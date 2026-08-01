"""Telegram notifier wrapper around the original bot notifier implementation."""

from __future__ import annotations

from trading_bot import TelegramNotifier as LegacyTelegramNotifier
from html import unescape
import re
import requests


class V4TelegramNotifier(LegacyTelegramNotifier):
    """Compatibility wrapper for the original Telegram notifier."""

    def verify_connection(self) -> dict:
        response = requests.get(f"{self.base_url}/getMe", timeout=10)
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            raise RuntimeError(payload.get("description", "Telegram getMe failed"))
        return payload["result"]

    def get_updates_checked(self, offset: int, timeout: int = 10) -> list[dict]:
        response = requests.get(
            f"{self.base_url}/getUpdates",
            params={"offset": offset, "timeout": timeout},
            timeout=timeout + 5,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            raise RuntimeError(payload.get("description", "Telegram getUpdates failed"))
        return payload.get("result", [])

    def send_to(self, chat_id: str, text: str) -> None:
        response = requests.post(
            f"{self.base_url}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        # If Telegram rejects malformed HTML, deliver a readable plain-text
        # version rather than losing an operational command reply or alert.
        if response.status_code == 400:
            try:
                description = str(response.json().get("description", ""))
            except ValueError:
                description = ""
            if "parse entities" in description.lower():
                plain_text = unescape(re.sub(r"<[^>]+>", "", text))
                response = requests.post(
                    f"{self.base_url}/sendMessage",
                    json={"chat_id": chat_id, "text": plain_text},
                    timeout=10,
                )
        if response.status_code >= 400:
            try:
                description = response.json().get("description", response.text)
            except ValueError:
                description = response.text
            error = requests.HTTPError(
                f"Telegram sendMessage returned {response.status_code}: {description}",
                response=response,
            )
            raise error


TelegramNotifier = V4TelegramNotifier
