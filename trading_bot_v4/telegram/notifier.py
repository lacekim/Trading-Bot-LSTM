"""Telegram notifier wrapper around the original bot notifier implementation."""

from __future__ import annotations

from trading_bot import TelegramNotifier as LegacyTelegramNotifier
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
        response.raise_for_status()


TelegramNotifier = V4TelegramNotifier
