"""Check whether the ALPACA_STATUS_BRIDGE patch in freqtrade's vendored
telegram.py is still present and unmodified.

freqtrade-develop/freqtrade/rpc/telegram.py is third-party framework
code this project does not own. The /status and /status table
extension that appends Alpaca + --ma200-auto positions to freqtrade's
own Kraken-trade display was added directly into that file (see its
"PROJECT PATCH: ALPACA_STATUS_BRIDGE" comment blocks). If
freqtrade-develop is ever upgraded or re-pulled from upstream, that
file can be silently overwritten and this patch lost -- freqtrade would
keep running fine, /status would keep working, just quietly back to
Kraken-only again, with no error anywhere. This script exists so that
loss gets noticed instead of discovered by accident days later.

Usage:
    python -m tools.check_telegram_patch            # check, print result,
                                                      # exit 0 if OK else 1
    python -m tools.check_telegram_patch --record    # (re)record the
                                                      # expected hashes --
                                                      # run this once right
                                                      # after the patch is
                                                      # applied and verified
                                                      # working, and again
                                                      # any time you
                                                      # intentionally change
                                                      # the patch itself
    python -m tools.check_telegram_patch --notify    # on failure, also
                                                      # send a Telegram
                                                      # alert (via whatever
                                                      # bot .env currently
                                                      # points at)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TELEGRAM_PY = ROOT / "freqtrade-develop" / "freqtrade" / "rpc" / "telegram.py"
MANIFEST_PATH = ROOT / "tools" / "telegram_patch_manifest.json"

# Three independently-checked blocks: the bridge function itself (plus its
# big explanatory comment), and its two call sites inside _status_msg and
# _status_table. Checking them separately means a partial loss (e.g. only
# one call site clobbered by a merge) is still caught, not just a
# wholesale file replacement.
BLOCKS = {
    "bridge_function": (
        "# === PROJECT PATCH: ALPACA_STATUS_BRIDGE START ===",
        "# === PROJECT PATCH: ALPACA_STATUS_BRIDGE END ===",
    ),
    "status_msg_call": (
        "# === PROJECT PATCH: ALPACA_STATUS_BRIDGE CALL (status_msg) START ===",
        "# === PROJECT PATCH: ALPACA_STATUS_BRIDGE CALL (status_msg) END ===",
    ),
    "status_table_call": (
        "# === PROJECT PATCH: ALPACA_STATUS_BRIDGE CALL (status_table) START ===",
        "# === PROJECT PATCH: ALPACA_STATUS_BRIDGE CALL (status_table) END ===",
    ),
}


def _extract_block(text: str, start_marker: str, end_marker: str) -> str | None:
    start = text.find(start_marker)
    if start == -1:
        return None
    end = text.find(end_marker, start)
    if end == -1:
        return None
    return text[start : end + len(end_marker)]


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _current_hashes() -> dict[str, str | None]:
    if not TELEGRAM_PY.exists():
        return {name: None for name in BLOCKS}
    text = TELEGRAM_PY.read_text(encoding="utf-8")
    hashes: dict[str, str | None] = {}
    for name, (start_marker, end_marker) in BLOCKS.items():
        block = _extract_block(text, start_marker, end_marker)
        hashes[name] = _hash(block) if block is not None else None
    return hashes


def record() -> None:
    hashes = _current_hashes()
    missing = [name for name, h in hashes.items() if h is None]
    if missing:
        print(f"Refusing to record -- these blocks are missing right now: {missing}")
        print("Fix the patch first, then re-run --record.")
        sys.exit(1)
    MANIFEST_PATH.write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded expected hashes to {MANIFEST_PATH}")


def check(notify: bool) -> bool:
    if not MANIFEST_PATH.exists():
        print(
            f"No manifest at {MANIFEST_PATH} yet -- run with --record once "
            "right after the patch is applied and verified working."
        )
        return False

    expected = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    current = _current_hashes()

    problems: list[str] = []
    if not TELEGRAM_PY.exists():
        problems.append(f"{TELEGRAM_PY} does not exist at all.")
    for name in BLOCKS:
        exp = expected.get(name)
        cur = current.get(name)
        if cur is None:
            problems.append(f"'{name}' block is MISSING from telegram.py (markers not found).")
        elif exp is not None and cur != exp:
            problems.append(f"'{name}' block is PRESENT but MODIFIED (hash mismatch).")

    if not problems:
        print("OK -- ALPACA_STATUS_BRIDGE patch is intact in telegram.py.")
        return True

    message = (
        "ALPACA_STATUS_BRIDGE patch check FAILED for "
        "freqtrade-develop/freqtrade/rpc/telegram.py:\n  - "
        + "\n  - ".join(problems)
        + "\n\nMost likely cause: freqtrade-develop was upgraded/reinstalled "
        "and this vendored file got overwritten. /status may have quietly "
        "gone back to Kraken-only. Re-apply the patch (see FULL_HANDOFF.md "
        "for the exact edit)."
    )
    print(message)

    if notify:
        try:
            sys.path.insert(0, str(ROOT))
            from dotenv import load_dotenv

            load_dotenv(ROOT / ".env")
            from trading_bot_v4.config_v4 import V4Config as Config
            from trading_bot_v4.telegram.notifier import V4TelegramNotifier

            if getattr(Config, "TELEGRAM_ENABLED", False) and getattr(
                Config, "TELEGRAM_BOT_TOKEN", None
            ):
                V4TelegramNotifier().send_message("⚠️ " + message)
                print("[alert sent to Telegram]")
            else:
                print("[telegram disabled -- could not send alert]")
        except Exception as exc:  # noqa: BLE001 -- this is a best-effort alert
            print(f"[could not send Telegram alert: {exc}]")

    return False


def main() -> None:
    args = sys.argv[1:]
    if "--record" in args:
        record()
        return
    ok = check(notify="--notify" in args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
