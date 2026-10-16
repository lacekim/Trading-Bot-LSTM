"""Local web server for this project's dashboards -- run this, then open
http://localhost:8765/live_dashboard.html for the real, combined live
dashboard (actual Kraken/Alpaca/--ma200-auto positions, plus the market
snapshot backtest report embedded underneath), or
http://localhost:8765/cross_asset_dashboard.html for the older standalone
report on its own. No account, no external hosting, same pattern as
running `freqtrade webserver` and opening its local UI.

2026-09-21: added the /api/live_trades endpoint + live_dashboard.html.
Previously, "live" data only ever reached freqUI's own Dashboard page via
a hand-injected <script> in freqUI's compiled index.html (MutationObserver
+ a 500ms poll, gluing an iframe of the OLD static report onto freqUI's
real Vue-rendered page) -- fragile, and it already broke freqUI's own
Trade page rendering once before (see the removed injected script's own
comments in freqtrade-develop/.../ui/installed/index.html's git history /
FULL_HANDOFF.md Section G5). This replaces that: a real, self-owned page,
fed by a real JSON endpoint on this same server, no DOM hacking of
someone else's compiled app at all.
"""
from __future__ import annotations

import http.server
import json
import socketserver
import sys
from datetime import datetime, timezone
from pathlib import Path

PORT = 8765
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVE_DIR = PROJECT_ROOT / "runtime"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class UTF8HTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    # Without an explicit charset, the browser has to guess encoding --
    # confirmed this silently corrupted the ▲/▼ sort-arrow glyphs in the
    # dashboards. All three/four dashboards are UTF-8; declare it.
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".html": "text/html; charset=utf-8"}

    # live_dashboard.html (and, previously, freqUI's own page) fetches JSON
    # from here client-side -- different origin (port), so this needs an
    # explicit CORS allow or the browser silently blocks the fetch.
    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def do_GET(self) -> None:  # noqa: N802 -- stdlib handler method name
        if self.path == "/api/live_trades":
            self._serve_live_trades()
            return
        super().do_GET()

    def _serve_live_trades(self) -> None:
        """Real, live positions from all three systems this project
        actually trades through -- Kraken (both freqtrade strategies' own
        SQLite DBs), --ma200-auto's own tracked positions, and Alpaca's
        real account -- built from the exact same, already-tested
        functions tools/run_combined_status_report.py uses for Test_bot's
        /status merge (FULL_HANDOFF.md Section G4), so this can never
        drift from what Telegram already reports. Computed fresh on every
        request; nothing cached to go stale. Any failure still returns
        valid JSON with ok:false + the error, rather than a broken
        response the page can't parse."""
        payload: dict = {"generated_at": datetime.now(timezone.utc).isoformat()}
        try:
            from tools.run_combined_status_report import (
                FREQTRADE_DBS,
                _alpaca_positions,
                _freqtrade_open_trades,
                _ma200_auto_positions,
            )

            payload["kraken"] = {
                label: _freqtrade_open_trades(db_path)
                for label, db_path in FREQTRADE_DBS.items()
            }
            payload["ma200_auto"] = _ma200_auto_positions()
            payload["alpaca"] = _alpaca_positions()
            payload["ok"] = True
        except Exception as exc:  # noqa: BLE001 -- must always return valid JSON
            payload["ok"] = False
            payload["error"] = str(exc)

        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    handler = lambda *args, **kwargs: UTF8HTTPRequestHandler(
        *args, directory=str(SERVE_DIR), **kwargs
    )
    with socketserver.TCPServer(("127.0.0.1", PORT), handler) as httpd:
        print(f"serving {SERVE_DIR} at http://localhost:{PORT}/")
        print(f"live dashboard:   http://localhost:{PORT}/live_dashboard.html")
        print(f"old cross-asset:  http://localhost:{PORT}/cross_asset_dashboard.html")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
