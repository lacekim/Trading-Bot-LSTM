#!/usr/bin/env bash
# tools/restart_bots.sh
#
# One-command clean restart for this project's live/paper-trading
# processes. Replaces copy-pasting the ad-hoc restart snippets that had
# piled up across handoff.md/handoff2.md/handoff3.md/handoff4.md (each
# with its own incrementing log filename) -- run this one script instead.
#
# What it restarts:
#   - freqtrade, port 8080, GoldenCrossStrategy (Kraken; wired to
#     Test_bot's Telegram token -- see FULL_HANDOFF.md Section G3/G4)
#   - trading_bot_v4.main --ma200-auto   (Kraken, golden-cross + BMSB)
#   - trading_bot_v4.main --alpaca-auto  (Alpaca, equities + migrated crypto)
#
# What it deliberately does NOT touch:
#   - freqtrade, port 8081, GoldenCrossRegimeFilteredStrategy -- a
#     separate, independent forward-test strategy (FULL_HANDOFF.md
#     Section C). Its status is printed at the end so you can see if
#     it's running, but this script never kills or restarts it.
#
# Logs: one fixed-name file per process under logs/ at the project root
# (append mode, with a "=== restart <UTC timestamp> ===" separator each
# run) instead of a new incrementing filename every time.
#
# Usage:
#   bash tools/restart_bots.sh
#
set -uo pipefail

PROJECT_ROOT="/Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera"
JAVIER_BIN="/opt/homebrew/Caskroom/miniconda/base/envs/javier/bin"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

kill_and_wait() {
  local pattern="$1" label="$2"
  local pids
  pids=$(pgrep -f "$pattern" || true)
  if [ -z "$pids" ]; then
    echo "  $label: not running, nothing to kill"
    return
  fi
  echo "  $label: killing PID(s) $pids"
  pkill -f "$pattern" || true
  for _ in $(seq 1 15); do
    if ! pgrep -f "$pattern" >/dev/null 2>&1; then
      echo "  $label: confirmed stopped"
      return
    fi
    sleep 1
  done
  echo "  $label: still alive after 15s, forcing (kill -9)"
  pkill -9 -f "$pattern" || true
}

echo "=== stopping (if running) ==="
kill_and_wait "freqtrade trade --strategy GoldenCrossStrategy" "freqtrade (8080, GoldenCrossStrategy)"
kill_and_wait "trading_bot_v4.main --ma200-auto" "--ma200-auto"
kill_and_wait "trading_bot_v4.main --alpaca-auto" "--alpaca-auto"

echo
echo "=== starting ==="

{ echo; echo "=== restart $(stamp) ==="; } >> "$LOG_DIR/freqtrade_8080.log"
( cd "$PROJECT_ROOT/freqtrade-develop" && nohup "$JAVIER_BIN/freqtrade" trade --strategy GoldenCrossStrategy --config user_data/config.json >> "$LOG_DIR/freqtrade_8080.log" 2>&1 & disown )
echo "  freqtrade (8080): launched"

{ echo; echo "=== restart $(stamp) ==="; } >> "$LOG_DIR/ma200_auto.log"
( cd "$PROJECT_ROOT" && nohup "$JAVIER_BIN/python" -m trading_bot_v4.main --ma200-auto >> "$LOG_DIR/ma200_auto.log" 2>&1 & disown )
echo "  --ma200-auto: launched"

{ echo; echo "=== restart $(stamp) ==="; } >> "$LOG_DIR/alpaca_auto.log"
( cd "$PROJECT_ROOT" && nohup "$JAVIER_BIN/python" -m trading_bot_v4.main --alpaca-auto >> "$LOG_DIR/alpaca_auto.log" 2>&1 & disown )
echo "  --alpaca-auto: launched"

echo
echo "=== waiting 6s, then confirming ==="
sleep 6

echo
echo "--- live process check ---"
ps aux | grep -E "GoldenCrossStrategy|GoldenCrossRegimeFilteredStrategy|ma200-auto|alpaca-auto" | grep -v grep
if [ $? -ne 0 ]; then
  echo "(nothing matched -- something's wrong)"
fi

echo
echo "--- freqtrade (8080) last 20 log lines ---"
tail -20 "$LOG_DIR/freqtrade_8080.log"

echo
echo "--- checking freqtrade log for errors ---"
if ! grep -i "conflict\|traceback\|unauthorized" "$LOG_DIR/freqtrade_8080.log" | tail -10; then
  echo "(none found)"
fi

echo
echo "Done. Port 8081 (GoldenCrossRegimeFilteredStrategy) was left untouched by design -- check separately if needed."
echo "Logs now live at: $LOG_DIR/{freqtrade_8080,ma200_auto,alpaca_auto}.log"
