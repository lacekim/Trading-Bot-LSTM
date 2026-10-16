#!/usr/bin/env bash
# run_kraken_whale_monitor_loop.sh
#
# Runs tools/run_kraken_large_trade_monitor.py repeatedly on a fixed
# interval and stops itself automatically after a predeclared number of
# runs, so the "whale" verdicts can be watched across changing market
# conditions (time of day, weekday/weekend) before deciding whether/how
# to combine this with the golden-cross signal -- per the project's own
# discipline of observing a diagnostic before wiring it into anything.
#
# This script does NOT modify run_kraken_large_trade_monitor.py or any
# trading logic. It only orchestrates repeated calls to the existing
# read-only monitor and archives every run's output (the monitor itself
# only keeps the LATEST snapshot -- this loop is what builds the time
# series).
#
# INTERVAL_SECONDS and TOTAL_RUNS are predeclared below (overridable via
# environment variables) rather than picked after peeking at early
# results. Defaults: 1 run per hour for 48 runs = 48 hours of coverage,
# spanning two full day/night cycles and at least one weekday/weekend
# boundary depending on start time.
#
# Usage (run from the repo root, or anywhere -- paths are resolved
# relative to this script's own location):
#   bash tools/run_kraken_whale_monitor_loop.sh
#
# To let it keep running after closing the terminal:
#   nohup bash tools/run_kraken_whale_monitor_loop.sh > /dev/null 2>&1 &
#
# To override the defaults for one run without editing this file:
#   INTERVAL_SECONDS=1800 TOTAL_RUNS=96 bash tools/run_kraken_whale_monitor_loop.sh
#
# PYTHON below hardcodes the same conda interpreter this project's other
# loop scripts already use (see runtime/ma200_paper_loop.sh's own PYTHON=
# line) instead of relying on bare "python3" resolving correctly under
# nohup/background execution -- that bare-"python3" approach was tried
# first and broke silently (it resolved to an interpreter without numpy
# installed, unlike the javier conda env every other script here uses).
# Override via KRAKEN_MONITOR_PYTHON if this ever runs on a different
# machine or under a different env name.
set -euo pipefail

# ---- predeclared parameters (override via env vars if needed) ------------
INTERVAL_SECONDS="${INTERVAL_SECONDS:-3600}"   # seconds between runs (default: 1 hour)
TOTAL_RUNS="${TOTAL_RUNS:-48}"                  # number of runs before the loop stops itself
PYTHON="${KRAKEN_MONITOR_PYTHON:-/opt/homebrew/Caskroom/miniconda/base/envs/javier/bin/python}"
# ----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RUNTIME_DIR="$REPO_ROOT/runtime"
HISTORY_DIR="$RUNTIME_DIR/kraken_large_trade_monitor_history"
LATEST_JSON="$RUNTIME_DIR/kraken_large_trade_monitor_results.json"
LOG_PATH="$RUNTIME_DIR/kraken_large_trade_monitor_loop.log"
DONE_MARKER="$RUNTIME_DIR/kraken_large_trade_monitor_loop.done"

mkdir -p "$HISTORY_DIR"
rm -f "$DONE_MARKER"

log() {
    printf '%s %s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$*" | tee -a "$LOG_PATH"
}

# ---- preflight: fail fast, not slow -----------------------------------------
# $PYTHON is what every scheduled run will use for the next
# TOTAL_RUNS * INTERVAL_SECONDS worth of real time -- if the interpreter is
# missing, or missing a dependency the monitor needs, every single run would
# fail the same way. Check that ONCE, up front, and refuse to start the loop
# rather than quietly burning through hours of retries that can never succeed.
if [ ! -x "$PYTHON" ]; then
    {
        echo "preflight FAILED: PYTHON is set to '${PYTHON}' but that is not an executable file."
        echo "This project's other loops (see runtime/ma200_paper_loop.sh) use the javier conda env's"
        echo "interpreter at this exact path -- if it moved, or you're on a different machine, override it:"
        echo "  KRAKEN_MONITOR_PYTHON=/path/to/python bash $(basename "${BASH_SOURCE[0]}")"
        echo "Then re-run this script. The loop was NOT started -- nothing was scheduled."
    } | tee -a "$LOG_PATH" >&2
    exit 1
fi
if ! "$PYTHON" -c "import numpy, requests" >/dev/null 2>&1; then
    {
        echo "preflight FAILED: ${PYTHON} cannot import numpy and/or requests."
        echo "Fix: ${PYTHON} -m pip install numpy requests"
        echo "Then re-run this script. The loop was NOT started -- nothing was scheduled."
    } | tee -a "$LOG_PATH" >&2
    exit 1
fi

total_hours=$(( TOTAL_RUNS * INTERVAL_SECONDS / 3600 ))
log "=== preflight OK (PYTHON=${PYTHON}) ==="
log "=== starting loop: ${TOTAL_RUNS} runs, ${INTERVAL_SECONDS}s apart (~${total_hours}h total coverage) ==="

for i in $(seq 1 "$TOTAL_RUNS"); do
    ts="$(date -u +"%Y%m%dT%H%M%SZ")"
    log "--- run ${i} of ${TOTAL_RUNS} (timestamp ${ts}) ---"

    if "$PYTHON" "$SCRIPT_DIR/run_kraken_large_trade_monitor.py" >> "$LOG_PATH" 2>&1; then
        if [ -f "$LATEST_JSON" ]; then
            cp "$LATEST_JSON" "$HISTORY_DIR/kraken_large_trade_monitor_${ts}.json"
            log "run ${i} saved to ${HISTORY_DIR}/kraken_large_trade_monitor_${ts}.json"
        else
            log "run ${i} completed but ${LATEST_JSON} was not found -- nothing archived for this run"
        fi
    else
        log "run ${i} FAILED -- see the error above in this same log; continuing to the next scheduled run"
    fi

    if [ "$i" -lt "$TOTAL_RUNS" ]; then
        sleep "$INTERVAL_SECONDS"
    fi
done

log "=== loop finished: ${TOTAL_RUNS} runs collected in ${HISTORY_DIR} ==="
log "run: python3 tools/summarize_kraken_large_trade_monitor_history.py"
log "to see the aggregated time series before we discuss it."
echo "finished at $(date -u +"%Y-%m-%dT%H:%M:%SZ")" > "$DONE_MARKER"
