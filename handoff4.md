# Handoff #4: load the Alpaca /status merge into the running Test_bot

Context: G3 (swapping freqtrade's own Telegram integration onto
Test_bot's token) is live and confirmed working — PID 23539, keyboard
confirmed, `/status` confirmed showing Kraken trades. But the user's
original ask was that `/status` itself also show Alpaca + `--ma200-auto`
positions, not just the Kraken side — that part got missed in G3 and is
now built as G4. Full reasoning in `FULL_HANDOFF.md` Section G4.

All code edits are already made and committed. This is a restart +
verification pass, same pattern as the last few handoffs.

## 1. Verify the edits are actually in place

```
cd /Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera
grep -n "PROJECT PATCH" freqtrade-develop/freqtrade/rpc/telegram.py
```

Expected: 6 lines (one function block, one call site in `_status_msg`,
one in `_status_table`).

```
grep -n "def build_extra_only" tools/run_combined_status_report.py
```

Expected: present.

## 2. Sanity-check the direct import BEFORE restarting freqtrade

This is the one part not yet proven live — the whole point of G4 was
avoiding a subprocess bridge because freqtrade runs in the same conda
env as everything else. Confirm that's actually true by importing it the
same way `telegram.py` will:

```
/opt/homebrew/Caskroom/miniconda/base/envs/javier/bin/python -c "
import sys
sys.path.insert(0, '.')
from tools.run_combined_status_report import build_extra_only
print(build_extra_only())
"
```

Expected: prints the `--ma200-auto golden-cross`, `--ma200-auto BMSB`,
and `Alpaca` sections cleanly, no traceback. If this fails, fix it here
first — this exact import is what `telegram.py`'s `_alpaca_status_extra()`
does at runtime, just with `Path(__file__).resolve().parents[3]` computing
the project root instead of the hardcoded `.` used here for the sanity
check.

## 3. Restart freqtrade (port 8080 only)

```
pkill -f "freqtrade trade --strategy GoldenCrossStrategy"
ps aux | grep "GoldenCrossStrategy" | grep -v grep   # should return nothing
```

(Port 8081 is a different strategy/config, `telegram.enabled: false` —
untouched, leave it running.)

```
cd /Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop
nohup /opt/homebrew/Caskroom/miniconda/base/envs/javier/bin/freqtrade trade --strategy GoldenCrossStrategy --config user_data/config.json >> /private/tmp/freqtrade_live_trade_restart4.log 2>&1 &
disown
sleep 6
ps aux | grep "GoldenCrossStrategy" | grep -v grep
```

Record the new PID. Then check the log the same way as last time:

```
tail -60 /private/tmp/freqtrade_live_trade_restart4.log
grep -i "conflict\|error\|unauthorized\|traceback\|ALPACA_STATUS_BRIDGE" /private/tmp/freqtrade_live_trade_restart4.log
```

Clean startup should look the same as the last restart (Telegram
`Application started`, whitelist with 5 pairs, heartbeat) — nothing new
should appear here just from starting up; the bridge only runs when
`/status` is actually pressed.

## 4. Real test — needs the user

Ask the user to press `/status` in Test_bot on his phone. Expected: the
normal Kraken trades exactly as before (BTC/ETH/SOL/TRX — XMR closed via
the -8% stop, see `FULL_HANDOFF.md` Section E), **plus one more message**
after them with `--ma200-auto golden-cross`, `--ma200-auto BMSB`, and
`Alpaca` sections (this is the part that was missing before). Then try
`/status table` too — same expectation.

If the extra message doesn't show up, check the log immediately after
for an `ALPACA_STATUS_BRIDGE` warning line — that means the import or
the call itself failed at runtime even though step 2's standalone check
passed (different working directory/environment inside the running
process is the most likely culprit). Report back exactly what was seen
rather than guessing a fix.

## 5. Re-record the patch-checker baseline

Once step 4 confirms it's actually working:

```
/opt/homebrew/Caskroom/miniconda/base/envs/javier/bin/python -m tools.check_telegram_patch --record
/opt/homebrew/Caskroom/miniconda/base/envs/javier/bin/python -m tools.check_telegram_patch
```

Second command should print `OK`. (The manifest already has the right
hashes for the current file — this just re-confirms nothing drifted
between the file edit and now, cheap and worth doing.)

## When done

Report back (in chat, not by editing `FULL_HANDOFF.md` directly): new
PID, whether the direct-import sanity check worked, whether the extra
message actually arrived in `/status` and `/status table`, and whether
`check_telegram_patch` printed OK.
