# Handoff: finish the BTC-removal restart + combined-status test

Both code edits are already made and committed. What's left is mechanical:
restart two processes so the edits take effect, and run one test command.
Don't improvise beyond what's below — the edits themselves were already
worked out carefully (see `FULL_HANDOFF.md` Section B5 for the full
reasoning if you want context), this is just executing them.

## 1. Verify the edits are actually in place

```
cd /Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera
grep -A6 '"pair_whitelist"' freqtrade-develop/user_data/config.json
grep -n "KRAKEN_ENTRY_EXCLUDED_ASSETS" tools/run_ma200_paper_trader.py
```

Expected: no `"BTC/USD"` in the whitelist (ETH/XMR/PAXG/SOL/TRX still
there), and `KRAKEN_ENTRY_EXCLUDED_ASSETS = {"BTC"}` defined. If either
edit isn't there, stop and report back rather than re-deriving it —
something's wrong with what got committed.

## 2. Restart freqtrade (port 8080 only) and `--ma200-auto`

```
pkill -f "freqtrade trade --strategy GoldenCrossStrategy"
pkill -f "trading_bot_v4.main --ma200-auto"
```

Confirm both are actually dead before relaunching:

```
ps aux | grep -E "GoldenCrossStrategy|ma200-auto" | grep -v grep
```

Should return nothing. (Port 8081 / `GoldenCrossRegimeFilteredStrategy` is
a different strategy and this pattern won't touch it — leave it alone
either way.)

Then relaunch both, detached:

```
cd /Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera/freqtrade-develop
nohup /opt/homebrew/Caskroom/miniconda/base/envs/javier/bin/freqtrade trade --strategy GoldenCrossStrategy --config user_data/config.json >> /private/tmp/freqtrade_live_trade_restart.log 2>&1 &
disown

cd /Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera
nohup /opt/homebrew/Caskroom/miniconda/base/envs/javier/bin/python -m trading_bot_v4.main --ma200-auto >> /private/tmp/ma200_auto.log 2>&1 &
disown
```

Record the new PIDs (`ps aux | grep -E "GoldenCrossStrategy|ma200-auto" | grep -v grep`).

## 3. Test the combined-status extension

```
cd /Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera
/opt/homebrew/Caskroom/miniconda/base/envs/javier/bin/python -m trading_bot_v4.main --combined-status
```

Confirm the output includes a `--ma200-auto golden-cross (Kraken)` section
and a `--ma200-auto BMSB (Kraken)` section (in addition to the existing
freqtrade + Alpaca sections), and that the `✅ Telegram:` log line actually
appears — that's the only proof it really sent, not just printed.

## When done

Report the results back in chat (new PIDs, whether both edits/restarts
succeeded, whether the Telegram send confirmed) rather than editing
`FULL_HANDOFF.md` directly — that file's been silently reverted three
times tonight already, almost certainly from a stale in-context write.
Mike will relay the results and the master doc gets updated from there.
