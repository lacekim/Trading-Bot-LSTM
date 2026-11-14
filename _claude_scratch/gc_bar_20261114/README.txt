Golden cross bar on both venues - run 2026-11-14 (offline; no live file read-write, nothing committed)

KRAKEN  kraken_bar.py  = tools/run_golden_cross_variants.py run() unchanged; only the loader (CSV from
        runtime/venue_candles_v2/kraken_1h) and the windows differ. gc_run.py reproduces
        logs/v5_golden_cross_variants.json exactly on the January data (_claude_scratch/*_USD-1h.csv).
        BTC candles in venue_candles_v2 end 2026-01-05; the other five run to 2026-10-12.
ALPACA  alpaca_bar.py  = compute_trend_state / backtest_one / simulate_portfolio / robustness taken unchanged
        from tools/run_cross_asset_golden_cross_backtest_4h_alpaca.py, run on DAILY closes in
        runtime/venue_candles/alpaca_1d/alpaca_daily.csv.gz for the robust-on-daily universe.
        55 of 60 instruments are in the local file (missing: QQQ, VTI, DIA, GLD, PPLT).
CHECK   check.py       = separate code. Kraken: 6 of 6 rows match. Alpaca: window trade counts match (80, 19);
        whole-file differs (engine 404 closed trades, strict-cross count 355) because the engine opens a
        position on the first bar after the 200-bar warm-up when 50MA is already above 200MA.
        alpaca_bar_checked.json also marks still-open positions at the last close (the engine counts closed only).
