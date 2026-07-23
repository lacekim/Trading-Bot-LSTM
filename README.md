# V5 GMX CNN/LSTM Trading Research and Paper-Trading System

V5 is a continuously running cryptocurrency research, model-inference, asset-selection, and persistent paper-trading system built around GMX market data. It combines an original CNN/LSTM price-direction model with optional Smart Money Concepts (SMC), daily GO/NO-GO qualification, hourly signal processing, portfolio risk controls, a decision dashboard, Telegram monitoring, safe shutdown controls, and Web3 read-only readiness checks.

The Python package is still named `trading_bot_v4` for compatibility with earlier versions of the project. The current user-facing scheduler and dashboard are V5.

> **Safety status:** PAPER mode is the supported execution mode. Web3 can perform read-only Arbitrum health checks, but GMX transaction construction, signing, broadcasting, and on-chain position reconciliation are not implemented. `LIVE_SMALL` and `LIVE` fail safely instead of submitting transactions.

## What the system does

V5 runs a forward-testing workflow using newly synchronized market data rather than reconstructing paper results afterward:

```text
GMX market data
       |
       v
Feature engineering + SMC context
       |
       v
CNN/LSTM and SMC model inference
       |
       v
Daily GO / WATCH / NO-GO qualification
       |
       v
Hourly signal generation for GO assets
       |
       v
Persistent paper order and position engine
       |
       v
Risk management, accounting, dashboard, and Telegram
```

The automatic scheduler performs two types of work:

- **Daily research:** refreshes the broader GMX universe, ranks assets, evaluates recent readiness, builds the qualified whitelist, and writes the V5 Daily Decision Dashboard.
- **Hourly cycles:** refreshes qualified assets, updates SMC features, runs the model, processes the newest closed-candle signals, manages paper positions, persists accounting state, and sends Telegram updates.

## Main features

### GMX market-data synchronization

- Discovers locally supported GMX symbols.
- Refreshes OHLCV data through the configured GMX update script.
- Uses symbol-scoped hourly refreshes for the current GO list.
- Keeps cached data as a fallback when a remote refresh fails.
- Deduplicates timestamps when loading OHLCV files.
- Normalizes columns and numeric types.
- Excludes configured stablecoin or unsupported markets.
- Supports a configurable GMX symbol blacklist.
- Uses the configured timeframe, currently `1h` by default.

The GMX paths and update script are inherited from [config.py](config.py). Confirm that `GMX_OHLC_DIR`, `GMX_UPDATE_SCRIPT`, and `GMX_UPDATE_CONFIG` point to valid locations on the machine running the bot.

### Feature engineering

The original model uses 18 features:

1. Returns
2. Volatility
3. Normalized RSI
4. Normalized MACD
5. Momentum
6. Volume ratio
7. Normalized ATR
8. Green-candle behavior
9. Price range
10. Directional volume
11. Bollinger Band position
12. On-balance volume
13. Price versus the 50-period moving average
14. Volume change
15. High/low ratio
16. Close position within the candle
17. Short-term trend
18. Long-term trend

The SMC layer adds market-structure context such as:

- Confirmed swing highs and lows
- Break of Structure (BOS)
- Change of Character (CHOCH)
- Fair Value Gaps (FVG)
- Order blocks
- Liquidity sweeps
- Market regime context

SMC behavior is configurable through `V4_ENABLE_SMC`, `V4_SMC_MIN_CONFIDENCE`, `V4_USE_ORDER_BLOCKS`, `V4_USE_FVG`, `V4_USE_LIQUIDITY_SWEEPS`, and swing-related settings.

### Model training and inference

- Original CNN/LSTM model and scaler support.
- Separate optional SMC-enhanced model and scaler.
- Fixed sequence construction using the configured sequence length.
- Feature-order compatibility with saved scalers and model inputs.
- Probability-to-direction conversion for LONG, SHORT, and HOLD.
- Model/scaler caching in the long-running scheduler.
- Automatic model reload when artifact timestamps change.
- Manual reload requests through the local runtime request mechanism.
- Original-versus-SMC model comparison reports.
- Test-only model comparisons using the final portion of each asset's data.

### Asset ranking and daily qualification

The research pipeline separates ranking from hard trading eligibility. A high ranking alone does not make an asset tradable.

Available evaluation inputs include:

- Recent 7-, 14-, and 30-day return
- Profit factor
- Maximum drawdown
- Trade count
- Walk-forward stability
- SMC improvement over the baseline model
- Constrained SMC performance
- Liquidity
- Trend strength
- Model confidence
- Market momentum

Daily decisions are saved as GO, WATCH, or NO-GO. Only strict GO assets enter the automatic hourly trading universe.

### V5 Daily Decision Dashboard

Daily research writes:

```text
reports/v5_daily_decision_dashboard.html
```

The dashboard includes:

- Current paper-trading decision
- GO, WATCHLIST, and NO-GO counts
- Current whitelist
- Market regime and risk context
- Market breadth and trend strength
- Today's opportunity table
- Paper allocation view
- Qualification changes and alerts
- Recent readiness metrics
- Market-momentum leaders
- Paper-performance results
- Validated asset rankings

The qualification CSV is stored at `reports/v4_daily_go_status.csv`; its legacy filename is retained for compatibility.
The full recent, validated, walk-forward, and forward-demotion evidence is stored at `reports/v5_daily_qualification_audit.csv`.

### Persistent paper trading

Paper trading uses SQLite by default:

```text
data/v5_paper_trading.sqlite3
```

The paper engine provides:

- Configurable starting balance
- Persistent cash, equity, realized P&L, unrealized P&L, and fees
- Restart-safe open positions
- Persistent signals, orders, closed trades, and equity history
- Deterministic signal IDs for candle-level deduplication
- Unique paper order and position IDs
- One active position per symbol
- LONG and SHORT positions
- Simulated market fills
- Entry and exit fees
- Slippage
- Estimated price impact
- Optional funding and borrowing costs
- Initial stop loss and take profit
- Signal-reversal exits
- Stop-loss and take-profit exits
- Position restoration after restart
- Pending entry cancellation during destructive shutdowns
- Protective exit preservation until the position is closed
- Direction-aware GMX oracle monitoring every 10 seconds while positions are open
- LONG protection evaluated with GMX `minPrice`; SHORT protection with `maxPrice`
- Automatic one-minute GMX candle fallback if the live ticker is unavailable
- Persistent monitor heartbeats, fallback diagnostics, and immediate exit events
- Five-second feed request timeouts and a 45-second watchdog that distinguishes a stopped thread from a slow cycle
- Hourly portfolio summaries

### Promotion, demotion, and shadow validation

GO status requires both recent performance and separate validated evidence. Defaults require:

- Positive 7-, 14-, and 30-day constrained returns
- 30-day profit factor of at least 1.30
- Positive validated constrained return and profit factor of at least 1.30
- Validated drawdown better than -5%
- At least 100 validated trades
- Walk-forward stability of at least 70

An asset that passes recent readiness but fails the promotion layer becomes `WATCH`. WATCH assets cannot open baseline paper positions, but they continue through hourly data refresh, inference, and the persistent shadow challenger.

Forward demotion uses the persistent paper database. It does not react until a symbol has at least 20 closed forward trades. It then evaluates the latest 30 trades using profit factor, expectancy, and account-relative drawdown; a failing GO asset becomes WATCH.

The shadow challenger is research-only and cannot place baseline or live orders. LONG confirmation requires a green candle, a close near the top of the range, and price above its trailing trend. SHORT confirmation mirrors those rules: a red candle, a close near the bottom, and price below trend. Shadow signals, positions, and completed trades are stored in SQLite and compared on the daily dashboard.

V5 also has an independent bearish CNN/LSTM path. Its positive class is an actual future decline greater than 1%; it does not misuse the upside model's negative class as a SHORT prediction. Train and calibrate it with:

```bash
python -m trading_bot_v4.main --train-bearish-model --timeframe 1h
```

The scheduler loads this model only when `models/smc_bearish_calibration.json` records `promoted: true`. Training uses a chronological 70% training block and a separate 15% model-selection block. Promotion then uses only the final untouched 15%. Promotion is per asset: final-holdout precision must be at least 0.55 and all three non-overlapping final-holdout windows must have positive net return and profit factor of at least 1.30. The current artifact promotes `CHZ` at a `0.77` SHORT threshold; no other symbol can execute a model-based SHORT. Calibrated thresholds are stored with the model, while detailed results are written to `reports/v5_bearish_validation.csv` and `reports/v5_bearish_walk_forward.csv`.

Paper results are forward state. The account is not reset to its starting balance when the scheduler restarts.

### Position sizing and risk controls

Position size is constrained by:

- Current account equity
- Maximum risk per trade
- Stop distance
- Maximum position percentage
- Maximum portfolio exposure
- Maximum number of open positions
- Available cash
- Minimum cash buffer
- Minimum order size
- Configured leverage

Leverage changes collateral requirements but does not increase the configured permitted account risk.

Relevant defaults and overrides are defined in [trading_bot_v4/config_v4.py](trading_bot_v4/config_v4.py).

### Accounting and persistence

SQLite stores:

- Starting and available balance
- Realized and unrealized P&L
- Accumulated fees
- Signals and deduplication IDs
- Orders and order state
- Open positions
- Closed trades
- Equity history
- Stop and target prices
- Last processed candle per symbol
- Current execution mode
- Current whitelist
- Shutdown mode and timestamp
- Previous clean/unclean shutdown status
- Entry permission state

At startup, V5 blocks new entries, restores the database, reconciles paper state, and only then enables entries when the execution mode permits them.

### Telegram monitoring and control

The V5 listener reuses the project's original Telegram notifier transport and runs concurrently with the scheduler. It verifies the token through Telegram's `getMe` API before reporting that the listener started.

At startup, the bot sends:

```text
✅ Bot Operating Automatically

📱 Available Commands:
/status - View Status
/balance - View Balance
/positions - View Open Positions
/help - Help
```

Supported commands:

| Command | Behavior |
|---|---|
| `/status` | Execution mode, scheduler state, entries, qualified assets, portfolio metrics, schedule, Web3 status, and signing status |
| `/balance` | Paper equity, realized P&L, unrealized P&L, and fees |
| `/positions` | All open positions |
| `/pause_entries` | Blocks new entries without stopping position management |
| `/resume_entries` | Re-enables entries only after successful reconciliation and when no shutdown is active |
| `/shutdown_graceful` | Requests a safe state-preserving shutdown |
| `/shutdown_close` | Requests confirmation before closing all positions and stopping |
| `/shutdown_emergency` | Requests confirmation before an emergency flatten-and-stop operation |
| `/help` | Lists available commands |

Destructive confirmation must match exactly:

```text
CONFIRM CLOSE
CONFIRM EMERGENCY
```

Unauthorized chat IDs receive no portfolio information and cannot control the bot.

Automatic Telegram notifications include:

- Scheduler startup and command list
- Position opened with direction, entry, size, stop, and target
- Position closed
- Immediate stop-loss and take-profit exits
- Repeated live-ticker failures, fallback operation, and lost monitor heartbeats
- Hourly updates for every open position with current price and unrealized P&L

### Shutdown control

V5 uses one shared, thread-safe `ShutdownController`. Signals, Telegram, the scheduler, paper execution, and local IPC all submit requests to this controller rather than manipulating positions from background threads.

Shutdown priority is:

```text
EMERGENCY > CLOSE_POSITIONS > GRACEFUL > NONE
```

#### GRACEFUL

- Default for Ctrl+C, SIGINT, and SIGTERM.
- Immediately blocks new entries and future scheduler cycles.
- Lets the active cycle finish at a safe boundary.
- Preserves open positions, stops, targets, and compatible exits.
- Does not close positions.
- Saves state and commits SQLite.
- Stops Telegram and the local control socket.

#### CLOSE_POSITIONS

- Blocks new entries and scheduler cycles.
- Cancels pending entry orders.
- Simulates reduce-only paper closes using the normal fee, slippage, carrying-cost, and price-impact rules.
- Confirms that no paper positions remain.
- Cancels obsolete protective exits after positions are flat.
- Saves final state and stops.
- Never reports success if positions remain.

#### EMERGENCY

- Blocks all new strategy activity.
- Cancels pending entries.
- Attempts immediate paper market closes.
- Uses controlled retries.
- Sends alerts on failures.
- Records manual-intervention state if the account cannot be flattened.

### Local IPC and single-instance protection

The automatic scheduler owns:

```text
runtime/trading_bot_v4.sock
runtime/trading_bot_v4.pid
runtime/trading_bot_v4.lock
```

The lock prevents two automatic schedulers from controlling the same account. A second startup correctly exits with `another trading bot instance is already running`.

Shutdown requests from another terminal use the Unix-domain socket and do not load models or start another trading process.

### Web3 readiness

The Web3 read-only check supports:

- `web3.py`
- Arbitrum RPC connection
- Expected chain ID validation (`42161`)
- Current block number
- Optional wallet checksum validation
- Optional wallet ETH balance
- No private key
- No signing
- No transaction broadcasting

GMX contract reads, position discovery, pending-order discovery, ABIs, approvals, transaction construction, nonce management, and live reconciliation remain future work.

## Execution modes

| Mode | Current behavior |
|---|---|
| `RESEARCH` | Research and reporting without paper execution or Telegram control startup |
| `PAPER` | Full supported persistent paper lifecycle and Telegram control |
| `WEB3_READ_ONLY` | Public Arbitrum health checks; no signing or paper entries |
| `LIVE_SMALL` | Reserved; currently refused because live signing is disabled |
| `LIVE` | Reserved; currently refused because live signing is disabled |

## Installation

### 1. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate javier
```

If the environment already exists:

```bash
conda env update -n javier -f environment.yml --prune
conda activate javier
```

Major dependencies include TensorFlow, pandas, NumPy, scikit-learn, Matplotlib, seaborn, requests, python-dotenv, Krakenex, schedule, and Web3.

### 2. Configure `.env`

Create `.env` in the repository root. It is excluded from Git. Never commit a private key, seed phrase, API secret, or Telegram token.

Minimal paper and Telegram example:

```env
EXECUTION_MODE=PAPER

TELEGRAM_ENABLED=true
TELEGRAM_TOKEN=your_existing_bot_token
TELEGRAM_CHAT_ID=your_chat_id

PAPER_STARTING_BALANCE=10000
PAPER_MAX_RISK_PER_TRADE=0.25
PAPER_MAX_TRADES_PER_DAY=3
PAPER_MAX_DAILY_LOSS_PCT=3
PAPER_MAX_EQUITY_LOSS_PCT=3
PAPER_MAX_DRAWDOWN_PCT=15
PAPER_MAX_CONSECUTIVE_LOSSES=5
PAPER_MAX_CANDLE_AGE_MULTIPLIER=2.5
```

The preferred token name is also supported:

```env
TELEGRAM_BOT_TOKEN=your_bot_token
```

For multiple authorized Telegram users:

```env
TELEGRAM_ALLOWED_CHAT_IDS=123456789,987654321
```

`TELEGRAM_TOKEN` remains supported for compatibility. `TELEGRAM_ALLOWED_CHAT_IDS` falls back to `TELEGRAM_CHAT_ID` when it is not provided.

Optional Web3 read-only configuration:

```env
ARBITRUM_RPC_URL=https://your-arbitrum-rpc
ARBITRUM_BACKUP_RPC_URL=https://your-backup-rpc
WEB3_WALLET_ADDRESS=0xYourPublicAddress
```

A private key is not needed for read-only mode and is not consumed by the current V5 implementation.

### 3. Confirm model and data artifacts

The scheduler expects trained model/scaler artifacts under `models/` and GMX OHLCV data at the configured location. Common artifacts include:

```text
models/lstm_ada_model.h5
models/scaler_ada.pkl
models/lstm_smc_model.h5
models/scaler_smc.pkl
```

## Running V5

### Automatic scheduler

```bash
python -m trading_bot_v4.main --auto --timeframe 1h
```

Expected startup includes:

```text
Telegram: enabled
Telegram listener started.
Authorized Telegram chat IDs: 1
V5 Scheduler started.
Execution mode: PAPER
New entries: enabled
Model loaded.
Scaler loaded.
Qualified LONG: ...
Qualified SHORT: ...
Live signing: disabled.
Web3 read-only: ...
```

Only one automatic scheduler may run at a time.

### Gracefully stop with Ctrl+C

Press `Ctrl+C` in the scheduler terminal. Positions are preserved; they are not automatically closed.

### Request shutdown from a second terminal

```bash
python -m trading_bot_v4.main --request-shutdown graceful
python -m trading_bot_v4.main --request-shutdown close-positions
python -m trading_bot_v4.main --request-shutdown emergency
```

The shorter `--shutdown` form is an alias.

### Web3 read-only check

```bash
python -m trading_bot_v4.main --web3-read-only
```

### Refresh GMX data

```bash
python -m trading_bot_v4.main --refresh
```

### Daily research and dashboard

```bash
python -m trading_bot_v4.main --daily-research --timeframe 1h
```

### Train and predict

```bash
python -m trading_bot_v4.main --train
python -m trading_bot_v4.main --predict
```

### Build and train the SMC model

```bash
python -m trading_bot_v4.main --build-smc-training-data --all-assets --timeframe 1h
python -m trading_bot_v4.main --train-smc-model
```

### Backtesting

```bash
# One symbol
python -m trading_bot_v4.main --backtest --symbol BTC --timeframe 1h

# Entire available GMX universe
python -m trading_bot_v4.main --backtest --all-assets --timeframe 1h

# Rank backtest results
python -m trading_bot_v4.main --backtest-rank --timeframe 1h

# Compare original behavior with V5
python -m trading_bot_v4.main --compare-original --symbol BTC --timeframe 1h
```

### SMC research and walk-forward validation

```bash
python -m trading_bot_v4.main --analyze-smc --symbol BTC --timeframe 1h
python -m trading_bot_v4.main --smc-shadow-backtest --all-assets --timeframe 1h
python -m trading_bot_v4.main --walk-forward-smc --timeframe 1h
```

### Paper signal and readiness analysis

```bash
python -m trading_bot_v4.main --paper-trade-smc --all-assets --timeframe 1h
python -m trading_bot_v4.main --paper-trade-smc-model --all-assets --timeframe 1h
python -m trading_bot_v4.main --paper-readiness --timeframe 1h
python -m trading_bot_v4.main --go-assets-performance --timeframe 1h
```

### Asset ranking and model comparison

```bash
python -m trading_bot_v4.main --rank-assets --validated --timeframe 1h
python -m trading_bot_v4.main --validate-asset-rankings --timeframe 1h
python -m trading_bot_v4.main --compare-models --timeframe 1h
python -m trading_bot_v4.main --compare-paper-models --timeframe 1h
python -m trading_bot_v4.main --compare-paper-model-performance --timeframe 1h
```

Run `python -m trading_bot_v4.main --help` for the complete current CLI.

## Important files and directories

```text
trading_bot_v4/
  backtesting/          Backtests, walk-forward tests, rankings, and reports
  core/                 Features, indicators, signals, and market structure
  database/             Performance tracking helpers
  execution/            Paper execution, comparisons, shutdown operations
  features/             SMC training feature construction
  ml/                   Models, trainers, predictors, and caching
  research/             Market scanner, daily research, and scheduler
  risk/                 Sizing, stops, and portfolio-risk components
  telegram/             Legacy-backed notifier and V5 command listener
  tests/                Automated test suite
  utils/                Logging, helpers, and signal utilities

data/                    Persistent paper database and local state
logs/                    Scheduler, paper signals, audits, and reports
models/                  Model, scaler, ranking, and comparison artifacts
reports/                 V5 dashboard, qualification status, and SMC snapshots
runtime/                 PID, lock, and Unix socket while the scheduler runs
```

## Testing

Run the complete suite:

```bash
python -m unittest discover -s trading_bot_v4/tests -p 'test_*.py'
```

The suite covers signal direction, market scanning, comparison logic, daily decisions, paper persistence, deduplication, reversals, risk limits, shutdown policies, shutdown priority, Unix-socket IPC, single-instance locking, Telegram authorization and confirmation, concurrent polling, legacy Telegram commands, and automatic position notifications.

## Logs and troubleshooting

Primary scheduler log:

```text
logs/v4_scheduler.log
```

The filename is retained for compatibility even though the running system is V5.

### `another trading bot instance is already running`

This means a scheduler already owns the runtime lock. Check `runtime/trading_bot_v4.pid`. Use the second-terminal graceful shutdown request instead of starting another instance.

### Telegram does not start

Startup prints one of these explicit states:

```text
Telegram: enabled
Telegram: disabled — TELEGRAM_ENABLED is false
Telegram: unavailable — TELEGRAM_BOT_TOKEN is missing
Telegram startup failed: <actual error>
```

Check `.env`, the authorized chat ID, internet access, and `logs/v4_scheduler.log`. Only one process should poll a Telegram bot token at a time.

### Web3 says the RPC URL is missing

Set `ARBITRUM_RPC_URL` in `.env`. This does not enable live signing; it only enables the public read-only check.

### Matplotlib cache warnings

`sitecustomize.py` directs Matplotlib to the writable `logs/matplotlib` cache when Python is launched from this repository.

## Security and operational guidance

- Keep `.env` out of source control.
- Never store a seed phrase or raw private key in this repository.
- Use a dedicated wallet with limited funds for future live testing.
- Keep `EXECUTION_MODE=PAPER` until on-chain execution and reconciliation are implemented and audited.
- Treat Telegram as an operational control surface: authorize only known chat IDs.
- Use `GRACEFUL` for routine restarts and updates.
- Confirm database restoration and reconciliation before resuming entries.
- Do not delete the SQLite database if you want to preserve the forward test.
- Review dashboard and scheduler logs rather than relying only on data-sync output.

## Current limitations

- No live GMX order creation or execution.
- No private-key loading or transaction signing.
- No GMX contract ABI integration.
- No on-chain open-position or pending-order reconciliation.
- No token approval or nonce management.
- Funding and borrowing are configurable estimates rather than live GMX rate reads.
- Paper fills approximate execution using configured fees, slippage, and price impact.
- Several filenames and the Python package retain `v4` for backward compatibility.

V5 should currently be treated as a live-data research and persistent forward paper-trading system—not a production live-capital trading engine.
