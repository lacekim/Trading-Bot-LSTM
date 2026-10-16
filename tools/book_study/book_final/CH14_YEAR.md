# Chapter 14 backtest statistics, last 365 days, Kraken and Alpaca

Book strategy, walk-forward path, net of costs, HRP allocation. Series: the account's daily target position (sum over instruments of HRP weight x averaged bet signal at the day's last bar) and its daily net return. Snippets 14.1-14.4 as ported. 'n/a' = the snippet returns nothing (a HHI needs more than two bets, and negative-return bets must exist).

| | Kraken | Alpaca |
|---|---|---|
| Period | 2025-10-13 to 2026-10-12 | 2025-10-10 to 2026-10-09 |
| Gain over the year, net of costs, % | 0.50 | 0.0051 |
| Largest absolute account position (share of capital) | 0.0514 | 0.00106 |
| Mean absolute account position | 0.0131 | 0.00038 |
| Days with a non-zero position | 364 | 364 |
| Bets (Snippet 14.1) | 4 | 1 |
| Average holding period, days (14.2) | 16.2 | 18.1 |
| Return of each bet, % (14.1) | 0.5251, 0.0053, -0.0345, 0.0000 | 0.0051 |
| HHI of positive-return bets (14.3) | 0.9703 | n/a |
| HHI of negative-return bets (14.3) | n/a | n/a |
| HHI of bets per month (14.3) | 0.4375 | n/a |
| Drawdowns: count / max % / 95th pct % (14.4) | 24 / 0.595 / 0.247 | 7 / 0.0145 / 0.0109 |
| Time under water, years: max / 95th pct (14.4) | 0.178 / 0.119 | 0.033 / 0.032 |

Not computed: the other Chapter 14 measures (average AUM, capacity, leverage, turnover, hit ratio, implementation shortfall) are described in the book's text, not given as snippets in the extract.
