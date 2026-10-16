# Chapters 8 and 9 on the side-free book model, alpaca

Events with labels ending before 2024-01-01: 252,460.

## Feature importance (Ch.8)

| feature | MDI | MDA | SFI neg log-loss |
|---|---:|---:|---:|
| ffd | 0.2354 | 0.0009 | -0.69439 |
| log_trgt | 0.2057 | -0.0005 | -0.69507 |
| sadf | 0.1683 | -0.0004 | -0.69589 |
| ent_plugin | 0.1788 | 0.0001 | -0.69497 |
| ent_konto | 0.1417 | -0.0001 | -0.69519 |
| cs_spread | 0.0700 | 0.0000 | -0.69464 |

MDA baseline OOS neg log-loss -0.69448.

## Hyperparameter search (Ch.9, purged 3-fold)

| cand | min_weight_fraction_leaf | max_features | mean OOS neg log-loss |
|---:|---:|---|---:|
| 0 (untuned) | 0.0500 | sqrt | -0.695324 |
| 1 | 0.0155 | 1 | -0.695085 |
| 2 | 0.0309 | 6 | -0.696274 |
| 3 | 0.0212 | 2 | -0.695802 |
| 4 | 0.0111 | 3 | -0.697594 |
| 5 | 0.0703 | 6 | -0.695531 |
| 6 | 0.0052 | 3 | -0.698583 |
| 7 | 0.1654 | 1 | -0.694214 |
| 8 | 0.0868 | 1 | -0.694691 |
| 9 | 0.0096 | 2 | -0.696952 |
| 10 | 0.0364 | 6 | -0.696249 |

Chosen: candidate 7.