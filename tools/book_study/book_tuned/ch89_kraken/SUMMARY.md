# Chapters 8 and 9 on the side-free book model, kraken

Events with labels ending before 2024-01-01: 28,279.

## Feature importance (Ch.8)

| feature | MDI | MDA | SFI neg log-loss |
|---|---:|---:|---:|
| ffd | 0.1638 | -0.0002 | -0.70025 |
| log_trgt | 0.1876 | -0.0000 | -0.69909 |
| sadf | 0.2277 | 0.0009 | -0.69750 |
| ent_plugin | 0.1802 | -0.0009 | -0.69911 |
| ent_konto | 0.1333 | 0.0004 | -0.69898 |
| cs_spread | 0.1073 | 0.0004 | -0.69782 |

MDA baseline OOS neg log-loss -0.69633.

## Hyperparameter search (Ch.9, purged 3-fold)

| cand | min_weight_fraction_leaf | max_features | mean OOS neg log-loss |
|---:|---:|---|---:|
| 0 (untuned) | 0.0500 | sqrt | -0.699824 |
| 1 | 0.0155 | 1 | -0.701995 |
| 2 | 0.0309 | 6 | -0.703774 |
| 3 | 0.0212 | 2 | -0.701769 |
| 4 | 0.0111 | 3 | -0.706847 |
| 5 | 0.0703 | 6 | -0.701530 |
| 6 | 0.0052 | 3 | -0.710367 |
| 7 | 0.1654 | 1 | -0.698012 |
| 8 | 0.0868 | 1 | -0.698199 |
| 9 | 0.0096 | 2 | -0.706233 |
| 10 | 0.0364 | 6 | -0.703083 |

Chosen: candidate 7.