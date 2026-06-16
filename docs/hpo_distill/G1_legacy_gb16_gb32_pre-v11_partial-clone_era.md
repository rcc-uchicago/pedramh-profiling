# G1 — Legacy GB16/GB32 (pre-v11 partial-clone era)

**Hypothesis.** Whether GB16 or GB32 own-track training could beat GB4 (the eventual production baseline).

**Outcome.** GB4 wins. GB16 and GB32 (standalone) both worse on the own-track scorecard. Result memorialised in `project_zgplev_gb_decision` (2026-05-09).

## Runs

| run | verdict | best_val_loss_ema (epoch) | final_epoch | wall_time_h | reason |
|---|---|---|---|---|---|
| `sfno_zgplev_full_gb16_lr1e4_20260508` | **PRUNE** | 2.829e-03 (ep 49) | 50 | 8.69 | old + dominated (GB4 won per project_zgplev_gb_decision) |
| `sfno_zgplev_full_gb16_lr2e4_20260509` | **PRUNE** | — |  |  | old + dominated (GB4 won) |
| `sfno_zgplev_full_gb16_lr2e4_20260509_retry1` | **PRUNE** | 2.201e-03 (ep 45) | 50 | 8.60 | old + dominated (GB4 won) |
| `sfno_zgplev_full_gb32_20260508` | **PRUNE** | 2.330e-03 (ep 48) | 50 | 6.40 | old + dominated (GB4 won) |
| `sfno_zgplev_gbhpo40_gb16_lr2_0e-4_20260511` | **PRUNE** | 4.170e-03 (ep 49) | 50 | 3.63 | old + dominated (GB4 won) |
| `sfno_zgplev_gbhpo40_gb16_lr2_83e-4_20260511` | **PRUNE** | 3.805e-03 (ep 50) | 50 | 3.59 | old + dominated (GB4 won) |

## Eval headline (RMSE @ 24 h / 120 h / 336 h on emulator)

### `20260514_eval-8b395eb_data-e3c934b_family-sfno_zgplev_gbhpo40_gb16_lr2_0e-4_20260511_ckpt-best_ckpt_ema_mp0`

| channel | 24h | 120h | 336h |
|---|---|---|---|
| `tas` | 0.9703 ± 0.083 | 1.805 ± 0.21 | 3.547 ± 0.54 |
| `pr_6h` | 1.089e-07 ± 6.9e-09 | 2.203e-07 ± 1.9e-08 | 3.367e-07 ± 2.5e-08 |
| `zg500` | 6.875 ± 0.43 | 25.24 ± 3 | 75.67 ± 9.8 |
| `ua5` | 1.157 ± 0.065 | 3.605 ± 0.31 | 8.391 ± 0.71 |
| `ta5` | 0.7326 ± 0.031 | 1.378 ± 0.094 | 3.053 ± 0.35 |

### `20260514_eval-8b395eb_data-e3c934b_family-sfno_zgplev_gbhpo40_gb16_lr2_83e-4_20260511_ckpt-best_ckpt_ema_mp0`

| channel | 24h | 120h | 336h |
|---|---|---|---|
| `tas` | 0.9073 ± 0.078 | 1.69 ± 0.18 | 3.409 ± 0.51 |
| `pr_6h` | 1.049e-07 ± 6.4e-09 | 2.117e-07 ± 2.2e-08 | 3.347e-07 ± 2.5e-08 |
| `zg500` | 6.082 ± 0.4 | 23.31 ± 3 | 74.4 ± 9.7 |
| `ua5` | 1.074 ± 0.059 | 3.383 ± 0.32 | 8.332 ± 0.73 |
| `ta5` | 0.6818 ± 0.03 | 1.292 ± 0.092 | 2.978 ± 0.32 |

