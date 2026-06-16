# G8 — v11_gb32_lr8e4_minlr1e5 noise sweep (second round)

**Hypothesis.** Second-round input-noise σ sweep at the v11_gb32_lr8e4_minlr1e5 winner, plus a longer (epochs=75) variant.

**Outcome.** σ=0.070 fails the tas-6h persistence gate (per `project_v11_noise_sweep_result`). σ=0.020-with-epochs75 is too fresh to call (kept-for-review).

## Runs

| run | verdict | best_val_loss_ema (epoch) | final_epoch | wall_time_h | reason |
|---|---|---|---|---|---|
| `sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_noise0p020_epochs75` | **KEEP** | 1.805e-03 (ep 73) | 75 | 9.66 | 1d old, insufficient evidence; review at next prune pass |
| `sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_noise0p070` | **PRUNE** | 2.820e-03 (ep 50) | 50 | 6.46 | dominated (failed tas 6h persistence gate per project_v11_noise_sweep_result) |

## Eval headline (RMSE @ 24 h / 120 h / 336 h on emulator)

### `20260522_eval-867fead_data-8b395eb_family-sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_noise0p020_epochs75_ckpt-best_ckpt_ema_mp0`

| channel | 24h | 120h | 336h |
|---|---|---|---|
| `tas` | — | — | — |
| `pr_6h` | — | — | — |
| `zg500` | — | — | — |
| `ua5` | — | — | — |
| `ta5` | — | — | — |

### `20260522_eval-867fead_data-8b395eb_family-sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_noise0p020_epochs75_ckpt-best_ckpt_ema_mp0_h100retry`

| channel | 24h | 120h | 336h |
|---|---|---|---|
| `tas` | 0.6293 ± 0.056 | 1.198 ± 0.13 | 2.873 ± 0.4 |
| `pr_6h` | 7.758e-08 ± 4.9e-09 | 1.657e-07 ± 1.7e-08 | 3.21e-07 ± 2.7e-08 |
| `zg500` | 3.958 ± 0.27 | 16.11 ± 2.1 | 66.66 ± 9.4 |
| `ua5` | 0.6646 ± 0.033 | 2.432 ± 0.22 | 7.57 ± 0.75 |
| `ta5` | 0.4698 ± 0.021 | 0.9951 ± 0.069 | 2.755 ± 0.3 |

### `20260521_eval-867fead_data-8b395eb_family-sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_noise0p070_ckpt-best_ckpt_ema_mp0`

| channel | 24h | 120h | 336h |
|---|---|---|---|
| `tas` | 0.6988 ± 0.063 | 1.222 ± 0.14 | 2.81 ± 0.4 |
| `pr_6h` | 8.902e-08 ± 5.8e-09 | 1.858e-07 ± 1.9e-08 | 3.248e-07 ± 2.6e-08 |
| `zg500` | 4.563 ± 0.26 | 18.78 ± 2.3 | 69.62 ± 9.1 |
| `ua5` | 0.8785 ± 0.038 | 2.753 ± 0.24 | 7.839 ± 0.74 |
| `ta5` | 0.562 ± 0.023 | 1.079 ± 0.065 | 2.842 ± 0.32 |

### `20260521_eval-867fead_data-e3c934b_family-sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_noise0p070_ckpt-best_ckpt_ema_mp0`

| channel | 24h | 120h | 336h |
|---|---|---|---|
| `tas` | 2.658 ± 0.65 | 3.636 ± 0.98 | 4.624 ± 1.1 |
| `pr_6h` | 9.133e-08 ± 5.7e-09 | 1.913e-07 ± 1.9e-08 | 3.293e-07 ± 2.7e-08 |
| `zg500` | 5.306 ± 0.62 | 21.88 ± 2.8 | 72.38 ± 8.9 |
| `ua5` | 0.9037 ± 0.041 | 2.991 ± 0.27 | 8.025 ± 0.68 |
| `ta5` | 0.5754 ± 0.023 | 1.156 ± 0.072 | 2.956 ± 0.31 |

