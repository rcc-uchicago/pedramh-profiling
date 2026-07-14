# G7 — v11_gb32_lr8e4_minlr1e5 β₁ sweep (null result)

**Hypothesis.** Adam β₁ sweep at the v11_gb32_lr8e4_minlr1e5 winner.

**Outcome.** β₁ ∈ {0.9, 0.95, 0.97} produced no meaningful change; baseline 0.9 marginally best. Null result; see `project_v11_beta1_sweep_null` (2026-05-21).

## Runs

| run | verdict | best_val_loss_ema (epoch) | final_epoch | wall_time_h | reason |
|---|---|---|---|---|---|
| `sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_beta1_0p95` | **PRUNE** | 2.432e-03 (ep 50) | 50 | 6.43 | dominated by β₁=0.9 baseline (per project_v11_beta1_sweep_null) |
| `sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_beta1_0p97` | **PRUNE** | 2.425e-03 (ep 50) | 50 | 6.46 | dominated by β₁=0.9 baseline |

## Eval headline (RMSE @ 24 h / 120 h / 336 h on emulator)

### `20260520_eval-867fead_data-8b395eb_family-sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_beta1_0p95_ckpt-best_ckpt_ema_mp0`

| channel | 24h | 120h | 336h |
|---|---|---|---|
| `tas` | 0.6702 ± 0.059 | 1.218 ± 0.14 | 2.837 ± 0.38 |
| `pr_6h` | 8.501e-08 ± 5.7e-09 | 1.792e-07 ± 1.8e-08 | 3.218e-07 ± 2.2e-08 |
| `zg500` | 4.393 ± 0.27 | 17.85 ± 2.3 | 69.11 ± 10 |
| `ua5` | 0.8087 ± 0.037 | 2.657 ± 0.24 | 7.771 ± 0.8 |
| `ta5` | 0.5336 ± 0.022 | 1.054 ± 0.068 | 2.83 ± 0.32 |

### `20260520_eval-867fead_data-8b395eb_family-sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_beta1_0p97_ckpt-best_ckpt_ema_mp0`

| channel | 24h | 120h | 336h |
|---|---|---|---|
| `tas` | 0.6785 ± 0.061 | 1.212 ± 0.13 | 2.894 ± 0.39 |
| `pr_6h` | 8.454e-08 ± 5.4e-09 | 1.787e-07 ± 1.6e-08 | 3.252e-07 ± 2.5e-08 |
| `zg500` | 4.404 ± 0.28 | 17.82 ± 2.3 | 68.42 ± 9.8 |
| `ua5` | 0.8039 ± 0.036 | 2.615 ± 0.23 | 7.785 ± 0.78 |
| `ta5` | 0.528 ± 0.02 | 1.04 ± 0.069 | 2.813 ± 0.35 |

