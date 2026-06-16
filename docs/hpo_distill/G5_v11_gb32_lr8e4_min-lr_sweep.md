# G5 — v11_gb32_lr8e4 min-LR sweep

**Hypothesis.** Min learning-rate (cosine floor) sweep at the v11_gb32_lr8e4 winner.

**Outcome.** minlr=1e-5 (not 1e-4) was the better cosine-floor target at lr8e4.

## Runs

| run | verdict | best_val_loss_ema (epoch) | final_epoch | wall_time_h | reason |
|---|---|---|---|---|---|
| `sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e4` | **PRUNE** | 2.430e-03 (ep 50) | 50 | 6.48 | dominated by minlr1e5 |
| `sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5` | **KEEP** | 2.410e-03 (ep 50) | 50 | 6.44 | min-LR sweep winner; new operating point for downstream HPO |

## Eval headline (RMSE @ 24 h / 120 h / 336 h on emulator)

### `20260520_eval-867fead_data-8b395eb_family-sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e4_ckpt-best_ckpt_ema_mp0`

| channel | 24h | 120h | 336h |
|---|---|---|---|
| `tas` | 0.6793 ± 0.062 | 1.219 ± 0.14 | 2.902 ± 0.41 |
| `pr_6h` | 8.448e-08 ± 5.4e-09 | 1.772e-07 ± 1.9e-08 | 3.246e-07 ± 2.6e-08 |
| `zg500` | 4.329 ± 0.27 | 17.53 ± 2.4 | 68.19 ± 8.9 |
| `ua5` | 0.8079 ± 0.04 | 2.61 ± 0.24 | 7.688 ± 0.72 |
| `ta5` | 0.5344 ± 0.022 | 1.046 ± 0.081 | 2.765 ± 0.29 |

### `_INVALID_v10data_20260520_eval-867fead_data-e3c934b_family-sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e4_ckpt-best_ckpt_ema_mp0`

| channel | 24h | 120h | 336h |
|---|---|---|---|
| `tas` | 2.333 ± 0.54 | 3.451 ± 0.91 | 4.522 ± 1.1 |
| `pr_6h` | 8.658e-08 ± 5.5e-09 | 1.813e-07 ± 1.8e-08 | 3.288e-07 ± 2.7e-08 |
| `zg500` | 4.729 ± 0.38 | 19.61 ± 2.5 | 68.16 ± 8.4 |
| `ua5` | 0.8274 ± 0.042 | 2.789 ± 0.26 | 7.738 ± 0.68 |
| `ta5` | 0.5489 ± 0.025 | 1.11 ± 0.081 | 2.802 ± 0.27 |

### `20260520_eval-867fead_data-8b395eb_family-sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_ckpt-best_ckpt_ema_mp0`

| channel | 24h | 120h | 336h |
|---|---|---|---|
| `tas` | 0.6697 ± 0.058 | 1.189 ± 0.13 | 2.836 ± 0.41 |
| `pr_6h` | 8.433e-08 ± 5.2e-09 | 1.753e-07 ± 1.8e-08 | 3.228e-07 ± 2.4e-08 |
| `zg500` | 4.379 ± 0.29 | 17.77 ± 2.2 | 68.84 ± 9.5 |
| `ua5` | 0.8064 ± 0.037 | 2.603 ± 0.21 | 7.733 ± 0.78 |
| `ta5` | 0.5291 ± 0.022 | 1.044 ± 0.067 | 2.786 ± 0.32 |

### `_INVALID_v10data_20260520_eval-867fead_data-e3c934b_family-sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_ckpt-best_ckpt_ema_mp0`

| channel | 24h | 120h | 336h |
|---|---|---|---|
| `tas` | 2.26 ± 0.53 | 3.208 ± 0.8 | 4.305 ± 0.98 |
| `pr_6h` | 8.565e-08 ± 5.2e-09 | 1.777e-07 ± 1.7e-08 | 3.245e-07 ± 2.5e-08 |
| `zg500` | 4.788 ± 0.49 | 19.54 ± 2.2 | 69.37 ± 8.5 |
| `ua5` | 0.8219 ± 0.038 | 2.763 ± 0.21 | 7.812 ± 0.7 |
| `ta5` | 0.5376 ± 0.022 | 1.103 ± 0.065 | 2.861 ± 0.3 |

