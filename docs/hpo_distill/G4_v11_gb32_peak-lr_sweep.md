# G4 — v11_gb32 peak-LR sweep

**Hypothesis.** Peak learning-rate sweep on the v11_gb32 base.

**Outcome.** Sweeping peak LR found 8e-4 best so far; ~1e-3 (1.13e-3) degraded performance; 1.6e-3 made the loss itself unstable. Verbatim user note preserved.

## Runs

| run | verdict | best_val_loss_ema (epoch) | final_epoch | wall_time_h | reason |
|---|---|---|---|---|---|
| `sfno_zgplev_group_clone_v11_gb32` | **PRUNE** | 4.333e-03 (ep 48) | 50 | 6.21 | old; baseline (lr=2.83e-4) dominated by lr8e4 |
| `sfno_zgplev_group_clone_v11_gb32_lr1p13e3` | **PRUNE** | 2.389e-03 (ep 49) | 50 | 6.47 | dominated (the "~1e-3 degraded" probe) |
| `sfno_zgplev_group_clone_v11_gb32_lr1p13e3_noise0p035` | **PRUNE** | 3.264e-03 (ep 8) | 8 |  | dominated (1.13e-3 LR loser; noise=0.035 also a loser per G6) |
| `sfno_zgplev_group_clone_v11_gb32_lr1p6e3` | **PRUNE** | 4.961e-03 (ep 5) | 33 |  | dominated (the "1.6e-3 unstable" probe) |
| `sfno_zgplev_group_clone_v11_gb32_lr2p83e4` | **PRUNE** | 3.029e-03 (ep 48) | 50 | 6.19 | old + dominated by lr8e4 |
| `sfno_zgplev_group_clone_v11_gb32_lr4e4` | **PRUNE** | 2.760e-03 (ep 46) | 50 | 5.81 | dominated by lr8e4 |
| `sfno_zgplev_group_clone_v11_gb32_lr5p66e4` | **PRUNE** | 2.590e-03 (ep 48) | 50 | 6.47 | dominated by lr8e4 |
| `sfno_zgplev_group_clone_v11_gb32_lr8e4` | **KEEP** | 2.412e-03 (ep 47) | 50 | 6.47 | sweep winner. User-verbatim: "For gb32, sweeping peak learning rate found 8e-4 best so far; ~1e-3 degraded performance, and 1.6e-3 made the loss itself unstable." |

## Eval headline (RMSE @ 24 h / 120 h / 336 h on emulator)

### `20260515_eval-867fead_data-e3c934b_family-sfno_zgplev_group_clone_v11_gb32_ckpt-best_ckpt_ema_mp0`

| channel | 24h | 120h | 336h |
|---|---|---|---|
| `tas` | 1.017 ± 0.086 | 1.73 ± 0.17 | 3.15 ± 0.39 |
| `pr_6h` | 1.068e-07 ± 6.9e-09 | 2.307e-07 ± 2.3e-08 | 3.463e-07 ± 2.6e-08 |
| `zg500` | 7.559 ± 0.49 | 27 ± 3.4 | 76.56 ± 10 |
| `ua5` | 1.253 ± 0.069 | 3.807 ± 0.31 | 8.569 ± 0.81 |
| `ta5` | 0.7917 ± 0.036 | 1.448 ± 0.098 | 3.072 ± 0.31 |

### `20260517_eval-867fead_data-e3c934b_family-sfno_zgplev_group_clone_v11_gb32_lr1p13e3_ckpt-best_ckpt_ema_mp0`

| channel | 24h | 120h | 336h |
|---|---|---|---|
| `tas` | 2.525 ± 0.63 | 3.532 ± 0.94 | 4.497 ± 1 |
| `pr_6h` | 8.597e-08 ± 5.2e-09 | 1.799e-07 ± 1.8e-08 | 3.215e-07 ± 2.5e-08 |
| `zg500` | 4.901 ± 0.5 | 19.86 ± 2.1 | 69.22 ± 8.2 |
| `ua5` | 0.8197 ± 0.037 | 2.792 ± 0.24 | 7.758 ± 0.68 |
| `ta5` | 0.5362 ± 0.022 | 1.107 ± 0.068 | 2.854 ± 0.28 |

### `20260516_eval-867fead_data-8b395eb_family-sfno_zgplev_group_clone_v11_gb32_lr2p83e4_ckpt-best_ckpt_ema_mp0`

| channel | 24h | 120h | 336h |
|---|---|---|---|
| `tas` | 0.7968 ± 0.068 | 1.38 ± 0.14 | 2.925 ± 0.43 |
| `pr_6h` | 9.328e-08 ± 6.2e-09 | 1.975e-07 ± 2.1e-08 | 3.312e-07 ± 2.5e-08 |
| `zg500` | 5.403 ± 0.32 | 20.71 ± 2.6 | 71.25 ± 9.5 |
| `ua5` | 0.9578 ± 0.046 | 3.017 ± 0.26 | 8.002 ± 0.74 |
| `ta5` | 0.6293 ± 0.025 | 1.188 ± 0.085 | 2.873 ± 0.31 |

### `20260516_eval-867fead_data-8b395eb_family-sfno_zgplev_group_clone_v11_gb32_lr4e4_ckpt-best_ckpt_ema_mp0`

| channel | 24h | 120h | 336h |
|---|---|---|---|
| `tas` | 0.747 ± 0.067 | 1.307 ± 0.14 | 2.898 ± 0.41 |
| `pr_6h` | 8.931e-08 ± 6e-09 | 1.871e-07 ± 1.8e-08 | 3.288e-07 ± 2.5e-08 |
| `zg500` | 4.906 ± 0.31 | 19.32 ± 2.7 | 70.56 ± 9.1 |
| `ua5` | 0.8892 ± 0.043 | 2.849 ± 0.29 | 7.889 ± 0.76 |
| `ta5` | 0.5894 ± 0.026 | 1.121 ± 0.088 | 2.831 ± 0.32 |

### `20260516_eval-867fead_data-8b395eb_family-sfno_zgplev_group_clone_v11_gb32_lr5p66e4_ckpt-best_ckpt_ema_mp0`

| channel | 24h | 120h | 336h |
|---|---|---|---|
| `tas` | 0.7071 ± 0.064 | 1.254 ± 0.14 | 2.846 ± 0.46 |
| `pr_6h` | 8.703e-08 ± 6e-09 | 1.817e-07 ± 1.8e-08 | 3.254e-07 ± 2.5e-08 |
| `zg500` | 4.593 ± 0.29 | 18.4 ± 2.3 | 69.11 ± 9.8 |
| `ua5` | 0.8521 ± 0.042 | 2.73 ± 0.25 | 7.735 ± 0.78 |
| `ta5` | 0.5614 ± 0.024 | 1.077 ± 0.074 | 2.79 ± 0.31 |

### `20260516_eval-867fead_data-8b395eb_family-sfno_zgplev_group_clone_v11_gb32_lr8e4_ckpt-best_ckpt_ema_mp0`

| channel | 24h | 120h | 336h |
|---|---|---|---|
| `tas` | 0.668 ± 0.059 | 1.196 ± 0.14 | 2.86 ± 0.41 |
| `pr_6h` | 8.447e-08 ± 5.5e-09 | 1.771e-07 ± 1.9e-08 | 3.203e-07 ± 2.1e-08 |
| `zg500` | 4.398 ± 0.27 | 17.81 ± 2.1 | 70.16 ± 8.9 |
| `ua5` | 0.8045 ± 0.038 | 2.633 ± 0.23 | 7.835 ± 0.71 |
| `ta5` | 0.5342 ± 0.024 | 1.043 ± 0.067 | 2.81 ± 0.29 |

