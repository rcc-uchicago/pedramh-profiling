# G6 — v11_gb32_lr8e4 noise sweep (first round)

**Hypothesis.** Input-noise σ sweep at v11_gb32_lr8e4 (default minlr).

**Outcome.** Baseline σ=0.05 is the operating point; σ=0.020 and σ=0.035 are both worse on val. Confirms `project_v11_noise_sweep_result`.

## Runs

| run | verdict | best_val_loss_ema (epoch) | final_epoch | wall_time_h | reason |
|---|---|---|---|---|---|
| `sfno_zgplev_group_clone_v11_gb32_lr8e4_noise0p020` | **PRUNE** | 1.909e-03 (ep 50) | 50 | 6.49 | dominated by baseline noise=0.05 (per project_v11_noise_sweep_result) |
| `sfno_zgplev_group_clone_v11_gb32_lr8e4_noise0p035` | **PRUNE** | 2.181e-03 (ep 50) | 50 | 6.48 | dominated by baseline noise=0.05 |

## Eval headline (RMSE @ 24 h / 120 h / 336 h on emulator)

### `20260518_eval-867fead_data-e3c934b_family-sfno_zgplev_group_clone_v11_gb32_lr8e4_noise0p020_ckpt-best_ckpt_ema_mp0`

| channel | 24h | 120h | 336h |
|---|---|---|---|
| `tas` | 1.956 ± 0.44 | 3.039 ± 0.83 | 4.194 ± 1 |
| `pr_6h` | 8.165e-08 ± 5.3e-09 | 1.762e-07 ± 1.8e-08 | 3.248e-07 ± 2.5e-08 |
| `zg500` | 4.596 ± 0.36 | 19.29 ± 2.6 | 67.92 ± 8.3 |
| `ua5` | 0.7286 ± 0.042 | 2.731 ± 0.28 | 7.73 ± 0.73 |
| `ta5` | 0.4985 ± 0.022 | 1.115 ± 0.089 | 2.823 ± 0.29 |

### `20260521_eval-867fead_data-8b395eb_family-sfno_zgplev_group_clone_v11_gb32_lr8e4_noise0p020_ckpt-best_ckpt_ema_mp0`

| channel | 24h | 120h | 336h |
|---|---|---|---|
| `tas` | 0.6694 ± 0.059 | 1.24 ± 0.13 | 2.871 ± 0.37 |
| `pr_6h` | 7.972e-08 ± 5.5e-09 | 1.708e-07 ± 1.7e-08 | 3.199e-07 ± 2.4e-08 |
| `zg500` | 4.224 ± 0.27 | 17.24 ± 2.3 | 67.39 ± 9.6 |
| `ua5` | 0.7101 ± 0.04 | 2.562 ± 0.23 | 7.708 ± 0.79 |
| `ta5` | 0.486 ± 0.021 | 1.041 ± 0.081 | 2.774 ± 0.32 |

### `20260518_eval-867fead_data-e3c934b_family-sfno_zgplev_group_clone_v11_gb32_lr8e4_noise0p035_ckpt-best_ckpt_ema_mp0`

| channel | 24h | 120h | 336h |
|---|---|---|---|
| `tas` | 2.526 ± 0.62 | 3.563 ± 0.93 | 4.582 ± 1.1 |
| `pr_6h` | 8.472e-08 ± 5.4e-09 | 1.789e-07 ± 1.6e-08 | 3.282e-07 ± 2.6e-08 |
| `zg500` | 4.682 ± 0.43 | 20.33 ± 2.5 | 70.21 ± 8.4 |
| `ua5` | 0.78 ± 0.04 | 2.822 ± 0.28 | 7.949 ± 0.66 |
| `ta5` | 0.5241 ± 0.022 | 1.124 ± 0.082 | 2.893 ± 0.28 |

### `20260521_eval-867fead_data-8b395eb_family-sfno_zgplev_group_clone_v11_gb32_lr8e4_noise0p035_ckpt-best_ckpt_ema_mp0`

| channel | 24h | 120h | 336h |
|---|---|---|---|
| `tas` | 0.6646 ± 0.059 | 1.201 ± 0.12 | 2.835 ± 0.41 |
| `pr_6h` | 8.255e-08 ± 5.5e-09 | 1.735e-07 ± 1.5e-08 | 3.208e-07 ± 2.5e-08 |
| `zg500` | 4.241 ± 0.31 | 17.52 ± 2.5 | 69.3 ± 8.7 |
| `ua5` | 0.757 ± 0.038 | 2.594 ± 0.26 | 7.809 ± 0.71 |
| `ta5` | 0.5096 ± 0.022 | 1.039 ± 0.074 | 2.822 ± 0.3 |

