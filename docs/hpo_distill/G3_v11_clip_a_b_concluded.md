# G3 — v11 clip A/B (concluded)

**Hypothesis.** Restore gradient-norm clipping (max_grad_norm=32) on top of v11 to see if it stabilises long-lead loss.

**Outcome.** v11_clip restored gradient-norm clipping; A/B concluded and the line moved on to the v11_gb32 LR sweep. Live record in `project_v11_clip_experiment` (2026-05-12).

## Runs

| run | verdict | best_val_loss_ema (epoch) | final_epoch | wall_time_h | reason |
|---|---|---|---|---|---|
| `sfno_zgplev_group_clone_v11` | **PRUNE** | 3.161e-03 (ep 48) | 50 | 15.77 | old; v11 baseline obsoleted by v11_clip + v11_gb32 lineage |
| `sfno_zgplev_group_clone_v11_clip` | **PRUNE** | 3.140e-03 (ep 50) | 50 | 17.03 | old; clip A/B concluded |
| `sfno_zgplev_group_clone_v11_clip_warmstart` | **PRUNE** | 2.841e-03 (ep 50) | 50 | 16.71 | old; warmstart variant of obsoleted v11_clip |

## Eval headline (RMSE @ 24 h / 120 h / 336 h on emulator)

### `20260513_eval-8b395eb_data-e3c934b_family-sfno_zgplev_group_clone_v11_clip`

| channel | 24h | 120h | 336h |
|---|---|---|---|
| `tas` | 2.009 ± 0.45 | 3.099 ± 0.85 | 4.221 ± 0.94 |
| `pr_6h` | 9.591e-08 ± 6.3e-09 | 2.014e-07 ± 2.2e-08 | 3.317e-07 ± 2.3e-08 |
| `zg500` | 6.126 ± 0.59 | 22.59 ± 2.8 | 73.22 ± 9.5 |
| `ua5` | 0.993 ± 0.052 | 3.193 ± 0.27 | 8.153 ± 0.74 |
| `ta5` | 0.6453 ± 0.029 | 1.259 ± 0.084 | 2.954 ± 0.31 |

### `20260513_eval-8b395eb_data-e3c934b_family-sfno_zgplev_group_clone_v11_clip_ckpt-best_ckpt_ema_mp0`

| channel | 24h | 120h | 336h |
|---|---|---|---|
| `tas` | 1.999 ± 0.45 | 3.089 ± 0.85 | 4.243 ± 0.96 |
| `pr_6h` | 9.585e-08 ± 6.3e-09 | 2.014e-07 ± 2.1e-08 | 3.326e-07 ± 2.3e-08 |
| `zg500` | 6.225 ± 0.61 | 23.15 ± 2.8 | 74.43 ± 9.7 |
| `ua5` | 0.9945 ± 0.052 | 3.223 ± 0.27 | 8.251 ± 0.73 |
| `ta5` | 0.6467 ± 0.029 | 1.282 ± 0.086 | 3.002 ± 0.32 |

### `20260515_eval-867fead_data-8b395eb_train-8b395eb_family-sfno_zgplev_group_clone_v11_clip_warmstart_ckpt-best_ckpt_ema_mp0`

| channel | 24h | 120h | 336h |
|---|---|---|---|
| `tas` | 0.7392 ± 0.065 | 1.296 ± 0.14 | 2.917 ± 0.42 |
| `pr_6h` | 9.114e-08 ± 5.5e-09 | 1.871e-07 ± 1.8e-08 | 3.286e-07 ± 2.4e-08 |
| `zg500` | 4.832 ± 0.31 | 19.28 ± 2.6 | 71.36 ± 9.8 |
| `ua5` | 0.8802 ± 0.043 | 2.83 ± 0.24 | 7.996 ± 0.67 |
| `ta5` | 0.578 ± 0.025 | 1.114 ± 0.07 | 2.858 ± 0.33 |

