# G2 — Early group_clone exploration (pre-v11 / non-noise)

**Hypothesis.** Early baseline runs cloning the group-emulator config; explored input-noise on/off and GB.

**Outcome.** Baseline group-clone superseded by v11 lineage. `nonoise` was a clear loser, confirming input-noise is load-bearing (`feedback_input_noise_is_load_bearing`). GB32 group-clone dominated by both the GB4 winner and the v11_gb32 LR-sweep winner.

## Runs

| run | verdict | best_val_loss_ema (epoch) | final_epoch | wall_time_h | reason |
|---|---|---|---|---|---|
| `sfno_zgplev_group_clone` | **PRUNE** | 3.115e-03 (ep 50) | 50 | 0.95 | old; superseded by v11 lineage |
| `sfno_zgplev_group_clone_gb32` | **PRUNE** | 4.362e-03 (ep 48) | 50 | 6.19 | dominated by GB4 (G1) and superseded by v11_gb32 (G4) |
| `sfno_zgplev_group_clone_nonoise` | **PRUNE** | 2.346e-03 (ep 49) | 50 | 0.34 | old; nonoise is known loser per feedback_input_noise_is_load_bearing |
| `sfno_zgplev_group_clone_smoke` | **PRUNE** | 1.947e+00 (ep 1) | 1 | 0.01 | old smoke run; no scientific record needed |
| `sfno_zgplev_group_clone_v10_warmstart` | **KEEP** | 2.786e-03 (ep 50) | 50 | 15.84 | live v10 warm-start line of inquiry; not obsoleted by v11 |

## Eval headline (RMSE @ 24 h / 120 h / 336 h on emulator)

### `20260515_eval-867fead_data-e3c934b_family-sfno_zgplev_group_clone_gb32_ckpt-best_ckpt_ema_mp0`

| channel | 24h | 120h | 336h |
|---|---|---|---|
| `tas` | 1.045 ± 0.096 | 1.903 ± 0.23 | 3.567 ± 0.49 |
| `pr_6h` | 1.068e-07 ± 6.6e-09 | 2.36e-07 ± 2.4e-08 | 3.482e-07 ± 2.7e-08 |
| `zg500` | 7.584 ± 0.46 | 27.33 ± 3.3 | 77.09 ± 9.8 |
| `ua5` | 1.25 ± 0.074 | 3.857 ± 0.31 | 8.595 ± 0.72 |
| `ta5` | 0.7991 ± 0.036 | 1.463 ± 0.092 | 3.106 ± 0.32 |

### `20260515_eval-867fead_data-e3c934b_family-sfno_zgplev_group_clone_v10_warmstart_ckpt-best_ckpt_ema_mp0`

| channel | 24h | 120h | 336h |
|---|---|---|---|
| `tas` | 0.7781 ± 0.069 | 1.452 ± 0.18 | 3.214 ± 0.44 |
| `pr_6h` | 8.972e-08 ± 5.5e-09 | 1.863e-07 ± 1.9e-08 | 3.235e-07 ± 2.3e-08 |
| `zg500` | 4.742 ± 0.3 | 19.24 ± 2.5 | 69.08 ± 9.8 |
| `ua5` | 0.8753 ± 0.042 | 2.836 ± 0.27 | 7.84 ± 0.72 |
| `ta5` | 0.5783 ± 0.024 | 1.12 ± 0.078 | 2.826 ± 0.33 |

