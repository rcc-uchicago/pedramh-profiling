# G9 — v11_gb32_lr8e4_minlr1e5 epochs extension

**Hypothesis.** Epochs-extension probe (50 → 75) at the cumulative winner.

**Outcome.** Epochs=75 extension at the cumulative winner is the current candidate operating point; not yet superseded.

## Runs

| run | verdict | best_val_loss_ema (epoch) | final_epoch | wall_time_h | reason |
|---|---|---|---|---|---|
| `sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_epochs75` | **KEEP** | 2.322e-03 (ep 71) | 75 | 9.69 | current candidate operating point; not yet superseded |

## Eval headline (RMSE @ 24 h / 120 h / 336 h on emulator)

### `20260522_eval-867fead_data-8b395eb_family-sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_epochs75_ckpt-best_ckpt_ema_mp0`

| channel | 24h | 120h | 336h |
|---|---|---|---|
| `tas` | 0.6396 ± 0.055 | 1.154 ± 0.12 | 2.784 ± 0.41 |
| `pr_6h` | 8.349e-08 ± 5.3e-09 | 1.727e-07 ± 1.8e-08 | 3.206e-07 ± 2.6e-08 |
| `zg500` | 4.083 ± 0.25 | 16.8 ± 2 | 67.15 ± 8.8 |
| `ua5` | 0.7704 ± 0.035 | 2.501 ± 0.2 | 7.618 ± 0.75 |
| `ta5` | 0.5083 ± 0.021 | 0.9988 ± 0.06 | 2.743 ± 0.32 |

