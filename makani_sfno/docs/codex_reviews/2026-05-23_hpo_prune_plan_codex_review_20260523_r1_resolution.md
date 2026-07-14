# Round 1 resolution

## Applied

- P1 familyless-evals-need-explicit-listing — added §3 G0 with all 10 dirs + `FAMILYLESS_EVAL_PRUNE_ALLOWLIST` constant; new familyless evals default UNCLASSIFIED_PROTECT.
- P1 eval-schema-mismatch — §4.2 rewritten: leads `6,24,72,120,240,336`, channels `tas, pr_6h, zg500, ua5, ta5`, metrics `rmse, acc` (dropped `mae`, `t850/z500/u500/v500`, `168/360`).
- P1 slurm-matching-underspecified — §5 step 1 reworded to "queue must be empty for $USER" (matches `_active_slurm_jobs` behavior); cites generic job-name limitation.
- P1 train-ckpt-path-pseudocode-wrong — §5 step 3 rewritten: manifest stores exact resolved path, script calls `shutil.rmtree(Path(row["path"]))` directly.
- P1 ema-vs-raw-not-strong-enough — §4.2 now lists `best_val_loss`, `best_val_loss_ema`, `best_val_loss_ema_epoch` explicitly + cites `feedback_ema_is_canonical_ckpt`.
- P2 training-log-parse-too-schematic — §4.2 prose rewritten to describe multi-line `Epoch N summary:` block + actual column list (no `lr`).
- P2 stale-totals — §3 totals updated to 26 PRUNE training runs + 32 PRUNE evals + ~640 GB combined.

## Rejected

(none)

## Contested

(none)
