# Round 9 resolution

## Applied
- P1 smoke command sources wrong checkpoint key — §6 smoke snippet now sources `CKPT=` (not `CKPT_PATH=`) per `scripts/submit_eval_prelude.sh:190` and `scripts/submit_eval_5410.sh:131,:138`; maps `CKPT` → `--ckpt-path "$CKPT"` with inline comment explaining the rename
- P2 aggregation citation inaccurate — §2 (Why suppression over scalar conversion) reworded: per-IC RMSE/ACC computation at `score_nwp.py:160` (RMSE) and `:172` (ACC), IC-averaging + CSV write at `_summarize` `score_nwp.py:238-253`
- Self-audit (citation correctness class repeats from r8 BYO finding) — fixed two additional citation drifts:
  - §4.1 `render_eval_report.py:178` (per-cell `src.get`) → `:173-174` (the conditional benchmark-row append; suppression intervenes at the append, not the formatted row)
  - §5.2 shared loop citation `:166` (header construction) → `:163` (`for metric in ("rmse", "acc")`)
