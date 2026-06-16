# Round 1 resolution

## Applied
- P1-a eval RUN_DIR — rewrote §5 to use inner `/<config>/0` path per submit_eval_prelude.sh:26
- P1-b v11 data overrides — added TEST_HOLDOUT/TRAIN_DIR/PACKAGER_TEST_SRC v11 exports per submit_beta1_chains.sh:16
- P1-c collision semantics — clarified §4 partial-dir overwrite hole + mtime resume mechanism
- P2-a pr_6h cross-track caveat — added §5 unit-mismatch note referencing 2026-05-14 ticket
- P2-b epoch-75 inspection source — added out.log-only note to §2.2 (Makani keeps last 3 ckpts)
- P2-c stale doc ref — 2026-05-12 → 2026-05-14 v11_clip_warmstart_continuation_plan
- P2-d SLURM hygiene — added §4 directive to fix stale GB=8 comment on copy + epochs75 wallclock comment
