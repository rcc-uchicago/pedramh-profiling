# Round 8 resolution

## Applied
- R8-F1a Pre-flight gate was unrealistic — ~100 untracked text files have AI-RES content (verified via `git ls-files --others --exclude-standard | xargs grep -l AI-RES`). Replaced commit-required gate with **snapshot-and-review** (`/tmp/migration_untracked_active_with_airrs.txt`); Phase 5 pass 2's content-driven sed handles them in place without requiring tracking changes.
- R8-F1b Phase 5 pass 2 pathspec exempts `docs/hpo_distill/**`, `docs/run_log/**`, `analysis_outputs/**` (historical dirs that should NOT be sed-ed).
- R8-F1c Phase 6 skill count corrected (3 → 4: `plasim-makani-packager`, `plasim-postprocess`, `sfno-training`, `train-sfno-hpo`).
- R8-F2 `submit_eval.sh` rc-capture bug — Phase 10b adds a defensive `squeue -u $USER -h | wc -l` check immediately after `bash scripts/submit_eval.sh` that aborts if fewer than 4 jobs queued (would catch prelude silent-fail).
- R8-F3 report.md normalization can't catch all legitimate semantic drift — demoted to ADVISORY ONLY. CSV diff is the load-bearing acceptance gate; explicit "do NOT block Phase 11 on report.md" added.

## Rejected
None.

## Contested
None.

## Note
This loop has now spent 8 rounds × ~280k tokens ≈ 2.2M Codex tokens. Findings remain real but increasingly niche edge cases. Surfacing to user before firing round 9.
