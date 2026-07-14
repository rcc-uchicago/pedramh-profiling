# Round 1 resolution

## Applied
- F1 DSI sed exclusion — Phase 5 pathspec excludes `src/sfno_training/*.dsi*.slurm` and `docs/dsi_*.md` from both passes.
- F2 wrong artifact names — Phase 10b targets `scores/nwp_scorecard_summary.csv` + `report.md`; `match_complete()` selects baseline by artifact presence, not `head -1`.
- F3 untracked-file coverage — Phase 5 adds pass 2 via `git ls-files --others --exclude-standard` with `file(1)` mime gate, excluding `external/**` and `makani-src/**`.
- F4 run-dir config.json paths (contested → user accepted) — new Phase 5.5 sed-rewrites `config.json`/`provenance.txt`/`*.log`/`*.yaml` under `$SCRATCH/.../runs` and `$WORK2/.../results` after tar backup.
- F5 collision guards + rollback symlink unlink — Phase 1/2 add `test ! -e dst` before each `mv`; rollback unlinks transitional symlinks first.
- F6 dead SCORE_ONLY_K — removed from Phase 10b; default `--nwp-K=56` already matches baseline.
- Pre-flight `command -v gh` + `gh auth status` checks added.

## Contested
- F4 run-dir-artifacts rewrite — user: accepted (sed-rewrite config.json files; new Phase 5.5).
