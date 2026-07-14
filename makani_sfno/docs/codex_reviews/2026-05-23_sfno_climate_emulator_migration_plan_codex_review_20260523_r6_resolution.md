# Round 6 resolution

## Applied
- R6-F1 group env wrapper untested — Phase 10a now `source`s `src/sfno_training_group/env_activate.sh`, asserts `SFNO_Climate_Emulator` in `sys.executable`, then deactivates. Also adds `bash -n` for `src/sfno_training_group/slurm/*.slurm`.
- R6-F2 bare `"AI-RES"` Python path component — added two quote-bounded sed patterns (`"AI-RES"`/`'AI-RES'` → `"SFNO_Climate_Emulator"`/`'SFNO_Climate_Emulator'`) to `SED_EXPR`. Catches `scripts/hpo_prune.py:40,41,99,100` and any other Python `Path / "AI-RES"` idioms.
- R6-F3 preflight conflicts with vendored external/ — preflight gate now exempts `external/` and `makani-src/` in addition to `analysis_outputs/`/`docs/2026-`. Documents `external/PanguWeather_stampede3_env.txt` as intentionally stale.
- R6-F4 DSI audit misses `/home1/.../AI-RES` — extended positive-match regex to include the Stampede3 absolute home path.
- R6-F5 report.md raw diff false-positive — replaced with header/path-normalized Python differ (rewrites RUN_TAG token and `/scratch|/work2|/home1` absolute paths to `<PATH>`); raw diff still captured to `/tmp/report_md_raw_diff.txt` for the record.
- (self-audit) sed-pattern-completeness class: quote-bounded patterns added; unquoted-bare `AI-RES` left alone (too many false-positive risks; covered by manual Phase 6 edits if needed).

## Rejected
None.

## Contested
None.
