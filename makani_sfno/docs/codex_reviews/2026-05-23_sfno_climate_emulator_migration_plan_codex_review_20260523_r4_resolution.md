# Round 4 resolution

## Applied
- R4-F1 Phase 4b conda env stale paths — extended to `conda-meta/history`, `lib/pkgconfig/*.pc`, `lib/*Config.sh`, `lib/python*/_sysconfigdata*.py`. Verification now does `grep -rl '/AI-RES/' $GENV` and `python -c "import sysconfig; ..."` against env with transitional symlink temporarily down.
- R4-F2 stale symlinks in run/results — Phase 5.5 step 4 retargets all `find -type l -lname '*AI-RES*'` under `$NEW_SC/runs` and `$NEW_W2/results` (verified: 223 such symlinks exist today, mostly `training_checkpoints` and `inference/ic_nc/*.nc`).
- R4-F3 `git add -A` would stage vendored — Phase 11 now uses `git add -u` plus explicit add of the 3 active untracked files; vendored `external/`/`makani-src/` not staged.
- R4-F4 missing JSON/txt — Phase 5.5 mutation+backup set broadened to all `*.json` and `*.txt` under runs/results.
- R4-F5 DSI grep too strict — allowlist now also exempts bare `AI-RES/`, `cd AI-RES`, trailing/space-bounded `AI-RES` in prose; remaining lines written to `/tmp/dsi_doc_audit_remaining.txt` for manual classification rather than empty-grep gating.
- (self-audit) Coverage-completeness class swept yet again; conda-env file types audited; runs/results file types broadened from `{config.json, provenance.txt, *.log, *.yaml}` to `{*.json, *.txt, *.log, *.yaml}`; symlinks added as a new artifact class.

## Rejected
None.

## Contested
None.
