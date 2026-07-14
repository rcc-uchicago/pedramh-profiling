# Round 3 resolution

## Applied
- R3-F1 stash still removes added files — pre-flight step 3 now requires commit only; no stash alternative. Adds explicit untracked-files-with-`AI-RES` ABORT gate.
- R3-F2 group conda env not transparent — new Phase 4b: surgical sed-fix of `$WORK2/.../envs/group_pangu_sfno_v2/{bin,conda-meta,etc/conda,lib/**/*.pth}`; verification with transitional symlink temporarily removed.
- R3-F3 DSI_SCRATCH missing — added to Phase 6 PRESERVE table and the post-audit grep allowlist.
- R3-F4 ~/AI-RES not covered — added `s|~/AI-RES|~/projects/SFNO_Climate_Emulator|g` to `SED_EXPR`.
- R3-F5 earth2studio conditional precedence — replaced `[ ] || [ ] && pip` with `if [ ] || [ ]; then pip; fi`.
- R3-F6 NEW_OUT_ROOT lost across shells — Phase 10b appends `NEW_OUT_ROOT` and `NEW_RUN_TAG` to `/tmp/migration_baseline.txt` at submit time; diff block sources them and aborts if NEW_OUT_ROOT missing.
- (self-audit) Cross-shell state persistence class swept; all transient vars now persisted to `/tmp/migration_*.txt` sidecars.
- (self-audit) Path-variant coverage class swept; tilde added, no other variants found.
- Estimated-time and Critical-files sections updated to reference Phase 4b.

## Rejected
None.

## Contested
None.
