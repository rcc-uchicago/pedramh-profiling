# Round 2 resolution

## Applied
- R2-F1 SED_EXPR scoping — exported in parent shell + passed as `$1` positional to `sh -c`; verify block now `exit 1`s on stale paths.
- R2-F1 (related) — pre-flight step 3 no longer offers bare stash; must commit or `git add` active untracked files before stash.
- R2-F2 Phase 6 DSI-doc contradiction — Phase 6 now has explicit per-pattern audit table (PRESERVE `$HOME/AI-RES`/`$DSI_PROJECT/AI-RES`/`AI-RES-dsi-bootstrap`; REWRITE `/scratch/.../AI-RES`, `/work2/.../AI-RES`, env-var Stampede3 paths, GitHub URL).
- R2-F3 Phase 5.5 backup coverage — backup tar now includes `*.log` and `*.yaml` (same find pattern as the mutation); rollback restores from updated `scratch_run_text.tar.gz` / `work2_results_text.tar.gz`.
- R2-F4 Phase 10b baseline handoff — pre-flight persists `BASELINE_COPY`, `OLD_EVAL_DIR`, `OLD_RUN_DIR` to `/tmp/migration_baseline.txt`; Phase 10b sources that file. No more `head -1`/glob.
- (self-audit) DSI-handling class swept across plan; no other implicit-auto-migration claims found.

## Rejected
None.

## Contested
None.
