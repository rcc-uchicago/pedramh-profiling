# Round 5 resolution

## Applied
- R5-F1 Phase 4b ordering — verification now activates the env directly by absolute prefix (`conda activate "$GENV"`) instead of sourcing the still-stale `env_activate.sh`. Wrapper-script end-to-end check deferred to Phase 10a.
- R5-F2 DSI grep over-allowlisted — replaced with positive-match audit: `grep -nE` for must-rewrite patterns (`/scratch/.../AI-RES`, `/work2/.../AI-RES`, `$SCRATCH/AI-RES`, `$WORK2/AI-RES`, `feynmanliu214/AI-RES-Stampede3`); hits go to `/tmp/dsi_doc_audit_to_rewrite.txt` for manual edit.
- R5-F3 symlink targets not backed up — Phase 5.5 now manifests `find -type l -lname '*AI-RES*' -printf '%p\t%l\n'` to `$BACKUP/symlink_manifest.tsv` before retargeting; rollback restores from manifest.
- R5-F4 *.md missed — Phase 5.5 mutation set now discovered by content via `grep -rIl '/AI-RES'` (text files only — `-I` skips binary), not by extension list. Includes `*.md`, `*.json`, `*.txt`, `*.log`, `*.yaml`, and anything else with the pattern.
- (self-audit, TERMINATING the class) Phase 5.5 discovery is now content-based, not extension-based. The "coverage-completeness" class that repeated R1→R3→R4→R5 cannot recur for this phase: any text file with `/AI-RES/` will be found and rewritten regardless of extension. Phase 4b retains extension-based discovery (necessary because conda env has many files, most don't have old paths) but is gated by a final `grep -rl` check that hard-aborts if anything is left.

## Rejected
None.

## Contested
None.
