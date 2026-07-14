# Round 3 resolution

## Applied

- P1 plan-§5-still-says-report.md-OR-scores/ — updated §5 wording to match code (require report.md OR specific score artifacts; empty dir not sufficient).
- P1 5410-pr_6h-rows-still-in-eval_scores — `parse_eval_report` now drops `(model="5410 benchmark", channel="pr_6h")`; §4.2 documents the filter.
- P1 _INVALID_-evals-outside-§3 — added §3 G-INVALID subsection documenting the prefix-override rule (PRUNE always wins over family verdict); enumerates both currently present `_INVALID_v10data_*` dirs.
- P1 apply-not-all-manifest-precheck — refactored `cmd_prune` into Phase A (`_validate_manifest_row`, zero-side-effect) + Phase B (delete); --apply aborts with exit 3 if any row fails validation. §5 rewritten to match.
- P2 missing-protect-roots — added `sfno_short_ddp` and `sfno_short_ddp_sweep` to `PROTECTED_TRAIN_DIRS` (defensive even if absent today).
- P2 legacy-leads-not-noted — §4.2 now flags that CSV fallback may surface additional legacy leads (e.g. 360h) and channels (e.g. hus1).
- P2 dry-run-audit-side-effect — audit JSONL writes are now gated behind `--apply`; verified by `ls -la prune_audit.jsonl` after dry-run returning ENOENT.

## Applied (self-audit)

- §6 prose drift — updated subcommand listing to mention `--force-active` and the two-phase exit-3-on-validation-fail semantics.
- §7 prose drift — safety summary checklist updated: dry-run no side effects, two-phase validation, squeue fail-closed, audit only on --apply.

## Rejected

(none)

## Contested

(none)
