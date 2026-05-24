# Round 7 resolution

## Applied
- R7-F1 DSI regex matched `AI-RES-dsi-bootstrap` — each `/AI-RES` in the audit regex is now bounded with `(/|[[:space:]]|$|`|")` to ensure it terminates on a real word/path separator; `AI-RES-Stampede3` (GitHub URL) kept as an explicit alternation pattern.
- R7-F2 Phase 10a fail-open — added `set -euo pipefail` to 10a; `bash -n` failures now `exit 1`; stale-path grep replaced with `if git grep ...; then exit 1; fi`.
- R7-F3a preflight gate now exempts `docs/codex_reviews/` (review artifacts intentionally preserve old paths).
- R7-F3b Phase 5 pathspec excludes `docs/codex_reviews/**` and `docs/2026-05-23_sfno_climate_emulator_migration_plan.md` (the plan itself). Verify-grep and "must be EMPTY" diff statements updated.
- R7-F4 `report.md` normalization broadened — label-pattern regex `(**Run tag:** \`)…(\`)` and same for Checkpoint catch any RUN_TAG form (date-prefixed historical baselines included).

## Rejected
None.

## Contested
None.
