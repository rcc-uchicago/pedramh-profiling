# Round 1 resolution

## Applied
- P1 diagnostic_transform chain misstated — §1 now cites line 348-349 forward call with file:line evidence
- P1 upstream paths wrong — §3.1/§3.2 now point at `artifacts/derecho_blocking/source_trees/forecast_modules/PanguPlasim/`
- P1 unit factor inconsistent (×21600=m/6h not mm/6h) — §4.4 keeps own in native m/s, no conflicting label
- P1 ACC over-broad — §4.2 suppresses ACC under the same rule as RMSE with explicit climatology-coherence rationale
- P2 CLI flag wording — §5.1 now uses `--pr6h-unit-align` consistently; `--mode` references removed
- P2 smoke clobber risk — §6 smoke uses `--report-out /tmp/...`

## Contested
- P0 scalar conversion cannot recover RMSE/ACC — user: accepted suppression (option 1 of 3); plan §§2/4/5/6/8/9/10 rewritten to drop conversion path and adopt suppression with banner
