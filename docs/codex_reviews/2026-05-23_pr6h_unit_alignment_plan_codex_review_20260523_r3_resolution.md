# Round 3 resolution

## Applied
- P1 guard not track-scoped — §5.2 guard now reads `mode == "suppress" and track == "own" and ch == "pr_6h"`; rationale paragraph added explaining why 5410-track mode must not be suppressed
- P2 stale help/banner — §5.3 expanded to require updating CLI help at `render_eval_report.py:65-70` and `_load_benchmark` banner at `:391-396`; existing `:158-160` disclaimer replaced under suppress+own condition only
- P2 figure-unit wording — §4.4 fixed to say mm/day scaling is for bias maps (not line plots); §5.5 corrected to cite `LINE_PLOT_CHANNELS` at `:33` (the actual filter), distinguishing it from `REPORT_CHANNELS` at `:28` (bias-map iteration only)
