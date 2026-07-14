# Round 4 resolution

## Applied
- P1 _load_benchmark not track-scoped — §5.3 now specifies signature extension `_load_benchmark(bench_root, *, track, pr6h_unit_align)` and gates the suppression-mention only on `suppress+own`; existing wording preserved in `5410`/`none` modes (with stale "line plots" reference dropped)
- P2 metadata key misstated — §6 regression-test fixture now specifies exact shape `{"coords": {"channel": [...]}}` per `_eval_utils.resolve_channel_names`; cross-references existing `_write_metadata_json` helper in `test_render_eval_report_warmstart.py`
- P2 Phase-1 validation conflict — second-IC ratio check moved from §7 risk row into §3.3 Phase 1 as belt-and-suspenders; §7 row simplified to point at §3.3
- P2 dirty-branch overlap stale — §7 row updated to note commits `99d0180`/`e3beb57` did touch the renderer and to require a `git diff main` inspection before patching
