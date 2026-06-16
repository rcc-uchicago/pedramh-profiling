# Round 5 resolution

## Applied
- P1 `mode`/`pr6h_unit_align` not threaded into `_render_table` — §5.2 now specifies explicit signature extension (`pr6h_unit_align="suppress"`) and call-site update at `:425` to pass `pr6h_unit_align=args.pr6h_unit_align`; `mode` references replaced with `pr6h_unit_align` throughout
- P2 Phase-1 validation conflict still present — §3.5 "does not attempt a sanity-check ratio test" line removed; §6 "No Phase-1 sanity check" removed; cross-reference between §3.3 and §6 added so the second-IC check is unambiguously a Phase-1 item, not a §6 test
- P2 second-IC underspecified — §3.3 now explicitly scopes the second-IC check to within-5410 only (no own-side cross-run matching), citing `nwp_ic_offsets` at `rollout_driver.py:309` vs `IC_OFFSETS` at `infer_sfno5410_blocking_h100_packed.py:45` as the reason cross-run IC matching is fragile
- P2 byte-for-byte conflicts with banner change — §5.3 banner update now restricted to `suppress+own`; `none`/`5410` modes keep `_load_benchmark` banner unchanged (line-plot pedantry deferred); §6 byte-for-byte scope clarified to scorecard table only
- P2 skill update misses env-table — §8 Phase 6 expanded: update `BENCHMARK_5410_OUT_ROOT` row at `SKILL.md:71`, add new `PR6H_UNIT_ALIGN` row, then rewrite the `:224` paragraph; §7 dirty-branch checklist extended to include `SKILL.md`
