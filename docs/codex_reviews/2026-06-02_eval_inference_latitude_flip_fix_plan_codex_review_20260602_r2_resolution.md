# Round 2 resolution

## Applied
- P1-a `--limit-ics 1` not 1 IC total — §4 step 3 + Goal 2: smoke is now `--limit-files 1 --limit-ics 1`; baselines pinned to same single (file,IC) subset.
- P1-b climatology recompute on fresh root — §4 preflight: reuse existing same-family --clim-nc (or symlink); rebuild only if intentional.
- P2-a 5410/group writer source overbroad — §3.2: replaced blanket "h5-sourced/correct" with per-writer verified sources (converters=h5, group_emulator=params.lat/yaml, score_adapter=raw 5410 NC); all marked out-of-scope.
- P2-b shared helper must exclude compute_climatology — §6 Q4: helper limited to degree-coordinate writers; compute_climatology.py explicitly excluded (writes integer np.arange(H)).
- (self-audit) writer-lat-source class — swept all to_netcdf writers; stopped making per-writer correctness claims, scoped them out with verified sources instead of patching one at a time.

## Rejected
(none)

## Contested
(none — all correctness/clarity/compute-safety refinements; no scope/experiment/path/science change)
