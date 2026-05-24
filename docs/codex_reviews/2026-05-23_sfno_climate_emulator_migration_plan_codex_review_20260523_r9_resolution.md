# Round 9 resolution

## Applied
- R9-F1 packed Derecho env stale paths — new Phase 4c handles `$WORK2/.../artifacts/derecho_runtime/aires_env_20260509/unpacked/` (verified 601 stale paths). Sed covers `bin/`, `conda-meta/`, `etc/`, `lib/pkgconfig/`, `lib/cmake/`, `lib/python*/_sysconfigdata*.py`, `.pth`. Verification grep + `gdal-config --prefix` smoke. Estimated-time updated.
- R9-F2a Phase 5 verify no longer assumes only-3 untracked — now iterates `/tmp/migration_untracked_active_with_airrs.txt` and reports every stale file.
- R9-F2b Phase 11 git add — now lists the inventory explicitly, adds the load-bearing untracked files deliberately (3 from earlier + 2 untracked SKILL.md files Codex flagged in R8), leaves true-WIP/scratch untracked.
- R9-F3 squeue count too broad — replaced with `grep -cE '^\[submit_eval\] (inference|scoring|report|figures)' $SUBMIT_LOG`, which targets the four job-submission log lines `submit_eval.sh` always emits.
- R9-F4 Phase 6 narrative audit scope — `After Phase 6` block now explicitly classifies bare-name historical narrative mentions in `docs/2026-05-04_makani_local_patches.md`, `docs/plasim_expansion_and_adaptor_plan.md` etc. as preserved (historical record); only *path* hits are migration misses.
- R9-suggest-5 stale "Update Pre-flight step 6" block removed (was R5 leftover; Pre-flight already persists BASELINE_COPY).

## Rejected
None.

## Contested
None.

## Note
- Packed env was a NEW coverage class my prior self-audits missed (treated regular conda env and `.venv` as the universe of envs; forgot `artifacts/derecho_runtime/.../unpacked` is a third one). Real P1 from Codex.
- Cumulative cost: ~2.5M tokens.
