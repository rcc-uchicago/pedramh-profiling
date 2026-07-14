**Strengths**
- User-approved intent is explicit: DSI naming stays out of scope, venv rebuild is avoided, generated run artifacts are intentionally rewritten, and `report.md` is advisory.
- R8 applied items are mostly present: historical untracked dirs are excluded from pass 2, skill count is corrected to 4, Phase 10a fails hard, and CSV is the load-bearing eval gate.
- Baseline selection now matches the actual v11 eval layout: the non-`h100retry` result is incomplete, while the `h100retry` result has `scores/nwp_scorecard_summary.csv` and `report.md`.

**Issues**
- **P1 - Packed Derecho 5410 env is moved but never repointed.**  
  Phase 4 fixes `.venv`, Phase 4b fixes the group conda env, and Phase 5.5 rewrites only `runs/` + `results/`. But 5410 production uses the packed Derecho env at `artifacts/derecho_runtime/.../unpacked/bin/python` from [scripts/submit_eval_inference_5410_packed.slurm](/home1/11114/zhixingliu/AI-RES/scripts/submit_eval_inference_5410_packed.slurm:34), and that env has embedded old prefixes such as `CONFIG_PREFIX="/work2/.../AI-RES/..."` in `/work2/11114/zhixingliu/stampede3/AI-RES/artifacts/derecho_runtime/aires_env_20260509/unpacked/bin/gdal-config:4`. After the two-week symlink is removed, 5410 tooling can still resolve stale `AI-RES` internals.

- **P2 - R8-F1 is only partially resolved: untracked active files are rewritten, but final audit/staging still assumes only 3.**  
  The plan acknowledges many active untracked files ([plan](/home1/11114/zhixingliu/AI-RES/docs/2026-05-23_sfno_climate_emulator_migration_plan.md:21)), but Phase 5 only spot-checks three files ([plan](/home1/11114/zhixingliu/AI-RES/docs/2026-05-23_sfno_climate_emulator_migration_plan.md:293)) and Phase 11 adds only those same three ([plan](/home1/11114/zhixingliu/AI-RES/docs/2026-05-23_sfno_climate_emulator_migration_plan.md:639)). Actual load-bearing untracked files include inline eval bodies sourced by SLURM, e.g. [scripts/eval_run_inference_inline.sh](/home1/11114/zhixingliu/AI-RES/scripts/eval_run_inference_inline.sh:12), plus untracked skills like [.claude/skills/eval-sfno-5410/SKILL.md](/home1/11114/zhixingliu/AI-RES/.claude/skills/eval-sfno-5410/SKILL.md:22). The “status should show only tracked files modified and the 3 explicitly-added files” expectation is not true for this repo.

- **P2 - R8-F2 guard is not chain-specific.**  
  The underlying fail-open bug remains in [scripts/submit_eval.sh](/home1/11114/zhixingliu/AI-RES/scripts/submit_eval.sh:37). The added guard counts all user jobs with `squeue -u "$USER"` ([plan](/home1/11114/zhixingliu/AI-RES/docs/2026-05-23_sfno_climate_emulator_migration_plan.md:550)), so unrelated queued jobs can mask “submit_eval queued nothing.” The script already emits four job IDs at [scripts/submit_eval.sh](/home1/11114/zhixingliu/AI-RES/scripts/submit_eval.sh:65), which is a better validation target.

- **P2 - Bare prose audit expectation is still too narrow.**  
  The plan says bare `AI-RES` prose is handled in Phase 6 ([plan](/home1/11114/zhixingliu/AI-RES/docs/2026-05-23_sfno_climate_emulator_migration_plan.md:282)), but Phase 6 only lists README/skills/DSI docs. Tracked non-DSI docs still contain bare narrative references that sed will not change, e.g. [docs/2026-05-04_makani_local_patches.md](/home1/11114/zhixingliu/AI-RES/docs/2026-05-04_makani_local_patches.md:164) and [docs/plasim_expansion_and_adaptor_plan.md](/home1/11114/zhixingliu/AI-RES/docs/plasim_expansion_and_adaptor_plan.md:118). The final expected-remainder statement at [plan](/home1/11114/zhixingliu/AI-RES/docs/2026-05-23_sfno_climate_emulator_migration_plan.md:408) will produce extra hits.

**Suggested Edits**
- Add a Phase 4c for the packed Derecho env under `$WORK2/.../artifacts/derecho_runtime/aires_env_20260509/unpacked`: run `conda-unpack` from the new prefix or sed the same text/prefix files as Phase 4b, then grep for old `AI-RES` prefixes and add a cheap 5410 smoke import.
- Replace the Phase 10b `squeue` count with either a real fix to `scripts/submit_eval.sh` return-code capture or a wrapper that captures/parses the four submitted job IDs.
- After Phase 5 pass 2, verify every file in `/tmp/migration_untracked_active_with_airrs.txt` is clean, and update Phase 11 to say which untracked files intentionally remain untracked.
- Either expand Phase 6’s narrative-doc audit beyond README/skills/DSI docs, or explicitly classify remaining historical prose `AI-RES` mentions as preserved.
- Remove the stale “Update Pre-flight step 6” block; `BASELINE_COPY` is already persisted earlier in the current plan.

verdict: CHANGES_REQUESTED