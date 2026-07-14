**Strengths**
- R2 fixes for `SED_EXPR` scoping are materially present: exported and passed as `$1`, with fail-hard stale-path checks.
- R2 backup coverage for Phase 5.5 now matches the mutation set.
- Phase 10b now sources `/tmp/migration_baseline.txt`, so the old baseline/run handles are no longer shell-local only.

**Issues**
- **P0:** None.

- **P1 - R2-F1 is still not fully resolved: the stash option removes the active untracked files.** The plan says `git add` the untracked files, then `git stash push -u` so they survive into Phase 5 pass 1 ([plan](/home1/11114/zhixingliu/AI-RES/docs/2026-05-23_sfno_climate_emulator_migration_plan.md:21)). Staged new files are still removed by `git stash push -u`; Phase 5 then will not see uncommitted `scripts/submit_eval_prelude.sh`, `scripts/submit_eval_5410.sh`, or `src/sfno_training_group/env_activate.sh`, which contain load-bearing old paths ([submit_eval_prelude.sh](/home1/11114/zhixingliu/AI-RES/scripts/submit_eval_prelude.sh:26), [submit_eval_5410.sh](/home1/11114/zhixingliu/AI-RES/scripts/submit_eval_5410.sh:32), [env_activate.sh](/home1/11114/zhixingliu/AI-RES/src/sfno_training_group/env_activate.sh:13)).

- **P1 - challenges the surgical-env scope for the separate group conda env: `mv` is not transparent.** The plan only fixes `.venv` and says the group conda env is renamed transparently by Phase 1 ([plan](/home1/11114/zhixingliu/AI-RES/docs/2026-05-23_sfno_climate_emulator_migration_plan.md:484)). But the repo activates that env by absolute prefix ([env_activate.sh](/home1/11114/zhixingliu/AI-RES/src/sfno_training_group/env_activate.sh:13)), and the env itself has old-prefix shebangs such as [/work2/.../bin/tqdm](/work2/11114/zhixingliu/stampede3/AI-RES/envs/group_pangu_sfno_v2/bin/tqdm:1). After the old symlink expires, group-track tools can still point back to `AI-RES`.

- **P1 - R2-F2 DSI audit omits `$DSI_SCRATCH/AI-RES`.** The new table preserves `$HOME/AI-RES`, `$DSI_PROJECT/AI-RES`, and the bootstrap name, but not DSI scratch ([plan](/home1/11114/zhixingliu/AI-RES/docs/2026-05-23_sfno_climate_emulator_migration_plan.md:270)). The DSI smoke doc uses `$DSI_SCRATCH/AI-RES` as real DSI layout ([dsi_smoke_backup_plan.md](/home1/11114/zhixingliu/AI-RES/docs/dsi_smoke_backup_plan.md:469), [dsi_smoke_backup_plan.md](/home1/11114/zhixingliu/AI-RES/docs/dsi_smoke_backup_plan.md:483)), so the post-audit grep would either fail or encourage rewriting an out-of-scope DSI path.

- **P2 - `~/AI-RES` paths are not covered by Phase 5.** `SED_EXPR` handles absolute paths and `$HOME/AI-RES`, but not tilde paths ([plan](/home1/11114/zhixingliu/AI-RES/docs/2026-05-23_sfno_climate_emulator_migration_plan.md:153)). Several user-facing skill commands use `~/AI-RES` ([plasim-makani-packager](/home1/11114/zhixingliu/AI-RES/skills/plasim-makani-packager/SKILL.md:56), [plasim-postprocess](/home1/11114/zhixingliu/AI-RES/skills/plasim-postprocess/SKILL.md:83)). Phase 6 should not say paths are already covered.

- **P2 - Earth2Studio editable reinstall conditional is shell-precedence wrong.** The plan’s `[ -f setup.py ] || [ -f pyproject.toml ] && pip install ...` skips install when `setup.py` exists ([plan](/home1/11114/zhixingliu/AI-RES/docs/2026-05-23_sfno_climate_emulator_migration_plan.md:133)); `external/earth2studio/setup.py` does exist ([setup.py](/home1/11114/zhixingliu/AI-RES/external/earth2studio/setup.py:18)).

- **P2 - Phase 10b still loses the new eval root across fresh shells.** `OUT_ROOT` is exported before submission ([plan](/home1/11114/zhixingliu/AI-RES/docs/2026-05-23_sfno_climate_emulator_migration_plan.md:398)), but the diff block only re-sources baseline vars and then uses `$OUT_ROOT` ([plan](/home1/11114/zhixingliu/AI-RES/docs/2026-05-23_sfno_climate_emulator_migration_plan.md:409)). Persist `NEW_OUT_ROOT` or recompute it from `RUN_TAG`.

**Suggested Edits**
- Remove the stash alternative, or require `git stash pop/apply` before Phase 5; safest is “commit WIP before rename.”
- Add a group conda-env fix/verification phase: rebuild, conda-pack/unpack, or explicitly rewrite old prefixes and verify with the old symlink temporarily absent.
- Add `$DSI_SCRATCH/AI-RES` to the DSI preserve table and grep allowlist.
- Add `~/AI-RES` rewrite coverage or explicit Phase 6 manual edits.
- Replace the Earth2Studio conditional with a grouped `if [ -f ... ] || [ -f ... ]; then ...; fi`.
- Persist `NEW_OUT_ROOT` in `/tmp/migration_baseline.txt` or another sidecar before waiting on SLURM.

verdict: CHANGES_REQUESTED