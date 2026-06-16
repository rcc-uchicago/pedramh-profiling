**Strengths**
- Parent YAML knobs match the plan: GB=32, `lr=8e-4`, `scheduler_min_lr=1e-5`, `max_epochs=50`, `scheduler_T_max=45`, `input_noise.sigma=0.05`, EMA enabled. See [parent config](/home1/11114/zhixingliu/AI-RES/src/sfno_training/config/plasim_sim52_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5.yaml:126).
- Channel/layout claims match the repo: 52 state + `pr_6h`, 6 forcing channels. See [parent config](/home1/11114/zhixingliu/AI-RES/src/sfno_training/config/plasim_sim52_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5.yaml:58) and [channels.py](/home1/11114/zhixingliu/AI-RES/src/plasim_makani_packager/channels.py:47).
- The epoch/T_max probe matches the runtime scheduler model: `CosineAnnealingLR(T_max=...)` wrapped by `LinearLR` warmup. See [driver.py](/home1/11114/zhixingliu/AI-RES/makani-src/makani/utils/driver.py:696).

**Issues**
**P0**
- None.

**P1**
- Eval `RUN_DIR` in the plan is wrong. The plan passes `$SCRATCH/.../runs/<family>`, but training writes the actual run under `$EXP_DIR/<config>/0`; eval requires `config.json` and checkpoints inside that run dir. See [train_plasim.py](/home1/11114/zhixingliu/AI-RES/src/sfno_training/train_plasim.py:300), [checkpoint_loader.py](/home1/11114/zhixingliu/AI-RES/src/sfno_inference/checkpoint_loader.py:65), and [submit_eval_prelude.sh](/home1/11114/zhixingliu/AI-RES/scripts/submit_eval_prelude.sh:29). This would submit eval jobs that fail after queue time.
- Eval commands omit v11 data overrides. `submit_eval_prelude.sh` defaults `TEST_HOLDOUT`, `TRAIN_DIR`, and `PACKAGER_TEST_SRC` to v10 paths, while these v11 runs need `sim52_zgplev_full_v11` / `sim52_astro_64x128_zgplev_v11`. See [submit_eval_prelude.sh](/home1/11114/zhixingliu/AI-RES/scripts/submit_eval_prelude.sh:37) and the existing v11 beta chain override pattern in [submit_beta1_chains.sh](/home1/11114/zhixingliu/AI-RES/scripts/submit_beta1_chains.sh:16). This is exactly the kind of silent invalid comparison the plan is trying to avoid.
- Resume/collision semantics are underspecified. `RESUME=1` only bypasses the Slurm guard; actual resume is inferred from `ckpt_mp0_v0.tar`, then latest checkpoint is chosen by mtime. See [parent slurm](/home1/11114/zhixingliu/AI-RES/src/sfno_training/submit_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5.slurm:53), [train_plasim.py](/home1/11114/zhixingliu/AI-RES/src/sfno_training/train_plasim.py:316), and [checkpoint_helpers.py](/home1/11114/zhixingliu/AI-RES/makani-src/makani/utils/checkpoint_helpers.py:33). Partial run dirs without checkpoints are not refused.

**P2**
- `pr_6h` cross-track interpretation is missing. The repo has an open ticket documenting a ~3,600-4,400x `pr_6h` stats/unit mismatch vs 5410 and says direct 5410 overlay comparisons are meaningless without annotation/conversion/drop. See [ticket](/home1/11114/zhixingliu/AI-RES/docs/2026-05-14_pr_6h_units_mismatch_ticket.md:13) and [ticket action](/home1/11114/zhixingliu/AI-RES/docs/2026-05-14_pr_6h_units_mismatch_ticket.md:132).
- Epoch-75 curve inspection depends on logs, not checkpoints. Makani rotates only 3 checkpoint versions by default, so epoch 55 state will not be retained unless separately archived. See [driver.py](/home1/11114/zhixingliu/AI-RES/makani-src/makani/utils/driver.py:146) and [deterministic_trainer.py](/home1/11114/zhixingliu/AI-RES/makani-src/makani/utils/training/deterministic_trainer.py:395).
- Related doc reference is stale: the plan names `docs/2026-05-12_v11_clip_warmstart_continuation_plan.md`, but repo references/use the 2026-05-14 file. See [v11 warmstart config](/home1/11114/zhixingliu/AI-RES/src/sfno_training/config/plasim_sim52_zgplev_group_clone_v11_clip_warmstart.yaml:36).
- Existing GB=32 parent Slurm comments still say `GB=8 -> per-rank=2`; copied scripts should fix that. Runtime uses global batch divided by data-parallel size. See [parent slurm](/home1/11114/zhixingliu/AI-RES/src/sfno_training/submit_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5.slurm:95), [parent config](/home1/11114/zhixingliu/AI-RES/src/sfno_training/config/plasim_sim52_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5.yaml:129), and [train_plasim.py](/home1/11114/zhixingliu/AI-RES/src/sfno_training/train_plasim.py:51).

**Suggested Edits**
- Replace eval commands with actual run dirs, explicit EMA ckpt, and v11 data env vars.
- Add a fresh-run preflight: refuse if `$RUN_DIR` exists at all unless intentionally resuming; for resume, assert `ckpt_mp0_v0.tar` exists and list selected latest checkpoint.
- Add `pr_6h` reporting caveat: HPO decisions should use tas/zg500/etc.; do not interpret `pr_6h` vs 5410 until the unit ticket is resolved.
- Spell out new config filenames/top-level config keys and fix copied Slurm comments/budgets.

verdict: CHANGES_REQUESTED