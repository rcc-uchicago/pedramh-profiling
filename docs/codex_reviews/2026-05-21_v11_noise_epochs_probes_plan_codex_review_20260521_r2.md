**Strengths**
- R1 eval fixes are resolved: the plan now uses inner `/<config>/0` run dirs and explicit v11 `TEST_HOLDOUT` / `TRAIN_DIR` / `PACKAGER_TEST_SRC`, matching [submit_eval_prelude.sh](/home1/11114/zhixingliu/AI-RES/scripts/submit_eval_prelude.sh:26) and [submit_beta1_chains.sh](/home1/11114/zhixingliu/AI-RES/scripts/submit_beta1_chains.sh:16).
- Collision/resume semantics now match the parent launcher and trainer: guard checks versioned ckpts, resume sentinel is `ckpt_mp0_v0.tar`, latest checkpoint is mtime-selected. See [parent slurm](/home1/11114/zhixingliu/AI-RES/src/sfno_training/submit_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5.slurm:58), [train_plasim.py](/home1/11114/zhixingliu/AI-RES/src/sfno_training/train_plasim.py:316), and [checkpoint_helpers.py](/home1/11114/zhixingliu/AI-RES/makani-src/makani/utils/checkpoint_helpers.py:33).
- Epoch-75 log-source note is correct: checkpoint rotation keeps 3 versions and `validation loss ema` is emitted into `out.log`. See [driver.py](/home1/11114/zhixingliu/AI-RES/makani-src/makani/utils/driver.py:146), [deterministic_trainer.py](/home1/11114/zhixingliu/AI-RES/makani-src/makani/utils/training/deterministic_trainer.py:395), and [plasim_trainer.py](/home1/11114/zhixingliu/AI-RES/src/sfno_training/trainer/plasim_trainer.py:657).

**Issues**

**P0**
- None.

**P1**
- None.

**P2**
- Warm-start rationale overstates the existing template schedule. The plan says the current warm-start templates re-warm to peak `8e-4`, but the actual v11/v10 warm-start configs are older GB=8, peak `lr: 1.0E-4`, `scheduler_min_lr: 1.0E-8` recipes. See [v11 warmstart config](/home1/11114/zhixingliu/AI-RES/src/sfno_training/config/plasim_sim52_zgplev_group_clone_v11_clip_warmstart.yaml:128), [v11 schedule](/home1/11114/zhixingliu/AI-RES/src/sfno_training/config/plasim_sim52_zgplev_group_clone_v11_clip_warmstart.yaml:142), [v10 warmstart config](/home1/11114/zhixingliu/AI-RES/src/sfno_training/config/plasim_sim52_zgplev_group_clone_v10_warmstart.yaml:116). This does not break the no-warmstart decision, but the explanation should say the existing templates answer an older lower-LR recipe, not a fresh 8e-4 phase.
- The `pr_6h` caveat now resolves the RMSE unit issue, but the added “`pr_6h` ACC trajectory remains comparable” language is too strong. The repo’s figure path intentionally omits `pr_6h` line plots because both RMSE and ACC are misleading for intermittent precipitation, even before cross-track unit issues. See [render_eval_figures.py](/home1/11114/zhixingliu/AI-RES/scripts/render_eval_figures.py:28) and [render_eval_figures.py](/home1/11114/zhixingliu/AI-RES/scripts/render_eval_figures.py:278). Keep `pr_6h` as a secondary within-track diagnostic at most, not an HPO decision input.
- Eval commands still lack a cheap pre-submit existence check for the chosen EMA checkpoint. `submit_eval_prelude.sh` chooses or accepts `CKPT` but does not validate that it exists before queuing the SLURM chain; the failure moves to the inference job. See [submit_eval_prelude.sh](/home1/11114/zhixingliu/AI-RES/scripts/submit_eval_prelude.sh:29) and [eval_run_inference_inline.sh](/home1/11114/zhixingliu/AI-RES/scripts/eval_run_inference_inline.sh:73).
- The plan says to use the beta-chain precedent for v11 eval overrides, but omits the repo’s run convention that `sbatch` chains should be submitted from a login node. See [submit_beta1_chains.sh](/home1/11114/zhixingliu/AI-RES/scripts/submit_beta1_chains.sh:7).

**Suggested Edits**
- Replace the warm-start sentence with: “Existing warm-start templates are fresh 50-epoch lower-LR GB=8 recipes (`lr=1e-4`, `T_max=45`, `min_lr=1e-8`), so they do not answer this parent’s 8e-4/min_lr1e-5 tail-extension question.”
- Change the `pr_6h` decision text to: “Do not use cross-track `pr_6h`; within-track `pr_6h` may be inspected as secondary diagnostics only. HPO decisions use tas, zg500, ua5, ta5.”
- Add before each eval command: `test -s "$CKPT"` and “run from a login node.”
- Spell exact new YAML/slurm filenames and config keys in the implementation checklist to reduce copy/paste risk from the parent launcher.

verdict: CHANGES_REQUESTED