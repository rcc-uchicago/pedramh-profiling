**Strengths**
- Round-3 applied fixes are resolved. The eval blocks now export `CKPT`, test it, then call `scripts/submit_eval.sh`, which matches the repo reality that prelude only selects a default and inference consumes `--ckpt` later: [submit_eval_prelude.sh](/home1/11114/zhixingliu/AI-RES/scripts/submit_eval_prelude.sh:29), [eval_run_inference_inline.sh](/home1/11114/zhixingliu/AI-RES/scripts/eval_run_inference_inline.sh:73).
- The narrowed noise-history wording now matches the configs: active lower-noise probes are on `lr8e4`, while historical `sigma=0.075` is on retired `lr1p13e3`: [noise0p035 config](/home1/11114/zhixingliu/AI-RES/src/sfno_training/config/plasim_sim52_zgplev_group_clone_v11_gb32_lr8e4_noise0p035.yaml:4), [lr1p13e3 noise0p075 config](/home1/11114/zhixingliu/AI-RES/src/sfno_training/config/plasim_sim52_zgplev_group_clone_v11_gb32_lr1p13e3_noise0p075.yaml:1).
- Planned config deltas match the parent: `lr=8e-4`, `max_epochs=50`, `batch_size=32`, `scheduler_T_max=45`, `scheduler_min_lr=1e-5`, `input_noise.sigma=0.05`, EMA enabled: [parent config](/home1/11114/zhixingliu/AI-RES/src/sfno_training/config/plasim_sim52_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5.yaml:126).
- Collision/resume and checkpoint-rotation notes match the launcher/trainer behavior: [parent SLURM guard](/home1/11114/zhixingliu/AI-RES/src/sfno_training/submit_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5.slurm:58), [train resume sentinel](/home1/11114/zhixingliu/AI-RES/src/sfno_training/train_plasim.py:316), [checkpoint version rotation](/home1/11114/zhixingliu/AI-RES/makani-src/makani/utils/training/deterministic_trainer.py:395).

**Issues**

**P0**
- None.

**P1**
- None.

**P2**
- None.

**Suggested Edits**
- None required.

verdict: APPROVED