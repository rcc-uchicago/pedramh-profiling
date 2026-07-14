**Strengths**
- Warm-start rationale, `pr_6h` downgrade, login-node note, exact artifact names, inner `RUN_DIR`, v11 eval overrides, and GB=32 SLURM comment guidance are now aligned with the repo.
- No target run-dir collision currently exists under `$SCRATCH/AI-RES/runs`.

**Issues**

**P0**
- None.

**P1**
- None.

**P2**
- CKPT preflight is still not correctly applied to the eval commands. The plan’s eval snippets use one-shot `CKPT=... scripts/submit_eval.sh`; the later standalone `test -s "$CKPT"` will not see that assignment unless the user separately sets/export it. The repo still does not validate explicit `CKPT` before queuing; it only selects a default when unset, then inference consumes it later. See [submit_eval_prelude.sh](/home1/11114/zhixingliu/AI-RES/scripts/submit_eval_prelude.sh:29), [submit_eval.sh](/home1/11114/zhixingliu/AI-RES/scripts/submit_eval.sh:65), and [eval_run_inference_inline.sh](/home1/11114/zhixingliu/AI-RES/scripts/eval_run_inference_inline.sh:73).
- The motivation still overstates “never tested σ above 0.05.” The repo has historical σ=0.075 artifacts, albeit on the now-inactive lr1p13e3 branch. See [lr1p13e3 noise0p075 config](/home1/11114/zhixingliu/AI-RES/src/sfno_training/config/plasim_sim52_zgplev_group_clone_v11_gb32_lr1p13e3_noise0p075.yaml:17), [sigma line](/home1/11114/zhixingliu/AI-RES/src/sfno_training/config/plasim_sim52_zgplev_group_clone_v11_gb32_lr1p13e3_noise0p075.yaml:173), and the later note that those lr1p13e3 noise templates are historical, not active, in [lr8e4 noise0p035](/home1/11114/zhixingliu/AI-RES/src/sfno_training/config/plasim_sim52_zgplev_group_clone_v11_gb32_lr8e4_noise0p035.yaml:4).

**Suggested Edits**
- Rewrite each eval snippet as a small block that assigns `RUN_DIR` and `CKPT` in the shell first, runs `test -s "$CKPT"`, then calls `TRACK=own MODE=nwp ... scripts/submit_eval.sh`.
- Narrow the noise motivation to: “No σ above 0.05 has been tested on the active lr8e4/minlr1e5 branch; historical σ=0.075 existed only on the retired lr1p13e3 branch.”

verdict: CHANGES_REQUESTED