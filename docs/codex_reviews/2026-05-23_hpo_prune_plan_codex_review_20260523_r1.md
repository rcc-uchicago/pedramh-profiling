**Strengths**
- Intent markers are clear: “from interview,” memory slugs, proposed protect list, proposed sweep groups, and explicit sign-off gates. I treated those as user-approved.
- The distill-before-delete sequence, manifest-only deletion, runtime protect guard, and audit log are the right safety shape.

**Issues**
**P0**
- None found.

**P1**
- Unassigned old eval dirs can be swept by age without explicit §3 sign-off. The existing dry-run manifest already includes familyless eval roots like `20260509_gb4_ema` and `20260509_y11valid_gb4_k60` as prune targets, but those are not in the sweep tables/protect list: [prune_manifest.csv](/home1/11114/zhixingliu/AI-RES/docs/hpo_distill/prune_manifest.csv:28). Newer evals get `_family-...` tags from `submit_eval_prelude`, but older ones do not: [submit_eval_prelude.sh](/home1/11114/zhixingliu/AI-RES/scripts/submit_eval_prelude.sh:126).

- Eval distill schema does not match repo scoring. The repo scores leads `6,24,72,120,240,336`, not `168,360`: [score_nwp.py](/home1/11114/zhixingliu/AI-RES/scripts/score_nwp.py:53). Key channels are `tas, pr_6h, zg500|zg5, ua5, ta5`, not `t850,z500,u500,v500`: [_eval_utils.py](/home1/11114/zhixingliu/AI-RES/scripts/_eval_utils.py:83), [plasim_sim52_zgplev_group_clone_v11_gb32_lr8e4.yaml](/home1/11114/zhixingliu/AI-RES/src/sfno_training/config/plasim_sim52_zgplev_group_clone_v11_gb32_lr8e4.yaml:58). Repo scorecards emit `rmse` and `acc`, not `mae`: [score_nwp.py](/home1/11114/zhixingliu/AI-RES/scripts/score_nwp.py:158).

- Active SLURM matching is under-specified. Eval jobs have generic names like `sfno_eval_inf`, so `squeue -u $USER`/job-name matching cannot reliably map jobs to manifest paths: [submit_eval_inference.slurm](/home1/11114/zhixingliu/AI-RES/scripts/submit_eval_inference.slurm:2). The current script only asks `squeue` for job id/name, confirming the limitation: [hpo_prune.py](/home1/11114/zhixingliu/AI-RES/scripts/hpo_prune.py:1031).

- Training checkpoint deletion target should be an exact manifest path, not `path / "plasim_*" / "0" / "training_checkpoints"`. Actual training output is `<EXP_DIR>/<config>/0/training_checkpoints`: [train_plasim.py](/home1/11114/zhixingliu/AI-RES/src/sfno_training/train_plasim.py:300), [submit_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_epochs75.slurm](/home1/11114/zhixingliu/AI-RES/src/sfno_training/submit_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_epochs75.slurm:58). Literal `Path / "plasim_*"` will not glob.

- EMA vs raw is not preserved strongly enough in the planned summaries. Eval defaults prefer `best_ckpt_ema_mp0.tar`: [submit_eval_prelude.sh](/home1/11114/zhixingliu/AI-RES/scripts/submit_eval_prelude.sh:27). Trainer logs and writes separate EMA-best state: [plasim_trainer.py](/home1/11114/zhixingliu/AI-RES/src/sfno_training/trainer/plasim_trainer.py:651), [plasim_trainer.py](/home1/11114/zhixingliu/AI-RES/src/sfno_training/trainer/plasim_trainer.py:714).

**P2**
- Training log parse description is too schematic. Actual logs are multiline `Epoch N summary` blocks with `epoch time [s]`, `training loss`, `validation loss`, and `validation loss ema`: [out.log](/scratch/11114/zhixingliu/AI-RES/runs/sfno_zgplev_group_clone_v11_gb32_lr8e4/plasim_sim52_zgplev_group_clone_v11_gb32_lr8e4/0/out.log:280). Planned columns include `lr`, which is not emitted per epoch.
- Totals look stale. The sweep assignment set in the repo contains 26 PRUNE training runs, not 24: [hpo_prune.py](/home1/11114/zhixingliu/AI-RES/scripts/hpo_prune.py:89).

**Suggested Edits**
- Default all unclassified eval dirs to protect unless individually listed in §3 or explicitly approved in the dry-run manifest; add own-track production/canonical eval aliases to the protect list.
- Replace eval canonical extraction with repo schema: leads `6,24,72,120,240,336`; channels `tas, pr_6h, zg500|zg5, ua5, ta5` plus optional `tas_no_ice`; metrics `rmse,acc`.
- Store exact delete targets in the manifest (`ckpt_dir`, `inference`, `baselines`) and separate `target_mtime_iso` from `run_mtime_iso`.
- Make the SLURM guard either “queue must be empty” or implement explicit lock files/scontrol inspection; do not rely on generic `squeue` path matching.
- Add `best_val_loss_ema`, `best_val_loss_raw`, `best_epoch_*`, and `ckpt_flavor` to train/eval summaries; hash eval `provenance.txt` and scorecard, not only training `metadata.json/config.json`.

verdict: CHANGES_REQUESTED