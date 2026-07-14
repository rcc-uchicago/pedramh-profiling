**Strengths**
- R5-F3 and R5-F4 look resolved: Phase 5.5 now has a symlink manifest/rollback path and content-based text discovery.
- Phase 10b still respects own-track eval conventions: fresh `OUT_ROOT`, EMA-default checkpoint behavior, and no fake `SCORE_ONLY_K`.
- DSI preservation is explicitly documented, and the new DSI audit is much safer than the prior broad allowlist.

**Issues**

**P1 - R5-F1 is not fully resolved: Phase 10a still does not test the group env wrapper.**  
Phase 4b now activates the conda env directly, but the resolution said wrapper testing was deferred to Phase 10a. Phase 10a only checks own-track/eval shell scripts; it never sources `src/sfno_training_group/env_activate.sh` or syntax-checks `src/sfno_training_group/slurm/*.slurm`. That wrapper is what group jobs actually use: [submit_train_full.slurm](/home1/11114/zhixingliu/AI-RES/src/sfno_training_group/slurm/submit_train_full.slurm:25), [submit_inference_smoke.slurm](/home1/11114/zhixingliu/AI-RES/src/sfno_training_group/slurm/submit_inference_smoke.slurm:21), with the absolute env prefix in [env_activate.sh](/home1/11114/zhixingliu/AI-RES/src/sfno_training_group/env_activate.sh:13).

**P1 - Phase 5 misses load-bearing bare `"AI-RES"` path components.**  
`SED_EXPR` handles absolute/env-var path strings, but not Python path construction like [hpo_prune.py](/home1/11114/zhixingliu/AI-RES/scripts/hpo_prune.py:40), [hpo_prune.py](/home1/11114/zhixingliu/AI-RES/scripts/hpo_prune.py:41), and [hpo_prune.py](/home1/11114/zhixingliu/AI-RES/scripts/hpo_prune.py:99). The Phase 5 stale-path grep also won’t catch these because the source text is `SCRATCH / "AI-RES"` rather than `/AI-RES` or `$SCRATCH/AI-RES`.

**P1 - Preflight untracked gate conflicts with out-of-scope vendored files.**  
The plan excludes `external/**` from Phase 5 and says vendored trees are out of scope, but the preflight aborts on untracked `AI-RES` hits except `analysis_outputs/` and `docs/2026-`. Current out-of-scope `external/PanguWeather_stampede3_env.txt` contains old paths at [external/PanguWeather_stampede3_env.txt](/home1/11114/zhixingliu/AI-RES/external/PanguWeather_stampede3_env.txt:6), [external/PanguWeather_stampede3_env.txt](/home1/11114/zhixingliu/AI-RES/external/PanguWeather_stampede3_env.txt:8), and [external/PanguWeather_stampede3_env.txt](/home1/11114/zhixingliu/AI-RES/external/PanguWeather_stampede3_env.txt:10), so the current repo can hit an abort unless the plan explicitly exempts or disposes of this class.

**P2 - DSI positive audit still misses Stampede3 home paths.**  
This does not challenge the user-approved DSI `$HOME/AI-RES` preservation. The missed cases are absolute Stampede3 home paths, e.g. [dsi_full_training_plan.md](/home1/11114/zhixingliu/AI-RES/docs/dsi_full_training_plan.md:57), [dsi_smoke_backup_plan.md](/home1/11114/zhixingliu/AI-RES/docs/dsi_smoke_backup_plan.md:80), and [dsi_smoke_backup_plan.md](/home1/11114/zhixingliu/AI-RES/docs/dsi_smoke_backup_plan.md:243). The current DSI audit regex only covers scratch/work/work2/GitHub patterns.

**P2 - `report.md` raw diff is expected to differ.**  
`render_eval_report.py` writes the run tag and checkpoint path into the report header at [render_eval_report.py](/home1/11114/zhixingliu/AI-RES/scripts/render_eval_report.py:387) and [render_eval_report.py](/home1/11114/zhixingliu/AI-RES/scripts/render_eval_report.py:394), and the report stage passes `--ckpt-path "$CKPT"` at [eval_run_report_inline.sh](/home1/11114/zhixingliu/AI-RES/scripts/eval_run_report_inline.sh:35). Phase 10b intentionally changes both `RUN_TAG` and old/new paths, so `diff -u "$OLD_BASE/report.md" "$NEW_REPORT"` will produce expected noise.

**Suggested Edits**
- Add a Phase 10a command that sources `src/sfno_training_group/env_activate.sh`, checks `sys.executable`, then deactivates; also include `bash -n src/sfno_training_group/slurm/*.slurm`.
- Add an explicit code audit for bare path components, e.g. `rg -n '"AI-RES"|/ "AI-RES"' scripts src tests`, and manually update `scripts/hpo_prune.py`.
- Align preflight with out-of-scope rules: exempt `external/` and `makani-src/`, or explicitly require moving/removing those untracked files before migration.
- Extend the DSI-doc rewrite audit to include `/home1/11114/zhixingliu/AI-RES`.
- Replace raw `report.md` diff with a filtered comparison or document expected header/path differences.

verdict: CHANGES_REQUESTED