**Strengths**
- User-approved intent is clear: suppression is explicitly chosen by the user, permanent by default, with upstream fixes and display-unit harmonization deferred.
- The prior round’s main rationale fixes are mostly applied: the plan no longer claims a clean `×21600` conversion, uses “6-hour precip proxy” in the banner, and cites both packed and BYO 5410 paths.
- The suppression site matches the renderer: one guarded append in `_render_table` covers both RMSE and ACC; `_render_masked_tas` is correctly out of scope.

**Issues**

**P1 — Smoke command still sources the wrong checkpoint key.**  
The plan’s smoke snippet reads `CKPT_PATH` from `provenance.txt` and passes `--ckpt-path "$CKPT_PATH"`: [plan](/home1/11114/zhixingliu/AI-RES/docs/2026-05-23_pr6h_unit_alignment_plan.md:493), [plan](/home1/11114/zhixingliu/AI-RES/docs/2026-05-23_pr6h_unit_alignment_plan.md:502). But own-track provenance writes `CKPT=...`, not `CKPT_PATH=...`: [submit_eval_prelude.sh](/home1/11114/zhixingliu/AI-RES/scripts/submit_eval_prelude.sh:184), [submit_eval_prelude.sh](/home1/11114/zhixingliu/AI-RES/scripts/submit_eval_prelude.sh:190). The 5410 provenance writer also uses `CKPT=`: [submit_eval_5410.sh](/home1/11114/zhixingliu/AI-RES/scripts/submit_eval_5410.sh:131), [submit_eval_5410.sh](/home1/11114/zhixingliu/AI-RES/scripts/submit_eval_5410.sh:138). This means the smoke can fail under `set -u` or silently render an empty checkpoint string, despite `--ckpt-path` being required: [render_eval_report.py](/home1/11114/zhixingliu/AI-RES/scripts/render_eval_report.py:51).

**P2 — Aggregation citation is still inaccurate.**  
The plan says `scores/nwp_scorecard_summary.csv` is aggregated at `score_nwp.py:125, :160, :172`: [plan](/home1/11114/zhixingliu/AI-RES/docs/2026-05-23_pr6h_unit_alignment_plan.md:100). Those lines load fields and compute per-IC RMSE/ACC: [score_nwp.py](/home1/11114/zhixingliu/AI-RES/scripts/score_nwp.py:125), [score_nwp.py](/home1/11114/zhixingliu/AI-RES/scripts/score_nwp.py:160), [score_nwp.py](/home1/11114/zhixingliu/AI-RES/scripts/score_nwp.py:172). The IC-averaging and summary CSV write happen in `_summarize`: [score_nwp.py](/home1/11114/zhixingliu/AI-RES/scripts/score_nwp.py:238), [score_nwp.py](/home1/11114/zhixingliu/AI-RES/scripts/score_nwp.py:253).

**Suggested edits**
- In the smoke snippet, source `CKPT` and pass `--ckpt-path "$CKPT"`, or map `CKPT_PATH="$CKPT"` explicitly after sourcing.
- Change the scalar-conversion rationale citation to cite `score_nwp.py:238-253` for aggregation; keep `:160` and `:172` only as citations for per-field metric computation.

verdict: CHANGES_REQUESTED