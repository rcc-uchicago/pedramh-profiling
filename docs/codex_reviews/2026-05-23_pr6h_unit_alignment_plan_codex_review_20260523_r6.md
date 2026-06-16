**Strengths**
- R5 applied items are resolved in the plan text: `_render_table` gets `pr6h_unit_align`, Phase-1 check wording is consistent, second-IC check is scoped within 5410, `none` banner behavior is preserved, and the skill env-table update is included.
- Main suppression site matches the real shared RMSE/ACC loop in [render_eval_report.py](/home1/11114/zhixingliu/AI-RES/scripts/render_eval_report.py:163).
- `_render_masked_tas` is correctly left untouched; it is `tas_no_ice`-only in [render_eval_report.py](/home1/11114/zhixingliu/AI-RES/scripts/render_eval_report.py:189).
- The second-IC empirical check is feasible: `Y121_s0122.nc` exists and shows the same transformed-prediction signature as `Y121_s0000.nc`.

**Issues**
P0: None.

P1 — `track == "own"` is load-bearing, but group/5410 report entrypoints do not reliably set `TRACK=5410`.  
The plan says `--track 5410` is used for group/direct 5410 cases and must avoid suppression ([plan](/home1/11114/zhixingliu/AI-RES/docs/2026-05-23_pr6h_unit_alignment_plan.md:278)), but `eval_run_report_inline.sh` defaults `TRACK=own` and defaults the benchmark overlay on ([eval_run_report_inline.sh](/home1/11114/zhixingliu/AI-RES/scripts/eval_run_report_inline.sh:18)), `submit_eval_group_prod.sh` calls the shared report SLURM without exporting `TRACK` ([submit_eval_group_prod.sh](/home1/11114/zhixingliu/AI-RES/scripts/submit_eval_group_prod.sh:100)), and `submit_eval_report_5410.slurm` invokes the renderer without `--track 5410` ([submit_eval_report_5410.slurm](/home1/11114/zhixingliu/AI-RES/scripts/submit_eval_report_5410.slurm:40)). Literal implementation can suppress a valid group-native `pr_6h` benchmark row on shared group report paths.

P2 — Tests do not cover the load-bearing `track=5410` branch.  
§6 tests own+suppress and `none`, but not `track="5410"` with `pr6h_unit_align="suppress"` preserving the `pr_6h` benchmark row ([plan](/home1/11114/zhixingliu/AI-RES/docs/2026-05-23_pr6h_unit_alignment_plan.md:379)). That is the exact guard preventing valid 5410-track suppression.

P2 — Future-fixed 5410 wording overstates `none`.  
The risk row says a fixed 5410 build can use `--pr6h-unit-align none` to restore a valid comparison ([plan](/home1/11114/zhixingliu/AI-RES/docs/2026-05-23_pr6h_unit_alignment_plan.md:440)), but the truth-side unit gap remains own m/s vs 5410 m/6h ([plan](/home1/11114/zhixingliu/AI-RES/docs/2026-05-23_pr6h_unit_alignment_plan.md:33)). `none` restores the row plus disclaimer; it does not make own-track RMSE directly numeric-comparable without a scalar unit conversion or recompute.

**Suggested Edits**
- Add a call-site audit/update step: set `TRACK=5410` or pass `--track 5410` for actual group/5410 report paths, or explicitly declare those paths out of scope and ensure they do not inherit benchmark+suppress defaults.
- Add a unit test asserting `track="5410", pr6h_unit_align="suppress"` keeps `| pr_6h | 5410 benchmark |`.
- Reword the future-fixed risk: `none` restores the displayed row; direct own-vs-5410 RMSE comparability still requires addressing the m/s vs m/6h scale.

verdict: CHANGES_REQUESTED