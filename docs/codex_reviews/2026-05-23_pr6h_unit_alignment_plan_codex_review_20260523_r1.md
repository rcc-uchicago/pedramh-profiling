**Strengths**
- The plan correctly treats “no on-disk NetCDF conversion” and “5410 group conventions are sacred” as user-approved constraints.
- It identifies the right main renderer surface: `scripts/render_eval_report.py` loads benchmark summaries and formats the scorecard table.
- The Phase 1 fail-loudly stance is appropriate; a guessed pr_6h conversion would be worse than suppression.

**Issues**
**P0 — Scalar report-time correction cannot make 5410 pr_6h RMSE/ACC comparable.**  
The renderer only reads aggregated `mean,std,n_ics` from CSVs ([scripts/render_eval_report.py:100](/home1/11114/zhixingliu/AI-RES/scripts/render_eval_report.py:100), [scripts/render_eval_report.py:374](/home1/11114/zhixingliu/AI-RES/scripts/render_eval_report.py:374)), then formats scalar cells ([scripts/render_eval_report.py:178](/home1/11114/zhixingliu/AI-RES/scripts/render_eval_report.py:178)). But RMSE/ACC are computed upstream from full prediction/truth fields ([scripts/score_nwp.py:125](/home1/11114/zhixingliu/AI-RES/scripts/score_nwp.py:125), [scripts/score_nwp.py:160](/home1/11114/zhixingliu/AI-RES/scripts/score_nwp.py:160), [scripts/score_nwp.py:172](/home1/11114/zhixingliu/AI-RES/scripts/score_nwp.py:172)). If 5410 predictions need inverse normalization, `RMSE(f(pred), truth)` cannot be recovered from an already-aggregated `RMSE(pred, truth)`. This challenges the user-approved “report rendering only” choice based on feasibility: renderer-only is possible only if it recomputes pr_6h metrics from fields, or suppresses 5410 pr_6h.

**P1 — The 5410 diagnostic transform chain is misstated.**  
The plan says the packed inference writes `diagnostic_prediction[:, 0]` directly, but the script first applies `stepper.dataset.diagnostic_transform(out_diagnostic)` ([scripts/infer_sfno5410_blocking_h100_packed.py:348](/home1/11114/zhixingliu/AI-RES/scripts/infer_sfno5410_blocking_h100_packed.py:348)). In the upstream loader, `diagnostic_transform` is the forward z-score transform and `diagnostic_inv_transform` is the inverse (`.../utils/data_loader_multifiles.py:705`, `:722`). Phase 1 should start from this exact chain; one inverse may not be enough if the raw NetCDF contains a second forward transform.

**P1 — Upstream paths in Phase 1 are wrong for this repo/runtime.**  
The plan points at `/work2/.../code/blocking/...`, but the production script defaults to `/work2/.../artifacts/derecho_blocking/source_trees/forecast_modules/PanguPlasim` ([scripts/infer_sfno5410_blocking_h100_packed.py:25](/home1/11114/zhixingliu/AI-RES/scripts/infer_sfno5410_blocking_h100_packed.py:25), [scripts/infer_sfno5410_blocking_h100_packed.py:32](/home1/11114/zhixingliu/AI-RES/scripts/infer_sfno5410_blocking_h100_packed.py:32)). The requested `makani/preprocessor.py` / `deterministic_trainer.py` are not present in that tree; the relevant transform code is in `utils/data_loader_multifiles.py` and `ensemble_inference.py`.

**P1 — Unit label/factor is inconsistent.**  
Repo comments treat own-track `pr_6h` as `m s^-1` ([scripts/render_eval_report.py:119](/home1/11114/zhixingliu/AI-RES/scripts/render_eval_report.py:119)). Existing figure code converts own precip from `m s^-1` to `mm day^-1` using `86400 * 1000` ([scripts/render_eval_figures.py:46](/home1/11114/zhixingliu/AI-RES/scripts/render_eval_figures.py:46), [scripts/render_eval_figures.py:58](/home1/11114/zhixingliu/AI-RES/scripts/render_eval_figures.py:58)). Therefore `×21600` yields `m/6h`, not `kg m^-2/6h` or `mm/6h`. If the table label is `kg m^-2/6h` / `mm/6h`, own RMSE needs `×21600×1000`; if the factor remains `×21600`, label it `m/6h`.

**P1 — “ACC needs no conversion” is over-broad.**  
ACC is invariant only when prediction, truth, and climatology are transformed consistently. Current ACC subtracts the same physical climatology from both prediction and truth ([src/sfno_eval/metrics.py:169](/home1/11114/zhixingliu/AI-RES/src/sfno_eval/metrics.py:169)), and `score_nwp.py` supplies that climatology per channel ([scripts/score_nwp.py:167](/home1/11114/zhixingliu/AI-RES/scripts/score_nwp.py:167)). If 5410 pr_6h prediction is in transformed space, keeping its ACC row is not automatically valid. Suppress or recompute pr_6h ACC together with RMSE.

**P2 — CLI/test wording mismatch.**  
The plan proposes `--pr6h-unit-align`, but the validation section calls `render_eval_report.py --mode own_to_5410_native`. Existing renderer has `--track`, not `--mode` ([scripts/render_eval_report.py:39](/home1/11114/zhixingliu/AI-RES/scripts/render_eval_report.py:39), [scripts/render_eval_report.py:87](/home1/11114/zhixingliu/AI-RES/scripts/render_eval_report.py:87)). Also thread the new flag through `eval_run_report_inline.sh` if default is initially `none` ([scripts/eval_run_report_inline.sh:28](/home1/11114/zhixingliu/AI-RES/scripts/eval_run_report_inline.sh:28)).

**P2 — Smoke re-render can clobber a production report.**  
`render_eval_report.py` defaults to overwriting `$OUT_ROOT/report.md` ([scripts/render_eval_report.py:434](/home1/11114/zhixingliu/AI-RES/scripts/render_eval_report.py:434), [scripts/render_eval_report.py:436](/home1/11114/zhixingliu/AI-RES/scripts/render_eval_report.py:436)). The smoke check should require `--report-out` to a scratch path or explicitly back up the existing report.

**Suggested edits**
- Change Phase 2 to either “suppress 5410 pr_6h unless field-level recompute is implemented” or “renderer recomputes pr_6h metrics from benchmark NetCDF + climatology.”
- Fix the transform audit paths and document the actual `diagnostic_transform` / `diagnostic_inv_transform` chain.
- Decide the display unit precisely: `m/6h` with `×21600`, or `mm/6h = kg m^-2/6h` with `×21600×1000`.
- Suppress/recompute pr_6h ACC under the same rule as RMSE.
- Use `--pr6h-unit-align` consistently, update report-stage scripts if needed, and run smoke output via `--report-out`.

verdict: CHANGES_REQUESTED