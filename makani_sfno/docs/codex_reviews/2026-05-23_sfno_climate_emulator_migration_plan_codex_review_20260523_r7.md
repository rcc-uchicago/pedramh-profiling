**Strengths**
- R6-F2 is materially addressed: the quoted `AI-RES` sed patterns now cover `scripts/hpo_prune.py` path components like [hpo_prune.py:40](/home1/11114/zhixingliu/AI-RES/scripts/hpo_prune.py:40).
- The v10/v11 eval path still respects EMA defaulting and K=56: [submit_eval_prelude.sh:29](/home1/11114/zhixingliu/AI-RES/scripts/submit_eval_prelude.sh:29), [eval_inference.py:61](/home1/11114/zhixingliu/AI-RES/scripts/eval_inference.py:61).
- External vendored dirs are now aligned with the out-of-scope decision.

**Issues**
**P1 - DSI audit regex now conflicts with the preserve table.**  
The plan says `AI-RES-dsi-bootstrap` must be preserved [plan:363](/home1/11114/zhixingliu/AI-RES/docs/2026-05-23_sfno_climate_emulator_migration_plan.md:363), but the new `/home1/.../AI-RES` audit regex [plan:376](/home1/11114/zhixingliu/AI-RES/docs/2026-05-23_sfno_climate_emulator_migration_plan.md:376) matches the preserved bootstrap path in [dsi_full_training_plan.md:250](/home1/11114/zhixingliu/AI-RES/docs/dsi_full_training_plan.md:250). That makes the “must be hand-edited” audit either force a wrong edit or never go empty.

**P1 - Phase 10a checks are fail-open.**  
`bash -n "$f" || echo BROKEN: $f` does not fail the smoke block [plan:448](/home1/11114/zhixingliu/AI-RES/docs/2026-05-23_sfno_climate_emulator_migration_plan.md:448). The stale-path greps also only echo and continue [plan:461](/home1/11114/zhixingliu/AI-RES/docs/2026-05-23_sfno_climate_emulator_migration_plan.md:461). This weakens the R6-F1 fix because group scripts really depend on the wrapper: [submit_train_full.slurm:25](/home1/11114/zhixingliu/AI-RES/src/sfno_training_group/slurm/submit_train_full.slurm:25), [env_activate.sh:13](/home1/11114/zhixingliu/AI-RES/src/sfno_training_group/env_activate.sh:13).

**P1 - Review/plan artifacts are not consistently protected.**  
The preflight gate exempts `docs/2026-` but not `docs/codex_reviews/` [plan:29](/home1/11114/zhixingliu/AI-RES/docs/2026-05-23_sfno_climate_emulator_migration_plan.md:29), while the plan later expects historical `docs/codex_reviews/` hits to remain [plan:384](/home1/11114/zhixingliu/AI-RES/docs/2026-05-23_sfno_climate_emulator_migration_plan.md:384). Current R6 sidecars contain `AI-RES`, e.g. [r6_resolution.md:5](/home1/11114/zhixingliu/AI-RES/docs/codex_reviews/2026-05-23_sfno_climate_emulator_migration_plan_codex_review_20260523_r6_resolution.md:5), so the gate can abort or Phase 5 can rewrite review history. The migration plan itself is also under `docs/2026-` and contains executable old-path commands like [plan:71](/home1/11114/zhixingliu/AI-RES/docs/2026-05-23_sfno_climate_emulator_migration_plan.md:71), yet Phase 12 wants to preserve/copy it [plan:613](/home1/11114/zhixingliu/AI-RES/docs/2026-05-23_sfno_climate_emulator_migration_plan.md:613).

**P2 - `report.md` normalization still leaves expected run-tag noise.**  
`render_eval_report.py` writes whatever run tag it receives [render_eval_report.py:388](/home1/11114/zhixingliu/AI-RES/scripts/render_eval_report.py:388), and default historical tags are date-prefixed from [submit_eval_prelude.sh:132](/home1/11114/zhixingliu/AI-RES/scripts/submit_eval_prelude.sh:132). The plan normalizes only `postrename_*` [plan:545](/home1/11114/zhixingliu/AI-RES/docs/2026-05-23_sfno_climate_emulator_migration_plan.md:545), so the old baseline run tag will still diff.

**Suggested Edits**
- Bound the DSI home-path regex so it does not match `AI-RES-dsi-bootstrap`, e.g. require `/AI-RES(/|$|[[:space:]\`])`, or explicitly exclude the bootstrap line from the rewrite audit.
- Make Phase 10a fail hard: add `set -euo pipefail`, use `bash -n "$f" || { echo "BROKEN: $f"; exit 1; }`, and replace the `! git grep ... || echo` checks with `if git grep ...; then exit 1; fi`.
- Add `docs/codex_reviews/` and the migration-plan file itself to the preflight/sed exclusions, or explicitly require moving those audit artifacts outside the tree before Phase 5.
- Normalize the whole report run-tag line, not just `postrename_*`, e.g. substitute `(**Run tag:** \`)[^\`]+(\`)` to `<RUN_TAG>`.

verdict: CHANGES_REQUESTED