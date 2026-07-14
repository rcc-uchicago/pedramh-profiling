# v11 EMA / gbhpo40 RUN_TAG collision — 2026-05-12

**Eval RUN_TAG:** `20260512_eval-8b395eb_data-e3c934b_ckpt-best_ckpt_ema_mp0`
**Conflicting training runs:**
- `sfno_zgplev_group_clone_v11` (v11)
- `sfno_zgplev_gbhpo40_gb16_lr2_0e-4_20260511` (gbhpo40)

**Author / debugger:** Zhixing Liu / Claude
**Date:** 2026-05-12
**Status:** mitigated (auto-derived `_family-<train_family>` suffix + collision-guard merged)
**Severity:** correctness (mis-attributed scorecard)

## TL;DR

Two eval chains targeting different training runs at the same eval/data SHA
on the same day both resolved to one `$OUT_ROOT`. The v11 EMA chain
re-populated the inference NCs but the gbhpo40 chain's scorecard CSV was
not regenerated, so for a window the scorecard appeared to belong to v11
EMA but had actually been computed on the wrong inference dir. Fixed by
adding `_family-<TRAIN_FAMILY>` to RUN_TAG and a hard collision-guard in
`scripts/submit_eval.sh`.

## Symptoms

- `$WORK2/AI-RES/results/sfno_eval/20260512_…_ckpt-best_ckpt_ema_mp0/scores/nwp_scorecard_summary.csv`
  was bit-identical (md5) to a snapshot under
  `_snapshot_v11_pre_collision_resolve_20260512/nwp_scorecard_summary.csv`.
- The snapshot's `provenance_currently_chain1.txt` pointed at gbhpo40:
  `CKPT=…/sfno_zgplev_gbhpo40_gb16_lr2_0e-4_20260511/.../best_ckpt_ema_mp0.tar`.
- The current `provenance.txt` pointed at v11 EMA, with
  `RESTORED_NOTE=chain1_gbhpo_lr2.0e-4_collided_on_RUN_TAG_when_v11_REP_was_still_PENDING`.
- The inference NCs in `inference/nwp/` carried the v11 EMA `ckpt_path`
  attribute (timestamp 16:42 CDT), but score CSV (18:06 CDT) reflected
  scoring done against ... actually the v11 inference dir, just with the
  collision metadata noise around it.

The user-facing risk: the first attempt at root-causing the v11 regression
used the snapshot file and almost concluded that v11 EMA was producing
**gbhpo40-quality** numbers, because the snapshot provenance pointed at
gbhpo40. The numbers in scores/ were in fact from v11 EMA; the snapshot
was a defensive copy made during cleanup, not the active CSV. Confusion
multiplied because both files had identical md5 (same data, different
provenance labels co-existed in the same parent dir).

## What we checked (and ruled out)

1. **Are the score CSVs actually identical bytes?** `md5sum`: identical.
   So the two filenames refer to the same data.
2. **Does the inference dir match v11 EMA?** `ds.attrs['ckpt_path']` on
   `inference/nwp/MOST.0121_ic000.nc` and `MOST.0128_ic000.nc` → both point
   at the v11 EMA file. Inference outputs are legitimate v11 EMA.
3. **Are the two training ckpts actually different?** md5sum on
   `…/sfno_zgplev_group_clone_v11/…/best_ckpt_ema_mp0.tar` vs
   `…/sfno_zgplev_gbhpo40_gb16_lr2_0e-4_20260511/…/best_ckpt_ema_mp0.tar` —
   different hashes, different val_loss (v11 ≈ 0.00316, gbhpo40 ≈ 0.00417).
4. **Could scoring have been done against the gbhpo40 inference NCs and then
   the NCs got overwritten?** Timestamps say no — inference NCs at 16:42 CDT,
   scoring at 18:06 CDT, monotone.

The data is from v11 EMA. The collision artifact is the *provenance*
labeling — chain1 (gbhpo40) tried to write to the same RUN_TAG slot after
v11's eval landed, the chain1 provenance.txt clobbered the v11 one
temporarily, then someone restored the v11 provenance and moved chain1's
provenance into the `_snapshot_*` dir.

## Root cause

`submit_eval.sh`'s default RUN_TAG template was
`${DATE_STR}_eval-${EVAL_SHA7}_data-${DATA_SHA7}[_ckpt-...]`. Two different
training runs (v11, gbhpo40) at the same eval+data SHA on the same day
land in the same `$OUT_ROOT`. The `_ckpt-` suffix didn't differentiate them
because both used the EMA-best (same basename). No collision detection.

## Fix

`scripts/submit_eval.sh` (committed 2026-05-12):

1. **Auto-derive `_family-<TRAIN_FAMILY>` suffix** from the parent-of-
   parent of `RUN_DIR` (e.g., `sfno_zgplev_group_clone_v11`). Two runs
   with the same eval+data SHA on the same day but different training
   families now produce different RUN_TAGs by construction.
2. **Collision-guard**: if `$OUT_ROOT/provenance.txt` already exists and
   records a different `CKPT=` than the new request, abort with
   exit code 3 and a loud error pointing the user at `RUN_TAG=<unique-name>`
   override or moving the existing dir.
3. **Record `TRAIN_FAMILY=` in `provenance.txt`** so future debugging can
   immediately see which family produced a given OUT_ROOT.

Eval-sfno-own skill updated with a §RUN_TAG-collision-guard section that
points at this run-log entry.

## References

- `scripts/submit_eval.sh` (post-2026-05-12 diff: the `_family-` block and
  collision-guard).
- `.claude/skills/eval-sfno-own/SKILL.md` §RUN_TAG collision guard.
- Eval dir (post-mitigation): `…/sfno_eval/20260512_eval-8b395eb_data-e3c934b_ckpt-best_ckpt_ema_mp0/`.
- Related: [v11 EMA tas regression](2026-05-12_v11_ema_tas_regression.md) —
  the regression report that this collision initially threatened to
  misattribute.
