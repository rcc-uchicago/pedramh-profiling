# v11 + grad-clip-restored training plan (v11_clip)

**Status:** proposed, ready to submit. Single-knob A/B against
`plasim_sim52_zgplev_group_clone_v11`.
**Author:** Zhixing Liu / Claude (post-EMA-debug session)
**Date:** 2026-05-12

---

## 1. Why this run exists

v11's eval (using `best_ckpt_ema_mp0.tar`) showed a sharp **tas-specific** ACC
regression at short leads — tas@6h ACC ≈ 0.919 vs ≈ 0.995 in
`group_clone_nonoise` and ≈ 0.987 in the older v10 `group_clone`. zg500, ua5,
ta5, pr_6h were comparable. The user asked whether the EMA implementation was
to blame.

Debug findings (full notes in the 2026-05-12 session transcript):

1. **Checkpoint structural diff** — `best_ckpt_mp0` and `best_ckpt_ema_mp0`
   `model_state` dicts have identical 128-key sets, identical shapes, identical
   dtypes (116 fp32 + 12 complex64). All 128 keys are in the EMA shadow — no
   buffer is left unshadowed (instance_norm has no running stats; SHT tables
   are non-persistent buffers).

2. **Per-parameter delta** — max single-element |raw−EMA| = 9.59e-4; max
   per-layer relative L2 delta = 0.15% (in `model.blocks.2.mlp.fwd.3.bias`).
   Output head `model.decoder.fwd.2.weight` per-channel row L2 for tas (idx 1):
   raw 0.3733 vs EMA 0.3732 — Δ = 0.03%, *smaller* than ua5's or pr_6h's
   relative delta. No tas-specific weight anomaly.

3. **Validation loss trajectory** — at the EMA-best save point (ep47),
   raw val L2 = 3.166e-3, EMA val L2 = 3.161e-3 (Δ = 0.16%). A 7+pp ACC drop
   at the first 6h forecast step cannot be explained by a 0.16% normalized-L2
   weight difference.

4. **Provenance hygiene** — the `_snapshot_v11_pre_collision_resolve_20260512`
   directory's `provenance_currently_chain1.txt` revealed a near-collision
   with a separate `sfno_zgplev_gbhpo40_gb16_lr2_0e-4_20260511` chain on the
   same RUN_TAG. The inference NCs in `inference/nwp/` carry the v11 EMA
   `ckpt_path` attribute, so the 0.919 number IS from v11 EMA (not gbhpo40),
   but the collision class is real and worth fixing. Addressed in
   §3 below.

**Conclusion: the EMA implementation is sound. The tas regression is a
training-recipe artifact.**

## 2. Hypothesis under test

The v11 recipe disabled gradient clipping (`optimizer_max_grad_norm: 0.0`)
while keeping `input_noise.sigma: 0.05` on the 52 state channels. The
`group_clone_nonoise` run was identical except (a) input noise disabled and
(b) `optimizer_max_grad_norm: 32.0` retained.

Two candidate causes for the tas regression:
- (A) input_noise σ=0.05 itself smooths first-step predictions toward the
  conditional mean, hurting high-variance surface fields.
- (B) the **combination** of input_noise + no-clip lets occasional
  perturbation-amplified gradient spikes contaminate training, especially in
  layers that drive the surface (tas) output.

The 2026-05-11 `group_clone_nonoise` eval already showed that removing input
noise outright degrades **long-lead** ACC by 21–33% at 336 h — input noise is
the empirically validated long-lead stabilizer and we do not want to remove
it. So this plan tests hypothesis (B) by changing **only grad clip**.

If `v11_clip` recovers tas@6h to ~0.99 with long-lead ACC comparable to v11,
the bug was (B) — interaction of input noise and unclamped gradients. If
`v11_clip` still shows the tas regression, then (A) is the real cause and we
need a different mitigation (e.g. lower sigma, or a per-channel mask).

## 3. Changes

### 3.1 New training config (single knob change vs v11)

**File:** `src/sfno_training/config/plasim_sim52_zgplev_group_clone_v11_clip.yaml`

```diff
- optimizer_max_grad_norm: 0.0      # disabled
+ optimizer_max_grad_norm: 32.0     # restored (matches group_clone_nonoise)
```

All other knobs — `input_noise.sigma: 0.05`, `target: state`, `batch_size: 8`,
`lr: 1e-4`, EMA decay 0.999 + warmup, betas (0.9, 0.999), 50 epochs — held
fixed.

### 3.2 New submit slurm

**File:** `src/sfno_training/submit_zgplev_group_clone_v11_clip.slurm`

Mirror of `submit_zgplev_group_clone_v11.slurm`. Defaults:
- `OUTPUT_ROOT`: `$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full_v11` (same
  v11 packager output as the original v11 run; this is the SST-handling
  dataset we are training on).
- `EXP_DIR`: `$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_group_clone_v11_clip` (fresh
  dir, never reuses the v11 EXP_DIR — protect-prior-runs).

### 3.3 Eval-pipeline fixes

`scripts/submit_eval.sh` now:
- Defaults `CKPT` to `best_ckpt_ema_mp0.tar` when present (EMA is canonical),
  falling back to `best_ckpt_mp0.tar` only when EMA is absent.
- Auto-includes `_family-<train_family>` in RUN_TAG so two evals against
  different training runs at the same SHAs land in distinct `OUT_ROOT`s.
- Aborts with a loud error when an existing `$OUT_ROOT/provenance.txt`
  records a different `CKPT=` than the one being submitted.
- Records `TRAIN_FAMILY=` in `provenance.txt`.

`.claude/skills/eval-sfno-own/SKILL.md` updated to reflect the EMA-as-default
policy (the prior "always best_ckpt_mp0" instruction was incorrect and is
removed).

## 4. Eval plan after training finishes

1. Submit the canonical `v11_clip` eval (EMA-best, NWP mode, 8-yr ×
   12-IC × K=56 scorecard):
   ```bash
   RUN_DIR=$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_group_clone_v11_clip/plasim_sim52_zgplev_group_clone_v11_clip/0 \
     scripts/submit_eval.sh
   ```
   This will resolve to `best_ckpt_ema_mp0.tar` automatically.

2. Diagnostic raw-vs-EMA control (optional, only if EMA result is surprising):
   ```bash
   RUN_DIR=...v11_clip.../0 \
     CKPT=...v11_clip.../training_checkpoints/best_ckpt_mp0.tar \
     scripts/submit_eval.sh
   ```
   Lands in a separate `_ckpt-best_ckpt_mp0`-tagged OUT_ROOT.

3. Decision criteria:
   - **tas@6h ACC ≥ 0.99 AND zg500@336h ACC ≥ v11**: hypothesis (B) confirmed,
     adopt `v11_clip` as the new canonical recipe.
   - **tas@6h ACC stays ≤ 0.93**: hypothesis (A) confirmed, the input-noise
     itself is the root cause. Open a follow-up to test reduced σ (e.g. 0.01)
     or per-channel masking that excludes the surface row.

## 5. Backout

Trivial. The v11 run and its eval are untouched. The new YAML / slurm sit
alongside the existing v11 files. If `v11_clip` regresses on long leads
(unlikely — clip=32 never observed to fire in `nonoise` training), revert to
v11 by re-pointing scripts at the v11 EXP_DIR.

## 6. References

- 2026-05-02 EMA implementation plan: `docs/2026-05-02_ema_implementation_plan.md`
- v11 SST migration plan: `docs/2026-05-10_sst_sea_ice_handling_fix_plan.md`
- v11 nonoise eval (2026-05-11): `…/sfno_eval/20260511_eval-8b395eb_data-e3c934b/`
- v11 EMA eval (2026-05-12): `…/sfno_eval/20260512_eval-8b395eb_data-e3c934b_ckpt-best_ckpt_ema_mp0/`
- Eval-sfno-own skill: `.claude/skills/eval-sfno-own/SKILL.md`
