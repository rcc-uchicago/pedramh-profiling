# v11 EMA tas regression — not an EMA bug — 2026-05-12

**Run / RUN_TAG / EXP_DIR:** `$SCRATCH/AI-RES/runs/sfno_zgplev_group_clone_v11/plasim_sim52_zgplev_group_clone_v11/0`
**Eval RUN_TAG:** `20260512_eval-8b395eb_data-e3c934b_ckpt-best_ckpt_ema_mp0`
**Author / debugger:** Zhixing Liu / Claude
**Date:** 2026-05-12
**Status:** root-caused (hypothesis); A/B `v11_clip` queued for confirmation
**Severity:** regression (own-track headline metric)

## TL;DR

v11 EMA eval shows tas@6h ACC ≈ 0.919, vs ≈ 0.995 in `group_clone_nonoise`
and ≈ 0.987 in older v10. Checkpoint inspection ruled out an EMA
implementation bug: raw vs EMA `model_state` keys match 128/128, per-layer
relative L2 delta ≤ 0.15%, raw vs EMA validation loss within 0.16%. Leading
hypothesis: `optimizer_max_grad_norm=0` + `input_noise.sigma=0.05`
interaction. Mitigation: `v11_clip` variant restores clip=32 (see plan
`docs/2026-05-12_v11_clip_restore_plan.md`).

## Symptoms

ACC at lead 6h / 24h / 336h (v11 EMA vs v11_nonoise raw vs v10 group_clone raw):

| Channel | v11 EMA @ 6h | v11_nonoise @ 6h | v10 group_clone @ 6h |
|---------|-------------:|-----------------:|---------------------:|
| tas     | **0.919**    | 0.995            | 0.987                |
| zg500   | 0.998        | 0.999            | 0.998                |
| ua5     | (comparable) | (comparable)     | (comparable)         |
| ta5     | (comparable) | (comparable)     | (comparable)         |
| pr_6h   | (comparable) | (comparable)     | (comparable)         |

Tas-only, near-channel-isolated regression — zg500/ua5/ta5/pr_6h essentially
unaffected. Drop persists at 24h (0.669 vs 0.951) and 72h (0.515 vs 0.871).

Source scorecards:
- v11 EMA: `…/sfno_eval/20260512_eval-8b395eb_data-e3c934b_ckpt-best_ckpt_ema_mp0/scores/nwp_scorecard_summary.csv`
- v11_nonoise raw: `…/sfno_eval/20260511_eval-8b395eb_data-e3c934b/scores/nwp_scorecard_summary.csv`
- v10 group_clone raw: `…/sfno_eval/20260510_eval-8b395eb_data-e3c934b/scores/nwp_scorecard_summary.csv`

## What we checked (and ruled out)

1. **EMA shadow coverage** — `model_state` has 128 keys (116 fp32 + 12
   complex64). The EMA shadow (`ema_state` embedded in the raw best
   checkpoint) has 128 matching keys. No buffer slips through; the model has
   no persistent buffers (instance_norm = no running stats, SHT tables are
   non-persistent). EMA covers every trainable parameter. ✅
2. **State_dict key diff between raw and EMA best** — identical sets,
   identical shapes, identical dtypes. ✅
3. **Per-parameter Frobenius / Linf delta** — max single-element |raw−EMA| =
   9.59e-4; max per-layer relative L2 delta = 0.15% (block-2 MLP fc-3 bias).
   Output head (`model.decoder.fwd.2.weight`) per-channel row L2 for tas
   (idx 1): raw 0.3733 vs EMA 0.3732, Δ = 0.03% — smaller than ua5's or
   pr_6h's relative delta. ✅
4. **EMA validation loss tracking raw** — at the EMA-best save point (ep47),
   raw val L2 = 3.166e-3 vs EMA val L2 = 3.161e-3 (Δ = 0.16%). EMA was
   consistently lower than raw throughout training. ✅
5. **Save/load round-trip** — `ema_state` in raw best at ep50 vs
   `model_state` in EMA best at ep48 differs by relative L2 = 4.5e-4, the
   expected drift over ~36k optimizer steps with decay 0.999. ✅
6. **Inference loader path** — `Driver._restore_checkpoint_legacy(...,
   strict=True)` succeeds on both files (no missing/unexpected keys).
   `model.inp_chans == 58`, `model.out_chans == 53` after load. ✅

A 7+ pp ACC drop at the **first** forecast step (6h, one forward pass)
cannot be produced by weights that differ by ≤ 0.15% per layer with no
tas-row anomaly. So the regression is not coming from the EMA snapshot
itself.

## Root cause (hypothesis)

v11's training config diverged from the previous stable recipe in **two**
places — only one of which is intentional:

| Knob                     | v11_nonoise | v11    | Note                  |
|--------------------------|------------:|-------:|-----------------------|
| `input_noise.sigma`      | 0.0         | 0.05   | intentional           |
| `optimizer_max_grad_norm`| 32.0        | **0.0**| disabled in v11       |

`input_noise σ=0.05` is the empirically validated long-lead stabilizer
(removing it cost 21–33% ACC at 336h in the 2026-05-11 `nonoise` eval). It
adds Gaussian noise on the 52 state channels at the input.

`optimizer_max_grad_norm=0` disables gradient clipping. v11 inherited this
from a different lineage where peak observed grad norm was ≈ 2.5 (well below
the 32 threshold), so disabling it looked free.

**Leading hypothesis (UNCONFIRMED until v11_clip lands):** input_noise can
occasionally amplify the gradient on perturbation-sensitive directions, and
without clipping a single bad step can shove the weights along a
high-curvature ridge. The effect is most visible on high-variance surface
fields (tas), where the next-state prediction is most sensitive to a
mis-tuned downstream of normalization layers. Mid-troposphere channels
(ua5, ta5, zg500) average out spatial noise more and are less affected.

## Fix / next experiment

- New variant `group_clone_v11_clip`:
  - `src/sfno_training/config/plasim_sim52_zgplev_group_clone_v11_clip.yaml`
  - `src/sfno_training/submit_zgplev_group_clone_v11_clip.slurm`
  - `docs/2026-05-12_v11_clip_restore_plan.md` for the full A/B + decision criteria.
  - Single knob delta: `optimizer_max_grad_norm: 0.0 → 32.0`.
  - Keeps `input_noise.sigma=0.05` (do NOT remove — see
    [[feedback-input-noise-is-load-bearing]]).

Decision rule when `v11_clip` eval lands:
- tas@6h ≥ 0.99 AND zg500@336h ≥ v11 → hypothesis confirmed, adopt as canon.
- tas@6h still ≤ 0.93 → reopen, this entry stays open with a follow-up A/B.

## References

- v11 training run: `$SCRATCH/AI-RES/runs/sfno_zgplev_group_clone_v11/plasim_sim52_zgplev_group_clone_v11/0/`
- v11 EMA eval: `…/sfno_eval/20260512_eval-8b395eb_data-e3c934b_ckpt-best_ckpt_ema_mp0/`
- EMA implementation plan: `docs/2026-05-02_ema_implementation_plan.md`
- v11 clip-restore plan: `docs/2026-05-12_v11_clip_restore_plan.md`
- Related: [v11 / gbhpo40 RUN_TAG collision](2026-05-12_v11_gbhpo40_run_tag_collision.md)
  — the original report cited the wrong scorecard until provenance was
  clarified; the numbers above are confirmed from the actual v11 EMA NCs.
- Memory: [[feedback-ema-is-canonical-ckpt]], [[feedback-input-noise-is-load-bearing]],
  [[project-v11-clip-experiment]].
