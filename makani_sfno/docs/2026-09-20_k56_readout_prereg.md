# PRE-REGISTRATION — reading the K=56 (14-day) curve

Written **2026-09-20, before any of job 7633207's output was scored.** The 24
NetCDFs have existed since 2026-09-19 00:02; nothing in this document was chosen
after looking at a number derived from them. That is the whole point of writing
it: the K=56 sweep is the decision point between two arms that cost very
different things, and the temptation to pick the threshold that gives the
answer you already like is exactly what a pre-registration exists to remove.

Companion to `2026-09-10_lagged_ensemble_endtoend_plan.md` §4.5 (which failure
mode this separates) and `polaris_makani_accuracy_handoff.md` §3 (the original,
coarser statement of the rule). Where the two differ, this one is the operative
version — it replaces "saturating / still climbing", read by eye, with numbers a
script computes.

---

## 1. What is being decided

At lead 126 h the rollout's RMSE has grown ×4.65 from 6 h, **linearly, with no
saturation**, and ACC is **0.878** — nowhere near climatology. At that horizon
**exposure bias** (the model has not seen enough of its own output) and
**mode-averaging** (MSE training produces the conditional mean, so the forecast
blurs) are not separable, and they point at opposite fixes:

| dominant failure | next arm | cost |
|---|---|---|
| mode-averaging / blurring | **CRPS** — `PlasimEnsembleTrainer`, `polaris/e3sm_alldata_crps.yaml` | ⚠ changes the loss ⇒ **jesswan's sign-off** (CLAUDE.md, division of labor) |
| exposure bias | **`n_future=8`** | memory audit first — depth 4 at global 16 is already 22.29 GiB/GPU at local 2 |

K=56 (336 h) is twice the horizon and is inference-only: the members are already
paid for.

## 2. What is measured

Per channel `c` (all 101 — a headline subset is jesswan's choice, not the
scorer's) and per lead `k = 1…56` (lead hours `6k`), averaged over the 24 ICs:

| symbol | definition |
|---|---|
| `RMSE(k,c)` | latitude-weighted RMSE, **equiangular** weights, physical units |
| `ACC(k,c)` | anomaly correlation against `stats/time_means.npy` |
| `A_pred(k,c)` | predicted anomaly amplitude, `sqrt(Σ_lat w · mean_lon (pred − clim)²)` |
| `A_truth(k,c)` | the same for truth — the channel's own anomaly scale |
| `NRMSE(k,c)` | `RMSE / A_truth` — dimensionless, so 101 channels can be pooled |
| `VR(k,c)` | `A_pred / A_truth` — the **variance ratio**, i.e. the blurring diagnostic |
| `bias(k,c)` | latitude-weighted mean of `pred − truth` |

Headline aggregate is the **median over the 101 channels**, not the mean: the
channels are on wildly different scales and a mean would be whatever the worst
few do.

**Climatology.** `stats/time_means.npy` is `(1, 101, 180, 360)` float32 in raw
physical units — the per-cell time-mean over the **training** split, written by
`convert_e3sm_to_makani_alldata.py`'s stats pass. It is the same climatology
makani's own validation ACC uses, which is what makes these numbers comparable
with the existing 126 h curve.

⚠ **Stated before the fact, because it bounds what ACC can prove here:** that
file is a *single annual* time-mean, not a calendar-binned one. The anomaly it
defines therefore still contains the seasonal cycle, so **ACC is biased high** —
a model that merely reproduced the seasonal march would score well above 0. ACC
is consequently the **secondary** discriminator in §3; `VR` and `NRMSE` are
primary, and neither has that defect.

**Quadrature.** Equiangular (`lat_weights(180, "equiangular")`), never
Gauss-Legendre. Change G measured GL over-weighting the polar row by **1.50×** on
this grid, and the only guard downstream is a shape check that 180-vs-180
passes. `scripts/score_nwp.py` still calls `legendre_gauss_lat_weights` directly
and is **not** the scorer used here.

## 3. The decision rule — fixed now

Let `R_slope = slope(240→336 h) / slope(30→126 h)` on the median NRMSE curve,
and let `VR336`, `NRMSE336`, `ACC336` be the 336 h medians.

**Three reference levels, which are the reason this rule can be numeric at all.**
If the forecast decays to the climatological *mean* (blurring), `NRMSE → 1.0`
and `VR → 0`. If it decorrelates but keeps realistic amplitude, `NRMSE → √2 =
1.414` and `VR → 1`. Above ≈1.6 it is neither — that is drift.

| test | verdict A | verdict B |
|---|---|---|
| **Blurring** (primary) | `VR336 ≤ 0.60` ⇒ amplitude collapsed | `VR336 ≥ 0.85` ⇒ amplitude preserved |
| **Level** | `NRMSE336 ≤ 1.15` ⇒ at the conditional-mean level | `NRMSE336 ≥ 1.30` ⇒ past it |
| **Saturation** | `R_slope ≤ 0.25` ⇒ saturating | `R_slope ≥ 0.60` ⇒ still climbing |
| **ACC** (secondary) | `ACC336 ≤ 0.30` ⇒ decaying to 0 | `ACC336 ≥ 0.50` ⇒ well above 0 |

Branches, evaluated in this order:

1. **DRIFT_FIRST** — `NRMSE336 > 1.6`, or any channel's median NRMSE > 3.
   The rollout is not merely losing skill, it is leaving the attractor. Neither
   arm is the right next spend; find the drift.
2. **CRPS** — `VR336 ≤ 0.60` **and** `NRMSE336 ≤ 1.15`.
   The forecast has collapsed onto the climatological mean. That is
   mode-averaging, and no amount of rollout depth or deterministic ensembling
   repairs it (plan §4.5). ⇒ raise the loss change with jesswan.
3. **NFUTURE8** — `VR336 ≥ 0.85` **and** (`R_slope ≥ 0.60` **or**
   `NRMSE336 ≥ 1.30`).
   Error still growing with realistic amplitude ⇒ exposure bias ⇒ more depth.
4. **AMBIGUOUS** — anything else. Report both curves, take them to jesswan, and
   **do not** spend node-hours on a coin flip. An ambiguous read is a real
   result, not a failure of the measurement.

`ACC336` does not gate any branch; it is reported alongside and must be
*consistent* with the branch taken. An inconsistency (e.g. CRPS fires while
`ACC336 ≥ 0.5`) is itself a finding and downgrades the result to AMBIGUOUS
pending explanation.

This rule is implemented as `classify_regime()` in
`polaris/score_rollout_nc.py` and pinned by `polaris/test_score_rollout_nc.py`,
so it cannot quietly move after the numbers are in.

## 4. What the same pass also delivers

`RMSE(k,c)` over `k = 1…56` **is** the `σ(k)` that the lagged ensemble's
combination stage needs — `w_k ∝ 1/σ(k)²`, plan task 13. Today's curves stop at
lead 20, so depths 21–56 have no weights at all. The scorer writes the full
`(56 × 101)` arrays to `k56_metrics.h5` in the form task 13 consumes, whichever
branch §3 selects.

## 5. Limits, stated up front

* **n = 1.** One sweep, 24 ICs, 2 test years (2048, 2049), one checkpoint. No
  repeats.
* The 12 ICs within a year are 117 samples apart — plenty for lead-dependence,
  **not** 24 independent draws of the climate.
* `PRECT` is the diagnostic channel and is expected to behave unlike the 100
  state channels; it is reported separately. At 1 of 101 it cannot move a median.
* Channel bookkeeping: the loader reports the pack as `180 × 360 × 100` while the
  contract is quoted as 101. The scorer asserts `len(channel) ==
  time_means.shape[1]` and fails loud on a mismatch, which settles handoff §7
  either way.

---

# RESULT — job 7639537, 2026-09-20

Everything above was fixed before this section existed. `K56_SCORE_OK`, 24 ICs ×
56 leads × 101 channels, **0 non-finite cells**, 22.8 min CPU-only on one debug
node. Outputs in `runs/makani_eval/prod1n_b32_sgdr_K56/scores/`.

| quantity (median over 101 channels) | 126 h | **336 h** |
|---|---|---|
| NRMSE (1.0 = climatological mean, 1.414 = decorrelated) | 0.519 | **0.970** |
| VR = predicted / true anomaly amplitude | 0.999 | **1.019** |
| ACC (secondary — annual-mean climatology flatters this) | 0.867 | **0.546** |
| `R_slope` (240→336 h) / (30→126 h) | — | **0.473** |

## Branch, as the rule states it: **DRIFT_FIRST**

It fired on the worst-channel clause — `max_channel_nrmse336 = 43.18 > 3.0` —
and on **exactly one** channel of 101. Every other channel is ≤ 1.371 at 336 h.
The median NRMSE, 0.970, is nowhere near the 1.60 global-drift threshold.

Had that clause not fired the verdict would have been **AMBIGUOUS**: `VR336 =
1.019 ≥ 0.85` satisfies NFUTURE8's amplitude test, but neither of its triggers
holds (`R_slope 0.473 < 0.60`, `NRMSE336 0.970 < 1.30`).

## What is nonetheless established, and it is the answer to the CRPS question

**Mode-averaging is ruled out.** Blurring means amplitude collapse, and at 14
days the model's anomaly amplitude is not merely preserved but 2 % *above*
truth's (VR 1.019, against ≤ 0.60 for the CRPS branch). Across leads VR never
leaves 0.99–1.02. Whatever the model is getting wrong at 336 h, it is not
producing the conditional mean. **A distributional loss is not indicated by this
evidence** — which is a real result for a question that had none at 126 h.

What K=56 did **not** deliver is the saturation regime: NRMSE 0.970 is still
short of 1.0, let alone 1.414, and the curve is decelerating (`R_slope` 0.473,
continuing the 0.779 late/early ratio measured at lead 21 in
`2026-09-10_longroll_blowup_analysis.md` §2). 336 h is still the early regime.
Doubling the horizon moved us from "29 % of the ceiling" to roughly 69 %.

## The one channel: `Z3_l17`

| | `Z3_l17` | `Z3_l16` | `Z3_l00` |
|---|---|---|---|
| truth anomaly amplitude `a_truth` | **0.271 m** | 12.83 m | 563.1 m |
| predicted amplitude at 336 h | **10.22 m** | 16.19 m | 576.6 m |
| VR at 6 h → 336 h | **5.06 → 43.16** | ~1.0 | ~1.02 |
| RMSE at 6 h → 336 h | 1.23 → 10.22 m | 1.77 → 13.01 m | 16.4 → 191.8 m |
| bias at 6 h → 336 h | −0.11 → **−5.15 m** | — | — |

`l17` is the near-surface terrain-following level, so its geopotential height is
nearly time-invariant — 27 cm of temporal variability about the per-cell mean.
**The model manufactures 1.27 m of variance in a single step and 10.2 m by 14
days, and half the final error is a systematic −5.15 m drift.**

Two things this is *not*. It is not a small-denominator artifact alone: the
numerator, `a_pred`, genuinely grows 40× while truth's stays flat. And it is not
large in absolute terms — 10.2 m is the *smallest* 336 h RMSE of the Z3 family.

⚠ It was invisible to the earlier analysis by construction: that one normalised
by `global_stds.npy`, which for `Z3_l17` is dominated by **spatial** terrain
variation and therefore hides a temporal drift completely. This scorer
normalises by per-cell temporal anomaly amplitude, which is the forecast-skill
question.

`Z3_l17` is a fed-back state channel, so it is a candidate seed for the ~500-step
blow-up — a different candidate from `U10`, which the long-roll analysis named on
the `global_stds` normalisation. ⚠ **Channel semantics are science-owned**
(CLAUDE.md, division of labor): whether a near-constant near-surface geopotential
level should be prognostic at all is jesswan's call, not ours.

## A defect in this pre-registration, disclosed rather than patched

§5 says `PRECT` "is reported separately" because a diagnostic channel behaves
unlike the 100 state channels, and argues that at 1 of 101 it cannot move a
median. That reasoning covers the medians and **not** the drift clause, which is
a max over all 101 channels — so any channel with a degenerate denominator can
select the branch on its own, which is what happened. The rule is left exactly as
written and the verdict is reported as it came out; **changing the clause now,
having seen which channel trips it, would forfeit the point of writing it down
first.** The next pre-registration should state the drift clause over channels
whose `a_truth` clears a floor, and report the excluded ones by name.

## Also delivered

`k56_metrics.h5` carries RMSE for all **56** depths × 101 channels — the `σ(k)`
that the lagged ensemble's weighted combination needs (`w_k ∝ 1/σ(k)²`, task 13).
Depths 21–56 had no weights before today.
