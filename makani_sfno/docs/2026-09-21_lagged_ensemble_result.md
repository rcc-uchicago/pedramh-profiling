# The lagged ensemble, finished — and it does not improve the deterministic forecast

Stages 1–5 (tasks 11–14) of `2026-09-10_lagged_ensemble_endtoend_plan.md` are built,
verified against real output, and measured. **The plan's own check 1 fails, and it
fails decisively.** Check 4 passes. This document is the result; the machinery is
`src/sfno_ensemble/` and `polaris/score_lagged_ensemble.py`.

Jobs: **7643069** (`EVAL_INFERENCE_OK`, the sweep), **7643103** (`ALIGN_CHECK_OK`),
**7643249** (`E3SM_PORT_OK`, 100 passed / 4 skipped), **7643271**
(`LAGGED_ENSEMBLE_OK`, the scorecard).

---

## 1. What was measured

One stagger-4 sweep over `2048.h5`: 21 rollouts at K=56, 39 GB, 26.1 min on one
A100. Regrouped by absolute target index into **8 targets × 14 members**, members at
depths 4, 8, … 56 — i.e. forecasts aged **24 h to 336 h**, all valid at the same time.
Weights `w_k ∝ 1/σ(k, c)²` per channel, with σ from the *monthly* K=56 sweep's
`k56_metrics.h5`, so the weights come from an independent sample. Equiangular
quadrature throughout, 101 channels.

⚠ Only the **modal depth pattern** is scored — 8 of the 32 covered targets. The other
24 sit in the three other residue classes mod 4 and have minimum depths 1, 2, 3, i.e.
their freshest member is a 6, 12 or 18 h forecast. Those are legitimate lagged
ensembles at a *different lead*, and averaging them together would compare a 14-member
ensemble against a 6 h baseline on one target and a 24 h baseline on the next. The
first run of this scorer did exactly that, by breaking a four-way tie on encounter
order; the headline moved 9 pp when it was fixed.

**Alignment is verified, not assumed:** all 14 members of every target carry a
bitwise-identical truth field — `truth_max_disagreement = 0.000e+00`. Different valid
times in one bucket would be the silent failure that invalidates every number here,
and it is checked rather than argued.

## 2. The four checks

| # | plan §4.3 check | result | |
|---|---|---|---|
| 1 | ensemble mean beats the best single member | **NO** — 31.7 % worse | 🔴 |
| 2 | SSR ≈ 1 | **1.66** — over-dispersed, not under | ⚠ |
| 3 | rank histogram flat | not read — diagnostic only (§4.4) | — |
| 4 | CRPS beats the deterministic baseline | **YES** — 28.7 % better | ✅ |

### Check 1 fails, and the failure is monotone

| | median RMSE over 101 channels |
|---|---|
| shallowest member alone (24 h forecast) | **2.3375** |
| weighted ensemble, `w_k ∝ 1/σ²` | 3.2212 (**+31.7 %**) |
| uniform ensemble | 5.1556 (+120.6 %) |

**0 of 101 channels improved.** The weighting is doing real work — it recovers 42.1 %
against the uniform mean — but not nearly enough to reach the single freshest member.

The nested sub-ensemble sweep says why, and it is not "the old members are too old":

```
m:      1       2       3       4       5       6       7   …      14
rmse:  2.3375  2.4529  2.5770  2.7094  2.8156  2.9052  2.9835 …  3.2212
```

**Monotonically increasing. Every member added makes it worse, starting from the
second** — which is a 48 h forecast, not a stale one. 101 of 101 channels prefer
`m = 1`. There is no truncation of this ensemble that beats the single member.

The mechanism is not mysterious. Inverse-variance weighting is optimal for
**independent** errors; these members share one set of weights, one model and
overlapping initial conditions, so their errors are strongly correlated, and the
optimal combination of correlated estimators is not a positive-weight average. Combine
that with a 9× spread in member error between 24 h and 336 h and the best achievable
convex combination is close to "use the best member".

### Check 2 fails in the opposite direction to the one predicted

Plan §4.5 expects **under**-dispersion (SSR < 1) because a lagged ensemble shares one
set of weights and so samples initial-condition and rollout error but not model error.
Measured: **SSR 1.66** (raw 1.40) — clearly **over**-dispersed.

That is consistent with check 1 rather than separate from it. The spread across
members (4.378) is dominated by *error growth with lead*, not by forecast uncertainty
at a fixed lead. The construction is measuring the wrong axis: it spreads members over
forecast age, and forecast age is a deterministic property of the schedule, not a draw
from a predictive distribution.

⚠ 83 (channel, target) cells had the fair SSR correction go non-positive and are
reported NaN rather than clamped. Clamping is what makani does — correctly, for a
differentiable training-time loss — and offline it turned an undefined ratio into a
finite-looking one: an earlier run of this scorer printed a median SSR of **599581**,
which reads as a value rather than as "not defined here".

### Check 4 passes

| | median CRPS |
|---|---|
| ensemble | **1.2665** |
| shallowest member (a point forecast, so CRPS = MAE) | 1.7233 |

**28.7 % better.** As a *probabilistic* forecast the ensemble is genuinely better than
the single member it loses to on RMSE. A distribution that brackets the truth beats a
sharp point estimate under a proper scoring rule even when its mean is further off.

## 3. What this means

**The lagged ensemble does not improve makani's deterministic forecast, and no
sub-ensemble of it does.** That is a real answer to the question the plan posed, from
39 GB and about 35 minutes of A100 time, and it should stop anyone spending a 685 GB
full-year sweep on the same construction.

What it *is* good for is uncertainty: CRPS improves by 28.7 %, and there is now a
calibration number (SSR 1.66) where before there was none.

If the deterministic forecast is the goal, the evidence points away from combining
forecasts of *different ages* and toward members that differ at *the same lead* —
which is the checkpoint ensemble the 243 on-disk snapshots make possible (different
weights, same data, same lead), or a distributional objective. ⚠ Neither is implied by
this result alone, and the CRPS-arm question was separately answered in the negative by
the K=56 read-out (`2026-09-20_k56_readout_prereg.md`): mode-averaging is ruled out, so
a distributional loss is not indicated by *that* evidence either.

## 4. Limits

* **n = 1, and small.** One sweep, one holdout year (2048), 8 targets, one checkpoint.
  The 8 targets are consecutive in time and therefore heavily correlated with each
  other — they are not 8 independent draws.
* **Only the 24 h-lead ensemble was scored.** The three other residue classes were
  dropped rather than averaged in; a 6 h-lead ensemble is a different experiment and
  has not been run.
* **The nested sweep varies only ensemble size, not the weights' functional form.** It
  shows no *prefix* of the members helps; it does not prove no weighting helps. An
  inverse-*covariance* combination is the untested alternative, and it would need the
  member error covariance, which this sweep can estimate only from 8 targets.
* **`Z3_l17` is in these medians.** The K=56 read-out found the model injects 43× that
  channel's natural variance; at 1 of 101 it cannot move a median, but it is not
  excluded.
