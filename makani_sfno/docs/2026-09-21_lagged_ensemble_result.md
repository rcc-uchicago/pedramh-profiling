# The lagged ensemble, finished — and it does not improve the deterministic forecast

Stages 1–5 (tasks 11–14) of `2026-09-10_lagged_ensemble_endtoend_plan.md` are built,
verified against real output, and measured. **The plan's own check 1 fails, and it
fails decisively.** Check 4 passes. This document is the result; the machinery is
`src/sfno_ensemble/` and `polaris/score_lagged_ensemble.py`.

Jobs: **7643069** (`EVAL_INFERENCE_OK`, the sweep), **7643103** (`ALIGN_CHECK_OK`),
**7643249** (`E3SM_PORT_OK`, 100 passed / 4 skipped), **7643271**
(`LAGGED_ENSEMBLE_OK`, the scorecard).

> **§5 supersedes the framing of §2.** Checks 1–4 below are this repo's own
> RMSE/CRPS scorecard; the framework this ensemble should be scored under is **ACE2
> eq 8** (arXiv:2411.11268 §4.3), a *bias* metric on normalised fields with a plain
> unweighted ensemble mean. It is implemented and measured in §5 (job **7643591**).
> The conclusion is unchanged and considerably stronger: **−213 %**, not −31.7 %.
> §2's numbers are retained unaltered — they reproduce exactly — because the CRPS and
> SSR readings are still the only calibration evidence available.

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

---

## 5. The framework above is not the one this should be scored under

Checks 1–4 are an RMSE/CRPS scorecard of our own construction. The framework it should
follow is **ACE2, Watt-Meyer et al., arXiv:2411.11268 §4.3, eq 8**:

```
alpha = (1/C) sum_c sqrt( sum_{phi,lambda} w_{phi,lambda} ( MEAN_{t,ens}[ y_c - yhat_c ] )^2 )
```

with eq 9 the single-variable form. Three differences from §2, each of which changes
the answer rather than decorating it:

1. **The overbar is inside the square.** The time- *and ensemble*-average is taken on
   the **signed** error and only then squared. So alpha is the RMS of the *mean* error
   — a **bias** metric. Every number in §2 squares per snapshot first and so keeps the
   random component. The two differ by exactly the variance of the error about its own
   time mean, and averaging signed errors is precisely what cancels random error. **A
   construction that loses on instantaneous RMSE can still win on alpha.** That is not
   a loophole; it is why the paper scores this way.
2. **ACE2's ensemble in eq 8 is itself a lagged ensemble — combined with a plain,
   unweighted mean.** "An ensemble of eight 5-year long simulations, initialized at
   evenly spaced intervals", each contributing its prediction "for the corresponding
   time, from a simulation initialized at some previous time". There is no
   inverse-variance weighting anywhere in the paper's construction. Our `w_k ∝ 1/σ²` is
   an invention of this repo, and §2's check 1 is a verdict on *it*, not on lagged
   ensembling — which matters, because that rule is knowably wrong for correlated
   members (§4, third bullet) and it is not what we were supposed to be testing.
3. **eq 8 is defined on normalised fields**, which is what makes its arithmetic
   `(1/C) sum_c` channel reduction legal. §2 works in physical units and is therefore
   forced into a median over 101 disparate scales — a reduction that cannot see a
   single blown-up channel (`Z3_l17`, §4). The mean can, which is why ACE2 has to
   explicitly downweight `q0` by 10× rather than let it dominate alpha.

### What changed in the code

`scores.ace2_alpha` / `bias_rms_per_channel` / `time_mean_bias` implement eq 8 and
eq 9; `score_lagged_ensemble.py` reports alpha as the headline with checks 1–4 kept as
secondary. No new input file is needed: eq 8's standard-scaling **mean cancels in
`y − yhat`**, so only a per-channel sigma is required, and `a_truth_mean` in the
independent `k56_metrics.h5` is already that (it is what `score_rollout_nc.py` divides
by to form NRMSE), averaged over its 56 leads.

The driver accumulates one running field sum **per member** rather than per ensemble
rule. `member_weights` depends only on `(depth, channel)` and never on the target, so
every combination rule — uniform, `1/σ²`, any nested prefix — is an exact linear
recombination of those sums afterwards, at no extra read. That is what makes "does the
paper's uniform mean beat our weighting?" free to answer.

### Measured — job **7643591**, `scores_ace2/`

Checks 1–4 reproduce job 7643271 **exactly** (3.2212 / −31.69 % / CRPS +28.73 % /
SSR 1.6604 / the same 14-point nested curve), so alpha is an addition, not a
perturbation of what was there.

| alpha (lower is better), 101 channels, time average = 8 targets | |
|---|---|
| single freshest member — the deterministic baseline | **0.14035** |
| `1/σ²`-weighted 14-member ensemble | 0.23832 |
| **uniform 14-member ensemble — ACE2's own rule** | **0.43886** (**−212.7 %**) |

```
alpha by m (uniform):
0.1404  0.1638  0.1885  0.2133  0.2384  0.2638  0.2888  0.3124
0.3347  0.3561  0.3774  0.3982  0.4189  0.4389
```

**The negative result does not merely survive the reframing — it roughly septuples.**
Under RMSE the ensemble was 31.7 % worse than the freshest member; under the paper's
own metric it is **213 %** worse, and the curve is again monotone with `best m = 1`.

The reason is the one thing the reframing was supposed to cut the other way. Alpha
scores *bias*, and bias is exactly the component that **grows with forecast lead** and
that a time average **cannot** cancel — unlike random error, which is what averaging
signed errors does kill. So mixing in 336 h members injects systematic drift straight
into the metric. The hypothesis that "a construction that loses on instantaneous RMSE
can still win on alpha" is testable, was tested, and **is false here**.

Note also that **ACE2's uniform mean is the worse of the two combination rules** on its
own metric (0.439 vs 0.239). Our `w_k ∝ 1/σ²` invention was helping — just nowhere near
enough, and it remains the wrong estimator for correlated members either way.

⚠ **One channel owns 42 % of alpha.** `Z3_l17` scores 18.524 against 0.513 for the next
worst (`RELHUM_l05`) — the same channel the K=56 read-out flagged for 43× variance
injection. This is precisely ACE2's `q0` situation, so the driver applies the paper's
remedy as an automatic **sensitivity, never as the headline**: at 0.1× weight the
comparison becomes 0.27626 vs 0.10149, **−172.2 %, and the verdict holds**. The median
reduction used by checks 1–4 could not have surfaced this at all.

### Why it works for ACE2 and not for us — and the two are not in conflict

**ACE2 never uses a lagged ensemble to improve a forecast.** The word "lagged" does not
appear in the paper. Its *forecast* skill (§2.2.6) comes from "48 initializations
equally spaced across 2020", each scored as a **deterministic** forecast with RMSE
averaged *over* forecasts — never combined at a common valid time. The staggered-init
ensemble appears only inside eq 8, where its job is to average away **internal
variability** so that a 5-year **time-mean climate bias** can be estimated cleanly
enough to select a checkpoint. Different target, different job.

That difference is decisive, and it reduces to one inequality. For two estimators with
error magnitudes σ₁ < σ_k and correlation ρ, the optimal weight on the second is
positive — i.e. adding it helps **at all** — iff

> **ρ < σ₁ / σ_k**

*(a\* = (σ_k² − ρσ₁σ_k)/(σ₁² + σ_k² − 2ρσ₁σ_k); a\* < 1 ⟺ ρσ₁σ_k < σ₁².)*

- **ACE2's eq-8 members are 5-year simulations**, far beyond any predictability horizon,
  so every member estimates the climate **equally well**: σ₁/σ_k = 1, the threshold is
  ρ < 1, and *any* non-degenerate ensemble clears it. Ensembling is close to free.
- **Our members are forecasts at 24…336 h**, so their skill is wildly **unequal** and
  the threshold collapses. Measured, per nested step (median over 101 channels):

| m → m+1 | lead of incoming member | σ_k / √V_m | ρ measured | threshold ρ < | channels helped |
|---|---|---|---|---|---|
| 1→2 | 48 h | 1.397 | **0.740** | 0.716 | 39 % |
| 2→3 | 72 h | 1.692 | 0.689 | 0.591 | 2 % |
| 3→4 | 96 h | 1.978 | 0.648 | 0.505 | 0 % |
| 5→6 | 144 h | 2.536 | 0.556 | 0.394 | 0 % |
| 9→10 | 240 h | 3.234 | 0.443 | 0.309 | 1 % |
| 13→14 | 336 h | 3.841 | **0.418** | 0.260 | 0 % |

**The gap never closes — it widens.** The first step misses by a hair (0.740 vs 0.716),
which is why the curve turns over immediately; after that the error ratio grows far
faster than the correlation decays.

The load-bearing number is the last one. **ρ plateaus near 0.42 and does not go to
zero**, even between a 24 h and a 336 h forecast. That floor is the **shared systematic
model bias**: both members come from the same weights, so a component of their error is
identical no matter how far apart their initializations are. Averaging cannot touch it.

⚠ **This floor is NOT caused by `Z3_l17`, and an earlier draft of this section wrongly
said it was.** ρ = 0.418 is a *median over 101 channels* and one channel cannot move a
median. Measured: median ρ is **0.4183** over all channels and **0.4149** with `Z3_l17`
removed, and **101/101** channels fail the `ρ < σ₁/σ_k` threshold — **100/100** without
it. So there are **two separate problems** here, related in kind but not the same defect:

* **A — shared systematic bias in essentially every channel.** The ρ≈0.42 floor. This is
  what kills the lagged ensemble, it is a property of averaging one deterministic model
  against itself, and **no single-channel fix touches it.**
* **B — `Z3_l17` specifically**, which is the most extreme instance of A: the highest ρ
  of any channel (0.779, against a 90th percentile of 0.604), 43× drift at 336 h, and
  41.8 % of alpha where the other 100 channels have a median alpha of 0.26. It drives
  the `DRIFT_FIRST` verdict and dominates alpha.

**Fixing B would not rescue the ensemble**, and that is measured rather than argued: at
ACE2's 10× downweight the ensemble is still **−172 %** against the single member.

This also explains why re-scoring under alpha made things *worse* rather than better.
Averaging removes the independent part of the error and leaves the shared part;
alpha measures precisely the shared part. Asking a same-model ensemble to reduce alpha
is asking it to cancel the one component it structurally cannot — and old members
contribute *more* of it, because bias grows with lead. ACE2 does not make that mistake:
it uses the ensemble to *measure* the bias, then changes the checkpoint.

**What follows.** Ensembling helps here only where members are of comparable skill. That
points at a **fixed-lead** ensemble — the 243 on-disk checkpoints, or seeds — where
σ₁/σ_k ≈ 1 restores the threshold to ~1. It also means ACE2's eq-8 construction is
directly reproducible for us, but as a *climate* statistic over long rollouts from
well-separated ICs, not over 8 targets in one week.

### What this does *not* fix

* **The time average is 8 targets over 7 days.** ACE2 computes alpha over eight
  **5-year** simulations. A bias metric is exactly the kind of statistic that needs a
  long time average — the random component falls off as `1/sqrt(n)`, and at `n = 8` a
  good fraction of what alpha reports as "bias" is still sampling noise. Alpha over 8
  consecutive targets is a weak estimate and must not be read as ACE2 reads it.
* Checks 2 and 3 remain unfixed and unfixable in this construction: SSR and the rank
  histogram both assume exchangeable members, and lagged members are ordered by
  construction. §2 disclaims the rank histogram on exactly this ground and then reports
  SSR anyway; both should be read as diagnostics, not calibration.
