# ACE2 does 7300-step stable rollouts at the SAME 2-step training depth. The difference is a corrector.

**2026-09-10.** Prompted by the observation that makani's `n_future = 1` should
be comparable to ACE2's two 6-hourly training steps. It is — exactly — and
following that comparison answers the long-rollout blow-up.

---

## 1. The comparison holds exactly

| | training rollout window | equivalent |
|---|---|---|
| ACE2 (`fme`) | `stepper_training.n_forward_steps: 2` | 2 forward passes |
| makani C1 | `multistep_count: 2` -> `n_future: 1` | 2 forward passes |

**Identical.** `n_forward_steps: 2` appears in every ACE2 baseline config
checked — `config_polaris.yaml:119`, `baselines/era5/ace-train-config.yaml:68`,
`baselines/amip-c96-shield/train-ace2.yaml:76`,
`baselines/shield-som/ace-train-config.yaml:79`.

## 2. And ACE2 runs 7300 steps stably at that depth

From the same configs: `inference.n_forward_steps: 7300` — 7300 x 6 h = **5
years**. The SHiELD-SOM config uses **10220 steps (7 years)**.

⇒ **This is an existence proof.** A 2-step training window is sufficient for
multi-year stable rollouts. So the ~500-step divergence in `longroll.py` is
**not** caused by insufficient rollout-training depth, and increasing
`n_future` is not the fix.

This independently confirms what was measured earlier the same day: going from
`n_future` 0 -> 1 improved accuracy 3-4.7 % at every lead but left every
stability statistic unchanged (`2026-09-10_longroll_blowup_analysis.md` §3).
Two independent lines of evidence, same conclusion.

## 3. What ACE2 has that makani has ZERO of

`config_polaris.yaml:169-188` — part of the **stepper**, so it is applied at
**every step of inference as well as training**:

```yaml
corrector:
  conserve_dry_air: true
  moisture_budget_correction: advection_and_precipitation
  force_positive_names:
  - specific_total_water_0 ... specific_total_water_7   # all 8 water species
  - Q2m
  - PRATEsfc                                            # precipitation
  - ULWRFsfc, ULWRFtoa, DLWRFsfc, DSWRFsfc, USWRFsfc, USWRFtoa   # radiative fluxes
```

Three enforced constraints, every step:
1. **Global dry-air-mass conservation**
2. **Global moisture budget correction** (advection and precipitation)
3. **Hard non-negativity** on 16 named variables

**Searched the entire makani upstream checkout for `corrector`,
`conserve_dry_air`, `force_positive`, `moisture_budget`: ZERO hits across every
`.py` and `.yaml`.** makani has no conservation machinery, no positivity
clamping, and no budget correction, anywhere.

## 4. This matches the measured failure exactly

`2026-09-10_rollout_spectra_and_drift.md` measured, on the real rollout:

> `SOILWATER_10CM` global mean falls **6.36 % in 14 days**, decreasing on 76 %
> of steps, linear trend extrapolating to **-59 % by step 500** — the step count
> at which the rollout diverges.

A slow water reservoir walking monotonically toward zero is **precisely** the
failure mode that `conserve_dry_air`, `moisture_budget_correction` and
`force_positive_names` exist to prevent. ACE2 clamps every water species
non-negative and closes the global moisture budget at every step. makani lets
soil moisture drift wherever the network sends it, with nothing to stop it going
to zero or negative.

⇒ **Candidate causal chain, now supported from two directions:** single-step
MSE gives a slow reservoir variable no restoring force -> it accumulates a small
per-step bias -> over hundreds of steps it goes unphysical -> the land-surface
coupling feeds garbage into temperature and precipitation -> divergence.

## 5. Two further differences in the same config

| | ACE2 | ours |
|---|---|---|
| loss channel weights | explicit per-channel dict spanning **20x** (`h500: 10`, `TMP850: 5`, `ULWRFsfc: 5`, most 0.5-2, `specific_total_water_1: 0.25`) | `channel_weights: constant` (all equal) |
| normalization | **dual** — `network` uses `scaling-full-field.nc`, `residual` uses `scaling-residual.nc` | single full-field; `temp_diff_normalization: False` |

The second is the same gap flagged in
`2026-09-10_rollout_spectra_and_drift.md` §3: ACE2 carries a **separate
residual/tendency scaling**, which is what makani's `temp_diff_normalization`
provides and which we have off because our converter never produced
`time_diff_stds.npy`.

⇒ Both ACE2 and FCN3 up-weight or re-scale by tendency. **We are the only one of
the three that does neither.**

## 6. Why this is good news

**The corrector is an inference-time mechanism.** It lives in the stepper and
runs on every forward step, so a minimal version can be tested in
`longroll.py` **with zero retraining**:

> Clamp the non-negative-by-physics channels after each step — `SOILWATER_10CM`,
> `PRECT`, `RHREFHT`, the humidity levels — and re-run to divergence. If the
> blow-up step moves substantially, the causal chain in §4 is confirmed and the
> fix is scoped.

That is one job, no training, and it discriminates between the remaining
hypotheses better than anything else available.

## 7. Revised priority order

1. **Minimal inference-time corrector in `longroll.py`** (positivity clamp on
   the physically non-negative channels). One job, no retraining. ← **do this
   first**
2. **Re-run the long rollout sampling `U10` and a level channel**, to
   divergence — the four sampled variables include none of the dynamically
   active ones (§5.1 of the drift doc).
3. **Compute `time_diff_stds.npy`; enable `temp_diff_normalization`;
   reconsider `channel_weights`.** Closes the gap against *both* ACE2 and FCN3.
4. Spectral loss term — for blurring, which is real but is not the divergence.
5. `n_future = 3` — **demoted further.** §2 is an existence proof that depth is
   not the lever.

## 8. Caveats

- ACE2 and makani are different architectures on different data; "ACE2 does X"
  is a strong prior, not proof that X fixes makani.
- The causal chain in §4 is a hypothesis consistent with two measurements, not
  a demonstrated mechanism. §6 is the test.
- ACE2's corrector operates on a variable set with an explicit moisture budget
  (`specific_total_water_*`, `PRATEsfc`, advection tendency). Our 101-channel
  E3SM contract is not obviously budget-closable in the same way — a positivity
  clamp ports trivially, a moisture-budget correction may not.
