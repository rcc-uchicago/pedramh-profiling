# Measured: the rollout BLURS and DRIFTS — it does not accumulate small-scale energy

**2026-09-10.** Measured on the 6-hourly field subset from jesswan's long
rollout, `runs/makani_longroll/parity_7603079/parity__6hourly_subset.zarr`
(56 steps = 14 days, 1 IC, 4 sampled variables).

> ⚠ **This refutes the mechanism proposed earlier the same day** in
> `2026-09-10_fcn3_recipe_vs_ours.md` §4, which argued that the missing
> `ensemble_spectral_crps` mattered because *"uncontrolled accumulation of
> small-scale energy is the classic autoregressive blow-up mode."* Measured on
> real rollout output, the model does the **opposite**: it loses small-scale
> energy. That section's *recommendation* survives, but its *reason* was wrong.

---

## 1. What was measured

Zonal power spectrum (FFT along longitude, `cos(lat)`-weighted over latitude),
split into large scales (k=1-4) and small scales (upper half, k>=90), plus
global mean and spatial standard deviation, at every one of the 56 steps.

| variable | small/large spectral ratio, step 0 -> 55 | change |
|---|---|---|
| `TREFHT` | 0.001372 -> 0.000681 | **x0.50 — halved** |
| `TSOI_10CM` | 0.09263 -> 0.06865 | x0.74 |
| `SOILWATER_10CM` | 0.06435 -> 0.05851 | x0.91 |
| `PRECT` | 0.04312 -> 0.05245 | x1.22 (noisy, no monotone trend) |

⇒ **Three of four variables LOSE relative small-scale energy.** `TREFHT` halves
its in 14 days. This is the **blurring / mode-averaging** signature — the
MSE-optimal hedge — not energy accumulation.

## 2. The stronger finding: `SOILWATER_10CM` drifts

| variable | global mean, step 0 -> 55 | change | linear trend/step | extrapolated to 500 steps |
|---|---|---|---|---|
| **`SOILWATER_10CM`** | 8.108 -> 7.593 | **-6.36 %** | **-0.00958** | **-4.79, i.e. -59 %** |
| `PRECT` | 3.549e-08 -> 3.620e-08 | +1.99 % | +9.48e-12 | +13.4 % |
| `TREFHT` | 286.595 -> 286.309 | -0.10 % | +0.00372 | +0.6 % |
| `TSOI_10CM` | 273.873 -> 273.708 | -0.06 % | +0.00119 | +0.2 % |

`SOILWATER_10CM` decreases on **76 % of steps**. Its spatial standard deviation
simultaneously **grows +5.8 %**, so the field is drying non-uniformly — the
pattern amplifies while the mean falls.

⇒ **Soil moisture is on a large, persistent, one-directional drift.** Linear
extrapolation puts it at **-59 % of its initial value by step 500** — the step
count at which the rollout is observed to diverge.

This is the strongest blow-up candidate in the sampled data, and it is a
different failure from either blurring or spectral accumulation. **Soil moisture
is a slow reservoir**: it has long memory, changes little per step, and has no
restoring force in a single-step loss. A small per-step bias integrates.

## 3. A concrete candidate cause — and it is a config difference from upstream

`makani/utils/loss.py:181-184`:

```python
if loss.get("temp_diff_normalization", False):
    time_diff_scale = get_time_diff_stds(params).flatten()
    time_diff_scale = torch.clamp(torch.from_numpy(time_diff_scale[params.out_channels]), min=1e-4)
    time_diff_scale = scale.flatten() / time_diff_scale
```

The weight is **(climatological std) / (per-timestep change std)**. A channel
that changes *slowly per step* but varies a lot climatologically gets a **large**
weight. That is precisely the profile of a slow reservoir variable like soil
moisture — and precisely the channel we measure drifting.

| | upstream FCN3 | ours |
|---|---|---|
| `temp_diff_normalization` | **`True`** (every stage) | **`False`** |
| `channel_weights` | **`auto`** | **`constant`** |

⚠ **And ours is `False` because the statistic does not exist.** The pack's
`stats/` directory contains `global_means/stds`, `time_means`, and the three
forcing equivalents — but **no `time_diff_stds.npy`**. Our converter never
computed it, so `temp_diff_normalization: True` was not available.

⇒ Under `constant` channel weights and no temporal-difference normalization,
**slowly-varying reservoir channels are systematically under-penalised in our
loss relative to upstream's recipe** — and that is exactly where the measured
drift is.

Not proven to be the cause. But it is a specific, testable mechanism that
matches the observation, and the fix is small.

## 4. Does the spectral-loss recommendation survive?

**Yes — but for the opposite reason to the one originally given.**

A spectral loss scores the magnitude of every spherical-harmonic mode
(`SpectralCRPSLoss`, `crps_loss.py:533-538`). It penalises a **deficit** of
small-scale energy exactly as much as an excess. Countering MSE-induced
blurring is its canonical use, and blurring is what we measure.

So the term is still worth having. It just addresses §1 (blurring), **not** §2
(the drift). Nothing about a spectral penalty prevents soil moisture from
walking away.

## 5. Caveats — these matter

1. **Only 4 variables were sampled, and none is dynamically active.**
   `PRECT`, `SOILWATER_10CM`, `TREFHT`, `TSOI_10CM`. The independently measured
   fastest-degrading *fed-back* channel is **`U10`**
   (`2026-09-10_longroll_blowup_analysis.md` §2) and it is **not in this
   subset**. The blow-up may well be driven by wind/level channels that were
   never looked at.
2. **56 steps, not 500.** Early blurring does not exclude a different
   instability taking over later.
3. Zonal FFT along longitude, not the spherical-harmonic spectrum the model
   and its losses actually operate in.
4. **n=1** — one initial condition, one member.

## 6. Revised priorities

1. **Re-run the long rollout with `U10` and a level-temperature/humidity
   channel in `--sample-vars`,** and let it run to divergence. Costs one job and
   answers §5.1 and §5.2 together. Highest information per unit cost by a wide
   margin.
2. **Track `SOILWATER_10CM`'s drift all the way to the blow-up.** If it goes
   unphysical (zero/negative) before the NaN, the causal chain is established.
3. **Compute `time_diff_stds.npy` in the converter and enable
   `temp_diff_normalization: True`** (and consider `channel_weights: auto`).
   Small change, directly targets the measured drift, and closes a real gap
   against upstream's recipe.
4. Spectral loss term — still worthwhile, for blurring (§4), but demoted below
   1-3 because the drift is the larger measured effect.
5. `n_future = 3` (C2) — unchanged priority: expect accuracy, not stability
   (`2026-09-10_longroll_blowup_analysis.md` §3).

## 7. Reproduce

```bash
bash <repo>/.claude/worktrees/.../run_py.sh drift.py     # global mean / std / spectra per step
```
Scripts: `$CLAUDE_JOB_DIR/tmp/{spectra,drift}.py`. numpy + zarr only; no torch
on a login node (CLAUDE.md #3). Set `OPENBLAS_NUM_THREADS=1` — the default
64-thread OpenBLAS init exceeds the login node's process cap.
