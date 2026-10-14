# Why the 500-step rollout blows up — what the error curve says

**Question from jesswan, 2026-09-10:** long rollouts diverge beyond ~120 days
(~500 steps). Was `n_future = 0` the training default, and could that be the
cause?

**Short answers.** Yes, `n_future = 0` — confirmed. Yes, it is a real
contributor — measured. **But `n_future = 1` does not fix it**, and that is the
non-obvious part.

Evidence: the per-lead curves measured 2026-09-10 for both checkpoints
(jobs 7603089 / 7603119, 21 leads x 101 channels), against the pack's own
`global_stds.npy`.

---

## 1. `n_future = 0` is confirmed

From `prod1n_b32_sgdr/config.json`: `n_future = 0`. The 243-epoch base
checkpoint was trained **purely single-step with teacher forcing** — it was
never once shown its own output as input. Only `c1_rollout_full_b16` saw
`n_future = 1` (a 2-step unroll), and that was a 24-epoch fine-tune.

---

## 2. The stability argument, quantified

A **stable** emulator's RMSE must plateau. Once a forecast decorrelates from
truth, error stops growing and settles between **1.0x** the climatological
standard deviation (forecast collapsed to the climatological mean) and
**1.41x** (decorrelated but with correct variance). A curve that never bends
toward that ceiling has to either saturate later or diverge.

Measured at lead 21 (126 h), base checkpoint:

| quantity | value |
|---|---|
| fraction of climatological std reached | median **0.290** (p10 0.065, p90 0.699) |
| linear extrapolation reaches 1.0x std at | median **95 steps (24 days)** |
| linear extrapolation reaches 1.41x std at | median **154 steps (38 days)** |
| channels crossing 1.0x std before step 500 | **96 / 101** |

⇒ At 126 h we are only ~29 % of the way to the ceiling, so **126 h is firmly
the early regime**. And under linear extrapolation the median channel would
exceed climatological error by **step ~95**, long before the ~500 steps at
which the rollout is observed to blow up. By 120 days the model is deep past
the point where its own error exceeds climatology.

### Is the curve bending toward the ceiling?

Ratio of the last-5-lead slope to the first-5-lead slope (< 1 = decelerating,
heading for a plateau; ~1 = still linear):

| model | median | p25 | p75 |
|---|---|---|---|
| base | **0.779** | 0.414 | 1.101 |
| C1 | 0.774 | 0.491 | 1.182 |

Median **0.78 means the typical channel IS decelerating** — the healthy
signature — so the linear crossing steps above are **pessimistic upper bounds**
on error growth, not predictions.

⚠ **But p75 > 1.1: roughly a quarter of the 101 channels are still
accelerating at lead 21.** Those are the runaway candidates, and a rollout is
only as stable as its worst feedback channel.

### Which channels go first

| channel | RMSE @126 h | clim std | fraction | linear crossing of 1.0x std |
|---|---|---|---|---|
| `PRECT` | 9.79e-08 | 8.31e-08 | **1.179** | 13 steps |
| `U10` | 2.336 | 3.686 | **0.634** | **36 steps** |
| `PSL` | 435.8 | 1510 | 0.289 | 66 steps |
| `TMQ` | 5.092 | 17.55 | 0.290 | 79 steps |
| `T_l00` | 3.133 | 11.87 | 0.264 | 160 steps |
| `T_l17` | 2.053 | 20.98 | 0.098 | 282 steps |
| `TREFHT` | 2.053 | 21.22 | 0.097 | 294 steps |
| `PS` | 480.5 | 9353 | 0.051 | 392 steps |

- **`PRECT` already exceeds its climatological std at 126 h (1.18x).** It is
  the *diagnostic* channel and is **not fed back**, so it cannot drive the
  instability directly — but it is a strong signal that the moisture side is
  unphysical well before the blow-up.
- **`U10` is the fastest-degrading prognostic channel** — 63 % of the ceiling
  by 126 h, linear crossing at **36 steps**. Winds are fed back and are the
  natural driver of dynamical runaway. **This is where to look first.**
- Surface temperature and pressure are the *slowest*, which is why a
  `TREFHT_mean` monitor (what `longroll.py:419` currently logs) is the least
  sensitive early-warning channel available.

---

## 3. The non-obvious result: `n_future = 1` did not change stability

C1 is unambiguously **more accurate** at every lead beyond the first —
3-4.7 % lower RMSE, 77-89 of 101 channels improved
(→ `2026-09-10_c1_verdict.md`). But on every *stability* statistic it is
indistinguishable from the base model:

| statistic | base (`n_future=0`) | C1 (`n_future=1`) |
|---|---|---|
| fraction of ceiling at 126 h | 0.290 | 0.278 |
| linear crossing of 1.0x std | 95 steps | **94 steps** |
| linear crossing of 1.41x std | 154 steps | 143 steps |
| late/early slope ratio | 0.779 | **0.774** |

⇒ **A 2-step unroll shifts the whole error curve down without changing its
growth character.** More accuracy, same runaway. So `n_future = 1` is the right
*direction* but empirically nowhere near sufficient for a 500-step rollout —
and this is measured here, not argued from the literature.

---

## 4. What would actually be worth trying, in cost order

1. **Find out which channel diverges first.** `longroll.py:413` catches the
   blow-up with an `isfinite` check, but only once it is total; the drift
   usually starts hundreds of steps earlier. Log, every N steps, each channel's
   global mean and std as a **z-score against `global_means/global_stds`**, and
   flag any channel exceeding a few std. That converts "it blows up around 500"
   into "channel X starts drifting at step Y" for the cost of one reduction per
   step. Given §2, watch `U10` and the wind/level channels rather than
   `TREFHT`.
2. **Much longer unroll training (C2).** `n_future = 2-3` is affordable — the
   measured memory model gives ~26 GiB of a 39.49 GiB card at `n_future=1`,
   batch 4/GPU, and allows up to ~3 at that batch
   (`polaris_makani_1node_production_handoff.md` §C2). Upstream uses 4. Given
   §3, expect this to help gradually rather than to switch stability on.
3. **Attack the growth character directly, not just the error level.** The
   standard tools for long stable rollouts act on the accumulating
   high-frequency energy — spectral damping / filtering per step, or a
   stability-oriented regularizer. Cheapest experiment: apply a mild spectral
   filter each rollout step at *inference only* and see whether the divergence
   step moves. That costs one run and no retraining.
   ⚠ Noise injection during training is the other standard fix and is
   **explicitly out of scope** by the science owner's decision (the lagged
   ensemble exists to avoid it).
4. **Keep AMP off.** `longroll.py` already disables it. Correct — bf16 rounding
   compounds over hundreds of autoregressive steps.

⚠ **Caveat on all of the above.** Extrapolating a 21-step measurement to 500
steps is a long reach. The deceleration (§2) says the real curve bends more
than linearly, so the crossing steps are upper bounds on error growth. The
*ranking* of channels and the base-vs-C1 comparison are much more robust than
the absolute step numbers.

---

## 5. Two things to check on the E3SM inference code

`src/sfno_inference_e3sm/` (jesswan) supersedes most of the port scoped in
`2026-09-10_e3sm_inference_port_scope.md` — that document was written before
this module was known here, and its items A-D are largely moot. `longroll.py`
streams **real** prescribed forcing every step via `_advance_forcing`, so stale
covariates are ruled out as a cause of the blow-up.

Two open questions, neither verified:

1. **Which scorer produced the RMSE maps and ACC curves?** `nwp_rollout.py`
   writes the per-IC NetCDFs but does not score them. If the group's
   `src/sfno_eval/metrics.py` was used, its latitude weights are
   **Gauss-Legendre** (`cache_lat_weights(nlat=64)`), which is correct for the
   PLaSim T21 64x128 *Gaussian* grid and **wrong for the equiangular 180x360
   E3SM grid**, where the area weight goes as `cos(lat)`. The only guard is a
   length check, so 180 weights on 180 latitudes passes silently and returns a
   mis-weighted number, worst near the poles.
   → `2026-09-10_e3sm_inference_port_scope.md` §2.5.
2. **`_extract_truth_sic`-style positional lookups.** In the fork's older
   `sfno_inference/rollout_driver.py:281-302` a forcing channel is selected by
   **positional index 5** behind a length guard of `< 6`. E3SM's forcing vector
   is `[lsm, topo, glacier, natveg, sst, solin, ice]` — index 5 is `solin`,
   sea ice is 6 — so it passes the guard and returns the wrong variable under a
   confident label. Worth checking whether any equivalent positional lookup
   survived into `sfno_inference_e3sm`.
