# Tile-rows + IF-AB3 at 25 steps per frame over 50 time units: predictions (written 2026-11-12, before any run)

**Ask (user, 2026-11-12):** "how can we try out tile-rows+IF-ab3, can we submit in capacity", then option A, "50-TU
stability". Later: "use 2 code reviewers and a fable principal scientist to make sure it is accurate".

**Why.**
- The time-step probe (job 7839988) showed IF-AB3 at 25 steps per frame is more accurate than production at 40 over
  a 0.02-TU lag.
- The tile-row step (job 7840376) showed 1.58 ms/step with `lean32ek_fast`'s accuracy.
- Neither tested long runs. At 25 steps per frame (h = 8e-6 = 1.6 DT) the CFL is ~0.6–0.67, against A-SQG_data.py's
  abort at 0.72 and warning at 0.60:
  - the 16-state archive's max is 0.379 at DT, i.e. 0.607 at h;
  - production's own maximum is 0.42 at DT (A-SQG_data.py L133: "DT=5e-6 stayed <= 0.42"), i.e. 0.672 at h.

## Scope: what this job can and cannot show (added after the review)

- **Rare blow-ups.** 4 runs × 50 TU = 200 TU of trajectory. Zero events bounds an event rate only to ≲ 1 per 67 TU
  (95 %, rule of three), or ~1 per 200 TU as a point estimate. An o1.2 schedule integrates ~480 TU per GPU (~6,700 TU
  in total) and production integrated 4,400 TU. A pass does not rule out rare aborts over a full o1.2 schedule.
- **Climate.** S1 / S2 compare 40-TU window means against the [min, max] of 16 instantaneous archive snapshots (SPE
  24.0–48.3, std 6.8). They only detect shifts of order ~25 %.
- **What the CFL numbers mean.** 0.7236 is AB3's imaginary-axis limit for the single fastest-advected mode at KMAX.
  That mode's viscosity removes most of it every step (E ≈ 0.05), so a brief excursion past 0.72 is not a blow-up.
  o1.2's training solver has no CFL abort; the 0.72 abort is production's heuristic. This job therefore stops only
  on non-finite values or CFL_h ≥ 1.0, and reports production's policy as a separate verdict.
- **Not the science gate.** This job checks consistency with production's own climate; it is not an equivalence test.
  The trained-model test (option B, gate-D style) is the science gate for the target change.

## Design

`stability_run.py`, `analyze_stability.py`; `run_stability_polaris.pbs` on capacity, 1 node, 6 h. The code is frozen
by `snapshot_src.py` at qsub time.

- **Production reference first:** the same diagnostics on all 16 staged archive states (members 1–4 × t = 100–400),
  alone on GPU 0.
- **Runs:** GPU w runs member w + 1 at t = 100 (w + 1), the energy job's four states (member 3 at t = 300 is the
  archive's highest CFL, 0.607 at h). Each worker has its own Warp kernel cache.
- **Per run:**
  1. load with o1.2's recipe;
  2. two float64 IF-AB3 switching steps to h = 8e-6 (`if_ab3.make_transition`, exact elapsed-time factors);
  3. **gate at h** (TileRows' first use at any dt but DT):
     - TileRows(if_mode) after 1 and 40 steps vs the float64 IF-AB3 reference at h (`make_advance_rt` "if" with
       `rt_consts(h)`), next to the JAX float32 IF-AB3 at h (`make_call_if32` with the h table);
     - pass if rel(TileRows, if64) ≤ min(1e-4, 2 · rel(if32, if64)) at both n (the tile-rows probe's GIF form), and
       max |(1 − e_h) − E(h)| ≤ 1e-7;
     - octave-band errors of the 40-step increment are reported;
     - a failure stops the worker (GATE_FAIL, counted as INCONCLUSIVE);
  4. `TileRows(if_mode=True)` at dt = h for 50 TU = 6.25 M steps.
- **Sampling:** every 12 frames (0.0024 TU; production samples every 0.0025 TU), the full state goes through
  A-SQG_data.py's own `diagnostics`: SPE, VIE, ε_ν, ε_μ, P, E(k) (shell = round(|k|)) and umax.
  - CFL_h = umax · KMAX · h.
  - Every sample's values + umax are stored (the fine series, ~21 k rows).
  - The run stops only on a non-finite sample or CFL_h ≥ 1.0.
  - Samples ≥ 0.60 and ≥ 0.72 are counted, as is the longest run of consecutive samples ≥ 0.72.
- **Stored every TU (frame count):** the values, E(k) and a 256² sub frame. run_w<w>.json / .npz are rewritten each
  TU and on a crash.

## Rule (fixed now; `analyze_stability.py`)

**Run status** (each of workers 0–3; a missing file is MISSING):
- COMPLETE = the full 50 TU, finite, CFL_h < 1.0.
- UNSTABLE_NONFINITE / UNSTABLE_CFL_HARD = the run's own stops.
- Anything else (MISSING, GATE_FAIL, RUNNING, CRASHED, DEADLINE, SHORT) is INCONCLUSIVE.

**Verdicts:**
- **STABILITY:** STABLE if all 4 runs are COMPLETE; UNSTABLE if any run is UNSTABLE_*; else INCONCLUSIVE.
  Token `STAB_FINITE`.
- **PROD_POLICY:** OK if all 4 runs are COMPLETE with max CFL_h < 0.72; EXCEEDED if any sample is ≥ 0.72; else
  INCONCLUSIVE. Token `STAB_PROD_POLICY_OK`.
- **STATS** (window tu_frame ∈ [10, 50], 41 per-TU samples per run, asserted):
  - **S1:** each run's window-mean SPE and VIE lie inside [min, max] of the 16 archive states.
  - **S2:** the pooled window-mean energy in every octave band of E(k) (k 1–2 … 1024–1366) lies inside [min, max] of
    the 16 states.
  - **S3 (no common drift):** b_w = OLS slope of SPE vs tu on run w's fine series (every 12 frames, window [10, 50]).
    Pass if |mean b| ≤ t₀.₉₇₅,₃ · sd(b) / √4, with t₀.₉₇₅,₃ = 3.182446.
    - The 4 runs are independent, so no model of SPE's autocorrelation is needed.
    - Simulated (`calibrate_s3_common.py`, AR(1), n = 16,667 per run, τ_I = 0.5–3 TU): false positives 5.0–7.7 %;
      power 0.31–0.40 / 0.76–0.85 / 1.00 for a common drift of 1 / 2 / 4 sd(b).
    - A scheme-induced drift would move every run the same way.
  - STATS = PASS if S1 ∧ S2 ∧ S3 over 4 complete runs; FAIL if any is False over 4 complete runs; else INCONCLUSIVE.
- **Report only** (in no verdict):
  - per-run |slope| ≤ 2 SE with SE inflated by the estimated g = 1 + 2 Σρ_k. This is anti-conservative at this
    length: `calibrate_s3_2se.py` gives 6–20 % false positives per run at τ_I 0.5–3 TU, because the ACF estimate is
    biased low;
  - the original S3 (|slope| × 40 ≤ archive SPE std, ddof 1);
  - energy budget over the window: SPE(50) − SPE(10) vs ∫(P − ε_ν − ε_μ) dt, as a fraction of ∫P. This is exact in
    continuous time except for the KMAX truncation;
  - per-shell ratios run / archive mean for k > 500;
  - an exponential tail fit of CFL_h above its 90th percentile;
  - ms/step; the gate numbers.
- No threshold changes after the run; a changed rule gets a dated, post-hoc addendum.

## Predictions (P = probability)

First written in commit da2008a7; revised 2026-11-12 after the review, **before any run** (definitions changed as
above; no data seen).

| # | prediction | P first | P revised |
|---|---|---|---|
| 0 | the gate at h passes on all 4 workers | — | 0.90 |
| 1 | STABILITY = STABLE (first: also CFL_h < 0.72) | 0.80 | 0.80 |
| 1b | PROD_POLICY = OK (max CFL_h < 0.72 everywhere) | (in 1) | 0.80 |
| 2 | max CFL_h over all samples is 0.55–0.70 | 0.70 | 0.80 |
| 3 | some samples reach ≥ 0.60, in at least one run (member 3 starts at 0.607) | 0.55 | 0.90 |
| 4 | S1 holds | 0.80 | 0.90 |
| 5 | S2 holds | 0.75 | 0.85 |
| 6 | S3 holds (first: archive-std yardstick, now the common-drift test) | 0.80 | 0.90 |
| 7 | sustained wall time is ≤ 1.75 ms/step including diagnostics (job 7840376: 1.578 without) | 0.70 | 0.80 |
| 8 | each run finishes in ≤ 3.5 h of wall time | 0.70 | 0.80 |

## Review → changes (wf_8c6d4a2d-cde, 2026-11-12; 2 code reviewers + a principal scientist, all Fable)

- **Required:**
  - the gate at h;
  - every diagnostics sample stored;
  - S3 recalibrated:
    - the review suggested 2 SE with N_eff;
    - the pre-run simulation showed that test is miscalibrated at τ_I ≥ 2 TU, hence the between-run test (that
      simulation result stays on record as report-only);
  - the expected worker set with INCONCLUSIVE statuses;
  - this scope paragraph.
- **Robustness:**
  - NaN strings coerced;
  - a missing npz handled;
  - the 41-sample window asserted;
  - the archive pass before the runs;
  - the end token on every exit path;
  - per-worker Warp caches;
  - the code frozen at qsub time.
- **Adopted from the recommendations:**
  - stop only on non-finite or CFL ≥ 1.0;
  - counts of 0.60 / 0.72 crossings;
  - the FINITE and PROD_POLICY tokens;
  - the energy budget, high-k shell ratios and CFL tail fit (report only).
- **Not adopted:**
  - 75 TU instead of 50: the user chose 50;
  - asking Midway for production's NetCDF scalar series: a future reference, not this job.

## Tokens

- Workers: `STAB_STATE`, `STAB_GATE`, `STAB_TU`, `STAB_UNSTABLE`, `STAB_DEADLINE`, `STAB_RUN_DONE`; archive
  `STAB_ARCHIVE_DONE`.
- Analysis: `STAB_RUN`, `STAB_BANDS`, `STAB_DRIFT`, `STAB_FINITE`, `STAB_PROD_POLICY_OK`,
  `STAB_VERDICT stability=… prod_policy=… stats=… runs=…`, `STAB_ANALYSIS_DONE`.
- Job: `STAB_JOB_DONE failures=N` (printed on every exit path).
- Results → `SQG/analysis/results/polaris_stability_<jobid>/`.
- npz files (spectra, sub frames, fine series, the final compact state) stay in
  `$MEMBER_ROOT/sqg_profile/stability_<jobid>/`.
