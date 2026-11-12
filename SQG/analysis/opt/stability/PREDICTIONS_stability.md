# Tile-rows + IF-AB3 at 25 steps per frame over 50 time units: predictions (written 2026-11-12, before any run)

**Ask (user, 2026-11-12):** "how can we try out tile-rows+IF-ab3, can we submit in capacity", then option A, "50-TU
stability". Later: "use 2 code reviewers and a fable principal scientist to make sure it is accurate".

**Why.**
- The time-step probe (job 7839988) showed IF-AB3 at 25 steps per frame is more accurate than production at 40 over
  a 0.02-TU lag.
- The tile-row step (job 7840376) showed 1.58 ms/step with `lean32ek_fast`'s accuracy.
- Neither tested long runs. At 25 steps per frame the CFL is ~0.6 (archive max 0.379 at DT → 0.607), against
  A-SQG_data.py's abort at 0.72 and warning at 0.60.

**Design** (`stability_run.py`, `analyze_stability.py`; `run_stability_polaris.pbs` on capacity, 1 node).
- **Runs:** GPU w runs member w + 1 at t = 100 (w + 1), the energy job's four states (member 3 at t = 300 is the
  archive's highest CFL).
- **Per run:**
  - load with o1.2's recipe;
  - two float64 IF-AB3 switching steps to h = 8e-6 (if_ab3.make_transition);
  - `TileRows(if_mode=True)` at dt = h, with the float32 damping table for h;
  - 50 TU = 6.25 M steps.
- **Sampling:** every 12 frames (0.0024 TU; production samples every 0.0025 TU), the full state goes through
  A-SQG_data.py's own `diagnostics`: SPE, VIE, ε_ν, ε_μ, P, E(k) (shell = round(|k|)) and umax.
  CFL_h = umax · KMAX · h. Non-finite values or CFL_h ≥ 0.72 stop the run.
- **Stored every TU:** the values, E(k) and a 256² sub frame.
- **Production reference:** the same diagnostics on all 16 staged archive states (members 1–4 × t = 100–400).

## Rule (fixed now; `analyze_stability.py`)

- **STABLE:** every run completes 50 TU with every sample finite, and max CFL_h < 0.72.
- **STATS** (window tu ∈ [10, 50], 41 samples per run):
  - **S1:** each run's mean SPE and VIE lie inside [min, max] of the 16 archive states.
  - **S2:** the pooled mean energy in every octave band of E(k) (k 1–2 … 1024–1366) lies inside [min, max] of the 16
    states.
  - **S3:** per run, |slope of SPE vs tu| × 40 TU ≤ the 16 states' standard deviation of SPE.
  - **STATS_OK** = S1 ∧ S2 ∧ S3.
- **What these are:**
  - consistency checks against production's own climate, not an equivalence test;
  - the trained-model test (option B, gate-D style) is the science gate for the target change;
  - a [min, max] envelope of 16 states is loose by construction (stated here so nobody reads more into a pass).
- No threshold changes after the run; a changed rule gets a dated, post-hoc addendum.

## Predictions (P = probability)

| # | prediction | P |
|---|---|---|
| 1 | all 4 runs complete 50 TU, finite (STABLE) | 0.80 |
| 2 | max CFL_h over all samples is 0.55–0.70 | 0.70 |
| 3 | some samples reach ≥ 0.60 (production's warning level), in at least one run | 0.55 |
| 4 | S1 holds (every run's mean SPE and VIE inside the archive envelope) | 0.80 |
| 5 | S2 holds (every octave band inside the envelope) | 0.75 |
| 6 | S3 holds (no SPE drift beyond the archive std over 40 TU) | 0.80 |
| 7 | sustained wall time is ≤ 1.75 ms/step including diagnostics (job 7840376: 1.578 without) | 0.70 |
| 8 | each run finishes in ≤ 3.5 h of wall time | 0.70 |

## Tokens

- Workers: `STAB_STATE`, `STAB_TU`, `STAB_UNSTABLE`, `STAB_RUN_DONE`; archive `STAB_ARCHIVE_DONE`.
- Analysis: `STAB_RUN`, `STAB_BANDS`, `STAB_VERDICT stable=… stats=…`, `STAB_ANALYSIS_DONE`.
- Job: `STAB_JOB_DONE failures=N`.
- Results → `SQG/analysis/results/polaris_stability_<jobid>/`; npz (spectra, sub frames, CFL samples, the final
  compact state) stay in `$MEMBER_ROOT/sqg_profile/stability_<jobid>/`.
