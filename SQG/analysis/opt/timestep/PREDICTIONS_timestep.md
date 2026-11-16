# How far can the SQG solver's time step grow? Predictions for the time-step probe (written 2026-11-12, before any run)

**Ask (user, 2026-11-12):** "continue" after research round 2
(`../research_2026-11-12/SOLVER_SPEED_RESEARCH.md`). Its biggest lever is the time step:
- an exact integrating-factor AB3 (IF-AB3) removes production's first-order dt error;
- that allows 28 or 25 steps per frame instead of 40, i.e. 1.30–1.40× per optimizer step.
Its "Decisions" section proposed one 1-node debug job for the first probes. This is that job (the overlap retest is
not in it: `profile_overlap.py` knows only the `ref`/`fixed` solvers; it is left for later).

**What this job decides and what it does not.**
- It measures accuracy, stability margin and cost, and names the smallest admissible steps-per-frame `n_adm` under
  the rule below.
- Adopting IF-AB3 for o1.2 stays the user's call. It changes o1.2's targets by about production's own dt error
  (prediction 12), and research round 2's decision 1 (the gating route) is open.

## Design (`timestep_probe.py`, `if_ab3.py`; `run_timestep_polaris.pbs`, one node)

- **States:** archived checkpoints, loaded with o1.2's recipe. GPU w ∈ {0, 1, 2}: member w + 1 at t = 100 (w + 1).
  GPU 3 (aux): all 16 staged checkpoints for the CFL scan; member 4 at t = 400 for the float32 check.
- **One 0.02 lag** (o1.2's pair: 100 frames of `H_FRAME` = 2e-4) from each ladder state, at h = 2e-4 / n:
  - **prod:** production's AB3, θ' = E(θ + b0 N_n + b1 N_{n−1} + b2 N_{n−2});
  - **if:** IF-AB3, θ' = E(θ + b0 N_n + b1 E N_{n−1} + b2 E² N_{n−2});
  - n ∈ {80, 40, 32, 28, 25, 20} for both schemes;
  - **ref:** if at n = 160 (h = 1.25e-6). Its CFL is sampled after every 40-step call.
- **Switching steps:** off n = 40, the first two steps after the start use variable-step AB3 weights (dt_ladder's
  `ab3_coeffs`). For if, the older tendencies are multiplied by E over their real elapsed times.
- **Control:** prod at n = 40 through the run-time-weight call vs `lean_u1` (bitwise production).
- **Error per band** (taxonomy's 8 bands), on o1.2's sub frames (irfft2[::16, ::16] / 8, kept float64 here) and on
  af frames: e_b = rms_b(X_arm − X_ref) / rms_b(Δ_ref). This is the arm's error in the 0.02 delta, the quantity
  Part XII's dt ladder measured.
- **Aux:**
  - CFL at DT of the 16 states;
  - float32 IF-AB3 (`lean32ek_fast` + the IF factors written as T − e·T and T − e(2 − e)·T) vs float64 IF-AB3,
    next to `lean32ek_fast` vs `lean_u1` (n = 40, 0.02 lag);
  - ms/step of `lean_u1`, prod-rt, if-rt, `lean32ek_fast`, if32;
  - the batched-RFFT lowering check (one `rfft2` over a leading dim of 2 vs two calls: HLO fft ops and time;
    research #4).

## Background numbers these predictions lean on

- **Part XII P5:** production's dt order ratio D80/D30 was −1.42 to −1.51 in all 8 bands; first order is −1.50.
  Fitting e = A h + B h³ to −1.42 puts the h³ part at ≈ 2.5% of the first-order part at n = 40. At −1.51 it is ≈ 0.
- **Part XII's table:** production's dt sensitivity |D(2.5e-6)|/|Δ| = 2.5e-5 (sub, k 1–2) … 1.0e-4 (sub, k 128–182);
  9.3e-7 … 2.6e-5 alias-free. For first order, e(prod40) ≈ 2|D(2.5e-6)|.
- **CFL at DT:** 0.374 (4 archived states), 0.42 (production-sampled), 0.45 (a synthetic spin-up). CFL(n) =
  CFL(40)·40/n. AB3's imaginary-axis limit is ~0.72, and A-SQG_data.py aborts at 0.72.

## Decision rule (`analyze_timestep.py`; fixed now)

- **VALID:**
  - on every ladder worker, |prod40 − lean_u1| / |lean_u1| ≤ 1e-12;
  - ref is finite;
  - the reference is adequate: e(if80) ≤ 0.1 e(prod40) in every band and frame kind.
- **ACC_OK(n):** every ladder worker ran if<n> finite, and e_b(if<n>) ≤ e_b(prod40) in every band b, sub and af.
  IF-AB3 at n is no less accurate than production at 40, against the converged reference.
- **CFL_OK(n):** CFL_max · 40 / n ≤ 0.72. CFL_max is the largest CFL at DT over the 16 states, the ladder states and
  the reference runs' samples.
- **n_adm:** the smallest n ∈ {32, 28, 25, 20} with ACC_OK and CFL_OK.
  - Token `TS_VERDICT n_adm=<n|none> valid=<bool>`.
  - Projection per optimizer step: s(n) = 12 n t_step + 0.40 s (lean32ek_fast 2.698 ms, lean_u1 6.409 ms).
- **Out of scope:** a 0.02-lag run shows short-time stability only. Long-run stability at n_adm needs its own test,
  e.g. 50 TU like gate D, before any o1.2 run. Thresholds do not change after the run; a changed rule gets a dated,
  post-hoc-labelled addendum below.

## Predictions (P = probability I give it)

| # | prediction | P |
|---|---|---|
| 1 | control: prod40 via the run-time-weight call matches lean_u1 within 1e-12 on all 3 states (bitwise: reported, P 0.4) | 0.90 |
| 2 | VALID (incl. the reference adequacy check e(if80) ≤ 0.1 e(prod40) everywhere) | 0.75 |
| 3 | prod is first order: e(prod80)/e(prod40) ∈ [0.4, 0.6] in every band, sub and af; e(prod n)/e(prod40) within ±20% of 40/n for n ∈ {32, 28, 25, 20} | 0.70 |
| 4 | IF removes the split error: e(if40) ≤ 0.1 e(prod40) in every band, sub and af | 0.75 |
| 5 | IF is third order where measurable: e(if80)/e(if40) ∈ [0.08, 0.25] in every band, sub and af | 0.55 |
| 6 | ACC_OK at n = 28 | 0.85 |
| 7 | ACC_OK at n = 25 | 0.80 |
| 8 | ACC_OK at n = 20 | 0.65 |
| 9 | every run with n ≥ 25 is finite; n = 20 is finite on all 3 states | 0.90 / 0.75 |
| 10 | CFL_max ≤ 0.45, so CFL_OK at n = 25 (≤ 0.72) and not at n = 20 | 0.75 |
| 11 | n_adm = 25 (alternatives: 28 0.10, 20 0.10, 32 0.05, none 0.15) | 0.60 |
| 12 | adopting IF at n = 40 changes the delta by about production's own error: e(if40 vs prod40)/e(prod40) ∈ [0.8, 1.2] in every band | 0.75 |
| 13 | IF's cost: if_rt ≤ 1.02 × prod_rt and if32 ≤ 1.02 × lean32ek_fast (ms/step) | 0.75 |
| 14 | float32 IF keeps lean32ek's precision: e(if32 vs if64)/e(lean32ek_fast vs lean_u1) ∈ [0.33, 3] in every band, sub and af | 0.70 |
| 15 | batched RFFT: the leading-dim-2 `rfft2` is one fft op of batch 2 in the HLO | 0.70 |
| 16 | batched RFFT saves 0.02–0.08 ms vs two calls in float32 (bitwise: reported) | 0.50 |

## Tokens

- Workers: `TS_STATE`, `TS_CONTROL`, `TS_ARM`, `TS_ERR`, `TS_CFL`, `TS_F32`, `TS_BENCH`, `TS_RFFT`,
  `TS_WORKER_DONE w=N`.
- Analysis: `TS_VALID`, `TS_ORDER`, `TS_ACC`, `TS_CFL`, `TS_VERDICT n_adm=… valid=…`, `TS_ANALYSIS_DONE`.
- Job: `TS_JOB_DONE failures=N`.
- Results → `SQG/analysis/results/polaris_timestep_<jobid>/`; frames npz (~10 MB) stay in
  `$MEMBER_ROOT/sqg_profile/timestep_<jobid>/`.
