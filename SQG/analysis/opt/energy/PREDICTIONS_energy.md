# Predictions for the MD-style energy-budget test: does `lean32ek_fast` close the energy budget as float64 does?

Written 2026-11-11 by the main session, before any energy-budget job was submitted. Do not edit after submission.
Scored in the notes (`polaris_sqg_notes.md`, next Part) after the job.

## Why

`team_2026-11-06/ANALOGUES.md` §2.6: molecular-dynamics GPU codes (AMBER SPFP, OpenMM mixed, GROMACS mixed, LAMMPS
mixed) accept a precision model on **long-run energy conservation** against double precision, not on short paired
errors. `lean32ek` is SPFP's design (single-precision forces, exact accumulation). Our system is forced and
dissipative, so the analogue of NVE drift is the **closure of the energy budget**. The user, 2026-11-11: "yes do the MD
debug job".

**The budgets.** SQG advection conserves two quadratic invariants exactly (with the 2/3 truncation), so in continuous
time

    d SPE/dt = P_S − ε_S,   SPE = ½ Σ a,          P_S = Σ w Re(θ̂* F̂)/N⁴,         ε_S = Σ λ a
    d VIE/dt = P_V − ε_V,   VIE = ½ Σ a / |k|,    P_V = Σ w Re(θ̂* F̂)/(|k| N⁴),   ε_V = Σ λ a / |k|

with a = w |θ̂/N²|² and λ = ν k² + μ/k² (`A-SQG_data.py`'s `diagnostics()` definitions; VIE's terms are the same sums
weighted by 1/|k|). Anything else that changes an invariant is the numerics: the time scheme, or the precision.

**Change from the offer** (ANALOGUES §2.6 said "≥ 10 time units"): one time unit is 200,000 steps, so 10 TU of
`lean_u1` alone is 3.6 h, and a debug job has 1 h. A 10-TU comparison of the climate means would also only resolve
~1% differences. The closure residual is a per-step identity, so a paired run resolves it to far below that in 0.5 TU:
- 4 archived states (independent members) × 0.5 TU (100,000 steps, 25× the delta job's longest window) × 5 arms.
- The closure residual is the deciding statistic.
- The climate means are reported only.

## What runs (`run_energy_polaris.pbs`, one Polaris node, 4 A100s, `debug`)

Worker w (one per GPU) loads the archived checkpoint of member w + 1 at t = 100 (w + 1) (o1.2's loading recipe) and
runs each arm from it for 2,500 calls of 40 steps (0.5 TU). After every call, `energy_budget.py` evaluates the budget
terms in float64 on the arm's complex128 state.

| arm | what | role |
|---|---|---|
| `lean_u1` | float64, bitwise production (jobs 7815637, 7824121, 7835278) | the reference |
| `lean32ek_fast` | float32 tendencies + compensated (hi + lo) state, 2.70 ms/step | **the candidate** |
| `f64ic32` | `lean_u1` from the state rounded once to complex64 | the null: a round-off perturbation, no precision source |
| `dquant` | `lean_u1` with only its decay table rounded to float32 (mechanism A, in float64) | calibration: its source is known exactly |
| `lean32` | all float32 (AMBER's deprecated SPSP / OpenMM single) | positive control |

Per arm, IC and τ ∈ {0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5}, with Simpson's rule on the 40-step samples (h = 2e-4 TU):
- closure residual R_X(τ) = X(τ) − X(0) − ∫₀^τ (P_X − ε_X) dt, normalised r_X = R_X / ∫₀^τ P_X dt (the fraction of
  the injected invariant that the numerics created or destroyed);
- **paired spurious source** s_X = r_X(arm) − r_X(`lean_u1`), same IC, same τ.
- s is also, to first order, the fractional shift of the invariant's steady-state mean: a source s·P_X shifts X by
  ≈ s·X if the relaxation time is X/P_X.

Exact predictions computed along each arm's own trajectory:
- **Integrating-factor term** (the float64 scheme's own residual). The scheme multiplies the AB3 increment by
  e^{−λΔt} and not by ≈ 1 − λΔt/2. That is a tendency error −(λΔt/2)·T, so r̂_IF = −(Δt/2) ∫ Σ λ w Re(θ̂* T̂)/N⁴ dt /
  ∫P, with T̂ the stored newest tendency. Also r̂_bal, which uses the steady-state balance Σ λ² a instead.
- **Mechanism A** (dquant, lean32): ŝ_A = −∫ Σ (λ_eff − λ) a dt / ∫P, with λ_eff = −ln(float32(decay))/Δt.
- **lean32ek's damping table**: ŝ_E, the same with λ_eff = −ln(1 − float32(e))/Δt.

Also checked:
- `lean_u1`'s o1.2 frames at calls 0, 1, 10, 100, 1,000 and 2,500, bit for bit against gate D's stored production
  frames (`sqg_profile/gateD_frames`, read only);
- the budget function against `GD.diagnostics` every 50 calls (SPE, VIE, ε_ν, ε_μ, P), with the CFL number.

## Decision rule (fixed now)

**Tolerance.** T_E = 1e-4 of the injection, for each invariant.
- By the linear-response reading above, it is a ~1e-4 shift of the mean SPE or VIE.
- That is about 100× below the sampling error of a 500-TU member's own mean SPE (SPE ranges 32–46 over the 16
  archived states). This is an estimate, not a measurement.

**Valid** only if all hold at τ_eval:
- V1 `lean_u1` frames bitwise equal to gate D's production frames at every checked call.
- V2 The budget function equals `GD.diagnostics` to ≤ 1e-12 relative (P relative to ε) at every check.
- V3 Null: max over ICs |s(`f64ic32`)| ≤ 0.1·T_E, for both invariants.
- V4 Calibration: for each invariant where |mean ŝ_A(`dquant`)| ≥ 10 × max |s(`f64ic32`)|, the ratio mean s(`dquant`) /
  mean ŝ_A(`dquant`) is within ±10% of 1. At least one invariant must qualify.
- V5 All required arms (`lean_u1`, `lean32ek_fast`, `f64ic32`, `dquant`) finite, CFL < 0.72 (the solver's abort
  threshold), and τ_eval ≥ 0.2 TU.

τ_eval = 0.5 TU, or, if the job's deadline guard cut an arm short, the largest listed τ that every required arm reached
on every IC.

**Verdicts.**
- **PASS_E** if valid and max over ICs |s(`lean32ek_fast`)| ≤ T_E for both SPE and VIE.
- **FAIL_E** if valid and not PASS_E.
- **INCONCLUSIVE_<reason>** if not valid: PROD_MISMATCH (V1), DIAG (V2), NOISE (V3), CALIBRATION (V4), RUN (V5).

**Reported, not deciding:**
- EQUIV_DOUBLE_X: MD's "conserves as well as double", |mean s(`lean32ek_fast`)| ≤ 0.1·|mean r(`lean_u1`)|.
- DETECTED_X: |mean s(`lean32ek_fast`)| > 3 × max |s(`f64ic32`)|.
- s(τ) at every τ.
- `lean32`'s numbers.
- The paired time means of SPE, VIE, P, ε.
- ms/step.

**What a PASS_E means.** Over 100,000 steps from 4 archived states, `lean32ek_fast`'s float32 arithmetic creates or
destroys less than 1e-4 of the injected SPE and VIE. It is MD's acceptance test in our terms: the solver-side half of
gate C (ANALOGUES §2.6). It says nothing about the ML model (that is gate D, job 7835786).

## Predictions

| # | prediction | P |
|---|---|---|
| 1 | `lean_u1` frames bitwise equal to gate D's at every checked call, on all 4 ICs | 0.95 |
| 2 | Budget function equals `GD.diagnostics` to ≤ 1e-12 at every check | 0.95 |
| 3 | r_SPE(`lean_u1`) ∈ [−5e-4, −5e-5]: the scheme loses SPE through the integrating factor at the dissipation scale | 0.6 |
| 4 | r_SPE(`lean_u1`) / r̂_IF ∈ [0.8, 1.2]: that first-order term is the float64 scheme's whole SPE residual | 0.6 |
| 5 | \|r_VIE(`lean_u1`)\| ≤ 1e-4 | 0.8 |
| 6 | Null: max \|s(`f64ic32`)\| ≤ 1e-6 for both invariants | 0.9 |
| 7 | `dquant` calibration within ±10% on every qualifying invariant | 0.75 |
| 8 | \|mean ŝ_A(`dquant`)\| ≥ T_E on at least one invariant (the known source is big enough to fail the gate) | 0.7 |
| 9 | `lean32`: \|mean s\| ≥ T_E on at least one invariant (plain float32 fails, as MD's SPSP) | 0.7 |
| 10 | `lean32`: s_VIE / ŝ_A,VIE ∈ [0.5, 2] (mechanism A dominates, as for the learnable bias in Part XII) | 0.5 |
| 11 | `lean32ek_fast`: max \|s\| ≤ 1e-6 for both invariants (≥ 100× below T_E) | 0.7 |
| 12 | **PASS_E** | 0.9 |
| 13 | EQUIV_DOUBLE for both invariants | 0.5 |
| 14 | DETECTED for at least one invariant (`lean32ek_fast`'s source is real but tiny, as in gate B) | 0.5 |
| 15 | `dquant`'s s_X(τ) within ±20% of its τ = 0.5 value for every τ ≥ 0.1 (a steady source, not a growing one) | 0.7 |
| 16 | ms/step (budget included) within ±5% of 6.40 / 2.70 / 3.20 for `lean_u1` / `lean32ek_fast` / `lean32` | 0.8 |
| 17 | Paired time means of SPE and VIE over 0.5 TU: `lean32ek_fast` within 1e-3 relative of `lean_u1` on every IC | 0.7 |

## Limits

- **Length.**
  - 0.5 TU is ~5 e-folding times (λ ≈ 9–10 per TU) but shorter than the ~1.7 TU it takes a 1e-7 perturbation to
    reach O(1).
  - The paired trajectories stay close, which is what makes the residual sharp. For the same reason the climate
    means are not independent samples here.
  - A source that grows over tens of TU (plain float32's mechanism B grew 160× over 100× steps, Part XI) would be
    underestimated. `lean32ek` compensates that mechanism.
- **4 ICs.** One state per member, so the IC spread is from 4 values.
- **Quadrature.** Simpson on 40-step samples. Its error is the same functional for paired arms, so it cancels in s,
  but not in r(`lean_u1`). |Simpson − trapezoid| is reported as its bound.
- **Total budgets only.** A per-shell budget needs the nonlinear transfer per shell, which is not computed. Where a
  source sits in k is inferred from ŝ_A's tables, not measured.

## Erratum (2026-11-11, after the run; nothing above is changed)

- §Decision rule, "Tolerance": "SPE ranges 32–46 over the 16 archived states". 32–46 is the range over the 4 states
  this test uses. Over the 16 it is 24–48 (taxonomy job 7824121's worker JSONs). The rule, T_E and every prediction
  are unchanged. Found by the MATH_energy.md checker.
