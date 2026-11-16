# Gates A, B, E and D for `tilerows_if25`: predictions (written 2026-11-13, before any of these jobs)

**Ask (user, 2026-11-13):**
- "did we also test if it avoids drifting and overfitting". Only `lean32ek_fast` had been through gate D.
- Then: "okay do the same gates for tilerows_if25".

**The candidate.** `tilerows_if25` is IF-AB3 at 25 steps per frame (h = 8e-6) with the tile-row step, as o1.2 runs
it: `o12run/tilerows_advance.py` (Part XVII F, 0.926 s/step).
- `if32_25` (pure JAX, same scheme and step) is run beside it and **reported**, not decided.

**Same gates, same thresholds.** Each gate keeps its rule file unchanged:
- A and B: `taxonomy/ACCEPTANCE_gates_AB.md`;
- E: `energy/PREDICTIONS_energy.md`;
- D: `gateD/PREDICTIONS_gateD.md`.

Only the arm and its bias pattern change.

**The reference stays production (`lean_u1`, bitwise production).**
- Each gate asks whether the arm's training data are interchangeable with production's for o1.2. o1.2 is validated
  against production-generated data.
- Unlike `lean32ek`, this arm changes the time scheme. IF-AB3 removes production's first-order damping-split
  error (Part XVII A), so part of every arm–production difference is production's own error.
- For information only, each gate A/B state's first pair is also run with float64 IF-AB3 at 160 steps per frame (the
  time-step probe's converged reference). Each arm's error and production's error against it are reported side by
  side. They decide nothing.

## Gates A and B (`run_tr_ab_polaris.pbs`: `tr_pairs.py` + `gates_ab_tr.py`; debug)

**Data.**
- The taxonomy job 7824121's 16 archived states, 4 consecutive 0.02 pairs each (64 pairs). Same worker/state
  assignment; S_{j+1} = lean_u1 4,000 steps after S_j.
- From every pair start: `tilerows_if25` and `if32_25`, 100 frames.
- First pair only: ref160.
- Frames `sub` / `af` as taxonomy.py.

**Validity (all must hold, else INCONCLUSIVE):**
- V_frames: every pair's regenerated `lean_u1` frames (sub and af, every lag) are bitwise equal to job 7824121's stored
  pair.
- V_sealed: `lean32ek`'s level-1 / level-2 sub statistics, recomputed by `gates_ab_tr.py` from the stored pairs, equal
  the opened sealed file's to 1e-9 relative. Checked 2026-11-13 before submission: 72/72 equal (login-node dry run).
- V_floor: the alias floor matches ACCEPTANCE's protocol table to its digits (dry run: matches).

**Rules (ACCEPTANCE_gates_AB.md, unchanged):**
- **Gate A:** Var_b(e) / A_b < 0.01 in every band, with e = F_arm(100 frames) − F_lean_u1(4,000 steps).
- **Gate B** per band: B1 |B_x|, |B_d| < T_b; B2 learnable bias < T_b; B3 shell fraction ≤ p₀ + 2√(p₀(1−p₀)/n) where
  the band holds ≥ 1% of Δ's energy; power MDE < T_b.
  - PASS if all hold.
  - Otherwise FAIL_FLOOR, which sends the arm to gate D.
  - INCONCLUSIVE without power.

**Gate D input** (`gates_ab_tr.py`):
- `bias_pattern_tilerows_if25.json`: level-1 alpha / beta, `bias_pattern_lean32ek.json`'s format.
- size_ratio R = rms B̂_tr / rms B̂_lean32ek over the 64 pairs.

## Gate E (`run_tr_energy_polaris.pbs`: `energy_budget.py` + `analyze_energy.py --candidate`; debug-scaling)

- **Design:** job 7836358's. Member w + 1 at t = 100 (w + 1); 0.5 TU in calls of one frame (2e-4 TU); Simpson.
- **Arms:** `lean_u1` (reference), `tilerows_if25` (candidate), `f64ic32` (null), `dquant` (calibration), `if32_25`
  (reported).
- The IF arms run 25 steps per call. ms/step and CFL use their own step h.
- **Rule (PREDICTIONS_energy.md, unchanged):** T_E = 1e-4.
  - PASS_E if valid and max over ICs |s(candidate)| ≤ T_E for SPE and VIE.
  - V1–V5 as written, with V5's CFL < 0.72 read at each arm's own step.
- The modified analysis reproduces job 7836358's verdict on its own data: PASS_E, max |s| 5.54e-6 (checked
  2026-11-13).

## Gate D (capacity, after A/B: `gateD/` with `DOSE_BIAS_PATTERN` = the arm's pattern)

- **Same design as job 7835786:**
  - o1.2's UViT, loss, optimiser and OneCycle; 195,250 steps at batch 16;
  - train members 1–3 on the stored production frames;
  - target y + α·B̂_tr(x, y);
  - rollouts from 16 member-4 frames × 50 TU; best.pt decides, last.pt is reported.
- **Doses (rule fixed now).** The arm's own bias has α = 1. The injected magnitudes are kept equal to job 7835786's:
  - α_lo = max(1, 1000 / R), rounded to 2 significant figures;
  - α_hi = 10·α_lo;
  - arms a0_sA, a0_sB, a⟨α_lo⟩_sA, a⟨α_hi⟩_sA.
- **PASS_D (unchanged):** |R_αlo − R_0A| / α_lo < 0.1·N(w) in every usable 5-TU window, for E_low and all 8 bands.
  - INCONCLUSIVE_NO_POWER and INCONCLUSIVE_UNSTABLE as written.
  - Overfitting: val_clean final / min ≥ 1.10.
- Gate D runs whatever gate B says: the user asked about drift and overfitting directly.

## What the result decides (the user's rule, as for `lean32ek_fast`)

- Gates A and B pass: the arm is admitted to generate o1.2's training data.
- B fails on the floor stand-in: gate D decides.
- E is MD's solver-side check. Admission stays the user's call (Part XV's wording: "the case for admitting").

## Predictions

Calibration: the production scheme at 30 / 32 steps per frame (`dt30`, `dt32`) passed gate A at Var/A_b ≈ 1–4e-8 and
failed gate B on B3 detection only, with ≥ 100× margin on B1 / B2 (job 7824121). `tilerows_if25`'s difference from
production should be ~3× `dt30`'s (all of production's first-order error, not a third of it) plus `lean32ek`-sized
float32 noise.

| # | prediction | P |
|---|---|---|
| A1 | gate A/B validity holds (V_frames 64/64, V_sealed, V_floor) | 0.92 |
| A2 | gate A PASS for `tilerows_if25`, every band Var_b(e)/A_b ≤ 1e-5 | 0.90 |
| A3 | gate B: B1 and B2 hold in every band (bias ≥ 10× below T_b) | 0.85 |
| A4 | gate B NOT_PASS on B3 detection in at least one band (as `lean32ek` and the dt arms) | 0.85 |
| A5 | vs ref160: `tilerows_if25`'s median e_b below `lean_u1`'s in ≥ 6 of 8 bands | 0.75 |
| A6 | `if32_25` and `tilerows_if25` reach the same gate A and B verdicts | 0.85 |
| A7 | size_ratio R in [0.5, 20] (so α_lo between 50 and 2,000) | 0.6 |
| E1 | gate E valid (V1–V5) | 0.85 |
| E2 | **PASS_E** for `tilerows_if25` (max \|s\| ≤ 1e-4) | 0.85 |
| E3 | max \|s_SPE\| ≤ 3e-5: about production's own −1.1e-5 residual, which IF-AB3 does not have, plus float32 | 0.7 |
| E4 | DETECTED for at least one invariant | 0.8 |
| E5 | `tilerows_if25` ms/step (budget included) ≤ 2.0 | 0.7 |
| D1 | **PASS_D** on best.pt | 0.7 |
| D2 | no overfitting in the α = 0 arms (final / min < 1.10), as job 7835786 | 0.85 |
| D3 | the α_hi positive control drifts beyond N(w) in ≥ 1 window on E_low or a band | 0.6 |

## Tokens and outputs

- **A/B:** `TRAB_STATE`, `TRAB_PAIR`, `TRAB_WORKER_DONE`, `TRAB_VALID`, `GATE_A`, `GATE_B`, `TRAB_ACC`,
  `TRAB_PATTERN`, `TRAB_VERDICT`, `GATES_AB_TR_DONE`, `TRAB_JOB_DONE failures=N` → `results/polaris_trab_<job>/`.
- **E:** `ENERGY_*` lines, `ENERGY_VERDICT` per candidate, `TRE_JOB_DONE failures=N` → `results/polaris_tre_<job>/`.
- **D:** as job 7835786 (`DOSE_*`).
- Scoring: `score_gates_tilerows.py`, written before the results are read.
