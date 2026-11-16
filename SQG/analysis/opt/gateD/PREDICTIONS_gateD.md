# Predictions for gate D: do rollouts drift, or overfit, when the training data carry lean32ek's bias?

Written 2026-11-10 by the main session, before any gate-D job was submitted. Do not edit after submission. Scored in
the notes (`polaris_sqg_notes.md`, next Part) after the job.

## Why

Gates A and B (notes Part XII, "Unsealed", 2026-11-10) gave this result for `lean32ek`:
- Its noise is negligible.
- Its bias is ≥ 100× below the gate-B tolerance, but statistically detectable.
- Under the pre-set rules that sends it to gate D.

The user asked: "can we check if rollouts overfit and drift?" They chose the training dose-response, and the longer
run on `capacity` ("we can submit on capacity").

## What runs (`run_gateD_polaris.pbs`, one Polaris node, 4 A100s)

1. **Stored frames.** Production (`lean_u1`, bitwise) frames from the 16 staged archived checkpoints, members 1–4 ×
   t = 100, 200, 300, 400, 1 time unit each (`gen_frames.py`). Frames 0 and 100–400 are checked bit for bit against
   job 7824121's production frames.
2. **Four training arms**, one per GPU (`train_dose.py`):
   - o1.2's network, loss, optimiser and OneCycle;
   - 195,250 steps, o1.2's own count, at batch 16;
   - trained on members 1–3 (12 segments, 58,812 pairs);
   - validated on member 4.

   | arm | dose α | seed |
   |---|---|---|
   | `a0_sA` | 0 | 1 |
   | `a0_sB` | 0 | 2 (the seed-to-seed spread) |
   | `a1000_sA` | 1,000 | 1 |
   | `a10000_sA` | 10,000 | 1 (positive control) |

   - The target is y + α·B̂(x, y). B̂ is `lean32ek`'s band-level bias pattern (`bias_pattern_lean32ek.json`, from
     job 7824121's opened sealed file).
   - α = 1,000 is about plain float32's bias size.
   - The same seed gives the same initial weights and batch sequence, so dose arms are paired with `a0_sA`.
   - **Change from the plan given to the user** (doses 0, 0, 100, 1,000), made before any run. A null result at both
     100 and 1,000 could not tell "the bias does not matter" from "the test cannot see". α = 10,000 is a positive
     control that should be visible, and 1,000 vs 10,000 checks linearity.
3. **Rollouts** (`rollout_dose.py`):
   - from 16 held-out member-4 frames, for 2,500 steps (50 time units), with best.pt and last.pt;
   - per-step band energies;
   - blow-ups (non-finite, or max|x| > 50× the initial max).
4. **Analysis** (`analyze_dose.py`; rules in its docstring, repeated below). It runs on best.pt (the deciding
   checkpoint) and on last.pt (reported).

## Decision rule (fixed now)

**Drift.**
- r_a(t) = E(rollout) / E(climatology) − 1, for E_low (k ≤ 4) and for each of the 8 band energies.
- Climatology: all stored frames, every 0.02.
- Windows: 5 time units.
- Usable time ends at the first step where any arm has fewer than half its initial conditions alive.
- Noise: N(w) = max(|R_0B − R_0A|, 2·SE_IC(R_0A)).
- Real-bias effect: |R_1000 − R_0A| / 1000.

**Verdicts.**
- **PASS_D** if the real-bias effect < 0.1·N(w) in every usable window, for E_low and every band.
- **INCONCLUSIVE_NO_POWER** if the α = 10,000 arm neither absorbs the bias (κ < 0.3) nor drifts beyond N(w) in any
  window.
- **INCONCLUSIVE_UNSTABLE** if no full window is usable.
- Otherwise **FAIL_D**.

**Absorption.** κ_a = Σ(f_a − f_0A)·B̂ / (α_a Σ B̂·B̂) on the fixed held-out pairs. It is 1 if a model learns the
injected bias one for one.

**Overfitting.** val_clean final / minimum ≥ 1.10 counts as overfitting.

**What a PASS_D means (the user's rule).** `lean32ek`'s bias does not change how a model trained on it rolls out, at
this training level. With gates A and B, that is the case for admitting `lean32ek_fast`.

## Predictions

| # | prediction | P |
|---|---|---|
| 1 | All 16 stored segments are bitwise equal to job 7824121's production frames at frames 0, 100, 200, 300, 400 | 0.9 |
| 2 | All 4 arms finish 195,250 steps at 0.38–0.46 s/step (o1.2 loop's ML step: 0.40–0.42 s) | 0.8 |
| 3 | The α = 0 arms overfit: val_clean final ≥ 1.10 × its minimum (≈ 53 passes over 58,812 pairs) | 0.6 |
| 4 | The α = 0 best.pt rollouts stay finite for 50 time units in ≥ 12 of 16 initial conditions | 0.5 |
| 5 | Absorption κ ∈ [0.5, 1.5] for both α = 1,000 and α = 10,000 | 0.6 |
| 6 | The α = 10,000 arm drifts beyond N(w) in at least one window (the positive control works) | 0.6 |
| 7 | **PASS_D** (best.pt) | 0.8 |
| 8 | The α = 0 model's own E_low stays within ±20% of the climatology, in every usable window (R_0A) | 0.5 |

## Limits

- **Training scale.** One A100 per arm at batch 16: about 1/14 of o1.2's samples, on 12 time units of data (o1.2: up
  to 3,281). Our models are less trained and less diverse. Their own errors and drift are larger, and the noise N(w)
  includes that.
- **The pattern.** Only lean32ek's *band-level* bias is injected; the per-shell detail was not kept. The linear part
  (α_b x + β_b d) is what the gate-B regression measured.
- **Two seeds** give one seed-spread sample per window. The IC standard error is a floor under it.
- **Model selection.** best.pt is chosen on each arm's own val target, as o1.2 does.
