# Prereg: B22/B24 + inference-only dry-air fix at 5 yr, and fidelity of FSEMA and B+fix (2026-10-06)

Written before submission. Operator, 2026-10-06: "do 3 debug jobs: score the EMA checkpoint 5 year
roll out, the inference only b22/24 and B + fix number".

## 1. B22/B24 with the dry-air fix at inference only, 5 yr

**Why:** DRYAIR24 (−0.21 hPa) differs from B22/B24 in three ways: the fix was trained in, `T_max 22`
annealing, and a restart from B_e01. This run isolates the fix: same B weights, fix applied only at
inference.

- `polaris_climate_run.pbs`, `ARMS="B22dryair B24dryair"`, checkpoints
  `nf4_prod_b16_r1/.../ckpt_mp0_e22_stable.tar` and `ckpt_mp0_e24_stable.tar` (the 7707597 files),
  `FLAGS=--dry-air-fix=on`. 8 starts to 2049-12-31. Then `polaris_climate_readout.pbs` and
  `polaris_dryair_budget_readout.pbs`.
- References on the same protocol:

  | arm | survived | dPS (hPa) | dDRY (hPa) |
  |---|---|---|---|
  | B22 | 8/8 | −6.0..−8.5 | −5.9..−8.4 |
  | B24 | 8/8 | +12.7..+17.7 | +12.4..+17.4 |
  | DRYAIR24 | 8/8 | −0.14..−0.24 | 0.00 |
  | B_e01 + inference fix | 0/8 | — | — |

  B_e01 with the fix failed earlier than without it.
- **Prediction:** both arms 8/8, dDRY ≈ 0 (max one-step |dDRY| < 0.01 hPa), median |dPS| ≤ 1 hPa.
- **Outcomes, per arm:**
  - **PH1, the fix alone suffices:** 8/8 and median |dPS| ≤ 1 hPa. DRYAIR24's small drift then
    comes from the constraint, not from its retraining or annealing.
  - **PH2, it pins mass but costs stability:** fewer than 8/8.
  - **PH3, it doesn't pin:** 8/8 but median |dDRY| > 0.1 hPa. The fix is not acting as
    built. Debug it; don't interpret.

## 2. Fidelity (time-mean pattern RMSE vs the true 2045–2049 climate, σ units)

`polaris_climate_fidelity_run.pbs`, now parametrised (defaults reproduce 7709968). Metric:
per-member **mean `rmse_sigma` over channels**. Reference: B22 0.246–0.259 σ, B24 0.310–0.327 σ
(101 ch, `climate_fidelity_7709968/fidelity_b22_b24.csv`).

- **FSEMA** (`climate_protocol_7720705`, 8 members): truth built from
  `f_nosoil_2n_b32_e23_scratch` (99 ch). It uses the same stats files as B, with `out_channels`
  skipping 8 and 9, so per-channel σ scores are comparable. **Compare on the 99 shared channels:** B22/B24's
  7709968 rows restricted to the 99 channel names.
  - Prediction: FSEMA is less accurate than B22 (immature single-step: val 0.01438 vs A's best
    0.01284).
  - Rule: Δ = median(FSEMA) − median(B22), over members. **Comparable** if |Δ| ≤ 0.02 σ. B22's own
    member range spans 0.013 σ. Otherwise better or worse by the sign of Δ. Same rule vs B24.
- **B22dryair / B24dryair** (the run in §1): truth `climate_fidelity_7709968/true_climatology_b.npz`
  (101 ch, reused). Δ = median(B+fix) − median(plain B), same checkpoint. **Accuracy cost** if
  Δ > +0.02 σ, **free** if |Δ| ≤ 0.02 σ, **gain** if Δ < −0.02 σ. Report PS and TMQ rows separately.
  The fix acts on them directly.

## Not decided

- Accuracy of DRYAIR24 itself (never scored; not in this batch).
- Whether FSEMA's skill holds for other seeds or epochs.
