# Item 4 acceptance gates A and B for the non-bitwise generators (rules fixed 2026-11-10, before unsealing)

**Decisions (user, 2026-11-10):**
- The user decides item 4 ("ignore weidong/jesswan").
- Results are not to be hidden: "do not seal anything we don't need to hide results".
- Gate C is deferred ("later, if needed"). Gate D runs only if B fails.

This file and `gates_ab.py` are committed **before** job 7824121's sealed file is opened. That is not to hide
anything: it is so the thresholds visibly predate the numbers. They are not changed after it is read.

## Inputs

- **Job 7824121's raw pairs** (`$MEMBER_ROOT/sqg_profile/taxonomy_7824121/pair_w*_s*_p*.npz`). These are the 16 archived
  states:
  - 64 pairs for `lean32ek` (= `lean32ek_fast`, bitwise, at 40 / 400 / 4,000 steps);
  - 8 pairs for each dt arm.
- **The sealed file** `sealed/sealed_taxonomy.json`. Its sha256 must equal
  `6f1d0e5153918a3b14faf6e29795d91263a032479121bf87a4417fb173b78be1`, or nothing is evaluated.
- **Frames.** `sub` (o1.2's training frames) decides the gates. `af` is reported.

The 8 coarse bands are those of `analyze_taxonomy.py`.

## The alias floor per band (production frames only; computed before unsealing)

Definitions, for production's own (`lean_u1`) frames over all 64 pairs:
- D_b is the band's contribution to the per-pixel mean square of Δ (o1.2 units; Parseval on the rfft half-plane).
- A_b is the same for the aliased part Δ_sub − Δ_af, on the modes where `af` is defined.
- ε_floor,b = √(A_b / D_b,valid) is the smallest relative error ANY model can reach in band b, if the aliased part is
  unpredictable from x (the team's "lo" floor).
- g_b = 0.1/√n_b, with n_b = 53, 56, 17, 4.9, 1.8, 1, 1, 1: the band memory in 0.02-steps, from `TEAM_REPORT.md`
  item 2/3, R4.
- T_b = g_b · ε_floor,b is the gate B threshold.

| band | D_b | A_b | ε_floor,b | g_b | T_b | ≥ 1% of Δ's energy |
|---|---|---|---|---|---|---|
| k 1–2 | 2.83e-5 | 2.16e-7 | 0.087 | 0.0137 | 1.2e-3 | no (0.04%) |
| k 2–4 | 1.07e-3 | 1.16e-6 | 0.033 | 0.0134 | 4.4e-4 | yes |
| k 4–8 | 5.68e-3 | 4.90e-6 | 0.029 | 0.0243 | 7.1e-4 | yes |
| k 8–16 | 1.12e-2 | 1.94e-5 | 0.042 | 0.0452 | 1.9e-3 | yes |
| k 16–32 | 2.00e-2 | 8.11e-5 | 0.064 | 0.0745 | 4.7e-3 | yes |
| k 32–64 | 2.18e-2 | 3.80e-4 | 0.132 | 0.1 | 1.3e-2 | yes |
| k 64–128 | 1.54e-2 | 2.62e-3 | 0.412 | 0.1 | 4.1e-2 | yes |
| k 128–182 | 2.32e-3 | 1.19e-3 | 0.730 | 0.1 | 7.3e-2 | yes |

The script recomputes these numbers from the same data. They must agree to the digits shown.

**Cross-check against the team table.** The alias fractions agree with TEAM_REPORT's floor table: A/(S+A) = 0.013 /
0.161 / 0.520 there, against 0.017 / 0.170 / 0.533 here. But A_b here is ≈ 1.9× the table's in every band. That is
unresolved: the state sample differs, or a half-plane Parseval factor was missed. The gates use one self-consistent
calculation, so the factor cancels.

## Gate A (noise)

- **Statistic.** Var_b(e) is the mean over the arm's pairs of band b's contribution to the per-pixel mean square of
  e = F_arm(4,000) − F_prod(4,000). It is computed by the same Parseval code as A_b.
- **Per band:** PASS if Var_b(e) / A_b < 0.01.
- **The arm passes gate A** if every band passes.

## Gate B (learnable bias), with the alias floor in place of the trained model's own error

All conditions use the arm's `sub` statistics from the sealed file.

| condition | rule | where it applies |
|---|---|---|
| B1, band coefficients | \|B_x,b\| < T_b and \|B_d,b\| < T_b | every band |
| B2, per-shell bias | the debiased level-2 learnable bias < T_b | every band |
| B3, detection | fraction of shells at p < 0.0027 ≤ p₀ + 2·√(p₀(1 − p₀)/n_shells,b), with p₀ = 0.0046 (the pooled negative-control rate of job 7824121, f64ic32 + lean32sr, 11 / 2,392 shells) | only bands holding ≥ 1% of Δ's energy (k ≥ 2) |
| power | MDE_x,b < T_b and MDE_d,b < T_b, from the arm's own scatter | every band; if this fails the band reads INCONCLUSIVE, not PASS |

- **Per band:** PASS if B1, B2 and B3 hold.
- **Otherwise: FAIL_FLOOR.** It fails against the floor stand-in. That is conservative, so it is not a final
  rejection: it then needs a trained model's own per-band error (the real ε_model,b ≥ ε_floor,b) or gate D.
- **The arm passes gate B** if every band passes.

## Arms and predictions

- **Arms:** `lean32ek` (64 pairs, which also stands for `lean32ek_fast`), `dt30` and `dt32` (8 pairs each).
- **Predictions** from `PREDICTIONS_taxonomy.md` are scored as written:
  - **P2.** The addendum's reading rule did not trigger, because the pooled controls were 0.46% < 1%. So "≤ 1% of
    shells in each band up to k 64" is read per band.
  - **P5.** The dt order ratio in [−1.6, −0.55] in every band (sub).

## What the result decides (the user's rule)

An arm that passes gates A and B is admitted to generate o1.2's training data. A FAIL_FLOOR band leads either to a
trained model's error or to gate D, before any decision. Gate C (one-time-unit data statistics) is deferred.

## Limits

- **The floor stand-in assumes the aliased part of Δ is unpredictable from x** (the "lo" assumption). If part of it
  were predictable, ε_floor would overstate the floor and the gate would be less strict than claimed.
- **Paired errors only.** A distributional bias over ≥ 1 time unit is gate C.
- **The dt arms have only 8 pairs.** Their power is checked per band.
