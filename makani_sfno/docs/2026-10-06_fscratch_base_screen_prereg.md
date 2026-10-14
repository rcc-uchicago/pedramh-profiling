# Prereg: 1-yr screen of the F-scratch base vs A at matched epochs (2026-10-06)

Written before submission. Operator request, 2026-10-06: "Screen F-scratch 1 yr".

## Question

F-scratch (`f_nosoil_2n_b32_e23_scratch`, job 7718436) is A's single-step recipe trained from
random init on 99 channels (no `SOILWATER_10CM`/`TSOI_10CM`). At epoch 23 its single-step val is
0.01438 vs A's 0.01402 (+2.5 %). **Does it roll out for one year any differently from A at the same
epochs?** This is the only A comparison where recipe, epoch and init are matched and only the channel
set differs.

## Members

`polaris/polaris_climate_screen.pbs`, `DRY_AIR_FIX=off` (neither run trained with it), 1460 leads,
`_tmq` truth files. One debug job per start: 2044 f1092, then f1156.

| label | run | file | epoch |
|---|---|---|---|
| fs_e21 / fs_e22 / fs_e23 | `f_nosoil_2n_b32_e23_scratch` | `ckpt_mp0_v20/v21/v22.tar` | 21 / 22 / 23 |
| fs_best | same | `best_ckpt_mp0.tar` | expected 23 |
| fs_ema | same | `best_ckpt_ema_mp0.tar` | expected 22 (log) |
| A_e21 / A_e22 / A_e23 | `prod1n_b32_sgdr` | `ckpt_mp0_v20/v21/v22.tar` | 21 / 22 / 23 |

A has no EMA file, so the EMA member has no matched A partner.

## Priors

- A e200/e220/e243 truncate at steps 490–664 on f1092 (`docs/2026-09-24_climate_screen_7649647.csv`).
- F-warm, all 8 checkpoints, truncate at 281–1243 across both starts (F3, 7709775/7709776).
- **Prediction:** all 16 member-runs truncate before step 1460.

## Read-out rule

- Measure: `truncated_at_step` per member and start. A survivor counts as 1461.
- Matched pairs: (fs_eN, A_eN) for N = 21, 22, 23 at both starts, so 6 pairs. A pair is a tie
  if both survive or the steps are equal.
- **"F-scratch less stable than A"** only if F truncates earlier in **6 of 6** pairs (two-sided
  sign test p ≈ 0.03). **"More stable"** is the mirror case. 5 of 6 is reported as leaning, not
  as a finding. Anything else: **indistinguishable at n = 6**.
- Noise check before reading anything into it: F-warm's own spread was 281–1243 steps.
- Any member that survives both starts is reported as a 5-yr protocol candidate. It is not run
  without the operator's word.

## Side checks (descriptive, no ranking)

- **`fs_best` must report `ckpt_epoch` = 23.** It is 7719538's base. If it reads otherwise, flag the
  operator before 7719538 starts (`qhold` is the lever). This is the only actionable gate here.
- `fs_ema` vs `fs_e22`: same epoch, raw vs EMA. One pair per start; report it, don't conclude.
- PS / TREFHT drift at steps 600 and 1460, only where both members of a pair reach them.

## What this does not decide

- Whether soil matters under B's 5-step recipe. That is 7719538 (`2026-10-06_fsF_nf4_bsched_prereg.md`).
- The channel-count share of the +2.5 % val gap (needs a validate-only A run on the 99 shared channels).

## Addendum: 5-yr protocol on fs_e23 and fs_ema (2026-10-06, written before submission)

The screen found every F-scratch member a candidate, and the operator said "yes run it for the 5 year
roll out". This addendum is dated after the screen results (`973958ef`) and before the 5-yr run.

- `polaris/polaris_climate_run.pbs`: 8 starts (Oct 1–29 2044, frames 1092 + 16i) to 2049-12-31,
  `FLAGS` empty (the dry-air fix stays off because the config has none). Arms: **FS23** = `ckpt_mp0_v22.tar`
  (epoch 23) and **FSEMA** = `best_ckpt_ema_mp0.tar` (epoch 23, EMA).
- Read-outs: `polaris_climate_readout.pbs` (survival, truncation lead) and
  `polaris_dryair_budget_readout.pbs` (dPS = lead L − lead 1, dDRY, TMQ, per member).
- References on the same protocol: B22 8/8, dPS −6.0..−8.5 hPa. B24 8/8, +12.7..+17.7. DRYAIR24 8/8,
  −0.14..−0.24. A e243 0/8 survive (non-finite at leads 317–849). F-warm has never run at 5 yr; it died in < 1 yr.
- **Prediction:** at least one arm survives fewer than 8 of 8 members, because no single-step
  checkpoint has survived 5 yr yet. FSEMA survives ≥ FS23.
- Outcomes, per arm:
  - **S5:** 8/8 and median |dPS| ≤ 6.8 hPa (B22's). Single-step soil-free from scratch matches
    B's 5-yr stability without a multistep fine-tune.
  - **P:** 8/8, median |dPS| > 6.8 hPa. It survives but drifts more than B22.
  - **D:** < 8/8. Report counts and truncation leads against A's 317–849.
  - FS23 vs FSEMA is descriptive only.
- Not decided here: soil vs training maturity (A e22/e23 collapsed at the same epoch, but this is
  one training seed per arm). Accuracy/fidelity scoring is a separate step.
