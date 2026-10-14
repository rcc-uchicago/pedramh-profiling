# A, B and F side by side (2026-10-08, revised 16:45 UTC; channel drift added 2026-10-09)

Every number comes from a job output, read on 2026-10-08: the rendered yamls and training logs in
`RUNS/`, the 1-yr `screen_summary.csv` files in `EVAL/climate_screen_<job>/`, and the 5-yr
`member_*.log` / `readout.log` files in `EVAL/climate_protocol_<job>/`. The 5-yr PS drifts come from the
budget read-outs 7719173 / 7720728. One step = 6 h, so 1460 steps = 1 yr.

## 1. Main table: how the models differ, and how they hold up

**Bold** in the recipe rows marks a cell that differs from A.

| | **A** | **B** | **F-scratch** | **F-depth4** (fsF) | F-warm |
|---|---|---|---|---|---|
| run tag | `prod1n_b32_sgdr` | `nf4_prod_b16_r1` | `f_nosoil_2n_b32_e23_scratch` | `fsF_anneal_nf4_b16_rtmax100_scratch` | `f_nosoil_2n_b32_e43_warm` |
| **Recipe** | | | | | |
| output channels | 101 | 101 | **99** | **99** | **99** |
| soil (`SOILWATER_10CM`, `TSOI_10CM`) | kept | kept | **dropped** | **dropped** | **dropped** |
| starts from | random init | **A e243** | random init | **F-scratch raw e23** | **A e243 sliced to 99 ch** |
| training depth (`multistep_count`) | 1 | **5** | 1 | **5** | 1 |
| peak LR | 2e-3 | **4e-4** | 2e-3 | **4e-4** | 2e-3 |
| LR schedule | SGDR T₀ 20, 3-ep warmup | **cosine `T_max 100`, never anneals** | SGDR T₀ 20 | **cosine `T_max 100`** | SGDR T₀ 20 |
| global batch (GPUs) | 32 (4) | **16** (8) | 32 (**8**) | **16** (8) | 32 (**8**) |
| EMA | none | none | **0.9995** | **0.9995** | **0.9995** |
| epochs trained | 243 | 24 | 43 | 21 so far, running to 63 (7725884) | 43 |
| **1-yr screen** | | | | | |
| stable (within ±10 hPa, finite) at both starts | none of the 5 epochs screened at both. e83 and e143 were stable at the one start tried | e22, e24 (e01/e21/e23 finite, drift 31–70 hPa) | e21, e22, e23, EMA e23. Not e41–e43 or EMA e41 | e14. The other 5 are finite but drift 16–78 hPa | none (every checkpoint blows up in 2–10 months) |
| **5-yr test** (8 runs, Oct-2044 starts → end 2049) | | | | | |
| checkpoints tested | e243 | e01, e22, e24 | raw e20–e23, raw e41–e43, EMA e23 | e14, e22, e24 **queued** (debug-scaling 7727731, starts after 17:45 UTC) | not tested |
| runs that finish | **0/8** | e01 1/8 · **e22 8/8** · **e24 8/8** | **EMA e23 8/8** · raw e22 2/8 · raw e20, e21, e23, e41, e42, e43 all 0/8 | pending | — |
| when the failing runs blow up | 2.6–7.0 months (steps 317–849) | e01: 2.5–5.0 yr (steps 3639–7339) | raw cycle 1 (e20–e23): 0.9–5.1 yr · raw cycle 2 (e41–e43): 0.3–1.8 yr | pending | — |
| PS drift at 5 yr (hPa) | — | e01 survivor −269 · e22 −6.0 to −8.5 · e24 +12.7 to +17.7 | EMA e23 −0.83 to +0.37 (dips −3.3 to −4.8 in months 1–5, then recovers) | pending | — |
| **Accuracy** | | | | | |
| climate error, 99 shared ch (σ, lower is better) | not scored | e22 0.239 · e24 0.308 | EMA e23 0.234 | not scored | — |
| RMSE vs B22 at 6 h / 24 h, 99 shared ch | — | reference | EMA e23 +4.3 / +19.0 % · raw e43 +0.9 / +14.5 % | EMA e16 +8.7 / +4.4 % | — |

What each pairwise comparison isolates:
- **A vs F-scratch** is the closest match: soil, EMA, and data-parallel width (4 → 8 GPUs at the same
  global batch). Depth, LR, schedule and batch are the same.
- **B vs F-depth4** differs in soil, EMA and base maturity. B's base is A e243 at val 0.01284;
  F-depth4's base is F-scratch e23 at 0.01438. The depth-4 recipe is the same.
- **A vs B** differs in depth, LR, schedule, batch, and starting point (B starts from A e243).
- **F-warm vs F-scratch** differs in starting point: A-derived weights vs random init.

Sources, 5-yr: 7649597 (A e243, B e01), 7707597 (B22, B24), 7720705 (F-scratch raw/EMA e23). 7727625
(F-scratch raw by epoch, prereg `8c8f9048`). Queued: 7727731 (F-depth4, prereg amendment 3 `295aae40`).

## 2. A, per checkpoint: 101 ch, soil kept, depth 1, random init, raw weights (A keeps no EMA)

1-yr cells: PS drift vs truth at step 1460 in hPa, with channels past 3σ in brackets. `✗ N` means
non-finite at step N. "—" means not run.

| epoch | single-step val | 1-yr f1092 | 1-yr f1156 | 5-yr (8 runs) |
|---|---|---|---|---|
| 14 | 0.015280 | — | — | — |
| 21 | 0.014106 | +14.8 (14) | +12.1 (13) | — |
| 22 | 0.014052 | −527 (81), finite but collapsed | −937 (84), finite but collapsed | — |
| 23 | 0.014024 | −1025 (86), finite but collapsed | −1442 (90), finite but collapsed | — |
| 43 | 0.013407 | ✗ 875 | ✗ 859 | — |
| 63 | — | ✗ 906 | — | — |
| 83 | 0.013043 | −6.8 (1) | — | — |
| 103 / 123 | — | ✗ 619 / ✗ 911 | — | — |
| 143 | 0.012910 | +8.3 (0) | — | — |
| 163 / 183 | — | ✗ 890 / ✗ 1291 | — | — |
| 200 / 220 | — | ✗ 664 / ✗ 490 | — | — |
| 243 (production) | 0.012839 | ✗ 595 | ✗ 490 | **0/8**, ✗ 317–849 |

Sources: 7720637/7720644 (e21–23), 7725618/7725885 (e43), 7721629 (e63–e183), 7649647 (e200–243
f1092), `surgical_soil_screen_7671841/f1156` (e243 f1156), 7649597 (5-yr).

## 3. B, per checkpoint: 101 ch, soil kept, depth 5, from A e243, raw weights (no EMA)

| epoch | single-step val (A e243 = 0.012839) | 1-yr f1092 | 1-yr f1156 | 5-yr (8 runs) |
|---|---|---|---|---|
| 1 (= `best_ckpt`) | 0.013411 (+4.5 %) | −31.0 (1) | −32.0 (1) | 1/8, ✗ 3639–7339; survivor −269 hPa |
| 21 | 0.013620 (+6.1 %) | −58.0 (0) | −56.5 (1) | — |
| 22 | 0.013591 (+5.9 %) | +0.7 (0) | −0.5 (0) | **8/8**, dPS −6.0 to −8.5 |
| 23 | 0.013597 (+5.9 %) | −68.1 (1) | −69.5 (3) | — |
| 24 | 0.013612 (+6.0 %) | +4.6 (0) | +4.5 (0) | **8/8**, dPS +12.7 to +17.7 |

Sources: 7649647 (f1092), 7649792 (f1156), 7649597 (e01 5-yr), 7707597 + 7719173 (e22/e24 5-yr). Depth-4
training raises single-step val by design, so a rising val here does not mean a worse model.

## 4. F, per checkpoint: 99 ch, soil dropped, EMA kept. Three variants

The `starts from` and `depth` columns are the differences between the variants. F averages val over
99 channels and A/B over 101, so part of the F − A val gap may be the denominator.

| variant | starts from | depth | checkpoint | single-step val | 1-yr f1092 | 1-yr f1156 | 5-yr (8 runs) |
|---|---|---|---|---|---|---|---|
| F-warm | A e243 sliced | 1 | raw e01, e22, e39–e43 | e43 0.013064 | ✗ 281–735 | ✗ 308–637 | — |
| F-warm | A e243 sliced | 1 | EMA e30 | 0.012891 | ✗ 1243 | ✗ 909 | — |
| F-scratch | random | 1 | raw e14 | 0.015654 (A +2.4 %) | — | — | — |
| F-scratch | random | 1 | raw e20 | 0.014536 (A +2.5 %) | — | — | 0/8, ✗ 2007–2904 |
| F-scratch | random | 1 | raw e21 | 0.014467 (A +2.6 %) | −9.6 (4) | −8.1 (5) | 0/8, ✗ 1250–3483 |
| F-scratch | random | 1 | raw e22 | 0.014414 (A +2.6 %) | −4.0 (0) | −2.8 (0) | 2/8, ✗ 3668–7477 |
| F-scratch | random | 1 | raw e23 | 0.014380 (A +2.5 %) | +7.5 (0) | +6.8 (1) | 0/8, ✗ 1766–4830 |
| F-scratch | random | 1 | **EMA e23** | 0.014381 | −2.3 (0) | −2.4 (0) | **8/8**, dPS −0.83 to +0.37 |
| F-scratch | random | 1 | raw e41 | 0.013824 (A +2.8 %) | −197 (72) | ✗ 517 | 0/8, ✗ 517–2673 |
| F-scratch | random | 1 | raw e42 | 0.013794 (A +2.8 %) | ✗ 1098 | ✗ 745 | 0/8, ✗ 662–1098 |
| F-scratch | random | 1 | raw e43 | 0.013776 (A +2.8 %) | ✗ 1060 | +6.1 (1) | 0/8, ✗ 496–2511 |
| F-scratch | random | 1 | EMA e41 | 0.013758 | −1.4e15 (99), finite only | ✗ 718 | — |
| F-depth4 | F-scratch raw e23 | 5 | e1 (= best before the resume) | 0.015667 | −72.5 (0) | −78.1 (0) | — |
| F-depth4 | F-scratch raw e23 | 5 | EMA (= e1) | 0.015267 | −78.4 (0) | −78.0 (0) | — |
| F-depth4 | F-scratch raw e23 | 5 | e8 | 0.015870 | +49.8 (0) | +49.0 (0) | — |
| F-depth4 | F-scratch raw e23 | 5 | **e14** | 0.015850 | −4.5 (0) | −2.4 (0) | queued, 7727731 |
| F-depth4 | F-scratch raw e23 | 5 | e15 | 0.015873 | −17.4 (0) | −15.6 (0) | — |
| F-depth4 | F-scratch raw e23 | 5 | e16 | 0.015837 | −40.0 (0) | −38.9 (0) | — |
| F-depth4 | F-scratch raw e23 | 5 | e22 / e24 | in training | — | — | queued, 7727731 |

Sources: 7709775/7709776 (F-warm), 7720637/7720644 (F-scratch e21–23), 7725618/7725885 (F-scratch
e41–43), 7725803/7725885 (F-depth4), 7720705 and 7727625 (5-yr). F-depth4's live run (7725884) logged e21 val
0.015759, EMA 0.015452.

**F-scratch raw at 5 yr, by epoch (7727625, `CLIMATE_RUN_OK`): outcome CONSISTENT-FAIL, as predicted
(prereg `8c8f9048`).**
- Cycle 1: e20, e21 and e23 are 0/8, and e22 is 2/8. Every failure is at PS.
- Cycle 2 fails sooner: e41, e42 and e43 are all 0/8, most within about 1.5 yr.
- No checkpoint reaches the ≥ 6/8 that LOTTERY needs. Unlike A at 1 yr, raw F-scratch does not survive at
  random epochs: it fails at every epoch. How long it lasts does vary by epoch (e22 lasts longest).
- The only F-scratch checkpoint that holds 5 yr is **EMA e23** (8/8), so the EMA averaging is what survives.
- The PS drift of e22's two survivors has not been read yet (needs the budget read-out).

## Log traps found while building this

- A resume overwrites the run's `RUNS/<tag>.log`. `f_nosoil_2n_b32_e23_scratch.log` now starts at e24, and
  `fsF_anneal_nf4_b16_rtmax100_scratch.log` starts at e17. The earlier epochs survive only in the job stdout
  files: `makani-ace2-ports/makani_sfno/makani_mn_scaling.o7718436` and
  `arm-fsF-nf4-bsched/makani_sfno/makani_mn_scaling.o7719538`.
- A resume resets the raw best-val tracker. F-depth4's `best_ckpt_mp0.tar` is now e20; the e1 best is
  `best_ckpt_mp0_e1_stable.tar`.

## Best checkpoints so far (2026-10-08 16:45 UTC)

Every checkpoint that finishes the 5-yr test in all 8 runs. The climate errors were re-derived from the
scoring CSVs (7709968, 7721213, 7721221, 7721578): median over the 8 runs of the mean `rmse_sigma`. The
range is across the runs.

| checkpoint | soil | 5-yr runs finish | PS drift at 5 yr (hPa) | climate error, 99 shared ch (σ) | climate error, 101 ch (σ) | forecast RMSE vs B22, 6 h / 24 h |
|---|---|---|---|---|---|---|
| **B22** | kept | 8/8 | −6.0 to −8.5 | **0.239** (0.235–0.247) | 0.251 | **reference** |
| B22 + inference-only dry-air fix | kept | 8/8 | **−0.08 to −0.17** | 0.245 (0.242–0.248) | 0.256 | not measured |
| B22 + inference-only RELHUM clamp | kept | 8/8 | −3.6 to −7.1 | 0.239 (0.235–0.244) | 0.250 | not measured |
| **F-scratch EMA e23** (FSEMA) | dropped | 8/8 | −0.83 to +0.37 (dips −3.3 to −4.8 in months 1–5, then recovers) | **0.234** (0.231–0.241) | — | +4.3 % / **+19.0 %** |
| DRYAIR24 (dry-air fix trained in) | kept | 8/8 | −0.14 to −0.24 | 0.325 (0.314–0.332) | 0.332 | not measured |
| B24 + inference-only dry-air fix | kept | 8/8 | +0.28 to +0.41 | 0.307 (0.303–0.311) | 0.315 | not measured |
| B24 | kept | 8/8 | +12.7 to +17.7 | 0.308 (0.301–0.318) | 0.317 | not measured |

**Best by category:**
- **5-yr stability:** all seven finish 8/8, so PS drift separates them. The smallest is B22 + the
  inference-only dry-air fix (about −0.1 hPa). The smallest without any corrector is FSEMA. None is fully clean:
  every arm's model top (`V_l00`) leaves 3σ within about 1–1.5 yr. FSEMA is the only one with runs that
  never do (2 of 8).
- **Climate accuracy:** FSEMA (0.234) and B22 (0.239) are tied, because their 8-run ranges overlap. B22 + fix
  is close (0.245), but it makes PSL worse and TREFHT colder. B24 and DRYAIR24 are clearly worse.
- **Short-range forecast RMSE:** B22 is the best measured, but only F checkpoints have been compared with it.
  B24, DRYAIR24 and the inference-only variants are unmeasured.

**Overall: B22 is the best all-rounder.** It is the only checkpoint at the top on all three. Run it with the
inference-only dry-air fix if PS drift matters more than a small accuracy cost. **FSEMA is the best
soil-free checkpoint:** B22-level climate and the smallest uncorrected drift, but 19 % worse at 24 h.

Tested at 5 yr and **not** surviving:
- A e243 0/8.
- B e01 1/8.
- B e01 + inference-only dry-air fix 0/8.
- DRYAIR01 0/8.
- SOILFIX21 7/8 and SOILFIX24 5/8.
- FSEMA + inference-only dry-air fix: 0/8 healthy.
- F-scratch raw at every epoch tested (e20–e23, e41–e43): ≤ 2/8.

**F-depth4 (2026-10-09):** e14, e22 and e24 also finish 8/8 (7727731, outcome S). PS drifts +11..+14,
+35..+40 and −105..−108 hPa, all of it dry air (budget 7731434). That is the worst of any 8/8 arm. With the
inference-only dry-air fix (7731448) PS holds to +0.3 / −0.6 hPa, but TREFHT then drifts +3.4 / −4.2 K. See
"5-yr channel drift" below. Its fidelity has not been run, so it is not ranked above.

**Caveats:**
- Each checkpoint comes from one training seed.
- The climate score has no noise floor yet: two real 5-yr periods have never been scored against each
  other. Read a 0.005 σ gap as a tie.

## 5-yr channel drift, every channel (2026-10-09)

What each checkpoint's climate drifts *to*, channel by channel. Drift = the annual global mean in 2049 minus
the one in 2045. Both are full scored years, so the seasonal cycle cancels. ⚠ `readout.log`'s "lead L − lead 1"
does not cancel it, because runs start in October and end in December. It reads F-depth4 e24's TREFHT as −3 K
when the drift is −0.16 K. σ = the pack's `global_stds`. Each cell is the median of the 8 runs.

Sources: debug jobs 7731449 / 7731614 read the member NetCDFs of 7720705 (FSEMA), 7711720 (DRYAIR24),
7707597 (B22, B24), 7727731 (F-depth4 e14/e22/e24) and 7731448 (F-depth4 e22/e24 + inference-only dry-air
fix). Every channel, with the min/max over the runs: `2026-10-09_channel_drift_2045_2049.csv` (next to this
doc). The script is in `EVAL/channel_drift_2045_2049/`. FD = F-depth4.

| checkpoint | ch | > 0.1σ | > 0.2σ | largest (σ) | Z3_l17 / PS (σ/σ) | PS (hPa) | PSL (hPa) | Z3_l17 (m) | Z3_l10 (m) | TREFHT (K) | T_l00 (K) | T_l08 (K) | U_l00 (m/s) | TMQ (kg/m²) | RHREFHT (%) | PRECT (mm/d) | FSNTOA (W/m²) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FSEMA | 99 | 3 | 0 | U_l04 -0.16 | -1.34 | +3.1 | -0.0 | -36 | -23 | -0.14 | -0.51 | +0.30 | -0.8 | -0.19 | +0.35 | -0.10 | -2.0 |
| DRYAIR24 | 101 | 12 | 5 | U_l00 -0.35 | n/a (PS pinned) | -0.0 | +1.7 | +58 | +23 | -0.20 | +1.16 | -0.35 | -9.0 | -0.24 | +1.50 | -0.18 | -0.3 |
| B22 | 101 | 15 | 8 | U_l00 -0.57 | -1.01 | -6.9 | +0.3 | +61 | +25 | +0.09 | +4.24 | -0.68 | -14.5 | +0.16 | +0.87 | -0.21 | -1.6 |
| B24 | 101 | 34 | 6 | U_l01 +0.32 | -1.12 | +14.2 | -0.4 | -139 | -67 | +0.78 | +2.16 | -0.80 | -4.6 | +0.79 | +0.73 | -0.11 | +3.0 |
| FD14 | 99 | 31 | 9 | U_l02 -0.37 | -1.24 | +13.4 | -0.8 | -145 | -37 | +1.73 | +1.62 | -1.08 | -4.8 | +1.85 | -0.34 | -0.22 | +3.9 |
| FD22 | 99 | 28 | 14 | U_l00 +0.58 | -1.21 | +24.3 | -1.2 | -256 | -91 | +1.75 | -1.22 | -1.38 | +14.8 | +1.26 | -0.54 | -0.44 | +2.6 |
| FD24 | 99 | 52 | 39 | Z3_l12 +1.26 | -1.23 | -74.4 | +3.5 | +799 | +488 | -0.16 | -7.90 | +3.37 | -7.7 | +3.42 | +5.48 | +0.61 | -21.8 |
| FD22dryair | 99 | 54 | 24 | U_l02 -0.67 | n/a (PS pinned) | +0.3 | -3.6 | -327 | -1 | +3.35 | +0.63 | +0.03 | +7.1 | +3.03 | -1.24 | -0.44 | -0.6 |
| FD24dryair | 99 | 54 | 30 | Z3_l17 +0.87 | n/a (PS pinned) | -0.3 | +8.5 | +708 | +148 | -4.18 | -1.75 | +1.16 | -1.9 | -3.27 | +6.53 | +0.22 | -8.4 |

**FSEMA: ten largest drifts** (σ; physical units; runs with the median's sign)

| channel | σ | physical | same sign |
|---|---|---|---|
| U_l04 | -0.159 | -2.224 | 7/8 |
| U_l06 | -0.117 | -1.54 | 8/8 |
| U_l05 | -0.111 | -1.416 | 7/8 |
| U_l01 | +0.062 | +1.377 | 7/8 |
| RELHUM_l00 | +0.059 | +0.003271 | 7/8 |
| Z3_l12 | -0.056 | -28.68 | 8/8 |
| Z3_l11 | -0.056 | -26.17 | 8/8 |
| Z3_l13 | -0.055 | -32.34 | 8/8 |
| Z3_l14 | -0.052 | -34.28 | 8/8 |
| Z3_l15 | -0.048 | -35.45 | 8/8 |

**DRYAIR24: ten largest drifts** (σ; physical units; runs with the median's sign)

| channel | σ | physical | same sign |
|---|---|---|---|
| U_l00 | -0.352 | -8.956 | 8/8 |
| RELHUM_l05 | +0.331 | +5.174 | 8/8 |
| RELHUM_l04 | +0.327 | +2.598 | 8/8 |
| T_l04 | -0.247 | -2.165 | 8/8 |
| T_l03 | -0.231 | -2.063 | 8/8 |
| U_l01 | -0.156 | -3.459 | 7/8 |
| SOILWATER_10CM | -0.143 | -2.035 | 8/8 |
| U_l03 | -0.133 | -2.309 | 8/8 |
| PSL | +0.116 | +174.6 | 8/8 |
| RELHUM_l10 | +0.105 | +3.637 | 8/8 |

**Readings:**
- **FSEMA is the cleanest of the nine.** No channel moves more than 0.2σ, and only three move more than 0.1σ
  (`U_l04`–`l06`, about −1.5..−2.2 m/s). Its +3.1 hPa PS here, against −0.83..+0.37 hPa in the budget, is the
  months 1–5 dip recovering: 2045 is still low.
- **DRYAIR24 pins PS (−0.0 hPa) but the upper troposphere drifts:**
  - `U_l00` −9.0 m/s
  - `T_l03`/`l04` −2.1 K
  - `RELHUM_l04`/`l05` +2.6 / +5.2
  - `SOILWATER_10CM` −2.0

  That is 5 channels past 0.2σ (B22 has 8). Its lowest-level height `Z3_l17` still rises 58 m with PS held.
- **Without a corrector, PS and the lower heights move together as one pattern.** `Z3_l17` drifts −1.0 to
  −1.3× PS, in σ units, in all six unpinned arms (FSEMA, B22, B24, FD14/22/24). Over the same years PSL barely
  moves. These are the terrain-dominated channels (R² ≥ 0.94 against `topo`, static audit 7646252), so the
  drift is a change in the terrain-shaped part of the state.
- **F-depth4 drifts most.** e24 has 39 channels past 0.2σ:
  - lower-level heights +490..+800 m
  - model top −7.9 K, mid-troposphere +3.4 K
  - TMQ +3.4 kg/m², FSNTOA −22 W/m²
- **The inference-only dry-air fix on F-depth4 (7731448) is PH1 by prereg amendment 4:**
  - Both arms are 8/8 complete.
  - Global TMQ stays > 0 at every lead. Its minimum is 25.6 kg/m² for e22 and 19.9 kg/m² for e24.
  - dDRY is 0.00. dPS is +0.30..+0.38 hPa (e22) and −0.56..−0.69 hPa (e24), all of it water
    (budget in the 7731614 log).
  - The prediction "e22 at risk of PH2" was wrong.
- **But the fix does not hold F-depth4's climate.** It moves the drift into temperature and water:
  - Channels past 0.1σ: 54 / 54, against 28 / 52 without the fix.
  - TREFHT: +3.35 K (e22) and −4.18 K (e24).
  - Global TMQ, first to last lead: 27 → 30.5 (e22) and 27 → 20 kg/m² (e24).
  - `Z3_l17` still drifts −327 / +708 m, because the fix shifts PS uniformly and leaves the terrain pattern
    alone.
  - So on this checkpoint the fix pins mass but leaves the climate drifting. Fidelity has not been run.
