# A, B and F side by side (2026-10-08, revised 15:45 UTC)

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
| checkpoints tested | e243 | e01, e22, e24 | raw e23, EMA e23. Raw e20–e22 and e41–e43 **queued** (debug 7727625) | e14, e22, e24 **queued** (debug-scaling 7727731, starts after 17:45 UTC) | not tested |
| runs that finish | **0/8** | e01 1/8 · **e22 8/8** · **e24 8/8** | raw e23 0/8 · **EMA e23 8/8** · the rest pending | pending | — |
| when the failing runs blow up | 2.6–7.0 months (steps 317–849) | e01: 2.5–5.0 yr (steps 3639–7339) | raw e23: 1.2–3.3 yr (steps 1766–4830) | pending | — |
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

Sources, 5-yr: 7649597 (A e243, B e01), 7707597 (B22, B24), 7720705 (F-scratch raw/EMA e23). Queued: 7727625
(F-scratch by epoch, prereg `8c8f9048`) and 7727731 (F-depth4, prereg amendment 3 `295aae40`).

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
| F-scratch | random | 1 | raw e20 | 0.014536 (A +2.5 %) | — | — | queued, 7727625 |
| F-scratch | random | 1 | raw e21 | 0.014467 (A +2.6 %) | −9.6 (4) | −8.1 (5) | queued, 7727625 |
| F-scratch | random | 1 | raw e22 | 0.014414 (A +2.6 %) | −4.0 (0) | −2.8 (0) | queued, 7727625 |
| F-scratch | random | 1 | raw e23 | 0.014380 (A +2.5 %) | +7.5 (0) | +6.8 (1) | 0/8, ✗ 1766–4830 |
| F-scratch | random | 1 | **EMA e23** | 0.014381 | −2.3 (0) | −2.4 (0) | **8/8**, dPS −0.83 to +0.37 |
| F-scratch | random | 1 | raw e41 | 0.013824 (A +2.8 %) | −197 (72) | ✗ 517 | queued, 7727625 |
| F-scratch | random | 1 | raw e42 | 0.013794 (A +2.8 %) | ✗ 1098 | ✗ 745 | queued, 7727625 |
| F-scratch | random | 1 | raw e43 | 0.013776 (A +2.8 %) | ✗ 1060 | +6.1 (1) | queued, 7727625 |
| F-scratch | random | 1 | EMA e41 | 0.013758 | −1.4e15 (99), finite only | ✗ 718 | — |
| F-depth4 | F-scratch raw e23 | 5 | e1 (= best before the resume) | 0.015667 | −72.5 (0) | −78.1 (0) | — |
| F-depth4 | F-scratch raw e23 | 5 | EMA (= e1) | 0.015267 | −78.4 (0) | −78.0 (0) | — |
| F-depth4 | F-scratch raw e23 | 5 | e8 | 0.015870 | +49.8 (0) | +49.0 (0) | — |
| F-depth4 | F-scratch raw e23 | 5 | **e14** | 0.015850 | −4.5 (0) | −2.4 (0) | queued, 7727731 |
| F-depth4 | F-scratch raw e23 | 5 | e15 | 0.015873 | −17.4 (0) | −15.6 (0) | — |
| F-depth4 | F-scratch raw e23 | 5 | e16 | 0.015837 | −40.0 (0) | −38.9 (0) | — |
| F-depth4 | F-scratch raw e23 | 5 | e22 / e24 | in training | — | — | queued, 7727731 |

Sources: 7709775/7709776 (F-warm), 7720637/7720644 (F-scratch e21–23), 7725618/7725885 (F-scratch
e41–43), 7725803/7725885 (F-depth4), 7720705 (5-yr). F-depth4's live run (7725884) logged e21 val
0.015759, EMA 0.015452.

## Log traps found while building this

- A resume overwrites the run's `RUNS/<tag>.log`. `f_nosoil_2n_b32_e23_scratch.log` now starts at e24, and
  `fsF_anneal_nf4_b16_rtmax100_scratch.log` starts at e17. The earlier epochs survive only in the job stdout
  files: `makani-ace2-ports/makani_sfno/makani_mn_scaling.o7718436` and
  `arm-fsF-nf4-bsched/makani_sfno/makani_mn_scaling.o7719538`.
- A resume resets the raw best-val tracker. F-depth4's `best_ckpt_mp0.tar` is now e20; the e1 best is
  `best_ckpt_mp0_e1_stable.tar`.
