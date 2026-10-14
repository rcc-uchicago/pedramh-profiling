| arm | nodes | steps per epoch | global batch (samples per update) | epochs | updates | compute (node-hours) | median epoch time (s) | final validation loss | best validation loss (epoch) | best EMA validation loss (epoch) |
|---|---|---|---|---|---|---|---|---|---|---|
| 128-node run | 128 | 85 | 512 | 100 | 8,500 | 204.3 | 58 | 0.018297 | 0.018297 (e100) | no EMA kept |
| A | 1 | 1368 | 32 | 243 | 332,424 | 46.3 | 672 | 0.012839 | 0.012839 (e243) | no EMA kept |
| B | 2 | 2737 | 16 | 24 | 65,688 | 17.4 | 1305 | 0.013612 | 0.013411 (e1) | no EMA kept |
| F-warm | 2 | 1368 | 32 | 43 | 58,824 | 21.4 | 894 | 0.013064 | 0.013064 (e43) | 0.012891 (e30) |
| F-scratch | 2 | 1368 | 32 | 43 | 58,824 | 23.0 | 931 | 0.013776 | 0.013776 (e43) | 0.013758 (e41) |
| F-depth4 | 2 | 2737 | 16 | 63 | 172,431 | 95.1 | 2717 | 0.015320 | 0.015320 (e63) | 0.015220 (e63) |
| G1 | 2 | 1368 | 32 | 23 | 31,464 | 11.4 | 886 | 0.014311 | 0.014311 (e23) | 0.014308 (e23) |
| G1-dry | 2 | 1368 | 32 | 63 | 86,184 | 30.3 | 864 | 0.013361 | 0.013361 (e63) | 0.013331 (e59) |

| checkpoint | rollouts finite to 2049-12-31 | 95% interval for the finite fraction | first non-finite step, range (years) |
|---|---|---|---|
| A e243 (raw) | 0/8 | 0.00 to 0.37 | 317 to 849 (0.22 to 0.58) |
| B e01 (raw; its lowest-validation-loss checkpoint) | 1/8 | 0.00 to 0.53 | 3639 to 7339 (2.49 to 5.03) |
| B e22 (raw) | 8/8 | 0.63 to 1.00 | none |
| B e24 (raw) | 8/8 | 0.63 to 1.00 | none |
| F-scratch e20 (raw) | 0/8 | 0.00 to 0.37 | 2007 to 2904 (1.37 to 1.99) |
| F-scratch e21 (raw) | 0/8 | 0.00 to 0.37 | 1250 to 3483 (0.86 to 2.39) |
| F-scratch e22 (raw) | 2/8 | 0.03 to 0.65 | 3668 to 7477 (2.51 to 5.12) |
| F-scratch e23 (raw) | 0/8 | 0.00 to 0.37 | 1766 to 4830 (1.21 to 3.31) |
| F-scratch e41 (raw) | 0/8 | 0.00 to 0.37 | 517 to 2673 (0.35 to 1.83) |
| F-scratch e42 (raw) | 0/8 | 0.00 to 0.37 | 662 to 1098 (0.45 to 0.75) |
| F-scratch e43 (raw) | 0/8 | 0.00 to 0.37 | 496 to 2511 (0.34 to 1.72) |
| F-scratch e23 (EMA) = FSEMA | 8/8 | 0.63 to 1.00 | none |
| F-depth4 e14 (raw) | 8/8 | 0.63 to 1.00 | none |
| F-depth4 e22 (raw) | 8/8 | 0.63 to 1.00 | none |
| F-depth4 e24 (raw) | 8/8 | 0.63 to 1.00 | none |

| candidate | finite (of 8) | score A | worst group W | A minus B22 (paired mean ± SE; p) | gate 2: water physical | gate 3: \|dPS\| ≤ 1 hPa (range, hPa) | drifting groups (gate 4: ≤ 3) |
|---|---|---|---|---|---|---|---|
| B22 | 8 | 3.47 | 6.35 | reference | pass | fail (-8.49 to -5.97) | 21 |
| B24 | 8 | 4.26 | 6.36 | +0.807 ± 0.048; 6.1e-07 | pass | fail (+12.74 to +17.71) | 23 |
| B22dryair | 8 | 3.52 | 6.29 | +0.067 ± 0.061; 0.311 | pass | pass (-0.17 to -0.08) | 21 |
| B24dryair | 8 | 4.29 | 6.18 | +0.827 ± 0.057; 1.7e-06 | pass | pass (+0.28 to +0.41) | 22 |
| B22posclamp | 8 | 3.46 | 6.36 | -0.002 ± 0.055; 0.966 | pass | fail (-7.09 to -3.59) | 21 |
| DRYAIR24 | 8 | 4.35 | 7.45 | +0.886 ± 0.070; 4.5e-06 | pass | pass (-0.24 to -0.14) | 22 |
| FSEMA | 8 | 3.10 | 4.55 | -0.348 ± 0.055; 4.0e-04 | pass | pass (-0.83 to +0.37) | 18 |
| FSE26 | 8 | 2.20 | 3.16 | -1.261 ± 0.044; 1.7e-08 | pass | fail (+1.74 to +4.01) | 12 |
| FSE30 | 8 | 2.16 | 4.30 | -1.296 ± 0.044; 1.4e-08 | fail (6 of 8 members) | pass (-0.38 to +0.70) | 9 |
| FD22 | 8 | 4.46 | 13.04 | +0.981 ± 0.033; 1.4e-08 | fail (8 of 8 members) | fail (+35.14 to +40.04) | 22 |
| FD24 | 8 | 7.17 | 21.32 | +3.703 ± 0.049; 1.9e-11 | pass | fail (-107.94 to -106.04) | 23 |
| FD24dryair | 8 | 6.86 | 19.38 | +3.421 ± 0.056; 8.3e-11 | pass | pass (-0.69 to -0.56) | 21 |
| FDE28 | 8 | 2.02 | 4.81 | -1.438 ± 0.050; 1.5e-08 | fail (4 of 8 members) | fail (-33.14 to -31.21) | 17 |
| FDE32 | 8 | 1.99 | 4.56 | -1.483 ± 0.055; 2.6e-08 | fail (8 of 8 members) | fail (-34.38 to -32.95) | 18 |
| G1E15 | 8 | 1.59 | 2.63 | -1.874 ± 0.045; 1.3e-09 | pass | fail (-2.40 to -1.36) | 15 |
| G1E15fix | 8 | 1.68 | 2.57 | -1.783 ± 0.039; 6.0e-10 | pass | pass (-0.44 to -0.35) | 17 |
| G1E18 | 8 | 1.71 | 2.91 | -1.755 ± 0.053; 6.2e-09 | pass | fail (+1.04 to +2.45) | 7 |
| G1D20 | 8 | 2.09 | 3.32 | -1.387 ± 0.072; 2.5e-07 | fail (6 of 8 members) | pass (+0.10 to +0.71) | 19 |
| G1D39 | 8 | 1.52 | 2.52 | -1.942 ± 0.051; 2.1e-09 | pass | pass (-0.30 to -0.18) | 14 |
| G1D59 | 8 | 1.52 | 2.34 | -1.937 ± 0.037; 2.4e-10 | fail (7 of 8 members) | fail (-2.3e+20 to +0.88) | 16 |

| checkpoint | A without fix | A with fix | with minus without (paired ± SE; p) | \|dPS\| without → with (max over members, hPa) |
|---|---|---|---|---|
| B22 | 3.47 | 3.52 | +0.067 ± 0.061; 0.311 | 8.49 → 0.17 |
| B24 | 4.26 | 4.29 | +0.020 ± 0.042; 0.650 | 17.71 → 0.41 |
| FD22 | 4.46 | 4.63 | +0.187 ± 0.033; 7.1e-04 | 40.04 → 0.38 |
| FD24 | 7.17 | 6.86 | -0.282 ± 0.019; 1.4e-06 | 107.94 → 0.69 |
| G1E15 | 1.59 | 1.68 | +0.090 ± 0.021; 0.004 | 2.40 → 0.44 |
| G1E16 | 1.60 | 1.82 | +0.221 ± 0.019; 7.5e-06 | 8.00 → 0.67 |
| G1E19 | 1.63 | 1.74 | +0.117 ± 0.022; 0.001 | 4.55 → 0.34 |

**F-scratch, EMA weights, bf16 rollouts**

| epoch | cycle position (0–19) | EMA validation loss | finite (of 8) | score A | max \|dPS\| (hPa) | water gate |
|---|---|---|---|---|---|---|
| 15 | 11 | 0.014898 | 8 | 3.62 | 8.23 | pass |
| 16 | 12 | 0.014753 | 8 | 3.70 | 14.75 | fail |
| 17 | 13 | 0.014639 | 4 | 26.92 | not finite | fail |
| 18 | 14 | 0.014550 | 8 | 3.45 | 17.29 | fail |
| 19 | 15 | 0.014483 | 8 | 3.25 | 11.70 | pass |
| 20 | 16 | 0.014436 | 0 | > 400 | not finite | fail |
| 21 | 17 | 0.014407 | 0 | > 400 | not finite | fail |
| 22 | 18 | 0.014389 | 0 | > 400 | not finite | fail |
| 23 | 19 | 0.014381 | 8 | 3.10 | 0.83 | pass |
| 24 | 0 | 0.015244 | 8 | 4.31 | 7.37 | fail |
| 25 | 1 | 0.014858 | 8 | 2.71 | 20.28 | pass |
| 26 | 2 | 0.014697 | 8 | 2.20 | 4.01 | pass |
| 27 | 3 | 0.014587 | 8 | 2.46 | 15.90 | pass |
| 28 | 4 | 0.014483 | 8 | 2.84 | 1.09 | fail |
| 29 | 5 | 0.014379 | 8 | 2.95 | 2.86 | fail |
| 30 | 6 | 0.014278 | 8 | 2.16 | 0.70 | fail |
| 31 | 7 | 0.014188 | 8 | 3.45 | 6.93 | pass |
| 32 | 8 | 0.014101 | 8 | 2.68 | 15.42 | fail |
| 33 | 9 | 0.014028 | 0 | 10.31 | not finite | fail |
| 34 | 10 | 0.013959 | 0 | 11.13 | not finite | fail |
| 35 | 11 | 0.013901 | 7 | 3.24 | not finite | fail |
| 36 | 12 | 0.013854 | 3 | 4.67 | not finite | fail |
| 37 | 13 | 0.013820 | 5 | 3.20 | not finite | fail |
| 38 | 14 | 0.013789 | 0 | > 400 | not finite | fail |
| 39 | 15 | 0.013772 | 0 | 5.61 | not finite | fail |
| 40 | 16 | 0.013761 | 0 | > 400 | not finite | fail |
| 41 | 17 | 0.013758 | 0 | > 400 | not finite | fail |
| 42 | 18 | 0.013761 | 0 | > 400 | not finite | fail |
| 43 | 19 | 0.013763 | 0 | > 400 | not finite | fail |

**G1, EMA weights, fp32 rollouts**

| epoch | cycle position (0–19) | EMA validation loss | finite (of 8) | score A | max \|dPS\| (hPa) | water gate |
|---|---|---|---|---|---|---|
| 15 | 11 | 0.014827 | 8 | 1.59 | 2.40 | pass |
| 16 | 12 | 0.014684 | 8 | 1.60 | 8.00 | pass |
| 17 | 13 | 0.014570 | 8 | 1.72 | 7.11 | pass |
| 18 | 14 | 0.014480 | 8 | 1.71 | 2.45 | pass |
| 19 | 15 | 0.014412 | 8 | 1.63 | 4.55 | pass |
| 20 | 16 | 0.014364 | 8 | 2.25 | 1417.31 | fail |
| 21 | 17 | 0.014334 | 8 | 41.03 | 3144.48 | fail |
| 22 | 18 | 0.014316 | 8 | 99.69 | not finite | fail |
| 23 | 19 | 0.014308 | 6 | 144.97 | not finite | fail |

**G1-dry, EMA weights, fp32 rollouts with the dry-air constraint**

| epoch | cycle position (0–19) | EMA validation loss | finite (of 8) | score A | max \|dPS\| (hPa) | water gate |
|---|---|---|---|---|---|---|
| 15 | 11 | 0.014754 | 8 | 2.22 | 1.08 | fail |
| 16 | 12 | 0.014608 | 8 | 2.11 | 0.48 | fail |
| 17 | 13 | 0.014491 | 8 | 2.83 | 0.57 | fail |
| 18 | 14 | 0.014399 | 8 | 2.29 | 0.92 | fail |
| 19 | 15 | 0.014329 | 8 | 2.65 | 1.16 | fail |
| 20 | 16 | 0.014279 | 8 | 2.09 | 0.71 | fail |
| 21 | 17 | 0.014246 | 8 | 2.79 | 0.25 | fail |
| 22 | 18 | 0.014227 | 8 | 2.76 | 0.23 | fail |
| 23 | 19 | 0.014218 | 8 | 2.26 | 0.45 | fail |
| 39 | 15 | 0.013594 | 8 | 1.52 | 0.30 | pass |
| 40 | 16 | 0.013582 | 7 | 1.77 | not finite | fail |
| 41 | 17 | 0.013577 | 7 | 1.76 | not finite | fail |
| 42 | 18 | 0.013577 | 6 | 1.77 | not finite | fail |
| 43 | 19 | 0.013579 | 0 | > 400 | not finite | fail |
| 59 | 15 | 0.013331 | 8 | 1.52 | not finite | fail |
| 60 | 16 | 0.013331 | 1 | 1.60 | not finite | fail |
| 61 | 17 | 0.013335 | 0 | > 400 | not finite | fail |
| 62 | 18 | 0.013340 | 2 | > 400 | not finite | fail |
| 63 | 19 | 0.013345 | 0 | > 400 | not finite | fail |

**F-depth4, EMA weights, bf16 rollouts (no warm restarts)**

| epoch | EMA validation loss | finite (of 8) | score A | max \|dPS\| (hPa) | water gate |
|---|---|---|---|---|---|
| 4 | 0.015429 | 8 | 4.29 | 54.51 | pass |
| 8 | 0.015484 | 8 | 3.30 | 53.11 | pass |
| 12 | 0.015494 | 8 | 3.74 | 28.19 | pass |
| 16 | 0.015483 | 8 | 3.12 | 38.15 | pass |
| 20 | 0.015462 | 8 | 2.61 | 54.90 | pass |
| 24 | 0.015434 | 8 | 2.95 | 31.94 | pass |
| 28 | 0.015409 | 8 | 2.02 | 33.14 | fail |
| 32 | 0.015382 | 8 | 1.99 | 34.38 | fail |
| 36 | 0.015357 | 8 | 2.71 | 48.13 | pass |
| 40 | 0.015331 | 8 | 2.22 | 43.73 | pass |
| 44 | 0.015308 | 8 | 2.34 | 42.84 | fail |
| 48 | 0.015287 | 6 | 3.53 | not finite | fail |
| 52 | 0.015264 | 8 | 3.22 | 28.56 | pass |
| 56 | 0.015248 | 8 | 2.63 | 19.70 | pass |
| 60 | 0.015233 | 8 | 3.26 | 17.99 | pass |
| 63 | 0.015220 | 8 | 4.07 | 13.50 | pass |

