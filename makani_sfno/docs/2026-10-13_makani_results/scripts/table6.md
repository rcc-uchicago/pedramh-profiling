| candidate | training precision | depth | dry-air constraint | EMA | finite (of 8) | score A | A minus pick (paired ± SE; p) | worst group W | max \|dPS\| (hPa) | max negative-water area (%) | gates 1–3 | drifting groups (≤ 3) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **G1-dry EMA e39** | fp32 | 1 | trained in | yes | 8 | 1.52 | **pick** | 2.52 | 0.30 | 0.4 | pass | 14 |
| G1-dry EMA e59 | fp32 | 1 | trained in | yes | 8 | 1.52 | +0.005 ± 0.023; 0.842 | 2.34 | not finite | 57.0 | fail (2, 3) | 16 |
| G1 EMA e15 | fp32 | 1 | none | yes | 8 | 1.59 | +0.068 ± 0.021; 0.013 | 2.63 | 2.40 | 0.2 | fail (3) | 15 |
| G1 EMA e15 + fix | fp32 | 1 | at prediction | yes | 8 | 1.68 | +0.158 ± 0.021; 1.4e-04 | 2.57 | 0.44 | 0.8 | pass | 17 |
| G1 EMA e18 | fp32 | 1 | none | yes | 8 | 1.71 | +0.187 ± 0.010; 4.2e-07 | 2.91 | 2.45 | 0.1 | fail (3) | 7 |
| G1 EMA e19 + fix | fp32 | 1 | at prediction | yes | 8 | 1.74 | +0.231 ± 0.026; 4.1e-05 | 2.71 | 0.34 | 0.7 | pass | 16 |
| F-depth4 EMA e32 | bf16 | 5 | none | yes | 8 | 1.99 | +0.459 ± 0.028; 6.9e-07 | 4.56 | 34.38 | 1.3 | fail (2, 3) | 18 |
| F-scratch EMA e30 | bf16 | 1 | none | yes | 8 | 2.16 | +0.646 ± 0.020; 6.8e-09 | 4.30 | 0.70 | 1.7 | fail (2) | 9 |
| F-scratch EMA e23 | bf16 | 1 | none | yes | 8 | 3.10 | +1.594 ± 0.024; 5.3e-11 | 4.55 | 0.83 | 1.0 | pass | 18 |
| B e22 | bf16 | 5 | none | no | 8 | 3.47 | +1.942 ± 0.051; 2.1e-09 | 6.35 | 8.49 | 0.0 | fail (3) | 21 |
| B e22 + fix | bf16 | 5 | at prediction | no | 8 | 3.52 | +2.009 ± 0.034; 1.0e-10 | 6.29 | 0.17 | 0.1 | pass | 21 |
| B e24 | bf16 | 5 | none | no | 8 | 4.26 | +2.749 ± 0.036; 1.7e-11 | 6.36 | 17.71 | 0.1 | fail (3) | 23 |
| DRYAIR e24 | bf16 | 5 | trained in | no | 8 | 4.35 | +2.828 ± 0.041; 3.6e-11 | 7.45 | 0.24 | 0.1 | pass | 22 |
