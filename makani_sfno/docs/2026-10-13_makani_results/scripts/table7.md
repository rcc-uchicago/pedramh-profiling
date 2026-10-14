| effect (negative = fp32 better) | pooled, epochs 15/16/18/19 (mean ± SE) | p | rollouts | per epoch: 15 / 16 / 17 / 18 / 19 |
|---|---|---|---|---|
| rollout fp32 − bf16, bf16-trained (R\|T=bf16) | -1.026 ± 0.068 | 1.3e-06 | 8 | -1.77 / -1.94 / -12.16 (n4) / -0.30 / -0.10 |
| rollout fp32 − bf16, fp32-trained (R\|T=fp32) | -2.160 ± 0.011 | 2.7e-14 | 8 | — / -2.93 / -0.55 / -1.39 / — |
| training fp32 − bf16, bf16 rollouts (T\|R=bf16) | +0.232 ± 0.018 | 4.2e-06 | 8 | — / +0.81 / -11.74 (n4) / -0.35 / — |
| training fp32 − bf16, fp32 rollouts (T\|R=fp32) | -0.826 ± 0.072 | 8.5e-06 | 8 | -0.32 / -0.18 / -0.12 / -1.44 / -1.36 |
| total: fp32/fp32 − bf16/bf16 | -1.852 ± 0.055 | 5.4e-09 | 8 | -2.09 / -2.12 / -12.30 (n4) / -1.74 / -1.46 |
