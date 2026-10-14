# Pre-registration — does a 1-epoch depth-4 fine-tune rescue the soil-free model?

Written 2026-09-30, **committed before submission** (operator: "okay submit the 1-epoch depth-4
fine-tune"). The job's `stime` must be later than this file's commit time.

## Why

The surgical soil-free checkpoint (`surgical_nosoil_7646690`, A e243 minus soil + 20 steps) died
**earlier** than A: leads 293 / 343 vs 595 / 490 (7671841, outcome D). From A, one epoch of
`n_future=4` turned a ~day-150 blow-up into a one-year survivor (`nf4_proxy_b8_r1`: survived, 0
channels past 3σ, PS −63 hPa at 1 yr; 7649647, one start). Does the same fine-tune rescue the
soil-free model? It is the cheapest soil-free + multi-step evidence before F (7660250) exists.

## Design — one `debug` job, `polaris/polaris_fsurg_nf4_proxy.pbs`

1. **Train** `fsurg_nf4_proxy_b8_r1` through the production launcher with **`nf4_proxy_b8_r1`'s exact
   recipe** (its `config.json`): warm start = `surgical_nosoil_7646690/best_ckpt_mp0.tar`,
   `e3sm_alldata_nosoil.yaml` (99 out / 105 in, channel-subset gate), production pack,
   `MULTISTEP=5` (`n_future 4`), 1 node × `LOCAL_BATCH=2` = global 8, **1 full epoch**, LR 4e-4,
   `CosineAnnealingLR` (T_max 100, min 1e-6), **warmup 0** (r1; r2's 1-epoch warmup ran the whole
   epoch at 4e-6), `LR_START=0.01`, fresh optimizer/scheduler/counters/loss, no EMA.
   Expected ≈ 35 min (r1: 2,101.8 s/epoch at the same shape).
2. **Screen** (unchanged `polaris_climate_screen.pbs`, `_tmq` truths, 1,460 leads) from 2044 f1092
   and f1156: **`FsurgNF4`** (step 1's `best_ckpt_mp0.tar`) and the control **`nf4p_r1`**
   (`nf4_proxy_b8_r1:best_ckpt_mp0.tar`, the A-based twin).

Control sanity: `nf4p_r1` from f1092 must again **survive** the year with `n_past_3sigma` 0
(7649647). Otherwise the pipeline changed and the job is not read.

## Outcomes for `FsurgNF4` (`L` = truncation lead; a survivor counts as 1,460)

| outcome | rule | reading |
|---|---|---|
| **R — rescued** | survives the year at **both** starts | Multi-step training rescues the soil-free model as it rescues A; soil is not needed for one-year stability |
| **P — partial** | `L ≥ 1.5 × L(Fsurg)` at both starts (i.e. ≥ 440 / 515), but dies within the year at ≥ 1 | Helps, not enough in one epoch; the full arms (T-anneal-F) remain the test |
| **N — not rescued** | `L < 1.5 × L(Fsurg)` at ≥ 1 start | The soil-free base does not respond like A; a flag for F's arms and for G/H, which also drop soil |

Reported beside the class, not used to decide it: `n_past_3sigma`, PS drift at 1 yr, first channel
past 3σ, each vs `nf4p_r1` at the same start; validation loss (never used to select).

## Strength

One checkpoint, one seed, two starts, one epoch. The base is A minus soil with almost no
retraining, not F. A pointer for the F arms, not a result about them.

PASS = `FSURG_NF4_PROXY_OK` (training `MAKANI_MN_SCALING_OK`, both `CLIMATE_SCREEN_OK`, control ok).
