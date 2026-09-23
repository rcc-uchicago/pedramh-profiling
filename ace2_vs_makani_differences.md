# ACE2 vs makani — what actually differs

*Written 2026-09-23.* Two SFNO climate emulators are trained in this repo on the same
cluster, on the same `180×360` equiangular grid, at the same 6-hourly cadence, with the
same `embed_dim=384` / `num_layers=8` trunk — and they behave completely differently at
long rollout. This document is the list of what is actually different, with the file and
line each row comes from.

**Scope.** This is the **science axis**: what each model computes, what constrains it, and
how it is trained and scored. The **architecture and performance axis** is already covered
by `ACE2_retrain/polaris/ace2_polaris_results.md` **Table 8** (`scale_factor` 1 vs 3, 3.08×
the parameters, 17.3× the activation memory per sample, and why ACE2 cannot fit its
production batch on one node while makani can). The two documents are complementary; this
one does not repeat those rows.

**Provenance.** Every row is read from a config or from a recorded measurement — nothing
here is inferred from a paper. ACE2 rows come from `ACE2_retrain/config_polaris.yaml`
(parsed, not eyeballed); makani rows from `makani_sfno/polaris/e3sm_alldata_full.yaml` and
`makani_sfno/polaris/polaris_pack_alldata_production.pbs`; measurements from
`makani_bench_report.md`, `makani_sfno/docs/2026-09-20_k56_readout_prereg.md`,
`makani_sfno/docs/2026-09-21_lagged_ensemble_result.md` and
`makani_sfno/docs/2026-09-10_ace2_comparison_the_corrector.md`.

---

## 0. The summary, in one paragraph

The two models are far more alike than the results suggest. Same architecture family, same
grid, same cadence, same optimizer family, and both are boundary-forced by prescribed sea
surface temperature, sea ice and insolation. What ACE2 has that our makani run has **zero**
of is: a prognostic variable that is **overwritten with truth at every step** over ~71 % of
the globe, an explicit **conservation and positivity corrector** applied at every step, a
loss with **explicit per-channel weights**, and a **separate tendency normalization**. It
also feeds back **38** channels where we feed back **100**, and keeps precipitation and all
six radiative fluxes as diagnostics that never re-enter the model. ACE2 runs 7300 steps
(5 years) stably; our rollout diverges near step 500.

---

## 1. What each model computes — the channel contract

| | ACE2 | makani (E3SM ALLDATA) | source |
|---|---|---|---|
| inputs | **44** | **107** | `config_polaris.yaml` `stepper.step.config.in_names`; `e3sm_alldata_full.yaml:66-70` |
| outputs | **50** | **101** | same |
| **prognostic** (in ∩ out, fed back every step) | **38** | **100** | parsed from the two configs |
| **forcing** (input only, read from file each step) | **6** — `land_fraction`, `ocean_fraction`, `sea_ice_fraction`, `DSWRFtoa`, `HGTsfc`, `global_mean_co2` | **7** — `lsm`, `topo`, `glacier`, `natveg`, `sst`, `solin`, `ice` | same |
| **diagnostic** (output only, never fed back) | **12** — `PRATEsfc`, `ULWRFsfc`, `ULWRFtoa`, `DLWRFsfc`, `DSWRFsfc`, `USWRFsfc`, `USWRFtoa`, `LHTFLsfc`, `SHTFLsfc`, `TMP850`, `h500`, `tendency_of_total_water_path_due_to_advection` | **1** — `PRECT` | same |
| vertical structure | 8 layers per variable | **18 terrain-following (sigma-like) levels** per variable | `e3sm_alldata_full.yaml:19-22` |
| land-surface reservoirs in the state | **none** | `SOILWATER_10CM`, `TSOI_10CM` | `in_names` vs `channel_names` |
| channels excluded from the pack | — | 54 cloud channels (16 of them exactly constant); 108 of the archive's 162 | `e3sm_alldata_full.yaml:3-7,107-109` |

**Why this row set matters.** Every prognostic channel is a path by which an error at step
`k` becomes an input at step `k+1`. ACE2 keeps precipitation and the entire radiative-flux
set *out* of that loop; we keep everything in it, plus two slow land reservoirs that have no
restoring force. Our measured failure is exactly there: `SOILWATER_10CM`'s global mean falls
6.36 % in 14 days, decreasing on 76 % of steps, on a linear trend extrapolating to −59 % by
step 500 — which is the step at which the rollout diverges
(`2026-09-10_rollout_spectra_and_drift.md`).

The 18 terrain-following levels produce one more artefact with no ACE2 analogue: `Z3_l17`,
a **prognostic channel with 0.271 m of temporal variability**, against which the model
manufactures 1.27 m of anomaly in a single step and 10.22 m by 336 h
(`2026-09-20_k56_readout_prereg.md`).

## 2. Constraints applied at every step

| | ACE2 | makani | source |
|---|---|---|---|
| **prognostic state overwritten with truth** | **yes** — `ocean:` block, `type = "prescribed"`; `surface_temperature` is replaced by the target over `ocean_fraction` **every step** (~71 % of the grid) | **none** — nothing is ever overwritten | `config_polaris.yaml` `stepper.step.config.ocean`; `fme/core/ocean.py:95,119` |
| global dry-air-mass conservation | `conserve_dry_air: true` | absent | `corrector:` block |
| moisture budget correction | `advection_and_precipitation` | absent | same |
| hard non-negativity | **16 named variables** (8 water species, `Q2m`, `PRATEsfc`, 6 radiative fluxes) | absent | same |
| any `corrector` / `conserve_dry_air` / `force_positive` / `moisture_budget` in the codebase | present | **zero hits across every `.py` and `.yaml`** | searched 2026-09-10 |

This is the single largest difference in the document. ACE2 removes error from the feedback
loop on three independent paths at every step; our rollout has none of them.

## 3. Training objective and normalization

| | ACE2 | makani | source |
|---|---|---|---|
| loss | `MSE` | `l2`, `squared: True` | `stepper_training.loss.type`; `e3sm_alldata_full.yaml:110-115` |
| **channel weights** | **17 explicit weights spanning 0.25 → 10** (`h500: 10`, `TMP850: 5`, `ULWRFsfc: 5`, `specific_total_water_1: 0.25`) | **`channel_weights: "constant"`** — all equal | same |
| **normalization** | **dual** — `network` uses `scaling-full-field.nc`, `residual` uses **`scaling-residual.nc`** | single full-field z-score; **`temp_diff_normalization: False`** | `stepper.step.config.normalization`; `e3sm_alldata_full.yaml:75-83,113` |
| why ours is off | — | the converter never wrote `time_diff_stds.npy` | `2026-09-10_rollout_spectra_and_drift.md` §3 |
| prediction target | state | **tendency** (`target: "tendency"`, `normalize_residual: False`) | `e3sm_alldata_full.yaml:158-159` |
| optimizer | `FusedAdam`, weight decay 0.01, AMP on | `AdamW`, weight decay 0.0, `β₂ 0.95`, grad-norm clip 32 | `optimization:`; `e3sm_alldata_full.yaml:138-141` |
| learning rate | 1e-4 (config), 3e-4 production peak | **2.0e-3** production — ceiling measured at (2e-3, 3e-3], does not move with batch size | `makani_bench_report.md` §7e |
| schedule | warm restarts, `T_0=9` over 27 epochs | `CosineAnnealingWarmRestarts` | launcher / config |
| **EMA** | `decay: 0.999`, **`validate_using_ema: true`** | **none** | `ema:`; C1/D1 `warmstart_provenance.txt` |

⚠ Both ACE2 and FCN3 up-weight or re-scale by tendency. **We are the only one of the three
that does neither.** Under `zscore` against `global_stds.npy` with constant channel weights,
each channel's effective weight in the loss is `1/σ_global²`, and for `Z3_l17` that σ is
dominated by *spatial* terrain variation — which is why a 43× inflation of that channel's
temporal variance left no trace in the training loss.

## 4. Data

| | ACE2 | makani | source |
|---|---|---|---|
| source | ERA5 reanalysis | E3SMv3 SSP245-AMIP CTL, **one** SST/REST realization | config paths; `polaris_data_prep_decisions.md:15-16` |
| grid | 180 × 360 | 180 × 360 | `ace2_polaris_results.md` Table 2; `e3sm_alldata_full.yaml:44-45` |
| **cadence** | **6-hourly** | **6-hourly** | store shape `121,262 × 180 × 360`; `dhours: 6` |
| calendar | Gregorian | **noleap**, 1460 steps/year | `ai_rossby_e3sm_zarr_schema.md:67` |
| archive size | one **2,388.77 GB** NetCDF, `lmm_stripe_count 1` | ~1.4 TB pack, one HDF5 per year; 51,100 source files = 35 yr × 1460 | `config_polaris.yaml:14-15`; `polaris_e3sm_variable_reference.md:7` |
| span | ~1940–2023 | **2015–2049** (a forward-looking scenario) | store shape; pack years |
| train split | ≤1995 + 2011–2019 + 2021– | **2015–2044** (30 yr = 43,800 samples/epoch) | `train_loader.dataset.concat`; `polaris_pack_alldata_production.pbs:48-50` |
| validation | 1996–1997 | 2045–2047 | same |
| test | — (2001–2010 plausibly the paper's) | **2048–2049** | same |
| ⚠ gap | **1998–2010 is in NEITHER split** | none | CHANGELOG 2026-09-18 |
| ⚠ trend across the split | in-sample period | train and test sit at **different points of a warming scenario**, and `global_means/stds` + `time_means.npy` are all computed on the train split | `polaris_pack_alldata_production.pbs`; `e3sm_alldata_full.yaml:77-79` |

## 5. Rollout and evaluation

| | ACE2 | makani | source |
|---|---|---|---|
| **training rollout window** | **2 forward applications**, always (`n_forward_steps: 2`) | **1** for the shipped production model (`n_future: 0`); 2 for the C1 fine-tune; 3 and 4 on the ladder | `stepper_training.n_forward_steps`; `e3sm_alldata_full.yaml:127` |
| **stable inference length** | **7300 steps = 5 years** (SHiELD-SOM baseline 10220 = 7 years) | **diverges near step 500** | `inference.n_forward_steps`; `2026-09-10_longroll_blowup_analysis.md` |
| longest scored rollout | 5 years, 16 ICs across 1996 | **K = 56 = 336 h**, 24 ICs | `inference.loader.start_indices`; `2026-09-20_k56_readout_prereg.md` |
| checkpoint selection metric | validation over 1996–1997, on **EMA** weights | **single-step** validation loss (0.01284) | `validate_using_ema: true`; TODO item 3 |
| production cost | 1 node, global batch 8, 27 epochs, ~68–70 node-h | 1 node, global batch 32, 243 epochs, **332,424 updates, 46.3 node-h** | CHANGELOG 2026-09-18; `makani_bench_report.md` §5k |

**Scored skill, makani** (median over 101 channels, `2026-09-20_k56_readout_prereg.md`):
NRMSE **0.519** at 126 h and **0.970** at 336 h; VR **1.019** at 336 h; ACC **0.546** at
336 h; `R_slope` 0.473. The variance ratio staying at ~1.0 rules out mode-averaging
(blurring), so a distributional loss is not indicated for us by that evidence.

## 6. Ensembles — neither model ensembles to improve the deterministic forecast

| | ACE2 | makani | source |
|---|---|---|---|
| ensembles the method actually uses | **three, none of them a snapshot or lagged ensemble** — (1) 8 × 5-year inference runs for checkpoint selection, (2) **3 ICs one day apart** for error bars, (3) SST perturbation (`constant` / `greens_function`) for forced response | lagged ensemble (built and measured), snapshot ensemble (243 checkpoints on disk, unmeasured) | CHANGELOG 2026-09-10; `2026-09-21_lagged_ensemble_result.md` |
| ⚠ IC-ensemble spread | **near zero by construction** — SST is prescribed, so all members share the slow component; fme's Nino3.4 index is computed inside a box that is 100 % ocean and therefore overwritten with truth, making it bit-identical across members | — | `fme/core/ocean.py`; `enso/dynamic_index.py` |
| ⚠ replication trap | `n_ensemble_per_ic` replicates members **exactly**; with N identical members `SSRBiasMetric` returns **−1.0** and CRPS collapses to MAE, logged without a warning | — | `data_loading/perturbation.py:89,109` |
| lagged-ensemble result | not attempted | **−31.7 % RMSE (worse), 0 of 101 channels improved**; monotone in ensemble size; CRPS +28.7 %; SSR 1.66 | job 7643271, `lagged_readout.json` |

## 7. What is NOT different — worth stating, because each was assumed at some point

| | status |
|---|---|
| **architecture family** | both are the Modulus/makani SFNO: `embed_dim 384`, `num_layers 8`, `operator_type dhconv`, `filter_type linear`, `normalization_layer instance_norm`, `hard_thresholding_fraction 1.0`, `use_mlp`, `separable False`. Only `scale_factor` (1 vs 3) and `pos_embed` differ → Table 8 |
| **grid** | 180 × 360 equiangular, both |
| **cadence** | 6-hourly, both. ERA5 is hourly at source and ai2 chose 6-hourly anyway |
| **prescribed boundary forcings** | both read SST, sea ice and insolation from file at every step |
| **training depth, for the C1 fine-tune** | ACE2's `n_forward_steps: 2` and makani's `multistep_count: 2 → n_future: 1` are both **2 forward passes** — exactly equal. ⚠ But the *shipped* makani production checkpoint is `n_future: 0`, i.e. **one** application |
| **cluster, scheduler, fabric** | same Polaris nodes, same PBS, same Slingshot |

## 8. What the differences imply

Rollout error follows roughly `e_{k+1} ≈ J·e_k + ε`, so the quantities that matter are how
much of the state is inside the feedback loop and whether anything projects error back out
of it. ACE2 removes error on three paths (ocean overwrite, corrector, 12 diagnostics kept
out of the loop) and weights its loss so that a channel's influence is a stated choice
rather than a by-product of its spatial variance. We do none of that and feed back 2.6× as
many channels, two of which are unforced slow reservoirs.

The cheap ports, in order of cost:

| port | cost | what it tests |
|---|---|---|
| positivity clamp at inference on the physically non-negative channels (`SOILWATER_10CM`, `PRECT`, `RHREFHT`, humidity levels) | one job, **no retraining** | if the divergence step moves, the reservoir-drift chain is confirmed |
| build `time_diff_stds.npy`, set `temp_diff_normalization: True` | one converter pass + one training arm | closes the gap against **both** ACE2 and FCN3 |
| per-channel loss weights | one training arm | makes near-constant and reservoir channels visible to the loss |
| EMA | config-only | weight-space averaging, the free version of a checkpoint ensemble |
| move fluxes / precipitation out of the feedback loop | **science decision** | 12 diagnostics to our 1; every one is an amplification path we have and they do not |

⚠ A positivity clamp ports trivially; a **moisture-budget correction may not**. Our
101-channel E3SM contract is not obviously budget-closable the way ACE2's
`specific_total_water_*` + `PRATEsfc` + advective-tendency set is. Which channels should be
prognostic at all, and what a closable budget looks like on this variable set, is a science
decision (CLAUDE.md, division of labor) — nothing above is a proposal to change what the
model computes without that sign-off.

## 9. Open items and one discrepancy to reconcile

1. **Channel counts disagree with Table 8.** `ace2_polaris_results.md` Table 8 records ACE2
   as "56 (43 in / 40 out)". Parsing `ACE2_retrain/config_polaris.yaml` today gives
   **44 in / 50 out** (38 prognostic + 6 forcing / 38 + 12 diagnostic). Both cannot be right
   for the same config; most likely Table 8 recorded an earlier variable list. **Table 8's
   parameter, memory and throughput rows are unaffected** — they depend on `scale_factor`
   and the trunk, not on the encoder/decoder width — but the channel row should be corrected
   at its source rather than here.
2. **ACE2 has never been scored at long lead by us**, so the "7300 steps stably" figure is
   ai2's configuration, not our measurement. It is an existence proof for the *recipe*, not
   a result of this project.
3. ACE2 and makani are different-sized models on different data. "ACE2 does X" is a strong
   prior, not proof that X fixes makani. Every row above is a difference; which ones are
   *causes* is what the ports in §8 are designed to discriminate.

## 10. Sources

| what | where |
|---|---|
| ACE2 config (all ACE2 rows) | `ACE2_retrain/config_polaris.yaml` |
| makani config (all makani rows) | `makani_sfno/polaris/e3sm_alldata_full.yaml` |
| makani data split | `makani_sfno/polaris/polaris_pack_alldata_production.pbs:48-50` |
| architecture / performance comparison | `ACE2_retrain/polaris/ace2_polaris_results.md` **Table 8** |
| the corrector finding | `makani_sfno/docs/2026-09-10_ace2_comparison_the_corrector.md` |
| long-rollout divergence and drift | `makani_sfno/docs/2026-09-10_longroll_blowup_analysis.md`, `..._rollout_spectra_and_drift.md` |
| 14-day skill curve and the `Z3_l17` drift | `makani_sfno/docs/2026-09-20_k56_readout_prereg.md` |
| lagged-ensemble result | `makani_sfno/docs/2026-09-21_lagged_ensemble_result.md` |
| rollout-depth ladder | `makani_sfno/docs/2026-09-11_nfuture_ladder_result.md`, `..._2026-09-15_nfuture_ladder_d1_and_replication.md` |
| ACE2 ensemble critique | CHANGELOG `2026-09-10 (cont. 2)`, job 7602723 |

⚠ `ACE2_retrain/` is not on `main` — it lives on `feat/multinode-ddp-port` (PR #12). Paths
above resolve on that branch.
