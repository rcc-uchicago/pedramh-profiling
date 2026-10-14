<!-- Provenance: debug job 7703518 (makani_sfno/polaris/polaris_makani_ace2_audit.pbs @ b8644d57),
     4 parallel Fable 5.1 section auditors + 1 synthesis pass, read-only tools; MAKANI_ACE2_AUDIT_OK.
     Raw sections: $MEMBER_ROOT/runs/audit/makani_vs_ace2/7703518/section_{1..4}.md -->

> **Reviewer spot-checks (operator session, 2026-10-01).** Re-verified in code before landing:
> F1-F9 (this session's facts); §8 item 10 — internal grid is `180//3 = 60` rows, `modes_lat = 60`,
> `self.trans` uses our `sht_grid_type: "equiangular"` override (`OLD/models/networks/sfnonet.py:305,524,546`),
> and makani's own bandlimit for a 60-row equiangular grid is 29 (`OLD/utils/grids.py:37-42`) — the
> *aliasing* consequence stays INFERRED (torch_harmonics quadrature not inspected); §8 item 8 — inference
> autocast follows the checkpoint's `amp_mode` (`SRC/sfno_inference/checkpoint_loader.py:125-138`) and
> production A trained in bf16 (`PRODDOC:165`), so our rollouts run under bf16 autocast; the fed-back
> dtype itself is still open (§10 Q12). Everything else is as the auditors wrote it.

# Audit: ACE2 (ai2 fme) vs makani SFNO climate emulators, code-level synthesis

All four section audits ended with `SECTION_OK`; none was cut off. This document merges them. Where two sections disagreed I read the code; the resolutions are listed in §1.

Path aliases used in every `source` column (all expand to absolute paths):

| alias | path |
|---|---|
| `FME` | `/lus/eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling/.claude/worktrees/makani-port/ACE2_retrain/ace_exp/fme` |
| `MOD` | `FME/ace/models/modulus` |
| `OLD` | `/lus/eagle/projects/lighthouse-uchicago/members/mehta5/conda-envs/sfno-venv/lib/python3.12/site-packages/makani` (pin c9704308) |
| `MAIN` | `/lus/eagle/projects/lighthouse-uchicago/members/mehta5/conda-envs/sfno-venv-main/lib/python3.12/site-packages/makani` (main a0aa4c4f) |
| `WT` | `/lus/eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling/.claude/worktrees/makani-port` |
| `OURS` / `SRC` / `POL` | `WT/makani_sfno` / `OURS/src` / `OURS/polaris` |
| `JSON` | `/lus/eagle/projects/lighthouse-uchicago/members/mehta5/ace2_eamv3/ace2_eamv3_stepper_config.json` |
| `ERA5` | `WT/ACE2_retrain/config_polaris.yaml`; `ERA5INF` = `/lus/eagle/projects/lighthouse-uchicago/members/mehta5/ace2_era5/inference_config.yaml` |
| `A` / `G` | `POL/e3sm_alldata_full.yaml` / `POL/e3sm_alldata_ace2vars.yaml` |
| `PRODDOC` | `OURS/docs/2026-09-03_prod1n_b32_sgdr_checkpoint_usage.md` |
| `PAPER` | `/lus/eagle/projects/lighthouse-uchicago/members/mehta5/papers/duncan2024_readout.md` |
| `PROBE` | `/lus/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_probe/tendency_norm/7646192/tendency_norm_probe.csv` |
| `APPROVALS` | `/lus/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_port/jesswan_approvals.md` |

## 0. Summary

Same in both systems: an SFNO with embed 384, 8 blocks, dhconv zonal filters, GELU, InstanceNorm, MLP ratio 2, on a 180×360 grid at a 6 h step; full-state output at the network (neither predicts a tendency, F1 and `JSON:401`); per-channel z-score normalization; AdamW; bf16 training; no dropout, no noise, no effective gradient clipping.

Different: ACE2 instantiates the ai2 modulus SFNO (concat big skip, learned position embedding, scale_factor 1, Legendre–Gauss SHT, inner linear skip), ours the makani SFNO (additive linear big skip, no pos_embed, scale_factor 3, equiangular SHT at 60 rows). ACE2 divides prognostic errors by tendency-like stds in the loss, runs a corrector and an ocean overwrite inside the training graph, feeds back 34 channels, conditions on next-step insolation, carries an fp32 state at inference, and keeps a checkpoint selected on a 5-year rollout metric. Ours does none of these.

The five differences most likely to explain ACE2's multi-year stability versus our ~500-step divergence, ranked:
1. Loss normalizer in tendency units (ACE2 effective weight up to 1437× ours on PS, `JSON:197,304`) vs full-field σ on every channel, on a pack where A carries channels with r = δ/σ down to 1e-4 (`PROBE:83`) that the loss cannot see.
2. What is fed back and reset: ACE2 pins TS to truth over ocean, clamps water, shifts PS for dry-air mass every step in training and inference (`FME/core/step/single_module.py:443-451`); we feed back 100 (A) / 76 (G) raw channels including net fluxes, constant-fill land fields and quasi-static Z3, with no state edit.
3. Network internals: all learned spatial mixing truncated at degree 59 with only a linear additive full-resolution bypass (`OLD/models/networks/sfnonet.py:639-640`), no position embedding, and an internal equiangular 60-row grid that cannot integrate the modes it keeps, vs lmax 180 everywhere, concat skip into a nonlinear decoder, learned pos_embed, LG quadrature.
4. Selection and averaging: ACE2 keeps a best-inference checkpoint by time-mean RMSE over 7300 steps (`FME/core/generics/trainer.py:746-787`), ERA5 recipe on EMA weights; we select on one-step validation loss of raw weights.
5. Step-level inputs and precision: SOLIN at t+1 (`JSON:127-129`) vs our solin at t; fp32 fed-back state vs bf16 autocast output in our drivers.

## 1. Scope and versions

| item | value | source | tag |
|---|---|---|---|
| ACE2 code | fme 2026.5.1 in-tree at `FME` | `FME/__init__.py:1` | VERIFIED |
| ACE2-EAMv3 (primary) | stepper config + tensor shapes of the pretrained checkpoint, epoch 49, 228 046 batches; training config NOT included | `JSON:1-531` | CONFIG |
| ACE2-ERA5 recipe (secondary) | `ERA5` Polaris port of the ai2 retrain config; `ERA5INF` inference config | `ERA5:1-287`; `ERA5INF:1-22` | CONFIG |
| makani OLD | pin c9704308, production checkpoint A (`prod1n_b32_sgdr`) trained here | `PRODDOC:20-22` | CONFIG |
| makani MAIN | a0aa4c4f, port target; SFNO forward identical to OLD under configs A/G (§3 Table 3.6) | `MAIN/models/networks/sfnonet.py:385-408,656-698` | VERIFIED |
| our configs | A = `A` + launcher overrides in `PRODDOC`; G = `G` as written (ACE2-EAMv3 variable set) | `A:1-160`; `G:1-185` | CONFIG |
| PAPER | ACE on EAMv2 readout, labelled PAPER wherever used | `PAPER` | PAPER |

Resolved disagreements between the section audits (checked in code for this synthesis):
- Ocean overwrite rule: Section 3 said `round(OCNFRAC) == 1`, Section 4 said `OCNFRAC > 0`. Code: `replace_on_mask` rounds the mask and compares to `mask_value=1` (`FME/core/masking.py:26-28`; `FME/core/ocean.py:87-92`), and the docstring says `ocean_fraction>=0.5` (`FME/core/ocean.py:39-41`). Section 3 is right. VERIFIED.
- ERA5 output count: Section 2 wrote "12 of 46"; the list is 50 names = 38 prognostic + 12 diagnostic (`ERA5:236-286`). Sections 1 and 4 are right. VERIFIED.
- fme AMP default: `enable_automatic_mixed_precision: bool = False` (`FME/core/optimization.py:304`). Section 4 is right. VERIFIED.
- makani bandlimit rule used by Section 1 item 5(b): `compute_spherical_bandlimit` gives `(nlat-1)//2` for equiangular and `nlat-1` for legendre-gauss (`OLD/utils/grids.py:37-48`). VERIFIED.
- fme builder defaults: `data_grid="legendre-gauss"`, `pos_embed=True`, `big_skip=True`, `operator_type="diagonal"`, `embed_dim=256`, `num_layers=12`, `spectral_layers=1` (`FME/ace/registry/sfno.py:21-42`). VERIFIED.
- The "~70 %" ocean-overwrite area quoted by Section 4 is not in any file in scope; UNDETERMINED (it is the fraction of cells with OCNFRAC ≥ 0.5, which needs the forcing data).
- EAMv3 EMA use: Section 2's "EMA weights per recipe" applies to the ERA5 recipe only; for EAMv3 it is UNDETERMINED (Section 4).

Session facts F1 to F9 were re-checked by every section and hold.

## 2. Data contract

| aspect | ACE2-EAMv3 | ACE2-ERA5 recipe | makani (ours, A / G) | makani stock default | source | tag | relevance to long-rollout stability |
|---|---|---|---|---|---|---|---|
| grid | 180×360; `area_weights (180,360)` stored, latitude values not in json; builder omits `data_grid` so the SHT is built on `legendre-gauss` | same builder, same default | 180×360, cell-centred lat 89.5…−89.5 descending, lon 0.5…359.5; `model_grid_type` / `sht_grid_type: equiangular` | fme `data_grid="legendre-gauss"`; makani `equiangular` data grid, `legendre-gauss` internal | `JSON:409-419`; `FME/ace/registry/sfno.py:42`; `MOD/sfnonet.py:463,489-508`; `POL/convert_e3sm_to_makani_alldata.py:173-174`; `A:51-53` | CONFIG / VERIFIED | Data-grid vs SHT-grid mismatch changes polar round-trip error; EAMv3's real latitude layout UNDETERMINED (§10 Q2) |
| data pre-filter | UNDETERMINED | UNDETERMINED | none; converter flips rows and NaN-fills, no SHT round-trip | — | `POL/convert_e3sm_to_makani_alldata.py:178-191`; `PAPER:242` (EAMv2 was SHT round-tripped) | VERIFIED / PAPER | ACE-EAMv2 removed polar artefacts before training; we train on raw regridded fields |
| timestep | 6 h | 6 h | 6 h (`dhours 6`, `dt 1`, `STEP_SECONDS 21600`) | — | `JSON:420`; `A:46-47`; `POL/convert_e3sm_to_makani_alldata.py:98` | CONFIG / VERIFIED | same |
| calendar / samples | UNDETERMINED | Gregorian 1940–2022 | noleap, 1460/yr; 30 yr train = 43 800 samples; 1368 updates/epoch at batch 32 | — | `PRODDOC:20-21`; `A:5-8` | INFERRED | epoch arithmetic only |
| vertical coordinate | 8 hybrid layers, `ak/bk` shape (9,), values not printed; used by the corrector | 8 layers, `ak/bk` from data | 18 terrain-following hybrid model levels `l00..l17`, nominal labels 5…1000 hPa; no `hyam/hybm/P0` in the pack | — | `JSON:421-424`; `ERA5:204-235`; `POL/convert_e3sm_to_makani_alldata.py:30-45,108-120`; `APPROVALS:26-27` | CONFIG / VERIFIED | Our lowest level is terrain-following, which makes `Z3_l17` quasi-static (r = 1.05e-4, `PROBE:83`); ACE2 has no such channel |
| inputs | 39 = 34 prognostic + 5 forcing | 44 = 38 + 6 | A: 107 = 100 + 7; G: 83 = 76 + 7 | — | `JSON:86-126`; `ERA5:191-235`; `A:63-70`; `G:73-81`; `SRC/sfno_training/trainer/plasim_trainer.py:671-673` | CONFIG / VERIFIED | feedback-loop width |
| outputs | 44 = 34 + 10 diagnostic | 50 = 38 + 12 | A: 101 = 100 + 1 (PRECT); G: 77 = 76 + 1 | — | `JSON:355-400`; `ERA5:236-286`; `A:66-67`; `G:77-78` | CONFIG | idem |
| prognostic list | PS, TS, T_0..7, specific_total_water_0..7, U_0..7, V_0..7 | PRESsfc, surface_temperature, TMP2m, Q2m, UGRD10m, VGRD10m, air_temperature_0..7, specific_total_water_0..7, eastward/northward_wind_0..7 | A: PS, TREFHT, U10, RHREFHT, PSL, TMQ, FSNT, FSNTOA, SOILWATER_10CM, TSOI_10CM, T/U/V/Z3/RELHUM ×18. G: A minus U10, RHREFHT, PSL, TMQ, SOILWATER_10CM, TSOI_10CM, Z3×18 | — | as above | CONFIG | — |
| diagnostic (out-only) list | LHFLX, SHFLX, surface_precipitation_rate, surface_upward_longwave_flux, FLUT, FLDS, FSDS, surface_upward_shortwave_flux, top_of_atmos_upward_shortwave_flux, tendency_of_total_water_path_due_to_advection | the 10 ERA5 equivalents + TMP850 + h500 | PRECT only; FSNT/FSNTOA are **prognostic** in A and G; our preprocessor strips the diagnostic tail before feedback and requires `n_diagnostic_channels == 1` | stock `append_history` feeds back the whole prediction | `JSON:390-399`; `ERA5:271-282`; `SRC/sfno_training/models/preprocessor.py:98-115`; `A:22-23`; `G:16-18`; `OLD/models/preprocessor.py:236-238` | CONFIG / VERIFIED | All six radiative fluxes are outside ACE2's loop; our two net-SW fluxes are inside it |
| forcing list | LANDFRAC, OCNFRAC, ICEFRAC, PHIS, SOLIN | land_fraction, ocean_fraction, sea_ice_fraction, DSWRFtoa, HGTsfc, global_mean_co2 | lsm (PFTDATA_MASK 0/1), topo (m), glacier (%), natveg (%), sst (°C, land → −1.8), solin, ice | — | `JSON:86-91`; `ERA5:193-198`; `POL/convert_e3sm_to_makani_alldata.py:150-158` | CONFIG / VERIFIED | — |
| forcing time alignment | SOLIN at t+1 (`next_step_forcing_names`); others at t | DSWRFtoa at t+1; others at t | all 7 at the **input** time t in train, validation and rollout: `append_unpredicted_features` uses `unpredicted_inp`, which `append_history` overwrites with the target-time frame after each step | makani has no next-step-forcing concept | `JSON:127-129`; `FME/ace/stepper/single_module.py:1098-1105`; `FME/core/step/single_module.py:79-87`; `OLD/models/preprocessor.py:212-224,236-238,448-463`; `SRC/sfno_training/data/plasim_forcing_dataset.py:395-442`; `SRC/sfno_inference/climate_driver.py:367` | VERIFIED | ACE2 gives the net the insolation of the interval it predicts; ours must extrapolate it: a 6 h phase lag on the diurnal driver |
| SST / skin temperature | TS prognostic but overwritten by target TS where `round(OCNFRAC) == 1` every step (`slab: null`, `interpolate: false`) | same via `surface_temperature` / `ocean_fraction` | sst is a forcing channel; nothing in the state is overwritten; TREFHT (2 m T) is the nearest prognostic and is not a skin temperature | fme `ocean: None`; makani none | `JSON:349-354`; `FME/core/ocean.py:39-48,87-92,118-119,133-137`; `FME/core/masking.py:26-31`; `ERA5:166-168`; `A:69-70` | CONFIG / VERIFIED | ACE2 re-anchors its surface boundary over ocean to truth every 6 h; we never do |
| moisture variable | specific total water (vapour + condensate), 8 layers | same | RELHUM ×18 (A, G), TMQ column vapour (A only); no q, no condensate | — | `JSON:102-109`; `A:63`; `APPROVALS:22-25` | CONFIG | No closable moisture budget on our set, so ACE2's corrector is not portable as-is |
| network normalization | per-channel z-score, means/stds inline | `centering.nc` + `scaling-full-field.nc` | z-score from `global_means/stds.npy`; forcings from `forcing_global_means/stds.npy` | `normalization: "zscore"`; absent → identity | `JSON:239-346`; `ERA5:160-162`; `A:76-79`; `OLD/utils/dataloaders/data_helpers.py:31-44` | CONFIG / VERIFIED | — |
| fills / masks | UNDETERMINED | UNDETERMINED | SOILWATER_10CM NaN→0, TSOI_10CM NaN→270 K over ocean; land masks NaN→0; sst NaN→−1.8 | — | `POL/convert_e3sm_to_makani_alldata.py:135-158` | VERIFIED | Constant-fill regions are inside A's feedback loop (removed in G) |
| NaN handling | `fill_nans_on_normalize/denormalize: false`; loss zeroes pred and target where target is NaN | same defaults | none; a NaN target propagates | fme `False`; makani none | `JSON:132-133,240-241`; `FME/core/normalizer.py:41-42`; `FME/core/loss.py:146-149`; `OLD/utils/losses/lp_loss.py:61-75` | VERIFIED | none (our pack is filled) |

EAMv3 variable → nearest makani channel:

| EAMv3 name | role | makani A | makani G | note | tag |
|---|---|---|---|---|---|
| LANDFRAC | forcing | lsm (0/1 mask, not a fraction) | lsm | `POL/convert_e3sm_to_makani_alldata.py:151` | VERIFIED |
| OCNFRAC | forcing | ABSENT | ABSENT | not derivable: lsm is a mask, ice separate | VERIFIED |
| ICEFRAC | forcing | ice | ice | — | VERIFIED |
| PHIS (m² s⁻²) | forcing | topo (m) | topo | units differ by g | VERIFIED |
| SOLIN at t+1 | forcing | solin at t | solin at t | timing differs | VERIFIED |
| PS | prognostic | PS | PS | — | CONFIG |
| TS | prognostic, ocean-overwritten | ABSENT; nearest TREFHT + sst forcing | same | — | CONFIG |
| T_0..7 | prognostic, 8 mass-weighted layers | T_l00..l17 | same | different vertical basis | CONFIG |
| specific_total_water_0..7 | prognostic | ABSENT; RELHUM_l00..l17, TMQ | RELHUM_l00..l17 | RH not q; no condensate | CONFIG |
| U/V_0..7 | prognostic | U/V_l00..l17 | same | — | CONFIG |
| LHFLX, SHFLX | diagnostic | ABSENT | ABSENT | — | CONFIG |
| surface_precipitation_rate | diagnostic | PRECT (diagnostic) | PRECT | only shared diagnostic; σ 8.3e-8 (m s⁻¹) vs ACE2 mean 2.8e-5 (`JSON:181`) so units differ, which is UNDETERMINED | CONFIG / INFERRED |
| FLUT, FLDS, FSDS, upward LW/SW fluxes | diagnostic | ABSENT; nearest FSNT, FSNTOA (prognostic, net not upward) | same | — | CONFIG |
| tendency_of_total_water_path_due_to_advection | diagnostic | ABSENT | ABSENT | — | CONFIG |

makani channels with no EAMv3 counterpart: A: U10, RHREFHT, PSL, TMQ, SOILWATER_10CM, TSOI_10CM, Z3_l00..l17, FSNT, FSNTOA (as prognostic), glacier, natveg (forcing). G: FSNT, FSNTOA, glacier, natveg (`G:73-74,81`). ACE2-ERA5 adds global_mean_co2, TMP2m/Q2m/UGRD10m/VGRD10m, TMP850/h500 relative to EAMv3 (`ERA5:197-202,281-286`).

## 3. Network

### 3.1 Which class runs

| aspect | ACE2-EAMv3 | ACE2-ERA5 recipe | makani (ours, A / G) | makani stock default | source | tag | relevance to long-rollout stability |
|---|---|---|---|---|---|---|---|
| builder type string | `SphericalFourierNeuralOperatorNet` | same | `nettype: "SFNO"` | n/a | `JSON:60`; `ERA5:145`; `A:82`; `G:93` | CONFIG | — |
| class instantiated | `fme.ace.models.modulus.sfnonet.SphericalFourierNeuralOperatorNet` (ai2cm/modulus fork), NOT the makani variant | same | `makani.models.networks.sfnonet.SphericalFourierNeuralOperatorNet` via entry-point registry, wrapped by physicsnemo `Module.from_torch` | same | `FME/ace/registry/sfno.py:7,14,54-59`; `OLD/models/model_registry.py:158-187`; `OLD/models/networks/sfnonet.py:656`; `MAIN/models/model_registry.py:190-235`; `MAIN/models/networks/sfnonet.py:973-977` | VERIFIED | Different block designs (3.3); "same architecture" claims in the docs are wrong |
| makani variant inside fme | builder `SFNO-v0.1.0` (`FME/ace/models/makani/sfnonet.py`) exists, unused by both ACE2 configs | unused | — | — | `FME/ace/registry/sfno.py:64-128`; `FME/ace/models/makani/sfnonet.py:2,363-364,411-418` | VERIFIED | Checkpoint shapes prove the modulus variant: `inner_skip.*` present, `decoder.0.weight` in-width 423 (`JSON:431-432,526`) |
| how config keys reach the net | `dacite.from_dict(strict)` into the builder dataclass; absent keys take dataclass defaults; net reads `params.<attr>` by `hasattr` | same | `params.to_dict()` splatted as kwargs; unknown keys (`mlp_mode`, `target`) swallowed by `**kwargs` | — | `FME/core/registry/module.py:50-58,139-152`; `MOD/sfnonet.py:377-463`; `OLD/models/model_registry.py:168,187`; `OLD/models/networks/sfnonet.py:292` | VERIFIED | ACE2's `pos_embed`, `big_skip`, `data_grid` are defaults, not config |
| image shape fed to the net | (180, 360) from `dataset_info.img_shape` | same | (180, 360), subsampling factor 1 | — | `JSON:416-419`; `FME/ace/registry/sfno.py:58`; `SRC/sfno_training/data/plasim_forcing_dataset.py:247-250`; `A:44-45` | VERIFIED | — |

### 3.2 Constructor arguments as resolved

| aspect | ACE2-EAMv3 | ACE2-ERA5 recipe | makani (ours, A / G) | makani stock default | source | tag | relevance to long-rollout stability |
|---|---|---|---|---|---|---|---|
| spectral_transform | `sht` | `sht` | `sht` (default) | `sht` | `JSON:57`; `OLD/models/networks/sfnonet.py:260`; `MAIN/models/networks/sfnonet.py:520` | CONFIG/VERIFIED | — |
| filter_type | `linear` → `SpectralConvS2` | `linear` | `linear` → `SpectralConv` | `linear` | `JSON:49`; `MOD/sfnonet.py:99-114`; `A:83`; `OLD/models/networks/sfnonet.py:87-97` | VERIFIED | — |
| operator_type | `dhconv` (config; builder default `diagonal`) | `dhconv` | `dhconv` | `dhconv` | `JSON:53`; `FME/ace/registry/sfno.py:23`; `A:94`; `OLD/models/networks/sfnonet.py:264` | VERIFIED | Both are zonal (l-only) kernels, rotation-equivariant mixing |
| scale_factor | 1 | 1 | **3** | 8 | `JSON:54`; `A:84`; `OLD/models/networks/sfnonet.py:267` | CONFIG/VERIFIED | Internal grid 180×360 vs 60×120, lmax 180 vs 60 (§8 item 3) |
| embed_dim / num_layers | 384 / 8 | 384 / 8 | 384 / 8 | 32 / 4 (makani); 256 / 12 (fme builder) | `JSON:48,52`; `A:85-86`; `OLD/models/networks/sfnonet.py:270-271`; `FME/ace/registry/sfno.py:26-27` | CONFIG | — |
| hard_thresholding_fraction | 1.0 | 1.0 | 1.0 | 1.0 | `JSON:50`; `A:89` | CONFIG | — |
| normalization_layer | `instance_norm` = `nn.InstanceNorm2d(384, eps=1e-6, affine=True, track_running_stats=False)` | same | same class and settings | `instance_norm` | `MOD/sfnonet.py:593-601`; `OLD/models/networks/sfnonet.py:355-360`; `MAIN/models/networks/sfnonet.py:614-621` | VERIFIED | Same op, different placement (3.3) |
| use_mlp / mlp_ratio | True / 2.0 (`mlp_ratio` not a builder field, constructor default) | True / 2.0 | True / 2 | True / 2.0 | `JSON:58`; `MOD/sfnonet.py:355`; `A:90,92`; `OLD/models/networks/sfnonet.py:272-273` | VERIFIED | — |
| activation_function | `gelu` (builder default) | `gelu` | `gelu` | `gelu` | `FME/ace/registry/sfno.py:31`; `MOD/sfnonet.py:557-564`; `A:95` | VERIFIED | — |
| encoder_layers | 1 (builder default) | 1 | 1 (default) | 1 | `FME/ace/registry/sfno.py:32`; `OLD/models/networks/sfnonet.py:277` | VERIFIED | — |
| pos_embed | **True** (builder default), learned `(1,384,180,360)`, confirmed by `module.pos_embed` tensor | **true** (explicit) | **"none"** | `"none"` | `FME/ace/registry/sfno.py:33`; `MOD/sfnonet.py:674-683`; `JSON:531`; `ERA5:158`; `A:96`; `OLD/models/networks/sfnonet.py:278` | VERIFIED | §8 item 8 |
| big_skip | **True** (default), **concatenation** into the decoder; decoder in-width 423 = 384 + 39 | True, concat (428) | **True** (constructor default, not in config), **additive** 1×1 conv `residual_transform` in→out | True (additive) | `FME/ace/registry/sfno.py:34`; `MOD/sfnonet.py:662,715-716,741-742`; `JSON:526`; `OLD/models/networks/sfnonet.py:467-472,639-640` | VERIFIED | §8 item 3 |
| spectral_layers / complex_activation | 3 / `real`; dead for `linear` filter (only `SpectralAttentionS2` reads them) | dead | 3 / `real`, dead | 3 / `real` (makani); 1 (fme builder) | `MOD/sfnonet.py:99-114`; `OLD/models/networks/sfnonet.py:72-97`; `A:87`; `FME/ace/registry/sfno.py:40` | VERIFIED | none |
| separable / rank / factorization | False / 1.0 / None → dense complex weight | same | False / 1.0 | False | `JSON:55`; `MOD/sfnonet.py:103-114`; `MOD/s2convolutions.py:135-148`; `MAIN/models/networks/spectral_convolution.py:80-81` | VERIFIED | — |
| data grid of the outer SHT | `legendre-gauss` (builder default, absent from JSON) | same | `model_grid_type: "equiangular"` | makani `equiangular`; fme `legendre-gauss` | `FME/ace/registry/sfno.py:42`; `MOD/sfnonet.py:463,504-509`; `A:51-53`; `OLD/models/networks/sfnonet.py:261,544-545` | VERIFIED; whether EAMv3 data is Gaussian UNDETERMINED (§10 Q2) | §8 item 10 |
| grid of the internal SHT | hardcoded `legendre-gauss`; with scale_factor 1 all four transforms are the same 180×360 LG transform | same | `sht_grid_type: "equiangular"` (config **overrides** makani's default `legendre-gauss`) on the 60×120 internal grid | `legendre-gauss` | `MOD/sfnonet.py:510-515`; `A:53`; `OLD/models/networks/sfnonet.py:262,546-547` | VERIFIED | §8 item 10 |
| modes kept (lmax, mmax) | (180, 181) | (180, 181) | (60, 61) | from h, w | `MOD/sfnonet.py:467-472`; `OLD/models/networks/sfnonet.py:305-306,524-525` | VERIFIED | Spectral truncation at degree 59 for all learned spatial mixing in ours |
| residual_filter_factor | 1 → `Identity` | 1 | no such option | — | `FME/ace/registry/sfno.py:25`; `MOD/sfnonet.py:481-483,716` | VERIFIED | — |
| spectral-conv bias | **True** (hardwired), tensor `(1,384,1,1)` | True | **False** (default, not in config) | False | `MOD/sfnonet.py:112`; `JSON:429`; `OLD/models/networks/sfnonet.py:290`; `OLD/models/networks/spectral_convolution.py:109-112` | VERIFIED | minor |
| inner_skip / outer_skip | `linear` (Conv2d 384→384 with bias) / `identity` | same | `none` / `linear` (Conv2d 384→384, no bias) | same as ours | `MOD/sfnonet.py:619-620,179-182,209-212`; `OLD/models/networks/sfnonet.py:402-403,150-160,189-199` | VERIFIED | Block wiring differs (3.3) |
| drop_path / dropout | 0 / 0 | 0 / 0 | 0 / 0 | 0 | `MOD/sfnonet.py:359-361,580-581`; `OLD/models/networks/sfnonet.py:279-281,348-349` | VERIFIED | none |
| checkpointing | 0 | 0 | not passed by the production launcher; memory only | 0 | `FME/ace/registry/sfno.py:41`; `MOD/sfnonet.py:704-711,718-721`; `POL/polaris_makani_multinode_scaling.pbs:654-668`; `SRC/sfno_training/train_plasim.py:338` | VERIFIED | none |
| mixed precision around the net | UNDETERMINED for the EAMv3 run (not in JSON); SHT and spectral contraction always fp32/complex64 | `enable_automatic_mixed_precision: true` | bf16 autocast (`--amp_mode bf16`); SHT forced fp32, conv/MLP bf16 | fme False; makani `none` | `MOD/s2convolutions.py:165-169,200-201`; `ERA5:111`; `FME/core/optimization.py:304`; `POL/polaris_makani_multinode_scaling.pbs:668`; `OLD/models/networks/spectral_convolution.py:118-122,133-137` | VERIFIED/CONFIG | — |

### 3.3 The forward pass each system executes

| step | ACE2-EAMv3 (modulus variant) | ACE2-ERA5 recipe | makani (ours, A / G) | makani stock default | source | tag | relevance |
|---|---|---|---|---|---|---|---|
| 1 save big-skip residual | `residual = x` (normalized 39-ch input) | same | `residual = x` (normalized 107/83-ch input) | same | `MOD/sfnonet.py:715-716`; `OLD/models/networks/sfnonet.py:591-604` | VERIFIED | — |
| 2 encoder | `Conv2d(39→384, bias)`, GELU, `Conv2d(384→384, no bias)` | same, 44 in | same structure, 107/83 in | same | `MOD/sfnonet.py:566-577`; `JSON:528-530`; `OLD/models/networks/layers.py:280-339`; `OLD/models/networks/sfnonet.py:336-343` | VERIFIED | identical |
| 3 position embedding | `x = x + pos_embed` | same | none | none | `MOD/sfnonet.py:723-733`; `OLD/models/networks/sfnonet.py:614-623` | VERIFIED | §8 item 8 |
| 4 block grids | all 8 blocks 180×360 LG in and out; `scale_residual=False` everywhere | same | block 0: 180×360 equiangular → 60×120; blocks 1–6: 60×120; block 7: 60×120 → 180×360; `scale_residual=True` in blocks 0 and 7, so their skip residual is the SHT-truncated resampled input | same pattern | `MOD/sfnonet.py:616-617`; `MOD/s2convolutions.py:81-85`; `OLD/models/networks/sfnonet.py:399-400`; `OLD/models/networks/spectral_convolution.py:54-56,118-122` | VERIFIED | §8 item 3 |
| 5 block wiring | `xn = norm0(x)`; `(y, r) = SConv(xn)`, `r = xn`; `y += inner_skip(r)`; GELU; `norm1`; MLP; `out = y + r` (identity skip of the **normed** input) | same | `(y, r) = SConv(x)`, `r = x` (or resampled); `norm0(y)`; GELU; MLP; `norm1`; `out = y + W_o r` (learned 1×1, no bias); no identity path; MLP branch normed **after** the MLP | same | `MOD/sfnonet.py:217-252`; `OLD/models/networks/sfnonet.py:222-250`; `MAIN/models/networks/sfnonet.py:385-408` | VERIFIED | §8 item 12 |
| 6 spectral conv | fp32 `RealSHT(lmax=180, mmax=181, grid=LG)`; dense complex weight stored `(384,384,180,2)`; einsum `bixy,iox->boxy`; inverse SHT; `+ bias` | same | fp32 `RealSHT(lmax=60, mmax=61, grid=equiangular)`; complex64 `(1,384,384,60)`; einsum `bgixy,giox->bgoxy`; no bias | same | `MOD/s2convolutions.py:162-208`; `MOD/contractions.py:184-195`; `MOD/factorizations.py:194-217,240-241`; `JSON:430`; `OLD/models/networks/spectral_convolution.py:65-94,114-142`; `OLD/models/networks/contractions.py:25-28,67-98` | VERIFIED | Same operator, different lmax and grid |
| 7 MLP | `Conv2d(384→768, bias)`, GELU, `Conv2d(768→384, bias)` | same | identical shapes via makani `MLP` | same | `MOD/layers.py:97-137`; `JSON:433-436`; `OLD/models/networks/layers.py:341-425` | VERIFIED | — |
| 8 after blocks | `cat([x, residual])` → 423 ch | 428 | nothing | nothing | `MOD/sfnonet.py:741-742`; `JSON:526` | VERIFIED | §8 item 3 |
| 9 decoder | `Conv2d(423→384, bias)`, GELU, `Conv2d(384→44, no bias)` | `428→384→50` | `Conv2d(384→384, bias)`, GELU, `Conv2d(384→101, no bias)`, output gain 0.5 because big_skip | same | `MOD/sfnonet.py:660-671`; `JSON:525-527`; `OLD/models/networks/sfnonet.py:456-464` | VERIFIED | — |
| 10 output | `decoder(...)` | same | `decoder(...) + residual_transform(residual)`, `Conv2d(107→101, 1×1, no bias)` | same | `MOD/sfnonet.py:744-749`; `OLD/models/networks/sfnonet.py:639-642`; `MAIN/models/networks/sfnonet.py:931-934` | VERIFIED | §8 item 3 |
| 11 what reaches degrees l ≥ 60 of the output | everything (all mixing at lmax 180) | same | only the linear map `W x` of the raw input and harmonics generated by pointwise GELU/MLP/norm in block 7 and the decoder; no learned spectral filter touches l ≥ 60 | same for any scale_factor > 1 | derived from rows 4, 6, 10 | INFERRED | §8 item 3 |

### 3.4 Initialization

| aspect | ACE2-EAMv3 | ACE2-ERA5 recipe | makani (ours, A / G) | makani stock default | source | tag | relevance |
|---|---|---|---|---|---|---|---|
| spectral weights | `scale * randn` with `scale = 1/(in*out) ≈ 6.8e-6`; filter branch effectively zero at init | same | complex normal std `sqrt(2/384) ≈ 0.072`, l=0 row ×√2 | same | `MOD/s2convolutions.py:72-73,148`; `OLD/models/networks/spectral_convolution.py:89-94`; `OLD/models/networks/sfnonet.py:145-160` | VERIFIED | ~10⁴× scale difference at the start; affects training trajectory, not rollout math |
| Conv2d / Linear | `trunc_normal_(std=0.02)`, bias 0, via `self.apply(_init_weights)` | same | He `N(0, 2/fan_in)` hidden; output convs gain 0.5; `residual_transform` std 0.068 (A) / 0.078 (G) | same | `MOD/sfnonet.py:685-697`; `OLD/models/networks/layers.py:309-311,329-331,374-376,399-402`; `OLD/models/networks/sfnonet.py:189-192,467-472` | VERIFIED | Our additive big skip starts as a random dense map, not identity/persistence |
| pos_embed | `trunc_normal_(std=0.02)` | same | n/a | n/a | `MOD/sfnonet.py:683` | VERIFIED | — |
| InstanceNorm affine | weight 1, bias 0 | same | torch default | same | `MOD/sfnonet.py:693-697` | VERIFIED | — |
| fme `parameter_init` | `weights_path: null` → no-op | same | n/a | — | `JSON:32-39`; `FME/ace/stepper/parameter_init.py:122,198-215` | VERIFIED | — |

### 3.5 Parameter counts from verified shapes

| aspect | ACE2-EAMv3 | ACE2-ERA5 recipe | makani (ours, A / G) | source | tag |
|---|---|---|---|---|---|
| encoder | 15,360 + 147,456 | 17,280 + 147,456 | A: 41,472 + 147,456; G: 32,256 + 147,456 | `JSON:528-530` | VERIFIED (arithmetic) |
| pos_embed | 24,883,200 (5.5 %) | same | 0 | `JSON:531` | VERIFIED |
| one block | spectral 53,084,160 real + bias 384 + inner_skip 147,840 + MLP 590,976 + norms 1,536 = 53,824,896 | same | spectral 8,847,360 complex64 (17,694,720 real) + outer_skip 147,456 + MLP 590,976 + norms 1,536 = 9,587,328 numel / 18,434,688 real-equiv | `JSON:429-440`; `OLD/models/networks/spectral_convolution.py:65-94` | VERIFIED |
| decoder + output skip | 162,816 + 16,896 | 164,736 + 19,200 | A: 147,840 + 38,784 + 10,807; G: 147,840 + 29,568 + 6,391 | `JSON:525-527` | VERIFIED |
| total | 455,824,896 real, 1.823 GB fp32, 93.2 % spectral | 455,831,040 | A: 77,084,983 torch numel = 147,863,863 real-equiv, 0.591 GB; G: 77,062,135 numel; 95.7 % spectral | cross-check `WT/ACE2_retrain/polaris/ace2_polaris_results.md:195-197` | VERIFIED |
| checkpoint sanity | — | — | 0.591 GB + 2 AdamW moments ≈ 1.77 GB, matches `best_ckpt_mp0.tar` 1.65 GiB | `PRODDOC:48` | INFERRED |

### 3.6 makani OLD vs MAIN, SFNO path

| aspect | OLD | MAIN | numerics-relevant for A/G? | source | tag |
|---|---|---|---|---|---|
| block wiring, norms, skips, gains | as 3.3 | identical | no | `OLD/models/networks/sfnonet.py:222-250,394-436`; `MAIN/models/networks/sfnonet.py:385-408,656-698` | VERIFIED |
| spectral contraction kernels | einsum under `@torch.compile`, `x.contiguous()` | plain einsum, `contiguous_complex_safe`, `view_as_real(...).reshape(...)` | no (layout only) | `OLD/models/networks/contractions.py:25-28,67-98`; `MAIN/models/networks/contractions.py:21-56`; `MAIN/models/networks/spectral_convolution.py:237-256` | VERIFIED |
| SpectralConv bias add / pos_embed add | `x + bias`, `x + pos_embed` | `.to(x.dtype)` added | no (both off in A/G) | `OLD/models/networks/spectral_convolution.py:139-140`; `MAIN/models/networks/spectral_convolution.py:264-268`; `OLD/models/networks/sfnonet.py:623`; `MAIN/models/networks/sfnonet.py:915` | VERIFIED |
| model-parallel rewrite | `fin`/`fout` comm names, input scatter when `fin > 1` | single `matmul` comm; scatter/gather removed | no (mp size 1 in production) | `OLD/models/networks/sfnonet.py:322-345,606-607,636-637`; `MAIN/models/networks/sfnonet.py:584-602,689,896-898`; `PRODDOC:164-165` | VERIFIED |
| MLP | standard | optional TransformerEngine path, `use_te=False` | no | `MAIN/models/networks/layers.py:739-823` | VERIFIED |
| checkpoint restore | — | `module_from_torch`, `ensure_resampled_shapes` | no | `MAIN/models/model_registry.py:162-167`; `MAIN/models/networks/sfnonet.py:973-977` | VERIFIED |
| grid-type verification | none | `verify_grid_type` rejects our cell-centred rows as not `equiangular` (expects `linspace(-90, 90, 180)`); our `compat.py` waives it | documents the node mismatch, no change to the SHT | `MAIN/utils/grid_types.py:49-66,88-113`; `SRC/sfno_training/compat.py:155-209`; `POL/convert_e3sm_to_makani_alldata.py:173` | VERIFIED |

Conclusion: under configs A/G the SFNO forward pass is mathematically identical at both pins; an OLD checkpoint runs unchanged on MAIN.

### 3.7 Output parameterization

| aspect | ACE2-EAMv3 | ACE2-ERA5 recipe | makani (ours, A / G) | makani stock default | source | tag | relevance |
|---|---|---|---|---|---|---|---|
| residual_prediction / target | `residual_prediction: false` → full normalized next state; the only "residual" is the loss normalizer (F5) | key absent → default False | `target: "tendency"` never read (F1); full state | full state | `JSON:401`; `FME/core/step/single_module.py:69,445-447`; `ERA5:159-165`; `A:158`; `OLD/models/stepper.py:28-52` | VERIFIED | Both nets must learn persistence |
| where persistence can live | decoder MLP over `cat(features, x)`: nonlinear, pointwise, full resolution | same | the additive `residual_transform` (linear, dense across channels, no spatial mixing) plus the decoder | same | `MOD/sfnonet.py:741-747`; `OLD/models/networks/sfnonet.py:639-640` | VERIFIED | §8 item 3 |

## 4. Normalization and loss

### 4.1 Input/output normalization

| aspect | ACE2-EAMv3 | ACE2-ERA5 recipe | makani (ours, A / G) | makani stock default | source | tag | relevance |
|---|---|---|---|---|---|---|---|
| normalizer form | per-variable `(x-mean)/std`, same dict for inputs and outputs; denormalize `x*std+mean` | same | per-channel z-score `(x-bias)/scale` in the dataloader for inputs and targets | absent `params.normalization` → identity | `FME/core/normalizer.py:187-214`; `JSON:239-345`; `OLD/utils/dataloaders/data_helpers.py:31-46`; `OLD/utils/dataloaders/data_loader_multifiles.py:125-136,398-400`; `SRC/sfno_training/data/plasim_forcing_dataset.py:411,432`; `A:76-79`; `G:87-90`; `OLD/utils/loss.py:101-108` | VERIFIED | stds feed the loss weight below |
| network std values | full-field (PS 9380 Pa, TS 22.7 K, T_0 8.62 K) | full-field (`scaling-full-field.nc`, values outside allowed dirs) | full-field `global_stds.npy` (PS σ 9353 Pa in the probe) | n/a | `JSON:304-339`; `ERA5:160-162`; `PROBE` row PS | CONFIG (values) / VERIFIED (mechanism) | identical philosophy |
| other modes | none | none | `zscore` only | `minmax` and per-channel dict exist | `OLD/utils/dataloaders/data_helpers.py:37-41,48-65` | VERIFIED | none |
| forcing channels | own mean/std in the same dict; never in the loss | same | 7 forcings via `forcing_global_means/stds.npy`; never in the loss | identity if no forcing stats passed; ours always passes them | `JSON:86-126,136-147`; `FME/core/loss.py:123,140-145`; `ERA5:191-235`; `SRC/sfno_training/data/plasim_forcing_dataset.py:100-105,413-414`; `SRC/sfno_training/trainer/plasim_trainer.py:188-190`; `A:68-73` | VERIFIED | none |
| diagnostic channels | 10 of 44; in the loss with full-field std | 12 of 50 | PRECT (slot 100); in the loss; never fed back | n/a | `JSON:355-400`; `ERA5:271-282`; `SRC/sfno_training/data/plasim_forcing_dataset.py:421-432`; `SRC/sfno_training/trainer/plasim_trainer.py:150-157` | VERIFIED | none directly |
| history normalization | n/a (`n_ic_timesteps = 1`) | same | `none`; our trainer asserts `none` | `none`, `timediff`, `exponential`, `mean` | `FME/core/step/single_module.py:99-101`; `OLD/models/preprocessor.py:40-53,270-319`; `SRC/sfno_training/trainer/plasim_trainer.py:628-633`; `A:104` | VERIFIED | none for A/G |

### 4.2 Loss definition

Notation: `e_c` physical error, `s_net` network std, `s_loss` loss-normalizer std, `w_c` fme weight, `sigma_c`/`delta_c` our full-field / 6 h-change stds, `r_c = delta_c/sigma_c`, `C` loss channels, `q_ij` area-quadrature weights.

| aspect | ACE2-EAMv3 | ACE2-ERA5 recipe | makani (ours, A / G) | makani stock default | source | tag | relevance |
|---|---|---|---|---|---|---|---|
| loss type | `MSE` = `torch.nn.MSELoss(reduction="none")` then `.mean()` over all dims | `MSE` | `l2` = `GeometricLpLoss(p=2)`, `squared: True`: per (sample, channel) `sum_ij q_ij e²`, `chw`-weighted channel sum, batch mean | fme `MSE`; makani `squared False` (per-channel area-RMSE), `relative False` | `JSON:14`; `ERA5:121`; `FME/core/loss.py:71-75,484-485,589,615-617`; `OLD/utils/loss.py:36`; `OLD/utils/losses/lp_loss.py:38-44,61-75`; `A:110-115`; `G:121-126` | VERIFIED | same functional once squared is on |
| area weighting | none; every cell equal (polar rows over-weighted). `AreaWeightedMSE` exists, unused | none | yes: `equiangular` → `naive` rule `sin(linspace(0, π, nlat))`, rows 0 and 179 (our ±89.5 cell centres) get weight exactly 0 | same rule | `FME/core/loss.py:225-231,486-489`; `FME/core/metrics.py:14-29`; `OLD/utils/grids.py:29,99-107,135-136,171-177`; `OLD/utils/losses/base_loss.py:313-321` | VERIFIED | low; ACE2 is stable without it; our pole rows are unpenalised (`APPROVALS:18` row 3) |
| loss normalizer | explicit `normalization.loss`: prognostics tendency-like (PS 247 Pa, T_0 0.499 K, TS 3.92 K), diagnostics full-field; `residual: null` | `residual: scaling-residual.nc` overrides network stds of every prognostic name via `_combine_normalizers`; diagnostics full-field | implicit loss std = `sigma_c` for every channel | fme loss std = network std if both None; makani same as ours | `JSON:131-238`; `FME/core/normalizer.py:268-286,317-330`; `FME/core/step/single_module.py:106-118`; `ERA5:159-165`; `OLD/utils/loss.py:99-108` | VERIFIED (mechanism) / CONFIG (values) | largest loss difference (4.3, 4.4) |
| static per-channel weights | 14 names, 0.25 to 5; default 1.0 for 30; applied as `loss(w*x, w*y)` → `w²` | 17 names, 0.25 to 10 → `w²` | `channel_weights: "constant"` = `1/C` | fme `{}`; makani `constant` | `JSON:15-30`; `ERA5:122-139`; `FME/core/loss.py:189,305-318`; `OLD/utils/loss.py:146-149`; `OLD/utils/losses/base_loss.py:42-44,233` | VERIFIED | modest next to the normalizer |
| `channel_weights: auto/new auto/custom/pangu` on E3SM names | n/a | n/a | silent no-op: matcher tests lowercase ERA5 names; every E3SM name falls to `else` → normalized to exactly `constant` | same | `OLD/utils/losses/base_loss.py:46-57,59-70,150-151,227-228,233` | VERIFIED | none |
| explicit weight list | n/a | n/a | `channel_weights: [[w_0 … w_{C-1}]]` used as-is (NOT sum-normalized), × `time_diff_scale` if on; must be nested (`chw.shape[1]`) | same (MAIN raises `ValueError` on count mismatch) | `OLD/utils/loss.py:160-164`; `MAIN/utils/loss.py:193-196` | VERIFIED | only makani path to an ACE2-like weighting |
| `temp_diff_normalization` | n/a | n/a | off. If on: `chw_c *= sigma_c / max(delta_c, 1e-4)` with `delta_c` in physical units from `time_diff_stds_path` (required) | False | `OLD/utils/loss.py:152-157,162-163`; `OLD/utils/dataloaders/data_helpers.py:69-78`; `OLD/utils/losses/base_loss.py:236-237` | VERIFIED | gives `1/r_c`, not ACE2's `1/r_c²`; physical-unit clamp hits PRECT |
| per-loss `tendency: True` | n/a | n/a | off; if on, input cancels exactly for absolute squared L2 | False | `OLD/utils/loss.py:120,360-386,398-402` | VERIFIED | none; with `relative: True` would be sample-adaptive but `eps=1e-6` dominates Z3_l17 (INFERRED) |
| dynamic weighting knobs | none | none | all off | `uncertainty_weighting`, `balanced_weighting`, `randomized_loss_weights`, `random_slice_loss`, `relative_weight`, all off | `OLD/utils/loss.py:79-82,172-173,410-434` | VERIFIED | none |
| effective weight on squared error (physical) | `(1/C) w_c² / s_loss,c²` | same with residual stds (values UNDETERMINED) | `(1/C) / sigma_c²` (area-weighted) | makani stock `(1/C)/sigma_c` per sample-RMSE | `FME/core/loss.py:140-151,317-318,71-75`; `OLD/utils/loss.py:443`; `OLD/utils/losses/lp_loss.py:64-75`; `OLD/utils/losses/base_loss.py:233` | INFERRED (algebra) | 4.3 |
| same, relative to full-field z-scoring | `(s_net/s_loss)² w²` = 1437 for PS, 0.104 for `specific_total_water_1`, span 1.4e4 | UNDETERMINED numerically | 1 every channel | 1 | 4.3 | INFERRED | ACE2 pays up to 1400× more for an equal normalized PS error |
| multistep weighting | depth UNDETERMINED; per-step `.total()` summed, weights `(1 + a·step)^-0.5`, `a = sqrt_loss_step_decay_constant` | 2 steps, summed, `a = 0` | A: `n_future 0`; fine-tune arms via `--multistep_count`; `multistep: constant` = `1/(n_future+1)` (average) | fme `a 0.0`, `optimize_last_step_only False`; makani `constant` | `FME/core/loss.py:553,596`; `FME/ace/stepper/single_module.py:1411-1413,1640-1670`; `FME/core/optimization.py:164-166,185-188`; `ERA5:119`; `OLD/train.py:118-119`; `OLD/utils/argument_parser.py:65`; `OLD/utils/loss.py:190-191,206-239,437-440`; `A:127` | VERIFIED | sum vs mean only matters for grad-clip (INFERRED) |
| global-mean loss term | `global_mean_type: null` | default None | none available | fme `LpLoss` on area-weighted global mean, weight 1.0 | `JSON:10-12`; `FME/core/loss.py:258-283,449-453,505-515` | VERIFIED | a direct drift guard; unused by both |
| ensemble / CRPS | off, `n_ensemble` 1 | same | not used | fme `EnsembleLoss`; makani `ensemble_crps` etc. | `JSON:8,85`; `FME/core/loss.py:386-425,492-503`; `OLD/utils/loss.py:42-51` | VERIFIED | not in play |
| what the loss sees | denormalize → corrector → ocean → re-normalize with loss stds; TS over `round(OCNFRAC)==1` cells equals target, error 0 there | same | raw network output; no corrector or SST overwrite in training (opt-ins default off) | fme empty corrector, `ocean None`; makani none | `FME/core/step/single_module.py:443-451`; `FME/core/ocean.py:87-92,118-119,133-137`; `FME/core/prescriber.py:12-14,101-108`; `JSON:62-84,349-354`; `APPROVALS:15-17` | VERIFIED | ACE2 trains through its fixes and never pays for ocean TS |
| checkpoint-selection loss | `best_validation_loss 0.0889`; EMA use UNDETERMINED | `validate_using_ema: true` | single-step validation loss on raw weights (A); G adds an EMA shadow | n/a | `JSON:3`; `ERA5:47-49`; `G:182`; `PRODDOC:24` | CONFIG | none |
| loss state in checkpoints | n/a | n/a | `LossHandler` running-stat buffers saved; restoring across different `n_future` or `C` needs `LOAD_LOSS=0` | `load_loss True` | `OLD/utils/loss.py:184-187`; `OLD/utils/training/deterministic_trainer.py:252`; `POL/submit_c1_rollout_finetune.sh:81-89` | VERIFIED | operational trap only |
| PAPER (ACE-EAMv2) | relative L2 on the full next state, one step, no weights stated | | | fme `LpLoss` is exactly that | `PAPER:209-216,221,225`; `FME/core/loss.py:198-222` | PAPER / VERIFIED | not comparable to either production recipe |

### 4.3 ACE2-EAMv3 effective per-channel weight (all 44 outputs)

Source: `JSON:15-30` (weights), `JSON:187-237` (loss stds), `JSON:295-345` (network stds), `JSON:86-126,355-400` (P/D). Mechanism `FME/core/loss.py:140-151,317-318`. Last two columns are arithmetic (INFERRED from CONFIG): `w²/s_loss²` in inverse physical units squared, and `(s_net/s_loss)²·w²`, the weight on an equal full-field-normalized squared error (ours is 1 everywhere).

| output | P/D | network std | loss std | net/loss | w | w²/loss_std² | (net/loss)²·w² |
|---|---|---|---|---|---|---|---|
| PS | P | 9380 Pa | 247.4 Pa | 37.9 | 1 | 1.63e-5 | **1437** |
| TS | P | 22.69 K | 3.920 K | 5.79 | 1 | 0.0651 | 33.5 |
| T_0 | P | 8.620 | 0.4990 | 17.3 | 0.5 | 1.004 | 74.6 |
| T_1 | P | 10.62 | 0.5890 | 18.0 | 0.5 | 0.721 | 81.3 |
| T_2 | P | 6.064 | 1.103 | 5.50 | 1 | 0.823 | 30.3 |
| T_3 | P | 12.14 | 0.8985 | 13.5 | 1 | 1.239 | 182.6 |
| T_4 | P | 14.77 | 1.152 | 12.8 | 1 | 0.754 | 164.6 |
| T_5 | P | 15.38 | 1.144 | 13.5 | 1 | 0.765 | 180.9 |
| T_6 | P | 15.90 | 1.206 | 13.2 | 1 | 0.687 | 173.8 |
| T_7 | P | 17.83 | 1.276 | 14.0 | 1 | 0.615 | 195.3 |
| U_0 | P | 17.68 | 1.419 | 12.5 | 0.5 | 0.124 | 38.8 |
| U_1 | P | 12.63 | 1.656 | 7.63 | 1 | 0.365 | 58.2 |
| U_2 | P | 16.99 | 3.349 | 5.07 | 1 | 0.0892 | 25.7 |
| U_3 | P | 15.40 | 4.267 | 3.61 | 1 | 0.0549 | 13.0 |
| U_4 | P | 11.90 | 3.119 | 3.81 | 1 | 0.103 | 14.6 |
| U_5 | P | 9.708 | 2.423 | 4.01 | 1 | 0.170 | 16.1 |
| U_6 | P | 8.729 | 2.477 | 3.52 | 1 | 0.163 | 12.4 |
| U_7 | P | 8.235 | 2.825 | 2.91 | 1 | 0.125 | 8.50 |
| V_0 | P | 7.369 | 1.580 | 4.67 | 0.5 | 0.100 | 5.44 |
| V_1 | P | 6.784 | 1.932 | 3.51 | 1 | 0.268 | 12.3 |
| V_2 | P | 11.18 | 4.190 | 2.67 | 1 | 0.0570 | 7.13 |
| V_3 | P | 11.62 | 5.473 | 2.12 | 1 | 0.0334 | 4.50 |
| V_4 | P | 8.772 | 3.963 | 2.21 | 1 | 0.0637 | 4.90 |
| V_5 | P | 6.892 | 2.982 | 2.31 | 1 | 0.112 | 5.34 |
| V_6 | P | 6.159 | 2.878 | 2.14 | 1 | 0.121 | 4.58 |
| V_7 | P | 6.419 | 3.258 | 1.97 | 1 | 0.0942 | 3.88 |
| specific_total_water_0 | P | 2.493e-8 | 3.954e-9 | 6.31 | 0.5 | 1.60e16 | 9.94 |
| specific_total_water_1 | P | 1.018e-6 | 7.875e-7 | 1.29 | 0.25 | 1.01e11 | **0.104** |
| specific_total_water_2 | P | 4.079e-5 | 1.580e-5 | 2.58 | 0.5 | 1.00e9 | 1.67 |
| specific_total_water_3 | P | 4.076e-4 | 1.265e-4 | 3.22 | 1 | 6.25e7 | 10.4 |
| specific_total_water_4 | P | 1.344e-3 | 3.581e-4 | 3.75 | 1 | 7.80e6 | 14.1 |
| specific_total_water_5 | P | 2.334e-3 | 5.651e-4 | 4.13 | 1 | 3.13e6 | 17.1 |
| specific_total_water_6 | P | 3.683e-3 | 6.728e-4 | 5.47 | 1 | 2.21e6 | 30.0 |
| specific_total_water_7 | P | 5.351e-3 | 5.531e-4 | 9.67 | 1 | 3.27e6 | 93.6 |
| LHFLX | D | 80.13 | 80.13 | 1 | 1 | 1.56e-4 | 1 |
| SHFLX | D | 43.45 | 43.45 | 1 | 1 | 5.30e-4 | 1 |
| surface_precipitation_rate | D | 7.284e-5 | 7.284e-5 | 1 | 0.5 | 4.71e7 | 0.25 |
| surface_upward_longwave_flux | D | 99.35 | 99.35 | 1 | 5 | 2.53e-3 | 25 |
| FLUT | D | 48.34 | 48.34 | 1 | 1 | 4.28e-4 | 1 |
| FLDS | D | 94.77 | 94.77 | 1 | 2 | 4.45e-4 | 4 |
| FSDS | D | 224.5 | 224.5 | 1 | 2 | 7.94e-5 | 4 |
| surface_upward_shortwave_flux | D | 67.23 | 67.23 | 1 | 2 | 8.85e-4 | 4 |
| top_of_atmos_upward_shortwave_flux | D | 128.5 | 128.5 | 1 | 2 | 2.42e-4 | 4 |
| tendency_of_total_water_path_due_to_advection | D | 1.127e-4 | 1.127e-4 | 1 | 0.5 | 1.97e7 | 0.25 |

Group shares of the last column (total 3011; uniform would be 2.27 % per channel), INFERRED:

| group | n | share |
|---|---|---|
| PS | 1 | 47.7 % |
| T_0..7 | 8 | 36.0 % |
| U_0..7 | 8 | 6.2 % |
| specific_total_water_0..7 | 8 | 5.9 % |
| V_0..7 | 8 | 1.6 % |
| TS | 1 | 1.1 % (0 over ocean cells) |
| 10 diagnostics | 10 | 1.5 % |

Checks: every diagnostic has loss std == network std; every prognostic has a smaller loss std (`JSON:187-237` vs `295-345`). Max/min = 1437 / 0.104 = 1.4e4.

### 4.4 What the two tendency schemes would do to our channels (from `PROBE`)

`1/r` is what `temp_diff_normalization` multiplies into `chw` (`OLD/utils/loss.py:152-155`); `1/r²` is the ACE2-style `(s_net/s_loss)²`. Shares are of the sum over the set; uniform is 0.99 % (A) and 1.30 % (G). Arithmetic INFERRED from the csv.

| channel set | scheme | total | top channel | top share | next | PS | T group | U | V | RELHUM | PRECT | max/min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A (101) | `1/r` | 10 860 | Z3_l17 9518 | 87.6 % | Z3_l16 2.5 %, Z3_l15 1.3 % | 0.37 % | 1.6 % | 0.76 % | 0.39 % | 0.69 % | 0.0096 % | 1.3e4 |
| A (101) | `1/r²` | 9.07e7 | Z3_l17 9.06e7 | **99.87 %** | Z3_l16 0.08 % | 0.0018 % | 0.002 % | 0.0005 % | 0.0001 % | 0.0006 % | 1e-6 | 1.7e8 |
| G (77) | `1/r` | 425 | PS 40.3 | 9.5 % | T_l16 3.1 %, T_l13 3.1 % | 9.5 % | 40.7 % | 19.5 % | 10.0 % | 17.6 % | 0.25 % | 55 |
| G (77) | `1/r²` | 4536 | PS 1624 | 35.8 % | T_l16 3.9 %, RELHUM_l03 3.9 % | 35.8 % | 39.7 % | 9.2 % | 2.3 % | 11.3 % | 0.024 % | 3.1e3 |
| ACE2-EAMv3 actual | `(s_net/s_loss)² w²` | 3011 | PS 1437 | 47.7 % | T_7 6.5 % | 47.7 % | 36.0 % | 6.2 % | 1.6 % | water 5.9 % | precip 0.008 % | 1.4e4 |

| item | value | source | tag |
|---|---|---|---|
| physical-unit clamp under `temp_diff`: PRECT `delta_c` 7.95e-8 m/s < 1e-4, weight becomes `sigma/1e-4` = 8.3e-4 instead of 1.044; only channel affected | `PROBE` row PRECT; `OLD/utils/loss.py:154` | VERIFIED + INFERRED |
| in G, `temp_diff` is benign: span 55×, PRECT 0.25 % before the clamp, 2e-4 % after | `PROBE`; arithmetic | INFERRED |
| in G, `1/r²` reproduces ACE2-EAMv3's structure (PS 36 % vs 48 %, T 40 % vs 36 %, U 9 % vs 6 %, V 2 % vs 2 %, moisture 11 % vs 6 %) with no hand weights | 4.3 and this table | INFERRED |
| ACE2-equivalent in makani: explicit nested list `[[ (sigma_c/delta_c)² … ]]` (scale to sum 1 yourself) or a `time_diff_stds.npy` holding `delta²/sigma`; `temp_diff` alone gives `1/r` | `OLD/utils/loss.py:152-164` | VERIFIED |
| a `(sigma/delta)²` list on A hands Z3_l17 99.9 %; `make_capped_weights.py` caps `1/r`, not `1/r²` | `POL/make_capped_weights.py:34-39` | VERIFIED |

### 4.5 Which loss differences plausibly matter

| difference | ACE2-EAMv3 | ours A / G | why it matters for drift | tag |
|---|---|---|---|---|
| error in tendency units for prognostics | yes, `(s_net/s_loss)²` 4 to 1437 | no | per-step bias `eps·delta_c` costs `(eps·r_c)²/C` under our loss (invisible for r ~ 1e-2 to 1e-4) but `eps² w²/C` under ACE2's; after N steps the bias is `N·eps·r_c` σ; probe ties 336 h bias share to small r (Spearman −0.65, `WT/polaris_makani_ace2_ports_handoff.md:143`) | INFERRED |
| which slow channels exist | none below r ≈ 0.026 (PS) | A: Z3 ×18 with r to 1e-4, SOILWATER_10CM 0.023; G: slowest is PS 0.025 | on A no reweighting is usable without a cap; on G ACE2-style weighting is well-conditioned (span 3e3) | INFERRED |
| hand weights | 20× span on 14 names, effective 400× | none | second-order | INFERRED |
| corrector and ocean in the training graph | yes (F7) | no | network learns the pre-fix state that makes the fixed state right; an inference-only constraint does not have this property | INFERRED |
| area weighting | none | naive sin rule, pole rows 0 | no evidence either way | INFERRED |
| multi-step loss, sum vs mean | UNDETERMINED / ERA5 2 steps summed | 1 step | not a normalization effect; §7 | VERIFIED (configs) |

## 5. Constraints, corrector, ocean

### 5.1 fme corrector

| aspect | ACE2-EAMv3 | ACE2-ERA5 recipe | makani (ours, A / G) | makani stock default | source | tag | relevance |
|---|---|---|---|---|---|---|---|
| corrector config | `conserve_dry_air: true`, `moisture_budget_correction: advection_and_precipitation`, 15 `force_positive_names` (8 water levels, precip, 5 radiative fluxes, FLUT), energy `null`, `zero_global_mean_moisture_advection: false` | same three on; 16 names (adds Q2m, ULWRFtoa); others default | A: none; opt-in `DryAirFix` and opt-in inference clamp. G: `DryAirFix` impossible (TMQ dropped) | fme dataclass all off; `SingleModuleStepConfig.corrector` builds the empty corrector. makani OLD/MAIN: no corrector concept | `FME/core/corrector/atmosphere.py:121-133`; `FME/core/step/single_module.py:64-66`; `JSON:62-84`; `ERA5:169-188` | VERIFIED + CONFIG | every fme step edits the fed-back state; makani never does unless opt-ins are on |
| order inside `AtmosphereCorrector.__call__` | 1 `force_positive` → 2 `conserve_dry_air` → 3 zero-mean advection (off) → 4 moisture budget → 5 energy (off); positivity first so conservation is not violated later | same | `DryAirFix` is a single hook, no clamp before it | — | `FME/core/corrector/atmosphere.py:181-234` | VERIFIED | dry-air solve uses clamped water; ours uses raw TMQ |
| `force_positive_names` op | `torch.clamp(x, min=0)` in physical units, train and inference; prognostic water (fed back) and diagnostics | same, 16 names | **inference only, opt-in**: `rollout_one_ic` reads `eval_params.force_positive_names` (absent in every config → OFF), applies `torch.maximum(pred, -mean/std)` in z-space. **Absent from `climate_driver.stream_rollout`**. Separate script wraps `longroll` with arms none/soil/moist | OLD nothing. MAIN `NonNegativeConstraint` (train: `x·sigmoid(x/eps)` silu, eps 0.1, leak 0.02; eval: hard clamp) but `ConstraintsWrapper` accepts only `hydrostatic_balance`, so **unreachable from config** | `FME/core/corrector/utils.py:6-11`; `FME/ace/stepper/single_module.py:1110-1125`; `SRC/sfno_inference/rollout_driver.py:123-150,256,269`; `SRC/sfno_inference/climate_driver.py:345-371`; `POL/longroll_force_positive.py:37-41,52-69`; `MAIN/utils/constraints.py:27-113`; `MAIN/models/parametrizations.py:292-303`; `OLD/utils/constraints.py:19-40` | VERIFIED | fme bounds humidity every step; our long-rollout driver has no bound. In G the only non-negative prognostics fed back are the RELHUM levels (`APPROVALS:15`) |
| `conserve_dry_air` math | `dry = ps − g·TWP`, `TWP = (1/g) Σ q_k Δp_k`, `Δp_k = Δak + Δbk·ps`; `error = ⟨dry_gen⟩ − ⟨dry_input⟩` (area-weighted, cos-lat at cell centres); additive uniform shift, ps re-solved holding q fixed: `ps = (dry + Σ Δak·q)/(1 − Σ Δbk·q)`; float64; input = previous corrected state, so global dry air is constant along the trajectory; only PS changes | identical | A opt-in (default OFF): `dry = PS − g·TMQ`, uniform `PS ← PS − err`, **TMQ held fixed** (fme holds q fixed); float64 in z-space; output promoted to ≥ fp32 because a few-Pa shift vanishes in bf16; two `PlasimPreprocessor` hooks called by both stock wrappers → active in training (pre-loss) and inference when `params.conserve_dry_air`; refuses `h/w_parallel_size ≠ 1`. G: constructor raises | none | `FME/core/corrector/atmosphere.py:160-163,243-297`; `FME/core/metrics.py:14-32,283-296`; `FME/core/coordinates.py:244-287`; `FME/core/atmosphere_data.py:168-185`; `JSON:410-424`; `SRC/sfno_training/models/mass_fix.py:37-48,86-88,101-123`; `SRC/sfno_training/models/preprocessor.py:33-48,72-75,84-96`; `OLD/models/stepper.py:34,50,82,100,125,143`; `MAIN/models/stepper.py:94,134,240,267,295,313`; `OLD/utils/training/deterministic_trainer.py:492-493`; `G:14-15` | VERIFIED | removes the one global-mean drift mode the net cannot self-correct; our opt-in matches it on A only; not in `APPROVALS` rows 1-5 |
| `zero_global_mean_moisture_advection` | false | default False | — | subtract area-mean from the advection tendency; with `advection_*` the recomputed advection has zero mean by construction | `FME/core/corrector/atmosphere.py:86-87,300-323,407-415`; `JSON:83` | VERIFIED + INFERRED | none |
| `moisture_budget_correction` | `twp_tend = (TWP_gen − TWP_in)/Δt`; global: `P ← P × (⟨E⟩ − ⟨twp_tend⟩)/⟨P⟩` (multiplicative uniform), `E = LHFLX/2.5e6` untouched; column: `advection ← twp_tend − (E − P)`; no guard on `⟨P⟩ ≈ 0`; both edited fields are **diagnostics**, never fed back | identical | — (no q levels, no advection channel) | — | `FME/core/corrector/atmosphere.py:326-416`; `FME/core/atmosphere_data.py:18-40,255-264`; `FME/core/constants.py:1`; `JSON:420` | VERIFIED | lower than the docs claim: shapes gradients, does not touch the rolled state |
| `total_energy_budget_correction` | null | None | — | `constant_temperature`: uniform ΔT on every `T_k`; needs `DSWRFtoa`/`HGTsfc` | `FME/core/corrector/atmosphere.py:419-501`; `JSON:82` | VERIFIED + CONFIG | none |
| applied in training AND inference? | yes: `step_with_adjustments` inside `predict_generator`, consumed by `train_on_batch → _accumulate_loss`; `StepperOverrideConfig` can override ocean, multi_call, derived forcings, prescribed prognostics but **not the corrector** | same; `ERA5INF` sets no override | `DryAirFix`: both. Inference clamp: inference only, one driver | — | `FME/core/step/single_module.py:406-459`; `FME/ace/stepper/single_module.py:1088-1125,1599-1671,1775-1798,1837-1885`; `ERA5INF:1-22` | VERIFIED | the network is trained through the corrector |

### 5.2 Ocean and prescribed prognostics

| aspect | ACE2-EAMv3 | ACE2-ERA5 recipe | makani (ours, A / G) | makani stock default | source | tag | relevance |
|---|---|---|---|---|---|---|---|
| ocean / SST prescription | `ocean: {TS, OCNFRAC, interpolate: false, slab: null}`; after the corrector, `TS_gen` ← truth TS at t+1 where `round(OCNFRAC) == 1` (OCNFRAC ≥ 0.5); `interpolate: true` would blend; requires TS in both in/out names; runs in training and inference | same with `surface_temperature`/`ocean_fraction`; `interpolate` default False | no prognostic ever overwritten; sst/ice/lsm/topo/glacier/natveg/solin are input-only forcings; TREFHT prognostic everywhere | — | `FME/core/ocean.py:31-48,56-73,87-92,102-137`; `FME/core/prescriber.py:94-108`; `FME/core/masking.py:12-31`; `FME/core/step/single_module.py:450-451`; `JSON:349-354`; `ERA5:166-168`; `A:69-70`; `POL/convert_e3sm_to_makani_alldata.py:151-157,252`; `G:7` | VERIFIED + CONFIG | largest single difference by area; fraction of cells affected UNDETERMINED. PAPER: SST exogenous, T_s prognostic over land/sea-ice (`PAPER:197-204`) |
| slab ocean | not used | not used | — | `T_{t+1} = T_t + (F_net + q_flux)/(ρ d c_p)·Δt`, then prescribed through the same mask | `FME/core/ocean.py:120-129,145-163` | VERIFIED | none |
| `prescribed_prognostic_names` | `[]` | not set | — | overwrite listed prognostics from `next_step_input_data` (inference tool) | `FME/core/step/single_module.py:68,452-458` | VERIFIED | none |

### 5.3 makani-side constraints

| aspect | ACE2 | makani (ours, A / G) | makani stock OLD / MAIN | source | tag | relevance |
|---|---|---|---|---|---|---|
| hydrostatic balance (reparametrizing wrapper) | — | not wired (no `constraints` key in A or G); would not work on our names: needs `^z\d+$`/`^t\d+$` with integer hPa, ours are `T_l00`/`Z3_l00` on hybrid levels; G has no Z | OLD and MAIN: `params.constraints = [{type: hydrostatic_balance}]` → `ConstraintsWrapper` → `_HydrostaticBalanceWrapper` (net outputs L−1 fewer channels, T column rebuilt from Z and T_0); MAIN adds autocast-off fp32 block | `OLD/models/model_registry.py:142-155,191-200`; `OLD/models/parametrizations.py:25-212,217-242`; `MAIN/models/model_registry.py:158-187,239-255`; `MAIN/models/parametrizations.py:26-231,234-328`; `OLD/utils/constraints.py:19-40`; `MAIN/utils/constraints.py:325-348`; `APPROVALS:16,26-27` | VERIFIED | not usable as-is |
| hydrostatic balance (soft projection) | — | not wired; `APPROVALS:16-17` row 2/2b approved verbally, opt-in; needs `hyam/hybm/P0` (pack lacks them); moot for G | MAIN only: `HydrostaticBalanceProjection` (min-norm projection, `strength λ`, climatology offset, moist variant); **not reachable through `ConstraintsWrapper`**; MAIN `model_registry` passes `hydrostatic_balance_means` kwarg when path set (consumer not identified) | `MAIN/utils/constraints.py:116-320`; `MAIN/models/parametrizations.py:292-303`; `MAIN/models/model_registry.py:220-225`; `/lus/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_port/monitor_notes.md:223` | VERIFIED; kwarg consumer UNDETERMINED | A only |
| hydrostatic loss term | — | not in `losses:` | OLD and MAIN loss type `"hydrostatic"`; same name requirement | `OLD/utils/loss.py:32,41`; `MAIN/utils/loss.py:32,41`; `MAIN/utils/losses/hydrostatic_loss.py:25-177` | VERIFIED | same blocker |
| non-negativity | 15 / 16 names hard clamp | `APPROVALS:15` row 1: `NonNegativeConstraint` on PRECT, SOILWATER_10CM, TMQ approved opt-in, **not wired**; in G two are dropped and PRECT is never fed back → zero effect on the G state | MAIN unwired; OLD nothing | `MAIN/utils/constraints.py:27-113`; `/lus/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_port/review/7681379/review.md:55`; `.../review/order_panel/task_inventory.md:47-48`; `G:73` | VERIFIED + CONFIG; G effect INFERRED | approved list constrains nothing fed back in G |
| grid / area weights | training MSE unweighted (F6); cos-lat cell-centre weights for global means, metrics, corrector | loss quadrature `naive` on pole-inclusive nodes, rows 0/179 weight 0 while our rows are cell-centred; measurement script only: channel-mean l2 −7.4e-4 rel, `RELHUM_l00` +2.45 %; eval/inference code already uses cell-centred weights; `APPROVALS:18` row 3 approved opt-in, not implemented | stock as described; MAIN `grids.py` identical | `FME/core/metrics.py:14-32`; `FME/core/gridded_ops.py:281-306`; `OLD/utils/loss.py:127-138`; `OLD/utils/losses/base_loss.py:313-316`; `OLD/utils/grids.py:27-34,99-107`; `OURS/scripts/port_grid_weight_delta.py:1-17,49-53`; `SRC/sfno_inference/climate_driver.py:770-775`; `SRC/sfno_training/compat.py:155-205`; `monitor_notes.md:53` | VERIFIED | small (L1 0.63 %), polar rows unpenalised |
| `weight_decay_mode` | UNDETERMINED (EAMv3); ERA5 `AdamW(fused)` wd 0.01 on all params | wd 0.0 → mode irrelevant; `APPROVALS:19` row 4 approved opt-in | MAIN `get_parameter_groups("full"|"transformer")`, default `"full"`; OLD absent | `FME/core/optimization.py:94-95`; `ERA5:113-115`; `A:125,138`; `MAIN/utils/training/training_helpers.py:43-84` | VERIFIED + CONFIG | none with wd 0 |
| bias correction / static features | — | `add_*` False; no `bias_correction` → identity | subtracts a precomputed bias if configured | `OLD/models/preprocessor.py:76-80,494-497`; `A:99-103` | VERIFIED + CONFIG | none |

## 6. Step and rollout path

| aspect | ACE2-EAMv3 | ACE2-ERA5 recipe | makani (ours, A / G) | makani stock default | source | tag | relevance |
|---|---|---|---|---|---|---|---|
| step wrapper | `multi_call` with `config: null` wraps `single_module`; MultiCall is None, pure pass-through; with a config it re-runs with one forcing scaled (CO2 × m) | `single_module` | `SingleStepWrapper` (n_future 0) or `MultiStepWrapper`; one `self.model(x)` per step | same | `JSON:40-44,406`; `FME/core/step/multi_call.py:67-83,105-112,242-256,291-303`; `FME/core/step/_multi_call.py:148-184`; `OLD/models/stepper.py:22-52,55-117` | VERIFIED | none |
| one step | `step_with_adjustments`: `{**state, **input_forcing}` → normalize → Packer → `module(x)` → (no residual add) → denormalize → corrector → ocean; `next_step_input_data` (t+1 LANDFRAC, OCNFRAC, ICEFRAC, PHIS, SOLIN, TS) only to corrector/ocean | same | preprocessor: `_append_channels` forcing → z-score → `model(x)` → bias-correct (off) → denormalize | same | `FME/core/step/single_module.py:161-169,351-374,443-451`; `FME/core/step/args.py:12-19`; `OLD/models/stepper.py:28-52`; `OLD/models/preprocessor.py:241-268,448-463` | VERIFIED | equivalent information apart from SOLIN timing |
| state fed back | output dict after corrector + ocean: PS shifted, water clamped, TS truth over ocean | same | `inpt = append_history(inpt, pred, step) = pred[:, :n_state]`, unmodified unless `DryAirFix` (wrapper) or `force_positive_names` (`rollout_one_ic` only) | `append_history` returns `x2` | `FME/ace/stepper/single_module.py:1110-1125`; `SRC/sfno_training/models/preprocessor.py:113-115`; `OLD/models/preprocessor.py:236-238`; `SRC/sfno_inference/climate_driver.py:355-367` | VERIFIED | three state resets per step vs none |
| forcing refresh | window per `forward_steps_in_memory`; `forcing_dict[k][:, step]` / `[:, step+1]` | same | `rollout_one_ic`: whole K-frame `tar_forcing` cached once. `climate_driver`: chunks of `chunk_len` read with the dataset's own ops; re-caches input forcing at each boundary; guards two stock traps (`step ≥ cached length` keeps stale forcing; `cache_unpredicted_features(xz=None)` wipes the cache; MAIN keeps the same None-clears semantics) | stock traps | `SRC/sfno_inference/climate_driver.py:216-268,329-337`; `OLD/models/preprocessor.py:219,386-389`; `MAIN/models/preprocessor.py:689-707,739-746` | VERIFIED | correctness of the forcing timeline |
| inference stepper vs training stepper | same `Stepper`; `main()` under `torch.no_grad`, `set_eval()`, `NullOptimization` → autocast no-op → **fp32 inference** regardless of training AMP; state between steps is the fp32 denormalized dict | same; `n_forward_steps 400`, `forward_steps_in_memory 50` | drivers mirror `validate_one_epoch`: `inference_mode` + `autocast(enabled=amp_enabled)` from `config.json`'s `amp_mode`; every launcher trains with bf16, so inference runs **bf16 autocast** and the fed-back tensor is the wrapper's autocast output; `feedback_dtype` logged, not forced | same autocast in stock validation | `FME/ace/inference/inference.py:284,293-363`; `FME/ace/stepper/single_module.py:1127-1210,1382-1384`; `FME/core/optimization.py:103-106,115-120,396-402`; `FME/core/normalizer.py:170,202-214`; `FME/core/generics/inference.py:61-66,117-166`; `ERA5INF:2-3`; `SRC/sfno_inference/checkpoint_loader.py:125-138`; `SRC/sfno_inference/rollout_driver.py:253-268`; `SRC/sfno_inference/climate_driver.py:321-322,350-367`; `OLD/utils/training/deterministic_trainer.py:84-97,626`; `PRODDOC:166` | VERIFIED; fp32-after-denormalize INFERRED from torch promotion | fme carries fp32; ours carries bf16 output (`mass_fix.py:115-119` records bf16 feedback). Repeated rounding of a slowly varying state is a plausible drift source |
| training-rollout graph | `detach_if_using_gradient_accumulation` only detaches when accumulation is on; `optimize_last_step_only` option; depth UNDETERMINED | 2 steps, full BPTT (`use_gradient_accumulation: false`) | A: n_future 0; `push_forward` detach only if `params.multistep.push_forward` (default False) | stock | `FME/core/optimization.py:159-162`; `FME/ace/stepper/single_module.py:1125,1641-1647`; `ERA5:110,119`; `A:127`; `POL/polaris_sfno_alldata_full.pbs:232`; `OLD/models/stepper.py:60-61,77-79`; `MAIN/models/stepper.py:209-210,236-237` | VERIFIED + CONFIG | §7 |
| non-finite handling | none in the loop | same | `ClimateReducer.update` stops at the first non-finite prediction, writes last-finite and first-nonfinite snapshots | — | `SRC/sfno_inference/climate_driver.py:490-496,808-812`; `FME/core/generics/inference.py:117-166` | VERIFIED | diagnostic only |
| inline inference / validation rollout | `best_inference_error 0.0689` implies inline inference ran; length UNDETERMINED | inline inference 7300 steps, 16 ICs, every epoch | `validate_one_epoch`: 4 steps (`valid_autoreg_steps 3`), bf16; `rollout_one_ic` is a copy of that body | stock | `JSON:2`; `ERA5:51-78`; `OLD/utils/training/deterministic_trainer.py:612-661`; `A:128`; `SRC/sfno_inference/rollout_driver.py:239-290` | VERIFIED + CONFIG | §7 selection row |
| PAPER (ACE on EAMv2) | no corrector, no positivity, no budget fixer; precipitation has small negatives from the SHT round-trip; SST exogenous | — | — | — | `PAPER:182-192,197-204` | PAPER | ACE v1 was stable without a corrector; corrector is an ACE2 addition |

## 7. Training recipe

| aspect | ACE2-EAMv3 | ACE2-ERA5 recipe | makani (ours, A / G) | makani stock default | source | tag | relevance |
|---|---|---|---|---|---|---|---|
| optimizer | UNDETERMINED (no training config in json); EAMv2: Adam | `FusedAdam` = `torch.optim.AdamW(fused=True)`, wd 0.01, betas (0.9, 0.999), eps 1e-8 | AdamW, betas (0.9, **0.95**), eps 1e-8, wd 0.0, `foreach=True` | fme `Adam`, lr 1e-3; makani `optimizer_type` required; MAIN `weight_decay_mode "full"` | `JSON:1-408`; `PAPER:136-142`; `ERA5:109-115`; `FME/core/optimization.py:94-99,301-312`; `A:125,138-141`; `OLD/utils/driver.py:646-675`; `MAIN/utils/driver.py:875-927` | CONFIG / VERIFIED / PAPER | β₂ 0.95 is load-bearing for training stability (`PRODDOC:41-44`: 0.999 collapsed), not a rollout lever |
| learning rate | UNDETERMINED; EAMv2 3e-4 | 1e-4 constant (no `scheduler` key → fme default no scheduler) | A: 2.0e-3 peak (config 1e-3 overridden by launcher); G: 1e-3 | fme 1e-3; makani 1e-3 | `ERA5:112`; `FME/core/scheduler.py:23,31-32`; `PRODDOC:22`; `A:120`; `OLD/utils/driver.py:657` | CONFIG / VERIFIED | ours 20× ERA5's, one rung under a measured collapse ceiling |
| schedule + warmup | UNDETERMINED; EAMv2 cosine to zero | none | A: `CosineAnnealingWarmRestarts` T_0 20 epochs, T_mult 1, η_min 1e-6, 3-epoch LinearLR warmup from 0.01×, stepped per epoch; G: `ReduceLROnPlateau` factor 0.5 patience 1 | fme none; makani OLD `lr_start` 0.0, MAIN 1e-5 | `PRODDOC:22`; `OLD/utils/driver.py:695-706`; `MAIN/utils/driver.py:958-981`; `OLD/utils/training/deterministic_trainer.py:377-380`; `A:131-137` | CONFIG / VERIFIED | warm restarts re-raise LR to 2e-3 twelve times |
| epochs | ≥ 49 completed; total UNDETERMINED; EAMv2 50 | `max_epochs 3` in this Polaris port (not in the "byte-identical" list) | A: 243; G: 100 | required | `JSON:4`; `PAPER:140`; `ERA5:1-8,50`; `PRODDOC:21`; `A:123` | CONFIG / PAPER | — |
| batch / updates | 228 046 batches / 49 epochs = 4654 per epoch; batch size UNDETERMINED (×16 ≈ 51 yr, ×8 ≈ 25.5 yr) | global 16 train, 128 validation | A: global 32, 1368 updates/epoch, 332 424 total; G: config 8 (launcher overrides) | makani `--batch_size` global, split over data ranks | `JSON:4-5`; `PAPER:141`; `ERA5:86,102`; `PRODDOC:20-21`; `SRC/sfno_training/train_plasim.py:292-294` | INFERRED / CONFIG | EMA arithmetic |
| n_forward_steps / n_future | UNDETERMINED; EAMv2 1 step | 2 steps, both optimized, summed | A: n_future 0; G: `n_future: 0` but launcher `--multistep_count` (default 1) overwrites it | fme required; makani default 1 | `ERA5:119`; `FME/ace/stepper/single_module.py:1410-1416,1640-1670`; `FME/core/loss.py:596`; `FME/ace/train/train_config.py:269-273`; `A:127`; `OLD/train.py:118-119`; `OLD/utils/argument_parser.py:65` | CONFIG / VERIFIED | ERA5 trains through two applications; our shipped checkpoint through one |
| gradient through all steps | n/a | yes (full BPTT) | at n_future > 0, `MultiStepWrapper` back-propagates through all steps unless `push_forward` | fme no accumulation; makani push_forward False | `FME/core/optimization.py:159-162`; `ERA5:110`; `OLD/models/stepper.py:60-61,78-79`; `MAIN/models/stepper.py:209-210,236-237` | VERIFIED | — |
| corrector / ocean in the training graph | yes (§5) | yes | none; opt-in `DryAirFix` | fme all off; makani none | `JSON:62-84,349-354`; `ERA5:166-188`; `FME/core/step/single_module.py:58-69,443-451` | VERIFIED | trained to produce states the corrector will fix |
| AMP / scaler | UNDETERMINED | bf16 autocast + `GradScaler("cuda")` constructed | bf16 autocast; GradScaler disabled for bf16 (OLD) / `AutocastManager` (MAIN) | fme False; makani `none` | `ERA5:111`; `FME/core/optimization.py:103-106,115-120,304`; `POL/polaris_makani_multinode_scaling.pbs:668`; `OLD/utils/training/deterministic_trainer.py:83-97,186`; `MAIN/utils/training/deterministic_trainer.py:85-91,193`; `OLD/utils/argument_parser.py:39` | CONFIG / VERIFIED | equivalent precision in training |
| grad clipping | none in the fme training path | none | `optimizer_max_grad_norm` 32; measured ‖g‖ ≈ 0.004 so it never engages | makani entrypoints default 1.0 when absent | `OLD/utils/training/deterministic_trainer.py:189,513-517`; `OLD/utils/training/training_helpers.py:100-115`; `OLD/train.py:74-75`; `MAIN/train.py:81-82`; `SRC/sfno_training/train_plasim.py:296-297`; `PRODDOC:65-78` | VERIFIED | effectively unclipped everywhere |
| EMA | UNDETERMINED (fme always writes an `ema` key, decay default 0.9999, so presence would not prove use); EAMv2 used EMA | decay 0.999 per batch, warm-start `min(0.999, (1+n)/(10+n))`; `validate_using_ema: true` | A: none. G: `ema: {enabled: true, decay: 0.9995}`, warmup, post-step hook; raw weights still train/save; second EMA validation writes `best_ckpt_ema_mp0.tar` | fme 0.9999, `validate_using_ema False`; stock makani trainers have no EMA | `FME/core/ema.py:66,70,132-134`; `FME/core/generics/trainer.py:550,708-716`; `ERA5:47-49`; `PAPER:136`; `SRC/sfno_training/trainer/plasim_trainer.py:449-501,887-948`; `SRC/sfno_training/trainer/ema.py:99-103,137-138,167`; `G:176-182`; `FME/ace/train/train_config.py:211,215` | VERIFIED / CONFIG / PAPER | ERA5 averages ≈ 0.16 epoch; G ≈ 1.5 epochs |
| checkpoint selection | json carries `best_validation_loss 0.0889` and `best_inference_error 0.0689`; which file this is UNDETERMINED | two bests: `best_ckpt` by `val/mean/loss`, `best_inference_ckpt` by `inference/time_mean_norm/rmse/channel_mean`; both under EMA weights | `best_ckpt_mp0.tar` by validation loss; G adds `best_ckpt_ema_mp0.tar`; resume scores the newest epoch file, not best | fme as described; makani best-by-val-loss | `JSON:2-3`; `FME/core/generics/trainer.py:448-450,746-787`; `OLD/utils/training/deterministic_trainer.py:402-406`; `MAIN/utils/training/deterministic_trainer.py:443-458`; `SRC/sfno_training/trainer/plasim_trainer.py:937-939`; `PRODDOC:206-213` | VERIFIED | ACE2 can select against drift; we cannot see drift at selection time |
| inline inference metric | time-mean of network-normalized fields over the rollout; per-channel area-weighted RMSE of time-mean vs target; mean over 44 `out_names` | same; 7300 steps, 16 ICs in 1996, `forward_steps_in_memory` 40 | no inline long rollout; 4-step validation per-lead metrics go to HDF5 but `validation loss` sums only `idt == 0` | fme `inference: None` allowed; makani `valid_autoreg_steps` 0 | `FME/ace/aggregator/inference/main.py:252-258,398-400`; `FME/ace/aggregator/inference/time_mean.py:29-39,345-353`; `FME/ace/train/train.py:130-131`; `FME/core/step/single_module.py:172-173`; `ERA5:51-78`; `OLD/utils/metric.py:605-607,636,647`; `A:128`; `OLD/utils/training/deterministic_trainer.py:168` | VERIFIED | a climate-drift metric drove ACE2 selection |
| validation | `val/mean/loss` on validation windows; scheduler stepped per epoch | 1996–1997 | 4-step rollout over 4 380 samples (2045–2047); loss scalar single-step; ReduceLROnPlateau (G) steps on it | — | `FME/core/generics/trainer.py:437,447,453`; `PRODDOC:201-203`; `OLD/utils/training/deterministic_trainer.py:598-661` | VERIFIED | — |
| noise / augmentation | none; `rotate_probability` 0.0 | none | none; yaml `perturb/add_noise` keys are **dead**; live knob `input_noise` unset | fme off; makani `input_noise` None | `FME/ace/data_loading/augmentation.py:23`; `FME/ace/data_loading/config.py:55-57`; `A:153-155`; `OLD/models/preprocessor.py:93-120`; `OLD/utils/driver.py:187-190` | VERIFIED | no model is noise-regularised |
| parameter_init / warm start | from random init, no L2-SP | default | A from scratch; fork supports `--pretrained_checkpoint_path` (weights only) | fme weights_path None; makani `pretrained False` | `JSON:32-39`; `FME/ace/stepper/parameter_init.py:96-120`; `SRC/sfno_training/train_plasim.py:244-255`; `SRC/sfno_training/trainer/plasim_trainer.py:425-446` | CONFIG / VERIFIED | — |
| seed | UNDETERMINED | 3 | not set | fme None | `ERA5:41`; `FME/ace/train/train_config.py:207` | CONFIG | — |
| data splits | UNDETERMINED; EAMv2: 42 yr train after 11 yr spin-up, F2010 repeating SST, next 10 yr validation | train 1940–1995, 2011–2019, 2021–; val 1996–1997; 16 ICs in 1996 | train 2015–2044 (SSP245-AMIP), val 2045–2047, test 2048–2049; stats from train only | — | `PAPER:200,241`; `ERA5:55-72,89-108`; `POL/polaris_pack_alldata_production.pbs:48-50`; `G:23-26` | PAPER / CONFIG | train and test at different points of a warming scenario (INFERRED) |
| normalization stats provenance | inline; computation UNDETERMINED | `centering.nc`, `scaling-full-field.nc`, `scaling-residual.nc`, `time-mean.nc`; computation UNDETERMINED | `global_means/stds.npy` (1×101), `forcing_*` (1×7), `time_means.npy` from packed train files; no `time_diff_stds.npy` | — | `JSON:136-346`; `ERA5:78,161-165`; `POL/convert_e3sm_to_makani_alldata.py:64-71`; `A:71-79` | CONFIG / VERIFIED | — |

## 8. ACE2 mechanisms not in our production stack, ranked

| # | mechanism | what it does (source) | expected effect on stability | cost to port | changes what the model computes (jesswan) | status in our code |
|---|---|---|---|---|---|---|
| 1 | Loss normalizer in tendency units for prognostics | errors divided by residual-like stds, effective `(s_net/s_loss)² w²` up to 1437 on PS (`JSON:187-237,295-345`; `FME/core/normalizer.py:317-330`) | makes per-step bias on slow channels visible to the optimizer; most plausible loss-side driver (§4.5). On A it must be capped (Z3_l17 takes 99.9 % under `1/r²`); on G the uncapped `1/r²` reproduces ACE2's structure | config (explicit nested `channel_weights` list, not sum-normalized, `OLD/utils/loss.py:160-164`) + a `time_diff_stds.npy` built over the full train split + retrain | yes: changes the trained map (objective); inference operator unchanged. Sign-off on the weighting | built-off: `temp_diff_normalization` exists but gives `1/r`; `POL/make_capped_weights.py` caps `1/r`, not `1/r²`; approved-pending status not recorded |
| 2 | Ocean skin-temperature prescription | TS ← truth at t+1 where `round(OCNFRAC)==1`, train and inference (`FME/core/ocean.py:87-92,118-137`; `FME/core/step/single_module.py:450-451`) | resets the surface boundary over ocean every 6 h; surface-driven error cannot accumulate there | contract change (TS and OCNFRAC absent from A/G) + code (prescriber hook in `PlasimPreprocessor`) + retrain | yes | absent |
| 3 | lmax 180 everywhere (scale_factor 1) with concat big skip into a nonlinear decoder, learned pos_embed | all learned mixing at full resolution (`MOD/sfnonet.py:467-472`); `cat([x, residual])` → decoder (`MOD/sfnonet.py:741-747`); `pos_embed` (`MOD/sfnonet.py:674-683`) | in ours scales l ≥ 60 evolve only through a fixed linear map per step (§3.3 row 11), which the one-step loss need not make contractive; pos_embed gives latitude identity to a rotation-equivariant trunk | scale_factor: config-only + retrain (≈ 3× spectral params per block, 8 blocks at 180×360, memory); pos_embed: config-only `pos_embed: "direct"` (`OLD/models/networks/sfnonet.py:475-483`) + retrain; concat skip: code (no makani option) + retrain | yes, all three | absent; `scale_factor` listed as jesswan's call via `WT/ACE2_retrain/polaris/ace2_polaris_results.md` Table 8; PlaSim configs already use `pos_embed: direct` (`SRC/sfno_training/config/plasim_sim52_zgplev_full.yaml:76`) |
| 4 | Hard clamp on the fed-back water state, trained through | `torch.clamp(min=0)` on 15 names before the loss (`FME/core/corrector/utils.py:6-11`; `atmosphere.py:181-234`) | never rolls negative humidity forward | code: wire MAIN `NonNegativeConstraint` into `ConstraintsWrapper` or a preprocessor hook; add the hook to `climate_driver.stream_rollout`; retrain if trained-with | yes | approved-pending (`APPROVALS:15` row 1, opt-in, unwired); inference-only clamp exists in `rollout_one_ic` only; in G the approved list constrains nothing fed back (RELHUM is the candidate, her call) |
| 5 | Global dry-air mass conservation (PS shift) | uniform PS shift so `⟨ps − g·TWP⟩` is constant along the trajectory, float64 (`FME/core/corrector/atmosphere.py:243-297`) | removes the one global-mean drift mode the loss cannot see | A: config-only (`conserve_dry_air: true`) + retrain if trained-with; G: impossible (no TMQ) | yes | built-off: `DryAirFix` (`SRC/sfno_training/models/mass_fix.py`), "diagnostic arm only", TMQ held fixed (fme holds q fixed); not in `APPROVALS` rows 1-5 |
| 6 | Checkpoint selection on a multi-year rollout metric | `best_inference_ckpt` by time-mean normalized RMSE over 7300 steps × 16 ICs (`FME/core/generics/trainer.py:746-787`; `ERA5:51-78`) | selects against drift directly | code (periodic long rollout in the trainer, or offline re-scoring of saved epoch checkpoints with `climate_driver`); no retrain | no (selection only) | absent; `valid_autoreg_steps 3` is the only rollout seen at validation |
| 7 | Next-step insolation | SOLIN fed at the output time (`JSON:127-129`; `FME/ace/stepper/single_module.py:1098-1105`) | removes a 6 h phase lag on the dominant diurnal driver; plausible source of small per-step bias in T and net SW | code in the dataset (append the t+1 forcing frame) + driver forcing timeline + retrain | yes (inputs change) | absent |
| 8 | fp32 state between steps at inference | `NullOptimization` makes autocast a no-op; denormalized fp32 dict carried (`FME/core/optimization.py:396-402`; `FME/core/normalizer.py:202-214`) | few-Pa PS signals are below bf16 resolution (`mass_fix.py:115-119`); avoids accumulated rounding of slow channels | code in our drivers (disable autocast or cast feedback to fp32); no retrain | no (numerical precision of an existing model) | absent: drivers run bf16 autocast from `config.json` (`SRC/sfno_inference/checkpoint_loader.py:125-138`) |
| 9 | EMA weights for validation and shipping | decay 0.999 per batch, `validate_using_ema` (`FME/core/ema.py:66-134`; `ERA5:47-49`) | reduces variance of the shipped map; does not change the loss blind spot | config (G already has it); code for EMA-based selection | no (weight averaging), but changes shipped weights, flag it | built-off in A (no `ema:` block), on in G with a sibling EMA checkpoint; selection still on raw |
| 10 | Legendre–Gauss SHT (data grid and internal grid) | `data_grid` default LG, internal LG (`FME/ace/registry/sfno.py:42`; `MOD/sfnonet.py:504-515`) | our internal equiangular 60-row grid resolves only degree 29 by makani's own rule (`OLD/utils/grids.py:37-48`) while 59 are kept, so degrees 30–59 alias in seven of eight forward SHTs (INFERRED, torch_harmonics not inspected, §10 Q3) | config-only: delete the `sht_grid_type` override (makani default is LG) + retrain | yes | absent (`A:53` overrides to equiangular) |
| 11 | Two-step training with full BPTT (ERA5 recipe only; EAMv3 UNDETERMINED) | `n_forward_steps 2`, losses summed (`ERA5:119`; `FME/core/optimization.py:159-162`) | secondary: EAMv2 was stable at 1 step (`PAPER:225`); our n_future ladder gave accuracy without stability | config (`--multistep_count 2`) + retrain | yes (objective) | built, run (ladder arms) |
| 12 | Block wiring (inner linear skip, identity outer skip of the normed input, norm before SConv) | `MOD/sfnonet.py:217-252` vs `OLD/models/networks/sfnonet.py:222-250` | not determinable from code which is more drift-resistant | code (fork) + retrain | yes | absent |
| 13 | Per-channel hand weights (w², 14 names) | `loss(w*x, w*y)` (`FME/core/loss.py:317-318`; `JSON:15-30`) | second-order next to item 1 | config (explicit list) + retrain | yes | absent |
| 14 | Diagnostics kept out of the loop (all radiative fluxes out-only) | out-only names ignored by the input packer (`FME/core/step/step.py:69-70`) | FSNT/FSNTOA errors cannot feed back | contract + code (`n_diagnostic_channels` must be 1 in our preprocessor, `SRC/sfno_training/models/preprocessor.py:98-115`) + retrain | yes | absent |
| 15 | Moisture budget correction | rescales precipitation and recomputes advection, diagnostics only (`FME/core/corrector/atmosphere.py:326-416`) | low: never touches the fed-back state | not portable on our contract (no q levels, no advection channel) | yes | absent, not portable |
| 16 | Weight decay 0.01 (ERA5 only) | `AdamW(fused=True)` wd on all params (`FME/core/optimization.py:94-95`; `ERA5:113-115`) | unknown; mild regularization | config-only + retrain | training recipe; `APPROVALS:19` row 4 covers the mode | approved-pending (mode), wd value unset |

Not an ACE2 mechanism but in the same decision space: cell-centred area weights in the training loss (`APPROVALS:18` row 3, approved opt-in, unimplemented). ACE2 uses no area weighting at all.

## 9. Corrections to existing docs

| file:line | wrong claim | correct claim | evidence |
|---|---|---|---|
| `WT/ace2_vs_makani_differences.md:87`; `WT/ace2_e3sm_channel_contract.md:141,230-231`; `OURS/docs/2026-09-10_ace2_comparison_the_corrector.md:141,230` | makani predicts a tendency; "porting means dividing our tendency error by tendency σ" | `target: "tendency"` is never read; both systems emit the full state; for absolute squared L2 the full-state error equals the tendency error, so the port is a direct reweighting | `OLD/models/stepper.py:28-52`; `JSON:401`; F1 |
| `WT/ACE2_retrain/polaris/ace2_polaris_results.md:179-192`; `ace2_vs_makani_differences.md:143` | "SAME SFNO, one config knob apart"; "only `scale_factor` and `pos_embed` differ" | ACE2 instantiates the ai2cm/modulus SFNO, ours the makani SFNO: big skip concat vs additive linear, inner linear skip + identity outer skip vs none + learned outer skip, norm placement, spectral-conv bias, data/internal grids, initialization | §3.2 to 3.4; `FME/ace/registry/sfno.py:7,14`; `JSON:431-432,526` |
| `ace2_vs_makani_differences.md:4,144`; `ace2_polaris_results.md:181` | "180×360 equiangular, both" | ACE2's SHT is built with `data_grid="legendre-gauss"` (builder default, not set in JSON) | `FME/ace/registry/sfno.py:42`; `MOD/sfnonet.py:504-509` |
| `ace2_polaris_results.md:193` | ERA5 "56 (43 in / 40 out)" | ERA5 recipe 44 in / 50 out; EAMv3 39 in / 44 out | `ERA5:191-286`; `JSON:86-126,355-400` |
| `ace2_polaris_results.md:195` | makani "147,860,000" params | correct only as real-equivalent; torch `numel()` is 77,084,983 (complex64 dhconv weights) | §3.5 |
| `WT/ACE2_retrain/config_polaris.yaml:33-34` | "2.67 GB of parameters" | 455,831,040 fp32 params = 1.823 GB | §3.5; `ace2_polaris_results.md:196` agrees |
| `A:50-53`; `G:55-58` comment | "E3SM is equiangular, NOT legendre-gauss" applied to `sht_grid_type` | `sht_grid_type` is the internal 60-row grid, not the data grid; makani's default LG is the exact-quadrature choice there | `OLD/models/networks/sfnonet.py:262,546-547`; `OLD/utils/grids.py:37-48` |
| `ace2_vs_makani_differences.md:85,87`; `2026-09-10_ace2_comparison_the_corrector.md:86-87` | "dual normalization, `residual` uses `scaling-residual.nc`" for ACE2 generally | that is the ERA5 recipe; EAMv3 has `residual: null` and an explicit `loss` dict | `JSON:131,347`; `ERA5:159-165` |
| `ace2_e3sm_channel_contract.md` §5.3 table | T_0 effective 300× | omits `w`: T_0 is 74.6× (w = 0.5); U_0, V_0, water_0..2 likewise reduced | `JSON:18-24`; §4.3 |
| `OURS/docs/2026-09-23_architect_review_ace2_ports.md:94-95`; `WT/polaris_makani_ace2_ports_handoff.md:177` | "ACE2's hand-picked span is 40×" | EAMv3's effective per-channel weight spans 1.4e4 once loss stds are included; a `W_max=30` cap on `1/r` is far more conservative than ACE2, and `1/r` is the square root of ACE2's scaling | §4.3, 4.4 |
| `polaris_makani_ace2_ports_handoff.md:43,126` | `temp_diff` block at `loss.py:181-184`; `temp_diff` "= ACE2's scheme" | old pin `:152-157`; `temp_diff` gives `1/r_c`, ACE2's is `1/r_c²` | `OLD/utils/loss.py:152-157` |
| `APPROVALS:28` | "ACE2 trains with an UNWEIGHTED MSE" | true only of area weighting; EAMv3 is per-channel weighted by `w²/s_loss²` | `JSON:15-30,187-237`; `FME/core/loss.py:317-318` |
| `ace2_vs_makani_differences.md:84`; `2026-09-10_ace2_comparison_the_corrector.md:86` | "17 explicit weights spanning 0.25 to 10" (as ACE2); "20×" | ERA5 recipe: 17 weights, 40× nominal, 1600× effective; EAMv3: 14 weights, 20× nominal, 400× effective (F6) | `JSON:15-30`; `ERA5:122-139` |
| `2026-09-10_ace2_comparison_the_corrector.md:57-60`; `ace2_vs_makani_differences.md:74`; `2026-09-23_architect_review_ace2_ports.md:41`; `polaris_makani_ace2_ports_handoff.md:41` | "makani has zero constraint/corrector machinery anywhere" | OLD has `ConstraintsWrapper`/`_HydrostaticBalanceWrapper` and the `"hydrostatic"` loss; MAIN adds `NonNegativeConstraint` and `HydrostaticBalanceProjection` (unwired); the fork has opt-in `DryAirFix`. No dry-air, moisture-budget or positivity corrector in OLD | `OLD/models/parametrizations.py:25-242`; `OLD/utils/loss.py:41`; `MAIN/utils/constraints.py:27,116`; `SRC/sfno_training/models/preprocessor.py:33-48` |
| `2026-09-10_ace2_comparison_the_corrector.md:70-75` (§3, §4) | corrector touches a land reservoir; global moisture budget; "8 water species" | only PS and `specific_total_water_0..7` are fed-back prognostics the corrector edits; the precipitation rescale is global, the advection recompute column-local; 8 levels of one species | `FME/core/corrector/atmosphere.py:185,296,385-397,407-415` |
| `2026-09-10_ace2_comparison_the_corrector.md:100` | "the corrector is an inference-time mechanism" | runs before the loss in training | `FME/core/step/single_module.py:448-451`; `FME/ace/stepper/single_module.py:1649-1660` |
| `ace2_vs_makani_differences.md:70` | ocean `type = "prescribed"` field; replaced "over `ocean_fraction`" | `OceanConfig` fields are `surface_temperature_name`, `ocean_fraction_name`, `interpolate`, `slab`; rule is `round(OCNFRAC) == 1`; the "~71 %" figure is UNDETERMINED | `FME/core/ocean.py:45-48`; `FME/core/masking.py:26-31` |
| `2026-09-23_architect_review_ace2_ports.md:78-93` | where the clamp goes | `force_positive_names` exists only in `rollout_one_ic`; `climate_driver.stream_rollout` applies no clamp | `SRC/sfno_inference/rollout_driver.py:123-150`; `SRC/sfno_inference/climate_driver.py:345-371` |
| `APPROVALS` rows 1-5 | — | omit the implemented `DryAirFix`; its approval status is unrecorded | `SRC/sfno_training/models/mass_fix.py` |
| `APPROVALS:16,24` | "our plev data is interpolated"; "RELHUM at 10 pressure levels" | levels are 18 terrain-following hybrid levels `l00..l17` (the doc's own line 26 says so); 100 = 10 + 5×18 | `POL/convert_e3sm_to_makani_alldata.py:30-45`; `A:63` |
| `ace2_vs_makani_differences.md:89-90`; `polaris_makani_ace2_ports_handoff.md:277` | ERA5 "1e-4 config, 3e-4 production peak", "warm restarts T_0=9"; "0.999 at ~12,234 updates/epoch" | `ERA5` has no `scheduler` key → no schedule; batch 16 → ≈ 6.1k updates/epoch; 3e-4 is EAMv2's (PAPER). The cited CHANGELOG is outside scope, so UNVERIFIED rather than disproved | `FME/core/scheduler.py:23,31-32`; `ERA5:86,112`; `PAPER:139` |
| `ace2_vs_makani_differences.md:122` | ACE2 selection = "validation over 1996–1997 on EMA weights" | fme also keeps `best_inference_ckpt` by the 5-year time-mean metric; EAMv3's `best_inference_error` shows it ran | `FME/core/generics/trainer.py:746-787`; `JSON:2` |
| `ace2_e3sm_channel_contract.md:145-146` | "EMA present in the checkpoint" proves use; "50 epochs" | fme writes an `ema` key regardless; `epoch: 49` is completed epochs at write time, possibly a best from a longer run | `FME/core/generics/trainer.py:711,715`; `JSON:4` |
| `ace2_vs_makani_differences.md:88` | "same optimizer family" | both AdamW (`FusedAdam` = `AdamW(fused=True)`); difference is wd 0.01 vs 0 | `FME/core/optimization.py:94-95` |
| `2026-09-10_ace2_comparison_the_corrector.md:16-19` | `n_forward_steps: 2` in "every ACE2 baseline config" | verified for `ERA5` only; three other configs are outside scope; EAMv3 depth UNDETERMINED | `ERA5:119` |
| `ace2_e3sm_channel_contract.md:142` | training depth not stored | correct, kept for the record | `JSON` |

## 10. Open questions / UNDETERMINED

1. EAMv3 training recipe: optimizer, LR, schedule, total epochs, batch size, `n_forward_steps`, `optimize_last_step_only`, EMA, AMP, data years, inline-inference length. Only 4654 batches/epoch is recoverable (`JSON:4-5`). Needs ai2's training config or the tar's `optimization`/`ema` keys.
2. EAMv3 latitude layout (Gaussian vs cell-centred 1°) and the `ak/bk` values: shape-only in the json (`JSON:410-424`). Decides whether the LG SHT is exact or misregistered by up to ~0.25°.
3. torch_harmonics node and quadrature conventions for `grid="equiangular"` were not inspected (package outside the allowed paths); §8 item 10 and the 0.5° polar misregistration of our outer transform (`MAIN/utils/grid_types.py:55-56`; `POL/convert_e3sm_to_makani_alldata.py:173`) rest on makani's own `linspace` assumption.
4. Was the EAMv3 checkpoint produced by an fme whose builder already had `data_grid`? If not, the modulus fallback was `"equiangular"` (`MOD/sfnonet.py:463`). Affects only the ACE2 column of item 10.
5. Which selection produced the published EAMv3 file (best validation vs best inference): both running bests are in the json, the filename is not.
6. ERA5 `scaling-residual.nc` / `scaling-full-field.nc` values live outside the audited directories, so ERA5 effective weights are UNDETERMINED numerically; its production epochs and batch are also UNDETERMINED (`max_epochs 3`, batch 16 are this Polaris port's values).
7. Whether ACE2's loss stds are exactly 6 h-change stds or smoothed/clipped (PS 247 Pa vs our probe δ 232 Pa is consistent; `specific_total_water_0` at 0.16 of the network std less obviously so).
8. `PROBE` δ comes from 4×60 samples of one file (`polaris_makani_ace2_ports_handoff.md:188-190`); the ordering is robust, the shares in §4.4 are not production numbers until `time_diff_stds.npy` is built over the full train split.
9. Fraction of cells with `OCNFRAC ≥ 0.5` (the ocean-overwrite area) needs the forcing data.
10. Units of `surface_precipitation_rate` vs PRECT: inferred from magnitudes only.
11. Whether ai2 SHT-round-tripped the EAMv3 data as for EAMv2 (PAPER only).
12. The exact dtype our drivers feed back under bf16 autocast: read `feedback_dtype` from an existing run's NetCDF rather than infer.
13. Should `climate_driver.stream_rollout` get the same opt-in `force_positive_names` hook as `rollout_one_ic` before any long-rollout clamp screen is trusted?
14. For G, which prognostics are non-negative by physics (RELHUM levels are the only fed-back candidates); whether FSNT/FSNTOA (prognostic in G, diagnostic in ACE2) and TREFHT (fully scored and fed back, where ACE2's TS is prescribed over ocean) need their own treatment. Science calls for jesswan.
15. Is `DryAirFix` approved, and does its "TMQ held fixed" variant (vs fme's "q held fixed") need sign-off?
16. Consumer of MAIN's `hydrostatic_balance_means` model kwarg (`MAIN/models/model_registry.py:220-225`) not identified; hydrostatic projection on A needs `hyam/hybm/P0` per column, and which pack build would carry them is open.
17. fme's precipitation rescale divides by `⟨P⟩` with no guard; whether ACE2 ever hits `⟨P⟩ ≈ 0` is not determinable from code.
18. Whether feeding `solin` at t+1 changes our one-step bias is testable in the dataset alone and has not been measured.

AUDIT_OK
