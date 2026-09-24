# ACE on E3SM vs makani on E3SM — channel contract, loss scaling, and what they imply

*Written 2026-09-24.* This note compares how three emulators of the E3SM atmosphere treat
their variables and score their training loss:

- **ACE v1 on EAMv2**, as published by Duncan et al. (2024) [1];
- **ACE2 on EAMv3**, as read directly from ai2's released checkpoint [2];
- **our makani SFNO on EAMv3**, as configured in this repository [3].

It extends `ace2_vs_makani_differences.md` [4], which compared makani against ACE2 trained
on **ERA5**. The comparison here holds the source model fixed (E3SM's atmosphere, EAM), so
differences in how variables are handled can no longer be attributed to the dataset.

**Scope.** The science axis only: which variables are fed back, which are output-only, how
the loss is normalized, and what constraints run at each step. Architecture and throughput
are covered by `ACE2_retrain/polaris/ace2_polaris_results.md` Table 8. Nothing here is a
proposal to change what our model computes without the science owner's sign-off
(CLAUDE.md, division of labor).

---

## 1. Summary

ai2 has built E3SM emulators in two model generations, and both use the same variable
contract: **34 prognostic channels** (surface pressure, surface skin temperature, and
temperature, total water and horizontal wind on 8 layers), with **10 fluxes and
precipitation kept as output-only diagnostics**. Our makani model feeds back **100
channels** and keeps only precipitation out of the loop.

Three findings follow from placing the three side by side:

1. **The difference in feedback-loop size is a modelling choice, not a dataset effect.**
   ai2 made the same 34/10 split on EAMv2 and on EAMv3. About half of our fed-back channels
   (geopotential, relative humidity, sea-level pressure, column water, near-surface
   fields, two radiative fluxes and two soil variables) have no counterpart in either ACE
   model.
2. **ACE v1 was stable for a 10-year rollout without a corrector, per-channel loss
   weights, or multi-step training** [1]. Our production model also trains on a single
   step and also lacks a corrector, yet diverges near step 500. Those two features are
   therefore unlikely to be the primary cause of our divergence.
3. **ACE2-EAMv3 scores prognostic errors against 6-hour tendency variability, not
   full-field variability** [2]. For surface pressure this makes a given error roughly
   1,400 times more costly in their loss than in ours. This matches the mechanism already
   observed in our `Z3_l17` channel, and supports the tendency-normalization port as the
   cheapest next experiment.

---

## 2. Terms

In an autoregressive emulator, each step's output becomes the next step's input. Every
variable plays one of three roles:

| role | predicted | in the loss | fed back as next input |
|---|---|---|---|
| **prognostic** | yes | yes | yes |
| **diagnostic** | yes | yes | no |
| **forcing** | no (read from file) | no | supplied fresh each step |

The distinction matters at rollout time. Rollout error evolves roughly as
`e_{k+1} ≈ J·e_k + ε`: a prognostic channel's error becomes part of the next input and can
compound, while a diagnostic channel's error is scored and then discarded. In our makani
code the split is implemented by `PlasimPreprocessor.append_history`, which slices the
diagnostic tail off the prediction before feedback
(`makani_sfno/src/sfno_training/models/preprocessor.py:20`); which channels fall on each
side is set by the converter (`makani_sfno/polaris/convert_e3sm_to_makani_alldata.py:149`).

---

## 3. Channel contracts

### 3.1 Counts

| | ACE v1 / EAMv2 [1] | ACE2 / EAMv3 [2] | makani / EAMv3 [3] |
|---|---|---|---|
| inputs | 40 | 39 | 107 |
| outputs | 44 | 44 | 101 |
| **prognostic** | **34** | **34** | **100** |
| forcing | 6 | 5 | 7 |
| **diagnostic** | **10** | **10** | **1** |
| vertical structure | 8 layers, mass-weighted from 72 levels | 8 layers (9 `ak`/`bk` interfaces) | 18 terrain-following levels |

### 3.2 Prognostic (fed back)

| quantity | ACE v1 / EAMv2 | ACE2 / EAMv3 | makani / EAMv3 |
|---|---|---|---|
| surface pressure | `p_s` | `PS` | `PS` |
| surface skin temperature | `T_s` (land and sea ice) | `TS` (overwritten over ocean, §4) | — |
| air temperature | `T_k` ×8 | `T_0..7` | `T_l00..17` |
| total water | `q^T_k` ×8 | `specific_total_water_0..7` | — (see `RELHUM`) |
| eastward / northward wind | `U_k`, `V_k` ×8 | `U_0..7`, `V_0..7` | `U_l00..17`, `V_l00..17` |
| geopotential | — | — | `Z3_l00..17` |
| relative humidity | — | — | `RELHUM_l00..17` |
| near-surface | — | — | `TREFHT`, `U10`, `RHREFHT` |
| column / sea-level | — | — | `TMQ`, `PSL` |
| radiative flux | diagnostic | diagnostic | **`FSNT`, `FSNTOA`** |
| land surface | — | — | **`SOILWATER_10CM`, `TSOI_10CM`** |

Sources: Table S2 [1, SI p.11]; `in_names`/`out_names` in the checkpoint config [2];
`channel_names` in `e3sm_alldata_full.yaml:63` [3].

### 3.3 Diagnostic (output only)

| quantity | ACE v1 / EAMv2 | ACE2 / EAMv3 | makani / EAMv3 |
|---|---|---|---|
| precipitation | `P` | `surface_precipitation_rate` | `PRECT` |
| latent / sensible heat | `LHF`, `SHF` | `LHFLX`, `SHFLX` | — |
| TOA longwave / shortwave up | `OLR`, `RSW` | `FLUT`, `top_of_atmos_upward_shortwave_flux` | — (`FSNTOA` is prognostic) |
| surface longwave up / down | `ULW_sfc`, `DLW_sfc` | `surface_upward_longwave_flux`, `FLDS` | — |
| surface shortwave up / down | `USW_sfc`, `DSW_sfc` | `surface_upward_shortwave_flux`, `FSDS` | — (`FSNT` is prognostic) |
| advective water tendency | `∂TWP/∂t|adv` | `tendency_of_total_water_path_due_to_advection` | — |

Table S2 in [1] labels every diagnostic as a 6-hour **mean** and every prognostic as a
**snapshot**. That is the physical basis of the split: an interval-averaged flux describes
what happened during the step, not a state the next step starts from. ai2's EAMv3
processing config draws the same line, reading state from E3SM's `6hourly_instant` stream
and fluxes from its `6hourly` (averaged) stream [5].

### 3.4 Forcing

| quantity | ACE v1 / EAMv2 | ACE2 / EAMv3 | makani / EAMv3 |
|---|---|---|---|
| insolation | yes | `SOLIN` | `solin` |
| sea surface temperature | explicit input | via `TS` overwrite (§4) | `sst` |
| surface geopotential | yes | `PHIS` | `topo` |
| land / ocean / sea-ice fraction | yes / yes / yes | `LANDFRAC` / `OCNFRAC` / `ICEFRAC` | `lsm` / — / `ice` |
| other | — | — | `glacier`, `natveg` |
| CO2 | no | **no** | no |

ACE2-EAMv3 has no CO2 input despite being trained on historical forcing (1970–2020).

---

## 4. Training objective and per-step constraints

| | ACE v1 / EAMv2 [1] | ACE2 / EAMv3 [2] | makani / EAMv3 [3] |
|---|---|---|---|
| loss | relative L2, one step (main p.3) | MSE | L2, squared |
| per-channel weights | none described | 14 explicit, 0.25–5 | `constant` (`:112`) |
| **loss normalization** | not described | **prognostics by 6 h tendency σ; diagnostics by full-field σ** | full-field z-score for all (`:76`, `:113`) |
| network target | full state | full state (`residual_prediction: False`) | tendency (`target: "tendency"`, `:158`) |
| training depth | 1 step | not stored in the checkpoint | 1 step in production (`n_future: 0`, `:127`) |
| corrector | none described | dry-air conservation; moisture budget (`advection_and_precipitation`); non-negativity on 15 variables | none |
| ocean surface | SST as a forcing input | `TS` overwritten with truth where `OCNFRAC` marks ocean | `sst` as a forcing input |
| EMA | yes (SI Table S4) | present in the checkpoint | none in production |
| optimizer | Adam, lr 3e-4, cosine, 50 epochs, batch 8 | 50 epochs (`epoch = 49`, 228,046 batches) | AdamW, lr 2e-3, 243 epochs, batch 32 |
| SFNO width | — | `embed_dim` 384, 8 layers, `scale_factor` 1 | `embed_dim` 384, 8 layers, `scale_factor` 3 (`:84`) |
| scenario | repeating 2005–2014 SST, fixed 2010 forcing | historical AMIP, 1970–2020 | SSP245-AMIP, 2015–2049 |
| reported stability | 10-year rollout, no drift reported | multi-year (ai2's claim; not measured by us) | diverges near step 500 [4] |

Line references in the makani column are to `makani_sfno/polaris/e3sm_alldata_full.yaml`.
In the ACE v1 column, "not described" means neither the paper nor its supplement states
it; the paper defers such details to Watt-Meyer et al. (2023) [6].

---

## 5. Findings

### 5.1 The feedback-loop size is our choice, not the data's

The earlier comparison [4] could not separate the effect of the dataset from the effect of
modelling choices, because ACE2 was trained on ERA5. With E3SM held fixed, the result is
unambiguous: ai2 chose the same 34-prognostic, 10-diagnostic contract for EAMv2 and for
EAMv3, and it is the physical state plus interval-averaged fluxes, as E3SM itself
distinguishes them.

Our 100 fed-back channels decompose as follows:

| group | channels | status in both ACE models |
|---|---|---|
| T, U, V on 18 levels | 54 | prognostic, on 8 layers |
| `PS` | 1 | prognostic |
| `Z3` on 18 levels | 18 | absent |
| `RELHUM` on 18 levels | 18 | absent; total water is prognostic instead |
| `TREFHT`, `U10`, `RHREFHT`, `PSL`, `TMQ` | 5 | absent |
| `FSNT`, `FSNTOA` | 2 | **diagnostic** |
| `SOILWATER_10CM`, `TSOI_10CM` | 2 | absent |

Forty-three of our fed-back channels have no counterpart in ACE, and two more are ones ACE
deliberately keeps out of the loop. Our measured failure modes sit in exactly these
groups: `SOILWATER_10CM` drifts to negative values and is on a trajectory that reaches
the divergence step [4, §1]; `Z3_l17` produces a 1.27 m anomaly in one step against
0.271 m of natural temporal variability [4, §1].

### 5.2 Stability without the ACE2 machinery

Duncan et al. describe no corrector, no positivity enforcement, no per-channel weights and
no multi-step loss for ACE v1 [1, main p.3; SI Table S4]. The model is trained on a single
6-hour step and evaluated as one continuous 10-year rollout with no reported drift
[1, main pp.4–6].

Our production model shares the single-step training and the absence of a corrector. The
largest remaining differences are the size of the feedback loop (§5.1) and the loss
normalization (§5.3). This does not prove either one is the cause: ACE v1 was forced with
a repeating SST climatology, an easier target than our warming scenario, and several of
its implementation details are not stated in the paper. But it lowers the expected payoff
of adding multi-step training or a corrector alone, relative to the two changes below.

### 5.3 ACE2-EAMv3 normalizes the loss by tendency variability

The checkpoint carries two normalization sets, `network` (applied to inputs and outputs)
and `loss` (applied when scoring). For diagnostic channels the two are identical. For
prognostic channels the loss uses a much smaller standard deviation, consistent with the
spread of 6-hour changes rather than of the full field:

| channel | network σ | loss σ | ratio | relative weight of an equal error in their loss vs ours |
|---|---|---|---|---|
| `PS` | 9,380 Pa | 247 Pa | 0.026 | ~1,400× |
| `TS` | 22.7 K | 3.92 K | 0.173 | ~33× |
| `T_0` | 8.62 K | 0.499 K | 0.058 | ~300× |
| `T_7` | 17.8 K | 1.28 K | 0.072 | ~190× |
| `specific_total_water_7` | 5.35e-3 | 5.53e-4 | 0.103 | ~94× |
| `U_4` | 11.9 m/s | 3.12 m/s | 0.262 | ~15× |
| `V_7` | 6.42 m/s | 3.26 m/s | 0.507 | ~4× |
| `FLUT` (diagnostic) | 48.3 | 48.3 | 1.000 | 1× |

The last column is `(network σ / loss σ)²`, the factor by which a squared error is
up-weighted. Our loss uses the full-field σ for every channel
(`temp_diff_normalization: False`), so a 6-hour error in surface pressure is measured
against a σ dominated by topography rather than by weather. The `Z3_l17` result in [4]
is the same effect: a 43-fold inflation of that channel's temporal variance left no trace
in our training loss.

This corrects one statement in [4]. It describes ACE2 as using "dual normalization
(full-field + residual)". In the EAMv3 checkpoint `normalization.residual` is `None` and
`residual_prediction` is `False`; the network predicts the full state, and the tendency
scaling enters through the separate `loss` normalization. The practical effect on
training is the same.

Two caveats. First, we predict a tendency and ACE2 predicts the full state, so porting
the idea means dividing our tendency error by tendency σ, not copying ACE2's config.
Second, the σ values above come from ai2's EAMv3 statistics, not ours; our own values must
be computed from our pack.

---

## 6. Implications for the port plan

| lever | evidence from this note | cost | decision owner |
|---|---|---|---|
| **tendency-normalized loss** (`temp_diff_normalization: True`, build `time_diff_stds.npy`) | §5.3: used by ACE2 on the same source model; explains the `Z3_l17` blind spot | one converter pass plus one training arm | us: changes how training is scored, not what the model outputs |
| **Port F**: drop `SOILWATER_10CM`, `TSOI_10CM` | §5.1: absent from both ACE models | retrain | decided 2026-09-23 |
| **Port E**: reduce the feedback loop (`Z3`, `RELHUM`, `FSNT`/`FSNTOA`, near-surface fields to diagnostic or removed) | §5.1: both ACE generations use the 34/10 contract | converter change plus retrain | **science owner** |
| corrector / non-negativity clamp | ACE2 only; ACE v1 stable without it (§5.2) | inference-only clamp exists as a diagnostic | us, for the diagnostic arm |
| multi-step training | ACE v1 stable with one step (§5.2) | training arm | us; lower priority than the rows above |

The recommended order is: tendency-normalized loss first, because it is cheap, within our
remit, and directly supported by the same-model evidence; then bring §5.1 to the science
owner together with the `Z3_l17` question already queued for her [7, §6].

---

## 7. Caveats and open questions

- **Training depth of ACE2-EAMv3 is unknown.** The checkpoint stores the stepper config but
  not the training config, so `n_forward_steps` is not recoverable from it.
- **ACE2-EAMv3 stability is ai2's claim.** We have not rolled out their checkpoint. Doing
  so on our hardware would turn the claim into a measurement and give a same-data
  reference for our divergence step.
- **ACE v1 details are partly unstated.** Normalization, loss weighting and treatment of
  `T_s` over open ocean are not given in [1]; the paper refers to [6].
- **E3SM variable names for ACE v1** are not printed in [1]. The mapping in §3 uses the
  ACE2-EAMv3 names from [2], which are authoritative for that model only.
- **Different forcing regimes.** ACE v1 used a repeating climatology; ACE2-EAMv3 used
  historical AMIP; we use a warming scenario with train and test at different points of
  it [4, §4]. Stability comparisons across the three are indicative, not controlled.
- **Unit question in [1].** Table S2 gives total water in g/kg; the readout flags that
  Figure 1 may use kg/kg. Not material to this note.

---

## 8. Reproducing this note

| artefact | location | produced by |
|---|---|---|
| ACE2-EAMv3 checkpoint (1.82 GB, not in git) | `/eagle/projects/lighthouse-uchicago/members/mehta5/ace2_eamv3/ace2_EAMv3_ckpt.tar` | job 7649301 |
| stepper config, weights stripped | `ACE2_retrain/polaris/ace2_eamv3_stepper_config.json` (copy of the file beside the checkpoint) | `ACE2_retrain/polaris/ace2_eamv3_read_config.py` |
| job script | `ACE2_retrain/polaris/polaris_ace2_eamv3_config.pbs` | PASS = `ACE2_EAMV3_CONFIG_OK` |
| Duncan et al. paper and supplement (not in git) | `/eagle/projects/lighthouse-uchicago/members/mehta5/papers/` | manual download |
| full-text readout of [1] | `/eagle/projects/lighthouse-uchicago/members/mehta5/papers/duncan2024_readout.md` | job 7649397, `ACE2_retrain/polaris/polaris_read_duncan2024.pbs`, PASS = `DUNCAN_READOUT_OK` |

The σ table in §5.3 is read from `stepper.config.step.config.wrapped_step.config.normalization`
in the stepper config JSON.

---

## References

1. Duncan, J. P. C., Wu, E., Golaz, J.-C., et al. (2024). Application of the AI2 Climate
   Emulator to E3SMv2's global atmosphere model, with a focus on precipitation fidelity.
   *JGR: Machine Learning and Computation*, 1, e2024JH000136.
   https://doi.org/10.1029/2024JH000136 — main text and Supporting Information (Tables
   S1–S5).
2. Allen Institute for AI. ACE2-EAMv3 model checkpoint, `ace2_EAMv3_ckpt.tar`.
   https://huggingface.co/allenai/ACE2-EAMv3 — config read by job 7649301.
3. `makani_sfno/polaris/e3sm_alldata_full.yaml` and
   `makani_sfno/polaris/convert_e3sm_to_makani_alldata.py`, this repository.
4. `ace2_vs_makani_differences.md`, this repository (2026-09-23).
5. `ACE2_retrain/ace_exp/scripts/data_process/configs/e3sm-1deg-8layer-v3-AMIP.yaml`,
   ai2's EAMv3 data-processing config, vendored in this repository.
6. Watt-Meyer, O., Dresdner, G., McGibbon, J., et al. (2023). ACE: A fast, skillful
   learned global atmospheric model for climate prediction. arXiv:2310.02074.
7. `polaris_makani_ace2_ports_handoff.md`, this repository.
