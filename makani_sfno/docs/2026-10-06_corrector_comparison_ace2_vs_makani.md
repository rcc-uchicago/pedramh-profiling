# ACE2 corrector vs the makani fixes: side-by-side check (2026-10-06)

Read from the code, not from earlier docs. The 2026-10-01 audit
(`docs/2026-10-01_makani_vs_ace2_code_audit.md` rows 352–357, 376, 437) was cross-checked against this
and agrees.

Paths:
- `FME` = `ACE2_retrain/ace_exp/fme` (main checkout). `ace_exp_updated` has a byte-identical
  `core/corrector/atmosphere.py` and `utils.py`.
- `MK` = `makani_sfno` on `feat/makani-b-continuation-dryair`.
- `STEPPER` = installed makani `models/stepper.py` in `sfno-venv`.

Legend: ✅ same · ⚠ differs by design (stated) · ❌ not ported · — not applicable.

## 1. What each side runs

| corrector step (ACE2 order) | ACE2-EAMv3 shipped checkpoint | our ACE2 retrain | makani |
|---|---|---|---|
| 1 `force_positive_names` | on: 15 names (8 water levels, precipitation, 6 radiative fluxes) | on: 16 names (8 water levels, Q2m, precipitation, 6 radiative fluxes) | inference-only, opt-in: `--force-positive=relhum` (RELHUM_l00..l17) |
| 2 `conserve_dry_air` | on | on | opt-in: trained in (DRYAIR arm) or inference-only (`--dry-air-fix=on`) |
| 3 `zero_global_mean_moisture_advection` | off | off (default) | — |
| 4 `moisture_budget_correction` | `advection_and_precipitation` | same | ❌ not portable: no q levels, no advection channel |
| 5 `total_energy_budget_correction` | off (`null`) | off | — |
| soil moisture | no corrector in the vendored fme | — | opt-in `SoilMoistureFix` (SOILFIX arm) |

Sources:
- ACE2-EAMv3: `members/mehta5/ace2_eamv3/ace2_eamv3_stepper_config.json` (`stepper/.../corrector`).
- Our ACE2 retrain: `ACE2_retrain/config_polaris.yaml:178-195`.
- makani: `MK/scripts/climate_rollout.py:42,71-78,120-128`.

## 2. Dry-air conservation, element by element

| element | ACE2 (`FME`) | makani `DryAirFix` (`MK/src/sfno_training/models/mass_fix.py`) | status |
|---|---|---|---|
| conserved quantity | `ps − Σ_k(Δa_k + Δb_k·ps)·q_k` = `ps − g·TWP` (`corrector/atmosphere.py:243-274`, `metrics.py:283-296`, `coordinates.py:260-287`) | `PS − g·TMQ` (`:101-108`) | ✅ same form |
| what counts as water | `specific_total_water_0..7`: vapour + condensate, 8 layers | `TMQ`: column water **vapour** only. There is no condensate in the pack. | ⚠ the dry air includes cloud condensate mass. That is a definition, not a bug. |
| reference state | global mean of the step's **input** (`atmosphere.py:281-285`) | global mean of the step's input, cached before the model (`preprocessor.py:103-106`, `mass_fix.py:113`) | ✅ |
| global mean | cos-lat area weights normalised over the grid (`metrics.py:14-32`, `gridded_ops.py:281-353`) | cos-lat at cell-centred equiangular latitudes, normalised (`:40-48`) | ✅ **identical on our grid.** The pack's 180 latitudes are 89.5..−89.5, so the max weight difference is 0.0 (checked 2026-10-06 against `climate_fidelity_7709968/true_climatology_b.npz` `lat`). |
| gravity | 9.80665 (`constants.py:3`) | 9.80665 (`:37`) | ✅ |
| how the error is removed | subtract the global error from each column's dry air, then solve `ps` with `q_k` held fixed: `ps = (dry − err + ΣΔa·q)/(1 − ΣΔb·q)` (`atmosphere.py:285-296`). So ΔPS = −err/(1 − ΣΔb·q), slightly non-uniform. | uniform ΔPS = −err with `TMQ` held fixed, applied as a z-shift `err/σ_PS` (`:110-123`) | ⚠ by design (docstring `:19-24`). **ΔPS_ACE·(1 − ΣΔb·q) = ΔPS_makani exactly.** The difference is ≈ g·TWP/ps ≈ 0.3 % of the correction (global mean), up to ~0.7 % in the moistest columns. |
| water used in the solve | **after** `force_positive` clamps the 8 water levels (`atmosphere.py:181-198`) | raw predicted `TMQ`, unclamped. TMQ is never clamped in makani. | ⚠ our 5-yr runs show local negative TMQ: monthly cell minima −5.5..−33 kg/m² (B24+fix the most negative), ≤ 1 % of the area in the last month (`dryair_budget_readout`, 7719173/7720728/7721211). It enters the dry-air mean as predicted. |
| precision | float64 (float32 on MPS), cast back to the input dtype (`atmosphere.py:160-163,296`) | float64, output promoted to ≥ float32 (`:114-122`). bf16 output had silently undone the fix (7650512); fixed in `6c29bf18`. | ✅ |
| where it runs | denormalised output, after every step, in training **and** inference (`core/step/single_module.py:443-449`) | z-space, after `history_denormalize(target=True)` on every predicted step, including each unrolled multistep step. The corrected prediction is fed back (`STEPPER:34-50, 82-112, 125-143`; `preprocessor.py:96-121`). | ✅. Two conditions: `history_normalization_mode` must be `none`/`timediff` (KNOWLEDGE §8b), and `h/w_parallel_size = 1` is enforced (`preprocessor.py:36-39`). ACE2 handles spatial parallel. |
| on in training? | yes, whenever configured | only in the DRYAIR arm (`conserve_dry_air: True`). `--dry-air-fix=on` is inference-only (`climate_rollout.py:120-123`). | ⚠ by choice |

## 3. Positivity clamp

| element | ACE2 (`FME/core/corrector/utils.py:6-11`) | makani (`MK/src/sfno_inference/rollout_driver.py:123-150`) | status |
|---|---|---|---|
| operation | `clamp(x, min=0)` in physical units | `max(z, (0 − μ)/σ)` in z-space, i.e. physical 0 | ✅ same math. If the prediction is bf16, the floor is cast to bf16, so it is exact only to one bf16 step. |
| which fields | 15 (EAMv3) / 16 (retrain) | RELHUM_l00..l17 (`climate_rollout.py:42`). ACE2's names mostly don't exist in the makani pack. | ⚠ different fields |
| order | **first**, before dry air (`atmosphere.py:181-185`) | after the model wrapper, i.e. **after** `DryAirFix` (`climate_driver.py:376-377`) | ⚠ harmless today: RELHUM does not enter PS − g·TMQ. It would matter if TMQ were clamped. |
| training | yes | no (inference only) | ⚠ |

## 4. Soil moisture (makani only)

`MK/src/sfno_training/models/soil_moisture_fix.py:92-101`. Where the input `TSOI_10CM < 272 K`, the
predicted `SOILWATER_10CM` is replaced with the input value, then clamped ≥ 0. **No reference
implementation exists:** the vendored fme has no soil corrector, and its strict config loader would
refuse the `soil_moisture_corrector` key in the lsm experiment yaml (docstring `:9-23`). It is a
reimplementation of the stated intent, so it can be checked for internal consistency only.

## 5. Evidence that each fix does what it claims

| fix | evidence | result |
|---|---|---|
| makani `DryAirFix` (trained in) | 5-yr budget readout, DRYAIR24 (7711720 → 7719173) | max one-step \|ΔDRY\| 0.000 hPa, 5-yr dDRY 0.00, 8/8 |
| makani `DryAirFix` (inference only) | B22/B24 + fix (7721150 → 7721211) | max one-step \|ΔDRY\| 0.000 hPa, dDRY 0.00, 16/16 |
| makani vs ACE2, same synthetic state | `polaris/test_dryair_vs_fme.py` (`polaris_dryair_vs_fme.pbs`, fme-venv) | see §6 |
| makani RELHUM clamp | B22posclamp (7707597) | not checked for negatives after the clamp |
| makani `SoilMoistureFix` | SOILFIX screen 7719183 | not checked numerically against its rule |
| ACE2 corrector | `FME/core/corrector/test_atmosphere.py` | exists, not run by us |

## 6. Numerical cross-check (makani `DryAirFix` vs fme `_force_conserve_dry_air`)

Pending: job id and result are filled in when it runs.
