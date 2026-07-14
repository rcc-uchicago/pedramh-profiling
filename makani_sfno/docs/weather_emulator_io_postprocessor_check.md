# Weather Emulator IO and PlaSim Postprocessor Coverage

## Scope

This note summarizes the current weather emulator input/output variable contract and checks whether the current post-processing code in `src/plasim_postprocessor` can produce all required emulator inputs.

The emulator configs inspected were:

- `/work2/09979/awikner/stampede3/PanguWeather/v2.0/config/SFNO_PLASIM_H5_DERECHO_5410.yaml`
- `/work2/09979/awikner/stampede3/PanguWeather/v2.0/config/SFNO_PLASIM_H5_DERECHO_5411.yaml`
- `/work2/09979/awikner/stampede3/PanguWeather/v2.0/config/SFNO_PLASIM_H5_DERECHO_5412.yaml`

The post-processing code inspected was:

- `src/plasim_postprocessor/plasim_postprocessor.py`
- `src/plasim_postprocessor/burn7/burn7.cpp`

## Emulator inputs

The current SFNO emulator family uses the following inputs.

### Dynamic state inputs

| Group | Variables | Notes |
|---|---|---|
| Upper air | `ta`, `ua`, `va`, `hus`, `zg` | Configured as `upper_air_variables` |
| Surface | `pl`, `tas` | Configured as `surface_variables` |
| Constant boundary | `lsm`, `sg`, `z0` | Configured as `constant_boundary_variables` |
| Varying boundary | `sst`, `rsdt`, `sic` | Configured as `varying_boundary_variables` |

### Tensor interface used by the model

Based on the readable test harness, the model call is:

`model(surface, constant_boundary, varying_boundary, upper_air)`

with the following shapes:

- Surface input: `[B, 2, 64, 128]` for `pl, tas`
- Constant boundary input: `[B, 3, 64, 128]` for `lsm, sg, z0`
- Varying boundary input: `[B, 3, 64, 128]` for `sst, rsdt, sic`
- Upper-air input: `[B, 5, 10, 64, 128]` for `ta, ua, va, hus, zg`

Common config settings across the inspected runs:

- Horizontal resolution: `64 x 128`
- Timestep: `6` hours
- `num_levels: 10`
- `use_sigma_levels: True`

## Emulator outputs

The current emulator predicts:

| Group | Variables | Notes |
|---|---|---|
| Upper air output | `ta`, `ua`, `va`, `hus`, `zg` | Same variable family as upper-air inputs |
| Surface / diagnostic output | `pl`, `tas`, `pr_6h` | `pr_6h` is configured as a diagnostic variable |

The readable training/test code uses:

- `output_surface, output_upper_air = model(...)`
- Loss against `target_surface` and `target_upper_air`

The readable legacy network implementation shows that when `diagnostic_head` is `False`, diagnostic variables are folded into the 2D surface output head rather than emitted as a separate tensor. In practice, that means the 2D output head covers `surface_variables + diagnostic_variables`, i.e. `pl, tas, pr_6h`.

## Current postprocessor output contract

The current `src/plasim_postprocessor/plasim_postprocessor.py` emits:

- Sigma-level variables from `SIGMA_CODES`:
  - `ta`, `ua`, `va`, `hus`
  - `tas`, `td2m`
  - `ts`, `ps`, `psl`, `clt`
  - `mrso`
  - `rss`, `rls`, `rst`, `rlut`, `rsut`, `hfss`, `hfls`
  - `lsm`, `z0`, `sg`
  - `pr`
- Pressure-level `zg` from a separate burn7 namelist at:
  - `50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000` hPa
- Derived `pr_6h` from `cdo runsum,6`
- Optional `sic` only when `--with-sea-ice` is enabled

Important consequences:

1. `pl` is not currently emitted by `plasim_postprocessor.py`, even though burn7 itself supports `pl`.
2. `zg` is currently emitted on pressure levels, not as a sigma-level field.
3. `sst` and `rsdt` are not emitted by the current postprocessor, and no burn7 code with those names exists in `burn7.cpp`.

## Coverage check against emulator inputs

| Emulator input variable | Current postprocessor status | Evidence | Verdict |
|---|---|---|---|
| `ta` | In `SIGMA_CODES` | Current script requests it directly | Exact support |
| `ua` | In `SIGMA_CODES` | Current script requests it directly | Exact support |
| `va` | In `SIGMA_CODES` | Current script requests it directly | Exact support |
| `hus` | In `SIGMA_CODES` | Current script requests it directly | Exact support |
| `zg` | Produced only via separate pressure-level namelist | Current script writes `code=zg` with fixed pressure levels | Partial / mismatched |
| `pl` | Supported by burn7, but not requested by current script | burn7 has `pl`; current script does not include it in `SIGMA_CODES` | Missing in current postprocessor output |
| `tas` | In `SIGMA_CODES` | Current script requests it directly | Exact support |
| `lsm` | In `SIGMA_CODES` | Current script requests it directly | Exact support |
| `sg` | In `SIGMA_CODES` | Current script requests it directly | Exact support |
| `z0` | In `SIGMA_CODES` | Current script requests it directly | Exact support |
| `sst` | Not present in current script or burn7 variable catalog | No `sst` burn7 variable found | Missing |
| `rsdt` | Not present in current script or burn7 variable catalog | No `rsdt` burn7 variable found | Missing |
| `sic` | Optional only | Current script appends it only under `--with-sea-ice` | Conditional support |

## Coverage check against emulator outputs

| Emulator output variable | Current postprocessor status | Verdict |
|---|---|---|
| `ta`, `ua`, `va`, `hus` | Available as sigma-level fields | Exact support |
| `zg` | Available only as pressure-level field | Partial / mismatched |
| `pl` | Not currently emitted by the script | Missing in current output |
| `tas` | Available | Exact support |
| `pr_6h` | Derived with CDO `runsum,6` | Exact support |

## Overall verdict

The current post-processing code in `src/plasim_postprocessor` does **not** handle all current emulator input variables.

### Fully covered

- `ta`
- `ua`
- `va`
- `hus`
- `tas`
- `lsm`
- `sg`
- `z0`

### Conditionally covered

- `sic`
  - Only if `--with-sea-ice` is enabled
  - Still depends on the source PlaSim simulation actually containing sea-ice data

### Available but not in the required emulator form

- `zg`
  - Current postprocessor emits pressure-level `zg`
  - Emulator input path expects `zg` in the same 10-level upper-air tensor family used by the other upper-air inputs

### Not currently handled

- `pl`
  - burn7 supports it as `log_surface_pressure`
  - current `plasim_postprocessor.py` does not request it
- `sst`
  - not found in the current postprocessor or burn7 variable catalog
- `rsdt`
  - not found in the current postprocessor or burn7 variable catalog

## Practical conclusion

If this postprocessor is used as-is to feed the current emulator contract, the blocking gaps are:

- `pl` missing from the emitted files
- `sst` missing
- `rsdt` missing
- `zg` emitted on the wrong vertical coordinate family for the current emulator tensor contract

`pr_6h` is available for the surface output target, and `sic` is only available behind the sea-ice flag.
