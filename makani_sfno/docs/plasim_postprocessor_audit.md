# PlaSim Post-processor Audit

Records the empirical output of the existing `src/plasim_postprocessor/plasim_postprocessor.py` for each of the three experiment profiles (EXP15/25/26) on a sample input. The new script (Step 2 of the refactor) must reproduce these outputs exactly. Locked snapshots live alongside this doc at `docs/audit_snapshots/{exp15,exp25,exp26}_manifest.txt`.

## Method

- **Sample input**: `/scratch/10000/amarchakitus/PLASIM/data/sim30/MOST.0012` (smallest year present in sim30).
- **Toolchain**: existing `plasim_postprocessor.py` v3.0 driver, calling `burn7_wrappers/stampede3.sh` (which runs the prebuilt burn7 at `burn7/Stampede3/burn7`) plus `cdo` for merging and precipitation accumulation.
- **Modules loaded**: `intel/24.0 cdo netcdf python/3.12.11`. `LD_LIBRARY_PATH` extended with `/opt/apps/intel24/netcdf/4.9.2/x86_64/lib` and `/home1/09979/awikner/netcdf-4.2/lib` (for `libnetcdf_c++.so.4` that burn7 links against).
- **Pangu disabled** for EXP25/26 (out of scope; grid file isn't on Stampede3).
- **NCL not validated** — the legacy NCL scripts (`compute_zg.ncl`, `interpolate_data.ncl`) were never copied into this checkout. Per `README.md:233` they introduce a documented −20 to −30 m bias relative to burn7 VTYPE=P, so `zg_source: "burn7"` is the correct path. NCL refs are deleted as dead code in Step 4.
- **Manifest format**: `dimensions:` block, sorted variable declarations, `lev` (sigma) coordinate values, `lev_2` (pressure) coordinate values, lat/lon dim sizes. Time values are excluded since they're calendar-dependent and not part of the variable contract.

## Predicted vs empirical (reconciliation)

For each profile, "predicted" comes from a static trace of `_sigma_variables()`, `_compute_z500_burn7()`, and `accumulate_precipitation()`; "empirical" comes from the audit run.

### EXP15

| Field | Predicted | Empirical | Status |
|---|---|---|---|
| Sigma namelist `code=` | `ta,ua,va,hus,pl,tas` | `ta,ua,va,hus,pl,tas` | ✓ |
| zg pressure levels | `[500]` | `lev_2 = [500]` | ✓ |
| `pr_6h` present | no | no | ✓ |
| `mrso` present | no | no | ✓ |
| Sigma `lev` size | 10 (MODLEV 10..0 minus the 11th non-sigma surface entry) | 10 | ✓ |
| Time steps | ~1460 (4×365 native; no precip-accum trimming) | 1464 | ✓ (no surprises — PlaSim calendar) |

### EXP25

| Field | Predicted | Empirical | Status |
|---|---|---|---|
| Sigma namelist `code=` | `ta,ua,va,hus,pl,tas,pr` (pr added by `_sigma_variables` due to `accumulate_precip=True`) | `ta,ua,va,hus,pl,tas,pr` | ✓ |
| zg pressure levels | `[50,100,150,200,250,300,400,500,600,700,850,925,1000]` | `lev_2 = [50,100,…,1000]` (13) | ✓ |
| `pr` in output | yes (instantaneous, from burn7) | yes | ✓ |
| `pr_6h` in output | yes (CDO `runsum,6` then `chname,pr,pr_6h`) | yes | ✓ |
| `mrso` present | no | no | ✓ |
| Time steps | 1464 native − 5 dropped by `runsum,6` window = 1459 | 1459 | ✓ |

### EXP26

| Field | Predicted | Empirical | Status |
|---|---|---|---|
| Sigma namelist `code=` | `ta,ua,va,hus,pl,tas,mrso,pr` | `ta,ua,va,hus,pl,tas,mrso,pr` | ✓ |
| zg pressure levels | same as EXP25 | `lev_2 = [50,100,…,1000]` (13) | ✓ |
| `mrso` in output | yes (from `land_vars=["mrso"]`) | yes | ✓ |
| `pr_6h` in output | yes | yes | ✓ |
| Time steps | 1459 (same trim as EXP25) | 1459 | ✓ |

## Locked variable sets

These are the variable lists every successful run for each profile must produce. Sourced from `docs/audit_snapshots/{profile}_manifest.txt`.

| Profile | Variables (alphabetical) |
|---|---|
| `exp15` | `hus, lat, lev, lev_2, lon, pl, ta, tas, time, ua, va, zg` |
| `exp25` | `hus, lat, lev, lev_2, lon, pl, pr, pr_6h, ta, tas, time, ua, va, zg` |
| `exp26` | `hus, lat, lev, lev_2, lon, mrso, pl, pr, pr_6h, ta, tas, time, ua, va, zg` |

Note that `pr` (instantaneous) is retained in EXP25/26 outputs alongside `pr_6h` — the existing `accumulate_precipitation()` does not drop the source variable. The new script must preserve this.

## Findings

1. **`lev` is sigma, `lev_2` is pressure.** burn7 emits two distinct vertical dimensions: `lev` (10 sigma levels, identical across profiles) for sigma-level fields, and `lev_2` (1 entry for EXP15, 13 entries for EXP25/26) for pressure-level fields (zg only). The new script's verification must compare both `lev` and `lev_2` values.
2. **Native cadence preserved.** EXP15 has 1464 timesteps for a 366-day year (T42 PlaSim default = 4 outputs/day). No daily aggregation occurs in any existing profile.
3. **`runsum,6` trims 5 timesteps.** EXP25/26 outputs have 1459 timesteps — 5 fewer than EXP15 — because `cdo runsum,6` requires a 6-step accumulation window. The new script must produce the same trimming.
4. **`pr` is retained alongside `pr_6h`.** `accumulate_precipitation()` adds `pr_6h` via merge but doesn't `selname` it back out. Both variables are in the EXP25/26 output.
5. **`pl` is single-level, time-varying.** Captured as `float pl(time, lat, lon)` in all three profiles. Confirms it's surface pressure, not multi-level.

## Module-load contract for the new script

For the SLURM template and any direct user invocation on Stampede3, the required module set is:

```bash
module purge
module load intel/24.0 cdo netcdf python/3.12.11
export LD_LIBRARY_PATH=/opt/apps/intel24/netcdf/4.9.2/x86_64/lib:/home1/09979/awikner/netcdf-4.2/lib:${LD_LIBRARY_PATH}
```

Notes:
- `intel/24.0` is required to load `cdo/2.3.0` (toolchain dependency discovered during audit).
- `python/3.12.11` is required for the existing `plasim_postprocessor.py` (uses Python 3.10+ union syntax `str | None`); the new script will keep the same minimum.
- The two `LD_LIBRARY_PATH` entries are needed by burn7's runtime linkage (`libnetcdf_c++.so.4` lives at `/home1/09979/awikner/netcdf-4.2/lib`; the system `netcdf/4.9.2` doesn't include the legacy C++ bindings).

This module set replaces the original `gcc netcdf` line in `burn7_wrappers/stampede3.sh:20` (which was incomplete — it didn't load `cdo` or `python`).

## Reproducing the audit

```bash
cd ~/projects/SFNO_Climate_Emulator
module purge
module load intel/24.0 cdo netcdf python/3.12.11
export LD_LIBRARY_PATH=/opt/apps/intel24/netcdf/4.9.2/x86_64/lib:/home1/09979/awikner/netcdf-4.2/lib:${LD_LIBRARY_PATH}

INPUT=/scratch/10000/amarchakitus/PLASIM/data/sim30/MOST.0012
mkdir -p /tmp/audit_runs

for prof in EXP15 EXP25 EXP26; do
    python3 src/plasim_postprocessor/plasim_postprocessor.py \
        --config src/plasim_postprocessor/config/${prof}_postproc.yaml \
        --input "$INPUT" \
        --plasim_output /tmp/audit_runs/${prof,,}.nc
done
```

Then run the `manifest()` helper from `docs/plasim_postprocessor_refactor_plan.md` against each `/tmp/audit_runs/*.nc` and `diff` against the corresponding `docs/audit_snapshots/*_manifest.txt`.

## Addendum: `aires_rad` profile (audited)

Added per `docs/aires_rad_profile_plan.md`. **Status: audited** — snapshot at `docs/audit_snapshots/aires_rad_manifest.txt` (locked from sim30/MOST.0012).

### Variable list

`aires_rad` is a strict superset of `exp26`, adding 7 surface radiation + heat-flux variables:

| Group | Variables | Producing tool |
|---|---|---|
| Sigma 11-level | ta, ua, va, hus | burn7 sigma namelist |
| Surface (single-level via sigma namelist) | pl, tas | burn7 sigma namelist |
| Land (surface) | mrso | burn7 sigma namelist |
| Radiation (surface) | rss (176, sfc net SW), rls (177, sfc net LW), rst (178, TOA net SW), rlut (179, TOA net LW), rsut (203, TOA outgoing SW) | burn7 sigma namelist |
| Heat fluxes (surface) | hfss (146, sfc sensible), hfls (147, sfc latent) | burn7 sigma namelist |
| Pressure (zg only) | zg @ 13 hPa levels | burn7 zg namelist |
| Precip accumulation | pr_6h (and pr retained) | CDO `runsum,6` + `chname` |

### Audit method

Departure from the EXP15/25/26 audit method: the legacy `plasim_postprocessor.py` v3.0 driver never emitted radiation, so there's no legacy pipeline to diff against. The new script's first verified run became the locked contract, but only after the semantic-validation step (below) confirmed each variable's sign convention and global mean lies in expected climate ranges. burn7's metadata (`burn7.cpp:5508-5547`) only stores label/units (CF-style names like `surface_net_shortwave_flux`); it does not encode sign conventions. PlaSim sets the sign — pinned down empirically below.

### B.0 probe outcome (sim30/MOST.0012)

- burn7 exit 0 with `code=rss,rls,rst,rlut,rsut,hfss,hfls`.
- All 7 variables present in the output NetCDF (`ncdump -h` count = 7).
- `cdo infon` per-variable: every variable has `Miss = 0` (no missing values) over 1464 timesteps × 8192 gridpoints; min ≠ max for all 7.

### B.1 semantic validation

Global temporal means (sim30/MOST.0012, `cdo fldmean -timmean`):

| Variable | burn7 label | Global mean (W m⁻²) | Min (W m⁻²) | Max (W m⁻²) | Expected magnitude | Verdict |
|---|---|---|---|---|---|---|
| `rss` | surface_net_shortwave_flux | **+176.46** | 0 | +1014 | 150–180 (positive: net into sfc) | ✓ in range |
| `rls` | surface_net_longwave_flux | **−60.49** | −395 | +106 | 40–60 magnitude (negative: sfc loses LW) | ✓ in range |
| `rst` | toa_net_shortwave_flux | **+242.49** | 0 | +1179 | 230–250 (positive: atm absorbs) | ✓ in range |
| `rlut` | toa_net_longwave_flux | **−234.95** | −429 | −55 | 230–250 magnitude | ✓ in range — **OLR = −rlut ≈ +235 W m⁻²** |
| `rsut` | toa_outgoing_shortwave_flux | **−99.31** | −888 | 0 | 90–110 magnitude | ✓ in range — outgoing SW magnitude ≈ +99 W m⁻² |
| `hfss` | surface_sensible_heat_flux | **−21.77** | −1191 | +507 | 15–25 magnitude (negative: sfc → atm) | ✓ in range |
| `hfls` | surface_latent_heat_flux | **−85.94** | −828 | 0 | 75–90 magnitude | ✓ in range |

**Sign convention discovered (PlaSim): positive = into receiver, negative = leaving receiver.** This is uniform across all 7 variables and is **opposite to CF conventions** for outgoing fluxes. Specifically:

- `rlut` is *negative* when LW is leaving the TOA (i.e. when there is OLR). `OLR = −rlut`. Despite burn7 labeling it `toa_net_longwave_flux` (CF would imply positive-down), PlaSim's sign matches the physical convention "positive = absorbed."
- `rsut` is named "outgoing" but PlaSim outputs it as *negative* (loss-from-atmosphere convention), not as a positive magnitude. To get the conventional outgoing-SW magnitude, take `−rsut`.
- `hfss` and `hfls` are negative because surface loses sensible/latent heat to the atmosphere; this matches the CF "downward = positive" convention but means typical climate values are negative.

All 7 variables are within ±30% of the expected magnitudes; no variable was rejected.

### B.2 / B.3 snapshot lock + round-trip

Snapshot file: `docs/audit_snapshots/aires_rad_manifest.txt` — 5 dimensions (lat=64, lev=10, lev_2=13, lon=128, time=1459) and 22 variables (5 coordinate + 17 data: hfls, hfss, hus, mrso, pl, pr, pr_6h, rls, rlut, rss, rst, rsut, ta, tas, ua, va, zg). Time count = 1459 (matches exp25/exp26: same 5-step trim from `runsum,6`).

Round-trip diff (`/tmp/aires_rad_verify.txt` vs snapshot): empty — exact match.

### B.4 flip commit

Atomic edits performed: `PROVISIONAL_PROFILES` reset to `set()`; module docstring's Provisional block merged into Audited; argparse description updated to list `aires_rad` as audited; `--profile` help text reverted to "Audited variable-set profile to produce."; SKILL.md tables merged; provisional callout and failure-mode row removed; verification loop expanded to include `aires_rad`. After this commit, the runtime PROVISIONAL warning no longer fires for `aires_rad`.

---

## Expanded contract audit (2026-04-21)

Captured in support of commits `7fc8a7c` (burn7 patch), `789caf8` (postprocessor rewrite), `eda188f` (snapshot lock). The driver is now single-purpose (no `--profile`), so there is one locked variable set instead of per-profile sets. The EXP15/25/26/aires_rad sections above are kept for historical record; the current contract lives here.

### Method

- **Sample input**: `/scratch/10000/amarchakitus/PLASIM/data/sim30/MOST.0012`, same as the prior audit.
- **burn7 binary**: `src/plasim_postprocessor/burn7/Stampede3/burn7`, rebuilt from the patched `burn7/burn7.cpp` with the three edits in Commit 1 (relax the zg-on-sigma guard, widen the hydrostatic trigger, add the midpoint hydrostatic-integration block).
- **Driver**: `src/plasim_postprocessor/plasim_postprocessor.py` run with `--sims 30 --years 12 12 --with-sea-ice`.
- **Module-load stanza**: unchanged from the prior audit — `intel/24.0 cdo netcdf python/3.12.11` plus the two `LD_LIBRARY_PATH` entries for burn7's C++ runtime linkage.
- **Snapshots locked**: `docs/audit_snapshots/manifest.txt` (default run, 30 vars) and `docs/audit_snapshots/manifest_with_sea_ice.txt` (31 vars, adds `sic`).
- **Audit run**: SLURM job 3048472 on `c476-003`, elapsed 01:48. Full log at `/scratch/11114/zhixingliu/SFNO_Climate_Emulator/audit/commit4/audit_3048472.out`.

### Breaking changes vs the prior per-profile contracts

1. **`zg` is now sigma-level** (10 midpoints, co-located with ta/ua/va/hus). The pressure-level field is renamed to `zg_plev` (13 levels [50, 100, …, 1000] hPa). Consumers that read `zg` by name expecting pressure levels must update to `zg_plev` — known consumers: `PanguWeather/v2.0/config/SFNO_PLASIM_H5_*.yaml` (the `diagnostic_*_var_dict` entries at 50000 Pa) and `netcdf-to-h5-eventwise.py`'s `PRESSURE_LEVEL_VARS`.
2. **`pl` is re-introduced** as `float pl(time, lat, lon)` = `log_surface_pressure` (code 152 = `ln(ps in Pa)`, not `ln(ps/1000hPa)`). The SFNO emulator's `SINGLE_LEVEL_VARS` list names this variable.
3. **`td2m` is removed** from the sigma code set. sim30 does not emit code 168, and no SFNO emulator config references `td2m`. Kept out of the contract until both conditions flip.

### Per-variable sanity (sim30/MOST.0012, area-weighted global+time mean unless noted)

| Field | Bound | Measured | Status |
|---|---|---|---|
| `zg_plev` @ 500 hPa | 5400–5700 m | **5673.98** | ✓ |
| `zg` near-top (sigma midpoint 0.0383, p ≈ 37.7 hPa) | 20000–24000 m | **22541.3** | ✓ |
| `zg` near-surface (sigma midpoint 0.9833, p ≈ 968 hPa) | 100–600 m | **373.3** | ✓ |
| `pl` (= ln(ps in Pa), ps ≈ 98500 Pa → ln(ps) ≈ 11.498) | 11.0–12.0 | **11.4964** | ✓ (earlier draft's `≈ −0.015` target was computed in normalized units and is wrong; burn7's actual output is `ln(ps in Pa)`) |
| `ts` global mean | 280–290 K | **287.86** | ✓ |
| `sic` global mean (polar-only) | 0.03–0.08 | **0.0434** | ✓ |

### Cross-validation: patched sigma-zg vs patched pressure-zg

Recorded during Commit 1's audit gate (SLURM job 3048339, `audit/commit4/../smoke/commit1/verify2_3048339.out`):

- pressure-zg @ 50 hPa: 20,778 m (standard atm 20,580 m — within 1 %)
- pressure-zg @ 100 hPa: 16,472 m (standard atm 16,180 m — within 2 %)
- sigma-zg @ lev=1 (p ≈ 37.7 hPa): 22,395 m
- Difference `sigma-zg(37.7 hPa) − pressure-zg(50 hPa)`: 1,617 m
- Predicted hypsometric range for T_avg ∈ [200, 240] K over the layer: [1,649, 1,979] m
- Agreement: measured falls 32 m below the T=200 K lower bound, consistent with T_avg ≈ 196 K for the upper stratosphere. The patched integration is hydrostatically consistent with the existing pressure-level path.

The pre-swap behavioral-equivalence test (old vs new binary on the pre-patch code set) was also clean: identical variable lists, 13/14 vars bit-identical on sim30/MOST.0012, `ua` relative diff 7.2e-8 / absolute 1.1e-13 (well under the 1e-5/1e-10 bounds — expected compiler instruction-reorder).

### Known downstream consumers to update (post-rename)

The `zg → zg_plev` rename is the only breaking change to the NetCDF variable namespace. Update sites:

- `$WORK2/awikner/stampede3/PanguWeather/v2.0/config/SFNO_PLASIM_H5_*.yaml`: replace `"zg": [50000]` in `diagnostic_gif_var_dict`, `diagnostic_acc_var_dict`, `diagnostic_spectrum_var_dict`, `diagnostic_bias_var_dict` with `"zg_plev": [50000]`.
- `$WORK2/awikner/stampede3/PanguWeather/v2.0/utils/data/netcdf-to-h5-eventwise.py`: replace the `"zg"` entry in `PRESSURE_LEVEL_VARS` with `"zg_plev"`.
- Any hand-written diagnostic/plotting script that does `ncfile['zg'][:]` expecting pressure-level zg.

These are cross-repo changes, not gated by commits here. The expanded-contract manifest lock means any consumer that diffs against the old per-profile snapshots will fail loudly on the variable-list check, so the rename will surface quickly.
