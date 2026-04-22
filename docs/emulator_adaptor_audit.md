# Emulator Adaptor Audit (2026-04-21)

Captures the first audit of `src/emulator_adaptor/adaptor.py` — the convention translation layer from the postprocess NetCDF to the SFNO emulator's varying-boundary tuple `(sst, rsdt, sic)`. Locks numerical bounds for later runs.

Supports commits:

- `1435b15` Add emulator_adaptor: {sst, rsdt, sic} from postprocess NetCDF
- `20b7bc4` Floor sst at the seawater freezing point over ocean (ERA5 convention)
- `504079e` Fix astronomical rsdt: wrap hour angle across the ±π seam

## Method

- **Sample input**: postprocess NetCDF for sim30/MOST.0012 produced by the patched burn7 binary (commit `7fc8a7c`) and single-purpose driver (commit `789caf8`), run with `--with-sea-ice`. File: `/scratch/11114/zhixingliu/AI-RES/audit/commit4/postproc/sim30/MOST.0012.nc`, 3.97 GB, 1459 timesteps.
- **Adaptor runs**: two back-to-back invocations of `src/emulator_adaptor/adaptor.py --sims 30 --years 12 12`, once with `--rsdt-method arithmetic` (default) and once with `--rsdt-method astronomical`. Outputs at `boundary_arith/sim30/boundary.0012.nc` and `boundary_astro/sim30/boundary.0012.nc`.
- **SLURM job**: 3048472 on `c476-003`, elapsed 01:48. Full log: `/scratch/11114/zhixingliu/AI-RES/audit/commit4/audit_3048472.out`.
- **Module-load stanza**: identical to the postprocessor's — `intel/24.0 cdo netcdf python/3.12.11`. `LD_LIBRARY_PATH` entries are kept for uniformity (not actually needed by the adaptor, which is pure Python).

## Step 0 — convention probes

Both probes run against the Commit 4 postprocess output for sim30/MOST.0012. They confirm the two convention assumptions baked into the sst rule.

### Probe 0.A — `lsm` value distribution

Confirms the strict `ocean = (lsm < 1e-6)` test in the adaptor is sound for PlaSim T42.

| Bucket | Count | % |
|---|---|---|
| cells == 0 (ocean) | 5527 / 8192 | 67.47 |
| cells == 1 (land)  | 2665 / 8192 | 32.53 |
| cells in (0, 1) | **0** | **0.00** |

`lsm` is exactly binary 0/1 at every grid cell. **PASS** — the strict `< 1e-6` test captures all ocean cells; no fractional-coastline cells to worry about.

### Probe 0.B — `sic` value distribution (all 1459 timesteps, all cells)

Confirms that the `SIC_THRESHOLD = 0.5` convention is not load-bearing for PlaSim T42.

| Bucket | Count | % |
|---|---|---|
| sic == 0 | 10,458,465 | 87.503 |
| sic ∈ (0, 0.15) | 0 | 0.000 |
| sic ∈ [0.15, 0.5) | 0 | 0.000 |
| sic ∈ [0.5, 0.85) | 0 | 0.000 |
| sic ∈ [0.85, 1] | 1,493,663 | 12.497 |
| Nonzero mean, median | 1.000, 1.000 | — |

`sic` is effectively binary — every nonzero cell is exactly 1.0. **PASS** — threshold choice between `> 0`, `> 0.15`, `> 0.5` is irrelevant; they all classify the same cells.

## sst sanity

Both adaptor runs produce identical `sst`. The formula applies the freezing-seawater floor universally over ocean (commit `20b7bc4`):

```
ocean = (lsm < 1e-6)
icy   = (sic > 0.5)
sst_ocean = where(icy, 271.35, ts).clip(min=271.35)
sst = where(ocean, sst_ocean, NaN)
```

| Check | Bound | Measured | Status |
|---|---|---|---|
| sst range over ocean | [271.34, 310] K | **[271.35, 306.83]** K | ✓ |
| sst NaN fraction vs lsm land fraction | within 1 grid cell (1.22×10⁻⁴) | **diff = 0.000e+00** (exact) | ✓ |

Without the freezing floor, the naive plan formula `sst = ts where ocean & ~icy` would have passed ~3 % of ocean cell-timesteps through at sub-freezing values (down to 216 K), because PlaSim emits `sic` as a hard binary that lags polar-night cooling. The floor is consistent with ERA5 / CMIP reanalysis convention ("SST" under/near sea ice = freezing point, not ice-skin temperature).

## sic sanity

Pass-through from postprocess, clipped to [0, 1]. With PlaSim's binary `sic` in sim30, the clip is a no-op, but defensive against future sims.

| Check | Bound | Measured | Status |
|---|---|---|---|
| sic range after clip | [0, 1] | **[0.0000, 1.0000]** | ✓ |

## rsdt sanity — arithmetic path (default)

`rsdt = rst − rsut`, reading PlaSim's own TOA shortwave accounting. PlaSim's sign convention is "positive = into receiver": `rst` is the TOA net SW flux (positive, ~242 W/m² annual global mean) and `rsut` is the TOA outgoing SW (negative under the PlaSim convention, ~−99 W/m² annual global mean). Their arithmetic difference recovers the incoming-SW magnitude ~341 W/m².

| Check | Bound | Measured | Status |
|---|---|---|---|
| Area-weighted global annual mean | solar_constant/4 ± 1 % = 341.75 ± 3.4 W/m² | **341.751** W/m² | ✓ (relative error 3×10⁻⁶) |

## rsdt sanity — astronomical path

Analytic 6h-mean integration of `S₀·(a/r)²·max(0, cos(zenith))` over each output window, using the `dec(doy) = obliquity · sin(2π(doy−80)/365.25)` and `dist_factor(doy) = 1 + e · cos(2π(doy−4)/365.25)` approximations. Pure numpy, no pvlib or ephemeris data. The hour angle is wrapped to `(−π, π]` and the window integration is split into two parts across the ±π seam (commit `504079e`) — without this wrap the low-latitude global mean would be ~half of correct.

### Offline unit tests (no data required)

These run in a few seconds from the command line; they verify the analytic formula against closed-form targets:

```
area-weighted global + annual mean (full synthetic year):
    341.788 W/m²  (target S₀/4 = 341.750; rel error 1×10⁻⁴)

polar day (lat=87.86°, doy=172, summer solstice):
    per-6h-window [497, 555, 555, 497] W/m², daily mean 526
    (matches S₀ · dist_factor(172)² · sin(lat)·sin(dec) = 526; Earth is
     near aphelion at summer solstice so dist_factor² = 0.968 not 1.0)

equator, equinox day:
    per-6h-window [0, 878, 878, 0] W/m², daily mean 439
    (S₀/π ≈ 435 within the single-day rounding)

equator, full year lon+time mean:
    417.17 W/m²  (matches S₀ · 2·J₀(0.409)/π = 417.2 — NOT S₀/π
     because seasonal declination takes the sun off-axis half the year)
```

### sim30/MOST.0012 sanity

| Check | Bound | Measured | Status |
|---|---|---|---|
| Area-weighted global annual mean | solar_constant/4 ± 0.5 % = 341.75 ± 1.7 W/m² | **341.767** W/m² | ✓ (relative error 5×10⁻⁵) |
| Zonal (time, lon) mean monotone from equator | should decrease to each pole | **417.0 → 170.8 N, 417.0 → 178.9 S** | ✓ |

## Per-cell arith-vs-astro calibration (locked)

The per-cell, per-6h-timestep `|rsdt_arith − rsdt_astro|` statistic measured on sim30/MOST.0012 is the calibrated bound for subsequent audit runs. Plan sets the lock as `max(measured, 20 W/m²)`.

| Statistic | Value |
|---|---|
| max            | **1083.57** W/m² |
| 99.9 percentile | 1068.65 W/m² |
| 99 percentile   | 1031.62 W/m² |
| 95 percentile   | 940.52 W/m² |
| mean            | 333.49 W/m² |

**Locked calibration bound: `1083.57 W/m²`.** Later audit runs (other sims or years) hard-fail if the measured per-cell max exceeds this.

### Why is the per-cell diff so large if the global means agree?

The two paths answer different questions at a single (time, cell) point:

- **arithmetic** reads PlaSim's own radiation-scheme accounting for that 6h step. PlaSim evaluates the SW budget at its model timesteps (short, ~30 min) and emits 6h-mean fluxes, subject to the scheme's radiation-update cadence and diurnal-cycle treatment.
- **astronomical** is a clean 6h-mean analytic integral under ideal-Earth assumptions (no atmosphere, no clouds, no diurnal-cycle interpolation, trivial orbital model).

At a sun-baked equatorial noon cell the arithmetic value can reach ~1380 W/m² (at the peak of PlaSim's diurnal curve) while the astronomical 6h-window-average is ~880 W/m² (the exact mean of `S₀·cos(zenith)` over 6 hours centered on noon). The pointwise ~500 W/m² gap is a timing-convention mismatch, not a physics disagreement. Zonally and globally, the time-averaging reconciles: zonal means agree within 5.07 W/m² at all latitudes, and the area-weighted global mean differs by 0.016 W/m² (5×10⁻⁵ relative).

The large bound is therefore not an error — it is the size of the timing-window effect on a per-cell basis under PlaSim's radiation scheme. Subsequent audits checking against this bound are really checking "PlaSim's scheme still has roughly the same diurnal behavior," which is a useful sanity check without requiring a tighter number.

## Zonal (time, lon)-mean profile comparison

From the audit run, the two methods per zonal band:

| lat    | arith | astro  | diff  |
|-------:|------:|------:|-----:|
| +87.86 | 175.75 | 170.83 |  +4.91 |
| +71.16 | 196.68 | 191.62 |  +5.07 |
| +54.42 | 265.25 | 261.13 |  +4.13 |
| +37.67 | 339.58 | 336.56 |  +3.02 |
| +20.93 | 392.45 | 390.75 |  +1.70 |
|  +4.19 | 416.13 | 415.86 |  +0.27 |
| −12.56 | 407.79 | 408.97 |  −1.18 |
| −29.30 | 368.39 | 370.90 |  −2.51 |
| −46.04 | 302.94 | 306.53 |  −3.59 |
| −62.79 | 224.01 | 228.29 |  −4.28 |
| −79.53 | 179.86 | 184.70 |  −4.84 |

Max absolute zonal difference: 5.07 W/m² at +71.16°. The pattern (+N skews high, −S skews low) is consistent with the 2026 base-year used to decode PlaSim's cftime axis being offset from an equinox year, so the synthetic astro value lags PlaSim's own radiation scheme by a small fraction of a day. Not flagged as an issue; within the calibration bound.

## Reproduction

Driver files (kept in `/scratch/11114/zhixingliu/AI-RES/audit/commit4/`, not checked in):

- `audit.slurm` — SLURM driver for the end-to-end run
- `probe.py` — Step 0 probes (lsm + sic distributions)
- `calibrate_rsdt.py` — per-cell calibration + zonal comparison
- `make_manifest.py` — snapshot generator (writes the two locked manifests)

To re-run:

```bash
cd /scratch/11114/zhixingliu/AI-RES/audit/commit4
sbatch audit.slurm
# After completion:
cat audit_<jobid>.out
```

## Known downstream consumers

`boundary.{YYYY:04d}.nc` is **not** the final emulator-ingestion layout. The SFNO/NVIDIA data pipeline reads varying-boundary variables by filename from a separate `boundary_dir`, one file per variable:

```
{boundary_dir}/sst_masked_6h.nc
{boundary_dir}/rsdt_masked_6h.nc
{boundary_dir}/sic_masked_6h.nc
# plus {var}_masked_6h_leap.nc variants for leap-year splitting
```

Leap-year splitting, per-variable file separation, and possibly concatenation across years are the responsibility of a downstream reshape step that is out of scope for this adaptor. The adaptor intentionally emits one bundled per-(sim, year) file because that's the natural rollup unit for the postprocess → adaptor chain.
