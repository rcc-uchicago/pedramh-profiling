# Standalone prompt: build the SFNO-5410 climatology on Derecho/Glade

> Copy everything between the `==== BEGIN PROMPT ====` and `==== END PROMPT ====` lines into the Derecho/Glade agent. The prompt is self-contained and assumes the agent has no context from the Stampede3 evaluation conversation.

==== BEGIN PROMPT ====

You are a Derecho/Glade agent on the NCAR Derecho cluster. Your job is to **build a single climatology NetCDF file** that will be transferred to Stampede3 and used as the **final** climatology for ACC scoring of the group's SFNO-5410 PlaSim emulator. This is the production climatology used in the published cross-emulator scorecard — **not a smoke test, not a placeholder**.

A separate AI-RES emulator on Stampede3 will be scored using a single, unified metric stack (`src/sfno_eval/`, Gauss–Legendre lat-weighted RMSE/ACC, time-of-year-proleptic climatology binning). For apples-to-apples ACC, the climatology used to score SFNO-5410 must come from the **same group post-processing pipeline** SFNO-5410 was trained against (your local Derecho data at `/glade/derecho/scratch/awikner/PLASIM/data/2100_year_sims_rerun/sim52/`), **not** from a re-packaged version. The group post-processing differs from the AI-RES Makani repackager, so values are not guaranteed numerically identical and the climatology must be regenerated from the group source files.

## 1. Source years — training/reference window only

- Build the climatology from group-processed years **`12-111`** (100 years total). These are the SFNO-5410 training reference window (`train_year_start=12, train_year_end=112` in `SFNO_PLASIM_H5_DERECHO_5410.yaml`).
- **Do NOT use years 121-129.** Those are the held-out evaluation years; including them would leak test-set information into ACC. The 9-year window 121-129 is also too noisy to act as a reference distribution. Reject any input list that overlaps `121-129` with a hard error.
- If any of years 12-111 are missing on Derecho/Glade, **report exactly which years are missing as a BLOCKER** in the manifest and stop. Do **not** silently substitute other years. Do **not** silently fall back to a smaller window. Do **not** fall back to `121-129`. Do **not** pull from any AI-RES `sim52_full/` re-packaged tree even if visible.

## 2. Source data path

- **Primary source (now confirmed bit-equivalent to per-timestep h5):** the per-year aggregated NetCDFs under
  `/glade/derecho/scratch/awikner/PLASIM/data/2100_year_sims_rerun/sim52/sigma_data/<year>_gaussian.nc`
  (or whichever sibling directory holds the yearly aggregates with the 5410 channel layout). Derecho-side spot-checks on sampled channels confirmed bit-for-bit equivalence to the per-timestep h5 series, so building from the yearly NetCDFs is faster (fewer file opens, no per-timestep concatenation overhead) and produces the same climatology.
- **Acceptable alternate:** the per-timestep h5 files under
  `/glade/derecho/scratch/awikner/PLASIM/data/2100_year_sims_rerun/sim52/h5/sigma_data/`
  (naming pattern `<year>_<timestep>.h5`, one timestep per file). Use this only if the yearly-NetCDF source is unavailable for some years; in that case the manifest must record which years came from which source and re-confirm the bit-equivalence on at least one shared sample.
- **Do not mix sources within a single year.** Pick one source family per year and document it; mixing within a year breaks reproducibility.
- **Do not** read `/glade/.../sim52/sigma_data/climatology.nc` and re-shape it — that file uses the group's day-of-year axis and may differ from our scorer's expected schema; we are deliberately rebuilding to control the binning.

## 3. Channel slate (53 channels, this exact order, with **group-processing units**)

Match the SFNO-5410 evaluation target slate exactly. **The unit conventions below are non-standard and must be preserved verbatim from the source files** — see §3a for what *not* to do.

```
0      pl                                    # log surface pressure (dimensionless): pl = ln(p_s),
                                             # NOT surface pressure in Pa
1      tas                                   # near-surface air temperature (K)
2..11  ta1..ta10                             # T (K) on 10 sigma levels
                                             #   sigma=[0.0383, 0.1191, 0.2109, 0.3169, 0.4368,
                                             #          0.5668, 0.6994, 0.8234, 0.9241, 0.9833]
                                             #   (TOA → surface)
12..21 ua1..ua10                             # u (m/s) on the same 10 sigma levels
22..31 va1..va10                             # v (m/s) on the same 10 sigma levels
32..41 hus1..hus10                           # specific humidity (kg/kg) on the same 10 sigma levels
42..51 zg200, zg250, zg300, zg400, zg500,
       zg600, zg700, zg850, zg925, zg1000    # geopotential height (gpm = geopotential meters),
                                             # NOT geopotential in m² s⁻². On 10 pressure levels
                                             #   plev=[20000, 25000, 30000, 40000, 50000,
                                             #         60000, 70000, 85000, 92500, 100000] Pa
                                             #   (TOA → surface; zg500 sits at index 46)
52     pr_6h                                 # snapshot precipitation rate × 6 h (mm or kg m⁻²
                                             # depending on group writer): this is the instantaneous
                                             # rate at the timestep multiplied by 6 hours, NOT a true
                                             # ∫₀⁶ rate dt accumulator. Treat as a 6-hour proxy.
```

Use **physical units as written by the group post-processor** — do **not** apply the group's `data_12-132_*_sigma.nc` mean/std normalization.

If any of these 53 channels is missing or has a different vertical-coord convention in the source files, **stop and report as a BLOCKER**. Do not silently substitute alternate levels.

### 3a. Conventions that must be preserved (do **NOT** convert)

These are group-pipeline conventions, not standard CF/ERA5 conventions. Downstream Stampede3 scoring explicitly assumes the climatology preserves them.

- **`pl` is dimensionless `ln(p_s)`.** Do not multiply or otherwise convert to surface pressure in Pa. If the source files store something else under the name `pl`, **stop and report as a BLOCKER**.
- **`zg*` channels are geopotential height in geopotential metres (`gpm`).** Do not multiply by g₀ ≈ 9.80665 to recover m² s⁻². Numerical magnitude (e.g., zg500 ≈ 5500–5900 gpm at midlatitudes) is the same as standard geopotential height in metres; the unit string is `gpm` (or whatever the source files write — record it verbatim in the manifest).
- **`pr_6h` is `instantaneous_pr_rate(t) × 6h`, not `∫₀⁶ pr_rate dt`.** Do not re-derive from a higher-cadence accumulator or rate field. Use the channel as written. The Stampede3 scoring report will label this channel as a "6-hour proxy", not as a true accumulator.

Set the per-variable `units` attribute on each variable in the output NetCDF to the unit string the source files use (e.g., `units = "gpm"` for `zg*`, `units = "1"` or `units = ""` for `pl`, `units = "kg m-2"` or `units = "mm"` for `pr_6h` — whatever the group post-processor writes). Do not invent unit strings.

## 4. Output schema (must match the Stampede3 scorer's expected format)

Single NetCDF file (uncompressed, fp32 accumulators). The scorer expects this schema:

- **Dims:**
  - `time_of_year=366`
  - `hour_quarter=4`
  - `channel=53`
  - `lat=64`
  - `lon=128`

- **Coords:**
  - `time_of_year` — int, day-of-leap-year ordinal in `[0, 365]`. Define ordinal so that day 0 = Jan 1, day 59 = Feb 29 (leap day), day 365 = Dec 31. Document this rule in a `time_of_year_convention` global attribute. Bins for non-existent dates in non-leap years (only Feb 29 in non-leap years) are simply assigned 0 contributors from those years; no NaNs in the time axis itself.
  - `hour_quarter` — int, values `[0, 6, 12, 18]` UTC.
  - `channel` — 53 string labels in the exact order listed in §3 (`pl`, `tas`, `ta1`, …, `ta10`, `ua1`, …, `ua10`, `va1`, …, `va10`, `hus1`, …, `hus10`, `zg200`, `zg250`, `zg300`, `zg400`, `zg500`, `zg600`, `zg700`, `zg850`, `zg925`, `zg1000`, `pr_6h`).
  - `lat` — 64 Gauss-quadrature latitudes from the source file, north-pole first (`+87.864…` first, `-87.864…` last). Use the latitude array from the SFNO-5410 yaml verbatim if the source files do not carry one.
  - `lon` — 128 equiangular longitudes `0, 2.8125, …, 357.1875`.

- **Variables (fp32, physical units):**
  - `mean(time_of_year, hour_quarter, channel, lat, lon)` — climatological mean per bin.
  - `std(time_of_year, hour_quarter, channel, lat, lon)` — climatological standard deviation per bin.
  - `n_contributors(time_of_year, hour_quarter)` — int32, number of source-file timesteps that fell into each `(day, hour_quarter)` bin across all years used.

- **Global attributes (required):**
  - `calendar = 'proleptic_gregorian'`
  - `has_year_zero = True`
  - `source_pipeline = 'derecho_glade_group_post_processing'`
  - `source_path` — absolute Derecho path of the input directory.
  - `years_used` — explicit comma-separated year list, e.g. `12,13,14,...,111`.
  - `n_years_used`
  - `n_leap_years_used`
  - `time_of_year_convention` — string explaining the day-0 = Jan 1 / day-59 = Feb 29 mapping.
  - `time_indexing = 'time_of_year_proleptic'`
  - `created_at` — ISO 8601 timestamp.
  - `built_by_sha7` — git short SHA of the build script (`unknown` if not under git).
  - `intended_consumer = 'AI-RES src/sfno_eval/ scorer (Stampede3)'`
  - `purpose = 'final ACC scoring for SFNO-5410, not a smoke test'`

Approximate file size: 366 × 4 × 53 × 64 × 128 × 4 B × 2 vars ≈ 5 GB. Do **not** apply zlib compression unless the resulting file exceeds 10 GB — uncompressed reads are faster on Stampede3's Lustre.

## 5. Calendar / leap-day handling

- Each source timestep has an absolute datetime in the proleptic-Gregorian calendar (`calendar: 'proleptic_gregorian'` per the SFNO-5410 yaml; data cadence is 6 hours from `data_timedelta_hours: 6`).
- For each timestep, compute the absolute datetime, derive `(month, day, hour)` where `hour ∈ {0, 6, 12, 18}` (snap to the nearest 6-hour quarter), then map `(month, day)` → the day-of-leap-year ordinal `time_of_year ∈ [0, 365]` defined in §4.
- **Cross-check the calendar mapping before running the full pass.** Read the calendar metadata (`plasim_time_units`, `time_plasim`, or whatever the group h5/NetCDF source carries) on at least 3 files spanning leap and non-leap years, and verify your computed `(year, month, day, hour)` ladder matches what those metadata imply. If you cannot independently confirm the mapping, **stop and report as a BLOCKER**.
- Feb-29 bins (`time_of_year=59, hour_quarter ∈ {0,6,12,18}`) will receive only ~24 contributors (one per leap-year file in the 12-111 window). That is expected, not a bug. Verify in §7.

## 6. Algorithm

- Use **Welford's online algorithm** to compute mean and std in a single pass over disk. fp32 accumulators are sufficient.
- Memory: two accumulators (`mean`, `M2`) at `(366, 4, 53, 64, 128)` × 4 B = ~5 GB each → ~10 GB resident, plus `(366, 4)` int32 count → negligible. Run on a CPU node with at least 32 GB RAM.
- Single-process is fine — no MPI/DDP needed; this is I/O bound. If you parallelize, ensure deterministic reduction order (e.g., per-year partial accumulators reduced sequentially) so the output is reproducible.
- Read each source h5/NetCDF exactly once. For each timestep: compute the bin, update `(mean, M2, count)` in place.
- **Reproducibility:** seed nothing; the result is deterministic given the source-file order. Print the source-file order at the start of the run and store it as a sidecar log.

## 7. Validation (must run before declaring success)

After the file is built, the script must run these checks and the manifest must record PASS/FAIL for each:

1. **Shape:** `mean.shape == (366, 4, 53, 64, 128)`, `std.shape == (366, 4, 53, 64, 128)`, `n_contributors.shape == (366, 4)`.
2. **Channel coord:** `len(channel) == 53` and the labels match the slate in §3 exactly (string comparison).
3. **Lat/lon coord:** `lat.shape == (64,)`, `lon.shape == (128,)`, `lat[0] > 0` (north pole first), `lat[-1] < 0`, `lat[0] ≈ 87.864`, `lon[0] == 0.0`, `lon[-1] ≈ 357.1875`.
4. **Finiteness on populated bins:** for every `(t, q)` where `n_contributors[t, q] > 0`: `np.isfinite(mean[t, q, :, :, :]).all()` and `np.isfinite(std[t, q, :, :, :]).all()` and `(std[t, q, :, :, :] >= 0).all()`.
5. **Non-leap-day population:** for `time_of_year ∉ {59}` (i.e., not Feb 29), `n_contributors[t, q]` should equal `n_years_used` for all `q` (e.g., 100 if 12-111 used). Tolerate up to 1% missing per bin; anything larger is a FAIL.
6. **Leap-day population:** for `time_of_year == 59` (Feb 29), `n_contributors[59, q]` should equal `n_leap_years_used` for all `q` (e.g., 24 if 12-111 used; verify against the calendar trace from §5).
7. **Sanity ranges (per-channel min/max of `mean` averaged over `(t, q, lat, lon)`)**, given the group conventions in §3 / §3a:
   - `tas` mean ≈ 250-310 K
   - `pl` mean ≈ 11.3-11.6 (dimensionless; `ln(p_s)` ≈ ln(1.0e5 Pa) = 11.51). **NOT** ~10⁵.
   - `ta1..ta10` mean ≈ 200-300 K (cooler at low sigma, warmer near surface)
   - `pr_6h` mean ≥ 0
   - `hus1..hus10` mean ≥ 0
   - `zg500` mean ≈ 5400-5900 gpm at midlatitudes. **NOT** ~5×10⁴ m² s⁻². If the channel comes out near 5×10⁴ this almost certainly means a g₀ multiplication slipped in — STOP and investigate.
   - `zg1000` mean ≈ 0-200 gpm (near surface).
   - `zg200` mean ≈ 11500-12500 gpm (upper troposphere).
   Print all 53 channel means and stds; flag any channel whose values look off by an order of magnitude or imply an accidental unit conversion.
8. **Round-trip:** open the freshly written file with `xarray.open_dataset(...)` (close-and-reopen) and re-run checks 1-3. The file must be readable by xarray without warnings.

If any check fails, the manifest must record `BUILD_STATUS=FAIL` with the failing checks listed, and the file must be moved to a `failed/` subdirectory (not delivered for transfer).

## 8. Deliverables

1. **Climatology NetCDF** at a Derecho path you choose; recommended:
   `/glade/derecho/scratch/<your_user>/AI-RES/climatology/sfno_5410_clim_yr12-111_proleptic_<YYYYMMDD>.nc`

2. **Manifest** (markdown or JSON) at the same directory, recording:
   - Input source path(s) used.
   - Explicit list of years used (and any years skipped, with reason).
   - Source-file order (or a checksum thereof).
   - Channel order (all 53 names, in order).
   - Calendar handling chosen, plus the cross-check log from §5.
   - Output schema (the dims, coords, variables, attributes — basically `ncdump -h` output of the final file).
   - Output absolute path, sha256, file size in bytes.
   - Validation PASS/FAIL for each of the 8 checks in §7, with diagnostics for any FAIL.
   - Build wallclock duration.
   - Build script path and git SHA (or `unknown`).
   - `BUILD_STATUS=PASS` or `BUILD_STATUS=FAIL`.

3. **Build script** committed in your local working tree. The script must:
   - Accept `--years 12-111` (or equivalent) as a required argument.
   - Reject any year list overlapping `121-129` with a hard error (exit non-zero, no output written).
   - Print the source-file order before processing.
   - Run all validation checks in §7 and exit non-zero on any failure.

## 9. Transfer to Stampede3

After publishing (only on `BUILD_STATUS=PASS`), report to the user:
- The Derecho absolute path of the NetCDF.
- The sha256.
- The file size.
- A one-line transfer invocation suitable for Stampede3 destination:
  `/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/baselines/climatology_proleptic_5410.nc`
  (Use Globus, `scp`, or `rsync` — pick whatever your local tooling already supports. The Stampede3 directory `baselines/` may not exist yet — `mkdir -p` is fine.)

The Stampede3-side scorer reads via `xarray.open_dataset(...)`. No further format conversion is expected.

## 10. Out of scope (do NOT do these)

- Do **not** z-score / normalize. Use physical units as written by the group post-processor.
- Do **not** convert `pl` from `ln(p_s)` to Pa.
- Do **not** convert `zg*` from gpm to m² s⁻² (no multiplication by g₀ ≈ 9.80665).
- Do **not** re-derive `pr_6h` from a higher-cadence rate or true 6-hour accumulator. Use the channel as the group writer wrote it.
- Do **not** include any data from years 121-129.
- Do **not** read or re-shape `/glade/.../sim52/sigma_data/climatology.nc` — that file's day-of-year axis and channel layout do not match our scorer's expected schema.
- Do **not** mix in data from any AI-RES re-packaging.
- Do **not** skip the calendar cross-check in §5. The Aug-1 anchor offset (or whatever the group's anchor convention is) is the most fragile assumption in this build.
- Do **not** silently aggregate fewer than 100 years if 12-111 is the requested window. Report as a BLOCKER and stop.
- Do **not** treat this as a smoke test. The output is the published-scorecard climatology.

---

If anything in this prompt conflicts with what you find on Derecho — for example, the channel slate doesn't match the actual 5410 source files, or the calendar anchor convention is different from what's implied above — **stop and ask before proceeding**. The Stampede3 evaluation is downstream of this artifact, so silent inconsistency will produce wrong ACC numbers in the published comparison between the two emulators.

==== END PROMPT ====
