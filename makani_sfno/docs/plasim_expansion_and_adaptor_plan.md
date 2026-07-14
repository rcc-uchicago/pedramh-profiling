# PlaSim Postprocessor Expansion + Emulator Adaptor — Implementation Plan

## Context

The current postprocessor at `src/plasim_postprocessor/plasim_postprocessor.py` emits a locked variable set that covers most SFNO emulator inputs but has four gaps: `pl` is missing, `zg` is only on pressure levels (emulator tensor wants sigma), and `sst` + `rsdt` have no burn7 equivalents. The emulator contract is moving toward NVIDIA's implementation, so the resolution needs to be physically correct rather than shaped to the current SFNO file layout.

After an architectural review, the four variables split across two layers:

- **Postprocess**: `pl` (native burn7), `zg` on sigma (requires a small burn7 patch). Kept inside the postprocess because both are native PlaSim fields that physically belong with `ta/ua/va/hus`.
- **Downstream adaptor** (new): `sst` and `rsdt` — convention-dependent derivations that belong in an emulator-contract translation layer, not in the "native PlaSim fields" postprocess.

This plan implements both layers, retains the existing pressure-level `zg` data (under the new name `zg_plev`), and re-audits the postprocess to lock the expanded contract. See "Breaking contract changes" below — the pressure-level `zg` rename is a breaking change for any consumer that reads that variable by name, which is called out explicitly so downstream configs get updated as part of this rollout.

## Locked decisions

| Decision | Choice |
|---|---|
| zg on sigma — method | Patch burn7 (`burn7.cpp`); rebuild Stampede3 binary |
| Pressure-level zg | Keep both (renamed to `zg_plev`) |
| Sigma-zg vertical representation | Direct hydrostatic integration to arithmetic-mean sigma midpoints (co-located with ta/ua/va/hus) |
| `pl` | Add to `SIGMA_CODES`; reverses the draft expand-plan decision |
| Adaptor location | New `src/emulator_adaptor/` module |
| sst convention | `sst = ts` where `lsm==0 & sic<=SIC_THRESHOLD`; `sst = 271.35 K` where `lsm==0 & sic>SIC_THRESHOLD`; NaN `_FillValue` elsewhere |
| `SIC_THRESHOLD` for "ice-covered" | `0.5` — majority-ice convention (ERA5/CMIP-style). **Explicitly signed off 2026-04-21** as a semantic convention, separate from the earlier sst-layout question. The user chose `sic > 0.5` over `sic > 0` and `sic >= 0.15` after Codex flagged that the threshold was a convention choice that needed its own sign-off. |
| Adaptor requires `sic` | Yes — fails fast if not present |
| rsdt method | Both implemented; default `arithmetic` (`rst − rsut`); `--rsdt-method astronomical` optional |
| Adaptor output | Per-(sim, year) NetCDF at `{output-root}/sim{NN}/boundary.{YYYY:04d}.nc` |
| Audit strategy | Full audit: new locked manifest + per-variable sanity |

## Breaking contract changes

These changes intentionally break the previous variable-name contract. They are listed here so the matching downstream updates are part of this rollout, not silent fallout.

1. **`zg` on pressure levels is renamed to `zg_plev`.** The variable slot named `zg` in the postprocess NetCDF now holds the sigma-level field (10 sigma midpoints). Every consumer that previously read pressure-level `zg` by name must update.
   - Known consumer: SFNO emulator configs at `/work2/09979/awikner/stampede3/PanguWeather/v2.0/config/SFNO_PLASIM_H5_*.yaml`. Fields `diagnostic_gif_var_dict`, `diagnostic_acc_var_dict`, `diagnostic_spectrum_var_dict`, `diagnostic_bias_var_dict` contain `"zg": [50000]` entries that need to become `"zg_plev": [50000]` (those entries index by Pa on the pressure coordinate).
   - Known consumer: SFNO's NetCDF→H5 converter (`netcdf-to-h5-eventwise.py`) indexes pressure-level variables by `ds.plev.values`; after this change, the `zg` that lives on `plev` will be named `zg_plev`. The H5 converter's `PRESSURE_LEVEL_VARS` list entry `"zg"` must become `"zg_plev"`.
   - Any user-written diagnostic or plotting script that does `ncfile['zg'][:]` expecting the pressure-level field will now read the sigma field. Users need to be notified.
   - Downstream updates are not part of this repo and are therefore not gated by this plan's commits, but the audit doc Part D flags them as "known downstream consumers to update" so the rollout is visible.

2. **`pl` reappears in the postprocess output.** The currently-in-draft `docs/plasim_postprocessor_expand_plan.md` proposed removing `pl` (under the misreading that `ps` is a substitute). This plan reverses that decision. Consumers that were building against the draft's "no pl" output will see a new variable — additive, not breaking, but worth calling out.

3. **New `zg` on sigma appears where a pressure-level `zg` used to live.** Same variable name, different units / coordinate — worse than a pure rename in that the *interpretation* changes silently for a consumer that doesn't read the coordinate dimension. Mitigate by also bumping the output-contract docstring in `plasim_postprocessor.py` and locking the new manifest in the audit snapshot; anything that diffs against the old manifest will fail loudly.

## Critical files

To modify:
- `src/plasim_postprocessor/burn7/burn7.cpp` — relax zg-on-sigma guard + direct hydrostatic integration to sigma midpoints
- `src/plasim_postprocessor/burn7/Stampede3/burn7` — rebuilt binary artifact
- `src/plasim_postprocessor/plasim_postprocessor.py` — add `pl`, new zg routing, rename to `zg_plev`
- `docs/plasim_postprocessor_expand_plan.md` — update to reflect locked decisions (don't delete; preserve history)
- `docs/plasim_postprocessor_audit.md` — add expanded-contract audit section
- `docs/audit_snapshots/manifest.txt` — new locked manifest (supersedes per-profile snapshots only after sign-off)

To create:
- `src/emulator_adaptor/__init__.py`
- `src/emulator_adaptor/adaptor.py`
- `src/emulator_adaptor/submit.slurm`
- `docs/emulator_adaptor_audit.md`

Existing functions / constants to reuse:
- `_run_burn7`, `_run_cdo` in `plasim_postprocessor.py:96–119` — wrap subprocess, identical error surface
- `enumerate_tasks` in `plasim_postprocessor.py:72` — (sim, year) enumeration pattern
- `process_one` / `main` / argparse pattern in `plasim_postprocessor.py:122–244` — adaptor CLI mirrors this (same `--sims`/`--years`/`--task-index`/`--count-tasks`/`--overwrite`/`--dry-run` surface, same SLURM-array dispatch model)
- `MakeGeopotHeight` in `burn7.cpp:4395–4420` — hydrostatic integrator, unchanged; only the call-site wrapping changes
- Module-load contract documented in `docs/plasim_postprocessor_audit.md:72–78` — reused verbatim for both burn7 rebuild and adaptor runtime

## Part A — burn7 C++ patch (Commit 1)

Three edits to `src/plasim_postprocessor/burn7/burn7.cpp`:

1. **Relax the zg-on-sigma guard (line 6252–6258).** Replace the `printf(" * Geopotential height (156) requires pressure level *") ; exit(1)` block with a no-op. Rationale: `MakeGeopotHeight` already runs correctly on sigma inputs — the guard is a historical restriction, not a correctness requirement.

2. **Widen the hydrostatic-integration trigger (line 4652).** Current: `if (VerType == 'p' || Omega->needed)`. Change to: `if (VerType == 'p' || Omega->needed || GeopotHeight->needed)`. Ensures the `presh` / `MakeGeopotHeight` block runs when `code=zg` is requested in sigma mode.

3. **Integrate hydrostatic equation directly to sigma midpoints after MakeGeopotHeight (insert after line 4670).** Burn7's MakeGeopotHeight leaves zg on 11 half-level sigmas (top interface, 9 interior, surface-orography). The 10 emitted `lev` slots are midpoints at sigma = `0.5 · (sigma_half[k] + sigma_half[k+1])` (per `burn7.cpp:869–871`) — hence midpoint pressure `p_mid[k] = 0.5 · (ph[k] + ph[k+1])`. We need zg at those midpoint pressures, not at half-level interfaces.

   Simple averaging `0.5·(z_half[k] + z_half[k+1])` corresponds to the hydrostatic integral to a *log-mean* pressure midpoint, not the arithmetic-mean midpoint that burn7's `lev` encodes. The difference is small (~0.5 % of layer thickness) but it is an approximation. Codex flagged this — the correct move is to use the *same hydrostatic formula* that `MakeGeopotHeight` already uses, integrating from the half-level below each layer up to the arithmetic-mean midpoint:

   ```cpp
   if (GeopotHeight->needed && VerType != 'p') {
       // Replace half-level zg with midpoint zg by direct hydrostatic integration
       // from each layer's bottom half-level up to p_mid = 0.5*(ph[k] + ph[k+1]),
       // using the SAME virtual-temperature hydrostatic step as MakeGeopotHeight.
       // VTMP is scoped to MakeGeopotHeight's body (see line 4399) and is not
       // accessible from the caller — redeclare locally, identical definition.
       const double VTMP_local = (RV / RD) - 1.0;   // or use the global RETV if it exists in this build
       const double zrg        = 1.0 / Grav;
       double *T  = &Temperature->hgp[0];   // layer (midpoint) temperature
       double *Q  = &Humidity->hgp[0];      // layer (midpoint) specific humidity
       double *ph = &HalfPress->hgp[0];     // half-level pressures, SigLevs+1 per column
       double *z  = &GeopotHeight->hgp[0];  // input: half-level z; output: midpoint z
       for (int k = 0; k < SigLevs; ++k) {
           for (int ig = 0; ig < DimGP; ++ig) {
               int i_layer    = k * DimGP + ig;          // layer index (also = upper half-level)
               int i_halfbot  = (k + 1) * DimGP + ig;    // lower half-level
               double p_bot   = ph[i_halfbot];
               double p_top   = ph[i_layer];
               double p_mid   = 0.5 * (p_bot + p_top);
               double virt    = 1.0 + VTMP_local * Q[i_layer];
               double dz      = RD * T[i_layer] * virt * log(p_bot / p_mid) * zrg;
               z[i_layer]     = z[i_halfbot] + dz;       // z at midpoint = z at lower half + layer-up-to-midpoint thickness
           }
       }
       GeopotHeight->hlev = SigLevs;
       GeopotHeight->plev = SigLevs;
   }
   ```

   **Compile note** (Codex item 2): `VTMP` is declared as a local `double` inside `MakeGeopotHeight` at `burn7.cpp:4399`, so it is not visible at this insertion site. The patch declares a fresh local `VTMP_local = (RV / RD) - 1.0` using the same `RV`/`RD` file-scope constants already used by `MakeGeopotHeight`. If the build exposes a global `RETV` (a synonym used in some burn7 forks for the same quantity), the code should prefer `RETV` and drop the local declaration. Resolve during the build by grepping for `RETV` in `burn7.cpp`; if present, use it verbatim.

   In-place safe: at step k the code reads `z[(k+1)*DimGP + ig]` (a half-level value still untouched because the loop runs k=0 upward and writes k before reaching k+1). `z[(SigLevs)*DimGP + ig]` is the surface orography — pre-loaded at line 4669, still a valid boundary value.

   This is NOT averaging; it is the identical hydrostatic integral that MakeGeopotHeight uses, applied to a thinner sub-layer. No new physics introduced.

**Multi-cluster binary handling** (Codex item 4 — rewritten around actual repo state). In the current checkout, `git status` shows the Derecho and Jean-Zay prebuilt binaries (`burn7/derecho/burn7`, `burn7/jeanzay/burn7`) and their `submit_burn.sh` wrappers are already staged for deletion in a separate cleanup, not part of this plan. Only `burn7/Stampede3/burn7` is live in the working tree. That reframes the scope here:

- **In-repo**: rebuild `burn7/Stampede3/burn7` from the patched source on Stampede3 (recipe below). Nothing else in the `burn7/` subtree is touched by this plan — the pre-existing staged deletions land on their own schedule.
- **Out-of-repo / operational**: if someone later needs a Derecho or Jean-Zay binary for this patched `burn7.cpp`, they must rebuild on that cluster using the patched source. `make_burn.sh` (Derecho) and `makefile` (Jean-Zay) in the `burn7/` directory are historical build recipes from the pre-patch era and are kept as starting points; their line numbers may drift after this patch lands. This is a distribution concern, not a commit-gating one for the AI-RES repo.
- **Rationale for not re-introducing deleted binaries**: the staged deletions already in-flight reflect the fact that those binaries were pre-patch artifacts with no live audience. Re-adding rebuilt versions in this plan would contradict an in-progress cleanup. The operational rollout text above replaces a repo-level action with a per-cluster rebuild guide.

**Rebuild on Stampede3** (single compile line; existing `make_burn.sh`/`makefile` target Derecho/Jean-Zay and are not reused):

```bash
module purge
module load intel/24.0 cdo netcdf python/3.12.11
export LD_LIBRARY_PATH=/opt/apps/intel24/netcdf/4.9.2/x86_64/lib:/home1/09979/awikner/netcdf-4.2/lib:$LD_LIBRARY_PATH

cd src/plasim_postprocessor/burn7
# First match the existing binary's compiler:
file Stampede3/burn7
strings Stampede3/burn7 | grep -iE 'gcc|icc|icpx|oneapi' | head
# Then rebuild — try icpx first under intel/24.0 (OneAPI); fall back to g++ if that fails:
icpx -O2 burn7.cpp -o Stampede3/burn7.new \
  -I/opt/apps/intel24/netcdf/4.9.2/x86_64/include \
  -I/home1/09979/awikner/netcdf-4.2/include \
  -L/opt/apps/intel24/netcdf/4.9.2/x86_64/lib \
  -L/home1/09979/awikner/netcdf-4.2/lib \
  -lnetcdf_c++ -lnetcdf -lm
# Verify; only then replace:
mv Stampede3/burn7.new Stampede3/burn7
```

**Pre-swap smoke test** (required before committing the new binary; Codex item 5 — byte-for-byte is too brittle for a compiler rebuild):

- **Structural equality (hard).** Old-SIGMA_CODES namelist run (no zg) on both old and new binaries against the same input. Manifest diff using the `manifest()` helper from `docs/plasim_postprocessor_audit.md` (dimensions, variable list, `lev`/`lev_2` coords, time-coordinate summary). Must be empty.
- **Numerical tolerance per variable (hard).** For each data variable `v` in the old output, compute `cdo -s sub old.nc new.nc diff.nc` then `cdo -s output -fldmax -timmax -abs -selname,v diff.nc`. Absolute bound: `< 1e-6` for coordinate variables; `< 1e-5` relative (via `cdo -s output -div diff_abs v_abs`) for data variables. Rationale: compiler/library version churn can reorder float-ops and change lowest-bit results; a relative bound catches true semantic divergence while tolerating harmless reorder.
- **New sigma-zg presence (hard).** New sigma namelist with `code=...,zg` must succeed; `ncdump -h` shows `float zg(time, lev, lat, lon)` with `lev=10`.
- **New sigma-zg physical sanity (hard).** Global-mean check: near-surface `zg` (`sellev` at sigma=0.983) in the range 100–600 m; near-top `zg` (sigma=0.038) in the range 20000–24000 m. (Earlier draft said 15000–20000 m; corrected during the 2026-04-21 Commit 1 audit after measuring 22,395 m on sim30/MOST.0012. The earlier bound implicitly assumed `lev=1` sat at ~50-80 hPa; burn7's actual midpoint is at sigma 0.0383 → p ≈ 37.7 hPa, where standard-atmosphere zg is ~22,500 m. The new window brackets T_avg ∈ [200, 240] K from surface to 37.7 hPa.)

**Cross-validation** (three layers; Codex flagged that interpolation-based validation alone is weak):

1. **Self-consistency (analytic, no PlaSim data).** Before rebuilding, write a minimal C or Python reproduction of `MakeGeopotHeight` + the midpoint integration above applied to an isothermal dry atmosphere (T=270 K, q=0, ps=1000 hPa, pure sigma). Compare against the exact analytic solution `z(p) = R·T/g · log(ps/p)`. Agreement should be numerical-precision (< 0.01 m).

2. **Pre-patch vs patched at matching pressures.** Run the pre-patch binary with `vtype=p` on sim30/MOST.0012 to get pressure-zg on the 13 standard levels. Run the patched binary with `vtype=sigma` to get sigma-zg on the 10 midpoints. At each grid cell and timestep, compute `p_mid[k] = 0.5·(ph[k] + ph[k+1])` and linearly-interpolate the pressure-zg to `p_mid[k]` in log-pressure. Compare to the patched sigma-zg. Report RMSE per level (pressure-level zg is itself an interpolation, so this measures interpolation-vs-integration divergence); acceptance bound < 5 m mid-troposphere, < 20 m near surface. This is a bound on the *interpolation scheme in burn7*, not on the new integration.

3. **Hydrostatic self-check (most important).** In the patched binary's sigma output, for each (cell, time) and each k, compute `z[k] − z_half[k+1]` and verify it equals `R·T[k]·(1+virt·q[k])·log(p_bot/p_mid)/g` to machine precision. This is tautological w.r.t. the patch code but catches any in-place aliasing bug or accidental buffer reuse. Implement via a one-shot offline CDO+Python script against a single year.

Records go in the audit doc as offline validation; not part of the locked manifest diff.

## Part B — `plasim_postprocessor.py` changes (Commit 2)

Edits to `src/plasim_postprocessor/plasim_postprocessor.py`:

1. **Add `"pl"` to SIGMA_CODES** (line 51–66). Burn7 code 152; `All[152].Init("pl","log_surface_pressure","1",1)` — twod=1 so it emits as `float pl(time, lat, lon)`. Output-contract docstring (lines 14–27) should list it under the 2D surface block, not 3D.

2. **Add `"zg"` to SIGMA_CODES**. With the burn7 patch, sigma-zg is emitted in the same sigma call — no separate third burn7 invocation needed. This keeps the call count at 2 (sigma, pressure-zg) and avoids a third CDO merge.

3. **Route pressure-zg through a rename step to avoid NetCDF variable-name collision.** In `process_one` (lines 142–171), between the pressure-zg burn7 call and the sigma/pressure merge, add:

   ```python
   zg_raw = td / "zg_plev_raw.nc"
   zg_renamed = td / "zg_plev.nc"
   _run_burn7(opts.burn7_binary, zg_nl, input_path, zg_raw)
   _run_cdo(["chname,zg,zg_plev", str(zg_raw), str(zg_renamed)])
   _run_cdo(["merge", str(sigma_nc), str(zg_renamed), str(current)])
   ```

   `zg` (on `lev`, sigma, 10 levels) and `zg_plev` (on `lev_2`, pressure, 13 levels) will coexist in the merged NetCDF because CDO merge keys on dimension size, not value.

4. **Update the module docstring Output contract block (lines 14–27)** to list:
   - 3D sigma (10 levels): `ta, ua, va, hus, zg`
   - 3D pressure (13 levels): `zg_plev`
   - 2D surface (single-level): `pl, tas, td2m, ts, ps, psl, clt, mrso, lsm, z0, sg`
   - Radiation: `rss, rls, rst, rlut, rsut, hfss, hfls`
   - Precipitation: `pr, pr_6h`
   - Sea ice (conditional): `sic`

`submit.slurm` is not changed — no new CLI flag is added.

## Part C — `src/emulator_adaptor/` module (Commit 3)

Three new files. CLI driver follows the postprocess pattern (`_run_burn7`/`_run_cdo`-style subprocess wrappers not needed — pure Python).

### `adaptor.py`

Core variables and constants:

```python
FREEZING_SEAWATER_K = 271.35
SIC_THRESHOLD       = 0.5        # sic > this → ice-covered
SOLAR_CONSTANT_W_M2 = 1367.0     # default PlaSim; override via --solar-constant
# No LAND_THRESHOLD constant: we use the strict locked convention (lsm == 0 → ocean).
# Implemented as `ocean = (ds["lsm"] < 1e-6)` for float-safe equality to zero.
# A pre-audit step (Step 0 below) confirms PlaSim T42 lsm is binary (0 or 1);
# if fractional coastline cells appear, this is surfaced as a plan revision, not silently handled.
```

CLI (argparse, mirroring postprocess):

```
--sims INT [INT ...]             (required)
--years START END                (required)
--input-root  PATH               (required; postprocess output root)
--output-root PATH               (required; boundary output root)
--rsdt-method {arithmetic,astronomical}   default: arithmetic
--solar-constant FLOAT           default: 1367.0   (astronomical only)
--eccentricity   FLOAT           default: 0.0167   (astronomical only)
--obliquity-deg  FLOAT           default: 23.441   (astronomical only)
--task-index INT
--count-tasks
--overwrite
--dry-run
-v/--verbose
```

Core per-(sim, year) logic using xarray:

```python
ds = xr.open_dataset(src)
for req in ("ts", "lsm", "sic"):
    if req not in ds.data_vars:
        raise RuntimeError(
            f"Adaptor requires {req}. Rerun postprocess with --with-sea-ice "
            f"for sim{sim}/{year}."
        )

ocean = ds["lsm"] < 1e-6                  # strict lsm == 0 (float-safe)
icy   = ds["sic"] > SIC_THRESHOLD
sst = xr.where(
    ocean & ~icy, ds["ts"],
    xr.where(ocean & icy, FREEZING_SEAWATER_K, np.nan),
)
sst.attrs = {
    "units": "K",
    "long_name": "sea_surface_temperature",
    "standard_name": "sea_surface_temperature",
}

if opts.rsdt_method == "arithmetic":
    rsdt = ds["rst"] - ds["rsut"]
else:
    rsdt = compute_astronomical_rsdt(ds, opts)

rsdt.attrs = {
    "units": "W m-2",
    "long_name": "toa_incident_shortwave_flux",
    "standard_name": "toa_incoming_shortwave_flux",
}

# Pass sic through to the adaptor output so the full emulator varying-boundary
# tuple (sst, rsdt, sic) lives in one file. Clip to [0, 1] defensively; PlaSim
# can emit sic marginally outside that range at the edges of sea-ice cells.
sic_out = ds["sic"].clip(min=0.0, max=1.0)
sic_out.attrs = {
    "units": "1",
    "long_name": "sea_ice_area_fraction",
    "standard_name": "sea_ice_area_fraction",
}

out = xr.Dataset({"sst": sst, "rsdt": rsdt, "sic": sic_out})
enc = {
    "sst":  {"dtype": "float32", "_FillValue": np.float32("nan")},
    "rsdt": {"dtype": "float32", "_FillValue": np.float32("nan")},
    "sic":  {"dtype": "float32", "_FillValue": np.float32("nan")},
}
out.to_netcdf(dst, encoding=enc)
```

The emulator varying-boundary tuple in the SFNO/NVIDIA contract is `(sst, rsdt, sic)`. Emitting all three from the adaptor (rather than leaving `sic` in the postprocess NetCDF) keeps the "emulator varying-boundary state lives in one place" invariant and removes the split-file trap Codex flagged.

Astronomical rsdt — pure numpy (no `pvlib`). **6-hour mean over each output window**, not instantaneous at the timestamp, so it is semantically aligned with the arithmetic path (`rst − rsut` is PlaSim's 6h-mean TOA shortwave accounting). Codex flagged that instantaneous sampling could match the annual global mean while still giving wrong diurnal/local fields — making the two rsdt paths non-interchangeable per cell and per time. 6h-mean integration fixes this.

Formula (per 6 h window [t0, t0+Δt], Δt = 6 h, at each grid cell of (lat, lon)):

```
doy         = day_of_year(t0)                               # constant within the 6h window
dec         = obliquity * sin(2π (doy-80)/365.25)
dist_factor = 1 + eccentricity * cos(2π (doy-4)/365.25)     # quasi-constant within 6h

# Hour angles at window endpoints, in radians
h1 = (utc_hour(t0)       - 12) * π/12 + lon_rad
h2 = (utc_hour(t0 + Δt)  - 12) * π/12 + lon_rad

# Sunrise/sunset hour angles (where cos_zen = 0):
cos_h0 = −tan(lat) * tan(dec)          # clipped: if > 1 → polar night (rsdt=0); if < −1 → polar day (no sunrise/sunset in window)
h0     = arccos(clip(cos_h0, −1, 1))

# Clip the integration bounds to the daylit portion:
h1_lit = clip(h1, −h0, +h0)
h2_lit = clip(h2, −h0, +h0)
# If h1_lit == h2_lit, the window is entirely within night → integrand = 0.

# Analytic integral of max(0, sin(lat)sin(dec) + cos(lat)cos(dec)cos(h)) over [h1_lit, h2_lit]:
integ = sin(lat)*sin(dec) * (h2_lit − h1_lit)
      + cos(lat)*cos(dec) * (sin(h2_lit) − sin(h1_lit))

rsdt_6h_mean = solar_constant * dist_factor**2 * integ / Δh     # Δh = 6h in radians
```

Handle the three edge cases explicitly: polar night (cos_h0 > 1 → `rsdt = 0`), polar day (cos_h0 < −1 → `rsdt = solar_constant · dist_factor² · (sin(lat)sin(dec) + cos(lat)cos(dec)·(sin(h2)−sin(h1))/(h2−h1))`), and normal day (clip bounds as above). Full vectorisation over (time, lat, lon) is straightforward in numpy.

**Validation on write** (stronger than global-mean alone — Codex pointed out that matching the annual global mean is necessary but not sufficient):
- **Per-cell, per-timestep agreement — calibration then lock (Codex item 3).** Compute `|rsdt_arithmetic − rsdt_astronomical|` at every grid cell and every 6h step. The first audit run against sim30/MOST.0012 is a **calibration step**: it reports the max absolute difference, the 99.9th percentile, and the zonal-mean-difference profile — no hard-fail. Those numbers are then recorded in `docs/emulator_adaptor_audit.md` as the locked acceptance bound (e.g. "max abs diff over sim30/MOST.0012 = X W m⁻²; subsequent audits on other sims must stay within max(X, 20) W m⁻²"). The 20 W m⁻² floor exists because PlaSim's arithmetic path carries radiation-scheme accounting noise that is independent of sample size — a tighter bound would be overfitting to the pilot sim. Commit 4 locks the bound in the audit doc; all later adaptor runs hard-fail if the bound is exceeded.
- **Area-weighted annual-mean (soft check).** Area-weighted global annual mean rsdt must match `solar_constant / 4` within 0.5 % for astronomical and within 1 % for arithmetic (arithmetic is looser because it carries PlaSim radiation-scheme accounting noise).
- **Zonal-mean structure (soft check).** Zonal annual mean of astronomical rsdt should be monotone decreasing from equator to each pole (simple sanity against declination bias).
- **sst range (hard check).** Over ocean cells: `sst.min() >= FREEZING_SEAWATER_K − 0.01`, `sst.max() <= 310`.
- **Mask application (hard check).** NaN fraction of `sst` equals global land fraction from `lsm` to within one grid cell.

### `submit.slurm`

Mirror `src/plasim_postprocessor/submit.slurm`. Same module-load block (no LD_LIBRARY_PATH required — no burn7). Adds optional `RSDT_METHOD=` env variable; if set, the dispatch line appends `--rsdt-method "$RSDT_METHOD"`.

### Python dependencies

New stack: `xarray`, `numpy`, `netCDF4`. None currently used by the postprocessor. Document in `adaptor.py` module docstring; rely on Stampede3's `python/3.12.11` site packages (xarray ships there by default for the scientific modules list).

## Part D — docs + audit (Commit 4)

### Step 0 — convention probes (Codex item 4, referenced throughout plan)

Before Commit 3's adaptor code is audited, run two narrow probes on the Commit 2 postprocess output for sim30/MOST.0012 (with `--with-sea-ice`). These confirm the two convention assumptions baked into the sst rule. Probe outputs go into `docs/plasim_postprocessor_audit.md` as a new "Step 0 — convention probes" subsection.

**Probe 0.A — `lsm` value distribution.** Confirms the strict `ocean = (lsm < 1e-6)` test in the adaptor matches PlaSim T42 behavior.

```bash
POSTPROC_NC=/tmp/audit_expanded/sim30/MOST.0012.nc

# Distinct values of lsm (over a single timestep — lsm is static):
cdo -s output -seltimestep,1 -selname,lsm $POSTPROC_NC | tr -s ' \n' '\n' | sort -u > /tmp/lsm_unique.txt
head -20 /tmp/lsm_unique.txt

# Count how many cells are strictly 0, strictly 1, or fractional:
python3 -c "
import xarray as xr, numpy as np
lsm = xr.open_dataset('$POSTPROC_NC')['lsm'].isel(time=0).values
print(f'cells == 0:             {np.sum(lsm == 0)}')
print(f'cells == 1:             {np.sum(lsm == 1)}')
print(f'cells in (0, 1) strict: {np.sum((lsm > 0) & (lsm < 1))}')
print(f'min, max:               {lsm.min():.6f}, {lsm.max():.6f}')
"
```

Pass: fractional count is 0, or small enough (<1 % of total cells) that the strict `lsm == 0` test captures essentially all open-ocean cells. If fractional count is a meaningful fraction (say >5 %), surface this as a plan revision — the sst convention needs rediscussion before Commit 3.

**Probe 0.B — `sic` value distribution.** Confirms the `sic > 0.5` threshold is in a sensible regime for PlaSim T42. If most nonzero sic cells are near 1 (binary-like), threshold choice barely matters. If many are in (0, 0.5), the choice is load-bearing and needs re-confirmation.

```bash
python3 -c "
import xarray as xr, numpy as np
sic = xr.open_dataset('$POSTPROC_NC')['sic'].values
nonzero = sic[sic > 0]
print(f'total cells:              {sic.size}')
print(f'sic == 0:                 {np.sum(sic == 0)}')
print(f'sic in (0, 0.15):         {np.sum((sic > 0) & (sic < 0.15))}')
print(f'sic in [0.15, 0.5):       {np.sum((sic >= 0.15) & (sic < 0.5))}')
print(f'sic in [0.5, 0.85):       {np.sum((sic >= 0.5) & (sic < 0.85))}')
print(f'sic in [0.85, 1]:         {np.sum(sic >= 0.85)}')
print(f'nonzero mean, median:     {nonzero.mean():.3f}, {np.median(nonzero):.3f}')
"
```

Pass: either (a) the `(0, 0.5)` bucket is small compared to the `≥ 0.5` bucket (threshold choice doesn't flip many cells), or (b) the nonzero distribution is bimodal with a clear valley near 0.5. If neither is true — if the distribution is broad and smooth across (0, 1) — surface as a plan revision; threshold needs re-confirmation.

Both probes run in Commit 4's audit sequence before any adaptor validation numbers are locked.

### Doc edits

1. **Update `docs/plasim_postprocessor_expand_plan.md`**: add a "SUPERSEDED 2026-04-21" header block noting that `pl` is kept (reverses the v1 decision), zg is dual-output (sigma + `zg_plev`), and the adaptor is now a separate module. Preserve v1 text beneath for history.

2. **Rewrite the "Locked variable set" section of `docs/plasim_postprocessor_audit.md`**: method (sim30/MOST.0012 with patched binary, module set from existing doc), expected manifest listing all 29 variables (alphabetic + `sic` conditional), per-variable sanity sub-table for the three new/changed fields:
   - `zg` on sigma (midpoints): near-surface level mean 100–600 m; near-top level mean 20000–24000 m (see Part A audit note for the corrected bound).
   - `zg_plev` at 500 hPa: 5400–5700 m (carry-forward from prior audit).
   - `pl`: global-mean ≈ ln(985/1000) ≈ −0.015, range ≈ [−0.05, +0.01].
   - burn7 sigma-zg vs pressure-zg cross-validation note (RMSE <5 m mid-troposphere).

3. **New `docs/audit_snapshots/manifest.txt`**: locked snapshot of the expanded contract (default run, no sea ice). Variables (alphabetic): `clt, hfls, hfss, hus, lat, lev, lev_2, lon, lsm, mrso, pl, pr, pr_6h, ps, psl, rls, rlut, rss, rst, rsut, sg, ta, tas, td2m, time, ts, ua, va, z0, zg, zg_plev`. Keep existing per-profile snapshots (`exp15/25/26/aires_rad`) in place until the user signs off on deletion.

4. **New `docs/audit_snapshots/manifest_with_sea_ice.txt`**: snapshot with `sic`. Needed as the adaptor's upstream input (adaptor requires `--with-sea-ice`).

5. **New `docs/emulator_adaptor_audit.md`**:
   - Method: run postprocess on sim30/MOST.0012 with `--with-sea-ice`, then adaptor with default rsdt, then separately with `--rsdt-method astronomical` for the cross-check run.
   - sst sanity: ocean-cell fldmean 285–290 K; min ≥ 271.35; max ≤ 310; NaN-fraction ≈ global land fraction (to within one grid cell).
   - sic sanity (pass-through): post-clip min ≥ 0; max ≤ 1; global-mean fldmean-timmean matches the postprocess input's sic within 1e-6.
   - rsdt sanity (arithmetic): area-weighted fldmean-timmean within ±1 % of `solar_constant/4` = 341.75 W m⁻² (for default `solar_constant = 1367` W m⁻²); zonal pattern monotone from equator.
   - rsdt sanity (astronomical): area-weighted global-mean within ±0.5 % of `solar_constant/4` = 341.75 W m⁻²; zonal pattern monotone from equator; **per-cell-per-time max abs difference vs arithmetic ≤ locked bound `max(measured_on_sim30, 20)` W m⁻²** (Codex item 3, calibrated during this same audit run and recorded here; the initial pilot run measures the bound and Commit 4's version of this doc records the final number).
   - Known downstream consumers to update post-rename (Codex item 3): SFNO config files listed under "Breaking contract changes" above.

## Commit structure

Four commits, each standalone-testable:

- **Commit 1**: `burn7.cpp` edits + rebuilt `Stampede3/burn7`. Old postprocessor still works against the new binary (superset of old behaviors).
- **Commit 2**: `plasim_postprocessor.py` SIGMA_CODES/process_one changes. Tested against Commit 1's binary.
- **Commit 3**: `src/emulator_adaptor/` module + slurm template. Tested against Commit 2's postprocess output.
- **Commit 4**: Docs rewrite + new audit snapshots. Tested by running the full verification recipe below.

## Verification

End-to-end on sim30/MOST.0012:

```bash
module purge
module load intel/24.0 cdo netcdf python/3.12.11
export LD_LIBRARY_PATH=/opt/apps/intel24/netcdf/4.9.2/x86_64/lib:/home1/09979/awikner/netcdf-4.2/lib:$LD_LIBRARY_PATH

# 1. Postprocess
python3 src/plasim_postprocessor/plasim_postprocessor.py \
    --sims 30 --years 12 12 --with-sea-ice \
    --input-root /scratch/10000/amarchakitus/PLASIM/data \
    --output-root /tmp/audit_expanded

# 2. Manifest round-trip (must be empty)
manifest /tmp/audit_expanded/sim30/MOST.0012.nc > /tmp/new_manifest.txt
diff /tmp/new_manifest.txt docs/audit_snapshots/manifest_with_sea_ice.txt

# 3. New-variable sanity
cdo -s output -fldmean -timmean -sellevel,500 -selname,zg_plev /tmp/audit_expanded/sim30/MOST.0012.nc   # expect 5400–5700 m
cdo -s output -fldmean -timmean -sellev,1       -selname,zg      /tmp/audit_expanded/sim30/MOST.0012.nc   # near-top; expect 20000–24000 m
cdo -s output -fldmean -timmean -sellev,10      -selname,zg      /tmp/audit_expanded/sim30/MOST.0012.nc   # near-surface; expect 100–600 m
cdo -s output -fldmean -timmean -selname,pl     /tmp/audit_expanded/sim30/MOST.0012.nc                    # expect ≈ −0.015

# 4. Adaptor
python3 src/emulator_adaptor/adaptor.py \
    --sims 30 --years 12 12 \
    --input-root /tmp/audit_expanded \
    --output-root /tmp/audit_adaptor

# 5. Adaptor sanity
cdo -s output -fldmean -timmean -selname,rsdt /tmp/audit_adaptor/sim30/boundary.0012.nc   # expect 341.75 W m⁻² ± 1 % (= solar_constant/4 for default S0=1367)
cdo -s output -fldmean -timmean -selname,sst  /tmp/audit_adaptor/sim30/boundary.0012.nc   # expect 285–290 K (NaN-skipping)
cdo -s output -fldmean -timmean -selname,sic  /tmp/audit_adaptor/sim30/boundary.0012.nc   # expect ~0.05 (polar only); min=0, max≤1

# 6. Astronomical rsdt cross-check (not optional — required by Codex item 4)
python3 src/emulator_adaptor/adaptor.py \
    --sims 30 --years 12 12 \
    --input-root /tmp/audit_expanded \
    --output-root /tmp/audit_adaptor_astro \
    --rsdt-method astronomical
cdo -s output -fldmean -timmean -selname,rsdt /tmp/audit_adaptor_astro/sim30/boundary.0012.nc
# Annual area-weighted global mean must match arithmetic within 0.5 %.

# 7. Per-cell, per-time rsdt agreement — Commit 4 CALIBRATION run (Codex item 3).
#    This pilot measures the max-abs-diff statistic on sim30/MOST.0012; the locked
#    acceptance bound recorded in docs/emulator_adaptor_audit.md is max(measured, 20) W m⁻².
#    Subsequent audit runs on other sims hard-fail if they exceed the recorded bound.
cdo -s -sub /tmp/audit_adaptor/sim30/boundary.0012.nc \
             /tmp/audit_adaptor_astro/sim30/boundary.0012.nc \
             /tmp/rsdt_diff.nc
cdo -s output -fldmax -timmax -abs -selname,rsdt /tmp/rsdt_diff.nc   # record for the audit doc
```

## Deferred / implementation-time lookups (not decisions)

- Stampede3 compiler driver for rebuild — `file` the existing binary before picking `icpx` vs `g++`.
- Freezing-point constant 271.35 K — cross-check PlaSim sea-ice module if that source is accessible; not user-blocking.
- PlaSim T42 `lsm` values — confirm binary (0/1) in the Step 0 audit probe. If fractional-coastline cells exist, surface as a plan revision (Codex item 2) rather than switching silently to a threshold.
- Per-cell rsdt arithmetic-vs-astronomical bound — Commit 4's audit run records the measured max-abs-diff on sim30/MOST.0012 and locks the bound as `max(measured, 20)` W m⁻². This is a one-time calibration, not a deferred decision.

## Plan-document housekeeping

After ExitPlanMode, copy this plan to `docs/plasim_expansion_and_adaptor_plan.md` per the project's "save plans to docs/ for external review" convention.
