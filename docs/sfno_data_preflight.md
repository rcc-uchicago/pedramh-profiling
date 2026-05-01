# SFNO Data Preflight — Phase 0 Gate Artifact

> - **v1 (2026-04-22):** initial pass. Executed Phase 0 of `docs/sfno_training_extraction_plan.md` against currently-staged AI-RES data (year 0012 of sim30 from the `emulator_adaptor` audit).

## Scope

This is the artifact produced by **Phase 0** of the SFNO training extraction plan. It records the state of the three data preflight gates and the dataset-scoped layout adopted for this port.

All paths are dataset-scoped under `<sim_tag> = sim30_astro` (PlaSim run "sim30", astronomical-`rsdt` boundary variant). The `_astro` suffix is part of the tag because the audit emits both arithmetic- and astronomical-rsdt outputs and which one trains the emulator is a scientific choice that should be explicit in the path.

## Chosen `<sim_tag>/<profile>`

- `<sim_tag>`: `sim30_astro`
- Source postprocess output: `/scratch/11114/zhixingliu/AI-RES/audit/commit4/postproc/sim30/MOST.{YYYY:04d}.nc`
- Source boundary output: `/scratch/11114/zhixingliu/AI-RES/audit/commit4/boundary_astro/sim30/boundary.{YYYY:04d}.nc`
- Dataset-scoped root (target): `/scratch/11114/zhixingliu/AI-RES/data/plasim_postprocess/sim30_astro/`

Only year **0012** is currently staged. The source SFNO config uses years 12–112 for training and 11–12 for validation. **The AI-RES staging is nowhere near that range yet.** Phase 0 records this rather than blocking on it; full-scale training data staging is a separate project workstream.

## Gate 0a — Atmospheric contract in the postprocess NetCDF

**Status:** PASS (with one caveat about constant-boundary handling).

Inspected: `/scratch/11114/zhixingliu/AI-RES/audit/commit4/postproc/sim30/MOST.0012.nc` — 1459 timesteps at 6h cadence on a 64×128 grid.

| Contract group | Required | Present in MOST.0012.nc |
|---|---|---|
| Upper-air (sigma) | `ta, ua, va, hus, zg` on 10 sigma levels | All five present on `lev=10` ✓ |
| Surface | `pl, tas` | Both present ✓ |
| Constant boundary | `lsm, sg, z0` | All three present ✓ (see caveat) |
| Varying boundary | `sst, rsdt, sic` | Not in this file by design — live in the boundary bundle (Gate 0b) |
| Diagnostic | `pr_6h` | Present ✓ |

**Caveat:** `lsm, sg, z0` are stored with a `time` dim (shape `[1459, 64, 128]`) even though they're static by nature. The Pass 2 dataloader must read "take the first timestep, broadcast across time" to build the `constant_boundary` tensor, and assert static-across-time (optionally under a verbose flag).

**Note on `docs/weather_emulator_io_postprocessor_check.md`:** that doc (pre-April audit) lists `pl` as missing from the postprocessor and `zg` as pressure-level-only. Both have since been fixed — the current `plasim_postprocessor.py` emits `pl` on the surface and `zg` on 10 sigma levels (with `zg_plev` retained on 13 pressure levels as a separate variable). That doc should be updated with a revision block when convenient; for the purposes of this port, the contract is satisfied.

## Gate 0b — Per-variable boundary files

**Status:** PASS (script shipped; year-12 demo output generated).

Shipped: `scripts/build_boundary_dir.py` (standalone; *not* a new mode on `src/emulator_adaptor/adaptor.py`, per the audited-contract rationale in the extraction plan). Reshapes one-bundled-per-(sim,year) adaptor output into the per-variable layout the SFNO dataloader expects.

Demo run — processed the single staged year (0012):

```
.venv/bin/python scripts/build_boundary_dir.py \
  --sims 30 --years 12 12 \
  --input-root /scratch/11114/zhixingliu/AI-RES/audit/commit4/boundary_astro \
  --output-dir /scratch/11114/zhixingliu/AI-RES/data/plasim_postprocess/sim30_astro/boundaries
```

Output:
```
sim30_astro/boundaries/
├── sst_masked_6h.nc     (1459 timesteps, 64×128, float32)
├── rsdt_masked_6h.nc
└── sic_masked_6h.nc
```

No `_leap.nc` variants were produced — the single staged year (0012) has a uniform 1459-timestep count (proleptic Gregorian calendar, Jan 1 00:00 → Dec 30 12:00). The script groups by timestep-count per year and emits `_leap.nc` only when a second length group appears. When more years come online, rerun the script over the full year range and `_leap.nc` files will appear automatically if any PlaSim years produce a different timestep count.

Provenance attrs from the adaptor (`rsdt_method=astronomical`, solar constant, obliquity, etc.) are preserved on the per-variable output files and tagged with `reshape_source=scripts/build_boundary_dir.py`.

## Gate 0c — Normalization stats copy

**Status:** PARTIAL PASS — fallback stats copied; two documented divergences from the source's exact stats file.

The source SFNO config (`SFNO_PLASIM_H5_DERECHO_5310.yaml`) references `data_12-132_mean_sigma.nc` and `data_12-132_std_sigma.nc`, which live on Derecho at `/glade/derecho/scratch/marchakitus/PLASIM/data/sim52/capped_10mm/h5/sigma_data/`. Those files are not accessible from Stampede3.

A near-match pair exists on Stampede3 and was copied into the dataset-scoped `norm/` dir:

```
/scratch/11114/zhixingliu/AI-RES/data/plasim_postprocess/sim30_astro/norm/
├── data_12-111_sigma_mean.nc    (from awikner@stampede3/PlaSim-emulator-diagnosis/Pangu-PlaSim-postprocessor/)
└── data_12-111_sigma_std.nc
```

These cover all four variable groups (surface, constant_boundary, varying_boundary, upper_air, plus diagnostic `pr_6h`). Both files are used as a smoke-test-only fallback with the following caveats flagged for Pass 2:

1. **Year range divergence.** Stats were computed over years 12–111; the source's `data_12-132_*` is years 12–132. For smoke-test purposes the distributions should be close enough; for parity with source-trained checkpoints, we'll need either a Derecho transfer (Globus / scp) or a fresh recomputation from the AI-RES data once more years are staged.
2. **`zg` stats are on the wrong vertical coord.** The Stampede3 fallback has `zg: dims=('Z',), shape=(13,)` — i.e. 13 pressure levels — but the staged `MOST.0012.nc` emits `zg` on **10 sigma levels** (shape `[1459, 10, 64, 128]`). Applying these 13-level stats to the 10-level data is a type error. For Phase 5 smoke we'll need either (a) recomputed per-sigma-level `zg` stats from the AI-RES data, (b) a Derecho transfer of the correct file, or (c) a temporary smoke-test shortcut that zero-normalizes `zg` (clearly marked as non-production).
3. **Filename ordering.** Source uses `mean_sigma` / `std_sigma`; Stampede3 fallback uses `sigma_mean` / `sigma_std`. Cosmetic only — the YAML config fields just need updating to match whichever pair is actually in `norm/`.

**Pass 2 requirement:** the AIRES YAML configs must point `surface_*`, `upper_air_*`, `boundary_*`, `diagnostic_*` at the file(s) actually in `norm/`. The **normalization is still a single shared file** (contract requirement); we're just changing *which* shared file.

**Escalation path:** if zg parity matters for the first real training run (not just smoke), request a Derecho → Stampede3 Globus transfer of `data_12-132_{mean,std}_sigma.nc` from marchakitus, or recompute stats from the full year range once `sim30_astro` has 100+ years staged.

## Blockers still outstanding after Phase 0

1. **Only 1 year of data is staged.** 0012 is a single audit year. SFNO training wants 100+ years (source uses 12–112). Full-scale data staging is separate from this port.
2. **`zg` sigma-level normalization stats not available on Stampede3.** See Gate 0c note 2 — smoke test needs a workaround; full training needs the correct file.
3. **Source `data_loader_multifiles.py` still permission-restricted.** Phase 1 item — requesting `chmod -R o+r` from awikner is out of scope for Pass 1.
4. **Format is NetCDF, not H5.** The source pipeline reads H5. The Pass 2 dataloader can be written to read NetCDF directly (simpler) or an H5 converter (`utils/data/netcdf-to-h5-new.py` exists in awikner's tree, 755-readable) can be ported. Decision deferred to Pass 2.

## Dataset-scoped layout (locked by this preflight)

```
/scratch/11114/zhixingliu/AI-RES/data/plasim_postprocess/sim30_astro/
├── postproc/                  ← (future) copy of MOST.{YYYY:04d}.nc staged here, or symlinked from audit/commit4/
├── boundaries/                ← Gate 0b output (populated)
│   ├── sst_masked_6h.nc
│   ├── rsdt_masked_6h.nc
│   └── sic_masked_6h.nc
└── norm/                      ← Gate 0c output (populated with 12-111 fallback)
    ├── data_12-111_sigma_mean.nc
    └── data_12-111_sigma_std.nc
```

The Pass 2 YAML config fields:
- `data_dir: /scratch/11114/zhixingliu/AI-RES/data/plasim_postprocess/sim30_astro/postproc/`
  (or wherever the training NetCDFs end up being staged under `sim30_astro/`)
- `boundary_dir: /scratch/11114/zhixingliu/AI-RES/data/plasim_postprocess/sim30_astro/boundaries/`
- `normalization_dir: /scratch/11114/zhixingliu/AI-RES/data/plasim_postprocess/sim30_astro/norm/`
- `surface_mean: data_12-111_sigma_mean.nc` (and equivalent for the other three group × {mean,std} fields)
- `surface_std: data_12-111_sigma_std.nc`

## Verdict

Phase 0 preflight: **PASS for Gate 0a and Gate 0b; PARTIAL for Gate 0c**. The port can proceed into Pass 2 once awikner grants read access to `data_loader_multifiles.py` and `train.py`. Smoke-test-scale runs are feasible today with the shipped fallback stats (modulo the `zg` workaround); full-scale training requires the Derecho transfer and more staged years.
