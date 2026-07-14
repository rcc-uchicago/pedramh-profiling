---
name: plasim-postprocess
description: Use when the user wants to convert raw PlaSim binary output files (MOST.NNNN, one per simulated year) into per-sim-year NetCDF files using the audited burn7-based post-processor at src/plasim_postprocessor/. Covers four audited profiles (aires_rad, exp15, exp25, exp26), the Stampede3 module-load contract, SLURM array dispatch, and how to verify changes against locked audit snapshots. Invoke for any task involving plasim_postprocessor.py, the burn7 binary at src/plasim_postprocessor/burn7/Stampede3/burn7, or any change to the variable set / pressure levels / precip accumulation / radiation outputs.
---

# PlaSim post-processor — usage and contract

The single-purpose post-processor lives at `src/plasim_postprocessor/plasim_postprocessor.py`. It converts raw PlaSim binary output (one `MOST.NNNN` file per simulated year) into per-sim-year NetCDF files at native 6-hourly cadence. The variable set is selected by `--profile` and is **frozen** by audit snapshots in `docs/audit_snapshots/`.

## When to use this skill

- Running post-processing on `MOST.NNNN` files (single-shot or SLURM array).
- Sizing a SLURM array job for a given (sims × years) request.
- Modifying the script (always re-verify against the locked audit snapshots after any change).
- Adding a new profile (requires a fresh audit run + new snapshot file).
- Debugging burn7/CDO failures, missing modules, or `LD_LIBRARY_PATH` issues on Stampede3.

## Profiles (locked by audit snapshots)

Edit-then-verify rule: profile definitions in `PROFILES` must always match `docs/audit_snapshots/{profile}_manifest.txt`. If you change a profile, regenerate the snapshot from a fresh audit run on the existing pipeline and update the snapshot file in the same commit.

| Profile | Sigma + radiation/flux vars | Land vars | zg pressure levels (hPa) | Precip |
|---------|------------------------------|-----------|---------------------------|--------|
| `exp15` | ta, ua, va, hus, pl, tas | — | 500 | — |
| `exp25` | ta, ua, va, hus, pl, tas | — | 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 | pr_6h |
| `exp26` | ta, ua, va, hus, pl, tas | mrso | 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 | pr_6h |
| `aires_rad` | ta, ua, va, hus, pl, tas, **rss, rls, rst, rlut, rsut, hfss, hfls** | mrso | 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 | pr_6h |

Radiation and surface heat-flux variables (`rss, rls, rst, rlut, rsut, hfss, hfls`) are surface 2D fields produced via the sigma namelist — they come out as `float <name>(time, lat, lon)`, the same shape as `tas` and `pl`. burn7 emits 2D vs 3D based on its own internal code metadata, not the `vtype` flag.

**PlaSim radiation sign convention** (validated empirically — see `docs/plasim_postprocessor_audit.md`): **positive = into receiver, negative = leaving receiver**. So `rlut` (TOA net LW) is negative when there's outgoing longwave; OLR magnitude = `-rlut`. Same for `rsut`, `hfss`, `hfls` — all negative-mean for typical climate. `rss` (surface net SW) and `rst` (TOA net SW) are positive (atmosphere absorbs SW from above).

Output dim names: `lev` (10 sigma levels, identical across profiles) and `lev_2` (1 entry for `exp15`, 13 for `exp25`/`exp26`/`aires_rad` — the zg pressure levels).

`pr` (instantaneous, from burn7) is retained alongside `pr_6h` (CDO `runsum,6` then `chname,pr,pr_6h`) in `exp25`/`exp26`/`aires_rad` outputs. Those three profiles have 5 fewer timesteps than `exp15` because `runsum,6` requires a 6-step warm-up window.

## Stampede3 module-load contract

Always load this exact set before invoking the script (interactive or via SLURM):

```bash
module purge
module load intel/24.0 cdo netcdf python/3.12.11
export LD_LIBRARY_PATH=/opt/apps/intel24/netcdf/4.9.2/x86_64/lib:/home1/09979/awikner/netcdf-4.2/lib:${LD_LIBRARY_PATH}
```

Why each piece:
- `intel/24.0` — toolchain that `cdo/2.3.0` requires.
- `cdo` — used for merging burn7 outputs and the `runsum,6` precip accumulation.
- `netcdf` — burn7 reads/writes NetCDF.
- `python/3.12.11` — script uses `str | None` union syntax (Python 3.10+).
- `LD_LIBRARY_PATH` — burn7 dynamically links against `libnetcdf_c++.so.4`, which lives in `/home1/09979/awikner/netcdf-4.2/lib` (the system `netcdf/4.9.2` doesn't ship the legacy C++ bindings).

Skipping any of these will fail with library-load or "module not found" errors.

## CLI

```
plasim_postprocessor.py
  --profile           {aires_rad,exp15,exp25,exp26}    selects audited variable set
  --sims              INT [INT ...]           e.g. 30 31 32 33 34 35 36
  --years             START END               inclusive, e.g. 1 100

  --input-root        PATH    required for processing; root containing sim{NN}/MOST.{YYYY:04d}
  --output-root       PATH    required for processing; root for {output-root}/sim{NN}/MOST.{YYYY:04d}.nc

  --burn7-binary      PATH    default: <script_dir>/burn7/Stampede3/burn7
  --task-index        INT     run only the Nth (sim, year) pair (for SLURM array dispatch)
  --count-tasks               print number of (sim, year) pairs and exit
                              (does not require --input-root/--output-root)
  --overwrite                 force re-write of existing output files
  --dry-run                   print actions without executing
  -v, --verbose
```

Conditional-required: `--input-root` and `--output-root` are needed for any processing run, but **omitted with `--count-tasks`** (count mode only enumerates tasks).

## Recipes

### 1. Single-shot interactive run (one sim-year)

```bash
cd ~/projects/SFNO_Climate_Emulator
module purge && module load intel/24.0 cdo netcdf python/3.12.11
export LD_LIBRARY_PATH=/opt/apps/intel24/netcdf/4.9.2/x86_64/lib:/home1/09979/awikner/netcdf-4.2/lib:$LD_LIBRARY_PATH

python3 src/plasim_postprocessor/plasim_postprocessor.py \
    --profile exp15 --sims 30 --years 12 12 \
    --input-root /scratch/10000/amarchakitus/PLASIM/data \
    --output-root $SCRATCH/SFNO_Climate_Emulator/data/postproc
```

Output: `$SCRATCH/SFNO_Climate_Emulator/data/postproc/sim30/MOST.0012.nc`.

### 2. SLURM array job (recommended for any non-trivial run)

The template at `src/plasim_postprocessor/submit.slurm` has placeholders for `PROFILE`, `SIMS`, `YEAR_START`, `YEAR_END`, `INPUT_ROOT`, `OUTPUT_ROOT`. Edit them, then:

```bash
cd ~/projects/SFNO_Climate_Emulator

# Step 1: compute task count (no --input/output-root needed)
N=$(python3 src/plasim_postprocessor/plasim_postprocessor.py \
        --profile exp15 --sims 30 31 32 33 34 35 36 --years 1 100 --count-tasks)
echo "Will dispatch $N tasks"

# Step 2: submit with the right array size
sbatch --array=0-$((N-1)) src/plasim_postprocessor/submit.slurm
```

Each array task processes exactly one `(sim, year)` pair. Failures retry per-task (don't poison the whole job). The default `--array=0-0` in submit.slurm is a single-task smoke test — always resize before launching the real run.

### 3. Re-run only failed tasks

The script is idempotent: re-running over an existing output is a skip-with-log. To force re-processing of specific tasks, find their `--task-index` values and use `--overwrite`:

```bash
python3 src/plasim_postprocessor/plasim_postprocessor.py \
    --profile exp25 --sims 30 31 32 33 34 35 36 --years 1 100 \
    --task-index 42 --overwrite \
    --input-root ... --output-root ...
```

## Verification — required after any change to the script

If you touch `plasim_postprocessor.py`, namelist construction, the `PROFILES` dict, or the burn7 invocation, re-verify all four profiles against the locked snapshots **before declaring the change done**. Use the same `manifest()` helper the audit used:

```bash
manifest() {
    local nc="$1"
    echo "=== dimensions ==="
    ncdump -h "$nc" | awk '/^dimensions:/{flag=1; next} /^variables:/{flag=0} flag' \
        | sed 's/^[[:space:]]*//' | sort
    echo "=== variables ==="
    ncdump -h "$nc" | awk '/^variables:/{flag=1; next} /^\/\/ global attributes:|^data:/{flag=0} flag' \
        | grep -E "^[[:space:]]*(float|double|int|short|char) " | sort
    echo "=== lev (sigma) coordinate values ==="
    ncdump -v lev "$nc" 2>/dev/null | awk '/^ lev =/,/;/' | tr -d '\n ' | sed 's/;.*//'
    echo
    echo "=== lev_2 (pressure) coordinate values ==="
    ncdump -v lev_2 "$nc" 2>/dev/null | awk '/^ lev_2 =/,/;/' | tr -d '\n ' | sed 's/;.*//'
    echo
    echo "=== lat/lon shape (sanity) ==="
    ncdump -h "$nc" | grep -E "^[[:space:]]+(lat|lon) =" | sort
}

for prof in aires_rad exp15 exp25 exp26; do
    out=/tmp/verify_${prof}/sim30/MOST.0012.nc
    rm -f "$out"
    python3 src/plasim_postprocessor/plasim_postprocessor.py \
        --profile $prof --sims 30 --years 12 12 \
        --input-root /scratch/10000/amarchakitus/PLASIM/data \
        --output-root /tmp/verify_${prof}
    manifest "$out" > /tmp/${prof}_check.txt
    diff /tmp/${prof}_check.txt docs/audit_snapshots/${prof}_manifest.txt \
        && echo "$prof: OK" || echo "$prof: DRIFT"
done
```

All four diffs must be empty. If any prints `DRIFT`, either the change broke the contract (revert or fix) or the contract genuinely needs to update (then regenerate the snapshot from the audit run on the *previous* version of the code, document the drift in `docs/plasim_postprocessor_audit.md`, and get user sign-off).

## Adding a new profile (provisional → audited workflow)

A new profile lands provisional first (no snapshot yet) and is flipped to audited once the snapshot is locked. Two phases:

**Phase 1 — Provisional ship:**

1. Define the new entry in `PROFILES` (sigma_vars, land_vars, pressure_levels, accumulate_precip_hours).
2. Add the profile name to the module-level `PROVISIONAL_PROFILES = {…}` set in `plasim_postprocessor.py`. The CLI emits a stderr warning whenever `--profile <name>` is selected for a name in this set.
3. argparse `--profile choices` come from `sorted(PROFILES.keys())` — auto-picked-up; nothing to add.
4. Add the new profile to the **Provisional profiles** table in this skill, plus a "Provisional addendum" section in `docs/plasim_postprocessor_audit.md` documenting the variable list and a TODO list of remaining audit steps.
5. Update the docstring's "Provisional" block in `plasim_postprocessor.py` and the argparse description / `--profile` help text to mention the new provisional profile.

**Phase 2 — Lock snapshot, flip to audited:**

6. Pick a reference MOST file with the variables present (use env vars `REF_INPUT_ROOT` / `REF_SIM` / `REF_YEAR` rather than hardcoding).
7. Run a probe (`burn7` directly with the new vars) to confirm the source data has them. Use `cdo infon` for per-var min/max/missing rather than grep heuristics.
8. For radiation/flux variables: run a semantic-validation step (`cdo fldmean -timmean`) and compare global means against textbook climate ranges before locking. Document sign conventions in the audit doc.
9. Run the new script with the new profile on the reference file; manifest the output to `docs/audit_snapshots/{newprofile}_manifest.txt`. This becomes the locked contract.
10. Round-trip verify: re-run, manifest, diff against snapshot — must be empty.
11. **Atomic flip commit**: remove the profile from `PROVISIONAL_PROFILES`, merge the docstring's Provisional block into the Audited block, drop "provisional" wording from argparse description / --profile help / SKILL.md (move the table row from Provisional to Audited; remove the provisional callout entry for that profile), replace the audit-doc TODO with the Phase 2 findings.
12. Re-verify all (now-audited) profiles per the verification section above.

## What NOT to do (deliberate exclusions)

These were removed in the v4 refactor and should **not** be reintroduced without explicit user direction. Each was deliberated and resolved against:

- **YAML configs / wrapper scripts**: the script is intentionally CLI-only with internal `PROFILES`. Adding YAML/wrappers re-creates the multi-experiment scaffolding the refactor eliminated.
- **NCL geopotential path**: the legacy `compute_zg.ncl`/`interpolate_data.ncl` had a documented −20 to −30 m bias relative to burn7 `VTYPE=P`. The scripts are not in this repo.
- **Free-form CLI flags** (`--sigma-vars`, `--pressure-levels`, `--accumulate-precip-hours`): rejected during plan review because they make verification open-ended. Profiles only.
- **Pangu regridding / second output tree**: out of scope; the contract is one NetCDF per `MOST.NNNN`. If Pangu inputs are ever needed, build a separate downstream tool that reads these outputs.
- **Temporal aggregation in the post-processor** (daymean, monthly means, etc.): the post-processor is lossless. Do aggregation downstream in xarray/CDO when needed for analysis.

## Where to read more

- **Plan**: `docs/plasim_postprocessor_refactor_plan.md` — full design rationale and revision history (v1 → v4).
- **Audit**: `docs/plasim_postprocessor_audit.md` — empirical-vs-predicted reconciliation, findings (e.g. `lev` vs `lev_2`, `runsum,6` timestep trim).
- **Locked snapshots**: `docs/audit_snapshots/{aires_rad,exp15,exp25,exp26}_manifest.txt` — the verification target.
- **`aires_rad` plan**: `docs/aires_rad_profile_plan.md` — full plan for the radiation+heat-flux profile, the empirical sign-convention findings, and the documented provisional → audited flip workflow (reusable for future profiles).
- **Script**: `src/plasim_postprocessor/plasim_postprocessor.py` — module docstring summarizes the toolchain split (burn7 for variables, CDO for plumbing).
- **SLURM template**: `src/plasim_postprocessor/submit.slurm`.

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `error while loading shared libraries: libnetcdf_c++.so.4` | Missing `LD_LIBRARY_PATH` entry for `/home1/09979/awikner/netcdf-4.2/lib` | Re-export the full `LD_LIBRARY_PATH` from the module-load contract above |
| `module(s) ... cannot be loaded as requested: "cdo"` | `intel/24.0` not loaded first | `module purge && module load intel/24.0 cdo netcdf python/3.12.11` |
| `TypeError: unsupported operand type(s) for |: 'type'` | System `python3` (3.9.x) instead of `python/3.12.11` module | Load `python/3.12.11`; do not use `/usr/bin/python3` |
| Output diff shows different `lev_2` values | Wrong profile selected, or `PROFILES` was edited without snapshot update | Re-run with the right `--profile`; if profile def changed, follow "Adding a new profile" workflow |
| `--task-index N out of range` | Mismatch between `--count-tasks` value and `--array` size | Re-run `--count-tasks` and resize the SLURM array |
| All tasks log "skipping (input missing: ...)" | Wrong `--input-root` path | Verify `ls $INPUT_ROOT/sim{NN}/MOST.{YYYY:04d}` exists |
