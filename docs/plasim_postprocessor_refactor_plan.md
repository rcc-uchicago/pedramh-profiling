# PlaSim Post-processor Refactor — Plan (v4)

> Revision history:
> - **v1**: EXP15-only, hardcoded variable set.
> - **v2**: scope widened to EXP15+25+26 union; introduced free-form CLI flags and optional Pangu regridding.
> - **v3**: dropped Pangu and free-form flags; constrained CLI to `--profile {exp15|exp25|exp26}`; removed unrunnable NCL parity check; verification against machine-readable audit snapshots.
> - **v4**: responds to Codex round 3:
>   1. Reframes the contract as **"burn7 for variable derivation; CDO for plumbing"** (the original `pr_6h` is not burn7-derivable; it requires `cdo runsum,6`, which the original pipeline also uses). The phrase "burn7-only" referred to replacing NCL for zg, not to displacing CDO for merge/accumulation.
>   2. Strengthens the audit manifest to include the `dimensions:` block and coordinate-variable values (especially `lev`), so wrong dimension sizes or wrong pressure-level values can't slip through verification.
>   3. Clarifies that `--input-root`/`--output-root` are **conditionally** required — needed for processing, not for `--count-tasks`.
> - **v5** (separate doc): adds `aires_rad` profile (radiation + heat fluxes) as a strict superset of `exp26`, lands provisional first then flipped to audited. See `docs/aires_rad_profile_plan.md` for the v5 plan and Stream A / Stream B workflow. The v4 plan body below remains unchanged.

## Context

The current `src/plasim_postprocessor/` directory holds the upstream amaurylancelin/AI-RES post-processor with three experiment configs (EXP15/25/26), 11 namelist variants, a cluster wrapper layer, and an inactive (and uncopiable) NCL geopotential path. You want a single, generic, lossless extractor that:

- Takes `--profile {exp15|exp25|exp26}`, `--sims`, `--years`, `--input-root`, `--output-root` from the CLI.
- Writes one NetCDF per sim-year at native 6-hourly cadence to `<output-root>/sim{NN}/MOST.{YYYY}.nc`.
- Reproduces the audited variable set for the chosen profile **exactly** — no additions, no drops.
- Ships with a SLURM array template whose array size is derived from `--count-tasks`, not hardcoded.

**Toolchain contract**: every output variable is derived by **burn7** (replacing the inactive NCL geopotential path). **CDO** is used as plumbing only — to merge burn7 outputs into a single NetCDF and to compute `pr_6h` via `runsum,6` (burn7 has no temporal-accumulation primitive; the original pipeline uses CDO for this, and there is no burn7 equivalent to switch to). This is a deliberate scope choice for v4: the goal is to displace NCL, not CDO.

A clarifying note on the original framing: the existing `plasim_postprocessor.py` already routes Z500 through burn7 (`zg_source: "burn7"`) by default. The NCL path is unreachable in this repo (the .ncl scripts aren't present); deleting it is dead-code removal, not a behavior change.

## Audited profile contract

Profile definitions are **internal constants** in the script (not user-overridable). The audit (Step 1) validates this table empirically.

| Profile | Sigma vars (11 sigma levels) — burn7 sigma | Land vars — burn7 sigma | zg pressure levels (hPa) — burn7 VTYPE=P | Precip — CDO runsum |
|---------|---------------------------------------------|--------------------------|--------------------------------------------|---------------------|
| `exp15` | ta, ua, va, hus, pl, tas | — | 500 | — |
| `exp25` | ta, ua, va, hus, pl, tas | — | 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 | pr_6h |
| `exp26` | ta, ua, va, hus, pl, tas | mrso | 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 | pr_6h |

Per-variable producing tool:

| Variable family | Tool | How |
|---|---|---|
| ta, ua, va, hus, pl, tas, mrso | burn7 | sigma namelist (`vtype=sigma`, MODLEV=10..0) |
| zg | burn7 | pressure namelist (`vtype=p`, `hpa=…`) |
| pr_6h | CDO | `selname,pr` → `runsum,6` → `chname,pr,pr_6h`; merged into output |
| (file assembly) | CDO | `merge` of the burn7 sigma + burn7 zg + (optional) pr_6h NetCDFs |

All profiles produce a single T42 NetCDF per sim-year at native 6-hourly cadence. Pangu regridding is out of scope (separate downstream tool if ever needed).

## Approach

### Step 1 — Exhaustive audit (preservation contract validation)

Goal: confirm the contract table above is correct, then snapshot each profile's expected output to a machine-readable file rich enough to catch dimension and pressure-level errors.

1. **Static trace.** For each YAML (`config/EXP15_postproc.yaml:13-22`, `config/EXP25_postproc.yaml:14-26`, `config/EXP26_postproc.yaml:7-16`), follow into `plasim_postprocessor.py`:
   - `_sigma_variables()` (line 93) — predicts the sigma namelist `code=` string per profile.
   - `_compute_z500_burn7()` (line 204) — predicts the zg namelist (`code=zg, hpa=…`).
   - `accumulate_precipitation()` (line 251) — predicts `pr_{H}h` additions when `accumulate_precip=True`.

2. **Empirical run.** On the same input `/scratch/10000/amarchakitus/PLASIM/data/sim30/MOST.0012`, run the existing pipeline three times (one per profile):
   - Patch `EXP{15,25,26}_postproc.yaml` to point `burn7_wrapper:` at `src/plasim_postprocessor/burn7_wrappers/stampede3.sh`.
   - Patch `burn7_wrappers/stampede3.sh:24` `BURN7_DIR` from `../../postprocessor2.0/burn7/Stampede3` to `../burn7/Stampede3` (matches our flattened layout).
   - For EXP25/26: set `outputs.pangu.enabled: false` (Pangu out of scope; grid file isn't on Stampede3 anyway).
   - For each run, extract a normalized manifest with three sections:

     ```bash
     manifest() {
         local nc="$1"
         echo "=== dimensions ==="
         ncdump -h "$nc" \
             | awk '/^dimensions:/{flag=1; next} /^variables:/{flag=0} flag' \
             | sed 's/^[[:space:]]*//' | sort
         echo "=== variables ==="
         ncdump -h "$nc" \
             | awk '/^variables:/{flag=1; next} /^\/\/ global attributes:|^data:/{flag=0} flag' \
             | grep -E "^\s*(float|double|int|short|char) " \
             | sort
         echo "=== lev coordinate values ==="
         ncdump -v lev "$nc" 2>/dev/null \
             | awk '/^ lev =/,/;/' \
             | tr -d '\n ' | sed 's/;.*//' || true
         echo
         echo "=== lat/lon shape (sanity) ==="
         ncdump -h "$nc" | grep -E "^\s+(lat|lon) =" | sort
     }
     ```

     Captures: `dimensions:` block (catches wrong sizes), variable declarations (catches wrong dtype/shape), the literal `lev` coordinate values (catches wrong pressure-level set), and lat/lon dim sizes (sanity check on the T42 grid). Time values are excluded since they're calendar-dependent and not part of the variable contract.

   - Save: `docs/audit_snapshots/exp15_manifest.txt`, `exp25_manifest.txt`, `exp26_manifest.txt`.

3. **Reconcile.** Diff predicted (1) vs empirical (2) per profile. Any cell that diverges from the contract table is a finding logged in `docs/plasim_postprocessor_audit.md`. The new script targets the **empirical** snapshots, since empirical is what real code produces.

4. **Lock the contract.** Commit the three manifest files. They are the verification targets the new script's output must match exactly.

5. **NCL: not validated, just removed.** The NCL scripts are absent from this repo (upstream lives at `RES/postprocessor2.0/src/`, never copied). The README documents a −20 to −30 m bias. NCL code paths and config fields are deleted in Step 4 as dead code, with a note in the audit report.

The audit modifies only the wrapper-path patches needed to make the existing pipeline runnable on Stampede3. No deletions until Step 4.

### Step 2 — Implement the new script

Single self-contained Python file: `src/plasim_postprocessor/plasim_postprocessor.py`. No YAML, no wrapper layer, no cluster abstractions. Profile selection is the only variable knob.

**CLI surface (full):**

```
plasim_postprocessor.py

  # --- Required ---
  --profile           {exp15,exp25,exp26}     selects audited variable set
  --sims              INT [INT ...]            e.g. 30 31 32 33 34 35 36
  --years             START END                inclusive, e.g. 1 100

  # --- Conditionally required (needed for processing, not for --count-tasks) ---
  --input-root        PATH                     root containing sim{NN}/MOST.{YYYY:04d}
  --output-root       PATH                     root for output NCs

  # --- Execution ---
  --burn7-binary      PATH                     default: <script_dir>/burn7/Stampede3/burn7
  --task-index        INT                      run only the Nth (sim, year) pair (for SLURM array)
  --count-tasks                                print number of (sim, year) pairs and exit
  --overwrite                                  force re-write of existing output files
  --dry-run                                    print actions without executing
  -v / --verbose                               debug logging
```

**Argument-validation rules** (enforced in `main()` before any processing):

| Mode | Required args |
|---|---|
| `--count-tasks` | `--profile`, `--sims`, `--years` only |
| Any processing run (default, `--task-index`, `--dry-run`) | All of the above plus `--input-root`, `--output-root` |

`argparse` declares `--input-root`/`--output-root` with `default=None`; a post-parse check raises `argparse.ArgumentError` with a clear message ("--input-root and --output-root are required unless --count-tasks is set") if missing in a processing mode. `--help` documents this in the option help text.

**Profile definitions (internal constants):**

```python
PROFILES = {
    "exp15": {
        "sigma_vars": ["ta", "ua", "va", "hus", "pl", "tas"],
        "land_vars": [],
        "pressure_levels": [500],
        "accumulate_precip_hours": [],
    },
    "exp25": {
        "sigma_vars": ["ta", "ua", "va", "hus", "pl", "tas"],
        "land_vars": [],
        "pressure_levels": [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
        "accumulate_precip_hours": [6],
    },
    "exp26": {
        "sigma_vars": ["ta", "ua", "va", "hus", "pl", "tas"],
        "land_vars": ["mrso"],
        "pressure_levels": [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
        "accumulate_precip_hours": [6],
    },
}
```

These mirror `config/EXP{15,25,26}_postproc.yaml` exactly and are validated against the audit snapshots.

**Internal structure:**

- `enumerate_tasks(sims, year_start, year_end)` → ordered `[(sim, year), …]`.
- `process_one(sim, year, profile, opts)`:
  1. Resolve input `{input_root}/sim{sim}/MOST.{year:04d}`. Skip with warning if missing.
  2. Resolve output `{output_root}/sim{sim}/MOST.{year:04d}.nc`. Skip if exists and not `--overwrite`.
  3. In a `tempfile.TemporaryDirectory`:
     - Build sigma namelist: `code = sigma_vars + land_vars + (["pr"] if accumulate_precip_hours else [])`, `MODLEV=10,9,…,0`, `vtype=sigma,htype=g,mean=0,netcdf=1`. Reproduces `_sigma_variables()` (line 93) inclusion logic.
     - Run `burn7 < sigma.nl input sigma.nc`.
     - Build zg namelist: `code=zg, hpa={pressure_levels}`, `vtype=p,htype=g,mean=0,netcdf=1`. Run burn7 → `zg.nc`. CDO merge into `sigma.nc`.
     - For each `H` in `accumulate_precip_hours`: `cdo selname,pr` → `cdo runsum,H` → `cdo chname,pr,pr_{H}h`; CDO merge into `sigma.nc`.
     - Move final to `{output_root}/sim{sim}/MOST.{year:04d}.nc`.
- `main()`:
  - Parse args; enforce conditional-requirement rule above.
  - `--count-tasks` → print `len(enumerate_tasks(...))`, exit 0.
  - `--task-index N` → process `tasks[N]` only.
  - else → sequential loop.

**Reuse from existing code (read patterns; don't import):**

- Sigma + Z500 namelist text format: `plasim_postprocessor.py:123-141`.
- burn7 subprocess + error reporting: `plasim_postprocessor.py:143-159`.
- CDO helper (`cdo -s -O <op>` + stderr capture): `plasim_postprocessor.py:358-366`.
- Precip accumulation chain: `plasim_postprocessor.py:251-282`.
- Stampede3 module + `LD_LIBRARY_PATH`: `burn7_wrappers/stampede3.sh:19-22` → moves verbatim into `submit.slurm`.

### Step 3 — SLURM template (placeholders, fail-fast)

`src/plasim_postprocessor/submit.slurm`:

```bash
#!/bin/bash
#SBATCH -J plasim_postproc
#SBATCH -p skx                      # adjust queue as needed
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --array=0-0                 # REQUIRED — set after running --count-tasks
#SBATCH -o logs/postproc_%A_%a.out

# === REQUIRED — fill in for your run (sample values shown only as comments) ===
PROFILE=""             # e.g. "exp15" / "exp25" / "exp26"
SIMS=""                # e.g. "30 31 32 33 34 35 36"
YEAR_START=            # e.g. 1
YEAR_END=              # e.g. 100
INPUT_ROOT=            # e.g. /scratch/10000/amarchakitus/PLASIM/data
OUTPUT_ROOT=           # e.g. $SCRATCH/AI-RES/data/postproc

# === SIZING THE ARRAY ===
# 1. Compute task count (no --input-root/--output-root needed for this):
#      N=$(python3 plasim_postprocessor.py --profile $PROFILE \
#                  --sims $SIMS --years $YEAR_START $YEAR_END --count-tasks)
# 2. Set #SBATCH --array=0-$((N-1)) above, OR submit with:
#      sbatch --array=0-$((N-1)) submit.slurm
# Default --array=0-0 processes only the first (sim, year) pair (smoke test).

# === ENVIRONMENT (Stampede3-specific) ===
module purge
module load gcc netcdf cdo
export LD_LIBRARY_PATH=/opt/apps/intel24/netcdf/4.9.2/x86_64/lib:/home1/09979/awikner/netcdf-4.2/lib:${LD_LIBRARY_PATH}

# === FAIL FAST IF UNSET ===
: "${PROFILE:?PROFILE is required — edit submit.slurm}"
: "${SIMS:?SIMS is required}"
: "${YEAR_START:?YEAR_START is required}"
: "${YEAR_END:?YEAR_END is required}"
: "${INPUT_ROOT:?INPUT_ROOT is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

# === DISPATCH ===
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python3 "$SCRIPT_DIR/plasim_postprocessor.py" \
    --profile "$PROFILE" \
    --sims $SIMS \
    --years $YEAR_START $YEAR_END \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --task-index "$SLURM_ARRAY_TASK_ID"
```

The `:${VAR:?msg}` guards make an unedited template fail loudly before burning a queue slot. `LD_LIBRARY_PATH` is the only Stampede3-specific magic, copied verbatim from `burn7_wrappers/stampede3.sh:22`.

### Step 4 — Cleanup (delete obsolete files)

Performed only after Step 1 audit snapshots are committed and Step 2/3 verification passes against them.

Delete:
- `src/plasim_postprocessor/plasim_postprocessor.py` (replaced)
- `src/plasim_postprocessor/README.md` (obsolete; new docs live in module docstring + `--help` + `docs/plasim_postprocessor_audit.md`)
- `src/plasim_postprocessor/burn7_wrappers/` (entire dir)
- `src/plasim_postprocessor/config/` (entire dir — all 3 EXP YAMLs replaced by internal `PROFILES` dict)
- `src/plasim_postprocessor/namelists/` (entire dir — all 11 .nl files; namelists are now generated in tmpdir)
- `src/plasim_postprocessor/burn7/derecho/` (other-cluster binary)
- `src/plasim_postprocessor/burn7/jeanzay/` (other-cluster binary)
- `src/plasim_postprocessor/burn7/Stampede3/submit_burn.sh` (obsolete standalone submit script)

Keep:
- `src/plasim_postprocessor/burn7/` source files (`burn7.cpp`, `makefile`, `make_burn.sh`, `example.nl`, `README_POSTPROCESSOR`, `readme.txt`, `ExampleCompile`, `burn7qr.pdf`) — needed if the binary breaks and burn7 must be rebuilt on Stampede3.
- `src/plasim_postprocessor/burn7/Stampede3/burn7` — prebuilt binary the new script depends on.

### Final directory layout

```
src/plasim_postprocessor/
├── plasim_postprocessor.py     # new single-purpose script, profile-driven
├── submit.slurm                # SLURM array template, placeholders + fail-fast guards
└── burn7/                      # burn7 source + Stampede3 binary
    ├── burn7.cpp
    ├── makefile
    ├── make_burn.sh
    ├── example.nl
    ├── README_POSTPROCESSOR
    ├── readme.txt
    ├── ExampleCompile
    ├── burn7qr.pdf
    └── Stampede3/
        └── burn7

docs/
├── plasim_postprocessor_refactor_plan.md   # this doc
├── plasim_postprocessor_audit.md           # produced by Step 1 (narrative)
└── audit_snapshots/                        # produced by Step 1 (machine-checkable)
    ├── exp15_manifest.txt
    ├── exp25_manifest.txt
    └── exp26_manifest.txt
```

## Critical files

- **Create:** `src/plasim_postprocessor/plasim_postprocessor.py` (~200 lines, profile-driven).
- **Create:** `src/plasim_postprocessor/submit.slurm`.
- **Create:** `docs/plasim_postprocessor_audit.md` + `docs/audit_snapshots/{exp15,exp25,exp26}_manifest.txt` (Step 1 output).
- **Delete:** files listed in Step 4.
- **Read for reference (do not import):** `plasim_postprocessor.py:93` (sigma var resolution), `:204` (Z500 burn7 path), `:251` (precip accumulation), `:358` (CDO helper); `burn7_wrappers/stampede3.sh:19-22` (env setup).

## Verification

End-to-end after Step 2/3 are in place — every check is machine-checkable against the audit snapshots.

```bash
module load gcc netcdf cdo
export LD_LIBRARY_PATH=/opt/apps/intel24/netcdf/4.9.2/x86_64/lib:/home1/09979/awikner/netcdf-4.2/lib:$LD_LIBRARY_PATH
cd ~/AI-RES

# Same manifest function used by the audit
manifest() {
    local nc="$1"
    echo "=== dimensions ==="
    ncdump -h "$nc" \
        | awk '/^dimensions:/{flag=1; next} /^variables:/{flag=0} flag' \
        | sed 's/^[[:space:]]*//' | sort
    echo "=== variables ==="
    ncdump -h "$nc" \
        | awk '/^variables:/{flag=1; next} /^\/\/ global attributes:|^data:/{flag=0} flag' \
        | grep -E "^\s*(float|double|int|short|char) " \
        | sort
    echo "=== lev coordinate values ==="
    ncdump -v lev "$nc" 2>/dev/null \
        | awk '/^ lev =/,/;/' \
        | tr -d '\n ' | sed 's/;.*//' || true
    echo
    echo "=== lat/lon shape (sanity) ==="
    ncdump -h "$nc" | grep -E "^\s+(lat|lon) =" | sort
}

# 1. --count-tasks math (no --input-root/--output-root required here)
python3 src/plasim_postprocessor/plasim_postprocessor.py \
    --profile exp15 --sims 30 31 --years 12 13 --count-tasks
# expect: 4   (2 sims × 2 years)

# 2. exp15 round-trip → must match audit snapshot byte-for-byte
python3 src/plasim_postprocessor/plasim_postprocessor.py \
    --profile exp15 --sims 30 --years 12 12 \
    --input-root /scratch/10000/amarchakitus/PLASIM/data \
    --output-root /tmp/postproc_exp15
manifest /tmp/postproc_exp15/sim30/MOST.0012.nc > /tmp/exp15_new.txt
diff /tmp/exp15_new.txt docs/audit_snapshots/exp15_manifest.txt   # must be empty

# 3. exp25 round-trip (must show lev with 13 pressure values; must include pr_6h)
python3 src/plasim_postprocessor/plasim_postprocessor.py \
    --profile exp25 --sims 30 --years 12 12 \
    --input-root /scratch/10000/amarchakitus/PLASIM/data \
    --output-root /tmp/postproc_exp25
manifest /tmp/postproc_exp25/sim30/MOST.0012.nc > /tmp/exp25_new.txt
diff /tmp/exp25_new.txt docs/audit_snapshots/exp25_manifest.txt   # must be empty

# 4. exp26 round-trip (adds mrso)
python3 src/plasim_postprocessor/plasim_postprocessor.py \
    --profile exp26 --sims 30 --years 12 12 \
    --input-root /scratch/10000/amarchakitus/PLASIM/data \
    --output-root /tmp/postproc_exp26
manifest /tmp/postproc_exp26/sim30/MOST.0012.nc > /tmp/exp26_new.txt
diff /tmp/exp26_new.txt docs/audit_snapshots/exp26_manifest.txt   # must be empty

# 5. Native cadence preserved (no aggregation)
ncdump -h /tmp/postproc_exp15/sim30/MOST.0012.nc | grep "time = "
# expect ~1460 (4 per day × 365 days), NOT 365.

# 6. Idempotency: re-run without --overwrite is a no-op
python3 src/plasim_postprocessor/plasim_postprocessor.py \
    --profile exp15 --sims 30 --years 12 12 \
    --input-root /scratch/10000/amarchakitus/PLASIM/data --output-root /tmp/postproc_exp15
# expect log: "skipping sim30/MOST.0012.nc (exists; pass --overwrite to force)"

# 7. Conditional-required check: processing without --input-root/--output-root must fail loudly
python3 src/plasim_postprocessor/plasim_postprocessor.py \
    --profile exp15 --sims 30 --years 12 12
# expect: argparse error "--input-root and --output-root are required unless --count-tasks is set"

# 8. SLURM array smoke test (after editing submit.slurm placeholders)
sbatch src/plasim_postprocessor/submit.slurm    # default --array=0-0 → one task
```

If any of (2)/(3)/(4) `diff` outputs are non-empty, the new script's namelist generation diverges from the locked contract and must be fixed before deletion in Step 4. Verification is binary: matches snapshot or doesn't.
