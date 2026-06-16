# SFNO-5410 inference from your own NetCDF — Stampede3 user guide

**Audience.** A group member of G-819272 on TACC Stampede3 who wants to roll
out the SFNO-5410 emulator from her **own initial-condition NetCDF**, for an
arbitrary forecast horizon, deterministic or as a perturbation ensemble. No
prior knowledge of this repo is assumed.

**Last verified.** 2026-05-09 against the H100 + packed-Derecho-env path that
the production eval pipeline uses.

---

## 0. TL;DR

```bash
# Once: confirm you can read the public group workspace.
ls /work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/group/sfno5410_byo/
# Should show: infer_sfno5410_byo_ic.py, byo_ic.py, submit_sfno5410_byo_inference.slurm,
#              __init__.py

# Per-run: prepare your IC NetCDF (single timestep, sim52 grid; see §5).
#          Then submit:

cd $SCRATCH         # somewhere your job can write outputs
mkdir -p byo_run01

IC_NC=/your/abs/path/to/your_ic.nc \
INIT_DATETIME=2020-01-01_00:00:00 \
HORIZON_DAYS=15 \
OUT_DIR=$SCRATCH/byo_run01 \
sbatch /work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/group/sfno5410_byo/submit_sfno5410_byo_inference.slurm
```

Outputs: `byo_member000.nc` + `byo_inference_metadata.json` in `OUT_DIR`.

For an ensemble, add `NUM_MEMBERS=8 EPSILON_FACTOR=1e-3 PERTURBATION_TYPE=gaussian_noise`
to the env line.

The rest of this document explains every piece.

---

## 1. What this gives you

This pipeline runs **one forward integration** of the group's SFNO-5410
emulator starting from your IC, for `HORIZON_DAYS` days, with 6-hour output
cadence. You get:

- **Deterministic mode** (`NUM_MEMBERS=1`): a single forecast trajectory.
- **Ensemble mode** (`NUM_MEMBERS=N>1` + `EPSILON_FACTOR>0`): N trajectories,
  each starting from your IC with independent Gaussian (or Perlin) noise added.

It does **not** do any scoring, climatology comparison, or evaluation against
truth — that is handled by a separate, sim52-only pipeline. This guide is for
arbitrary-IC forecasting only.

### What "boundary forcing" means here

SFNO-5410 needs prescribed boundary fields at every step (sea-surface
temperature, top-of-atmosphere insolation, sea-ice cover). It does **not**
read those from your IC NetCDF. They come from a fixed sim52 template year
on disk:

- `51_*.h5` — non-leap template, used when your IC year is non-leap.
- `52_*.h5` — leap template, used when your IC year is leap.

So if you give the model two ICs with the same `MM-DD HH` but different
years (one leap, one not), you will get different boundary forcing.

### What you cannot do (yet)

- **Multi-year rollouts.** Capped at 365 days. The in-process loop has no
  year-rollover boundary handling. (The yaml-driven `long_inference.py`
  path supports multi-year, but is not wired into this CLI.)
- **Bring-your-own boundary forcing.** Boundary is sim52-only.
- **Run on CPU.** Only the H100 partition is supported.
- **Use ICs not on the sim52 64×128 grid.** You must regrid first.

---

## 2. Prerequisites

| What | Notes |
|------|-------|
| Stampede3 account | https://accounts.tacc.utexas.edu/ |
| Allocation on H100 | `idev -p h100` should succeed; otherwise see TACC docs |
| Group membership in **G-819272** | Required to read the model + data. Check with `id` (look for `G-819272` in groups) |
| `sbatch` queue access | Standard for any TACC user |

If your username is not in `G-819272`, ask the PI; until then the paths in
§3 will return "Permission denied."

---

## 3. Where everything lives (paths you'll touch)

You do not need to copy anything to your home directory. Everything is
read directly from `$WORK` and `$SCRATCH` of the maintainer's account.

| What | Absolute path |
|------|---------------|
| **Group workspace** (scripts + reader you'll use) | `/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/group/sfno5410_byo/` |
| **Inference driver** | `…/group/sfno5410_byo/infer_sfno5410_byo_ic.py` |
| **SLURM submit wrapper** | `…/group/sfno5410_byo/submit_sfno5410_byo_inference.slurm` |
| **NetCDF schema reader** (read for the schema; not invoked directly) | `…/group/sfno5410_byo/byo_ic.py` |
| **Packed Derecho Python env** (CUDA 12, torch, torch_harmonics, etc.) | `/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_runtime/aires_env_20260509/unpacked/bin/python` |
| **Blocking source tree** (PanguPlasim) | `/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_blocking/source_trees/forecast_modules/PanguPlasim` |
| **Pangu source tree** (data loader, perturber) | `/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0` |
| **Epoch-48 checkpoint** | `/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_blocking/sfno5410_blocking_epoch48_20260509/checkpoints/ckpt_epoch_48.tar` |
| **Default yaml** | `…/derecho_blocking/source_trees/forecast_modules/PanguPlasim/yaml_config/SFNO_PLASIM_H5_DERECHO_5410_deterministic.yaml` |
| **Sim52 boundary H5 root** | `/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/h5/sigma_data` |
| **Bias dir** | `/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/bias` |
| **Climatology** | `/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/sigma_data/climatology.nc` |

You will normally only ever type the **group workspace** path. Everything
else is wired into the SLURM defaults.

### 3.1 Will my output disappear?

`$SCRATCH` is purged on TACC's published schedule (typically files older
than ~10 days). Move your outputs to `$WORK` (or your own `$WORK2`)
before TACC's purge if you want to keep them.

---

## 4. Environment setup on Stampede3

You do **not** need to `pip install` anything or build a conda env. The
packed Derecho env is ready-to-use and is what the `submit_*.slurm`
wrapper invokes.

If you want to verify it works *before* submitting, on a login node:

```bash
PACKED_PY=/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_runtime/aires_env_20260509/unpacked/bin/python
"$PACKED_PY" -c "import torch; print('torch', torch.__version__, 'cuda compiled:', torch.version.cuda)"
```

You should see `torch 2.x.x cuda compiled: 12.x`. (`torch.cuda.is_available()`
will be **False** on a login node — that is expected; CUDA only initialises
on the H100 compute node where the SLURM job lands.)

---

## 5. Preparing your IC NetCDF

The model is trained on a **fixed grid and variable set**. Your IC must
match that contract exactly. The reader in
`/work2/.../group/sfno5410_byo/byo_ic.py` enforces it and will refuse a
malformed file with a clear error message.

### 5.1 Required schema

| Variable | Dims | Shape | Units / convention |
|----------|------|-------|--------------------|
| `pl`     | `(lat, lon)` or `(time=1, lat, lon)` | `(64, 128)` | log of surface pressure (`ln(p_s)`), dimensionless |
| `tas`    | same | `(64, 128)` | 2 m air temperature, K |
| `pr_6h`  | same | `(64, 128)` | precip integrated over 6 h (rate × 6 h, group convention) |
| `ta`     | `(lev, lat, lon)` or `(time=1, lev, lat, lon)` | `(10, 64, 128)` | air temperature on σ-levels, K |
| `ua`     | same | `(10, 64, 128)` | zonal wind on σ-levels, m/s |
| `va`     | same | `(10, 64, 128)` | meridional wind on σ-levels, m/s |
| `hus`    | same | `(10, 64, 128)` | specific humidity on σ-levels, kg/kg |
| `zg`     | `(plev, lat, lon)` or `(time=1, plev, lat, lon)` | `(10, 64, 128)` | geopotential height on pressure levels, **gpm** (NOT m²/s²) |

### 5.2 Required coordinates

- `lat`: length 64 (the model is on a Gaussian grid; spacing is fixed).
- `lon`: length 128.
- `lev`: length 10 (10 σ-levels).
- `plev`: length 10 (10 pressure levels: 200, 250, 300, 400, 500, 600,
  700, 850, 925, 1000 hPa).
- `time` (optional): if present, must have length 1.

The reader does **not** check `lat` / `lon` / `lev` / `plev` *values* —
only their lengths. Your interpolation onto the sim52 grid is your
responsibility. If your IC is on a different grid, regrid it first
(e.g. with `xesmf` or `cdo`).

### 5.3 Conventions you must NOT re-convert

The group's pipeline emits these in non-standard units. Your IC must
match the same conventions:

- `pl = ln(p_s)` — log of surface pressure, **dimensionless**. Do not
  pass raw Pa.
- `zg` — **geopotential metres** (gpm). Do not multiply by g; do not
  divide by 9.81.
- `pr_6h` — **rate × 6 h** = mass per 6-hour interval. Do not multiply
  by 6 × 3600.

Getting any of these wrong silently produces garbage forecasts.

### 5.4 Quick schema check on the login node

```bash
PACKED_PY=/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_runtime/aires_env_20260509/unpacked/bin/python
"$PACKED_PY" - <<'EOF'
import sys
sys.path.insert(0, "/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/group/sfno5410_byo")
from byo_ic import validate_byo_ic
raw = validate_byo_ic("/your/path/to/your_ic.nc")
for k, v in raw.items():
    print(f"{k:8s} shape={v.shape}  min={v.min():.3g}  max={v.max():.3g}")
EOF
```

If this returns sensible min/max for every variable, you're good to submit.

---

## 6. Running deterministic inference

This is the simplest mode. One IC, one forecast trajectory, no perturbation.

### 6.1 Submit

```bash
cd $SCRATCH                  # any dir your job can write to
mkdir -p byo_det_run01

IC_NC=/your/abs/path/your_ic.nc \
INIT_DATETIME=2020-01-01_00:00:00 \
HORIZON_DAYS=15 \
OUT_DIR=$SCRATCH/byo_det_run01 \
sbatch /work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/group/sfno5410_byo/submit_sfno5410_byo_inference.slurm
```

`sbatch` will print a job ID. Track it with `squeue -u $USER`.

### 6.2 What gets produced

On success, `OUT_DIR` will contain:

```
byo_member000.nc                    # the forecast (60 timesteps for 15 days)
byo_inference_metadata.json         # run provenance (sha256s, paths, timing)
```

And in the **directory you ran `sbatch` from**:

```
slurm-byo-<jobid>.out       # stdout
slurm-byo-<jobid>.err       # stderr
```

(So `cd` into a writable dir like a fresh `$SCRATCH/byo_runXX` before
`sbatch`. The wrapper figures out the driver path on its own — you do
not need an AI-RES checkout in your home.)

### 6.3 Expected wall time

- 15 days (60 steps), 1 member: **<2 minutes** of GPU time after the
  one-shot model load (~30 s).
- Full SLURM walltime budget is 2 hours by default — generous so you
  can stretch to 365 days without bumping `-t`.

---

## 7. Running ensemble inference

Ensemble mode runs N forward passes from your IC, each with an
independent IC perturbation. The *boundary forcing* is identical across
members (same sim52 template year), so spread comes purely from the IC
perturbation.

### 7.1 Submit (8-member Gaussian ensemble)

```bash
cd $SCRATCH
mkdir -p byo_ens_run01

IC_NC=/your/abs/path/your_ic.nc \
INIT_DATETIME=2020-01-01_00:00:00 \
HORIZON_DAYS=15 \
NUM_MEMBERS=8 \
EPSILON_FACTOR=1e-3 \
PERTURBATION_TYPE=gaussian_noise \
RANDOM_SEED=42 \
OUT_DIR=$SCRATCH/byo_ens_run01 \
sbatch /work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/group/sfno5410_byo/submit_sfno5410_byo_inference.slurm
```

### 7.2 Knobs explained

- `NUM_MEMBERS` — number of ensemble trajectories. 1–32 is reasonable;
  beyond ~16 you may need to bump SLURM walltime.
- `EPSILON_FACTOR` — perturbation magnitude. The actual noise standard
  deviation on each variable is
  `EPSILON_FACTOR × surface_ff_std / surface_std` (residual-norm scaling
  is on by default — see `utils/perturbation.py:48-58`). Typical values:
  - `1e-4` — very small, ensemble spread builds slowly
  - `1e-3` — middling, the value used in upstream tests
  - `1e-2` — large, members diverge fast
- `PERTURBATION_TYPE` — choose one:
  - `gaussian_noise` — independent N(0, ε) on every member, every grid
    cell, every variable. Default choice.
  - `gaussian_noise_n_minus_1` — first member is unperturbed (control);
    members 1..N-1 are Gaussian. Use when you want a "control + N-1
    perturbed" layout.
  - `perlin_noise` — spatially correlated noise (Perlin spectrum). Use
    if you want perturbations that don't average out spatially.
- `RANDOM_SEED` — perturber seed. **Set this if you want
  reproducibility.** Different seeds → different ensemble realisations.

### 7.3 What gets produced

```
byo_member000.nc
byo_member001.nc
…
byo_member007.nc
byo_inference_metadata.json
```

Each member is a full forecast trajectory of the same shape as the
deterministic case.

### 7.4 Validations the CLI will refuse

The driver fails fast (before model load) on these:

- `NUM_MEMBERS > 1` with `EPSILON_FACTOR=0` — all members would be
  bit-identical, which is never what you want. **Fix:** set
  `EPSILON_FACTOR>0` or drop to `NUM_MEMBERS=1`.
- `EPSILON_FACTOR > 0` without `PERTURBATION_TYPE` — ambiguous. **Fix:**
  set `PERTURBATION_TYPE`.
- `HORIZON_DAYS > 365` — boundary forcing would overrun the template
  year. **Fix:** lower the horizon (or use the long_inference.py path,
  not yet wired here).
- `INIT_DATETIME` not on a 6-hour grid (00, 06, 12, 18). **Fix:** round.

---

## 8. Output format

Each `*_member<NNN>.nc` is a CF-style NetCDF with:

| Coord | Dim size | Notes |
|-------|----------|-------|
| `time` | `K + 1` | `K = HORIZON_DAYS × 4`. Index 0 is your IC; indices 1..K are forecast leads at +6 h, +12 h, …, in your IC's calendar. |
| `lat`, `lon` | 64, 128 | sim52 grid. |
| `lev`  | 10 | σ-levels (sigma values from the upstream dataset). |
| `plev` | 10 | pressure levels in Pa (200–1000 hPa). |

| Variable | Dims | Same units / convention as the input schema (§5.1). |
|----------|------|---------------------------------------------------|
| `pl`     | `(time, lat, lon)` | |
| `tas`    | `(time, lat, lon)` | |
| `pr_6h`  | `(time, lat, lon)` | |
| `ta`     | `(time, lev, lat, lon)` | |
| `ua`     | `(time, lev, lat, lon)` | |
| `va`     | `(time, lev, lat, lon)` | |
| `hus`    | `(time, lev, lat, lon)` | |
| `zg`     | `(time, plev, lat, lon)` | |

### 8.1 Reading the time axis

Outputs use `cftime.DatetimeProlepticGregorian` because PLASIM uses a
proleptic-Gregorian calendar. To inspect:

```python
import xarray as xr
ds = xr.open_dataset("byo_member000.nc", use_cftime=True)
print(ds.time.values[0])      # your IC datetime
print(ds.time.values[-1])     # IC + HORIZON_DAYS days
print(ds.tas.isel(time=10).mean("lon").values)  # zonal-mean tas at +60h
```

### 8.2 Run provenance

`byo_inference_metadata.json` records:

- The exact command line, IC file sha256, checkpoint sha256, yaml sha256.
- `boundary_template_year`, `K_steps`, `horizon_hours`, all CLI knobs.
- Per-member output paths + sha256s + sizes.
- Wall-clock seconds.

Keep this file alongside your outputs — it is the audit trail.

---

## 9. Adjusting the rollout length

`HORIZON_DAYS` accepts any positive number ≤ 365 days. Internally
the driver computes `K = round(HORIZON_DAYS × 24 / 6)` 6-hour steps, so:

| `HORIZON_DAYS` | `K` (steps) | Output `time` length |
|----------------|-------------|----------------------|
| 0.25 | 1 | 2 |
| 1    | 4 | 5 |
| 5    | 20 | 21 |
| 15   | 60 | 61 (the production eval default) |
| 30   | 120 | 121 |
| 90   | 360 | 361 |
| 365  | 1460 | 1461 |

Multiply runtime by ~`K/60`. SLURM walltime defaults to 2 h; for
horizons ≥ 180 days bump it: `sbatch -t 04:00:00 …`.

---

## 10. Failure modes & debugging

### 10.1 "Permission denied"

You are not in `G-819272`, or one of the parent directories has been
re-locked. Check:

```bash
id | grep -o G-819272                        # must print the group
namei -l /work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_blocking/sfno5410_blocking_epoch48_20260509/checkpoints/ckpt_epoch_48.tar
```

The `namei -l` output should show every path component as readable by
your user or your group. If a component is `drwx------`, ask the
maintainer to widen it.

### 10.2 "FATAL: required env var IC_NC is unset"

The wrapper requires `IC_NC`, `INIT_DATETIME`, `HORIZON_DAYS`, `OUT_DIR`.
You probably forgot one. Re-run with all four set as `KEY=val` env
prefixes on the `sbatch` line.

### 10.3 Schema errors

The driver fails fast with `ValueError: <ic_path>: missing required
variables [...]`, or `dim X has size N, expected M`. Re-run §5.4 to
isolate which variable is wrong.

Common causes:
- **Multi-timestep file.** Strip to a single timestep:
  `cdo -seltimestep,1 in.nc out.nc` or `xr.Dataset.isel(time=[0])`.
- **Wrong vertical-level count.** SFNO-5410 needs exactly 10 σ-levels for
  `ta/ua/va/hus` and 10 pressure levels for `zg`. Pre-interpolate.
- **Variable named differently.** Rename to the exact spellings in §5.1
  (case-sensitive).

### 10.4 "CUDA is required for SFNO-5410 inference"

You ran the driver outside a SLURM allocation that owns a GPU. Either:
- Submit through `submit_sfno5410_byo_inference.slurm` (recommended), or
- `idev -p h100 -t 02:00:00` first, then run interactively.

### 10.5 "perturber reseeded with X" but my members are still identical

`PERTURBATION_TYPE=gaussian_noise_n_minus_1` makes member 0 unperturbed
by design. Members 1..N-1 should differ. If *all* members match, check:

- `EPSILON_FACTOR` was actually >0 (the launch echo line shows the args).
- The output members really come from the same job ID (not an old
  deterministic run still in the dir — use `--force` or a fresh
  `OUT_DIR`).

### 10.6 Job rejected at submit time

TACC enforces a per-user pending-job limit. If `sbatch` returns
"QOSMaxJobsPerUserLimit" or similar, wait for a pending job to clear,
then resubmit. `squeue -u $USER` shows your queue.

### 10.7 "OUT_DIR already contains byo_member*.nc files"

Safety guard against silently overwriting prior outputs. Either:
- Use a fresh `OUT_DIR` (recommended — provenance stays clean), or
- Add `FORCE=1` to the env prefix.

### 10.8 "torch_harmonics" or similar import error

Means you are not using the packed Derecho Python. Check the launch
echo line in `slurm-byo-<jobid>.out` — it should print
`python=…/aires_env_20260509/unpacked/bin/python`. If it shows your
own conda interpreter, override `PACKED_PYTHON` back to the default
(or unset it).

### 10.9 "RuntimeError: model.training stayed True after eval()"

Should not happen. If it does, the upstream code path changed. Capture
`logs/5410_byo_inf_<jobid>.err` and ping the maintainer.

---

## 11. Where to dig deeper

- The driver itself: `infer_sfno5410_byo_ic.py` is heavily commented.
  Read `main()` top-to-bottom for the exact flow.
- The IC schema enforcement: `byo_ic.py` is short — every check is a
  one-liner with the expected value.
- The model + dataset upstream: `forecast_modules/PanguPlasim/ensemble_inference.py`
  defines the `Stepper` class that owns the model, dataset, and
  perturber. The dataset (`utils/data_loader_multifiles.py`) defines
  `surface_transform` / `upper_air_transform` and the boundary-forcing
  loader.
- Group emulator background, training conventions, and the eval-track
  pipeline: see this repo's `.claude/skills/eval-sfno-5410/SKILL.md`.

If something here is wrong or unclear, drop a note in the group's
shared channel — this guide is a living document.
