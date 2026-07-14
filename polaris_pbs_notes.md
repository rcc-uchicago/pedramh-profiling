# Polaris (ALCF) PBS bring-up notes

The Polaris/PBS analog of the Midway notes docs. This is the **proof deliverable**
for the `polaris_handoff_prompt.md` bring-up: it records the confirmed Polaris
cluster facts, the env/module strategy, the per-model GREEN matrix
(probe → 1-GPU → 4-GPU), the repointed-path map, the two SFNO data-conversion
recipes, and a dated decisions log.

See **CLAUDE.md** for how to work here, **DESIGN.md** for what/why, **CHANGELOG.md**
for cross-cutting status, and **polaris_handoff_prompt.md** for the bring-up brief
this doc discharges.

> Style mirrors `si/bench_midway_notes.md`: a narrative + a dated decisions log.
> Everything under "confirmed" was verified **on a Polaris compute node** (probe
> job `7251974`, 2026-07-14), not assumed from the Midway-authored handoff.

---

## 1. Cluster facts (confirmed on Polaris, 2026-07-14)

| Item | Value (confirmed) | How confirmed |
|---|---|---|
| Login node | `polaris-login-01` | `hostname` |
| Scheduler | PBS Pro (`qsub`/`qstat`/`qdel`) | `which qsub` |
| **Account (`-A`)** | **`lighthouse-uchicago`** (17,128 node-h avail) | `sbank` |
| Compute node | `x3001c0s13b1n0` (1 node = **4× A100-SXM4-40GB**, sm80) | probe `nvidia-smi -L` |
| GPU memory | **40960 MiB/GPU** (~40 GiB); driver **570.124.06** | probe `nvidia-smi` |
| CPU | AMD EPYC Milan, **nproc=64** (32 cores × 2 SMT) | probe `nproc` |
| Node RAM | **not captured** — the probe's `free -g` line had a shell-quoting bug (fixed in `polaris_probe.pbs` after job 7251974); re-run the probe to record it | ⚠️ unverified |
| **Node-local scratch** | **`/local/scratch` = 2.8 TB free** (also `/tmp` = 252 GB) | probe |
| Queue (smoke) | `debug` (1–2 nodes, ≤1 h, 1 running job/user) | `qstat -Q` |
| Node/GPU directive | `-l select=1:system=polaris -l place=scatter` (whole node = 4 GPU) | probe accepted |
| **Filesystems** | `-l filesystems=home:eagle` (REQUIRED; jobs rejected without it) | probe accepted |
| Job id | `$PBS_JOBID` = `7251974.polaris-pbs-01...`; `${PBS_JOBID%%.*}` for a numeric tag | probe |
| Submit dir | `$PBS_O_WORKDIR` (PBS analog of `$SLURM_SUBMIT_DIR`) | probe |
| Nodefile | `$PBS_NODEFILE`; `NNODES=$(wc -l < $PBS_NODEFILE)` | probe |
| Project storage | `/eagle/projects/lighthouse-uchicago` = `/eagle/lighthouse-uchicago` (both → `/lus/eagle/projects/lighthouse-uchicago`) | `readlink -f` |

**PBS output-file gotchas (both bit us):**
1. `#PBS -o polaris_logs/` (a *directory*) makes PBS write `polaris_logs/<full_jobid>.OU`
   (with `-j oe`), **not** `<jobname>.o<seq>`. The per-model scripts instead pass an
   explicit `-o polaris_logs/<name>.log` for a predictable filename.
2. **PBS APPENDS to a fixed `-o` path** — it does not truncate. So a re-run's output is
   concatenated after the previous run's, and `<name>.log` accumulates *several jobs*.
   `pangu_e3sm_sfno.log` holds the OOM run 7252261 **and** the green 7252271;
   `si_polaris_bench.log` holds the crashed 7252286 **and** the green 7252700. **When
   reading these logs, always anchor on the `PBS_JOBID=` header of the run you care
   about** — a naive `grep -c OutOfMemory` on the file will report a green run as failed.

## 2. Environment strategy — ALCF base conda + `--user` top-ups

Decision: **use the ALCF base conda** (fastest per handoff), and `pip install --user`
the few packages it lacks. No fresh env was built — base already carries a
CUDA-12.9-matched torch and Lightning.

Canonical env block (identical in every `polaris_*.pbs`; validated by the green probe):

```bash
#!/bin/bash -l
module use /soft/modulefiles
module load conda            # → conda/2025-09-25: python 3.12, torch 2.8.0 (cu12.9),
conda activate base          #   lightning 2.5.5, wandb, h5py, xarray, einops, timm
export WANDB_MODE=offline
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

`module load conda` sources `.../mconda3/etc/profile.d/conda.sh`, sets `CUDA_HOME`
to cuda-12.9.1, and prepends the cuda/nccl/cudnn libs to `LD_LIBRARY_PATH` — this is
why `h5py` (built against `libcudart.so.12`) imports only *after* the full module
load, not a bare `source conda.sh`.

**`pip install --user` top-ups** (done once on the **login node** — compute nodes
have no outbound network; the conda module sets `http_proxy` for the login node).
They land in `PYTHONUSERBASE=/home/rmehta1987/.local/polaris/conda/2025-09-25`:

| Package | Version | Needed by | Note |
|---|---|---|---|
| `netCDF4` | 1.7.4 | S2S/SI/Pangu `.nc` stats | base lacks it |
| `h5netcdf` | — | xarray `.nc` backend | |
| `zarr` | 2.18.7 (`<3`) | PhysicsNeMo zarr store | pinned <3 |
| `torch_harmonics` | **0.7.4** in base conda | S2S-family, Pangu-SFNO, SI | the SFNO frameworks need 0.9.x — see the version box + §6 venv |

**The `torch_harmonics` version box (a genuine 3-way version squeeze):**
- `0.9.1` **wheel** → `import torch_harmonics` dies: `undefined symbol:
  _ZNK3c1010TensorImpl15incref_pyobjectEv` in `attention/_C.so` (built against a different
  torch ABI than 2.8). It also has **no sdist on PyPI**, so `--no-binary :all:` cannot
  build it.
- `0.7.4` / `0.8.0` → import fine, but expose only the *private* `_precompute_latitudes`;
  **makani 0.2.0 imports the public `precompute_latitudes`** and fails.
- Only a **GitHub source build** gives both (public API + an ABI-matched `_C`).
- ✅ **Resolution — split the envs (§6):** base conda keeps **0.7.4**, which is what the
  GREEN Pangu-SFNO (7252271) and SI (7252700) smokes actually ran on; the SFNO frameworks
  get an **isolated venv** carrying the source-built **0.9.2a**. This avoids re-validating
  the two greens against a new torch_harmonics, at the cost of one extra env.

## 3. Toolchain probe (gate 1) — **GREEN**

`polaris_probe.pbs` + `polaris_probe.py` (repo root). One node, single process,
imports each repo in its own subprocess/PYTHONPATH (the repos share colliding
`utils`/`networks`/`data`/`modules` package names).

Result (job `7251974`, node `x3001c0s13b1n0`): **`PROBE_OK`**. 4× A100-40GB visible,
torch sees CUDA (device_count=4), all core libs import, and **all four in-repo
models import** — S2S (`PanguModel_Plasim`), S2S-Lightning, SI, PanguWeather.
`makani`/`physicsnemo` are not yet importable (need their own installs — see §6).

## 4. Data availability on Polaris (the binding constraint)

| Dataset | On Polaris? | Consumers | Path |
|---|---|---|---|
| **E3SMv3 SSP245-AMIP** per-sample HDF5 | ✅ staged, readable | SI (AMIP), makani, physicsnemo, PanguWeather-SFNO | `/eagle/lighthouse-uchicago/members/jesswan/AI4SRM/data/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/h5/plev_data/{year}_{index:04d}.h5` (1460/yr, 2015–2049) + `normalize_mean.npz`/`normalize_std.npz` |
| **ERA5 HDF5** (`pangu_s2s_1979-2018_*.nc` + h5) | ❌ NOT staged | S2S, S2S-Lightning, PanguWeather-deterministic (`exp2.yaml`) | Midway had `/project/pedramh/h5data/h5data`; needs a Globus stage to eagle |

**Consequence:** the S2S / S2S-Lightning real-data smokes are **blocked on an ERA5
Globus stage**. Their PBS scripts are delivered and the launcher/env is proven by
the probe, but the 4-GPU *data* smoke cannot close until ERA5 lands on eagle. The
SI + SFNO models target the **staged E3SM data**, so they are the runnable path.

## 5. Per-model GREEN matrix (probe → 4-GPU)

> **Ladder note:** the handoff's ladder is probe → 1-GPU → 4-GPU. The **1-GPU rung was
> skipped** for the two green models — the probe already proved single-device torch/CUDA
> + imports, and both scripts take `-v NPROC=1` to run the 1-GPU rung on demand. Recorded
> here rather than implied.

| Model | Probe | 1-GPU smoke | 4-GPU smoke | Blocker |
|---|---|---|---|---|
| Toolchain probe | ✅ `PROBE_OK` | — | — | — |
| **PanguWeather SFNO** (torchrun) | ✅ | (via 4-GPU) | ✅ **GREEN** (job 7252271) | — runs on staged E3SM |
| **SI** (Lightning DDP) | ✅ imports | (via 4-GPU) | ✅ **GREEN** (job 7252700) | — runs on converted E3SM |
| S2S (torchrun) | ✅ imports | ⬜ | ⬜ | **ERA5 not staged** (Globus) |
| S2S-Lightning | ✅ imports | ⬜ | ⬜ | **ERA5 not staged** (Globus) |
| Makani SFNO | ✅ (venv) | ⬜ | 🟡 in progress | pack ✅ **built** (`CONVERT_OK`, job 7252736); training needs the SFNO venv (§6) |
| PhysicsNeMo SFNO | ✅ (venv) | ⬜ | ⬜ | converter authored (**zarr store not yet built**); hydra wiring unproven |
| PanguWeather deterministic | ✅ imports | ⬜ | ⬜ | PLASIM h5 not staged (NCAR glade) |

**PanguWeather SFNO 4-GPU smoke (GREEN, job 7252271):** one bounded epoch (year 2015,
1460 samples → 365 steps/rank on 4 A100), **train loss 0.3411**, validation ran
(`valid_loss 0.7049` + per-variable ACC/RMSE across cuda:0–3, so DDP all-reduce works),
`rc=0`, all ranks `DONE`. `HPC_scripts/polaris_train_e3sm_sfno.pbs` auto-preps the 16 GB
climatology (compute node) on first run. Two traps found & fixed:
- **`--debug` is single-GPU only** — it hardcodes `world_size=1`, so under
  `torchrun --nproc_per_node=4` all ranks became rank-0-on-GPU-0 and OOMed. Bound the
  smoke with **`--epochs 1`** instead (lets `setup_distributed` bind `device = rank %
  device_count`). A single rank is ~14.5 GB (fits 40 GB).
- **HDF5 file locking on Lustre** — the climatology CDF-5→NETCDF4 re-encode fails with
  `BlockingIOError: unable to lock file` unless `HDF5_USE_FILE_LOCKING=FALSE` (set in
  every script).

**SI 4-GPU smoke (GREEN, job 7252700):** the ~60 GB stage was converted by the *earlier*
job **7252286** (`CONVERT_OK`, 1464 files), which then died on the `noleap` calendar bug
below; **7252700** reused the cached stage (the `bench_polaris.pbs` staging step is
idempotent — it skips when `normalize_mean.nc` + `h5/2015_0000.h5` exist) and benched.
CSV row:

| n_gpus | batch/gpu | precision | step_med | samples_per_s_wall | peak_mem_gb_max_rank | steps |
|---|---|---|---|---|---|---|
| 4 | 1 | bf16-mixed | 0.400 s | 10.29 | **30.98** | 20 |

Peak memory matches the A100-40GB prediction (~31 GB) — batch 1 + bf16 fits with ~9 GB
headroom; do **not** raise batch_size without re-checking `peak_mem_gb_max_rank`.
Trap found & fixed: **`calendar: 'noleap'` crashes** the loader —
`noleap` is an *idealized* cftime calendar that forces `has_year_zero=True`, clashing
with `has_year_zero: False` at `amip_new.py:667` (`TypeError: cannot compute the time
difference between dates with year zero conventions`). Use `calendar: 'standard'`: the
smoke spans only 2015 + early-Jan 2016, which index identically (2015 is non-leap). A
full run crossing a leap year would need a loader fix (E3SM has 1460 files/yr always).
**Comparability caveats** (this is a bring-up smoke, NOT a Midway-comparable bench):
- Channel counts differ from the Midway bench (153 out/306 in, 18 levels, 3 diagnostics),
  so these numbers are same-order but NOT directly comparable to Midway
  `bench_results.csv` rows (the CSV records `config_sha16=2d0818b131b67f83`).
- The bench ran **warmup=5 / steps=20**, not the Midway convention of **20 / 80**
  (`bench_polaris.pbs` defaults). Raise both before quoting any throughput number.

⚠️ **Known latent bug — SI validation/rollout is broken with this config (not hit by the
smoke).** `si/modules/train_module.py` calls `disassemble_input(y_last, nlevels=...)`
relying on the hardcoded defaults `nsurface=6, ndiagnostic=15`; this config has **3**
diagnostics, so the channel split would be wrong. The bench never sees it because
`bench.py` forces `limit_val_batches=0`. Before running SI *training/validation* (as
opposed to the bench) on E3SM, plumb `ndiagnostic=len(diagnostic_variables)` and
`nsurface=len(surface_variables)` into that call — a shared-code change, so re-run both
the S2S and port smokes with it (CLAUDE.md rule #5).

## 6. SFNO framework installs (login node only — compute nodes have no network)

`makani` and `physicsnemo` are NOT in base conda. Install once on a **login node**
(the conda module sets an http proxy there); `--no-deps` protects base torch 2.8.0:

```bash
module use /soft/modulefiles && module load conda && conda activate base
# Makani:
pip install --user --no-deps 'makani @ git+https://github.com/NVIDIA/makani.git@c97043086e60d44a3adc3bede9a6b3dc71f5005d'
pip install --user nvidia-physicsnemo moviepy
# PhysicsNeMo (unified_recipe SFNO):
pip install --user warp-lang s3fs treelib
pip install --user --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda120
cd physicsnemo_sfno && pip install --user --no-deps -e .   # registers physicsnemo.models
pip install --user --no-deps 'makani @ git+https://github.com/NVIDIA/makani.git'  # 'SFNO' entry point
```

Each `polaris_*.pbs` preflights the imports and exits with `*_NOT_INSTALLED` + this
block if missing.

**STATUS (2026-07-14): installs DONE and verified.** `makani 0.2.0`,
`nvidia-physicsnemo 2.2.0a0` (editable, from the in-repo `physicsnemo_sfno/` tree —
this one install satisfies *both* physicsnemo's own example and makani's
`from physicsnemo.distributed.manager import DistributedManager`), `warp-lang 1.15.0`,
`nvidia-dali-cuda120 2.2.0`, `s3fs`, `treelib`, `moviepy`, plus **torch_harmonics
0.9.2.dev74 built from GitHub source** (§2 — the pinned PyPI releases cannot satisfy
makani). `import makani, physicsnemo, warp` all succeed, and Pangu/SI still import.

⚠️ Do **not** `pip install nvidia-physicsnemo` from PyPI without `--no-deps`: its
pyproject pins `torch>=2.10` and would upgrade the base torch 2.8 out from under every
model. The editable in-repo install (`--no-deps`) is the safe path (runtime only
enforces `torch>=2.4`).

## 7. Repointed-path map (per model)

| Model | Original (Midway/Derecho) | Polaris target | Config file |
|---|---|---|---|
| S2S / port | `data_dir=/project/pedramh/h5data/h5data` + `pangu_s2s_1979-2018_*.nc` | `/eagle/.../mehta5/era5_h5data/h5data` (**stage ERA5**) | `s2s/v2.0/config/exp2_polaris.yaml`, `test_polaris.yaml` |
| SI | `/project/pedramh/AMIP/h5` + `.nc` stats | `/eagle/.../mehta5/si_e3sm_stage/{h5,normalize_*.nc}` (converter output) | `si/configs/bench_polaris_e3sm.yaml` |
| PanguWeather SFNO | Derecho glade paths | E3SM `.../jesswan/AI4SRM/...` + `PanguWeather/v2.0/polaris_data/*.nc` (prep output) | `PanguWeather/v2.0/config/E3SM_SFNO_H5_POLARIS.yaml` |
| Makani SFNO | Stampede3 `$SCRATCH` | `/eagle/.../mehta5/data/e3sm_makani/{train,valid,test,stats,metadata}` (packer output) | `makani_sfno/polaris/e3sm_smoke.yaml` |
| PhysicsNeMo SFNO | ARCO-ERA5 zarr | `/eagle/.../mehta5/e3sm_seqzarr/e3sm_{train,val}.zarr` (converter output) | hydra CLI overrides on `unified_recipe/conf/config.yaml` |

All output/stage roots live on **eagle** (persistent), and every script sets
`TMPDIR`/`TORCHINDUCTOR_CACHE_DIR`/`TRITON_CACHE_DIR` → `/eagle/.../mehta5/{tmp,torchinductor_cache,triton_cache}`
(not node-local `/local/scratch`, which is wiped at job end).

## 8. Data-conversion recipes (Makani multifiles / PhysicsNeMo zarr / SI rename)

Three converters turn the E3SM per-sample h5 (`input` group, 162 float32 (180,360)
datasets: 8 upper-air × 18 plev + 18 surface; lat S→N; `normalize_{mean,std}.npz`
keys == h5 keys). Two data hazards every converter must handle:
- **16 zero-std keys** in `normalize_std.npz` (CLDLIQ ×8, CLDICE ×4, CLOUD ×4 — the
  condensate fields are identically zero in the upper stratosphere). SI and PhysicsNeMo
  clamp `std==0 → 1.0`; makani sidesteps it by computing its own stats from the packed
  split with a `1e-12` floor.
- **NaN masks**: ocean-only fields (`SST`, `ICE`) are NaN over land; land-only fields
  (`TOPO`, `PFTDATA_MASK`, `PCT_GLACIER`, `PCT_NATVEG`, `SOILWATER_10CM`, `TSOI_10CM`)
  are NaN over ocean. Every converter must fill them (SI does it via the config's
  `mask_fill`; makani and physicsnemo fill in the converter) or the model trains on NaN.

- **SI** — `si/convert_e3sm_for_si.py`: (A) `npz → normalize_{mean,std}.nc` with a
  `level` dim; (B) repack each h5 renaming upper-air keys `T_849.66… → T_850.0`
  (SI builds `f'{var}_{int(level)}.0'`). Stages 2015 + 2016_0000..0003 (~60 GB).
- **Makani** — `makani_sfno/polaris/convert_e3sm_to_makani.py`: packs into the PlaSim
  3-dataset contract `{split}/{year}.h5` (`fields_state (T,52,H,W)` + `fields_diagnostic
  (T,1,…)` + `forcing (T,6,…)` + timestamp/lat/lon dim-scales) + `stats/*.npy` +
  `metadata/data.json`. Rows flipped to **descending** lat; SST land-fill −1.8 °C.
- **PhysicsNeMo** — `physicsnemo_sfno/polaris/e3sm_h5_to_seqzarr.py`: SeqZarr store
  `predicted (T,157,H,W)` + `unpredicted (T,5,…)` + int-hours `time` (DALI cannot ingest
  datetime64/bytes) + means/stds; static/forcing fields
  (PCT_GLACIER/PCT_NATVEG/PFTDATA_MASK/TOPO/sol_in) → unpredicted. Fills the E3SM
  land/ocean NaN masks via `NAN_FILL` and **hard-fails** if any NaN lacks a fill entry;
  `--validate` does an exact round-trip on an unfilled channel + an all-finite gate.
  ⚠️ **Status: authored, store NOT yet built and the smoke NOT yet run** — the hydra
  SeqZarr/transform wiring (`curated_dataset.*`, `transform.transformed_shape`) was
  authored from the code map and needs live verification.

Each converter runs **inside its PBS job** (compute node); the makani/physicsnemo
smokes additionally require the §6 installs.

---

## Decisions / changes log

- **2026-07-14** — Confirmed Polaris facts (account `lighthouse-uchicago`, 4× A100-40GB
  sm80, `/local/scratch` 2.8 TB, `debug` queue, `filesystems=home:eagle`). Chose
  **base conda + `--user`** env strategy; pinned `torch_harmonics==0.7.4` (0.9.1
  ABI-breaks on torch 2.8). **Probe `7251974` = `PROBE_OK`** (gate 1 green): 4 GPUs +
  all 4 in-repo models import. Found ERA5 is **not** staged (S2S/port blocked on a
  Globus stage); E3SM AMIP data **is** staged (SI/SFNO path is runnable).
