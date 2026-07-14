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

**Compute-node networking — CORRECTED 2026-07-14.** Several places in this repo said
"compute nodes have no outbound network". **That is wrong.** Per ALCF's docs the proxy is the
*only* route out, but it does work from compute nodes:

```bash
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"
```

`module load conda` already exports `http_proxy`/`https_proxy` (not `ftp_proxy`), so every
`polaris_*.pbs` has them. Measured on-node (job **7253810**, `x3206c0s7b1n0`):

| from a compute node | result |
|---|---|
| `https://api.wandb.ai/healthz` via proxy | **HTTP 200** |
| `https://pypi.org/simple/` via proxy | **HTTP 200** |
| same, with the proxy unset | **HTTP 000** (no route) |

Consequences: **W&B can log online from a job** (`qsub -v WANDB_MODE=online …`, see §9), and
a pip install from a job *would* work — we still keep installs on the login node because they
waste allocation, not because they are impossible.

> ⚠️ A trap this session walked into: the FIRST version of that probe called
> `HTTP 404` a failure. A 404 is the server *answering* — it is proof of connectivity, and
> `urllib` raising `HTTPError` means the same. The probe reported "compute nodes CANNOT reach
> W&B" while its own output showed they could. Read what the check actually measured.

**W&B online logging: PROVEN from a compute node** (job **7253823**, `x3206c0s7b1n0`). A real
run uploaded live through the proxy and was read back from the server afterwards:
`polaris-connectivity-check`, state `finished`, 5 history points —
`wandb.ai/rmehta1987-the-university-of-chicago/pedramh-profiling/runs/rh6142uo`. So
`qsub -v WANDB_MODE=online …` works; offline remains the default (a network hiccup can never
fail a run, and a preempted job still leaves a clean local record to `wandb sync` later).

> Two traps found while proving it, both worth knowing:
> * **`wandb.Api().viewer` is a PROPERTY, not a method.** Calling `viewer()` raises
>   `TypeError: 'User' object is not callable` — which reads exactly like an auth failure but
>   is auth *succeeding*. Our setup script had this bug; fixed.
> * **A bare key in `~/.netrc` is not a netrc.** wandb writes
>   `machine api.wandb.ai / login user / password <key>`; a raw token on line 1 makes
>   `netrc.netrc()` raise `NetrcParseError` **with the key in the message** — i.e. the
>   malformed file leaks the secret into any log that parses it. Keep `~/.netrc` at mode 600
>   and in proper format; prefer `wandb login` over hand-editing.

**PBS output-file gotchas (both bit us):**
1. `#PBS -o polaris_logs/` (a *directory*) makes PBS write `polaris_logs/<full_jobid>.OU`
   (with `-j oe`), **not** `<jobname>.o<seq>` — this is why the probe's log is named
   `polaris_logs/7251974….OU`.
   **The per-model scripts no longer pass `-o` at all** (an absolute path can only ever name
   ONE user's dir, so a second member's job could not write its own log). With `-j oe` and no
   `-o`, PBS writes `<jobname>.o<jobid>` into the SUBMIT dir — per-user for free.
2. **HISTORICAL (pre-`7eacdb31`): PBS APPENDS to a fixed `-o` path** — it does not truncate,
   so the older `polaris_logs/<name>.log` archives accumulate *several jobs* each. Current
   runs get one file per job, so this only matters when reading the old archives.
   `pangu_e3sm_sfno.log` holds the OOM run 7252261 **and** the green 7252271;
   `si_polaris_bench.log` holds *three* runs — the crashed 7252286, the green 7252700, and a
   later green re-run 7252946 (rc=0, step_med 0.399). **When
   reading these logs, always anchor on the `PBS_JOBID=` header of the run you care
   about** — a naive `grep -c OutOfMemory` on the file will report a green run as failed.

## 2. Environment strategy — ALCF base conda + a SHARED top-ups dir

Decision: **use the ALCF base conda** (fastest per handoff) and add the few packages it
lacks. No fresh env was built — base already carries a CUDA-12.9-matched torch and Lightning.

**Those top-ups must NOT be `pip install --user`.** That was the original decision and it was
wrong: ALCF homes are mode `0700`, so `--user` packages are readable by one person and every
other member's job dies on `ModuleNotFoundError` (see the box below and the decisions log).
They now live in a shared, world-readable `$POLARIS_TOPUPS`, built by
`polaris_setup_base_topups.sh`.

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

**Base-conda top-ups** — `polaris_setup_base_topups.sh`, once on the **login node**
(compute nodes have no outbound network; the conda module sets `http_proxy` for the login
node). PASS = `TOPUPS_OK`. They install to the **shared, world-readable**
`$POLARIS_TOPUPS` = `<member>/conda-envs/polaris-topups`, and Pangu/SI/S2S prepend it to
`PYTHONPATH` themselves.

> ⚠️ **This was originally `pip install --user`, and that was a silent, project-wide bug**
> (found 2026-07-14 by the cold audit; see the decisions log). `--user` lands in
> `PYTHONUSERBASE=/home/<installer>/.local/...`, and **ALCF home dirs are mode `0700`** — so
> those packages were readable by exactly one person. Every "GREEN" Pangu/SI result was green
> *only for rmehta1987*; a second member running the identical script got
> `ModuleNotFoundError: No module named 'torch_harmonics'`
> (`networks/modulus_sfno/sfnonet.py:24`, a bare import) or `netCDF4`
> (`utils/data_loader_multifiles.py:71`). That defeats the entire point of the deliverable.
> **Never `pip install --user` a dependency the project is supposed to share.**
>
> Two traps in the fix itself, both now guarded in the script:
> 1. **`pip install --target` cannot see the base conda**, so it re-resolves *everything* —
>    the first attempt silently pulled **torch 2.13.0 + a CUDA-13 stack + numpy 2.5.1**
>    (4.1 GB). Since `$POLARIS_TOPUPS` goes on `PYTHONPATH`, which *outranks*
>    site-packages, that torch would have **shadowed the base's torch 2.8.0/cu12.9** and
>    moved every smoke onto an untested toolchain. Fixed with `--no-deps` + only the four
>    deps base genuinely lacks (`cftime`, `numcodecs`, `asciitree`, `fasteners`); the
>    script now hard-fails if `torch`/`numpy`/`nvidia`/`triton` land in the target.
>    **64 MB** (cartopy dominates), and it asserts torch/numpy still resolve to base.
> 2. `$POLARIS_TOPUPS` **must not** be added to `PYTHONPATH` globally in `polaris_env.sh`:
>    its torch_harmonics **0.7.4** would shadow the SFNO venv's **0.9.x** and re-break
>    makani. `PYTHONNOUSERSITE=1` does *not* block `PYTHONPATH`. Both SFNO scripts now
>    assert torch_harmonics resolves inside their venv (`ERROR TORCH_HARMONICS_SHADOWED`).

Contents (all verified importable with the user site DISABLED, i.e. as another member):

| Package | Version | Needed by | Note |
|---|---|---|---|
| `netCDF4` | 1.7.4 | S2S/SI/Pangu `.nc` stats | base lacks it |
| `h5netcdf` | — | xarray `.nc` backend | |
| `zarr` | 2.18.7 (`<3`) | PhysicsNeMo zarr store | pinned <3 |
| `torch_harmonics` | **0.7.4** (top-ups dir, NOT base conda) | S2S-family, Pangu-SFNO, SI | the SFNO frameworks need 0.9.x — see the version box + §6 venv |

**The `torch_harmonics` version box (a genuine 3-way version squeeze):**
- `0.9.1` **wheel** → `import torch_harmonics` dies: `undefined symbol:
  _ZNK3c1010TensorImpl15incref_pyobjectEv` in `attention/_C.so` (built against a different
  torch ABI than 2.8). It also has **no sdist on PyPI**, so `--no-binary :all:` cannot
  build it.
- `0.7.4` / `0.8.0` → import fine, but expose only the *private* `_precompute_latitudes`;
  **makani 0.2.0 does `from torch_harmonics.quadrature import precompute_latitudes`**
  (`makani/utils/grids.py:20`) and fails. Verified: `def precompute_latitudes` is present in
  the venv's 0.9.2a `torch_harmonics/quadrature.py` and absent from 0.7.4. Note it is a
  **submodule** symbol — `from torch_harmonics import precompute_latitudes` fails even on
  0.9.2a, so probe `torch_harmonics.quadrature`, not the top-level package.
- Only a **GitHub source build** gives both (public API + an ABI-matched `_C`).
- ✅ **Resolution — split the envs (§6):** the base top-ups keep **0.7.4**, which is what the
  GREEN Pangu-SFNO (7252271) and SI (7252700) smokes actually ran on; the SFNO frameworks
  get an **isolated venv** carrying the source-built **0.9.2a**. This avoids re-validating
  the two greens against a new torch_harmonics, at the cost of one extra env.

## 3. Toolchain probe (gate 1) — **GREEN**

`polaris_probe.pbs` + `polaris_probe.py` (repo root). One node, single process,
imports each repo in its own subprocess/PYTHONPATH (the repos share colliding
`utils`/`networks`/`data`/`modules` package names).

Result (job `7251974`, node `x3001c0s13b1n0`): **`PROBE_OK`**. 4× A100-40GB visible,
torch sees CUDA (device_count=4), all core libs import, and the four in-repo models import.
`makani`/`physicsnemo` are not yet importable (they live in the §6 venv — non-blocking).

**Re-run 2026-07-14 after two corrections — job `7253681`, and only this one is worth
quoting:**
1. The port check imported `common, data, modules`, which have **no `__init__.py`** — they
   are namespace packages, so it executed none of the smoke's code and would pass over any
   missing import. It now imports **`modules.train_module`**, the module the entrypoint
   actually loads. (This is exactly what let the missing `cf_xarray` hide behind a green
   `PROBE_OK` — see the §2 box.)
2. The probe resolved its imports from the author's private `~/.local`, so `PROBE_OK` was a
   statement about one account. It now imports through `$POLARIS_TOPUPS`.

`7253681` ran with `PYTHONNOUSERSITE=1` and reports:
`sys.path is free of ~/.local -> imports below are reproducible by any member`,
`[OK ] S2S-Lightning  modules.train_module`, then **`PROBE_OK`**.

## 4. Data availability on Polaris (the binding constraint)

| Dataset | On Polaris? | Consumers | Path |
|---|---|---|---|
| **E3SMv3 SSP245-AMIP** per-sample HDF5 | ✅ staged, readable | SI (AMIP), makani, physicsnemo, PanguWeather-SFNO | `/eagle/lighthouse-uchicago/members/jesswan/AI4SRM/data/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/h5/plev_data/{year}_{index:04d}.h5` (1460/yr, 2015–2049) + `normalize_mean.npz`/`normalize_std.npz` |
| **ERA5 HDF5** (`pangu_s2s_1979-2018_*.nc` + h5) | ❌ NOT staged | S2S, S2S-Lightning, PanguWeather-deterministic (`exp2.yaml`) | Midway had `/project/pedramh/h5data/h5data`; needs a Globus stage to eagle |

**Consequence:** the S2S / S2S-Lightning real-data smokes are **blocked on an ERA5
Globus stage**. Their PBS scripts are delivered and the **import chain is verified**
(the probe now imports `modules.train_module` — the module the entrypoint really loads),
but the 4-GPU *data* smoke has **never executed on Polaris** and cannot close until ERA5
lands on eagle. Do not read "delivered" as "proven".

> ⚠️ **The old wording here said the port's env was "proven by the probe". It wasn't.**
> The probe imported `common, data, modules` — and those dirs have **no `__init__.py`**, so
> they are *namespace packages*: the import succeeds without running a line of the smoke's
> code. It therefore never touched `s2s-lightning/modules/train_module.py:52`, a bare
> `import cf_xarray` that was missing from the base conda, from the top-ups, and from every
> `~/.local` — i.e. **broken for everyone**. Both port entrypoints reach it
> (`smoke_train_module.py:20`, `bench.py:67`), so the port would have died at import the
> moment ERA5 landed, *after* a multi-TB Globus stage, while this doc insisted only data was
> missing. Fixed: `cf_xarray` is in the top-ups and the probe imports the real entrypoint.
> **Lesson: a green import check that imports a namespace package proves nothing.**

The
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
| **PanguWeather SFNO** (2nd-user *simulation*) | — | — | ✅ **7253591** — rc=0, loss **0.3411 = identical** to 7253401 | see the caveat below |
| **SI** (2nd-user *simulation*) | — | — | ✅ **7253603** — rc=0, step_med 0.399 / peak 30.69 GB (vs 0.400 / 30.98 as-installer: noise) | torch_harmonics now resolves; it previously warned "could not be imported" and silently degraded |

> **What "2nd-user" above does and does NOT mean.** These ran from **rmehta1987's account**
> with `PYTHONNOUSERSITE=1`, which removes `/home/rmehta1987/.local` from `sys.path` and so
> reproduces *the dependency-resolution view* another member gets. That is what makes them
> evidence about the private-deps bug, and it is genuinely strong evidence: Pangu came back
> **bit-identical** (0.3411).
> It is **not** the same as jesswan running them. Still untested by simulation: her UID's
> write permissions, her `MEMBER_ROOT` resolution, her quota, her fresh clone, and her own
> SFNO venv build. **The first real second-user run is still jesswan's** — treat her first
> attempt as the actual test, not a formality.
| **guard regression test** | — | — | ✅ Pangu **7253616** (rc=0, loss 0.3411) + SI **7253627** (rc=0, step_med 0.3985) — the COMMITTED scripts, with `polaris_require_topups` in the launch path | the guard is silent on a good env and costs nothing; both failure branches verified by hand on the login node: `POLARIS_TOPUPS=/nonexistent` → `ERROR TOPUPS_MISSING` rc=3, and unsetting `PYTHONPATH` (which reproduces the original bug) → `ERROR PRIVATE_DEPS_ON_PATH` rc=3 naming all four offending packages |
| **Makani SFNO** (venv) | ✅ | ⬜ | ✅ **GREEN** (job **7253465** on the current script, train 2.61 / val 2.38; first green 7252769 pre-rework) | — pack ✅ `CONVERT_OK` (7252728) |
| **PhysicsNeMo SFNO** (venv) | ✅ | ✅ **GREEN** (7252816) | ✅ **GREEN** (job 7252933, `rc=0`) | — zarr store ✅ `CONVERT_OK` |
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

⚠️ **Latent bug — `disassemble_input`'s hardcoded channel split. FIXED in
`train_module.py` (commit `1fef2473`); STILL OPEN in four other call sites.**
`disassemble_input` defaults to `nsurface=6, ndiagnostic=15` — silently baked to the Midway
AMIP config. The E3SM config has **3** diagnostics, so any caller that relies on the
defaults splits the channels wrongly and produces *plausible but wrong* tensors rather than
raising. `si/modules/train_module.py:269` now passes the real
`len(self.surface_variables)` / `len(self.diagnostic_variables)` / `self.nlevels`.

The bench never exercised it (`bench.py` forces `limit_val_batches=0`), which is why the
GREEN 7252700 is still valid. **These callers remain unfixed** and will mis-split on E3SM:

| Call site | Passes |
|---|---|
| `si/bias.py:226` | `nlevels` only |
| `si/modules/ae_module.py:68` | `nlevels` only |
| `si/modules/combined_module.py:185` | `nlevels` only |
| `si/modules/combined_module.py:287` | `nlevels` only |

(`si/modules/models/old/*` is dead code — out of scope.) Fix these before running SI
*training/validation or the bias/AE/combined paths* on E3SM. The real repair is to make
`disassemble_input` **require** the counts instead of defaulting them, so a missed caller
fails loudly; that is a shared-code change → re-run both the S2S and port smokes
(CLAUDE.md rule #5).

### Trap: makani's smoke re-run is a silent no-op (`rc=0`, zero steps)

`train_plasim` **auto-resumes** from `${EXP_DIR}/e3sm_smoke/<run_num>/training_checkpoints/
ckpt_mp0_v0.tar` whenever it exists. The smoke sets `max_epochs: 1`, which a finished run's
checkpoint has *already satisfied* — so a second run with the same `--run_num` loads it,
prints `Total training time is 0.00 sec`, trains **nothing**, and still exits **`rc=0`**.
Job **7253454** looked green this way and proved nothing.

Two fixes, both in `makani_sfno/polaris/polaris_sfno_smoke.pbs`:
- `RUN_NUM` now defaults to `${PBS_JOBID%%.*}` (was a hardcoded `0`), so every job trains
  from scratch into its own attributable run dir.
- A **PASS gate** after the launcher: `rc=0` with no `ckpt_mp0_v0.tar` written by *this*
  job's `RUN_NUM` is forced to `rc=4` with `ERROR NO_CHECKPOINT`.

**Generalise this** (CLAUDE.md "never claim a step passed without reading the output"):
`rc=0` is not a PASS criterion for any resumable trainer. Key on the work token — a loss
line, a written checkpoint — not the exit code.

Note also: the smoke has **no seed knob** (that's a DESIGN §4.0 prerequisite), so its losses
move run-to-run — 7252769 gave 2.19/2.05 and 7253465 gave 2.61/2.38 on identical code. Treat
these as "finite and roughly O(1)", never as an equivalence baseline.

### Known gap: the *inference* entrypoints are not import-clean

`s2s/v2.0/inference.py:21` and `PanguWeather/v2.0/long_inference.py:34` both do a bare
`import dask`, which is in neither the base conda nor `$POLARIS_TOPUPS`. **No Polaris script
launches either file** — this bring-up delivers *training* smokes — so `dask` is deliberately
not in the top-ups (it is a heavy dep, and adding unused packages to a dir that sits on
everyone's `PYTHONPATH` is its own risk).

Recorded because it is the same shape as the `cf_xarray` bug (§2): an unrun path's missing
import is invisible until someone runs it. **Before the first Polaris inference run**, add
`dask` to `polaris_setup_base_topups.sh` and check the chain with:

```bash
cd s2s/v2.0 && PYTHONNOUSERSITE=1 PYTHONPATH=.:$POLARIS_TOPUPS python -c "import inference"
```

## 6. SFNO frameworks — an ISOLATED venv (`polaris_setup_sfno_venv.sh`)

`makani` + `physicsnemo` do **not** run in the base conda: makani needs the *public*
`torch_harmonics.quadrature.precompute_latitudes`, which only exists in 0.9.x, while the
base must keep **0.7.4** (the version the GREEN Pangu/SI smokes ran on — see the §2
version box). Resolution: a dedicated venv, built once on a **login node** (compute nodes
have no outbound network):

```bash
bash polaris_setup_sfno_venv.sh          # PASS = "SFNO_VENV_OK"
# -> ${MEMBER_ROOT}/conda-envs/sfno-venv   (YOUR member dir — e.g. .../members/jesswan/...;
#    override with POLARIS_SFNO_VENV=<dir>. It must be yours: physicsnemo is installed
#    editable, so a shared venv would import someone else's checkout — see the guard in
#    physicsnemo_sfno/polaris/polaris_sfno_smoke.pbs.)
```

It is a `--system-site-packages` venv layered on the base conda, so it **inherits the
CUDA-12.9-matched torch 2.8** (no 2.5 GB reinstall) and adds (see the script for the full
pinned list, which also includes **mlflow** — §8 explains why it is not optional — plus
`tensorly-torch`): `torch_harmonics`
**0.9.2a built from GitHub source** (ABI-matched `_C` + the public API), `makani 0.2.0`
(pinned `c970430…`, mandated by the makani_sfno README), `nvidia-physicsnemo 2.2.0a0`
(editable, from the in-repo `physicsnemo_sfno/` tree — one install satisfies *both*
physicsnemo's own example and makani's `from physicsnemo.distributed.manager import
DistributedManager`), `warp-lang`, `nvidia-dali-cuda120`, `s3fs`, `treelib`, `moviepy`,
`tensorly`, `tensorly-torch`. Verified: `SFNO_VENV_OK` + a provenance gate asserting
`torch_harmonics`/`makani` resolve **from the venv**.

**Four traps this env cost us — all encoded in the scripts:**
1. **User-site shadowing.** `--system-site-packages` re-enables the USER site (`~/.local`),
   and `site.py` adds it **before** the venv's own site-packages — so the base's `--user`
   torch_harmonics 0.7.4 shadowed the venv's 0.9.x and makani still failed with
   `cannot import name 'precompute_latitudes'`. Fix: **`PYTHONNOUSERSITE=1`** in the venv
   *and* in both SFNO PBS scripts. (The conda base is the SYSTEM site, so torch survives.)
2. **`torchrun` is the wrong launcher here.** torch is inherited from the conda base, so
   the venv has no `torchrun` of its own; the bare name resolves to the **base** conda
   `torchrun`, whose shebang pins the **base** python — the spawned ranks then die with
   `No module named 'makani'`. Use **`python -m torch.distributed.run`**.
3. **`pip install tltorch` does not exist** — the module `tltorch` ships as the PyPI
   package **`tensorly-torch`**. (It silently "worked" only by leaking from the user site.)
4. **Never `pip install nvidia-physicsnemo` from PyPI without `--no-deps`**: its pyproject
   pins `torch>=2.10` and would upgrade the base torch 2.8 out from under every model. The
   editable in-repo install (`--no-deps`) is the safe path (runtime only enforces `torch>=2.4`).

**Upstream makani bug (patched in our wrapper, not in makani):** at pin `c970430…`,
`utils/driver.py` assigns `self.logger` **only when `log_to_screen` is truthy** (makani
disables it on non-zero ranks), but `utils/training/deterministic_trainer.py` then calls
`self.logger.info("No channels to visualize, skipping visualization.")` **unconditionally**
whenever `init_visualizer()` returns `None` — true for any config without visualization
channels, including our smoke. Every non-zero rank dies with
`AttributeError: 'PlasimTrainer' object has no attribute 'logger'`. Fixed by setting
`self.logger` before `super().__init__()` in `sfno_training/trainer/plasim_trainer.py`
(the wrapper whose stated job is patching stock makani; zero edits to makani itself).

## 6b. Repo layout: the 3 SFNO codebases are **git subtrees** (full provenance)

`PanguWeather/`, `makani_sfno/` and `physicsnemo_sfno/` are **not** separate checkouts any
more (the handoff assumed they were) and **not** submodules. They are `git subtree`
imports, added **unsquashed** so the complete upstream history is preserved:

| prefix | imported from | upstream authorship reachable |
|---|---|---|
| `PanguWeather/` | `git@github.com:envfluids/PanguWeather.git` (`sfno_e3sm`) | jesswan-uc (8) |
| `makani_sfno/` | `git@github.com:feynmanliu214/SFNO_Climate_Emulator.git` (`snapshot/sfno-climate-emulator`) | feynmanliu214 (38) |
| `physicsnemo_sfno/` | `https://github.com/awikner/physicsnemo.git` (`main`) | ktangsali (203) + NVIDIA |

Each landed as a real **merge commit with two parents**, so upstream commits are genuine
ancestors of `HEAD` (`git merge-base --is-ancestor <their-sha> HEAD` succeeds). Cost:
the repo went **313 → 4,769 files**, **.git 3.9 MB → 306 MB**. That is the deliberate
price of full provenance.

### Merging in both directions

**Them → us** (take upstream's new work):
```bash
git subtree pull --prefix=makani_sfno git@github.com:feynmanliu214/SFNO_Climate_Emulator.git snapshot/sfno-climate-emulator
git subtree pull --prefix=PanguWeather git@github.com:envfluids/PanguWeather.git sfno_e3sm
git subtree pull --prefix=physicsnemo_sfno https://github.com/awikner/physicsnemo.git main
```
Conflicts are ordinary git conflicts, only where we touched the same lines.

**Us → them** (send our fixes upstream). We have **no write access** to those repos, so:
```bash
git subtree split --prefix=makani_sfno -b makani-polaris   # history of ONLY that subdir,
                                                           # re-rooted as if it were the repo
```
then push `makani-polaris` to a fork you control and open a PR (or hand them the branch /
`git format-patch` series). Do NOT `git subtree push` to their URLs — we cannot write there.

⚠️ **Keep the subtrees pristine.** Upstream ships files this repo would normally reject
(13 `*.npy`, 50 `*.o`/`*.e` job logs). They came in with the history and are **already in
`.git` permanently** — deleting them from the tree would NOT shrink the clone, and would
make every future `subtree pull` fight to re-add them and every `subtree split` look like
it deletes upstream's files. So they stay, as a **deliberate, informed exception to
CLAUDE.md rule #8**, scoped to imported third-party history. Each subtree's own
`.gitignore` stops *new* junk (`polaris_data/`, `core.*`, `checkpoints/`, `mlruns/`,
per-channel validation PNGs).

## 7. Repointed-path map (per model)

| Model | Original (Midway/Derecho) | Polaris target | Config file |
|---|---|---|---|
| S2S / port | `data_dir=/project/pedramh/h5data/h5data` + `pangu_s2s_1979-2018_*.nc` | `/eagle/.../mehta5/era5_h5data/h5data` (**stage ERA5**) | `s2s/v2.0/config/exp2_polaris.yaml`, `test_polaris.yaml` |
| SI | `/project/pedramh/AMIP/h5` + `.nc` stats | `/eagle/.../mehta5/si_e3sm_stage/{h5,normalize_*.nc}` (converter output) | `si/configs/bench_polaris_e3sm.yaml` |
| PanguWeather SFNO | Derecho glade paths | E3SM `.../jesswan/AI4SRM/...` + `$PANGU_AUX/*.nc` = `/eagle/.../mehta5/pangu_polaris_data` (prep output, shared read-only) | `PanguWeather/v2.0/config/E3SM_SFNO_H5_POLARIS.yaml` |
| Makani SFNO | Stampede3 `$SCRATCH` | `/eagle/.../mehta5/data/e3sm_makani/{train,valid,test,stats,metadata}` (packer output) | `makani_sfno/polaris/e3sm_smoke.yaml` |
| PhysicsNeMo SFNO | ARCO-ERA5 zarr | `/eagle/.../mehta5/e3sm_seqzarr/e3sm_{train,val}.zarr` (converter output) | `physicsnemo_sfno/examples/weather/unified_recipe/conf/config_e3sm_sfno.yaml` (+ CLI overrides) |

All output/stage roots live on **eagle** (persistent). Every per-model `*.pbs` sources
`polaris_env.sh`, which sets `TMPDIR`/`TORCHINDUCTOR_CACHE_DIR`/`TRITON_CACHE_DIR` →
`${MEMBER_ROOT}/{tmp,torchinductor_cache,triton_cache}` — **your own** member dir, resolved
per user, not a hardcoded `mehta5` (not node-local `/local/scratch`, which is wiped at job
end). `polaris_probe.pbs` sources it too. (Until 2026-07-14 five scripts — the S2S/port pair
and the PLASIM one — set none of this; the audit caught the doc claiming otherwise.)

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
  ✅ **Store PROVEN on the real data** (job 7252780): `validation: max|zarr-h5|
  = 0.000e+00 … sample0 all-finite = True` + `CONVERT_OK` for both the 64-sample train
  and 16-sample val stores (`predicted=157 unpredicted=5`, 180×360).

  **PhysicsNeMo training traps** (each cost a job; all now encoded):
  - **`model=tiny_sfno` on the CLI is impossible.** The recipe's `defaults:` use hydra's
    *path* form (`- model/tiny_afno`), which registers no overridable `model` **group** →
    `Could not override 'model'. No match in the defaults list.` Fix: the 2-line
    `conf/config_e3sm_sfno.yaml` (tiny_afno→tiny_sfno, training/afno→training/sfno) and
    `--config-name=config_e3sm_sfno` with *value* overrides only.
  - **mlflow is NOT optional** — `train.py` calls `initialize_mlflow()` unconditionally and
    `physicsnemo/utils/logging/mlflow.py` **raises** `ImportError` without it (it does not
    degrade to a no-op). Then mlflow 3.x **refuses its own `./mlruns` file store**
    ("maintenance mode") → `export MLFLOW_ALLOW_FILE_STORE=true`.
  - **Debug via the 1-GPU rung** (`qsub -v NPROC=1`): it uses plain `python`, so the real
    traceback reaches the log. Under `torchrun` the child stderr is swallowed and you only
    get a bare `ChildFailedError` with `error_file: <N/A>` — hence `--redirects/--tee 3`.
  - **`datapipe.parallel=false` is a broken "fallback"** — `seq_zarr_datapipe.py` passes
    `prefetch_queue_depth` to `dali.fn.external_source` unconditionally and DALI rejects it
    unless `parallel=True`. Leave the datapipe defaults alone.
  - **`validation.num_steps` must be ≥ 2** — `train.py`'s post-validation plotting indexes
    `ax[0, t]`, but matplotlib squeezes `plt.subplots(...)` to a 1-D axes array when a grid
    dim is 1, so `num_steps=1` dies *after* a successful epoch with `IndexError`.
  - **`dataset.dataset_filename` must be repointed too** — it is the RAW (pre-curation)
    store; a trailing step opens it even though we supply pre-curated stores, and its stock
    value is a nonexistent relative ARCO-ERA5 path → `zarr PathNotFoundError` *after* a
    successful train+validate+checkpoint.

  **PhysicsNeMo SFNO smokes (GREEN):** 1-GPU job 7252816 (`rc=0`, loss 1.082, validation
  error 0.776) and **4-GPU job 7252933** (`rc=0`, 4 ranks `[default0..3]`, loss 0.889,
  validation error 0.541), both writing `checkpoint.0.0.pt` +
  `SphericalFourierNeuralOperatorNet.0.0.mdlus`. fp32 — the recipe's `training/sfno` sets
  `amp_supported: False` on purpose, so do NOT force bf16.

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
