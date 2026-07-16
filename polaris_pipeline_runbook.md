# Running the three E3SM model-training pipelines on Polaris

*A step-by-step guide covering all three models — **PanguWeather**, **makani**, and
**PhysicsNeMo**. Written 2026-07-16; **all four confirmation runs (§2) passed the same
day**, so every command below sits on a demonstrated-green software path. The
engineering companion (costs, failure-mode analysis, open decisions) is
`polaris_pipelines_plan.md`.*

---

## 1. What these pipelines do

Each pipeline trains a neural network to advance a simulated atmospheric state forward
by six hours: the model receives the global state at time *t* and learns to reproduce
the state at *t + 6 h*. All three models are variants of the **Spherical Fourier
Neural Operator (SFNO)** — a network that operates on fields defined on the sphere —
implemented in three different code bases whose relative performance this project
measures.

**Source data (shared by all three).** A 35-year E3SM version 3 atmosphere simulation
(SSP2-4.5 scenario with prescribed ocean, years 2015–2049), archived as one HDF5 file
per 6-hour interval — 1,460 files per year, 51,100 in all — on a 1° × 1° global grid
(180 × 360 points). The archive lives, read-only, at:

```
/eagle/lighthouse-uchicago/members/jesswan/AI4SRM/data/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/h5/plev_data
```

**Variables.** Of the archive's 162 fields, all three pipelines retain **108** — the
three cloud condensate variables (54 fields across 18 model levels) are excluded from
every model by decision of the science owner (2026-07-16). The three models partition
those 108 fields slightly differently (which fields are forecast versus merely
supplied), which is why each has its own data-preparation step:

| model | data preparation | prepared dataset |
|---|---|---|
| PanguWeather | **none** — reads the archive directly | — (plus a small shared statistics/climatology directory, reused automatically) |
| makani | repacks into per-year HDF5 files (100 forecast + 1 diagnosed + 7 supplied fields) | ~1.43 TB, built by a batch job |
| PhysicsNeMo | rewrites into two Zarr stores (103 forecast + 5 supplied fields) | ~1.43 TB — **this is the transferred dataset** |

Fields that E3SM masks with missing values over land or ocean are filled with fixed
constants during preparation, and those constants are recorded inside each prepared
dataset.

---

## 2. The confirmation tests (all passed 2026-07-16)

Every stage below has a brief confirmation run (a "smoke test": the complete software
path exercised on a small data subset, minutes instead of days). **Success is always a
specific printed line in the job's log — never just a clean exit code.** The full
sequence was run 2026-07-16 and **all four passed**:

| test | job | result |
|---|---|---|
| PanguWeather validation-memory test | 7259271 | ✅ `PANGU_VAL_SMOKE_OK` — validation peaks at 25.0 of 39.5 GiB per GPU (13.9 GiB headroom); the previously feared out-of-memory failure mode is refuted |
| PanguWeather training test (108 fields) | 7259296 | ✅ `ALLDATA_SMOKE_OK` — peak 27.0 GB, median step 0.64 s |
| PhysicsNeMo pipeline test (103+5 fields) | 7259303 | ✅ `ALLYEARS_SMOKE_OK` — year-seam data verified bit-for-bit; the per-step metrics file also confirmed working (`PHYSICSNEMO_CSV_OK`) |
| makani pipeline test (100/1/7 fields) | 7259321 | ✅ `ALLDATA_SMOKE_OK` — fresh 108-field pack built and trained; the "expected 58" warning in its log is a benign, by-design notice |

To re-run any of them (recommended after pulling new code), each model's section below
names its test.

---

## 3. One-time setup (per person)

Any member of the `lighthouse-uchicago` project can run these pipelines. Three things
are per-person:

1. **Your own copy of the repository, updated to the newest code.** The scripts
   resolve all writable paths to *your* member directory automatically, but they must
   be *submitted from your own checkout*. jesswan's copy is at:

   ```
   /lus/eagle/projects/lighthouse-uchicago/members/jesswan/pedramh-profiling
   ```

   **Getting the newest code from GitHub.** The repository is
   `https://github.com/rcc-uchicago/pedramh-profiling`, and everything this guide
   describes lives on the branch **`polaris-data-prep`** (an open pull request; once
   it is merged, substitute `main` below). On a login node:

   ```
   # if you already have a copy (jesswan does):
   cd /lus/eagle/projects/lighthouse-uchicago/members/jesswan/pedramh-profiling
   git fetch origin
   git checkout polaris-data-prep
   git pull origin polaris-data-prep

   # if you are starting from nothing:
   cd /lus/eagle/projects/lighthouse-uchicago/members/<you>
   git clone https://github.com/rcc-uchicago/pedramh-profiling.git
   cd pedramh-profiling
   git checkout polaris-data-prep
   ```

   Two practical notes. First, after a `git pull`, updated code takes effect
   immediately — the Python environment links to your checkout rather than copying
   from it, so no reinstallation is needed (rebuild the environment only if
   `polaris_setup_sfno_venv.sh` itself changed). Second, if you ever *push* from a
   Polaris login node, use `git -c pack.threads=1 push` — the login nodes cap
   per-user processes and multi-threaded git pushes are killed mid-transfer
   (ordinary `git pull` is not affected).

2. **Your own Python environment for makani and PhysicsNeMo.** Build it once, on a
   login node (about 15 minutes; success line `SFNO_VENV_OK`):

   ```
   cd /lus/eagle/projects/lighthouse-uchicago/members/jesswan/pedramh-profiling
   bash polaris_setup_sfno_venv.sh
   ```

   This is deliberately per-person: the environment links the PhysicsNeMo source in
   your own checkout, and the batch scripts refuse to run against somebody else's
   working copy (error `PHYSICSNEMO_WRONG_CHECKOUT`) so that no one unknowingly
   trains with another person's uncommitted edits.

3. **Nothing extra for PanguWeather** — it uses the system-provided Python plus a
   shared, group-readable package directory that the scripts locate automatically
   (they fail loudly with `TOPUPS_MISSING` if it is ever absent, with the fix printed).

All compute runs through the PBS batch scheduler (`qsub` to submit, `qstat -u $USER`
to watch, `qdel` to cancel). Nothing heavy runs on login nodes. Job logs are written
to the directory the job was submitted from (file `<job-name>.o<job-id>`), except
where a script states otherwise. The `debug` queue (used by the tests) allows one job
per person, up to one hour; long jobs use `preemptable` and resume automatically.

**Experiment tracking (Weights & Biases).** All jobs default to *offline* tracking —
nothing is sent anywhere and no account is needed. For **PanguWeather and makani**,
live dashboards are one flag away: submit any training job with
`qsub -v WANDB_MODE=online <script>` and the launcher enables the config's own logging
switch for you (setting the mode alone is not enough; the scripts handle both). This
requires a Weights & Biases API key registered on Polaris — **jesswan already has one
set up**, so she needs nothing further; anyone who has never run `wandb login` here
runs `bash polaris_setup_wandb.sh` once on a login node first. Runs land in your
default W&B account under the project `pedramh-profiling` (override with
`-v WANDB_ENTITY=<account-or-team>` / `-v WANDB_PROJECT=<name>`). Offline runs
accumulate under `members/<you>/wandb/` and can be uploaded later from a login node
with `wandb sync`. **PhysicsNeMo does not use Weights & Biases at all** — its record
is the local MLflow store plus the plain `metrics.csv` described in §6.

---

## 4. PanguWeather

### Dataset
None to prepare: training reads the archive directly. The small auxiliary directory
(field statistics + a 16.7 GB climatology used during validation) already exists,
group-readable, at:

```
/lus/eagle/projects/lighthouse-uchicago/members/mehta5/pangu_polaris_data
```

and is found automatically (if you have your own copy under your member directory, it
is preferred; otherwise the shared one is used).

### Confirmation tests
Both already green (jobs 7259271 and 7259296; §2). To re-run, submit **from
`PanguWeather/v2.0/` inside your checkout** — the scripts fail within a second if
submitted from anywhere else:

```
cd /lus/eagle/projects/lighthouse-uchicago/members/jesswan/pedramh-profiling/PanguWeather/v2.0
qsub HPC_scripts/polaris_val_e3sm_sfno_alldata_smoke.pbs      # success: PANGU_VAL_SMOKE_OK
qsub HPC_scripts/polaris_train_e3sm_sfno_alldata_smoke.pbs    # success: ALLDATA_SMOKE_OK
```

### Full training
```
cd /lus/eagle/projects/lighthouse-uchicago/members/jesswan/pedramh-profiling/PanguWeather/v2.0
qsub HPC_scripts/polaris_train_e3sm_sfno_alldata_full.pbs
```

- **Scale:** a 1.18-billion-parameter model; 100 epochs over years 2015–2044 with
  validation on 2045–2048. Expect roughly **1.8 hours of computation plus ~27 minutes
  of validation per epoch** (measured), i.e. **8+ days of wall-clock time** delivered
  in 24-hour scheduler slices that resume automatically from the latest checkpoint.
- **Storage:** each checkpoint is **18.9 GB**; the job keeps the ten most recent plus
  the latest and best — budget **~230 GB** under your member directory
  (`.../members/<you>/runs/pangu_sfno_alldata_full/`).
- **What progress looks like:** `Starting epoch N/100` advancing in the log across
  resubmissions. A non-zero exit code on the preemptable queue usually means the
  scheduler reclaimed the node, which is normal — the next run resumes.
- **One caveat to know:** the "best" checkpoint is the best *since the last
  interruption*, not the best overall (the running-best value is not carried across
  resumptions). Until that is changed, treat `ckpt_epoch_N.tar` files, not
  `best_ckpt.tar`, as the record of training history.

### Forecasts (inference)
Not yet wired into a batch script. The working path is `ensemble_inference.py`, which
can start forecasts directly from the archive's own validation years; the exact
command and its caveats are in `polaris_pipelines_plan.md` §1. Producing the first
forecast is the next piece of engineering after training starts.

---

## 5. makani

### Dataset (its own pack; the transferred Zarr dataset is NOT used here)
```
cd /lus/eagle/projects/lighthouse-uchicago/members/jesswan/pedramh-profiling/makani_sfno
qsub polaris/polaris_pack_e3sm_alldata_full.pbs
```
- Writes ~**1.43 TB** of per-year HDF5 files plus normalization statistics to
  `/lus/eagle/projects/lighthouse-uchicago/members/<you>/data/e3sm_makani_alldata_full/`
  (check your quota with `myquota` first). Up to 24 hours; safe to resubmit — finished
  years are kept, not rewritten.
- **Success = both lines `CONVERT_ALLDATA_OK` and `PACK_ALLDATA_COMPLETE`.**

### Confirmation test
Already green (job 7259321; §2). To re-run:
```
qsub polaris/polaris_sfno_alldata_smoke.pbs        # success: ALLDATA_SMOKE_OK
```

### Full training
```
qsub polaris/polaris_sfno_alldata_full.pbs
```
- Resumable 24-hour slices, like the others; progress = "Total training time" > 0 and
  the epoch counter advancing across resubmissions; done at epoch 100.
- **Two cautions, stated plainly.** (a) This launcher is new and has not yet had a
  production run; its memory use with the widened 107-field input is unmeasured — if
  it runs out of GPU memory, resubmit with a smaller batch
  (`qsub -v BATCH=4 ...`). (b) **A model trained in this configuration cannot
  currently produce forecasts**: the evaluation code is fixed to the older 58-field
  configuration and rejects anything else. Until that is generalized (a pending
  decision, tracked in `polaris_pipelines_plan.md` §2), makani training here is
  train-only. Expect one benign warning per process in the log
  (`N_in_channels=107 ... expected 58`) — it is a notice, not an error.

---

## 6. PhysicsNeMo — the pipeline that uses the transferred dataset

### Dataset — two routes

**Route A (preferred): verify the transferred dataset, then use it in place.** It is
expected as a directory containing two Zarr stores, e.g.:

```
/lus/eagle/projects/lighthouse-uchicago/members/jesswan/e3sm_seqzarr_allyears/
    e3sm_train.zarr/      (training years 2015–2046)
    e3sm_val.zarr/        (validation years 2047–2049)
```

(Substitute the actual directory name below if it differs. The two store names,
however, must be exactly `e3sm_train.zarr` and `e3sm_val.zarr` — rename or symlink if
necessary.)

**A dataset is never trusted just because it arrived.** Submit the verification job
(under one hour, no GPUs; it reads the dataset in place and writes nothing into it):

```
cd /lus/eagle/projects/lighthouse-uchicago/members/jesswan/pedramh-profiling/physicsnemo_sfno
qsub -v STORE=/lus/eagle/projects/lighthouse-uchicago/members/jesswan/e3sm_seqzarr_allyears \
    polaris/polaris_verify_store.pbs
```

The job checks, in order of increasing cost:

1. **Provenance and generation** — the dataset's own recorded metadata must match the
   current pipeline: the cloud-variable exclusion, the five supplied fields, the
   completion marker written only when a conversion finishes.
2. **Completeness** — every expected data chunk file is present on disk. This catches
   an interrupted transfer, whose missing pieces would otherwise read back silently as
   fields of zeros.
3. **Fidelity** — a sample of ~480 time steps, spread across all 35 years, is compared
   **bit for bit** against the original archive, including the placement and value of
   every land/ocean fill constant and the correctness of the time axis.

**Success = the line `STORE_VERIFY_OK`** (preceded by `SEQZARR_VERIFIED` for each of
the two stores) in the job log.

*Honest scope:* the sampled fidelity check reads ~1 % of the data, so a transfer that
*corrupted* (rather than dropped) a chunk among the unsampled 99 % would not be seen.
If the transfer itself ran with checksums (Globus "verify checksum", or `rsync -c`),
the sampled check is sufficient. If it did not, either re-verify exhaustively —

```
qsub -q preemptable -l walltime=24:00:00 \
    -v STORE=<same path>,EXHAUSTIVE=1 polaris/polaris_verify_store.pbs
```

— which reads every value (several hours), or accept the 1 % sample.

**If verification fails**, the log names the reason precisely:

| error line | meaning | remedy |
|---|---|---|
| `STORE_WRONG_GENERATION` | built before (or under different) variable decisions | Route B |
| `STORE_INCOMPLETE` / `CHUNKS_MISSING` | conversion or transfer never finished | re-transfer, else Route B |
| `roundtrip/…` failures | data do not match the archive | Route B |
| permission errors | files not group-readable | from jesswan's account: `chmod -R g+rX <dataset dir>` (only needed if someone other than jesswan runs the pipeline) |

**Route B (fallback): recreate the dataset from the archive.** Two submissions from
the same directory — first the confirmation test (already green as job 7259303), then
the full conversion, which deliberately requires an explicit confirmation flag because
it writes ~1.43 TB over up to ~11 hours:

```
qsub polaris/polaris_sfno_allyears_smoke.pbs                  # success: ALLYEARS_SMOKE_OK
qsub -v CONFIRM_ALLYEARS=1 polaris/polaris_zarr_e3sm_allyears.pbs
```

**Success = `CONVERT_OK` for each store and `ZARR_ALLYEARS_COMPLETE` on the last
line.** Output lands in
`/lus/eagle/projects/lighthouse-uchicago/members/<you>/e3sm_seqzarr_allyears/`. Safe
to resubmit if killed partway: each finished store is recognized and skipped.

### Full training
With the transferred (Route A) dataset:

```
cd /lus/eagle/projects/lighthouse-uchicago/members/jesswan/pedramh-profiling/physicsnemo_sfno
qsub -v SEQZARR_ALLYEARS_DATA=/lus/eagle/projects/lighthouse-uchicago/members/jesswan/e3sm_seqzarr_allyears \
    polaris/polaris_sfno_allyears.pbs
```

With a Route-B dataset in your own member directory, omit the `-v` argument — that
location is the default.

Before training a single step, the job re-reads the dataset's recorded metadata and
**refuses loudly** if anything is wrong — wrong variable generation, missing
completion marker, non-contiguous years, wrong sample counts, overlapping
train/validation years, or mismatched train/validation variable lists. The number of
model input and output fields is then *derived from the dataset itself* (103 + 5),
never restated by hand.

- **Where results go** (all under your own member directory):

  ```
  /lus/eagle/projects/lighthouse-uchicago/members/<you>/runs/physicsnemo_sfno_allyears/allyears01/
      checkpoints/            model + optimizer state, saved at epochs 1, 5, 10, …
      metrics.csv             one row per training step (loss) and per epoch
                              (learning rate, data throughput, validation error)
      model_package_*/        a self-contained inference package written at each save
      forcast_validation_*.png   per-field validation figures, refreshed each epoch
  ```

- **What progress looks like:** lines of the form `Epoch N Metrics ... loss = ...`
  advancing in the job log across resubmissions, and `metrics.csv` growing. The
  configured run length is 500 epochs; **treat the first epoch as the timing
  measurement** — no production-scale wall-clock number exists yet for this model,
  and `metrics.csv` records exactly what is needed to extrapolate.

- **Watch the first resumption, once.** Checkpoint *saving* is proven; resumption
  *across a preemption* has not yet been observed for this trainer. When the job is
  first requeued, confirm the epoch counter continues where it stopped; if it restarts
  at zero, stop the run and report it. Also note checkpoints are written only every
  5 epochs, so a preemption can cost up to 5 epochs of work.

---

## 7. Known scientific limitations (bound what the trained models can be used for)

1. **PhysicsNeMo: precipitation is currently not learnable.** The trainer normalizes
   each field by statistics computed on the fly in the field's native physical units.
   Precipitation's native unit (m s⁻¹) makes its variability ~40,000 times smaller
   than the numerical floor of that normalization, so its contribution to the training
   objective is effectively zero: the model converges normally and outputs
   near-climatological precipitation with no error message. The fix (precomputed
   per-field statistics) is pending a decision; every epoch trained before it lands is
   an epoch of zero-skill precipitation. Other retained fields are unaffected.
2. **PhysicsNeMo: no forecast driver exists yet.** Training exports a self-contained
   model package at every save, but nothing in the repository reads it to produce
   forecasts yet. Whoever writes that driver must know: the package's `*_stds.npy`
   files actually contain **variances** (the training code saves the squared quantity
   under the "stds" name), and the normalization statistics baked into each package
   inherit the precipitation issue above.
3. **makani: training in the widened configuration is train-only** until the
   evaluation code's fixed 58-field assumption is generalized (§5 above).
4. **PanguWeather: the "best" checkpoint is best-since-last-interruption**, not best
   overall (§4 above).
5. **All models: the prescribed ocean repeats one year.** Sea-surface temperature, sea
   ice, and solar forcing in the archive are bitwise-identical across all 35 years (a
   property of the archive, not of these pipelines; whether it is intended is a
   question only the science owner can answer). Within-year seasonality is intact; the
   interannual axis of those fields is not.

---

## 8. Quick reference

| item | full path |
|---|---|
| E3SM source archive (read-only, all models) | `/eagle/lighthouse-uchicago/members/jesswan/AI4SRM/data/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/h5/plev_data` |
| repository on GitHub (branch `polaris-data-prep`) | `https://github.com/rcc-uchicago/pedramh-profiling` |
| jesswan's repository copy | `/lus/eagle/projects/lighthouse-uchicago/members/jesswan/pedramh-profiling` |
| reference repository copy | `/lus/eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling` |
| **PanguWeather** training job (submit from `PanguWeather/v2.0/`) | `HPC_scripts/polaris_train_e3sm_sfno_alldata_full.pbs` |
| PanguWeather auxiliary statistics (shared, found automatically) | `/lus/eagle/projects/lighthouse-uchicago/members/mehta5/pangu_polaris_data` |
| **makani** dataset pack (submit from `makani_sfno/`) | `polaris/polaris_pack_e3sm_alldata_full.pbs` (success: `CONVERT_ALLDATA_OK` + `PACK_ALLDATA_COMPLETE`) |
| makani training job | `polaris/polaris_sfno_alldata_full.pbs` |
| **PhysicsNeMo** transferred dataset (expected location) | `/lus/eagle/projects/lighthouse-uchicago/members/jesswan/e3sm_seqzarr_allyears/{e3sm_train.zarr, e3sm_val.zarr}` |
| PhysicsNeMo dataset verification (submit from `physicsnemo_sfno/`) | `polaris/polaris_verify_store.pbs` (success: `STORE_VERIFY_OK`) |
| PhysicsNeMo dataset rebuild (Route B) | `polaris/polaris_zarr_e3sm_allyears.pbs` (success: `ZARR_ALLYEARS_COMPLETE`; needs `-v CONFIRM_ALLYEARS=1`) |
| PhysicsNeMo training job | `polaris/polaris_sfno_allyears.pbs` |
| training outputs (every model) | `/lus/eagle/projects/lighthouse-uchicago/members/<you>/runs/…` |
| engineering companion (costs, failure modes, open decisions) | `polaris_pipelines_plan.md` |
