# Running the E3SM → SFNO training pipeline on Polaris

*A step-by-step guide. Written 2026-07-16; **all four confirmation runs (§2) passed
the same day**, so every command below sits on a demonstrated-green software path. For
the two companion model pipelines (PanguWeather and makani), see
`polaris_pipelines_plan.md` — this document covers the PhysicsNeMo pipeline, the one
that consumes the transferred dataset.*

---

## 1. What this pipeline does

The pipeline trains a **Spherical Fourier Neural Operator (SFNO)** — a neural network
that operates on fields defined on the sphere — to advance the simulated atmospheric
state forward by six hours. One training example is therefore a pair of consecutive
states: the model receives the state at time *t* and learns to reproduce the state at
*t + 6 h*.

**Source data.** The training data come from a 35-year E3SM version 3 atmosphere
simulation (SSP2-4.5 scenario with prescribed ocean, years 2015–2049), archived as one
HDF5 file per 6-hour interval — 1,460 files per year, 51,100 files in all — on a
1° × 1° global grid (180 × 360 points). The archive lives, read-only, at:

```
/eagle/lighthouse-uchicago/members/jesswan/AI4SRM/data/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/h5/plev_data
```

**Variables.** Of the archive's 162 fields, the pipeline retains **108**: 103
*predicted* fields (the model forecasts them: temperature, winds, geopotential height,
relative humidity on 18 model levels, plus surface and land fields) and 5 *prescribed*
fields (supplied to the model but never forecast: land/glacier/vegetation masks,
topography, incoming solar radiation). The three cloud condensate variables (54 fields
across levels) are excluded from all models by decision of the science owner
(2026-07-16). Fields that E3SM masks with missing values over land or ocean are filled
with fixed constants recorded inside the dataset itself.

**Dataset format.** Training does not read the 51,100 HDF5 files directly. A one-time
conversion step rewrites them into two **Zarr stores** — a chunked array format suited
to fast sequential reads — named exactly `e3sm_train.zarr` (years 2015–2046) and
`e3sm_val.zarr` (years 2047–2049, held out for validation). Together they occupy about
**1.43 terabytes**.

---

## 2. Before anything large runs: the confirmation tests

Every stage below has a brief confirmation run (a "smoke test": the complete software
path exercised on a small data subset, minutes instead of days). **Success is always a
specific printed line in the job's log — never just a clean exit code.** The full
confirmation sequence was run 2026-07-16 and **all four passed**:

| test | job | result |
|---|---|---|
| PanguWeather validation-memory test | 7259271 | ✅ `PANGU_VAL_SMOKE_OK` — validation peaks at 25.0 of 39.5 GiB per GPU (13.9 GiB headroom); the previously feared out-of-memory failure mode is refuted |
| PanguWeather training test (108 fields) | 7259296 | ✅ `ALLDATA_SMOKE_OK` — peak 27.0 GB, median step 0.64 s |
| PhysicsNeMo pipeline test (103+5 fields) | 7259303 | ✅ `ALLYEARS_SMOKE_OK` — year-seam data verified bit-for-bit; the per-step metrics file also confirmed working (`PHYSICSNEMO_CSV_OK`) |
| makani pipeline test (100/1/7 fields) | 7259321 | ✅ `ALLDATA_SMOKE_OK` — fresh 108-field pack built and trained; the "expected 58" warning in its log is a benign, by-design notice |

---

## 3. One-time setup (per person)

Any member of the `lighthouse-uchicago` project can run this pipeline. Two things are
per-person:

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

   Two practical notes. First, after a `git pull`, the updated PhysicsNeMo code takes
   effect immediately — the Python environment links to your checkout rather than
   copying from it, so no reinstallation is needed (rebuild the environment only if
   `polaris_setup_sfno_venv.sh` itself changed). Second, if you ever *push* from a
   Polaris login node, use `git -c pack.threads=1 push` — the login nodes cap
   per-user processes and multi-threaded git pushes are killed mid-transfer
   (ordinary `git pull` is not affected).

2. **Your own Python environment.** Build it once, on a login node (about 15 minutes;
   success line `SFNO_VENV_OK`):

   ```
   cd /lus/eagle/projects/lighthouse-uchicago/members/jesswan/pedramh-profiling
   bash polaris_setup_sfno_venv.sh
   ```

   This is deliberately per-person: the environment links the PhysicsNeMo source in
   your own checkout, and the batch scripts refuse to run against somebody else's
   working copy (error `PHYSICSNEMO_WRONG_CHECKOUT`) so that no one unknowingly trains
   with another person's uncommitted edits.

All compute below runs through the PBS batch scheduler (`qsub` to submit, `qstat` to
watch, `qdel` to cancel). Nothing heavy runs on login nodes.

---

## 4. Step one — obtain the training dataset

There are two routes. **Route A is preferred**: it reuses the already-built dataset
being transferred into jesswan's directory and saves ~11 hours of compute and 1.43 TB
of duplicate storage. Route B rebuilds the dataset from the archive and is the fallback
if Route A's verification fails.

### Route A: verify the transferred dataset, then use it in place

The transferred dataset is expected as a directory containing the two stores, e.g.:

```
/lus/eagle/projects/lighthouse-uchicago/members/jesswan/e3sm_seqzarr_allyears/
    e3sm_train.zarr/
    e3sm_val.zarr/
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
   current pipeline: the cloud-variable exclusion, the five prescribed fields, the
   completion marker written only when a conversion finishes.
2. **Completeness** — every expected data chunk file is present on disk. This is the
   check that catches an interrupted transfer, whose missing pieces would otherwise
   read back silently as fields of zeros.
3. **Fidelity** — a sample of ~480 time steps, spread across all 35 years, is compared
   **bit for bit** against the original archive, including the placement and value of
   every land/ocean fill constant and the correctness of the time axis.

**Success = the line `STORE_VERIFY_OK` (preceded by `SEQZARR_VERIFIED` for each of the
two stores) in the job log** (`pangu`-style logs land beside the submission directory;
the job name is `physicsnemo_verify_store`).

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

### Route B: recreate the dataset from the archive

Two submissions from the same directory. First the pipeline's own confirmation test
(minutes; also part of §2):

```
cd /lus/eagle/projects/lighthouse-uchicago/members/jesswan/pedramh-profiling/physicsnemo_sfno
qsub polaris/polaris_sfno_allyears_smoke.pbs          # success line: ALLYEARS_SMOKE_OK
```

Then the full conversion (deliberately requires an explicit confirmation flag, because
it writes ~1.43 TB over up to ~11 hours — check your quota with `myquota` first):

```
qsub -v CONFIRM_ALLYEARS=1 polaris/polaris_zarr_e3sm_allyears.pbs
```

**Success = `CONVERT_OK` for each store and `ZARR_ALLYEARS_COMPLETE` on the last
line.** The output lands in your member directory:

```
/lus/eagle/projects/lighthouse-uchicago/members/<you>/e3sm_seqzarr_allyears/
```

The job is safe to resubmit if the queue kills it partway: each finished store is
recognized by its completion marker and skipped; only the store that was in flight is
redone.

---

## 5. Step two — train

One submission. With the transferred (Route A) dataset:

```
cd /lus/eagle/projects/lighthouse-uchicago/members/jesswan/pedramh-profiling/physicsnemo_sfno
qsub -v SEQZARR_ALLYEARS_DATA=/lus/eagle/projects/lighthouse-uchicago/members/jesswan/e3sm_seqzarr_allyears \
    polaris/polaris_sfno_allyears.pbs
```

With a Route-B dataset in your own member directory, omit the `-v` argument entirely —
that location is the default.

What the job does before training a single step: it re-reads the dataset's recorded
metadata and **refuses loudly** if anything is wrong — wrong variable generation,
missing completion marker, non-contiguous years, wrong sample counts, overlapping
train/validation years, or mismatched train/validation variable lists. The number of
model input and output fields is then *derived from the dataset itself* (103 predicted
+ 5 prescribed), never restated by hand.

Operational facts:

- **Hardware:** one node, 4 × NVIDIA A100 (40 GB each). The job runs on the
  `preemptable` queue in 24-hour slices and is marked re-runnable: when the scheduler
  preempts it, it requeues itself and **resumes from its last checkpoint**. Simply
  leave it in the queue; resubmitting the same command later is also safe.
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
  measurement** — no production-scale wall-clock number exists yet for this model, and
  `metrics.csv` now records exactly what is needed to extrapolate.

---

## 6. Step three — watch the first resumption (once), then let it run

Two things merit human eyes early in the run; both are known, documented limitations
rather than surprises:

1. **The first preemption.** Checkpoint *saving* is proven; checkpoint *resumption
   across a preemption* has not yet been observed for this trainer. When the job is
   first requeued, confirm in the log that the epoch counter continues from where it
   stopped. If it restarts at zero, stop the run and report it — continuing would
   silently retrain from scratch in 24-hour fragments.
2. **Checkpoint cadence.** States are saved at epochs 1, 5, 10, … — a preemption can
   therefore lose up to five epochs of work. Acceptable for now; recorded here so the
   cost is a known trade, not a discovery.

---

## 7. Known scientific limitations of the current training path

Stated plainly, because they bound what a trained model can be used for:

1. **Precipitation is currently not learnable.** The trainer normalizes each field by
   statistics computed on the fly in the field's native physical units. Precipitation's
   native unit (m s⁻¹) makes its variability ~40,000 times smaller than the numerical
   floor of that normalization, so its contribution to the training objective is
   effectively zero: the model will converge normally and output near-climatological
   precipitation with no error message. The fix (precomputed per-field statistics)
   belongs in the training code, is pending a decision, and every epoch trained before
   it lands is an epoch of zero-skill precipitation. All other retained fields are
   unaffected — the ~23 worst-affected fields were the cloud variables, already
   excluded.
2. **No forecast (inference) driver exists yet for this model.** Training exports a
   self-contained model package at every save, but nothing in the repository currently
   reads it to produce forecasts. Two defects to carry into whoever writes that driver:
   the package's `*_stds.npy` files actually contain **variances** (the training code
   saves the squared quantity under the "stds" name), and the normalization statistics
   baked into each package inherit the precipitation issue above.
3. **The prescribed ocean repeats one year.** Sea-surface temperature, sea ice, and
   solar forcing in the archive are bitwise-identical across all 35 years (a property
   of the archive, not of this pipeline; whether it is intended is a question only the
   science owner can answer). Within-year seasonality is intact; the interannual axis
   of those fields is not.

---

## 8. Quick reference

| item | full path |
|---|---|
| E3SM source archive (read-only) | `/eagle/lighthouse-uchicago/members/jesswan/AI4SRM/data/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/h5/plev_data` |
| jesswan's repository copy | `/lus/eagle/projects/lighthouse-uchicago/members/jesswan/pedramh-profiling` |
| reference repository copy (branch `polaris-data-prep`) | `/lus/eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling` |
| transferred dataset (expected location) | `/lus/eagle/projects/lighthouse-uchicago/members/jesswan/e3sm_seqzarr_allyears/{e3sm_train.zarr, e3sm_val.zarr}` |
| dataset verification job | `physicsnemo_sfno/polaris/polaris_verify_store.pbs` (success: `STORE_VERIFY_OK`) |
| pipeline confirmation test | `physicsnemo_sfno/polaris/polaris_sfno_allyears_smoke.pbs` (success: `ALLYEARS_SMOKE_OK`) |
| dataset rebuild job (Route B) | `physicsnemo_sfno/polaris/polaris_zarr_e3sm_allyears.pbs` (success: `ZARR_ALLYEARS_COMPLETE`; needs `-v CONFIRM_ALLYEARS=1`) |
| training job | `physicsnemo_sfno/polaris/polaris_sfno_allyears.pbs` |
| training outputs | `/lus/eagle/projects/lighthouse-uchicago/members/<you>/runs/physicsnemo_sfno_allyears/allyears01/` |
| companion runbook (all three models, costs, open decisions) | `polaris_pipelines_plan.md` |

*Scheduler basics: `qstat -u $USER` to see your jobs; job logs are written to the
directory the job was submitted from (file `<job-name>.o<job-id>`), except where a
script states otherwise. The `debug` queue (used by the tests) allows one job per
person, up to one hour; long jobs use `preemptable`.*
