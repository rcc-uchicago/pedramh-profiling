# Running the three E3SM model-training pipelines on Polaris

*A step-by-step guide covering all three models — **PanguWeather**, **makani**, and
**PhysicsNeMo**. Written 2026-07-16; **all four confirmation runs (§2) passed the same
day**, and this revision incorporates the findings of a three-agent adversarial review
run the same evening. The engineering companion (costs, failure-mode analysis, open
decisions) is `polaris_pipelines_plan.md`.*

*Contacts: science decisions (variable sets, fills, whether archive properties are
intended) — **jesswan**. Pipeline/bring-up engineering and this repository —
**rmehta1987**.*

---

## 1. What these pipelines do

Each pipeline trains a neural network to advance a simulated atmospheric state forward
by six hours: the model receives the global state at time *t* and learns to reproduce
the state at *t + 6 h*.

**All three models are variants of the Spherical Fourier Neural Operator (SFNO)** — a
network that operates on fields defined on the sphere. The pipeline *names* refer to
the code bases, not the architectures: the "PanguWeather" repository also contains the
original Pangu 3-D transformer, but on Polaris it trains an SFNO (1.18 billion
parameters); makani and PhysicsNeMo train smaller SFNOs of identical width (embedding
384, 8 layers; parameter counts unmeasured, order 10⁸). The project measures how the
three implementations compare — see §7.6 for what "compare" can and cannot mean today.

**Source data (shared by all three).** A 35-year E3SM version 3 atmosphere simulation
(SSP2-4.5 scenario with prescribed ocean, years 2015–2049), archived as one HDF5 file
per 6-hour interval — 1,460 files per year, 51,100 in all — on a 1° × 1° global grid
(180 × 360 points). The archive lives, read-only, at:

```
/eagle/lighthouse-uchicago/members/jesswan/AI4SRM/data/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/h5/plev_data
```

(Paths beginning `/eagle/...` and `/lus/eagle/projects/...` are the **same
filesystem** — two names for one location; this document uses both interchangeably.)

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

### ⚠ Storage budget — read before building anything

Disk quota on this filesystem is **shared by the whole project, not per person**. As
of 2026-07-16 the project had used **13.14 of 15 TB — about 1.86 TB free** (`myquota`
reports it). The full plan wants far more than that: the makani pack (~1.43 TB) + a
PhysicsNeMo dataset (~1.43 TB, whether transferred in or rebuilt) + Pangu checkpoints
(~230 GB) + run outputs ≈ **3.1+ TB**. Consequences:

- **Never hold two copies of the PhysicsNeMo dataset** — either the transferred one
  *or* a rebuild, not both.
- Before the makani pack or a Route-B rebuild, run `myquota` and confirm ≥1.5 TB free;
  a job that hits the quota dies mid-write hours in, with `Disk quota exceeded`.
- Freeing space or requesting a quota increase is a project decision (rmehta1987 /
  jesswan) that should be made **before** the first terabyte-scale job — and note the
  incoming dataset transfer itself needs ~1.43 TB of this same quota to land.

---

## 2. The confirmation tests (all passed 2026-07-16)

Every stage below has a brief confirmation run (a "smoke test": the complete software
path exercised on a small data subset, minutes instead of days). **Success is always a
specific printed line in the job's log — never just a clean exit code.** The full
sequence was run 2026-07-16 and **all four passed**:

| test | job | result |
|---|---|---|
| PanguWeather validation-memory test | 7259271 | ✅ `PANGU_VAL_SMOKE_OK` — validation used 25.0 GiB (25.6 GiB reserved by the allocator) of 39.5 GiB per GPU, i.e. **13.9 GiB of headroom against the reserved peak**; the previously feared out-of-memory failure mode is refuted |
| PanguWeather training test (108 fields) | 7259296 | ✅ `ALLDATA_SMOKE_OK` — peak 27.0 GB, median step 0.64 s |
| PhysicsNeMo pipeline test (103+5 fields) | 7259303 | ✅ `ALLYEARS_SMOKE_OK` — year-seam data verified bit-for-bit; the per-step metrics file also confirmed working (`PHYSICSNEMO_CSV_OK`) |
| makani pipeline test (100/1/7 fields) | 7259321 | ✅ `ALLDATA_SMOKE_OK` — fresh 108-field pack built and trained; the "expected 58" warning in its log is a benign, by-design notice |
| Weights & Biases online logging (Pangu) | 7259364 | ✅ `ALLDATA_SMOKE_OK` + live run synced to wandb.ai — the online path the full run uses is proven end-to-end |
| chain submission mechanics | 7259371/72 | ✅ link 2 held on `depend=afterany:` link 1; whole chain cancelled cleanly with `qdel` |

The job logs live in the *submitter's* directories, so they are not independently
readable by others. If you were not the submitter, re-run what you rely on: each
model's section names its test. Note the tests use the `debug` queue (one job per
person at a time), so re-running all of them serializes to roughly 2–3 hours.

---

## 3. One-time setup (per person)

Any member of the `lighthouse-uchicago` project can run these pipelines. Three things
are per-person:

1. **Your own copy of the repository, updated to the newest code.** The scripts
   resolve all writable paths to *your* member directory automatically, but they must
   be *submitted from your own checkout*.

   **Getting the newest code from GitHub.** The repository is
   `https://github.com/rcc-uchicago/pedramh-profiling`, and everything this guide
   describes lives on the branch **`polaris-data-prep`** (an open pull request; once
   it is merged, substitute `main` below). On a login node:

   ```
   # if you already have a copy (jesswan's is at members/jesswan/pedramh-profiling):
   cd /lus/eagle/projects/lighthouse-uchicago/members/<you>/pedramh-profiling
   git fetch origin
   git checkout polaris-data-prep
   git pull origin polaris-data-prep

   # if you are starting from nothing:
   cd /lus/eagle/projects/lighthouse-uchicago/members/<you>
   git clone https://github.com/rcc-uchicago/pedramh-profiling.git
   cd pedramh-profiling
   git checkout polaris-data-prep
   ```

   **Confirm the update actually landed** — the fetch is load-bearing (an older clone
   contains *none* of the scripts this guide names):

   ```
   ls polaris_pipeline_runbook.md polaris_submit_chain.sh \
      physicsnemo_sfno/polaris/polaris_verify_store.pbs   # all three must exist
   ```

   Two practical notes. First, after a `git pull`, updated code takes effect
   immediately — the Python environment links to your checkout rather than copying
   from it, so no reinstallation is needed (rebuild the environment only if
   `polaris_setup_sfno_venv.sh` itself changed). Second, if you ever *push* from a
   Polaris login node, use `git -c pack.threads=1 push` — the login nodes cap
   per-user processes and multi-threaded git pushes are killed mid-transfer
   (ordinary `git pull` is not affected).

2. **Your own Python environment for makani and PhysicsNeMo.** Build it once, on a
   login node (~10–20 minutes; success line `SFNO_VENV_OK`) — **from your own
   checkout**, because the build links the PhysicsNeMo source of whatever checkout it
   is run from:

   ```
   cd /lus/eagle/projects/lighthouse-uchicago/members/<you>/pedramh-profiling
   bash polaris_setup_sfno_venv.sh
   ```

   Why this matters, honestly stated: the **PhysicsNeMo** batch jobs verify the match
   and refuse to run against somebody else's working copy (error
   `PHYSICSNEMO_WRONG_CHECKOUT`). The **makani** jobs carry no such guard — and if you
   skip this build entirely, they silently fall back to a shared environment that
   executes *its builder's* working tree, uncommitted edits and all. So for makani the
   only protection is doing this step. (jesswan already has her own environment.)

3. **Nothing extra for PanguWeather** — it uses the system-provided Python plus a
   shared, group-readable package directory that the scripts locate automatically
   (they fail loudly with `TOPUPS_MISSING` if it is ever absent, with the fix printed).

**Scheduler basics.** All compute runs through PBS (`qsub` to submit, `qstat -u $USER`
to watch, `qdel` to cancel); nothing heavy runs on login nodes. Job logs land in the
directory the job was submitted from (file `<job-name>.o<job-id>`; these are
git-ignored). The `debug` queue (the tests) allows one job per person, up to one hour.
Long jobs use `preemptable` (up to **72 h** walltime, 10 running / 20 queued per
person). **Two different interruptions, two different behaviors:** if the scheduler
*preempts* a job, it requeues and resumes from its checkpoint automatically; if a job
reaches its **walltime, it is simply killed** — nothing resubmits it. For runs longer
than 72 h, pre-submit a dependency chain so no one has to babysit:

```
bash <repo>/polaris_submit_chain.sh <n-links> <script.pbs> [extra qsub args...]
```

Each link starts only when the previous one has terminated (works with preemption
too), re-runs the same script, and resumes from the checkpoint; a link that starts
after training has already finished exits within minutes, so over-provisioning the
chain is cheap. Mechanics proven 2026-07-16 (jobs 7259371/72: link 2 held on link 1,
clean `qdel`). Cancel a chain by deleting **all** its links in one `qdel` (deleting
only an early link releases the next one).

**Experiment tracking (Weights & Biases).** All jobs default to *offline* tracking —
nothing is sent anywhere and no account is needed. For **PanguWeather and makani**,
live dashboards are one flag away: submit any training job with
`qsub -v WANDB_MODE=online <script>` (or add `-v WANDB_MODE=online` to the chain
command) and the launcher enables the config's own logging switch for you. **Proven
live 2026-07-16** (job 7259364 synced a real training run to wandb.ai through the ALCF
proxy). This requires a Weights & Biases API key registered on Polaris — **jesswan
already has one**, so she needs nothing further; anyone who has never run
`wandb login` here runs `bash polaris_setup_wandb.sh` once on a login node first. Runs
land in your default W&B account under the project `pedramh-profiling` (override with
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
is preferred; otherwise the shared one is used — the job log prints which).

### Confirmation tests
Both already green (jobs 7259271 and 7259296; §2). To re-run, submit **from
`PanguWeather/v2.0/` inside your checkout** (the scripts resolve shared helpers
relative to the submission directory and abort early — before any training — if
submitted from the wrong place):

```
cd /lus/eagle/projects/lighthouse-uchicago/members/<you>/pedramh-profiling/PanguWeather/v2.0
qsub HPC_scripts/polaris_val_e3sm_sfno_alldata_smoke.pbs      # success: PANGU_VAL_SMOKE_OK
qsub HPC_scripts/polaris_train_e3sm_sfno_alldata_smoke.pbs    # success: ALLDATA_SMOKE_OK
```

### Full training
```
cd /lus/eagle/projects/lighthouse-uchicago/members/<you>/pedramh-profiling/PanguWeather/v2.0
bash ../../polaris_submit_chain.sh 3 HPC_scripts/polaris_train_e3sm_sfno_alldata_full.pbs
# add wandb:  bash ../../polaris_submit_chain.sh 3 HPC_scripts/polaris_train_e3sm_sfno_alldata_full.pbs -v WANDB_MODE=online
```

- **Scale:** a 1.18-billion-parameter model; 100 epochs over years 2015–2044 with
  validation on 2045–2048, one node with 4 GPUs. Per epoch: ~1.8–2.0 hours of training
  (from the measured 0.60–0.64 s/step) **plus validation, which is unmeasured at
  production scope** — the confirmation test measured 112.6 s at 9 initial conditions;
  linear scaling to the production 129 suggests **budget 20–30 minutes per epoch**.
  Total: **~8–9 days of wall-clock time**, which the 3-link × 72 h chain above covers
  unattended.
- **Storage:** each checkpoint is **18.9 GB**; the job keeps the ten most recent plus
  the latest and best — budget **~230 GB** under
  `.../members/<you>/runs/pangu_sfno_alldata_full/`. (Keeping more history is
  `-v MAX_CKPTS_KEEP=<n>` at 18.9 GB per epoch kept — mind §1's storage budget.)
- **What progress looks like:** `Starting epoch N/100` advancing in the log across
  links. **Finished** = the final log ends with `DONE ---- rank 0` after epoch
  100/100. A non-zero exit on preemptable usually means preemption — normal.
- **One caveat to know:** the "best" checkpoint is the best *since the last
  interruption*, not the best overall (the running-best value is not carried across
  resumptions) — and the rolling keep-10 window means an early true-best epoch can be
  deleted. Until the pending fix lands, treat the numbered `ckpt_epoch_N.tar` files
  plus the validation-loss curve (in the log / W&B) as the record, and consider
  raising `MAX_CKPTS_KEEP` if quota allows.

### Forecasts (inference)
Not yet wired into a batch script. The working path is `ensemble_inference.py`, which
can start forecasts directly from the archive's own validation years; the exact
command and its caveats are in `polaris_pipelines_plan.md` §1. Producing the first
forecast is the next piece of engineering after training starts.

---

## 5. makani

### Dataset (its own pack; the transferred Zarr dataset is NOT used here)
```
cd /lus/eagle/projects/lighthouse-uchicago/members/<you>/pedramh-profiling/makani_sfno
qsub polaris/polaris_pack_e3sm_alldata_full.pbs
```
- Writes ~**1.43 TB** (check §1's storage budget first) of per-year HDF5 files plus
  normalization statistics to
  `/lus/eagle/projects/lighthouse-uchicago/members/<you>/data/e3sm_makani_alldata_full/`.
  Up to 24 hours; safe to resubmit — finished years are kept, not rewritten.
- **Success = both lines `CONVERT_ALLDATA_OK` and `PACK_ALLDATA_COMPLETE`.**
- *Honest scope:* the pack job checks shapes, channel names, sample counts, and
  statistics presence — but unlike the PhysicsNeMo dataset (§6), **nothing re-verifies
  the packed values bit-for-bit against the archive**; a value-level verifier for this
  pack has not been built. The packed statistics are likewise not independently
  recomputed.

### Confirmation test
Already green (job 7259321; §2). To re-run (from the same `makani_sfno/` directory):
```
qsub polaris/polaris_sfno_alldata_smoke.pbs        # success: ALLDATA_SMOKE_OK
```

### Full training
```
cd /lus/eagle/projects/lighthouse-uchicago/members/<you>/pedramh-profiling/makani_sfno
bash ../polaris_submit_chain.sh 2 polaris/polaris_sfno_alldata_full.pbs
```
- Duration is **unmeasured at production scale** (only a tiny-model test has run);
  start with a 2-link chain and extend once the first epochs give a rate. The global
  batch size defaults to 8 and must remain divisible by the 4 GPUs — if memory runs
  out, resubmit with `-v BATCH=4`.
- Progress = "Total training time" > 0 and the epoch counter advancing across links;
  **finished** = a later link starts, finds nothing left to train (epoch counter at
  100), and exits almost immediately.
- **Two cautions, stated plainly.** (a) This launcher is new; its first production
  submission should be watched for the first half hour. (b) **A model trained in this
  configuration cannot currently produce forecasts**: the evaluation code is fixed to
  the older 58-field configuration and rejects anything else. Until that is
  generalized (a pending decision, tracked in `polaris_pipelines_plan.md` §2), makani
  training here is train-only. Expect one benign warning per process in the log
  (`N_in_channels=107 ... expected 58`) — a notice, not an error.

---

## 6. PhysicsNeMo — the pipeline that uses the transferred dataset

### Dataset — two routes, and you use exactly one (§1 storage budget)

**Route A (preferred): verify the transferred dataset, then use it in place.**
**Status 2026-07-16: the transfer is announced but not yet on disk** — until it lands,
the commands below fail immediately with `NO_STORE_FOUND` (verification) or
`ALLYEARS_ZARR_MISSING` (training); that is those errors' meaning today. When it
lands, it is expected as a directory containing two Zarr stores, e.g.:

```
/lus/eagle/projects/lighthouse-uchicago/members/jesswan/e3sm_seqzarr_allyears/
    e3sm_train.zarr/      (training years 2015–2046)
    e3sm_val.zarr/        (validation years 2047–2049)
```

(Substitute the actual directory if it differs. The two store names, however, must be
exactly `e3sm_train.zarr` and `e3sm_val.zarr` — rename or symlink if necessary. If the
transfer stalls or fails verification and the dataset is needed sooner, fall back to
Route B — after checking the storage budget.)

**A dataset is never trusted just because it arrived.** Submit the verification job
(under one hour, no GPUs; it reads the dataset in place and writes nothing into it):

```
cd /lus/eagle/projects/lighthouse-uchicago/members/<you>/pedramh-profiling/physicsnemo_sfno
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
3. **Fidelity** — a sample of ~480 time steps *per store* (about 1 % of the training
   store, 11 % of the smaller validation store), spread across each store's years, is
   compared **bit for bit** against the original archive, including the placement and
   value of every land/ocean fill constant and the correctness of the time axis.

**Success = the line `STORE_VERIFY_OK`** (preceded by `SEQZARR_VERIFIED` for each of
the two stores) in the job log.

*Honest scope:* the sampled fidelity check leaves ~99 % of the training store unread,
so a transfer that *corrupted* (rather than dropped) a chunk there would not be seen.
If the transfer ran with checksums (Globus "verify checksum", or `rsync -c`), the
sampled check is sufficient. **If you do not know how the transfer was run, treat it
as unchecksummed and verify exhaustively** (several hours; safe to leave unattended —
it requeues itself if preempted):

```
qsub -q preemptable -l walltime=24:00:00 \
    -v STORE=<same path>,EXHAUSTIVE=1 polaris/polaris_verify_store.pbs
```

**If verification fails**, the log names the reason precisely:

| error line | meaning | remedy |
|---|---|---|
| `STORE_WRONG_GENERATION` | built before (or under different) variable decisions | reconcile with jesswan, else Route B |
| `STORE_INCOMPLETE` / `CHUNKS_MISSING` | conversion or transfer never finished | re-transfer, else Route B |
| `roundtrip/…` failures | data do not match the archive | Route B |
| permission errors | files not group-readable | from jesswan's account: `chmod -R g+rX <dataset dir>` (only needed if someone other than jesswan runs the pipeline) |

**Route B (fallback): recreate the dataset from the archive.** Two submissions, both
from the `physicsnemo_sfno/` directory of your checkout — first the confirmation test
(already green as job 7259303), then the full conversion, which deliberately requires
an explicit confirmation flag because it writes ~1.43 TB over up to ~11 hours:

```
cd /lus/eagle/projects/lighthouse-uchicago/members/<you>/pedramh-profiling/physicsnemo_sfno
qsub polaris/polaris_sfno_allyears_smoke.pbs                  # success: ALLYEARS_SMOKE_OK
qsub -v CONFIRM_ALLYEARS=1 polaris/polaris_zarr_e3sm_allyears.pbs
```

**Success = the line `ZARR_ALLYEARS_COMPLETE` near the end of the log.** On a first,
uninterrupted run you will also see `CONVERT_OK` once per store; on a legitimate
resubmit after an interruption, an already-finished store prints a "skip" line
*instead of* `CONVERT_OK` — that is correct behavior, not a failure. Output lands in
`/lus/eagle/projects/lighthouse-uchicago/members/<you>/e3sm_seqzarr_allyears/`.

### Full training
With the transferred (Route A) dataset:

```
cd /lus/eagle/projects/lighthouse-uchicago/members/<you>/pedramh-profiling/physicsnemo_sfno
bash ../polaris_submit_chain.sh 3 polaris/polaris_sfno_allyears.pbs \
    -v SEQZARR_ALLYEARS_DATA=/lus/eagle/projects/lighthouse-uchicago/members/jesswan/e3sm_seqzarr_allyears
```

With a Route-B dataset in your own member directory, omit the `-v` argument — that
location is the default. (Start with a 3-link chain; the configured 500 epochs will
need more once the first epochs establish the rate — submitting another chain later
continues the same run.)

Before training a single step, the job re-reads the dataset's recorded metadata and
**refuses loudly** if anything is wrong — wrong variable generation, missing
completion marker, non-contiguous years, wrong sample counts, overlapping
train/validation years, or mismatched train/validation variable lists. The number of
model input and output fields is then *derived from the dataset itself* (103 + 5),
never restated by hand.

- **Where results go** (all under your own member directory):

  ```
  /lus/eagle/projects/lighthouse-uchicago/members/<you>/runs/physicsnemo_sfno_allyears/allyears01/
      checkpoints/            model + optimizer state, saved at epochs 0, 1, 5, 10, …
      metrics.csv             one row per training step (loss) and per epoch
                              (learning rate, data throughput, validation error)
      model_package_*/        ONE self-contained inference package, overwritten at
                              each save (earlier versions are not kept)
      forcast_validation_*.png   ("forcast" is the code's own spelling) — ~103 NEW
                              figures per epoch, one per field; they accumulate, so a
                              long run writes tens of thousands of small files
  ```

- **What progress looks like:** lines of the form `Epoch N Metrics ... loss = ...`
  advancing in the job log across links, and `metrics.csv` growing. **Finished** = the
  log line `Finished training!`. The configured run length is 500 epochs; **treat the
  first epoch as the timing measurement** — no production-scale wall-clock number
  exists yet for this model, and `metrics.csv` records exactly what is needed to
  extrapolate.

- **Watch the first resumption, once.** Checkpoint *saving* is proven; resumption
  *across an interruption* has not yet been observed for this trainer. When the second
  chain link (or a requeue) starts, confirm the epoch counter continues where it
  stopped; if it restarts at zero, `qdel` the remaining links and report it. Also note
  checkpoints are written only every 5 epochs, so an interruption can cost up to 5
  epochs of work.

---

## 7. Known limitations (bound what the trained models can be used for)

1. **PhysicsNeMo: precipitation is currently not learnable.** The trainer normalizes
   each field by statistics computed on the fly in the field's native physical units.
   Precipitation's native unit (m s⁻¹) makes its variability ~40,000 times smaller
   than the numerical floor of that normalization, so its contribution to the training
   objective is effectively zero: the model converges normally and outputs
   near-climatological precipitation with no error message. The fix (precomputed
   per-field statistics) is pending a decision; every epoch trained before it lands is
   an epoch of zero-skill precipitation. (No systematic audit of the other 102 fields'
   susceptibility exists yet; precipitation is the one *demonstrated* case.)
2. **PhysicsNeMo: no forecast driver exists yet.** Training exports a self-contained
   model package at every save (overwriting the previous one), but nothing in the
   repository reads it to produce forecasts yet. Whoever writes that driver must know:
   the package's `*_stds.npy` files actually contain **variances** (the training code
   saves the squared quantity under the "stds" name), and the normalization statistics
   baked into the package inherit the precipitation issue above.
3. **makani: training in the widened configuration is train-only** until the
   evaluation code's fixed 58-field assumption is generalized (§5 above).
4. **PanguWeather: the "best" checkpoint is best-since-last-interruption**, not best
   overall (§4 above).
5. **All models: the prescribed ocean repeats one year.** Sea-surface temperature, sea
   ice, and solar forcing in the archive are bitwise-identical across all 35 years (a
   property of the archive, not of these pipelines; whether it is intended is a
   question only jesswan can answer). Within-year seasonality is intact; the
   interannual axis of those fields is not.
6. **The three pipelines are not yet directly comparable.** Their train/validation
   splits differ — PhysicsNeMo trains through 2046, i.e. *on years the other two
   validate on*; only **2047–2049 is untouched by all three trainings** — and their
   "epochs" are incommensurable (makani draws 8,192 samples per epoch, Pangu a full
   43,800-sample pass, PhysicsNeMo 500 passes of 46,720 — total training budgets
   differing by ~30×). Any cross-model skill comparison needs a common evaluation
   protocol on the shared held-out years, which does not exist yet and is a science
   decision (jesswan).

---

## 8. Quick reference

| item | full path |
|---|---|
| E3SM source archive (read-only, all models) | `/eagle/lighthouse-uchicago/members/jesswan/AI4SRM/data/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/h5/plev_data` |
| repository on GitHub (branch `polaris-data-prep`) | `https://github.com/rcc-uchicago/pedramh-profiling` |
| your repository copy | `/lus/eagle/projects/lighthouse-uchicago/members/<you>/pedramh-profiling` |
| chain submitter (multi-day runs, no babysitting) | `polaris_submit_chain.sh` (repo root) |
| **PanguWeather** training (submit from `PanguWeather/v2.0/`) | `HPC_scripts/polaris_train_e3sm_sfno_alldata_full.pbs` — chain 3 × 72 h |
| PanguWeather auxiliary statistics (shared, found automatically) | `/lus/eagle/projects/lighthouse-uchicago/members/mehta5/pangu_polaris_data` |
| **makani** dataset pack (submit from `makani_sfno/`) | `polaris/polaris_pack_e3sm_alldata_full.pbs` (success: `CONVERT_ALLDATA_OK` + `PACK_ALLDATA_COMPLETE`) |
| makani training | `polaris/polaris_sfno_alldata_full.pbs` — chain, extend as measured |
| **PhysicsNeMo** transferred dataset (expected location; not yet on disk) | `/lus/eagle/projects/lighthouse-uchicago/members/jesswan/e3sm_seqzarr_allyears/{e3sm_train.zarr, e3sm_val.zarr}` |
| PhysicsNeMo dataset verification (submit from `physicsnemo_sfno/`) | `polaris/polaris_verify_store.pbs` (success: `STORE_VERIFY_OK`) |
| PhysicsNeMo dataset rebuild (Route B) | `polaris/polaris_zarr_e3sm_allyears.pbs` (success: `ZARR_ALLYEARS_COMPLETE`; needs `-v CONFIRM_ALLYEARS=1`) |
| PhysicsNeMo training | `polaris/polaris_sfno_allyears.pbs` — chain, extend as measured |
| training outputs (every model) | `/lus/eagle/projects/lighthouse-uchicago/members/<you>/runs/…` |
| project disk quota (shared!) | `myquota` — see §1 storage budget |
| engineering companion (costs, failure modes, open decisions) | `polaris_pipelines_plan.md` |
