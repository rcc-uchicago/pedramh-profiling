# Running the four test jobs on Polaris

This page walks you through running four of our weather models on Polaris. Each one
is a short **test run**: it trains for a few minutes on one machine, just far enough to prove
the model still runs from start to finish. 

You are not training a real model here, and you
don't need to change any code.

There are four to run: **PanguWeather**, **SI**, **Makani**, and **PhysicsNeMo**. Each is a
single command. All four have been run successfully, and the results near the bottom come from
those real runs, so you have something to compare against.

Everything you run writes only into your own folder. You can't break anyone else's work, and
nothing here can damage the shared data.

**If you get stuck at any point, email Rahul (rmehta1987@gmail.com)** with the location of the
log file and its last 30 lines (`tail -30 <the log file>`). That's always the right move —
none of this is worth spending an afternoon on.

---

## Step 1 — Get the code (once)

Log in to Polaris, then:

```bash
cd /eagle/projects/lighthouse-uchicago/members/jesswan
git clone -b polaris-pbs-bringup https://github.com/rcc-uchicago/pedramh-profiling.git
cd pedramh-profiling
```

This copies the code into your own folder. It takes a minute or two.

## Step 2 — Set up an environment (once, about 10 minutes)

Two of the four models (Makani and PhysicsNeMo) need extra software. This installs it into
your folder:

```bash
bash polaris_setup_sfno_venv.sh
```

Wait for it to finish, and check that the last line says:

```
SFNO_VENV_OK
```

## Step 3 — Run the four jobs

Run these **from the `pedramh-profiling` folder** you just created. Copy them one at a time.
The brackets matter: they keep each command independent, so the order doesn't matter and one
doesn't affect the next.

```bash
( cd PanguWeather/v2.0 && qsub HPC_scripts/polaris_train_e3sm_sfno.pbs )
( cd si                && qsub bench_polaris.pbs )
( cd makani_sfno       && qsub polaris/polaris_sfno_smoke.pbs )
( cd physicsnemo_sfno  && qsub polaris/polaris_sfno_smoke.pbs )
```

Each prints a job number like `7253330.polaris-pbs-01...`. Note it down — it's how you find
the results.

**Only one of your jobs runs at a time.** You can submit all four; they'll simply queue and
run one after another.

### Where the data actually is

You don't prepare any data — it already exists and the scripts find it for you. This is only
so you know what is being read, and can tell whether a path in a log looks right.

**The source archive** — the raw E3SM climate simulation everything derives from. Read-only,
shared, ~2 TB, 51,100 files covering 2015–2049 (four snapshots a day, 1,460 per year):

```
/eagle/lighthouse-uchicago/members/jesswan/AI4SRM/data/
    E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/h5/plev_data/{year}_{0000..1459}.h5
```

That one is in *your* AI4SRM folder — it's the group's copy of the simulation output.

**The prepared copies** — each model wants the same data in its own format, so it was
converted once and shared. About 92 GB in total, all read-only to you:

| Used by | What it is | Where |
|---|---|---|
| PanguWeather | statistics + climatology it needs | `.../members/mehta5/pangu_polaris_data` (16 GB) |
| SI | renamed/restructured copy | `.../members/mehta5/si_e3sm_stage` (56 GB) |
| Makani | packed into Makani's layout | `.../members/mehta5/data/e3sm_makani` (18 GB) |
| PhysicsNeMo | converted to Zarr | `.../members/mehta5/e3sm_seqzarr` (1.7 GB) |

(all under `/eagle/projects/lighthouse-uchicago/`)

**Two things worth knowing about those paths:**

- Seeing `mehta5` in a log is **normal and correct** — that's the shared, read-only prepared
  data. What must say `jesswan` is anything the job *writes*: your logs, checkpoints and
  results, which go to `/eagle/projects/lighthouse-uchicago/members/jesswan/`.
- These prepared copies are **small subsets** made for testing — Makani's holds 400 of 2015's
  1,460 snapshots, PhysicsNeMo's holds 80. That's on purpose: they're for checking the code
  runs. Real training uses different, much larger copies (see the full-training section).

## Step 4 — Watch it, and find the results

Check on your jobs:

```bash
qstat -u $USER
```

`Q` means waiting in the queue, `R` means running. When a job disappears from that list, it's
finished. Each test takes roughly 2–10 minutes once it starts.

When it finishes, a log file appears **in the folder you submitted from**, named after the
job — for example `PanguWeather/v2.0/pangu_e3sm_sfno.o7253330`. It only appears at the *end*,
so don't worry if it's missing while the job is still running.

To read it:

```bash
tail -30 PanguWeather/v2.0/pangu_e3sm_sfno.o7253330
```

(If you're already inside the folder you submitted from, just the file name is enough.)

## What "it worked" looks like

| Model | You submitted from | Look for this in the log |
|---|---|---|
| **PanguWeather** | `PanguWeather/v2.0` | `DONE ---- rank 0` (and ranks 1, 2, 3), plus a `Loss:` number |
| **SI** | `si` | a block headed `BENCH RESULT` |
| **Makani** | `makani_sfno` | `Saving checkpoint`, plus a `loss=` number |
| **PhysicsNeMo** | `physicsnemo_sfno` | `Saved training checkpoint`, plus a `loss =` number |

Each log also has an `rc=0` line near the end, which means success — any other number means
it failed. For PanguWeather, Makani and PhysicsNeMo it's the very last line. **SI is the
exception**: it prints a table of numbers *after* its `rc=0` line, so look a little further
up rather than only at the bottom. If you see anything other than `rc=0`, send Rahul the log.

Don't rely on `rc=0` alone for Makani. A Makani run that has already finished once can start
up, decide there's nothing left to do, and stop straight away still reporting `rc=0` — so
check that a `loss=` number is actually there too.

SI also writes a small table of timings here:

```
/eagle/projects/lighthouse-uchicago/members/jesswan/polaris_logs/si_bench_polaris_<jobnumber>.csv
```

## Results from Rahul's runs, for comparison

Yours should land in the same ballpark. They won't match exactly, and that's normal.

| Model | Job | What it produced |
|---|---|---|
| PanguWeather | 7252271 | training loss 0.34, validation loss 0.70, about 4 minutes |
| SI | 7252700 | 0.40 seconds per step, used 31 GB of the GPU's 40 GB |
| Makani | 7253465 | training loss 2.61, validation loss 2.38 |
| PhysicsNeMo | 7252933 | loss 0.89, validation error 0.54 |

## Tracking your runs in Weights & Biases

Optional for the test runs, worth doing before any real training.

First get your API key: sign in at https://wandb.ai/authorize and copy it.

Then **pick either one** of these, on a login node. They do the same thing.

**Option A — let wandb do it (simplest):**

```bash
module use /soft/modulefiles && module load conda && conda activate base
wandb login
```

Paste the key when prompted. That's it.

**Option B — write the file yourself.** Create `~/.netrc` containing exactly:

```
machine api.wandb.ai
  login user
  password YOUR_KEY_HERE
```

then lock it down — wandb won't use it otherwise, and neither should you:

```bash
chmod 600 ~/.netrc
```

The word `login user` is literal; leave it as-is. (`machine api.wandb.ai` is the part that
actually matters — verified.)

> **The one mistake that bites.** Don't put the bare key in `~/.netrc` on its own line with
> nothing else. It looks like it should work and it doesn't — and worse, the error it causes
> **prints your key into the log**, so anyone reading that log now has it. It happened here.
> If you do it by accident, assume the key is compromised and get a new one from
> https://wandb.ai/authorize.

Either way, **never put the key in a file inside the repo** — that would publish it to
everyone with access.

**To check it worked** (optional):

```bash
bash polaris_setup_wandb.sh          # prints WANDB_OK, tells you which account you're on
```

You can also just put your API key into ~/.netrc without using the bash script.

**Then add one thing to any submit line:**

```bash
( cd PanguWeather/v2.0 && qsub -v WANDB_MODE=online HPC_scripts/polaris_train_e3sm_sfno.pbs )
( cd makani_sfno       && qsub -v WANDB_MODE=online polaris/polaris_sfno_smoke.pbs )
```

Your run appears live at wandb.ai, under the project `pedramh-profiling` and **your own**
account.

**Only PanguWeather and Makani can do this.** The other two can't, and it isn't worth your
time trying:

| Model | Live tracking? |
|---|---|
| PanguWeather | yes |
| Makani | yes |
| SI (the test job) | **no** — its benchmark has W&B switched off in the code itself |
| PhysicsNeMo | **no** — it records to MLflow instead, and doesn't use W&B at all |

**Without that, runs are "offline"** — still fully recorded, just written to your own folder
(`/eagle/projects/lighthouse-uchicago/members/jesswan/wandb/`) instead of uploaded. That's
the default on purpose: an offline run can't stall or fail because of a network problem. You
can upload one later from a login node:

```bash
wandb sync /eagle/projects/lighthouse-uchicago/members/jesswan/wandb/offline-run-*
```

> Compute nodes do reach the internet, but only through ALCF's proxy — which the job scripts
> set up for you. So this works from inside a job; you don't need to do anything special.

## Running the real thing (full training)

Everything above is a **test run**. The full-training jobs are separate scripts, named
`*_full`, so the test scripts stay quick and safe to re-run:

| Model | Data prep first (hours, big) | Then training |
|---|---|---|
| PanguWeather | *none needed* | `( cd PanguWeather/v2.0 && qsub HPC_scripts/polaris_train_e3sm_sfno_full.pbs )` |
| Makani | `qsub polaris/polaris_pack_e3sm_full.pbs` (~750 GB) | `( cd makani_sfno && qsub polaris/polaris_sfno_full.pbs )` |
| PhysicsNeMo | `qsub polaris/polaris_zarr_e3sm_full.pbs` (~1 TB) | `( cd physicsnemo_sfno && qsub polaris/polaris_sfno_full.pbs )` |
| SI | — | **not ready — see below** |


Four things that differ from the test runs, and will surprise you if you don't expect them:

1. **A "failed" job is usually normal.** Full training runs on the `preemptable` queue, which
   is the only way to get more than an hour on one machine. Your job **will be killed**
   without warning, sometimes minutes in, and report a failure. That's the deal, not a bug.
   Just submit the same command again — it picks up from its last checkpoint. Only worry if
   there's an actual error message in the log.
2. **The models are much bigger.** Makani's test model is a toy of ~54,000 numbers; the real
   one is thousands of times larger. The test isn't a small version of the real thing — it's
   a different thing that happens to exercise the same code.
3. **The data is different too.** Full training needs its own much larger prepared copies,
   which is why there's a prep job first. The training script refuses to start if it finds
   the small test data instead — that check exists so a multi-day run can't quietly train on
   400 snapshots and look fine.
4. **SI can't do full training yet.** Its script exists but deliberately stops with
   `ERROR SI_CALENDAR_LEAP_MISMATCH`. The climate data has 365 days in every year, but the
   settings tell it to use a normal calendar with leap days — so from March 2016 onward it
   would read the *wrong day's* file and never notice. Nothing looks broken; the numbers just
   quietly stop meaning anything. It needs a code fix first. Please don't work around it.

## If something goes wrong

1. **Find the `rc=` line near the end of the log.** `rc=0` is success; anything else is a
   failure. (It's the last line for every model except SI, which prints a table after it.)
2. **Look near the top of the log.** There's a short block listing the folders the job used.
   Anything the job *writes* — its logs, its results, its run folder — should have **your**
   name in it. Some paths will legitimately say `mehta5`: that's the shared, read-only data
   and software everyone uses, and it's meant to look like that. If the folders being written
   to aren't yours, submit again naming your folder:
   ```bash
   qsub -v POLARIS_MEMBER=jesswan <the same script>
   ```
   (It has to be passed to `qsub` like this — setting it in your shell beforehand won't reach
   the job.)
3. **For a clearer error message**, re-run the same job on a single GPU:
   ```bash
   qsub -v NPROC=1 <the same script>
   ```
   Slower, but the error is usually much easier to read.
4. **If the log says `PHYSICSNEMO_WRONG_CHECKOUT`**, Step 2 didn't finish. Re-run
   `bash polaris_setup_sfno_venv.sh` and check for `SFNO_VENV_OK`.
5. **If the log says `No module named` something** (for PanguWeather or SI), a piece of
   shared software has gone missing. It lives in a folder everyone can read, so this
   shouldn't happen — but if it does, tell Rahul; the one-line repair on his side is
   `bash polaris_setup_base_topups.sh`. Please don't try to install anything yourself.
6. **Still stuck? Email Rahul** (rmehta1987@gmail.com) with the full path of the log and its
   last 30 lines. Don't spend more than a few minutes on it.

## A few things worth knowing

- **Don't add `--debug`** to the PanguWeather job. It sounds helpful, but it forces everything
  onto a single GPU and all four processes crash by piling onto the same one.
- **The first PanguWeather run may pause for several minutes** near the start while it prepares
  a large data file. That's expected, and it only happens once.
- **These test runs use deliberately small models** (except PanguWeather, which is full-size
  but only runs for one pass). They prove the code works; they say nothing about speed or
  accuracy. Don't read anything into the loss numbers.
- **SI's test uses data from 2015 and the first few days of 2016.** It would need a code fix
  to run across a leap year, so please don't change the years in the settings file.
- **S2S and S2S-Lightning aren't in this list.** Their data hasn't been copied to Polaris yet,
  so they can't run. If you try one it stops immediately with `ERROR ERA5_NOT_STAGED` —
  expected, and not something you did.

