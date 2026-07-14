# Running the four test jobs on Polaris

Hi Jess — this page walks you through running four of our weather models on Polaris. Each one
is a short **test run**: it trains for a few minutes on one machine, just far enough to prove
the model still runs from start to finish. You are not training a real model here, and you
don't need to change any code.

There are four to run: **PanguWeather**, **SI**, **Makani**, and **PhysicsNeMo**. Each is a
single command. All four already work — and PanguWeather and SI have been run from a second
person's account specifically to check nothing was tied to Rahul's — the results near the
bottom are from real runs, so
you have something to compare against.

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

If it says anything else, send Rahul the output.

> **Why you can't just borrow Rahul's copy.** His installation is wired to *his* copy of the
> code. If you used it, your runs would quietly execute whatever he happens to be editing
> that day, and your results could change for reasons that have nothing to do with you.
> Your own copy makes your runs yours.
>
> PanguWeather and SI don't need this step — they use software already set up for the whole
> project, in a shared folder you can read.

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

You don't need to prepare any data. About 75 GB of it is already prepared and shared with you
automatically.

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

## If something goes wrong

1. **Check the last line of the log.** `rc=0` is success; anything else is a failure.
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
5. **Still stuck? Email Rahul** (rmehta1987@gmail.com) with the full path of the log and its
   last 30 lines. Don't spend more than a few minutes on it.

## A few things worth knowing

- **Don't add `--debug`** to the PanguWeather job. It sounds helpful, but it forces everything
  onto a single GPU and all four processes crash by piling onto the same one.
- **The first PanguWeather run may pause for several minutes** near the start while it prepares
  a large data file. That's expected, and it only happens once.
- **These are deliberately tiny models.** They prove the code runs; they say nothing about
  speed or accuracy. Don't read anything into the loss numbers.
- **SI's test uses data from 2015 and the first few days of 2016.** It would need a code fix
  to run across a leap year, so please don't change the years in the settings file.
- **S2S and S2S-Lightning aren't in this list.** Their data hasn't been copied to Polaris yet,
  so they can't run. If you try one it stops immediately with `ERROR ERA5_NOT_STAGED` —
  expected, and not something you did.

---

*Technical background — why the environment is per-user, what's shared, the known bugs and
traps, and the cluster details — is in `polaris_pbs_notes.md`. You don't need any of it to run
these.*
