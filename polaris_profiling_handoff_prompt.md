# Handoff: start profiling on Polaris

You are picking up the pedramh-profiling project on **Polaris (ALCF, PBS Pro)**. Bring-up is
done. Your job is to **find out where the time actually goes** in the four runnable models,
and to build the two remaining pieces that gate optimization.

Read **DESIGN.md** (what/why) and **CLAUDE.md** (how to work here) first. **CHANGELOG.md** is
the living document — read it before you touch anything, and update it before you stop.
Cluster facts and every trap found so far: **`polaris_pbs_notes.md`**.

---

## 1. The one distinction that decides what you may do

**Profiling is unblocked. Optimizing is not.**

- **Measuring** where time goes — nsys, step timings, memory, data-loader idle — changes
  nothing the model computes, so you can do it today, on the models that are green.
- **Changing the hot path** — `torch.compile`, precision, fused kernels, DDP tuning — is
  gated on DESIGN §4's equivalence oracle, which is **not executable yet**. Do not start.

Of §4.0's three prerequisites, one is done:

| prerequisite | state |
|---|---|
| seed knob | ✅ **DONE** — `--seed`/`$S2S_SEED`/YAML + `--deterministic`, in `s2s/v2.0/utils/seeding.py`. 10 tests (`SEEDING_OK`) on CPU **and A100** (job 7253738). Opt-in: no seed ⇒ legacy path byte-for-byte. |
| `tiny_baseline.yaml` | ⬜ **blocks baseline capture** — no small config exists; `test.yaml` is the full ~79M-param model despite the name (it OOMed a 93 GiB H100 at its defaults) |
| VAE noise-fixing hook | ⬜ **blocks baseline capture** — and it is the subtle one, see §5 |

So: **profile now, and build those two in parallel.** The moment they exist, capture
baselines (§4.1) and only then touch the ladder (§5).

---

## 2. What is green, and what those greens are worth

All verified from logs, and re-verified **as a second user** (`PYTHONNOUSERSITE=1`, which
reproduces another member's view of the filesystem):

| model | 4-GPU green | reproducible by another member |
|---|---|---|
| PanguWeather SFNO | 7252271 | ✅ **7253591** — loss **0.3411, bit-identical** to the installer's run |
| SI (bench) | 7252700 | ✅ **7253603** — step_med 0.399 vs 0.400, peak 30.69 GB |
| Makani SFNO | 7253465 | model probe 7253837 |
| PhysicsNeMo SFNO | 7252933 | model probe 7253862 |
| Toolchain probe | **7253681** | imports the real `modules.train_module`, `sys.path` free of `~/.local` |

**Blocked, honestly:**
- **S2S + S2S-Lightning** — ERA5 is not staged on Polaris. Scripts are delivered and their
  *import chain* is verified, but they have **never run here**. "Delivered" ≠ "proven".
- **SI full training** — a correctness bug, not a resource problem (§5).
- **Makani/PhysicsNeMo full training** — needs ~1.75 TB of one-time data prep, and the
  project was at **15.34 TB of a 15 TB quota** at handoff. Check `myquota` before assuming.

---

## 3. The guardrails. These are not style preferences — each one is a bug that already happened

Every item below cost real time this session. They are the reason the greens above can be
trusted. Violating them does not produce a failure; it produces a **confident wrong answer**,
which is worse and much harder to find later.

**1. `rc=0` is not a pass. Key on the work token.**
Makani's re-run resumed a finished checkpoint, trained **zero steps**, printed
`Total training time is 0.00 sec`, and exited **0** (job 7253454). Every future re-run would
have done the same. For any resumable trainer, look for the loss line / the written
checkpoint — not the exit code. The smoke now forces a fresh `RUN_NUM` and gates on
`ERROR NO_CHECKPOINT`; full training deliberately does the opposite (it *wants* the resume).

**2. A green is only green for whoever ran it.**
Pangu/SI depended on `pip install --user` packages under `/home/rmehta1987/.local`. **ALCF
homes are mode 0700**, so those greens were true for exactly one person while the whole point
of the deliverable is that the project can reproduce them. It is invisible to the one person
who could fix it, because their own runs pass. **When a result must be reproducible by
others, prove it as them:** re-run with `PYTHONNOUSERSITE=1`. Shared deps live in
`$POLARIS_TOPUPS`; `polaris_require_topups` (all 8 base-conda jobs) now hard-fails on a
regression, and its two branches are tested.

**3. A check that cannot fail proves nothing. Ask what the check would do if the thing were broken.**
This bit four separate times in one session:
- the probe imported `common, data, modules` — **namespace packages with no `__init__.py`**,
  so it executed none of the smoke's code and sailed over a missing `cf_xarray` while the
  docs claimed the port's env was "proven by the probe";
- a network probe called **HTTP 404 a failure** and reported "compute nodes CANNOT reach
  W&B" while its own output showed they could (a 404 is the server *answering*);
- `wandb.Api().viewer()` raised `TypeError: 'User' object is not callable` — which reads as
  an auth failure but is auth **succeeding** (`viewer` is a property);
- a gate was "tested" against an **empty file** because the extraction silently produced
  0 lines, so it returned rc=0 having run nothing.

**4. "Missing for everyone" only means "unused" for code that has actually RUN.**
I dropped `cf_xarray` from the top-ups on the reasoning *"the installer lacks it too, so it's
off the smoke path"*. The port smoke has **never run on Polaris**, so for it "missing for
everyone" meant **broken for everyone** — it would have died at import right after a
multi-TB Globus stage. Absence is only evidence where execution happened.

**5. Blocklists only catch what you already thought of. Enumerate instead.**
My top-ups guard blocklisted `torch/numpy/nvidia/triton` and therefore missed that
**`h5netcdf` was already in the base conda** — the shared dir was silently upgrading it
1.6.4 → 1.8.1 for every job. It now enumerates the directory and asks a clean interpreter
about each name.

**6. Never `pip install --user` anything the project must share** (see #2). And never commit a
key: a **bare token in `~/.netrc` makes `netrc.netrc()` raise a parse error with the key in
the message**, leaking it into any log that touches it. That happened here.

**7. A wrong number is a wrong term. Refuse; never fudge.** (CLAUDE.md #1/#11.)
SI's config says `calendar: 'standard'`, but E3SM is **noleap** — 1460 files/year for *every*
year, 2016 and 2020 included. `si/data/amip_new.py:666-670` derives the **filename** from the
date, so from Mar 2016 every sample reads the wrong day (`2016_0240.h5` where the correct
file is `2016_0236.h5`) and December overruns to a file that does not exist. **The loss still
falls.** `si/train_polaris_full.pbs` therefore *refuses to start*
(`ERROR SI_CALENDAR_LEAP_MISMATCH`) rather than train on misaligned data. Do not "fix" this
by clamping the index or trimming December.

**8. Measure; do not guess — especially resources.**
`e3sm_full.yaml`'s `batch_size: 8` comes from the group's baseline targeting **H100 (80 GB)**;
Polaris is **A100 (40 GB)**. Rather than argue, probe: full-size makani is
**147,818,882 params** (the smoke's is 54,258 — a 2,724× toy) at **8.82 GB of 40** — it fits
with room. The `*_full_probe.pbs` scripts exist for exactly this; they also caught two bugs
that would each have burned days of queue (a `PYTHONPATH` missing `/src`; PhysicsNeMo's
smoke and full training **sharing a checkpoint dir inside the repo**).

**9. Read the log, not the exit code; and never claim a step passed without reading its output.**

---

## 4. Do this first

**Profiling — the actual assignment.** nsys is on Polaris:
`/soft/compilers/cudatoolkit/cuda-12.9.1/bin/nsys` (2025.1.3). You inherit real
infrastructure — use it rather than reinventing:

- knobs already plumbed: `S2S_BENCH_{WARMUP,STEPS,CSV}`, `S2S_NVTX`, `S2S_AMP_DTYPE`,
  `TORCH_COMPILE_MODE`, `SI_BENCH_*`, `SI_NVTX`, `S2S_PRECISION`
- parsers: `s2s/v2.0/HPC_scripts/parse_nsys.py`, `compare_nsys.py`, `si/parse_nsys.py`
- prior art + the reporting style to match: `s2s/v2.0/bench_report.md`,
  `si/bench_midway_notes.md`

1. **Profile the four green models on Polaris A100s** and write `polaris_bench_report.md` in
   the style of `bench_midway_notes.md`. For each: step-time distribution, peak memory,
   data-loader idle fraction, and the top kernels. The Midway numbers are **H100-NVL**; these
   are **A100-40GB** — different node class, so they are *not* comparable. Say so rather than
   putting them in one table.
2. Mirror the existing nsys scripts for Polaris (`midway_bench_nsys.sh` →
   `polaris_bench_nsys.pbs`), following CLAUDE.md #7: add beside, never edit the Midway path.
   **Do not rename or drop an NVTX range or CSV column** (#10) — that silently invalidates
   every prior comparison and breaks `parse_nsys.py`. S2S and SI use *different* range names;
   don't cross them.
3. Start with the cheap, high-value question: **is the hot path GPU-bound or input-bound?**
   The smokes ran `num_data_workers: 0/1`; if the answer is "waiting on data", every kernel
   optimization on the ladder is premature.

**The two §4.0 prerequisites**, in parallel:

4. **`tiny_baseline.yaml`** — few layers/channels, `batch_size 1`, `num_data_workers 0`, no
   wandb, no checkpoint. It must be genuinely small: `test.yaml` is *not* a small config.
5. **The VAE noise hook — read DESIGN §4.0 carefully.** The reparameterization draw is
   stochastic, and `torch.compile`/FlexAttention can change RNG kernel selection and
   consumption order — so **a correct optimization can still produce different ensemble
   outputs**. A seed alone does not fix this. Fix the noise (a dedicated `torch.Generator`
   for the reparam draw, or inject a fixed epsilon) or compare a deterministic pre-sample
   quantity. **Never** compare a bitwise hash of a stochastic output. This is the piece most
   likely to manufacture a "failure" that isn't one — and under guardrail #7 the response to
   a mismatch is to trace it, never to widen the tolerance.

---

## 5. Known-open bugs (documented, not fixed)

- **SI `disassemble_input`** — fixed in `train_module.py` (`1fef2473`) but **still open** in
  `si/bias.py:226`, `si/modules/ae_module.py:68`, `si/modules/combined_module.py:185` and
  `:287`. They rely on defaults `nsurface=6, ndiagnostic=15` baked to the Midway AMIP config;
  E3SM has **3** diagnostics, so the channel split is wrong — plausible tensors, wrong
  contents. The bench never hit it (`limit_val_batches=0`); training validates, so it will.
  The real repair is to make `disassemble_input` **require** the counts so a missed caller
  fails loudly.
- **SI calendar** — guardrail #7. The fix: make `has_year_zero` follow the calendar instead
  of being forced `False` (`amip_new.py:612-615, 622, 668`), then set `calendar: 'noleap'`.
  Ship it with a test pinning the date→filename mapping across Feb/Mar of a leap year
  (`2016_0236`, not `2016_0240`).
- **SI SST normalization** is degenerate (npz mean ≈110 vs °C land-filled data).
- **`s2s/v2.0/inference.py:21`** and **`PanguWeather/v2.0/long_inference.py:34`** bare-import
  `dask`, which is in neither the base conda nor `$POLARIS_TOPUPS`. No Polaris script runs
  them, so it is deliberately not installed — but per guardrail #4 that is an **unrun path**,
  so check the chain before the first inference run.
- **No seed knob on the Pangu/makani/physicsnemo paths** — their smoke losses move run to run
  (makani: 2.19/2.05 vs 2.61/2.38 on identical code). Fine as proof-of-life; **not** an
  equivalence baseline.

## 6. Cluster facts you will otherwise rediscover the hard way

- **Queues:** `debug` = 1 h, ≤2 nodes, **1 running + 1 queued job per user** (this will
  serialize your day). `preemptable` = up to 72 h, 1–10 nodes, **and your job will be killed
  without warning**. `prod` needs **≥10 nodes**. 
- **Preemption is self-healing only with `#PBS -r y`.** `preempt_order = RD` — PBS requeues
  before deleting — but only for rerunnable jobs, and it **defaults to `Rerunable = False`**
  (checked on our own jobs). With `-r y` PBS re-runs the script **from the top**, which is
  safe only because the full scripts are idempotent and resume from a stable `RUN_NUM`.
  ⚠️ **The resume itself is verified by INSPECTION, not observation.** Pangu resumes via
  `train.py:3851  elif checkpoint_exists: params['resuming'] = True`, and the full script
  correctly omits the `--fresh_start` that the smoke passes — so the logic is right. But at
  handoff the empirical test (run → kill → resubmit → see it resume) had **not** come back
  from the queue. **Confirm it yourself before trusting a multi-day run to it:**
  `qsub -q debug -l walltime=00:50:00 -v TRAIN_YEAR_END=2016,RUN_NUM=resumetest <full script>`
  twice; the second must log `Resuming from existing checkpoint`. Per guardrail #9, do not
  take my word for it.
- **Network:** compute nodes reach the internet **only via the ALCF proxy**
  (`https_proxy=http://proxy.alcf.anl.gov:3128`), which `module load conda` exports. Verified
  on-node (job 7253810: `api.wandb.ai/healthz` → 200; direct → 000). Earlier notes claiming
  "no outbound network" were wrong.
- **W&B works from a compute node** (job 7253874: `rc=0`, loss unchanged at 0.3411, run
  synced). But `WANDB_MODE` only drives the *client* — Pangu/makani also need `log_to_wandb`
  flipped (the launchers do it), SI's bench hardcodes `wandb_mode="disabled"`, and
  PhysicsNeMo uses MLflow. Pangu's config hardcodes **someone else's entity**; the launcher
  rewrites it, or every member's runs land in one account.
- **Login node has a process cap** — `git push` needs `-c pack.threads=1`, and python with
  default OMP threads can fail to fork. Use `OMP_NUM_THREADS=1` for quick login-node checks.
- Never `find /` or scan outside the repo (CLAUDE.md #2). Lustre needs
  `HDF5_USE_FILE_LOCKING=FALSE`.

## 7. Ground rules for the work itself

Small commits, each gated on the check it can run. Every change ships its test. Update
`CHANGELOG.md` before you stop — what you did, the **measured** result, and what you learned
or is now blocked. Record failed approaches so they are not re-tried. `main` is
branch-protected: branch → PR, and a solo session cannot self-approve.

**And the meta-rule behind all of §3:** when something passes, ask *what would this have done
if it were broken?* If you cannot answer, you have not tested it yet.
