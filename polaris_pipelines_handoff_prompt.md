# Handoff: design how to run the three E3SM pipelines (convert → train → infer)

**Your job is to DESIGN, not to launch.** Produce a plan for running data conversion, training,
and inference for **PanguWeather**, **makani**, and **PhysicsNeMo** on Polaris — plus one concrete
implementation task (§6, the PhysicsNeMo CSV tee). Nothing here is submitted until a human says so.

Read first: **CLAUDE.md** (house rules; it owns the cluster facts), **DESIGN.md** (§1 goals — note
it was rewritten 2026-07-16 to *implement-first, then-profile*), **CHANGELOG.md** (status).

---

## 1. Where things actually stand (2026-07-16)

Everything below was measured this session or read from a cited `file:line`. **Do not trust a
comment; verify against code or data.** This project has repeatedly been burned by docs asserting
things the code contradicts — including several this session had to retract.

### Settled by the science owner
**The three cloud variables (`CLDICE`, `CLDLIQ`, `CLOUD`) are excluded from ALL models**
(confirmed 2026-07-16). All three pipelines now agree: **108 of the archive's 162 channels**.

- PanguWeather: already excluded (`E3SM_SFNO_H5_POLARIS.yaml:52`, commented out).
- makani: never had them (its channel map is a PlaSim-inherited contract).
- PhysicsNeMo: `e3sm_h5_to_seqzarr.py` now has `EXCLUDED_VARS`; **162 → 108, predicted 157 → 103**,
  store shrinks 33.3%. Verified against `2015_0000.h5`: 54 channels dropped, 0 clouds survive.

### The state of the tree

**All committed** on branch `polaris-data-prep` (`2fdcb0ff`, `3eb9d839`) — not pushed, not on
main. Nothing here has been submitted since job 7258626. `git log --oneline -3` first; the two
commit messages are the honest record, including what is unproven.

| area | what exists |
|---|---|
| `physicsnemo_sfno/polaris/e3sm_h5_to_seqzarr.py` + `verify_seqzarr.py` | cloud exclusion; verifier kept in lockstep |
| `PanguWeather/v2.0/utils/metrics.py` | climatology on CPU — **the unproven fix (§4)** |
| `PanguWeather/v2.0/config/E3SM_SFNO_H5_POLARIS_ALLDATA.yaml` | full years (2015-2044 train / 2045-2048 val), 108 ch |
| `PanguWeather/v2.0/HPC_scripts/polaris_train_e3sm_sfno_alldata_{smoke,full}.pbs` | training smoke + production launcher |
| `PanguWeather/v2.0/HPC_scripts/polaris_val_e3sm_sfno_alldata_smoke.pbs` + `run_validation_smoke.py` | **the validation smoke — built, NEVER RUN (§4)** |
| `makani_sfno/polaris/convert_e3sm_to_makani_alldata.py` + 2 yaml + 2 pbs | all-data pack: 100 state + 1 diag + 7 forcing |
| `physicsnemo_sfno/polaris/{allyears_split.sh,polaris_*_allyears*.pbs}` | all-years convert + train + smoke |
| `polaris_e3sm_variable_reference.md`, `data_for_training.md` | the measured data facts |

⚠ **`_alldata` is a misleading name.** Those files were built to enable clouds (162 ch); the
science decision reverted that, so they now carry only the **full-year range**. The Pangu config's
header already says "ALL-YEARS variant"; the *filenames* still say `_alldata`. Renaming to
`_allyears` would be honest — if you do, grep every reference (PBS, docs, CHANGELOG).

### Proven on Polaris
- **Pangu, 162 ch: job 7258382 — `ALLDATA_SMOKE_OK … peak_mem_gb=27.045`** on a 40 GB A100.
  Since 108 ch is strictly smaller, memory is a solved problem for Pangu training.
  It also proved the model **auto-sizes from the config lists** (`sfnonet.py:760-765` derives
  `in_chans`/`out_chans` from `dataset.variable_list_in/_out` — nothing hardcoded).
- **Pangu 108 ch bench: `step_med` 0.602 s (n=80)** — the trustworthy number.
  The `_alldata` smoke CSV (n=12) says on its face never to compare it; don't.

---

## 2. What you must design

For **each** of the three pipelines, the full path: **convert → train → infer**, on Polaris.

| pipeline | data prep | status |
|---|---|---|
| **PanguWeather** | **none** — reads `h5/plev_data` directly | training smoke green **at 162 ch (7258626)**; config is now 108 → **re-run**. Validation + inference **never run** |
| **makani** | pack `convert_e3sm_to_makani_alldata.py` → its own h5 contract | pack+train green **at 154 ch (7258407)**; converter is now 100 → **re-run** (guards will reject the stale pack: `WRONG_CONTRACT`) |
| **PhysicsNeMo** | convert h5 → zarr (`e3sm_h5_to_seqzarr.py`) | converter verified at 103 ✓ (ran it: `CONVERT_OK` → `SEQZARR_VERIFIED` 6×108 bitwise). **Training never run at 103** — last green was 157 ch (7252933) |

> ⚠ **Every green above except the converter predates the change it would validate.** That is not
> bad luck — it is the same structural problem as the bench harness (§4a): the gates ran before
> the thing they would catch existed. **Trust no job id in this repo without checking what
> contract it ran at.** First four jobs, in order: fixed Pangu smoke (108) → makani (100) →
> PhysicsNeMo (103) → the validation smoke.

**Inference is the biggest gap.** This session was all training-side. Nothing here has produced a
forecast. Each model's inference path is a different beast:
- Pangu: `inference.py` / `long_inference.py` / `ensemble_inference.py` (note `long_validation`
  must stay False — the bias `.npy` files are NOT staged on Polaris, only `bias/old/TREFHT`).
- makani: `src/sfno_inference/` — ⚠ `checkpoint_loader.py:75` **hard-asserts
  `N_in_channels == 58`**. Any checkpoint from a non-52/1/6 contract is rejected. Two independent
  agents found this. If makani trains on a widened contract, **it cannot currently be evaluated.**
- PhysicsNeMo: `model_packages.py` writes an inference package; the BatchNorm state is baked into
  it (§5).

---

## 3. The launcher rules (get these wrong and nothing runs)

**Polaris/PBS: never `srun`.** The launcher is **per-model**:
- PanguWeather → `torchrun --standalone --nproc_per_node=$NPROC`
- makani / PhysicsNeMo → `python -m torch.distributed.run`
  (their venv has no `torchrun`; the bare name resolves to the BASE conda's, whose shebang pins
  the wrong python)
- Every Polaris script must `source polaris_env.sh`.
- PBS needs `-A lighthouse-uchicago`, `-l select=1:system=polaris`, **`-l filesystems=home:eagle`**
  (jobs are **rejected** without it). `-q debug` ≤1 h and **1 job in `Q` per user**; long runs use
  `-q preemptable`.
- Copy the env-bootstrap block **verbatim** from the same model's existing script — module
  ordering differs per model on purpose.

⚠ **Submit from the directory the script expects.** Both scripts resolve `polaris_env.sh`
relative to `$PBS_O_WORKDIR` (`cd PanguWeather/v2.0 && qsub HPC_scripts/…`). Submitting from the
repo root fails in 1 second with `No such file or directory`. This bit me twice.

---

## 4. Open problems the design must confront

These are real, measured, and unsolved. **Do not design around them silently.**

**a) Production validation is expected to OOM — and until now no smoke could see it.**
`train.py::validate_one_epoch` is called (`:969`) **before** the first `save_checkpoint` (`:1001`).
Adversarial arithmetic: 22.0 GiB training state + 13.48 GiB GPU-resident climatology + 2.26 GiB
lead-60 targets + transients ≈ **38.8–40.8 GiB vs ~38–39.3 usable**. The production run would
train ~2 h, die at validation, requeue, repeat — **forever, zero progress, never checkpointing.**

**Attempted fix, UNVERIFIED:** `utils/metrics.py` now holds the climatology on CPU and gathers
the `[batch]` slice per call (~9 MB/batch instead of 13.5 GiB resident). Job 7258626 passed but
proves only that *training* still works — the bench harness `sys.exit(0)`s (`train.py:1305`,
inside `_bench_finalize`) upstream of validation, so **nothing has executed the changed code.**
Blast radius is **PanguWeather only**: `s2s/v2.0/utils/metrics.py` does not exist and the sole
importers are PanguWeather's own `train.py`/`train_optimized.py` (verified). CLAUDE.md #5 governs
`s2s/v2.0/` — which the port *imports* — and does not reach here.

**The tool now exists: `HPC_scripts/polaris_val_e3sm_sfno_alldata_smoke.pbs` + `run_validation_smoke.py`.
RUN IT FIRST — it answers the question this whole section is about.** It uses **no bench harness**
(that is the point), keeps production `forecast_lead_times [1,12,20,40,60]` and all 5 upper-air
vars so the climatology allocates at full size, and shrinks only duration (1 train year, 1 val
year, 9 ICs). `run_validation_smoke.py` patches `create_metrics_aggregator_new` before
`runpy`-exec'ing `train.py`, so **train.py runs unmodified and no tracked file is touched**; it
snapshots the pre-validation peak, then `reset_peak_memory_stats()`, so the reported number is the
**validation window** and not the cumulative ~27 GB training peak. PASS = `PANGU_VAL_SMOKE_OK`,
emitted only from inside `MetricsAggregator.compute()` (once per validation, after the batch loop)
— a run that never validates cannot print it (verified against a synthetic skip log). OOM prints
`ERROR PANGU_VAL_SMOKE_OOM` with the per-rank pre-validation peaks. Either way **the measured
number is the deliverable.** Watchlist: it asks `-q debug` for `00:59:00` against a 1 h cap —
~60 s of margin, and validation I/O (~80 GB of h5 reads at `num_data_workers=1`) is unmeasured.

**The blind spot was structural — remember why, because it generalises:** every *training* smoke
keys on `PANGU_BENCH`, whose `_bench_finalize` ends in `sys.exit(0)` (`train.py:1305`) **upstream
of validation**. The harness exists to time training steps, and exiting early is right for that
job — but it means every gate built on it is blind to everything after the loop. The validation
smoke above exists precisely because no gate can test code it never reaches. **Ask that question
of every check you inherit here.**

**b) Numbers the docs got wrong (measured):**
- **Checkpoints are 18.9 GB, not ~3.5 GB.** Every `ckpt_epoch_N.tar` in
  `runs/pangu_sfno_full/SFNO/resume2/checkpoints/` is 18,913,987,526 bytes; that dir is already
  177 GB after ~9 epochs. Keep-10 ⇒ ~230–250 GB.
- **The model is 1.18B params**, not the ~79M CLAUDE.md #12 implies
  (`Number of trainable model parameters: 1182191104`, from 7258382's own log).

**c) `best_ckpt.tar` is best-of-final-segment, not global best.** `best_valid_loss` is a function
local (`PanguWeather/v2.0/train.py:883`, init `1e6`) and is **not** in the checkpoint, so the first
validation after every requeue always "improves". Across the ~10–25 requeues an 8-day run needs,
plus rolling keep-10 pruning, an early global best is lost. Decide whether to fix or accept.

**d) ~90% of validation I/O is dead weight.** Each val sample loads a full target for every step
1..60 (`data_loader_multifiles.py:1151-1153`) but only the 5 `forecast_lead_times` are consumed;
the all-steps consumer (GIF accumulation) is force-disabled for sigma-level runs
(`PanguWeather/v2.0/train.py:4283`).

> **Line numbers rot — this doc's did.** `train.py:939/971/1433/853/4251` were all correct when
> written and all wrong within a day (the `PANGU_BENCH` rename shifted them). Corrected above;
> **re-verify before trusting any of them**, and prefer the symbol name. Note also that **two
> different files are called `train.py`** here: `PanguWeather/v2.0/train.py` and
> `physicsnemo_sfno/examples/weather/unified_recipe/train.py` (§5). They are unrelated.

**e) The 157-literal cascade.** Nothing derives `nr_predicted_variables` from `UNPREDICTED` —
it is restated across the PBS files. After the cloud exclusion the correct value is **103**.
`polaris_sfno_{full,full_probe,smoke}.pbs` are **tracked** and target the OLD 162-ch stores;
they must keep working. **A stale 162-ch store on disk + a new launcher = a silent shape
mismatch.** The store's attrs should be the arbiter; verify a mismatch fails LOUDLY.

**f) Two science recommendations are pending with jesswan** (`data_for_training.md`), both in
`e3sm_h5_to_seqzarr.py`, both **baked into the store** so they cannot be changed in config later:
1. `UNPREDICTED += ["SST", "ICE"]` — AMIP prescribes the ocean by definition; Pangu and makani
   already do. Cascades to the 157/103 literals.
2. `"TSOI_10CM": 0.0 → 270.0` — 0.0 is absolute zero in a Kelvin field over 61% of the globe.
   Measured: with 0.0 fill only **1.5%** of that channel's variance is real soil temperature and
   98.5% is the coastline step; with 270.0 it is **99.6%**. Its stated justification (matching the
   npz) is **void** — PhysicsNeMo never reads the npz, it uses BatchNorm.
**Decide whether the conversion waits.** The rest of the archive is fine either way.

---

## 5. PhysicsNeMo's normalizer erases precipitation (R1) — know this before you design training

All refs here are **`physicsnemo_sfno/examples/weather/unified_recipe/train.py`** — a different
file from PanguWeather's `train.py`.

`:121-126` normalizes with `nn.BatchNorm2d(momentum=None, affine=False)` on **raw physical
units**, `eps=1e-5`. Amplitude is `σ/√(σ²+eps)`; `√eps ≈ 3.16e-3`. **`PRECT`'s σ is 7.7e-8 m/s**
→ amplitude **2.5e-5**. The loss (`batch_normalized_mse`, `:51`) is a **global** L2 over all
channels flattened, so a channel's gradient share scales as amplitude²: PRECT's is **~6e-10**.
The model converges normally and forecasts **climatological-mean precipitation** — zero skill, no
error raised, and the BatchNorm state is exported into the inference package (`:444-445`),
making it permanent.

**Not a data defect.** In mm/day, PRECT's σ is 6.72 → amplitude 1.0000. Fix in the **training
path** (precomputed per-channel stats instead of online BatchNorm), NOT the converter — the
archive is healthy. **This does not gate conversion.** The cloud exclusion already removed ~23 of
the other crushed channels; PRECT remains.

---

## 6. Implementation task: tee PhysicsNeMo's metrics to CSV

**Decided by the owner:** tee to CSV **in `train.py`**, with **its own minimal schema**.

**What exists.** MLflow already runs **offline to a local file store** — `initialize_mlflow(…,
mode="offline")` (`train.py:78-85`) and every Polaris script exports
`MLFLOW_ALLOW_FILE_STORE=true`, because `initialize_mlflow()` is called **unconditionally** and
raises `ImportError` without mlflow rather than degrading to a no-op. So this is not "replace
MLflow" — it is "also write a CSV".

**The whole logged surface is four metrics** (verified — this is all `LaunchLogger` is given):

| metric | site | cadence |
|---|---|---|
| `loss` | `log.log_minibatch({"loss": loss.detach()})` `:335` | per minibatch |
| `Learning Rate` | `log.log_epoch(...)` `:344` | per epoch |
| `GB/s` | `log.log_epoch(...)` `:347` | per epoch |
| `Validation error` | `log.log_epoch(...)` `:419` | per epoch |

**Design constraints:**
- **Own minimal schema.** Do NOT reuse the 19-column `S2S_BENCH` schema — PhysicsNeMo measures
  almost none of it and the columns would be empty. Suggested: `timestamp, epoch, step, loss, lr,
  gb_per_s, valid_error, n_gpus, git_sha, run_name`. CLAUDE.md #10 forbids letting instrumentation
  drift **within** a benchmark's schema; a *new, separate* CSV for a different model is fine —
  but once you fix these columns, they are frozen.
- **Rank 0 only.** `Validation error` is already inside `if dist.rank == 0` (`:350`). A CSV
  written by 4 ranks is a corrupted CSV.
- **Env-gated, like the rest of the repo** (`PHYSICSNEMO_BENCH_CSV=<path>`, unset ⇒ no-op).
  A vendored file that behaves identically when the knob is unset is far easier to defend.
- ⚠ **This is a vendor divergence.** `physicsnemo_sfno/` came in via `git subtree`
  (commit `94f9e4dd`, upstream `a8eedb65`). Editing `examples/weather/unified_recipe/train.py`
  means the next subtree pull conflicts. **Say so in the file header**, and keep the edit as
  small and as clearly-delimited as possible.
- **Ship its test.** A smoke that greps a real CSV row — not exit code (CLAUDE.md).

---

## 7. Method (this is what worked)

1. **Measure; do not infer.** A variable name is not evidence — and on this archive, *neither is
   the unit attribute*: `SST`'s `long_name` is literally "potential temperature",
   `RHREFHT`'s units say `1` but it is percent. See `polaris_e3sm_variable_reference.md` R11.
2. **Ask of every check: what would it do if the thing were broken?** `CONVERT_OK` verified
   1 channel of 1 sample — 0.01% — and was trusted for weeks. The green smoke that never
   executes the failing code is this repo's signature bug (§4a).
3. **Use cold + adversarial Fable 5 agents** (CLAUDE.md model policy: subagents = `claude-fable-5`).
   This session: 3 adversaries confirmed the arithmetic but **corrected two interpretations**, and
   **the cold agent — given no conclusions — found the single worst issue (§5), which nobody was
   looking for.** Give the cold one no conclusions at all.
4. **Additive only.** New config/script *beside* the working one, never replacing it
   (CLAUDE.md #7). Every change so far is additive except 3 tracked files, all listed in §1.
5. **Never claim a step passed without reading the output.** Key on the token, not `rc=0`.

## 8. Deliverable

A written plan covering, per pipeline: the convert command, the train command, the inference
command, what PASS looks like at each stage (a greppable token or a CSV row — *not* an exit code),
what it costs (hours/TB), and **what each gate would fail to catch**. Plus the §6 CSV tee,
implemented with its smoke.

State plainly what you did not verify. The most valuable thing in this document is the list of
things it admits are unproven.
