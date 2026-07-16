# The three E3SM pipelines on Polaris: convert → train → infer (the plan)

Written 2026-07-16, the deliverable of `polaris_pipelines_handoff_prompt.md` §8. Every
claim below was read from code this session (two Fable-5 agents swept PanguWeather and
makani; PhysicsNeMo was read directly) or is marked **UNMEASURED / NEVER RUN**. Nothing
in this document has been submitted.

**What was implemented alongside this plan (all additive, committed on `polaris-data-prep`):**

| artifact | status |
|---|---|
| PhysicsNeMo CSV tee (`bench_csv.py` + 5 delimited blocks in `train.py`) — handoff §6 | unit test green: `BENCH_CSV_OK (23 tests)`; end-to-end grep wired into the allyears smoke (`PHYSICSNEMO_CSV_OK`), **job not yet run** |
| `physicsnemo_sfno/polaris/polaris_verify_store.pbs` — verifies an EXISTING SeqZarr store at any path (the jesswan transfer) | attrs/chunk logic exercised against a real old-generation store (correctly refused it); **PBS never submitted** |
| `makani_sfno/polaris/polaris_sfno_alldata_full.pbs` — the missing production launcher for `e3sm_alldata_full.yaml` | mirrors the green locked launcher + the alldata smoke's gates; **NEVER RUN** |

---

## 0. The job order — RUN 2026-07-16, ALL FOUR GREEN

Every earlier green predated the contract it would now validate (handoff §2). The
re-establishing sequence below was executed 2026-07-16 with the owner's go-ahead:

1. **Pangu validation smoke** — ✅ **job 7259271: `PANGU_VAL_SMOKE_OK`**.
   `valid_loss=0.6989` (finite), val peak **25.048 GiB** / train peak 25.127 / device
   39.495 ⇒ **13.86 GiB headroom**; climatology confirmed on CPU on all 4 ranks. The
   §4a requeue-forever OOM scenario is **refuted** — the `utils/metrics.py` fix is now
   proven, not just written. Planning number: validation took 112.6 s at 9 ICs ⇒
   production's 129 ICs ≈ **~27 min of validation per epoch**.
2. **Pangu training smoke at 108** — ✅ **job 7259296: `ALLDATA_SMOKE_OK`**,
   105-in/101-out, peak 26.98 GB, `step_med` 0.643 s (n=12 — small-n, don't compare to
   the n=80 0.602 s).
3. **makani alldata smoke at 100/1/7** — ✅ **job 7259321: `ALLDATA_SMOKE_OK`** after the
   stale 154-ch pack was moved aside (now at
   `${MEMBER_ROOT}/data/e3sm_makani_alldata_smoke_154ch_stale`; delete when convenient).
   Fresh pack `CONVERT_ALLDATA_OK`, contract + channel names verified against the
   converter, real training (6.79 s) + checkpoint; the benign 107-vs-58 watchdog warning
   appeared exactly as designed.
4. **PhysicsNeMo allyears smoke at 103** — ✅ **job 7259303: `ALLYEARS_SMOKE_OK` +
   `PHYSICSNEMO_CSV_OK`** (7 metric rows: 5 minibatch with decreasing loss, 1 epoch,
   1 validation — the §6 tee is proven end-to-end).
5. **When the transferred zarr lands in jesswan's folder: verify it** (§4 below), before
   any conversion job — if it verifies and is the right generation, our own ~11 h /
   ~1.43 TB conversion is unnecessary. **Still pending the transfer.**
6. Then the conversions that are still needed, then training, per pipeline below.
   The operator-facing guide for the PhysicsNeMo path is `polaris_pipeline_runbook.md`.

---

## 1. PanguWeather (the focus)

### Convert
**None.** Reads `E3SM_ROOT/h5/plev_data` directly. One auxiliary input: `PANGU_AUX`
(~17 GB: Z→Z_2 stats + the CDF-5→NETCDF4 climatology), auto-prepared by the smoke if
missing and hard-gated on `climatology.nc` having exactly 365 time steps
(`ERROR CLIM_NOT_READY` otherwise).

### Train
Submit **from `PanguWeather/v2.0/`** (both scripts resolve `polaris_env.sh` via
`$PBS_O_WORKDIR` and die in ~1 s otherwise):

```bash
qsub HPC_scripts/polaris_train_e3sm_sfno_alldata_smoke.pbs        # debug, ≤50 min
qsub HPC_scripts/polaris_train_e3sm_sfno_alldata_full.pbs         # preemptable, 24 h, -r y
```

- Config `E3SM_SFNO_H5_POLARIS_ALLDATA.yaml`: 108 channels (105-in/101-out, model
  auto-sizes from the lists), train 2015–2044 (`train_year_end: 2045`, exclusive),
  val 2045–2048, leads `[1,12,20,40,60]`, 129 ICs, bf16 (the Trainer default — neither
  yaml sets `amp_dtype`; note the CLI help's "float16 if unset" is wrong, `train.py:265`
  is bf16).
- Launcher `torchrun --standalone --nproc_per_node=4`. Full run: stable
  `RUN_NUM=alldata_full01`, resume via `ckpt_latest.tar`, keep-10 pruning patched in.
- **PASS**: smoke = `ALLDATA_SMOKE_OK` + a new row in
  `${MEMBER_ROOT}/bench/pangu_sfno_polaris_alldata_smoke.csv` (`ERROR ALLDATA_SMOKE_NO_ROW`
  otherwise, rc=4). Full = **no token of its own**; progress is `Starting epoch N/100`
  advancing across requeues, completion is `DONE ---- rank 0`. rc≠0 on preemptable is
  normally preemption.
- **Cost**: 1.18 B params; `step_med` 0.602 s (measured, 108 ch, 4×A100, global batch 4)
  ⇒ 43,800 samples / 4 = 10,950 steps ≈ **1.8 h compute per epoch + validation**
  (validation cost UNMEASURED — that is what job 1 measures). 100 epochs ≈ **8+ days
  wall, 10–25 preemption requeues**. Checkpoints **18.9 GB each**; keep-10 + latest +
  best ≈ **~230 GB** (the full PBS header still says "~3.5 GB" — stale, do not budget
  from it).
- **What the training gates fail to catch**:
  - The smoke `sys.exit(0)`s inside `_bench_finalize` **before validation** — it can
    never see a validation OOM (§4a of the handoff). Only job 1 can.
  - `best_ckpt.tar` is best-of-current-segment: `best_valid_loss` is a function-local
    (`train.py:883`, init 1e6) and not in the checkpoint dict (`train.py:3565-3575`),
    so every requeue's first validation "improves". Confirmed at current lines. An early
    global best can be pruned. **Open decision: fix (persist it in the checkpoint) or
    accept.**
  - ~90 % of validation I/O is dead (all 60 lead targets loaded, 5 consumed) — a cost,
    not a correctness issue.
  - No gate watches for the requeue-forever mode: if validation dies post-train,
    the run trains ~2 h, dies, requeues, repeats, and **never checkpoints** (save is at
    `train.py:1001`, after `validate_one_epoch` at `:969`). If job 1 fails, do not
    submit the full run.

### Infer
**No launcher exists on any cluster in this repo, and two of the three scripts cannot
serve this model.** Verified:

- `inference.py` — `pangu_plasim` only (`inference.py:106,131-132`) *and* loads
  `cwd/results/<cfg>/<run>/training_checkpoints/ckpt.tar`, a layout train.py never
  writes. Unusable for E3SM-SFNO, full stop.
- `long_inference.py` — has an `sfno_plasim` path but **requires `--init_nc_filepaths`**
  (a `*_Combined_EAM_ELM.nc`-style IC file, not staged on Polaris). Blocked on staging.
- `ensemble_inference.py` — has an `sfno_plasim` path and, **when `--init_nc_filepaths`
  is omitted, takes ICs from the h5 validation dataset** (`ensemble_inference.py:185-193`)
  — the only path runnable from what is staged today. Checkpoint priority
  `best_ckpt.tar` > `ckpt_latest.tar` > newest epoch, with a clean FileNotFoundError.

**The command to design the first inference smoke around** (single GPU, `--debug` sets
world_size=1; datetime format uses a **space**, unlike long_inference's underscore):

```bash
cd PanguWeather/v2.0
python ensemble_inference.py \
    --yaml_config=${MEMBER_ROOT}/runs/pangu_sfno_alldata_full/E3SM_SFNO_H5_POLARIS_ALLDATA.full.rendered.yaml \
    --config=SFNO --run_num=alldata_full01 --debug \
    --init_datetimes="2045-01-01 00:00:00" \
    --ensemble_inference_hours=336 --num_ensemble_members=1 --epsilon_factor=0.0 \
    --output_dirs=${MEMBER_ROOT}/runs/pangu_sfno_alldata_full/inference \
    --save_basenames=pangu_alldata_336h
```

(mirrors `submit_scripts/derecho/derecho_ensemble_inference_jsw.sh` minus its
`--init_nc_filepaths`; the rendered yaml is the SAME one training used, so the model
auto-sizes identically and `exp_dir` already points at the trained run.)

- **PASS**: there is **no success token in any inference script** (`DONE ---- rank N`
  only). A PBS wrapper must key on the output NetCDF existing
  (`<output_dir>/<basename>_run.0000-0000_output.nc`) and its variables being finite.
  Writing that gated wrapper (`polaris_ensemble_inference_e3sm_sfno.pbs`, mirroring the
  Derecho script per CLAUDE.md #7) is the first inference work item — **deliberately not
  written this session**: the h5-IC ensemble path has never been exercised anywhere, and
  encoding guesses about it into a script would manufacture false confidence. Run the
  command above once, interactively, on a debug node first.
- **Watch**: precision asymmetry — training ran bf16, both inference scripts default
  their AMP dtype to **float16**. Decide `--enable_amp` on/off (off = fp32 rollout,
  safest first) before comparing anything to training-time validation numbers.
- The `long_validation: False` / bias-`.npy` caveat binds **train.py's config**, not the
  inference scripts — `_load_bias` (99 files, not staged) is called only from Trainer
  init. Keep it False in training configs; inference is unaffected.
- **Cost**: 1 GPU, minutes per 336 h rollout (UNMEASURED).

---

## 2. makani

### Convert (pack)
From `makani_sfno/`:

```bash
qsub polaris/polaris_pack_e3sm_alldata_full.pbs      # preemptable, 24 h, -r y, no GPUs
```

- Contract **100 state + 1 diag + 7 forcing = 108 kept channels** (107-in/101-out),
  counts and names **derived** in `convert_e3sm_to_makani_alldata.py` (asserted at import;
  the 154-literal era is over). Default split train 2015-2044 / valid 2045-2047 / test
  2048-2049. Per-year files written atomically (`.tmp` + `os.replace`), so `-r y` resume
  is per-year safe. Stats are a second pass over the **packed train bytes** (float64,
  σ floored at 1e-12) — they describe exactly what the trainer reads, fills included.
- **PASS = `CONVERT_ALLDATA_OK` AND `PACK_ALLDATA_COMPLETE`** (the latter after
  MISSING_YEARS / WRONG_CONTRACT / SHORT_YEAR / MISSING_STAT checks).
- **Cost: ~1.43 TB** (40.87 GB/yr × 35), pack ≤24 h budget + 1–2 h stats re-read.
  Check `myquota` first — this plus the PhysicsNeMo store plus Pangu checkpoints is
  ~3+ TB if everything is built under `mehta5/`.
- **What the gate fails to catch**: nothing verifies the packed **values** against the
  h5 the way `verify_seqzarr.py` does for PhysicsNeMo (`--validate` re-reads one sample
  per split). A makani-side analogue of the exhaustive verifier remains unbuilt
  (CHANGELOG: "makani's 367-line converter is unaudited"). The stats are also unverified
  against an independent computation.

### Train
```bash
qsub polaris/polaris_sfno_alldata_smoke.pbs          # debug; PASS = ALLDATA_SMOKE_OK
qsub polaris/polaris_sfno_alldata_full.pbs           # preemptable; NEW THIS SESSION, NEVER RUN
```

- The full launcher did not exist before this session (the handoff's "2 pbs" were
  pack-full + train-smoke). The new one mirrors the green locked `polaris_sfno_full.pbs`:
  stable `RUN_NUM=alldata_full01`, resume from `ckpt_mp0_v0.tar`, global `--batch_size 8`
  with a divisibility gate, `python -m torch.distributed.run` (never `torchrun`), plus
  the alldata smoke's derived `WRONG_CONTRACT` / `WRONG_CHANNEL_NAMES` /
  `CONFIG_CONVERTER_DRIFT` gates.
- **PASS**: progress = "Total training time" > 0 and the epoch counter advancing across
  requeues; done = epoch 100. Expect the benign watchdog warning
  `N_in_channels=107 ... expected 58` (it is a warning by design, not a gate).
- **Cost**: UNMEASURED at production scale — the only alldata green (7258407) was a
  **tiny** model (embed_dim 16) at the **stale 154 contract**, ~3 min total. Production
  dims are the group baseline (embed_dim 384 / 8 layers, 8192 samples/epoch × 100
  epochs); memory at 107-in on 40 GB A100s is unmeasured (the yaml's own header says
  so). First submit needs watching; if OOM, drop `-v BATCH=` first.
- **What the gates fail to catch**: same-width channel *value* corruption (names and
  widths are checked; bytes are not — see the converter gap above), and everything after
  training (below).

### Infer — **BLOCKED, decision required before training 100 epochs**
`src/sfno_inference/checkpoint_loader.py:74-82` hard-asserts the locked contract
(`N_in_channels == 58`, i.e. 52+6, plus 53-out and the 52/1/6 params), and re-asserts
58/53 on the built module (`:234-239`). Stock makani inference is separately hard-gated
off (`plasim_trainer.py:127-131` — the stock Inferencer has no slot for our forcing
channels). **A checkpoint trained on 100/1/7 cannot currently be evaluated by anything
in this repo.** The new launcher's header carries this warning.

Options, in preference order:
1. **Generalize `sfno_inference`** to read the contract from the run's `config.json`
   instead of asserting 52/1/6: `rollout_driver.py` and `nc_writer.py` are already
   width-generic (they read `n_state`/`n_out` from params); the loader asserts (and one
   `n_out=53` default at `_load_run_norm_stats:83`) are the only hard blocks. Mechanical
   and ours — but ship a test that the locked 58/53 path still passes its asserts
   (that contract guards the group's PlaSim-comparable evaluations).
2. Accept alldata-makani as train-only until jesswan needs its forecasts.

The eval entry point (once unblocked) is `scripts/eval_inference.py`
(`--run-dir --ckpt --test-holdout --out-root --mode nwp|climate`), which writes
physical-units NetCDF per (file, IC); its wrappers are Stampede3 SLURM — a Polaris PBS
mirror is a to-write item. Note the two `makani_sfno/.claude/skills` eval skills target
**Stampede3**, not Polaris.

---

## 3. PhysicsNeMo

### Convert
The converter is the one thing already proven at the current contract
(`SEQZARR_VERIFIED`, 6×108 bitwise). Full build, from `physicsnemo_sfno/`:

```bash
qsub polaris/polaris_sfno_allyears_smoke.pbs                       # PASS = ALLYEARS_SMOKE_OK (+ PHYSICSNEMO_CSV_OK)
qsub -v CONFIRM_ALLYEARS=1 polaris/polaris_zarr_e3sm_allyears.pbs  # PASS = CONVERT_OK ×2 + ZARR_ALLYEARS_COMPLETE
```

~1.43 TB (1.31 train + 0.12 val), ≤ ~11.3 h, per-store sentinel-skip resume, split
2015-2046 / 2047-2049 gated for contiguity/overlap/all-years.

**⚠ If the store being transferred into jesswan's folder is this store, do not run the
conversion — verify the transfer instead (§4) and point the trainer at it.** That saves
~11 h and 1.43 TB of quota under `mehta5/`.

### Train
```bash
qsub polaris/polaris_sfno_allyears.pbs               # preemptable, 24 h, -r y
# or, against the verified transferred store:
qsub -v SEQZARR_ALLYEARS_DATA=/eagle/projects/lighthouse-uchicago/members/jesswan/<dir> \
    polaris/polaris_sfno_allyears.pbs
```

(the `<dir>` must hold stores named exactly `e3sm_train.zarr` and `e3sm_val.zarr`;
`SEQZARR_ALLYEARS_DATA` is honoured because `polaris_env.sh` never touches that name —
unlike `SEQZARR_DATA`, whose `_pick` clobbers `qsub -v` overrides by design.)

- Preflight gates on the stores' own attrs (generation, sentinel, contiguity, exact
  counts, overlap, channel-list agreement), then **derives** 103+5. The launcher passes
  no channel literals.
- **PASS**: progress = `Epoch N Metrics ... loss = ...` advancing across requeues — and
  now also `metrics.csv` growing in the stable run dir (the §6 tee, on by default in
  this launcher; disable with `-v PHYSICSNEMO_BENCH_CSV=`).
- **Cost**: UNMEASURED at the production SFNO (embed_dim 384/8 layers; the 157-ch green
  7252933 was a smoke). Config: 500 epochs × (46,720 samples / global batch 8) ≈ 5,840
  steps/epoch. **Measure one epoch before believing any wall-clock estimate** — the CSV
  tee's `gb_per_s` + epoch rows now give that for free.
- **What the gates fail to catch**:
  - **Resume is UNVERIFIED** (the launcher's own header says so): `load_checkpoint` runs
    at startup but only saves have ever been observed. Watch the first requeue; if the
    epoch counter restarts at 0, stop and wire it before burning allocation.
  - Checkpoint cadence is `epoch % 5 == 0 or epoch == 1` (`train.py:429`) — a preemption
    can lose up to 5 epochs (vs Pangu's every-epoch saves).
  - **R1 (handoff §5): the BatchNorm normalizer erases PRECT** (amplitude 2.5e-5,
    gradient share ~6e-10). Training will converge and look green while learning
    climatological rain. This is a *training-path* defect, pending a decision
    (precomputed per-channel stats vs online BatchNorm); it does not gate conversion,
    but every epoch trained before it is decided is an epoch of zero-skill precipitation.

### Infer
There is **no inference script** in the unified recipe. What exists is the **inference
model package** written at each checkpoint save (`model.mdlus`, `metadata.json`, the
BatchNorm running stats as `.npy`, lat/lon). Two facts to carry into any consumer:

1. **The package's `*_stds.npy` files actually contain VARIANCES** —
   `model_packages.py:85-100` saves `running_var` under the "stds" name. A consumer
   normalizing with them as-is mis-scales every channel by √var. Fix the consumer or the
   name, but flag it now, before a consumer exists to inherit the bug.
2. The R1 PRECT crush is baked into those stats — fixing training later invalidates
   previously exported packages.

A forecast driver (load package → unroll like `train.py::unroll` → write NetCDF) is a
to-write item; nothing in-repo consumes the package today. State plainly: **PhysicsNeMo
has no runnable inference path yet**, only its ingredients.

---

## 4. Verifying the transferred zarr store (the jesswan folder)

The store is not there yet (checked 2026-07-16: no zarr under
`/eagle/projects/lighthouse-uchicago/members/jesswan/`; note `../jesswan` relative to
this repo does not exist — the member folder is the plausible target). The moment it
lands:

```bash
cd physicsnemo_sfno
qsub -v STORE=/eagle/projects/lighthouse-uchicago/members/jesswan/<dir-or-store.zarr> \
    polaris/polaris_verify_store.pbs
```

(`STORE` may be one `.zarr` or a directory of them — both `e3sm_train.zarr` and
`e3sm_val.zarr` get verified in one job.)

**PASS = `SEQZARR_VERIFIED` for every store + `STORE_VERIFY_OK (n stores)`.** What the
job proves, in three layers:

1. **Attrs / generation**: `conversion_complete` sentinel, `excluded_vars` ==
   the converter's current `[CLDICE, CLDLIQ, CLOUD]`, unpredicted set == the declared 5.
   A store built under *different science decisions* (e.g. the two §4f pendings: SST/ICE
   → unpredicted, TSOI_10CM fill 270) fails **here, loudly** — which is correct: if
   jesswan's store embodies decisions our converter hasn't adopted, the converter's
   declared lists must be updated *first*, then verification re-run. The converter is
   the single source of truth by design; a store must never smuggle a decision in.
2. **Exact chunk-key completeness** per array — every expected chunk file present, no
   `.partial` aliens. This is the transfer-truncation check, and it is deliberately NOT
   `nchunks_initialized` (which counts by prefix and blesses partials — measured
   2026-07-15). Catches every *dropped* byte of an interrupted rsync/Globus.
3. **Bitwise round-trip vs the h5 archive**, sampled at an auto-stride (~480 samples
   nudged off divisors of 1460 so it never locks onto the same calendar dates; full
   train store → stride 97). Values, NaN-fill placement, channel map, year-reconstructed
   time axis.

**What a PASS does NOT prove**: the bytes of unsampled samples (~99 % of a full store).
A transfer that *corrupted* (rather than dropped) a chunk in the unsampled set is
invisible at stride 97. For full fidelity either:
`qsub -q preemptable -l walltime=24:00:00 -v STORE=...,EXHAUSTIVE=1 polaris/polaris_verify_store.pbs`
(reads the whole store + archive, hours), or checksum the transfer itself (Globus
"verify checksum" on, or `rsync -c`). **Recommendation: have the transfer itself run
with checksums, then the stride job is sufficient.**

Tested this session: the attrs+chunk layer, run against the old 162-generation store on
disk, correctly reported `STORE_INCOMPLETE` + `STORE_WRONG_GENERATION` (and confirmed its
chunk keys). The full PBS has never been submitted (needs a compute node).

---

## 5. Cross-cutting: costs, decisions, and what is not verified

### Storage ledger (before submitting anything big, run `myquota`)
| item | size |
|---|---|
| makani full pack | ~1.43 TB |
| PhysicsNeMo allyears stores | ~1.43 TB — **skippable if the transferred store verifies** |
| Pangu checkpoints (keep-10 + latest + best @ 18.9 GB) | ~230 GB |
| makani / PhysicsNeMo checkpoints | smaller models; unmeasured |

### Open decisions (nothing below is silently designed around)
1. **§4f (jesswan)**: SST/ICE → `UNPREDICTED`, and `TSOI_10CM` fill 0.0 → 270.0. Both
   are baked into any store at conversion time. **The transferred store forces this**:
   whatever it embodies either matches our converter or the mismatch fails verification
   and must be reconciled first.
2. **§4c (us)**: persist `best_valid_loss` in the Pangu checkpoint, or accept
   best-of-segment. ~5-line change + a test if fixed; decide before the 8-day run.
3. **makani inference contract**: generalize `sfno_inference` off the 58-assert or
   accept train-only. Decide **before** the 100-epoch run.
4. **R1 PRECT normalizer** (jesswan + us): precomputed stats vs BatchNorm in
   PhysicsNeMo's training path. Decide before long training; does not gate conversion.
5. **`_alldata` naming**: the Pangu files still say `_alldata` while meaning
   "all-years, 108 ch". Rename to `_allyears` only with a full grep of every reference
   (PBS, docs, CHANGELOG) — deferred, not forgotten.

### The honest list (updated after the 2026-07-16 smoke sequence)

Proven today (job ids in §0): the Pangu validation memory fix, Pangu 108-ch training,
the makani 100/1/7 pack + tiny-model training, PhysicsNeMo 103-ch tiny-model training,
and the §6 CSV tee end-to-end.

Still unproven:
- **PhysicsNeMo resume across preemption**: never observed; watch the first requeue.
- **`polaris_verify_store.pbs` as a job**: logic tested on a login node against a real
  store; the PBS wrapper itself never submitted (needs the transferred store to exist).
- **The new makani full launcher**: never run; production-model memory at 107-in
  unmeasured (the smokes train a tiny model).
- **The PhysicsNeMo production SFNO at 103 ch**: has never trained a step (the smoke's
  model is tiny).
- **Every inference command in this document**: none has ever been executed on Polaris.
  The Pangu ensemble h5-IC path has never been executed anywhere.
- **Production wall-clock for makani and PhysicsNeMo**: no production-scale s/step
  exists; the first epochs are the measurement (PhysicsNeMo's `metrics.csv` now records
  it automatically).

### Corrections to the handoff, found while verifying it (update it or trust this doc)
- `inference.py` is not merely "one of three Pangu inference paths" — it **cannot run
  this model** (nettype gate + a checkpoint layout train.py never writes).
- The bias-`.npy` / `long_validation` caveat binds **train.py**, not the inference
  scripts (their only reference is dead code inside a string literal).
- makani's `checkpoint_loader` assert is at line **74** (75 is the message), and the
  handoff omits that **`e3sm_alldata_full.yaml` had no launcher** (now written).
- Pangu's full/smoke PBS headers still carry pre-exclusion 162-ch numbers and "~3.5 GB"
  checkpoints (18.9 GB measured) — stale comments in tracked files, listed here rather
  than edited to keep this session's diff reviewable.
- train.py's `--amp_dtype` CLI help says "float16 if unset"; the code default is
  bfloat16 (`train.py:265`).
