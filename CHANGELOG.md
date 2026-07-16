# CHANGELOG — pedramh-profiling living document

This is the **living document**: the shared memory across sessions. It records
what's done, what's in progress, what's blocked, measured results, and — most
importantly — **failed approaches so they aren't re-attempted**. Update it before
you stop working. Newest entries at the top of each section.

See **CLAUDE.md** for how to work here and **DESIGN.md** for what/why.

Format for entries: `YYYY-MM-DD — <what happened> — <result/measurement> — <what it means / next>`.

---

## Status at a glance

| Track | State |
|---|---|
| Repo published (s2s / s2s-lightning / si) | ✅ done |
| SNFO → SI rename (repo-wide) | ✅ done |
| Polaris (PBS) bring-up | ✅ **all 4 runnable models GREEN on 4×A100**, and Pangu is now proven **reproducible by a second user** (7253591, loss identical to the installer's run); **SI too** (7253603). Their deps were private to rmehta1987 until today's shared top-ups (PanguWeather-SFNO, SI, Makani-SFNO, PhysicsNeMo) + probe + all 3 data converters proven on real data. S2S/port scripts delivered but blocked on an ERA5 Globus stage. See `polaris_pbs_notes.md`. |
| **Profiling (PanguWeather SFNO on A100)** | ✅ **DONE — see `polaris_bench_report.md`.** Harness ported (PanguWeather had **zero** instrumentation), loader sweep + nsys captured. **VERDICT: GPU-bound** (loader idle **0.7%**) and **elementwise-bound** (61% of GPU time pointwise vs 15% GEMM) ⇒ `torch.compile` (§5 rung 1) is the right first lever, now on evidence. Model is **1.18 B params**, not ~79M. SI/makani/physicsnemo **not yet profiled**. |
| §4.0 prerequisites — **`s2s/v2.0`** | 🟡 **seed knob DONE + GPU-verified** (`--seed`/`$S2S_SEED`/YAML + `--deterministic`, `s2s/v2.0/utils/seeding.py`; 10 tests `SEEDING_OK` on CPU **and on an A100**, job 7253738 rc=0); tiny config + VAE noise-fix still **block baseline capture** |
| §4.0 prerequisites — **`PanguWeather`** (the focus; a separate fork, nothing propagates) | ✅ **ALL THREE MET.** seed knob ✅ **already existed — do NOT port `seeding.py` here** (`--global_seed`→`seed_torch`, seeds numpy+torch+CUDA, forces `cudnn.deterministic`; stronger than s2s's legacy path). VAE noise hook ✅ **built** (`utils/vae_noise.py`, 16 tests `VAE_NOISE_OK`) but **inert on `sfno_plasim`** (no VAE). `tiny_baseline.yaml` ✅ **written AND run** — job 7255583: **7,166,656 params** (165× smaller than the real 1.18 B), 0.023 s/step, **1.00 GB**. ⇒ **baseline capture is no longer blocked on building anything** |
| **E3SM data prep (PhysicsNeMo zarr)** | 🟡 **7 defects found, 5 fixed, 4 open**; verified `SEQZARR_VERIFIED` on a 24-year random fixture (job 7257786). **The full ~1 TB conversion is NOT cleared to run** — 4 open defects + 5 decisions. `polaris_data_prep_handoff_prompt.md`. makani's converter **unaudited**; Pangu's stats prep audited (clean, metadata-only). |
| Correctness baselines captured (DESIGN.md §4) | ⬜ not started — **blocks all optimization** |
| Test harness (tier-1 equivalence/unit + `--fast`) | 🟡 3 test files now exist + self-run (`SEEDING_OK`, `BENCH_INSTR_OK`, `VAE_NOISE_OK`); no `conftest.py`/`--fast` yet |
| Optimization ladder (DESIGN.md §5) | ⬜ not started — **deliberately**: profiling is unblocked, optimizing is not |

### Smoke status matrix (probe → 1-GPU → 4-GPU)

| Model | Midway | Polaris |
|---|---|---|
| Toolchain probe | — | ✅ `PROBE_OK` (job **7253681**, the fixed probe run as a second user: `sys.path` free of `~/.local`, and it imports the REAL `modules.train_module` rather than an empty namespace package — the hollow check that hid `cf_xarray`. 4×A100-40GB; makani/physicsnemo need the §6 venv, non-blocking) |
| S2S (`torchrun`) | ✅ runs (Midway scripts GREEN) | ⛔ blocked on ERA5 stage (scripts ready) |
| S2S-Lightning | ⚠️ standalone smoke config-path fixed 2026-07-13 — **needs a Midway run to reconfirm** | ⛔ blocked on ERA5 stage (scripts ready) |
| SI | ✅ runs (Midway scripts GREEN) | ✅ **4-GPU GREEN** (7252700: step_med 0.400 s, peak 30.98 GB) **and reproducible by a SECOND USER** — job **7253603** (`PYTHONNOUSERSITE=1`): step_med 0.399, peak 30.69 GB, rc=0 |
| PanguWeather SFNO | — | ✅ **4-GPU GREEN** (7252271) **and reproducible by a SECOND USER** — job **7253591** (`PYTHONNOUSERSITE=1`) rc=0 with loss **0.3411, identical** to the as-installer run |
| Makani SFNO | — | ✅ **4-GPU GREEN** (job **7253465**, current script: train loss 2.61 / val 2.38 + ckpt; first green 7252769 pre-rework; pack `CONVERT_OK` 7252728) — runs from the isolated SFNO venv |
| PhysicsNeMo SFNO | — | ✅ **4-GPU GREEN** (job 7252933, rc=0: loss 0.889, val err 0.541, ckpt saved; 1-GPU 7252816 also green; zarr `CONVERT_OK`) |

## Next actions (pick from the top)

0. **FOCUS (2026-07-15): the work is on `PanguWeather/`, not `s2s/v2.0`.** They are
   95%-identical forks (DESIGN §2c) — but **copies, not shared imports**, so nothing
   propagates between them. Both consequences the earlier handoff listed here are now
   **resolved, and one of them was wrong**:
   - PanguWeather's missing NVTX/`S2S_BENCH` instrumentation: ✅ **ported** (2026-07-15,
     range names identical per CLAUDE.md #10), and the legacy path is proven unchanged
     (job 7255505 reproduces the green 0.3411 bit-identically).
   - "the `--seed` knob lives in `s2s/v2.0/utils/seeding.py` only — port it deliberately":
     ❌ **do NOT.** PanguWeather already has `--global_seed` → `seed_torch()`, which is
     *more* complete than s2s's legacy path. See the Known-issues entry.


1. **Capture the §4.1 baseline** on TINY — **nothing blocks it any more.** All three §4.0
   prerequisites are met on PanguWeather: the seed knob already existed, the VAE hook is
   built, and `tiny_baseline.yaml` is written *and run* (job 7255583: **7,166,656 params**,
   165× smaller than the real 1.18 B; 0.023 s/step; **1.00 GB**), so a K=20 baseline is
   ~0.5 s of compute.
   Procedure: world size 1, fixed seed (`--global_seed`), K=20 steps, per-step loss
   trajectory + output summary stats → `baselines/pangu_sfno/` as JSON/CSV (§4.2 — text
   only, never tensors).
2. **Then, and only then, rung 1 of the §5 ladder: `torch.compile`.** The profile now says
   this is the right lever on evidence — **61% of GPU time is elementwise over ~1506
   launches/step vs 15% GEMM** (`polaris_bench_report.md` §4.2), i.e. fusion-starved.
   `TORCH_COMPILE_MODE` is now genuinely wired and deliberately unset — it was **not**
   plumbed in PanguWeather despite an earlier claim in this file (see the correction entry
   below). Expect longer warmup (raise `S2S_BENCH_WARMUP` to 40); profile eager, bench
   compiled.
3. **Profile the other three models** (`polaris_bench_report.md` covers only PanguWeather
   SFNO). SI is cheapest — it already has `SI_BENCH_*`/`SI_NVTX` and a green Polaris bench
   (7252700/7253603). makani/physicsnemo have no comparable harness.
4. **Fix the loader's missing `worker_init_fn`** — it would make `num_data_workers` an
   output-neutral knob and unlock a measured **+9% wall throughput with 10× less jitter**
   (`1 → 8`). Today the worker count changes the noise realization, so the win cannot pass
   the §4 bitwise gate. Ship it with a test pinning sample→noise independence from worker count.
5. **Stand up the test harness proper** — three self-running test files now exist
   (`SEEDING_OK`, `BENCH_INSTR_OK`, `VAE_NOISE_OK`) but there is no `conftest.py` or
   `--fast`. Add CRPS/KL numerical checks and the normalize↔inverse round-trip.
6. **Unblock `s2s`/the port** — still the ERA5 Globus stage.

## In progress

- **Data-prep PR open for review — https://github.com/rcc-uchicago/pedramh-profiling/pull/11**
  (branch `polaris-data-prep`). ⚠️ **Stacked on `polaris-profiling` (#10), itself stacked on
  `polaris-pbs-bringup`** — merge in that order. Carries the 7 converter defects (5 fixed, 4
  open) and **5 decisions** needing jesswan/us: `polaris_data_prep_decisions.md`.
  **The analysis is 1/3 done** — makani's 367-line converter is unaudited (§8).
- **Profiling PR open for review — https://github.com/rcc-uchicago/pedramh-profiling/pull/10**
  (branch `polaris-profiling`). ⚠️ **Stacked on `polaris-pbs-bringup`**, which is still
  unmerged, so PR #10's diff includes those commits until it lands. **Merge the bring-up PR
  first.** A solo session cannot self-approve — maintainer review/merge needed.
- **Polaris bring-up PR open for review** — branch `polaris-pbs-bringup` pushed; open at
  https://github.com/rcc-uchicago/pedramh-profiling/compare/main...polaris-pbs-bringup
  (a solo session cannot self-approve — maintainer review/merge needed).
- **Layout change: the 3 SFNO codebases are now `git subtree`s of this repo** (not
  separate checkouts as the handoff assumed, not submodules). Imported **unsquashed for
  full provenance**: upstream commits are real ancestors of HEAD (jesswan-uc 8,
  feynmanliu214 38, ktangsali 203). Cost: 313 → 4,769 files, .git 3.9 MB → 306 MB.
  Bidirectional merging (`subtree pull` from them, `subtree split` + PR to them) and the
  rule-#8 exception for imported third-party junk are documented in
  `polaris_pbs_notes.md` §6b. Note: pushing this repo now needs
  `git -c pack.threads=1 push` — the ALCF login node's process cap kills multi-threaded
  pack (`unable to create thread` / `git-pack-objects died`); the same cap forced the
  physicsnemo `subtree add` onto a compute node.
- **Deferred, ready:** **ERA5 Globus stage** → unblocks the S2S + S2S-Lightning smokes
  (scripts already preflight `ERA5_NOT_STAGED`).
- **makani / physicsnemo — torch_harmonics conflict RESOLVED via an isolated venv.**
  makani 0.2.0 needs the *public* `precompute_latitudes`, absent from every torch-2.8-safe
  release (0.7.4/0.8.0); 0.9.1 ships wheels only (no sdist) and its `attention/_C.so`
  ABI-breaks torch 2.8. Resolution (per user): `polaris_setup_sfno_venv.sh` builds an
  isolated `--system-site-packages` venv with **torch_harmonics 0.9.x from GitHub source**,
  so the base conda keeps 0.7.4 and the GREEN Pangu/SI smokes need no re-validation.
  **Trap:** a `--system-site-packages` venv re-enables the USER site, which `site.py` puts
  *before* the venv — the base's `--user` 0.7.4 shadowed the venv and makani still failed;
  fixed with `PYTHONNOUSERSITE=1` in the venv + both SFNO PBS scripts.
  Two more launch traps (both encoded in the scripts): `torchrun` resolves to the BASE
  conda launcher (whose shebang pins the base python) because the venv inherits torch and
  has no torchrun — use `python -m torch.distributed.run`; and makani's `--batch_size` is
  GLOBAL, so the **rank count must divide it** (`global_batch_size % data_parallel_size == 0`
  — `--batch_size=1` on 4 ranks fails). Plus an **upstream makani bug** (pin
  `c970430`): `self.logger` is assigned only when `log_to_screen` is truthy (rank-0 only)
  yet `deterministic_trainer.py` calls it unconditionally → every non-zero rank died;
  patched in our `plasim_trainer.py` wrapper, not in makani.
  **RESULT: Makani pack GREEN (`CONVERT_OK`, 7252728) and Makani SFNO 4-GPU smoke GREEN
  (7252769: train loss 2.19, val 2.05, checkpoint written, rc=0), **re-confirmed on the
  post-rework script by 7253465** (train 2.61 / val 2.38, 7.10 s of real training).**
  **PhysicsNeMo is ALSO GREEN** (1-GPU, job 7252816, rc=0: loss 1.082, val err 0.776,
  checkpoint saved; zarr store `CONVERT_OK` with max|zarr-h5|=0 + all-finite). Its four
  traps: hydra's PATH-form defaults make `model=tiny_sfno` impossible (added
  `conf/config_e3sm_sfno.yaml` + `--config-name`); mlflow is NOT optional and mlflow 3.x
  refuses its own file store (`MLFLOW_ALLOW_FILE_STORE=true`); `datapipe.parallel=false`
  is a broken fallback (DALI rejects `prefetch_queue_depth`); `validation.num_steps` must
  be >=2 (matplotlib squeezes the axes array) and `dataset.dataset_filename` must be
  repointed. **PhysicsNeMo 4-GPU is GREEN too** (job 7252933, rc=0: 4 ranks, loss 0.889,
  val err 0.541) — so all four runnable models are green on 4 GPUs.

## Decisions / changes log

- **2026-07-16** — **🔴 The E3SM archive replays ONE year of ocean forcing 35 times, and two of
  three pipelines mis-normalize a channel. Measured, adversarially verified, and written up in
  `polaris_e3sm_variable_reference.md` (per-variable reference + risk register R1–R12) and
  `data_for_training.md` (which risks actually affect *training*).** Method that produced these:
  4 agents — 3 adversarial (told to refute) + **1 cold, given no conclusions at all**. The cold
  one found the single worst issue; nobody was looking for it. Two of my own interpretations were
  corrected by the adversaries. Do this again.
  - **`SST`/`ICE`/`sol_in` are BITWISE identical across all 35 years** at the same index-in-year.
    1,224 md5 comparisons (12 indices × 34 years × 3 vars) + 480 random: **0 mismatches**; distinct
    inodes (not hardlinks); valid cells compared by value. Control: atmospheric fields **never**
    matched (1,632 comparisons; `TREFHT` differs by up to 30.6 K). Global SST mean is
    `14.574015 °C` in 2015, 2020, 2030, 2040 **and** 2049. Cause: `boundary_data/*_masked.nc` hold
    exactly **1460 steps = one year**; `netcdf-to-h5_e3sm.py` re-slices from `chunk_id=0` every
    year. The frozen 2015 in-file timestamp is the SAME bug, not a second one.
    **Intent is unresolved — only jesswan can say.** `CTL_SST0051` reads as a deliberate fixed-SST
    control; `SSP245AMIP` reads as a transient scenario that should warm. Both readings fit the
    name. Within-year seasonality is intact and strong (σ: SST 11.5, `sol_in` 403) — only the
    *interannual* axis is dead. **Not a training blocker** (it is a valid prescribed boundary);
    it contaminates metrics if a model *forecasts* those fields, which PhysicsNeMo does.
  - **PhysicsNeMo's normalizer silently erases precipitation.** `BatchNorm2d(momentum=None,
    affine=False)` on **raw physical units**, `eps=1e-5` → amplitude `σ/√(σ²+eps)`. `PRECT`'s σ is
    **7.7e-8 m/s** (m/s!), ~40,000× below `√eps` → amplitude **2.5e-5**. The loss is a *global* L2
    over all channels, so gradient share scales as amplitude²: PRECT's is **~6e-10**. The model
    converges and forecasts **climatological-mean rain**. Zero skill, no error, and the BatchNorm
    state is exported into the inference package. **NOT a data defect** — in mm/day σ is 6.72 →
    amplitude 1.0000. Fix belongs in the training path; **does not gate conversion**.
  - **PanguWeather's `TSOI_10CM` is normalized with stats that don't match its fill.** Config fills
    **270**; `compute_normalization_e3sm.py` never sets that key so its stats encode a **0** fill
    (npz 105.229/133.802 vs predicted 0-fill 105.266/133.857; 270-fill would be 271.13/16.43).
    A *predicted* channel ends up ~**26×** under-weighted. **Inherited from jesswan's own
    `_DERECHO_jsw.yaml:66` / `_STAMPEDE_jsw.yaml:66` — live in the group's existing runs**, not
    introduced here. The 270 fill itself is *good* (0.02σ from the valid mean).
  - **The `SST` 270 fill is inherited from a Kelvin ancestor.** `compute_normalization{,_plasim}.py`
    both carry `mask_fill['sst'] = 270.` / `['ts'] = 270.` — ERA5/PlaSim names for **Kelvin** fields.
    The E3SM copy renamed them mechanically; `SST` is **degC** (metadata says so). Fingerprint that
    it was mechanical: the same edit produced `mask_fill['TREFHT'] = 270.`, and TREFHT has **zero
    NaN** — dead code. 270-filling `SST_masked.nc` reproduces the shipped npz **exactly** (1e-7).
  - **Units metadata lies on 4 fields.** `SST`'s `long_name` is literally *"potential temperature"*;
    `RHREFHT` says units `1` but is percent; `PCT_*` say `unitless` but run 0–100. On this archive a
    variable name is not evidence **and neither is the attribute**. Measure.
  - Also corrected: the store is **2.15 TB**, not the "~1 TB" the older docs repeat; checkpoints are
    **18.9 GB**, not ~3.5 GB (`runs/pangu_sfno_full/.../checkpoints/` was already 177 GB at ~9
    epochs); the E3SM SFNO model is **1.18 B params**, not ~79 M.

- **2026-07-16** — **Cloud variables excluded from ALL three models (science owner).** All three
  pipelines now agree on the same **108 of 162** channels. PanguWeather already excluded them;
  makani's ALLDATA converter and PhysicsNeMo's `e3sm_h5_to_seqzarr.py` (`EXCLUDED_VARS`) now do too.
  PhysicsNeMo: **162 → 108, predicted 157 → 103**, store **2.15 TB → ~1.43 TB**. Verified by running
  the converter on the real archive: `excluded 54 channels`, `max|zarr-h5| = 0.0`, `CONVERT_OK`, then
  `SEQZARR_VERIFIED (EXHAUSTIVE: 6 × 108 = 648 channel-samples, bitwise)`, and the store's own attrs
  inspected independently (103+5+54 = 162, zero clouds survive, `TSOI_10CM`/`SOILWATER_10CM` intact).
  - **The duplication bug this exposed is the real lesson.** Nothing derived the channel counts:
    `157`/`5` was restated across ~13 sites, and makani's converter had `N_STATE, N_TARGET,
    N_FORCING = 154, 155, 7` as literals the asserts did **not** check — it would have written a
    correct 100-channel pack advertising 154 in its metadata. Both are now **derived**. The
    trainer reads counts from the store's attrs behind a `STORE_WRONG_GENERATION` gate; a stale
    162-channel store can no longer be silently adopted.
  - The biggest miss was **not** a `157` literal: `verify_seqzarr.py:107-135` structurally assumed
    the store partitions all 162 h5 keys, so it would have hard-failed **every** new store —
    including the tracked `polaris_verify_data_prep.pbs`, which rebuilds its fixture each run.

- **2026-07-16** — **Scope widened (owner): training is now Phase 1, not out of bounds.** DESIGN.md
  rewritten (375 → 314 lines) around *implement-first, then-profile*; three overlapping model tables
  merged into one six-codebase table; CLAUDE.md's "Why we're here" updated to match (it said "NOT
  retraining" and is auto-loaded, so it outranked DESIGN). **The division of labor is the line that
  matters and did not change: bring-up and training are ours; the science — variable sets, fill
  values, channel roles, physics — is jesswan's.** The rewrite also found ~10 stale claims in the
  old DESIGN (it said the §4 prerequisites "NONE exist yet"; all three exist).

- **2026-07-16** — **PanguWeather's bench knobs renamed `S2S_*` → `PANGU_*`.** PanguWeather is its
  own project — a fork of `s2s/v2.0` by **copy, not import** — and nothing outside it ever read
  these (verified: the only cross-project bench consumers are `s2s-lightning/common/bench_callback.py`,
  which genuinely imports `s2s/v2.0`, and `si/parse_nsys.py`). The `S2S_` prefix was decoration the
  copy carried along, and it kept inviting the conflation (it led me to cite CLAUDE.md #5 — which
  governs `s2s/v2.0/` — about a PanguWeather-only file).
  - **`train.py` now errors `LEGACY_BENCH_ENV` if an `S2S_BENCH*`/`S2S_NVTX` knob is set.** Required,
    not politeness: `BENCH = os.environ.get(...) == "1"` means *unset ⇒ silently no benchmarking*, so
    a stale script or doc (e.g. `polaris_bench_report.md`'s reproduction command) would have produced
    a run that measured nothing and exited 0. Verified the guard fires on `S2S_BENCH=1`/`S2S_NVTX=1`
    and passes on `PANGU_BENCH=1`/nothing-set.
  - **NVTX range names and CSV columns are UNCHANGED** (CLAUDE.md #10) — only the env knobs moved, so
    every prior bench row stays comparable. `s2s/` and `s2s-lightning/` keep `S2S_*`; historical
    CHANGELOG entries below are left as written (they record what was true then — the guard is what
    catches anyone following them).

- **2026-07-15** — **🔴 E3SM data prep: the PhysicsNeMo converter had 7 defects, and the smoke
  could not see the worst 3. Full analysis + the 5 open decisions:
  `polaris_data_prep_handoff_prompt.md`.** Prompted by "how do we confirm the conversion is
  correct before the ~1 TB run" — the answer was that we couldn't.
  - **The smoke was green because it was the one configuration where the bugs were invisible.**
    It converts 64+16 samples from **2015 only**: the archive's frozen in-file year is
    *correct* for 2015; `--max-samples` defaulted to **64**, which is what the smoke passes
    explicitly; and the zero-fill-on-interrupt trap needs an interrupt. Guardrail #4 exactly.
  - **Fixed (5):** `--max-samples` default 64→None (the "full" run wrote **64 of 51,100** and
    printed CONVERT_OK); the time axis now takes the **year from the filename** (measured: the
    archive stamps 2015 into EVERY file — `2049_1459.h5` → `2015-12-31 18:00:00`); `longitude`
    0..359 → **0.5..359.5** cell centres (verified vs `boundary_data/TOPO.nc`; `train.py:447`
    reads it into the inference model package, so it georegistered every product half a degree
    west); a `conversion_complete` sentinel (zarr pre-allocates with `fill_value: 0.0`, so a
    preempted run leaves a **right-shaped store of silent zero slabs** and the trainer gate only
    checked `shape[0] >= 1000`); and the **four `means_/stds_` arrays DELETED** — dead
    (nothing reads them: the datapipe asks only for `time/predicted/unpredicted`, `train.py`
    normalizes with `BatchNorm2d(momentum=None, affine=False)`, `model_packages.py:85-86` saves
    *that* batchnorm's stats) **and wrong** (npz SST assumes a 270 fill; the store fills −1.8).
  - **Dead-and-wrong metadata is worse than none** — it fooled **two independent auditors**
    (me and a cold Fable 5 agent) into "the model can't see SST". It can; the arrays are never
    read. Deleted, not corrected. The converter has **no npz dependency** now: layout + fill.
  - **New `--random-sample N --seed S`** + `polaris/polaris_verify_data_prep.pbs`: a small store
    spanning EVERY year, verified **exhaustively**. Green: **`SEQZARR_VERIFIED (EXHAUSTIVE:
    40 samples × 162 channels = 6480 channel-samples, bitwise)`**, job **7257786**, 24 distinct
    years. The full script now needs `-v CONFIRM_FULL=1` (the `--max-samples` fix *armed* it).
  - **STILL OPEN (4, documented not fixed):** a non-contiguous `--years` store passes every gate
    with **8766 h seams**; `nchunks_initialized` is defeatable (zarr 2.18.7 counts `.partial`
    files by prefix — measured 6/6 while a sample was a zero slab); a preemption during the
    *val* conversion **destroys the completed 9 h train store**; `--random-sample` ignores
    `--start-sample` but records it.
  - **5 DECISIONS for jesswan/us** (§0 of the handoff). The sharpest: **PhysicsNeMo forecasts
    SST and sea ice; Pangu and makani prescribe them — and this is an AMIP run, where they are
    prescribed by definition.** `UNPREDICTED` is *our* list, so that one is ours to fix.
  - **The npz "Kelvin" open issue is RESOLVED and its recorded diagnosis was WRONG.** SST is
    **°C** (measured `[−1.80, 32.21]`). The npz mean of 109.963 is arithmetically a **270
    land-fill of °C data**: `0.374×270 + 0.626×14.70 = 110.06`. The npz is not broken — it
    *encodes* Pangu's `SST: 270.` fill. Do not "fix" it to Kelvin.
  - **Analysis is 1/3 done:** PhysicsNeMo's converter audited, Pangu's stats prep audited (64
    lines, metadata-only, clean). **makani's 367-line converter is unaudited** — it flips
    latitude, truncates to 10 of 18 levels, renames channels, and its stats *are* live. §8 of
    the handoff scopes it.

- **2026-07-15** — **🔴 CORRECTION + fork-drift audit: `TORCH_COMPILE_MODE` was NOT plumbed in
  PanguWeather, and this file said it was.** Prompted by the question "have the
  `bench_report.md` optimizations — especially the ViT ones — already been done in
  PanguWeather?". Checked instead of assumed (DESIGN §2c: the forks share code by **copy**, so
  "nothing tells you the other copy drifted"). Full table: `polaris_bench_report.md` §6b.
  - **The error:** the harness port brought `S2S_BENCH`/NVTX across but **not** the compile
    knob. PanguWeather had only a commented-out `torch.compile(self.model, mode='default')`
    (`train.py:639`) and no env read — exactly as DESIGN §2c's table already said
    (`TORCH_COMPILE_MODE`: s2s **2**, PanguWeather **0**). I should have read my own table.
    The commented-out `export TORCH_COMPILE_MODE=…` in both new bench scripts was therefore a
    **live trap**: uncomment it → no compile, no error → "torch.compile doesn't help this
    model". **Now genuinely wired** in `get_model()` (gated; unset ⇒ legacy path) + a test
    that fails if the knob is ever disconnected again.
  - **The drift is BIDIRECTIONAL** — each fork has something the other lacks:
    **`static_graph=True`** is in s2s and **missing in PanguWeather**;
    **`gradient_as_bucket_view=True`** is in PanguWeather and **missing in s2s**.
  - **PanguWeather is AHEAD on several**: bf16 is the YAML default (`amp_dtype: bfloat16`)
    rather than an env knob defaulting to fp16; `--async_save` reaches more files; there are
    more `os.path.isfile` checkpoint guards. The per-iteration `empty_cache()` removal and the
    per-param `.item()` removal landed in both, independently.
  - **`static_graph=True` is a candidate, NOT a known win — do not just copy it.**
    `bench_report.md` §4 changed bf16 and `find_unused_parameters=False`+`static_graph=True`
    **together** (+5.3% for the pair), so `static_graph`'s isolated contribution was never
    measured, and PanguWeather already has the expensive half. Worse, s2s needs a
    **dead-module freeze** (`layer_perturbation2`/`layer_purturbation_e2`, `train.py:437-444`)
    to make `static_graph` legal; PanguWeather has no such freeze, so copying it across could
    fail at runtime on `pangu_plasim`.
  - **The ViT/Swin optimizations: there are none to port.** `bench_report.md` §3's findings
    (LayerNorm-backward 2nd-largest, layout conversions, `roll`, matmul only 6th) are
    **profiler observations explicitly deferred to `torch.compile`**, and rung 2
    (FlexAttention) is unstarted — in *either* fork. A diff of the two `networks/pangu.py`
    shows the **only** perf-relevant divergence is s2s's NVTX ranges: SDPA is already in
    `EarthAttention3D` in both, and both have the same 2 `torch.roll`s, 13 `LayerNorm`s and
    identically commented-out block checkpointing. **And the ViT does not run on Polaris** —
    the green path is `sfno_plasim` → `networks/modulus_sfno/`, which never touches
    `pangu.py`. The ViT (and its VAE, and the new `vae_noise` hook) belong to `pangu_plasim`,
    blocked on PLASIM data.
  - **Worth noting:** the H100 **ViT** and the A100 **SFNO** — different architectures —
    profile the *same way*: elementwise-dominated, matmul secondary. Two independent
    measurements, one conclusion: `torch.compile` is rung 1.

- **2026-07-15** — **Profiling phase: PanguWeather SFNO profiled on 4×A100. Full report:
  `polaris_bench_report.md`.** Branch `polaris-profiling` (stacked on the still-unmerged
  `polaris-pbs-bringup`). Headlines:
  - **Instrumentation had to be built first** — PanguWeather carried 0 NVTX ranges and no
    `S2S_BENCH` (DESIGN §2c). Ported from `s2s/v2.0` with range names + CSV columns
    byte-identical (CLAUDE.md #10), gated so unset knobs ⇒ legacy path byte-for-byte.
    **Proven, not asserted:** job **7255505** (no `S2S_BENCH`) reproduced the GREEN
    reference **7253591** exactly — train loss **0.3411**, valid_loss
    **0.7049359679222107**, bit-identical. Adapted where the fork differs: the scaler can be
    `None` (bf16), EMA is real hot-path work s2s lacks (own `ema` range), and `amp_dtype` is
    recorded from the dtype actually used (PanguWeather takes precision from the YAML, so
    reading `$S2S_AMP_DTYPE` would have mislabelled every row).
  - **VERDICT 1 — GPU-bound.** Loader idle is **0.7%** at the shipped `num_data_workers: 1`
    (job 7255410). The §5 kernel ladder is **not** premature.
  - **VERDICT 2 — elementwise-bound, not matmul-bound.** **61%** of GPU time is
    pointwise/elementwise over ~1506 launches/step; GEMM is only **15%**; NCCL 10.5%;
    cuFFT/SHT just 3.3% (job 7255503). Memory-bandwidth bound and fusion-starved ⇒
    `torch.compile` is the right first lever **on evidence**, not assumption.
  - **The model is 1,182,108,160 params** — 1.18 B, **not** the "~79M" DESIGN/CLAUDE.md
    assume (that figure is the Pangu/Swin model, not the E3SM SFNO). 26.98 GB peak of 40 GB.
  - **`cpu_prep_frac` is NOT the data-loader idle fraction** — a trap worth remembering. It
    times `_prepare_inputs_batch` on an **already-fetched** batch (0.3–0.6% of the step even
    with the loader deliberately starved). The blocking fetch happens in `__next__`,
    *between* steps, inside no step window. Worse, it was **fatal**: the elapsed-vs-sum
    self-check fires on an input-bound run and **refuses the row**, i.e. the harness aborts
    exactly when the loader is the finding. Now measured (`loader_wait_med`/`loader_wait_frac`,
    appended after s2s's 19 columns) and folded into the check, which makes it *tighter*.
    **Falsified before believed:** `workers=0` moved it 0.7% → 14.8% (21×) while
    `cpu_prep_frac` stayed flat — a metric that cannot move proves nothing.
  - **`elapsed` was sampled AFTER `cudaProfilerStop()`** (inherited from s2s), folding the
    profiler's buffer flush into the measured wall time. Job **7255503** read `elapsed=51.8s`
    vs `sum=25.7s` and threw away a good bench row — on **every** nsys run. The timers were
    fine; the clock was stopped in the wrong place. Fixed; the re-run **7255557** records
    cleanly at rc=0.
  - **`samples_per_s` is a STEP RATE, not throughput** — it excludes the loader gap. At
    `workers=0` it reads 6.50 while the truth is 5.53; quoting it would have ranked the
    **slower** config first. Convert: `wall = samples_per_s × (1 − loader_wait_frac)`.
  - **`num_data_workers` is NOT output-neutral here** — `data_loader_multifiles.py:1031/1102`
    draws per-sample gaussian noise **inside the workers** (`epsilon_factor: 0.1`) and there
    is **no `worker_init_fn`**, so the worker count changes the noise realization and moves
    the loss. `1 → 8` is **+9% wall throughput and 10× less jitter** (step_p90 0.826→0.603) —
    recorded as a **finding, not a recommendation**. Clean fix: a seeded `worker_init_fn`.
  - **Nothing was optimized.** `TORCH_COMPILE_MODE` is wired and left unset; the §4 gate is
    not executable until a baseline is captured.

- **2026-07-15** — **§4.0 on PanguWeather: the seed prerequisite was already satisfied.**
  The handoff implied `--seed` needed porting from `s2s/v2.0/utils/seeding.py`. It does
  **not**: `train.py:3825` has `--global_seed` (default 0) → `seed_torch()` (`:3742`, called
  at `:3785`), which seeds `PYTHONHASHSEED`/numpy/torch/CUDA and sets `cudnn.benchmark=False`
  + `cudnn.deterministic=True`. That is **stronger than s2s's legacy path** — the numpy gap
  that made s2s's baselines irreproducible does not exist here, which is *why* 0.3411 is
  bit-reproducible. **Porting `seeding.py` would create two competing seed mechanisms — don't.**
  Remaining gaps: Python's `random` unseeded, `torch.use_deterministic_algorithms(True)` never
  set. Side note: `cudnn.benchmark=False`/`deterministic=True` are therefore **always on** —
  a performance fact hiding inside a reproducibility mechanism.

- **2026-07-14** — **DESIGN §4.0 seed knob: DONE.** `s2s/v2.0/train.py` gains `--seed` and
  `--deterministic`; the logic lives in the new shared `s2s/v2.0/utils/seeding.py` (imported
  by S2S *and* the port, CLAUDE.md #5 — additive, nothing existing changed).
  Precedence: `--seed` > `$S2S_SEED` > the YAML's `seed:` > **legacy**.
  - **Opt-in by design.** No seed => the historical path is preserved *byte-for-byte*
    (`torch.manual_seed(world_rank)`, `cudnn.benchmark=True`), which is what lets this ship
    without re-validating the existing greens. A test pins that property.
  - **What was actually broken:** `torch.manual_seed(world_rank)` seeded torch only.
    **numpy was never seeded** — and `train.py:1251` draws the validation sample from it
    (`np.random.randint`) — so two runs of the "same" config diverged. `random` was unseeded
    too, and `cudnn.benchmark=True` picks kernels by timing. A "reproducible baseline" on
    that footing was not reproducible.
  - **Rank offset:** `seed + world_rank`, preserving the legacy intent (distinct streams per
    rank, so the loader's per-sample noise — `data_loader_multifiles.py:474-481`, drawn in
    the workers — doesn't correlate across ranks). At rank 0 the applied seed IS the seed,
    so a §4.1 world-size-1 baseline is comparable with the port's `seed_everything(s)`. A
    multi-rank baseline is NOT comparable across launchers — documented in
    `seeding.equivalent_to_seed_everything`.
  - **Tests:** `s2s/v2.0/test/seeding_test.py` — 10 assertions, **`SEEDING_OK`**, runs with
    no ERA5/GPU/cluster (deliberate: the S2S+port data smokes are blocked on the ERA5 stage,
    so the mechanism is proven without them). Covers same-seed reproduction, different-seed
    divergence, the numpy gap, byte-identical legacy, precedence (incl. **seed 0**, the
    classic falsy bug), loud failure on a bad seed, rank offsets, and model-level identical
    init+forward+backward. `polaris_seeding_test.pbs` runs the CUDA half on a real GPU and
    **fails rc=4 if CUDA was not visible** — a skipped test must never read as a pass.
    **GPU-verified: job 7253738, rc=0, `SEEDING_OK (10 tests)` +
    `CUDA was visible -> the CUDA RNG assertion really ran`.** So CUDA RNG reproducibility is
    demonstrated on the device a baseline would actually be captured on, not just on CPU.
  - **Still blocking baseline capture:** `tiny_baseline.yaml` and the VAE noise-fix hook.
    Also note `--deterministic` needs `CUBLAS_WORKSPACE_CONFIG=:4096:8` exported *before*
    python starts (the PBS script does it); `enable_determinism()` warns rather than
    pretending when it is missing.


- **2026-07-14** — **🔴 The "GREEN" smokes were green for ONE PERSON. Fixed.** A cold
  5-agent audit of the *fixed* tree (the second gauntlet) surfaced that Pangu/SI depended on
  `pip install --user` packages living in `PYTHONUSERBASE=/home/rmehta1987/.local/...`.
  **ALCF home dirs are mode `0700`**, so those packages are readable by their owner alone.
  Every Pangu/SI "GREEN" was therefore unreproducible by the rest of the project — the exact
  opposite of this deliverable's purpose — and `polaris_running_the_smokes.md` told jesswan
  "they use software already installed on Polaris", which was false.
  - **Proof, not inference:** job **7253539** re-ran Pangu with `PYTHONNOUSERSITE=1` (which
    reproduces a second member's view of the filesystem) and died on
    `ModuleNotFoundError: No module named 'tensorly'`. Impersonating the other user is the
    only way to catch this class of bug; a normal re-run by the installer always passes.
  - **Fix:** `polaris_setup_base_topups.sh` installs netCDF4 / zarr / torch_harmonics 0.7.4 /
    tensorly / tltorch / cftime / numcodecs into the **shared, world-readable**
    `$POLARIS_TOPUPS` on eagle; Pangu/SI/S2S/probe prepend it to `PYTHONPATH`.
  - **Two traps inside the fix**, both now guarded:
    (1) `pip install --target` can't see the base conda, so it re-resolved the world and
    silently pulled **torch 2.13 + CUDA 13 + numpy 2.5.1** (4.1 GB) — which, being on
    PYTHONPATH, would have **shadowed the base's torch 2.8/cu12.9** and moved every smoke
    onto an untested toolchain. `--no-deps` + a hard fail if `torch|numpy|nvidia|triton`
    land in the target; now **64 MB**, and it asserts torch/numpy still come from base.
    (2) `$POLARIS_TOPUPS` must NEVER go on PYTHONPATH in an SFNO job — its
    torch_harmonics 0.7.4 would shadow the venv's 0.9.x and re-break makani
    (`PYTHONNOUSERSITE` does **not** block PYTHONPATH). Both SFNO scripts now assert
    torch_harmonics resolves inside their venv (`ERROR TORCH_HARMONICS_SHADOWED`).
  - **Proven fixed, not assumed:** Pangu **7253591** (`PYTHONNOUSERSITE=1`) rc=0 with loss
    **0.3411 — bit-identical** to the installer's 7253401, and SI **7253603** step_med 0.399 /
    peak 30.69 GB (vs 0.400 / 30.98: noise). Identical rather than merely similar matters: it
    shows the shared top-ups serve the *same code* the greens ran on. Version pins in
    `$POLARIS_TOPUPS` match the old `~/.local` ones exactly.
  - **Regression-proofed:** `polaris_require_topups()` (in `polaris_env.sh`, called by **all 8**
    base-conda jobs — the SFNO pair is deliberately exempt) fails the run with `ERROR TOPUPS_MISSING` or `ERROR PRIVATE_DEPS_ON_PATH`
    if a dep ever resolves from a private home again. Both branches tested: unsetting
    PYTHONPATH reproduces the original bug and the guard catches it. **Note the asymmetry the
    guard exists for — this bug is invisible to the one person who could fix it**, because
    their own runs pass.
  - **A reasoning error worth remembering.** When deriving the top-ups list I used the rule
    *"missing for the installer too => off the smoke path"*. That is only valid for code that
    has **actually run green**. The S2S/port smokes have never run on Polaris (blocked on
    ERA5), so for them "missing for everyone" means **broken for everyone** — and I dropped
    `cf_xarray` on that basis. It is a bare import at `s2s-lightning/modules/train_module.py:52`
    reached from both port entrypoints. The port would have died at import right after a
    multi-TB Globus stage. Caught by the third cold gauntlet; `cf_xarray` is now in the
    top-ups and all 5 entrypoint chains import as a second user.
  - **The probe's port check was hollow.** It imported `common, data, modules`, which have no
    `__init__.py` — namespace packages, so the import succeeds without executing any of the
    smoke's code. That is how a bare missing import survived a green `PROBE_OK` while the
    docs claimed the port's env was "proven by the probe". The probe now imports
    `modules.train_module`, and `polaris_require_topups` covers **8/8** base-conda jobs (it
    was 5/8; the probe and both port jobs were missed, contradicting this very entry's
    earlier claim of "every base-conda job").
  - **Lesson (generalise):** never `pip install --user` a dependency the project must share,
    and never accept "it's green" from the environment that installed it. The probe
    (7251974) had the same blind spot — it certified "all models import" while importing
    from a private home; it now imports through the shared dir and warns if `~/.local` is on
    `sys.path`.

- **2026-07-14** — **Makani's re-run was a silent no-op (`rc=0`, zero steps).** With a
  hardcoded `--run_num 0`, `train_plasim` auto-resumed from a checkpoint that already
  satisfied the smoke's `max_epochs=1`: job **7253454** printed `Total training time is
  0.00 sec` and exited **0**. `RUN_NUM` now defaults to `${PBS_JOBID%%.*}`, plus a gate that
  forces `rc=4`/`ERROR NO_CHECKPOINT` when a run exits 0 without writing its checkpoint.
  Revalidated by **7253465** (train 2.61 / val 2.38, 7.10 s of real training).
  **`rc=0` is not a PASS criterion for a resumable trainer** — key on the work token.
  Related: the smokes have **no seed knob** (DESIGN §4.0), so their losses move run-to-run
  (7252769: 2.19/2.05 vs 7253465: 2.61/2.38 on identical code) — they are **not** an
  equivalence baseline.

- **2026-07-14** — **Audit fixes (docs + scripts).** `CONVERT_OK` re-attributed **7252736 →
  7252728** (7252736 packed nothing and failed rc=1; verified from the log). The
  `disassemble_input` note corrected: **fixed** in `train_module.py` (`1fef2473`) but still
  **open** in `bias.py:226`, `ae_module.py:68`, `combined_module.py:185/287`. All 3
  converters now honour the advertised `$E3SM_ROOT` (only makani did). All 10 `*.pbs` now
  source `polaris_env.sh` (5 didn't, so the notes' "every script pins the caches" was false).
  `polaris_logs/.gitkeep` committed — the dir is gitignored, so the probe's `#PBS -o
  polaris_logs/` had nowhere to deliver in a fresh clone. CLAUDE.md's "Polaris/PBS = single
  `python`" corrected (6 of 10 use torchrun). Cleanup doc's `.npy` "loaded by" column fixed
  (`pangu_lite.py` only mentions the masks in a comment) — the conclusion (never blanket-
  ignore `*.npy`) was right, the evidence wasn't.


- **2026-07-14** — **Polaris (PBS) bring-up.** Confirmed cluster facts (`-A
  lighthouse-uchicago`, 4×A100-40GB sm80, `debug` queue, `filesystems=home:eagle`,
  `/local/scratch`); env = base ALCF conda (`module load conda`, torch 2.8/cu12.9) +
  `pip install --user` netCDF4/zarr/**torch_harmonics 0.7.4** (0.9.1 ABI-breaks torch 2.8).
  **Probe GREEN** (job 7251974). **PanguWeather-SFNO 4-GPU smoke GREEN** (job 7252271):
  climatology CDF-5→NETCDF4 auto-prep + 1 bounded epoch, train loss 0.3411, DDP
  validation, rc=0. Two traps recorded in `polaris_pbs_notes.md`: (1) Pangu `--debug`
  hardcodes `world_size=1` → OOMs under `torchrun -n4` (bound with `--epochs 1`
  instead); (2) Lustre needs `HDF5_USE_FILE_LOCKING=FALSE`. Authored all
  `polaris_*.pbs` (S2S/port/SI/Pangu/makani/physicsnemo) + 3 data converters +
  repointed configs. **S2S/port blocked on an ERA5 Globus stage** (not on Polaris).
  **SI, Makani and PhysicsNeMo also went GREEN** (7252700 / 7252769 / 7252816) — the
  latter two from an isolated SFNO venv (see the In-progress entry). Caches/TMPDIR
  pinned to eagle (persistent), not node-local scratch (per user). A 5-agent cold
  adversarial audit independently re-confirmed every GREEN claim against the raw logs and
  surfaced the fixes applied in `3c0b4e5`. Full detail: `polaris_pbs_notes.md`.
- **2026-07-13** — Model policy set to **main = Opus 4.7 (xhigh effort), subagents =
  Fable 5**. Trimmed CLAUDE.md to stay <200 lines while adding: filled the real
  Midway env paths, a per-model smoke table (what to run + PASS signal), the
  launcher-shape + env-bootstrap rules, the `test.yaml` trap (rule #12), and a
  "where to look" doc map. Ran two cold Fable-5 agents to source the additions.
- **2026-07-13** — PR #4 (`polaris-pbs-handoff`) merged to `main` (`4c283f2`);
  `polaris_handoff_prompt.md` is on `main`.
- **2026-07-13** — Cold adversarial review of the docs (three Fable-5 agents); applied
  the findings (SI `bench.py --config <path>` command, DESIGN §2 launch table,
  `data_prep` NVTX name, a concrete §4 + its §4.0 prerequisites, baseline
  `.pt`-vs-`.gitignore` fix, interactive-allocation preface, `pytest --fast` hedge).
  **Also fixed a real regression:** the port smokes hardcoded a cwd-relative
  `v2.0/config/test.yaml` (pre-monorepo) → now resolved relative to `__file__`.
- **2026-07-13** — Added `DESIGN.md`, `CLAUDE.md`, `CHANGELOG.md` (design spec,
  working guide, living doc) patterned on `smsharma/clax` + the MARSHAL/decrypto
  playbooks. Establishes the **numerical-equivalence-vs-baseline** gate as the oracle.
- **2026-07-13** — Published the repo; repo-wide **SNFO → SI** rename (SI is correct;
  SNFO a mislabel). NGC key scrubbed to `$NGC_API_KEY`. `main` branch-protected.

## Known issues / failed approaches (do NOT re-attempt)

Each is attributed to its source doc — verify there before acting.

- **(PanguWeather) Don't assume a `bench_report.md` optimization reached this fork — the drift
  is bidirectional.** `static_graph=True` is in s2s only; `gradient_as_bucket_view=True` is in
  PanguWeather only; bf16/`--async_save`/checkpoint-guards are *ahead* in PanguWeather. Full
  table before acting: `polaris_bench_report.md` §6b.
- **(PanguWeather) Do NOT copy `static_graph=True` across without the dead-module freeze** —
  s2s freezes `layer_perturbation2`/`layer_purturbation_e2` (`train.py:437-444`) to make it
  legal, and PanguWeather has no such freeze. Its isolated gain was also never measured
  (`bench_report.md` §4 changed it together with bf16). — `polaris_bench_report.md` §6b.
- **(PanguWeather) There are no ViT/Swin optimizations to port — they were never implemented
  in either fork**, and the ViT doesn't run on Polaris anyway (`sfno_plasim` uses
  `networks/modulus_sfno/`, never `networks/pangu.py`). `bench_report.md` §3's LayerNorm/
  layout/`roll` findings are observations deferred to `torch.compile`; SDPA is already in both.
  Don't go hunting for a missing port that doesn't exist. — `polaris_bench_report.md` §6c.
- **(polaris_env.sh) `-v SEQZARR_DATA=…` (and the other `_pick` vars) CANNOT be overridden** —
  `_pick` never reads its first argument, and `polaris_env.sh:155` exports unconditionally. Job
  7257791 was submitted with `-v SEQZARR_DATA=…_fresh` to force the PhysicsNeMo smoke to rebuild
  its store with a changed converter; it silently used the OLD cached store and passed. A gate
  that cannot be pointed at fresh data is not a gate.
  — `polaris_data_prep_handoff_prompt.md` §4.
- **(E3SM archive) The .h5 construction is NOT in this repo** — it lives in
  `/eagle/.../jesswan/PanguWeather/data_utils/` (`netcdf_to_h5*.py` ×3, `get_stats.py`, adapted
  "from FourCastNet repo"). Read it before deciding the fill questions: it is the ground truth
  for the 270 sea-surface-temperature fill, the 19 constant cloud channels, and the frozen
  `time` year. Three `netcdf_to_h5` variants exist and nothing records which built the archive.
  — `polaris_data_prep_handoff_prompt.md` §8c-bis.
- **(E3SM data prep) `CONVERT_OK` is NOT a verification** — it checks 1 channel of 1 sample
  (0.01%) and is blind to the NaN fill by construction (its probe channel is chosen as one
  with no fill). Require `SEQZARR_VERIFIED` from `polaris/verify_seqzarr.py`.
  — `polaris_data_prep_handoff_prompt.md` §4.
- **(E3SM data prep) The PhysicsNeMo smoke store CANNOT validate the full conversion** — it is
  64+16 samples of **2015 only**, and all three worst defects were invisible at exactly that
  scale. Use `polaris/polaris_verify_data_prep.pbs` (`--random-sample`, spans every year).
  — `polaris_data_prep_handoff_prompt.md` §5.
- **(E3SM archive) `input/time` is FROZEN AT 2015 in every file** — `2049_1459.h5` carries
  `2015-12-31 18:00:00`. Month/day/hour track the index; only the year is wrong. Never build a
  time axis from the in-file label; take the year from the filename. Upstream defect.
  — `polaris_data_prep_handoff_prompt.md` §1.
- **(E3SM stats) The npz SST mean of ~110 is NOT "Kelvin data"** — that inference (previously
  recorded here as an open issue) is refuted. SST is °C; 110 is arithmetically a 270 land-fill:
  `0.374×270 + 0.626×14.70 = 110.06`. The npz encodes Pangu's `SST: 270.` fill and is
  self-consistent. Do not "fix" it. — `polaris_data_prep_handoff_prompt.md` §3.
- **(zarr) `nchunks_initialized` is not a completeness check** in zarr 2.18.7 — it counts chunk
  keys by *prefix* regex, so a `.partial` left by a kill mid-write counts as written (measured
  6/6 while a sample was an all-zero slab). Compare the exact expected key set.
  — `polaris_data_prep_handoff_prompt.md` §4.
- **(PanguWeather) Do NOT port `s2s/v2.0/utils/seeding.py` into PanguWeather** — it already
  has `--global_seed` → `seed_torch()`, which is more complete than s2s's legacy path. Two
  seed mechanisms racing to set the same global RNG is a regression, not a port.
  — `polaris_bench_report.md` §6.
- **(PanguWeather) `cpu_prep_frac` is NOT the data-loader idle fraction** — it times
  `_prepare_inputs_batch` on an already-fetched batch and stays at ~0.4% even with the loader
  deliberately starved to a 14.8% stall. Use `loader_wait_frac`. — `polaris_bench_report.md` §3.
- **(PanguWeather) `samples_per_s` is a step rate, not wall throughput** — it excludes the
  between-step loader fetch. Convert with `× (1 − loader_wait_frac)` before comparing two
  configurations, or you will rank the slower one first. — `polaris_bench_report.md` §3.
- **(PanguWeather) Don't read the NVTX sub-ranges as GPU time** — they are pushed/popped on
  the CPU thread and measure *enqueue*; they sum to 55% of the step, and the rest is the
  terminal `cuda.synchronize()` draining the GPU. `backward = 280 ms` is CPU launch work, not
  47% of GPU time. Attribute GPU time from the kernel table. — `polaris_bench_report.md` §4.1.
- **(PanguWeather) Bumping `num_data_workers` changes the LOSS, not just the speed** — the
  loader draws per-sample gaussian noise inside the workers (`epsilon_factor: 0.1`) with no
  `worker_init_fn`. The +9% from `1 → 8` is real but cannot be validated by the §4 bitwise
  gate. Fix `worker_init_fn` first. — `polaris_bench_report.md` §3.
- **The E3SM SFNO is 1.18 B params, not "~79M"** — the 79M figure (DESIGN, CLAUDE.md #12)
  describes the Pangu/Swin model. Don't carry 79M-era resource intuition onto the SFNO path.
  — `polaris_bench_report.md` §1.

- **(Polaris) `torch_harmonics` version box** — makani 0.2.0 imports the *public*
  `torch_harmonics.quadrature.precompute_latitudes`, which does NOT exist in 0.7.4 or
  0.8.0 (private `_precompute_latitudes`). 0.9.1 has it but ships **wheels only** (no
  sdist on PyPI — `--no-binary :all:` cannot build it) and its prebuilt
  `attention/_C.so` fails on torch 2.8 with `undefined symbol:
  _ZNK3c1010TensorImpl15incref_pyobjectEv`, so `import torch_harmonics` dies outright.
  Don't re-try pinning a PyPI version — install from the GitHub source (compiles `_C`
  against the local torch) **and re-verify the green Pangu-SFNO smoke**, or isolate the
  SFNO frameworks in their own venv. — `polaris_pbs_notes.md` §6.
- **(Polaris) Pangu `--debug` is single-GPU ONLY** — it hardcodes `world_size=1`, so
  under `torchrun --nproc_per_node=4` all 4 ranks init as rank-0-on-GPU-0 and OOM the
  40 GB A100. Bound a smoke with `--epochs 1` instead. — `polaris_pbs_notes.md` §5.
- **(Polaris) SI `calendar: 'noleap'` crashes the loader** — noleap is an *idealized*
  cftime calendar that forces `has_year_zero=True`, clashing with `has_year_zero: False`
  at `si/data/amip_new.py:667` (`cannot compute the time difference between dates with
  year zero conventions`). Use `'standard'` (correct for a non-leap-year smoke); a full
  run crossing a leap year needs a loader fix. — `polaris_pbs_notes.md` §5.
- **Port standalone smokes had a stale cwd-relative config path** (`v2.0/config/test.yaml`,
  pre-monorepo) → `FileNotFoundError` before any GPU work. Fixed 2026-07-13 (resolve
  relative to `__file__`). If a port smoke can't find the config, check this first.
- **"Missing kernel tables" are NOT a profiler/ptrace limit** — an unconditional
  `restore_checkpoint()` crashed on `FileNotFoundError` before any GPU work (a
  byte-identical CUDA-API fingerprint). Fixed with the `os.path.isfile` guard. If a
  profile has no kernel table, **read the `.err` first.** — `bench_report.md` §II.7.
- **S2S batch ≥3/card (bf16) is a trap** — throughput collapses near allocator
  saturation and 4/card OOMs; the known-good ceiling on ~94 GB cards is **2/card**.
  — `bench_report.md` §II.4.
- **`num_data_workers=0` fakes a GPU-idle "bottleneck"** (large idle %; SI's first
  4-GPU bench failed its sanity check on HDF5 reads). Known-good: 8 workers +
  `--cpus-per-task=8`. — `bench_report.md` §II.7 / `si/bench_midway_notes.md`.
- **Inference: always pass `--async_save`** — synchronous NetCDF saving throttles
  rank 0 well below the other ranks. — `bench_report.md` §II.7.
- **Don't remove SI's fp32 island around the spherical-harmonic transform** — bf16
  breaks `torch_harmonics` (`view_as_complex` rejects bf16); it's wrapped in
  `autocast(enabled=False)` on purpose, cost ≈ 0. — `si/bench_midway_notes.md` §3–4.
- **SI + `torch.compile max-autotune` / nsys-on-compiled-SI** — reported to
  crash/segfault (CUDA-graph capture; tracing the compiled DDP backward). Use
  `default` compile mode; profile eager, bench compiled. — `si/bench_optim_sweep.sh`
  header (verify before relying on it).
- **DSI handoff-latency investigation** — several hypotheses already investigated;
  the current lead is a **driver/CUDA mismatch** (not interconnect). Read
  `bench_report.md` §I.5/§I.7 and don't re-run the ruled-out ones.

## Open questions (answer + record here)

- **Baseline node class.** The SI optimization-sweep CSVs (`si/bench_optim_*.csv`)
  ran on the **test partition H100**, a different node class from the pedramh-gpu
  H100-NVL numbers — re-measure a pedramh-gpu baseline to compare like-for-like.
- ~~**A100 (Polaris) memory**~~ — **RESOLVED** by probe 7251974: `nvidia-smi` on-node
  reports **40960 MiB/GPU** (4× A100-SXM4-40GB, driver 570.124.06). See
  `polaris_pbs_notes.md` §1.
- **SI compile gain** — reported ~+62% (`default` mode) but a `*_postfix` re-run is
  lower; quote it as a range until re-measured on pedramh-gpu.

## Benchmark results

**Read the existing evidence before capturing baselines or claiming a speedup**
(compare only within a cluster, never A100 vs H100 NVL):
- `s2s/v2.0/bench_report.md` — S2S H100-NVL baselines + the step-time / VAE-encoder split.
- `si/bench_midway_notes.md` — SI bench + decisions log (refutes the "H200" label).
- `si/bench_optim_*.csv` + `si/bench_optim_sweep.sh` header — the 2026-05 one-lever-
  at-a-time SI optimization sweep (test-partition H100 — a different node class).
- `s2s-lightning/LIGHTNING_PORT.md` — the port's DDP/AMP/bench wiring + per-phase
  smoke-id table. **The port-vs-v2.0 nsys caveat** is in the header of
  `s2s-lightning/midway_bench_nsys_port.sh`: the port's per-step NVTX window opens at
  `on_train_batch_start` (after H2D), so its `step_med` excludes the transfer —
  compare throughput via `samples_per_s_wall`, never `step_med`.
- `s2s/v2.0/HPC_scripts/bench_methodology.md` — what every `bench_results.csv` column
  means and why timing is `cuda.synchronize`-bracketed.

**Hardware identity (do not reintroduce refuted labels):** `pedramh-gpu` is
**H100 NVL (~93 GB)**, NOT "H200" (a commit message said so; refuted in
`si/bench_midway_notes.md`) and NOT "80 GB H100". NVLink is within socket-pairs only
(GPU0↔1, GPU2↔3); the host is PCIe Gen4. The Midway H200 *test* partition is a
separate node class (full-mesh NVLink, PCIe Gen5).

**How to capture a baseline (BLOCKED on §4.0):** procedure = DESIGN.md §4.1, storage
= §4.2 (JSON/CSV summary in git, tensors on cluster storage), metric definitions =
`bench_methodology.md`. Record each capture as a dated row here.

_(record new per-cluster bench deltas below — model, cluster, config, samples/s,
peak mem, and the equivalence result for any optimization.)_
