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
| **Profiling (PanguWeather SFNO on A100)** | 🟡 **first pass done, then RE-OPENED** by `PANGU_POLARIS_PROFILING_PLAN.md` (21 items; **1, 2, 3, 4, 5, 6, 6b, 7 done**) — **271 ms/rank-step (47% of compute time)** is `direct_copy`+`conj`, kernels that compute nothing, split **72.9% `backward`** (§4.3). Quote the ms, not a share of GPU-kernel time: that share is not reproducible (§4.4c). See `polaris_bench_report.md`. Harness ported (PanguWeather had **zero** instrumentation), loader sweep + nsys captured. **VERDICT: GPU-bound** (loader idle **0.7%**) and **elementwise-bound** (**68% of *compute* pointwise vs 17% GEMM**, 392 vs 97 ms/rank-step) ⇒ `torch.compile` (§5 rung 1) is the right first lever, now on evidence. **2026-08-21 (item 7, job 7550715, prereg 4/4): those copies are CONTIGUITY-bound, not bandwidth-bound** — store side at exactly the ideal sectors/request, load side at the hardware maximum of 32.00, and the 377 MB spectral weight reads **2043 MB to move 377 MB**. Nothing is saturated (DRAM 24–51% of peak, SM 5–20%). ⇒ the lever is a **layout fix**, and §4.5's "only dominant kernel with no mechanism" now has one (§4.8). Model is **1.18 B params**, not ~79M. SI/makani/physicsnemo **not yet profiled**. |
| §4.0 prerequisites — **`s2s/v2.0`** | 🟡 **seed knob DONE + GPU-verified** (`--seed`/`$S2S_SEED`/YAML + `--deterministic`, `s2s/v2.0/utils/seeding.py`; 10 tests `SEEDING_OK` on CPU **and on an A100**, job 7253738 rc=0); tiny config + VAE noise-fix still **block baseline capture** |
| §4.0 prerequisites — **`PanguWeather`** (the focus; a separate fork, nothing propagates) | ✅ **ALL THREE MET.** seed knob ✅ **already existed — do NOT port `seeding.py` here** (`--global_seed`→`seed_torch`, seeds numpy+torch+CUDA, forces `cudnn.deterministic`; stronger than s2s's legacy path). VAE noise hook ✅ **built** (`utils/vae_noise.py`, 16 tests `VAE_NOISE_OK`) but **inert on `sfno_plasim`** (no VAE). `tiny_baseline.yaml` ✅ **written AND run** — job 7255583: **7,166,656 params** (165× smaller than the real 1.18 B), 0.023 s/step, **1.00 GB**. ⇒ **baseline capture is no longer blocked on building anything** |
| **E3SM data prep (PhysicsNeMo zarr)** | 🟡 **7 defects found, 5 fixed, 4 open**; verified `SEQZARR_VERIFIED` on a 24-year random fixture (job 7257786). **The full ~1 TB conversion is NOT cleared to run** — 4 open defects + 5 decisions. `polaris_data_prep_handoff_prompt.md`. makani's converter **unaudited**; Pangu's stats prep audited (clean, metadata-only). |
| Correctness baselines captured (DESIGN.md §4) | 🟡 **machinery BUILT and first baselines captured** (`baselines/ai_rossby_pangu_plasim/{eager,compiled_default}.json`, job 7353187) — `equivalence.py` + `compare_baselines.py` + a PBS gate. **First verdict: `torch.compile` FAILS** (4.02e-01 > 1e-2), so rung 1 is measured (1.40×) but **not adopted**. Open question for the owner: §4.1 gates on a 20-step bf16 *training* trajectory, which compounds — step 0 agrees to 8.3e-4. Other models still have no baseline |
| Test harness (tier-1 equivalence/unit + `--fast`) | 🟡 3 test files now exist + self-run (`SEEDING_OK`, `BENCH_INSTR_OK`, `VAE_NOISE_OK`); no `conftest.py`/`--fast` yet |
| Optimization ladder (DESIGN.md §5) | 🟡 **rung 1 MEASURED on ai-rossby, not adopted**: `torch.compile` (default mode) = **1.401×** (step_med 449.6→320.8 ms, peak mem 24.98→21.07 GB; jobs 7352022 vs 7352948). Still **not enabled anywhere** — §4 equivalence gate does not exist yet. Measuring is unblocked; adopting is not. PanguWeather/SI/s2s rungs untouched |

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

> **2026-08-20 — Pangu-on-Polaris profiling to-do list lives in
> `PANGU_POLARIS_PROFILING_PLAN.md`.** Tiered by cost, with what each item unblocks.
> Top three, in order: (1) **capture the missing PanguWeather §4.1 baseline** — it gates
> every hot-path change; (2) **ncu single-rank on the top six kernels** — settles whether
> the 42%-of-GPU-time copies are bandwidth-bound or contiguity-bound, i.e. whether
> "we're at the maximum" is even true; (3) **multi-node scaling**, the one axis never
> profiled here, where the single-node "comms are free" result (1.2% exposed) will not
> hold. Deprioritized on new evidence: `broadcast_buffers=False` (0.11%, not 33%) and
> the single-node `NCCL_PROTO` sweep (<=1.2% available).
>
> **Updated 2026-08-20 (items 1 and 5 done, no GPU time):** all three of the top three survive — item (2),
> ncu, is *narrowed* by a measured in-capture ceiling (D2D memcpy reaches **82% of HBM peak**, so the
> hardware is not the limit) but not replaced. Two things are now sized rather than guessed: the copy time
> is **72.9% `backward`**, and **checkpointing headroom against the shipped `ckpt2` config is ≈4%, not
> ≈25%** — so the `ckpt` ladder (item 10) is worth *measuring* but is not the prize the "61% elementwise"
> share suggests. Next unchecked: **6b**, then **7** — but see the blocker.
>
> **🚨 2026-08-20 — BLOCKED: `module load conda` AND the base-conda torch are both broken cluster-side.**
> Every PBS script here uses the plain module bootstrap. Item 6 landed only because it needs *torch alone*
> and could borrow the ai-rossby venv; **items 7-17 need the real Pangu env and cannot run.** Details and
> the three failed workarounds: `polaris_pbs_notes.md` §1. Needs a decision: port the proven torch-aware
> bootstrap to the Pangu scripts, or file an ALCF ticket and wait.
>
> **RESOLVED 2026-08-21 — it is not a choice: the base conda must be REPAIRED (ALCF ticket).**
> Substituting the one working environment is **disqualified**, not merely unattractive: it carries
> torch **2.10.0+cu129** where every number in the profile (§0d, §4.3, §4.4, §4.5) was measured on
> **2.8.0**, and items 7-10 exist to refine that same picture. §4.4a already measured that kernel
> selection is not bit-reproducible even within one torch version, so a minor-version bump would
> silently break comparability. (Item 6 was fine on that env because a *topology* measurement is
> torch-independent — it measures hardware links, not model kernels.)
>
> **Updated again 2026-08-20 (item 3 done, no GPU time):** the copy problem is **half a weight problem**.
> **133 ms/rank-step (~21% of the step) moves one 377 MB spectral weight in four places**, all from a
> single layout mismatch, and **`torch.compile` reaches none of it** (Inductor fails outright on the
> complex64 SFNO). DESIGN §5 gains rung **1b** — store the parameter pre-permuted — gated on item 18.
> ⚠ The finding is **batch-1 specific**: weight:activation = `E/(B·mmax)` inverts at batch 4.
>
> **Updated again 2026-08-20 (item 2 done, no GPU time):** **stop quoting shares of GPU-kernel time.** On
> two identical-config runs the copies read **42.2% and 37.4%** while the absolute moved **0.09%** (§4.4c)
> — quote **271 ms/rank-step**. And the single-node "comms are free" result (1.2% exposed) is
> **conditional on rank balance**: the same config on another node gives **4.9–8.8% exposed** with 17 of 40
> steps stalled. ⇒ fix rank placement *before* attributing a multi-node scaling loss to Slingshot. One
> stall is already diagnosed and **output-neutral to fix**: CPython gen-2 GC (item 6b(A)).

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

- **ai-rossby PanguPlasim bring-up — ✅ ALL THREE HANDOFF-v2 STEPS GREEN (2026-08-05).**
  Branch `fix/tsoi-fill-270`. The smoke-scale pipeline is end-to-end proven: corrected
  statistics → rebuilt norm zarr → a 4×A100 training run that converges.
  1. **Normalization regen** — job **7340945**, exit 0, **34:54** over all 51,100 h5 files
     (~2.15 TB). `MOMENTS_OK` → `NORM_NC_OK` → `NORMALIZATION_OK` → `E3SM_NORM_REGEN_OK`.
  2. **Norm zarr rebuilt** — bitwise-identical to the source `.nc` across all 26 vars/levels.
  3. **Training smoke** — job **7341412**, exit 0, **9:36**. `PREFLIGHT_OK`
     (`VARIABLE_PARITY_OK 21/21` ×3 stores) + `PANGU_PLASIM_RUN_OK`.
  4. **Production conversion** — ✅ **DONE 2026-08-06**: 35 stores (train 2015–2044,
     val 2045–2048 + 2049 tail), 43,800 training samples, verified on disk. Took
     3 h through `debug`; `preemptable` never scheduled it (see the 2026-08-06 entry).
     No inode problem materialised — `lfs quota` shows files quota/limit **0 = no cap**.
  **Next: the long training run — and use `-q capacity`, NOT `preemptable`.**
  `capacity` takes **1–4 nodes for ≤168 h** at `Priority=150` and is actively
  scheduling (12 running); the project's slot is currently free. At the *measured*
  449.6 ms/step (not the stale 537) the 100-epoch run is **~150 h — it fits in ONE
  job**, versus 3× 72 h `preemptable` links that cost jesswan ~80 h of inter-link
  queue wait plus a lost partial epoch per kill. Caveat: `capacity`'s
  max_run 1 / max_queued 2 are **per PROJECT**, so coordinate before taking the slot.

- **Pipelines runbook delivered (`polaris_pipelines_plan.md` + operator guide
  `polaris_pipeline_runbook.md`); the §0 smoke sequence is DONE (all four green, see the
  decisions log). Remaining wait: jesswan's zarr transfer** (announced 2026-07-16, not yet on
  disk). The moment it lands:
  `cd physicsnemo_sfno && qsub -v STORE=<transferred-path> polaris/polaris_verify_store.pbs`
  (PASS = `SEQZARR_VERIFIED` per store + `STORE_VERIFY_OK`), then train against it via
  `qsub -v SEQZARR_ALLYEARS_DATA=<dir> polaris/polaris_sfno_allyears.pbs` — if it verifies at
  the current generation, our own ~11 h / ~1.43 TB conversion is unnecessary.
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
- **Layout change: the SFNO codebases are now `git subtree`s of this repo** (not
  separate checkouts as the handoff assumed, not submodules). Imported **unsquashed for
  full provenance**: upstream commits are real ancestors of HEAD (jesswan-uc 8,
  feynmanliu214 38, ktangsali 203). Cost: 313 → 4,769 files, .git 3.9 MB → 306 MB.
  **Was 3; `physicsnemo_ai_rossby/` made it 4 on 2026-08-04** (`0777be0f` ← `87002adb`) —
  and its `examples/weather/ai_rossby/` recipe is subtree-owned too, despite reading as
  fork-owned. See `polaris_pbs_notes.md` §6b's warning box before editing anything there.
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

- **2026-08-27 (cont. 3)** — **MULTI-NODE DDP PORT, HARNESS 1 OF 3: ai-rossby bring-up BUILT
  and committed; measurement started.** Branch `feat/multinode-ddp-port`, stacked on the
  unmerged `profile/pangu-polaris-profiling` (**merge order: `profile/pangu-polaris-profiling`
  FIRST, then this** — the repo has been burned by silent stacking before). Working from
  `polaris_multinode_ddp_port_handoff.md` §3a; prereg in the new
  `ai_rossby_multinode_ddp_plan.md`, committed **before** the first ladder job.
  - **Two dead spots found in `polaris_sfno_e3sm_multinode.pbs`, not one.** The handoff flags
    it as the origin of the stale fabric block. It was *also* opening with a bare
    `module load conda`, broken cluster-side since the 2026-08 PE roll (notes §1) — i.e. the
    script could not start at all, and back-porting only the fabric block would have produced
    a launcher that still failed. New `polaris/polaris_ai_rossby_env.sh` (sibling of
    `polaris_makani_env.sh`) tries the sanctioned module first, reconstructs `2025-09-25.lua`
    by hand when it fails, and reports which path it took in `AI_ROSSBY_ENV_SOURCE` so two runs
    from different sources are never tabled together. **Verified on a login node: it lands on
    `manual-reconstruction`**, i.e. the modulefile is still broken today.
    Three deliberate deltas vs the makani file, all documented in its header: **no** torch-2.8
    `DT_NEEDED` repair (this venv's torch 2.10 ships the whole nvidia stack as wheels), **no**
    h5py overlay (zarr harness; `include-system-site-packages=false`), and **no**
    `/soft/libraries/nccl` prepend — that last one would put NCCL 2.28.3 ahead of the venv's
    own bundled NCCL, i.e. silently change the comm library under a scaling measurement.
  - **Fabric block corrected in place** (§3a sanctions the back-port here): the shipped pin was
    `v1.9.1-aws` + `/opt/cray/libfabric/2.2.0rc1`, both `ls`-verified 2026-08-14 and both gone
    by 2026-08-23. A nonexistent dir on `LD_LIBRARY_PATH` is IGNORED, not honoured — the
    original 7553811 defect. Now the measured pin (self-built **v1.21.1 +
    `OFI_NCCL_PROGRESS_MODEL=AUTO`** + cray 2.3.1) **plus a hard exit on any missing path**,
    which is the half that makes the failure class impossible rather than fixing one instance.
    Explicitly NOT v1.6.0+Simple: handoff §1 forbids it for a new model (its progress engine
    drops small broadcasts *by tensor size*, job 7565896, and DDP's initial parameter broadcast
    on 1.18 B params is exactly that traffic).
  - **New:** `polaris_ai_rossby_multinode_scaling.pbs` — one script, any node count, carrying
    MASTER provisioning, the corrected fabric block, the `TARGET_NODES`+spare GPU preflight,
    `--cpu-bind depth -d 8`, `OMP_NUM_THREADS=1`, `TMPDIR=/tmp`, write-side isolation from the
    production run (all `realpath -m`-normalised, because `/eagle` is a symlink), and an
    **exact** wrap guard: the condition is `steps_per_epoch < STEPS`, and `steps_per_epoch` is
    the loader's own per-rank count after sharding — the real sample count, not a proxy.
  - **Parser + 24 tests green with no allocation** (`AI_ROSSBY_SCALING_PARSE_OK`). Guards kept
    from makani, grammar replaced. Two are new and earn their keep: `world_size` is read
    **only** off `train.py`'s own per-rank banner (a launcher that echoed `world_size=16` in
    its header would otherwise satisfy, by construction, the guard that exists to catch N
    independent world_size=1 trainers — and an absent banner is an ERROR, not a silent pass);
    and `ranks_reporting` counts distinct PALS labels, because world_size can be right while
    four ranks are missing. Timings come from the **epoch-telemetry CSV**, an existing
    cross-project contract, not from its printed line (rounded to 0.1 ms, no p90/mean/std/wall).
  - **Store decided, on the handoff's own criterion:** `$AI_ROSSBY_DATA/e3sm`, the **per-year**
    lineage (30 train stores 2015–2044), read off the production run's
    `.hydra/config.yaml` and cross-checked against its log (`steps_per_epoch=10950` ×
    world 4 = **43,800 samples**). Largest ladder arm needs 1,920 — 22× headroom.
  - **Three places the handoff contradicts the repo** (its own instruction: trust the repo,
    log it) — detail in `ai_rossby_multinode_ddp_plan.md` §6: (a) the store verifier it names
    (`polaris_verify_store.pbs`, `SEQZARR_VERIFIED`) is for the **SeqZarr** lineage, not this
    one; the matching verifier is `ai_rossby_variable_contract.py --check-artifacts`, already
    wired as PREFLIGHT 2; (b) the `module load conda` breakage above is absent from its asset
    table; (c) "rerun the probes per venv" needed an addition — makani's fabric probe tests six
    `/soft` pairings and does **not** contain the self-built v1.21.1 the handoff mandates, so
    the sibling adds combo **G**.
  - ⭐ **FABRIC ANSWER, job 7568561 — `FABRIC_PROBE_OK`, and the AUTO pin DOES carry across
    the NCCL version.** Under **torch 2.10.0+cu129 / NCCL 2.27.5** (makani measured
    2.8.0 / 2.28.3), exactly two of seven combos work — **E** (`/soft` v1.6.0-libfabric-1.22.0
    + cray 2.3.1) and **G** (self-built **v1.21.1 + `OFI_NCCL_PROGRESS_MODEL=AUTO`** + cray
    2.3.1) — both reporting `Using network AWS Libfabric`. A–D and F fail, three of them with
    `Failed to initialize any NET plugin`. **Same shape as makani's answer on a different
    NCCL**, which is direct evidence the ENOSYS mechanism is CXI-provider-side rather than
    NCCL-version-side, i.e. the ALCF ticket's premise (§4b) holds beyond the env it was found
    in. Single-node, so it proves the domain OPENS, not that inter-node traffic flows —
    that is the ≥2-node probe.
  - ⭐ **KNOB MATRIX, job 7568618 — `OFI_MATRIX_OK working combos: C_progress_auto`, ALONE.**
    Six combos, one winner, and it is the same single winner makani got on NCCL 2.28.3
    (7563894). Everything else fails `fi_domain` with **RC −38 ENOSYS** — default, MANUAL,
    RDMA, RDMA+MANUAL — with 8 domain errors each. ⇒ **`OFI_NCCL_PROGRESS_MODEL=AUTO` is
    NECESSARY here, not merely sufficient**, which is the question the fabric probe could not
    answer and the one that decides whether the pin may be dropped anywhere.
    **PREREG PREDICTION 2 — SCORED, HIT** (both halves: `C_progress_auto` alone, and
    `transport = AWS Libfabric` on every working combo).
    Two independent environments (torch 2.8.0/NCCL 2.28.3 and torch 2.10.0+cu129/NCCL 2.27.5)
    now give the identical answer, so the ALCF ticket §4b can state the ENOSYS root cause as
    **CXI-provider-side and NCCL-version-independent** rather than as one env's observation.
    ⚠ Still single-node: this proves the domain opens, NOT that inter-node traffic flows.
  - ⚠ **CLUSTER FACT, and a correction to the first version of this entry: the limit is ONE
    JOB PER QUEUE in Q/H state, not "no dependency chains".** First reading was wrong. A
    second `debug` submission is refused ("would exceed queue generic's per-user limit of jobs
    in 'Q' state") — but a `-W depend=afterany:` job lands in **`H`**, and a `debug-scaling`
    job in `Q` **plus** a `debug` job in `H` is accepted (measured: 7568641 Q + 7568642 H).
    ⇒ the handoff §0.8 dependency-chain recipe **does work**, one link per queue, and the
    ladder can be driven two-queues-deep rather than strictly one job at a time.
  - **Blocker found and fixed: the SFNO parity gate was failing, so PREFLIGHT 1 of BOTH
    ai-rossby multi-node launchers was dead** — the pre-existing 4-node test as well as the
    new ladder. `compare_sfno_parity.py --static` reported `SFNO_SOURCE_DIVERGED 3/7`, and all
    40 differing lines were **one thing**: the §4.9 **dhconv-XIO layout knob**
    (`PANGU_DHCONV_XIO`, default `"0"`) added to PanguWeather during the profiling work and
    never vendored into ai-rossby's copy. With the knob OFF the selected path is the pre-knob
    one, so the trees compute the same arithmetic and the difference is textual; with it ON
    they are **different models** (dhconv weight `[modes_lat,in,out]` vs `[in,out,modes_lat]`
    — 95.8% of parameters change shape). Hence a **conditional** carve-out, not an entry in
    `ALLOWED_DIFF_LINES`: knob set ⇒ `check_source` returns early with its own token
    `SFNO_XIO_KNOB_SET` and says *different models*, because reporting it as 40 line diffs is
    what would tempt the next reader to widen the allowlist. Listed line by line so it fails
    closed on any edit inside the knob. **Not a loosening** — nothing that is a real numerical
    difference is accepted now that was rejected before. `compare_sfno_parity_test.py` pins
    both directions (12 checks, `SFNO_PARITY_XIO_TEST_OK`, stdlib only), including the
    invariant that the two sets stay disjoint (`ALLOWED_DIFF_LINES` is consulted first, so an
    overlap would make the conditional half silently dead). `--static` now `SFNO_PARITY_OK`.
  - **Both launcher preflights verified from a login node before any trainer allocation:**
    PREFLIGHT 1 `SFNO_PARITY_OK`; PREFLIGHT 2 `VARIABLE_PARITY_OK 19/19` on
    `train/2015.zarr` in **0.13 s** — so the 30-store loop costs ~4 s, not a reason to trim it.
  - **▶ IN FLIGHT — submitted 2026-08-27 21:09 to run unattended, ANALYSE THESE FIRST:**
    | job | what | queue | PASS token | reads |
    |---|---|---|---|---|
    | ✅ **7568641** | 4-node app-free NCCL probe, v1.21.1+AUTO | `debug-scaling` | **`MN_NCCL_PROBE_OK ranks=16 nodes=4`, 16/16 ranks, `transport: AWS Libfabric`, rc=0** | `physicsnemo_ai_rossby/ai_rossby_nccl_mn_probe.o7568641` |
    | **7568642** | **1-node ladder smoke (arm A)** — held on 7568641 | `debug` | `AI_ROSSBY_MN_SCALING_OK` + a CSV row | `.../ai_rossby_mn_scaling.o7568642` |
    | ⛔ **7568724** | 2-node smoke (arm B) | `debug` | **HUNG — NCCL watchdog, `train_rc=134`, row REFUSED (`csv_rc=4`)** | `.../ai_rossby_mn_scaling.o7568724` |
    - ⭐ **7568641 GREEN — the fabric gate is fully cleared.** `MN_NCCL_PROBE_OK ranks=16
      nodes=4`, 16/16 ranks, `AWS Libfabric`, rc=0, in ~1 min. This is the first evidence that
      inter-node traffic actually **flows** under torch 2.10 — the two 1-node probes only
      proved the CXI domain opens. Note what it passed *without*: the small-broadcast storm,
      the 100 MB DDP-sized broadcast and the checked all-reduce all completed at **the default
      NCCL protocol, no `NCCL_PROTO=Simple` pin** — the workaround makani needed at ≥3 nodes
      on the old plugin, and which handoff §4 says not to port. **All three ai-rossby fabric
      probes are now green and the ladder is unblocked on the transport side.**
    - ✅ **7568642 GREEN — the first ai-rossby scaling row ever.** `AI_ROSSBY_MN_SCALING_OK`,
      1 node / 4 ranks / global batch 4: **step_med 698.9 ms**, p90 700.1, samples/s 1.43 per
      rank (5.72 total), **gpu_busy 97.5%**, peak **25.91 GB**, `AWS Libfabric`,
      `world_sizes_seen=4`, `ranks_reporting=4`, `n_steps=60`, wrap guard 10,950 ≥ 60.
      Every guard fired and passed. (gpu_busy 97.5% already part-scores prereg
      prediction 5 at 1 node: the loss is inside the step window, not the loader.)
    - ⛔ **7568724 HUNG — arm B does not run, and the ladder is BLOCKED.** 2 nodes / 8 ranks.
      All 8 ranks came up correctly (`world_sizes_seen=8`, `ranks_reporting=8`,
      `Using network AWS Libfabric` ×8) and NCCL setup **completed** — `Connected all rings`
      ×8, `Connected all trees` ×8. Last trainer line is
      `stage 0 'default' starting at global_epoch=1` at 21:28:25; then **silence**. Heartbeat
      monitor fired 21:44:06, watchdog terminated all ranks 21:52:06, `rank 2 died from
      signal 6`, `train_rc=134`. ⇒ **the wedge is the FIRST collective after communicator
      setup**, which for DDP is `_sync_params_and_buffers` broadcasting the whole model from
      rank 0. **The guards worked exactly as designed**: `NO_TELEMETRY_ROW` → `csv_rc=4` →
      `ERROR AI_ROSSBY_MN_SCALING_FAILED`, and the CSV row carries blank timings rather than
      a plausible number (contrast CLAUDE.md #14 — `rc` alone would have said nothing useful).
    - ⭐ **THE FABRIC IS EXONERATED ON EVERY AXIS TESTED — three hypotheses raised and all
      three REFUTED by app-free probes.** Recorded as misses, per the method:
      | job | nodes | traffic | result |
      |---|---|---|---|
      | 7569520 | **2** | one **4700 MB** broadcast | ✅ `MN_NCCL_PROBE_OK` |
      | 7569521 | **8** | one **4700 MB** broadcast | ✅ `MN_NCCL_PROBE_OK ranks=32` |
      | 7569540 | **2** | **190 × 25 MB all-reduces** (DDP's real gradient traffic) | ✅ `MN_NCCL_PROBE_OK` |
      - **H1 broadcast size — REFUTED.** 4.7 GB moves fine at the exact node count that hung.
      - **H2 node count — REFUTED, and it never had legs:** the trainer wedged at **2** nodes
        while the probe passed at **4**, i.e. the failing case had FEWER nodes than the
        passing one. Node count was exonerated before the probe ran; 7569521 is coverage.
      - **H3 all-reduce storm — REFUTED.** 190 × 25 MB back-to-back all-reduces, the DDP
        bucket pattern for a 1.18 B-param model (`ddp_bucket_cap_mb: null` ⇒ PyTorch's 25 MB
        default), completes app-free at 2 nodes.
      ⇒ **the transport carries every traffic pattern the trainer needs, at the node count
      that fails. The trainer's own path is implicated.**
    - ⚠ **CORRECTION to the first version of this entry, twice over.** (a) "The fabric gate is
      fully cleared" after 7568641 was **overclaimed**: that probe's broadcast was 100 MB
      against this model's ~4.7 GB, 47× undersized, so it could not have exonerated the size.
      (b) "The wedge is `_sync_params_and_buffers`" was **wrong**. The raw log ORDERING
      settles it: `Connected all rings` at idx 1313-1320 (**before** the stage banner at
      1338), `Connected all trees` at idx 1523-1530 (**after** it), first error at 1531. NCCL
      builds rings for the broadcast — which therefore **succeeded** — and trees for
      **all-reduce**. ⇒ the hang is the **first gradient all-reduce of the backward pass**,
      not DDP setup. Ordering, not narrative, is what decided this.
    - **The precise error is not a plain collective timeout:** *"ProcessGroupNCCL's watchdog
      got stuck for 480 seconds without making progress in monitoring enqueued collectives.
      This typically indicates a NCCL/CUDA API (e.g. CudaEventDestroy) hang blocking the
      watchdog, and could be triggered by another thread holding..."* — the MONITOR thread
      reporting the WATCHDOG thread stuck. Worth keeping distinct from "a collective timed
      out": it admits causes on the CUDA/host side, not only the network.
    - **Superseded hypothesis, kept because the method says to keep misses:**
      SfnoPlasim is **1.18 B params ⇒ ~4.7 GB** in that one broadcast. The app-free probe's
      "DDP-sized" broadcast was **100 MB — 47× too small**, an inherited default sized for
      makani's ~150 M-param model. So `MN_NCCL_PROBE_OK` at 4 nodes says nothing about 4.7 GB.
      Precedent for size dependence on this fabric is already in the ledger: the OLD plugin
      lost broadcasts **by tensor size** (384×58 survived, 384×107 wedged, job 7565896).
      Probe now takes **`-v BCAST_MB=`** (default 100) so this is a single-variable,
      app-free bisect — 100 → 1000 → 2000 → 4700 at 2 nodes — instead of a trainer bring-up
      per data point. **Next job to run.**
    - ⭐ **REPRODUCED, and one more hypothesis killed: job 7569539** (2 nodes, arm-B repro with
      `OMP_THREADS=64` so instrumentation was the ONLY change vs 7568724). Identical
      signature: `stage 0` at 03:25:12, monitor fired 03:40:51, all 8 ranks. **So the hang is
      deterministic, not a transient or a sick node** — which is what makes it worth chasing
      rather than retrying (#12 in spirit: never resubmit without diagnosing).
      **H4 extra process groups — REFUTED.** Every rank reports `PG ID 0 PG GUID
      0(default_pg)`, and a scan of the whole 1,557-line log finds **exactly one** PG:
      `[('0','0','default_pg')]`. ⇒ this is **pure DDP on one communicator**, so the
      multi-communicator ordering hazard of the mistakes ledger (#12, makani's
      `_serialized_sync_params` lesson) **does not apply here** and must not be ported on
      spec. Four hypotheses raised, four refuted.
    - ⭐⭐ **THE DECISIVE NARROWING, and it was free — the per-batch TSV already on disk says
      BOTH hung runs completed ZERO steps.** `bench/per_batch_ar_mn2n_b8_rep1_{7568724,
      7569539}.tsv` contain the header and **nothing else**; the 1-node run's has all 60 rows.
      ⇒ the wedge is in the **FIRST training step**, in its first gradient all-reduce (matching
      `Connected all trees` immediately preceding the silence). I chased four fabric
      hypotheses across five jobs before reading a file that was already sitting there.
      **H5 asymmetric rank-0 write — REFUTED, and by this same file.** The TSV write happens
      at the END of a step, so rank 0 never reached it; the asymmetry never occurred before
      the hang and cannot be the trigger. (7569626 was already in flight testing it — kept as
      a control, expected to hang.) Five hypotheses, five refuted.
    - ⭐⭐⭐ **THE STUCK COLLECTIVE, NAMED AT LAST — job 7569690's flight-recorder dumps.**
      All 8 ranks are byte-identical:
      ```
      seq=14  nccl:broadcast    in=[[95687168]]     state=completed
      seq=15  nccl:broadcast    in=[[1682432]]      state=completed
      seq=16  nccl:all_reduce   in=[[1182108160]]   state=scheduled   <-- never started
      ```
      - **1,182,108,160 elements is EXACTLY `EXPECTED_PARAMS`** in `compare_sfno_parity.py`,
        i.e. the whole model — **4.73 GB in fp32** — and its state is `scheduled`: enqueued
        and never launched.
      - **NOT a rank desync.** Every rank enqueued the SAME collective at the SAME seq with
        the SAME size. That retires the whole "uneven steps / one rank never arrives" family
        (the usual first answer for this error, and the one the community threads push).
      - The 15 preceding broadcasts all `completed` ⇒ DDP setup is fine, again.
    - **H7 DDP bucket count — REFUTED.** 7569690 ran `-v DDP_BUCKET_MB=5000` (one bucket,
      makani's shape, the thing both working reference trainers do) and hung anyway, rc=134.
      Seven hypotheses, seven refuted.
    - 🏁🏁 **ROOT CAUSE FOUND, app-free and general: THIS STACK CANNOT COMPLETE AN INTER-NODE
      `all_reduce` OF ~4.7 GB.** Two independent confirmations:
      - **7569767 — app-free, `ranks OK: 0/8`.** One `all_reduce` of 1,232,076,800 fp32
        elements (4.93 GB) at 2 nodes: `Watchdog caught collective operation timeout:
        WorkNCCL(SeqNum=23, OpType=ALLREDUCE, NumelIn=1232076800, Timeout(ms)=180000)`. **No
        trainer, no DDP, no physicsnemo** — just NCCL. (The earlier run of this same probe,
        7569743, printed `ranks OK: 8/8` because MY correctness check was vacuous; fixed, and
        with a real check it is 0/8. The first version would have let a broken collective
        pass as green.)
      - **7569744 — the trainer at DEFAULT 25 MB buckets hangs on the SAME collective.** All
        8 ranks: `seq=16 nccl:all_reduce numel=[1182108160] state=scheduled`, fp32, 4.728 GB,
        `timeout_ms=600000`, default_pg. Identical to the forced-one-bucket run 7569690 ⇒
        **not an artifact of my `DDP_BUCKET_MB=5000` override**, and the bucket knob is not
        what governs this collective.
      **The discriminating table:**
      | traffic | size | 2 nodes |
      |---|---|---|
      | broadcast | 4.7–4.9 GB | ✅ completes (7569520, 7569521 incl. 8 nodes) |
      | all-reduce | 190 × 25 MB | ✅ completes (7569540) |
      | **all-reduce** | **4.7–4.9 GB** | ❌ **never starts** (7569767 app-free; 7569690, 7569744 in-trainer) |
      ⇒ ai-rossby's gradient reduction IS such an all-reduce (1.18 B params × fp32), so the
      harness cannot train multi-node until the reduction is kept under the threshold or the
      stack is fixed. **This also explains why makani was never affected:** its ~150 M-param
      model reduces ~0.6 GB, an order of magnitude below the boundary.
      **⭐ SHARPENED: IT IS NOT A CLEAN STALL — THE ALL-REDUCE PARTIALLY COMPLETES, WHICH
      MAKES THIS A SILENT-CORRUPTION BUG, NOT ONLY A HANG.** The repaired correctness check
      earned its keep on its first run (7569805, 2000 MB):
      `GRAD_STORM_RESULT first=8.000 last=1.000 want=8 WRONG` — the START of the buffer is
      correctly reduced (8 ranks × 1.0), the END is still at its input value, untouched. And
      it differs BY RANK: `ranks OK: 1/8` (7569805) and `3/8` (7569806). Signature of a
      chunked transfer whose loop ends early or whose offset arithmetic overflows partway.
      ⇒ **had the watchdog timeout not fired, training would have proceeded on gradients
      correct at one end of the tensor and stale at the other.** This belongs in the ALCF
      ticket as a correctness defect, not a performance one.
      **🏁🏁🏁 IT IS THE TREE ALL-REDUCE PATH, NOT SIZE — AND `NCCL_ALGO=Ring` FIXES IT.**
      Same size, same nodes, same everything, one env var apart:
      | job | size | algorithm | result |
      |---|---|---|---|
      | 7569817 | 1000 MB | default (**Tree**) | ❌ `first=8 last=1 WRONG`, 2/8 ranks |
      | **7569818** | **2000 MB** | **`NCCL_ALGO=Ring`** | ✅ `first=8.000 last=8.000 OK`, **8/8**, `MN_NCCL_PROBE_OK` |
      A size that fails on the default algorithm completes CORRECTLY on Ring. Size mattered
      only because NCCL switches Ring→Tree above a threshold; the defect lives in the tree
      path of this plugin/NCCL combination. Consistent with the very first observation of the
      hang, which nobody read correctly at the time: `Connected all rings` preceded the stage
      banner (the broadcasts, which always worked) and `Connected all trees` followed it (the
      all-reduce, which never did).
      **This also explains makani without needing an NCCL-version theory** (operator's point:
      makani reduced ~0.6 GB at 512 ranks for 100 epochs and was fine) — at that size its
      all-reduce would stay on Ring and never enter the broken path. ⚠ The version confound
      is still real and unretired: makani runs NCCL **2.28.3**, ai-rossby **2.27.5**, same
      plugin/libfabric/progress model. Ring-vs-Tree is now the better-evidenced explanation,
      but a cross-venv run at one size would settle it outright.
      **✅✅ THE FIX WORKS END-TO-END. `AI_ROSSBY_MN_SCALING_OK` — job 7569831, 2 nodes,
      8 ranks, `NCCL_ALGO=Ring`: the FIRST ai-rossby multi-node training run that has ever
      completed in this project.** 60/60 steps, `step_med 2082.6 ms`, `gpu_busy 99.1%`,
      peak 25.91 GB, `world_sizes_seen=8`, `ranks_reporting=8`, `transport AWS Libfabric`,
      wrap guard clear, rc=0 — every guard passed and the row is in the CSV.
      Confirmed first at the model's real size app-free (**7569832**: 4700 MB on Ring,
      `first=8.000 last=8.000 OK`, 8/8) — larger than the 4.73 GB the model actually reduces.
      - **PREREG PREDICTION 3 — SCORED, MISS, on the high side (the consequential
        direction I flagged when writing it).** Predicted arm B − arm A in **190–750 ms** on
        the theory that the multi-node penalty is a roughly CONSTANT cost (makani's +375 ms).
        Measured **+1384 ms** (698.9 → 2082.6), ~3.7× the makani constant; weak-scaling
        efficiency **33.6%** vs the predicted 50–70%. The prereg said a high miss "would mean
        the penalty scales with the 8× gradient volume, i.e. bandwidth after all, and that is
        what would cap the climb to prod rungs." That is what happened, and it stands as
        written rather than being reinterpreted after the fact.
      - ⚠ **BUT THOSE TWO NUMBERS MAY NOT SHARE A TABLE YET (rule #4).** Arm A (7568642) and
        arm B (7569831) differ in **three** variables, not one: OMP 64 vs 1, per-batch TSV on
        vs off, and NCCL_ALGO default vs Ring. The +1384 ms is therefore CONFOUNDED, and Ring
        is being *forced*, so arm B may also carry a Ring penalty that Tree would not have.
        **Arm A re-run under the matched config: job 7569856 → `AI_ROSSBY_MN_SCALING_OK`,
        `step_med 699.103 ms`.** The original arm A read **698.868 ms**, so all three
        variables together (OMP 64→1, TSV on→off, algo default→Ring) moved the 1-node step
        time by **0.03%** — the confound is real in principle and NEGLIGIBLE in fact. ⇒ the
        A↔B comparison is valid and **prediction 3's miss stands with matched configs**:
        | arm | step_med | per-rank samples/s |
        |---|---|---|
        | A (1 node, 7569856) | 699.1 ms | 1.430 |
        | B (2 nodes, 7569831) | 2082.6 ms | 0.480 |
        | **penalty / efficiency** | **+1383.5 ms** | **33.6%** |
        Side finding: OMP 64 vs 1 is worth ~nothing at 1 node, consistent with `OMP_THREADS=1`
        having failed to fix the 2-node hang. The `${OMP_NUM_THREADS:-1}` defect remains worth
        fixing for correctness of the record, but it was never a performance factor here.
    - **LADDER LAUNCHED (operator request): A/B/C/8n × 3 INTERLEAVED reps, one config.**
      Driver `physicsnemo_ai_rossby/polaris/run_ai_rossby_ladder.sh`, log at
      `$MEMBER_ROOT/polaris_logs/ai_rossby_ladder.log`. Every arm carries
      **`NCCL_ALGO=Ring`** (required, not tuning — the default tree path corrupts and hangs
      above ~1 GB and this model reduces 4.73 GB), OMP=1, TSV off, 60 steps, one store.
      Interleaved A,B,C,8n,A,B,C,8n,… per rule #16, never batched. C requests
      `select=5 TARGET_NODES=4` and 8n `select=9 TARGET_NODES=8` so a zombie-GPU node is
      pruned rather than killing the arm (mandatory at ≥8, ledger #4).
      Ordering is enforced by PBS `-W depend=afterany`, not by the driver's timing; the driver
      only retries **qsub** (never `qstat`) because Polaris allows one job per queue in Q/H.
      ⚠ **Watch for at 4 and 8 nodes:** (a) 43,800 samples does NOT divide evenly at world 16
      or 32 (2737.5 / 1368.75 per rank) — the sampler pads, but it is a real difference from
      arms A/B; (b) **Ring's latency grows linearly with rank count where Tree's grows
      logarithmically**, so the workaround that unblocked 2 nodes may itself cost throughput
      at the upper rungs. Neither is a reason not to measure; both are reasons not to
      extrapolate from arm B.
      **Threshold on the TREE path** (of interest for the ticket, not for the fix) — lower
      than either boundary I guessed, not 2^32 bytes and not 2^31:
      | all-reduce | elements | bytes | result |
      |---|---|---|---|
      | 4700 / 4200 / 4000 MB | 1.23 G / 1.10 G / 1.05 G | 4.9–4.2 GB | ❌ |
      | **2100 MB** | 550,502,400 | 2.20 GB | ❌ |
      | **2000 MB** | 524,288,000 | **1.95 GiB** | ❌ (partial, first=8 last=1) |
      | 190 × 25 MB | 6.5 M each | 25 MB each | completed, no timeout (⚠ see caveat) |
      Bisecting between 25 MB and 1.95 GiB: **7569817 (1000 MB)**. In parallel, **7569818**
      re-runs the known-failing 2000 MB with **`NCCL_ALGO=Ring`** — the logs show broadcast
      used RINGS and all-reduce used TREES (`Connected all rings` precedes the stage banner,
      `Connected all trees` follows it), so if Ring survives where Tree fails the defect is
      localised to the tree path and Ring is a candidate workaround needing no numerics change.
      ⚠ **CAVEAT I MUST CARRY: the 190 × 25 MB result (7569540) was measured with the VACUOUS
      check.** What survives is that it did not hang (rc=0, no watchdog timeout); its
      NUMERICAL correctness was never verified and is now an open question of my own making.
      Re-run it before treating "small all-reduces are fine" as established.
      **Also open:** which code issues the single full-model all-reduce. `bucket_cap_mb` at 25
      MB (7569744) and 5000 MB (7569690) produced BYTE-IDENTICAL stuck collectives, so either
      the knob never reaches DDP's reducer or this is not DDP's gradient reduction at all —
      and zero training steps completed, which argues for the latter. `TORCH_DISTRIBUTED_DEBUG
      =DETAIL` prints the reducer's bucket layout with no code change and separates the two.
    - **Superseded framing — H8, the one shape never tested: an all-reduce above 2^32 bytes.**
      4.73 GB > 4.29 GB. The app-free probe covered a >4 GB **broadcast** (passed, 7569520/1)
      and **small** all-reduces (passed, 7569540) — it never covered a **>4 GB all-reduce**,
      which is precisely the stuck op. ⚠ Caveat kept deliberately: 7569690 is the run where
      **I forced one bucket**, so this giant all-reduce is partly my own doing; the
      default-bucket runs (~190 × 25 MB) hung too and have no dump. Both gaps now queued —
      **7569743** (app-free, ONE 4700 MB all-reduce) and **7569744** (trainer at DEFAULT
      buckets, with dump capture working).
    - **Superseded hypothesis — DDP bucket count, which the codebase itself pointed at:**
      `train.py:918-921`'s own comment records that **both** working reference trainers
      consolidate buckets: **makani uses ONE bucket** sized to the whole local parameter set,
      and the **PanguWeather-e3sm reference uses `bucket_cap_mb=250`**. This config leaves
      `ddp_bucket_cap_mb: null` ⇒ PyTorch's 25 MB default ⇒ **~190 buckets** for 1.18 B
      params, each fired from inside autograd as gradients become ready and **overlapped with
      backward compute**. That last clause is why probe 7569540 passed while the trainer
      hangs on the same pattern: the probe issued its 190 all-reduces **serially from the main
      thread with no compute overlapping**, which is a materially easier case.
      Test ready, `-v DDP_BUCKET_MB=250` (and `=5000` for the makani one-bucket shape);
      **could not be submitted yet — both queue slots occupied.**
    - ⚠ **Trade-off I introduced, worth knowing before queueing many arms:**
      `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800` bought the diagnosis but **cost fail-fast** — a
      hung arm no longer aborts at ~8 min, it sits until the 50-minute walltime. Fine for
      single-variable debugging, wrong for a ladder. Revert it once the hang is understood.
    - 🐛 **MY INSTRUMENTATION WAS INCOMPLETE — the flight recorder captured nothing readable,
      and it took a wasted job to find out.** `TORCH_NCCL_TRACE_BUFFER_SIZE` alone is not
      enough; **two** further defaults defeat it, and both had to be overridden:
      1. **The dump is a FILE, not a log line.** `TORCH_NCCL_DEBUG_INFO_TEMP_FILE` defaults to
         `/tmp/nccl_trace_rank_<n>` — and `/tmp` is NODE-LOCAL and cleaned at job end (this
         launcher deliberately sets `TMPDIR=/tmp` for the AF_UNIX path-length reason), so the
         evidence died with the allocation. Now redirected to `${EXP_DIR}/nccl_trace/`.
      2. **The informative message lost a race with the killer.** The text that NAMES the
         stuck collective — `Watchdog caught collective operation timeout: WorkNCCL(SeqNum=..,
         OpType=ALLREDUCE, NumelIn=..)` — comes from the **600 s collective timeout**, but the
         **heartbeat monitor fires at 480 s** and kills the process first, leaving only the
         opaque *"watchdog got stuck"* text. `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800` lets the
         useful message win. Costs nothing: the 600 s collective timeout still aborts the job,
         ~2 min later, with the diagnosis attached.
      Both are now launcher defaults. **Lesson worth keeping: enabling a diagnostic is not the
      same as capturing it** — verify the artifact exists before spending a job on it.
    - **Superseded: instrumentation first added for the retry** — the scaling launcher now exports
      `TORCH_NCCL_TRACE_BUFFER_SIZE=2000` + `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` by default.
      7568724's log named no collective; the flight recorder is what turned makani's 4-node
      hang into "rank 11, last enqueued 84, last completed 83". Ledger #7: **not**
      `CUDA_LAUNCH_BLOCKING=1`, which legitimately deadlocks NCCL.
    - 🐛 **DEFECT, cross-project: `export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"` NEVER
      PINNED ANYTHING.** The idiom reads and writes the same name, and **PBS Pro exports
      `OMP_NUM_THREADS=<ncpus>` (64 on Polaris) into every job** — so the variable is always
      already set, `:-1` is dead code, and the statement reduces to `export
      OMP_NUM_THREADS=64`. Measured: 7568642 logged `omp=64`, and it is **unset** on a login
      node after sourcing `polaris_env.sh` + the ai-rossby env script, so nothing of ours set
      it. **This is not confined to the new launcher — all 30 rows of
      `bench/makani_multinode_scaling.csv` record `omp_threads=64`, the 128-node production
      run included.** Consequence, stated in two halves: **comparability is intact** (the
      value was constant on every row of both ladders), but **the absolute numbers were taken
      under 8× CPU oversubscription** — 64 OpenMP threads on the 8 cores `--cpu-bind depth
      -d 8` reserves, which are the same cores aws-ofi-nccl's progress engine runs on, i.e.
      exactly the resource the bind exists to protect. **No evidence it caused the hang; not
      claimed.** Fixed here by taking the override from a name we own
      (`export OMP_NUM_THREADS="${OMP_THREADS:-1}"`, ledger #11) and echoing the value.
      ⚠ **makani's launcher deliberately NOT changed** — flipping it would make future rows
      incomparable with the existing 30, which is the owner's call, not a drive-by fix.
      ⇒ arm A rep1 (7568642) ran at omp=64 and is therefore a **different config** from
      anything the fixed launcher produces (rule #4): **re-run it before tabling the ladder.**
    - The 2-node arm could not be queued immediately (one job per queue, above). A detached
      one-shot helper retries `qsub` every 15 min, up to 10 tries, always with
      `-W depend=afterany:7568642` so the arms cannot overlap and contend for nodes:
      `physicsnemo_ai_rossby/polaris/submit_2n_smoke_when_slot_frees.sh`, log at
      `$MEMBER_ROOT/polaris_logs/submit_2n_smoke.log` (`SMOKE_2N_QUEUED` = it got in).
      It retries **qsub**, never `qstat` — CLAUDE.md forbids a qstat poll loop on a login
      node. If it gave up, the fallback command is printed in its log.
    Approved scope was three probes + the 1n/2n smokes, **then stop before the ladder**. Two
    probes are done and green; these three close it out.
  - **Reading the results when they land:** PASS is the token, never `rc` (#14). For the
    smokes also check, in the log: `transport` = `AWS Libfabric`, `world_sizes_seen` = 4 then
    8, `ranks_reporting` = the same, `n_steps` = 60, and the wrap-guard line. The parser
    refuses a row that fails any of these, so a written CSV row with `csv_rc=0` already
    carries them — the CSV lives at `$MEMBER_ROOT/bench/ai_rossby_multinode_scaling.csv`.
  - **Open:** ≥2-node NCCL probe → 1n then 2n launcher smokes → the
    ladder. Prereg prediction 3 is the one that matters: makani paid a roughly *constant*
    +375 ms whenever the fabric was touched, so ai-rossby should pay a similar absolute
    penalty on a ~4× longer step and scale visibly better for no better reason than step
    length. A miss on the high side means the penalty tracks the 8× gradient volume, and
    that is what would cap any climb to prod rungs.

- **2026-08-24** — **The paper's parallel layouts are IN the official repo, and the 512-A100 run
  factorizes exactly — plan §1's inference is now upstream-confirmed fact.** Cloned
  NVIDIA/makani to `${MEMBER_ROOT}/external/makani-upstream` (outside the git tree; the pip
  wheel ships no `config/`).
  - **`config/fourcastnet3.yaml` states the decompositions in comments:** pretrain1 =
    `ensemble 16 × batch 16` + *"requires a model-parallelism of h=2, w=2"* → 16·16·4 = **1024
    H100** ✓; **pretrain2 (the 512-A100 stage)** = `ensemble 2 × batch 32` + *"h=2, w=4 to fit
    into memory on 80GB GPUs"* → 32·2·8 = **512** ✓; finetune = `ensemble 4 × batch 4` × 16-fold
    spatial = **256** ✓. There is deliberately **no "512-GPU config file"** — YAML carries
    batch/ensemble, the spatial split is CLI (`--h_parallel_size/--w_parallel_size`), which is
    exactly the shape our `-v HPAR=/WPAR=` harness already has.
  - **Deterministic SFNO at scale, per upstream's README:** 256 GPUs = `--h_parallel_size=4
    --w_parallel_size=1 --batch_size=64` bf16 → **1 sample per data group, split 4-way spatially**.
    NVIDIA's scale-out lever for our trainer class is spatial parallelism at small per-GPU batch,
    not bigger local batches. (README's named config `sfno_linear_73chq_sc3_layers8_edim384_asgl2`
    no longer exists in `sfnonet.yaml` — the base config, batch 64, is the surviving recipe.)
  - **Operator correction accepted:** the per-rank=1 under-utilisation doc
    (`docs/2026-05-08_ddp_throughput_fix_resolution.md`) is **other-cluster H100 evidence**, not
    Polaris — and upstream's own recipe runs at fractional samples/GPU. The planned
    `LOCAL_BATCH` arm is dropped as the next move; the paper-faithful axis is `h/w`.
  - **Ensemble parallelism needs `ensemble.py`** (stochastic path); upstream's deterministic
    `train.py` inits comm without a data split, same as our 0.2.0. ⚠ Upstream renamed the
    model-parallel axes `[h,w,fin,fout]` → `[h,w,matmul]` — a venv upgrade is API drift and
    must not be done casually against the 7253465 comparability baseline.
  - **Submitted 7554129 — arm F: 4 nodes, `HPAR=2,WPAR=2`, scaling pack.** One job, three
    questions: (a) the paper's pretrain1 spatial layout, first exercise of the spatial path in
    this repo; (b) the arm-C deadlock hypothesis — h2w2 makes the currently-degenerate
    spatial/model groups real, so if the setup BROADCAST deadlock is a degenerate-group
    problem, this run dodges it; (c) whether spatial parallelism works at all on our
    E3SM-shaped SFNO.

- **2026-08-24 (cont.)** — ⭐ **4-NODE DEADLOCK DIAGNOSED AND FIXED: `NCCL_PROTO=Simple`. Arm C
  is GREEN (7554216, step_ms 199.5) and the full A/B/C ladder is being re-run under the pin.**
  The diagnosis chain, because each refuted hypothesis is a fact about this stack:
  - **7554129 (arm F, h2w2) HUNG the same way** ⇒ degenerate-group hypothesis **refuted** —
    making the spatial groups real changes nothing. (Also: first-ever exercise of makani's
    spatial path here reached setup; whether it *trains* is still unknown.)
  - **7554143 (arm C + `NCCL_ALGO=Ring`) HUNG** ⇒ tree-algorithm hypothesis **refuted**. But the
    newly-enabled flight recorder (`TORCH_NCCL_TRACE_BUFFER_SIZE=2000`) produced the first
    stack: torch DDP `_sync_params_and_buffers` → the initial weight broadcast on the DEFAULT
    PG — the same broadcast that succeeds at 2 nodes.
  - **7554170 `MN_NCCL_PROBE_OK` — the fabric passes APP-FREE at 16 ranks / 4 nodes** on the
    same chassis that hung (new `polaris_makani_nccl_mn_probe.pbs`: 20-broadcast storm +
    100 MB DDP-shaped broadcast + verified all-reduce, 3-min fail-fast timeout). ⇒ not a raw
    fabric failure; the trainer's *usage pattern* is implicated. This probe is now the cheap
    vehicle for fabric knob tests — minutes per data point instead of a trainer bring-up.
  - **7554185 (arm C + `FI_CXI_RX_MATCH_MODE=software`) HUNG** ⇒ CXI match-entry exhaustion
    **refuted**. Flight recorder: this time makani's own `sync_params` (`mpu/helpers.py:84`,
    ~86 per-parameter broadcasts on PG 5) — *earlier* than the Ring run's wedge point. Pattern
    across all four hangs: setup broadcast storms of **small (384-float, LL-eligible)**
    messages, wedging nondeterministically partway (18–23 of ~86 complete), on varying PGs.
  - **7554216 (arm C + `NCCL_PROTO=Simple`) → `MAKANI_MN_SCALING_OK`.** Root cause: the
    **LL/LL128 small-message protocol paths of the 2024-era `v1.6.0-libfabric-1.22.0` plugin
    wedge against NCCL 2.28.3 at ≥3 nodes** — the classic aws-ofi-nccl version-mismatch
    failure. Not guessable from any single hang; the probe + flight-recorder narrowing bought
    it in 5 jobs.
  - **Pin landed in the launcher** (default `NCCL_PROTO=Simple`, overridable, printed in every
    log header — a launcher pin, not a CSV column, per #10). ⚠️ **The pin is itself a
    measurement config:** arm A reads **114.9 ms under default proto** vs **144.7 ms under
    Simple** (7554222) — ~26% at 1 node, where LL actually works and helps. So pre-pin rows
    (7553890 A, 7553897 B) are **protocol-confounded vs any 4-node row** and both were re-run
    under the pin.
  - ⭐ **Rep-1 ladder complete under the pin — and the pin REWRITES the scaling story:**
    | arm | job | nodes | step_ms | vs A | samples/s total | weak-scaling eff | wireup_s |
    |---|---|---|---|---|---|---|---|
    | A | 7554222 | 1 | 144.7 | — | 27.6 | 100% | 2.76 |
    | B | 7554241 | 2 | **145.7** | **+0.7%** | 54.9 | 99.3% | 8.54 |
    | C | 7554216 | 4 | **199.5** | **+37.9%** | 80.2 | 72.5% | 21.38 |
    Under default proto the story read "first Slingshot hop costs +52%" (114.9 → 174.6). Under
    the uniform Simple pin **the first hop is free (+0.7%) and the cost concentrates at 4
    nodes** — i.e. the +52% was substantially a protocol artifact (LL helping the 1-node arm,
    unavailable-or-neutral at 2). Wireup grows ~linearly in nodes (2.8 → 8.5 → 21.4 s).
    **Prereg scored (rep 1): prediction 1 CONFIRMED at 4 nodes (+38% ≥ 10%) but REFUTED at 2;
    prediction 3 CONFIRMED (7.7× ≥ 2×).** The 2→4 cliff (free → −27% efficiency) is the next
    question: more-peers ring latency vs dragonfly-group crossing vs plugin Simple-path
    degradation — reps 2–3 first, then arm D separates I/O from comms.
  - **Consequence for the ALCF ticket:** the only initializable plugin pairing needs its LL
    paths avoided — one more line of evidence that `/soft/libraries/aws-ofi-nccl` needs a
    rebuild against libfabric 2.3.1 / current NCCL. The app-free probe (7554170 pass, and its
    hang mode under default proto is reproducible in minutes) is the ticket artifact.
  - 🚨 **Arm F under the pin (7554253): the spatial path CRASHES for real — `CUDA illegal
    memory access` on rank 1 during the FIRST model-parallel collectives** (mid P2P connection
    setup on the h/w groups; wireup itself clean: 22.01 s, 16 ranks, AWS Libfabric). CSV
    refused the row (`NO_STEP_TIMING`), correctly.
  - **Arm F DIAGNOSED by a 4-way split, then FIXED in the fork (pending verification):**
    | probe | config | outcome |
    |---|---|---|
    | 7554253 | 4 nodes h2w2, Simple + GDR on | **IMA** in NCCL `transport.cc` during setup |
    | 7554283 | + `CUDA_LAUNCH_BLOCKING=1` | silent CPU-side wedge, no stack (expected — blocking launch legitimately deadlocks NCCL); qdel'd |
    | 7554351 | **1 node** h2w2 (NVLink only) | ✅ **TRAINS** — 60 steps complete ⇒ makani's spatial kernels + torch_harmonics 0.9.2a on our grid are EXONERATED |
    | 7554367 | 4 nodes h2w2, Simple + **GDR off** (`NCCL_NET_GDR_LEVEL=LOC`, launcher's hard-coded PHB made overridable) | **hang** in `sync_params` broadcasts (31/88 on PG 5) — same wedge shape as the LL deadlock, different knob |
    ⇒ works on NVLink, fails over the fabric under every knob ⇒ not a kernel bug, a
    **collective-ordering bug**. Reading the code closed it:
    `makani/mpu/helpers.py:82-90` — ``sync_params`` **interleaves per-parameter broadcasts
    across DIFFERENT communicators** (``data`` first, then each ``param.is_shared_mp`` group).
    Pure DDP degenerates to one communicator (model groups size 1 → skipped) → fine, which is
    why arms A–C never saw it. With h2w2 multi-node, the alternation violates NCCL's
    concurrent-collectives ordering constraint, with rank-dependent lazy connection setup —
    IMA or hang depending on timing. **Upstream knows:** the call site carries
    ``# DEBUG: this also needs to be fixed in NCCL`` (`deterministic_trainer.py:142`).
    Single-node dodges it because every group is NVLink and nothing is lazy.
  - **Fix: `_serialized_sync_params` in `plasim_trainer.py`** — the fork's existing
    patch-installer pattern (4th rebind, alongside the dataloader/wrapper rebinds), NOT a venv
    edit. Byte-identical semantics (same groups, order, roots, complex handling); the only
    delta is `torch.cuda.synchronize()` after each broadcast so no two communicators ever have
    kernels in flight at once. Setup-path only — the training step never runs through it, so it
    is output-neutral and adds seconds at startup.

- **2026-08-26** — **Arm F VERDICT: the serialization patch is correct but NOT sufficient — the
  wedge is a transport-level progress loss in the old plugin, and arm F is now BLOCKED on a
  plugin rebuild, not on anything ours.**
  - **7563723 (4 nodes h2w2, patch active) still hung — but its flight recorder is the
    decisive evidence:** rank 11 wedged with `last enqueued 84, last completed 83` inside
    `_serialized_sync_params`. Under full serialization there is no ordering freedom left:
    **83 broadcasts on that same subgroup communicator succeeded, then one identical 384-float
    broadcast never completed.** Not first-use connection setup (83 worked), not ordering
    (serialized), not LL (Simple pinned), not GDR (both settings tried). ⇒ the
    `v1.6.0-libfabric-1.22.0` plugin **loses progress under sustained small-message load on
    multiple subgroup communicators** at ≥3 nodes. No application-side change can reach that.
  - **7563780 (1 node h2w2, patch active): `MAKANI_MN_SCALING_OK`** — 219.5 ms at global
    batch 1, the repo's **first working spatial-parallel measurement**, and the patch's
    regression smoke: the working case is unharmed. Patch kept as NCCL-contract hygiene
    (upstream's own `# DEBUG: this also needs to be fixed in NCCL` concedes the stock
    interleaving is wrong), explicitly documented as *not* the fix for 4 nodes.
  - **Where this leaves the axes:** pure-DDP scaling (arms A–E) fully working under the Simple
    pin — the A/B/C ladder stands. **Spatial parallelism multi-node is blocked on system
    software**: the fix is a CURRENT aws-ofi-nccl (v1.9+/1.16 era) built against libfabric
    2.3.1 + NCCL 2.28 — either via the ALCF ticket (their `/soft` builds are 2024-vintage and
    the newest predates the current libfabric) or a self-build in a debug job into
    `$MEMBER_ROOT` (feasible: autotools + libfabric headers + CUDA, all present). A modern
    plugin would likely also lift the Simple pin and its ~26% single-node cost.
  - **Untried knobs, noted for completeness, low prior:** `NCCL_CROSS_NIC=0`,
    `NCCL_MAX_NCHANNELS`. Neither addresses a mid-storm progress loss.

- **2026-08-26 (cont.)** — ⭐⭐ **ARM F IS GREEN. Self-built aws-ofi-nccl v1.21.1 +
  `OFI_NCCL_PROGRESS_MODEL=AUTO` unblocks multi-node spatial parallelism — job 7564035,
  `MAKANI_MN_SCALING_OK`, 4 nodes h2w2, 569.9 ms @ global batch 4, world 16.** The identical
  config that survived no workaround on the old plugin trains end-to-end. Chain:
  - **Build (7563854):** v1.21.1 source tarball → `$MEMBER_ROOT/sw/aws-ofi-nccl-1.21.1`
    (autotools vs libfabric 2.3.1/CUDA 12.9/hwloc, ~2 min in a debug job;
    `polaris_build_aws_ofi_nccl.pbs`). Loads (net API v11), ldd clean — **and still fails
    `fi_domain` with the same ENOSYS as every /soft build ≥v1.9.** So staleness was never the
    mechanism.
  - **The actual unlock (7563894, `polaris_ofi_plugin_matrix.pbs`): ONE env var.** Six-combo
    matrix; only **`OFI_NCCL_PROGRESS_MODEL=AUTO`** works. libfabric 2.3.1's CXI provider
    refuses domain creation unless the caller requests **auto progress**; newer plugins default
    to hints CXI rejects, and 2024's v1.6 happened to ask for the right thing. **This is
    almost certainly why ALCF's own 2025-09 plugin rebuilds sit broken on /soft — hand them
    this in the ticket.**
  - **Verification ladder:** 7563925 `MN_NCCL_PROBE_OK` 16 ranks/4 nodes app-free under the
    **DEFAULT protocol** (no Simple pin — the LL deadlock is at least absent app-free); then
    arm F real training, green. One transient en route: node `x3111c0s37b1n0` failed NCCL init
    twice (`CUDA device busy` — zombie GPU state; report to ALCF), dodged by parking a
    throwaway `host=`-pinned holder job on it while arm F took healthy nodes.
  - **Measurement config note:** the plugin is now part of the config. Old-plugin rows
    (A/B/C ladder, Simple pin) and new-plugin rows must never be mixed in one table; the log
    header records the pin (`fabric pin:` lines + `OFI_PLUGIN` override).

- **2026-08-26 (cont. 2)** — **8-node pure DDP GREEN (first 32-rank row ever) + the new-plugin
  ladder re-measured — and it exposes a real TRADE-OFF, not a win.**
  - **Sick-node epidemic + the harness answer.** Three distinct nodes with zombie GPU state
    (`CUDA-capable device(s) is/are busy or unavailable` at init) killed four runs:
    `x3111c0s37b1n0` (**3 strikes** — 7563960, 7563991, then arm B 7564227), `x3201c0s1b1n0`
    (7564075), `x3109c0s1b0n0` (7564123). A fast-failing node returns to the idle pool
    immediately, so the scheduler re-deals it — per-node holder jobs don't scale. **New
    launcher capability:** `-v TARGET_NODES=N` with `-l select=N+1` runs a per-node 4-GPU
    touch, names sick hosts in the log, prunes them, and runs on the first N healthy nodes
    (no TARGET_NODES → prior behaviour). Proven by 7564137: pruned `x3109c0s1b0n0`, ran clean.
    At 100 nodes a bad draw is near-certain, so `select=101,TARGET_NODES=100` becomes standard.
    **Report all three nodes to ALCF.** ⚠ A 2-node arm cannot carry a spare inside `debug`'s
    2-node cap — run arm B as `debug-scaling select=3,TARGET_NODES=2` (7564264 submitted).
  - **New-plugin ladder (v1.21.1 + AUTO + Simple pin), rep 1:**
    | nodes | job | step_ms | per-rank s/s | weak-scaling eff | total s/s | wireup_s |
    |---|---|---|---|---|---|---|
    | 1 | 7564184 | 115.3 | 8.67 | 100% | 34.7 | 3.2 |
    | 4 | 7564185 | 460.5 | 2.17 | 25% | 34.7 | 12.5 |
    | 8 | 7564137 | 545.0 | 1.83 | 21% | 58.7 | 13.4 |
  - **8-node analysis:** the cliff **saturates rather than compounds** (4→8 = +18% after the
    ×4.0 at 4 nodes), wireup flattens (12.5→13.4 s), I/O is exonerated (io_gbs rose 0.88→1.49
    while per-rank throughput fell). But the plateau is bad: per-GPU compute is ~115 ms, so
    **~430 ms ≈ 79% of the 8-node step is exposed comms** — 16 GPUs deliver exactly 4 GPUs'
    throughput at 4 nodes. Signature matches progress-engine CPU starvation: the new plugin's
    forced `FI_PROGRESS_AUTO` runs libfabric's own progress threads, which the
    `--cpu-bind depth -d 8` layout (tuned for the old plugin's manual progress) never
    accounted for. Tuning problem, plausibly recoverable — not an architectural wall.
  - **The trade-off, stated plainly:** old plugin (v1.6+Simple) = fast pure-DDP
    (199.5 ms / 80.2 s/s at 4 nodes) but spatial-parallel broken and 8-node untested; new
    plugin (v1.21.1+AUTO) = everything *works* (incl. spatial: arm F 569.9 ms) and fastest
    single-node (115.3), but inter-node throughput ~2.3× worse. Old-plugin 4 nodes currently
    out-delivers new-plugin 8 nodes. **Cross-plugin rows must never share a table** — arm A is
    now measured at 196.5 (smoke pack) / 114.9 (default proto) / 144.7 (Simple) / 115.3 (new
    plugin): four "arm A"s, four configs, which is the whole point of recording config per row.
  - **Queue geography (verified `qstat -Qf`):** `debug-scaling` max **10 nodes** / 1 h /
    1 job per user. Beyond: `prod` routes ~16→`small`, 25–99→`medium`, **100–496→`large`**.
    Settle all tuning at ≤10 nodes; climb prod with the settled config only.
  - **Open next:** (a) 8-node on the OLD plugin — one cheap job that decides which stack
    carries the pure-DDP study; (b) progress-thread/cpu-bind tuning sweep for the new plugin
    at 4 nodes; (c) reps 2–3 of the chosen ladder; (d) ALCF ticket (ENOSYS root cause +
    `OFI_NCCL_PROGRESS_MODEL=AUTO` fix + 3 sick nodes).
  - **(a) ANSWERED — arm B new-plugin completes its ladder, then the old plugin takes 8 nodes
    decisively:**
    - 7564264 (arm B, new plugin, `debug-scaling select=3,TARGET_NODES=2`): **490.7 ms** ⇒ on
      the new plugin the collapse is at the FIRST hop (×4.3 at 2 nodes, then flat 460–545
      through 8). A large, roughly constant per-step cost whenever the fabric is touched —
      progress-model overhead, not a bandwidth wall.
    - **7564288 — OLD plugin, 8 nodes pure DDP: `MAKANI_MN_SCALING_OK`, step_ms 215.3,**
      148.6 samples/s total, world 32, preflight 9/9 healthy. **v1.6+Simple survives 32 ranks**
      (its known defects need multi-communicator small-message storms, which pure DDP never
      creates). Completed rep-1 ladder: 144.7 / 145.7 / 199.5 / **215.3** — the 4-node cliff
      **saturates** (4→8 = +8%), efficiency 72.5%→67.2%, wireup sublinear (21.4→25.1 s).
    - ⇒ **STACK DECISION: old plugin + Simple carries the pure-DDP scaling study; the
      self-built v1.21.1+AUTO is reserved for spatial parallelism** until its progress tuning
      is solved (it is 2.5× slower at 8 nodes: 545.0 vs 215.3). If the plateau holds, naive
      100-node weak scaling projects ~65% efficiency ≈ 1,800 samples/s — the *shape* supports
      the climb; reps 2–3 must confirm before any prod-queue rung is bought.

- **2026-08-26 (cont. 3)** — ⭐ **makani now logs the Pangu/ai-rossby wandb metrics contract —
  built, smoke-run, and verified from the offline datastore (7564401) before any production
  launch.** Contract: `wandb_metrics_report.md` + `ai_rossby_finegrained_wandb_handoff.md` —
  per-variable(-level) lat-weighted RMSE in physical units, every iteration, flat keys.
  - **What "same metrics" means here, precisely:** makani-E3SM trains a different channel
    contract (53 ch, 10 plev levels) than the Pangu/ai-rossby pair (101 ch, 18 hybrid levels),
    so identical keys are impossible where variables differ. What is matched is the **scheme**
    — Pangu's exact formula (`N·cos(lat)/Σ` weights, lat/lon-only reduction, `×std`
    denormalization), cadence (every iteration), flat naming (`train_{var}_lwrmse`,
    `train_{var}_level{level:.4f}_lwrmse`), and the same project (`pedramh-profiling`) — plus
    Pangu's base names where the physics is shared: channels `RH{p}`→`RELHUM`, `Z{p}`→`Z3`.
  - **Mechanics** (`sfno_training/trainer/wandb_diagnostics.py` + 3 wiring points in
    `plasim_trainer.py`): a **forward hook on `trainer.loss_obj`** — the one per-step site
    holding the normalized `(pred, tar)` (deterministic_trainer.py:493). A hook, not a
    wrapper, because `loss_obj` is checkpointed (lines 252/281/399) and a wrapper would change
    state-dict keys. Gated `module.training ∧ torch.is_grad_enabled()` (validation never
    fires) and installed only under `log_to_wandb` — which every bench render forces False, so
    **the scaling-CSV path executes zero new instructions** (#10). fp32 math under
    `autocast(enabled=False)`; shape/std-length mismatches disable LOUDLY, never mislabel.
  - **wandb step-axis conflict handled:** makani natively logs epochs with `step=self.epoch`,
    which wandb drops as out-of-order once per-iteration auto-steps run ahead. The existing
    `log_epoch` override now re-logs everything flat (`train_loss_epoch`, `valid_loss`,
    `learning_rate`, `epoch`, + all native scalars) — verified present in the datastore.
  - **Launcher:** `-v WANDB=1` renders `log_to_wandb True` + the four `wandb_*` identity keys
    `Driver._init_wandb` hard-requires (project pinned to `pedramh-profiling`). ⚠ A wandb'd
    row's `step_ms` includes the diagnostics overhead — never table it against non-wandb rows.
  - **Verification (handoff §5, all passed):** smoke 7564401 green; datastore readback via
    `wandb_local_history.py`: **53/53 keys, 60/60 iterations**, physical magnitudes sane
    (TREFHT 17→5 K, PS 7860→1949 Pa, Z3 O(100 m), RELHUM O(10 %)) — i.e. no missed-`std` bug;
    `train_loss` ×60 flat; `valid_loss` == native "validation loss". Handoff §3's
    lat-weighting equivalence: makani reads the pack's real lat grid (the same 1° E3SM grid
    Pangu's `params.lat` carries) and uses Pangu's formula verbatim.
  - Key-scheme unit test run pre-submit (53 keys, no dupes, exact `.4f` formatting).
    **Production note:** launch with `WANDB=1` (or set `log_to_wandb`+identity keys in the
    production config); without it the contract silently stays off.

- **2026-08-26 (cont. 4)** — **"Does makani need the Pangu/ai-rossby variable regeneration?"
  — AUDITED against the corrected artifacts: NO, and now proven numerically, not just
  argued structurally. `MAKANI_PACK_AUDIT_OK`, all 53 channels.**
  - **What "most current data" turned out to be:** the regeneration did NOT produce new source
    variables. `${E3SM_ROOT}/h5/plev_data` is unchanged since 2026-07-08; what regenerated
    (2026-08-05, in `$PANGU_AUX`) is the **normalization**: `moments_2015-2050.json` +
    `data_2015-2050_{mean,std_corr}.nc`, computed **mask-aware** (moments over valid pixels
    only — SST/ICE over ocean, TSOI/SOILWATER over land; per-var shifts for stability), with
    the pre-fix files preserved under `pre_fix/`. ⇒ makani's packs, built from the same
    archive, are already "the most current data" at the variable level.
  - **The audit** (`polaris/audit_makani_pack_vs_reference.py`, new, reusable): every channel
    makani trains on, compared against stats reconstructed from the corrected moments —
    surface/diagnostic direct, upper-air by mapping makani's 10 plev channels onto the LAST 10
    of the reference's 18 hybrid levels (~200..1000 hPa; verified the level lists coincide).
    **Result: all 53 stds within 2.14%** (2-year pack vs 35-year reference window — pure
    sampling drift), **all means within 0.02 std**. Near-zero-mean wind channels show huge
    *percent* mean deltas (U1000 "−970%") that are 0.01σ absolute — the script therefore
    scores means in std units, not percent, so nobody re-trips over that.
  - **Why makani was immune** (2026-08-04 audit, now confirmed against the fix): the defect
    class needs an externally-precomputed stats file that can disagree with a
    separately-configured fill. makani computes stats **in-stream from the packed data**
    (cannot disagree by construction), already fills SST with **−1.8 °C** (the audit's
    "unifies all three pipelines" option), and **has no TSOI_10CM/SOILWATER_10CM channels at
    all** — the 8.2×-underweighted-loss defect cannot map onto its contract. SST/ICE are
    forcings (inputs, never scored) in makani.
  - **Remaining convention question (jesswan's call, not blocking):** the corrected reference
    is mask-aware for masked fields; makani's forcing stats are filled-field stats. Self-
    consistent and input-only, so not a defect — but if a single cross-pipeline convention is
    mandated (or SST fill moves to 0.0), the response is one constant + a repack (~5–10
    min/yr), then re-run this audit. Same procedure if the SOURCE archive is ever regenerated.
  - ⚠ Standing caveat unchanged: `convert_e3sm_to_makani_alldata.py` (the 154-ch ALLDATA
    path) was never fully audited — finish that before building a production pack with it.

- **2026-08-27 (cont. 2)** — 🏁 **PRODUCTION COMPLETE: `prod128_alldata_v2` (7566145) ran ALL
  100 EPOCHS to `MAKANI_MN_SCALING_OK`, rc=0 — the first full makani-E3SM production training
  ever, at 128 nodes / 512 ranks, on the Pangu/ai-rossby-parity 101-channel contract.**
  - **Numbers:** ~1 h 40 m wall ≈ **~215 node-hours** (vs 774 cap); step_ms 576.5 avg
    (incl. warmup; marginal ~440–510), **888 samples/s** total, io 43.1 GB/s aggregate,
    `world_sizes_seen=512`, AWS Libfabric (v1.21.1+AUTO), wireup 315.7 s.
  - **Training result: final train loss 0.0160, validation 0.0183 — and epoch 100 IS the
    validation minimum: still improving at the schedule bound, no overfit onset** (contrast
    the Pangu/ai-rossby runs' plateau-then-drift at ~epoch 25 — though absolute loss values
    are NOT cross-harness comparable; the shared per-channel lwrmse panels are the comparison
    vehicle, which is exactly why the key parity mattered). ⚠ batch-512 convergence quality
    remains jesswan's read.
  - **Artifacts:** `best_ckpt_mp0.tar` (= epoch 100) + last 3 versioned checkpoints (1.77 GB
    each; cap worked); 100 epoch summaries in `prod128_alldata_v2.log`; production CSV row;
    wandb run synced to `pedramh-profiling` with the 102-key contract (102/102 byte-match to
    Pangu, superset of ai-rossby).

- **2026-08-27 (cont.)** — ⚠→✅ **OPERATOR CAUGHT A KEY-SCHEMA MISS ON THE LIVE RUN; fixed,
  byte-verified against BOTH reference datastores, and production restarted clean as
  `prod128_alldata_v2` (job 7566145; v1 7566045 qdel'd ~50 min in, ~110 node-hours).**
  - **The miss:** makani's upper-air wandb keys formatted the archive's ACTUAL hybrid level
    values (`train_RELHUM_level849.6612_lwrmse`) while Pangu/ai-rossby format **NOMINAL
    round-hPa labels** (`train_Z3_level300.0000_lwrmse`) — every one of the 90 upper-air
    panels silently failed to overlay. My assumption ("configs share the literal level list")
    was wrong at the alldata level set; the surface + 53-ch nominal keys had matched, which
    masked it. **Lesson recorded: verify key parity against the reference DATASTORE, not
    against documentation** — assumptions about a contract are not the contract.
  - **The fix + proof:** `_NOMINAL_LEVELS` (5..1000, Pangu's 18 labels; index↔label per the
    archive's level_table) + Pangu's 102nd key `train_mean_norm_lwrmse` (channel-mean
    normalized rmse). Byte-comparison of generated keys vs the REAL datastores:
    **Pangu 102/102 exact, ai-rossby 101/101** (its schema lacks mean_norm; makani is now a
    strict superset = identical where defined). This comparison is the new gate for any
    future key change.
  - v1's local wandb data (hybrid-float keys) remains on eagle + the early-synced snapshot
    run 5lkf6o1c on wandb.ai — mark it superseded; v2 is the run of record.

- **2026-08-27** — 🚀🚀 **PIVOT TO ALLDATA (operator call: full channel parity with
  Pangu/ai-rossby) EXECUTED END-TO-END; 128-node ALLDATA production submitted: job 7566045.**
  7565573 (53-ch) was qdel'd before start — zero node-hours spent on the superseded contract.
  - **The contract:** ALLDATA = the **108-field set all three pipelines agree on** (100 state
    incl. U10/RHREFHT/PSL/TMQ/FSNT/FSNTOA/SOILWATER/TSOI + all 18 hybrid levels of
    T/U/V/Z3/RELHUM, PRECT diag, 7 forcings; clouds excluded per science owner). Converter
    audit (the standing caveat) resolved clean: Pangu's fills verbatim (SST −1.8, TSOI 270,
    SOILWATER 0), honest index-named levels, **resumable second-pass stats**.
  - **~1.4 TB production pack in two 1-h debug jobs** (7565605 pack: 12 parallel workers,
    30/30 train years; 7565734 finish): new `--skip-stats`/`--stats-only` converter flags with
    a completeness guard. `CONVERT_ALLDATA_OK` + (after fixes below)
    **`MAKANI_PACK_AUDIT_OK 101 channels`** vs the corrected Pangu moments — every level of
    every var within 5%/0.05σ; worst = TOA RELHUM −4.6% std; TSOI/SOILWATER informational
    (mask-aware ref vs filled pack, convention not defect).
  - **Two of my own bugs caught by token discipline, not luck** (#14: the finish job printed
    `PACK_ALLDATA_OK` while the audit had CRASHED): (a) `audit | tail` masked the audit's exit
    code — pipe removed, rc checked directly; (b) **`U10` misparsed as upper-air "U at 10 hPa"
    by the nominal-hPa regex** — in the audit script AND in `wandb_diagnostics.build_keys`
    (the 7565972 datastore check showed `train_U_level10.0000_lwrmse` where
    `train_U10_lwrmse` belonged). Fixed both with exact-surface-name-first dispatch;
    unit-tested on both contracts.
  - **⚠ STACK DECISION REVERSED FOR PRODUCTION: the old plugin is DISQUALIFIED, not just
    slower.** The first ALLDATA smoke (7565896, old plugin+Simple) wedged at **SeqNum=2 on a
    41,088-element broadcast = 384×107, the ALLDATA encoder weight** — the known
    small-message progress defect at a message size the 53-ch model (384×58) happened to
    dodge. The plugin's "working" regime is message-size lottery. Production runs on
    **v1.21.1+AUTO** (correct everywhere, measured cost accepted).
  - **8-node ALLDATA FULL smoke on the new plugin (7565972): GREEN.** 492.7 ms/step steady
    (2×1,368 full-pass steps), 512-sample validations, `valid_loss` 0.0323→0.0264, and the
    wandb datastore shows **101 lwrmse keys × 2,736 iterations** with hybrid-level keys
    formatted character-identical to Pangu's (`train_RELHUM_level145.0428_lwrmse`, ...).
  - **RUNNING (started 05:23, ~7 min queue wait; preflight 129/129 healthy). Epoch-1 facts:**
    wireup at 512 ranks = **299 s** (once-per-job; 18 s at 32 ranks); marginal step time
    **~509 ms — statistically identical to the 8-node smoke's 492.7 ms**, i.e. **~96%
    weak-scaling efficiency 8→128 nodes** on the new plugin (its per-step overhead is
    ~constant, so its relative cost shrinks at scale; ~1,000 samples/s at 512 ranks).
    Epoch 1 = 73.6 s incl. warmup; validation 14.5 s; steady-state epoch ~60–65 s ⇒
    **projected completion ≈ 2 h total ≈ 260 node-hours** (walltime cap 6 h). Monitor left
    running: waits for job end, records the tail, syncs the wandb run so the 101 panels are
    live in `pedramh-profiling` alongside Pangu/ai-rossby.
  - **7566045 = `prod`→`large`, 129 nodes (spare+preflight), 6 h walltime,
    `RUN_NUM=prod128_alldata_v1`** (pinned ⇒ requeue auto-resumes, same wandb run), FULL=1
    ⇒ 43,520-sample full-pass epochs (85 steps @ global batch 512) × ≤100 epochs,
    EVAL_SAMPLES=512, WANDB on, rows → `makani_production.csv`. Estimate **~3.5–4.5 h ≈
    450–580 node-hours** (cap 774). ⚠ Batch 512 at shipped LR remains the flagged science
    question — the 101 per-channel panels vs Pangu/ai-rossby are how jesswan judges it.

- **2026-08-26 (cont. 6)** — 🚀 **128-NODE PRODUCTION TRAINING SUBMITTED: job 7565573, `large`
  queue (routed from `prod`), 129 nodes (1 spare + preflight), 6 h walltime.** Every gate
  passed first; nothing here is untested machinery:
  - **Production pack built in ONE 1-h debug job (7564621): `PACK_PRODUCTION_OK`.** Canonical
    split (train 2015–2044 = **43,800 samples**, valid 2045–2047, test 2048–2049) via **7
    parallel chunk workers + exact accumulator merge** — new converter modes
    (`--chunk-accum-out` / `--merge-accums` / `--timestamp-offset-samples`) whose merge is
    bit-for-bit the single-pass arithmetic, with a coverage check refusing gaps/double-counts.
    `CONVERT_OK` (read-back validation) + **`MAKANI_PACK_AUDIT_OK`** vs the corrected Pangu
    reference (PRECT std within 0.08% — the 30-yr window nearly matches the 35-yr reference).
  - **FULL mode added to the launcher and smoked at 8 nodes (7564631, green):** `-v FULL=1`
    makes an epoch the whole train split (trimmed to a global-batch multiple — 43,800→43,520
    at batch 512), `EPOCHS` bounds the schedule, `EVAL_SAMPLES=512` production validation;
    wrap guard correctly bypassed (epochs re-serve by design).
  - **Requeue = resume, verified:** per-epoch versioned checkpoints (measured 1.77 GB each:
    model+optimizer+scheduler+counters) + `best_ckpt`; `-v RUN_NUM=` now pins the experiment
    dir (default embeds the jobid — right for scaling reps, wrong for production) so a
    resubmission auto-resumes, including the same wandb run (`makani_restart.yaml`).
    `checkpoint_num_versions: 3` capped for `EPOCHS>2` (100 uncapped epochs ≈ 177 GB of tars).
  - **The run:** `RUN_NUM=prod128_full_v1`, old-plugin stack (v1.6 + Simple — the fast, proven
    pure-DDP config), global batch 512 (1/GPU), ~85 steps/epoch × ≤100 epochs, W&B on
    (Pangu-parity contract), rows → `makani_production.csv`. Expected ~2–3 h ≈ **300–450
    node-hours** (774 worst case at the 6 h cap). ⚠ Global batch 512 vs the shipped config's
    8 is a training-regime change made on operator instruction — LR left as shipped (no
    unilateral scaling); convergence quality at this batch is jesswan's read, and the wandb
    panels exist precisely so she can make it.
  - Also synced the 4 functional-ladder runs to wandb.ai (`pedramh-profiling` project) at the
    operator's request — links in session log; 8-node run: runs/45275krs.

- **2026-08-26 (cont. 5)** — ✅ **Pre-production functional ladder COMPLETE: 1/2/4/8 nodes ×
  2 epochs × validation × wandb — all four rungs green, all four datastores verified.** The
  "does everything actually work end-to-end before we buy prod time" check the operator asked
  for, run on the ladder-of-record stack (old plugin + Simple).
  - **New launcher knobs:** `-v EPOCHS=` (multi-epoch renders exist for functional checks — a
    multi-epoch step average spans dataloader re-warmup and must never sit in a scaling table)
    and `-v EVAL_SAMPLES=` (default 8). These rows route to a **separate CSV**
    (`makani_wandb_check.csv` via `-v MAKANI_SCALING_CSV=`) so wandb-overhead rows cannot
    contaminate the scaling record.
  - **Results (60 steps/epoch × 2 epochs, 32 eval samples w/ 3-step autoregressive rollout):**
    | rung | job | ranks | wandb iters | valid_loss e1 → e2 |
    |---|---|---|---|---|
    | 1 node | 7564492 | 4 | 120 = 60×2 ✓ | 0.1476 → 0.1027 |
    | 2 nodes | 7564493 | 8 | 120 ✓ | 0.1536 → 0.0981 |
    | 4 nodes | 7564555 | 16 | 120 ✓ | 0.1488 → 0.0973 |
    | 8 nodes | 7564566 | 32 | 120 ✓ | 0.1443 → 0.0941 |
    All 53 lwrmse keys present on every iteration of both epochs at every rank count;
    `valid_loss` ×2 each; epochs 1,2 logged.
  - **What this proves for production:** the train dataloader survives the epoch boundary at
    every world size; the validation dataloader + autoregressive rollout runs per epoch; the
    ReduceLROnPlateau scheduler steps on real validation values; the Pangu-parity wandb
    contract holds across epochs and node counts; and **validation loss genuinely decreases
    epoch-over-epoch at every scale, with consistent values across world sizes** (e2 ≈
    0.094–0.103 everywhere) — the loop is learning, not merely executing. En route, 7564493
    sat through a `queue_tags` no-nodes window and was correctly left alone per rule #12
    (eligible_time preserved; it ran).
  - **Production-launch checklist now reads:** pack the production year range (audited
    converter, ~5–10 min/yr) → `audit_makani_pack_vs_reference.py` on the new pack →
    reps 2–3 of the timing ladder → prod rungs 16/32/64/128 (flag cost per rung) →
    production job with `WANDB=1`, `TARGET_NODES=N` spare-node preflight, old-plugin pin.

- **2026-08-23** — **makani multi-node work merged into the Pangu branch by FILE COPY, and the
  DDP sweep is live: arm A rep 1 = job 7553811.**
  - **The makani worktree is integrated and deleted.** Its 6 commits branched from an ancestor of
    this branch and touched 10 files, of which **only `CHANGELOG.md` overlapped** — everything
    else (8 new `makani_sfno/polaris/*` files, `makani_multinode_ddp_plan.md`,
    `polaris_pbs_notes.md`) was disjoint, so a copy was exact rather than approximate. The
    CHANGELOG block (91 contiguous lines) was spliced in date order: below the §4.1-gate entry
    (Pangu jobs 7551401+), above `SelectBackward0` — makani's 7551240 sits between them.
  - **⚠️ Why not `git merge`: working-tree git commands WEDGE on this repo's Lustre mount.**
    `git status` / `diff-files` (and therefore `merge` / `checkout`) block in `cl_sync_io_wait`
    **indefinitely** — while `log`, `show`, `diff <sha> <sha>`, `ls-files` and `merge-base` all
    return normally, and stat'ing all 9,927 tracked files by hand takes <90 s. So it is neither
    the object DB nor the working tree. Each wedged process **holds a login-node process slot
    permanently**, and enough of them made **every `fork` fail** (`Resource temporarily
    unavailable`) — no shell, no `qsub`, for ~2 h. `pkill -9` does **not** recover it: the signal
    cannot be delivered until the blocked I/O returns. ⇒ **move files between worktrees with
    `cp`; get the file list from `git diff --stat <base> <branch>`, which is safe.** Deleting a
    worktree loses nothing — the commits live in the shared object DB under
    `worktree-makani-multinode-ddp-profiling` (also on `origin`).
  - **`module load conda` re-confirmed BROKEN for the third time** (2026-08-20, -21, -23): same
    `gcc-native/14.2` + `cray-hdf5-parallel/1.14.3.5` dead pins, `which python` → none. The ALCF
    ticket is unfixed 3 days on, so `polaris_makani_env.sh` is the path, not a stopgap.
  - **New cluster fact: the login node's bare `python3` is 3.6.15** and dies on `from __future__
    import annotations`. Repo tooling needs **`/usr/bin/python3.11`** (3.13 also present).
    `MAKANI_SCALING_PARSE_OK 11 tests` re-confirmed green under it.
  - **7553811 — arm A rep 1 FAILED, and the harness behaved exactly as designed:** `rc=139`,
    every rank `NET/OFI Couldn't open a fabric access domain. RC: -38 (ENOSYS)`, rank 2 SIGSEGV.
    No measurement row was written — `ERROR NO_STEP_TIMING` + `csv_rc=4`. A row that would have
    read plausibly was refused, which is the whole point of §6.
  - **⚠️ ROOT CAUSE — a third stale ALCF pin, and this one failed SILENTLY.**
    `polaris_makani_multinode_scaling.pbs` pinned
    `LD_LIBRARY_PATH=/opt/cray/libfabric/**2.2.0rc1**/lib64`, ls-verified 2026-08-14. ALCF has
    since rolled libfabric to **2.3.1 only**. **A nonexistent directory on `LD_LIBRARY_PATH` is
    ignored, not an error** — so the pin no-opped, `aws-ofi-nccl 1.9.1` (built 2024-04-30) bound
    to libfabric 2.3.1, and `fi_domain` on `cxi` returned `ENOSYS`. Joins
    `gcc-native/14.2`→`14` and `cray-hdf5-parallel/1.14.3.5`→`1.14.3.9`: **the PE moved and every
    pinned path in this repo died quietly.**
  - **7553823 `FABRIC_PROBE_OK` — all six installed pairings tested in one job**
    (`polaris_makani_fabric_probe.pbs`, new). Exactly **one works**:
    | plugin | libfabric | result |
    |---|---|---|
    | `v1.9.1-aws` (the repo's pin) | cray 2.3.1 | rc=139 segfault |
    | `v1.9.1-aws-libfabric-1.22.0` | cray 2.3.1 | rc=139 segfault |
    | `v1.9.1-aws` / `v1.6.0` | bundled "1.15.2.0" (really **1.18.2**) | `libjson-c.so.3` absent; OS ships `.so.5` |
    | **`v1.6.0-libfabric-1.22.0`** | **cray 2.3.1** | **rc=0, `ALLREDUCE OK`, `Using network AWS Libfabric`** |
    **The OLDER plugin is the working one.** Both `-libfabric-1.22.0` builds date 2025-09-03, but
    only the v1.6.0 one survives libfabric 2.3.1 — not guessable, hence the probe.
  - **Fix landed + the defect that allowed it.** Pin is now `v1.6.0-libfabric-1.22.0` + 2.3.1,
    overridable via `-v OFI_PLUGIN=` / `-v OFI_LIBFABRIC=`, and **every fabric dir is now
    existence-checked with a hard `exit 3`**. The stale path was not the real bug; the real bug
    was that a load-bearing pin could vanish without stopping the run. **7553824** resubmitted.
  - **Prereg prediction 2 scored — FALSIFIED as worded, capability intact.** It predicted the
    *ported block* would still select `AWS Libfabric` under 2.8.0's NCCL. The ported block does
    not work at all; a *corrected* pin does, and reports `AWS Libfabric`. Recorded as a miss —
    the plan called this "the one most likely to fail for a boring reason", and it was right.
  - **Two more launcher bugs, both found only by running:** (a) `--skip_validation` is
    incompatible with the shipped config — `e3sm_full.yaml` sets `ReduceLROnPlateau` and
    `makani/utils/driver.py:684` raises on the combination, killing 7553824 at trainer
    construction. Fixed by **dropping the flag, not by swapping the scheduler**: validation runs
    after the single timed epoch and contributes nothing to `step_ms`, so this keeps the training
    config the shipped one (`n_eval_samples` trimmed 512→8 to bound the tail). (b) `num_data_workers=8`
    makes torch share storage by **file descriptor**, whose `resource_sharer` opens a Unix socket
    at `$TMPDIR/pymp-XXXXXXXX/listener-XXXXXXXX`; `polaris_env.sh` points TMPDIR at a 54-byte
    eagle path and every worker died on `OSError: AF_UNIX path too long` (sun_path caps at 108).
    Fixed **locally** with `export TMPDIR=/tmp` — that file is shared with every other model, so
    it is not edited to suit this one, and the caches that must persist
    (`TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`) are set separately and untouched.
  - ✅ **7553836 — `MAKANI_MN_SCALING_OK`, the first makani scaling row this repo has ever had.**
    1 node / 4 ranks / batch 4: `step_ms` **196.5**, `wireup_s` 2.77, `transport` **AWS Libfabric**,
    `world_sizes_seen` **4** (i.e. genuinely one 4-rank job, not 4 solo trainers).
  - **⚠️ Then the wrap guard refused arms B and C — and it was RIGHT, for a reason worth stating
    plainly: the pack is a 400-sample SMOKE pack.** `train samples available: 400 ; required: 480`
    (B) and `960` (C). Not 1460 — `2015.h5` was built with `--max-samples-per-year 400`. So the
    blocker on the *deliverable itself* was never fabric, model or launcher; it was data volume.
  - **The operator's correction was right and a prior draft of plan §9 was wrong.** §9 had framed
    the 100-node ceiling as needing ~14 training years as if that were out of reach. **It is the
    same E3SM data in a different format:** `${E3SM_ROOT}/h5/plev_data` holds **51,104 per-sample
    files covering 2015–2049** (1460/yr) — exactly `convert_e3sm_to_makani.py`'s documented input.
    The gap is a **repack**, at ~5–10 min and ~22 GB per year, against 32 T free on eagle. §9 rewritten.
  - ✅ **7553851 `CONVERT_OK` + `PACK_SCALING_OK` — the repack works, and it proves §7's swap.**
    New launcher `polaris_pack_e3sm_scaling.pbs` applies the two-line bootstrap swap §7 specifies
    (rather than editing seven files inside a git subtree), and packed **4 full years in 23m36s**:
    train 2015–2016 = **2,920 samples**, valid 2017, test 2018 → supports STEPS=60 to **12 nodes**.
    Written to a **new root** (`data/e3sm_makani_scaling`); the pack behind 7253465 is untouched.
    New `-v PACK=` knob selects it — `-v MAKANI_DATA=` cannot work, because `polaris_env.sh`
    reassigns that variable via `_pick()` unconditionally and would silently overwrite it.
  - ⚠️ **The pack changes the answer: arm A re-ran at 114.9 ms on the new pack vs 196.5 ms on the
    smoke pack — 41%, from data alone** (`io_gbs` 0.52 → 0.88). Every arm in a comparison must
    therefore share one pack, and arm A's first row is **not** comparable to B/C. Re-run as 7553890.
  - **First real scaling numbers (rep 1, all on `e3sm_makani_scaling`, all `AWS Libfabric`, all
    `world_size` correct):**
    | arm | job | nodes | ranks | batch | step_ms | vs A | wireup_s | vs A | io_gbs |
    |---|---|---|---|---|---|---|---|---|---|
    | A | 7553890 | 1 | 4 | 4 | 114.9 | — | 2.78 | — | 0.88 |
    | B | 7553897 | 2 | 8 | 8 | **174.6** | **+52%** | **17.85** | **6.4×** | 1.16 |
    | C | 7553891 | 4 | 16 | 16 | *queued* | | | | |
    **Prediction 1 (arm C ≥10% above arm A) is already exceeded at 2 nodes** — the single-node
    "comms are free" result (1.2% exposed) does not survive the first Slingshot hop.
    **Prediction 3 (wireup > 2×) is exceeded 3-fold at 2 nodes.** Both remain formally unscored
    until C lands, and **all of this is rep 1 of ≥3** — §4.4c measured 42.2% vs 37.4% for two runs
    of an identical config, so no single-rep number is a result yet.
  - **Arm C waited ~1 h to start** — `comment = Not Running: Job would conflict with reservation
    or top job`, `eligible_time` 00:40+. Backfill contention, **not** the `queue_tags` no-nodes
    case, so it was left to accrue eligibility rather than resubmitted (CLAUDE.md #12).
  - 🚨 **ARM C (4 nodes / 16 ranks) HUNG — new blocker, and the first thing here that 2 nodes did
    not predict.** NCCL watchdog on ranks 1, 2, 9, 13:
    `Watchdog caught collective operation timeout: WorkNCCL(SeqNum=23/24, OpType=BROADCAST,
    NumelIn=384 / 294912, Timeout(ms)=600000) ran for 660091 ms`, with
    `last enqueued work: 88, last completed work: 23` on `PG ID 5`. So ranks enqueued 88
    operations and completed 23: a **deadlocked broadcast during setup**, not a slow one, and it
    burned the 50 min walltime. Arm A (4 ranks, 1 node) and arm B (8 ranks, 2 nodes) are green on
    the identical build, so this appears **only past 2 nodes**.
  - **No arm C row was written, and that is correct** — the walltime kill pre-empted the parser
    entirely, so the CSV has 5 rows and none of them claims to be a 4-node measurement. The two
    failed single-node attempts (7553811, 7553824) do appear, with empty `step_ms` and
    `transport UNKNOWN`/no timing — visibly non-measurements rather than absent history.
  - **Not yet diagnosed. Candidates, cheapest first:** (a) the working plugin is
    **`v1.6.0`**-vintage — it survives libfabric 2.3.1 at 4–8 ranks but may not at 16 across 4
    NICs; (b) `FI_CXI_DEFAULT_CQ_SIZE=131072` and `FI_MR_CACHE_MONITOR=userfaultfd` are inherited
    from the ported block and were never validated at this rank count; (c) a genuine ordering
    deadlock in makani's group setup at 16 ranks (`PG ID 5` is a sub-group, not the world).
    Next: re-run arm C with `NCCL_DEBUG=INFO` retained plus `TORCH_NCCL_TRACE_BUFFER_SIZE` set so
    the stalled collective has a stack, and a 3-node arm to bracket where it starts.
  - **⚠️ The 100-node target is capped by the DATA PACK, not the fabric** — new plan §9.
    `e3sm_makani` is 1 train year / ~1460 samples, so at 100 nodes the global batch is 400 and an
    epoch is **under 4 steps**: the loader wraps and re-serves cached samples. The wrap point
    moves *along the scaling axis*, so it would manufacture a favourable curve; the harness
    rejects such a row, meaning **no 100-node row is obtainable from this pack at all**. ~50
    steps/epoch at batch 400 needs ≈**14 training years**, and the packers that would build it
    are among the seven dead launchers (plan §7). Ladder: arms A–C now → revive packers → build a
    multi-year pack → climb.

2026-08-21 — ⭐ **PanguWeather §4.1 equivalence gate NOW EXISTS, floor = 0.000e+00**
(plan item 18 CLOSED; jobs 7551401/7551411/7551439). Two independent runs of the 1.18 B-param
bf16 model are **bitwise identical** over 20 steps —
`EQUIVALENCE_OK 0.000e+00 <= 2.5e-07 (52 quantities)`, with 20/20 distinct losses so the
model is genuinely training. Capture side is new
(`PanguWeather/v2.0/utils/equivalence.py` + a 3-insertion hook in `train.py`, 16 tests);
comparison reuses `compare_baselines.py` unchanged. Artifact:
`baselines/pangu_sfno_e3sm/{eager,eager_repeat}.json`. **`batch_grad_norm` is recorded but
NOT gated** — a sum-of-squares reduction over 1.18 B gradients that drifted 17/20 steps at
1.255e-06 in one job and 0/20 in two others, while its order-independent control `grad_max`
was bitwise stable throughout; gating on it would impose a ~1.3e-6 *noise* floor and mask a
real 5e-7 change. ⇒ **unblocks §4.9's weight-layout fix, §4.9's zero-pad buffer, §4.11's
activation, and item 19 (`torch.compile`)** — and the bar is absolute: any movement in a
gating quantity is a real change. Not measured: cross-architecture (~1e-5) or multi-rank
reproducibility. → `polaris_bench_report.md` §4.12.

- **2026-08-21** — **`MAKANI_ENV_OK` — job 7551240, 84 s. The env repair WORKS on hardware, and
  makani can span nodes.** The one thing a login node could not settle is settled: with the
  `cuda-13.0.1` path + the `libmpi_gnu_123.so.12` symlink, **`import torch` succeeds** —
  torch 2.8.0 / CUDA 12.9 / **NCCL 2.28.3**, `is_available()=True`, `device_count=4`, a working
  cuBLAS matmul and `RealSHT(180,360)` → `(1,58,90,90) complex64`. Not "caught and survived":
  `ldd` reports **0** dangling libs.
  - **The PALS rank shim works: 4/4 ranks, `world_size=4`, `data_group=4`, all-reduce numerically
    correct.** ⇒ makani is no longer confined to `torch.distributed.run --standalone`, which is the
    single thing that made every green makani run on Polaris single-node-only.
  - Also green in the same job: **h5py 3.16.0 from the overlay** (the base conda's stays broken and
    is now shadowed), `torch_harmonics 0.9.2a` from the venv — *not* the top-ups' 0.7.4 — `makani
    0.2.0`, and **`physicsnemo 2.2.0a0` resolving out of the editable `physicsnemo_sfno/`
    checkout**, i.e. a live demonstration of the coupling CLAUDE.md flags: an edit there changes
    what makani jobs execute.
  - ⚠️ **Single-node.** Prereg prediction 2 — that the ported NCCL/CXI block still selects
    `AWS Libfabric` under 2.8.0's NCCL rather than the 2.10.0+cu129 it was measured on — is
    **untouched by this job** and needs ≥2 nodes. Predictions 1, 3, 4, 5 likewise unscored.
  - ⚠️ **`module load conda` is still broken and the ALCF ticket is still the right fix.** This
    unblocks *makani*, not the cluster: the seven pre-existing makani launchers (**both E3SM
    packers included**) still open with the bare module and still fail, so the full E3SM pack
    cannot be built today and the recorded greens (7253465, `CONVERT_OK` 7252728) are not
    reproducible until the two-line bootstrap swap lands. → `makani_multinode_ddp_plan.md` §7.
  - **Two of my own bugs found and fixed this tick, neither by a test run:** the probe's stage 3
    fed its rank script over **stdin**, which PALS delivers to one rank only (ranks 1-3 would have
    exited without calling `comm.init`, and the job would have hung or "passed" proving nothing) —
    caught while re-reading the launcher, so 7551222 was cancelled *queued* rather than wasted; and
    `makani_env_report`'s libtorch check looked in `sysconfig`'s **purelib**, the venv's
    site-packages, while torch lives in the base conda's — so it printed
    `<could not locate libtorch_global_deps.so>` on every run. A check that silently answers
    nothing is worse than no check. Now `0`.

- **2026-08-21** — **makani multi-node DDP: harness built, and the env blocker is CLEARED for
  makani** (plan item 12, on a different model). Target is the parallel decomposition of
  **FourCastNet 3** (arXiv:2507.12144 §E.2) — makani's *own* paper. No GPU time spent; nothing
  submitted. → `makani_multinode_ddp_plan.md`.
  - **The paper's stage-1 rank budget factorises exactly: 16 batch × 16 ensemble × 4 spatial =
    1024 H100.** All three are the same `makani/utils/comm.py::init` machinery. Of the three we
    can run **batch/data parallelism** (4 nodes × 4 A100 = 16 ranks at 1 sample/GPU = **global
    batch 16**, numerically the paper's value) and, untested, **spatial** (`HPAR`/`WPAR`).
    **Ensemble parallelism we cannot** — FCN3 is probabilistic (`ensemble_trainer`, CRPS) and our
    fork runs the *deterministic* trainer, so that group is size 1. Different model, different
    data, 1/64 the scale ⇒ **scaling behaviour transfers, absolute numbers do not. Not an FCN3
    reproduction**, and the plan says so in the table rather than in a footnote.
  - **`module load conda` re-confirmed BROKEN today** (same `cray-hdf5-parallel/1.14.3.5` +
    `gcc-native/14.2` pins; only `1.14.3.9` and `14` installed). `$SFNO_VENV` has **no torch of
    its own** — `--system-site-packages` inheriting the base conda's 2.8.0 — so makani was as
    blocked as Pangu items 7-17.
  - **⚠️ CORRECTION to `polaris_pbs_notes.md` §1: `libmpi_gnu_123.so.12` is NOT "not present
    anywhere".** That is true of the *filename* and false of the *library*. `_123` was the old
    cray-mpich's spelling of the **gcc-12.3** build, and `mpich/9.1.0` ships that build as
    `libmpi_gnu.so.12` under `.../ofi/gnu/12.3/lib`. SONAME and soversion both stay `12`, so a
    symlink under the former name is a **rename, not an ABI substitution** — and the conda
    modulefile's own last line names that soname in a commented-out PyTorch hotfix. With that
    plus `cuda-13.0.1` for `libcudart.so.13`, **`ldd libtorch_global_deps.so` → 0 unresolved**
    (login-node link check; an *import* still needs the probe job).
  - **h5py is a separate problem and is NOT shimmable:** 1.14.3.5's
    `libhdf5_parallel_gnu_123.so.200` → 1.14.3.9's `...gnu.so.310` is a real soversion bump.
    And it cannot be dodged — `makani/utils/metric.py:19` is a bare `import h5py`, so it is on
    the import path of *every* makani entrypoint **including `--enable_synthetic_data`**.
    Fixed with a minimal PyPI overlay (h5py 3.16.0, vendored `libhdf5-*.so.320`), the same
    `$POLARIS_TOPUPS` pattern, **h5py only** because PYTHONPATH outranks site-packages.
    `MAKANI_H5PY_OVERLAY_OK`.
  - **Why substituting an env is legitimate here but not for Pangu:** the 2026-08-21 entry below
    disqualifies substitution *because every Pangu number was measured on torch 2.8.0*. **makani
    has no prior profile at all**, so that constraint does not bind it. 2.8.0 was kept anyway —
    for the opposite reason: it is what the green makani run 7253465 used, so the 1-node arm stays
    comparable to the only makani evidence that exists. Every CSV row records `torch` and
    `env_source`.
  - **Shipped:** `polaris_makani_env.sh` (tries `module load conda` FIRST, reports which path it
    took, so it self-heals when ALCF fixes the modulefile), `polaris_setup_makani_h5py_overlay.sh`,
    `polaris_rank_env.sh` (PALS `PMI_*` → `RANK/WORLD_SIZE/LOCAL_RANK`; applies to makani because
    `makani/utils/comm.py` delegates to *physicsnemo's* `DistributedManager`),
    `polaris_makani_env_probe.pbs` (3 stages → `MAKANI_ENV_OK`),
    `polaris_makani_multinode_scaling.pbs` (**one file for 1/2/4 nodes** — `NNODES` from
    `$PBS_NODEFILE`), `parse_makani_scaling.py` + **11 passing tests**
    (`MAKANI_SCALING_PARSE_OK`).
  - **Prereg recorded before any job** (plan §4, 5 falsifiable predictions). The load-bearing one:
    arm C (4 nodes) `step_ms` ≥ **10%** above arm A (1 node) — i.e. §0b's single-node "comms are
    free" (1.2% exposed) does *not* survive Slingshot. The one most likely to fail for a boring
    reason: that the ported fabric block, whose values were measured on **torch 2.10.0+cu129**,
    still selects `AWS Libfabric` under **2.8.0**'s different bundled NCCL.
  - **Two guards the harness enforces, because both produce plausible numbers that mean nothing:**
    (a) N independent `world_size=1` trainers if the rank shim is missing — pinned to a hard error
    via `PHYSICSNEMO_DISTRIBUTED_INITIALIZATION_METHOD=ENV` *and* rejected by the parser;
    (b) an epoch that **wraps**, re-serving cached samples — and the wrap point moves with the
    global batch, i.e. *along the scaling axis*, so it would bias the result in its own direction.
    Gated on the real sample count read from the pack.
  - **Still needed:** `MAKANI_ENV_OK` from the probe job, then arms A/B/C. **Nothing submitted —
    multi-node and any `qsub` need explicit approval.** Note only smoke-sized makani packs exist
    (`e3sm_makani`: 1 train year, 18 GB, ~1460 samples), which caps `STEPS × global_batch`.

2026-08-21 — **`SelectBackward0`'s 30.8% (§4.10) traced to `ComplexReLU(mode="real")`** —
`activations.py:65-68`, the only `select` in the SFNO path, active by config
(`complex_activation: 'real'`). Activating only the real component costs **three
full-tensor traversals of 66.4 MB**: a `clone` existing solely to preserve the imaginary
half, a strided half-write, and a backward that embeds the gradient in zeros of the full
base shape. Math-preserving alternative listed and **explicitly not measured**.
`mode="real"` itself is jesswan's modelling choice; the candidate does not change it. →
`polaris_bench_report.md` §4.11.

2026-08-21 — ⚠ **item 18 (the PanguWeather equivalence baseline) is a BUILD, not a job —
and it is now the critical path.** The profiling loop has produced **three** math-preserving
optimization candidates (§4.9 weight layout — two lines in-repo; §4.9 zero-pad buffer;
§4.11 activation) and **none can be adopted**, because DESIGN §4 gates every hot-path change
on an equivalence baseline that does not exist for Pangu. The blocker is sharper than
recorded: the capture machinery lives **only** in the ai-rossby subtree (`equivalence.py`,
305 lines, Hydra-driven), and **`PanguWeather/v2.0/train.py` has no baseline hook
whatsoever**. The status-row phrasing that Pangu's §4.0 prerequisites are all met so
"baseline capture is no longer blocked on building anything" is true of the *prerequisites*
(seed knob, VAE hook, tiny config) and **false of the capture script**.
`compare_baselines.py` is reusable as-is.

2026-08-21 — **PanguWeather profiling item 7 (ncu) — CLOSED, prereg 4/4: the copy
kernels are CONTIGUITY-bound, not bandwidth-bound** — job 7550715, single rank, 80
launches, 11 metrics. Store side sits at **exactly** the ideal sectors/request (4.00 /
8.00 / 8.20); load side at or near the hardware maximum of **32.00** — one distinct 32 B
sector per lane. `TensorIterator` reorders iteration to make the *output* contiguous, so a
layout-changing copy pays the entire cost on the **read**. Whether that scatter costs
bandwidth depends on fitting A100's 40 MB L2: the 66/133 MB tensors hit 82% in L2 and pay
~1.0–1.46× DRAM (cost is request-issue latency, SM 6.4%); the **377 MB spectral weight**
hits only 61% and reads **2043 MB to move 377 MB** (3.21× overall). Nothing is saturated —
DRAM 24–51% of peak, SM 5–20%, occupancy 84–90%. ⇒ **the lever is a layout fix, not
"fewer bytes"**; §4.5's "only dominant kernel with no mechanism at all" now has one
(sec/req 31.50 vs an ideal of 4); the middle population is **bimodal (7.19–31.99)**, so the
same kernel at the same geometry is sometimes already perfectly coalesced and the fix is
feasible rather than hypothetical. **GPU counter access on Polaris works** (open risk
retired). Caveats that must travel with the numbers: `--cache-control` defaults to
flushing, so DRAM figures are cold-cache and 3.21× is an **upper bound** — the verdict
rests on sectors/request, which is cache-independent; **`conj` was 0 of 80 launches
sampled**; single rank (DDP deadlocks under ncu kernel replay); n=1. Cost 2 attempts, both
self-inflicted (`--kernel-name-base` defaults to `function`, which strips the template args
the regex needed). → `polaris_bench_report.md` §4.8, journal ticks 17–19.

2026-08-21 — **profiling item 8 is NOT executable on any capture on disk** — checked free
while 7550606 ran. `CUPTI_ACTIVITY_KIND_RUNTIME` *has* a `callchainId` column and it is **0
for all 551,346 rows** (`--cudabacktrace` was never enabled), and `SAMPLING_CALLCHAINS`'
top frames are `_PyEval_EvalFrameDefault`/`method_vectorcall` — CPython internals, not
Python source lines (`--python-sampling` was off). Replacement recipe, all flags verified
present in the Polaris nsys 2025.1.3: `--cudabacktrace=kernel --python-backtrace=cuda
--python-sampling=true`, keeping CPU sampling. **Better** than the plan's `with_stack`: it
reaches the *backward* launches that are 72.9% of the target. Two traps recorded — never
use the `kernel:<ns>` threshold (it is on **host-side API duration**, so it preferentially
samples queue-stalled launches, biasing the population under attribution), and the capture
is **attribution-only**: read *where*, never *how long*.

- **2026-08-21** — **BLOCKED, terminally — and the available workaround is DISQUALIFIED rather than
  merely unattractive: it would silently break comparability with every number in the profile.** No
  GPU time spent this tick.
  - **Environment blocker unchanged** — `module load conda` errors on every modulefile, and that
    install's torch has 2 unresolved libs. Both ALCF-side; broke between **2026-08-07** and
    **2026-08-20**.
  - **I was about to port the proven torch-aware bootstrap to the PanguWeather PBS scripts** — the
    operator had not answered across two ticks but kept re-firing the loop and had said to
    auto-submit, so a third round of asking was the wrong move and I treated it as a routine call.
    **Checking whether it would help stopped it:** the only working environment carries **torch
    2.10.0+cu129**, while the captures — and therefore **§0d, §4.3, §4.4, §4.5** — were all measured
    on **torch 2.8.0**. Items 7-10 exist to refine that *same* picture, and **§4.4a already measured
    that kernel selection is not bit-reproducible even within one torch version** (cuDNN chose a
    different `Conv2dWgrad` tile between two runs of an identical config). ⇒ **the port is rejected
    on analysis, not deferred.** Recorded as a reversal, because two earlier entries offered it as a
    live option.
  - **Why item 6 was legitimately fine on that env and items 7-17 are not:** a *topology* measurement
    is torch-independent — it measures hardware link bandwidth, not model kernels, so any working
    torch yields the same 83 GB/s. The test is whether the quantity depends on the framework's kernel
    choices. Item 6's does not; items 7-10's are entirely that.
  - **⇒ One ask: an ALCF ticket for the base conda**, both halves — the modulefiles pin
    `cray-hdf5-parallel/1.14.3.5` + `gcc-native/14.2` against a PE shipping **1.14.3.9** + **14**,
    **and** `libtorch_global_deps.so` links both `libcudart.so.12` and `.so.13` with only 12 present,
    so `import torch` fails **even with a working module**.
  - **State: 6 of 21 items done** (**1-6** — all of Tier 0 plus the topology); **7-17 blocked**;
    18-20 always listed-not-run. **Zero optimisations attempted; nothing pushed.** Branch
    `profile/pangu-polaris-profiling`, **28 commits**, left for review — a solo session cannot
    self-approve (CLAUDE.md #9). Infra-failure count **4/5**, stopped deliberately before the 5th.

- **2026-08-20** — **Polaris is a full NVLink mesh: `NV4` on every pair, 82.9-83.1 GB/s uniform. Plan item
  6 done — but the real story of the tick is that `module load conda` and the base-conda torch are BOTH
  broken cluster-side, which blocks every other GPU job in this repo.** Job **7533457**, `TOPO_OK`.
  Preregistered at `5063d221` before submission — **5/5 hit**. First submissions of the loop; operator
  narrowed submission authority mid-tick so single-node `debug` jobs are now auto.
  - **The measurement.** 4× A100-SXM4-40GB, 256 MiB unidirectional `copy_`, all 12 off-diagonal cells
    **82.9-83.1 GB/s (spread 0.24%)**; `nvidia-smi topo -m` agrees with `NV4` in every cell. No 2×2 block
    structure (the H100-NVL pair-bridge pattern) and no PCIe-class pair. ⇒ **handoff §4's OPEN topology
    cell closes with a measurement**, and §0b's "comms are free inside a node *when balanced*" now has a
    mechanism instead of an inference.
  - **It validates the §4.4 method fix sharply, which is worth more than the number itself.** The
    *minimum*-NCCL anchor implied **≥79 GB/s** against a measured **83.0** — within **5%**, so on the
    balanced capture DDP's all-reduce runs at essentially link speed. The stall-carrying **mean** would
    have implied ~32 GB/s and "discovered" a PCIe hop that does not exist. Quoting the minimum was the
    right call.
  - **Item 6b's key input, measured twice on two different nodes: the GPU↔NUMA map is REVERSED.**
    GPU0→NUMA **3** (cores 24-31,56-63), GPU1→**2**, GPU2→**1**, GPU3→**0** — from sysfs on job 7531456
    and independently from `nvidia-smi`'s CPU-Affinity column on job 7533457. ⇒ **a naive
    `--cpu-bind depth -d 8` puts local rank 0 on cores 0-7 = NUMA 0, whose GPU is GPU3, so every rank
    lands maximally far from its own GPU.** That is a concrete candidate mechanism for the *undiagnosed*
    host-CPU stall of §4.4e. Recorded in `polaris_pbs_notes.md` §1, which had it as NOT CAPTURED.
  - **🚨 BLOCKER, and it is bigger than this item.** (i) **All `conda/*` modulefiles are stale**: they pin
    `cray-hdf5-parallel/1.14.3.5` and `gcc-native/14.2`, and the live Cray PE ships only **1.14.3.9** and
    **14**, so Lmod errors and `python` never appears. (ii) **The base-conda torch is separately broken**:
    `ldd` on its `libtorch_global_deps.so` shows it links **both** `libcudart.so.12` and `libcudart.so.13`
    (only 12 exists), so `import torch` fails **even with a working module** — no `LD_LIBRARY_PATH` fix
    exists. Both ALCF-side; broke between **2026-08-07** (jobs 7366939/7366940 fine) and today. **Every
    PBS script in this repo uses the plain module bootstrap**, so items 7-17 cannot run: item 6 only got
    through because it needs *torch alone* and could borrow the ai-rossby venv (torch 2.10.0+cu129 with
    bundled `nvidia/*/lib` wheels — `ldd`: 0 unresolved).
  - **Workarounds tried and FAILED, recorded so they are not re-tried:** `module --ignore_cache load
    conda`; the older `conda/2024-04-29` and `conda/2024-10-30-workshop`; and pre-loading the dependency
    versions that do exist (the modulefile pins exact patch versions). **The fix that worked** was found
    from **repo state, not by probing `/soft`** — the venv `bin/python` symlinks the project's own setup
    scripts created.
  - **Cost, stated plainly: 4 attempts, 3 wasted — and one was mine, not the cluster's.** 7531456 (conda
    module), 7533451 (base torch), **7533454 — my error: an edit script aborted on an assertion before
    writing and I submitted the unmodified file in the same command**. Fix adopted: edits and submissions
    are separate steps, and the file is verified changed before any `qsub`. Infra-failure count **3/5**.
  - **Next:** the torch-aware bootstrap is proven and would port to the other PBS scripts, **but most live
    in `git subtree`s**, so mass-editing them is the operator's call. Either (a) port it to the Pangu
    scripts only, or (b) file the ALCF ticket and wait. Until then items **7-17 are blocked**; item **6b**
    now has a specific hypothesis to test rather than "maybe affinity".

- **2026-08-20** — **The largest single line item in the Pangu step is one weight in the wrong layout:
  133.15 ms/rank-step moves a 377 MB spectral weight in four places, and fusing pointwise ops touches
  none of them.** Plan item **3**, from captures already on disk. No GPU time, no queue, no `qsub`.
  Written up as `polaris_bench_report.md` **§4.5**.
  **Preregistered at `45cbd7de` before `--match` was run — 0/4 size predictions hit, and the miss is the
  deliverable: the inventory omitted the weights.**
  - **What the copies move.** For `operator_type: dhconv` the spectral weight is
    `[in_channels, out_channels, modes_lat]` complex = **512×512×180 = 377.49 MB per layer**, and 12
    layers are **1,132,462,080 of 1,182,108,160 params = 95.8%** — the **largest tensor in the model**,
    1.42× the largest activation. `num_blocks: 16` is vestigial and does **not** block-diagonalise it
    (checked, not assumed). Nine kernels covering **~99.6%** of copy time now match an analytic tensor.
  - **The finding: four copies, one root cause.** Splitting the `gridX=184320` population by duration
    and NVTX phase (both captures agree to **0.04 ms**): forward permute **35.60**, its `ckpt3`
    recompute **35.61**, adjoint `conj` **35.93**, and **`grad_w` → DDP bucket 26.01** ms/rank-step.
    So **36 = 2 × `num_layers` + 1 × `num_layers`**. All four follow from one mismatch — stored
    `(in,out,lmax)`, contracted by `einsum("bixy,iox->boxy")` which must permute to `(x,i,o)` for
    `bmm`. ⇒ **`gradient_as_bucket_view=True` is already set** (`train.py:302`), which is precisely
    what makes the *layout* the culprit rather than the DDP config.
  - **⇒ The optimisation picture reorders.** ~71.5 ms is addressable by storing the parameter
    pre-permuted (hot-path **and checkpoint-format** change ⇒ full DESIGN §4 gate, blocked on item 18);
    **35.61 ms exists only because `ckpt3` re-runs the block**, i.e. **47.7%** of what §4.3d calls
    "removable recompute" is weight re-permutation, not activation recompute; 26.01 ms is gradient
    movement. **DESIGN §5 gains rung 1b**, because rung 1 (`torch.compile`) cannot reach any of it —
    and ACE2 already measured `InductorError: KeyError: 'complex64'` on the whole SFNO.
  - **⚠ Scoped, because the headline inverts.** weight:activation = `E/(B·mmax)` = **2.83 at batch 1**
    and **0.71 at batch 4** — the base config's own commented default (`batch_size: 1 #4`) — where the
    weight share of copy time falls to ~19%. "This model *is* its spectral weights" is a batch-1
    statement. `hard_thresholding_fraction < 1` makes it *worse* (weight ∝ thf, activation ∝ thf²).
  - **⚠ OPEN, and it blocks every mechanism claim.** With `factorization: None` (and `YParams.py:20`
    converts `'None'` → `None` for every key), `use_tensorly=False` lands on
    `assert factorization == "ComplexDense"` (`s2convolutions.py:151`) — **that assert must fail, yet
    both jobs ran 40 measured steps.** Until it is resolved we do not know whether the weight is an
    `nn.Parameter` or a `FactorizedTensor`, and the layout argument depends on which. **No measured
    number depends on it.** Free to answer; folded into item 8.
  - **Two FATAL strikes, both mine, both held.** (i) I attributed the 3 copies/layer to
    `spectral_layers: 3`. **It never reaches this module** — `filter_type: 'linear'` builds
    `SpectralConvS2`, which is not passed it and does not accept it; the agreement with 3 was
    numerology. (ii) **26.01 ms of the 133 is the GRADIENT, not an invariant weight** — a distinct
    operation: 27% faster at identical grid, `backward`-only, and each call followed by an
    `ncclDevKernel` at a **0.871 ms median gap with a 9 µs p10–p90 spread**. A 377 MB parameter
    exceeds DDP's 25 MB bucket cap, so each gets its own bucket (15 all-reduces + 1 broadcast = the 16
    NCCL kernels/rank-step §4.2 already recorded).
  - **Also withdrawn** (§4.5d lists all seven): the `view_as_complex` explanation (a fresh
    `nn.Parameter(…,2)` is contiguous, so that view is **free** — the *permutation* is the strided
    thing); "2.8× the largest activation" (that is 2.83× the *spectral* activation; vs the MLP hidden
    it is **1.42×**); `C_in = 108` (it is **105** — the parameter total forces `2·in + out = 311`);
    and "22.0% of the step" quoted on a single basis when the report's own bases give **20.7–23.1%**.
  - **Two tool bugs, both found by review, both now tested.** Elements-per-block is **launch-path
    dependent**: reading `vectorized_elementwise_kernel<(int)vec,…>`'s leading `(int)` as `nt`
    under-counted the non-legacy paths by **exactly 4×** — with it fixed, a row published as
    "1.185×, no clean tensor" is the **fp32 latent exactly**. And the channel counts now come from the
    loader and are pinned by a test against the logged parameter total.
  - **Docs corrected in the same commit.** `POLARIS_PROFILING_HANDOFF.md` §4's spectral no-op-copy
    guard was shelved as "sub-1% class" on a **0.18%** figure that counted only the zero-fills and used
    a 74.2%-NCCL denominator; §4.5b sizes the *copy* it removes at **5.3% of copy time ≈ 2.4% of the
    step** ⇒ **re-ranked as one of the cheapest levers.** And the weight finding is **structural, not
    Pangu-specific**: ACE2's SFNO has the same `dhconv` weight class (384×384×180 = **212.34
    MB/layer**, ~93% of its params) and **has never been checked** — its `aten::clone` = 45.3%
    autocast-boundary story now has a competitor that reports as the same op, distinguishable for free
    by shape (26,542,080 = weight vs 12,510,720 = activation).
  - **Next:** plan item **4** (`kernel_census.py`) is the last free item. Tier 0 is then exhausted and
    the first submission request will be item **6** (`gpu_topology_check.py`, `debug`, ~1 min) or
    **6b**. Item **8** gains three free targets, the largest being the **42.7%** row — still the only
    dominant kernel with no mechanism at all.

- **2026-08-20** — **A share of the GPU-kernel total is not a reproducible quantity: the same measurement
  moved 4.77 points between two runs of an identical config while its numerator moved 0.09%.** Plan item
  **2**, from captures already on disk. No GPU time, no queue, no `qsub`. Jobs **7255557** vs **7255503** —
  and they ran on **different nodes** (`x3001c0s19b0n0` / `x3001c0s1b1n0`, disjoint GPU UUIDs), so this is
  node-to-node. Written up as `polaris_bench_report.md` **§4.4**.
  **Preregistered at `952fcb8d` before the second capture was queried — 5/5 predictions hit.**
  - **What moved and what did not.** `direct_copy`+`conj` **271.19 → 270.94 ms/rank-step (−0.09%)**, its
    `backward` share **72.86 → 72.83% (−0.03 pt)**, compute-only **+0.08%** on quiet steps (union **0.27%**
    across 8 devices on 2 nodes). But **NCCL 67.82 → 145.65 ms/rank-step (+114.8%)**, so the denominator
    moved **+12.7%** and the copies read **42.2%** then **37.4%**. ⇒ **quote 271 ms/rank-step, never "42.2%
    of GPU kernel time."** §4.2's category table and every share table in `PROFILING_TABLES.md` now say so.
  - **This refines the repo's own rule.** "A cross-JOB ratio on Polaris is not a measurement" (2026-08-06)
    becomes: **cross-job *compute* comparisons are sound; cross-job comparisons of anything containing NCCL
    are not.** The standing 10.5% node-to-node figure is a *wall-time* spread — it is comms, not compute.
  - **The reproducible stall is CPython generation-2 garbage collection — not NUMA, not affinity.** At the
    **same training iteration** in both captures a rank's `forward_loss` blows out to ~7× its median, and
    CPU sampling names it: **`gc_collect_main` 116 / 88 / 88 samples**, then `visit_reachable`,
    `dict_traverse`; nothing else in either capture exceeds **5**. Thread state `Running`; blocking-syscall
    time **0.6 ms** of 247 ms. A gen-2 collection fires on a deterministic function of **allocation count**,
    which is exactly why it lands on the same iteration on different hardware — **no affinity hypothesis can
    predict an iteration index.** ⇒ **Output-neutral fix to try** (no arithmetic changes, so outside the
    DESIGN §4 gate and **no jesswan sign-off**): `gc.freeze()` after model/optimizer construction, or
    `gc.disable()` around the bench loop. It is a **global** cost — every other rank waits at the
    collective. Plan item **6b(A)**.
  - **A second stall pattern is NOT diagnosed.** On 16 of 7255557's 17 stalled steps **dev0 alone waits
    ~600 ms** while the other three sit at 60–70 ms — dev0 *out of phase with the group*, not one rank
    straggling — with the late work in the **inter-step gap** and `_PyEval_EvalFrameDefault` /
    `__libc_malloc` frames. **Candidate, unestablished:** the Pangu nsys script sets `OMP_NUM_THREADS=8`
    and **no CPU binding**, so 4 unbound ranks put ~32 OpenMP threads plus 4 main threads plus loader
    workers on 32 physical cores. Plan item **6b(B)**.
  - **"Comms are essentially free on Polaris" is conditional on rank balance — plan §0b retitled.** The
    88.7%-overlapped / **1.2%-exposed** result came from the capture with **1 stalled step of 40**. On
    7255557, identical config, exposed NCCL is **4.9–8.8% of span** with **17 of 40** stalled. The
    `NCCL_PROTO`/`NCCL_ALGO` agenda stays deprioritized, but for a sharper reason: **what varies is
    placement, not protocol**, and no protocol change recovers time spent waiting for a late rank.
  - **`broadcast_buffers`: the attribution retraction STANDS, its sizing does NOT.** `ncclBroadcast` went
    **0.70 → 28.90 ms/rank-step (×41)** at an unchanged **160 launches** — same Midway mechanism, so a big
    broadcast number is still never a broadcast *cost*. But my claim that removing it "would only move the
    wait" is **refuted by our own union column**: `forward_loss` self-overlap is **0.0%** (that wait is 100%
    exposed) while `backward`'s NCCL is **83–87% overlapped**, so moving the skew would **hide** most of it.
    Sizing is **OPEN**; the change is still jesswan-gated (BN running statistics).
  - **Two more published numbers corrected.** §0a's "3.5–4.4% idle" counted **kernels only** — memcpy and
    memset are GPU work on the same stream, absent from the 643.2 ms kernel total; counting all GPU work
    idle is **1.4–1.5%**, which makes §0a's conclusion *stronger*. DESIGN §5 rung 1 restated as **68% of
    compute pointwise vs 17% GEMM** — the 4.0× ratio the rung rests on is invariant either way.
  - **An adversarial pass landed 11 strikes, 4 FATAL, and it found the GC cause.** §4.4f records all
    **seven** withdrawals rather than quietly dropping them. Mine, named: **"`deviceId 1` is the straggler
    in both captures" — WITHDRAWN.** It is one event per capture, not a rank property, and my per-rank
    ranking summed the **rooted** broadcast, whose root is ~0 by construction — handing dev0 a spurious
    "late" credit on every step. Also withdrawn: **"same `git_sha`"** (no env file exists for 7255503 —
    **my own prereg said exactly that and the write-up asserted the opposite**), "every launch count
    identical" (cuDNN chose a different wgrad tile: 64 vs 63 distinct kernels), "+20% warmup" (really
    **+6.7%**), "D2D bytes 2129.50 GB" (that is the above-L2 subset; all D2D is **2157.11 GB**), and
    "invisible in the kernel table" (a *waiting* rank's compute inflates ~5% via SM contention with the
    spinning ring kernel).
  - **New cluster facts in `polaris_pbs_notes.md` §1**, including a measured **9.1×** multi-node result
    (`mpiexec --cpu-bind depth -d 8` is mandatory; **4.08 vs 36.93 GB/s** busbw, colleague's job
    **7368993**, and it **fails silently**) that was living only in an **untracked** script comment.
    Single-node binding and the GPU↔NUMA map are recorded as **OPEN**.
  - **Tooling — every §4.4 table now has committed code behind it.** `nvtx_phase_attribution.py` gains
    `--per-rank` (straggler test, rooted collectives excluded) and `--stall-cause` (CPU leaf symbols inside
    elongated windows — this is what found the GC). Three of its own bugs fixed: a regex on `Reduce`
    misclassified `AllReduce` as rooted and silently emptied the ranking; SI and binary units were mixed in
    one table; and a **cursor-reuse** bug made `--stall-cause` return no samples at all, which reads exactly
    like "this capture has no sampling data." All three ship regression tests.
  - **Next:** plan item **3** (analytic bytes-per-step model, free), then **4** (`kernel_census.py`, free).
    Tier 0 is then exhausted and the first submission request will be item **6** (`gpu_topology_check.py`,
    `debug`, ~1 min) or **6b**.

- **2026-08-20** — **Pangu-on-Polaris plan items 1 and 5 are DONE, from captures already on disk: the
  42.2% copy time is `backward` 72.9% / `forward_loss` 27.1%, and warmup 20 was clean.** No GPU time, no
  queue, no `qsub`. Written up as `polaris_bench_report.md` **§4.3** (a new subsection joining §4.1's
  CPU-side ranges to §4.2's GPU kernel time for the first time); tool
  `ACE2_retrain/nvtx_phase_attribution.py` + passing test. Job **7255503**, 40 measured steps × 4 ranks.
  **Preregistered at `985214b5` before the number existed — 3/3 predictions hit.**
  - **The join had to be fixed first, and it had TWO bugs, not one.** (a) `correlationId` is unique per
    **process**, so on a 4-rank capture the bare join cross-products the ranks: **459,088 rows for 354,720
    kernels, +29.4% phantom**. The guard `k.globalPid = (r.globalTid & ~0xFFFFFF)` returns **exactly one row
    per kernel**, nothing orphaned (all rows `launchType = REGULAR`, `graphNodeId IS NULL`). Independently
    reproduces the **+30.8%** measured on ACE2's Midway capture. (b) attribution must be scoped to the
    **process, not the launching thread** — 62,680 of rank 0's 88,680 launches come from `pt_autograd_*`
    while all the house NVTX ranges sit on the main thread. **This is the whole origin of the "57% / 81% of
    GPU time lands outside any range" rows in `PROFILING_TABLES.md` and `ACE2_retrain/bench_midway_notes.md`
    — it was never an NVTX limitation.** Process-scoped, `(outside)` is **0.0%**. Both docs corrected in
    this commit; `ACE2_retrain/kernel_census.py:58` still carries both bugs (plan item 4).
  - **The NVTX text path is settled, and nothing of ours was ever missing.** House ranges are in the inline
    `NVTX_EVENTS.text` column (`domainId 0`, `eventType 59`), 160 rows each; the `textId → StringIds` path
    holds **only** NCCL's registered strings (`domainId 1`). `parse_nsys.py` was already on the right path.
  - **Result — measured.** `backward` **197.58** ms/rank-step (72.9%) / `forward_loss` **73.61** (27.1%) of
    **271.19** ms/rank-step. Union-safe: all 90,240 copy kernels are on one stream, so sum = union.
    `conj` (38.30 ms/rs, 14.1%) fires **only** in `backward`; the warrant is the **source**, not the split —
    there is no `conj` anywhere in `networks/modulus_sfno`, the contraction is `einsum` over
    `view_as_complex`, and it fires 24/rank-step = **2 × `num_layers`**. `ncclAllReduce` is 100% in
    `backward`, `ncclBroadcast` 100% in `forward_loss`, with **zero** cross-boundary leakage.
  - **`(outside)` = 0.0% closes §4.1's open question.** The step's "missing" ~268 ms contains **zero kernel
    launches**; the CPU is blocked in `cudaDeviceSynchronize` (119 calls, 10.72 s on rank 0). It is drain,
    not unaccounted work — measured, not inferred.
  - **What checkpointing is actually worth — ESTIMATED, and smaller than "61% elementwise" suggests.**
    Recompute lives *inside* `backward`, so that is the bucket that shrinks: ≈**74.6 ms/rs** of copy time
    (≈27%) is removable, and `forward_loss`'s 73.61 ms is removable by **no** level. Whole-forward recompute
    ≈**148–152 ms/rs** ⇒ ckpt-off ≈**1.34×**. **Not a bound:** kernels with equal fwd/bwd launch counts (the
    pure-recompute signature) run at **1.0136×, never below 1.0**, and recompute selects *different* GEMMs
    (+15%/call). Three qualifiers: `ckpt0` projects to **~41.7 GB > 40 GB** for Pangu (likely unreachable),
    production already ships **`ckpt2`**, and the levels are **cumulative** so a `ckpt3→ckpt2` delta measures
    **block-minus-MLP**. ⇒ **against the config we actually run, checkpointing headroom is ≈4%, not ≈25%.**
    Cross-check: ai-rossby's full ladder (1.307× = 23.5% of the step) sits within **1.8 points** of the
    estimate — two unrelated measurements agreeing, and very little slack. Prereg written into plan item 10.
  - **A measured HBM ceiling, which narrows plan item 7 without replacing it.** D2D memcpy **above L2** —
    16,000 copies, 98.7% of D2D bytes — sustains **1279 GB/s = 82% of peak** (per device 81.3–82.6%; peak
    and L2 read from `TARGET_INFO_GPU`). ⇒ the hardware plainly delivers, so §0d's *estimated* 17–27% for the
    copy kernels is about **the path those kernels take**, not an unreachable peak. **The `2 × bytes` rule
    only holds above L2:** sub-L2 buckets compute to **124.7% of peak**, which is the proof it fails there —
    quote the above-L2 population only. ⚠ **Intra-device HBM, NOT the interconnect** — it does **not** close
    the OPEN topology cell (handoff §4) or substitute for plan item 6.
  - **Item 5 (warmup) — DONE, and it found a different contamination.** No warmup regime: first measured step
    **640.26 ms** vs **634.36 ms** median (+0.9%). **But step index 30 is a comms stall** — 1222 ms on two
    ranks, **614 ms of NCCL** against a ~59 ms norm at identical launch counts. Excluding it, NCCL
    **67.82 → 59.67 ms/rs (−12.0%)** while phase shares move ≤0.4 points. ⇒ **§4.2's NCCL row carries the
    stall**; any sizing off it should use ~59.7 ms.
  - **Two published numbers in `polaris_bench_report.md` CORRECTED in the same commit.** (i) §4.2's "NCCL
    10.5% ⇒ §5 rung 3 targets ≈5% of the step" — NCCL is **88.7% overlapped / 1.2% exposed** (plan §0b), so
    rung 3's single-node ceiling is **~1.2%, not 5%**; it only becomes interesting multi-node (item 12).
    (ii) **§0a's "3.5–4.4% idle" counted kernels only** — memcpy and memset are GPU work on the *same*
    stream and are absent from the 643.2 ms kernel total; counting all GPU work, busy is **98.5–98.6%** and
    idle **1.4–1.5%**. §0a's conclusion (no idle to reclaim, nothing for CUDA Graphs) gets **stronger**.
    Full data-movement bill: **294.97 ms/rank-step**, not 271.19.
  - **Also RETRACTED as a Polaris item: handoff Tier B 5, "`broadcast_buffers=False` — the largest single
    item found on this project".** It is **0.11%** here (plan §0c) and §4.3c confirms it is 0.7 ms/rank-step,
    100% inside `forward_loss`. Midway's 33.14% was three ranks *waiting* on a straggler. **Do not open the
    jesswan BN-buffer gate for 0.1%.**
  - **An adversarial pass landed 15 strikes on the first draft of §4.3; all folded in.** The two that
    mattered: the removable/non-removable buckets were **inverted** (recompute is inside `backward`; the 27%
    magnitude was right only because recompute ≈ forward), and "backward is 77.4% of the step" was the
    **sum-vs-union confusion that plan §0 already lists as refuted** — the union is 408.63 ms = **67.7%**,
    and only `backward` self-overlaps (12.5%, NCCL on `streamId 19`). §4.3b now prints sum **and** union, and
    the test covers the union arithmetic. A drift auditor found 27 stale/contradicting passages across 8
    docs; the high-risk ones are fixed here.
  - **`parse_nsys.py` could not run on a Polaris login node at all** (commit `99378811`) —
    `sqlite3.connect(PosixPath)` needs Python ≥ 3.7 and the login default is **3.6.15**; `statistics.fmean`
    is 3.8+. Both fixed with regression tests. Also repaired the live **CLAUDE.md #10** drift: `unstack` was
    in the SQL but missing from the print loop, so its rows were fetched and **silently never printed** —
    the two lists are now one `RANGE_NAMES` constant, and the test asserts there is only one.
  - **⚠ FLAGGED, NOT FIXED — needs its own change with both smokes.** `s2s/v2.0/HPC_scripts/parse_nsys.py`
    is a **different copy carrying only 8 of the 19** range names, and it is the one the Polaris PBS scripts
    and `physicsnemo_ai_rossby/polaris/bench_instrumentation_test.py` actually invoke. So when **plan item
    16** adds SFNO-internal ranges, the Polaris analysis path will print **nothing** and look like the
    instrumentation never fired. Fixing it touches the **live-coupled** s2s pair, which this profiling loop
    must not do.
  - **Next:** plan item **2** (re-derive §0d on the second capture, job 7255557, as an n=2 — §4.3 is an
    independent *query path* on the *same* capture, so it does **not** close item 2), then item **3**
    (analytic bytes model) and item **4** (fix `kernel_census.py` by importing the guarded join). Tier 0 is
    still `qsub`-free; the first submission request will be item **6** (`gpu_topology_check.py`, `debug`,
    ~1 min).

- **2026-08-20** — **Live-session cluster loop for the Pangu-on-Polaris profiling plan.** Ported the hardened
  RCC/Midway autonomous-loop pattern (`L2LGWAS_DFE:prompts/_TEMPLATE_cluster_autonomous_loop.md` +
  `scripts/_TEMPLATE_cluster_loop.sbatch`) to Polaris. Three files:
  `prompts/_live_session_pangu_polaris_loop.md` (driver), `prompts/_live_session_loop_README.md` (setup +
  submission policy), `prompts/pangu_polaris_loop_journal.md` (the loop's durable state). Frozen plan =
  `PANGU_POLARIS_PROFILING_PLAN.md`. **Nothing submitted.**
  - **A batch orchestrator does NOT work on Polaris, and this is the design finding.** The Midway pattern runs a
    cheap `--partition=build` orchestrator that submits gate compute as nested `sbatch` jobs. Here `debug` is
    **`max_run 1` AND `max_queued 1` per USER** (a second `qsub` is rejected; a `-W depend=` job is rejected too,
    since a held job still counts) — so a PBS orchestrator on `debug` is its own competitor and cannot submit the
    jobs it exists to submit. `capacity` is **`max_run 1` per PROJECT**, so a nested design would consume
    `lighthouse-uchicago`'s only long slot twice, and `preemptable` started 0/9 jobs in 11.5 h. There is no
    `build`-equivalent queue. ⇒ **the live session is the orchestrator**, which is also exactly what
    `polaris_pbs_notes.md` §1b prescribes for `debug` ("a driver that submits chunk N+1 when chunk N finishes")
    and puts a human in front of every submission.
  - **Guardrails carried verbatim:** branch guard (never `main`, never push/amend), the per-change commit ratchet
    with reset-to-last-green, the **hashed pre-result prereg** (predictions committed before the job runs — the
    pattern that made the ZeRO sweep credible at 6/8 exact), explicit staging (**never `git add -A`** — each
    capture here is >120 MB), the infra-failure ≤5 ratchet, the stage-scoped reference fence, and the
    **completion-honesty rule**.
  - **Submission policy is absolute and not configurable: the loop NEVER submits without explicit
    per-submission approval.** No standing-approval mode exists. Starting the loop is not approval; approving a
    stage or a prediction is not approval; approving one `qsub` does not cover the next one or a resubmit of the
    same script. The loop prepares fully (script, static checks, node-hour arithmetic, prereg) and then stops on
    the literal `qsub` line. Consequence that makes this cheap: **the plan's whole Tier 0 needs no `qsub`** — it
    is re-analysis of captures already on disk, so the loop runs unattended to the end of Tier 0 before asking
    for anything.
  - **Guardrails added for this cluster:** at most **one job in flight** (the `debug` limit); **diagnose before
    ANY resubmit** — a `queue_tags` comment means the queue has no nodes and resubmitting destroys accrued
    `eligible_time` (CLAUDE.md #12, cost a day on 2026-08-05); key on the **PASS token / CSV row, never `rc`**
    (nsys writes a report file even when it captured nothing); science fenced to **jesswan**; one-tree
    `PYTHONPATH`; instrumentation-name contract; caches on `/eagle`.
  - **Three Midway guardrails deliberately dropped, each with its reason recorded** (§R of the driver): the batch
    orchestrator + `USR1` chain, the nested-job wait-loop with the name filter (**PBS truncates job names in
    `qstat`** — persisted job ids are authoritative instead), and the transient-`squeue` retry (a live session
    cannot double-submit while waiting on a human).
  - **Unverified, flagged in the README rather than assumed:** `qsub`-from-a-compute-node has never been
    exercised by this project, so a future headless variant must verify it before relying on a self-chain
    (fallback: `qsub -W depend=afterany:<jobid>` from a login node). PBS also has no `--signal` equivalent — a
    headless port needs a self-timer off `qstat -f $PBS_JOBID` → `Resource_List.walltime`.
  - **Next:** tick 1 = cut `profile/pangu-polaris-profiling` off `fix/tsoi-fill-270`, then plan item 1 (fix the
    NVTX↔kernel join). Tier 0 needs no GPU and no `qsub` at all.

- **2026-08-20** — **Polaris Pangu: 96% GPU-busy but only ~25% of HBM peak — the
  ceiling is our own data movement, not the A100.** Derived from the capture already on
  disk (job **7255503**), **zero GPU time spent**; plan and to-do list in
  **`PANGU_POLARIS_PROFILING_PLAN.md`**.
  - **GPU-busy union = 95.6-96.5%** across the four ranks (span 24.35 s). This
    *confirms* `polaris_bench_report.md` §4.2's saturation verdict, which had inferred
    it from `sum/span = 105%` — an NCCL-on-its-own-stream artifact. 3.5-4.4% idle ⇒ no
    launch-latency story, nothing for CUDA Graphs, on Polaris evidence now.
  - **Comms are NOT the Polaris problem: NCCL is 88.7% overlapped, exposed only 1.2% of
    span** (2.671 s union, 2.369 s overlapped, 0.302 s exposed, dev0). Against Midway's
    35.7% exposed. ⇒ **the `NCCL_PROTO`/`NCCL_ALGO` agenda in
    `POLARIS_PROFILING_HANDOFF.md` §2 has at most 1.2% to win on one node.** Deferred to
    the multi-node work, where it becomes interesting again.
  - **`ncclDevKernel_Broadcast_RING_LL` IS present** — 160 launches = exactly 1 per
    rank-step, confirming `broadcast_buffers=True` (`train.py:298-303`) — **but it is
    112 ms of 102.9 s = 0.11% of GPU kernel time.** Midway's 33.14% was three ranks
    *waiting* on a straggler, not the broadcast itself. ⇒ the handoff's "largest single
    item on the project" **does not transfer**; `broadcast_buffers=False` is a 0.1%
    change here and no longer worth opening the BN-buffer science gate for.
  - **42.2% of all GPU kernel time is `direct_copy` + `conj` — kernels that do zero
    arithmetic**: `direct_copy`(float) **18.9%** (121 ms/rank-step), `direct_copy`
    (complex64) **17.3%** (111 ms), `conj`(complex64) **6.0%** (38 ms). 271 ms of a
    603 ms step. All three take the **non-vectorized** `elementwise_kernel<128,2>` /
    `gpu_kernel_impl_nocast` TensorIterator path.
  - **Estimated 17-27% of A100 HBM2e peak (1555 GB/s) on those three, where a
    *vectorized* bf16 add in the same capture reaches 52%.** ⇒ **maximum GPU *time*
    occupancy and nowhere near maximum *bandwidth*, simultaneously** — which is exactly
    why the profile "looks maxed" and yet is unclear. **OPEN, method-limited:** bytes
    are estimated from CUPTI launch geometry (`grid x block x elts/thread x dtype x 2`),
    i.e. *useful* bytes. If access is uncoalesced, real DRAM traffic is higher and the
    defect is wasted traffic rather than unused bandwidth. Same class of fix, different
    mechanism. **Settle with ncu `dram__bytes_*` + sectors/request, single-rank only**
    (kernel replay deadlocks on `ncclDevKernel`). Do not quote "25% of peak" until then.
  - **The complex64 spectral island is 30.3% of GPU time** (complex64 copy + conj +
    `MulFunctor` + `cutlass_80_tensorop_c1688gemm_64x64_16x4_nt_**align1**`). The
    cutlass kernel's `align1` is unvectorized *loads* — **alignment is not precision**,
    so align4/align8 needs no science sign-off, unlike the deliberate fp32 SHT island.
  - **⚠ The only Polaris Pangu capture is `checkpointing: 3`; production ships
    `checkpointing: 2`** (jobs 7366939→7366940). On the same-model ai-rossby sweep
    `ckpt3 → ckpt2` is 1.274×, and recompute traffic is precisely what dominates the
    42% above. **Every percentage here is a `ckpt3` percentage** — re-capture at the
    production config, warmup >= 40, and long enough for **EMA to be active** (it has
    never fired in any capture in this repo).
  - **The NVTX join needs fixing before any phase attribution**: on this capture the
    naive `NVTX_EVENTS.textId -> StringIds` join surfaces only NCCL's own registered
    ranges (`ncclAllReduce` x600, `ncclBroadcast` x40) — the house ranges do not appear.
    So "how much of the 42% is activation recompute" is not yet answerable.
  - **Biggest structural blocker restated:** `baselines/` holds only
    `ai_rossby_pangu_plasim/` and `ai_rossby_sfno/`. **There is no PanguWeather
    baseline, so the DESIGN §4.1 gate cannot be closed for any Pangu hot-path change** —
    every optimization above is queued behind capturing it.

- **2026-08-19** — **ACE2 on H100/`pedramh-gpu`: `torch.compile` is NOT the lever; three
  bigger levers measured instead.** Detail: `ACE2_retrain/bench_midway_notes.md`.
  - **All runs restricted to `pedramh-gpu`** (midway3-0423, 4x H100). Every script
    retargeted; the two 2-node scripts are **parked** — that partition has one node, so
    `sbatch` now refuses them rather than silently taking nodes elsewhere.
  - **`torch.compile` swept across 7 regional targets. Best result: 2.2%** (the corrector,
    at 2.0e-5 drift). The **whole SFNO cannot be compiled at all** —
    `InductorError: KeyError: 'complex64'`, because the spherical-harmonic path is
    complex-valued. Compiling the **normalizer is actively bad** (no speedup, **8.4e-3**
    drift), and the **MLP blocks are 3.4% slower** (guard overhead beats fusion on small,
    frequently-called blocks). My ~20% estimate was too optimistic: it assumed the fusable
    pointwise work was contiguous, and it is not.
  - **Three larger, safer levers, all measured**: `validation_aggregator.log_snapshots=false`
    is **~30% off the epoch** with no numerics change (the rendered panels are discarded
    when wandb is off); **raising batch size** moves NCCL from 52.1% to 18.6%; and ending
    the `dict<->tensor` round trips (`stacker.py:121`) targets **28% of GPU kernel time**.
  - **CORRECTION to the 2026-08-18 entry**: the 8x H200 NCCL drop was read as proof the
    share "really was the PCIe interconnect". A H100 run at the *same* batch size refutes
    that — NCCL went **up**, 40.6% -> 52.1%. The H200 gain was mostly **batch 16 vs 4**.
    ⇒ the lever on communication is batch size, not interconnect.
  - **Two equivalence-tolerance floors measured** for DESIGN §4: **2.5e-7** same GPU/node,
    **~1e-5** across GPU architectures. A baseline must record which hardware captured it.
  - **Script bug fixed (rule #14)**: a windowed nsys capture was marked FAILED despite
    writing a valid 217 MB report — nsys SIGTERMs the target when `--duration` expires, and
    `set -e` aborted before the PASS gate. Now gated on the artifact.

- **2026-08-18** — **ACE2 runs on Midway and has its first profile.** Detail +
  caveats: `ACE2_retrain/bench_midway_notes.md`. Run on **`--account=rcc-staff -p test`**
  as requested, not the usual `pi-pedramh`/`pedramh-gpu`.
  - **Built the env** — `/project/rcc/mehta5/envs/fme` (torch 2.7.1+cu126, fme 2026.5.1),
    from `ace_exp/Makefile::create_environment` minus `[docs,graphcast]`/healpix. The Delta
    config's env (`/scratch/midway3/krucker01/envs/fme`) is **permission-denied to us**.
  - **Added** `config_midway.yaml` (port of the Delta `config_nsight.yaml`: only paths and
    wandb differ — model/loss/optimizer/variable lists are byte-identical), plus
    `midway_smoke_train.sh` (`ACE2_SMOKE_OK`) and `midway_bench_nsys.sh` (`ACE2_NSYS_OK`),
    beside the untouched Delta `train.sh` (rule #7).
  - **GREEN on 4×A100**: job **53478978** `ACE2_SMOKE_OK` train 35.783 / valid 36.213 (5:55);
    job **53478979** `ACE2_NSYS_OK`, 275 MB report (4:24); job **53479120** windowed
    re-capture `ACE2_NSYS_OK` (205 MB, 4:00).
  - **The model is 455,831,040 params** (455.8 M) — record it before anyone assumes otherwise,
    as happened with Pangu (1.18 B, not ~79 M).
  - **`tf32=True` is logged at startup** ⇒ the vendored `67242e348` perf commit is **live**,
    still with no equivalence baseline. Now confirmed by a run, not inferred from the diff.
  - **First profile (whole-run capture, batch 4, 4 ranks)**: NCCL **45.8%** of GPU kernel
    time, elementwise/copy **32.2%** (1.7 M launches), GEMM 10.4%, FFT/SHT 2.2%. **Occupancy
    is per-phase, not one number**: **91% during training, 3.3% during validation** (union of
    kernel intervals per device). An across-phase average (~59-70%) is meaningless and is NOT
    comparable to PanguWeather's "0.7% loader idle", which is `loader_wait_frac` over the
    training loop, a different quantity. The 9% training idle is **launch latency, not a
    stall** - 326k gaps, largest 9.76 ms, 7,222 kernel launches/s on one device. **Read the three caveats in the notes before quoting
    these** — NCCL ring kernels spin while waiting (max instance 4.16 s vs 11.2 ms median),
    so 45.8% is an upper bound on "not computing", not wire time; the capture includes
    startup + validation; and `batch_size=4` inflates the per-step all-reduce share.
    Surviving signal: **elementwise-bound** (same shape as PanguWeather) and fp32 gradient
    all-reduce is expensive on **A100-PCIE** (no NVLink). The windowed re-capture puts the
    training-only split at **NCCL 40.6% / elementwise 35.4% / GEMM 11.3%** — stable across
    all three views, so it is not an artifact of where the window fell.
  - **Validation is CPU-bound, not GPU-bound** — 280.2 s of the windowed capture's 282.5 s
    of kernel time falls in its first 71 s, so validation burns ~40% of the window's
    wall-clock (and ~80% of a cold epoch) for **~1% of GPU kernel time**. The aggregators,
    not the GPU, are what make an ACE2 epoch long ⇒ optimizing only the training step
    addresses the smaller half of the epoch.
  - **`fme` has ZERO instrumentation** — no `cudaProfilerApi`, no `torch.profiler`, no NVTX
    in the SFNO lat-lon path. So the house `--capture-range=cudaProfilerApi` flags would
    capture **nothing**, and `midway_bench_nsys.sh` uses a time window instead (its one
    deliberate departure from the s2s/SI/port scripts). `parse_nsys.py` yields nothing for
    ACE2 — the tables above come from querying the sqlite directly. Adding `ACE2_*` knobs +
    NVTX emitting the **shared** range names (#10) is the follow-up.
  - **`test` partition facts** (now in the notes): `AllowQos=test` makes `--qos=test`
    mandatory; `DefaultTime` is **5 minutes**; the partition is **hardware-mixed**
    (beagle3=A100, midway3-02xx=V100, 0320=A30) so **`--constraint=a100` is load-bearing**.
  - **Page cache dominates wall-clock again** (same lesson as the Pangu bench-vs-production
    gap): the warm re-run trained **8× more samples in half the wall-clock**. And **~80% of
    a cold epoch is validation**, not training.
  - **Open**: production `batch_size=16` unvalidated on 40 GB A100; no equivalence baseline
    for ACE2; `logging.metrics_log_dir` should be set — `GlobalTimer`'s category breakdown
    only reaches wandb, which the house rule disables.

- **2026-08-18** — **ACE2 (ai2cm `fme`) vendored into the repo** on branch `A2C` (off
  `fix/tsoi-fill-270`). Bring-up has not started; this is the code landing only.
  - **Provenance** (recorded here because the vendoring drops the nested `.git`):
    `cp -r /project/pedramh/shared/ACE2_retrain` (staged by krucker01). `ace_exp` was a
    clone of `https://github.com/ai2cm/ace.git` at **`1c3ebad80`** (2026-08-17), **4 commits
    ahead** of the fetched upstream `main` (merge-base `709c4c370e`).
  - ⚠️ **One of those 4 is a hot-path change that arrives already applied and has never been
    equivalence-checked**: `67242e348` "Perf: TF32, native SHT, spectral no-op copy, foreach
    EMA, DDP flags" (Katharine Rucker, 2026-08-14). **TF32 alone changes numerics.** Per
    DESIGN §4 this needs a baseline *before* it is trusted — and note the baseline would have
    to be captured against a build with it reverted, since it is already in the vendored tree.
  - **Vendored, not a submodule** — matches `makani_sfno`/`physicsnemo_*`/`PanguWeather`; the
    repo has no `.gitmodules`. `ace_exp/.git` was **moved, not deleted**, to
    `/project/rcc/mehta5/ace_exp_dotgit_backup_2026-08-18` (55 MB) — the 4 local commits are
    recoverable from there and nowhere else.
  - **`.gitignore` trap found and fixed (repo-wide).** Line 39's `core*` (intended for core
    dumps) also matches a **directory** named `core` at any depth, so it silently swallowed
    **`fme/core/` — 293 files / 13 MB, the heart of the `fme` package** — from a commit that
    otherwise looked clean at 686 files. It also defeated the `.npy` carve-out below, since git
    will not descend into an excluded directory. Replaced with `/core` + `core.[0-9]*`.
    The 26+26 already-tracked files under `physicsnemo_{ai_rossby,sfno}/**/core/` were
    unaffected (ignore rules don't apply to tracked files) — which is why this hid so long.
    **Any future vendored repo with a `core/` dir would have hit the same silent drop.**
  - **Second-ever `.gitignore` carve-out** (after `s2s-lightning/data/constant_mask/*.npy`):
    `!ACE2_retrain/ace_exp/fme/core/hpx/data/*.npy` — 7.5 MB of HEALPix reorder tables, without
    which the vendored `fme.core.hpx` is non-functional.
  - **70 scripts had lost their exec bit** (644 in the shared staging dir, 755 in `ace_exp`'s
    own index — so it was lost *before* this copy, not by it). Restored from the index so the
    vendored copy matches upstream; they are committed 100755.
  - **Deliberately NOT committed** (rule #8): `ace_training/merged_ACE2_ERA5_final.nc`
    (**2.39 TB**), `normalization/*.nc` (21 MB), and 36 `*.pt` regression fixtures (1.4 MB).
    ⇒ **`ace_exp`'s regression tests cannot run from a fresh clone** — re-copy the fixtures from
    `/project/pedramh/shared/ACE2_retrain`. Commit is 947 files / 20 MB.
  - **2.4 TB of data now lives in the repo working tree** at `ACE2_retrain/ace_training/`
    (ignored, so invisible to git). `/project` had 146 T free at the time. Worth relocating +
    symlinking if this becomes the pattern.
  - **`train.sh` targets a third cluster — Delta/NCSA**, not Midway or Polaris:
    `--account=bdiu-dtai-gh`, `--partition=ghx4` (GH200), `--qos=bdiu-dtai-gh`, `hsn0`/Slingshot
    NCCL vars, and mixed paths (`/project/pedramh/shared`, `/scratch/midway3/krucker01/envs/fme`,
    `/work/nvme/bdiu/...`). **Not runnable as-is on either of our clusters**; a Midway/Polaris
    sibling is the first bring-up task (rule #7: add beside it, don't edit in place).
  - **No smoke was run** — nothing of ours executes in this commit. Env, data paths and a
    scheduler script are the next steps before any PASS token exists to key on.

- **2026-08-14** — **Fixed: Pangu's `best_ckpt.tar` silently stops meaning "best" across any
  resume, and the true-best epoch's numbered checkpoint is already unrecoverable on the live
  run.** Found while answering "have we reached a best checkpoint" for the live production
  run (job chain 7368237 → 7368539 → ...).
  - **Root cause**: `best_valid_loss` and `early_stopping_counter` (`train.py`, `Trainer.train()`)
    were plain local variables, reset to `1.e6` / `0` on every call — and `train()` is called
    fresh every process start, while `restore_checkpoint()` (which runs once, in `__init__`,
    before `train()`) had no way to hand a resumed value into that local scope. Every resume's
    first validated epoch trivially satisfied `valid_loss <= 1.e6` and overwrote
    `best_ckpt.tar`, regardless of whether it was actually better than history.
  - **Measured on the live run**: `best_ckpt.tar`'s mtime matched epoch 42 (val_loss
    0.02690) — the first epoch validated after the most recent resume — not the run's true
    minimum, **epoch 27, val_loss 0.026548** (verified against the exact
    `Saved numbered checkpoint: ckpt_epoch_27.tar` log line). Epoch 27's own numbered
    checkpoint has since rotated out of the 10-file retention window (`_cleanup_old_checkpoints`)
    — **that checkpoint is gone**; this fix prevents recurrence, it does not recover it.
  - **Fix**: both become `self.` attributes, seeded via `getattr(self, ..., default)` in
    `train()` (fresh runs still start at 1.e6/0; a value already set by
    `restore_checkpoint()` survives), persisted in `save_checkpoint()`'s `checkpoint_data`
    dict, restored via `checkpoint.get(..., default)` (not `checkpoint[...]`) in
    `restore_checkpoint()` — `.get()` specifically so every checkpoint saved *before* this fix
    (which lacks the key) degrades to the old default instead of raising `KeyError` on load.
  - **Test**: `PanguWeather/v2.0/test/best_checkpoint_resume_test.py` — static AST analysis
    (no torch import, matches `bench_instrumentation_test.py`'s convention; this login node's
    `python3` is 3.6, so the test's own constant-node check has to handle the pre-3.8
    `ast.Str`/`ast.Num` split, not just `ast.Constant`). 6/6 `BEST_CKPT_RESUME_OK`, run
    directly on the login node (pure AST, no GPU/data needed).
  - **ai-rossby has no equivalent "best" mechanism at all** (`grep` for
    `is_best`/`best_checkpoint`/`best_valid_loss` in its `train.py`: zero hits) — only fixed
    interval-based numbered checkpoints, no best-selection logic, so this specific bug class
    doesn't apply there. Its own true-best epoch (~24-25, val_loss≈0.03258) happens to still be
    recoverable anyway, because of the OTHER known bug on that side
    (`max_checkpoints_to_keep` declared but never read — checkpoints accumulate unbounded,
    confirmed 50 files on disk back to epoch 2).
  - Not yet committed — this touches the live run's own source file
    (`PanguWeather/v2.0/train.py`); editing it is safe (the running process already has it
    loaded in memory, same reasoning as every other train.py edit this session), and the fix
    takes effect on Pangu's *next* resume, whenever that next queue rotation happens.

- **2026-08-11** — **Implemented `ai_rossby_finegrained_wandb_handoff.md`: ai-rossby now
  emits PanguWeather-identical per-var/per-level wandb keys.** Both production jobs
  (Pangu `7368237` on `capacity`, ai-rossby `7368547` on `preemptable`) are still RUNNING
  — nothing was launched; this lands in the working tree for the next queue-rotation
  restart per the handoff's own instruction. **Not yet GPU-verified** — see below.
  - New `physicsnemo_ai_rossby/examples/weather/ai_rossby/diagnostics.py`:
    `per_channel_lat_weighted_rmse` (mirrors Pangu's `weighted_rmse_torch_channels`/
    `_3D` exactly, keeping the channel/level axis) + `pangu_style_lwrmse_logs`
    (denormalize-and-key-format to Pangu's literal strings).
  - `train_loop.py::train_step` gained an opt-in `capture_outputs` dict param (default
    `None`, zero behavior change for the 4 existing callers) so `train.py` can read back
    the normalized-space output/target tensors after the timed step window.
  - `train.py`: new block after `telemetry.step_end()` (mirrors Pangu's placement +
    comment verbatim), runs on **every rank unconditionally** (no `wandb.enabled` gate,
    no frequency gate — matches Pangu's "whenever not BENCH"), calls `wandb.log(...)`
    directly (bypassing `LaunchLogger`, which both prefixes keys `train/` AND throttles
    to `mini_batch_log_freq=100` by default — either would have broken parity). Only
    rank 0 executes the final `wandb.log`; the `_ddp_mean_scalars` all-reduce before it
    is still a collective every rank joins, matching Pangu's own `dist.all_reduce`.
  - **Corrected a wrong claim in the handoff (§4.2):** it asserted ai-rossby's
    `cfg.model.levels` would format identically to Pangu's key strings. Traced instead —
    Pangu's `data_loader_multifiles.py:527-528` sets `self.levels = params['levels']`
    (the ROUNDED nominal list `[5, 10, 20, ..., 1000]`) UNCONDITIONALLY; `use_sigma_levels`
    only controls which list `load_mean_std` indexes stats with. ai-rossby's
    `cfg.model.levels` holds the OTHER list (full-precision hybrid values, needed for the
    normalizer's by-value matching) — using it directly would emit
    `train_T_level4.7150_lwrmse`, never merging with Pangu's `train_T_level5.0000_lwrmse`.
    Fixed with a hardcoded `PANGU_UPPER_AIR_LEVEL_LABELS` constant in `diagnostics.py`,
    pinned by a test asserting it equals Pangu's literal `SFNO.levels:` list.
  - **§3 latitude-grid check: DONE, reuse `cos_lat_weights` as-is (no new lat_values
    path needed).** Diffed Pangu's real production `lat:` array (180 cell-centered
    points, -89.5→89.5°, read from the live job's rendered config, not the template)
    against `cos_lat_weights`'s synthesized `phi` grid: agree to 1.4e-14° (float noise)
    after accounting for the ascending-vs-descending order. The `N·w/sum(w)` vs
    `w/mean(w)` normalization forms also agree to 6.9e-16.
  - New `physicsnemo_ai_rossby/test/recipes/ai_rossby/test_diagnostics.py` (the
    canonical test location — mirrors `test_loss.py`/`test_train_loop.py`'s
    `sys.path.insert` pattern to reach `examples/weather/ai_rossby/`; picked up by the
    existing `polaris/polaris_recipe_tests.pbs` runner via `TESTS=...`, not a new
    mechanism): 14 tests (shape, exact Pangu-formula match, the §3 grid check as a
    durable regression test, key-string parity, the FULL 101-channel E3SM variable
    contract — 8 surface + 5×18 upper-air + 3 diagnostic, matching `out_chans` — not
    just a toy 2-var example, de-normalization units, `train_step` capture wiring
    end-to-end). **RUN AND GREEN**: job 7413433 (13 tests, after fixing 3 self-inflicted
    test bugs — a dtype-strict `assert_close` against a function that's documented to
    always reduce in float32, and two `set(losses)` checks missing the always-present
    `vae_kl` placeholder key) then job 7413472 (14/14 after adding the full-contract
    test). `13 passed in 7.82s` / `14 passed in 1.73s`. Re-run with:
    `qsub -v TESTS="test/recipes/ai_rossby/test_diagnostics.py"
    physicsnemo_ai_rossby/polaris/polaris_recipe_tests.pbs`.
  - Left out of scope on purpose: `multistep_train_step`/`StaticCaptureTraining` paths
    (production runs `unroll_steps: 1`, `use_static_capture: False` — both dead code
    today); validation-side per-var metrics (handoff explicitly scopes these out, Pangu's
    own `long_validation` is off in production).

- **2026-08-07** — **BOTH PRODUCTION RUNS LAUNCHED. Read the QUEUE ROTATION below
  before touching anything in `qstat`.**

  | run | jobs | queue | config |
  |---|---|---|---|
  | PanguWeather | **7368237** → 7368539-41 | capacity 168 h → preemptable 3×72 h | `checkpointing: 2`, no ZeRO, wandb online |
  | ai-rossby | **7368547**-51 | preemptable 5×72 h | **`checkpointing: 3`, no ZeRO**, wandb online, `CKPT_INTERVAL=2` |
  | ai-rossby (reserved) | **7368536** | capacity 168 h | **USER-HELD — see below** |

  ### ⚠ QUEUE ROTATION — the one thing not reconstructable from the code
  `capacity` allows **one running job per PROJECT**. PanguWeather holds it now.
  `7368536` is ai-rossby's capacity reservation, held on `afterany:7368237` **plus a
  USER HOLD (`Hold_Types = ud`)**. The user hold is deliberate: both ai-rossby chains
  use `RUN_NAME=sfno_e3sm_parity01` and therefore the same `RUN_DIR`, so if the
  capacity job auto-started while the preemptable one was still running, **two
  processes would write checkpoints into the same directory and corrupt the run.**
  To hand ai-rossby the capacity slot, in this order:
  ```
  qdel 7368547 7368548 7368549 7368550 7368551   # stop the preemptable chain FIRST
  qrls 7368536                                    # then release the capacity job
  ```
  PanguWeather's continuation (7368539-41) auto-releases onto `preemptable` when
  7368237 ends — no action needed, and that queue suits it (`-r y`, per-epoch
  checkpoints, a header documenting the requeue contract).

  ### Measured on the live run (per-epoch telemetry, not a bench)
  `step_med` **755 / 776 / 767 ms** (stable ±1.4%), `peak` **32.63 GiB** fresh,
  `gpu_busy` **0.82-0.84**, epoch wall 3.23 / 2.99 / **2.79 h** ⇒ **~299 h projected**
  for 100 epochs. EMA switches on at **epoch 6** and its cost is still unmeasured.
  The remaining ~16-18% non-GPU wall is the biggest unclaimed lever (~30 h over the
  run); Fable's read is cold random zarr I/O over 2.15 TB, which the short
  cache-warm bench windows structurally could not see.

  ### RESUME SPIKES — measured on both harnesses, both fixed
  `peak_mem_gb` is run-to-date, so a resume spike cannot hide.
  * **PanguWeather +1.94 GiB** (32.63 → 34.57, headroom 6.86 → 4.92). Cause:
    `restore_checkpoint` loaded the 18.9 GB file with `map_location='cuda:N'`.
    Fixed → `'cpu'` (b039c483); `_migrate_optimizer_state_to_device()` already
    existed for exactly this. Every chain link paid it, and the run is 3-6 links.
  * **ai-rossby +5.32 GiB**, of which staging the optimizer on CPU recovered 1.05.
    The remaining **+4.28 GiB is still UNATTRIBUTED** — two hypotheses were wrong
    (the second, `Module.load`'s `map_location`, changed nothing). Does not block
    anything now that ai-rossby runs at `ckpt3` with ~13.6 GiB headroom.

  ### ai-rossby runs at `checkpointing: 3` — decision, and why
  `ckpt2` is ~1.26× faster but **OOMs without ZeRO**, so adopting it there is TWO
  changes behind ONE gate — and ZeRO's §4 result was withdrawn as confounded and
  never re-run. `ckpt3` is the config's own default, has zero ungated changes, and
  removes ZeRO from the critical path. A cold Fable review found no third option
  worth the work (hybrid per-block checkpointing buys ~1.12× for more new gating
  than ZeRO itself; EMA offload costs +30-60% step time or changes the deliverable).
  **Cost: ~47 h. Revisit only if the ZeRO bitwise gate passes.**

  ### Bugs found and fixed today (all mine, all would have bitten later)
  * **`import os` missing** in ai-rossby `train.py` — my probe instrumentation ran
    unconditionally at startup, so **every** ai-rossby run raised `NameError`. The
    queued production chain would have crashed ~30 h out and cascaded 4 dependents.
    Surfaced only because a measurement job hit the path first.
  * **`datetime.timedelta`** where line 8 binds `datetime` to the CLASS — would have
    crashed every future PanguWeather link at startup. Caught pre-commit.
  * **NCCL timeout 600 s → 30 min** — each epoch boundary costs 17.2 min with rank 0
    writing 18.9 GB alone while ranks 1-3 block in the next collective.
  * **`/usr/bin/time` does not exist on Polaris** (rc=127 killed a gate before it ran).
  * **`eligible_time` substring trap** — I quoted §1b about it, then wrote an
    unanchored `/eligible_time/` match into my own monitor, which reported the
    priority exponent `4` as a time. Anchor it.

- **2026-08-06** — **SFNO apples-to-apples: ARCHITECTURE GATE PASSED (job 7364984) — the
  two implementations are BITWISE identical, not merely same-sized.** Parity config,
  gate, per-epoch telemetry and group-write all in.
  > **Gate result (4×A100, ai-rossby venv, fp32, rc=0, `SFNO_PARITY_OK`):**
  > `in_chans=105 / out_chans=101` as predicted; **both models 1,182,108,160 params**,
  > matching the independently measured reference (job 7255410) exactly; and at
  > **identical weights on identical input the two agree to 0.000e+00 relative error**
  > on `surface`, `upper_air`, `diagnostic`, `loss` (5.26061580e-02) and `grad_norm`
  > (2.16819166e-01). Handoff §5 asked only for equal parameter counts — this is the
  > stronger fixed-weights form, so any remaining difference between the two runs is
  > **harness and data-path, not model.**
  Implements `polaris_sfno_comparison_handoff.md`. Owner decisions taken this session:
  results stay in `$MEMBER_ROOT` with the modes fixed (no new group-level dir),
  **`embed_dim 512`** (jesswan's production value), and the two long runs go to
  **different queues** — ai-rossby on `capacity`, PanguWeather on `preemptable` — which
  sidesteps `capacity`'s one-running-job-per-PROJECT cap (`max_run=[p:PBS_GENERIC=1]`,
  verified with `qstat -Qf`) instead of serialising them.
  - **The architecture claim is now PROVEN, not asserted** (`compare_sfno_parity.py`).
    `SfnoPlasim`'s docstring claimed fidelity to PanguWeather's
    `SphericalFourierNeuralOperatorNet_v2`; the vendored `modulus_sfno/` is in fact a
    near-verbatim copy — **5 of 7 files byte-identical**, the other two differing only by
    `@torch.compiler.disable` decorators (inert in eager) and `return_latent` plumbing
    (adds no module ⇒ no parameter). The gate keeps an **exact-line** allowlist, not
    regexes: a regex broad enough to cover the `return_latent` refactor (`^\s*else:$`)
    is broad enough to hide a real change. Negative control confirms an added
    `nn.Linear` still fails it. Config block: **26/26 architecture keys identical.**
  - **Two traps found that would each have silently broken the comparison:**
    1. **The effective learning rate is 3.0517578125e-05, NOT the `lr: 1e-4` written in
       the SFNO block.** That line is dead — `get_optimizer` (train.py:706-709) checks
       `hasattr(params,'loglr')` FIRST, and `loglr: -13` is inherited from the base
       config, so the peak LR is `2**-13 * global_batch_size/16` at 4 ranks × batch 1.
       Taking the file at face value would have run ai-rossby at **3.3×** the reference.
       It is also world-size-dependent, so it must be recomputed if ranks change.
    2. **The handoff's §4 claim that the checker already expects a folded config was
       wrong.** `--check-artifacts` compared `model.surface_variables` against the
       *unfolded* 6 and required `land_variables`/`ocean_variables` keys — which
       `SfnoPlasim.__init__` cannot accept (`build_model` forwards every config key as a
       kwarg). Checker now handles both shapes explicitly and reports which it checked;
       12 tests, incl. negative cases proving the folded branch is not a way to skip the
       land channels. **19/19 PASS folded, 21/21 PASS split (unchanged).**
  - **`torch.compile` stays OFF** for both runs — measured 1.40× but FAILS the §4 gate.
  - **Per-epoch telemetry added to BOTH PRODUCTION trainers** (`epoch_telemetry.py`,
    duplicated in each tree because they share code by copy; a drift test pins the
    copies identical). Opt-in (`{PANGU,AI_ROSSBY}_EPOCH_TELEMETRY=1`), on by default in
    the two run scripts. **Why not the bench harness:** it structurally cannot answer
    "what do epochs 1-6 cost" — `profile_train.py` builds **no EMA** and scopes its
    scheduler to the bench window, and `PANGU_BENCH` truncates the run and exits. Two
    CUDA events/step + one all-reduce/epoch, **no per-step sync** (a sync would remove
    CPU/GPU overlap and change the number being measured). 21 shared columns.
    - **This immediately exposed a real harness difference:** PanguWeather **skips the
      `update_ema` sweep entirely** until `ema_warmup_epochs` (6), so its step cost jumps
      at that epoch; ai-rossby calls `ema.update` from epoch 1 and warms only the *decay
      value*, paying the sweep throughout. On 1.18 B params that is ~4.7 GB of
      elementwise traffic per step. Recorded honestly in `ema_active` rather than
      smoothed over. Window boundaries are pinned by test so both sides span the same
      work (Pangu's per-iteration RMSE/wandb block is deliberately *outside* it — it has
      no ai-rossby counterpart).
  - **Group write fixed** (`umask 0002` in `polaris_env.sh`, before its `mkdir`s;
    `chmod -R g+rwX` + setgid over runs/bench/ai_rossby_data/pangu_polaris_data/
    baselines). The setgid bit was already correct — only the mode was wrong.
  - **Dead knob found:** `cfg.max_checkpoints_to_keep` is declared in ai-rossby's
    `conf/config.yaml` and **never read**, and physicsnemo's `save_checkpoint` does not
    prune ⇒ checkpoints accumulate unbounded at ~14 GB each (4.7 GB weights + 9.4 GB
    AdamW moments). `CKPT_INTERVAL` (default 5 ⇒ ~280 GB/100 epochs) is the only lever
    today. Eagle shows no quota cap, so this is a tidiness issue, not a blocker.
  - **New files:** `compare_sfno_parity.py`, `epoch_telemetry_test.py`,
    `ai_rossby_variable_contract_test.py`, `{PanguWeather/v2.0/utils,
    physicsnemo_ai_rossby/examples/weather/ai_rossby}/epoch_telemetry.py`,
    `conf/model/sfno_e3sm_parity.yaml`, `conf/training/sfno_e3sm_parity.yaml`,
    `polaris/polaris_sfno_{parity_gate,e3sm}.pbs`, `polaris/polaris_bench_sfno_e3sm.pbs`.
    Also corrected a **false comment** in `conf/model/sfno_e3sm.yaml` claiming the land
    variables are not written by the per-year converter — they are, folded into the
    store's `surface_variables` (verified on `train/2020.zarr`).
  - **ai-rossby SFNO training smoke GREEN — job 7364985, `SFNO_E3SM_SMOKE_OK`.** First
    time the vendored SFNO has ever been executed in the ai-rossby venv.
    `PREFLIGHT_OK` (`VARIABLE_PARITY_OK 19/19` against every one of the 35 stores),
    `steps_per_epoch=10950` — exactly the Pangu arithmetic (43,800/4) — finite loss
    **2.947**, val_loss **2.995**, and the first telemetry row:
    `epoch=1 n=20 step_med=751.1ms gpu_busy=94.9% peak=25.89GB ema=1`. Peak leaves
    ~14 GB of headroom on a 40 GB A100. `gpu_busy 94.9%` at `num_workers=1` puts loader
    idle at ~5%. (`step_mean` 1133 ms vs `step_med` 751 ms with std 1658 is first-step
    autotune in a 20-step window; it washes out over 10,950 steps — read the median.)
    Confirmed end-to-end that `umask 0002` works: the CSV landed `-rw-rw-r--`.
  - **`validation=off` does NOT turn validation off** — found by the smoke appearing to
    hang for 5 minutes after training finished. `conf/validation/off.yaml` disables only
    the ROLLOUT; a single-step `val_loss` pass still runs over every sample whenever
    `dataset.val_zarr_path` is set (train.py:832-842). Measured: **322.6 s per epoch**
    over the 4 val years (5,840 samples) — 13× the 20-step smoke it was validating.
    `dataset.val_zarr_path=null` is what actually skips it; the smoke path now does
    (override with `SMOKE_VALIDATE=1`). Two consequences: budget ~5.4 min/epoch of
    validation into the ai-rossby production run, and note this is a **disclosed
    non-matching axis** — PanguWeather's production validation is 129 ICs × 60-step
    rollouts, entirely different work. The telemetry row deliberately covers TRAINING
    only (`epoch_end` fires before validation), so the step comparison stays clean.
  - **MATCHED BENCH ROWS — the honest harness number is 1.166×, ai-rossby faster.**
    Jobs **7364997** (ai-rossby) and **7364998** (PanguWeather, `CONFIG_NAME=
    E3SM_SFNO_H5_POLARIS_ALLDATA`), same node type, 20+80 steps each. Every
    controlled axis identical in the CSVs: `n_gpus 4`, `batch_per_gpu 1`, bf16,
    `ddp_find_unused false`, **`n_loaders 1`**, `n_steps_counted 80`, and — proven by
    the gate — the same 1,182,108,160-param model.

    | | ai-rossby | PanguWeather | ratio |
    |---|---|---|---|
    | `step_med` | **0.71996 s** | **0.83975 s** | **1.166×** |
    | `samples_per_s` | 5.556 | 4.763 | **1.166×** |
    | `compute_med` | 0.71994 | 0.83753 | 1.163× |
    | `step_p90` | 0.72346 | 1.11494 | 1.541× |
    | `step_std` | 0.00814 | 0.19743 | 24× |
    | `peak_mem_gb_max_rank` | 21.40 GiB | 28.76 **"GB"** = 26.79 GiB | **1.25×** |
    | loader idle | 0.63% (`data_idle_frac`) | 9.05% (`loader_wait_frac`) | — |
    | `cpu_prep_med` | 1.5e-05 s | 2.2e-03 s | 149× |

    `step_med` and `samples_per_s` agree to three digits (internal consistency).

    > **⚠ UNIT MISMATCH in `peak_mem_gb_max_rank` — the two harnesses do not
    > report the same unit.** PanguWeather divides by **1e9** (decimal GB,
    > `train.py:1344`); ai-rossby's `profile_train.py:347` and
    > `epoch_telemetry.py:224` divide by **1024³** (GiB). Pangu's numbers are
    > therefore inflated ~7.4% relative to everything else in this document.
    > Pangu's `28.76 "GB"` is **26.79 GiB**, its headroom is **12.70 GiB** (not
    > 10.7), and the memory ratio is **1.25×** (not 1.34×). Third cross-harness
    > measurement inconsistency found today, after the missing EMA and the node
    > variance — every one of them made the two sides look more different than
    > they are. **Convert before comparing anything against a Pangu row.**

    > ### ⚠ RETRACTED SAME DAY — 1.166× IS NOT CONFIRMED. **Do not quote it.**
    > The two rows were measured in **separate jobs on different nodes** —
    > ai-rossby on `x3001c0s13b0n0`, PanguWeather on `x3003c0s37b0n0`. The sweep
    > (job 7365119, node `x3109c0s19b1n0`) then re-measured the **identical**
    > `ckpt3/sg0/w1/bs1` config at **0.6517 s** against the parity bench's
    > **0.7200 s** — **1.105×, from the node alone.**
    >
    > Node-to-node spread (10.5%) is therefore the same order as the effect
    > attributed to the harness (16.6%), and with n=2 nodes the *direction* of the
    > bias is unknown, so the true gap could be larger, smaller, or absent.
    >
    > This is the same failure class as the 1.51×→1.33× post-mortem below —
    > caught here only because the sweep happened to re-run the baseline config.
    > **Fix: bench both harnesses in ONE job on ONE node, alternating A/B/A/B/A/B
    > for repeats.** Until that runs, the only defensible statements are the
    > architecture gate (exact) and the within-job sweep ratios (same node,
    > back-to-back).
    >
    > **Lesson for every future comparison here: a cross-JOB ratio on Polaris is
    > not a measurement.** One node, one job, interleaved repeats, or it does not
    > count.
  - **Two things worth chasing, both now measurable:** (a) `compute_med` alone differs
    by 1.163×, i.e. the gap is NOT mostly loader — with an identical model, that is
    harness overhead inside the step; (b) PanguWeather is slower **while not yet paying
    for EMA** (`_ema_active()` is False until epoch 6) whereas ai-rossby sweeps from
    epoch 1. Expect the gap to WIDEN at Pangu's epoch 6 — which is exactly what the
    per-epoch telemetry was built to catch.
  - **Epoch budget arithmetic** (from the measured rows, for the still-open decision):
    ai-rossby **2.19 h/epoch** training + 0.09 h validation ⇒ 100 epochs ≈ **228 h**,
    which does NOT fit one 168 h `capacity` job. PanguWeather **2.55 h/epoch** training
    + production validation (129 ICs × 60-step rollouts, est. 0.2–0.7 h) ⇒ 100 epochs
    ≈ **275–325 h**, i.e. 4–5 uninterrupted 72 h `preemptable` links at best.
  - **BEST-CONFIG SWEEP for ai-rossby SFNO — job 7365119, 12 configs, ONE node
    (`x3109c0s19b1n0`), back-to-back.** These ratios are same-node/same-job and are
    therefore trustworthy in a way the cross-job parity ratio is not.

    | config | cls | step_med | samp/s | vs ckpt3 | peak GB |
    |---|---|---|---|---|---|
    | `ckpt3` (current) | N | 0.6517 | 6.138 | 1.000× | 21.40 |
    | `ckpt2` | N | 0.5116 | 7.819 | **1.274×** | 33.96 |
    | `ckpt1` | N | 0.4986 | 8.023 | **1.307×** | 36.06 |
    | `ckpt0` | N | 0.4987 | 8.020 | 1.307× | 36.11 |
    | `ckpt1`+static_graph | N | — | — | **CUDA OOM** | — |
    | `ckpt0`+static_graph | N | — | — | **CUDA OOM** | — |
    | `ckpt3`+workers=8 | N | 0.6579 | 6.080 | **0.991×** | 21.40 |
    | `ckpt1`+workers=8 | N | 0.5013 | 7.980 | 0.997× | 36.06 |
    | any `batch_size` 2 or 4 | C | — | — | **CUDA OOM** | — |

    - **`checkpointing` is the only lever that pays.** The expensive branch is `>=3`
      (checkpoint every block); dropping to 2 recovers 1.274× of the 1.307× available.
      **`ckpt0` and `ckpt1` are indistinguishable** (0.4986 vs 0.4987 s, 36.06 vs
      36.11 GB), so the `>=1` encoder/decoder branch costs and saves nothing — 1 is
      the sensible floor and there is no reason to run 0.
    - **`num_workers` is worthless here, and slightly negative** (0.991×, 0.997×) —
      exactly as the measured `data_idle_frac` of 0.0068 predicted. **The old "+9%
      throughput at 8 workers" does NOT transfer**: that was PanguPlasim at 449 ms/step;
      at ~500-650 ms with a 1.18 B model the loader is already fully hidden and extra
      workers only add startup cost. Do not re-try this knob on SFNO.
    - **`ddp_static_graph` OOMs** at ckpt1/ckpt0 — it retains extra state. The
      `sfno_plasim.yaml` comment calling it "safe to enable here" is not true at low
      checkpointing. Untested at ckpt2/ckpt3, where there is headroom.
    - **`batch_size` > 1 is unavailable** at any fast setting; even bs=2 at ckpt1 OOMs.
  - **ZeRO-1 SWEEP — job 7366778, real `train.py` (EMA on), 8 configs, one node.
    ZeRO is REQUIRED, not optional: without it the entire speedup is unreachable.**
    Predictions were written into the script before it ran; **6 of 8 exact**.

    | config | step | vs base | peak GB | predicted |
    |---|---|---|---|---|
    | `ckpt3` no-ZeRO | 711.5 ms | 1.000× | 25.91 | ~25.8 ✅ |
    | `ckpt2` no-ZeRO | — | — | **OOM** | "marginal ~38.4" ❌ |
    | `ckpt1` no-ZeRO | — | — | **OOM** | OOM ✅ |
    | `ckpt3` + ZeRO | 724.1 ms | 0.983× | **19.46** | fits ✅ |
    | `ckpt2` + ZeRO | 574.1 ms | **1.239×** | 32.02 | ~31.8 ✅ |
    | `ckpt1` + ZeRO | 561.0 ms | **1.268×** | 34.12 | ~33.9 ✅ |
    | `ckpt0` + ZeRO | 559.6 ms | 1.271× | 34.17 | ~33.9 ✅ |
    | `ckpt1` + ZeRO, bs=2 | — | — | OOM | "may now fit" ❌ |

    - **ZeRO-1 saves 6.45 GB** (25.91 → 19.46 at ckpt3; predicted 6.6) and costs
      **1.7%** in step time — the all-gather plus losing the fused AdamW kernel.
    - **Without ZeRO, ckpt2 and ckpt1 both OOM**, so on plain DDP this model is
      pinned at ckpt3 and none of the 1.24-1.27× is reachable. That makes ZeRO
      the enabling change, not a memory nicety.
    - **The EMA accounting is now empirically anchored**: `ckpt3` real training
      measured **25.91 GB** vs the bench's 21.40 GB and the earlier smoke's
      25.89 GB — a 0.02 GB reproduction. Bench peaks understate production by
      the 4.4 GB EMA copy, as derived.
    - **Bench step times also understate production by ~9%**: same `ckpt3`
      config, 651.7 ms in the bench vs 711.5 ms in the real trainer — the EMA
      sweep over 1.18 B params plus DDP metric reduction and logging.
    - **The one wrong prediction is the useful one.** `ckpt2`/no-ZeRO was called
      "marginal ~38.4 GB" and OOM'd ⇒ **these estimates run ~1-2 GB optimistic.**
      Do not trust a margin under ~2 GB.
    - **Not fragmentation.** The OOM reports 38.29 GiB allocated with only
      57.76 MiB reserved-but-unallocated, so `expandable_segments` would not have
      helped — these are genuine capacity limits. (The `PYTORCH_CUDA_ALLOC_CONF`
      vs torch-2.10 `PYTORCH_ALLOC_CONF` naming question is therefore hygiene,
      not a live defect.)
  - **PANGUWEATHER PRODUCTION LAUNCHED — 7366939 → 7366940, `capacity`, 2 × 168 h,
    100 epochs, `checkpointing: 2`, ZeRO OFF.** Pre-flight verified, not assumed:
    51,100 h5 files (1460/yr; train 2015–2044 = 43,800 samples ⇒ 10,950 steps/epoch
    at 4 ranks; val 2045–2048); corrected stats confirmed by value —
    `SST` **8.4407/12.0659**, `TSOI_10CM` **271.1259/16.3902**, and **no zero stds**
    anywhere (`PRECT` is 2.89e-08/8.30e-08, tiny but real); config dry-rendered and
    inspected (`embed_dim 512`, `epsilon_factor 0.01`, `checkpointing 2` in both
    places, batch 1, workers 1, 100 epochs, `train.py`). ZeRO is off because its
    gate is not rebuilt — an ungated change does not go into a 300 h run.
    At 0.4667 s/step ⇒ 1.42 h/epoch training — **WRONG, see below**.
  - **✅ THE BENCH-VS-PRODUCTION GAP IS SOLVED: it is COLD PAGE CACHE / loader I/O.**
    Two earlier entries called it unexplained and blamed first-CUDA-context clock
    ramp. **Both wrong, and the discriminating column was already in the CSV.**
    Four samples of the identical `ckpt3_zero0` config:
    | run | step_med | step_std | **loader_wait_frac** |
    |---|---|---|---|
    | `ckpt3_zero0` (first in job 7366921) | 0.8965 | 0.2458 | **0.0611** |
    | `ckpt3_zero0_r1` (first in job 7366932) | 0.8916 | 0.1349 | **0.0259** |
    | `ckpt3_zero0_r2` | 0.6014 | 0.0002 | 0.0003 |
    | `ckpt3_zero0_r3` | 0.6014 | 0.0002 | 0.0002 |
    Every config is its own `torchrun`, so a fresh CUDA context happens EVERY time —
    "first context" would make every row slow, but only **first-in-JOB** is. What
    survives across processes inside a job and not across jobs is the **page cache**,
    and `loader_wait_frac` is **100-200× higher** on exactly those rows. A clock ramp
    does not move a loader-wait metric by 200×.
    The magnitudes close in three independent comparisons:
    cold−warm same job **0.2927 s**, cold−warm other job **0.295 s**,
    production ckpt2 − warm bench ckpt2 **0.2995 s** — the same penalty at two
    checkpointing levels on two clocks.
    **The bench never touches cold data**: 50 steps × 4 ranks redraws the same 200
    samples (`DistributedSampler(seed=0, epoch=0)`), while production shuffles over
    **51,100 files / 2.0 TB, 42 MB per timestep, at `num_data_workers: 1`**.
    ⇒ Bench step times are a **warm-cache lower bound**, not a measurement artifact.
  - **WHERE THE 100-EPOCH TIME GOES — from `out.log` + telemetry, zero extra jobs.**
    Epoch 3 decomposes: train-loop wall 90.8% (in-window GPU 75.0%, gap 15.8% — of
    which **loader `next()` 12.3%** and **diagnostics 3.5%**), validation 8.2%,
    checkpoint write ~1.0%. Over 100 epochs: **loader 37.9 h, diagnostics 10.8 h,
    validation 25.4 h, checkpoints 3.0 h**, plus the in-window cold-I/O penalty of
    **~91 h** ⇒ **~129 h of a ~308 h projection is addressable I/O (42%)**.
    The 16-18% `gpu_busy` gap is the SECOND-order term; the in-window penalty is 2.4×
    larger.
  - **⚠ SUPERSEDED (kept for provenance): "BENCH STEP TIMES DO NOT PREDICT
    PRODUCTION, AND THE CAUSE IS UNKNOWN."**
    Live production at `ckpt2` measures **754.9 / 776.1 ms/step** (telemetry,
    epochs 1-2) against the bench's **466.7 ms** — a **1.62×** gap, and ~2.27× on
    epoch wall-clock. Real budget: **~3.4 h/epoch ⇒ ~340 h for 100 epochs**, so the
    2 × 168 h chain is ~4 h short and **needs a third link**.
    **A previous entry blamed the per-iteration diagnostics block. That was wrong
    and this repo's own code refutes it**: `train.py:1216-1222` closes the
    telemetry window *deliberately before* that block, precisely so a telemetry
    row and a PANGU_BENCH row span the same work. Both windows exclude it.
    Ruled out so far: the diagnostics block (above); EMA (`ema_active=0` in both
    epochs, warmup is 6); node variance (~10%, not 62%). Still open — candidates
    are the measurement method itself (bench uses `cuda.synchronize()` +
    `perf_counter`, telemetry uses CUDA events) and sustained-vs-burst clock
    behaviour over 10,950 steps rather than 50. **Until it is explained, treat
    every bench-derived step time in this document as a lower bound** — including
    the older 0.602 s/step. Bench MEMORY figures remain trustworthy (bench and
    production agree exactly at 32.63).
  - **⚠ COLD-CACHE CONTAMINATION (originally filed as "cold start") — the sweep's RATIOS were inflated
    (repeat job 7366932).** Four samples of the identical `ckpt3_zero0` config:
    **0.8965, 0.8916, 0.6014, 0.6014** — bimodal, 49% spread, while every other
    config held to **≤0.1%**. Tracing by job: both slow samples were the **first
    config of their job**; both fast ones came later *within* a job. Originally
    attributed to "first CUDA context / clock ramp" — **that was wrong**; it is the
    **page cache** (see the entry above: `loader_wait_frac` 100-200× higher on
    exactly those rows). The conclusion that the ratios were contaminated stands;
    only the mechanism changed. Corrected against the warm 0.6014:
    | config | sweep said | **actual** |
    |---|---|---|
    | `ckpt2` no-ZeRO | 1.909× | **1.288×** |
    | `ckpt1` no-ZeRO | 1.958× | **1.321×** |
    | `ckpt3` + ZeRO | 1.465× | **0.988×** |
    That now matches ai-rossby's independent 1.27–1.31×, and **ZeRO costs ~1.2% at
    every level on both harnesses** — the "1.47× ZeRO speedup" never existed.
    Absolute step times are unaffected (0.1% spread), so the production budget
    stands. **Discard each job's first config as warm-up**; and note that
    median-of-repeats *failed* here — the median of a bimodal sample picked a cold
    outlier. Printing every pass is what caught it.
  - **⚠ ai-rossby EMA WAS NEVER RESTORED ON RESUME — silent, and it degraded the
    DELIVERABLE.** `train.py` called `load_checkpoint()` without `metadata_dict=`,
    so the EMA written into every checkpoint (~4.4 GB) was never read back
    (`checkpoint.py:1164-1165` only surfaces metadata through that argument). And
    `ModelEMA` is built *before* the resume, so it deep-copied the random init and
    then self-healed to the live weights — no crash, no NaN, just a **reset decay
    warmup at every job boundary**. Since `validate_with_ema: True` and
    `inference.py` runs `use_ema=true`, **the delivered model is the EMA**: across a
    2-link chain it would have averaged only over link 2. PanguWeather restores its
    equivalent (`train.py:3737`); ai-rossby did not. Fixed, with a log line showing
    `n_averaged` so a regression is visible. **Gate: `polaris_resume_gate.pbs`** —
    the save→consolidate→resume path had **never executed once** (every ZeRO run set
    `checkpoint_save_interval=1000000`).
  - **ZeRO provenance, for the record:** it was added to PanguWeather by
    **Alexander Wikner on 2026-05-15** and has sat dormant since — **no config in
    the repo ever enabled it**, and **DESIGN §5's ladder does not list sharding at
    all** (it starts at `torch.compile`). It surfaced only because the memory
    question was asked; the ladder assumed compute was the lever, which the profile
    supported. ai-rossby had none until this session.
  - **⚠ ZeRO EQUIVALENCE: RESULT WITHDRAWN, TEST WAS CONFOUNDED (job 7366891).**
    An adversarial review refuted it. Two independent defects, both in the test:
    1. **The premise was false.** The instrument was built on "ZeRO reduce-scatters,
       so summation order changes, so a bitwise test would be wrong."
       `ZeroRedundancyOptimizer` does **not** touch gradient reduction — DDP's
       all-reduce runs unchanged, each rank runs the local optimizer over its
       shard, and updated parameters are **broadcast** (`step()` →
       `_local_step()` + `_sync_params()`). No reduce-scatter exists. So every
       rank holds bit-identical gradients after the all-reduce, AdamW is
       elementwise, and **correct ZeRO-1 must be BITWISE identical**. The whole
       noise-floor apparatus tolerated a difference that should not exist.
    2. **The arms used different kernels.** `sfno_e3sm_parity.yaml:90` sets
       `fused: True`; `_wrap_zero` (train_loop.py:175) drops it because the
       wrapper rejects it. So the DDP arms ran **fused** AdamW and the ZeRO arm
       ran **eager** AdamW. The measured **6.675e-06 is the kernel swap**, not
       sharding — the test compared fused-vs-eager and called it ZeRO-vs-DDP.
    Also wrong, from the same review: the "below one bf16 ulp" defence was a
    **category error** (the compared quantities are fp32 loss and **fp64**
    grad_norm; bf16 appears only in matmul intermediates); the 20-step window sat
    entirely inside LR warmup so it measured at **~4% of production peak LR**,
    where optimizer-path discrepancies are smallest; and recording only
    rank-**averaged** scalars structurally hides ZeRO-1's characteristic failure,
    replica divergence from a missed/stale broadcast.
    **Fixed** (commit 04447417): `--require-bitwise`, `fused=false` forced on all
    arms, per-step cross-rank parameter checksum, and the JSON now witnesses
    `optimizer_class` / `optimizer_fused` / `lr_first` / cross-rank delta so a
    confounded run is visible in the artifact. **Rerun pending. ZeRO is NOT
    cleared for adoption.**
  - **Lesson, and it is the fourth of its kind today** (after the cross-node
    parity ratio, the missing EMA, and the GB-vs-GiB mix-up): every one of these
    made a change look better or more different than it was, and every one was
    caught by a control or an outside check rather than by the measurement
    itself. **A number from an instrument nobody attacked is not a result.**
  - **RECOMMENDED ai-rossby production config: `model.checkpointing: 2` +
    `use_zero_optimizer: true`** — 1.239× at 32.02 GB (7.47 GB headroom), over
    `ckpt1`'s 1.268× at 34.12 GB (5.37 GB). `ckpt1` buys 2.3% for 2.1 GB less
    headroom, and the estimates just proved ~1-2 GB optimistic; a 170 h run that
    dies at hour 100 costs far more than 2.3%. `ckpt0` is strictly worse than
    `ckpt1` (same speed, more memory) — as the first sweep also found.
    At 1.75 h/epoch + 0.09 validation, 100 epochs ≈ **184 h**, down from ~225 h.
    **Still not adopted**: needs the §4 equivalence gate, and a validation-inclusive
    smoke (a 2-epoch/20-step run suffices — `peak_mem_gb` is run-to-date, so
    epoch 2's row already includes epoch 1's validation).
  - **⚠ THE BENCH SWEEP'S MEMORY NUMBERS UNDERSTATE PRODUCTION BY ~4.4 GB — and it flips the
    recommendation.** `profile_train.py` builds **no EMA**; `train.py` does, and
    `ModelEMA` holds a full fp32 shadow copy = 1,182,108,160 × 4 B = **4.40 GB**.
    Confirmed against the smoke: train.py peak 25.89 GB vs bench peak 21.40 GB, a
    **4.49 GB** delta. Adding it back, against 39.49 GiB usable:
    | config | bench peak | + EMA | verdict (before validation) |
    |---|---|---|---|
    | ckpt3 | 21.40 | ~25.8 | fine — matches the smoke's 25.89 |
    | ckpt2 | 33.96 | ~38.4 | **~1 GB margin — marginal** |
    | ckpt1 | 36.06 | ~40.5 | **would OOM** |
    So the fastest configs are **not actually reachable in production**. `ckpt2` is the
    only candidate and it is borderline before validation is even counted. **Nothing is
    adopted**: this needs a real short training run (EMA + validation) at ckpt2, then
    the §4 equivalence gate vs ckpt3, before it goes anywhere near a production run.
  - **Worth a cheap check:** torch 2.10 documents the allocator knob as
    `PYTORCH_ALLOC_CONF`, while `polaris_env.sh` exports `PYTORCH_CUDA_ALLOC_CONF`. If
    the old name is no longer honored in the ai-rossby venv, `expandable_segments:True`
    is silently inactive there — which would change every OOM verdict above.

  - **PRODUCTION RUNS (CANCELLED, pending the best-config decision) — 100 epochs each, to CONVERGENCE (owner's call: the goal
    is trained models, not only parity/profiling).** Both pre-chained with
    `polaris_submit_chain.sh`, so neither needs daily human resubmission; every link
    re-runs its script from the top and resumes from the run's checkpoint.
    - **ai-rossby** → `capacity`, **2 × 168 h**: **7365020** (Q) → 7365021 (H).
      ~228 h needed, so link 2 finishes it. This holds the project's ONLY capacity
      slot (`max_run=1` per project) — coordinate before queueing anything else there.
    - **PanguWeather** → `preemptable`, **6 × 72 h**: **7365022** (Q) → 7365023-27 (H).
      ~275–325 h needed; 6 links = 432 h of margin, deliberately over-provisioned
      because a link that starts after training completed just resumes a finished run
      and exits in minutes. `preemptable` is scheduling again (26 running at submit
      time), unlike 2026-08-05 when it never started a single job.
    - Both at `seed=0`, `num_data_workers/num_workers=1`, bf16, batch 1 × 4 ranks,
      `checkpointing: 3`, telemetry on for every epoch. Watch progress with the
      per-epoch CSVs under `$MEMBER_ROOT/bench/epoch_telemetry_*.csv`, not the exit
      code — a preempted link exits non-zero and that is normal.
  - **Group read+write DONE and verified**: `ai_rossby_data` stores are `drwxrwsr-x`
    with `-rw-rw-r--` chunks, likewise `runs/`, `bench/`, `pangu_polaris_data/`,
    `baselines/`. The `chmod -R` over the 35 zarr stores took ~50 min (millions of
    chunk files) — run it in the background, not inline.
  - **Still open:** jesswan's sign-off on the fills (DESIGN §1) before any resulting
    model's numbers are reported.

- **2026-08-06** — **PRODUCTION CONVERSION COMPLETE (35 stores) — after routing it
  through `debug`, because `preemptable` never scheduled it.**
  - **Final dataset**, verified on disk (not from logs): `train/` **30 stores,
    2015–2044**, contiguous, 1460 timesteps each = **43,800 training samples**;
    `val/` **2045–2048** full years + **2049** as the 1-sample tail store. Exactly
    the handoff §5 Step 4 target. Production training is no longer blocked on data.
  - **⚠ `val/2046.zarr` existed but was WRONG — and "it's on disk" would have hidden
    it.** It was the 1-sample tail store from the first smoke (time len **1**,
    22 MB) whereas production needs 2046 as a full val year and 2049 as the tail.
    Rebuilt to 1460. Skipping it would have given production a validation year of
    one sample — surfacing much later as a metrics oddity, not as a conversion bug.
    **Check `time` length, never mere existence.**
  - **The scheduling lesson, and a wrong diagnosis of mine.** I blamed queue times
    on requested walltime and resubmitted three times on that theory (18 h → 3 h →
    2 h). Wrong: the jobs had been **eligible 11 h 28 m** with
    `comment = Insufficient amount of resource: queue_tags`. `preemptable` only
    runs on nodes prod isn't using and had none (backlog grew 106 → **134 queued**
    while we waited). Walltime was never the constraint.

    | queue | evidence, 2026-08-05/06 |
    |---|---|
    | `debug` | **9/9 jobs started**, median wait **19 s**, worst 27 min |
    | `preemptable` | 18 h job: never started in 12 h. 8× 2 h jobs: never started in 11.5 h |

    **Every piece of real work that completed came through `debug`**; everything
    sent to `preemptable` after 16:08 did nothing. Check `eligible_time` (with an
    anchored grep — `wfp_eligible_time_exp` is a substring trap) before theorising
    about priority.
  - **How it was done:** `debug` allows one job at a time, so a driver submitted 5
    sequential chunks of ≤7 stores (6.2 min/store measured ⇒ ~43 min against a
    55 min wall). All five green on `CONVERT_ALL_OK`; **3 h wall total.** The driver
    stops on the first failed chunk rather than leaving a silent hole. This is only
    expressible because of the `${VAL_YEARS-}` fix (`b5f80060`) — with `:-`, every
    train-only chunk would have silently converted `val/2045` and raced.

- **2026-08-05** — **DESIGN §4 equivalence gate BUILT, first baselines captured — and
  `torch.compile` FAILS it, so the 1.40× is NOT adopted.** Commit `402d336e`,
  job **7353187**.
  - **What now exists.** `baselines/ai_rossby_pangu_plasim/{eager,compiled_default}.json`
    — the repo's first §4 artifacts — plus `equivalence.py` (deterministic K=20
    capture), `polaris/compare_baselines.py` (§4.1's metric, stdlib-only), and
    `polaris/polaris_equivalence_compile.pbs` (both captures in ONE job, so the
    only difference is the change under test). §4.1 was the item blocking every
    optimization; it is now runnable in ~3 min.
  - **The verdict, stated plainly: FAIL.**

    ```
    max rel err 4.019e-01 > 1e-2   at forward_output_stats.surface.mean
    loss trajectory alone          1.113e-01  (step 11, diagnostic)
    surface.std    0.44623 -> 0.43339   (2.9%)
    upper_air.max  1.44501 -> 1.28017  (11.4%)
    ```

    The `std`/`max` drifts are on well-scaled quantities, so this is **not** just
    the near-zero-mean artifact the headline number might suggest.
  - **The divergence COMPOUNDS — it does not start large:**

    | step | 0 | 2 | 3 | 10 | 15 |
    |---|---:|---:|---:|---:|---:|
    | rel err | **8.30e-04** ✓ | 3.37e-03 | 8.29e-03 | 1.82e-02 ✗ | 1.30e-02 ✗ |

    Step 0 — identical weights, one forward+backward — agrees to **8.3e-4**, well
    inside tolerance. So the per-step computation matches; what fails is the
    20-step trajectory, because each optimizer step feeds slightly different bf16
    weights forward. **That is expected of any change perturbing bit-level
    arithmetic, correct ones included.**
  - **Recorded as a failure anyway, and the tolerance stays 1e-2.**
  - **FOLLOW-UP (job 7353316, `MODE=fixed`): compile fails the GENEROUS test too.**
    Added `AI_ROSSBY_EQUIV_MODE=fixed` — K forward+backward passes with **no
    optimizer step**, so weights are identical every iteration and step *i* cannot
    inherit from step *i−1*. Backward still runs (63% of step time, where the
    fusion is) and `grad_norm` is recorded. This separates the two hypotheses, and
    **both turn out to be real**:

    | step | 0 | 5 | 10 | 19 |
    |---|---:|---:|---:|---:|
    | train mode | 8.3e-4 | 8.0e-3 | **1.8e-2** ↗ | 1.6e-3 |
    | **fixed mode** | 8.3e-4 | 1.2e-3 | 8.9e-4 | 1.8e-3 |

    Compounding was genuine (fixed-mode loss error is flat at ~1e-3). **But a real
    per-step difference remains, which the compounding had masked:**
    `grad_norm` **5.33e-02** at identical weights, `surface.std` 3.02e-02,
    `surface.min` 1.34e-01 (**0.606 of the field's own σ**), `upper_air.min`
    6.53e-02 (0.400 σ). The aggregate loss agrees to ~0.1% while the output
    *extremes* differ by 0.4–0.6 σ — an averaged scalar hiding pointwise
    disagreement, which is exactly why §4.1 asks for output stats *alongside* the
    losses.
  - **Verdict strengthened: NOT adopted.** The test built to give compile its best
    chance still fails on quantities unrelated to trajectory amplification.
  - Consequently the procedural question (should §4.1 gate fusion changes on fixed
    weights rather than a training trajectory?) is **still worth settling on its
    own merits, but it is not what blocks this lever** — and it was never something
    to settle by adjusting a number until it passed (rules #1/#11).
  - **Also open:** this is a **world-size-1** capture. `torch.compile` is applied
    beneath the DDP wrap, so §4.1 additionally requires a 4-GPU baseline before
    adoption could even be considered.
  - The comparator's failure modes were unit-tested **before** it gated anything:
    identical and 0.5%-drift pass; 5% drift, seed mismatch, output-shape drift and
    3% stat drift are all correctly rejected.

- **2026-08-05** — **DESIGN §5 rung 1 MEASURED on ai-rossby: `torch.compile` = 1.40×.
  Getting there exposed a silent correctness bug that would have inflated it.**
  Commit `6972a940`.
  - **The bug, first.** `train_step` introspects the model's forward signature to
    decide whether to pass `train=True` + targets. `torch.compile` returns an
    `OptimizedModule` whose forward is dynamo's wrapper, so introspection answered
    **`eager=True` vs `compiled=False`** (measured). Compiling therefore dropped
    `train=True`, flipping `if self.checkpointing > 0 and train:` false —
    activation checkpointing silently OFF and a different forward path. A speedup
    measured that way times a *different computation* (rule #1), **and flatters
    itself**, since disabled checkpointing is faster. Fixed with a bounded
    `_unwrap_model` peeling `_orig_mod` then `.module` (`_orig_mod` first —
    `OptimizedModule` forwards unknown attrs, so `hasattr(m,"module")` misses it).
  - **The measurement** (jobs **7352022** eager vs **7352948** compiled; 4×A100,
    bf16, `default` mode, 40 warmup + 20 steps, `train/2015.zarr`):

    | | eager | compiled | ratio |
    |---|---:|---:|---:|
    | `step_med` | 449.6 ms | **320.8 ms** | **1.401×** |
    | `samples_per_s` | 8.90 | 12.47 | 1.401× |
    | `samples_per_s_wall` | 8.80 | 12.29 | 1.397× |
    | `peak_mem_gb_max_rank` | 24.98 | 21.07 | 1.186× |
    | `n_params` | 60,708,112 | 60,708,112 | identical ✓ |

  - **The profile predicted this.** nsys put `backward` at **63%** of the step
    (288.2 ms vs `forward_loss` 37.0 ms — a 7.8× ratio where ~2× is healthy) with
    `elementwise_kernel` holding the top three kernel slots: fusion-starved, which
    is exactly what compile fixes. The memory drop is the same effect from another
    angle — fused kernels stop materialising intermediates. **`default` mode, NO
    cuda graphs**; `reduce-overhead` may add more given ~62k launches/step.
  - **NOT ENABLED.** DESIGN §4 requires equivalence vs a captured baseline and §5
    leaves rung 1 unset until it exists — it still doesn't. Shipped as a
    measurement + correctness fix, not a config change. **The gate is now cheap:**
    the harness already runs a real `train_step` and merely discards the losses;
    recording them would both capture the repo's first `baselines/<model>/` entry
    and gate this lever. That is the next job.
  - **Login nodes are not a test environment.** The same suite returned `rc=0`,
    `rc=1`, and `rc=130`-at-scipy-import within one hour, then passed cleanly in
    isolation minutes later — pure contention. Added
    `polaris/polaris_recipe_tests.pbs` (compute node, gates on pytest's summary
    LINE not the exit code, since a killed run can exit misleadingly):
    **`RECIPE_TESTS_OK 13 passed`** (job 7353083, 46 s). The login node had been
    displaying 6 dots for a 13-test suite — truncated output that I twice read as
    a result. **A truncated capture with `rc=0` is not evidence, and one failure
    on a shared login node is not evidence of a code defect.**

- **2026-08-05** — **ai-rossby profiling harness built and GREEN; `n_params` overturns
  the standing explanation for the SFNO-vs-PanguPlasim speed gap.**
  Commits `b5f80060` (converter fix) → `84dbacf6` (harness).
  - **The measurement** (job **7352022**, 4×A100, bf16, 5+20 steps on `train/2015.zarr`):
    `step_med` **449.6 ms**, p90 449.9, std 0.27 ms, `samples_per_s` 8.90,
    `peak_mem` **24.98 GB**, `data_idle_frac` 0.0107, **`n_params` 60,708,112**.
  - **`n_params` was never logged before, and it inverts the story.** ai-rossby is
    **60.7 M params vs SFNO's measured 1,182,108,160 — 19.5× SMALLER** — yet only
    ~1.6× faster per step. So the gap was never SFNO's spectral transforms (the
    profile puts cuFFT/SHT at **3.3%**). A 19.5× smaller model gaining only 1.6×
    means **ai-rossby is badly under-utilizing the GPU.** That is now the first
    thing to profile, and it is on evidence rather than assumption.
  - **The old step time was ~19% pessimistic.** `LaunchLogger`'s 537 ms is an epoch
    *mean* including first-step warmup; the synced median is **449.6 ms**. Every
    wall-clock estimate derived from 537 ms is stale.
  - **Memory was never the constraint we assumed.** 24.98 GB of 40 GB at
    `batch_size=1` — ~15 GB headroom, and `checkpointing`/`embed_dim` were never needed.
  - **Shape.** `profile_train.py` is a *wrapper*, not a fork: it imports
    `build_model`/`build_datapipe`/`build_loss`/`make_optimizer`/`make_scheduler`/
    `train_step` from the real trainer, so it cannot drift (rule #4). `train.py` is
    unchanged; `train_loop.py` gains gated NVTX only. CSV's first 19 columns are
    s2s's, in order, with 4 appended (`samples_per_s_wall`, `data_idle_frac`,
    `config_sha16`, `n_params`).
  - **Verification, in full:** existing `test_train_loop.py`/`test_multistep_train_step.py`
    6 pass **before** the edit, 6 after with NVTX unset, 6 with `AI_ROSSBY_NVTX=1`;
    the new drift guard is **negative-tested** (renaming `backward`→`backwards` fails
    it with the exact diff, revert byte-identical); PanguWeather's own guard still
    `BENCH_INSTR_OK (10 tests)`.
  - **Two bugs caught during implementation, both by reading the real call sites:**
    (a) `make_optimizer`/`make_scheduler` need `_flatten_{optimizer,scheduler}_cfg`
    first — passing raw nested config silently builds the WRONG optimizer;
    (b) `multistep_train_step`'s rollout gets ONE `forward_loss` range, not K, or
    `parse_nsys.py`'s per-range median becomes meaningless.
  - **A 4-reviewer adversarial pass rewrote the plan first**, and killed a claim I had
    repeated several times: the **"1.51× faster" figure was apples-to-oranges**
    (SFNO's `Time taken for epoch` includes validation + checkpoint; ai-rossby's was
    train-only). Like-for-like it is **1.33×**, and even that carries four
    uncontrolled confounds — chiefly that **`checkpointing: 3` is NOT a matched
    setting**: graduated in SFNO (`>= 3` does strictly more recompute), a flat
    boolean in ai-rossby (`> 0`, so 1/2/3 are byte-identical). Also uncontrolled:
    `num_data_workers` 1 vs 8, and her step time drifting 702→832 ms (+18.5%).
  - **Converter landmine fixed** (`b5f80060`): `${VAL_YEARS:-2045}` defaults on
    *empty* as well as unset, so splitting the conversion across parallel jobs gave
    every train-only shard `val/2045` — concurrent racing writers on one store with
    `--overwrite`. `qsub -v VAR=" "` does not dodge it (PBS strips it to empty;
    verified with `od`). Fixed `:-`→`-` plus a tail-year guard, four cases unit-tested.

- **2026-08-05** — **Handoff-v2 Steps 1-3 executed: corrected normalization regenerated,
  norm zarr rebuilt, and the ai-rossby PanguPlasim training smoke is GREEN on 4×A100.**
  The whole point of the exercise, measured: **`TSOI_10CM` normalizes to spread 0.9968,
  was 0.122.**
  - **Step 1 — regeneration** (job **7340945**, exit 0, **34:54**). Full 51,100-file /
    ~2.15 TB moments pass. Measured vs the handoff's predicted values:
    `SST` **8.4407 / 12.0659** (predicted 8.44 / 12.06), `TSOI_10CM` **271.1259 / 16.3902**
    (predicted 271.09 / 16.39). The gate `check_normalization.py` passes **all 23 contract
    channels** at mean ≈0 / spread ≈1. The two fill warnings are the documented false
    positives (`PFTDATA_MASK` is degenerate — valid values are only ever 1; `SST`'s −1.8
    fill sits exactly *on* the data minimum, so the `valid_lo <= fill` test trips on a
    float boundary).
  - **Step 2 — norm zarr rebuilt.** Old store moved to `…zarr.prefix` (rollback path).
    Verified beyond the handoff's 2-field spot-check: **all 26 vars × all levels are
    bitwise-identical to the source `.nc`** (worst abs diff 0.000e+00). Old→new:
    `SST` 109.963/123.908 → 8.441/12.066; `TSOI_10CM` 105.229/133.802 → 271.126/16.390.
  - **Step 3 — training smoke** (job **7341412**, exit 0, **9:36**, 4 ranks, bf16,
    365 steps/epoch, 537 ms/iter). `PREFLIGHT_OK` + `PANGU_PLASIM_RUN_OK`. Loss finite and
    decreasing within the epoch (8.122e-01 → 6.949e-01 → 6.870e-01), epoch loss 8.410e-01,
    **val_loss 6.708e-01**. **All four "never executed" unknowns from handoff §5 are now
    resolved:** the model constructs (the `sol_in` patch holds on a real config), the
    tensors line up, **it fits in 40 GB with no lever pulled** (`checkpointing: 3` and
    `embed_dim: 240` were NOT needed), and the windowing divides.
  - **The window_size question was answerable statically** — no run needed. 18 levels → 9
    after the level patch, **+1 for the surface row = 10**, which the vertical window of 2
    divides. (`pangu_plasim_e3sm.yaml:65`'s comment reaches the right conclusion by wrong
    arithmetic — it stops at 9.) Horizontally the downsampled 45×90 is *not*
    window-divisible, but `Transformer3DBlock` applies `get_pad3d`/`crop3d`, so it pads.
  - **Cross-check against jesswan's shipped `.nc` — this is the real validation, and it is
    stronger than her regeneration would have been.** The defect touches 2 channels; the
    other 24 carry 0% NaN and so have no fill dependence, meaning her Jul 8 file is an
    *independent* reference for them. **18 of 26 variables reproduce her numbers** (most to
    ~1e-8). `U`/`V`/`RELHUM` appear changed only under *relative* error on near-zero means —
    absolute diffs are ≤0.15 on values up to 83, inside §3's ±1%. Note this tests our
    arithmetic against a differently-computed source, whereas a second regeneration by her
    would share our fill assumptions rather than test them.
  - **⚠ NEW, latent — cloud std floors differ.** Shipped `_std_corr` floored **16**
    near-zero levels to 1.0; ours floors **9**. Where hers underflowed to exactly 0 (and so
    got floored), our float64 shifted accumulation leaves catastrophic-cancellation
    residue: `CLDICE` ×4 at **2.03e-12** and `CLDLIQ` ×3 at 7e-28…6.6e-24 instead of 1.0.
    **Inert today** — `CLDICE`/`CLDLIQ`/`CLOUD` are commented out of
    `E3SM_SFNO_H5_POLARIS.yaml:52` and absent from the 108-field contract — but **anyone
    re-enabling those channels would divide by ~1e-12.** Deliberately NOT "fixed": the
    floor is a science call, and rule/trap #8 forbids raising `STD_ZERO` globally because
    `PRECT`'s real std is ~8.3e-8. Flagged for jesswan alongside the fills.
  - **Scheduling: Step 1 was rerouted to `debug`.** The handoff's job 7337234 sat queued
    ~5 h behind **106 preemptable jobs**. Measured the cost first (0.62 s/file × 51,100 ÷ 32
    workers ≈ 16 min), so it fits debug's 1 h cap — it ran in 34:54. Submitted under
    `TAG=dbg` so its outputs (`data_dbg_*.nc`) could not collide with the queued job's,
    which stayed queued as a fallback; results were promoted to the canonical names
    afterward and 7337234 was cancelled (state F, never ran). **Reusable trick:** the `TAG`
    env var makes a racing submission safe.
  - **⚠ The handoff's backup step had never been performed.** `$PANGU_AUX/pre_fix/` did not
    exist, so job 7337234 would have destroyed the only local copy of the pre-fix statistics
    (the ones §2's analysis and the 85-epoch checkpoint rest on). Backed up before anything
    else. Anyone re-running this: do that step first, literally first.
  - **jesswan's own regeneration still has not started** (checked 2026-08-05 ~03:00 UTC):
    every stats file in `$E3SM_ROOT` is still dated Jul 8 17:01, and her newest PBS job
    (`7322826`, `physicsnemo_e3sm_sfno_peryear`, ended Aug 4 17:19 **exit 1**) is a training
    job. Caveat: ALCF homes are 0700, so this is "no evidence", not proof.
  - **Stale-doc sweep (handoff §8).** `e3sm_h5_to_seqzarr.py` said TSOI land mean **268 K**
    in two places; measured is **272 K** — corrected (comment-only, no behavior change, so
    the makani coupling in rule #5 is unaffected). The other two §8 items —
    `verify_pangu_store.py`'s "six checks" and the CHANGELOG's `train.py:636,739` anchors /
    "NOTHING SUBMITTED YET" line — were **already fixed** in `3cf6c6c4`.

- **2026-08-04** — **SST-fill investigation (jesswan): the degC/Kelvin bug is confirmed, my
  "it doesn't affect training" framing was WRONG, and an audit shows the defect class is
  structural — it exists only where stats are precomputed externally.**
  - **Correction to my own earlier claim.** I argued normalization is affine-invertible so a
    first-layer weight absorbs the bad fill. **That holds in fp32 only.** AMP was on
    (`enable_amp` defaults `not args.no_amp`; no launcher passes `--no_amp`), and under bf16
    the ocean SST range collapses to **75 distinct values** (fp16: ~575). Quantization is
    irreversible — no downstream weight recovers it. jesswan is right: SST is a *feature*, so
    training and inference are both affected. Measured: ocean signal **0.093σ**, ~**0.47 °C
    per bf16 level**.
  - **⚠ R2 CONFIRMED INDEPENDENTLY, and it reframes the cost.** `SST`, `ICE` and `sol_in` are
    **bit-identical across all 35 years** at the same day-of-year (`max|diff| = 0` for 2015 vs
    2020/2035/2049), while prognostic `TREFHT` varies normally. So this archive has **no
    interannual SST variability and no SSP245 warming trend** — measured global-mean ocean SST
    trend is *exactly* 0.000 °C/decade. The quantization therefore costs no ENSO/trend signal
    (there is none); it costs **spatial + seasonal** structure — SST fronts, upwelling, the ice
    edge — and because the field repeats exactly, that blurring is a *systematic* forcing bias
    seen 35 times, not averaging noise.
  - **Audit (jesswan's ask) — physicsnemo_sfno and makani are CLEAN, and immune by
    construction**, via two different mechanisms:
    | path | SST fill | stats | exposed? |
    |---|---|---|---|
    | PanguWeather | 270 (degC field) | precomputed `.nc`/`.npz` | **YES** |
    | **ai-rossby (ours)** | 270 (parity) | precomputed zarr **+ `amp: bf16`** | **YES — worst** |
    | physicsnemo_sfno | −1.8 ✅ | **writes none**, online BatchNorm | no |
    | makani | −1.8 ✅ | **in-stream from PACKED data** | no |
    Both already document SST as degC and explicitly rejected the npz because it "was computed
    under a land-fill convention of ~270 for SST"
    (`e3sm_h5_to_seqzarr.py:242`, `convert_e3sm_to_makani.py:56-57`). **Nothing to change there.**
    ⇒ **The bug class exists only where an externally-precomputed stats file is paired with a
    separately-configured fill.** That is exactly PanguWeather and our ai-rossby path.
  - **Corrected constants are exactly derivable — no raw recompute needed.** Because the land
    mask is static AND SST repeats bit-identically (R2), the 35-year stats equal any year's,
    and the shipped 270-fill pair inverts analytically to ocean-only moments (land fraction
    0.37352, ocean mean 14.546 °C, ocean std 11.514 — cross-checked against a direct 12-sample
    measurement: 14.598 / 11.488). Re-filling gives:
    | SST fill | mean | std | ocean signal |
    |---:|---:|---:|---:|
    | 270 (current) | 109.9630 | 123.9083 | 0.093σ |
    | **0.0** | **9.1130** | **11.5140** | **1.000σ** |
    | **−1.8** | **8.4407** | **12.0659** | **0.954σ** |
    Either is ~10× the current signal and ~36× the bf16 levels. **−1.8 additionally unifies
    all three pipelines**; 0.0 is marginally better numerically. jesswan regenerates the
    PanguWeather `.nc`; these are the cross-check.
  - **Deferred by jesswan's call: "fix SST first".** `TSOI_10CM`'s mismatch stands —
    shipped stats encode a **0-fill** while the config fills **270**, measured over 12
    samples (0.195σ vs 1.606σ if matched = **8.2× under-weighted**). Unlike SST this channel
    is **scored** (a `land_variable`, i.e. prognostic), so it hits a loss term, not an input.
    Fix in the same regeneration pass when SST is settled.

- **2026-08-04** — **ai-rossby PanguPlasim bring-up: Step-0 variable-parity gate PASSED
  (10/10), then steps 1-6 implemented. Nothing submitted to the scheduler yet.**
  Commits `b28b2e7c` (gate) → `b10d5e5c` (venv + code + configs) → `2924b90b` (norm store +
  conversion tooling).
  - **The gate.** `ai_rossby_panguweather_variable_parity.md` proves the run trains the
    **identical 108-field variable set** as jesswan's PanguWeather E3SM run — group for
    group, name for name, **order for order**, level for level, fill for fill. The contract
    lives once in `ai_rossby_variable_contract.py::PLANNED` and is machine-checked two ways:
    `--check-ground-truth` (vs jesswan's YAML) and `--check-artifacts` (vs our configs,
    converter and store attrs). **26/26 pass.** stdlib-only so it runs on a login node
    before any venv exists — but needs `python3.12`, since the bare `python3` there is 3.6.
  - **Ground truth identified from run artifacts, not assumed:** the trainer logged
    `E3SM_SFNO_H5_STAMPEDE_jsw.yaml` at startup in 7 of the 10 `e3sm_train_*.e` logs, from
    jesswan's Stampede3 `$WORK` (`jwan4`), with a descending loss. It turned out not to
    matter: **all four** E3SM configs (`_STAMPEDE_jsw`, `_DERECHO_jsw`, `_POLARIS`,
    `_POLARIS_ALLDATA`) carry a byte-equivalent variable contract.
  - **⚠ Trap that would have been a FALSE PASS:** PanguWeather carries two level lists.
    `levels: [5, 10, 20, …]` are nominal hPa *labels*; `sigma_levels: [4.714998…, …]` are
    the values actually embedded in the H5 keys, and `use_sigma_levels: True` makes the
    latter what reaches the loader. The check compares `sigma_levels` — verified identical
    to ai-rossby's 18 levels, `max|diff| = 0.0`.
  - **ai-rossby's shipped E3SM defaults were NOT the contract** — 3 surface vars vs 6 (+2
    land), 1 diagnostic vs 3, and **two groups in a different order**. Order is the silent
    one: `ClimateZarrDataset` stacks tensors in **store-attrs** order while fills and loss
    build from the **model-config** lists (`dataset.py:533` vs `train.py:657` fill / `:735`,`:770` loss, via `_surface_channel_names()` at `:92`), so a
    permutation is correctly-shaped and `torch.cat` raises nothing. Hence the preflight.
  - **Code edits (vendored tree).** `sol_in` added to `_solar_names` in *both*
    `pangu_plasim_legacy.py` and `pangu_plasim.py` (E3SM's solar field is neither `rsdt`
    nor `toa_incident_solar_radiation`, so the ctor raised `ValueError`). One
    `_surface_channel_names()` helper in `train.py` replaces four restatements of "the
    surface tensor is `[surface|land|ocean]`" — the two sites the handoff named, **plus two
    it did not**: the `ArchesWeatherLoss` branch (off our `loss=mae` path, but a landmine
    for a later loss switch) and `channel_equal_weight`'s `n_surf`, which would
    under-weight the surface term against the channels it scores.
  - **Fills — the user's Kelvin confirmation, and where it does not hold.** `TSOI_10CM =
    270 K` is right (soil temp is Kelvin; land mean 268 K ⇒ effectively a mean-fill).
    **`SST` is degC in this archive** (measured −1.80…32.92), so the same 270 is ~8× its
    maximum — R4, inherited from a mechanical rename of ERA5/PlaSim's
    `mask_fill['sst']=270.` where the field really was Kelvin. **Reproduced deliberately**
    (parity is the point, and the shipped stats were computed under the same 270-fill, so
    fill and stats agree — unlike `TSOI_10CM`, R3). `SST = -1.8` + masked stats remains the
    fast-follow. **Both still need jesswan's sign-off before any numbers are reported.**
  - **Normalization store BUILT + verified** (login node, seconds):
    `$AI_ROSSBY_DATA/e3sm/norm/normalization_2015-2050.zarr`, 26 vars / 18 levels, from
    `$PANGU_AUX/data_2015-2050_{mean,std_corr}.nc` — the *same constants jesswan's run
    used*, converted not recomputed. Round-trip vs source over all 23 contract vars:
    `max|diff| = 0`. The `Z`→`Z_2` fix the handoff anticipated was **already applied**
    upstream by `polaris_prepare_e3sm_stats.py`; the tool's rename branch is a no-op here.
    No zero/tiny std among the 23 (that is what `_std_corr` corrects).
  - **Handoff corrections worth keeping:** (a) `uv` *does* ship with the ALCF conda module,
    but only after **`conda activate base`** — `module load conda` alone puts neither
    `python` nor `uv` on PATH. (a2) **The extras list was incomplete: `--extra
    datapipes-extras` is REQUIRED.** The fork promoted `xarray`/`zarr`/`netCDF4` to core
    deps but **not `dask`**, which `pangu_h5_to_zarr.py` imports to allocate the Zarr
    template. Caught by running the converter on 8 timesteps locally; otherwise it would
    have died `ModuleNotFoundError` on the compute node **after queueing**. The venv script
    now installs and verifies it. (b) The dataset config is a **new** file,
    `conf/dataset/e3sm_pangu_parity.yaml`, **not** an overwrite of `conf/dataset/e3sm.yaml`
    — that one is the SFNO speed-bench config for a different channel set. Launch with
    `dataset=e3sm_pangu_parity`.
  - **Venv:** `polaris_setup_ai_rossby_venv.sh` → `AI_ROSSBY_VENV_OK`. torch 2.10.0+cu129,
    zarr 3.2.1, torch_harmonics 0.9.1, wandb 0.27.0; physicsnemo editable from **this**
    checkout (13 GB, `${MEMBER_ROOT}/conda-envs/ai-rossby-venv`, via
    `UV_PROJECT_ENVIRONMENT` so it stays out of the git tree). `AI_ROSSBY_VENV` +
    `AI_ROSSBY_DATA` added to `polaris_env.sh` — **no shared fallback for the venv**,
    deliberately: it is an editable install, and the PBS scripts hard-fail
    `AI_ROSSBY_WRONG_CHECKOUT` instead.
  - **Preflight is real, and proven so.** `polaris_pangu_plasim.pbs` refuses to launch
    unless every produced store's attrs AND the model/dataset configs match the contract
    (`PREFLIGHT_OK`). Negative-tested by pointing `--store` at the normalization store:
    5 store checks FAIL loudly rather than no-op'ing. Also wired
    `dataset.nan_fill_strict` (the transform supported `strict` but `train.py` never
    passed it) — on for the smoke, so a NaN reaching the model raises at its origin
    instead of surfacing later as an unattributable NaN loss.
  - **Subtree:** `physicsnemo_ai_rossby/` imported **unsquashed** from jesswan's local copy
    (`ai-rossby`, HEAD `87002adb`, a real ancestor of HEAD). 4,037 files; `.git` 310 → 315 MB
    only, because it shares objects with the existing `physicsnemo_sfno/` subtree (both fork
    awikner/physicsnemo). `subtree add` ran fine on the **login** node with
    `-c pack.threads=1` — the compute-node workaround the physicsnemo import needed was not
    required here (local path, no network fetch).

- **2026-08-04** — **Pivot: training moves to the ai-rossby recipe (PanguPlasim), for exact
  PanguWeather parity — and the SFNO 103-var conversion `7324098` was cancelled + its 463 GB
  partial store deleted.** The SFNO `unified_recipe` v2 packed store can't feed ai-rossby (which
  needs a per-variable **zarr v3** store from its own `tools/data/e3sm/pangu_h5_to_zarr.py`), so
  finishing it bought nothing for the chosen path. Decisions (with the user): model
  **PanguPlasimLegacy** on the exact PanguWeather 108-field groups (SST/ICE prescribed);
  **normalization parity-first** (reuse shipped stats + reference fills `SST=270, TSOI=270`),
  masked-stats recompute + `SST=-1.8` as a fast-follow; ai-rossby vendored as a git subtree
  `physicsnemo_ai_rossby/` on `fix/tsoi-fill-270` with its OWN uv venv (zarr≥3, torch≥2.10);
  cluster Polaris. **Full plan + a variable-parity assertion gate (Step 0) → handoff
  `polaris_ai_rossby_pangu_handoff_prompt.md`** (adversarial + cold reviewed; caught the silent
  channel-order trap, the `sol_in` solar-name blocker, and the OOM launch flags). The TSOI 0→270
  code change (`aa43824a`) stands — consistent with ai-rossby's parity fills, still pending
  jesswan sign-off.

- **2026-08-03** — **TSOI_10CM ocean fill changed `0.0 → 270 K`, and a 103-var E3SM SeqZarr
  regeneration submitted** — branch `fix/tsoi-fill-270`, commit `aa43824a`. rmehta1987's
  decision after establishing that `0.0` is 0 K over the ~61% ocean (out-of-distribution — the
  ~268 K coastline cliff dominates BatchNorm's σ and starves the channel ~26× of gradient, R3),
  while **270** is mid-distribution (land mean 268 K) and matches PanguWeather's own
  `mask_fill['TSOI_10CM']=270.`. Because PhysicsNeMo normalizes online (BatchNorm), fill and
  stats can't disagree — the old "keep 0.0" reason (npz stats at 0-fill) doesn't apply here.
  - **⚠ SCIENCE CHANGE — needs jesswan's sign-off** before any resulting model's numbers are
    reported (DESIGN §1). Made explicitly + recorded (store `nan_fill` attr + here), not silently.
    `verify_seqzarr.py` imports `NAN_FILL`, so it checks the 270 placement automatically.
  - **Jobs:** verify `7324097` (debug, N=120 random draw + exhaustive bitwise incl. TSOI=270;
    PASS=`SEQZARR_VERIFIED`) → full `7324098` (preemptable, **HELD** on
    `-W depend=afterok:7324097`; runs only if verify exits 0). **Output →
    `/eagle/.../members/mehta5/e3sm_seqzarr_allyears_tsoi270`** (`e3sm_train.zarr` 2015–2046 +
    `e3sm_val.zarr` 2047–2049, 103+5 ch, ~1.43 TB, ~11 h). **Quota OK: 16.17 / 50 TB used** (the
    old 15 TB / 1.86 TB-free note below is stale — quota was raised).
  - **⚠ Operational:** the PBS jobs read the converter from the **working tree at runtime**, so
    keep the checkout on `fix/tsoi-fill-270` (TSOI=270) until `7324098` finishes; the produced
    store's `nan_fill` attr is the post-hoc proof it used 270. This regen is the field-set that
    **matches PanguWeather** (clouds excluded + all 7: PS/TMQ/RHREFHT/FSNT/FSNTOA/SOILWATER/TSOI).

- **2026-07-16** — **The runbook survived a 3-agent gauntlet (2 adversarial + 1 cold, Fable 5),
  which refuted four of its claims and surfaced one operational blocker; all fixed, plus the
  no-babysitting mechanics the owner asked for.**
  - **🔴 QUOTA (operator-sim agent): the eagle project quota is 13.14 of 15 TB used — ~1.86 TB
    free, SHARED project-wide** — while the full plan needs ~3.1 TB and the incoming zarr
    transfer itself needs ~1.43 TB of the same pool. Runbook now leads with a storage-budget
    section (one PhysicsNeMo dataset copy ever; `myquota` ≥1.5 TB free before any TB-scale job;
    freeing/raising quota is an owner decision that precedes the first big write).
  - **Walltime + chains: the "resume automatically" claim was wrong for walltime expiry**
    (`-r y` requeues only PREEMPTED jobs; a walltime kill just stops). Fixed three ways: the 3
    production launchers (Pangu full, makani alldata full, PhysicsNeMo allyears) go 24 h → **72 h**
    (queue max, verified `qstat -Qf preemptable`: max walltime 72:00:00, 20 queued/10 running
    per user); new **`polaris_submit_chain.sh`** pre-submits N `-W depend=afterany` links so a
    multi-day run needs zero monitoring (links resume from checkpoint; post-completion links
    exit in minutes). **Mechanics proven**: jobs 7259371/72 — link 2 held on link 1's afterany,
    clean chain qdel, zero node-hours. Also `polaris_verify_store.pbs` gains `-r y` (read-only
    idempotent ⇒ a preempted EXHAUSTIVE verify requeues instead of being silently deleted).
  - **Refuted by the fact-check agent, fixed in the runbook:** the "~27 min (measured)"
    validation figure (it is an extrapolation from 112.6 s at 9 ICs; now stated as such, budget
    20–30 min); "PNGs refreshed each epoch" (they ACCUMULATE — ~103 new files/epoch, ~51,500
    over 500 epochs; `model_package_*` is the opposite: ONE dir overwritten every save);
    the Route B success rule (a resumed conversion legitimately prints a skip line INSTEAD of
    `CONVERT_OK` — keying on it would fail a correct run; key on `ZARR_ALLYEARS_COMPLETE`);
    the headroom arithmetic (13.9 GiB is vs the RESERVED peak 25.6, not the 25.0 used).
  - **Misleading-as-written, fixed:** `PHYSICSNEMO_WRONG_CHECKOUT` protects only the
    PhysicsNeMo jobs — **makani has no checkout guard** and silently falls back to the shared
    venv (= the builder's working tree) if the per-user venv build is skipped; the venv build
    command must run from YOUR OWN checkout (it editable-links whatever tree it runs from);
    jesswan-hardcoded paths became `<you>` placeholders; "fails within a second" softened
    (module loads come first, and two-levels-deep dirs pass the polaris_env.sh check).
  - **Cold-agent findings adopted:** train/val splits are NOT aligned across pipelines
    (PhysicsNeMo trains through 2046 — years Pangu/makani validate on; only **2047–2049 is
    unseen by all three**) and epoch budgets differ ~30× — recorded as runbook §7.6: no
    cross-model skill comparison without a protocol from jesswan; per-run "finished" log lines
    added (Pangu `DONE ---- rank 0`, PhysicsNeMo `Finished training!`, makani = a later link
    exits immediately); makani's pack has NO value-level verifier (stated in its section with
    the same honesty as PhysicsNeMo's); debug re-runs serialize (~2–3 h for all four); the
    stale 162-ch/3.5-GB narrative in the Pangu full launcher's header got a correction banner
    (18.9 GB measured, 108 ch, 72 h, chain pointer).
  - **W&B online logging: PROVEN live** (previous entry's smoke 7259364 synced run `gywtrgsr`
    to wandb.ai through the ALCF proxy mid-training; `ALLDATA_SMOKE_OK` unchanged). The Pangu
    smoke now carries the full launcher's online block — the block itself had never executed
    anywhere before this.

- **2026-07-16** — **🟢 The §0 smoke sequence ran with the owner's go-ahead: ALL FOUR GREEN at
  the current contracts, same day.** Every prior green predated the change it would validate;
  that debt is now paid. Read each verdict from the log token, not rc:
  - **Pangu validation smoke (7259271): `PANGU_VAL_SMOKE_OK`** — the headline. The unproven
    `utils/metrics.py` CPU-climatology fix EXECUTED for the first time (all 4 ranks reported
    `clim_*=cpu`): validation peak **25.048 GiB**, train peak 25.127, device 39.495 ⇒
    **13.86 GiB headroom**, `valid_loss=0.6989` finite, full production leads [1,12,20,40,60].
    **The §4a adversarial arithmetic (38.8–40.8 GiB ⇒ requeue-forever) is REFUTED with the fix
    in place** — Pangu production training is cleared on memory. Planning number: validation
    112.6 s at 9 ICs ⇒ ~27 min/epoch at production's 129 ICs.
  - **Pangu 108-ch training smoke (7259296): `ALLDATA_SMOKE_OK`** — 105-in/101-out, peak
    26.98 GB, `step_med` 0.643 s (n=12; the trustworthy number remains 0.602 s at n=80).
  - **PhysicsNeMo allyears smoke (7259303): `ALLYEARS_SMOKE_OK` + `PHYSICSNEMO_CSV_OK`** —
    seam stores bitwise-verified, 103+5 schema pinned, and the §6 CSV tee is now proven
    END-TO-END: 7 rows exactly as designed (5 minibatch, decreasing loss 1.518→1.330; 1 epoch
    row lr=1e-3 + GB/s; 1 validation row 1.883), git_sha stamped.
  - **makani 100/1/7 smoke (7259321): `ALLDATA_SMOKE_OK`** — fresh pack `CONVERT_ALLDATA_OK`,
    contract + names verified against the converter, real training (6.79 s) + checkpoint, and
    the benign `N_in_channels=107 ... expected 58` watchdog warning appeared exactly as
    designed. The stale 154-ch pack was MOVED (not deleted) to
    `${MEMBER_ROOT}/data/e3sm_makani_alldata_smoke_154ch_stale` — delete when convenient.
  - Also: `polaris_pipeline_runbook.md` added — the operator-facing guide (plain scientific
    language, full paths) for running the PhysicsNeMo pipeline off the transferred dataset
    (Route A: verify in place) or a rebuild (Route B). §0/§5 of the plan and the runbook's §2
    table updated with these results.

- **2026-07-16** — **The three-pipeline runbook is written (`polaris_pipelines_plan.md`) and the
  handoff's §6 tee + two gap-closing scripts are implemented. Nothing was submitted** (per the
  handoff: design, don't launch). Method: two Fable-5 agents swept PanguWeather and makani code;
  PhysicsNeMo read directly; every command in the plan carries its PASS token and what the gate
  cannot catch.
  - **§6 PhysicsNeMo CSV tee: DONE.** New `examples/weather/unified_recipe/bench_csv.py` +
    5 delimited vendor-divergence blocks in the vendored `train.py` (subtree-pull conflict
    surface is those blocks; a tripwire test greps for them). Frozen 10-column schema
    (`timestamp,epoch,step,loss,lr,gb_per_s,valid_error,n_gpus,git_sha,run_name`), rank-0 only,
    env-gated (`PHYSICSNEMO_BENCH_CSV`, unset ⇒ byte-identical no-op). **No per-step GPU sync
    added**: the loss is `.clone()`d (CUDA-graph static-buffer aliasing — N unclone'd refs would
    all read the last step) and buffered; conversion happens on a 100-row flush, the cadence at
    which LaunchLogger already syncs by string-formatting the tensor. Proven:
    `BENCH_CSV_OK (23 tests)` (login node, stdlib-only test). End-to-end proof wired but
    pending a job: the allyears smoke now also requires `PHYSICSNEMO_CSV_OK` (a real minibatch
    row) and the allyears launcher writes `${RUN_DIR}/metrics.csv` by default.
  - **`physicsnemo_sfno/polaris/polaris_verify_store.pbs` (new): verify an EXISTING SeqZarr
    store at any path** — built for the store jesswan is transferring in (not present yet,
    checked today). Three layers: attrs/generation vs the converter's current lists; **exact
    chunk-key set per array** (deliberately not `nchunks_initialized`, which counts `.partial`s
    by prefix); stride-sampled bitwise round-trip vs the h5 archive (auto-stride ≈480 samples,
    nudged off divisors of 1460 so it never re-samples the same calendar dates; `EXHAUSTIVE=1`
    escape for the full read). Attrs+chunk layer tested against the on-disk old-generation
    store — correctly refused it (`STORE_INCOMPLETE` + `STORE_WRONG_GENERATION`). The PBS has
    never been submitted. Note recorded while testing: **the stores at
    `${MEMBER_ROOT}/e3sm_seqzarr/` are pre-exclusion 157+5 smoke-scale relics** (CLDICE in
    attrs, no sentinel) — only the tracked 162-era launchers may use them.
  - **`makani_sfno/polaris/polaris_sfno_alldata_full.pbs` (new, NEVER RUN):**
    `e3sm_alldata_full.yaml` existed with **no launcher** (the "2 pbs" were pack-full +
    train-smoke — a gap the handoff missed). Mirrors the green locked launcher + the alldata
    smoke's derived contract gates. Its header carries the blocker in bold: **a 100/1/7
    checkpoint cannot currently be evaluated** (`sfno_inference/checkpoint_loader.py:74-82`
    hard-asserts 52/1/6; stock makani inference is hard-gated off) — decide the eval path
    before burning 100 epochs.
  - **Handoff corrections found by verification** (details in the plan §5): Pangu
    `inference.py` cannot run E3SM-SFNO at all (nettype gate + a checkpoint layout train.py
    never writes) — `ensemble_inference.py` with h5-derived ICs is the only path runnable from
    what is staged; the bias-`.npy`/`long_validation` caveat binds **train.py**, not the
    inference scripts; train.py's `--amp_dtype` help text says fp16 but the code default is
    bf16 (`train.py:265`) while both inference scripts default fp16 — a train/infer precision
    asymmetry to decide before comparing numbers; Pangu's full/smoke PBS headers still carry
    stale 162-ch/3.5-GB comments (flagged, deliberately not edited this session).
  - **Deliberately NOT done:** no Pangu inference PBS wrapper (the h5-IC ensemble path has
    never been exercised anywhere — the plan gives the exact first interactive command instead
    of encoding guesses into a script); no edit to `polaris_env.sh`'s `_pick` (the allyears
    smoke header records that as a deliberate stay-away; new scripts use their own env names).

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
