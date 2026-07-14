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
| §4.0 prerequisites (seed knob, tiny config, VAE noise-fix) | ⬜ not started — **blocks baseline capture** |
| Correctness baselines captured (DESIGN.md §4) | ⬜ not started — **blocks all optimization** |
| Test harness (tier-1 equivalence/unit + `--fast`) | ⬜ not started |
| Optimization ladder (DESIGN.md §5) | ⬜ not started |

### Smoke status matrix (probe → 1-GPU → 4-GPU)

| Model | Midway | Polaris |
|---|---|---|
| Toolchain probe | — | ✅ `PROBE_OK` (job 7251974: 4×A100-40GB; the 4 in-repo models import — makani/physicsnemo need the §6 venv, non-blocking) |
| S2S (`torchrun`) | ✅ runs (Midway scripts GREEN) | ⛔ blocked on ERA5 stage (scripts ready) |
| S2S-Lightning | ⚠️ standalone smoke config-path fixed 2026-07-13 — **needs a Midway run to reconfirm** | ⛔ blocked on ERA5 stage (scripts ready) |
| SI | ✅ runs (Midway scripts GREEN) | ✅ **4-GPU GREEN** (7252700: step_med 0.400 s, peak 30.98 GB) **and reproducible by a SECOND USER** — job **7253603** (`PYTHONNOUSERSITE=1`): step_med 0.399, peak 30.69 GB, rc=0 |
| PanguWeather SFNO | — | ✅ **4-GPU GREEN** (7252271) **and reproducible by a SECOND USER** — job **7253591** (`PYTHONNOUSERSITE=1`) rc=0 with loss **0.3411, identical** to the as-installer run |
| Makani SFNO | — | ✅ **4-GPU GREEN** (job **7253465**, current script: train loss 2.61 / val 2.38 + ckpt; first green 7252769 pre-rework; pack `CONVERT_OK` 7252728) — runs from the isolated SFNO venv |
| PhysicsNeMo SFNO | — | ✅ **4-GPU GREEN** (job 7252933, rc=0: loss 0.889, val err 0.541, ckpt saved; 1-GPU 7252816 also green; zarr `CONVERT_OK`) |

## Next actions (pick from the top)

1. **Polaris bring-up** — probe → 1-GPU → 4-GPU smoke for each model via PBS;
   write `polaris_pbs_notes.md`. Follow `polaris_handoff_prompt.md` (on `main`).
2. **Build the §4.0 prerequisites** — a `--seed` knob in `s2s/v2.0/train.py`, a
   `tiny_baseline.yaml`, and a VAE noise-fixing hook. Nothing can be optimized
   safely until the equivalence gate is executable.
3. **Capture correctness baselines** (DESIGN.md §4) for each model.
4. **Stand up the test harness** — CRPS/KL numerical checks, normalize↔inverse
   round-trip, tiny-model forward/backward, a `conftest`-registered `--fast`.
5. Then start the optimization ladder (torch.compile first — enable the existing
   `TORCH_COMPILE_MODE` plumbing), one gated commit each.

## In progress

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
  GLOBAL, so it must divide the rank count. Plus an **upstream makani bug** (pin
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
