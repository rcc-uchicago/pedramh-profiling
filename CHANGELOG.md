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
| Polaris (PBS) bring-up | ✅ **all 4 runnable models GREEN on 4×A100** (PanguWeather-SFNO, SI, Makani-SFNO, PhysicsNeMo) + probe + all 3 data converters proven on real data. S2S/port scripts delivered but blocked on an ERA5 Globus stage. See `polaris_pbs_notes.md`. |
| §4.0 prerequisites (seed knob, tiny config, VAE noise-fix) | ⬜ not started — **blocks baseline capture** |
| Correctness baselines captured (DESIGN.md §4) | ⬜ not started — **blocks all optimization** |
| Test harness (tier-1 equivalence/unit + `--fast`) | ⬜ not started |
| Optimization ladder (DESIGN.md §5) | ⬜ not started |

### Smoke status matrix (probe → 1-GPU → 4-GPU)

| Model | Midway | Polaris |
|---|---|---|
| Toolchain probe | — | ✅ `PROBE_OK` (job 7251974: 4×A100-40GB, all imports) |
| S2S (`torchrun`) | ✅ runs (Midway scripts GREEN) | ⛔ blocked on ERA5 stage (scripts ready) |
| S2S-Lightning | ⚠️ standalone smoke config-path fixed 2026-07-13 — **needs a Midway run to reconfirm** | ⛔ blocked on ERA5 stage (scripts ready) |
| SI | ✅ runs (Midway scripts GREEN) | ✅ **4-GPU GREEN** (job 7252700; converted E3SM; step_med 0.400 s, peak 30.98 GB) |
| PanguWeather SFNO | — | ✅ **4-GPU GREEN** (job 7252271; E3SM data) |
| Makani SFNO | — | ✅ **4-GPU GREEN** (job 7252769; train loss 2.19 / val 2.05 + ckpt; pack `CONVERT_OK` 7252736) — runs from the isolated SFNO venv |
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

- **SQG online-training profiling (jobs A/B/C, 2026-11-02)** — scripts in `SQG/analysis/`, Midway env =
  Weidong's `wdenv` (jax 0.4.28, torch 2.3.1, timm 0.9.12) + `module load cuda/12.9` (nsys/NVTX; the harness logs
  which cuFFT/cudart JAX loads since the module prepends LD_LIBRARY_PATH). No training data needed: A/B use one AB3
  checkpoint (`AllTrainData/...m01.../checkpoint/ckpt_t0100.00.npz`), C uses random input.
  - **A** `profile_solver_midway.sh` (solver ms/step, dispatch blocking, HLO op counts, donate / fixed-AB3-level
    variants checked bitwise, nsys) — **queued as job 62088260**.
  - **B** `profile_overlap_midway.sh` (does the solver overlap the UViT step? serial vs background thread, pyqg AB3 vs
    fixed level) and **C** `profile_uvit_midway.sh` (torch.profiler of the UViT step) — built, CPU-smoked (C), **not
    submitted**.
  - **Polaris route: DONE 2026-11-03** (branch `sqg-polaris-profiling`, draft PR against `sqg`; see the dated
    entry below and `SQG/polaris/polaris_sqg_notes.md`). All three jobs PASS on Polaris debug, n = 2 each.
  - **Blocked on GPUs:** pedramh-gpu midway3-0423 is fully held by Weidong's no-time-limit generator job; midway3-0608
    is reserved (`node_test`, user ndtrung) until 2026-11-04 00:00. The `gpu` partition is V100 (not representative).
  - Hypothesis under test (two independent Opus reviews): pyqg_jax `AB3Stepper.apply_updates` uses a runtime
    `lax.switch` on `_ablevel` (steppers.py:478) that may sync the host every step, which would serialise solver
    and network (2.6 s ≈ 2.2 + 0.4). Also found by review, no profiling needed: the host FIFO stores every frame twice
    (137 → ~69 GB/rank possible with identical samples); o1.2 cannot run on Midway as-is (torch 2.3.1 lacks
    `torch.backends.fp32_precision`; fixed_ring `DT_SAVE=0.0025` breaks o1.2's LAG assertion; hard-coded paths).

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
  **RESULT: Makani pack GREEN (`CONVERT_OK`, 7252736) and Makani SFNO 4-GPU smoke GREEN
  (7252769: train loss 2.19, val 2.05, checkpoint written, rc=0).**
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

- **2026-11-04** — **SQG job D: per-phase sync audit of the online optimizer step, tooling done, job 7813332
  submitted** (`debug`, 1 A100, 40 min; branch `sqg-polaris-profiling`; notes Part V). Jobs A/B counted syncs
  for the solver alone. D attributes every CUDA API call, and the GPU work it issued, to the NVTX phase open on its
  thread (`SQG_NVTX=1` in `profile_overlap.py`). Every sync and DtoH is also attributed to the XLA op that issued it
  (`summarize_nsys_phases.py`, stdlib sqlite3; `PHASES_TEST_OK`). **Validated on job A's capture:** 480.0
  `cuStreamSynchronize` + 480.0 4-byte pinned DtoH per 480 steps, all inside `conditional.277.0
  (while/body/cond)`, equal to A's totals. Five predictions are pre-registered (`flush` = 24 pageable DtoH + 24
  syncs/step, `loss_item` = 2, ML phases 0, …). Also found while explaining the solver: XLA's "68 GFLOP/step" is
  its FFT cost model, 8·(log₂4096)² flops per element (3 IRFFT × 8.4 M + 2 RFFT × 16.8 M = 67.6 GFLOP), not real
  arithmetic (≈5 GFLOP textbook). The cuFFT kernels run at ≈1.3 TB/s, close to the HBM limit, but
  `loop_add_multiply_fusion_1` (3.27 ms, 39.5%) is far below it. Nsight Compute on that kernel is the next solver
  lead. Next: read 7813332 (`PHASES_OK` ×3), score Part V.
- **2026-11-03** — **SQG online-training profiling ran on Polaris (jobs A/B/C, n = 2 each, `debug`, 1 A100 per
  job)** (branch `sqg-polaris-profiling`; write-up `SQG/polaris/polaris_sqg_notes.md`; summaries in
  `SQG/analysis/results/polaris_*`). Code findings only, with no cross-GPU ratios:
  - **The `lax.switch` hypothesis: mechanism confirmed, consequence incomplete.** `ref` does 1 DtoH + 1
    `cuStreamSynchronize` per solver step, so Python is blocked for every 40-step call. The `fixed` AB3 level is
    bitwise identical and has 0 syncs, but the host still cannot run more than ~1 call ahead: ~1,000 launches per
    call fill the CUDA launch queue (`cuLaunchKernel` 414 µs vs 2.7 µs).
  - **Overlap is limited by the GPU, not the host.** `serial_ref` = solver + ML exactly (4.887 s). A background-
    thread solver hides at most 21% of the ML step, because the solver alone keeps the GPU 99% busy. The best arm
    is −1.4%. `sample()` is 0.3% of the ML step.
  - **~5 × 134 MB loop-carry D2D copies per solver step** = 10.6% of solver time; `donate` removes a different
    3 × 134 MB copy per call.
  - **UViT:** batch 32 OOMs a 40 GB A100, so ML numbers are at batch 16 (416 ms/step). Kernel time is spread
    (GEMM 26%, elementwise 20%, native depthwise conv 19%, LayerNorm 15%; 3,562 launches/step). `channels_last`
    is +38.5%, and `cudnn.benchmark` −0.3%.
  - Of the 6 pre-registered predictions, #2, #3 and #5 are confirmed in full. #1, #4 and #6 are confirmed in part;
    the refuted sub-claims are #1's `fixed` half, #4's `thread_fixed` ≈ max, #6's "few kernels dominate" and #6's
    `channels_last` < 15%. The scorecard is in the notes. Ranked code changes with their
    gates are in the notes' Part IV. None are implemented.
  - **Env:** `module load conda` works again (ALCF fixed the modulefile 2026-09-16; makani jobs already log
    `MAKANI_ENV_SOURCE = module-conda`). SQG has its own `$MEMBER_ROOT/conda-envs/sqg-env` (py3.10, pip copy of
    wdenv; CUDA from the wheels, no modules at run time; `SQG/polaris/setup_sqg_env.sh`). The login-node import
    check was replaced by a GPU probe job (09-24 ruling).
  - **Gotchas:** `profile_uvit.py`'s `top_ops` percentages double-count (ops + kernels + ranges); use
    `summarize_uvit_trace.py`. `SQG/analysis/results/` is gitignored, so add results with `git add -f`. The brief's
    default `MEMBER_ROOT=/eagle/lighthouse-uchicago/members/$USER` doesn't exist for every user (rmehta1987 →
    mehta5).
  - Optional NUMA arm not run: the step is GPU-bound. Jobs: probe 7809334, C 7809340 (OOM @32) / 7809341 /
    7809408, A 7809356 / 7809412, B 7809385 / 7809433.

- **2026-11-02** — **SQG Polaris `debug`-queue profiling prompt** (`SQG/polaris/PROFILING_PROMPT.md`, commit
  41570a38). Self-contained handoff for a Polaris session to port jobs A/B/C to PBS. Scope is the **code**: solver
  host sync, copies, solver/UViT overlap. Not a GPU comparison (no A100-vs-H100 ratios), not data loading.
  Nothing has run on Polaris yet.
  - **No checkpoint needed:** `SQG_INIT=random` starts from the solver's own IC and runs `advance(…, 2)` to reach
    AB3 level 2. DT is fixed and the CFL checks are host-side, so every step costs the same 4096² FFTs whatever the
    field values. The default stays `checkpoint` (Midway behaviour).
  - **Vendored `SQG/fixed_ring/{A-SQG_data.py, sqg.py, SHA256SUMS}`**, byte-identical to Weidong's Midway
    `fixed_ring/` (≠ the older `SQG/A-SQG_data.py`). Found while writing the prompt:
    - The solver hashes its own bytes and `sqg.py`'s into `CONFIG_HASH`, so editing it, even its
      hard-coded `TRAINING_ROOT` (line 76), breaks checkpoint loading. The harness must patch that line in memory.
    - It `mkdir`s under `TRAINING_ROOT` at import.
    - It raises `JAX is not using the allocated GPU` on a CPU import (line 272), so a login-node import test is
      impossible.
  - **Env:** `SQG/polaris/requirements-sqg.txt` (top-level pins) + `requirements-wdenv.txt` (full wdenv pip freeze,
    used as constraints). Dry-run resolved against PyPI (cp310) to wdenv's exact CUDA stack: cuDNN 8.9.2.26, cuFFT
    11.0.2.54, nvcc/nvjitlink 12.9.86. Use `jax[cuda12]`; `jax-cuda12-plugin[with_cuda]` is not an extra in 0.4.28,
    and the nvcc wheel (ptxas) is easy to drop.
  - Batch size is a knob (`SQG_BATCH`/`SQG_BUF`; halve on OOM). B's `sample()` gather+H2D gets its own timer. The
    prompt's fallback table stops only on unpushed files or no GPU allocation.

- **2026-11-02** — **SQG decorrelation on the full 4096² grid vs the 1024²/256² views** (`decorrelation.py
  --factors 1 4 16`, caslake job 62086140 under pi-pedramh, `DECORR_OK`, 104 windows, 7 min; results
  `SQG/analysis/results/decorrelation_multigrid*`, figure `_bands.png`).
  - Whole-field numbers are identical on all grids (θ C(0.02)=0.963, e-fold 0.69; increment C(0.0025)=0.92,
    C(0.02)=−0.066), because θ variance is 89% at |k|<8.
  - **Per band, 256² == 4096² up to |k|=64.** The fast small-scale decorrelation is real physics, not an aliasing
    artifact. Aliasing only shows in the 256² top band |k| 64–128: its increment variance share goes 17.1%→20.5%
    and C(0.0025) 0.85→0.81. The 1024² view equals the full grid up to its own Nyquist.
  - **The target is small-scale:** 70% of the increment variance is at |k| 16–128 (e-fold 0.007–0.013 < Δt). Only
    23% is at |k|<16. Beyond |k|≈128, increments one step apart are at the −0.50 uncorrelated-differencing limit.
  - **Sweeping:** from |k|≈16 up, e-fold ∝ 1/k with 1/(k_mid·τ) ≈ 1.3 constant (1.23, 1.35, 1.31, 1.26, 1.26 for
    the 16–32 … 256–512 octaves). Small eddies are advected by a ~1.3 large-scale velocity, i.e. moved ≈0.026 ≈ one
    256² grid cell per Δt and ≈1% of a cell per online interval Δt/100. Deterministic and in principle predictable
    from the large scales of θ itself, so this is *not* an irreducible floor. (This corrects the 256²-only entry below.)
  - The genuinely unrepresentable part at 256² is |k|>128: 7.3% of increment variance, vs val MSE ≈ 5.6% of
    persistence (0.0043 / ≈0.076). Same order, so it is consistent with a resolution floor but not proof.
    Next: per-band val error of the trained UViT.
  - Caveat: e-folds for |k|≥512 come from one lag point (below the 0.0025 frame spacing).

- **2026-11-02** — **SQG online-training efficiency: temporal decorrelation measured** (`SQG/analysis/decorrelation.py`,
  caslake CPU job 62084323, `DECORR_OK`, 104 windows; results in `SQG/analysis/results/decorrelation_full*`).
  Context: online run `o1.2` costs ~142 h vs ~21 h offline, ~120 h of it solver time that re-integrates the 3500
  archived time units, sampling a pair every Δt/100 = 2e-4 (STRIDE 40 × DT 5e-6). Read-only from
  `/project/pedramh/weidong/SQG/SQG_deterForcing/{trainingdata (0.02 archive, 7 members), AllTrainData (0.0025 dense,
  4 members)}`, on the training view θ[::16, ::16]:
  - θ (network input): C(0.0025)=0.997, C(Δt=0.02)=0.963, C(1)=0.25, e-fold **0.69** (≈ τ_f 0.67). Plateaus at
    C≈0.06–0.08 beyond 5 units (probably the time-mean pattern of the frozen forcing; not verified), so use the e-fold
    time, not the integral time.
  - 0.02 increment (the residual target): C(0.0025)=**0.917**, C(0.005)=0.77, C(0.01)=0.47, C(0.02)=**−0.07**
    (consecutive offline targets are already uncorrelated). Below 0.0025 is extrapolated: a 1−C=aτ^p fit on
    0.0025–0.01 (p≈1.35) gives 1−C(2e-4)≈**3e-3** for targets and 1e-4 for inputs. That is an upper bound on novelty,
    since smooth-in-time fields go to p=2.
  - By band (256² grid): |k|<4 e-fold ≈1.1; |k|>32 decorrelates within Δt (θ C(0.02) 0.04–0.27; increment C(0.02)
    −0.35 to −0.48, close to the −0.5 signature of differencing uncorrelated frames). *Superseded by the 4096²
    multigrid entry above:* this is real physics (sweeping), not an aliasing artifact, and linear decorrelation ≠
    unpredictability.
  - Meaning: neighbouring Δt/100 pairs are near-duplicates (input changes ~1%, target ~3–8%). New *input* states cost
    ~0.7–1.4 time units of solver (~10–20 GPU-min each); the archive holds ≈2500 of them, and o1.2 re-generates the
    same ones. Online sampling at Δt/10–Δt/4 keeps most target diversity while spanning 10–25× more time in the same
    buffer. Withdrawn: sampling only every Δt (would drop intermediate targets with C_incr 0.5–0.9).

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
  run crossing a leap year needs a loader fix. — `polaris_pbs_notes.md` §8.
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
- **A100 (Polaris) memory** — the "40 GB" figure is from prose, not an on-node
  `nvidia-smi` (`si/bench_midway_notes.md` fn2); confirm during Polaris bring-up.
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
