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

- **2026-11-05** — **SQG Nsight job 7819082 DONE (`NSIGHT_DONE failures=0`), notes Part X.** Results in
  `SQG/analysis/results/polaris_nsight_lean_7819082/`. Predictions: 14 confirmed, 6 half, 1 refuted.
  - **`lean32` 3.198 ms/step, by op:** AB3 fusion 926 µs (28.8%), 2 RFFT 615, C2C 540, IRFFT 503, products 253,
    `build_w` 235, copies 99. GPU busy 99.4%. Ideal 2.82 GB/step ≈ 2.09 ms at 1.35 TB/s.
  - **The AB3 fusion over-fetches 2.2–2.9×** (4 elements per thread; ncu: 75% excessive sectors). **Seed 1's cos
    trigger fixes it:** the fusion reads exactly its 369 MB, at 89% of peak.
  - **Bitwise rewrites of `lean32`:** trigger **2.641 ms (1.211×)**; 3-slot ring 3.104 (0 copies); both
    **2.524 ms = 3.68× production**.
  - **cuFFT column passes run at 41–52% of DRAM peak** (uncoalesced global on the half-plane, shared bank
    conflicts on the full plane). Only a custom FFT reaches them (≈ 0.43 ms).
  - **cuFFT 11.4.1 via `LD_PRELOAD`:** confirmed mapped, no change (< 1%). **Not a lever.**
  - **Float64 `lean_u1` 6.40 ms:** 3 IRFFTs = 47%. Next is `lean64`'s C2C packing (≈ −0.9 ms) + the trigger on its
    last 4-element fusion (≈ −0.15) → ≈ 5.3–5.5 ms.
  - Ranked list: notes Part X. `lean32_u1_ring3`'s precision question is still Part XI (job 7819310).
- **2026-11-05** — **SQG float32 vs o1.2's training delta: two mechanisms found; delta-fidelity job built**
  (`SQG/analysis/opt/delta/`). The user: "we need to be sure that fp32 is not impacting δ as it is very small".
  Round 1b compared frames, not deltas.
  - **Damping quantization.** `decay` stored in float32 quantizes each mode's damping rate, because rate·DT ≈
    1e-7..5e-6 and float32 spacing below 1 is 6e-8. Rate error: up to ±20% at k = 4–16 (rms 11–13%), ±5% in
    the forcing ring, 0.1% at k = 1. It is deterministic per mode, applies every step, and is present in plain
    float32 too.
  - **Sub-ulp rounding.** A float32 state rounds sub-ulp per-step changes systematically wherever the increments
    vary slowly, i.e. the large scales, where Δ is smallest (k = 1–2: Δ ≈ 4% of θ).
    `test_f32_update.py` (emulated float32, one forced k = 8.4 mode, 100k steps): the 4,000-step delta is
    **31% off** for `lean32`'s update and **also for expm1 damping alone**. The compensated (TwoSum hi + lo)
    update gives 8.8e-8. So any float32 *state* has this; the fixes are a float64 state (`mixed`) or a
    compensated float32 state (`lean32ek`, +1 complex64 array).
  - **The GPU test** (`delta_fidelity.py`, `run_delta_polaris.pbs`):
    - spin-up of a synthetic −5/3 field (no archived checkpoint on Polaris);
    - per-band increments;
    - 8 × 4,000-step pairs on o1.2's coarse frames for `mixed` / `lean32` / `lean32e` / `lean32ek` vs float64,
      with a float64-from-rounded-state floor.

    Pre-set gate: e_Δ ≤ max(2 × floor, 1e-6), overall and in each band holding ≥ 1% of Δ energy.
    Predictions: `PREDICTIONS_delta.md` (written first). It is auto-submitted when 7819082 starts (debug allows
    one waiting job per user).
  - **Until it reports: do not ship `lean32` to o1.2.** `lean_u1` (float64, bitwise) is unaffected.
- **2026-11-05** — **SQG Nsight job on `lean32` submitted: 1-node debug job 7819082**
  (`SQG/analysis/opt/nsight/run_nsight_lean_polaris.pbs`, walltime 0:55). The user asked for an Nsight analysis of
  `lean32` and where the solver can still improve. End token `NSIGHT_DONE failures=N`; results →
  `SQG/analysis/results/polaris_nsight_lean_7819082/`. Predictions: `SQG/analysis/opt/nsight/PREDICTIONS_nsight.md`
  (written first). Don't edit `SQG/analysis/opt/nsight/` or the two new summarizers while it is queued.
  - **New tools.**
    - `summarize_nsys_ops.py`: GPU time per XLA op per step, via the XLA thunk NVTX ranges. Validated on job E's
      real f32 capture: AB3 fusion 649.8 µs/step (job E: 650), GPU busy 98.1%.
    - `summarize_ncu_window.py`: every kernel of one loop period, from an unfiltered ncu window.
    - `lean32_probes.py`: exact rewrites of `lean32` (seed 1's cos trigger, seed 1's 3-slot ring, both).
  - **Leads going in.**
    - `lean32`'s byte model is ≈ 2.8 GB/step ≈ 2.1 ms at 1.35 TB/s, vs 3.19 ms measured.
    - Its HLO still has 2 full-size tendency copies per 2-step iteration.
    - Job E's half-plane column FFT pass (`regular_fft_factor<4096>`) runs at ≈ 0.68 TB/s, half its neighbours.
      Hypothesis: a 2,049-wide complex64 row is 16,392 B, so rows are not 32 B-aligned.
    - `decay` is an exact circular 2/3 truncation, so most of the half-plane is zero every step.
    - CUDA 12.9.1's cuFFT 11.4.1 (the env has 11.0.2) can be swapped in by `LD_PRELOAD`: XLA dlopens
      `libcufft.so.11` by soname. Phase 3 is report-only.
- **2026-11-05** — **SQG round 1c (7815637) DONE, `OPT_R1C_DONE failures=0`.** Notes Part IX; summaries in
  `SQG/analysis/results/polaris_opt_round1c_7815637/`.
  - **Seed 1: all three pass, bitwise at 1/40/480/4,000 steps, `t` exact.** `lean` 7.970 ms (1.165×),
    **`lean_u1` 6.400 ms (1.451×; optimizer step 4.470 → 3.086 s)**, `lean_split` 7.078 (1.312×). So `lean_u1` is a
    bitwise float64 speed-up that needs no science sign-off. The run-time `dt` fixed `t` in both seeds.
  - **NCCL over CXI is solved.**
    - A (7815096's env + FI_LOG) fails again, and FI_LOG shows `cuda_hmem_dl_init(): Failed to dlopen
      libcudart.so` on 8/8 ranks. That confirms the cause.
    - **B, a dir holding `libcudart.so` → the env's own `libcudart.so.12`, passes at 31.5 GB/s:** GDRDMA on 24/24
      NET lines, CUDA HMEM up 8/8, 0 errors.
    - C (`GDR_LEVEL=LOC`) passes at 17.2 GB/s, host-staged; combine_handoffs' 2-node 18.0 GB/s was GDR 0.
    - D (`cuda-12.9.1` prepend) passes at 32.6 GB/s.
    - B and D log 24 harmless-looking `cuPointerSetAttribute … CUDA_ERROR_NOT_SUPPORTED` warnings each.
  - **Replicate vs decouple (DDP over B, GDR):**
    - replicate `lean32`: 1.946 s/step on 8 GPUs; DDP cost 5.2 ms. 0.514 rank-steps/s per GPU.
    - replicate `ref`: 4.897 s/step (vs 4.887 s on 1 GPU); DDP cost 4.8 ms.
    - decouple `lean32` (2 cross-node DDP trainers + 6 simulators): 0.4235 s/step free-running; ≈ 0.515 s paced to
      12 fresh pairs/step (derived). That is ≈ 0.485 rank-steps/s per GPU, 0.94× replicate, so equal GPU-seconds
      holds: 3.8× shorter steps at a 4× smaller global batch.
  - **Harness bug found and fixed (separate commit).** The paced phase was reported by its median (0.424 s, wait
    median 0.0). The simulators' bursts hide the waits, so the trainer now records the phase mean.
  - **Predictions:** 9 confirmed, 3 half (#6 busbw above range, #8 s/step 1–2% faster than the 1-GPU sum, #9
    paced s/step not measured as stated), 0 refuted.
  - **Next (the user's):** ship `lean_u1` (bitwise) to o1.2 while `lean32` waits for validation? Round 2 = the u1
    trigger + C2C packing. Size a decoupled run at ≥ 3.7 `lean32` simulators per trainer.
  - **Open, cheap (combine_handoffs):** were the 24 `cuPointerSetAttribute … NOT_SUPPORTED` warnings in B/D from
    NCCL's cuMem buffers? Test with an arm `B_shim` + `NCCL_CUMEM_ENABLE=0` in the next 2-node job; the warnings
    should disappear if so. Don't silence them with `disable_cuda_sync_memops` in gated runs before that A/B.
- **2026-11-05** — **SQG round 1c submitted: 2-node debug job 7815637** (`SQG/analysis/opt/run_round1c_polaris.pbs`,
  walltime 0:59).
  - **User decisions (11-05):** rerun seed 1; carry `lean32` toward o1.2 (precision in production still needs
    statistical validation + jesswan / Weidong's sign-off); measure both replicate and decouple; submit without a
    further approval once the critics' changes were in.
  - **Phase 1:** seed 1 (`lean` / `lean_u1` / `lean_split`) under round 1b's `t` rule (run-time `dt`; `t` within
    1e-6).
  - **Phase 2: four NCCL probe arms**, each judged by `multisim/probe_verdict.py` (NCCL_PROBE_OK needs a completed
    all_reduce + busbw, which fixes round 1b's token bug):
    - A: 7815096's env + `FI_LOG_LEVEL=info`;
    - B: a `libcudart.so` shim pointing at the env's own `.so.12` (combine_handoffs' preferred fix);
    - C: `GDR_LEVEL=LOC`;
    - D: makani/ACE2's `cuda-12.9.1/lib64` prepend.
  - **Phase 3:** replicate `lean32` / decouple `lean32` / replicate `ref`, with DDP over the first passing arm (order
    B, D, A, C).
  - **Library check (readelf):**
    - the aws-ofi-nccl plugin needs `libcudart.so.12`, and its RUNPATH dir no longer exists, so torch's
      already-loaded 12.1 copy satisfies it;
    - libfabric 2.3.1 dlopens the unversioned `libcudart.so` and `libgdrapi.so`, both absent.
  - **combine_handoffs' answers:**
    - their GDR runs were nccl-tests (NCCL 2.28.3, cudart 12.9), not torch, with libfabric resolving a
      `cuda-13.0.1` `libcudart.so`;
    - gdrcopy is opt-in and harmless; the RDZV vars are ruled out;
    - FI_LOG's `Number of CUDA devices detected` marks a working CUDA HMEM.
  - **Pre-submission review:** 2 read-only Fable critics; 3 minor harness fixes applied (receiver join timeout,
    guarded rank-0 aggregation, NCCL-timeout docstrings); the job script had 0 defects.
  - Predictions: `SQG/analysis/opt/PREDICTIONS_round1c.md`. Results →
    `SQG/analysis/results/polaris_opt_round1c_7815637/`. Don't edit `SQG/analysis/opt/` while it is queued.
- **2026-11-05** — **SQG job 7815096 DONE (`OPT_R1B_MS_DONE failures=1`).** Round 1b passed. Of the 2-node arms, 4/5
  passed; the NCCL-over-CXI probe failed. Notes Part VIII; summaries in
  `SQG/analysis/results/polaris_opt_round1b_ms_7815096/`.
  - **Round 1b (one A100): all three pass their pre-set gates.** ms per solver step vs ref 9.290:

    | variant | ms/step | speedup | optimizer step |
    |---|---|---|---|
    | `lean64` | 7.051 | 1.318× | 3.40 s |
    | `mixed` | 4.714 | 1.971× | 2.28 s |
    | `lean32` | 3.189 | 2.913× | 1.55 s (vs 4.47) |

    - `t` is now exact for all three (run-time `dt`).
    - `lean64`'s 256² float32 frames are **bitwise** equal to production's after 4,000 steps (fields ≤ 1e-13).
    - `lean32` is float32 itself: its 1-step and 4,000-step differences equal the original step run in float32 to 3
      digits (4.36e-4 coarse, −8.36e-5 SPE).
    - `mixed` keeps a float64 state and is 70× closer at 4,000 steps.
  - **2-node arms.**
    - **Independent simulators:** scaling efficiency 0.9999 (ref) / 1.0024 (lean32).
    - **Stream (7 simulators → 1 trainer, gloo over hsn0, RAM only, nothing written to disk):** the trainer stays
      at its standalone 0.419 s/step. A send is ≈1 ms per 6.3 MB message. Fresh pairs per trainer step:
      `ref` 8.14 (solver-bound), `lean32` 22.6 (trainer-bound). One trainer needs 10.7 / 8.1 / 5.4 / 3.7
      simulators (ref / lean64 / mixed / lean32) to get o1.2's 12 fresh pairs per step.
    - **Decoupling trades GPUs for wall-clock at equal GPU-hours.** Replicating o1.2 only grows the global batch.
  - **NCCL probe FAILED: cxi can't register NCCL's GPU buffers.**
    - Stack: torch 2.3.1's NCCL 2.20.5 + aws-ofi-nccl 1.6.0, under `NCCL_NET_GDR_LEVEL=PHB`.
    - Plugin v6 and the cxi provider loaded on 8/8 ranks. Intra-node is `P2P/CUMEM/read` ×24, SHM 0.
    - Every inter-node hop chose GDRDMA. On all 4 network ranks: `cxil_map: write error`, then
      `NET/OFI Unable to register memory (type = 2) … RC: -14, Error: Bad address`.
    - `Connected all rings` was never reached, so there is no busbw. No arm ran without `NCCL_PROTO=Simple`.
    - `NCCL_CXI_OK providers=8` only means the provider was selected.
    - These answers went to combine_handoffs (SendMessage, 11-05).
    - **Their reply.** GDR works on their stack (7629096: 4 nodes, NCCL 2.28.3, `GDRDMA`, `GDR 1`), with cuMem on
      too. Their leading cause: cray libfabric 2.3.1 dlopens the unversioned `libcudart.so`, and sqg-env has only
      `libcudart.so.12`, so CUDA HMEM init fails and cxi pins GPU addresses as host memory (EFAULT).
    - **Checked here:** the libfabric strings, and that no dir on the job's `LD_LIBRARY_PATH` has `libcudart.so`.
      The mechanism itself is unconfirmed (no `FI_LOG`).
    - **Next probe:** A `FI_LOG_LEVEL=info`; B a `libcudart.so` → env `libcudart.so.12` shim; C
      `NCCL_NET_GDR_LEVEL=LOC`. Gate the probe's token on a completed all_reduce + busbw.
  - **Predictions scored without edits.**
    - Round 1: the gates all mispredicted (cause: the gates); 2/6 speed ranges hit; every speed miss was slower.
    - Round 1b: 7 confirmed, 1 confirmed with two edge misses (≤ 2%), 1 half (#5: cxi yes, bandwidth never measured).
  - **Open (the user's):** a seed-1 `lean_u1` rerun under round 1b's `t` rule; which precision goes into o1.2
    (precision is jesswan / Weidong's call); replicate vs decouple.
- **2026-11-05** — **SQG handoff for a fresh session: `SQG/polaris/sqg_opt_handoff.md`.** It gives the state, the
  steps after 7815096 (scoring, the NCCL answers owed to combine_handoffs, notes Part VIII), the open decisions (seed 1
  rerun, which variant goes into o1.2, replicate vs decouple), and the Polaris multi-node recipe.
- **2026-11-05** — **SQG round 1b + 2-node independent-simulation smoke submitted: debug job 7815096** (2 nodes,
  8 A100s). Phase 1 reruns `lean64` / `mixed` / `lean32` under gates written before the rerun (user request):
  float32 variants ≤ min(1e-4, 2 × the ORIGINAL step's float32 error from the same state, same run); `lean64` ≤
  1e-12; `tc`/`_ablevel` exact, `t` within 1e-6 relative (exact `t` reported as information); `t` now advances
  by a run-time `dt`. Phase 2 (`SQG/analysis/opt/multisim/`) runs an NCCL-over-CXI probe (aws-ofi-nccl v1.6.0
  with torch 2.3.1's bundled NCCL, never tried before; recipe from combine_handoffs), 8 independent trajectories (ref
  and lean32), and an o1.2-style stream: 7 simulators send pairs over gloo/hsn0 to 1 trainer GPU running o1.2's
  UViT from an in-RAM FIFO, with nothing written to disk, as online training does. Predictions:
  `SQG/analysis/opt/PREDICTIONS_round1b.md`. Results → `SQG/analysis/results/polaris_opt_round1b_ms_7815096/`.
  **Do not edit `SQG/analysis/opt/` while it is queued** (the job reads it at start).
- **2026-11-05** — **SQG round 1 (7814300) DONE: every variant FAILED its pre-set gate; verdicts stand.** Measured
  ms/step vs ref 9.29: `lean` 8.00, `lean_u1` 6.40 (1.45×, bitwise fields), `lean_split` 7.07, `lean64` 7.07 (≤
  2.9e-14), `mixed` 4.68 (1.98×), `lean32` 3.20 (2.90×). Two causes. (1) All of them fail on the float32 time
  counter `t` (`tc`/`_ablevel` were fine), which the step never reads; the likely cause is XLA folding the
  unrolled `(t + dt) + dt`. (2) `mixed` and `lean32` (2.30e-5) also fail the 1e-5 float32 limit, which was below
  plain float32's own error: the original step run in float32 scores 2.48e-5 from the same state. So the limit
  measured float32, not the variants. Projected o1.2 online wall-clock on the H200 (7 days today, solver ≈ 87%):
  `lean_u1` ≈ 5.1 d, `mixed` ≈ 4.0 d, `lean32` ≈ 3.0 d; the offline floor is 21 h if the solver is hidden on
  extra simulation GPUs. Results: `SQG/analysis/results/polaris_opt_round1_7814300/`.
- **2026-11-04** — **SQG solver-optimization round 1 submitted: debug job 7814300.** Built by two Opus seeds and
  reviewed by a Fable principal scientist (user's design; user policy: bitwise not required, a tolerance fixed
  before the run decides). Six variants: seed 1 float64 JAX `lean` / `lean_u1` / `lean_split` (drop exact ×1/+0
  terms, 3-slot AB3 ring, unroll-1 and split tricks against the AB3 fusion's over-fetch); seed 2 `lean64` (u, v from
  one forward C2C FFT, 2-slot ring), `mixed` (float32 nonlinear term), `lean32`. Gates: ≤ 1e-12 (f64) / ≤ 1e-5 (f32)
  after 1 step from a random and a synthetic CFL-0.3 state. Report-only: per-|k|-band errors and a float32 baseline
  of the original step. Predictions in `SQG/analysis/opt/PREDICTIONS_round1.md`; results →
  `SQG/analysis/results/polaris_opt_round1_7814300/`.
- **2026-11-04** — **SQG: the 5 full-field copies per solver step are traced** (job A capture, each copy attributed to
  its issuing XLA thunk). 2/step are the AB3 history rotation inside the AB3 update graph (pyqg_jax
  `AB3Stepper.apply_updates`, steppers.py:501, under `lax.scan`'s one-buffer-per-slot carry). 3/step are the three
  inverse (C2R) FFTs copying their inputs (θ̂, û, v̂); the forward FFTs copy nothing. Removable: rotation ≈0.39 ms
  bitwise (unroll by 3 / ring index); û, v̂ ≈0.39 ms only with an input-overwriting FFT call; θ̂'s copy is needed.
  This **corrects** the "≈1 ms, all loop-carry" estimate to ≈0.4–0.8 ms. Notes Part IV addendum.
- **2026-11-04** — **SQG job E 7813816 PASS: a float32 solver step is 2.6× faster (9.30 → 3.61 ms; 4.47 → 1.74 s of
  solver per optimizer step).** Checks pass: the builder is bitwise equal to `GD.inner`, and there is no hidden f64 in
  the f32 HLO. Every kernel scales ≈2× with its bytes (cuFFT 3,059 → 1,512 µs/step) **except the AB3 fusion: 3,288 →
  650 µs (5.1×)**. So ≈2 ms/step of the f64 fusion is float64-specific, not data. Both precisions still evaluate the
  no-op filter's sqrt/exp, which is software-emulated in f64. Job F (Nsight Compute, **7814044**, submitted)
  separates that from register pressure. Prediction 2 (1.6–2.0×) refuted; notes Part VI. Float32 is timing only.
- **2026-11-04** — **SQG: a STRIDE-400 online run reportedly overfit** (user report; run details not seen here).
  This contradicts the decorrelation entry's advice to sample online pairs at Δt/10–Δt/4. In o1.2, `STRIDE` also sets
  the data *quantity*: pairs per segment 4,901 → 491, the start buffer (4 segments) 19,604 → 1,964 pairs, fresh pairs
  per optimizer step S/STRIDE 12 → 1 (S must be a multiple of STRIDE). The 262,144-pair buffer then never fills, so
  the first pairs are sampled ≈150× (vs ≈7× at STRIDE 40). The 40-step near-duplicates cost no extra solver time
  and probably also act as augmentation (untested). **Keep STRIDE 40 for online training; do not thin a saved frame
  store without a test.** No translation augmentation: the forcing is a fixed spatial pattern. A discriminating test
  is STRIDE 40 with every 10th pair buffered (same solver cost, STRIDE-400 density).
- **2026-11-04** — **SQG jobs E and F: the solver's GPU work.** E (`profile_solver_fp32.py`, debug **7813816**,
  queued) times the production step in float32 vs float64. It reports only if its builder is bitwise equal to
  `GD.inner` and the f32 HLO holds no full-size f64/c128 field. F (`profile_ab3_ncu*.py/.pbs`, `summarize_ncu.py`;
  written by a subagent, reviewed; `NCU_TEST_OK`) is Nsight Compute on the AB3 fusion and three contrast kernels. It
  is **not yet submitted**, because the debug queue allows 1 queued job per user (`queued_jobs_threshold`). Predictions
  in notes Part VI. Also answered: torchdiffeq / PyG do not touch the bottleneck (CFL-bound stepping; FFTs on a
  regular periodic grid).
- **2026-11-04** — **SQG job D 7813332 PASS (`PHASES_OK` × 3), 4 of 5 predictions confirmed, 1 half refuted.** In
  the production order there are 507 syncs and 506 DtoH per optimizer step. 480 of each come from one HLO op
  (`conditional.277.0`, the `lax.switch`) and block 4,412 of 4,920 ms. The other 27 are cheap (`flush` 24, 0.08 ms;
  `loss_item` 2, waiting on the network's own GPU work; harness 1). The network phases issue 0 syncs. Without the
  switch the wait moves into `cuLaunchKernel` (91% of launches slow, 3.84 s/step), not into `flush`. Parser counts
  equal nsys `cuda_api_sum` ÷ 6. Notes Part V; summaries in `SQG/analysis/results/polaris_phases_7813332/`.
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
  - **The largest solver kernel is the whole AB3 update** (notes Part IV addendum, from `hlo_ref.txt`). It moves
    ≈1.27 GB in 3.27 ms (≈390 GB/s, vs ≈1.3 TB/s for cuFFT). Three parts of it cannot change the answer: pyqg's
    filter (sqrt + f64 exp per coefficient; exactly 1 at `filterfac=0`), a dense 134 MB F̂ read every step
    (24 nonzero modes), and a probable zero background-gradient term (unverified). Candidate: a lean step in o1.2
    behind a bitwise gate, estimated headroom ≤ ≈2.5 ms/step; ncu on the fusion first.
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

- **(Polaris, SQG env) NCCL over CXI with torch 2.3.1's bundled NCCL 2.20.5 + aws-ofi-nccl v1.6.0 +
  `NCCL_NET_GDR_LEVEL=PHB` fails at connect.**
  - Symptom: `cxil_map: write error` → `NET/OFI Unable to register memory (type = 2) … RC: -14, Error: Bad address`
    on every rank that does an inter-node hop; the rest hang.
  - **Cause CONFIRMED (job 7815637):** libfabric's `cuda_hmem_dl_init()` cannot dlopen the unversioned
    `libcudart.so`, which a wheel-based env lacks.
  - **FIXED:** put a dir holding `libcudart.so` → the env's own `libcudart.so.12` first on `LD_LIBRARY_PATH`
    (`run_round1c_polaris.pbs`, `arm_env B_shim`). That gave 31.5 GB/s with GDRDMA, and DDP works. Fallbacks:
    `NCCL_NET_GDR_LEVEL=LOC` (17.2 GB/s), or prepending `cuda-12.9.1/lib64` (32.6 GB/s, but two CUDA runtimes).
  - Don't use the bare env block for cross-node NCCL from a wheel env.
  - Source: job 7815096; `SQG/polaris/polaris_sqg_notes.md` Part VIII.

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
