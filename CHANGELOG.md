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

- **2026-11-12 (night)** — **SQG tile-row step: Warp tile FFTs (cuFFTDx) for rows + cuFFT kept-column passes + Warp
  one-thread-per-element kernels → 1.578 ms/step vs `lean32ek_fast` 2.708 (1.72×), gates pass (job 7840376;
  11 confirmed / 0 refuted). Notes Part XVII D.**
  - The user's prompts: use Warp for the irregular memory access (`wp.tid()`), and use its tile FFTs.
  - **Design** (`SQG/analysis/opt/tilerows/`, standalone on torch buffers):
    - Warp per-element kernels: `build_w` with the mirrored gather (0.147 ms), expand (0.079), and the Hermitian split
      + compensated update (0.216);
    - cuFFT: strided column FFTs on the kept 2731 columns only (0.275 + 0.276) and the C2R (0.302);
    - one Warp tile kernel per row: tile_fft → conj(p)·θ → tile_fft (0.302).
  - **Accuracy:** tilerows32's error equals `lean32ek_fast`'s (2.33e-7 vs 2.27e-7 at 40 steps; bands 0.97–1.35×).
    The IF-AB3 variant costs nothing (1.577 ms).
  - **Per optimizer step:** 1.47× at n = 40, **1.95× with IF-AB3 at n = 25** (94 → ~48 h on 14 A100s).
  - **Caveats:**
    - standalone: o1.2's JAX solver needs a custom-call integration or a torch-side solver;
    - ran with cuFFT 11.4.1 preloaded (production wheel 11.0.2: rerun owed);
    - the statistical gates were run for `lean32ek_fast`, not this step (the user's call).
  - The first submission (7840360) died on Warp rejecting numpy-scalar constants (fixed; see "Known issues").
  - **Warp update kernels inside the JAX step (warpupd, job 7840397; 10 confirmed / 1 refuted; notes Part XVII E):**
    - all arms pass; `xla_compact`, `warp_full` and `warp_compact` are bitwise equal to `lean32ek_fast`;
    - ms/step: `warp_full` 2.639, `warp_compact` 2.533, `xla_compact` 2.505, **`warp_simt` 2.372 (1.14×)**,
      `lean32ek_fast` 2.697.
    - The update kernel's gain is the compact layout, which XLA does as well. Warp wins once it does all the
      irregular gathers.
    - Nsight: Warp `build_w` reads its unique 89.5 MB vs XLA's 190.5 MB; XLA's compact pad fusion is 25%
      sector-efficient; Warp's vec2f loads are scalarized (50% per instruction; DRAM bytes unique).
    - Far behind the tile-row step (1.578 ms).

- **2026-11-12 (evening)** — **SQG round 2 measured: IF-AB3 makes 25 steps/frame admissible (job 7839988), the Warp
  route works on jax 0.4.28 (jobs 7839932, 7840069), and a hybrid cuFFT + callbacks + Warp step is built (job 7840175
  queued). Notes Part XVII.**
  - **Time step (7839988, `TS_VERDICT n_adm=25 valid=True`; 12 confirmed / 2 half / 2 refuted):**
    - production's dt error is first order (prod80/prod40 0.49–0.51);
    - IF-AB3 removes it (≥ 10× smaller at n = 40);
    - IF-AB3 at 25 steps per frame is 2.4–4× more accurate than production at 40 in every band (worst 0.41, sub).
    - CFL_max at DT is 0.379, so n = 25 gives 0.607 vs the 0.72 abort (production-sampled 0.42 would give 0.672);
      n = 20 fails CFL.
    - Projection 1.38–1.40× per optimizer step (94 → 68 h).
    - Costs +1.8% per float32 step, and float32 precision is unchanged (ratio 0.99–1.01).
    - Adopting it shifts o1.2's targets by ≈ production's own dt error. That is the user's call; long-run stability at
      n = 25 is untested.
  - **Warp (user's suggestion):**
    - warp-lang 1.17.0's legacy custom-call path, via our `warp_xla.py`, works on jax 0.4.28: `BRIDGE_OK`, 8.6 µs
      per call.
    - Round 1's four failures were our bugs, now in "Known issues".
    - Round 2: `GO_F32` (projection 1.578 ms) / `GO_F64` (1.763 ms).
    - Warp's column FFT is 2× cuFFT's, so FFTs stay in cuFFT.
  - **Ecosystem survey:** no Warp QG/SQG/climate solver exists. XLB (LBM) reports Warp ≈ 8× JAX; pyspectral uses
    Warp around `torch.fft`; nvmath-python offers cuFFT LTO callbacks written in Python.
  - **Hybrid build (`SQG/analysis/opt/hybrid/`; user: "start building it"):**
    - 3 cuFFT calls with fused nvmath callbacks (`build_w` on load, θ̂ read on load, conj(p)θ on load with a kept-mode
      store) and 1 Warp update kernel on the compact block;
    - float32 = `lean32ek`'s compensated update (`fuse_fp=False`), optional IF-AB3; float64 = `lean_u1`'s;
    - index algebra verified on the callbacks' own source (`test_hybrid_math.py`);
    - gates G32 / GIF / G64 preregistered.
    - **Job 7840175 (`HYB_JOB_DONE failures=0`; 11 confirmed / 1 refuted): accurate but not faster.**
      - All gates pass: hybrid32's error equals `lean32ek_fast`'s (2.34e-7 vs 2.27e-7 at 40 steps; bands
        0.98–1.35×); hybrid64 = `lean_u1` to 5.3e-15 after 4,000 steps.
      - Speed: hybrid32 2.94 ms vs `lean32ek_fast` 2.71 (8% slower); hybrid64 4.28 vs `lean_u1` 6.40 (1.50×).
      - Cause: cuFFT LTO callbacks roughly doubled each C2C (1.09–1.13 vs 0.60 ms plain), more than the fused passes
        they remove. The CUDA graph changes nothing.
      - Pre-set "worth integrating": no.
      - The measured best stays `lean32ek_fast` + IF-AB3 at n = 25 (1.38× per optimizer step).
  - **Not done:** the solver/ML overlap retest (`profile_overlap.py` knows only ref/fixed).

- **2026-11-12** — **SQG solver speed, research round 2 (workflow wf_1a24fad4-309; no jobs). Notes Part XVI;
  write-up `SQG/analysis/opt/research_2026-11-12/SOLVER_SPEED_RESEARCH.md`.**
  - **Method.** 5 tracks (codebases, FFT kernels, time stepping, XLA-reachable, frontier), each adversarially
    verified against its sources and Part X's per-op table.
  - **Biggest lever: exact integrating-factor AB3.** It removes o1.2's first-order split error at ~0 cost and allows
    28/25 steps per frame instead of 40: 1.30–1.40× per optimizer step, gated.
  - **Bitwise:** compact kept-mode storage, −0.22 to −0.27 ms/step, pure JAX.
  - **Custom FFT:** cuFFT LTO callbacks (gyaradax pattern, LGPL-3.0) + cuFFTDx bring the step to ~1.6–2.1 ms, but jax
    0.4.28 has no FFI. That needs the legacy custom call, or a JAX upgrade that collides with torch 2.3.1's cuDNN pin.
  - **Stacked:** 1.40–1.78×, ≤ 2.1× with CFL-adaptive n.
  - **Dead ends recorded:** tensor-core / FP64-TC / Ozaki FFTs, ETDRK4/AB4/leapfrog, parareal/SDC, temporal blocking,
    VkFFT, cuFFT upgrades, porting other codes. Stochastic rounding was already measured worse.
  - **Correction:** Part III B's 21% overlap was 21% of the ML step, not of wall.
  - **Next:** the user's choice of IF-AB3 gating, compact storage, the custom-FFT route, and an overlap retest.

- **2026-11-12** — **SQG gate D: capacity job 7835786 DONE (`DOSE_JOB_DONE failures=0`), PASS_D on best.pt and
  last.pt. Notes Part XV.**
  - **The ask (user, 11-10).** "can we check if rollouts overfit and drift?": a training dose-response on
    `capacity` (prereg `SQG/analysis/opt/gateD/PREDICTIONS_gateD.md`, committed first).
  - **Design.**
    - 4 o1.2-UViT arms, 195,250 steps each at batch 16, trained on members 1–3 of 16 bitwise `lean_u1` 1-TU
      segments; member 4 held out.
    - Targets y + α·B̂ (`lean32ek`'s band bias pattern), α = 0 (2 seeds), 1,000, 10,000 (positive control).
    - 16 ICs × 50-TU rollouts per arm and checkpoint.
    - 24.8 node-h on one node, no restart.
  - **Validity.** All checks pass:
    - frames 16/16 bitwise vs 7824121;
    - 0/16 rollouts blown in every arm;
    - the positive control drifts beyond the noise in 52/90 window-band cells (best.pt), with the same sign in 10/10
      windows per band, following B̂'s sign for k < 64 (post hoc).
  - **Result.**
    - The real-bias effect, extrapolated from α = 1,000, is ≤ 2.4e-4 of a band energy, 55× below the pass line
      (0.1 × noise); 31× on last.pt.
    - The α = 1,000 effect itself is inside the seed spread (80/90 cells).
    - No overfitting: final/min val 1.0035 / 1.0022 for the α = 0 arms.
    - The trained models' own drift (E_low 16–26% low on best.pt; k ≥ 128 −70%) is 2–3 orders of magnitude above
      any precision effect.
    - Scored 5 confirmed / 3 refuted: 3 overfit, 5 κ in [0.5, 1.5] (one-step absorption is noise-dominated on
      best.pt), 8 E_low within ±20%.
  - **What it means.**
    - The pre-set chain for `lean32ek_fast` is now A PASS / B detection-only / energy PASS_E / D PASS_D, which is
      the prereg's "case for admitting `lean32ek_fast`". Its o1.2 schedule is 94 h vs 191 h (`lean_u1`) on 14 A100s.
    - Admitting it is the user's call. It is equivalent at the trained-model level, not numerically.
    - Limits: ~1/14 of o1.2's samples, band-level pattern only, 2 seeds.
  - **Results** in `SQG/analysis/results/polaris_gateD_7835786/`. Frames, checkpoints and rollouts stay in
    `$MEMBER_ROOT/sqg_profile/gateD_{frames,v1}/`.

- **2026-11-11** — **SQG MD-style energy-budget test: debug job 7836358 DONE (`ENERGY_JOB_DONE failures=0`),
  PASS_E. Notes Part XIV.**
  - **The ask (user).** "yes do the MD debug job": ANALOGUES.md §2.6's offer.
    - MD codes accept a precision model on long-run energy conservation.
    - Our forced-dissipative analogue is the closure of the SPE and VIE budgets.
  - **Design** (`SQG/analysis/opt/energy/`, prereg `PREDICTIONS_energy.md` committed first, bc296f00):
    - 4 archived states × 0.5 TU (100,000 steps) × 5 arms;
    - the paired spurious source s = r(arm) − r(`lean_u1`) as a fraction of the injection;
    - PASS_E if max |s(`lean32ek_fast`)| ≤ 1e-4.
    - Changed from the offered ≥ 10 TU before the run: 10 TU of `lean_u1` is 3.6 h per state.
  - **Validity.** All checks pass:
    - `lean_u1` frames bitwise equal to gate D's production frames, 24/24;
    - `budget()` equals `GD.diagnostics` to 4.2e-16;
    - the float64 null ≤ 6.2e-13;
    - the `dquant` calibration recovers its exactly known 1.1e-2 source to 6 digits.
  - **Result.**
    - `lean32ek_fast`: s = −5.5e-6 to +1.4e-6 (SPE) and −4.7e-6 to +1.2e-6 (VIE) over the 4 states, 18–21× below the
      tolerance. Its sign varies by state and it does not grow with time.
    - Its damping table contributes only ~1e-8, so the source is the float32 tendency arithmetic *(inference)*.
    - Plain `lean32`: −1.4e-2, about 140× over the tolerance; 82–85% of it is the float32 decay table.
    - The float64 scheme's own residual (−3.5e-6 to −2.9e-5 SPE, ~−6e-7 VIE) is the integrating-factor term, to
      0.92–1.13.
  - **MD's stricter "equivalent to double" (reported only).** Not met: the candidate's budget error is ≤ 2.4× float64's
    on SPE but up to 8.6× on VIE, whose float64 residual is only ~6e-7.
  - **Scorecard:** 14/17 confirmed. Refuted:
    - P3: SPE dissipation is 62–79% hypofriction, so the integrating-factor term is ~1e-5, not ~1e-4;
    - P11: the candidate's source is 5.5e-6, not ≤ 1e-6;
    - P13: EQUIV_DOUBLE.
  - **Speed** (budget included): `lean_u1` 6.443, `lean32ek_fast` 2.742, `lean32` 3.239 ms/step.
  - **Also:** `SQG/analysis/opt/energy/MATH_energy.md`, at the user's request: the SQG math step by step, why
    Fourier space, how we modified the original step (with drawings), and the MD comparison.
    - Written by a two-writer + two-checker workflow; the main session applied the checkers' 4 errors and 7 minors.
    - The fixes: a stale "job not run" status; a 30× slip in an absolute SPE change; the 16-state SPE range (24–48,
      not 32–46; the same slip is in the prereg, now a dated erratum); Part 5's line numbers moved to the pinned
      `fixed_ring/A-SQG_data.py`; two redrawn ASCII drawings.
    - §8.1 adds the results.
  - **Next:**
    - gate D (capacity 7835786) still running; its write-up becomes notes Part XV;
    - not tested: a ≥ 10-TU climate comparison and per-shell budgets.

- **2026-11-10 (late night)** — **SQG gate D: smoke 7835759 passed (failures=0); production submitted as capacity job
  7835786.**
  - **The ask (user).** "Can we check if rollouts overfit and drift": the training dose-response, on capacity.
  - **The harness:** `SQG/analysis/opt/gateD/`, with predictions and the decision rule in `PREDICTIONS_gateD.md`
    (committed first).
  - **What the smoke (debug, 6 min, tiny sizes) checked.**
    - The tests pass on the GPU; torch matches numpy to 5e-7.
    - Stored frames are bitwise equal to job 7824121's production frames.
    - Training runs at 0.434 s/step, so 195,250 steps ≈ 23.5 h.
    - Rollouts, blow-up handling and both analyses produce verdicts.
    - Its science is meaningless by design (300 steps on 453 pairs).
  - **The production run:**
    - 16 one-time-unit `lean_u1` segments, about 20 GB in `sqg_profile/gateD_frames`;
    - arms `a0_sA`, `a0_sB`, `a1000_sA`, `a10000_sA`, each 195,250 steps;
    - 16 × 50-time-unit rollouts per arm;
    - run name `gateD_v1` (resumable);
    - end token `DOSE_JOB_DONE`.
  - **Do not edit `SQG/analysis/opt/gateD/*.py` until it ends.** It reads them at each stage.
  - **Also added:** a molecular-dynamics section to ANALOGUES.md (§2.6). AMBER SPFP, OpenMM, GROMACS and LAMMPS all
    use single-precision forces with higher-precision accumulation, which is `lean32ek`'s design. AMBER deprecated its
    all-single mode, matching our rejection of plain float32. MD's acceptance test is long-run energy drift; our
    analogue is the steady-state energy budget.

- **2026-11-10 (night)** — **SQG item 4: gates A and B run by us, on the user's decision. Item 3's sealed file is
  opened. lean32ek passes A, and fails B on detection only.**
  - **User decisions:**
    - the user decides item 4 ("ignore weidong/jesswan");
    - results are not hidden ("do not seal anything we don't need to hide results");
    - gate C later.
  - **How it was run.** The rules were pushed before opening: `ACCEPTANCE_gates_AB.md` + `gates_ab.py`, 09970f75.
    The sha256 matched `6f1d0e51…`.
  - **The yardstick.** The alias floor of o1.2's frames (production only) stands in for the trained model's error:
    ε_floor = 0.087 … 0.730, T_b = (0.1/√n_b)·ε_floor = 1.2e-3 … 7.3e-2. That is conservative.
  - **Gate A (noise): PASS for all arms.** Var(e)/A_b is 8e-10 to 6e-8, against < 1e-2.
  - **Gate B (bias): NOT_PASS for all arms, on B3 (detection) only. Magnitudes pass by ≥ 100×.**
    - `lean32ek` has a real, detectable bias: |z| up to 16; on alias-free frames z_d −15 to −23 and 75–88% of shells
      flagged at k 4–32.
    - Its size is ≤ 4.4e-6 of Δ (sub) and ≈ 4–6e-7 (af), d-aligned: the delta is ≈ 5e-7 too small. That is 73–509×
      below plain float32 and ≥ 100× below T_b.
    - The dt arms miss B3 by 1–2 shells with only 8 pairs. Every |z| ≤ 3.
  - **Predictions:**
    - **P2 REFUTED.** lean32ek is not bias-free; the team's theory bound of ≤ 1e-7 was too low.
    - **P5 CONFIRMED.** The dt order ratio is −1.42 to −1.51 in all bands: o1.2's dt sensitivity is first order (the
      operator split), not AB3's third order.
    - **Item 3 final:** 9 confirmed / 2 half / 1 refuted.
  - **Decision state.**
    - Under the pre-set rules, lean32ek_fast is not admitted yet. The next step is gate D (dose-response fine-tune,
      which needs stored frames and a trained model).
    - Admitting it on magnitude would be a post-hoc rule change: the user's call, to be recorded as such.
  - **Unresolved.** Our alias floor is ≈ 1.9× TEAM_REPORT's table in every band, though the fractions agree. If the
    table missed a half-plane factor, o1.1's val 0.0043 would sit at ≈ 1.0× the floor, not 1.9×. That matters for
    item 2.

- **2026-11-10 (evening)** — **SQG: verified citations for `lean32ek` / `lean32ek_fast` / `lean32sr`, written into
  the code and docs.**
  - **What was checked, and how.** Seven references, each against its Crossref record. Klöwer et al. 2022 was also
    checked against its full text.
  - **Where they now appear:** the docstrings of `lean32_precise.py`, `f32_update.py` and `lean32_fast.py`;
    `MATH_delta.md` §6; notes Part XI; and ANALOGUES.md's sources.
  - **Corrections.** ANALOGUES.md had abbreviated or slightly wrong titles for Klöwer 2022, Paxton 2022 and Kimpson
    2023. They are fixed.
  - **The attribution, as recorded.**
    - The compensated state is published (Klöwer et al. 2022, doi:10.1029/2021MS002684).
    - So are TwoSum (Møller 1965 / Knuth), Fast2Sum (Dekker 1971) and compensated summation (Kahan 1965 / Higham 1993).
    - The expm1 damping table and the fast form's cos-trigger and 3-slot ring are ours, with no external source.
  - **Checks:** the edited files compile; `test_f32_update.py` prints `F32_UPDATE_TEST_OK`.

- **2026-11-10 (later)** — **SQG: o1.2 itself runs on Polaris A100s (job 7835278, `failures=0`, notes Part XIII).
  A full o1.2 schedule extrapolates to 268 h (production solver), 191 h (`lean_u1`) or 94 h (`lean32ek_fast`) on 14
  A100s.**
  - **The ask (user).** Run our own o1.2 and generate logs. Extrapolate the A100 speeds; no H200 comparison.
  - **The harness:** `SQG/analysis/opt/o12run/`.
    - `o12_polaris.py` execs o1.2's own file, unedited, with 34 exact-match, line-count-preserving patches: paths;
      torch 2.3.1 strict fp32; the vendored solver's BLOCK_T / CHECKPOINT_DIR; members 1–4; zarr / wandb / matplotlib
      stand-ins.
    - o1.2's first-segment check is re-pointed at job 7824121's production frames.
    - Knobs: batch 16, a short fill and schedule.
  - **Measured** (1 node, 4 A100s, strict fp32):
    - `ref` and `lean_u1` frames are **bitwise equal to production**.
    - Steady s/step: 4.897 / 3.4925 / 1.7107. That is solver (12 frames × 40 steps) + an ML step of 0.40–0.42 s: a
      pure sum, so solver speed-ups pass through one for one.
    - Fill 0.373 / 0.257 / 0.109 s per frame.
    - Validation 0.27 s and saves ~0.75 s per epoch; peak 20.4 GiB; 40.81 M parameters.
  - **Full o1.2 schedule** (14 A100s on 4 nodes, batch 16 = o1.2's global 224, 195,250 steps):

    | solver | wall-clock | node-h | GPU-h |
    |---|---|---|---|
    | `ref` | 268 h | 1,074 | 3,757 |
    | `lean_u1` | 191 h | 765 | 2,679 |
    | `lean32ek_fast` | 94 h | 376 | 1,315 |

    The step term is ≥ 98.9% of every projection, and GPU-h hardly depend on the layout. Restarts are not modelled:
    resume is from `best.pt`, which re-runs epochs.
  - **Scorecard:** 7 confirmed / 1 half / 0 refuted. The half is P2: `lean32ek_fast`'s 2.6e-3 max |Δθ| is above my
    band. It is the expected float32 drift: the taxonomy harness shows 2.6e-4 to 1.5e-3 at 4,000 steps, and θ's max is
    ≈ 44.
  - **A science-grade full Polaris run would still need:**
    - BUF de-duplication or a shrink (137 GB / rank × 4 > 512 GB);
    - ≈ 1.3 TB of checkpoints (we have 16);
    - a held-out val store;
    - ≈ 765 node-h preemptable with `lean_u1`.

    The user's call.

- **2026-11-10** — **SQG item 3: archive run 7824121 scored (notes Part XII). The run is valid. Plain float32 is a
  learnable bias. The candidates stay sealed.**
  - **The run.** 1-node debug, x3001c0s13b0n0, 2026-11-06 22:15–22:44. `TAXO_JOB_DONE failures=0`,
    `TAXO_ANALYSIS_DONE valid=True`. 16 archived states (members 1–4 × t = 100–400), 64 pairs. Summaries are in
    `SQG/analysis/results/polaris_taxonomy_7824121/`.
  - **Validity.** All gates pass.
    - V1: prod == lean_u1 bitwise.
    - V2: κ 1.036 / 0.997 / 0.992 at k < 8.
    - V3, paired: recovery 0.961–1.007, power z 5.2–17.6.
    - V4: dt40 == lean_u1 on 8 of 8 states.
  - **Sealed, unopened:** the bias statistics of lean32ek (and its fast form), dt 2e-4/30 and 2e-4/32, and the dt order
    ratio. sha256 `6f1d0e5153918a3b14faf6e29795d91263a032479121bf87a4417fb173b78be1`.
  - **Measured, unsealed.**
    - **lean32 is a learnable bias.** 67–74% of shells are flagged at k 2–16. Its learnable bias is 2.2e-3 of Δ at
      k 2–4, 5× its noise level.
    - **It is mostly mechanism A.** The float32 `decay` table alone (`dquant`) gives 2.14e-3. A ⊕ B in quadrature
      reproduces lean32's learnable bias to within 2% at k 2–8.
    - **Mechanism B (state rounding) is learnable too, but small.** It flags 35% / 26% of shells at k 4–8 / 8–16, at
      ≤ 1.9× its noise level.
    - **This corrects Part XI's "state rounding is the cause, not the damping table".** That line was read off
      lean32e, inside a float32 state, where the two mechanisms cannot be separated.
    - **lean32ek's size.** e_Δ 9.43e-6 overall, the same as on the synthetic state (8.8e-6). On o1.2's frames it is
      below production's own dt-halving sensitivity in every band (ratio 0.19–0.93). Its k 1–2 error is mostly folded
      small-scale noise: alias-free / sub = 0.068.
    - **lean32ek_fast:** 2.698 ms = 3.44× production, bitwise equal to lean32ek at 40 / 400 / 4,000 steps.
    - **Negative controls.**
      - Shells: 0.50% / 0.42% flagged (nominal 0.27%).
      - z_x is calibrated: 0 of 32 cells over 3.
      - z_d runs hot: 3 of 16 cells over 3. Most likely this is stochastic rounding's own small d-term (negative in 6
        of 8 bands, −4.6 / −5.5 at k 8–32).
  - **Scorecard.**
    - Confirmed (8): 1, 3, 6, 7, 9, 10, 12, 13.
    - Half (2):
      - 8: stochastic rounding's shell rate is above 1% at k 2–4 (1 of 6) and k 8–16 (1 of 68) per band, but 0.42%
        pooled.
      - 11: lean32ek's e_Δ, 9.43e-6, is just under the [1e-5, 4e-5] range.
    - Refuted: none.
    - Sealed: 2 and 5.
    - P4 (no prior): LEARNING.
  - **Raise before anyone unseals.** P2's "≤ 1% of shells in each band up to k 64" would fail a truly unbiased lean32ek
    with probability ≈ 25% at the nominal rate, or ≈ 43% at the observed 0.46%.
    - The reason is that bands to k 16 hold only 2 / 6 / 20 / 68 shells.
    - f64ic32, unbiased by construction, already fails it at k 8–16.
    - Any new reading has to go in a dated addendum before the sealed file is opened. That is the user's / owners'
      call.
  - **Next (waiting on people):**
    - item 4: the acceptance-protocol memo for jesswan / Weidong (gates A–D, the floor table, this MDE table, the
      controls);
    - item 1: the user's o1.2 / o1.1 / offline logs and `*_best.pt` / `*_last.pt` (gate B's eps_model,b needs items
      1 → 2);
    - where the 6.4 GB of staged checkpoints should live. They are in the main checkout's root, symlinked into
      `sqg_stage`.

- **2026-11-06 (later)** — **SQG item 3 (generator-error taxonomy): harness built, pre-registered and smoked (job
  7823653, `failures=0`); the archive run waits on staging.**
  - **Smoke facts** (synthetic state, 4 near-duplicate copies, unscored; `SQG/analysis/results/polaris_taxonomy_smoke_7823653/`).
    - Bitwise: prod == lean_u1; lean32ek_fast == lean32ek at 40/400/4,000 steps; dt-ladder n=40 == lean_u1.
    - **lean32ek_fast 2.694 ms/step = 3.45× production**, bitwise to lean32ek (3.340 ms). lean32sr 2.789,
      lean_u1 6.405.
    - Control a: κ 1.08 / 1.00 / 0.99 at k < 8.
    - Alias split 0.084: lean32ek's large-scale error in o1.2's frames is mostly folded small-scale noise.
    - Production's own dt sensitivity |D(2.5e-6)|/|Δ| is 2.3e-4 at k 1–2 on o1.2's subsampled frames, but 9e-7
      alias-free. The step error reaches the targets' large scales only through aliasing, at ≈ 8× lean32ek's
      float32 error there.
  - **Control-b design fixed after the smoke** (PREDICTIONS addendum, before the archive run).
    - The pre-set "z ≥ 3" was unsigned, but the planted shift damps (α < 0).
    - Its ratio α_inj/α_inj64 contained lean32ek's own coefficient, so it would void the run exactly when lean32ek is
      biased.
    - Now paired: recovery α(inj − lean32ek)/α(inj64) on the same pairs. Smoke: 0.987–1.079. Power: |α(inj64)|/SE.
      Smoke: 5.3–30.6. Both PASS.
    - Leak, noted honestly: on the near-duplicate smoke states, the old z values hinted at an x-aligned component in
      lean32ek's error at k 2–8. That is not evidence for archive states.
    - Negative controls flag 2.1–2.3% of shells on the 4-cluster smoke (nominal 0.27%; 16-cluster test 0.37%). A
      reading rule for P2/P8 is fixed in the addendum.
  - **Code** in `SQG/analysis/opt/taxonomy/` (commit 3854307c):
    - `taxonomy.py`, the per-GPU worker;
    - `lean32_fast.py`: lean32ek with the u1 trigger + 3-slot ring, which must be bitwise, and a stochastic-rounding
      control;
    - `dt_ladder.py`: a variable-step AB3 switch, then lean_u1's code at h;
    - `analyze_taxonomy.py`: per-band and per-shell clustered regressions of the paired error on (x, Δ), the MDE
      before any z, and two positive controls;
    - `PREDICTIONS_taxonomy.md`: validity gates V1–V4 + 13 predictions.
  - **Tests (stdlib / numpy, login node).**
    - dt-ladder order ratios −0.637 / −1.507 vs theory −0.639 / −1.500.
    - The variable-step switch adds 1e-9 vs 1e-6 for naive weights.
    - Analysis: the null flags 0.37% of shells; a planted band bias is recovered at 0.93 (z 7.5); a per-shell
      random-sign bias is flagged in 100% of its shells at 0.99 of its size; end-to-end passes on a fake job dir.
  - **Design facts worth keeping.**
    - float32's damping error is a per-SHELL bias with random sign. A per-band pooled fit under-reads it, so the
      analysis also fits per shell.
    - The x-coefficient of a damping-rate error is −δr·0.02 whether the band is static or swept. That makes it the
      robust control-a prediction; the d-coefficient is not.
  - **Sealing (user's choice).** Until jesswan/Weidong sign item 4's gates, the bias statistics of lean32ek, dt
    2e-4/30 and dt 2e-4/32 go to `$MEMBER_ROOT/sqg_profile/taxonomy_<job>/sealed/sealed_taxonomy.json`. Only its
    sha256 is committed.
  - **Blocked:** the 16 archived checkpoints (members 1–4 × `ckpt_t0{100,200,300,400}.00.npz`, ≈ 6.4 GB) must be
    copied from Midway `/project/pedramh/weidong/SQG/SQG_deterForcing/AllTrainData/<label>/checkpoint/` to
    `$MEMBER_ROOT/sqg_stage/AllTrainData/<label>/checkpoint/` (user, byte-for-byte). The job refuses to start without
    them (`TAXO_CKPTS_NOT_STAGED`).

- **2026-11-06** — **SQG: the math of the delta table, a three-agent "what else to test" plan, and analogues from
  other fields** (no jobs submitted). Everything is in `SQG/analysis/opt/team_2026-11-06/` and
  `SQG/analysis/opt/delta/MATH_delta.md`.
  - **Math (`MATH_delta.md`).**
    - e_Δ = e_frame · ‖F‖/‖Δ‖ exactly. The measured ratio is identical across variants: 32 at lag 40, 1.83 at 4,000.
    - The state-rounding estimate u|θ|/|i| reproduces `lean32`'s 1-step error band by band (6.1e-3 vs 5.3e-3 at
      k1–2).
    - `lean32`'s k1–2 error is flat or rising with lag, i.e. a bias. TwoSum/Fast2Sum leaves `lean32ek` with only
      the float32 tendency error.
  - **Team (workflow `wf_0391b43a-95e`: Opus NUMERICS + Opus LEARNING + Fable PS, 5 rounds, read-only).**
    - **The finding that reframes the program:** o1.2's `[::16]` subsample aliases 4096² k ≥ 128 content into the
      target. That puts **0.00222 (lo) / 0.00377 (central)** of o1.1's 0.0043 val MSE out of reach of any data,
      step or precision change. Main session independently got ≈ 0.0020.
    - **PS plan** (8 items, ~26 node-h; the decisive five ≈ 4 node-h):
      1. free stdout/checkpoint parse: pins 168 vs ~142 h, counts restarts, retrieves the final vals;
      2. per-band val vs the floor table: the pivot test, replay + fewer steps vs online value;
      3. noise-vs-bias taxonomy of generators on 16 archived states, with two positive controls;
      4. an owner-signed acceptance protocol before unblinding;
      5. a 4.3 TB tee/replay of the 256² stream;
      6. compile + activation checkpointing;
      7. resume from `last.pt`;
      8. compressed OneCycle + the first two-seed replication.
    - **Rejected:** 2048² restarts, densification, a custom f64 FFT, offset views.
    - **Main-session check:** the solver's own "2.2e8 steps ~287 h on an H100" gives 4.7 ms/step, which leaves
      ≈ 0.45 s/step (≈ 24 h of 168 h) of non-solver overhead.
    - **Correction:** channels_last is 38.5% *slower*, not a gain. The R1 brief had it backwards; memory fixed.
  - **Analogues (`ANALOGUES.md`, 14 systems).**
    - Klöwer et al. 2022 used compensated time integration in Float16 shallow water, i.e. `lean32ek`. It beat mixed
      precision there too (3.6× vs 2.75×).
    - Stochastic rounding (Paxton 2022) turns the bias into noise. Proposed as a calibration arm for item 3.
    - Melissa's online gain is new-data diversity, which o1.2 (re-integrating the archive) does not buy.
    - The online loop that adds information is on-policy (Rasp 2020; Frezat 2022).
  - **Next:** owners' answers to the 8 questions in `TEAM_REPORT.md`. Item 1 needs the user to point at the o1.2 /
    o1.1 logs.
- **2026-11-06** — **SQG delta-fidelity job 7819310 DONE (`DELTA_JOB_DONE failures=0`), notes Part XI. Plain float32
  damages o1.2's training delta; a compensated float32 state fixes 98% of it.** Results in
  `SQG/analysis/results/polaris_delta_7819310/`.
  - **Median e_Δ at the 4,000-step lag over 8 pairs** (spun-up synthetic state):

    | variant | all | k1–2 | k2–4 | ms/step |
    |---|---|---|---|---|
    | `lean32` | **5.5e-4** | **3.6e-3** | **2.0e-3** | 3.20 |
    | `lean32e` (expm1 damping) | 4.3e-4 | 3.2e-3 | 1.4e-3 | 3.20 |
    | `lean32ek` (compensated) | **8.8e-6** | 2.2e-5 | 5.2e-6 | **3.35** |
    | `mixed` | 8.7e-6 | 2.2e-5 | 5.2e-6 | 4.73 |
    | float64 from a once-rounded state (floor) | 2.3e-7 | | | |
  - **`lean32`'s error is systematic:** its frame error grows 160× over 100× the steps; `lean32ek`'s grows 13×.
    Its 1-step increment error at k1–2 is 5.3e-3, vs 4.3e-6 for `lean32ek` / `mixed`. The compensation survives
    XLA.
  - **Pre-set gate (≤ 2× floor): all four FAIL.** `mixed` / `lean32ek` are 37–38× the floor (float32 FFT term).
    Verdicts stand; the gate is not loosened.
  - **Predictions:** 5 confirmed, 2 half, 4 refuted. The floor and rms(Δ) were off, and `mixed` / `lean32ek` were
    predicted to pass.
  - **Decision is the user's / jesswan / Weidong's:**
    - **Do not use plain float32 (`lean32`, `lean32_u1_ring3`) for o1.2's targets.**
    - If a pre-set tolerance at ≈ 1e-5 of Δ is acceptable, use `lean32ek` (2.77×; with the trigger + ring
      ≈ 3.4×, untested).
    - Otherwise stay on float64: `lean_u1` 1.45×, round 2 ≈ 1.7×.
  - **Caveat:** the start state is not archive-like (rms(Δ) 0.825 vs 0.27). A definitive rerun needs 2–3 archived
    checkpoints staged from Midway (`delta_fidelity.py --state-in`), plus a long-run bias check.
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

- **(Polaris, SQG) NVIDIA Warp + jax 0.4.28: version box and four traps** (jobs 7839932, 7840069; notes Part XVII B).
  - **Version:** only warp-lang ≤ 1.17.0 has the legacy XLA custom call that JAX 0.4.25–0.7.x can use. 1.18 removed
    it, and the FFI wrappers need JAX ≥ 0.5. Warp's own legacy `jax_kernel` hard-codes block_dim 256 and
    launch-dim-shaped outputs, so it cannot run tile kernels; use `SQG/analysis/opt/warp/warp_xla.py`.
  - **Custom-call name:** jaxlib 0.4.28's `register_custom_call_target` takes the target name as `str`, not `bytes`.
  - **Tile storage:** a tile has one storage class per kernel. Transposing an FFT'd register tile moves it to shared
    memory and `tile_fft` stops compiling. Transpose a `tile_map(ident, ·)` copy instead.
  - **Vector class:** `wp.vec2f(...)` inside a `@wp.func` returns an anonymous `vector(2, float32)` class, which
    `tile_fft` rejects. Build results from a typed argument (`r = p * 0.0`, then set components).
  - **Limits:** register tiles of (K, 4096) at 512–1,024 threads, and R ≥ 4 row tiles, exceed the register limit.
    Strips with K = 4 need 147–279 KB of shared memory.
  - **Speed, not a bug:** Warp's column FFT is 2× cuFFT's (694 vs 348 µs, complex64 4096²). Keep cuFFT for
    transforms. Its row FFT is near cuFFT (232 vs 206 µs).
  - **FMA contraction:** Warp's module option `fuse_fp` defaults to True. Error-free transforms (TwoSum / Fast2Sum)
    need `module_options={"fuse_fp": False}`.
  - **Constants:** closure constants must be Python int / float (or Warp types), not numpy scalars. A
    `np.float32` gives "Invalid external reference type" at parse time (job 7840360). Python floats are typed
    float32 in kernels.

- **(Polaris, SQG) Fusing the SQG step's pointwise work into cuFFT via LTO callbacks (nvmath-python) is slower at
  4096², not faster** (job 7840175; notes Part XVII C).
  - Measured: each C2C went from 0.60 ms (plain) to 1.09–1.13 ms with our load/store callbacks; the C2R with a
    load callback took 0.52 ms. The whole hybrid step is 2.94 ms vs XLA's `lean32ek_fast` 2.71.
  - Not yet separated: the callback path's own cost vs our callbacks' work (64-bit index math, mirrored gathers,
    scattered kept-mode stores).
  - Before retrying, measure an identity-callback C2C at 4096². Don't re-run the full hybrid as is.
  - Setup that worked: nvmath-python 1.0.0 + numba-cuda 0.30.4 `--no-deps` beside sqg-env, plus `LD_PRELOAD` of
    CUDA 12.9.1's `libnvJitLink.so.12` and `libcufft.so.11` (11.4.1). LTO callbacks need cuFFT ≥ 11.3; the env's
    wheel is 11.0.2.

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
