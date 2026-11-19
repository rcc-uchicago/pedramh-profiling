# NVIDIA Warp skills run on the SQG project (2026-11-19)

**Ask (user, 2026-11-19):** install `warp-compile-time-optimizer`, `warp-debug-gradients` and `warp-eval` from
<https://github.com/NVIDIA/skills> "and run them on this project to see if there any other places we can improve warp".

- **Installed** to `~/.claude/skills/<name>/` from NVIDIA/skills `main` @ `4fbbff39da1a5780d8bfd796f1674618c6349ef7`
  (raw files; each directory has a `.source` line with the pinned URL).
- **Run statically, on evidence already in `SQG/analysis/results/`.** No GPU job was spent on them:
  - the login node may not run the venv Python (operator ruling);
  - `warp-eval` forbids measured work without a separate authorization;
  - every skill's first decision could be settled from existing measurements.
- **Code changed: none.** Every finding below is a measured fact or a bound computed from one.

**Verdict:** none of the three skills finds a material Warp opening left in this project.
- Compile time is ≤ 1.4% of the shortest job.
- No Warp gradients exist.
- Every remaining o1.2 stage is either a vendor-library / standard-NN region (`warp-eval` gate C) or ≤ 2.3% of the
  step (gate F).
- The largest remaining lever is the UViT ML step, about 48% of o1.2's step. It is outside Warp's scope. Probe 2
  already measured its precision knobs: tf32 −9.0%, bf16 −11.3% of the o1.2 step.

## 1. `warp-compile-time-optimizer`: compile time is not the bottleneck

The skill's step 1 says to measure the real command cold, and to stop if compilation is a small part of wall time.
Every SQG job already measures exactly this: each worker or rank gets a fresh per-job `WARP_CACHE_PATH`, so every job
is a cold start, and Warp logs each module's load. `warp_compile_census.py` tallies those lines from 18 jobs.

| job | what | Warp compile per worker / rank | job wall | share |
|---|---|---|---|---|
| 7854208 | gates A/B, 2 solver arms | 8.8 s (35.1 s over 4 workers) | 1,098 s | 0.8% |
| 7854210 | gate E, 2 arms | 8.5 s | 2,228 s | 0.4% |
| 7854382 | o1.2 probe 2, 3 arms (arms 2–3 hit the cache) | 8.2 s | ~10 min | 1.4% |
| 7859359 | mixprec probe 1 (crashed) | 7.8 s | 150 s | (crash) |
| production o1.2 | 49 h on 14 A100s | ~8 s once per rank | 49 h | 0.005% |

Per module (in-row solver, cold, A100):

| module | cold build | what it is |
|---|---|---|
| `rows_ir` | 3.4–6.4 s | the tile-row kernel: 3 cuFFTDx tile FFTs (MathDx LTO link) |
| `build_w` | 1.1–2.3 s | SIMT |
| `update` | 0.14–0.21 s | SIMT (the older tilerows `update` took 1.1 s, job 7840805) |
| `halve0` | 0.11–0.14 s | SIMT |

What the skill's mechanisms find:

| mechanism (skill) | status | evidence |
|---|---|---|
| backward codegen on forward-only kernels | **already off, correctly** | `module_options={"fuse_fp": False, "enable_backward": False}` on `module="unique"` kernels (the form the skill measured as also removing tile adjoint links), and `wp.config.enable_backward = False` before `wp.init()` in `o12_polaris.py`, `tr_pairs.py`, `stability_run.py` |
| identity churn (one module, several hashes) | **none** | the extra hashes are distinct kernels: dt / IF mode / order are compile-time constants per arm; within a process each identity builds once and later arms load it `(cached)` |
| block-dimension variants | **none** | one `block_dim` (256) per tile kernel |
| cache never reused | **by design** | per-job, per-worker caches; a persistent cache would save the ~8 s per job |
| overlapping independent builds | **ceiling 1.4×, ≤ 2.4 s per job** | 8.1 s summed / 5.65 s largest (`rows_ir`) in 7854382 |
| MathDx / LTO for `tile_fft` | **required** | the scalar fallback ran 4.6× slower at run time (`WARP_BUILTINS_AUDIT_2026-11-14.md`) |

**Measured and declined** (in the skill's ledger form):

| option | bound | why not taken | to take it |
|---|---|---|---|
| shared persistent `WARP_CACHE_PATH` per snapshot | −8 s per job (≤ 1.4% of a debug job) | not material; per-worker caches keep concurrent workers independent | point `WARP_CACHE_PATH` at `sqg_profile/warp_cache_<snapshot>` in the PBS scripts |
| `wp.force_load(..., max_workers=4)` before the first launch | ≤ −2.4 s per job (ceiling) | not material | one call after `wp.init()` at each entry point |

## 2. `warp-debug-gradients`: not applicable

The skill covers workflows trained through `wp.Tape`, and says not to activate for forward-only Warp work.
- **Nothing in SQG differentiates through Warp.** There are 0 `wp.Tape` uses, and no Warp array has
  `requires_grad=True`.
- `enable_backward` is off in every step module and at every entry point (part 1).
- **The boundary** is one DLPack hand-off of the solver state per frame, JAX ↔ torch
  (`o12run/tilerows_advance.py:267,276`). The UViT trains by torch autograd on frames that are data: DLPack imports
  are leaves with `requires_grad=False`.
- There is nothing for its finite-difference or overwrite-tracker checks to verify.

## 3. `warp-eval`: every remaining seam exits at a gate

**Warp is the option under evaluation. Per the skill, no adoption recommendation follows.**

**Contract (inferred; correct me if wrong):**
- o1.2 online training on Polaris A100 40 GB, NVIDIA-only;
- 1 rank per GPU, batch 16 per rank, 12 frames × 25 solver steps per optimizer step;
- strict FP32 network, float32 + compensation solver, no gradients through the solver.

Gates A and E do not fire: Warp is already a dependency of the solver.

**Stage census** of one o1.2 optimizer step, 0.8888 s. Sources:
- job 7854382 (`ieee` arm, `tilerows_inrow_if25`), s_frame = 38.76 ms;
- job 7853317's per-op split of the 1.48 ms solver step;
- job 7840909's per-frame glue breakdown;
- job 7809408's UViT kernel families.

| stage | ms / step | % of step | status | reason |
|---|---|---|---|---|
| solver, Warp kernels (`build_w`, `halve0`, `rows_ir`, `update`) | ~235 | 26.4% | excluded | already Warp: `warp-eval` excludes already-selected Warp kernels. They run at 67–96% of the copy roofline (audit) |
| solver, cuFFT column passes (W, θ, Z) | ~210 | 23.6% | **ABORT, gate C** | FFT on a vendor library. Measured: Warp tile columns 2× slower (694 vs 348 µs, 7840069); fusing via cuFFT callbacks doubled the FFT cost (7840175) |
| frame glue (entry sync 1.04 + hand-off/expand 0.54 + outside `advance()` 0.10 ms, × 12) | ~20 | 2.3% | **ABORT, gate F** | ceiling with the stage free: 0.8888 → 0.8686 s, **≤ 1.023×** |
| UViT forward + backward + AdamW (ML step) | ~424 | 47.7% | **ABORT, gate C** | standard NN layers in PyTorch (cuBLAS SGEMM, cuDNN and native depthwise conv, LayerNorm, GELU) |
| per-run / per-epoch (init, fill, segment loads, epoch overhead) | — | 1.4% of the 49.2 h schedule | **ABORT, gate F** | 0.69 h of 49.2 h, **≤ 1.014×** |

The UViT row, broken into kernel families (job 7809408; share of the UViT's GPU time, then of the o1.2 step):

| family | % of UViT | % of o1.2 step |
|---|---|---|
| GEMM (fp32 SGEMM) | 26.4 | 12.6 |
| elementwise | 19.8 | 9.4 |
| depthwise 7×7 conv (PyTorch native kernels) | 19.2 | 9.2 |
| LayerNorm (`GammaBetaBackward` 8.3% alone) | 15.3 | 7.3 |
| copies (circular `F.pad`, NCHW↔NHWC permutes; 1,012 launches/step) | 5.2 | 2.5 |
| cuDNN conv | 4.5 | 2.1 |
| memcpy / memset, reduce, other, optimizer | 9.7 | 4.6 |

**Memory**, the metric for the glue's one intermediate (the 134 MB complex128 `qh`):
- peak is 20.9 GiB of the 40 GB A100;
- the UViT's activations are 20.24 GiB at batch 16 (batch 32 OOMs);
- all solver buffers together are < 1 GB (2.5% of the device).
- Gate F on memory too.

**Exit lines** (the skill's format):
- `ABORT — Gate C: the cuFFT column passes are FFTs on a vendor library (cuFFT 11.0.2); Warp tile columns measured 2× slower (7840069).`
- `ABORT — Gate C: the UViT step is standard NN layers (SGEMM, conv, LayerNorm, GELU) in PyTorch; the tuned framework path is the baseline.`
- `ABORT — Gate F: o1.2's frame glue is 20 ms of a 0.889 s step (7840909, 7854382); with it free the step is ≤ 1.023× faster.`
- `ABORT — Gate F: init, fill, segment loads and epoch overhead total 0.69 h of the 49.2 h schedule (7854382); ≤ 1.014×.`

No measured work was done, so no `warp-evaluation-report/` directory exists (the skill creates none on an early
exit).

## 4. Inside our own Warp kernels: what is left (from the 11-14 audit, unchanged)

| item | bound | status |
|---|---|---|
| in-tile C2R (audit candidate 1) | −7.0% per solver step | **done**: `tilerows_inrow_if25`, gates A/B/E/D all pass (Part XVII I, J) |
| skip rewriting W's zero rows (candidate 2) | ≈ −0.03 ms per solver step ≈ −9 ms per o1.2 step (1.0%) | not run; needs a bitwise check (cuFFT out-of-place may pick other kernels) |
| CUDA-graph capture of the step (candidate 3) | 1% (1.467 vs 1.482 ms, 7853317) | measured |
| Warp 1.18 / 1.19 built-ins | nothing for this step | audit |

The step is now about half solver and half network. Every solver kernel except the cuFFT columns already runs near
the copy roofline, so any further Warp work on the solver is bounded at about 1% of o1.2 per item.
