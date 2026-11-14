# Can NVIDIA Warp carry the SQG solver's custom FFT? Predictions for the Warp probe (written 2026-11-12, before any run)

**Ask (user, 2026-11-12):** "another way of doing the custom FFT is using nvidia warp". Research round 2
(`../research_2026-11-12/SOLVER_SPEED_RESEARCH.md`, items #6–#8) found the custom-FFT routes blocked: jax 0.4.28 has
no FFI, and a JAX upgrade collides with torch 2.3.1's cuDNN in o1.2's process.

**Why Warp could unblock it (checked against the Warp 1.17.0 source before writing this):**
- **Bridge:** Warp ≤ 1.17.0 still ships a legacy XLA custom-call bridge
  (`warp/_src/jax/custom_call.py`: "requires JAX version 0.4.25 - 0.7.x"). Warp 1.18.0 removed it, and its FFI
  wrappers need JAX ≥ 0.5.0. So the probe pins **warp-lang 1.17.0** (PyPI wheel, CUDA 12 build), installed with
  `pip --target $MEMBER_ROOT/pylibs/warp-1.17.0`. sqg-env itself is not modified.
- **Our own wrapper:** that bridge hard-codes `block_dim = 256` and makes every output's shape equal the launch
  dims, so it cannot launch tile-FFT kernels. `warp_xla.py` (this directory) re-implements the same legacy
  mechanism with explicit output shapes and any tiled `block_dim`:
  - a ctypes callback registered with `xla_client.register_custom_call_target`;
  - `module.load(device, block_dim)` at lowering;
  - `wp_cuda_launch_kernel` on XLA's stream.
- **FFT engine:** `wp.tile_fft` / `wp.tile_ifft` are cuFFTDx (libmathdx), item #8's engine, written as Python
  kernels.
  - C2C only (`vec2f` / `vec2d`), along a register tile's last axis; ≥ 2 elements per thread.
  - A column pass is a shared-memory strip load → `tile_transpose` (aliasing view) → `tile_map` copy into
    registers → FFT → store.
- **Driver:** Polaris A100s run driver 580.65.06 (gate D job log), which covers the CUDA 12 wheel.

## What the probe measures (one A100, `warp_fft_probe.py`; everything 4096², the solver's N)

| id | quantity | how |
|---|---|---|
| M0 | attainable bandwidth `BW_copy` | torch `y.copy_(x)`, complex64 4096², 268.4 MB moved |
| M1 | cuFFT baselines | torch `fft(dim=-1)`, `fft(dim=0)`, `fft2`, complex64 and complex128. Also jax `jnp.fft.fft2` (complex64), the anchor to notes Part X's C2C (540 µs) |
| M2 | Warp row FFT | `(R, 4096)` tiles, R ∈ {1, 2}, block_dim ∈ {128, 256, 512} (vec2f); R = 1, block_dim ∈ {256, 512} (vec2d); best config also inverse, and with `enable_mathdx_fft = False` (scalar fallback) |
| M3 | Warp column FFT | strip kernels, K columns per block (K ∈ {1, 2, 4} vec2f, {1, 2} vec2d). `col_strided` stores back into the same layout; `col_tstore` stores the result transposed (contiguous rows). Also NVIDIA's own 3-kernel recipe (row FFT → 16×16 transpose → row FFT) as a 2-D C2C |
| M4 | fused row kernel (proxy for the step's row kernel) | per row: load 2 rows, 2 inverse FFTs, conj(a)·b, 1 forward FFT, store 1 row: 3 FFTs, 402.7 MB moved |
| M5 | accuracy | rel L2 error vs numpy complex128 of the same input: Warp vs torch cuFFT, row / column / 2-D |
| M6 | bridge | jit correctness vs `jnp.fft.fft`; a 50-iteration `lax.scan` vs the same jnp loop; per-call host overhead (tiny-kernel scan of 2,000 iterations vs a jnp scan); two Warp kernels in one jit; torch still works afterwards |

Timing: CUDA events around 20 back-to-back launches after 3 warm-ups, median of 5 batches, per launch. JAX: a jit
call ×20 then `block_until_ready`, median of 5 batches. "Bandwidth" = unique bytes read + written / time.

## Byte model of a fused Warp step (the projection the decision uses)

Three kernels per step, pruned to the kept 2731 × 1366 block (the decay table carries the 2/3 mask, research #3):
- **A, column inverse along ky:** reads θ̂ hi and `kinv`; builds ŵ = (k − i l) ψ̂ on load for +kx and −kx; writes
  W (4096 × 2731) and Θ (4096 × 1366) in mixed (y, kx) space. 4,097 column FFTs.
- **B, rows along x:** reads the W and Θ rows; computes u − i v and θ (2 inverse FFTs), Z = conj(w)·θ, 1 forward FFT,
  then the Hermitian split. Writes the kept 1366 columns of the two products' half-spectra. 3 × 4096 row FFTs.
- **C, column forward along ky:** reads the two products' columns; tendency, AB3 and update (compensated hi/lo for
  float32) on load. Writes the state and the tendency. 2 × 1366 column FFTs.

| | A | B | C | total | today's ideal (Part X) |
|---|---|---|---|---|---|
| float32 (lean32ek: hi + lo state) | 179.0 MB | 223.8 MB | 313.4 MB | **716.1 MB** | 2,819 MB |
| float64 (lean_u1's numerics) | 358.0 MB | 447.5 MB | 507.3 MB | **1,312.9 MB** | — |

(MB = 1e6 bytes. Block 2731 × 1366 = 3,730,546 modes; a mixed half-plane is 4096 × 1366, W is 4096 × 2731; tables are
real. The forcing patch is ignored. `warp_fft_probe.py` recomputes these from the shapes and prints them.)

- **Projection:** `t_step = bytes_A / BW_col + bytes_B / BW_fused_row + bytes_C / BW_col`.
  - `BW_col` = the best measured column-strip bandwidth (M3, same dtype);
  - `BW_fused_row` = M4's for float32, and M2's best row bandwidth for float64 (no float64 M4).
- **Float64 compute floor:** 19,117 length-4096 FFTs × 5·4096·12 flops = 4.70 GFLOP/step, i.e. ≥ 0.48 ms at the
  A100's 9.7 TFLOPS FP64. Measured float64 row bandwidth shows whether it binds.

## Predictions (P = probability I give it)

| # | prediction | P |
|---|---|---|
| 1 | the 1.17.0 wheel has libmathdx (`wp_is_mathdx_enabled() == 1`), and a 4096-point vec2f `tile_fft` compiles through it | 0.85 |
| 2 | warp 1.17.0 imports and runs kernels in one process with torch 2.3.1 + jax 0.4.28 (driver 580) | 0.90 |
| 3 | the bridge's row FFT under `jax.jit` matches `jnp.fft.fft` (rel L2 ≤ 1e-6), and the 50-iteration scan matches the jnp loop (rel L2 ≤ 1e-5) | 0.65 |
| 4 | bridge host overhead is 5–40 µs per custom call (tiny-kernel scan minus jnp scan, per iteration) | 0.70 |
| 5 | the best vec2f row FFT reaches 70–90 % of `BW_copy`, and is within ±15 % of torch `fft(dim=-1)` | 0.65 |
| 6 | the best vec2f column strip reaches 35–65 % of `BW_copy`, i.e. not more than 10 points above cuFFT's column pass (51 %, Part X) | 0.60 |
| 7 | K = 4 vec2f strips fail to compile or launch (shared memory > 163 KB per block) | 0.60 |
| 8 | the fused 3-FFT row kernel (M4) stays ≥ 60 % of `BW_copy` | 0.60 |
| 9 | Warp's complex64 errors are ≤ 2× torch cuFFT's (row and 2-D, vs complex128) | 0.80 |
| 10 | the best vec2d row FFT reaches ≥ 50 % of `BW_copy`, within ±25 % of torch complex128 `fft(dim=-1)` | 0.50 |
| 11 | the scalar fallback (`enable_mathdx_fft = False`) is ≥ 1.5× slower than cuFFTDx on the vec2f row FFT | 0.75 |
| 12 | anchor: `jnp.fft.fft2` complex64 is 0.46–0.62 ms (Part X's C2C 540 µs) | 0.85 |
| 13 | projected fused float32 step: 0.9–1.6 ms (today `lean32ek_fast` 2.698 ms) | 0.55 |
| 14 | projected fused float64 step: 1.8–3.5 ms (today `lean_u1` 6.409 ms) | 0.50 |

## Decision rule (fixed now; the probe only measures)

- **BRIDGE_OK** iff prediction 3's two correctness checks pass, the overhead is ≤ 50 µs per call, and torch still
  works afterwards.
- **GO_F32** iff BRIDGE_OK, the best vec2f column strip and the vec2f row FFT are each ≤ 2× torch cuFFT's error
  (M5), and the projected float32 step is ≤ 1.6 ms (≥ 1.69× today's 2.698).
  - Next step: build kernels A–C in float32 (lean32ek's compensated update).
  - It is gated like any float32 rounding change: the pre-set f32 tolerance, then the delta/taxonomy harness.
- **GO_F64** iff BRIDGE_OK and the projected float64 step is ≤ 2.70 ms (no slower than `lean32ek_fast`).
  - Next step: build kernels A–C in float64.
  - That path needs only the existing float64 gate (≤ 1e-12 relative after 1 step). No statistical-gate campaign,
    no precision question.
- **NO_GO** otherwise, with the failing condition named.
- **Not decided here:** the verdict tokens are computed by `warp_fft_probe.py` from the rule above. A projection is
  not a measured step; the A–C prototype job is what measures one.
- **Thresholds:** none of them changes after the run. A changed rule gets a dated, post-hoc-labelled addendum below.

## Tokens

- `WARP_ENV_OK`, `WARP_BRIDGE_OK` / `WARP_BRIDGE_FAIL <reason>`, `WARP_VERDICT f32=<GO_F32|NO_GO> f64=<GO_F64|NO_GO>`,
  `WARP_PROBE_DONE failures=N`.
- The job ends `WARP_JOB_DONE failures=N`.
- Results → `SQG/analysis/results/polaris_warp_<jobid>/` (`warp_probe.json`, logs).

## Addendum 2026-11-12, before the run (job 7839932 still queued; no number seen): reading notes for scoring

Wording only; no threshold changes.
- **Prediction 6** mixes two references. Its "35–65 %" is of `BW_copy`. Its "51 %, Part X" is a fraction of the
  A100-40GB DRAM peak (1,555 GB/s). Each clause is scored against its own reference: frac_copy ∈ [0.35, 0.65] and
  GB/s / 1,555 ≤ 0.61. The measured torch `fft(dim=0)` is reported next to them.
- **Prediction 7's "strips"** are the two shared-memory strip kernels (`col_strided`, `col_tstore`) at K = 4.
  `col_tview` loads through a strided array view and holds no strip.
- The scorer is `score_warp.py`. It moves into this directory after the job ends, so the queued job's inputs stay
  unchanged.

## Addendum 2026-11-12, AFTER round 1 (job 7839932) and BEFORE round 2: fixes, and predictions for round 2

**Round 1** (debug, x3001c0s7b1n0, 105 s, `WARP_JOB_DONE failures=0`). Verdict `NO_GO / NO_GO`, invalid: the bridge
never ran, because of my code (below). Score from `score_warp.py`: **7 confirmed, 3 half, 0 refuted, 4 untested**
(P3, P4, P8, P13 had no value to score). What it measured, A100, 4096² per pass:

| | Warp | cuFFT (torch) |
|---|---|---|
| row FFT complex64 (R 1, 256 threads) | 230.9 µs, 1,163 GB/s, 85% of copy (1,361 GB/s) | 206.2 µs |
| row FFT complex128 | 566.4 µs, 948 GB/s, 69% of copy | 403.5 µs |
| column complex64 (`tstore`, K 1) | 696.6 µs, 385 GB/s, 28% | 345.4 µs (`fft(dim=0)`) |
| column complex128 (`tstore`, K 1) | 923.8 µs, 581 GB/s, 42% | 674.9 µs |
| 32 × 32 transpose complex64 | 238.0 µs, 1,128 GB/s, 83% | — |

- `jnp.fft.fft2` 525.8 µs: the Part X anchor holds.
- Accuracy (rel L2 vs complex128): Warp rows 1.86e-7 vs torch 2.75e-7; Warp 2-D (NVIDIA's 3-pass) 2.77e-7 vs torch
  4.02e-7.
- Projected fused float64 step **1.961 ms**. The float32 projection had no fused-row number.

**Four failures, all bugs in this directory's code (fixed for round 2):**
1. **Bridge:** jaxlib 0.4.28's `register_custom_call_target` takes the target name as `str`; `warp_xla.py` passed
   `bytes`.
2. **`col_strided`:** transposing the FFT'd tile moved it to shared memory for the whole kernel, so `tile_fft` failed
   to compile. Round 2 transposes a register copy.
3. **`fused3`:** a `wp.vec2f(...)` constructor inside a `@wp.func` returns an anonymous `vector(2, float32)` class,
   and `tile_fft` accepts only the `vec2f` / `vec2d` classes. Round 2 builds the result from the typed argument
   (`mul(vector, scalar)` returns the argument's own class).
4. **`col_tview`:** "too many resources requested" at 512 / 1,024 threads (registers). Round 2 runs 256 / 128.

**What round 2 adds (new configs, chosen after seeing round 1):** row tiles R 2 / 4 at 512 threads, and `rowT`: a
row FFT of R = 2 / 4 rows stored transposed (the transpose fused into the store).

**Unchanged:** the decision rule, every threshold, the byte model, and P1–P14 (re-scored on round 2). Round 2's
verdict replaces round 1's.

**New predictions for round 2 (P = probability):**

| # | prediction | P |
|---|---|---|
| 15 | the fixed bridge gives BRIDGE_OK | 0.70 |
| 16 | fixed `col_strided` K = 1 (vec2f) is within 1.15× of `col_tstore` K = 1's 696.6 µs | 0.60 |
| 17 | `col_tview` launches at 256 threads for K ≤ 2, and its best vec2f point reaches ≥ 35% of copy | 0.40 |
| 18 | fixed `fused3` (vec2f) reaches ≥ 60% of copy (= P8, now testable) | 0.60 |
| 19 | `rowT` R = 4 (vec2f) reaches ≥ 50% of copy | 0.35 |
| 20 | round 2 reproduces round 1's row complex64 and column `tstore` K = 1 times within ±5% | 0.85 |
