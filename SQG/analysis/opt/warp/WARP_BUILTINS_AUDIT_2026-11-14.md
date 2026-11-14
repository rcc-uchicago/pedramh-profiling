# Warp built-ins audit for the SQG tile-row step (2026-11-14)

The user asked whether we had read Warp's built-ins reference in full and used every optimization it offers.
Before today there was no record of a full read (no SQG note or CHANGELOG entry cites the page). This note is
that read, checked against the step we actually run (`tilerows/tilerows_step.py`, the `tilerows_if25` arm).
No code was changed and no job was run.

## What was read

- **Page:** https://nvidia.github.io/warp/stable/language_reference/builtins.html (**Warp 1.19.0**), downloaded
  and read in full: the index lists 300 built-ins in 14 categories.
- **Detail pages read in full** (`_generated/warp.<name>.html`, 179 built-ins): Tile Primitives (64), Scalar Math
  (41), Vector Math (33), Utility (23), Operators (16), Other (1), Code Generation (1).
- **Index entries only** (121 built-ins): Quaternion, Transformations, Spatial Math, Geometry (BVH / mesh / hash
  grid), Volumes, Textures, Random. These are rigid-body, mesh, volume and RNG tools; none applies to a
  doubly periodic pseudo-spectral step.

## Our version: 1.17.0, and what 1.19 adds

- We run Warp **1.17.0** from `$MEMBER_ROOT/pylibs/warp-1.17.0` (pip `--target`; logged in every tile job).
- Diffing the 1.19 index against 1.17.0's `_src/builtins.py` registry: 19 built-ins are new since 1.17.0, all
  outside the solver's needs (`bit_count`, `norm_huber`, `norm_l1`, `norm_l2`, `norm_pseudo_huber`,
  `smooth_normalize`, six `quat_*` helpers, six `transform_*` helpers, `velocity_at_point`).
- One relevant 1.19-only option: `tile_matmul(precision="tf32")`. It changes the arithmetic, so it is excluded
  by CLAUDE.md rule 1 regardless.
- Already in 1.17.0: every tile primitive on the page, `tile_load` / `tile_store(aligned=...)`, the 3–8-argument
  `tile_map`, `tile_axpy`, `tile_empty`, `div_approx`, reversed tile views (`t[::-1, :]`), `tile_load_indexed`,
  and the cuFFTDx backend switch `enable_mathdx_fft` (default on).
- **Upgrading to 1.19 gains nothing for this step.**

## Where the step's time goes (job 7840376, A100, N = 4096, KM = 1365)

Unique bytes read + written per op (the convention of `warp_fft_probe.py`), against the 1,360 GB/s complex64
copy measured in job 7840069:

| op | kind | ms | unique MB | TB/s | % of copy |
|---|---|---|---|---|---|
| `build_w` | SIMT | 0.147 | 134.3 | 0.91 | 67% |
| `cufft_cols_w` | cuFFT, 2,731 kept columns | 0.275 | 179.0 | 0.65 | 48% |
| `expand` | SIMT | 0.079 | 97.0 | 1.23 | 90% |
| `cufft_c2r` | cuFFT 2-D C2R | 0.302 | 134.3 | 0.44 | 33% |
| `tile_rows` | Warp tile, 2 cuFFTDx FFTs/row | 0.302 | 335.5 | 1.11 | 82% |
| `cufft_cols_z` | cuFFT, 2,731 kept columns | 0.276 | 179.0 | 0.65 | 48% |
| `update` | SIMT | 0.216 | 283.5 | 1.31 | 96% |
| sum (eager step measured 1.578) | | 1.598 | | | |

The SIMT kernels and the tile kernel are near the copy roofline. The cuFFT passes are the least efficient per
unique byte because a 4096-point strided column FFT and a 2-D C2R each make more than one pass over memory
(standalone cuFFT columns ran at 772 GB/s, 57% of copy, in 7840069). **Any further gain has to come from moving
fewer bytes (fusion), not from faster arithmetic.**

## Every applicable built-in, against our step

| built-in / option | status | evidence |
|---|---|---|
| `tile_fft` / `tile_ifft` on the cuFFTDx backend | **used** | `"mathdx": 1` in jobs 7839932 / 7840069; the scalar fallback measured 4.6× slower (1,058 vs 231.7 µs) |
| `block_dim` (must be ≤ N/2 for cuFFTDx) | **tuned** | 128 / 256 / 512 measured; 256 best (231.7 µs, 85% of copy) |
| N-D batched `tile_fft` (several rows per block) | **measured, slower** | 2 rows/block 630–701 µs vs 1 row 231.7 µs (2.7–3×); 4 rows fail with "too many resources" |
| `tile_map` fusing the pointwise product between FFTs | **used** | `rows` kernel: FFT → `zmul` → FFT in one pass (the probe's fused 3-FFT row kernel was 40% cheaper than separate passes) |
| `tile_load` / `tile_store` (register tiles, as `tile_fft` requires) | **used** | `rows` kernel |
| column FFTs as tiles (shared panels, `tile_transpose`, `tile_view`) | **measured, slower** | best 694 µs vs cuFFT 348 µs; kept in cuFFT |
| `enable_backward=False` | **used** | all step kernels |
| `fuse_fp=False` (needed for `lean32ek`'s compensated update) | **used**; can cost at most ~4% in `rows` | `tile_rows` runs at 1.11 TB/s, 96% of the default-flag row probe's 1.16 TB/s |
| `aligned=True` (vectorized 16-byte loads/stores) | **not used; ~no expected gain** | applies only to 2D+ *shared* tiles; `tile_fft` needs register tiles; column panels lose to 32-byte DRAM sectors on 8–16 B strided segments, which wider instructions do not fix |
| `tile_empty`, `tile_zeros`, `tile_axpy` | **not applicable** | no tile accumulators in the hot path; `update` is SIMT at 96% of copy |
| `div_approx`, `inverse_approx` | **not applicable** | no divisions or matrix inverses in the step (`kinv` is a table); would also change the arithmetic |
| `tile_matmul` (incl. TF32) | **excluded** | DFT-as-GEMM would change the arithmetic (TF32 more so) |
| reductions, scans, sort, stack, scatter, Cholesky/triangular solves, BVH/mesh/volume, RNG | **not applicable** | nothing in a pseudo-spectral step uses them |
| `tile_load_indexed`, reversed `tile_view` | **not used — the one opening** | see candidate 1 |

## Remaining candidates (none run; all need a probe first)

1. **In-tile C2R (built-ins: `tile_load_indexed` or reversed `tile_view`, both in 1.17.0).**
   - **Change:** do the inverse column pass as cuFFT C2C on the 1,366 kept columns only. Then build each row's
     Hermitian-extended spectrum inside the `rows` kernel and inverse-transform it there with one more
     `tile_ifft`.
   - **Removes:** `expand`, the 2-D C2R (the 33% op) and the `q` round trip to memory.
   - **Estimate:** −0.381 ms (`expand` + `cufft_c2r`) +0.138 ms (kept-column C2C, scaled from `cufft_cols_w`)
     +0.05–0.25 ms (the third in-register FFT, plus the shared-memory reversal) −0.016 ms (45 MB instead of
     67 MB read). Net about **−0.2 to +0.02 ms/step: up to ~13% faster, or no gain.** The uncertainty is the
     third FFT, because `tile_rows` is already at 82% of copy and the 2-row batching that would avoid the
     waste measured 3× slower.
   - **Cost:** it changes the rounding (a different FFT path for θ), so if it wins it needs the full gate
     battery (A, B, E, D), as `tilerows_if25` did.
2. **Skip rewriting W's zero rows (not a built-in; found while checking).**
   - **Change:** make the forward column FFT out-of-place from a persistent zero-padded buffer, so `build_w`
     writes the 2,731 kept rows instead of all 4,096.
   - **Estimate:** −29.8 MB written, about −0.03 ms/step (~2%) at `build_w`'s measured rate.
   - **Cost:** cuFFT may choose different kernels out-of-place, so check bitwise equality first.
3. **CUDA graph capture.** Already measured in 7840376: 1.562 vs 1.578 ms (1%).

## Do not re-try

- Several rows per tile block (`tile_fft` on (2, N) or (4, N) tiles): 3× slower or fails at launch (7840069).
- Tile column FFTs in any of the three layouts measured: 2× slower than cuFFT (7840069).
- cuFFT callbacks to fuse SIMT work into the column passes: doubled the FFT cost (hybrid job 7840175).
- Upgrading Warp for built-ins: nothing in 1.18 / 1.19's built-in list helps this step.
