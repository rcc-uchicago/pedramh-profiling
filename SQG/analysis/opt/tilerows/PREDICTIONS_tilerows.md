# Warp tile FFTs for the rows, cuFFT for the columns: predictions (written 2026-11-12, before any run)

**Ask (user, 2026-11-12):** Warp's "Tile-Based FFTs via cuFFTDx … blocks of GPU threads cooperate on shared-memory
tiles and execute fast, vendor-tuned FFTs without leaving the [kernel]". Also "the algorithm relies on irregular
memory access, and the use of wp.tid() allows us to examine the thread-execution".

**What the earlier jobs say** (Warp probe 7840069, hybrid 7840175):
- Tile row FFTs ≈ cuFFT (231 vs 206 µs).
- A fused 3-FFT row kernel is 40% cheaper than separate passes.
- Tile column FFTs are 2× cuFFT's (694 vs 348 µs).
- cuFFT callbacks double the FFTs.

So this step puts each part where it measured well:
- columns: plain cuFFT strided plans, kept columns only;
- rows: one Warp tile kernel fusing FFT → product → FFT;
- irregular-access steps (build w with the mirror, expand, Hermitian split + update): one-thread-per-element Warp
  kernels.

`tilerows_step.py` has the per-step sequence. The decomposition is checked by `test_tilerows_math.py` (N = 48:
p = u − iv to 4.5e-16, the split to 3.2e-16; passed on the login node).

**Run** (`tilerows_probe.py`; `run_tilerows_polaris.pbs`, one A100):
- **State:** member 1, t = 100.
- **Arms:** tilerows32 and tilerows32_if (IF-AB3), next to `lean32ek_fast`, if32, `lean_u1`, if64.
- **Library:** cuFFT 11.4.1 is preloaded, as in the hybrid job, for every arm.
- **Measurements:**
  - rel at n = 1, 40, 4,000, and band errors at 4,000;
  - ms/step eager, as a 48-step CUDA graph, and the per-op split.

## Gates and rule (fixed now)

- **G32:** rel(tilerows32, `lean_u1`) ≤ min(1e-4, 2 · rel(`lean32ek_fast`, `lean_u1`)) at n = 1 and 40.
- **GIF:** the same for tilerows32_if with if64 / if32.
- **Worth integrating** (behind a JAX custom call) iff G32 passes and tilerows32 ≤ 0.97 × `lean32ek_fast` measured in
  the same job.

## Predictions (P = probability)

| # | prediction | P |
|---|---|---|
| 1 | both arms build and run finite; the strided cuFFT plans and the tile kernel compile and launch | 0.75 |
| 2 | G32 passes | 0.70 |
| 3 | GIF passes | 0.70 |
| 4 | at 4,000 steps, tilerows32's band errors are 0.5–2× `lean32ek_fast`'s (every band, sub and af) | 0.65 |
| 5 | tilerows32 eager is 1.4–2.2 ms/step | 0.45 |
| 6 | tilerows32 is faster than `lean32ek_fast` in the same job ("worth integrating" holds) | 0.55 |
| 7 | the tile row kernel (2 FFTs + product per row) takes 0.30–0.55 ms | 0.55 |
| 8 | each kept-column cuFFT pass (2 plans) takes 0.18–0.30 ms | 0.55 |
| 9 | the C2R θ takes 0.25–0.40 ms | 0.60 |
| 10 | the CUDA graph is within ±3% of eager | 0.60 |
| 11 | tilerows32_if ≤ 1.03 × tilerows32 | 0.75 |

## Tokens

- `TILEROWS_MATH_TEST_OK`, `TR_ENV_OK`, `TR_REL n=…`, `TR_GATE32` / `TR_GATEIF pass=…`, `TR_BANDS`, `TR_BENCH`,
  `TR_PROBE_DONE failures=N`.
- The job ends `TR_JOB_DONE failures=N`.
- Results → `SQG/analysis/results/polaris_tilerows_<jobid>/`.
