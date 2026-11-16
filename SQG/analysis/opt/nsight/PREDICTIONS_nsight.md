# Predictions for the `lean32` Nsight job (`run_nsight_lean_polaris.pbs`)

Written 2026-11-05 by the main session, before the job was submitted. Do not edit after submission; it is scored in
the notes (`polaris_sqg_notes.md`, Part X).

## Inputs

| input | value | source |
|---|---|---|
| `lean32` step | 3.189 ms/step | round 1b, 7815096 |
| `lean32` 2-step loop body | per step: `build_w` (`input_concatenate_fusion`), C2C, IRFFT (θ), products (`loop_multiply_fusion`, 2 outputs), 2 × RFFT, AB3 (`loop_add_multiply_fusion`); per 2 steps: 2 full-size tendency copies (`wrapped_copy.3/.4`) | `hlo_lean32.txt` |
| job E, float32 production step, per op per step | IRFFT 501 µs = input copy 99 + `regular_fft_factor<4096>` 197 + `vector_fft_symm_c2r` 101 + `scal` 104; RFFT 305 µs = `regular_fft_factor` 195 + `vector_fft_symm_r2c` 111 | job E's SQLite, re-read with the new `summarize_nsys_ops.py` |
| standalone c64 / f32 FFT costs | C2C 0.526, C2R 0.509, R2C 0.311 ms | seed 2 micro-table |

**Byte model of `lean32`'s current structure, per step** (MB read + written; c64 half-plane 67, full plane 134):

| op | MB |
|---|---|
| `build_w` | 235 |
| C2C | 537 (2 passes) |
| IRFFT | 536 (copy, 2 passes, `scal`) |
| products | 335 |
| 2 × RFFT | 537 |
| AB3 | 503 |
| tendency copy | 134 |
| **total** | **≈ 2.82 GB** |

At a practical 1.35 TB/s that is ≈ 2.1 ms; at the A100's 1.555 TB/s peak, ≈ 1.8 ms.

Job E's `regular_fft_factor<4096>` moves ≈ 134 MB in ≈ 196 µs (≈ 0.68 TB/s). Its neighbours move their bytes at
1.3 TB/s. **Hypothesis:** the half-plane is 2,049 complex64 wide. A row is 16,392 B, so rows are not 32 B-aligned, and
a block's 4 adjacent columns (32 B per row) straddle two sectors in 3 of 4 rows.

## Predictions

**Phase 1: checks and bench.**
1. `lean32_r2` is bitwise equal to seed 2's `lean32` at 40 and 480 steps (95%).
2. `lean32_u1`:
   - passes its check, and is bitwise at 40 and 480 steps (90%);
   - runs 2–8% faster than `lean32` (0.06–0.25 ms/step).
3. `lean32_ring3`:
   - passes its check; bitwise (80%: the fusion grouping may change);
   - its HLO has 0 full-size copies, vs `lean32`'s 2;
   - saves 80–130 µs/step (2.5–4%).
4. `lean32_u1_ring3`: the two savings add, ±50 µs.
5. Timings reproduce rounds 1b/1c within ±1.5%: `ref` 9.29, `lean_u1` 6.40, `lean32` 3.19 ms/step.

**Phase 2: nsys, `lean32` per XLA op, µs per step.**

6. The GPU is busy ≥ 97% of each range.

7. Per-op ranges:

   | op | µs/step |
   |---|---|
   | C2C | 480–600 |
   | IRFFT (same 4 parts as job E) | 480–530 |
   | 2 × RFFT | 580–640 |
   | products | 240–320 |
   | `build_w` | 230–380 |
   | AB3 | 380–600 |
   | tendency copies | 90–120 |
   | everything else | < 15 |

8. FFT ops (C2C + IRFFT + 2 RFFT, input copy and `scal` included) are the largest family, 50–58% of GPU time.

**Phase 4: ncu, every kernel of `lean32`'s loop.**

9. **The misalignment hypothesis holds:**
   - the half-plane `regular_fft_factor` passes (in IRFFT and RFFT) reach ≤ 55% of DRAM peak;
   - their global-load or -store sector efficiency is ≤ 60%.

   **Refuted if** both efficiencies are ≥ 90%. Then the cause is latency or occupancy, not alignment.
10. The C2C's column pass on the 4,096-wide plane has sector efficiency ≥ 90% and reaches ≥ 70% of DRAM peak.
11. The `vector_fft_symm_*` row passes, `scal` and the products fusion each reach ≥ 80% of DRAM peak.
12. AB3 fusion:
    - uses 4 elements per thread (grid ≈ 16,392 × 128);
    - reads ≥ 1.3× its 369 MB of unique input;
    - reaches ≥ 70% of DRAM peak.
13. `build_w` reads 1.3–2.2× its 100 MB of unique input (mirror re-reads of θ̂ and 1/|k|).
14. The loop period is found, and ncu's per-step kernel time is within 15% of nsys's.

**Phase 5: ncu, cos-trigger variants.**

15. `lean32_u1`'s AB3 fusion uses 1 element per thread (grid ≈ 65,5xx × 128), and its DRAM read is ≤ 1.15× unique.
16. `lean_u1`'s float64 AB3 fusion uses 1 element per thread, and its DRAM read is ≤ 1.15× its ≈ 740 MB unique (job
    F's production fusion: 3.99 GB).

**Phase 3: cuFFT 11.4.1 via `LD_PRELOAD`.**

17. The preloaded library is the one mapped (75%).
18. FFT micro-table: each c64 / f32 entry moves by ≤ ±15% vs the env's cuFFT 11.0.2. Direction unknown (50/50).
    `lean32` ms/step within ±8%.
19. Coarse frames vs the default build after 40 steps:
    - `ref` relative difference ≤ 1e-12, not bitwise (70%: a different FFT algorithm);
    - `lean32` relative difference ≤ 1e-5.

**Overall.**

20. **Headroom inside the current XLA structure is ≥ 0.8 ms/step** of `lean32`'s 3.19 ms. Of that, ≥ 0.4 ms is in
    the half-plane column FFT passes.
21. **The next-largest items are:** the AB3 over-fetch (if 12 holds) and the IRFFT's input copy + `scal`
    (≈ 0.2 ms, needs a custom FFT call).
