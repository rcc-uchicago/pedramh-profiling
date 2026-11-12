# Warp SIMT kernels inside the JAX `lean32ek_fast` step: predictions (written 2026-11-12, before any run)

**Ask (user, 2026-11-12):** "yes try it" — call a Warp update kernel from the JAX step through our bridge, since the
hybrid's Warp update took 0.21 ms (job 7840175). Mid-build the user added: "are [we] using warp correctly, the
algorithm relies on irregular memory access, and the use of wp.tid() allows us to examine the thread-execution".

**Design** (`warpupd.py`; bridge `../warp/warp_xla.py`, now taking JAX complex arrays as `vec2f` / `vec2d`).
- Warp is used as SIMT code: one thread per mode or grid point via `wp.tid()`, with the step's irregular accesses
  written out (the Hermitian mirror, the 2/3 mask, the compact gather).
- Every FFT stays XLA's cuFFT call. No callbacks: they doubled the FFTs in job 7840175.
- No `LD_PRELOAD`: production's cuFFT wheel.
- Arms, each a 40-step call with `lean32ek`'s arithmetic (`_if` adds the IF-AB3 factors):

| arm | what Warp does | what it isolates |
|---|---|---|
| `xla_compact`(`_if`) | nothing: pure XLA on the compact kept block | the layout gain alone (research #3); the control |
| `warp_full` | the tendency + compensated update, full half-plane | Warp's code vs XLA's fusion, same bytes |
| `warp_compact`(`_if`) | the update on the compact block, gathering the kept rows of XLA's rfft2 outputs | the user's idea |
| `warp_simt`(`_if`) | build w (mirrored gather from the compact state), Z = conj(p)θ, Hermitian split + update; XLA does fft2(w), irfft2(θ), fft2(Z) | Warp for every irregular-access step; one fft2(Z) in place of two rfft2s (research #4) |

**Gate** (the hybrid's G32 / GIF):
- rel(arm, `lean_u1`) ≤ min(1e-4, 2 · rel(`lean32ek_fast`, `lean_u1`)) at n = 1 and 40;
- `_if` arms: if64 / if32 in place of `lean_u1` / `lean32ek_fast`;
- bitwise vs `lean32ek_fast` / if32 is reported at n = 1, 40, 400.

**Timing:** ms/step, 3 × 30 calls of 40 steps, every arm, same GPU.

**Thread-level check (Nsight Compute):** one 40-step call each of `lean32ek_fast`, `warp_compact`, `warp_simt`
(first 40 launches). Per kernel: time, DRAM bytes, global load / store sectors per request, bytes used per sector,
registers, occupancy.

## Decision rule (fixed now)

- An arm is **admissible** iff its gate passes.
- **Warp is worth it** iff the fastest admissible Warp arm is ≥ 3% faster than both `lean32ek_fast` and
  `xla_compact`.
- If `xla_compact` is within 3% of the best Warp arm, the gain is the layout: recommend pure XLA (no Warp dependency).
- Report: s/optimizer step = 12 · 25 · t_if + 0.40 s for the best admissible `_if` arm (the IF-AB3 n = 25 path), vs
  `lean32ek_fast`'s 12 · 40 · t + 0.40.

## Predictions (P = probability)

| # | prediction | P |
|---|---|---|
| 1 | all 7 Warp-side arms build and run finite | 0.75 |
| 2 | the gate passes for all 7 | 0.75 |
| 3 | `xla_compact` is bitwise equal to `lean32ek_fast` at n = 1, 40, 400 | 0.50 |
| 4 | `warp_full` is bitwise equal to `lean32ek_fast` at n = 1, 40, 400 | 0.35 |
| 5 | `warp_full` is within ±3% of `lean32ek_fast` (ms/step) | 0.55 |
| 6 | `xla_compact` is 0.15–0.35 ms/step faster than `lean32ek_fast` | 0.55 |
| 7 | `warp_compact` is within ±0.05 ms/step of `xla_compact` | 0.50 |
| 8 | `warp_simt` is the fastest arm, at 2.2–2.5 ms/step | 0.40 |
| 9 | every `_if` arm costs ≤ +2% over its base arm | 0.70 |
| 10 | Nsight: the Warp update kernels' global loads are ≤ 8.5 sectors per request with ≥ 85% of each sector used | 0.60 |
| 11 | Nsight: Warp `build_w` (half its threads read the mirror in reverse order) averages ≤ 10 load sectors per request | 0.55 |

## Tokens

- `WU_ENV_OK`, `WU_REL n=…`, `WU_GATE <arm> pass=…`, `WU_BENCH`, `WU_PROBE_DONE failures=N`; per Nsight run
  `WU_NCU_DONE`, then `WU_NCU_SUMMARY_OK`.
- The job ends `WU_JOB_DONE failures=N`.
- Results → `SQG/analysis/results/polaris_warpupd_<jobid>/`.
