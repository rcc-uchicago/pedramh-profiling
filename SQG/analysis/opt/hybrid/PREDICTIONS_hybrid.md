# The hybrid solver step: cuFFT with fused callbacks + one Warp kernel. Predictions (written 2026-11-12, before any run)

**Ask (user, 2026-11-12):** "start building it while we wait for the debug, we can always submit to capacity or
preemptible, or debug scaling". "It" is the hybrid proposed after the Warp/ecosystem survey:
- cuFFT does every transform, because cuFFT's column passes beat Warp's 2× (Warp probe, job 7840069);
- nvmath-python LTO callbacks fuse the pointwise work into those transforms;
- one Warp kernel does the per-mode update.

**The step (`hybrid_step.py`; callbacks in `hybrid_callbacks.py`), per solver step:**
1. **C2C forward, in place:** its load callback builds w = s(−k_x + i k_y) conj(ψ̂) from the compact θ̂ and kinv
   (seed 2's `build_w`). Result p = u − i v.
2. **C2R:** its load callback reads the compact θ̂, with a scratch input buffer, so there is no input copy.
   Result q_u = N² θ.
3. **C2C forward, in place:** its load callback forms Z = conj(p) · q_u / N²; its store callback writes only the kept
   modes into a compact square.
4. **Warp kernel** on the compact kept block (2731 × 1366): the Hermitian split, the tendency, then the update.
   - float32: `lean32ek`'s compensated update, compiled `fuse_fp=False`; optional IF-AB3 factors;
   - float64: `lean_u1`'s update.

**Software:**
- nvmath-python 1.0.0 + numba-cuda 0.30.4 (and cuda-bindings / cuda-core / cuda-pathfinder / numba / llvmlite),
  installed `--no-deps` in `$MEMBER_ROOT/pylibs/nvmath-1.0.0`; sqg-env is unchanged.
- cuFFT LTO callbacks need cuFFT ≥ 11.3. The job preloads CUDA 12.9.1's `libcufft.so.11` (11.4.1), as Part X did,
  so every arm in the process uses it.

**Checked before the run:**
- `test_hybrid_math.py` (numpy, N = 48) runs the callbacks' own source:
  - p = u − i v (4.7e-16);
  - the w callback equals `build_w` exactly;
  - the C2R callback equals θ̂ exactly;
  - the Hermitian split equals rfft2(uθ), rfft2(vθ) (3.7e-16);
  - the tendency matches to 1.3e-15.
- The job runs it again.

## Design (`hybrid_probe.py`, one A100; `run_hybrid_polaris.pbs`)

- **State:** member 1, t = 100, loaded with o1.2's recipe.
- **H1:** a plain nvmath C2C vs `torch.fft.fft2` (complex64 / complex128).
- **H2:** n = 1, 40, 4,000 steps of hybrid32, hybrid32_if, hybrid64 and the JAX steps `lean_u1`, `lean32ek_fast`,
  `if32`, `if64`. Measures: rel = |θ_a − θ_b| / |θ_b| over the full state, and at 4,000 steps the band errors of the
  0.02 delta (sub and af).
- **H3:** ms/step: eager, a 48-step CUDA graph, the per-op split, and the JAX steps on the same GPU.

## Gates (fixed now; the round-1b form for float32 rounding changes)

- **G32:** rel(hybrid32, `lean_u1`) ≤ min(1e-4, 2 · rel(`lean32ek_fast`, `lean_u1`)), at n = 1 and n = 40.
- **GIF:** the same for hybrid32_if, with `if64` / `if32` in place of `lean_u1` / `lean32ek_fast`.
- **G64:** rel(hybrid64, `lean_u1`) ≤ 1e-12 at n = 1. This is the float64 gate of round 1; 40 / 4,000 are reported.

**What a pass means:**
- **G32:** the hybrid is a float32 rounding change no worse than 2× today's float32 error. Using it for o1.2 still
  needs the integration (a JAX custom call around the step, or a torch-side solver) and the user's call on the
  statistical gates, as for `lean32ek_fast`.
- **G64:** the hybrid is the float64 step to rounding; no precision question.
- **Speed:** reported against the JAX steps measured in the same job. "Worth integrating" is hybrid32 ≤ 2.2 ms
  (≥ 1.23× `lean32ek_fast`) or hybrid64 ≤ 2.70 ms.

## Predictions (P = probability)

| # | prediction | P |
|---|---|---|
| 1 | the job maps cuFFT ≥ 11.3 (version ≥ 11300) and nvmath + numba-cuda import beside torch / jax / warp | 0.85 |
| 2 | all 8 callbacks compile to LTO-IR and all 6 plans build with them | 0.75 |
| 3 | G32 passes at n = 1 and 40 | 0.70 |
| 4 | GIF passes at n = 1 and 40 | 0.70 |
| 5 | G64 passes | 0.75 |
| 6 | at 4,000 steps, e_Δ(hybrid32 vs `lean_u1`) / e_Δ(`lean32ek_fast` vs `lean_u1`) ∈ [0.5, 2] in every band, sub and af | 0.65 |
| 7 | hybrid32 eager is 1.5–2.2 ms/step (`lean32ek_fast` ≈ 2.70 in the same job) | 0.55 |
| 8 | the 48-step CUDA graph captures and replays, and is within ±3% of eager | 0.55 |
| 9 | hybrid64 eager is 3.0–4.5 ms/step (`lean_u1` ≈ 6.40) | 0.50 |
| 10 | hybrid32_if ≤ 1.03 × hybrid32 (ms/step) | 0.75 |
| 11 | the 3 cuFFT calls are ≥ 70% of hybrid32's step (per-op split) | 0.60 |
| 12 | plain nvmath C2C is within ±5% of `torch.fft.fft2` (same cuFFT) | 0.80 |

## Tokens

- `HYBRID_MATH_TEST_OK`, `HYB_ENV_OK`, `HYB_REL n=…`, `HYB_GATE32 pass=…`, `HYB_GATEIF pass=…`, `HYB_GATE64 pass=…`,
  `HYB_BANDS`, `HYB_BENCH`, `HYB_PROBE_DONE failures=N`.
- The job ends `HYB_JOB_DONE failures=N`.
- Results → `SQG/analysis/results/polaris_hybrid_<jobid>/`.
