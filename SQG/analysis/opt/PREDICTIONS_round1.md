# Round 1 predictions (pre-registered 2026-11-04, before any GPU run)

Written by the principal-scientist review of the two seeds' code, before `run_round1_polaris.pbs` was submitted.
Nothing below may be edited after the job reports; score it in CHANGELOG / `polaris_sqg_notes.md` against
`SQG/analysis/results/polaris_opt_round1_<jobid>/ranking_round1.txt`, `seed1.json` and `seed2.json`.

Reference numbers (jobs A/E/F, A100-SXM4-40GB, float64 production step 9.28–9.30 ms): cuFFT 3.06 ms, AB3 fusion
3.29 ms (DRAM-bound: 3.99 GB read for ~1.0 GB unique, grid 16392 × 128 = 4 c128 elements/thread, L1 hit 10 %, L2 hit
44 %, 88 % of DRAM peak, 58 regs, no spills, FP64 pipe 5.6 %), C2R input copies 3 × 0.195 ms, history-rotation copies
2 × 0.195 ms, û/v̂ build 0.64 ms (1.05 TB/s, 1.4× over-fetch), u·θ, v·θ 0.62 ms (ideal bytes), 1/N² `scal` 0.59 ms,
`loop_select_fusion` 0.07 ms.

## Gates

| variant | seed | precision | pre-set gate | predicted gate | predicted 1-step max rel L2 (synthetic / random) | predicted bitwise at 480 steps |
|---|---|---|---|---|---|---|
| `lean` | 1 | f64 | G1 ≤ 1e-12 per field; G2 ≤ 1e-9 at 40 steps; G3 finite at 480 | **PASS** | 0 to 1e-15 / 0 to 1e-16 | values equal: likely (≥ 60 %); bits equal: ≤ 50 % (signed zeros from the dropped `+0` terms and `F·1`) |
| `lean_u1` | 1 | f64 | same | **PASS** | same as `lean` (the cos(±0) factors are exactly 1) | same as `lean` |
| `lean_split` | 1 | f64 | same | **PASS** | same as `lean` | same as `lean` |
| `lean64` | 2 | f64 | ≤ 1e-12 max over fields; t/tc/level equal; finite at 1/40/480/4000; FFTs c128/f64 | **PASS** | 1e-16 to 3e-15 / 1e-18 to 1e-16 (a different FFT algorithm: C2C vs C2R) | **no** (certain) |
| `mixed` | 2 | f32 nonlinear term, f64 state | ≤ 1e-5; FFTs c64/f32 | **PASS** | 3e-7 to 2e-6 / 1e-9 to 1e-7 | no |
| `lean32` | 2 | f32 | ≤ 1e-5; FFTs c64/f32; 0 full-size f64/c128 in the 2-step body | **PASS** | 3e-7 to 2e-6 / 5e-8 to 1e-7 (θ̂ itself is rounded to f32: ~6e-8) | no |

Why the f64 variants should be at or near value-identity: every kept operation has the same operand order and the same
complex arithmetic as `GD.inner`; the dropped terms are an exact ×(1+0i) (filter, `exp(-0·x)`), an exact +(±0±0i)
(the β term, `(iota·0)·i` in the ref HLO), and a bit-identical forcing array. The only mechanisms left are the sign of
zeros and LLVM FMA-contraction choices inside a differently shaped fusion; both give ≤ 1e-15 relative. `lean64`
replaces the two C2R transforms of û, v̂ by one forward C2C of the Hermitian-mirrored plane, so its u, v differ in the
last bits (≈ 1e-16 rms), and the 1-step error is carried by T₋₁ (≈ 1e-16–1e-15), not θ̂ (≈ 1e-19, because one step
changes θ̂ by ~1.6e-4 of itself).

A FAIL on any gate would mean a bug in the ring/pack logic or a tracing error, not rounding: the f64 thresholds leave
≥ 3 decades of margin, the f32 ones ≥ 1 decade. The likeliest failure mode is an exception at first execution (never run
on a GPU), reported as `ERROR SEED*_GATE_FAIL <v> exception_*`; I give that ≈ 25 % per seed for at least one variant.

## Speed (ms per solver step, median of 3 × 60 jitted 40-step calls from the random IC; ref ≈ 9.30)

| variant | predicted ms/step (central, range) | speedup | mechanism |
|---|---|---|---|
| `lean` | **7.3** (6.6–7.9) | 1.27× | −0.39 rotation copies, −0.07 select, −0.1 û/v̂ build; AB3 fusion 3.29 → 1.5 (1.0–2.4): fewer operands (5 c128 + 1 f64) but the same 4-elements-per-thread pattern, so the L2 thrash is reduced, not removed; if the register count drops enough to double occupancy it can thrash again |
| `lean_u1` | **6.15** (5.9–6.6) | 1.51× | as `lean`, plus the cos(±0) factor switches XLA's loop emitter to 1 element/thread: AB3 fusion ≈ 0.85 ms (1.07 GB at ≈1.3 TB/s + f64 cos), û/v̂ build ≈ 0.40. If the cos does not change the unroll (`MayPreventVectorization` not containing kCos in this jaxlib), `lean_u1` ≈ `lean` |
| `lean_split` | **6.7** (6.3–7.1) | 1.39× | T fusion (3 operands) ≈ 0.4 + update fusion (5 operands) ≈ 0.7–1.0, both still unrolled ×4 |
| `lean64` | **5.8** (5.3–6.8) | 1.60× | −0.39 ring, −0.07 select, −0.78 (two C2R input copies + two `scal`), −0.4 (one c128 C2C ≈ 0.85 ms vs two C2R ≈ 1.22), −0.2 build_w vs û/v̂ build, +0.1 products (read the c128 u−iv), AB3 fusion as `lean` (1.0–2.4) |
| `mixed` | **3.3** (2.8–3.9) | 2.8× | f32 FFTs (C2C c64 ≈ 0.42, C2R + copy + scal ≈ 0.5, 2 R2C ≈ 0.6), f32 products 0.26, c128→c64 casts 0.45, AB3 fusion with 3 c128 + 2 c64 + f64 operands 0.65–1.6 |
| `lean32` | **2.5** (2.3–2.9) | 3.7× | everything c64/f32; AB3 fusion ≈ 0.4–0.65 (job E's f32 fusion with 8 operands took 0.65); the call-boundary casts cost ≈ 0.01 ms/step |

Optimizer step (12 × 40 steps + `coarse`): ref ≈ 4.47 s; `lean_u1` ≈ 2.97; `lean64` ≈ 2.8; `mixed` ≈ 1.6; `lean32` ≈ 1.2.

## Winner

- **Overall fastest passing variant: `lean32`** (seed 2), ≈ 2.5 ms, ≈ 3.7×. It passes its 1e-5 gate; whether a
  float32 state is acceptable is a science decision (DESIGN §4, Weidong / jesswan), which the ranking records as
  the `precision` column, not as a penalty.
- **Fastest float64-only variant: `lean64` over `lean_u1`, 55/45.** Central estimates 5.8 vs 6.15 ms, well inside
  each other's ranges. The decisive observable is the AB3 fusion in `lean64`'s nsys-free timing: if `lean64` lands
  above ≈ 6.3 ms its fusion still thrashes (≥ 1.8 ms) and `lean_u1` wins; if `lean_u1` lands above ≈ 7.0 ms the cos
  trigger did not fire. Either way the round-2 candidate is the union: C2C packing + ring + the unroll-1 trigger
  (or an equivalent `few_waves`-friendly fusion shape) ≈ 4.8–5.2 ms, i.e. the JAX-level floor of ≈ 4.9 ms that seed 1
  derived minus the C2R copies and `scal` that seed 2 removes.
- Ordering of the f64 class by speed: `lean64` < `lean_u1` < `lean_split` < `lean` (confidence ≈ 50 % for the full
  order; ≈ 85 % that both `lean_u1` and `lean64` beat `lean`).

## Information-only quantities (ungated)

- 4,000-step coarse-field relative L2 vs the reference (synthetic start, CFL 0.3): f64 variants 0 (if value-identical)
  or 1e-15–1e-13 (chaotic growth over 0.02 time units is at most a few e-folds); `mixed` 1e-6–1e-4; `lean32` 1e-5–1e-3.
- 4,000-step energy (0.5⟨θ²⟩) relative difference: f64 ≤ 1e-14; `mixed` 1e-9–1e-6; `lean32` 1e-7–1e-5. The reference's
  own energy change over 4,000 steps from the synthetic state is O(1e-2) (forcing vs dissipation out of balance for a
  non-attractor field) and is reported alongside, so the differences above must be read relative to it.
- Full-field copies per solver step inside the loop: ref 5 → seed-1 variants 3 (the C2R input copies stay) → seed-2
  variants 1 (θ̂'s C2R copy stays). Seed 2's FFT micro-table: `fft2_c2c_c128` ≈ 0.8–0.9 ms per call,
  `irfft2_c2r_c128` ≈ 0.95–1.05 (transform + input copy + scal), `rfft2_r2c_f64` ≈ 0.6; the c64 cases ≈ half.
- Both seeds' own `ref` timings agree with each other and with jobs A/E to ≤ 0.5 % (same GPU, sequential runs); a
  larger gap means the GPU's state changed between the seeds and cross-seed ms/step must be read with that error bar.
- Memory: seed 1 peak ≈ 5 GiB, seed 2 peak ≈ 7 GiB (it keeps three reference snapshots per start state).
- Wall: seed 1 ≈ 10 min (budget 14, timeout 22), seed 2 ≈ 7–8 min (worst 12.5, timeout 20); job ≈ 20 min, worst ≈ 45.

## What this job cannot decide

Long-run statistics. The flow is chaotic; pointwise divergence after 4,000 steps is expected even for `lean64`. Any
variant adopted for o1.2 needs the Midway twin run both seeds describe: ≥ 1–5 time units from an archived checkpoint,
production vs variant, comparing SPE/VIE/ε_ν/ε_μ/P time series, time-mean E(k), the CFL distribution and the coarse-field
PDFs, with the divergence rate checked against the 1-ulp twin of `SQG/analysis/decorrelation.py`. For `mixed`/`lean32`
that run is also the science case for a float32 solver, and o1.2's `assert check.item() == 0.0` (first segment bitwise
equal to `train_m16.zarr`) has to become a tolerance for any non-bitwise variant.
