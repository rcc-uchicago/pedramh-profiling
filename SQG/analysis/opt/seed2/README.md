# Seed 2: the fastest A-SQG solver step that keeps the physics

`seed2_opt.py` builds three drop-in replacements for o1.2's 40-step solver call,
`jax.jit(lambda st: lax.scan(GD.inner, st, None, length=40)[0])`. Each takes and returns the production `AB3State`
(the same pytree, complex128 fields). Each is gated against the float64 production step on one A100, then timed.
`A-SQG_data.py` is imported unmodified, through `profile_solver.import_asqg`. Nothing else in the repo is touched.

```
"$PY" SQG/analysis/opt/seed2/seed2_opt.py --out "$OUT/seed2"     # repo root, after the SQG Polaris env bootstrap
```
Tokens: `SEED2_GATE_OK <v>` or `ERROR SEED2_GATE_FAIL <v> <reason>`; `SEED2_BENCH <v> ms_ref= ms_cand= speedup=`;
`SEED2_OPTSTEP <fastest passing>`; `SEED2_DONE` (last line). Output: `seed2.json`, plus `hlo_{ref,<v>}.txt`.
No install and no build is needed (sqg-env as is). Expected wall time is about 7 min; the worst case, with no variant
faster than the reference, is about 12.5 min.

`test_build_w_stdlib.py` (stdlib only, so it can run on a login node) mirrors `build_w`'s lax ops on an 8×8 grid. It
checks `fft2(W) = u − i v` against the C2R path (u = irfft2(−i k_y ψ̂), v = irfft2(i k_x ψ̂)): max relative error 1.1e-15
(polaris-login, 2026-11-04). A flipped mirror sign gives 0.98 and a missing row flip gives 0.91, so a broken mapping
fails it. It prints `SEED2_BUILDW_TEST_OK`.

## Technique

Where the float64 step's 9.30 ms goes (jobs A/E, nsys): cuFFT 3.06 ms; AB3 fusion 3.29 ms (≈390 GB/s);
inverse-FFT input copies 0.59 ms; history-rotation copies 0.39 ms; û/v̂ build 0.64 ms; u·θ, v·θ 0.62 ms;
inverse-FFT 1/N² `scal` 0.59 ms. Everything except the FFTs is data movement that the physics does not need.

| change | what it removes (f64 numbers) | exact? |
|---|---|---|
| **u and v from one forward C2C FFT.** Build W = conj(û + i v̂)/N² on the full 4096² plane: the right half is the Hermitian mirror, row (−r) mod N, column N − j. Then `fft2(W)` = u − i v. | the 2 C2R input copies (≈0.39 ms) and the 2 `scal` (≈0.39 ms). A forward C2C has neither. It costs about as much as the two C2Rs together, because it moves the same bytes. θ keeps its C2R. | rounding only (a different FFT algorithm) |
| **AB3 history as a 2-slot ring.** The scan is unrolled by 2, and each step writes the new tendency over the older slot. | the 2 rotation copies per step (≈0.39 ms) | exact |
| **Lean AB3 fusion.** pyqg's filter is dropped (`filterfac=0`: exp(−0·x) = 1). So is the background-gradient term: the ref HLO shows `ikQy = (iota·0)·i`, i.e. 0, because β = U = 0. This verifies the "unverified" k_x term in the notes' Part IV addendum. The forcing is added from two 8×8 corner patches through `pad`, not read as a dense 134 MB array. `decay` is read as f64, not as a constant-folded c128 array. 1/N² is folded into W. | f64 `sqrt`/`exp` on 8.4 M modes; ≈0.27 GB/step of reads; the per-step `wv2i` rebuild (0.07 ms) | exact (×1, +0; up to the sign of zero) |
| **Tables are jit arguments.** | XLA constant-folds `pad`/`complex` of constant operands up to 45 M elements into dense arrays, which would undo the item above | — |

The three variants (at most 3):

| variant | precision | gate (rel. L2, 1 step) | expected ms/step (A100) |
|---|---|---|---|
| `lean64` | float64 everywhere | ≤ 1e-12 | ≈5.0–5.8 (≈1.6–1.85×) |
| `mixed` | f32 FFTs and products. θ̂, the AB3 history, F̂, the damping and the AB3 update stay f64 (gyaradax's split) | ≤ 1e-5 | ≈2.8–3.5 (≈2.7–3.3×) |
| `lean32` | float32 inside the call; the state is cast at entry and exit | ≤ 1e-5 | ≈2.4–3.0 (≈3.1–3.9×); job E's plain f32 step is 3.61 |

These are estimates from per-kernel bytes at the bandwidths measured in jobs A/E. They are not measurements. Of the
three, `mixed` changes the physics least among those that use float32: the AB3 increment (≈1.6e-4 of θ per step) is
still accumulated in float64. `lean32` rounds every increment to float32.

Not used:
- **cuFFT LTO callbacks.** They need cuFFT from CUDA ≥ 12.6u2, plus a C++ XLA custom call. The env's jaxlib 0.4.28
  loads the wheel's `libcufft.so.11` (11.0.2.54), and a second `libcufft.so.11` in the same process would collide.
  Not attempted.
- **Pallas/Triton.** `jax/_src/pallas/triton/lowering.py` (0.4.28) has float64 math externs but no complex types, so
  a complex128 state would need real/imag splits, which are extra passes. XLA's own fusion should already be
  bandwidth-bound once the f64 `exp`/`sqrt` and the dense constants are gone.
- **Donation.** Job A measured it as neutral, and it would invalidate the caller's state.

## The gate (fixed before any run; never loosened)

Two start states, both at AB3 level 2:
- **`random`:** the solver's IC (`GD.advance(sqg.init(...), 2)`, as `SQG_INIT=random`). θ is about 1e-3 there, so
  the tendency is almost all forcing. This state checks the forcing, AB3 and damping terms.
- **`synthetic`:** a broadband field built to test advection. Gaussian white noise (jax seed 20261104) is
  transformed with rfft2 and shaped by an amplitude A(k):
  - A(k) ∝ k^(−4/3)·exp(−(k − KF)/300) for k ≥ KF. This gives E(k) ∝ k^(−5/3), Blumen's range, with a dissipation
    fall-off placed from the documented kmax·l_ν = 1.76.
  - A(k) ∝ k for k < KF.
  - A(k) = 0 for k = 0 and for k > N/3.

  The field is then scaled so that umax·KMAX·DT = 0.30 (production CFL ≤ 0.42), and stepped twice by `GD.inner`.

A variant passes only if **all** of these hold:
1. After **1 step** from each start state, the max over (θ̂, T₋₁, T₋₂) of ‖x − x_ref‖₂/‖x_ref‖₂ against
   `GD.inner` is **≤ 1e-12** (`lean64`) or **≤ 1e-5** (`mixed`, `lean32`). The max over fields is stricter than
   one norm over the concatenated state.
2. t, tc and `_ablevel` equal the reference's, after 1, 40, 480 and 4,000 steps.
3. Every value is finite after 1, 40, 480 and 4,000 steps, at every point where the reference is finite.
4. The compiled call's FFTs run in the claimed precision, and `lean32`'s 2-step loop body holds no full-size f64/c128
   array. JAX runs with x64 on, so a stray constant promotes silently.

**Why 1e-12 for float64-only.** The reference is bitwise reproducible. Reordering, FMA contraction and a different
FFT algorithm in float64 give ≈ε·√log₂N ≈ 1e-16–1e-15 relative error per FFT, and below 1e-14 on a 1-step state.
1e-12 leaves ≥100× margin, yet sits about 5 decades below any float32 effect (ε₃₂ ≈ 6e-8). So a variant that is
accidentally float32 fails it.

**Why 1e-5 when float32 is involved.** A 4096² float32 FFT has an rms relative error of ≈2–5e-7, and five FFTs plus
the products compound to ≲1e-6 on the tendency. 1e-5 leaves about 10× margin. A physics bug is far larger:
- a sign, a missing term or a wrong wavenumber gives O(1e-2…1) on T;
- an AB3 coefficient error moves θ̂ by ≳ dt·|T|/|θ̂|, which is ≥ 7e-5 even for the T₋₂ coefficient.

The script also checks its own metric: a 1e-9 perturbation must read back as 1e-9.

**Information only, ungated:**
- bitwise equality after 480 steps;
- the relative L2 difference at 40, 480 and 4,000 steps;
- after 4,000 steps (one 0.02 interval): the relative L2 difference of the 256² `coarse()` field, and the
  0.5⟨θ²⟩ (`GD.diagnostics` SPE) difference and drift against the reference.

The flow is chaotic, so pointwise divergence over 4,000 steps is expected, and statistics decide.

**Timing.** `profile_solver.py`'s method: the median of 3 × 60 jitted 40-step calls after 3 warm-up calls, from the
random IC. Then o1.2's optimizer-step pattern (12 calls + `coarse()`) for each passing variant. A standalone
FFT-cost table (C2C vs C2R vs R2C, f64 and f32) shows directly whether the C2C path avoids the copy and the scal.

## Unverified / risks

- **XLA behaviour the speed-up depends on**, measured only by the job:
  - the forward C2C FFT thunk issues no input copy (the FFT table, `fft_ms_per_call_4096`);
  - the ring removes the rotation copies (`hlo.full_size_copies`);
  - `build_w` fuses into one kernel.

  If any of these fails, the variant is still correct, only slower.
- **Validity conditions.** Valid only at AB3 level 2, with filterfac = β = U = rek = 0 and the forcing inside the 8×8
  corners. The script checks all of these at start-up (`ERROR SEED2_CONFIG_CHANGED` /
  `ERROR SEED2_FORCING_OUTSIDE_PATCH`). The C2C path assumes θ̂ is zero on the Nyquist row and column, which the 2/3
  damping guarantees after every step; `ics.*.nyquist_lines_max_abs` records it.
- **o1.2 asserts its first segment is bitwise equal to `train_m16.zarr`** (`assert check.item() == 0.0`). Any
  non-bitwise variant fails that assert, so adopting one means replacing the assert with a tolerance or statistical
  check.
- **Science sign-off.** A float32 solver (`mixed`, `lean32`) changes what is computed, so it needs Weidong's/jesswan's
  sign-off (DESIGN §4). `lean64` changes rounding only.
- **What one debug job cannot check:** long-run statistics. That needs a Midway twin run: ≥ 1 time unit from archived
  checkpoints, the variant against the archive, comparing the SPE/VIE/ε_ν/ε_μ/P/E(k) time series, the CFL and the
  coarse-field PDFs/spectra statistically.
- **The synthetic field** is not a dynamically evolved state. Its amplitude and spectrum are modelled, not measured.
- **The 1-step gate runs a 1-step jit, not the timed 40-step scan program.** The two compile to different fusions.
  Their agreement is reported at 40/480/4,000 steps.
- **Job F (Nsight Compute on the f64 AB3 fusion)** is still pending. The lean fusion drops the `exp`/`sqrt` that are
  the likely cause, but this is not confirmed. *(Update, main session: job F found the fusion DRAM-bound with ~4×
  over-fetch from uncoalesced 4-per-thread complex128 access, FP64 pipe 6% — the exp/sqrt are not the cost.)*

## Report-only additions (main session, before the first run)

Neither gates anything, and each is wrapped so a failure prints a `WARN` and leaves the gates and the bench untouched:
- **Per-band 1-step error** (`band_rel_1step` per variant and start state; `SEED2_BANDS` line for the synthetic
  state): the relative error of θ̂ and the newest tendency per |k| octave [2^b, 2^(b+1)). A whole-field relative L2 is
  dominated by the energetic large scales, so this shows whether low-energy high-k modes carry larger errors.
- **The original step in float32** (`f32_original_baseline`; `SEED2_F32BASE` line): job E's
  `profile_solver_fp32.make_inner` through the same 1-step, per-band and 4,000-step comparisons.
  `rel1_over_f32_original` per variant divides its 1-step error by this baseline's. For `mixed` and `lean32`, ≈ 1 or
  below means the restructured code adds nothing beyond plain float32 rounding.
