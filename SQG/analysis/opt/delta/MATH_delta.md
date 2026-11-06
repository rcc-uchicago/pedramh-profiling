# The math behind the delta-fidelity table (job 7819310)

Written 2026-11-06 for the user's question "show the specific math behind" this table (notes Part XI):

| variant | e_Δ, overall | e_Δ, k = 1–2 | ms/step | vs production |
|---|---|---|---|---|
| `lean32` (all float32) | 5.5e-4 | 3.6e-3 | 3.20 | 2.9× |
| `lean32ek` (compensated float32 state) | 8.8e-6 | 2.2e-5 | 3.35 | 2.77× |
| `mixed` (float64 state) | 8.7e-6 | 2.2e-5 | 4.73 | 1.96× |
| `f64ic32`: float64 from a state rounded once to float32 (the floor) | 2.3e-7 | 1.2e-6 | — | — |

Every number below comes from `SQG/analysis/results/polaris_delta_7819310/delta_fidelity.{json,log}`,
`test_f32_update.log`, or `decay_quant.py`. The per-lag tables were re-read from the JSON on 2026-11-06.
Statements marked *(inference)* are explanations of the numbers, not measurements.

## 1. The step being computed

A-SQG in spectral space, one complex mode θ̂_k (4096², half plane, 2/3 dealiasing):

    dθ̂_k/dt = T_k(θ̂) − r_k θ̂_k,     T = −(i k_x (uθ)^ + i k_y (vθ)^) + F̂,     r_k = ν|k|² + μ/|k|²

with ν = 2e-4, μ = 1, Δt = 5e-6. pyqg_jax's AB3 (level 2) puts the damping in an integrating factor
d_k = exp(−r_k Δt) (and d_k = 0 beyond KMAX = N/3):

    θ̂ⁿ⁺¹ = (θ̂ⁿ + iⁿ) · d_k,     iⁿ = Δt (23/12 Tⁿ − 16/12 Tⁿ⁻¹ + 5/12 Tⁿ⁻²)

o1.2 samples every STRIDE = 40 steps and pairs frames LAG = 100 samples apart, so a training pair spans
L = 4,000 steps = 0.02 time units.

## 2. Float32, in one line

A float32 number has a 24-bit significand. Round-to-nearest gives fl(x) = x(1 + ε) with |ε| ≤ u = 2⁻²⁴ ≈ 5.96e-8.
The spacing between neighbours ("ulp") is between 2⁻²⁴|x| and 2⁻²³|x|. So any change smaller than about 6e-8 to
1.2e-7 of a value cannot be stored in that value.

## 3. What the table measures

For a variant v, run 4,000 steps from the same start state S_j. o1.2's frame is
F_v(ℓ) = float32(irfft2(θ̂)[::16, ::16]) / 8 (256²) after ℓ steps. The reference is float64 `lean_u1`, which is
bitwise equal to production on pair 0.

    Δ_v(ℓ) = F_v(ℓ) − F_ref(0),     e_Δ = ‖Δ_v(L) − Δ_ref(L)‖₂ / ‖Δ_ref(L)‖₂,     L = 4,000

- **Overall:** relative L2 over the 256² frame.
- **Per band ("k = 1–2"):** the same ratio restricted to the 256² rfft2 modes with 1 ≤ |k| < 2.
- **Aggregation:** the median over 8 start states, 0.02 apart on one float64 trajectory.

**Why a small Δ magnifies the error (the user's concern, made exact).** Both deltas subtract the same F_ref(0),
so Δ_v − Δ_ref = F_v(L) − F_ref(L). Hence

    e_Δ = e_frame · ‖F_ref(L)‖ / ‖Δ_ref(L)‖

The measured e_Δ / e_frame is the same for every variant at each lag. It is purely ‖F‖/‖Δ‖:

| lag ℓ (steps) | 40 | 400 | 1,000 | 2,000 | 4,000 |
|---|---|---|---|---|---|
| ‖F‖/‖Δ‖ (all 5 variants agree to ±2%) | 32.0 | 5.0 | 3.1 | 2.35 | 1.83 |
| rms(Δ) (o1.2 units) | 0.046 | 0.297 | 0.481 | 0.640 | 0.825 |

- At 4,000 steps this state's Δ is 55% of the frame, so frame errors are magnified 1.83×.
- In archive-like states, rms(Δ)/rms(θ) ≈ 0.26 (`decorrelation_full_summary.json`), so the factor would be ≈ 3.9.
- At k = 1–2, Δ is ≈ 4% of θ, so the factor in that band is ≈ 25.
- That is why the large scales are where a float32 error shows.

## 4. Mechanism A: float32 quantizes the damping rate (all of `lean32`)

d_k is just below 1, where float32 numbers are 2⁻²⁴ apart. Storing it adds an error δ with |δ| ≤ 2⁻²⁵ ≈ 2.98e-8.
The rate actually applied is r̃ = −ln(d + δ)/Δt, so

    (r̃ − r)/r ≈ −δ / (r Δt),     |(r̃ − r)/r| ≤ 2.98e-8 / (r Δt)

r_k is smallest at |k| = (μ/ν)^(1/4) = 8.41, where r = 2√(μν) = 0.0283 and rΔt = 1.41e-7. The bound there is
21%, and the measured error is −15.7%. At k = 1, rΔt = 5.0e-6: the bound is 0.6% and the measured error 0.12%.

| |k| band | 1–2 | 2–4 | 4–8 | 8–16 | 16–32 | 32–64 | ≥ 64 |
|---|---|---|---|---|---|---|---|
| max rate error, float32 `decay` | 0.12% | 5.2% | 20.0% | 20.0% | 9.0% | 2.8% | ≤ 0.7% |
| max rate error, float32 e = −expm1(−rΔt) | 3.6e-8 | 4.5e-8 | 4.7e-8 | 5.4e-8 | 5.9e-8 | 5.8e-8 | ≤ 6e-8 |

- **What `lean32e` changes.** It stores e = 1 − d instead. e has full float32 relative precision, so the rate is
  right to u, and the update is θ' = s − e·s.
- **Why that barely helps (§5).** e·s ≈ rΔt·|θ| is itself only 1–2 ulps of θ for the weakly damped modes, and it
  is rounded onto θ's grid. Measured: `lean32e` reaches 0.80× `lean32`'s e_Δ, a 20% gain.

## 5. Mechanism B: a float32 state rounds every increment, systematically (the dominant one)

**Per step.** θⁿ⁺¹ = fl(θⁿ + iⁿ) loses a rounding error ρⁿ with |ρⁿ| ≤ ½ ulp(θ) ≤ u|θ|. Relative to the increment,

    |ρ| / |i| ≲ u |θ| / |i|

The full-grid 1-step increment, as a fraction of θ, from the job:

| band | 1–2 | 2–4 | 4–8 | 8–16 | 16–32 | all |
|---|---|---|---|---|---|---|
| \|i\|/\|θ\| | 9.8e-6 | 1.6e-5 | 6.4e-5 | 1.7e-4 | 4.9e-4 | 8.4e-4 |
| u\|θ\|/\|i\| (estimate) | 6.1e-3 | 3.7e-3 | 9.3e-4 | 3.5e-4 | 1.2e-4 | 7e-5 |
| `lean32` measured, 1-step increment error | 5.3e-3 | 3.8e-3 | 9.3e-4 | 3.6e-4 | 1.1e-4 | 7.8e-5 |

- The one-line estimate reproduces the measurement band by band.
- At k = 1–2 the increment is only ≈ 80–160 ulps of θ, so each step stores it 0.3–0.6% wrong.
- The large scales are where the increment is the smallest fraction of θ (the slow modes): where Δ is smallest.

**Over L steps: why it does not average out.** Write iⁿ = (mₙ + fₙ)·ulp, with mₙ an integer and |fₙ| ≤ ½.
Round-to-nearest stores mₙ·ulp and drops fₙ·ulp. After L steps the error is E_L = −ulp·Σₙ fₙ.

- **If fₙ were independent noise:** |E_L| ~ ulp·√(L/12). Since Δ ≈ L·i, e_Δ ∝ L^(−1/2): it would **fall 10×**
  from L = 40 to 4,000.
- **If iⁿ varies slowly** (the large scales correlate 0.9993 over 0.02): fₙ keeps its value and sign for many
  steps, so |E_L| ≈ L·|f̄|·ulp. Then e_Δ ≈ |f̄|·ulp/|i|, **constant in L**: a bias.
- **The single forced-mode test** (`test_f32_update.py`, k = 8.4, increment ≈ 1–4 ulps): f̄/(m + f) is tens of
  percent. The 4,000-step Δ is **31% off** for `lean32`'s update and for the expm1 update alike.

The job's per-lag numbers show the bias. Coarse-frame e_Δ, median of 8:

| e_Δ at lag ℓ | 40 | 400 | 1,000 | 2,000 | 4,000 | ×(4,000/40) |
|---|---|---|---|---|---|---|
| `lean32`, k = 1–2 | 1.2e-3 | 1.8e-3 | 3.9e-3 | 4.8e-3 | 3.6e-3 | 3.1× (rises) |
| `lean32`, k = 2–4 | 2.0e-3 | 2.2e-3 | 2.1e-3 | 2.1e-3 | 2.0e-3 | 1.0× (flat) |
| `lean32ek`, k = 1–2 | 1.2e-4 | 4.9e-5 | 4.1e-5 | 3.4e-5 | 2.2e-5 | 0.19× (falls) |
| `f64ic32`, k = 1–2 | 2.1e-5 | 4.0e-6 | 3.2e-6 | 2.1e-6 | 1.2e-6 | 0.055× (falls) |
| `lean32` e_frame, all | 1.9e-6 | 1.6e-5 | 4.0e-5 | 9.8e-5 | 3.0e-4 | **160×** |
| `lean32ek` e_frame, all | 3.8e-7 | 1.1e-6 | 1.8e-6 | 2.9e-6 | 4.8e-6 | **13×** |

- **`lean32`:** its low-k delta error stays flat or rises with the lag, which is a bias.
- **`lean32ek` and the floor:** their errors fall as Δ grows.
- **`lean32` frame error:** it grows faster than linearly (160× over 100× steps): the bias, plus chaotic growth of
  it *(inference)*.
- **`lean32ek` frame error:** 13× ≈ √100 × 1.3, the random walk of independent per-step errors, slightly amplified
  *(inference)*.

## 6. `lean32ek`: the compensated (hi + lo) state

**The idea.** Carry θ as an unevaluated sum hi + lo of two complex64 arrays, with |lo| ≤ ½ ulp(hi). Then use two
error-free transformations, which are exact in IEEE round-to-nearest:

- **TwoSum(a, b)** (Knuth): s = fl(a + b), b' = s − a, r = (a − (s − b')) + (b − b'). Then s + r = a + b **exactly**.
- **Fast2Sum(a, b)** (Dekker), needs |a| ≥ |b|: s = fl(a + b), r = b − (s − a). Then s + r = a + b **exactly**.

**The step** (`f32_update.compensated_update`, componentwise on real and imaginary parts):

    y       = fl(inc + lo)                          error ≤ u|y| ≈ u|inc|                    (≈ 6e-13 |θ| at k=1–2)
    (a, r)  = TwoSum(hi, y)                         a + r = hi + y exactly
    ea      = fl(e·a)                               error ≤ u·e|a|                           (≈ 1e-14 |θ| at k=8.4)
    (u', r2)= Fast2Sum(a, −ea)                      u' + r2 = a − ea exactly   (|ea| ≤ |a| since 0 ≤ e ≤ 1)
    lo'     = fl(r2 + fl(r − fl(e·r)))              error ≤ u|lo| ≲ u²|θ|                    (≈ 3.6e-15 |θ|)
    hi'     = u'

    ⇒ hi' + lo' = (hi + lo + inc)(1 − e)  +  O(u|inc|) + O(u e|θ|) + O(u²|θ|)

- **What the rounding becomes.** It was u|θ| ≈ 6e-8|θ| per step. Now it is ≈ 1e-12|θ|. The state behaves like a
  ~48-bit number (u² ≈ 3.6e-15).
- **What is left.** Only the float32 tendency T (§7), as in `mixed`.
- **XLA keeps it.** XLA does not reassociate floating-point adds, so r is not simplified away. Measured: `lean32ek`
  equals `mixed` in every band at every lag, to 2 digits (1-step k = 1–2: 4.3e-6 vs 4.1e-6; e_Δ 8.84e-6 vs 8.67e-6).

**The cost: one more complex64 array read and written per step.**
- 4096 × 2049 × 8 B = 67.1 MB each way, 134 MB per step.
- Measured: 3.3514 − 3.2024 = 0.149 ms, i.e. ≈ 0.90 TB/s effective (A100 40 GB peak: 1.555 TB/s).
- The extra flops (TwoSum is 6, Fast2Sum is 3) are free in a memory-bound fusion.

## 7. `mixed` and `lean32ek`: what is left is the float32 tendency

`mixed` keeps θ̂, the AB3 history, the sum and the damping in float64, and computes only T with float32 FFTs and
products. Its per-step error is the float32 error of T, through the increment:

| 1-step increment rel. error | 1–2 | 2–4 | 4–64 | 128–256 | 512–1024 | 1024–2048 |
|---|---|---|---|---|---|---|
| `mixed` | 4.1e-6 | 2.2e-6 | 1.9–2.4e-6 | 5.8e-6 | 4.8e-5 | 1.4e-4 |
| `lean32ek` | 4.3e-6 | 2.2e-6 | 1.9–2.4e-6 | 5.8e-6 | 4.8e-5 | 1.4e-4 |

- **Where ≈ 2e-6 comes from.** A 4096² float32 FFT has rms relative error of a few × 1e-7 (u·√log₂N² ≈ 3e-7). The
  step chains 4–5 FFTs and two products, and the advection term is a small difference of larger products.
- **Why the error rises at high k.** FFT rounding is spread roughly evenly over wavenumbers. Where the tendency's
  own amplitude is small, the same absolute noise is a larger relative error *(standard FFT error model)*.
- **These errors decorrelate from step to step** because the field moves. So they random-walk:
  e_frame ∝ √L (13× over 100×, §5) and e_Δ ≈ 9e-6 at L = 4,000.

## 8. The floor `f64ic32`

Float64 `lean_u1` is started from fl₃₂(S_j), the state rounded once to complex64. Its initial error per component
is uniform in ±½ ulp, rms ≈ 2⁻²³ × 0.72 / √12 ≈ 2.5e-8 of θ. Measured e_frame:

    4.0e-8 (lag 40) → 4.3e-8 → 5.3e-8 → 7.6e-8 → 1.27e-7 (lag 4,000);     e_Δ = 1.27e-7 × 1.83 = 2.3e-7

- **Growth over the 0.02 time units: 3.1×.**
- **The global exponent would give only ≈ 1.2×.** λ ≈ 9–10 per time unit (Valadão et al. 2025, Figs 7–8), and
  e^(λ·0.02) ≈ 1.2.
- **The extra growth is plausibly transient** *(inference)*: a perturbation spread over all scales, growing fastest
  at the small scales.
- **Below this floor, nothing in float32 can go.** It is what one float32 description of the state already costs.
  The pre-set gate asked each variant to stay within 2× of it.

## 9. Gate arithmetic and the comparison with the UViT

**The gate** (pre-set; `PREDICTIONS_delta.md`):

    e_Δ ≤ max(2 × floor, 1e-6) = max(4.6e-7, 1e-6) = 1e-6

- **Scope.** It applies overall and in every 256² band with ≥ 1% of Δ's energy. Those bands are k = 4–8 (6.5%),
  8–16 (21%), 16–32 (30%), 32–64 (23%) and 64–182 (19%).
- **Not gated:** k = 1–2 (0.007%) and k = 2–4 (0.8%).
- **Ratio to the floor:** `mixed` 8.67e-6 / 2.32e-7 = 37×; `lean32ek` 38×; `lean32e` 1,900×; `lean32` 2,350×.
- **Verdict:** all fail, and the verdicts stand. Nothing with float32 tendencies reaches float64.

**The UViT comparison** (report only):
- **The model's error.** √(val MSE) = √0.0043 = 0.0656 in o1.2 units. Over rms(Δ) = 0.825 that is a relative
  error of 0.079 (with the archive's rms(Δ) = 0.27, 0.24).
- **Ratios:** `lean32ek` 8.8e-6 / 0.079 = 1.1e-4; `lean32` 5.5e-4 / 0.079 = 6.9e-3.
- **As noise in the loss, the solver error is invisible.** Added in quadrature, `lean32`'s
  (5.5e-4 × 0.825)² = 2.1e-7 changes an MSE of 0.0043 by 5e-5 relative.
- **The case against plain float32 is the bias, not the loss** *(inference)*.
  - `lean32`'s error is systematic and concentrated at k ≤ 4 (0.2–0.4% of Δ there).
  - A model trained on it learns that bias, and the loss cannot see it.
  - Autoregressive rollouts accumulate low-k bias step after step. That is what a long-rollout comparison would
    have to rule out.

## 10. Speed arithmetic

**Production** = o1.2's float64 call, 9.29 ms per solver step (round 1, job 7814300; job E: 9.30).

| variant | ms/step (this job, median of 3 × 30 calls) | 9.29 / ms |
|---|---|---|
| `lean32` | 3.2024 | 2.90× |
| `lean32ek` | 3.3514 | 2.77× |
| `mixed` | 4.7324 | 1.96× |

`mixed` pays for reading and writing θ̂ and the two tendencies in complex128 (134 MB each) instead of complex64:
4.73 − 3.35 = 1.38 ms.

## 11. Caveats on these numbers

- **The k = 1–2 band of a 256² frame holds 4 independent complex modes.** Its numbers are statistics of 8 real
  values. Treat factor-of-2 differences there as noise.
- **o1.2's coarse frame is θ[::16, ::16], a subsample, not a spectral filter.**
  - So 256² mode k receives every 4096² mode k + 256m (aliasing).
  - *(Inference)* This is why `lean32ek`'s coarse k = 1–2 e_Δ at lag 40 (1.2e-4) is 56× its full-grid 40-step
    k = 1–2 increment error (2.1e-6): fast-changing small scales fold into the coarse large-scale band.
  - Not tested.
- **The start state is not archive-like.** It is a synthetic −5/3 field spun up for 0.5 time units:
  rms(Δ) = 0.825 vs the archive's 0.27, with SPE still falling.
  - The definitive version reruns `delta_fidelity.py --state-in` on 2–3 archived checkpoints.
- **Paired errors only.** A bias in Δ's statistics (spectra, PDFs, low-k energy) needs ≥ 1 time unit per variant.
