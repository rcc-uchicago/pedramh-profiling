# Predictions for the delta-fidelity job (`run_delta_polaris.pbs`)

Written 2026-11-05 by the main session, before the job was submitted. Do not edit after submission. Scored in
the notes (`polaris_sqg_notes.md`, Part XI).

## Why this job

The user (2026-11-05): "we need to be sure that fp32 is not impacting δ as it is very small". o1.2 learns the
0.02 delta θ(t + 0.02) − θ(t) on 256² frames.

- **How small the delta is.** Overall rms(Δ)/rms(θ) = 2.13 / 8.32 = 0.26, from `decorrelation_full_summary.json`.
  At k = 1–2 the correlation over 0.02 is 0.9993, so Δ there is ≈ 4% of θ.
- **Round 1b only compared frames,** not deltas.
- **Two float32 mechanisms are already known:**
  - `decay` stored in float32 quantizes each mode's damping rate: up to ±20% at k = 4–16 and 0.1% at k = 1
    (`decay_quant.py` in the job notes).
  - A float32 state rounds sub-ulp per-step changes systematically. `test_f32_update.py`: for one forced
    k = 8.4 mode, the 4,000-step delta is 31% off for `lean32`'s update and also for the expm1 damping. The
    compensated update gets it to 8.8e-8.

## The gate

Fixed in `delta_fidelity.py` before submission.

- **Metric.** The lag is 4,000 steps (o1.2's pair). Take the median over 8 start states 0.02 apart on a float64
  trajectory. The metric is e_Δ = ‖Δ_v − Δ_f64‖ / ‖Δ_f64‖ on o1.2's own coarse float32 frames.
- **Limit.** e_Δ ≤ max(2 × e_Δ of `f64ic32`, 1e-6). It must hold overall and in every 256²-frame |k| octave that
  carries ≥ 1% of Δ's energy.
- **The `f64ic32` control.** It is float64 from the same state rounded once to complex64: the delta error that any
  float32 description of the state already implies.
- **Why this limit:**
  - It is strict on purpose. A variant passes only if its float32 arithmetic adds no more than that one rounding does.
  - The UViT's own error is reported next to it (val MSE 0.0043 → relative error ≈ 0.25), but it does not gate.

## Predictions

1. **Spin-up** (synthetic −5/3 field, float64 `lean_u1`, 0.5 time units) stays stable:
   - CFL ≤ 0.45 at every report;
   - `lean_u1` == production `ref` bitwise on pair 0 (95%).
2. **`edec` table** reproduces `decay` to ≤ 1e-9 absolute.
3. **Increments** (1 step, from an identical float32-representable start, vs float64):
   - **Size.** The 1-step increment is ≈ 1e-5–5e-5 of θ at k = 1–2.
   - **`lean32`:** its 1-step increment error at k = 1–2 is ≥ 1e-3, from state rounding over a small increment.
   - **`lean32e`:** within 2× of `lean32`.
   - **`lean32ek` and `mixed`:** ≤ 1e-5 at k ≤ 16.
   - If `lean32ek` ≈ `lean32`, the compiler has undone the compensation.
4. **Pairs, at lag 4,000, median over 8:**
   - **Floor:** `f64ic32` e_Δ is between 1e-6 and 1e-4 overall.
   - **`lean32` FAILS** the gate (85%), on the low-k bands. Its overall e_Δ is 1e-4–3e-3.
   - **`lean32e` FAILS** (85%), within 2× of `lean32`.
   - **`lean32ek` PASSES** (70%).
   - **`mixed` PASSES** (80%).
   - **Every variant's e_Δ is ≤ 1% of the UViT's relative error (90%).** Even `lean32`'s error would be far below
     the model's. The gate is stricter than that by design.
   - rms(Δ) in o1.2's units (θ/8) is 0.15–0.40 (archive: 0.27).
5. **Speed:**
   - `lean32e` = `lean32` ± 3%;
   - `lean32ek` = `lean32` + 0.08–0.25 ms/step (one more complex64 array read and written per step);
   - `mixed` ≈ 4.7 ms/step.

## Limits of this job

- **Start states.** They come from a 0.5-time-unit spin-up of a synthetic field, not the archived checkpoints
  (not staged on Polaris). The definitive version reruns `delta_fidelity.py --state-in` on 2–3 archived
  checkpoints converted to npz.
- **Pair fidelity only.** Distributional bias (long-run spectra, PDFs, energy) needs ≥ 1 time unit per variant:
  a later job, from the saved spin-up state.
