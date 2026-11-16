# Round 1b + 2-node smoke — predictions (written 2026-11-05, before job `run_round1b_multisim_polaris.pbs`)

Main session. **Round 1b's gates were written after round 1 (job 7814300) at the user's request**, so its predictions
below are informed by round 1's numbers. They are recorded anyway, and so is the reason each gate changed
(`seed2_opt.py`, ROUND 1b block):
- (a) Round 1's 1e-5 float32 limit sat below plain float32's own error. The ORIGINAL step run in float32 scored
  2.48e-5 on the synthetic state.
- (b) Every round-1 variant failed only on `t`, the float32 time counter, which the step never reads. Its fields were
  bitwise (seed 1) or ≤ 3e-14 (`lean64`).

Round 1's verdicts stand as recorded.

**New gates.** The float32 variants need a 1-step error ≤ min(1e-4, 2 × the original step's float32 error from the same
state, measured in the same run). `lean64` keeps ≤ 1e-12. All variants need `tc` and `_ablevel` exact and |t − t_ref| ≤
1e-6 |t_ref|. Exact `t` equality is reported as information. `t` now advances by a run-time `dt` (`c["tdt"]`), not a
constant, in case XLA folded the unrolled `(t + dt) + dt`.

## Phase 1 — round 1b (one A100 alone)

| # | prediction |
|---|---|
| 1 | `lean64` passes: 1-step ≈ 3e-14 (synthetic) / ≈ 1e-22 (random) ≤ 1e-12. `t` exact with the run-time `dt` (if XLA folding was the cause); within 1e-6 either way. |
| 2 | `mixed` passes: synthetic 1-step ≈ 2.3e-5 vs threshold 2 × 2.48e-5 ≈ 5.0e-5 (ratio to the float32 original ≈ 0.93); random ≈ 2e-13. |
| 3 | `lean32` passes: synthetic ≈ 2.3e-5 (ratio ≈ 0.93), random ≈ 6.8e-8 (ratio ≈ 1.0). |
| 4 | Timings within ±3% of round 1: `lean64` 7.07, `mixed` 4.68, `lean32` 3.20 ms/step (ref 9.29). |

## Phase 2 — 8-GPU smoke over Slingshot (2 nodes)

| # | prediction |
|---|---|
| 5 | NCCL probe: "Selected Provider is cxi" on both nodes with torch 2.3.1's bundled NCCL (2.20.x) + aws-ofi-nccl v1.6.0. **~60% confidence**: this pairing has never been run here (combine_handoffs used NCCL 2.28). 64 MB all_reduce bus bandwidth 10–20 GB/s (their 2-node nccl-tests: 18.0 GB/s on CXI). |
| 6 | `independent`: every rank within ±5% of its 1-GPU ms/step (ref 9.29, lean32 3.20). Independent GPUs share nothing on the hot path, so scaling efficiency ≥ 0.95. |
| 7 | `stream ref`: simulators ≈ 9.3–9.7 ms/step, ≈ 18–19 pairs/s into the trainer, trainer ≈ 0.42–0.45 s/step (batch 16), so ≈ 7.5–8.5 fresh pairs per trainer step: **solver-bound** (< o1.2's 12). |
| 8 | `stream lean32`: ≈ 52–55 pairs/s, ≈ 22–24 fresh pairs per trainer step: **trainer-bound** (≥ 12). Network ≈ 27–29 MB/s, trivial for Slingshot. Gloo send of a 6.3 MB chunk: < 5 ms same node, < 10 ms across nodes (combine_handoffs' TCP estimate ≈ 2 ms). |
| 9 | Nothing is written to disk by the smoke except its JSON/logs: pairs live only in the simulators' 101-frame rings and the trainer's in-RAM FIFO, as in online training. |
