# Mixed-precision probes with `tilerows_inrow_if25`: predictions (written 2026-11-16, before either job)

**Ask (user, 2026-11-16):**
- The user asked about "adaptive mixed precision as it choose between types but avoids error and has been applied
  in cholesky factorization".
- My answer: probably not for the solver step, but the ML step has real headroom. I offered two cheap probes.
- The user: "yes do both probes, use tilerows_inrow_if25 as the solves for these probes".

**Nothing is admitted by these probes.**
- A probe-1 PASS means a real 16-bit kernel is worth building and taking through gates A/B/E/D.
- A probe-2 speedup changes the trained model. o1.2 pins strict IEEE FP32 on purpose (L128-134, "checked against
  float64"), so using it needs a training-outcome study and the owners' agreement.

## Probe 1: 16-bit storage in the solver (`run_mixprec_ab_polaris.pbs`, debug)

**What is emulated** (`mixprec.py`).
- The in-row step is unchanged. After each op that writes an intermediate to GPU memory, that array's kept columns
  (k_x ≤ 1,365) are rounded to 16 bits. There are six such points: W_spec, W_mix, TH, Z_mix, Z_spec and the stored
  tendency T.
- This models 16-bit storage with fp32 compute: the version that saves bytes in a memory-bound step. It is optimistic
  for any real 16-bit FFT.
- The state (hi + lo) stays float32 with its compensation.
- **Formats:**
  - bf16 is torch's round-to-nearest-even: 8 significant bits, float32's range.
  - fp16 is block floating point: a power-of-two scale per column (column max < 2^15), then float16 RNE.

**Arms** (`tilerows_inrow_if25@<policy>`):

| policy | meaning |
|---|---|
| `none` | no rounding (the reference; must reproduce job 7854208's `tilerows_inrow_if25` frames bit for bit) |
| `bf16_all` | every kept column bf16 |
| `fp16_all` | every kept column fp16 |
| `fp16_kx128` | fp16 for \|k_x\| > 128 only: the scales the 256² frames do not resolve |
| `ad1`, `ad32`, `ad1024` | **adaptive.** Each array, at each write, gives each column the lowest of bf16 / fp16 / fp32 with u_p²‖X_c‖² ≤ (τ u₃₂)² ‖X‖² / n. The added rounding error is then ≤ τ × float32's own, for τ = 1, 32 and 1,024 (the per-tile bucketing of adaptive mixed-precision Cholesky). |

**Data and rules.** The A/B pipeline is unchanged:
- `tr_pairs.py` with the arms on the 16 archived states × 4 pairs (64);
- `gates_ab_tr.py` once per arm, as the candidate;
- `ACCEPTANCE_gates_AB.md`'s thresholds.

**Class per arm** (`analyze_mixprec.py`; the reference is `@none`):
- **PASS:** valid; gate A PASS; gate B PASS or FAIL_FLOOR on B3 alone in every band (as `tilerows_inrow_if25` itself);
  and the bias pattern's size (size_ratio) ≤ 2 × the reference's.
- **BIAS_UP:** as PASS, but size > 2 × the reference's.
- **FAIL_AB:** gate A FAIL, or any band failing B1 / B2 / power.
- **INVALID:** the shared validity (V_frames, V_sealed, V_floor) fails.

**Prize** (an upper bound): 0.5 × the mean over the six arrays of the 16-bit column fraction × 1.470 ms (the IF step,
job 7853317).

**Time.**
- The A/B job 7854208 took 1,098 s (two arms, 3.8 s per arm-pair). Each worker spends 26 s per pair regenerating
  lean_u1 and 111 s on each state's ref160.
- Seven arms at ~10 s per arm-pair (rounding included) gives ≈ 30–35 min, inside debug's hour.

## Probe 2: the network's precision inside o1.2 (`run_o12_tilerows_polaris.pbs`, debug-scaling)

- **Setup:** option C's harness and knobs, as job 7853365 (batch 16, 4 epochs × 30 steps, `O12_TR_GATE=1`). Solver
  `tilerows_inrow_if25`.
- **`O12_PREC`** (`o12_polaris.py`, two line-count-keeping patches, default = o1.2's own):

| arm | network |
|---|---|
| `tilerows_inrow_if25` | `ieee`: o1.2's strict FP32 (its check asserts) |
| `tilerows_inrow_if25_tf32` | TF32 matmul + cuDNN conv; the strict-FP32 check must fail (proof TF32 is on), and the failure is logged |
| `tilerows_inrow_if25_bf16` | strict flags; the training forward under bf16 autocast (the output dtype is logged); validation stays FP32 |

- **Timing** is `o12_extrapolate.py`'s s_step and full-schedule hours, unchanged.
- **Information only:** the epoch losses against `ieee` (same seeds, 120 steps; nothing is decided by them).

## Predictions (P = probability)

| # | prediction | P |
|---|---|---|
| V1 | probe 1 valid: tests pass (`MIXPREC_TEST_OK torch=ok`), V_frames 64/64, V_sealed, V_floor | 0.9 |
| V2 | V_wrap: `@none` reproduces job 7854208's frames bit for bit, 64/64 pairs | 0.85 |
| P1 | `bf16_all` is FAIL_AB | 0.75 |
| P2 | `fp16_all` is not PASS | 0.7 |
| P3 | `fp16_kx128` is not PASS | 0.6 |
| P4 | `ad1` is PASS | 0.85 |
| P5 | `ad32` is PASS | 0.7 |
| P6 | `ad1024` is not PASS | 0.55 |
| P7 | `ad1`'s mean 16-bit column fraction ≤ 0.25 | 0.7 |
| P8 | the largest prize among PASS arms ≤ 5% of the step (≤ 0.0735 ms) | 0.7 |
| M1 | probe 2: all three arms complete with finite losses, and the in-situ gate passes in each | 0.85 |
| M2 | the knobs took effect: tf32's strict check fails, and bf16's forward output is `torch.bfloat16` | 0.9 |
| M3 | the `ieee` arm's s_step is within ±2% of 0.8885 (job 7853365) | 0.85 |
| M4 | tf32 s_step ≤ 0.95 × ieee | 0.65 |
| M5 | bf16 s_step ≤ 0.90 × ieee | 0.6 |
| M6 | bf16 is faster than tf32 | 0.6 |

**Scoring:** `score_mixprec.py --p1 SQG/analysis/results/polaris_mpab_<job> --p2 SQG/analysis/results/polaris_o12tr_<job>`,
written before the results are read. End tokens: `MPAB_JOB_DONE failures=N`, `O12TR_JOB_DONE failures=N`.
