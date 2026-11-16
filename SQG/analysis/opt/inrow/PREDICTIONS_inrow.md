# In-tile C2R (the audit's "up to ~13%" item): predictions (written 2026-11-16, before any run)

**Ask (user, 2026-11-16):** "and what about including the last item from the documentation", then "show the
documentation analysis there was a 13% boost see the last session history". The item is candidate 1 of
`warp/WARP_BUILTINS_AUDIT_2026-11-14.md`: fold the inverse real FFT into the row kernel. The audit estimated −0.2 to
+0.02 ms per step, "up to ~13% faster, or no gain", and said it needs a probe first, then the full gate battery.

**What changes** (`inrow_step.py`; the arithmetic is otherwise `tilerows_step.py`'s):
- The state's `hi` is stored in FFT row order, (N, CC), with zero rows between ±KM. θ's inverse column pass becomes one
  out-of-place cuFFT C2C straight from the state. `expand` (0.079 ms) and the 2-D C2R (0.302 ms) are removed.
- `halve0` halves TH[:, 0], which is exact.
- The row kernel loads TH's one-sided row (k_x > KM read as zero) and runs `tile_ifft`, giving q_u = 2·Re G. It then
  does tilerows' FFT → product → FFT. That is 3 tile FFTs per row instead of 2.
- **Identity:** N²·irfft2 (Im of the DC bin dropped, as numpy's irfft does). `test_inrow_math.py` checks it to
  2.4e-16, including a non-Hermitian k_x = 0 column. It passed on the login node, 2026-11-16.
- θ is computed first and only its real tile stays live during W's FFT. Job 7840069's fused 3-FFT row kernel, with
  two complex tiles live, ran at 0.549 ms, against 0.302 ms for the 2-FFT kernel.

**Run** (`inrow_probe.py`; `run_inrow_polaris.pbs`, one A100, debug, from a frozen snapshot):
- **State:** member 1, t = 100 (as job 7840376).
- **Pass 1 (deciding): production cuFFT** (the wheels' 11.0.2, as o1.2 runs, option C).
- **Pass 2 (report only): cuFFT 11.4.1 preloaded**, as job 7840376.
- **Arms:**
  - `tilerows32` and `tilerows32_if`: the current step, re-measured in the same job;
  - `inrow32` and `inrow32_if`: the new step, kernel order `theta_first`, block 256 (fixed now as the deciding arm);
  - `inrow32_wfirst` (W's FFT first) and `inrow32_b512` (block 512): timing and rel only;
  - JAX references `lean_u1`, `lean32ek_fast`, if32, if64.
- **Measurements:**
  - rel at n = 1, 40, 4,000, and band errors of the 0.02 delta at 4,000;
  - ms/step eager (10 × 40 steps), a 48-step CUDA graph, and the per-op split, for `inrow32` and `tilerows32`.

## Validity, gates and rule (fixed now)

- **Validity** (else the arm is INVALID):
  - **THETA:** on the start state, the in-tile q_u against float64 N²·irfft2 has rms rel ≤ 1e-5. A coding error gives
    O(1). cuFFT C2R's q_u from `tilerows32` is reported beside it.
  - **HI_PRESERVED:** the out-of-place column pass leaves `hi` bit-identical.
  - **HI_ZERO_ROWS:** after the 4,000-step run, `hi` is still exactly 0 outside ±KM.
  - All states are finite.
- **G32:** rel(`inrow32`, `lean_u1`) ≤ min(1e-4, 2·rel(`lean32ek_fast`, `lean_u1`)) at n = 1 and 40. This is the
  tilerows probe's gate, unchanged.
- **GIF:** the same for `inrow32_if`, with if64 / if32.
- **WORTH (go to o1.2 integration):** valid, G32 and GIF pass, and `inrow32` eager ≤ 0.97 × `tilerows32` eager in pass
  1. The 0.97 is tilerows' own worth margin.
- **If WORTH:** next is a `tilerows_inrow_if25` arm in `o12run/tilerows_advance.py`, timed inside o1.2 (option C's
  harness). After that it needs the full gate battery A/B/E/D, as `tilerows_if25` did. Gate D runs on capacity, so it
  needs the user's go-ahead.
- **If not WORTH:** the measured split goes into the audit's "Do not re-try".

## Predictions (P = probability)

Prior: the bytes saved are 0.079 + 0.302 ms. The costs are θ's column pass (≈ 0.14 ms, half of a W/Z column pass)
and a third in-register FFT. The 2-FFT row kernel takes 0.302 ms and a 1-FFT row pass 0.232 ms. So if registers do
not spill, the new row kernel costs ≈ 0.36–0.41 ms, and the step ≈ 1.42–1.47 ms (7–10% faster). If it spills as
fused3 did, ≈ 0.55 ms (no gain).

| # | prediction | P |
|---|---|---|
| 1 | every tile arm builds (Warp + cuFFT plans) and runs finite | 0.75 |
| 2 | THETA valid: in-tile q_u rms rel vs float64 ≤ 1e-5 | 0.85 |
| 3 | the in-tile q_u error ≤ 1.5 × cuFFT C2R's (`tilerows32`) on the same state | 0.7 |
| 4 | HI_PRESERVED and HI_ZERO_ROWS hold | 0.9 |
| 5 | G32 passes | 0.75 |
| 6 | GIF passes | 0.75 |
| 7 | at 4,000 steps, `inrow32`'s band errors vs `lean_u1` are 0.5–2 × `tilerows32`'s (every band, sub and af) | 0.7 |
| 8 | `inrow32_wfirst` is bit-identical to `inrow32` at n = 40 | 0.7 |
| 9 | **WORTH:** `inrow32` eager ≤ 0.97 × `tilerows32` (pass 1) | 0.5 |
| 10 | `inrow32` eager ≤ 0.93 × `tilerows32` (the audit's mid estimate or better) | 0.3 |
| 11 | `tile_rows_ir` takes 0.33–0.45 ms | 0.5 |
| 12 | `cufft_cols_th` takes 0.10–0.18 ms | 0.6 |
| 13 | `halve0` takes ≤ 0.01 ms | 0.8 |
| 14 | `tilerows32` passes its own G32 / GIF again under production cuFFT | 0.85 |
| 15 | `inrow32_b512` is slower than `inrow32` | 0.6 |

## Tokens

- `INROW_MATH_TEST_OK`, `IR_ENV_OK`, `IR_BUILD`, `IR_THETA`, `IR_REL n=…`, `IR_GATE32` / `IR_GATEIF pass=…`,
  `IR_BANDS`, `IR_BENCH`, `IR_VERDICT`, `IR_PROBE_DONE failures=N`, per pass.
- The job ends `IR_JOB_DONE failures=N`.
- Results → `SQG/analysis/results/polaris_inrow_<jobid>/`.
- Scoring: `score_inrow.py <pass-1 json>`, written before the results are read.
