# `tilerows_inrow_if25` inside o1.2: predictions (written 2026-11-16, before the run)

**Why:** the in-tile C2R probe (job 7853317, notes Part XVII I) met its pre-set WORTH rule. It runs 1.470 vs 1.592
ms per IF-AB3 solver step. Its prereg's next step is this arm, timed inside o1.2 with option C's harness, to replace
the projection (`inrow/project_o12_inrow.py`: 0.890 s/step, 48.9 h on 14 A100s) with a measurement. The user asked
for "the resulting wall-clock time for tilerows_inrow_if25".

**What changes:** `tilerows_advance.py` gets the scheme `tilerows_inrow_if<n>` (kind `tri`).
- It builds `inrow_step.InRow` instead of `TileRows`.
- The compact `hi` goes into, and comes back from, the solver's FFT rows: `index_copy_` in, and `index_select` plus a
  stream sync out, before the DLPack hand-off to JAX.
- Everything else, including the in-situ gate, is option C's. The existing schemes run exactly the same code as
  before.
- `test_tilerows_advance.py` covers the parse and the hand-off identity. It passed on the login node, 2026-11-16.

**Run:** `run_o12_tilerows_polaris.pbs` with `O12TR_ARMS=tilerows_if25+tilerows_inrow_if25`, 1-node debug, from a
frozen snapshot.
- Option C's knobs: batch 16, 4 epochs × 30 steps, 600 start pairs, BUF 16384, `O12_TR_GATE=1`, production cuFFT.
- `tilerows_if25` is re-measured in the same job as the baseline.
- `o12_extrapolate.py` gives both arms' full-schedule hours by its fixed formula.

## Rules (fixed now)

- **Gate (unchanged, rank 0, first segment):** rel(arm, float64 IF-AB3 at h) ≤ min(1e-4, 2·rel(if32 twin, same)) at
  frames 1 and 4.
- **The measurement counts** if the gate passes, the run completes, the losses are finite, and the DLPack paths are
  `dlpack`.
- The measured hours then replace the projection in `table_o12_solvers.py` and notes Part XVII I.
- **Not decided here:** admission. The gate battery A/B/E/D for `tilerows_inrow_if25` comes next. Gate D runs on
  capacity, so it is the user's call.

## Predictions (P = probability)

| # | prediction | P |
|---|---|---|
| Q1 | both arms complete (`O12_RUN_DONE`) with finite losses | 0.85 |
| Q2 | the in-situ gate passes for `tilerows_inrow_if25` | 0.85 |
| Q3 | `tilerows_if25` re-measures within ±2% of 0.926 s/step | 0.8 |
| Q4 | s_step(`tilerows_inrow_if25`) / s_step(`tilerows_if25`) is 0.950–0.975 (projection 0.960) | 0.6 |
| Q5 | `tilerows_inrow_if25`'s 14-A100 projection is 48.0–49.8 h | 0.6 |
| Q6 | `tilerows_inrow_if25`'s DLPack paths are `dlpack` both ways | 0.85 |

Scoring: `score_o12_inrow.py --res SQG/analysis/results/polaris_o12tr_<job>`, written before the run. End token
`O12TR_JOB_DONE failures=N`.
