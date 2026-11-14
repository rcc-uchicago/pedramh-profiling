# Tile-rows (and IF-AB3) inside o1.2 itself: predictions (written 2026-11-12, before any run)

**Ask.** Option C of the 2026-11-12 plan. The user chose A, the 50-TU stability run (`stability/`, jobs 7840661 /
7840662), with "I build C (o1.2 integration) on debug-scaling meanwhile". The tile-row step measured 1.578 ms/step
standalone (job 7840376). This job runs it inside o1.2's own training loop and times what an o1.2 run would pay.

**Design** (`tilerows_advance.py`, wired into `o12_polaris.py`; `run_o12_tilerows_polaris.pbs`, 1 node,
debug-scaling).
- o1.2 compiled with the same 34 patches as job 7835278. Same knobs: batch 16, 600 fill pairs, 4 epochs × 30 steps,
  64 val pairs, members 1–4.
- **Arms, one after another:**
  - `lean32ek_fast`: same-node baseline (job 7835278: 1.7107 s/step);
  - `if32_25`: pure-JAX float32 IF-AB3 at 25 steps per frame;
  - `tilerows40`: the tile-row step at DT, 40 steps per frame;
  - `tilerows_if25`: tile-rows + IF-AB3 at 25 steps per frame;
  - `tilerows_if25_cufft114`: the same solver with CUDA 12.9.1's cuFFT 11.4.1 + nvJitLink preloaded, as in job
    7840376.
  - Every other arm uses the env's own cuFFT 11.0.2: the owed rerun without the preload.
- **Segment starts:** IF arms switch from the checkpoint's DT history with the 2 float64 IF-AB3 steps, so frames stay
  on o1.2's 2e-4 grid. Tile-row arms keep the true state on the GPU between frames and give o1.2's `coarse` a
  complex128 qh through DLPack.
- **Gate in situ** (O12_TR_GATE=1, rank 0, first segment): frame 1 and frame 4 vs the float64 reference of the same
  scheme, next to the float32 JAX twin.

## Gates (fixed now)

| gate | arms | rule |
|---|---|---|
| G_TR | tilerows40, tilerows_if25(_cufft114) | `O12_TR_GATE_VERDICT pass=True`: rel(arm, ref64) ≤ min(1e-4, 2 · rel(twin32, ref64)) at frames 1 and 4. ref64 / twin32 = lean_u1 / lean32ek_fast (tilerows40) or if64 / if32 at h (IF arms) |
| G_RUN | all | `O12_RUN_DONE`; finite train and val losses |
| WORTH | tilerows_if25 | s_step(tilerows_if25) ≤ 0.80 × s_step(if32_25), i.e. the Warp/cuFFT stack buys ≥ 1.25× over pure JAX at the same numerics family |

- First-segment frames vs production (`O12_BITWISE`) are INFO for every arm here: none is bitwise.
- Losses are in-sample (val store from the training members): not a science number.
- The science gates for IF-AB3 (option B, gate-D style) are the user's call, not this job.

## Predictions (s_step = steady seconds per optimizer step, rank 0)

Model: s_step = ML 0.4015 s + 12 frames × (steps per frame × ms/step + per-frame overhead). The ML term is
1.7107 − 12 × 0.10910 from job 7835278. Overhead: ~1.0 ms (coarse) for JAX arms, ~1.4 ms for tile-row arms (+ the
expand, DLPack, two syncs).

| # | prediction | P |
|---|---|---|
| 1 | G_TR passes for tilerows40 and tilerows_if25 | 0.85 |
| 2 | every arm passes G_RUN | 0.80 |
| 3 | lean32ek_fast s_step 1.68–1.75 (same as job 7835278 within 2%) | 0.90 |
| 4 | if32_25 s_step 1.17–1.31 (model 1.24) | 0.70 |
| 5 | tilerows40 s_step 1.10–1.25 (model 1.18) | 0.70 |
| 6 | tilerows_if25 s_step 0.84–0.97 (model 0.89) | 0.70 |
| 7 | WORTH holds (model ratio 0.72) | 0.75 |
| 8 | the arm without the preload runs (nvmath's raw cuFFT bindings work on cuFFT 11.0.2) | 0.75 |
| 9 | tilerows_if25 and tilerows_if25_cufft114 s_step within 3% of each other | 0.70 |
| 10 | DLPack path both ways (no host fallback) | 0.85 |
| 11 | o1.2's 14-A100 schedule with tilerows_if25: 45–54 h (model 49 h; lean32ek_fast 94 h, ref 268 h) | 0.65 |
| 12 | peak PyTorch memory ≤ 21.5 GiB (job 7835278: 20.4 GiB; tile-row buffers ~0.6 GiB) | 0.80 |
| 13 | final val loss of every new arm within 1% of the lean32ek_fast arm's (info; in-sample) | 0.80 |

## Tokens

- Per arm: `O12_TR_ENV`, `O12_TR_SCHEME`, `O12_TR_PATH`, `O12_TR_GATE`, `O12_TR_GATE_VERDICT`, plus the existing
  `O12_FILL`, `O12_BITWISE`, `O12_EPOCH`, `O12_PHASES`, `O12_RUN_DONE`.
- Job: `O12_EXTRAPOLATE_OK`, `O12TR_JOB_DONE failures=N`.
- Results → `SQG/analysis/results/polaris_o12tr_<jobid>/`.
- Scoring: `score_o12_tilerows.py` (mechanical, from the logs + `o12_extrapolation.json`).

## Result of job 7840805 (2026-11-12, scored by score_o12_tilerows.py)

- **Score:** 9 confirmed, 4 refuted (5, 6, 7, 11). G_TR and G_RUN pass for every arm.
- **WORTH fails:** tilerows_if25 1.2222 s/step vs if32_25 1.2404, ratio 0.985 > 0.80. This verdict stands.
- **Cause, from the fill frames:** 89.2 ms per 40-step frame vs 65.5 ms per 25-step frame. That is 1.58 ms/step
  (the standalone kernel) plus a fixed ~26 ms per frame in tilerows_advance.py's glue. The JAX arms pay ~1 ms.

## Follow-up (written 2026-11-12, after job 7840805, before the next job): diagnose and remove the glue overhead

**Suspect.** `expand` builds the (1, N, N/2+1) complex128 qh with an XLA scatter (`.at[rows, :CC].set`). Scatter
of 16-byte elements can take a slow serialised path on the GPU.

**Change.** `full_from_compact`: the same array from two contiguous row blocks around a zero block, then zero
columns.
- Pure copies, no scatter.
- Equals `tilerows_step.from_compact` exactly (numpy test on the login node).
- `O12_TR_EXPAND=concat` (new default) | `scatter` (job 7840805's).

**Arms** (one debug node, same knobs): `if32_25`, `tilerows_if25` (concat), `tilerows_if25_scatter_t` and
`tilerows_if25_concat_t`. The `_t` arms set `O12_TR_TIMING=1`. That blocks on expand, so the `_t` arms are for the
breakdown, not for s_step.

**Rules unchanged.**
- G_TR and G_RUN as above.
- WORTH as above: s_step(tilerows_if25) ≤ 0.80 × s_step(if32_25), both from this job.
- A pass here is a pass for the changed glue, reported next to job 7840805's fail, not in place of it.

| # | prediction | P |
|---|---|---|
| F1 | `O12_TR_EXPAND_CHECK concat_equals_scatter=True` | 0.97 |
| F2 | scatter arm: median `handoff_expand_ms` ≥ 15 (the overhead is the scatter) | 0.60 |
| F3 | concat arm: median `handoff_expand_ms` ≤ 3 | 0.75 |
| F4 | tilerows_if25 (concat) s_step ≤ 1.00 (model 0.40 + 12 × (25 × 1.58 + 1.5) ms = 0.89) | 0.60 |
| F5 | WORTH holds for the concat glue | 0.55 |
| F6 | if32_25 s_step within 1% of job 7840805's 1.2404 | 0.90 |
| F7 | G_TR passes for both tilerows_if25 arms that run the gate | 0.95 |

Scoring: `score_o12_tilerows_fix.py --res <dir>` (written before the job).
