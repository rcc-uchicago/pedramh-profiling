# Predictions for the o1.2-on-Polaris timing job (`run_o12_polaris.pbs`)

Written 2026-11-10 by the main session, before any o12 job was submitted. Do not edit after submission. Scored in the
notes (`polaris_sqg_notes.md`, next Part) after the job.

## Why this job

**The ask (user, 2026-11-10).** "Can we not run our own checkpoints and generate logs". Then: "we don't want to compare
it to the H200, we know we can't get the same results, but we can extrapolate the speeds from A100".

- **What item 1 of the team plan wanted from the production logs:**
  - a measured wall-clock: fill, steady-state s/step, validation, checkpoint saves, overhead;
  - the final validation losses.
- **What this job measures instead.** The timing half of that, for OUR runs on A100. It runs **o1.2 itself** (the
  file on disk, unedited) through `o12_polaris.py`, short, on one debug node, once per solver. From the pieces,
  `o12_extrapolate.py` projects a full o1.2 schedule's A100 wall-clock.
- **What it does not give.**
  - Any number about the H200 runs.
  - A trained model. Item 2's eps_model,b needs a full-length run, which is a separate decision for the user.

## What runs

- **Node.** 1 Polaris debug node, 4 A100s, `torchrun --standalone --nproc_per_node=4`, as o1.2's own launcher (one
  rank per GPU).
- **Three arms in turn**, each a fresh run directory:

  | arm | solver | why |
  |---|---|---|
  | `ref` | o1.2's own call | production |
  | `lean_u1` | float64, bitwise production | 7815637 / 7824121 |
  | `lean32ek_fast` | compensated float32 | candidate, not admitted; timed only, its frames are not used for anything |

- **Knobs** (`O12_*`, logged; everything else is o1.2's):

  | knob | value | o1.2 | why |
  |---|---|---|---|
  | BATCH | 16 | 32 | 32 does not fit a 40 GB A100 (profiling C) |
  | START_PAIRS | 600 | 19,604 | short fill; the steady fill rate is measured and extrapolated |
  | N_EPOCH × STEPS_PER_EPOCH | 4 × 30 | 250 × 781 | short schedule |
  | N_VAL | 64 | 200 | the stand-in val store has 64 pairs |
  | BUF | 16,384 | 262,144 | identical computation: the run makes < 3,000 pairs, so the buffer never wraps; avoids 137 GB / rank of host memory |

- **Patches** (`o12_polaris.PATCHES`). Each matches exactly once and keeps the line count:
  - paths;
  - the torch 2.3.1 spelling of strict IEEE fp32 (o1.2's own `check_fp32()` still asserts);
  - the vendored solver's names: BLOCK_T is the 0.02 pair spacing, `CHECKPOINT_DIR` layout;
  - the staged members 1–4;
  - zarr / wandb / matplotlib stand-ins (none of them is in sqg-env);
  - o1.2's first-segment bitwise check against `train_m16.zarr`, replaced by the same check against job 7824121's
    production frames of the same checkpoints.
- **Val store.** `build_val_store.py`: production frames of the 16 archived states, 0.02 apart; 64 pairs. They are
  in-sample, so the val loss is NOT a science number.

## Gates (pre-set)

- **G1.** `test_o12run.py` prints `O12_TEST_OK` under sqg-env: patches, compile under 3.10, stand-ins, parser and
  formula.
- **G2.** o1.2's own `check_fp32()` passes (strict IEEE fp32 under torch 2.3.1).
- **G3.** `ref` and `lean_u1`: `O12_BITWISE … PASS`. Their first-segment frames (k = 100, 200, 300, 400, 4 per rank)
  must equal the taxonomy job's production frames bit for bit. A FAIL aborts that arm, as o1.2 does, and voids its
  timings as "production".

## Predictions

The solver ms/step are job 7824121's medians: `ref` 9.29 (round 1, job 7814300), `lean_u1` 6.404, `lean32ek_fast` 2.698.
The ML step is t_ml ≈ 0.42 s at batch 16 (Part III 0.417; 7815637's free trainer 0.4235). Production's order is a
pure sum (Part III; round 1c's replicate lean32 1.946 = 0.480 × 3.196 + 0.41).

| # | prediction | P |
|---|---|---|
| 1 | All 3 arms reach `O12_RUN_DONE` with finite losses; G1 and G2 pass | 0.85 |
| 2 | G3 passes for `ref` and `lean_u1`; `lean32ek_fast` is INFO with 1e-7 ≤ max \|Δθ\| ≤ 1e-3 | 0.8 |
| 3 | Steady s/step (median of per-epoch medians, epochs ≥ 1) = 0.480 × ms_solver + 0.42 within ±7%: `ref` 4.88 [4.54, 5.22], `lean_u1` 3.49 [3.25, 3.73], `lean32ek_fast` 1.72 [1.60, 1.84] | 0.7 |
| 4 | Steady fill s/frame ∈ [1.00, 1.10] × 40 × ms_solver: `ref` [0.372, 0.409], `lean_u1` [0.256, 0.282], `lean32ek_fast` [0.108, 0.119] | 0.7 |
| 5 | Per epoch: validation ≤ 5 s (1 batch per rank) and `save_last` ≤ 10 s | 0.8 |
| 6 | Peak PyTorch GPU memory at batch 16 ∈ [19, 22] GiB (profiling C: 20.24–20.4) | 0.7 |
| 7 | t_init (start → fill done, minus the fill) ≤ 180 s | 0.6 |
| 8 | In the full-run projection, the optimizer-step term is ≥ 95% of the total for every solver and configuration | 0.8 |

**Projection** (`o12_extrapolate.py` docstring; reported, scored only through P8). Configurations:
- 1 node × 4 GPUs;
- 4 nodes × 14 GPUs at batch 16: o1.2's global batch 224 and its 195,250 steps;
- 4 nodes × 16 GPUs.

What the P3 values imply for the 14-GPU configuration: `ref` ≈ 269 h, `lean_u1` ≈ 193 h, `lean32ek_fast` ≈ 96 h
(roughly steps × s_step + 2 h fill + 1.3 h segment loads).

## Limits (true whatever the numbers)

- **Hardware.** A100 timings only. The H200 is not compared (the user's framing).
- **Not modelled:**
  - restarts: o1.2 resumes from `best.pt` and re-runs the epochs after it;
  - the decoupled layout (round 1c).
- **Cross-node DDP** is added as 5 ms/step (7815637), not measured here.
- **A full o1.2 run on Polaris as written does not fit.** The 137 GB / rank host buffer × 4 ranks exceeds a 512 GB
  node, so BUF must shrink (that changes sampling) or the frames must be de-duplicated (137 → ~69 GB). It also needs
  3,281 checkpoints (≈ 1.3 TB) where we have 16. With 16 it wraps around: fine for timing, not for science.
