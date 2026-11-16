# PREDICTIONS — the in-row SQG solver and o1.2 on Midway3's H100 NVL, timed as on Polaris

Written 2026-11-16, **before Job 2** (no H100 number for any of these arms exists yet). Scorer:
`SQG/midway3/score_midway3_timing.py` (stdlib; reads the JSON block at the end of this file, so the numbers below
are the only copy). Protocol: `MIDWAY3_INROW_PROMPT.md` §6 — the Polaris harnesses unchanged, one node, 4 GPUs.

## What is compared

Every A100 number below is measured on Polaris (A100-SXM4-40GB; jobs 7814300 / 7853317 / 7835278 / 7840909 /
7853365). The H100 runs are on Midway3 `pedramh-gpu` (midway3-0423: 4 × H100 NVL, ~94 GB, NVLink only within the
pairs GPU0↔1 and GPU2↔3, PCIe Gen4 host), with the same env pins as Polaris' `sqg-env` (wdenv: jax 0.4.28,
torch 2.3.1, cuFFT wheel 11.0.2.54, cuDNN 8.9.2; verified by `setup_sqg_midway.sh`, 2026-11-16), Warp 1.17.0 and
nvmath-python 1.0.0.

"Speedup" is always A100 time / H100 time (> 1 = H100 faster).

## Reasoning

- **Solver step (Level 1).** The f32 steps are memory-bound: the Warp tile kernels run at 82–96% of copy speed on
  A100 (`warp/WARP_BUILTINS_AUDIT_2026-11-14.md`). Peak HBM bandwidth: A100 40 GB 1.555 TB/s, H100 NVL 3.9 TB/s,
  ratio 2.5. Achieved copy bandwidth rarely scales fully, and short kernels (`halve0`, 8 µs) are launch-bound, so
  the point is **2.1×** with interval **[1.5, 2.7]** (the top allows the larger L2: 50 vs 40 MB).
  The f64 arms (`ref`, `lean_u1`) are different: XLA's own cost estimate for `ref` is 68 GFLOP and 4.2 GB per step
  (seed1.json, job 7814300), i.e. 7.3 TFLOP/s but only 0.46 TB/s at 9.29 ms. That is near A100's 9.7 TFLOP/s
  non-tensor FP64 peak, so these arms may scale with FP64 throughput instead (H100 NVL ~30 TFLOP/s on paper,
  3.1×). Point **2.3×**, interval **[1.5, 3.2]**.
- **o1.2 step (Level 2).** On A100, every arm satisfies `s_step ≈ 12 × s_frame + ML`, with ML ≈ 0.40–0.43 s
  (`ref` 4.897 = 12 × 0.3728 + 0.42; `tilerows_inrow_if25` 0.8885 = 12 × 0.0386 + 0.43). The solver frames scale
  as Level 1. The ML step is strict IEEE FP32 (o1.2 pins it; no TF32 or bf16 here), so it gains from H100's
  non-tensor FP32 throughput (~3× on paper) and bandwidth, but it is partly cuDNN-convolution and launch bound, and
  DDP's gradient all-reduce (40.81 M params, 163 MB per step) crosses PCIe between the two NVLink pairs on Midway
  where Polaris has NVLink everywhere. Point **2.0×**, interval **[1.3, 3.0]**.
  Predicted H100 `s_step` = `12 × s_frame_A100 / solver speedup + ML_A100 / ML speedup`; the interval combines the
  two intervals' ends.
- **Directional (low confidence).** The ML step shrinks *less* than the solver frames
  (ML speedup < frame speedup for `tilerows_inrow_if25`), for the PCIe all-reduce and launch-bound reasons above.
- **Full schedule (Level 3).** `o12_extrapolate.py`'s formula is fixed. At batch 16 on 1 node there are 683,500
  steps and the step term is > 98% of the total on A100, so 1-node hours ≈ 683,500 × s_step / 3600 plus 0–5 h of
  fill, validation, saves and segment loads.
- **Batch 32 (Job 3C).** o1.2 advances 12 ring frames per ML step whatever the batch (o1.2 L184), so the solver
  term per step stays the same while the ML step roughly doubles and the step count halves (341,750 steps). So
  `s_step(b32) / s_step(b16)` is near 1 for `ref` (solver-dominated) and near 1.4 for `tilerows_inrow_if25`. Peak
  memory: A100 b16 20.9 GiB → H100 b32 ~40 GiB (activations double; the JAX solver and the model do not).

## Gates (validity: a failed gate makes the run invalid, it is not a prediction miss)

| id | gate | source |
|---|---|---|
| E1 | Job 1: 4 × H100 in torch and jax, Warp `mathdx` = 1, the 4096-point tile FFT within 1e-5 of torch | `midway_env_extra.json` |
| S1 | seed 1: `lean`, `lean_u1`, `lean_split` pass G1–G3 (+ round 1b's `t` rule), as Polaris round 1c (7815637) | `seed1.json` |
| S2 | seed 1: `ref` drift between its first and last timing ≤ 1% | `seed1.json` |
| I1 | in-row probe: `valid`, G32, GIF, TR_G32, TR_GIF all true; `failures` = 0 | `inrow_probe_prod.json` |
| I2 | `lean_u1` timed by seed 1 and by the probe agree within 2% (A100: 0.07%) | both |
| O1 | every o1.2 arm logs `O12_RUN_DONE` and its 1-node projection | `o12_extrapolation.json` |
| O2 | every IF / tile-row arm logs `O12_TR_GATE_VERDICT ... pass=True` (O12_TR_GATE=1) | arm logs |
| O3 | batch 32 peak PyTorch memory ≤ 90 GiB | `o12_extrapolation.json` |

`O12_BITWISE ... NO_REFERENCE` is expected (no taxonomy frames on Midway) and is not a gate.

## Predictions

All in the JSON block. In words, for the six solvers (A100 → predicted H100 point [interval]):

| solver | ms / solver step | o1.2 s / opt step | 1 node, b16 (h) |
|---|---|---|---|
| `ref` | 9.290 → 4.04 [2.90, 6.19] | 4.897 → 2.16 [1.54, 3.31] | 933.1 → ~410 [292, 633] |
| `lean_u1` | 6.406 → 2.79 [2.00, 4.27] | 3.493 → 1.55 [1.10, 2.37] | 665.1 → ~293 [209, 455] |
| `lean32ek_fast` | 2.696 → 1.28 [1.00, 1.80] | 1.711 → 0.82 [0.62, 1.18] | 326.0 → ~157 [118, 229] |
| `if32_25` | 2.746 → 1.31 [1.02, 1.83] | 1.240 → 0.60 [0.44, 0.87] | 237.0 → ~114 [84, 170] |
| `tilerows_if25` | 1.592 → 0.76 [0.59, 1.06] | 0.926 → 0.45 [0.33, 0.66] | 176.6 → ~86 [62, 131] |
| `tilerows_inrow_if25` | 1.470 → 0.70 [0.54, 0.98] | 0.8885 → 0.43 [0.31, 0.64] | 169.4 → ~82 [60, 126] |

Also predicted: the in-row per-op split sums to within 5% of the eager `inrow32` step (A100: 1.1%); the nvmath
cuFFT version is 11002 (the same wheel as Polaris); `s_step(b32) / s_step(b16)` is in [1.0, 1.3] for `ref` and
[1.2, 1.9] for `tilerows_inrow_if25`; batch 32 needs fewer 1-node hours than batch 16 for both; the b32 peak is
in [30, 60] GiB.

```json PREDICTIONS
{
 "a100": {
  "ms": {"ref": 9.2897, "lean_u1": 6.4063, "lean32ek_fast": 2.6961, "if32_25": 2.7464, "tilerows_if25": 1.5922,
         "tilerows_inrow_if25": 1.4701},
  "s_step": {"ref": 4.8969, "lean_u1": 3.4925, "lean32ek_fast": 1.7107, "if32_25": 1.2395, "tilerows_if25": 0.9261,
             "tilerows_inrow_if25": 0.8885},
  "s_frame": {"ref": 0.3728, "lean_u1": 0.2571, "lean32ek_fast": 0.1091, "if32_25": 0.0700, "tilerows_if25": 0.0419,
              "tilerows_inrow_if25": 0.0386},
  "hours_1node_b16": {"ref": 933.1, "lean_u1": 665.1, "lean32ek_fast": 326.0, "if32_25": 237.0, "tilerows_if25": 176.6,
                      "tilerows_inrow_if25": 169.4},
  "inrow32_split_ms": {"build_w": 0.1475, "cufft_cols_w": 0.2877, "cufft_cols_th": 0.1290, "halve0": 0.0082,
                       "tile_rows_ir": 0.4178, "cufft_cols_z": 0.2908, "update": 0.2171},
  "inrow32_eager_ms": 1.4817, "inrow32_graph_ms": 1.4672,
  "sources": "seed1 7814300 (ref); inrow probe 7853317 (others, split); o1.2 7835278 / 7840909 / 7853365"
 },
 "f64_arms": ["ref", "lean_u1"],
 "solver_speedup": {"f64": [2.3, 1.5, 3.2], "f32": [2.1, 1.5, 2.7]},
 "ml_speedup": [2.0, 1.3, 3.0],
 "steps_1node_b16": 683500,
 "steps_1node_b32": 341750,
 "hours_overhead_max": 5.0,
 "split_sum_vs_eager_max": 0.05,
 "cufft_version": 11002,
 "b32_ratio": {"ref": [1.0, 1.3], "tilerows_inrow_if25": [1.2, 1.9]},
 "b32_peak_gib": [30, 60],
 "gates": {"lean_u1_agree_max": 0.02, "ref_drift_pct_max": 1.0, "b32_peak_gib_max": 90, "tile_fft_rel_max": 1e-5}
}
```
