# Round 1c — predictions (written 2026-11-05, before job `run_round1c_polaris.pbs`)

Main session. Scope comes from the user's decisions after job 7815096 (2026-11-05):
- rerun seed 1 under round 1b's `t` rule;
- carry `lean32` toward o1.2;
- measure both replicate and decouple;
- submit a 2-node debug job.

The NCCL arms follow combine_handoffs' plan. Reference numbers are from A100-40GB jobs 7814300 and 7815096 (notes
Part VIII). Nothing here may be edited after the job reports. Score it in notes Part IX / CHANGELOG.

**Differences from 7815096 in the NCCL phases:**
- `OMP_NUM_THREADS=1` (was 8);
- the NCCL group has a 90 s timeout (was none);
- arm A adds `FI_LOG_LEVEL=info`.

## Phase 1 — seed 1 rerun (one A100 alone)

| # | prediction |
|---|---|
| 1 | All three (`lean`, `lean_u1`, `lean_split`) pass. Their fields stay bitwise at 1, 40 and 480 steps from both start states (1-step error 0). `t` is within 1e-6, and **exact** with the run-time `dt` (≈ 85% confidence: seed 2's `t` became exact with the same change). |
| 2 | Timings within ±3% of round 1: `lean` 8.00, `lean_u1` 6.40, `lean_split` 7.07 ms/step (ref 9.29). A run-time `dt` scalar costs nothing measurable. |
| 3 | Best passing float64 variant: `lean_u1`, 1.45×. Its optimizer step (12 × 40 + coarse) is ≈ 3.1 s vs ref ≈ 4.47 s. That makes it faster than `lean64` (3.40 s) and the fastest float64 variant overall. |
| 4 | 4,000 steps from the synthetic state: coarse difference 0 and energy difference 0 for all three. |

## Phase 2 — NCCL over CXI, 8 ranks, 64 MB all_reduce

| # | prediction |
|---|---|
| 5 | **`A_filog` fails as 7815096 did** on the 4 GDR ranks: `cxil_map: write error`, then `Unable to register memory … RC: -14`, then no busbw. Its FI_LOG output contains `Failed to dlopen libcudart.so` and **no** `Number of CUDA devices detected` (≈ 70%: the file facts are verified, the mechanism is not). |
| 6 | **`B_shim` passes** (≈ 65%). All 16 inter-node channel lines read `GDRDMA`; there are 0 `cxil_map` errors, 0 registration failures and 0 `Failed to dlopen libcudart.so`. FI_LOG shows `Number of CUDA devices detected: 1` on each rank and no `Failed to find <cuda symbol>`. That marker comes from combine_handoffs' working 7629082 M1; a missing symbol in the 12.1 runtime is the named risk. busbw 12–20 GB/s (combine_handoffs' 2-node CXI: 18.0 GB/s at 64 MB). |
| 7 | **`C_loc` passes** (≈ 80%): host-staged, no GPU-memory registration. The NET lines carry no `/GDRDMA`. busbw 8–18 GB/s, ≤ `B_shim`'s if both pass. Caveat (script review): FI_LOG_LEVEL=info stays on during A/B/D's timed rounds but not C's, so the B-vs-C comparison is indicative, not like-for-like. |
| 7b | **`D_cuda129` passes** (≈ 60%, added before submission after a library check). This is makani/ACE2's prepend of `/soft/compilers/cudatoolkit/cuda-12.9.1/lib64`, which ships `libcudart.so` → 12.9.79. It behaves as `B_shim` (GDRDMA, no registration errors), with busbw within ±15% of `B_shim`'s. The risk is a 12.9 runtime loaded beside torch's 12.1. Phase 3 prefers `B_shim` (one runtime) if both pass. |

Library check (2026-11-05, readelf on polaris-login):
- **The plugin** (`libnccl-net.so.0.0.0`) needs `libcudart.so.12` and `libfabric.so.1`. Its RUNPATH
  (`/soft/applications/conda/2024-04-29/mconda3/lib`) no longer exists, so `libcudart.so.12` comes from torch's
  already-loaded copy (12.1.105).
- **libfabric 2.3.1** dlopens:
  - `libcuda.so.1` and `libnvidia-ml.so.1`, both present in `/usr/lib64`;
  - the unversioned `libcudart.so` and `libgdrapi.so`, both absent. Only `libgdrapi.so.2` exists; gdrcopy is optional.

## Phase 3 — DDP over the first passing config (A100, batch 16 per trainer rank, strict fp32)

| # | prediction |
|---|---|
| 8 | **`replicate lean32`** (8 ranks, each solver + DDP rank): s/step 1.99–2.10. That is the 1-GPU sum 1.546 + 0.419 = 1.965 s plus the DDP cost. The no_sync s/step is 1.95–2.02; the DDP cost (synced − no_sync) is 0.00–0.10 s (163 MB of gradients, ≈ 20 ms at 15 GB/s, partly overlapped with backward). 0.48–0.51 rank-steps/s per GPU. |
| 9 | **`decouple lean32`** (2 trainers at local rank 1, one per node, 3 simulators each): <ul><li>free-running s/step 0.42–0.47 (the stream arm's 0.419 + a 2-rank cross-node all_reduce);</li><li>matched s/step 0.50–0.54, solver-bound, from 12 / (3 × 7.73 pairs/s) = 0.517;</li><li>12.0 fresh pairs per step, 0 starved steps;</li><li>simulators 3.22–3.26 ms/step.</li></ul> |
| 10 | **Equal GPU-seconds.** `decouple`'s matched rate per GPU (2 / s_matched / 8) is within ±10% of `replicate lean32`'s per-GPU rate (≈ 0.48 vs ≈ 0.50 rank-steps/s per GPU). Wall-clock per step is ≈ 3.9× shorter for `decouple` (0.52 vs 2.0 s) at a 4× smaller global batch (32 vs 128). |
| 11 | **`replicate ref`**: s/step 4.88–5.05 (job B 1 GPU: 4.887); DDP cost 0.00–0.10 s; ≈ 0.20 rank-steps/s per GPU. |

## What this job cannot decide

- Whether decoupling's change to the freshness mix (pairs from 3 trajectories per trainer instead of 1) or `lean32`'s
  float32 solver changes what the UViT learns. That needs a training comparison and jesswan / Weidong's sign-off.
- H200 numbers, more than 2 nodes, and `OMP_NUM_THREADS` 1 vs 8: there is no A/B here.
- Whether arm A's failure has a second cause besides libcudart.
