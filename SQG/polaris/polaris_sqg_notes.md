# SQG online-training profiling on Polaris (jobs A/B/C)

**Dates:** 2026-11-03 –
**Cluster:** Polaris (ALCF), `debug` queue, 1 node, **one GPU per job** (GPU 0, unbound), NVIDIA A100-SXM4-40GB,
driver 580.65.06.
**Code under test:** the SQG online-training loop (`SQG/o1.2-TrainUViT_m16_tendency.py`). The JAX A-SQG solver is the
vendored `SQG/fixed_ring/A-SQG_data.py` + `sqg.py` (byte-identical to Weidong's Midway `fixed_ring/`, `SHA256SUMS` OK).
The network is the PyTorch `ConvNextUViT`. Also measured: how the two are sequenced. Harnesses `SQG/analysis/profile_{solver,overlap,uvit}.py`,
launched by the `*_polaris.pbs` siblings of the `*_midway.sh` scripts. Task brief: `SQG/polaris/PROFILING_PROMPT.md`.

This is a measurement of the **code**, not of the hardware. No number here is compared with an H100 or any other GPU,
and every finding is phrased as something that would hold on any GPU. No job reads training data either: A and B start from
the solver's own random IC, stepped to AB3 level 2 (`SQG_INIT=random`, §2b of the brief), and C uses random input.

---

## Summary

*(Filled in after jobs A/B/C; see Part III.)*

---

# Part I — Environment and bring-up

## 1. The interpreter and the CUDA stack

`module load conda` **works again**. Every Polaris doc in this repo still says the conda modulefile cannot load: as
of 2026-08 it pinned `cray-hdf5-parallel/1.14.3.5` and `gcc-native/14.2`, neither of which was installed. ALCF edited
`/soft/modulefiles/conda/2025-09-25.lua` on **2026-09-16**. It now depends on `cray-hdf5-parallel/1.14.3.9` and
`gcc-native/14`, which are the versions our hand reconstruction (`makani_sfno/polaris/polaris_makani_env.sh`) had
substituted, and it prepends `/soft/applications/conda/2025-09-25/pe-26.03-shim/` to `LD_LIBRARY_PATH`. On
polaris-login-02 on 2026-11-03, `module load conda` returned 0 and `conda activate base` gave
`/soft/applications/conda/2025-09-25/mconda3/bin/python` (3.12.11). A newer `conda/2026-10-01` also exists, unused here.
The makani jobs have in fact been on the module path since mid-September: their logs print
`MAKANI_ENV_SOURCE = module-conda` (7807790, 7809283; confirmed by the `combine_handoffs` session).

SQG uses conda only to get a Python 3.10 interpreter. Midway's `wdenv` is cp310, and the brief's dry run resolved on
cp310. `SQG/polaris/setup_sqg_env.sh` ran on the login node, with conda and pip single-threaded because of the
256-pid cgroup cap:

```
conda create -p $MEMBER_ROOT/conda-envs/sqg-env python=3.10        # -> Python 3.10.22
pip install -r SQG/polaris/requirements-sqg.txt -c SQG/polaris/requirements-wdenv.txt     # PIP_RC=0, constraints honoured
```

It resolved to wdenv's exact stack: jax / jaxlib / jax-cuda12-plugin / -pjrt 0.4.28, pyqg-jax 0.8.1, torch 2.3.1
(+cu121), torchvision 0.18.1, timm 0.9.12, numpy 1.26.4, netCDF4 1.7.1.post2, h5py 3.14.0; nvidia-cufft-cu12
11.0.2.54, nvidia-cudnn-cu12 8.9.2.26, nvidia-cublas-cu12 12.1.3.1, nvidia-cuda-nvcc-cu12 / nvjitlink 12.9.86. The
env is 5.9 GB, and its full freeze is `$MEMBER_ROOT/conda-envs/sqg-env/sqg-env-freeze.txt`. `$MEMBER_ROOT` is
`/eagle/projects/lighthouse-uchicago/members/mehta5`. The brief's default `/eagle/lighthouse-uchicago/members/$USER`
does not exist for user `rmehta1987`, so every script resolves the member dir as `polaris_env.sh` does: an explicit
`MEMBER_ROOT`, else `members/$USER`, else the writable dir the user owns.

The brief's login-node import check was **not run**. The operator ruled on 2026-09-24 that venv Python must not run on
login nodes (h5py/netCDF4 can hang there, and the pid cap kills sessions), so the probe job does the check instead.

## 2. Probe job 7809334: `SQG_ENV_OK` + `SQG_IMPORT_OK`

`SQG/polaris/sqg_env_probe.pbs` ran on x3001c0s19b0n0. Summary in
`SQG/analysis/results/polaris_probe_7809334/sqg_env_probe.json`.

- **Env:** JAX 0.4.28 sees `cuda:0` (backend `gpu`). Torch 2.3.1+cu121 (cuDNN 8902) sees the A100-SXM4-40GB. A
  4096² complex128 `jnp.fft.fft2` matches `numpy.fft.fft2` to a max relative error of **8.9e-16**. Every harness import
  succeeds.
- **CUDA libraries actually mapped** (§3.2 of the brief). All come from the env's own wheels, under
  `sqg-env/lib/python3.10/site-packages/`: `nvidia/cufft/lib/libcufft.so.11`, `nvidia/cudnn/lib/libcudnn.so.8` (+
  `_ops_infer`), `nvidia/cublas/lib/libcublas{,Lt}.so.12`, `nvidia/cuda_runtime/lib/libcudart.so.12`,
  `nvidia/cuda_cupti/lib/libcupti.so.12`, `nvidia/nvtx/lib/libnvToolsExt.so.1`, plus torchvision's vendored
  `torchvision.libs/libcudart.7ec1eba6.so.12`. Nothing loads from `/soft/compilers/cudatoolkit` or from NVHPC. The
  job's `LD_LIBRARY_PATH` carries only the login-shell defaults (darshan, pals, `hpc_sdk/25.9/compilers/lib`, papi,
  libfabric), and none of those provide a CUDA math library. No module is loaded by the scripts.
- **Patched solver import** (§3.1b): `import_asqg()` with `SQG_TRAINING_ROOT` took 3.3 s.
  `GD.TRAINING_ROOT` is the override (`$MEMBER_ROOT/sqg_stage/AllTrainData`), and its run dirs were created there.
  `GD.CONFIG["solver_sha256"]` equals the sha256 of `SQG/fixed_ring/A-SQG_data.py` on disk.
  `GD.advance(template, 2)` reaches `_ablevel == 2` with a finite field.

The login-node half of the import test (§3.1a, `SQG/polaris/test_training_root_patch.py`, pure text, no jax) printed
`PATCH_TEST_OK`. The patch changes exactly one line, line 76 (`TRAINING_ROOT`). Both harness copies produce the same
text, the patched source compiles, and with the `SQG_*` variables unset the module defaults are Midway's (solver path,
batch 32, buffer 8192).

## 3. What changed in the harnesses (all gated; unset = the Midway code path)

| variable | harness | effect when set |
|---|---|---|
| `SQG_ASQG_PY` | A, B | path of the solver (the vendored copy) |
| `SQG_TRAINING_ROOT` | A, B | exec the solver with only its `TRAINING_ROOT = …` line replaced in memory (`patch_training_root`) |
| `SQG_INIT=random` | A, B | `st0 = GD.advance(template, 2)` instead of `load_checkpoint`; JSON `"init": "random"`, `"checkpoint": null`, `"completed_blocks": null` |
| `SQG_BATCH` | B, C | batch (default 32) |
| `SQG_BUF` | B | host-buffer pairs (default 8192) |
| `PHASE` | A (PBS only) | `bench` / `nsys` / `all` (default) |
| `NUMA=1` | B (PBS only) | the optional `numactl --cpunodebind=3 --membind=3` arm, separate results dir |

One change is additive and not gated, as the brief asks (§3.5): B now times `sample()`, the host gather plus H2D
copy, on its own. It uses a host timer, an NVTX range `sample`, and a third CUDA event between the sample and the
network. The new keys are `sample_host_ms_median`, `sample_gpu_ms_median`, `net_gpu_ms_median` and
`sample_fraction_of_ml_only`. The existing `ml_gpu_ms_median` keeps its meaning, sample plus network, so earlier
JSONs stay comparable (CLAUDE.md #10). The compute number quoted for B is `net_gpu_ms_median`.

---

# Part II — Pre-registered predictions (written before job A; scored in Part III)

Copied from §7 of the brief, unchanged. Each one is about the code, and each will be scored confirmed or refuted
against a cited number.

1. **Host sync (A).** For `ref`, dispatching a jitted 40-step call takes about as long as completing it, i.e. Python is
   blocked. The optimized HLO has a `conditional` inside the `while` body. For `fixed`, dispatch is ≪ completion and
   there is no `conditional`. If `ref` does not block, the `lax.switch` hypothesis is refuted, and the sync must be
   found elsewhere (D2H copies, `cudaStreamSynchronize` in the nsys CUDA-API rows).
2. **Bitwise (A).** After 480 steps, `donate`, `fixed` and `fixed_donate` are bitwise equal to `ref`.
3. **Copies (A).** `donate` removes one ~400 MB device-to-device copy of the AB3 carry per call (HLO `copy` count,
   nsys memcpy rows), and changes ms/step by < 5%.
4. **Sequencing (B).** `serial_ref` ≈ `solver_only_ref` + `ml_only`, and `thread_fixed` ≈ max(solver, ML). Overlap
   efficiency for `ref` is < 20%. If `thread_ref` already overlaps well, the blocking is in o1.2's call order, not in
   pyqg_jax.
5. **The data path is not the cost (B).** The timed `sample()` gather + H2D is < 10% of `ml_only`.
6. **UViT (C).** A few conv/attention kernels dominate, and no single op exceeds 40% of CUDA time. Under strict fp32
   (TF32 off, as in production), `cudnn.benchmark` and `channels_last` each change step time by < 15%.

---

# Part III — Results

*(Job log, then per-job sections, filled in as each job lands.)*

| job | id | queue | node | batch | PASS token | result dir |
|---|---|---|---|---|---|---|
| probe | 7809334 | debug | x3001c0s19b0n0 | — | `SQG_ENV_OK`, `SQG_IMPORT_OK` | `results/polaris_probe_7809334/` |
| C | 7809340 | debug | x3001c0s19b0n0 | 32 | — (**CUDA OOM** in the UViT forward) | — |
| C | 7809341 | debug | x3001c0s19b0n0 | 16 | `UVIT_OK` | `results/polaris_uvit_7809341/` |
| A | 7809356 | debug | x3001c0s19b0n0 | — | `PROFILE_OK bench`, `NSYS_SUMMARY_OK` ×2 (ref, fixed) | `results/polaris_solver_7809356/` |
| B | 7809385 | debug | x3001c0s19b0n0 | 16 (buf 8192) | `OVERLAP_OK` | `results/polaris_overlap_7809385/` |
| C rep 2 | 7809408 | debug | x3001c0s19b0n0 | 16 | `UVIT_OK`, `UVIT_TRACE_OK` | `results/polaris_uvit_7809408/` |
| A rep 2 | 7809412 | debug | x3001c0s19b0n0 | — | `PROFILE_OK bench`, `NSYS_SUMMARY_OK` ×2 (ref, donate) | `results/polaris_solver_7809412/` |

## A — the solver step (random IC at AB3 level 2; 4096² float64)

Nsight Systems: `/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/compilers/bin/nsys`, version 2025.5.1. It is on `PATH`
through the login-shell default `nvidia/25.9` module, so the `/soft/compilers/cudatoolkit` fallback was not
needed. Compile time per variant is 1.9–2.2 s. Python's import of the solver takes 27.6 s, most of it the
import-time forcing setup on the GPU. JAX peak memory is 4.49 GiB.

### A1. What blocks the Python thread

| | `ref` (pyqg_jax AB3, as production) | `fixed` (AB3 level fixed at 2) |
|---|---|---|
| optimized HLO, per 40-step call | 5 fft, **1 conditional**, 1 while, 21 copy, 35 fusion | 5 fft, **0 conditional**, 1 while, 10 copy, 23 fusion |
| dispatch of 12 calls (+ coarse) vs completion | **4.469 s of 4.474 s (0.999)** | **4.163 s of 4.547 s (0.916)** |
| per-call dispatch, ms | 368, then 373 ×11 | **2.8**, then 371–379 ×11 |
| nsys: `cuStreamSynchronize` per solver step | **1.0** (98.8% of CUDA-API time) | **0** |
| nsys: DtoH copies per step | **1.0** | **0** |
| nsys: mean `cuLaunchKernel` | 2.7 µs | **414 µs** (81% of CUDA-API time) |
| nsys: mean `cuMemcpyDtoDAsync` / `cuGraphLaunch` | 4.2 / 9.6 µs | 204 / 188 µs |
| GPU busy / span, ms per step | 9.29 / 9.36 (0.992) | 9.47 / 9.50 (0.996) |

*Sources: `profile_solver_bench.json` (`hlo`, `dispatch`), `nsys_{ref,fixed}_summary.json`,
`nsys_{ref,fixed}_cuda_api_sum.csv`; capture = 5 × 12 × 40 = 2,400 steps per variant.*

There are **two separate reasons the host cannot run ahead of the solver, and the `lax.switch` is only the first.**

1. **`ref` syncs the host once per solver step.** pyqg_jax's `AB3Stepper.apply_updates` branches on the runtime
   `_ablevel` with `lax.switch`. XLA:GPU executes that `conditional` by copying the branch index to the host
   (1 DtoH per step) and blocking on the stream (1 `cuStreamSynchronize` per step, 2,400 calls averaging 9.2 ms,
   i.e. one whole step each). So every 40-step call returns only when its GPU work is done: the first call already
   takes 368 ms. The `while` loop of `lax.scan` does **not** sync: `fixed` keeps the `while` and shows 0 syncs, so
   XLA runs it with a known trip count. The hypothesis's mechanism is confirmed.
2. **Without the switch, the CUDA launch queue limits the host to about one call ahead.** `fixed` has no sync at all,
   and its first call returns in 2.8 ms. A solver step is ~25 launches (17 `cuLaunchKernel` + 3 `cudaLaunchKernel`
   + 2 `cuGraphLaunch` + 3 D2D memcpys), so a 40-step call is ~1,000. Enqueueing them fills the stream's launch queue.
   From the second call on, every launch waits for a free slot: `cuLaunchKernel` averages 414 µs instead of 2.7 µs.
   Each call then takes as long on the host as on the GPU (371–379 ms). This is not a JAX or pyqg_jax sync. It is
   the CUDA runtime pacing a thread that has queued too much work. **The consequence for o1.2's loop is the same as
   a sync:** a single Python thread that dispatches 12 × 40 solver steps cannot get to the ML step until about 11 of
   the 12 calls have finished on the GPU, with or without the `lax.switch`.

### A2. Equivalence and copies

- **Bitwise:** after 480 steps (12 calls), `donate`, `fixed` and `fixed_donate` are all `equal: True` vs `ref`,
  max |Δqh| = 0.0.
- **The loop carry is copied ~5× per solver step, in both variants.** nsys records 12,240 device-to-device memcpys
  of **134.283 MB**, which is exactly one `qh` field (1 × 4096 × 2049 complex128). That is 204 per 40-step call,
  i.e. 5 per step + 4 per call. They take **0.99 ms per step**, **10.7% of the 9.29 ms step**, at 1,378 GB/s
  (read + write). These are XLA's loop-carry copies, not something `donate` touches; the 4 per call are the
  likeliest candidates for what donation removes. `ref`'s conditional adds 3 more tiny D2D copies + 1 DtoH per step,
  which cost under 0.01 ms/step on the GPU (their cost is the host sync in A1).
- **`donate` removes exactly the 400 MB carry copy per call, and nothing inside the loop.** The repeat job 7809412
  captured `ref` and `donate` under nsys. Full-field D2D copies went from 12,240 to 12,060 over the same 60 calls:
  **3 × 134.283 MB = 403 MB fewer per 40-step call**, i.e. the AB3 carry (`qh` and its two update histories). Three
  tiny copies per call also go. The ~5 full-field copies per *step* inside the `while` are untouched (5.03/step,
  0.977 ms/step). The whole-module HLO `copy` count goes 21 → 15 (`ref`) and 10 → 4 (`fixed`). The time does not
  improve: ms/step +0.25% / +0.26% (n = 2), within the predicted 5%. 0.6 ms of copies per 373 ms call is too small
  to see.

### A3. Throughput (n = 2: jobs 7809356 / 7809412; each a median of 3 × 60 calls)

| variant | ms/step | ms/step with `coarse` | solver per optimizer step (480 steps + 12 coarse) |
|---|---|---|---|
| `ref` | 9.2905 / 9.2802 | 9.3175 / 9.3069 | 4.472 / 4.467 s |
| `donate` | 9.3137 / 9.3042 | — | — |
| `fixed` | 9.4521 / 9.4457 | 9.4775 / 9.4712 | 4.549 / 4.546 s |
| `fixed_donate` | 9.4599 / 9.4542 | — | — |

The two runs agree to ≤ 0.11%. The dispatch pattern repeats exactly (`ref` 4.464 s of 4.469 s; `fixed` 4.160 s of
4.544 s), and bitwise equality holds again in the repeat.

`coarse()` (irfft2 + `[::16, ::16]`) is 1.02 ms. **`fixed` is 1.7% slower than `ref` while computing the same
bits.** The kernel time per step is 8.47 vs 8.29 ms. The difference is one fusion: without the switch, XLA fuses
the AB3 update differently, and `loop_add_multiply_fusion_1` takes 3,407 µs vs 3,280 µs. So removing the switch
buys host freedom, not GPU time. The step is memory-bound: XLA's cost analysis gives 68.0 GFLOP and 4.23 GB per
step, i.e. 455 GB/s effective. Its biggest kernels are the one AB3/RHS fusion (~40% of GPU time) and the cuFFT
kernels (`regular_fft_factor` 24%, `vector_fft_c2r` 7.6%), plus a cuBLAS `scal` (7.2%), which is likely FFT
normalisation.

## B — does the solver overlap the UViT step? (batch 16, host buffer 8192, random IC)

One GPU, no DDP, 4 warm-up + 20 timed optimizer steps per arm. Each optimizer step is 480 solver steps (12 × 40 +
coarse) and one UViT training step. Wall time is the median per optimizer step. "ML GPU" is the CUDA-event time on
torch's stream. Overlap efficiency = (solver_only + ml_only − combined) / min(solver_only, ml_only): 1 = the ML step is
fully hidden, 0 = serial.

| arm | wall s | overlap eff. | ML GPU ms (of which `sample()`) | network-only GPU ms |
|---|---|---|---|---|
| `ml_only` | 0.4174 | — | 416.47 (1.22) | 415.27 |
| `solver_only_ref` | 4.4710 | — | — | — |
| `solver_only_fixed` | 4.5494 | — | — | — |
| **`serial_ref`** (o1.2's order and solver) | **4.8868** | **0.004** | 419.74 (1.27) | 418.44 |
| `serial_fixed` | 4.9386 | 0.068 | 773.11 (1.27) | 771.73 |
| `thread_ref` | 4.8169 | 0.171 | 1371.57 (1.42) | 1370.16 |
| `thread_fixed` | 4.8797 | 0.209 | **1910.05** (1.45) | 1908.52 |

*Source: `results/polaris_overlap_7809385/profile_overlap.json`. Peak memory: torch 20.26 GiB, JAX 3.23 GiB.*

What this says about the code:
- **o1.2's order is a pure sum.** `serial_ref` is 4.887 s, and `solver_only_ref` + `ml_only` is 4.888 s. The
  docstring's claim that the solver steps "run alongside" the ML step does not hold. A1 says why: with `ref`,
  `advance()` returns only after all 480 steps have run on the GPU.
- **Removing the host blocking does not produce the overlap either.** In `thread_fixed` nothing on the host
  serialises: the solver is dispatched and waited on in a background thread, and its calls have no sync. It still
  hides only 21% of the ML step, and is 0.15% faster than production. The ML step's own GPU time stretches 4.6×,
  from 416 to 1,910 ms, because its kernels now share the GPU with the solver's. A measured that the solver keeps
  the GPU busy 99.2–99.6% of the time on its own, so there is no idle GPU time for the network to fill. Concurrency
  can only come from two kernels sharing SMs and memory bandwidth, and that bought ≤ 21% here.
- `serial_fixed` sits in between: Python gets back the last ~1 call early (A1, launch queue), so the ML step
  overlaps that last call. The ML GPU time rises 1.9× to 773 ms, and the wall time does not improve (+1.1% vs
  `serial_ref`).
- **The fastest arm is `thread_ref`, 1.4% faster than production** (4.817 vs 4.887 s). `fixed` loses its host-side
  advantage to its 1.7% slower AB3 fusion (A3).
- **The data path is not the cost.** `sample()` (gather of 16 pairs from the pageable host buffer + H2D) is
  1.22 ms on the GPU and 1.18 ms on the host, **0.3% of `ml_only`**. That is about the same in every arm (1.27–1.45
  ms), including when a background thread is dispatching the solver.

## C — the UViT training step (random input; batch 16)

**Batch 32 does not fit.** With the production batch of 32, the UViT training step alone fails with a CUDA OOM on
the 40 GB A100, in the forward pass at the last conv of `ConvNextUViT.forward` (job 7809340). At batch 16 the
step peaks at **20.24 GiB** of torch allocations. The activations of this U-Net at 256² are the memory cost, not
the 40.8 M parameters. Every ML number in these notes is therefore at **batch 16**, and job B uses 16 as well.

| quantity (CUDA events, median of 20) | baseline (strict fp32, `cudnn.benchmark=False`, NCHW) | `cudnn.benchmark=True` | `channels_last` |
|---|---|---|---|
| fwd ms | 137.46 | 136.49 | 133.54 |
| bwd ms | 274.85 | 274.48 | **436.66** |
| clip ms | 0.50 | 0.49 | 0.49 |
| opt ms | 3.10 | 3.13 | 5.51 |
| **total ms** | **415.91** | 414.65 (−0.3%) | **576.22 (+38.5%)** |
| forward max rel. diff vs baseline | — | **0.0** (bitwise) | 9.3e-8 |

*Source: `results/polaris_uvit_7809341/profile_uvit.json`.*

What the code does with this:
- **Backward is two-thirds of the step** (275 of 416 ms). The optimizer (AdamW, 3.1 ms) and gradient clipping
  (0.5 ms) are negligible.
- **`cudnn.benchmark` buys nothing.** The autotuner picks the same algorithms as the heuristic (the forward is
  bitwise identical) and the step moves by −0.3%.
- **`channels_last` makes the step 38.5% slower, and all of the loss is in backward** (437 vs 275 ms; forward
  −3%). Which kernels slow down is not resolved here: only the NCHW baseline was traced.
- **The 7×7 depthwise convs run on PyTorch's native kernels, not cuDNN.** Under strict fp32 NCHW,
  `conv_depthwise2d_grad_weight_kernel` (41/step), `conv_depthwise2d_backward_kernel` (41/step) and
  `conv_depthwise2d_forward_kernel` (40/step) are all in the top 20 by GPU time.
- **Many small copies, and the model code explains them.** `aten::copy_` is called **1,271 times per step**,
  launching ~1,012 `direct_copy_kernel`s per step. There are two sources in `model_UViT.py`, both deliberate:
  - `ConvNextUViT.__init__` sets `padding_mode="circular"` on every padded `Conv2d`, because the domain is
    periodic. Every 3×3 / 7×7 conv therefore runs an `F.pad(..., mode="circular")` first, which is a copy of its
    input, in forward and again in backward.
  - `ConvNeXtBlock.forward` permutes NCHW → NHWC for the LayerNorm + MLP and back.

  Each copy is cheap, but the launch count is a code fact. Removing either source changes what the model computes
  at the boundary (circular → zeros) or needs an equivalence check (fusing the permutes), so it belongs to the
  ranked list, not this task.

**The op shares in `profile_uvit.json` (`top_ops`) are not shares of GPU time.** The harness divides each row's
self CUDA time by the sum over *all* `key_averages()` rows. Those rows mix aten ops, the kernels they launch, and
the `fwd`/`bwd`/`clip`/`opt` `record_function` ranges, so kernel time is counted more than once. The profiler's own
"Self CUDA time total" is 5.70 s for 10 steps that took 4.16 s by CUDA events, and the top "op" is the `fwd` range
itself (14.0%). Prediction 6 ("no single op > 40% of CUDA time") is therefore scored from `summarize_uvit_trace.py`,
which reads only the GPU-side events of the chrome trace. It ran inside the repeat job 7809408.

**Repeat 7809408 (n = 2).** The step is 416.27 ms (fwd 137.54, bwd 275.18), `cudnn.benchmark` 415.10 ms,
`channels_last` 576.45 ms, peak 20.24 GiB. Every number is within 0.1% of 7809341.

**Kernel-only breakdown** (`results/polaris_uvit_7809408/uvit_trace_summary.json`; 10 profiled steps; GPU time
426.8 ms/step under the profiler, busy 98.4% of the span):

| family | % of GPU time | biggest kernels (share, launches/step, mean) |
|---|---|---|
| GEMM (fp32 SGEMM: `Linear` pw1/pw2, attention qkv/proj) | 26.4 | `ampere_sgemm_128x64_tn` 6.5% (28, 988 µs), `_32x128_tn` 4.7%, `_64x32_sliced1x4_nt` 3.1% |
| elementwise | 19.8 | 250-launch vectorized functor 3.8%, GELU backward 2.7%, fill 1.7% (493/step) |
| depthwise 7×7 conv, PyTorch native kernels | 19.2 | `grad_weight` 7.9% (41, 815 µs), `backward` 6.7% (41, 693 µs), `forward` 4.7% (40, 500 µs) |
| LayerNorm | 15.3 | `GammaBetaBackward` **8.3%** (49, 720 µs), forward 3.5%, grad-input 2.4% |
| copies (`direct_copy_kernel`) | 5.2 | 1,012 launches/step at 21 µs |
| cuDNN conv (the dense 3×3 / 1×1 / stride-2 convs) | 4.5 | |
| memcpy / memset | 3.8 | |
| reduce | 3.1 | |
| other | 1.9 | `RowwiseMomentsCUDAKernel` (GroupNorm in `out`), **1 launch, 7.4 ms** |
| optimizer (AdamW, foreach) | 0.9 | |

- **3,562 kernel launches per training step**, 111 distinct kernels. The largest single kernel is 8.3% of GPU time.
- **Attention is not in the top 20.** The four `AttnBlock`s run at the bottleneck resolution, so their cost is
  small. The time goes into the ConvNeXt blocks at the high resolutions: depthwise conv, LayerNorm over few
  channels, the pointwise `Linear` GEMMs, and the elementwise and copy glue between them.
- **LayerNorm's weight/bias gradient (`GammaBetaBackward`) is the single most expensive kernel.** `ConvNeXtBlock`
  normalises over only `in_ch` channels at every pixel of every image, so this kernel reduces ~10⁶ rows into a
  handful of columns.
- **One GroupNorm statistics kernel takes 7.4 ms per step.** This is `RowwiseMomentsCUDAKernel` for
  `GroupNorm(8, n_feat)` in `ConvNextUViT.out`, at full 256² resolution: 1.7% of the step in one kernel.
