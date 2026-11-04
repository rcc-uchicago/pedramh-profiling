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

The hypothesis was that pyqg_jax's AB3 `lax.switch` on the runtime `_ablevel` syncs the host every step. Its
mechanism is **confirmed**. In the production solver (`ref`), each solver step copies the branch index to the host
and blocks on the stream: nsys records one DtoH and one `cuStreamSynchronize` per step. Python is therefore blocked
for the whole of every 40-step jitted call (dispatch 4.469 s of 4.474 s for 12 calls). The `while` loop of
`lax.scan` itself does not sync. Replacing the switch with its level-2 branch (`fixed`) gives bitwise-identical
results after 480 steps and removes every sync. Yet the host is still blocked after the first call. A 40-step call
is ~1,000 kernel launches, which fill CUDA's launch queue, and further launches then wait (414 µs each instead of
2.7 µs). **A single Python thread cannot run more than about one 40-step call ahead of this solver, with or without
the switch.**

Removing the host blocking does not produce the overlap production assumes, either. o1.2's order is a pure sum:
4.887 s per optimizer step = 4.471 s of solver + 0.417 s of UViT. Even when a background thread dispatches the
`fixed` solver, with nothing on the host serialising, only 21% of the ML step is hidden. The network's kernels then
take 4.6× longer, because they share a GPU that the solver alone keeps 99% busy. The best arm, `thread_ref`, is
1.4% faster than production. **Production's sum is GPU contention, not mainly host sync.** The data step
(`sample()`, gather + H2D) is 0.3% of the ML step.

The largest code-level cost the profile exposes is data movement inside the solver step: **~5 full-field (134 MB)
device-to-device copies of the loop carry per solver step**, 0.99 ms of 9.29 ms (10.6%). Buffer donation removes a
different, per-call 403 MB copy and does not touch them. On the UViT side, batch 32 does not fit a 40 GB GPU, so
every ML number here is at batch 16. The step's GPU time is spread over 3,562 kernel launches: GEMM 26%,
elementwise 20%, PyTorch-native depthwise conv 19%, LayerNorm 15%. `cudnn.benchmark` changes nothing, and
`channels_last` is 38.5% slower. Part IV ranks the changes the evidence supports, each with its equivalence gate.
None were implemented in this task.

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
| B rep 2 | 7809433 | debug | x3001c0s19b0n0 | 16 (buf 8192) | `OVERLAP_OK` | `results/polaris_overlap_7809433/` |

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

n = 2: each cell gives job 7809385 / repeat 7809433.

| arm | wall s | overlap eff. | ML GPU ms (of which `sample()`) | network-only GPU ms |
|---|---|---|---|---|
| `ml_only` | 0.4174 / 0.4174 | — | 416.47 (1.22) / 416.52 (1.55) | 415.27 / 414.95 |
| `solver_only_ref` | 4.4710 / 4.4718 | — | — | — |
| `solver_only_fixed` | 4.5494 / 4.5494 | — | — | — |
| **`serial_ref`** (o1.2's order and solver) | **4.8868 / 4.8871** | **0.004 / 0.005** | 419.74 / 419.71 | 418.44 / 418.39 |
| `serial_fixed` | 4.9386 / 4.9390 | 0.068 / 0.067 | 773.11 / 773.52 | 771.73 / 772.19 |
| `thread_ref` | 4.8169 / 4.8167 | 0.171 / 0.174 | 1371.57 / 1371.88 | 1370.16 / 1370.41 |
| `thread_fixed` | 4.8797 / 4.8803 | 0.209 / 0.207 | **1910.05 / 1897.86** | 1908.52 / 1896.37 |

*Sources: `results/polaris_overlap_{7809385,7809433}/profile_overlap.json`. Every wall time agrees to ≤ 0.02%.
Peak memory: torch 20.26 GiB, JAX 3.23 GiB, in both runs.*

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
  1.22 / 1.55 ms on the GPU and 1.18 / 1.51 ms on the host, **0.3–0.4% of `ml_only`**. That is about the same in
  every arm (1.27–1.45 ms), including when a background thread is dispatching the solver.

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

## Scorecard of the pre-registered predictions (Part II)

| # | prediction | verdict | the number |
|---|---|---|---|
| 1 | `ref` blocks Python, with a `conditional` inside the `while` body | **confirmed** | dispatch 4.469 s of 4.474 s; the one `conditional` is in `region_0.116`, the `while` body; 1 DtoH + 1 `cuStreamSynchronize` per step |
| 1 | `fixed`: no `conditional`, and dispatch ≪ completion | **half refuted** | 0 `conditional` and 0 syncs, and the first call returns in 2.8 ms. But over 12 calls dispatch is 4.163 s of 4.547 s (0.92), because the CUDA launch queue fills: `cuLaunchKernel` averages 414 µs vs 2.7 µs |
| 2 | `donate`, `fixed`, `fixed_donate` bitwise == `ref` after 480 steps | **confirmed** (n = 2) | `equal: True`, max \|Δqh\| = 0.0, in both A jobs |
| 3 | `donate` removes one ~400 MB D2D copy per call; ms/step changes < 5% | **confirmed** | −3 × 134.283 MB = −403 MB per call (nsys 7809412); +0.25% / +0.26% |
| 4 | `serial_ref` ≈ solver + ML | **confirmed** | 4.887 s vs 4.471 + 0.417 = 4.888 s |
| 4 | `thread_fixed` ≈ max(solver, ML) | **refuted** | 4.880 s vs max 4.549 s; overlap efficiency 0.21; ML GPU time 416 → 1,910 ms |
| 4 | overlap efficiency for `ref` < 20% | **confirmed** | `serial_ref` 0.004, `thread_ref` 0.171 |
| 5 | `sample()` gather + H2D < 10% of `ml_only` | **confirmed** (n = 2) | 1.22 ms of 416.47 ms = 0.3%; 1.55 of 416.52 = 0.4% |
| 6 | no single op > 40% of CUDA time | **confirmed** | the largest kernel is 8.3% (LayerNorm `GammaBetaBackward`) |
| 6 | a few conv/attention kernels dominate | **refuted** | the time is spread out: GEMM 26%, elementwise 20%, native depthwise conv 19%, LayerNorm 15%; 111 distinct kernels; attention is not in the top 20 |
| 6 | `cudnn.benchmark` changes step time < 15% | **confirmed** | −0.3% (n = 2), forward bitwise identical |
| 6 | `channels_last` changes step time < 15% | **refuted** | **+38.5%** (n = 2), all of it in backward |

The fallback the brief asked for under #1 ("if `ref` does not block, look for the sync elsewhere") was not needed
for `ref`, which does block, exactly as hypothesised. It *was* needed to explain `fixed`, and the cause is not a
sync at all: it is launch-queue back-pressure (A1). Under #4, "if `thread_ref` already overlaps well, the blocking
is in o1.2's call order": `thread_ref` does not overlap well (0.17). The second half of the explanation is GPU
contention, not call order (B).

---

# Part IV — Code changes the evidence supports (ranked; not implemented here)

Shares are of the production optimizer step as measured here (`serial_ref`, batch 16): the solver is 4.47 s of
4.89 s (**91%**), the UViT step 0.42 s (8.5%), `sample()` 0.001 s. Each item names the equivalence check it needs
(DESIGN §4). On the solver side the bar is **bitwise** state equality, the check job A already runs; that is stricter
than §4.1's fp32 tolerance.

1. **Remove the ~5 full-field loop-carry copies per solver step.** They cost 0.98–0.99 ms of 9.29 ms, **10.6% of
   solver time, ~9.7% of the optimizer step**, and are the largest single item the evidence shows. They are XLA
   copy-insertion copies inside the `lax.scan` body over the `AB3State` carry; `donate` does not reach them (A2).
   The first step is diagnosis: map the `copy` instructions of the `while` body (`hlo_ref.txt`, `region_0.116`) to
   carry leaves. The likely suspects are the `_updates` history rotation and the `state.update(...)` rebuilds.
   The fix belongs in o1.2's own step function, the way the harness builds `inner_fixed`, so the hash-pinned
   `A-SQG_data.py` stays untouched. **Gate:** bitwise full-state equality vs `ref` after 480 steps
   (`bitwise_vs_ref_480_steps`), then ms/step.
2. **Fuse the UViT's glue.** Elementwise (19.8%), LayerNorm (15.3%), copies (5.2%) and memcpy (3.8%) are ~44% of
   the UViT step's GPU time, spread over 3,562 launches per step. `torch.compile` of `ConvNeXtBlock` is the standard
   tool for this. The ceiling is a few percent of the optimizer step, since the UViT is 8.5% of it. **Gate:**
   DESIGN §4.1: fixed seed, K = 20-step loss trajectory, strict fp32 kept; compiled-path tolerance ≤ 1e-2, as §4.1
   states for compiled paths. It is not bitwise.
3. **Dispatch the solver from a background thread in o1.2** (B's `thread_ref` arm). Measured **−1.4%** (4.817 vs
   4.887 s). It is small because the solver alone already keeps the GPU 99% busy. **Gate:** bitwise solver state vs
   the serial loop after N optimizer steps, *and* the same sampled (x, y) pairs per step. The flush-previous-chunk-
   then-train order of `OnlinePairs` must be kept, which B's thread arm does. Check with the §4.1 loss trajectory,
   eager fp32 ≤ 1e-5.
4. **Replace pyqg_jax's runtime `lax.switch` with its level-2 branch once `_ablevel == 2`** (the `fixed` variant).
   It is bitwise identical (n = 2) and removes the per-step host sync, DtoH and 3 tiny D2D copies. **On its own it
   does not pay:** one AB3 fusion gets slower (3,407 vs 3,280 µs), so the GPU step is +1.7%. In B it is neutral or
   worse in every arm: `serial_fixed` +1.1%, `thread_fixed` vs `thread_ref` +1.3%. Adopt only together with item 3,
   and only if that fusion regression is fixed. **Gate:** bitwise (passes today). A fresh spin-up must still take
   levels 0 → 1 → 2 through the switch.
5. **Do not adopt `channels_last` (+38.5%) or `cudnn.benchmark` (−0.3%) for the UViT.** Both are measured n = 2.

**Addendum (2026-11-04): what the largest solver kernel computes.** `loop_add_multiply_fusion.1` (3.27 ms, 39.5%
of kernel time; body `%fused_add_multiply` in `hlo_ref.txt`) is the whole AB3 update in one pass. It assembles the
tendency T = −(i k_x (uθ)^ + i k_y (vθ)^ + a k_x-dependent linear term) + F̂, and writes T (the next history entry)
and θ̂ + dt₁T + dt₂T₋₁ + dt₃T₋₂, multiplied by filter × decay. It reads 7 full c128 fields and 1 f64 field and writes
2: ≈1.27 GB in 3.27 ms, **≈390 GB/s**. The cuFFT and `wrapped_multiply` kernels move their data at ≈1.3 TB/s on
the same A100. Three parts of the fusion cannot change the result:
- pyqg's exponential filter is evaluated every step: `sqrt`, compare with 0.65π, and float64 `exp` of
  −filterfac·(k·dx − 0.65π)⁴ for all 8.4 M coefficients. With `filterfac=0.0` it is exactly 1. XLA cannot fold it,
  because exp(−0·x) is not 1 for x = ±inf/NaN.
- F̂ is read as a dense 134 MB complex array every step. 24 of its 8.4 M entries are nonzero.
- The k_x-dependent linear term (`param_10 · sqrt(param_9) · qh`) has the shape of pyqg's background-gradient term
  −i k_x Q_y ψ̂, which is zero for β = 0, U = 0. **Its values are not verified.**

Removing an exact ×1 or +0 leaves every value unchanged, apart from the sign of zeros, which the checkpoints'
`qh_sha256` would see. So a lean step function in o1.2 (the way the harness builds `inner_fixed`; `A-SQG_data.py`
stays hash-pinned) is a candidate for the **bitwise** gate. Estimate, not measured: only the necessary ≈0.94 GB at
FFT-like bandwidth would take ≈0.7–0.8 ms, i.e. up to ≈2.5 ms/step (≈27%) of headroom. Nsight Compute on the fusion
should come first, to see why it sits at ≈390 GB/s (15 operands, c128 arithmetic, f64 sqrt/exp).

Two facts that are not code changes, recorded for the model owners:
- **Batch 32 does not fit a 40 GB GPU for the UViT training step** (job 7809340). Activation checkpointing would
  restore it without changing the math. **Gate:** §4.1 loss trajectory, eager fp32 ≤ 1e-5.
- **The solver dominates and saturates the GPU.** No scheduling change can hide it. Any large gain in online-
  training throughput has to reduce solver GPU work per optimizer step, e.g. STRIDE, solver steps per sample, or
  precision. That changes what is computed, so it is a science decision (jesswan / Weidong), not a code change.

The optional NUMA-bound arm of B (§3.6) was **not run**. B shows the step is GPU-bound: the solver keeps the GPU
99% busy, and the host-side data step is 0.3% of the ML step. A CPU/memory binding can only move host-side time,
and there is almost none to move.

---

# Part V — Per-phase sync audit (job D, 2026-11-04)

Jobs A and B counted syncs for the solver alone, divided over all its steps. Neither attributed the syncs, DtoH
copies and launches of the **whole** optimizer step to the phase that issued them. Job D does, on B's loop.

## Tooling

- **`profile_overlap.py`, `SQG_NVTX=1`:** NVTX ranges around every phase, on the thread that runs it.
  `step` > `advance` > `flush` | `dispatch` > `solver`, `coarse`; `step` > `sample`, `forward`, `backward`
  (zero_grad + backward), `clip`, `optimizer`, `loss_item`, `harness_sync`. In the thread arms the main thread has
  `join` and `flush`, and the worker has `dispatch` > `solver`, `coarse` and `wait`. `harness_sync` is the CUDA-event
  wait that times B's step. o1.2 has no such call. Two flags are added: `--arms` (a subset of the seven) and `--nsys`
  (one arm; the pipeline is drained after the warm-up, then `cudaProfilerStart`/`Stop` bracket the timed steps).
  With none of the three set, the code path and JSON are B's.
- **`summarize_nsys_phases.py`** (stdlib `sqlite3`) reads the capture's SQLite export. It assigns every CUDA API call
  to the stack of harness ranges (NVTX domain 0) open on its own thread. The kernels and copies the call issued follow
  it by `correlationId`. Each sync and DtoH also gets the innermost range of any *other* domain. Under JAX that is
  XLA's per-op `Thunk:#…hlo_op=…#` annotation, so **the HLO op that asked for the sync is named**. API calls are
  classed as sync (`*{Stream,Event,Ctx,Device}Synchronize`: the host blocks), stream wait (GPU-side, non-blocking),
  poll, launch ("slow" > 100 µs on the host = launch-queue back-pressure) or memcpy. Everything is per `step` range.
- **`test_summarize_nsys_phases.py`:** a synthetic two-step capture. It checks nesting, per-thread stacks, the
  XLA-domain issuer, that a `step` range in another domain is not a step, correlation of GPU work, nsys's copy and
  memory-kind enums, slow launches and syncs outside any step. `PHASES_TEST_OK` on polaris-login (2026-11-04).
- **Validated on job A's real capture** (`profile_solver_7809412/nsys_ref.sqlite`, `--step-regex _optstep_`).
  Per 480-step optimizer-equivalent it finds **480.0 `cuStreamSynchronize`** (4,410 of 4,492 ms blocked, 98.2%) and
  **480.0 DtoH of 4 B into pinned memory**. Both are issued inside **`conditional.277.0 (while/body/cond)`**, the
  `lax.switch`. It also finds 11,112 launches (none > 100 µs), 4,416 DtoD copies and 30.6 stream waits. These equal
  A's `nsys stats` totals divided by 5 (2,400 syncs, 55,560 launches, 22,080 DtoD, 153 waits), so the parser
  reproduces A1 and names the op behind it.
- **`profile_phases_polaris.pbs`** runs one nsys capture per arm (default `serial_ref serial_fixed thread_ref`;
  batch 16; 3 warm-up + 6 captured steps), exports SQLite, and runs the parser. PASS = one `PHASES_OK` per arm.

## Predictions (written before job D; batch 16, 12 pairs per step)

1. **`serial_ref`, `solver`:** 480 syncs + 480 DtoH (4 B, pinned) per step, all issued by `conditional.*`, as in A.
   No slow launches.
2. **`serial_ref`, `flush`:** 24 DtoH per step: 2 per pair (`bx[j].copy_`, `by[j].copy_`) × 12 pairs, of
   262,144 B each into **pageable** memory. Each is followed by a host sync, because `Tensor.copy_` into pageable
   CPU memory is synchronous: 24 syncs per step. The JAX → torch hand-off (`torch.from_dlpack`) appears as stream
   waits, not syncs.
3. **ML phases:** `loss_item` has 2 DtoH + 2 syncs per step. `sample` has 2 HtoD from pageable memory and 0 syncs.
   `forward`, `backward`, `clip` and `optimizer` have 0 syncs (torch 2.3 foreach AdamW keeps its step counts on the
   CPU, and `clip_grad_norm_` never reads the norm on the host).
4. **`serial_fixed`:** 0 syncs in `solver`. Slow launches appear in `dispatch` from the second call on (A1's
   back-pressure). The host wait moves into `flush`, whose first DtoH has to wait for the last solver call.
5. **`thread_ref`:** the 480 solver syncs move to the worker thread. The main thread keeps only `flush`,
   `loss_item` and `harness_sync`: ≤ 24 + 2 + 1 syncs per step.

## Results (job 7813332, `debug`, x3002c0s1b1n0, 2026-11-04 17:48–17:56; 6 captured steps per arm)

`PHASES_TEST_OK`, then `PHASES_OK` × 3. **Parser vs nsys:** the per-step API counts equal `cuda_api_sum` ÷ 6 for
every call name in all three arms, and 0 GPU activities are unattributed. Under nsys: `serial_ref` 4.919 s/step
(plain job B: 4.887 s, +0.7%), `serial_fixed` 4.956 s. `thread_ref` has a median of 4.838 s; its mean of 4.269 s is
low because the drained first step has no join. Summaries: `SQG/analysis/results/polaris_phases_7813332/`.

| per optimizer step | `serial_ref` (production) | `serial_fixed` | `thread_ref` |
|---|---|---|---|
| `solver`: syncs / DtoH (4 B, pinned) | **480 / 480**, 4,412 ms blocked, issuer `conditional.277.0` | 0 / 0 | 480 / 480 on the worker threads, ≈4,478 ms blocked |
| `solver`: launches (slow > 100 µs) | 11,052 (0) | 10,572 (**9,619**; 3,838 ms inside launch calls) | 11,052 on the workers |
| `flush`: syncs / pageable DtoH / stream waits | 24 / 24 (262,144 B each) / 12, **0.08 ms** blocked | same, 0.08 ms | same, 0.09 ms |
| `sample`: pageable HtoD | 2 (8.39 MB), 0 syncs | same | same |
| `forward`, `backward`, `clip`, `optimizer`: syncs | 0 | 0 | 0 |
| `loss_item`: syncs / DtoH, host wait | 2 / 2, 132 ms | 2 / 2, 132 ms | 2 / 2, 482 ms |
| `harness_sync` | 1 | 1 | 1 |
| **total syncs / DtoH / DtoD** | **507 / 506 / 4,723** | 27 / 26 / 3,283 | 507 / 506 / 4,723 |

What the code does:
- **The `lax.switch` is the only host sync that costs time.** In the production order, 480 of the 507 syncs per
  step come from one HLO op. They block 4,412 ms of the 4,920 ms step. The other 27 are cheap. The 24 `flush`
  syncs (`Tensor.copy_` into pageable memory) block 0.08 ms in total, because the solver has already finished.
  `loss_item` waits 132 ms, but for the network's own GPU work, whose result it needs. The data path (`flush` +
  `sample`) costs ≈5 ms of host time per step, and the network phases issue no sync at all.
- **Without the switch, the host wait moves into the launch calls, not into `flush`.** 91% of the solver's
  launches take > 100 µs (from the second call on, A1), for 3.84 s/step inside `cuLaunchKernel`. `flush` stays at
  0.08 ms: `loss_item` and `harness_sync` have already waited for a GPU that ran the last solver call alongside
  the network.
- **The JAX → torch hand-off is a GPU-side stream wait** (`cuStreamWaitEvent`, 12 per step in `flush`), not a host
  sync.
- **The backward kernels are launched from a thread without harness ranges**: 2,478 launches/step, 0 syncs, while
  the main thread waits in `backward` (256 ms). This fits PyTorch's autograd engine running CUDA backward work on
  its own device thread (an inference, not checked in the trace).
- `serial_fixed` has 1,440 fewer DtoD copies per step, i.e. 3 per solver step: the conditional's small copies of
  A2, now counted in the full loop.

**Scorecard:** 1 confirmed; 2 confirmed; 3 confirmed; 4 **half refuted** (0 solver syncs and the slow launches are
confirmed, but the host wait stays in the launch calls, not in `flush`); 5 confirmed.

**Consequence for Part IV:** the ranking is unchanged. The host side holds nothing to fix beyond item 4 (the
switch), which does not pay on its own (A3, B). The remaining target is the solver's GPU work (the addendum above).

---

# Part VI — The solver's GPU work: float32 (job E) and Nsight Compute on the AB3 fusion (job F)

**Job E** (`profile_solver_fp32.py`, `profile_solver_fp32_polaris.pbs`) times the production step in float32 against
float64 on one node. **Timing only**: a float32 solver computes something else. The f32 step is `GD.inner`'s four
lines with `sqg.aux(GD.p._replace(f64=False))`, `F0_hat` as complex64 and `decay` as float32. It reports only if the
builder is bitwise equal to `GD.inner` in float64, and only if neither the f32 state nor the optimized HLO (fused
bodies included) holds a full-size f64/c128 field. JAX runs with x64 enabled, so a stray constant would silently
time float64 again.

**Job F** (`profile_ab3_ncu.py`, `profile_ab3_ncu_polaris.pbs`, `summarize_ncu.py` + `test_summarize_ncu.py`; written
by a subagent, reviewed here) runs one `ncu --set full` pass per kernel: the AB3 fusion `loop_add_multiply_fusion_1`,
and for contrast `wrapped_multiply_1`, `loop_multiply_fusion_9` and one cuFFT `regular_fft_factor`. Each pass
profiles 3 steady-state launches at production clocks (`--clock-control none`). The parser reads the raw-page CSVs;
`NCU_TEST_OK` on polaris-login (system python3).

## Predictions (written before jobs E and F)

1. **E, checks:** both pass (builder bitwise; 0 full-size f64/c128 arrays in the f32 HLO).
2. **E, speed:** float64 / float32 ms per step is between **1.6 and 2.0**. The step is bandwidth-bound and float32
   halves the bytes; it stays below 2 if the AB3 fusion's inefficiency does not scale with the bytes.
3. **E, cost model:** XLA's bytes accessed per step roughly halve (4.23 → ≈2.1 GB). The AB3 fusion stays the largest
   kernel in the f32 nsys breakdown.
4. **F, bandwidth:** the AB3 fusion reaches ≤ 30% of peak DRAM throughput. `wrapped_multiply_1` and
   `regular_fft_factor` reach ≥ 75%.
5. **F, not compute-bound:** the AB3 fusion's SM throughput is < 40% of peak and its FP64 pipe < 30%.
6. **F, cause:** latency, not bandwidth. Achieved occupancy is low, from registers (≥ 64 per thread, theoretical
   occupancy ≤ 50%) and/or local-memory spills, and the top stall reason is waiting on memory (long scoreboard).

## Results

### Job E (7813816, `debug`, x3002c0s1b1n0, 2026-11-04 19:31–19:37)

`FP32_OK bench` and `NSYS_SUMMARY_OK` × 2. Checks: the builder is bitwise equal to `GD.inner` (40 steps); the f32 state
is `complex64`; the f32 HLO has 0 full-size f64/c128 arrays. Summaries: `SQG/analysis/results/polaris_fp32_7813816/`.

| | float64 (production) | float32 |
|---|---|---|
| ms per solver step | 9.2954 | **3.6105** (f64/f32 = **2.575**) |
| solver per optimizer step (480 steps + 12 coarse) | 4.474 s | **1.741 s** |
| XLA bytes accessed per step | 4.23 GB | 2.01 GB |
| HLO per 40-step call (fft / conditional / copy / fusion) | 5 / 1 / 21 / 35 | 5 / 1 / 21 / 31 |
| nsys kernels per step; GPU busy | 33.1; 9.29 of 9.37 ms | 24.1; 3.60 of 3.67 ms |

Per-kernel time per solver step, from nsys:

| kernel | f64 | f32 | ratio |
|---|---|---|---|
| AB3 fusion `loop_add_multiply_fusion_1` | 3,288 µs | **650 µs** | **5.1×** |
| cuFFT, all kernels | ≈3.06 ms (10 `regular_fft_factor` + c2r + r2c) | ≈1.5 ms (5 `regular_fft_factor<4096>` + `vector_fft_symm_c2r` + r2c) | ≈2× |
| û, v̂ build (`loop_multiply_fusion[_9]`) | 642 µs | 290 µs | 2.2× |
| u·θ, v·θ (`wrapped_multiply_1` × 2) | 616 µs | 310 µs | 2.0× |
| inverse-FFT 1/N² (`scal_kernel_val` × 3) | 594 µs | 314 µs | 1.9× |

What it says:
- **Every kernel scales with its bytes (≈2×) except the AB3 fusion, which is 5.1× faster in float32.** In float32
  it moves half of ≈1.27 GB in 650 µs, i.e. ≈1 TB/s, close to the cuFFT kernels. In float64 it runs at ≈390 GB/s.
  So the float64 fusion carries ≈2 ms per step of float64-specific cost, beyond its data.
- Both fusions still evaluate the switched-off filter: `sqrt` × 2 and `exponential` × 1 in f64, `sqrt` × 3 and
  `exponential` × 1 in f32. On an A100, float32 square roots and exponentials have special-function hardware;
  float64 ones are software sequences. That, or register pressure from 15 complex128 operands, are the candidates.
  Job F's FP64-pipe, occupancy and stall numbers separate them.
- In float32 the FFTs become the largest family (≈42%) and run near bandwidth.

**Scorecard (Part VI):** 1 confirmed (both checks pass). 2 **refuted**: f64/f32 = 2.575, above the predicted
1.6–2.0, because the AB3 fusion gains 5.1×. 3 half confirmed: bytes halve (4.23 → 2.01 GB), but the AB3 fusion
is no longer the largest kernel family in f32 (the cuFFT `regular_fft_factor` is: 31.8% vs 21.0%).

### Job F (7814044, submitted 2026-11-04)

*(to be filled in when it lands)*
