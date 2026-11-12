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

**Addendum (2026-11-04): where item 1's ≈5 full-field copies per step come from.** Each 134,283,264 B
device-to-device copy in job A's capture (7809412, `nsys_ref.sqlite`) was attributed to the XLA thunk range open
when it was issued (domain-1 NVTX, as `summarize_nsys_phases.py` does for syncs). The result is 5.000 per step:
- **2.000 from `call.14`** (`%command_buffer.2`, the AB3 update graph): `wrapped_copy.22` writes the old T₋₁ into
  the T₋₂ carry slot and `wrapped_copy.23` writes the new tendency into the T₋₁ slot. This is pyqg_jax's history
  rotation, `new_updates = (updates, updates_p)` in `AB3Stepper.apply_updates` (steppers.py:501). `lax.scan` keeps one
  buffer per carry slot, so the shift is done by copying.
- **1.000 each from `fft.0.0`, `fft.1.0`, `fft.3.0`**, the three inverse (C2R) FFTs of û, θ̂ and v̂: the FFT thunk
  copies its input before running. The two forward (R2C) FFTs issue none. This is consistent with cuFFT's
  multi-dimensional C2R transforms overwriting their input.
- Per call: `call.12` copies the 3 carry fields once on loop entry (what donation removes), and `coarse()`'s inverse
  FFT copies once. Also 4 tiny `wrapped_copy.12–15` per step from the conditional.

At ≈0.195 ms per copy, the rotation is ≈0.39 ms/step and is removable bitwise (unroll the scan by 3 so the slots
change roles, or a ring index). The û and v̂ FFT copies (≈0.39 ms) protect temporaries that are dead after the
FFT; avoiding them needs an FFT call that may overwrite its input (custom call). θ̂'s copy protects live state and
stays. So item 1 is ≈0.4 ms easily and ≈0.8 ms at most, not the ≈1 ms first estimated.

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

### Job F (7814044, `debug`, x3005c0s13b0n0, 2026-11-04 20:09–20:12; ncu 2025.3.0, `--set full`, production clocks)

`NCU_TEST_OK`, `NCU_HARNESS_OK` × 4, `NCU_SUMMARY_OK`. 3 steady-state launches per kernel. Summaries:
`SQG/analysis/results/polaris_ab3_ncu_7814044/`.

| kernel | µs | DRAM read / write | GB/s (% of 1,555 peak) | SM % | FP64 pipe % | regs; occupancy th/ach | L1 / L2 hit | grid × block |
|---|---|---|---|---|---|---|---|---|
| AB3 fusion `loop_add_multiply_fusion_1` | 3,275 | **3.99 GB / 0.50 GB** | 1,371 (**88%**) | 6 | 6 | 58; 50/47 | 10% / 44% | 16,392 × 128 |
| û/v̂ build `loop_multiply_fusion_9` | 635 | 0.32 / 0.35 GB | 1,048 (67%) | 7 | 7 | 42; 62.5/56 | 33% / 77% | 16,392 × 128 |
| u·θ `wrapped_multiply_1` | 293 | 0.27 / 0.12 GB | 1,340 (86%) | 2 | 1 | 26; 100/81 | 62% / 54% | 1,728 × 128 |
| cuFFT `regular_fft_factor<64>` | 187 | 0.13 / 0.12 GB | 1,342 (86%) | 26 | 26 | 60; 50/47 | 1% / 62% | 4,098 × 256 |

What it says:
- **The AB3 fusion is DRAM-bound at 88% of peak, but it moves ≈4× the data it needs.** It reads 3.99 GB for ≈1.0 GB of
  distinct inputs (7 c128 fields + 1 f64) and writes 0.50 GB for 0.27 GB of outputs. 16,392 × 128 threads over
  8.39 M elements is 4 complex128 elements per thread (64 B per operand), and with ~9 operands per thread the data
  in flight overruns the 40 MB L2 (L1 hit 10%, L2 hit 44%). This is the over-fetch the seeds' analysis predicted
  (round 1).
- **ncu's own rules name the access pattern** (`ncu_ab3_details.txt`): "uncoalesced global accesses resulting in a
  total of 52454400 excessive sectors (52% of the total 101308512 sectors)" (est. speedup 51.6%). Loads use "only 15.2
  of the 32 bytes transmitted per sector" (89.8% of L2-missed sectors) and stores 16.0 of 32 — "possibly caused by a
  stride between threads". With 4 consecutive complex128 per thread, one warp-wide load touches 32 addresses 64 B
  apart: each 32 B sector delivers the 16 B of one element, and its other half (the thread's next element) is
  evicted before reuse at L1 10% / L2 44% hit. Our estimate: ≈600 B in flight per thread × ≈960 resident threads per
  SM × 108 SMs ≈ 63 MB > the 40 MB L2. In f32 the same is ≈31 MB and fits, consistent with job E's 5.1×.
- **No flag in this jaxlib (0.4.28)** sets XLA's per-thread unroll: a byte search of `xla_extension.so` and
  `xla_cuda_plugin.so` finds no `xla_gpu_*unroll*` / `*vectoriz*` option, only compiler-internal symbols. A fix has to
  come from how the computation is written: one element per thread (round 1's `lean_u1` cos(±0) trigger, unverified),
  fewer operands per kernel (`lean`, `lean_split`), or no separate AB3 kernel at all (callbacks, newer CUDA).
- **Not FP64-bound:** SM 6%, FP64 pipe 6%. The no-op filter's f64 sqrt/exp, suspected after job E, is not the cost.
  float32 gained 5.1× because it halves the bytes per element and the in-flight working set, so it stops
  thrashing.
- The u·θ multiply and the cuFFT passes read and write the ideal bytes at 86% of peak: they are done. The û/v̂ build
  over-fetches ≈1.4×.

**Scorecard (Part VI):** 4 **half refuted** (the AB3 fusion is at 88% of DRAM peak, not ≤ 30%: busy, just over-fetching;
`wrapped_multiply_1` 86% and `regular_fft_factor` 86% ≥ 75% confirmed). 5 confirmed (SM 6% < 40%, FP64 6% < 30%).
6 **refuted**: 58 registers (< 64), no spills, theoretical occupancy 50%. The cause is DRAM over-fetch, not latency.

---

# Part VII — FFT libraries and pseudo-spectral techniques from other fields (literature, 2026-11-04)

Our float64 step (job E nsys): cuFFT 3.06 ms, inverse-FFT input copies 0.59, history-rotation copies 0.39, AB3 fusion
3.29, û/v̂ build 0.64, u·θ/v·θ products 0.62, inverse-FFT 1/N² `scal` 0.59. The FFT itself is about a third.

- **XLA's fusion barrier at cuFFT is a known limit, and others removed it.** gyaradax (plasma gyrokinetics in
  JAX, arXiv 2604.06085, read in full) says: "the FFTs are dispatched as opaque calls to cuFFT, which acts as a hard
  fusion barrier". They replaced the pipeline with 3 cuFFT launches using LTO callbacks (`cufftXtSetJITCallback`)
  behind the XLA FFI:
  - inverse-FFT load callback: spectral derivative, Bessel J₀, scaling;
  - forward-FFT load callback: the Poisson bracket in flight, with no gradient arrays in HBM;
  - forward-FFT store callback: writes only the kept modes, skipping the 59% dealiased.
  They also packed two real inverse FFTs into one C2C ("Z2Z packing", 4 → 2), which needed a Hermitian
  symmetrisation fix. Table 2 (B300, steps/s): JAX DP 12.49, JAX mixed 30.89, CUDA DP 16.6, CUDA mixed 60.54, i.e.
  CUDA/JAX = 1.33× in DP and 4.85× (mixed CUDA vs DP JAX). Not bitwise; validated statistically, as a chaotic
  solver must be.
- **pyspectral** (GPU pseudo-spectral NS, CuPy + hand-written CUDA) reports 99% of an "FFT roofline": every FFT at
  standalone cuFFT speed, plus the minimum pointwise bytes. It is 2.5× faster than JAX-CFD at 8192², fp32. Its
  techniques: fused kernels, writing the next FFT's input directly, in-place plans, a 1-D integrating-factor table,
  CUDA graphs, R2C. Its Basdevant 4-FFT trick is for 2-D NS vorticity. It does not carry over to SQG as far as we
  can see, because θ is not a local derivative of the velocity (θ̂ = |k|ψ̂), so 5 FFTs stay.
- **cuFFT docs** (§2.4): "Out-of-place complex-to-real FFT will always overwrite input buffer". That is the source of
  our 3 inverse-FFT copies. LTO callbacks need CUDA ≥ 12.6u2 and nvJitLink/NVRTC, support all types and precisions
  (§2.9), and "may be called more than once per element for complex-to-real". C2C inverse transforms (gyaradax's
  packing) do not overwrite their input.
- **VkFFT** (MIT; CUDA backend; FP64; R2C/C2R; native zero-padding; convolutions). Independent A100 FP64 C2C tests
  (TurboFFT, arXiv 2405.02520) find it ~11% slower than cuFFT on average (+3% to −51%). pyvkfft reports it faster than
  cuFFT for 2-D N > 1024 on a V100 but not on an A40. No JAX binding. It would need our own 4096² Z2D/D2Z benchmark; it
  is not a clear win.
- **nvmath-python**: cuFFT with prolog/epilog written as Python functions compiled to LTO-IR. Operands from NumPy,
  CuPy and PyTorch (JAX not stated; DLPack via CuPy). Also exposes cuFFTDx device FFTs. A prototyping route for
  callbacks without C++.
- **Pruning / implicit dealiasing.** Bowman & Roberts' implicit (and Murasko & Bowman's hybrid) dealiasing is
  ≈2× vs explicit padding in 2-D/3-D, but CPU-only (FFTW++); a 2026 paper calls a GPU version "a priority". Skipping
  all-zero rows on the GPU is mixed: SPECULA reports −20–40% (2-D), gNUFFTW little benefit. In a 4096² rfft half-plane
  with |k| ≤ 1365, about 35% of modes are kept, so the store-skip part comes with callbacks.

**Implication (estimates, to be measured).** With every pointwise op fused into the FFT loads and stores, the step
approaches its FFT roofline: ≈3.1–3.5 ms in float64 (now 9.3) and ≈1.5–1.8 ms in float32 (now 3.6). That route is not
bitwise (operation order, FMA contraction), so it needs gyaradax-style statistical validation. Mixed precision
(float32 FFTs and products, float64 state and AB3), as gyaradax does, sits between float64 and float32.

---

# Part VIII — Solver rewrites (rounds 1 and 1b) and the first 2-node run (jobs 7814300, 7815096; 2026-11-04/05)

Round 1 times six rewrites of the solver step, each against the production step from the same state. Two Opus seeds
wrote them, and a Fable principal scientist reviewed them. Each variant's tolerance was fixed before the run. User
policy (2026-11-04) is that bitwise equality is reported but not required. Round 1b reruns seed 2's three variants
under corrected gates, also set before the rerun. Phase 2 of the same job is SQG's first run across nodes. All code
is in `SQG/analysis/opt/`.

| variant | seed | precision | what changes vs the production step |
|---|---|---|---|
| `lean` | 1 | f64 | drops the exact ×1 / +0 terms (switched-off filter, β, U, rek); 3-slot AB3 ring (no history-rotation copies) |
| `lean_u1` | 1 | f64 | `lean` + a cos(±0) factor (exactly 1) meant to switch XLA's loop emitter to 1 element per thread |
| `lean_split` | 1 | f64 | `lean` with the AB3 update split into two smaller fusions |
| `lean64` | 2 | f64 | u, v from one forward C2C FFT of the Hermitian-mirrored plane (`build_w`) instead of two C2R; 2-slot ring |
| `mixed` | 2 | f32 nonlinear term, f64 state | `lean64` with the 4 FFTs and the products in float32 |
| `lean32` | 2 | f32 | `lean64` entirely in float32 inside each 40-step call (state cast at entry and exit) |

## Predictions

- **Round 1:** `SQG/analysis/opt/PREDICTIONS_round1.md`, written by the principal scientist before 7814300.
- **Round 1b + 2-node smoke:** `SQG/analysis/opt/PREDICTIONS_round1b.md`, written by the main session before
  7815096. Round 1's numbers informed its phase-1 items.

Neither file was edited after its job.

## Results

### Round 1 (7814300, `debug`, x3003c0s7b1n0, 2026-11-04 23:20–23:36; one A100)

The run ended `OPT_ROUND1_DONE failures=0` and `OPT_RANK_OK`, with `WARN OPT_RANK_NO_PASSING_VARIANT`.
- **Timing:** ms per solver step is the median of 3 × 60 jitted 40-step calls from the random start state. The ref
  is 9.290 (seed 1) / 9.291 (seed 2); a ref re-run at the end drifted 0.0%.
- **Error:** the 1-step error is the max over fields of the relative L2 vs the production step. There are two start
  states: a synthetic one at CFL 0.3 and the solver's random one.
- Summaries: `SQG/analysis/results/polaris_opt_round1_7814300/`.

| variant | 1-step error, synthetic / random | fields after 480 steps | ms/step | speedup | gate | failed on |
|---|---|---|---|---|---|---|
| `lean` | 0 / 0 | **bitwise** | 8.002 | 1.161× | FAIL | `t` |
| `lean_u1` | 0 / 0 | **bitwise** | **6.402** | **1.451×** | FAIL | `t` |
| `lean_split` | 0 / 0 | **bitwise** | 7.066 | 1.315× | FAIL | `t` |
| `lean64` | 2.90e-14 / 2.41e-22 | 3.9e-14 (synthetic), not bitwise | 7.073 | 1.314× | FAIL | `t` |
| `mixed` | 2.30e-5 / 1.75e-13 | — | 4.684 | 1.984× | FAIL | `t`; 2.30e-5 > 1e-5 |
| `lean32` | 2.30e-5 / 6.84e-8 | — | 3.205 | 2.899× | FAIL | `t`; 2.30e-5 > 1e-5 |
| *original step in float32* (report only) | 2.48e-5 / 6.84e-8 | — | — | — | — | — |

What it says:
- **Both failure causes were in the gates, not the variants; the verdicts stand as recorded.**
  - **(1) The `t` check.** Every variant failed the exact-`t` check. `t` is a float32 time counter that the step
    never reads; `tc` and `_ablevel` were exact.
  - **(2) The 1e-5 float32 limit.** It sits below float32's own error: the unmodified step run in float32 scores
    2.48e-5 from the same synthetic state.
- **Seed 1's dropped terms are exact no-ops.** All three seed-1 variants are bitwise in every field after 480 steps.
- **`lean_u1` is the fastest float64 variant.** The cos(±0) factor alone takes `lean` from 8.00 to 6.40 ms. That the
  saving comes from the per-thread layout is not verified (HLO / ncu not checked).
- **`lean64`'s C2C trick saves less than planned.** Seed 2's micro-table has one complex128 C2C at 1.198 ms and one
  C2R at 1.014 ms (the C2R time includes its input copy and the 1/N² scaling). Replacing two C2R with one C2C
  therefore saves ≈0.83 ms, not the ≈1.2 predicted. `lean64` also lands above the 6.3 ms marker that the predictions
  named for "its AB3 fusion still thrashes".

### Round 1b (7815096 phase 1, `debug`, x3001c0s1b0n0 GPU 0, 2026-11-05 03:39–03:47)

Tokens: `SEED2_BUILDW_TEST_OK`, `SEED2_F32BASE`, `SEED2_GATE_OK` × 3, `SEED2_BENCH` × 3, `SEED2_OPTSTEP`,
`SEED2_DONE` (wall 480 s), `OPT_RANK_OK`.
- **`build_w` check.** `build_w` matches the C2R path to 1.07e-15. Wrong-mirror-sign and no-row-flip mutants score
  0.98 / 0.905, so the test can fail.
- **Gates.**
  - `lean64`: ≤ 1e-12.
  - float32 variants: ≤ min(1e-4, 2 × the original step's float32 error from the same state, same run), which is
    1.4e-7 (random) / 5.0e-5 (synthetic).
  - all: `tc` / `_ablevel` exact and |t − t_ref| ≤ 1e-6 relative.
  - `t` now advances by a run-time `dt`.
- **The 4,000-step columns are ungated** (synthetic start). The ref's own SPE drift over those steps is −1.37e-2.
- Summaries: `SQG/analysis/results/polaris_opt_round1b_ms_7815096/`.

| variant | 1-step synthetic (gate) | 1-step random (gate) | `t` exact | coarse rel diff @ 4,000 | ΔSPE @ 4,000 | ms/step | speedup | optimizer step (ref 4.468 s) |
|---|---|---|---|---|---|---|---|---|
| `lean64` | 2.90e-14 (1e-12) | 2.41e-22 (1e-12) | yes | **0** | **0** | 7.051 | 1.318× | 3.396 s |
| `mixed` | 2.30e-5 (5.0e-5) | 1.75e-13 (1.4e-7) | yes | 6.37e-6 | −2.59e-8 | 4.714 | 1.971× | 2.278 s |
| `lean32` | 2.30e-5 (5.0e-5) | 6.84e-8 (1.4e-7) | yes | 4.36e-4 | −8.36e-5 | **3.189** | **2.913×** | **1.546 s (2.891×)** |
| *original in float32* | 2.48e-5 | 6.84e-8 | yes | 4.36e-4 | −8.36e-5 | — | — | — |

The per-|k|-band errors are report-only. The worst band is 1024–2048 for all three: `lean64` 2.10e-14,
`mixed` / `lean32` 1.89e-5.

What it says:
- **All three pass their pre-set gates.** `t` is now exact for all three (and for the float32 original). That fits
  XLA folding a compile-time `dt` in round 1. It is not proven, because round 1's HLO was not inspected for it.
- **`lean64`'s training frames are bitwise equal to production's.** After 4,000 steps its 256² frames match the
  reference exactly. Its ≤ 1e-13 field differences are below the frames' float32 rounding (`coarse()` casts to
  float32, as o1.2 stores frames), and its SPE difference is 0.
- **`lean32` is the original step run in float32, and adds nothing measurable.**
  - 1-step: 0.93× (synthetic) and 1.00× (random) the original-in-float32 error.
  - 4,000 steps: its coarse difference (4.357e-4) and SPE difference (−8.359e-5) equal the original-in-float32's
    (4.355e-4, −8.356e-5) to 3 digits.
  - So the restructuring is not the issue. Float32 itself is, and that is a science decision.
- **`mixed` sits close to float64.** It keeps the state in float64. Its 1-step synthetic error is float32-sized
  (2.30e-5, from the nonlinear term), but its 4,000-step coarse difference is 70× smaller than `lean32`'s (6.37e-6)
  and its ΔSPE is 3,000× smaller (−2.6e-8).
- **Timings reproduce round 1** to within 0.7%.

### Phase 2: 8 GPUs on 2 nodes (7815096, x3001c0s1b0n0 + x3001c0s1b1n0, 03:47–04:01)

**Setup.**
- **Launch:** `mpiexec --ppn 4 --cpu-bind depth -d 8` through `multisim/rank_env.sh`, which gives each rank one GPU
  (`CUDA_VISIBLE_DEVICES` = local rank). Rendezvous at the first node; the port is unique per job and per arm.
- **Stream transport:** gloo over `hsn0`.
- **Results:** 4 of 5 arms pass (`MULTISIM_OK` × 4). The NCCL probe failed (rc 124), so the job ended
  `OPT_R1B_MS_DONE failures=1`.
- **`NCCL_CXI_OK providers=8` is literal.** The line was printed and is true as defined (the cxi provider was
  selected on every rank). It does not mean the all_reduce ran; the probe's failure was counted separately.

| independent arm | ranks | ms/step, min–max over ranks | 1-GPU | scaling efficiency |
|---|---|---|---|---|
| `ref` | 8 simulators | 9.287–9.299 | 9.291 | 0.9999 |
| `lean32` | 8 simulators | 3.195–3.200 | 3.205 | 1.0024 |

**Stream arms.** Rank 0 trains o1.2's ConvNextUViT (40.8 M parameters; batch 16; strict fp32; AdamW; clip 1.0;
`loss.item()` every step). Its input is a 16,384-pair in-RAM replay buffer. Ranks 1–7 integrate their own
trajectories in o1.2's pattern (12 × 40 steps + `coarse`). Each sends 12 pairs per 6.3 MB message as soon as it has
them.

| | `stream ref` | `stream lean32` |
|---|---|---|
| simulator ms/step (7 ranks) | 9.321–9.332 | 3.229–3.237 |
| simulator period per 12 pairs (median) | 4.476 s | 1.552 s |
| pairs/s into the trainer: printed / from the period × 7 | 19.38 / **18.77** | 54.48 / **54.12** |
| MB/s into the trainer | 10.2 | 28.6 |
| gloo send per message, median same node / other node (max) | 0.97 / 1.52 ms (122 ms) | 1.04 / 1.27 ms (13 ms) |
| trainer s/step (job C, alone: 0.417) | 0.419 | 0.419 |
| **fresh pairs per trainer step** (o1.2: 12) | **8.14: solver-bound** | **22.59: trainer-bound** |
| wait for the first batch | 44.8 s | 15.5 s |
| written to disk | nothing but logs/JSON (0.46 MB for the whole job, HLO text included) | same |

The printed `pairs_per_s` is (messages − 1) / (last − first arrival). With 7 senders × 25 messages it overstates the
rate by ≈3.6% (ref), and by ≈1% with 88 messages (`lean32`). The second number is 7 × 12 / period.

**NCCL probe** (`multisim_ncclprobe_ref.log`, `NCCL_DEBUG=INFO`, INIT,NET; 64 MB all_reduce):
- **Stack:** torch 2.3.1's bundled NCCL 2.20.5+cuda12.4 (`cudaDriverVersion 13000`) with aws-ofi-nccl v1.6.0 and
  libfabric 2.3.1, using the combine_handoffs env block (`NCCL_NET_GDR_LEVEL=PHB`, `NCCL_PROTO=Simple`). No arm ran
  without `NCCL_PROTO=Simple`.
- **What loaded:** `Loaded net plugin AWS Libfabric (v6)` and `Selected Provider is cxi (found 2 nics)` on 8/8 ranks.
- **Intra-node:** all 24 channel lines are `via P2P/CUMEM/read`; 0 use SHM.
- **GDR:** under PHB, GDR is enabled only for GPUs 46000 / c7000 (distance 6 to their NIC), i.e. local ranks 1 and 3.
  The ring puts every inter-node hop on those GPUs, and all 16 inter-node lines read `via NET/AWS Libfabric/{0,1}/GDRDMA`.
- **The failure, on all 4 network ranks:**
  - `cxil_map: write error`, then
  - `NET/OFI Unable to register memory (type = 2) for device N. RC: -14, Error: Bad address`, then
  - a proxy `Connect` failure, then
  - `DistBackendError: unhandled system error`.
- **The other 4 ranks** hung to the 360 s timeout.
- **What never happened:** `Connected all rings` was never printed, and there is no bus-bandwidth number.
- **Leading hypothesis (from combine_handoffs, 2026-11-05; file facts checked here, mechanism untested): libfabric
  cannot load the CUDA runtime in sqg-env.**
  - `/opt/cray/libfabric/2.3.1/lib64/libfabric.so.1.29.1` dlopens the unversioned name `libcudart.so`. Its strings
    include `libcudart.so` and `Failed to dlopen libcudart.so`.
  - sqg-env ships only `site-packages/nvidia/cuda_runtime/lib/libcudart.so.12`. None of the job's 8
    `LD_LIBRARY_PATH` dirs has a `libcudart.so`, nor do `/usr/lib64` and `/lib64`.
  - makani and ACE2 prepend `/soft/compilers/cudatoolkit/cuda-12.9.1/lib64`, which ships one.
  - If CUDA HMEM init fails, cxi would pin the GPU address as host memory and get EFAULT, which matches the log.
    aws-ofi-nccl still reports "GPU Direct RDMA Enabled" because it only checks the provider's FI_HMEM capability.
  - This log has no `FI_LOG` lines, so the mechanism itself is unconfirmed.
- **combine_handoffs' working 4-node runs** (7629096, NCCL 2.28.3) show `GDRDMA` with `GDR 1` and no registration
  errors. **cuMem was on there too** (`P2P/CUMEM/read`), so my first guess (cuMem buffers) is unlikely.
- **Other differences from their runs, untested:**
  - NCCL 2.20.5;
  - one GPU per rank: their ranks see all 4, and their topology reports "distance 8 <= 8" where ours says 6 / 7.
- **Probe for the next job** (their proposal):
  - A: the current setup + `FI_LOG_LEVEL=info`, to look for the dlopen failure;
  - B: a shim dir with `libcudart.so` → sqg-env's own `libcudart.so.12`, prepended for the probe only (the same file
    torch loads, so there is no second CUDA runtime);
  - C: `NCCL_NET_GDR_LEVEL=LOC` (host-staged), as a fallback;
  - only if B fails: `NCCL_CUMEM_ENABLE=0`, then a newer NCCL wheel.
- **Token bug, to fix before the next probe:** the probe's PASS token must require a completed all_reduce plus a
  busbw line, not just provider selection.
- **2-node bandwidth reference:** combine_handoffs' `polaris_nccl_metrics.md` §5 (18.03 GB/s at 64 MB, 8 ranks).
  Check its GDR mode before comparing.

What it says:
- **Independent simulators scale perfectly over 2 nodes**, since nothing is shared.
- **Streaming pairs to a separate trainer costs almost nothing.** The simulators are 0.3–1% slower than alone, and
  the trainer is at its standalone speed. A send takes ≈1 ms per 1.5–4.5 s message, at 10–29 MB/s. The HSN is
  idle at this load.
- **Simulators per trainer.** For o1.2's 12 fresh pairs per step at 0.419 s/step (28.6 pairs/s), one trainer needs:
  - `ref`: 10.7 simulators (4.476 / 0.419);
  - `lean64`: 8.1;
  - `mixed`: 5.4;
  - `lean32`: 3.7.
  The `lean64` and `mixed` values come from their round-1b optimizer-step times; those two were not streamed.
- **Decoupling buys wall-clock with GPUs, not GPU-hours.** One trainer plus S = t_solver / t_train simulators costs
  (S + 1) · t_train = t_solver + t_train GPU-seconds per step. That is the same as one o1.2 rank doing both.
  Replicating o1.2 over more GPUs (option a) also leaves the wall-clock per step unchanged (≈4.9 s on an A100). It
  only grows the global batch, which changes the recipe.
- **For a fixed recipe the wall-clock levers are a faster solver and decoupling.** The fixed recipe is 195,250 steps,
  7 ranks × batch 32 and 12 fresh pairs per rank-step. A faster solver is rounds 1/1b; decoupling (option b) heads
  for the offline 21 h floor. The two compose.
- **H200 GPUs needed for ≈21 h (projection, not measured).** Assume the A100 ratios hold, with the H200 solver at
  ≈2.7 s and the trainer at ≈0.39 s per step. Then S ≈ 6.9 (ref), 5.3 (`lean64`), 3.5 (`mixed`) and 2.4 (`lean32`).
  For 7 trainers that is ≈55 / 44 / 31 / 24 GPUs.
- **Freshness changes under decoupling.** Each trainer's fresh pairs come from S trajectories instead of its own one.
  Each pair is the same kind (same STRIDE and LAG), but the mix per step differs. That is a science question for
  jesswan / Weidong.
- **Cross-node NCCL is broken in this env; intra-node NCCL works.** A decoupled layout with all trainers on one node
  (≤ 4 on Polaris) needs no cross-node NCCL, since the simulator → trainer traffic is gloo and works. Cross-node DDP
  needs the CXI fix first: 7 trainers, or option (a). The fp32 gradients are 163 MB per step. gloo was not measured
  for those.

## Scorecards

**Round 1 (`PREDICTIONS_round1.md`):**
- **Gates:** all six were predicted PASS and all six FAILED. Refuted, but the cause was the gates (the `t` check and
  the f32 limit), not the variants' arithmetic.
- **1-step errors:**
  - **Seed 1** "0 to 1e-15": confirmed (0). Bits equal at 480 steps (predicted ≤ 50% likely): yes for all three.
  - **`lean64`:** predicted 1e-16–3e-15 synthetic, measured 2.90e-14, refuted (10× above). Predicted 1e-18–1e-16
    random, measured 2.4e-22, refuted (below).
  - **`mixed`:** predicted 3e-7–2e-6 synthetic, measured 2.30e-5, refuted. Predicted 1e-9–1e-7 random, measured
    1.75e-13, refuted (below).
  - **`lean32`:** predicted 3e-7–2e-6 synthetic, measured 2.30e-5, refuted. Predicted 5e-8–1e-7 random, measured
    6.84e-8, confirmed.
  - The float32 synthetic ranges sat below float32's own floor on that state, the same mistake as the 1e-5 gate.
- **Speed:**
  - `lean`: 7.3 (6.6–7.9) predicted, 8.00 measured, refuted.
  - `lean_u1`: 6.15 (5.9–6.6) predicted, 6.40 measured, confirmed.
  - `lean_split`: 6.7 (6.3–7.1) predicted, 7.07 measured, confirmed.
  - `lean64`: 5.8 (5.3–6.8) predicted, 7.07 measured, refuted.
  - `mixed`: 3.3 (2.8–3.9) predicted, 4.68 measured, refuted.
  - `lean32`: 2.5 (2.3–2.9) predicted, 3.20 measured, refuted.
  - Every miss is on the slow side.
- **Optimizer step** (measured in round 1b): `lean64` 2.8 → 3.40 s, `mixed` 1.6 → 2.28, `lean32` 1.2 → 1.55.
- **Winner:**
  - Overall fastest, `lean32`: confirmed on speed. It passes only in round 1b.
  - Float64, "`lean64` over `lean_u1` 55/45": refuted; `lean_u1` wins by 0.67 ms, as the stated 6.3 ms marker
    implied.
  - Full order `lean64` < `lean_u1` < `lean_split` < `lean`: refuted (`lean_u1` < `lean_split` ≈ `lean64` < `lean`).
  - "≈85% both beat `lean`": confirmed.
- **Information-only:**
  - 4,000-step coarse: float64 0, confirmed; `mixed` 6.37e-6 in 1e-6–1e-4, confirmed; `lean32` 4.36e-4 in
    1e-5–1e-3, confirmed.
  - Energy: float64 0, confirmed; `mixed` 2.6e-8 in 1e-9–1e-6, confirmed; `lean32` 8.4e-5 vs 1e-7–1e-5, refuted. Plain
    float32 gives the same 8.4e-5.
  - FFT micro-table: C2C c128 0.8–0.9 predicted, 1.198 measured, refuted (this is `lean64`'s miss). C2R c128
    1.014, R2C f64 0.619 and the c64 ≈ halves: confirmed.
  - Seed refs agree to 0.015% (≤ 0.5%), confirmed.
  - Peak memory 3.79 / 8.39 GiB vs ≈5 / ≈7: half.
  - Wall: seed 1 7.1 min (≈10 predicted), seed 2 7.1 (7–8); job 15.7 min (≈20). Confirmed.
  - No first-execution exception (≈25% per seed predicted).

**Round 1b + smoke (`PREDICTIONS_round1b.md`):**
- **1, confirmed:** `lean64` 2.90e-14 / 2.41e-22; `t` exact.
- **2, confirmed:** `mixed` 2.30e-5 vs 4.96e-5, ratio 0.93; random 1.75e-13.
- **3, confirmed:** `lean32` ratio 0.93 / 1.00.
- **4, confirmed:** 7.051 / 4.714 / 3.189 ms, i.e. −0.3 / +0.6 / −0.5% vs round 1's measured 7.073 / 4.684 / 3.205.
- **5, half:** cxi was selected on both nodes, but the predicted 10–20 GB/s bus bandwidth is refuted. The all_reduce
  never ran.
- **6, confirmed:** every rank within 0.3% of 1 GPU; efficiency 0.9999 / 1.0024.
- **7, confirmed with two edge misses:** simulators 9.32–9.33 ms, 8.14 fresh pairs/step and solver-bound are all
  confirmed. The trainer's 0.419 s is 0.2% under the predicted 0.42–0.45, and the printed 19.38 pairs/s is 2% above
  18–19 (the period-based 18.77 is inside).
- **8, confirmed:** 54.5 pairs/s, 22.6 fresh, trainer-bound, 28.6 MB/s; median sends 1.04 / 1.27 ms (< 5 / < 10).
- **9, confirmed:** the job's outputs are 0.46 MB of logs/JSON/HLO text.

## Next (the user's decisions; handoff §4–5)

- **Seed-1 rerun.** `lean_u1` (bitwise fields, 1.45×) under round 1b's `t` rule.
- **Which precision goes into o1.2.**
  - float64: `lean64`, whose frames are bitwise, or `lean_u1`.
  - float32 / mixed: needs statistical validation and jesswan / Weidong's sign-off.
  - Round 2 is the union of the float64 tricks (C2C packing + ring + one element per thread), and later cuFFT callbacks.
- **The NCCL-over-CXI fix probe:** `FI_LOG` / `libcudart.so` shim / `GDR_LEVEL=LOC`, above. It comes before any
  cross-node DDP.
- **Replicate vs decouple.** The trade-off above is wall-clock vs GPUs at equal GPU-hours.

The user's decisions (2026-11-05) on all four became round 1c (Part IX).

---

# Part IX — Round 1c: seed 1 rerun, the NCCL-over-CXI fix, replicate vs decouple (job 7815637, 2026-11-05)

**Scope (user decisions after 7815096):** rerun seed 1 under round 1b's `t` rule; carry `lean32` toward o1.2; measure
both scale-out layouts.

**Run and harness.**
- **Job:** `SQG/analysis/opt/run_round1c_polaris.pbs`, 2-node debug, x3206c0s25b1n0 + x3206c0s31b0n0, 05:51–06:11
  (1,202 s). It ended `OPT_R1C_DONE failures=0`. Summaries: `SQG/analysis/results/polaris_opt_round1c_7815637/`.
- **Predictions:** `SQG/analysis/opt/PREDICTIONS_round1c.md`, written before submission.
- **NCCL arms:** from combine_handoffs, after a readelf check of the library stack (Part VIII, "Leading
  hypothesis").
- **Review:** two read-only Fable critics reviewed the code before submission.
- **Code:** `multisim/sqg_multisim.py` gained `--mode replicate` and `--mode decouple`; `multisim/probe_verdict.py` is
  the strict probe token.
- **NCCL phases** ran with `OMP_NUM_THREADS=1`.

## Phase 1 — seed 1 under round 1b's `t` rule (node 0, GPU 0)

`LEAN_RING_TEST_OK`, `SEED1_GATE_R1B_TEST_OK`, `SEED1_GATE_OK` × 3, `SEED1_BENCH` × 3, `SEED1_OPTSTEP`, `SEED1_LONG` ×
3, `SEED1_DONE` (9.0 min), `OPT_RANK_OK`. The ref is 9.288 ms/step and drifted 0.0% over the run.

| variant | gate | fields at 1 / 40 / 480 steps (both starts) | `t` exact | 4,000 steps: coarse / energy / θ̂ | ms/step | speedup | optimizer step |
|---|---|---|---|---|---|---|---|
| `lean` | PASS | bitwise | yes / yes | 0 / 0 / 0 | 7.970 | 1.165× | — |
| `lean_u1` | PASS | bitwise | yes / yes | 0 / 0 / 0 | **6.400** | **1.451×** | **3.086 s** (ref 4.470 s, 1.449×) |
| `lean_split` | PASS | bitwise | yes / yes | 0 / 0 / 0 | 7.078 | 1.312× | — |

What it says:
- **`lean_u1` computes exactly what production computes, 1.45× faster.** Its float64 fields are bitwise equal to the
  production step at every checkpoint, up to 4,000 steps from the synthetic state. So it needs no statistical
  validation and raises no precision question.
- **It beats every other float64 variant.** `lean64` (round 1b) runs 7.051 ms, frames bitwise but fields ≤ 1e-13.
- **The run-time `dt` made `t` exact here too.** In both seeds the only change to the step was where `dt` comes from,
  and `t` went from inexact (round 1) to exact. That supports XLA constant-folding the unrolled `(t + dt) + dt`; the
  HLO was not inspected.

## Phase 2 — NCCL over CXI, 8 ranks, 64 MB all_reduce (NCCL 2.20.5, aws-ofi-nccl v1.6.0, libfabric 2.3.1)

| arm | environment change vs 7815096 | verdict | busbw min (per-rank median ms) | inter-node lines `/GDRDMA` | `cxil_map` / regfail | FI_LOG |
|---|---|---|---|---|---|---|
| `A_filog` | `FI_LOG_LEVEL=info` | **FAIL** (rc 124, no JSON) | — | 16 / 16 | 30 / 12 | `cuda_hmem_dl_init(): Failed to dlopen libcudart.so` on 8/8 ranks; no CUDA HMEM |
| `B_shim` | + dir with `libcudart.so` → the env's `libcudart.so.12` (12.1.105) | **PASS** | **31.5 GB/s** (3.52 ms) | 24 / 24 | 0 / 0 | `Number of CUDA devices detected: 1` on 8/8; `cuda dmabuf support status: 0` |
| `C_loc` | `NCCL_NET_GDR_LEVEL=LOC` | **PASS** | 17.2 GB/s (6.67 ms) | 0 / 24 | 0 / 0 | (not set) |
| `D_cuda129` | + `/soft/compilers/cudatoolkit/cuda-12.9.1/lib64` | **PASS** | 32.6 GB/s (3.45 ms) | 24 / 24 | 0 / 0 | as B |

`B_shim` and `D_cuda129` each log 24 libfabric warnings: `cuda_set_sync_memops(): Failed to perform
cuPointerSetAttribute: CUDA_ERROR_NOT_SUPPORTED`. The all_reduce values were right in both. We infer, but did not
check, that libfabric asks for SYNC_MEMOPS on NCCL's cuMem (VMM) buffers, which do not support that attribute.
combine_handoffs (2026-11-05) has never seen these warnings, but their only FI_LOG run never registered CUDA memory, so
their data cannot rule on them. Their cheap test is one more probe arm: `B_shim` + `NCCL_CUMEM_ENABLE=0`. If the 24
warnings disappear, the cuMem attribution is confirmed. Until then, "benign" is an inference. Do not silence the
warnings with libfabric's `disable_cuda_sync_memops` in an equivalence-gated run without that A/B. Phase 3
ran on `B_shim`, the first in the preference order B, D, A, C. Its NET lines were `/GDRDMA` in every arm (24 / 8 / 24),
with 0 registration errors.

What it says:
- **The cause is confirmed.** In 7815096's environment, libfabric's CUDA HMEM layer cannot dlopen the unversioned
  `libcudart.so`, which the wheel-based env does not ship. The cxi provider then registers GPU addresses as host memory
  and gets EFAULT.
- **The fix is one symlink to the runtime file torch already loads**, so there is a single CUDA runtime in the
  process. It is now part of `run_round1c_polaris.pbs`'s `arm_env B_shim`.
- **GDR roughly doubles 2-node bandwidth over host-staged: 31.5 vs 17.2 GB/s.** Our host-staged 17.2 GB/s matches
  combine_handoffs' 2-node 18.0 GB/s, which ran GDR 0.
- **`D_cuda129` (makani/ACE2's recipe) works too, at +3%**, with a second CUDA runtime (12.9) in the process. `B_shim`
  stays the recommended setup.

## Phase 3 — replicate vs decouple, DDP over `B_shim` (A100, batch 16 per trainer rank, strict fp32)

| | `replicate lean32` | `decouple lean32` | `replicate ref` |
|---|---|---|---|
| layout | 8 GPUs, each solver + DDP rank | per node: 1 DDP trainer (local rank 1) + 3 simulators | 8 GPUs, each solver + DDP rank |
| global batch; fresh pairs per trainer step | 128; 12 | 32; **12.0**, 0 starved steps | 128; 12 |
| s/step | **1.946** (ranks 1.9455–1.9457) | 0.4235 free-running; **≈ 0.515 paced** (see below) | **4.897** (job B 1 GPU: 4.887) |
| same steps without the all_reduce (`no_sync`) | 1.941 → DDP cost **5.2 ms** | free 0.4235 vs the stream arm's no-DDP 0.419 (7815096, other nodes) → ≈ 4.5 ms | 4.892 → DDP cost **4.8 ms** |
| rank-steps/s per GPU | 0.514 | ≈ 0.485 (paced) | 0.204 |
| other | DDP init 0.42 s; buffer fill 17 s; torch peak 20.4 GiB | simulators 3.231–3.236 ms/step; 23.2 pairs/s per trainer | DDP init 0.37–0.59 s; fill 51 s |

**The paced number is derived, because the harness reported the wrong statistic.**
- **The printout is a median.** `decouple` printed paced s/step 0.424 and 0.590 rank-steps/s per GPU, both from the
  median step. The median wait was 0.0 s.
- **Why the median misleads.** The 3 simulators per trainer start within 0.03 s of each other, so their 12-pair
  messages arrive together, ≈ 36 pairs every 1.55 s. Two steps in three find pairs waiting; the third waits for the
  next burst.
- **The phase mean.** The phase got exactly 720 fresh pairs (12.0 × 60 steps) at 23.17 pairs/s, from the simulators'
  periods; the printed rate is 23.39. That puts the mean at 0.513–0.518 s per step.
- **The fix.** `sqg_multisim.py` now records each phase's wall time and mean, and the summary uses the means
  (`fix(sqg): decouple trainer -- report the paced phase by its mean`).

What it says:
- **Cross-node DDP is nearly free: ≈ 5 ms per step for 8 ranks and 163 MB of gradients**, 0.3% of the `lean32` step
  and 0.1% of `ref`'s.
- **Today's o1.2 replicates to 2 nodes at no per-step cost** (4.897 vs 4.887 s), but every added rank grows the
  global batch.
- **Equal GPU-seconds, measured: decouple ≈ 0.485 vs replicate 0.514 rank-steps/s per GPU** (ratio 0.94). The 6% is
  the trainers idling between bursts in a solver-bound layout: 3 `lean32` simulators give 23.2 pairs/s, and a 0.42 s
  trainer step needs 28.3.
- **Decoupling buys 3.8× shorter steps (1.946 → ≈ 0.515 s) for the same 12 fresh pairs per rank-step.** At this GPU
  count the global batch is 4× smaller. With ≥ 3.7 `lean32` simulators per trainer the step would reach the trainer's
  0.42 s.
- **`replicate lean32` is 1.946 s, slightly under the 1-GPU sum** of 1.546 + 0.419 = 1.965 s. Its solver has no
  host sync, so the next call's dispatch overlaps the end of the ML step a little: `advance()` host time is 1.36 s for
  ≈ 1.55 s of GPU work.

## Scorecard (`PREDICTIONS_round1c.md`)

- **1, confirmed:** all three pass, bitwise, `t` exact (the ≈ 85% call).
- **2, confirmed:** 7.970 / 6.400 / 7.078 ms, i.e. −0.4 / 0.0 / +0.2%.
- **3, confirmed:** `lean_u1` is the best float64 variant at 1.451×, optimizer step 3.086 s (≈ 3.1) vs `lean64`'s
  3.40.
- **4, confirmed:** all zero.
- **5, confirmed:** A fails with 7815096's signature; `Failed to dlopen libcudart.so` on 8/8 ranks; no CUDA HMEM line.
- **6, half:** every predicted marker is confirmed. GDRDMA is on all inter-node lines (24 here; the failing runs printed 16
  before their init aborted); there are 0 errors, CUDA HMEM is up on 8/8 and nothing reads "Failed to find". busbw 31.5
  GB/s refutes the 12–20 range: the 18.0 reference was host-staged.
- **7, confirmed:** 17.2 GB/s in 8–18, no GDRDMA, ≤ B.
- **7b, confirmed:** D passes with GDRDMA, at 32.6 GB/s = +3.3% of B (±15%).
- **8, half:** the DDP cost of 5.2 ms is confirmed (0–0.10 s). s/step 1.946 and no_sync 1.941 sit just below their
  ranges (1.99–2.10, 1.95–2.02), and 0.514 rank-steps/s per GPU just above (0.48–0.51). The 1-GPU sum missed the small
  overlap.
- **9, half:** free 0.4235 (0.42–0.47), 12.0 fresh, 0 starved and simulators 3.231–3.236 (3.22–3.26) are confirmed.
  The paced s/step was not measured as predicted: the printed median 0.424 is outside 0.50–0.54, and the derived mean
  of ≈ 0.515 is inside but is not a direct measurement.
- **10, confirmed on the derived mean:** 0.485 vs 0.514 per GPU (within ±10%), steps 3.8× shorter (≈ 3.9×
  predicted), global batch 32 vs 128.
- **11, confirmed:** 4.897 s in 4.88–5.05, DDP cost 4.8 ms, 0.204 rank-steps/s per GPU.

## Next (the user's decisions)

- **What goes into o1.2 first.**
  - `lean_u1` is now a bitwise float64 1.45× (optimizer step 4.47 → 3.09 s on an A100). It needs no science
    sign-off.
  - `lean32` (the user's 11-05 choice) is 2.9× (1.55 s), but needs statistical validation and jesswan / Weidong's
    sign-off.
  - They don't exclude each other: `lean_u1` can ship now while `lean32`'s validation runs.
- **Round 2 (float64):** `lean_u1`'s one-element-per-thread trigger + `lean64`'s C2C packing + the 2-slot ring.
  Separately, the same trigger on `lean32`.
- **Multi-node:** DDP over CXI now works (`B_shim`). The design choice is wall-clock per step (decouple) vs global
  batch (replicate) at equal GPU-hours. Sizing a decoupled run means ≥ 3.7 `lean32` (or ≥ 10.7 `ref`) simulators per
  trainer on A100s.
- **Cheap open check (combine_handoffs):** a probe arm `B_shim` + `NCCL_CUMEM_ENABLE=0`, to attribute the
  `cuPointerSetAttribute … NOT_SUPPORTED` warnings. It needs a 2-node job, so fold it into the next one.

---

# Part X — Nsight analysis of `lean32`, and where the solver can still improve (job 7819082, 2026-11-05)

**Ask (user, 2026-11-05):** an Nsight analysis of `lean32`, and where else the solver can improve.

**Run and harness.**
- **Job:** `SQG/analysis/opt/nsight/run_nsight_lean_polaris.pbs`, 1-node debug, x3001c0s19b0n0, 23:21–23:42 (1,261 s).
  It ended `NSIGHT_DONE failures=0`.
- **Predictions:** `SQG/analysis/opt/nsight/PREDICTIONS_nsight.md`, written before submission.
- **Summaries:** `SQG/analysis/results/polaris_nsight_lean_7819082/`. The tables below regenerate with
  `python3 SQG/analysis/opt/nsight/analyze_nsight.py SQG/analysis/results/polaris_nsight_lean_7819082`.
- **New tools.**
  - `summarize_nsys_ops.py`: GPU time per XLA op per step. Each kernel and memcpy follows its CUDA call to the XLA
    thunk NVTX range (`hlo_op=`) open on its thread. Validated on job E's real capture: AB3 fusion 649.8 µs/step
    (job E: 650).
  - `summarize_ncu_window.py`: every kernel of one loop period, from an unfiltered ncu window.
  - `lean32_probes.py`: exact rewrites of `lean32`.

## Probes: exact rewrites of `lean32` (phase 1, one A100)

Each probe passed its check vs seed 2's `lean32` (rel ≤ 1e-6 at 40 steps). **All are bitwise at 40 and 480 steps,
`t` exact.**

| variant | ms/step | vs `lean32` | vs production | full-size copies (HLO) |
|---|---|---|---|---|
| `ref` (production) | 9.288 | 0.344× | 1.000× | 5 |
| `lean_u1` (float64) | 6.409 | — | 1.449× | 3 |
| `lean32` | 3.198 | 1.000× | 2.904× | 2 |
| `lean32_u1` (+ cos(x·0) trigger) | **2.641** | **1.211×** | 3.517× | 2 |
| `lean32_ring3` (3-slot ring) | 3.104 | 1.030× | 2.992× | **0** |
| `lean32_u1_ring3` | **2.524** | **1.267×** | **3.681×** | 0 |

What it says:
- **Seed 1's trigger is worth 0.557 ms/step on `lean32` (−17%), bitwise.** That is far more than predicted. The
  ring saves 0.093 ms, and the two add up (0.674 vs 0.650 ms).
- **`lean32_u1_ring3` is the fastest exact form of `lean32`: 2.524 ms, 3.68× production.** It changes nothing
  `lean32` computes. The open question is `lean32`'s float32 numerics (Part XI), not this rewrite.

## Where `lean32`'s 3.21 ms go (nsys per XLA op, 2 ranges × 480 steps)

GPU busy 3,212 of 3,231 µs per step (99.4%).

| op | µs/step | share | ideal MB/step | achieved GB/s | at 1.35 TB/s | headroom |
|---|---|---|---|---|---|---|
| AB3 fusion (2 per body) | 926 | 28.8% | 504 | 544 | 373 | **553** |
| 2 × RFFT (uθ, vθ) | 615 | 19.1% | 537 | 874 | 398 | **217** |
| C2C (u − i v) | 540 | 16.8% | 537 | 994 | 398 | **143** |
| IRFFT (θ): copy + 2 passes + `scal` | 503 | 15.7% | 537 | 1,067 | 398 | 105 |
| products | 253 | 7.9% | 336 | 1,325 | 249 | 5 |
| `build_w` | 235 | 7.3% | 235 | 998 | 174 | 61 |
| tendency copies | 99 | 3.1% | 134 | 1,357 | 100 | 0 |
| entry/exit casts (`loop_convert_fusion`, `call`) | 34 | 1.0% | — | — | — | — |

- **Totals:** ideal 2,819 MB/step, i.e. 2.09 ms at 1.35 TB/s; summed headroom ≈ 1.08 ms.
- **How "ideal" is counted:** every operand read once and every output written once, from the HLO shapes. An FFT
  pass reads and writes its whole array. The 1.35 TB/s reference is what the products, `scal` and the cuFFT row
  passes reach.

## Kernel by kernel (ncu `--set full`, every kernel of a 120-launch window, production clocks)

The loop period was 30 launches = 2 steps. ncu's per-step kernel time is 2,852 µs: −11% vs nsys, because the memcpys
are not kernels and kernels run serialized with flushed caches.

| kernel | µs | DRAM read / write MB | % of DRAM peak | top stall | ncu's own rule |
|---|---|---|---|---|---|
| AB3 `loop_add_multiply_fusion` (16,392 × 128: 4 elements per thread) | 1,042 | **1,068 / 302** | 84.6 | long_scoreboard | 75% excessive sectors (uncoalesced) |
| AB3 `loop_add_multiply_fusion_1` | 795 | **808 / 221** | 83.3 | long_scoreboard | 75% excessive sectors |
| C2C column pass `regular_fft_factor` (1,024 × 1,024) | 317–320 | 134 / 121 | **51.2–51.6** | lg_throttle | 43% excessive shared wavefronts |
| C2C row pass `vector_fft` | 187–189 | 134 / 117 | 85.5–86.5 | long_scoreboard | — |
| half-plane column pass `regular_fft_factor` (513 × 1,024), × 3 per step | 186–190 | 67 / 52 | **40.4–41.2** | lg_throttle | 42% excessive global sectors; 4.2-wave tail |
| `vector_fft_symm_c2r` / `_r2c` | 96 / 98 | 67 / 50–58 | 78.1–81.9 | — | — |
| `scal` (1/N²) | 93 | 67 / 50 | 80.8 | — | — |
| products `loop_multiply_fusion_1` | 239–240 | 201 / 121 | 86.3–86.6 | — | — |
| `build_w` `input_concatenate_fusion_1` | 229–230 | **191** / 124 | 87.8–88.2 | — | 8% excessive sectors |

Unique data per AB3 launch: 369 MB read (5 complex64 fields + `decay`), 134 MB written.

**`lean32_u1`'s AB3 fusion uses 65,568 × 128 threads, one element each.** It reads **369.3 MB, exactly the unique
data**, and writes 126 MB, in 359–360 µs at 88.8% of peak. That is 2.2–2.9× less time than `lean32`'s.

What it says:
- **The AB3 fusion was the largest single cost, and its cause was over-fetch, as in float64 (job F).** XLA's
  4-elements-per-thread loop emitter makes a warp stride over complex64 elements, so each sector is fetched 2–3×.
  Job E's "float32 fixes it" was only partly true: float32 fits the L2 better, but the pattern remains. One element
  per thread removes it completely.
- **The cuFFT column passes are the next cost: 41–52% of DRAM peak.**
  - ncu blames uncoalesced global accesses on the 2,049-wide half-plane (42% excessive sectors), and shared-memory
    bank conflicts on the 4,096-wide full plane (43% excessive wavefronts).
  - So the 2,049-column misalignment is not the whole story (prediction 9's mechanism): the aligned full-plane
    column pass is slow too.
  - These kernels are inside cuFFT. Only a different transform layout or a custom FFT changes them.
- **cuFFT 11.4.1 (CUDA 12.9.1, phase 3) changes nothing.**
  - It really was loaded: the mapped file is `.../cuda-12.9.1/.../libcufft.so.11.4.1.4`.
  - The FFT micro-table moved < 1%, and `lean32` ran 3.200 vs 3.198 ms.
  - Production float64 frames were bitwise identical at 40/480 steps; `lean32`'s frames differed by 1.7e-10 / 6.7e-9.
  - A cuFFT upgrade is not a lever.
- **The rest is near bandwidth:** the products (87%), the row passes and `scal` (78–82%), and `build_w` (88%, but
  it reads 1.9× its unique data through the mirrored θ̂ reads).

## Float64 `lean_u1` (6.40 ms; bitwise production)

nsys: GPU busy 99.5%.

| part | ms/step | share |
|---|---|---|
| 3 IRFFTs (copy + 2 passes + `scal`, ≈ 1.0 ms each) | **3.01** | **47%** |
| 2 RFFTs | 1.22 | 19% |
| XLA fusions | the rest | |

The trigger worked on its AB3 fusion (`loop_multiply_fusion_1`): 65,568 × 128 threads, 671 MB read, the unique
data, at 89.8% of peak.

**One fusion still uses 4 elements per thread:** `loop_multiply_fusion_3`, once per step, 571–586 µs at 66% of peak,
lg_throttle, 279 MB read / 310 MB written.

Note: `ncu_window_lean_u1_xla.json`'s period of 5 XLA kernels is **one** step, because the 3-step body reuses one
kernel per fusion. Its `per_step` numbers are therefore 3× too small; the per-period numbers are right.

## Scorecard (`PREDICTIONS_nsight.md`)

| # | verdict | detail |
|---|---|---|
| 1 | confirmed | `lean32_r2` bitwise |
| 2 | half | check + bitwise confirmed. Speed 2–8% refuted: **17.4%** (1.211×) |
| 3 | confirmed | 0 copies; −93 µs (80–130) |
| 4 | confirmed | additive: 674 vs 650 µs (±50) |
| 5 | confirmed | timings reproduce: 9.288 / 6.409 / 3.198 ms |
| 6 | confirmed | GPU busy 99.4% (≥ 97) |
| 7 | half | 6 of 8 ranges hold. AB3 926 µs vs 380–600 refuted; "everything else" 36 µs vs < 15 (the c128↔c64 casts at each call's entry/exit) |
| 8 | confirmed | FFT family 51.6% (50–58) |
| 9 | half | half-plane column passes at 40–41% of peak (≤ 55) and 42% excessive global sectors (sector-efficiency metric not in this ncu's raw export). The mechanism is not only misalignment: the aligned full-plane pass is slow too (52%, shared-memory bank conflicts) |
| 10 | refuted | C2C column pass at 51–52% of peak, not ≥ 70 |
| 11 | half | row passes (78–82), `scal` (81), products (86–87) are ≥ 80, but `vector_fft_symm_c2r` is 78.1–78.6 |
| 12 | confirmed | AB3: 4 elements per thread, reads 2.2–2.9× unique, 83–85% of peak |
| 13 | confirmed | `build_w` reads 1.89× (1.3–2.2) |
| 14 | confirmed | period found (30); ncu −11% vs nsys (≤ 15) |
| 15 | confirmed | `lean32_u1` AB3: 1 element per thread, read = 1.00× unique |
| 16 | confirmed | `lean_u1` AB3: 1 element per thread, 671 MB = unique |
| 17 | confirmed | cuFFT 11.4.1 mapped |
| 18 | confirmed | micro-table ≤ 1%; `lean32` +0.06% |
| 19 | half | both differences ≤ their bounds, but `ref` came out bitwise (predicted not bitwise, 70%) |
| 20 | half | headroom 1.08 ms (≥ 0.8 confirmed), but the half-plane column passes hold ≈ 0.30 ms (< 0.4); the AB3 over-fetch (0.55 ms) is the largest |
| 21 | confirmed | AB3 over-fetch; the IRFFT copy + `scal` = 0.205 ms |

**Totals: 14 confirmed, 6 half, 1 refuted.**

## Where the solver can still improve (ranked by measured or bounded gain)

**Float32 path.** Precision caveat: see Part XI.

| # | change | gain | exact? | status |
|---|---|---|---|---|
| 1 | cos(x·0) trigger on the AB3 fusion | **−0.557 ms** | bitwise | measured; ready |
| 2 | 3-slot ring | −0.093 ms | bitwise | measured; ready |
| | → `lean32_u1_ring3` | **2.524 ms (3.68×)** | bitwise to `lean32` | |
| 3 | keep the complex64 state across calls (no c128↔c64 cast at each call's entry/exit) | ≈ −0.034 ms | bitwise to the continuous run | estimate from nsys |
| 4 | one forward C2C of (u − i v)·θ instead of 2 RFFTs (Hermitian separation folded into the AB3 fusion) | ≈ −0.075 ms (micro: 0.526 vs 2 × 0.311) | rounding only | estimate |
| 5 | cuFFT column passes (41–52% of peak): the remaining cuFFT headroom ≈ 0.43 ms | up to ≈ −0.43 ms | rounding only | needs a custom FFT (cuFFTDx / VkFFT / LTO callbacks); not reachable through `jnp.fft` |
| 6 | IRFFT input copy + `scal` (0.205 ms): fold 1/N² into the products, skip the copy | ≈ −0.1 to −0.2 ms | exact (power-of-2 scale) | needs a custom FFT call (XLA's thunk always copies and scales) |
| 7 | `build_w`'s mirrored re-read (1.9×) | ≈ −0.06 ms | exact | layout work |

- **Reachable in XLA:** items 1–4 give ≈ 2.4 ms.
- **With a custom fused FFT pipeline:** ≈ 1.8–2.0 ms. That is near Part VII's 1.5–1.8 ms FFT-roofline estimate.

**Float64 path** (bitwise or rounding-only; no precision question):

| # | change | gain | exact? |
|---|---|---|---|
| 1 | `lean64`'s C2C packing (two IRFFTs → one C2C) on top of `lean_u1` | ≈ −0.8 to −0.9 ms (measured `lean` → `lean64`: −0.93) | rounding only, ≤ 1e-13; `lean64`'s frames were bitwise |
| 2 | trigger on the remaining 4-element fusion (`loop_multiply_fusion_3`) | up to ≈ −0.15 ms | bitwise |

- **Expected round-2 float64 step:** ≈ 5.3–5.5 ms, ≈ 1.7× production.

**Not levers:**
- cuFFT 11.4.1;
- XLA unroll flags (none exist in this jaxlib; job F);
- host-side scheduling: the GPU is 99.4–99.5% busy in both variants.

---

# Part XI — Does float32 damage o1.2's training delta? (job 7819310, 2026-11-06)

**Ask (user, 2026-11-05):** "we need to be sure that fp32 is not impacting δ as it is very small".

o1.2 learns Δ = θ(t + 0.02) − θ(t) on 256² float32 frames (θ / 8), a lag of 4,000 solver steps. Round 1b had
compared frames, not deltas.

**Run.**
- **Job:** `SQG/analysis/opt/delta/run_delta_polaris.pbs`, 1-node debug, x3001c0s13b0n0, 2026-11-06 00:48–01:16
  (1,667 s). It ended `DELTA_JOB_DONE failures=0`.
- **Predictions and the gate:** `SQG/analysis/opt/delta/PREDICTIONS_delta.md`, written before submission.
- **Results:** `SQG/analysis/results/polaris_delta_7819310/`. The spun-up state (npz, 0.4 GB) stays in
  `$MEMBER_ROOT/sqg_profile/delta_7819310/` for reuse.

## Two mechanisms, found before the run

1. **`decay` stored in float32 quantizes each mode's damping rate.** rate·DT ≈ 1e-7–5e-6, and float32 numbers
   below 1 are 6e-8 apart.
   - Rate error: up to ±20% at k = 4–16 (rms 11–13%), ±5% in the forcing ring, 0.1% at k = 1.
   - It is deterministic per mode, and plain float32 has it too (`SQG/analysis/opt/delta/decay_quant.py`).
2. **A float32 state rounds sub-ulp per-step changes systematically wherever the increments vary slowly.**
   - `test_f32_update.py` (emulated float32; one forced k = 8.4 mode; 100k steps): the 4,000-step Δ is **31%
     off** for `lean32`'s update.
   - **It is the same 31% with expm1 damping.** The decrement is also sub-ulp and gets rounded onto the state's
     grid.
   - A compensated (TwoSum hi + lo) update gives **8.8e-8**.

## Variants and method

| variant | what | ms/step |
|---|---|---|
| `mixed` | seed 2: float32 FFTs and products; float64 state, AB3 and damping | 4.732 |
| `lean32` | seed 2: all float32 | 3.202 |
| `lean32e` | `lean32` + expm1 damping + AB3 terms summed first | 3.203 |
| `lean32ek` | `lean32e` + compensated hi + lo complex64 state (`f32_update.compensated_update`) | **3.351** |
| `f64ic32` (control) | float64 from the state rounded once to complex64: the floor | — |

**Prior art** (added 2026-11-10, checked against Crossref and the full text):
- **The compensation is published.** `lean32ek`'s hi + lo state is "compensated time integration", from Klöwer,
  Hatfield, Croci, Düben & Palmer (2022), *J. Adv. Model. Earth Syst.* 14(2), e2021MS002684,
  https://doi.org/10.1029/2021MS002684.
  - They used it for Float16 shallow water, with one compensation variable per prognostic variable.
  - Uncompensated, their error grew 100-fold in 150 days. Compensated, it matched mixed precision.
  - It ran 3.6× faster on A64FX, against ≈ 2.75× for mixed precision.
- **The building blocks:** TwoSum (Møller 1965; Knuth, *TAOCP* vol. 2 §4.2.2), Fast2Sum (Dekker 1971) and
  compensated summation (Kahan 1965; Higham 1993). Full list in `SQG/analysis/opt/team_2026-11-06/ANALOGUES.md`.
- **Ours, with no external source:** the expm1 damping table, and the fast form's cos(x·0) trigger and 3-slot ring.

**Method.**
- **Reference:** float64 `lean_u1`, confirmed bitwise equal to production on pair 0.
- **Start state:** seed 2's synthetic −5/3 field, spun up 0.5 time units in float64. CFL 0.31–0.45, SPE 106 → 75.
  No archived checkpoint is staged on Polaris.
- **Pairs:** 8 pairs, 0.02 apart along the float64 trajectory, through o1.2's own `coarse()` / 8.

## Per-step increments (full 4096² spectrum, identical float32-representable start)

The increment is a small fraction of θ: 9.8e-6 at k = 1–2, 1.6e-5 at k = 2–4, 1.7e-4 at k = 8–16, 8.4e-4 overall.

| 1-step increment rel. error | k 1–2 | k 2–4 | k 4–8 | k 8–16 | k 16–32 | all |
|---|---|---|---|---|---|---|
| `lean32` | **5.3e-3** | **3.8e-3** | 9.3e-4 | 3.6e-4 | 1.1e-4 | 7.8e-5 |
| `lean32e` | 5.4e-3 | 2.7e-3 | 6.1e-4 | 2.0e-4 | 7.2e-5 | 5.8e-5 |
| `lean32ek` | 4.3e-6 | 2.2e-6 | 1.9e-6 | 2.4e-6 | 2.4e-6 | 3.3e-5 |
| `mixed` | 4.1e-6 | 2.2e-6 | 1.9e-6 | 2.4e-6 | 2.3e-6 | 3.3e-5 |

`lean32` gets each large-scale increment 0.4–0.5% wrong, every step. That is state rounding (6e-8 of θ) on an
increment of 1e-5 of θ. **The compensation survives XLA:** `lean32ek` equals `mixed` in every band.

## The delta at o1.2's lag (median over 8 pairs)

| e_Δ at 4,000 steps | all | k 1–2 | k 2–4 | k 4–8 | k 8–16 | k 16–32 | k 32–64 | k 64–182 |
|---|---|---|---|---|---|---|---|---|
| share of Δ energy | — | 0.01% | 0.8% | 6.5% | 21% | 30% | 23% | 19% |
| `f64ic32` floor | 2.3e-7 | 1.2e-6 | 4.3e-7 | 1.1e-7 | 6.9e-8 | 9.7e-8 | 1.9e-7 | 4.5e-7 |
| `mixed` | 8.7e-6 | 2.2e-5 | 5.2e-6 | 3.1e-6 | 2.8e-6 | 3.9e-6 | 7.4e-6 | 1.7e-5 |
| `lean32ek` | 8.8e-6 | 2.2e-5 | 5.2e-6 | 3.1e-6 | 2.9e-6 | 3.9e-6 | 7.5e-6 | 1.8e-5 |
| `lean32e` | 4.3e-4 | 3.2e-3 | 1.4e-3 | 3.3e-4 | 1.4e-4 | 1.7e-4 | 3.5e-4 | 8.1e-4 |
| `lean32` | **5.5e-4** | **3.6e-3** | **2.0e-3** | 4.5e-4 | 2.1e-4 | 2.1e-4 | 4.3e-4 | 1.0e-3 |

How the error grows with lag (overall e_Δ / frame error; median):

| lag (steps) | 40 | 400 | 1,000 | 2,000 | 4,000 |
|---|---|---|---|---|---|
| `lean32` e_Δ | 5.8e-5 | 8.0e-5 | 1.3e-4 | 2.3e-4 | 5.5e-4 |
| `lean32` e_frame | 1.9e-6 | 1.6e-5 | 4.0e-5 | 9.8e-5 | **3.0e-4** (160×) |
| `lean32ek` e_frame | 3.8e-7 | 1.1e-6 | 1.8e-6 | 2.9e-6 | **4.8e-6** (13×) |

**Gate** (pre-set): e_Δ ≤ max(2 × floor, 1e-6), overall and in every band holding ≥ 1% of Δ's energy (k ≥ 4 here).

| variant | verdict | e_Δ / floor | e_Δ / UViT relative error |
|---|---|---|---|
| `mixed` | **FAIL** | 37× | 1.1e-4 |
| `lean32ek` | **FAIL** | 38× | 1.1e-4 |
| `lean32e` | **FAIL** | 1,900× | 5.5e-3 |
| `lean32` | **FAIL** | 2,400× | 6.9e-3 |

The UViT's relative error is taken as sqrt(0.0043)/rms(Δ) = 0.079 in this state; with the archive's rms(Δ), 0.24.
The verdicts stand as recorded; the gate is not loosened after the run.

What it says:
- **Plain float32 does damage the delta, worst where the delta is smallest.**
  - `lean32` (and the plain float32 production step it equals) gets the k = 1–4 part of the 0.02 delta 0.2–0.4%
    wrong.
  - Its frame error grows faster than linearly in the lag (160× over 100× the steps): accumulation that does not
    average out, as test_f32_update predicted.
  - The expm1 damping fix alone barely helps (−20%). State rounding is the cause, not the damping table.
- **A compensated float32 state removes 98% of it, for +0.15 ms/step.**
  - `lean32ek` matches `mixed` in every band and at every lag: ≈ 9e-6 overall, 2e-5 at k = 1–2.
  - Its error grows like a random walk (13× over 100× the steps).
  - It costs 3.35 ms vs `mixed`'s 4.73 (2.77× vs 1.96× production).
- **Nothing with float32 tendencies reaches the float64 floor.**
  - The floor is what one float32 rounding of the state does through float64 dynamics over 0.02: 2.3e-7. It sits
    far below my predicted 1e-6–1e-4, because chaos amplifies little in 0.02 time units.
  - `mixed`'s and `lean32ek`'s remaining 9e-6 is the float32 FFT / nonlinear term (≈ 2e-6 per step relative,
    1-step table).
  - So the gate as set asks "as good as float64?", and only the float64 path (`lean_u1`, `lean64`) is.
- **Limits.**
  - **The start state is not archive-like.** rms(Δ) is 0.825 vs the archive's 0.27, and the energy is still
    falling. Small scales dominate Δ here, so k ≤ 4 (where float32 errs most) holds < 1% of Δ's energy and was
    not gated.
  - **Paired, not distributional.** Paired errors do not show a bias in Δ's statistics; that needs long runs.

## Scorecard (`PREDICTIONS_delta.md`)

- **Confirmed:**
  - 2: `edec` = `decay` to 5.8e-11.
  - 4: `lean32` FAILS, overall 5.5e-4, in 1e-4–3e-3.
  - 4: `lean32e` FAILS, within 2× of `lean32` (0.80×).
  - 4: every variant's e_Δ ≤ 1% of the UViT's error (`lean32` 0.69%).
  - 5: speed. `lean32e` +0.003%; `lean32ek` +0.149 ms (0.08–0.25); `mixed` 4.73.
- **Half:**
  - 1: `lean_u1` == `ref` bitwise, but CFL reached 0.453 > 0.45 at t = 0.15.
  - 3: the error pattern is confirmed: `lean32` k1–2 5.3e-3 ≥ 1e-3, `lean32e` within 2×, `lean32ek` / `mixed`
    ≤ 1e-5 for k ≤ 16. But the k = 1–2 increment size was 9.8e-6, just under the predicted 1e-5–5e-5.
- **Refuted:**
  - 4: floor in 1e-6–1e-4 → 2.3e-7.
  - 4: `lean32ek` PASS → FAIL (38× floor).
  - 4: `mixed` PASS → FAIL (37× floor).
  - 4: rms(Δ) 0.15–0.40 → 0.825.

## Decision for the user (science; the pre-set gate settles only "as good as float64": nothing in float32 is)

| path | Δ error at the 0.02 lag | ms/step (A100) | vs production |
|---|---|---|---|
| float64 `lean_u1` | 0 (bitwise) | 6.40 | 1.45× |
| float64 round 2 (C2C + trigger) | ≤ 1e-13 fields, frames bitwise so far | ≈ 5.3–5.5 (estimate) | ≈ 1.7× |
| `lean32ek` | ≈ 9e-6 (2e-5 at k1–2), random-walk-like | 3.35 | 2.77× |
| `lean32ek` + trigger + 3-slot ring | same numerics, untested | ≈ 2.7 (estimate) | ≈ 3.4× |
| `mixed` | ≈ 9e-6 | 4.73 | 1.96× |
| `lean32` / `lean32_u1_ring3` | 5.5e-4 (3.6e-3 at k1–2), accumulating | 3.20 / 2.52 | 2.9× / 3.7× |

**Recommendation:**
- **Do not use plain float32 (`lean32`, with or without the trigger) for o1.2's targets.**
- If a pre-set tolerance at the `mixed` / `lean32ek` level (≈ 1e-5 of Δ) is acceptable to jesswan / Weidong,
  `lean32ek` is the float32 candidate, and the trigger and ring apply to it unchanged.
- Otherwise stay on float64: `lean_u1` now, round 2 next.

**Before either choice is final,** rerun `delta_fidelity.py --state-in` on 2–3 archived checkpoints (staged from
Midway), and add a long-run (≥ 1 time unit) bias comparison from the saved state.

---

# Part XII — Noise or learnable bias? The generator-error taxonomy on 16 archived states (job 7824121, 2026-11-06)

**Ask:** item 3 of the 2026-11-06 team plan (`SQG/analysis/opt/team_2026-11-06/TEAM_REPORT.md`), at the user's request
("do item 3"). Part XI left two things open:
- Its numbers came from one synthetic state (rms(Δ) 0.825 vs the archive's ≈ 0.27).
- Its gate asked "as good as float64?", and nothing in float32 is.

For o1.2 the question is different: is a faster solver's error noise or bias? Noise has zero conditional mean given the
learner's input and averages out in training. A bias can be learned by the UViT.

**Run.**
- **Job:** `SQG/analysis/opt/taxonomy/run_taxonomy_polaris.pbs`, 1-node debug, x3001c0s13b0n0, 2026-11-06 22:15–22:44.
  The workers took 1,631–1,692 s. It ended `TAXO_JOB_DONE failures=0` and `TAXO_ANALYSIS_DONE valid=True`.
- **Predictions and validity gates:** `SQG/analysis/opt/taxonomy/PREDICTIONS_taxonomy.md`, written before the smoke.
  Its addendum was written after the smoke and before this run.
- **Results:** `SQG/analysis/results/polaris_taxonomy_7824121/`. The raw pairs and the sealed file stay in
  `$MEMBER_ROOT/sqg_profile/taxonomy_7824121/`.
- **Sealed (the user's choice).** These stay sealed until jesswan / Weidong sign item 4's gates:
  - the bias statistics of `lean32ek` and its fast form, dt = 2e-4/30 and dt = 2e-4/32;
  - the dt order ratio.

  The file's sha256 is `6f1d0e5153918a3b14faf6e29795d91263a032479121bf87a4417fb173b78be1`, and nobody has opened it.
  Predictions 2 and 5 stay unscored.

## Design

- **States.** 16 archived A-SQG checkpoints, members 1–4 × t = 100, 200, 300, 400, loaded with o1.2's own
  `GD.load_checkpoint`.
  - They are spread over the 4 GPUs as a Latin square. CFL is 0.18–0.37 and SPE 24–48.
  - The user staged them from Midway. They are symlinked into `sqg_stage`.
- **Pairs.** 4 consecutive 0.02 pairs per state along float64 `lean_u1` (bitwise production): 64 pairs, clustered by
  state.

| arm | what it is | pairs |
|---|---|---|
| `lean_u1` | the reference (float64, bitwise production) | 64 |
| `lean32ek` | compensated float32 state: the candidate (sealed) | 64 |
| `inj` / `inj64` | control b: a planted damping shift, on `lean32ek` / on float64 (the truth) | 64 / 16 |
| `dquant` | control a: float64 with only `decay` rounded to float32 (mechanism A alone) | 16 |
| `lean32` | plain float32 (seed 2) | 16 |
| `f64ic32` | float64 from the state rounded once to complex64: the floor, and negative control 1 | 16 |
| `lean32sr` | float32 with stochastic rounding: negative control 2 | 16 |
| `prod`, `lean32ek_fast` | production (bitwise check); `lean32ek` + u1 trigger + 3-slot ring | 4 |
| dt ladder, n = 40 / 30 / 32 / 80 | n = 40 is the bitwise control; 30 and 32 are sealed; 80 (dt 2.5e-6) gives production's own dt sensitivity | 8 |

**Statistics** (`analyze_taxonomy.py` docstring).
- **Quantities.** x = F_ref(0), d = the reference Δ, and e = the variant's paired error at 4,000 steps, all on 256²
  spectra.
- **Level 1, per band.** e = αx + βd with cluster-robust SE (16 clusters), in units of the band's Δ. MDE = 2.8 SE.
- **Level 2, per shell** (one kx² + ky²). A Wald F(2, 15) p-value, then per band:
  - the fraction of shells at p < 0.0027;
  - the debiased learnable bias √(Σ(θᵀAθ − tr VA) / Σ|d|²);
  - its **noise level** √(Σ tr VA / Σ|d|²), which is what pure noise would show before debiasing.
- **Frames.** o1.2's `sub` (θ[::16, ::16] / 8) and the alias-free `af` (|kx|, |ky| ≤ 127).

## Validity: all four gates pass

Any failure would have voided the bias conclusions.

| gate | pre-set criterion | result |
|---|---|---|
| V1 | `prod` == `lean_u1` bit for bit, every lag, every GPU | ✅ at 40 / 400 / 1,000 / 2,000 / 4,000 steps on 4 of 4 GPUs |
| V2, control a | κ ∈ [0.8, 1.2] at k < 8 | ✅ 1.036 / 0.997 / 0.992; above k 8: 0.926, 0.646, 0.355, 0.140, 0.048 |
| V3, control b (paired, addendum) | recovery ∈ [0.7, 1.3] and power z ≥ 3 in every band | ✅ recovery 0.961–1.007; power z 5.2–17.6 |
| V4, dt ladder | n = 40 == `lean_u1` (frame and state hash) | ✅ on 8 of 8 states |

Control a falls off above k 8, as pre-registered. There 0.02 is no longer short against the flow, so the static
prediction −δr · 0.02 stops holding.

**MDE from `lean32ek`'s own scatter** (sub frames, x-coefficient, units of Δ), against the team's R4 prediction:

| band | k 1–2 | k 2–4 | k 4–8 | k 8–16 | k 16–32 | k 32–64 | k 64–128 | k 128–182 |
|---|---|---|---|---|---|---|---|---|
| MDE | 4.44e-6 | 9.35e-7 | 1.98e-7 | 1.39e-7 | 1.10e-7 | 7.04e-8 | 9.60e-8 | 1.17e-7 |
| predicted | 8.0e-6 | 8.8e-7 | 2.6e-7 | 1.2e-7 | 1.2e-7 | 1.2e-7 | 1.2e-7 | 1.2e-7 |
| ratio | 0.55 | 1.06 | 0.76 | 1.16 | 0.92 | 0.59 | 0.80 | 0.97 |

**Calibration on the two negative controls.** The nominal false-flag rate is 0.27%.
- **Shells (k < 64).**
  - f64ic32 flags 6 of 1,196 (0.50%), and lean32sr 5 of 1,196 (0.42%). Pooled, that is 0.46%.
  - That is below 1%, so the addendum's reading rule for P2 / P8 does not trigger.
  - The flags fall in small bands: f64ic32 has 2 of 68 at k 8–16, and lean32sr 1 of 6 at k 2–4.
- **Band cells, x-coefficient.** Calibrated: 0 of 16 cells per arm have |z_x| > 3. The largest is 2.74 (f64ic32,
  k 64–128).
- **Band cells, d-coefficient.** This one runs hot: 3 of 16 cells, against about 0.04 expected (0.14 with a t(15)
  reference).
  - f64ic32: one cell, −3.3 at k 64–128. That is unremarkable on its own.
  - lean32sr: two cells, **−4.6 at k 8–16 and −5.5 at k 16–32**. Its z_d is negative in 6 of 8 bands, from k 4 up.
  - So stochastic rounding most likely carries a small real d-aligned term, B_d ≈ −1e-7 to −9e-7 of Δ. A
    miscalibrated test is the less likely reading.
  - A single arm cannot separate the two. Keep both readings in mind when P2's |z_d| < 3 criterion is unsealed.

## Archive-like states, and how big each error is

**The states are archive-like.** Median rms(Δ) is 0.2733, range [0.173, 0.381], over 64 pairs. The synthetic state in
Part XI had 0.825.

| e_Δ at 4,000 steps (sub) | all | k 1–2 | k 2–4 | k 4–8 | k 8–16 | k 16–32 | k 32–64 | k 64–128 | k 128–182 |
|---|---|---|---|---|---|---|---|---|---|
| `f64ic32` floor | 2.05e-7 | 6.20e-7 | 5.91e-7 | 2.03e-7 | 9.14e-8 | 8.80e-8 | 1.41e-7 | 2.97e-7 | 4.32e-7 |
| `lean32ek` (size only) | **9.43e-6** | 1.91e-5 | 7.87e-6 | 5.31e-6 | 5.07e-6 | 5.69e-6 | 8.02e-6 | 1.50e-5 | 1.92e-5 |
| `lean32sr` | 1.43e-5 | 5.84e-5 | 4.89e-5 | 1.97e-5 | 8.20e-6 | 7.19e-6 | 9.90e-6 | 1.94e-5 | 2.69e-5 |
| `dquant` (mechanism A alone) | 3.79e-4 | 3.65e-4 | 2.23e-3 | 6.17e-4 | 2.23e-4 | 1.31e-4 | 1.85e-4 | 3.64e-4 | 5.18e-4 |
| `lean32` | **5.84e-4** | 1.88e-3 | 3.09e-3 | 9.50e-4 | 3.14e-4 | 1.82e-4 | 2.65e-4 | 5.20e-4 | 7.50e-4 |
| production's dt sensitivity \|D(2.5e-6)\|/\|Δ\|, sub | — | 2.49e-5 | 8.50e-6 | 7.77e-6 | 1.00e-5 | 1.52e-5 | 2.71e-5 | 6.61e-5 | 1.03e-4 |
| the same, alias-free | — | 9.27e-7 | 1.71e-7 | 5.64e-8 | 6.13e-8 | 2.75e-7 | 1.50e-6 | 9.48e-6 | 2.61e-5 |

What the table shows:
- **`lean32ek`'s error does not depend on the state.**
  - It is 9.43e-6 here and 8.8e-6 on the synthetic state, although rms(Δ) is 3× smaller here.
  - The magnification argument behind P11 (×3.9/1.83, ≈ 1.9e-5) did not hold.
- **On o1.2's own frames, `lean32ek`'s error is smaller than production's dt sensitivity in every band.** The ratio is
  0.77 at k 1–2, 0.93 at k 2–4, and falls to 0.19 at k 128–182.
  - D(2.5e-6) is how much production's targets change if dt is halved. The targets already carry that much
    time-discretization error.
  - This is context, not one of item 4's gates.
- **Both the float32 error and the dt error reach the targets' large scales mostly through aliasing.**
  - `lean32ek` at k 1–2: alias-free / sub = 1.31e-6 / 1.91e-5 = **0.068**. On alias-free large scales it is 2.2× the
    float64 floor (5.95e-7).
  - Production's dt sensitivity at k 1–2: sub 2.49e-5 vs alias-free 9.27e-7, a factor of 27.
  - `lean32` differs: alias-free and sub are both 1.88e-3 at k 1–2. Its large-scale error is real, not folded.
- **Error spectra.** The max/min spread of mean |e|² over bands is 29 for `lean32ek`, the flattest of all arms. It is
  2,639 for `lean32` (synthetic: 5 and 260), 697 for `lean32sr` and 353 for `f64ic32`.
- **How e_Δ changes with lag** (overall, 40 → 4,000 steps):
  - `lean32ek` 4.07e-5 → 9.43e-6. Its error grows more slowly than Δ does.
  - `lean32` 2.74e-4 → 5.84e-4. Its error grows faster than Δ does.

## Bias of the non-candidates (unsealed)

Each cell reads z_x / z_d · shells flagged · learnable bias (noise level). `lean32`, `lean32sr` and `f64ic32` are read
on sub frames, mechanism B on af frames.

| band | `lean32` | mechanism B (`lean32 − dquant`) | `lean32sr` | `f64ic32` |
|---|---|---|---|---|
| k 1–2 (2 shells) | −2.1 / −0.8 · 0 · 4.8e-4 (3.4e-4) | −1.2 / −0.6 · 0 · 2.1e-4 (3.5e-4) | 0.9 / 0.0 · 0 · 0 (1.9e-5) | 1.5 / −0.2 · 0 · 7e-8 (1.2e-7) |
| k 2–4 (6) | **−8.4** / 1.0 · **4** · **2.2e-3** (4.4e-4) | −3.0 / 0.5 · 0 · 4.8e-4 (4.4e-4) | −1.0 / 0.1 · 1 · 1.3e-5 (1.0e-5) | −2.7 / 0.1 · 0 · 0 (1.2e-7) |
| k 4–8 (20) | **−7.2** / −4.6 · **14** · **6.6e-4** (1.4e-4) | −3.0 / 0.4 · **7** · 2.1e-4 (1.4e-4) | −1.6 / −2.8 · 0 · 0 (3.5e-6) | 2.4 / 0.1 · 0 · 0 (3.8e-8) |
| k 8–16 (68) | **10.3** / −1.4 · **50** · **2.2e-4** (3.3e-5) | −3.0 / −3.9 · **18** · 5.4e-5 (2.9e-5) | −2.3 / **−4.6** · 1 · 7e-7 (1.1e-6) | −0.1 / −0.3 · 2 · 3e-9 (1.2e-8) |
| k 16–32 (240) | −1.4 / −7.4 · 74 · 5.2e-5 (2.1e-5) | −2.8 / −4.8 · 6 · 1.3e-5 (1.4e-5) | −0.4 / **−5.5** · 0 · 3e-7 (8.4e-7) | −1.1 / 0.0 · 1 · 1e-9 (9.7e-9) |
| k 32–64 (860) | 4.3 / −5.3 · 16 · 2.3e-5 (3.0e-5) | 2.4 / −4.4 · 3 · 5.6e-6 (1.9e-5) | −0.5 / −2.6 · 3 · 5e-8 (1.1e-6) | −2.0 / −0.8 · 3 · 9e-10 (1.5e-8) |

The learnable bias of mechanism A alone (`dquant`, af, control a) is 3.14e-4, 2.14e-3, 6.15e-4 and 1.87e-4 at k 1–2,
2–4, 4–8 and 8–16.

What it says:
- **Plain float32 is a learnable bias on archived states, not only on the synthetic one.**
  - 67–74% of shells are flagged at k 2–16.
  - Its learnable bias is 2.2e-3 of Δ at k 2–4, which is 5× its noise level; 6.6e-4 at k 4–8 (4.7×); and 2.2e-4 at
    k 8–16 (6.7×).
- **Most of it is mechanism A, the float32 `decay` table.** A and B add in quadrature to `lean32`'s learnable bias:

  | band | A ⊕ B | `lean32` |
  |---|---|---|
  | k 2–4 | 2.19e-3 | 2.22e-3 |
  | k 4–8 | 6.50e-4 | 6.57e-4 |
  | k 8–16 | 1.95e-4 | 2.21e-4 |
  | k 1–2 (2 shells) | 3.78e-4 | 4.80e-4 |

  `lean32` is read on sub frames and A and B on af frames. The two frames agree for `lean32` at k 1–2 (1.88e-3 in
  both), the only band where its af / sub split is printed.
- **This corrects one line of Part XI.** Part XI said "state rounding is the cause, not the damping table".
  - That was read off `lean32e`. Its expm1 decrement is still rounded onto a float32 state, so inside a float32 state
    the two mechanisms cannot be separated.
  - `dquant` keeps a float64 state and separates them. The rate table alone gives 72% of `lean32`'s e_Δ at k 2–4
    (2.23e-3 of 3.09e-3).
  - `lean32ek` is built to remove both: an expm1 decrement and a compensated state. Whether any bias is left in it is
    the sealed question.
- **Mechanism B (state rounding) is learnable too, but smaller.**
  - It flags 35% of shells at k 4–8 and 26% at k 8–16, and its band-level z_x is ≈ −3.0 from k 2 to k 16.
  - Its learnable bias is only 0.6–1.9× its noise level at k ≤ 16. It is detectable shell by shell but small in
    aggregate.
  - P4: LEARNING's prediction is the one the data favour.
- **Stochastic rounding removes the learnable x-bias, at a noise cost.**
  - Its |z_x| stays below 3 in every band, and its shell rate (0.42%) matches the float64 control's (0.50%).
  - Its error is 1.5× `lean32ek`'s overall and 3× at k 1–2.
  - It carries the small d-aligned term described under calibration.
  - It runs at 2.79 ms. It was not pre-registered as a candidate.

## Speed (A100; the median of 4 GPUs; GPU 3 ran 3–8% slower than the others on every arm)

| arm | ms/step | vs production 9.29 |
|---|---|---|
| `lean_u1` (bitwise production) | 6.404 | 1.45× |
| `lean32` | 3.196 | 2.91× |
| `lean32ek` | 3.346 | 2.78× |
| `lean32sr` | 2.793 | 3.33× |
| **`lean32ek_fast`** | **2.698** | **3.44×**, bitwise equal to `lean32ek` at 40 / 400 / 4,000 steps on all 4 GPUs |

If `lean32ek` is admitted, its fast form needs no further sign-off, because the two are bitwise equal.

## What the harness smoke (7823653) taught, before this run

- **Control b had two design flaws.** Both were fixed in the addendum before any archive state ran.
  - **Sign.** "z_x ≥ 3" was unsigned, but the planted shift adds damping, so α < 0. Every band failed only for that
    reason.
  - **Ratio.** α_inj / α_inj64 contains `lean32ek`'s own coefficient, so it would have voided the run exactly when
    `lean32ek` is biased.
  - **The leak.** On the smoke, z was 25.5 at k 2–4 (≈ 10 expected) and 58.7 at k 4–8 (≈ 24 expected). That hinted at
    an x-aligned part of `lean32ek`'s error on 4 near-duplicate states, and it was more leakage than the "±1 SE"
    accepted beforehand.
  - **The paired form fixed it.** Recovery α(inj − `lean32ek`) / α(inj64) cancels the candidate. It passed here at
    0.961–1.007, so the run now leaks nothing about `lean32ek`'s coefficient.
- **Negative-control calibration.**
  - With 4 clusters of near-duplicate states, the controls flagged 2.1–2.3% of shells. That motivated the reading rule.
  - With 16 archived clusters they flag 0.42–0.50%, and the rule does not trigger.
- **Production's dt sensitivity is aliasing.**
  - At k 1–2, sub vs alias-free: 2.3e-4 vs 9e-7 on the smoke, and 2.49e-5 vs 9.3e-7 here.
  - Aliasing, not the solver's time step, puts most of the large-scale dt variation into o1.2's targets.
  - On archived states the subsampled sensitivity is 9× smaller than on the smoke's state.

## Scorecard (`PREDICTIONS_taxonomy.md`, read with its addendum)

- **Confirmed (8):**
  - **1 (P 0.9).** V2 and V3 (paired form) both pass. κ is 0.99–1.04 at k < 8, recovery 0.961–1.007, power z
    5.2–17.6.
  - **3 (0.85).** `lean32` is detected at k ≤ 4.
    - Its learnable bias at k 1–2 is 4.8e-4 ≥ 1e-4.
    - 67% of shells are flagged at k 2–4 and 70% at k 4–8.
    - Caveats: the k 1–2 point estimate is only 1.4× its noise level (2 shells, |z_x| 2.1), and mechanism A there
      measured 3.1e-4, not the ≈ 7e-4 predicted. The detection itself is strong from k 2 up (z_x −8.4).
  - **6 (0.5).** `lean32ek_fast` is bitwise equal to `lean32ek` at 40 / 400 / 4,000 steps on 4 of 4 GPUs, at 2.698 ms
    (≤ 2.9).
  - **7 (0.6).** Alias split 0.068 ≤ 0.3.
  - **9 (0.6).** The MDE is within 3× of MDE_PRED in every band (ratio 0.55–1.16).
  - **10 (0.85).** Median rms(Δ) 0.2733 ∈ [0.2, 0.35].
  - **12 (0.7).** f64ic32 floor 2.05e-7 ∈ [1e-7, 2e-6].
  - **13 (0.7).** `lean_u1` 6.404 (6.40 ± 3%), `lean32ek` 3.346 (3.35 ± 3%), `lean32sr` 2.793 ≤ 5.
- **Half (2):**
  - **8 (0.6).** Stochastic rounding.
    - Passed: |z_x| < 3 in every band (max 2.3); k 1–2 e_Δ 5.84e-5 ∈ [3e-5, 1e-4]; overall error 1.52× `lean32ek`'s
      (≤ 2×).
    - The "≤ 1% of shells" criterion fails if read per band, as P2 words it: 1 of 6 at k 2–4 and 1 of 68 at k 8–16.
    - It passes if read pooled: 0.42%.
    - The addendum's rule does not trigger (pooled controls 0.46% < 1%). The criterion's wording does not settle the
      reading, so the score is half.
  - **11 (0.6).** `lean32` 5.84e-4 ∈ [5e-4, 3e-3] ✅. `lean32ek` 9.43e-6 is just under the [1e-5, 4e-5] range ❌: the
    predicted magnification did not happen.
- **Refuted:** none.
- **Sealed:** 2 (`lean32ek` has no learnable bias) and 5 (the dt order ratio). They wait for item 4's signatures.
- **Decided, no prior (4):** mechanism B has a learnable part (35% / 26% of shells at k 4–8 / 8–16), which favours
  LEARNING. NUMERICS' magnitude criterion (≤ 3× noise level) does hold; its shell criterion does not.

## What this means, and what waits on whom

- **Never use plain float32 (`lean32`, with or without the trigger / ring) for o1.2's targets.** It is now rejected on
  two independent grounds:
  - its size (Part XI);
  - its learnability (here).

  Any float32 path must avoid a float32-rounded rate table and must compensate the state. `lean32ek` does both.
- **`lean32ek` is the only float32 candidate.**
  - Its admission is item 4's decision. The unsealed facts that go into the memo: its size (≈ 9e-6 of Δ, below
    production's own dt sensitivity on o1.2's frames in every band), this run's MDE table, and both controls passing.
  - Gate B needs eps_model,b from item 2. Item 2 needs item 1's logs and checkpoints, which are still owed by the user.
- **One thing to settle before unsealing: P2's per-band shell criterion is very strict in small bands.**
  - Bands up to k 16 hold 2 / 6 / 20 / 68 shells, so one false flag is 50% / 17% / 5% / 1.5%.
  - A truly unbiased `lean32ek` would fail "≤ 1% of shells in each band up to k 64" with probability ≈ 25% at the
    nominal rate. At the observed pooled control rate of 0.46% that rises to ≈ 43%.
  - f64ic32, unbiased by construction, already fails it at k 8–16 (2 of 68).
  - Any change to that reading has to be fixed in a dated addendum before the sealed file is opened. It is the user's
    and the owners' call, not this session's.

## Unsealed (2026-11-10): gates A and B, P2 and P5

**Decisions (user):**
- The user decides item 4 ("ignore weidong/jesswan").
- Results are not hidden ("do not seal anything we don't need to hide results").
- Gate C later.

**The record.**
- **Rules:** fixed in `SQG/analysis/opt/taxonomy/ACCEPTANCE_gates_AB.md` and `gates_ab.py`, pushed as 09970f75
  **before** the file was opened.
- **The opening:** sha256 matched `6f1d0e51…78be1`.
- **Outputs:** `SQG/analysis/results/polaris_taxonomy_7824121/gates_ab/`, with `gates_ab.json` and the unsealed JSON.

**The yardstick.** The alias floor of o1.2's own frames stands in for the trained model's per-band error. No model
beats it, so gate B is conservative.
- **The numbers:** ε_floor = 0.087 / 0.033 / 0.029 / 0.042 / 0.064 / 0.132 / 0.412 / 0.730 from k 1–2 to the corner.
  The tolerance is T_b = (0.1/√n_b)·ε_floor = 1.2e-3 … 7.3e-2.
- **Cross-check:** the floor's alias fractions match TEAM_REPORT's table. Its absolute values are ≈ 1.9× the table's.
  That is unresolved (a different state sample, or a missed half-plane Parseval factor), and it cancels in these
  gates.

**Gate A (noise): every arm PASSES by more than five orders of magnitude.** Var_b(e)/A_b:

| arm | values per band | limit |
|---|---|---|
| `lean32ek` | 8e-10 to 6.1e-8 | < 1e-2 |
| dt30 | 1.4e-8 to 3.9e-8 | < 1e-2 |
| dt32 | 8e-9 to 2.2e-8 | < 1e-2 |

**Gate B (bias): every arm is NOT_PASS under the rules as written. The magnitude criteria pass; the detection
criterion fails.** For `lean32ek` (sub frames, 64 pairs, 16 clusters):

| band | T_b | \|B_x\| / \|B_d\| | z_x / z_d | learnable bias | shells flagged (limit) | verdict |
|---|---|---|---|---|---|---|
| k 1–2 | 1.2e-3 | 1.4e-6 / 6.3e-7 | −0.9 / +0.6 | 1.2e-6 | 0 of 2 (n/a) | PASS |
| k 2–4 | 4.4e-4 | 4.1e-6 / 2.3e-7 | **−12.3** / −0.6 | 4.4e-6 | **6 of 6** (0.060) | FAIL_FLOOR (B3) |
| k 4–8 | 7.1e-4 | 1.2e-6 / 7.5e-7 | **−16.4 / −6.2** | 1.4e-6 | **0.35** (0.035) | FAIL_FLOOR (B3) |
| k 8–16 | 1.9e-3 | 2.0e-7 / 5.9e-7 | **−3.9 / −10.6** | 5.8e-7 | **0.13** (0.021) | FAIL_FLOOR (B3) |
| k 16–32 | 4.7e-3 | 8e-9 / 3.8e-7 | +0.2 / **−12.2** | 3.9e-7 | **0.062** (0.013) | FAIL_FLOOR (B3) |
| k 32–64 | 1.3e-2 | 3.5e-8 / 1.9e-7 | +1.4 / **−7.7** | 1.9e-7 | **0.012** (0.009) | FAIL_FLOOR (B3) |
| k 64–128 | 4.1e-2 | 9.6e-8 / 1.6e-7 | −2.8 / −5.2 | 1.2e-7 | 0.004 (0.007) | PASS |
| k 128–182 | 7.3e-2 | 5.1e-8 / 1.6e-7 | −1.2 / −4.2 | 2.4e-7 | 0.004 (0.008) | PASS |

- **B1 and B2 pass in every band, by 100× to 3 × 10⁵×.** The bias is ≤ 4.4e-6 of Δ, against T_b ≥ 4.4e-4.
- **Every FAIL_FLOOR is B3, detection.** The bias is real and the test sees it plainly.
- **A trained model's ε cannot change a B3 failure,** because B3 does not depend on ε. Under the rules, only gate D
  (dose-response) can settle it.
- **The dt arms (8 pairs each) also read NOT_PASS on B3 alone,** in k 4–8, 32–64 and 128–182. Each miss is one or two
  shells over the limit. Every |z| ≤ 3.0, and B1/B2 hold everywhere. With 8 clusters these misses
  look like false flags; the negative-control rate was measured with 16.

**What the bias is** (`lean32ek` vs plain `lean32`, unsealed):
- **It is real, not aliasing.** On alias-free frames it is stronger in z: z_d −15.3 at k 4–8 and −23.0 at k 8–16, with
  75–88% of shells flagged at k 4–32.
- **Its shape.** The learnable bias is ≈ 4–6e-7 of Δ at k ≤ 32, mostly d-aligned and negative: the 0.02 delta comes
  out ≈ 5e-7 too small. `lean32sr` showed a similar d-term (B_d −1e-7 to −9e-7), which points at the float32 tendency
  they share. That is a hypothesis, not measured.
- **The x-aligned part at k 2–8 appears only on the subsampled frames** (af z_x −0.8 / +1.2), so it is folded in by
  aliasing.
- **The team's theory bound was too low.** It expected ≤ ~1e-7 of Δ. Measured: up to 4.4e-6 (sub, k 2–4) and ≈ 5e-7
  (af).
- **It is 73–509× smaller than plain float32's in every band** (`lean32` learnable 2.2e-3 at k 2–4, `lean32ek`
  4.4e-6).

**Scored:**
- **P2 (P 0.8): REFUTED.** |z| reaches 16.4; shells are flagged above 1% at k 2–64; learnable bias > MDE in every
  band except k 1–2.
  `lean32ek` does have a detectable learnable bias.
- **P5 (P 0.6): CONFIRMED.** The dt order ratio D80/D30 is −1.42 to −1.51 in all 8 bands, inside [−1.6, −0.55]. The
  value identifies **first order** (theory −1.50): o1.2's dt sensitivity is the split between the AB3 step and the
  damping step, not AB3's third-order truncation.
- **The final item-3 scorecard:** **9 confirmed** (1, 3, 5, 6, 7, 9, 10, 12, 13), 2 half (8, 11), **1 refuted (2)**.
  P4: LEARNING.

**What it means for the decision (the user's).**
- **Under the pre-set rules,** `lean32ek_fast` is **not admitted yet**. Gate A passes; gate B fails detection only.
  The rules send it to gate D: does a model trained on it absorb a ≈ 5e-7 to 4e-6 bias, and does that matter over
  5–50 time-unit rollouts? Gate D needs stored frames and a trained model.
- **Deciding on magnitude instead** (B1 and B2 pass by ≥ 100×) is a rule change made after seeing the data. It is the
  user's to make, but it must be recorded as post hoc.
- **What is at stake:** 191 h (`lean_u1`, bitwise) vs 94 h (`lean32ek_fast`) per full o1.2 schedule on 14 A100s
  (Part XIII).

---

# Part XIII — o1.2 itself on Polaris A100s, and a full run's A100 wall-clock (job 7835278, 2026-11-10)

**Ask (user, 2026-11-10):** "can we not run our own checkpoints and generate logs". Then: "we don't want to compare it
to the H200, we know we can't get the same results, but we can extrapolate the speeds from A100".

Until now, every Polaris number came from harnesses that mimic o1.2's loop. This job runs **o1.2's own file**
(`SQG/o1.2-TrainUViT_m16_tendency.py`, sha256 `94cbe1dc…`, unedited on disk). It runs it short, once per solver, on
one debug node. From the measured pieces it projects a full o1.2 schedule on A100s. No number here is compared with
the H200 runs.

**Run.**
- **Job:** `SQG/analysis/opt/o12run/run_o12_polaris.pbs`, 1-node debug, x3004c0s7b1n0, 2026-11-10 18:09–18:41
  (1,934 s). It ended `O12_JOB_DONE failures=0`.
- **Predictions and gates:** `SQG/analysis/opt/o12run/PREDICTIONS_o12.md`, committed before submission.
- **Results:** `SQG/analysis/results/polaris_o12_7835278/`: the three rank-0 logs, stderr, the wandb stand-in's
  jsonl, and `o12_extrapolation.json`.
- **Model checkpoints** (`*_best.pt` / `*_last.pt`, ~470 MB each) stay in `$MEMBER_ROOT/sqg_profile/o12_7835278/`.

## How o1.2 runs here (`o12_polaris.py`)

**Launch.** `torchrun --standalone --nproc_per_node=4`, one rank per A100, as o1.2's own launcher.
`CUDA_VISIBLE_DEVICES=0,1,2,3` must be set, because o1.2 remaps it to the rank's GPU only when it lists several
devices.

**Patches.** o1.2's source is compiled with 34 patches. Each must match exactly once and keeps the line count, so
tracebacks point at the file's real lines.
- **Environment.**
  - Paths.
  - Strict IEEE fp32 in torch 2.3.1's spelling (`allow_tf32 = False`, matmul precision "highest"), because o1.2's
    `fp32_precision` API is torch ≥ 2.6. o1.2's own `check_fp32()` still runs and passed (conv 2.2e-6, matmul 4.1e-7,
    attention 1.9e-6).
  - The vendored solver's names. o1.2 was written against a newer `A-SQG_data.py` (`RUN_CHECKPOINT_DIR`, and
    DT_SAVE = 0.02). Ours has DT_SAVE 0.0025 and BLOCK_T = 0.02, which is exactly 0.02 in float64, so o1.2's
    `LAG == 0.02` assert keeps its `==`.
  - The staged members 1–4.
- **Stand-ins.**
  - sqg-env has no zarr, wandb or matplotlib. The replacements: a read-only zarr stand-in for the val store; wandb
    writes a jsonl file; the per-epoch plot is skipped, since it only went to wandb.
  - o1.2's first-segment check against `train_m16.zarr` becomes **the same bitwise check against job 7824121's
    production frames** of the same checkpoints, at k = 100, 200, 300, 400.
- **Knobs** (logged as `O12_KNOB`):

  | knob | value | o1.2 |
  |---|---|---|
  | batch | 16 | 32, which runs out of memory on 40 GB |
  | fill | 600 pairs | 19,604 |
  | schedule | 4 epochs × 30 steps | 250 × 781 |
  | N_VAL | 64 | 200 |
  | BUF | 16,384 | 262,144 |
  | solver | `ref` (o1.2's own call), `lean_u1`, `lean32ek_fast` | — |

  BUF changes nothing computed: the run makes fewer than 3,000 pairs, so the buffer never wraps.
- **Val store** (`build_val_store.py`). The 16 archived states' production frames, 0.02 apart: 64 pairs. They come
  from the training members, so **the val losses below are in-sample and not a science number.**

## Results (A100; rank 0; medians of the per-epoch medians over epochs 1–3, 29 step intervals each)

| | `ref` | `lean_u1` | `lean32ek_fast` |
|---|---|---|---|
| first segment vs production frames (4 per rank) | **bitwise** (PASS) | **bitwise** (PASS) | max \|Δθ\| 2.6e-3 (info) |
| steady s / optimizer step | **4.897** | **3.4925** | **1.7107** |
| o1.2's own per-epoch s/step (epochs 1–3) | 4.900–4.901 | 3.494–3.496 | 1.713–1.715 |
| step p10 – p90 | 4.895 – 4.917 | 3.492 – 3.508 | 1.708 – 1.733 |
| steady fill s / ring frame (40 solver steps) | 0.37283 (9.32 ms/step) | 0.25709 (6.43 ms/step) | 0.10910 (2.73 ms/step) |
| implied ML step = s_step − 12 × fill s/frame | 0.423 | 0.407 | 0.402 |
| validation / `save_last` / `save_best` per epoch (s) | 0.27 / 0.75 / 0.73 | 0.27 / 0.74 / 0.72 | 0.27 / 0.78 / 0.74 |
| checkpoint load (first segment) / start-up to fill (s) | 2.6 / 49.8 | 1.0 / 13.9 | 1.0 / 12.7 |
| peak PyTorch GPU memory | 20.4 GiB | 20.4 GiB | 20.4 GiB |

The model has 40.81 M parameters. All three arms trained identically to 4 decimals: train loss 0.0861 → 0.055, val
≈ 0.058 (in-sample).

What the table shows:
- **The real loop is a pure sum, as Part III found with the harnesses.** Each step is 12 ring frames of solver work
  plus one ML step of 0.40–0.42 s. Nothing overlaps.
  - The in-loop solver is 0.3–1.1% slower than the isolated benchmark (9.32 vs 9.29 ms, 6.43 vs 6.40, 2.73 vs 2.70).
  - So a solver speed-up passes into o1.2's step time one for one.
- **The difference 2.6e-3 is the expected float32 drift, not a wiring fault.**
  - It is the largest pixel difference over frames 4,000–16,000 steps out, on 4 GPUs.
  - The taxonomy harness's lean32ek-vs-production difference after 4,000 steps is 2.6e-4 to 1.5e-3 per pair (median
    5.3e-4), and the fast form's alone is 4.0e-4 to 9.5e-4.
  - θ has rms 8.5 and a typical maximum of 44, so the rms difference is ≈ 2.5e-6 of θ.
- **Per-epoch overheads are negligible.** Validation, two saves and the checkpoint loads come to under 2 s per epoch.

## A full o1.2 schedule on A100s (`o12_extrapolate.py`; the formula is in its docstring and PREDICTIONS_o12.md)

The formula's terms:
- start-up;
- fill: 19,704 frames × fill rate;
- steps × (s_step + 5 ms for cross-node DDP, from 7815637);
- 250 × (validation of 200 pairs + both saves);
- one checkpoint load per segment.

| solver | 14 A100s / 4 nodes, batch 16: **o1.2's global batch 224 and its 195,250 steps** | 16 A100s / 4 nodes (170,750 steps, batch 256) | 4 A100s / 1 node (683,500 steps, batch 64) |
|---|---|---|---|
| `ref` (production) | **268.4 h** = 11.2 d; 1,074 node-h; 3,757 GPU-h | 235.0 h; 940 node-h | 933.1 h; 933 node-h |
| `lean_u1` (bitwise, no sign-off needed) | **191.4 h** = 8.0 d; 765 node-h; 2,679 GPU-h | 167.5 h; 670 node-h | 665.1 h; 665 node-h |
| `lean32ek_fast` (needs item 4) | **93.9 h** = 3.9 d; 376 node-h; 1,315 GPU-h | 82.2 h; 329 node-h | 326.0 h; 326 node-h |

How to read it:
- **The step term is 98.9–99.7% of every projection.** Fill is 0.6–2.0 h, segment loads 0.1–1.2 h, validation and
  saves 0.1–0.2 h.
- **GPU-hours per run hardly depend on the layout.**
  - `ref` ≈ 3,750, `lean_u1` ≈ 2,680, `lean32ek_fast` ≈ 1,315 GPU-h.
  - Every rank makes its own pairs, so a fixed sample budget costs fixed GPU-seconds.
  - More GPUs buy wall-clock, not GPU-hours.
- **Only the 14-GPU column is o1.2's schedule.** The other two change the global batch (64 / 256 vs 224), so they
  are different trainings, timed at the same per-step cost.
- **Savings against `ref` on 14 A100s:**
  - `lean_u1` −29% (268 → 191 h), with no numerics change.
  - `lean32ek_fast` −65% (268 → 94 h), if item 4 admits it.

**Not in the projection:**
- **Restarts.**
  - Polaris walltimes are well under 191–268 h (preemptable: 72 h), so a full run is a chain of ≥ 3–4 jobs.
  - o1.2 always resumes from `best.pt`. Each restart re-runs the epochs after the best one, rewinds OneCycle, and
    refills the buffer (2 h for `ref`).
  - Team-plan item 7 (resume from `last.pt`, guarded) removes most of that.
- **The decoupled layout** (round 1c: equal GPU-seconds, shorter steps).

## Scorecard (`PREDICTIONS_o12.md`)

- **Confirmed (7):**
  - **1 (0.85).** All 3 arms reached `O12_RUN_DONE` with finite losses; `O12_TEST_OK` compiled under 3.10; o1.2's
    `check_fp32()` passed.
  - **3 (0.7).** Steady s/step within ±7% of 0.480 × ms + 0.42: 4.897 vs 4.88; 3.4925 vs 3.49; 1.7107 vs 1.72. All
    within 0.6%.
  - **4 (0.7).** Fill s/frame ÷ (40 × ms) = 1.003 / 1.004 / 1.011, in [1.00, 1.10].
  - **5 (0.8).** Validation 0.26–0.84 s ≤ 5; `save_last` 0.74–0.80 s ≤ 10.
  - **6 (0.7).** Peak 20.4 GiB ∈ [19, 22].
  - **7 (0.6).** Start-up 12.7–49.8 s ≤ 180.
  - **8 (0.8).** The step term is ≥ 98.9% of every projection (≥ 95%).
- **Half (1):**
  - **2 (0.8).** G3 passed: `ref` and `lean_u1` are bitwise equal to production on all 4 ranks. But `lean32ek_fast`'s
    2.6e-3 is above my band [1e-7, 1e-3]. The band was set without θ's scale (max ≈ 44). The cross-check above shows
    the value is the expected drift.
- **Refuted:** none.

## What this means, and what a full Polaris run would still need

- **o1.2's real code now runs on Polaris A100s,** in strict fp32, with production-bitwise frames (`ref`, `lean_u1`).
  The logs it writes are o1.2's own format: `o12_extrapolate.parse()` reads them, and the same parser reads
  production logs too (item 1).
- **A science-grade full run of our own** would also need:
  1. **Host memory.** BUF 262,144 is 137 GB per rank, so 4 ranks need ~548 GB on a 512 GB node. Either shrink BUF
     (that changes sampling, so it's the owners' call) or store each frame once (137 → ~69 GB per rank, an exact
     fix).
  2. **Checkpoints.** 3,281 of the 3,500 archived ones, ≈ 1.3 TB. We have 16, which wrap around. Eagle had ≈ 4.4 TB
     free on 11-10.
  3. **A held-out val store**, for example `val_m16.zarr` from Weidong's machine, or val members we generate.
  4. **Allocation and restarts.** ≈ 765 node-h with `lean_u1` on 4 nodes, as a preemptable chain. Restart-from-best
     makes resume-from-last (item 7) worth doing first.

# Part XIV — The MD-style energy-budget test: does `lean32ek_fast` close the energy budget as float64 does? (job 7836358, 2026-11-11)

**Ask (user, 2026-11-11):** "yes do the MD debug job". The test was offered in ANALOGUES.md §2.6:
- Molecular-dynamics GPU codes accept a precision model on long-run energy conservation against double precision.
- AMBER SPFP, OpenMM, GROMACS and LAMMPS all use single-precision forces with higher-precision accumulation, which is
  `lean32ek`'s design.
- Our system is forced and dissipative, so nothing is conserved. The analogue is the **closure of the SPE and VIE
  budgets**.

The math (budgets, why they close exactly in continuous time, the integrating-factor term, the controls) is in
`SQG/analysis/opt/energy/MATH_energy.md`.

**Run.**
- **Job:** `SQG/analysis/opt/energy/run_energy_polaris.pbs`, 1-node debug, x3005c0s37b1n0, 2026-11-11 04:54–05:37
  (2,585 s). It ended `ENERGY_JOB_DONE failures=0`. The deadline guard (+52 min) was not needed: every arm finished.
- **Predictions and decision rule:** `SQG/analysis/opt/energy/PREDICTIONS_energy.md`, committed before submission
  (bc296f00).
- **Results:** `SQG/analysis/results/polaris_energy_7836358/`: `energy_summary.json`, `energy_table.csv`, the worker
  json and logs. The per-call budget npz (0.3 MB per arm) stay in `$MEMBER_ROOT/sqg_profile/energy_7836358/`.
- **Change from the offer, made before the run:** 4 archived states × 0.5 TU (100,000 steps) instead of ≥ 10 TU. Ten
  time units of `lean_u1` alone is 3.6 h per state.

## Design

- **States.** Each GPU starts from one archived checkpoint: members 1–4 at t = 100, 200, 300, 400 (o1.2's loading
  recipe).
- **Arms.** Each runs 2,500 calls of 40 steps:

  | arm | role |
  |---|---|
  | `lean_u1` | float64 reference, bitwise production |
  | `lean32ek_fast` | the candidate |
  | `f64ic32` | null: float64 from the state rounded once to complex64 |
  | `dquant` | calibration: `lean_u1` with only its decay table rounded to float32, so its energy source is known exactly |
  | `lean32` | plain float32 |

- **Budget terms.** After every call, `budget()` evaluates them in float64 on the arm's complex128 state.
- **Statistic.** For X = SPE and VIE:
  - R_X(τ) = X(τ) − X(0) − ∫(P_X − ε_X) dt (Simpson, h = 2e-4 TU);
  - r = R/∫P;
  - the paired spurious source s = r(arm) − r(`lean_u1`).
- **Rule.** PASS_E if max over states |s(`lean32ek_fast`)| ≤ T_E = 1e-4 for both invariants. The verdict is valid
  only if all of these hold:
  - the null ≤ 1e-5;
  - `dquant` within ±10% of its predicted source;
  - `lean_u1` frames bitwise equal to gate D's production frames;
  - `budget()` equal to `GD.diagnostics`.

## Validity: all checks pass

- **Production identity.** `lean_u1`'s o1.2 frames equal gate D's stored production frames bit for bit: 24 of 24
  checks, calls 0, 1, 10, 100, 1,000 and 2,500 on all 4 states.
- **`budget()` vs `GD.diagnostics`.** Max relative difference 4.2e-16 over every 50th call of every arm.
- **Stability.** Every arm stayed finite, with CFL ≤ 0.374.
- **Null `f64ic32`.** max |s| = 6.2e-13 (SPE), 2.9e-13 (VIE). The paired design resolves a source at ~1e-12 of the
  injection.
- **Calibration `dquant`.** Measured s / predicted ŝ_A = 0.999997 (SPE), 1.0000002 (VIE), on the means. Per state
  the ratio is 1.0000 to 4 decimals.
  - The harness recovers a known spurious source of ~1.1e-2 to six digits.

## Results (τ = 0.5 TU; 4 states: member 1 t100, member 2 t200, member 3 t300, member 4 t400)

| | SPE | VIE |
|---|---|---|
| r(`lean_u1`), the float64 scheme's own residual | −3.49e-6, −4.08e-6, −2.85e-5, −7.30e-6 (mean −1.09e-5) | −5.22e-7, −6.17e-7, −5.76e-7, −6.03e-7 (mean −5.80e-7) |
| r / r̂_IF (integrating-factor term) | 0.930, 0.921, 1.129, 0.942 | 0.995, 0.994, 1.031, 0.991 |
| quadrature bound \|Simpson − trapezoid\| / ∫P | ≤ 1.6e-8 | ≤ 1.9e-8 |
| s(`f64ic32`), null | max 6.2e-13 | max 2.9e-13 |
| s(`dquant`) = ŝ_A | −1.13e-2 | −1.18e-2 |
| s(`lean32`) | −1.37e-2 (s/ŝ_A = 1.22) | −1.38e-2 (s/ŝ_A = 1.17) |
| **s(`lean32ek_fast`)** | **−4.22e-6, −5.54e-6, +1.38e-6, +1.28e-6** (mean −1.77e-6; max 5.5e-6 = 0.055 T_E) | **−3.40e-6, −4.67e-6, +1.22e-6, +5.3e-7** (mean −1.58e-6; max 4.7e-6 = 0.047 T_E) |
| `lean32ek_fast`'s damping-table term ŝ_E | −3.9e-9 to −9.2e-9 | −4.8e-9 to −1.5e-8 |
| paired time means of X, `lean32ek_fast` vs `lean_u1` | ≤ 4.5e-7 relative | ≤ 3.7e-7 relative |

**Verdict: PASS_E.** `lean32ek_fast`'s float32 arithmetic creates or destroys at most 5.5e-6 of the injected SPE and
4.7e-6 of the injected VIE over 100,000 steps: 18–21× below T_E.

What the numbers show:
- **The float64 scheme's own budget error is the integrating factor's.**
  - The residual is −3.5e-6 to −2.9e-5 (SPE) and −5.2e-7 to −6.2e-7 (VIE) of the injection.
  - The first-order term −(Δt/2) ∫ Σ λ w Re(θ̂* T̂)/N⁴ dt / ∫P explains it to 0.92–1.13 (SPE) and 0.99–1.03 (VIE).
  - So the scheme itself loses slightly more energy than its diagnosed dissipation, as derived in MATH_energy.md
    Part 3. Quadrature is 1e-3 of that.
- **SPE's dissipation is mostly hypofriction in these states.**
  - Time means over the run, `lean_u1`: μ/k² does 62–79% of ε_SPE and 97–98% of ε_VIE.
  - The transfer-weighted damping rate ⟨Q⟩/⟨ε⟩ is 2.2–8.4 per TU for SPE and 0.21–0.34 for VIE.
  - That is why the scheme's residual is ~1e-5 and not the ~1e-4 I estimated from the viscous scale (prediction 3).
  - Relaxation times X̄/P̄: 2.4–3.4 TU for both invariants.
- **Plain float32 fails by a factor of ~140 over T_E.**
  - `lean32`'s budget error is ~1.4e-2 of the injection. That is ~1,300× (SPE) and ~24,000× (VIE) the float64
    scheme's own, and 2,500× the candidate's worst state.
  - 82–85% of it is mechanism A, the float32 decay table (s/ŝ_A = 1.17–1.22).
  - The remainder, state rounding (mechanism B), is −5.3e-3 to +1.9e-3 per state. Part XII found the same split
    for the learnable bias.
- **The candidate's source is real, small, state-dependent, and does not grow.**
  - Real: it is ≥ 10⁶ × the null.
  - Small: 9–93 float32 unit roundoffs (u = 5.96e-8) of the injection.
  - State-dependent: it is negative on members 1 and 2 and positive on members 3 and 4.
  - Not growing: on members 1 and 2 it is already −5.5e-6 / −4.4e-6 at τ = 0.02 and −4.2e-6 / −5.5e-6 at τ = 0.5. A
    steady fraction of the injection, not an accumulating drift.
  - Its damping table contributes only ~1e-8. So it comes from the float32 tendency arithmetic (FFTs, the products,
    ψ̂ from the float32 head) breaking the exact conservation of the nonlinear term *(inference: not separated by
    this job)*.
- **The climate effect is ≲ 6e-6 of the mean** *(estimate)*.
  - By the linear-response reading (MATH_energy.md Part 6), a source s·P shifts the mean of X by ≈ s·X̄ when the
    relaxation time is X̄/P̄.
  - The paired time means already differ by only ≤ 4.5e-7 over 0.5 TU.
  - SPE itself ranges 24–48 across the 16 archived states (32–46 over the 4 used here).

**MD's own standard ("energy conservation equivalent to double"), reported, not deciding: not met.**
- The candidate's budget error is r = r(`lean_u1`) + s.
- **SPE:** it is 2.2×, 2.4×, 0.95× and 0.82× float64's on the four states. That is comparable to double within a
  factor 2.4.
- **VIE:** float64's residual is only ~6e-7, so the candidate's is 7.5× and 8.6× it on members 1 and 2.
- EQUIV_DOUBLE (|mean s| ≤ 0.1·|mean r(`lean_u1`)|) is therefore false for both invariants.
- In absolute terms the candidate's added source is ≤ 5.5e-6 of the injection. Its total budget error, scheme
  included, is ≤ 2.8e-5 (SPE) and ≤ 5.3e-6 (VIE).
- The fair MD reading: by AMBER's "equivalent to double" standard, `lean32ek_fast` is comparable on SPE but not on
  VIE. Both are far below anything the ML step or the archive's own sampling error can see *(inference)*.

## Speed (A100, the median of 4 GPUs, budget evaluation included)

`lean_u1` 6.443 ms/step, `lean32ek_fast` 2.742, `f64ic32` 6.442, `dquant` 6.441, `lean32` 3.239.

These are 0.7%, 1.6% and 1.2% above the isolated benchmarks of `lean_u1`, `lean32ek_fast` and `lean32`. The ~1.7 ms
of budget evaluation and checks per 40-step call accounts for that.

## Scorecard (`PREDICTIONS_energy.md`)

- **Confirmed (14):**
  - **1 (0.95).** Bitwise vs gate D's frames: 24/24.
  - **2 (0.95).** `budget()` vs `GD.diagnostics`: 4.2e-16 ≤ 1e-12.
  - **4 (0.6).** r_SPE(`lean_u1`)/r̂_IF = 1.053 on the means (0.92–1.13 per state), in [0.8, 1.2].
  - **5 (0.8).** |r_VIE(`lean_u1`)| ≤ 6.2e-7 ≤ 1e-4.
  - **6 (0.9).** Null ≤ 6.2e-13 ≤ 1e-6.
  - **7 (0.75).** Calibration ratio 0.999997 / 1.0000002.
  - **8 (0.7).** |ŝ_A(`dquant`)| = 1.13e-2 / 1.18e-2 ≥ T_E.
  - **9 (0.7).** `lean32` |s| = 1.37e-2 / 1.38e-2 ≥ T_E.
  - **10 (0.5).** `lean32` s_VIE/ŝ_A = 1.17, in [0.5, 2].
  - **12 (0.9).** PASS_E.
  - **14 (0.5).** DETECTED on both invariants.
  - **15 (0.7).** `dquant` s(τ)/s(0.5) = 0.967–1.004 for τ ≥ 0.1.
  - **16 (0.8).** ms/step within ±5%: +0.7% / +1.6% / +1.2%.
  - **17 (0.7).** Paired time means ≤ 4.5e-7 ≤ 1e-3.
- **Refuted (3):**
  - **3 (0.6).** r_SPE(`lean_u1`) = −1.09e-5, not in [−5e-4, −5e-5]. I estimated the integrating-factor term with
    the viscous dissipation scale (λ ~ 100). In these states hypofriction does most of the SPE dissipation, at a
    transfer-weighted λ of 2–8. The first-order formula itself held (prediction 4).
  - **11 (0.7).** `lean32ek_fast` max |s| = 5.5e-6 / 4.7e-6, not ≤ 1e-6. The source is not the damping table (~1e-8);
    I did not estimate the float32 tendency arithmetic's share.
  - **13 (0.5).** EQUIV_DOUBLE false for both invariants (above): float64's own residual is smaller than I expected
    (prediction 3), so MD's 10% bar is tighter.

## What this means

- **The solver-side half of gate C passes for `lean32ek_fast`.** Over 100,000 steps its float32 arithmetic changes
  the SPE and VIE budgets by ≤ 5.5e-6 of the injection, with T_E = 1e-4.
- **Plain float32 fails by ~140×**, as MD's single-precision modes do. Most of that is the decay table, which
  `lean32ek`'s expm1 table fixes (ŝ_E ~ 1e-8).
- **Against MD's stricter "as good as double" bar, `lean32ek_fast` is comparable on SPE (≤ 2.4×) but not on VIE
  (up to 8.6×).** It is the user's call whether that bar matters. In absolute terms its added source is ≤ 5.5e-6 of the injection.
- **Not tested here:**
  - the ≥ 10-TU climate comparison: ~3.6 h of `lean_u1` per state, a preemptable or capacity job;
  - per-shell budgets;
  - the ML side, which is gate D (capacity job 7835786, running; its write-up becomes Part XV).

# Part XV — Gate D: do rollouts drift, or overfit, when the training data carry `lean32ek`'s bias? (job 7835786, 2026-11-10/11)

**Ask (user, 2026-11-10):** "can we check if rollouts overfit and drift?" The user chose the training dose-response
and the longer run on `capacity` ("we can submit on capacity"). Gates A and B (Part XII, "Unsealed") sent `lean32ek`
here: its bias is real and detectable, but ≥ 100× below the gate-B tolerance.

**Run.**
- **Job:** `SQG/analysis/opt/gateD/run_gateD_polaris.pbs`, 1-node `capacity`, x3004c0s37b0n0, 2026-11-10 22:40 to
  2026-11-11 23:27 (89,171 s = 24.8 node-h; run_count 1, no restart). It ended `DOSE_JOB_DONE failures=0`.
  - frames 5,285 s;
  - training 83,122 s (4 arms in parallel, one per GPU);
  - rollouts 340–360 s each, best + last per arm;
  - analysis < 1 min.
- **Predictions and decision rule:** `SQG/analysis/opt/gateD/PREDICTIONS_gateD.md`, committed before the smoke
  (smoke 7835759 passed first).
- **Results:** `SQG/analysis/results/polaris_gateD_7835786/`:
  - `best/` and `last/dose_summary.json`;
  - per-arm `*_history.json`, `*_config.json` and logs;
  - the 16 frame-segment json.
- **Left on eagle:** frames (~20 GB, `$MEMBER_ROOT/sqg_profile/gateD_frames/`), and checkpoints + rollout npz
  (`$MEMBER_ROOT/sqg_profile/gateD_v1/`).

## Design (full rule in `PREDICTIONS_gateD.md`; analysis in `analyze_dose.py`'s docstring)

- **Data.** 16 stored `lean_u1` (bitwise production) segments of 1 TU, from the archived checkpoints of members 1–4
  at t = 100, 200, 300, 400.
- **Split.** Members 1–3 train (12 segments, 58,812 pairs); member 4 is held out (val, rollout ICs).
- **Training.** o1.2's UViT, loss, optimiser and OneCycle; 195,250 steps at batch 16 (about 53 passes).
- **Arms.** Each is trained on the target y + α·B̂(x, y). B̂ is `lean32ek`'s band-level bias pattern.

  | arm | α | seed |
  |---|---|---|
  | `a0_sA` | 0 | 1 |
  | `a0_sB` | 0 | 2 (seed spread) |
  | `a1000_sA` | 1,000 (≈ plain float32's bias) | 1 |
  | `a10000_sA` | 10,000 (positive control) | 1 |

- **Rollouts.** 16 member-4 ICs × 2,500 steps (50 TU); best.pt decides, last.pt is reported.
- **PASS_D** if |Δ_1000(w)|/1000 < 0.1·N(w) in every 5-TU window, for E_low and all 8 bands. N(w) = max(seed
  spread, 2·SE_IC).

## Validity: all checks pass

- **Frames.** 16/16 segments bitwise equal to job 7824121's production frames at frames 0, 100, 200, 300 and 400.
- **Harness test.** `GATED_TEST_OK`:
  - pattern round trip 1.2e-14;
  - Parseval and indexing checks;
  - torch `bias_hat` vs numpy rel 5.5e-7 (float32).
- **Training.** All 4 arms finished 195,250 steps with no NaN or error lines.
- **Stability.** 0/16 ICs blew up in all 8 rollouts. All 2,500 steps and 10/10 windows are usable.
- **Power (the positive control).**
  - |Δ_10000| > N in 52/90 window-band cells on best.pt (E_low 9/10) and 71/90 on last.pt.
  - *Post hoc (not pre-registered):* Δ_10000 has the same sign in 10/10 windows in every band, on both checkpoints.
    The sign follows the injected α_b in the 6 bands below k = 64: energy removed at k < 16, added at 16–64. The two
    dissipation-range bands (k ≥ 64) go the other way.
  - At α = 1,000 the sign is mixed (|Δ_1000| ≤ N in 80/90 cells on best.pt), and the 10,000/1,000 ratio scatters
    (−17 to +6 in window 0). The α = 1,000 effect is below the test's resolution.
  - So the test can see a learned bias at 10,000×, but not at 1,000×.

## Results

**Training** (`DOSE_OVERFIT`; s/step = mean over the 79 val intervals):

| arm | val_clean min (step) | final / min | train final | val / train | best.pt step | s/step |
|---|---|---|---|---|---|---|
| `a0_sA` | 0.005540 (160,000) | 1.0035 | 0.001638 | 3.39 | 160,000 | 0.4249 |
| `a0_sB` | 0.005592 (165,000) | 1.0022 | 0.001679 | 3.34 | 165,000 | 0.4254 |
| `a1000_sA` | 0.005534 (160,000) | 1.0002 | 0.001612 | 3.43 | 160,000 | 0.4249 |
| `a10000_sA` | 0.005557 (195,000) | 1.00001 | 0.001644 | 3.38 | 195,000 | 0.4249 |

- **Seed spread vs dose.** The two seeds differ by 0.9% in val_clean. The doses move it by −0.1% (α = 1,000) and
  +0.3% (α = 10,000), both inside the seed spread.
- **No overfitting.** Val flattens after ~160,000 steps. The val/train gap of ~3.4 is steady and does not grow.

**Rollout drift, best.pt (deciding).** R = E/E_clim − 1 per 5-TU window; ranges are over the 10 windows.

| quantity | R_0A (model's own) | N(w) | Δ_1000 | Δ_10000 | windows \|Δ_10000\| > N | max real-bias effect | max real / (0.1 N) |
|---|---|---|---|---|---|---|---|
| E_low | −0.257 to −0.163 | 0.062–0.218 | −0.084 to +0.014 | −0.281 to −0.136 | 9/10 | 8.4e-05 | 0.009 |
| k1-2 | −0.129 to +0.086 | 0.106–0.193 | −0.048 to +0.237 | −0.452 to −0.085 | 8/10 | 2.4e-04 | 0.018 |
| k2-4 | −0.264 to −0.164 | 0.063–0.221 | −0.094 to +0.008 | −0.281 to −0.138 | 9/10 | 9.4e-05 | 0.010 |
| k4-8 | −0.215 to −0.115 | 0.095–0.425 | −0.087 to −0.009 | −0.307 to −0.162 | 3/10 | 8.7e-05 | 0.009 |
| k8-16 | −0.109 to −0.040 | 0.099–0.297 | −0.043 to +0.058 | −0.152 to −0.053 | 0/10 | 5.8e-05 | 0.002 |
| k16-32 | +0.010 to +0.091 | 0.033–0.084 | −0.019 to +0.074 | +0.038 to +0.199 | 10/10 | 7.4e-05 | 0.017 |
| k32-64 | −0.018 to +0.061 | 0.042–0.184 | −0.031 to +0.067 | +0.099 to +0.269 | 10/10 | 6.7e-05 | 0.015 |
| k64-128 | −0.352 to −0.307 | 0.055–0.136 | −0.034 to +0.015 | +0.027 to +0.102 | 1/10 | 3.4e-05 | 0.005 |
| k128-182 | −0.722 to −0.703 | 0.017–0.047 | −0.022 to −0.003 | +0.010 to +0.039 | 2/10 | 2.2e-05 | 0.009 |

**last.pt (reported).** Also PASS_D.
- The largest real / (0.1 N) is 0.032 (k8-16), a 31× margin.
- The largest real-bias effect is 4.1e-4 (k2-4).
- The model's own E_low is −0.161 to −0.072.
- Late windows have a large seed spread: E_low N reaches 0.80 at 45–50 TU, because the two seeds' last.pt rollouts
  diverge.

**Absorption κ (one step on the 512 held-out pairs).**

| | κ_1000 | κ_10000 | seed-noise reference (per 1,000 / per 10,000) |
|---|---|---|---|
| best.pt | −3.24 | 0.40 | −6.84 / −0.68 |
| last.pt | −1.70 | 0.73 | −1.98 / −0.20 |

- On best.pt, the cross-seed projection is larger than even the α = 10,000 signal. κ ≥ 0.3 meets the power rule's
  letter, but on best.pt the power rests on drift.
- On last.pt, κ_10000 = 0.73 is resolved: 3.7× its seed-noise reference.
- The paired noise (same seed, different target) was not measured separately. The α = 1,000 Δ's suggest it is about
  the size of the seed spread: training is chaotic, so pairing does not help much *(inference)*.

**Verdict: PASS_D** on best.pt (and on last.pt). With the injected bias scaled up 1,000×, the extrapolated per-unit
effect is ≤ 2.4e-4 of a band energy. That is 55× below the pass line (0.1·N) in the tightest window and band.

## Scorecard (`PREDICTIONS_gateD.md`)

- **Confirmed (5):**
  - **1 (0.9).** 16/16 segments bitwise at frames 0, 100, 200, 300, 400.
  - **2 (0.8).** All arms finished at 0.4249–0.4254 s/step mean, inside 0.38–0.46. Per-interval values were
    0.418–0.469: the first interval (startup) and a couple of others sat just over 0.46.
  - **4 (0.5).** α = 0 best.pt rollouts: 16/16 finite for 50 TU.
  - **6 (0.6).** The positive control drifts beyond N in 52/90 cells on best.pt (9/10 windows on E_low).
  - **7 (0.8).** PASS_D.
- **Refuted (3):**
  - **3 (0.6).** No overfitting: final/min = 1.0035 and 1.0022 for the α = 0 arms. Fifty-three passes over 58,812
    pairs did not overfit this UViT at o1.2's OneCycle.
  - **5 (0.6).** κ on best.pt = −3.24 and 0.40, not in [0.5, 1.5]. One-step absorption is not resolved at α =
    1,000: the seed noise projected on B̂ is ~7× the injection. At α = 10,000 it is resolved only on last.pt (0.73).
  - **8 (0.5).** On best.pt the α = 0 model's own E_low runs 16–26% low (8 of 10 windows beyond −20%). On last.pt it
    stays within ±20% (−7% to −16%).

## What this means

- **Gate D passes for `lean32ek`.** Training on data with `lean32ek`'s bias does not change how the trained model
  rolls out, at this training level.
  - Even at 1,000× the real size, the effect is inside the seed-to-seed noise.
  - The pass rests on an upper bound set by the test's noise, not on a measured effect. The true effect at 1× is at
    most ~2.4e-4 of a band energy, and probably much smaller.
- **The trained model's own errors are 2–3 orders of magnitude larger than any precision effect.**
  - E_low runs 7–26% low (16–26% on best.pt, 7–16% on last.pt).
  - The dissipation range loses 31–40% (k 64–128) and 70–73% (k 128–182) of its energy in rollouts.
  - The seed-to-seed spread alone is 0.9% in val, and the windowed-drift noise floor N(w) is 0.013–0.82.
  - Model and training choices, not solver precision, set the rollout quality here.
- **Where the gate chain stands for `lean32ek_fast`** (the pre-set chain; admitting it is the user's call):

  | test | result |
  |---|---|
  | gate A | PASS |
  | gate B | NOT_PASS on detection only (magnitude ≤ 4.4e-6 of Δ, ≥ 100× below tolerance) → gate D |
  | energy (Part XIV) | PASS_E (≤ 5.5e-6 of the injection; not "as good as double" on VIE) |
  | gate D | PASS_D |

  - Per `PREDICTIONS_gateD.md`, "With gates A and B, that is the case for admitting `lean32ek_fast`".
  - Speed (Part XIII, job 7835278): the full o1.2 schedule on 14 A100s is 94 h, vs 191 h for `lean_u1` and 268 h
    for `ref`.
- **`lean32ek` is not numerically equal to float64.** Its bias is detectable (gate B: |z| up to 16) and its energy
  residual is above float64's on VIE (Part XIV). The equivalence shown here is at the trained-model level.
- **Limits** (as pre-registered):
  - about 1/14 of o1.2's samples, on 12 TU of training data;
  - only the band-level bias pattern is injected;
  - two seeds;
  - the 1× effect is a linear extrapolation from α = 1,000.
  - A full-scale o1.2 run is the remaining test, and it is the user's call (Part XIII lists what it needs).

# Part XVI — Solver speed, research round 2: codebases, FFT kernels, time stepping, XLA, frontier (2026-11-12)

**Ask (user, 2026-11-12):** "do more deep-research explore github and other code bases and other nascent strategies
of how to further solver work can be done to decrease it's step time".

**Method.** Workflow `wf_1a24fad4-309`: 5 research tracks, each followed by an adversarial verifier that re-opened the
sources and re-did the arithmetic against Part X. No jobs were run. The full write-up and ranked plan are in
`SQG/analysis/opt/research_2026-11-12/SOLVER_SPEED_RESEARCH.md`; the raw verified findings are in
`research_findings.json` beside it.

**Findings (estimates from Part X's per-op table, not measurements):**
- **The time step is the biggest lever.**
  - o1.2's first-order dt error (Part XII, P5) comes from applying the decay once to all three AB3 history terms.
  - The exact integrating-factor AB3, `theta_{n+1} = E·[theta_n + dt·(23/12·N_n − 16/12·E·N_{n−1} + 5/12·E²·N_{n−2})]`,
    costs ~0 per step and allows 28 or 25 steps per frame instead of 40. The CFL then rises from ≤ 0.42 to ≈ 0.60 /
    0.67; dt 1e-5 hit 0.71, against an abort at 0.72.
  - Gain: 1.30–1.40× per optimizer step (94 → 67–72 h on 14 A100s). It is a numerics change, so it is gated.
- **Largest bitwise item: compact kept-mode storage**, −0.22 to −0.27 ms/step, pure JAX. The decay table carries the
  2/3 mask (`A-SQG_data.py` L326-329), so the state is exactly 0 outside the 2731 × 1366 block.
- **Custom fused FFT pipeline:** ~2.0–2.1 ms with cuFFT LTO callbacks and ~1.6–1.7 ms with cuFFTDx column passes.
  gyaradax's public LGPL-3.0 code is the template. It is blocked by jax 0.4.28 having no FFI: the options are the
  legacy custom call, or an upgrade that collides with torch 2.3.1's cuDNN pin.
- **Stacked:** 1.40–1.48× (pure JAX) to 1.56–1.78× (custom FFT), and ≤ 2.1× with CFL-adaptive n. The 0.40 s ML step is
  then half the optimizer step.
- **No surveyed open GPU QG/SQG code has a per-step technique we lack,** other than the exact-linear stepper.
  - Ruled out: tensor-core FFTs, FP64 tensor cores, Ozaki emulation, ETDRK4/AB4/leapfrog, parareal/SDC, temporal
    blocking, VkFFT, cuFFT upgrades.
  - Stochastic rounding was already measured (`lean32sr`, 2.793 ms, worse e_Δ).
- **Correction to Part III B's reading:** the 21% overlap was 21% *of the ML step* (~1.8% of wall at `ref` speed). With
  the fast solver the untested ceiling is ~23.6% of wall, so a retest is worth one debug run.

**Next (the user's decisions; listed in the write-up):**
1. gate IF-AB3 at n = 28/25 (after a wider dt ladder and an archive CFL scan), or rescue n = 32/30 first;
2. compact storage (bitwise);
3. the custom-FFT route (legacy custom call vs JAX/torch upgrade; clean-room vs LGPL);
4. the overlap retest.
