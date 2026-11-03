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
