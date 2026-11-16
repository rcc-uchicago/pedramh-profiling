# SQG on Midway3's H100 NVL: the in-row solver and o1.2, timed as on Polaris

**Dates:** 2026-11-16 –
**Cluster:** Midway3 (RCC), partition `pedramh-gpu`, one node (midway3-0423), 4 × H100 NVL. **One node only**: every
o1.2 number here is 1 node, 4 GPUs.
**Branch:** `sqg-midway3-inrow` (from `sqg-polaris-profiling` 7d0ddf50). Brief: `SQG/midway3/MIDWAY3_INROW_PROMPT.md`.
Predictions + gates: `SQG/midway3/PREDICTIONS_midway3_timing.md` (committed 1c6176a3, before any H100 job).

Style as `SQG/polaris/polaris_sqg_notes.md`: every fact below is marked **confirmed** (with how) or **unverified**.

---

## Summary (2026-11-16)

Everything is built, and Jobs 1–2 are queued. No H100 number exists yet.

- **The only `pedramh-gpu` node is fully held** by `UViT-N4096-dtnum` (job 63233495, user weidongli, 4 GPUs, time
  limit UNLIMITED; it had been running 15.5 h when checked on 2026-11-16). Our jobs pend until it ends. The same
  blocker stopped jobs A/B/C on 2026-11-02 (CHANGELOG). `pedramh-gpu2` (midway3-0608) was idle, but it is
  **H200**: a different GPU, not this H100-vs-A100 comparison. Using it is the user's call.
- **Jobs 3A–3C also need the val store**: the 20 MiB `o12tr_7853365/val_store/` exists only on Polaris. The user
  must copy it to `/project/rcc/mehta5/sqg_midway3/val_store/` (Globus or scp, their credentials).

## 1. Cluster facts

| fact | value | how |
|---|---|---|
| partition / node | `pedramh-gpu` = midway3-0423 only; 32 CPUs, 515,633 MB, `gpu:4`, features `gold-6346,512g,H100` | **confirmed** `sinfo`, `scontrol show node` (login) |
| partition limits | MaxTime UNLIMITED, OverSubscribe NO (so `--exclusive` is accepted), AllowAccounts pi-pedramh | **confirmed** `scontrol show partition` |
| GPU | H100 NVL ~94 GB; NVLink within the pairs GPU0↔1 and GPU2↔3 only; host PCIe Gen4 | from CHANGELOG (refuted "H200" label) and `si/bench_midway_notes.md`; Job 1 records `nvidia-smi topo -m` |
| driver, clocks, power limit | — | **unverified**: Job 1 `gpu.csv` |
| `pedramh-gpu2` | midway3-0608, 64 CPUs, 1.5 TB, `gpu:4`, **H200**, AllowGroups tmp-pedramh-gpu2 | **confirmed** `scontrol` (not used) |
| `sbatch` env | exports the submit shell (`--export=ALL`). The login shell's `LD_LIBRARY_PATH` holds only slurm + libevent, with no CUDA module | **confirmed** (login). Every Midway SQG script fails with `ERROR SQG_CUDA_ON_LD_LIBRARY_PATH` if a CUDA path appears |
| home quota | 24.7 of 30 GB used → pip caches go under `$SQG_ROOT/pip_cache`, never `~/.cache` | **confirmed** `quota` |

## 2. Environment (Option A: wdenv, read only)

- `PY=/project/pedramh/weidong/miniconda3/envs/wdenv/bin/python` (python 3.10.18). **Confirmed** by
  `setup_sqg_midway.sh`, which `pip freeze`s it (`$SQG_ROOT/wdenv-freeze.txt`) and compares it with
  `SQG/polaris/requirements-wdenv.txt`:
  - every key pin is identical: jax / jaxlib / jax-cuda12-plugin / jax-cuda12-pjrt 0.4.28, pyqg-jax 0.8.1, torch
    2.3.1, timm 0.9.12, numpy 1.26.4, nvidia-cufft-cu12 11.0.2.54, cudnn 8.9.2.26, cublas 12.1.3.1, cuda-nvcc
    12.9.86, cuda-runtime 12.1.105;
  - the 100 non-key "differences" are conda-installed packages that `pip freeze` lists as `@ file://` with no version.
    The 3 missing names are pip / setuptools / wheel. No real version drift.
  - Option B (`setup_sqg_env_midway.sh`) was therefore not needed and not written.
- **Warp / nvmath** (pip `--target`, `--no-deps`; wdenv untouched):
  `$SQG_ROOT/pylibs/warp-1.17.0` (warp-lang 1.17.0) and `$SQG_ROOT/pylibs/nvmath-1.0.0` (nvmath-python 1.0.0 +
  numba-cuda 0.30.4). **Confirmed** installed. `--no-deps` for warp too: its only dependency is numpy, which every
  harness imports from wdenv before it puts the Warp dir on `sys.path`.
- **cuFFT actually loaded, cuFFTDx (`mathdx`) and the 4096-point `tile_fft`**: **unverified** until Job 1
  (`midway_env_extra.json`). Every timed job repeats the check as its `env` stage.
- `SQG_ROOT=/project/rcc/mehta5/sqg_midway3` (group project space; 829 of 1,010 TB used by `pedramh`, plenty).

## 3. Data

- **Checkpoints confirmed:** all 16 o1.2 checkpoints (`ckpt_t0{1,2,3,4}00.00.npz` of members 1–4, 402,853,585 B
  each) exist under `/project/pedramh/weidong/SQG/SQG_deterForcing/AllTrainData/trainingdata_m0X<LABEL_TAIL>/checkpoint/`.
  They are symlinked (read only) into `$SQG_ROOT/AllTrainData/...`, and `SQG_TRAINING_ROOT` points there.
- **Val store missing:** `$SQG_ROOT/val_store/` is empty. **Taxonomy frames:** not copied, so
  `O12_BITWISE ... NO_REFERENCE` is expected.

## 4. What runs where

| job | script | GPUs | result dir | end token |
|---|---|---|---|---|
| 1 env | `SQG/midway3/sqg_env_probe_midway.sh` | 4 | `midway3_probe_<id>` | `SQG_MIDWAY_PROBE_DONE failures=0` (+ SQG_ENV_OK, SQG_IMPORT_OK, SQG_MIDWAY_ENV_OK) |
| 2 solver | `SQG/analysis/opt/inrow/run_inrow_midway.sh` | 1 (exclusive node) | `midway3_inrow_<id>` | `IR_JOB_DONE failures=N` |
| 3A o1.2 | `SQG/analysis/opt/o12run/run_o12_midway.sh` | 4 | `midway3_o12_<id>` | `O12_JOB_DONE failures=N` |
| 3B o1.2 | `SQG/analysis/opt/o12run/run_o12_tilerows_midway.sh` | 4 | `midway3_o12tr_<id>` | `O12TR_JOB_DONE failures=N` |
| 3C b32 | the same, `--export=ALL,O12TR_ARMS=ref_b32+tilerows_inrow_if25_b32` | 4 | `midway3_o12tr_<id>` | the same |

Scoring: `python3 SQG/midway3/score_midway3_timing.py --probe ... --inrow ... --o12 ... --o12tr ... --o12b32 ...`.
Table: `python3 SQG/analysis/opt/o12run/table_o12_solvers_midway.py --inrow ... --o12 ... --o12tr ... --o12b32 ...`.
Result dirs are gitignored (`results/`), so commit them with `git add -f`, as the Polaris ones were.

## 5. Job log

- **2026-11-16** — Job 1 = **63274057** (`sqg-env-probe`, 15 min), Job 2 = **63274058** (`sqg-inrow`, afterok:63274057).
  Both pending behind job 63233495.

## 6. What failed and why

- Nothing has run yet.
