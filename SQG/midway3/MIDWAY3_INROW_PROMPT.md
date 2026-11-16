# Prompt: run the in-row SQG solver (`tilerows_inrow_if25`) on Midway3's H100s

> Paste this into a fresh Claude Code session started in the Midway3 checkout,
> `/project/rcc/mehta5/pedramh-profiling` (written 2026-11-16 from the Polaris session; Polaris facts are measured,
> Midway3 facts marked **verify** are not yet checked on Midway3).

## 0. What you are doing, and the one hard limit

Bring up the SQG environment on Midway3 and measure the in-row solver on **H100**:
1. an environment check job;
2. the standalone in-row probe on one H100 (accuracy gates + ms/step), as Polaris job 7853317;
3. o1.2 itself with `tilerows_inrow_if25` on one node (4 × H100), timed, with its full-schedule extrapolation, as
   Polaris job 7853365.

**Hard limit: one node, 4 GPUs (H100 NVL, ~94 GB each).** No multi-node work. Every o1.2 projection you report is
the 1-node configuration. (Polaris numbers for 14 A100s exist only for context.)

Nothing here changes what the solver computes (CLAUDE.md #1). The solver code is already validated on A100; on H100
the same accuracy gates (G32 / GIF, against float64) decide, not bitwise equality with Polaris (a different GPU can
pick a different cuFFT / cuFFTDx kernel).

## 1. Read first

- `CLAUDE.md` (the "Things NOT to do" list; Midway cluster facts row) and the top of `CHANGELOG.md` (2026-11-16 entries).
- `SQG/polaris/polaris_sqg_notes.md` Part XVII I (the in-row step, its measurements and gates).
- `SQG/analysis/opt/inrow/inrow_step.py`: the solver, documented line by line (module docstring: model, buffers, ops).
- `SQG/analysis/opt/o12run/tilerows_advance.py` (how o1.2 runs it) and `o12_polaris.py` (the o1.2 harness:
  patches o1.2's source in memory, `O12_*` knobs; nothing in it is Polaris-specific except names).

## 2. Git

```bash
cd /project/rcc/mehta5/pedramh-profiling
git fetch origin
git switch -c sqg-midway3-inrow origin/sqg-polaris-profiling     # >= 881e8f6b; never push to main (CLAUDE.md #9)
```
Commit small (CLAUDE.md "Development principles"); write each job's `PREDICTIONS_*.md` and scorer and commit them
**before** submitting the job; push the branch.

## 3. Rules specific to this port

- **Compute only through `sbatch`** (CLAUDE.md #3). pip installs and file setup on the login node are fine; importing
  jax/torch to test them is not: do that in the env-check job.
- **Never write under `/project/pedramh/weidong/`** (Weidong's env and data): read only. The solver
  (`SQG/fixed_ring/A-SQG_data.py`) `mkdir`s run directories under its training root at import, so always set
  `SQG_TRAINING_ROOT` to a directory you own (the harnesses patch the hard-coded `TRAINING_ROOT` line in memory).
- **Add Midway scripts beside the Polaris ones, never edit the `*_polaris.pbs`** (CLAUDE.md #7, mirrored):
  `run_inrow_polaris.pbs` → `run_inrow_midway.sh`, `run_o12_tilerows_polaris.pbs` → `run_o12_tilerows_midway.sh`,
  `sqg_env_probe.pbs` → `SQG/midway3/sqg_env_probe_midway.sh`. Keep every stage, token and output name; change only
  the scheduler header and the env-bootstrap block (§4).
- **Drop the Polaris-only pieces:** the ALCF proxy, the login pid-cap lines, `-l filesystems`, `$MEMBER_ROOT`
  discovery under `/eagle`, the `/soft/compilers/...` cuFFT 11.4.1 `LD_PRELOAD` stage (report-only on Polaris; skip
  it), and the `libcudart.so` shim (multi-node NCCL over Slingshot only).
- **Do not `module load cuda` in the deciding runs.** The CUDA libraries come from the env's `nvidia-*-cu12` wheels
  (cuFFT 11.0.2 in the Polaris copy); a module prepends `LD_LIBRARY_PATH` and can change which cuFFT loads. Record
  `LD_LIBRARY_PATH` and the cuFFT version actually loaded in every result.
- **Key on tokens, not exit codes**; read `.err` first when a job fails; results are JSON/CSV/logs only (no npz,
  checkpoints or `.nsys-rep` in git: CLAUDE.md #8).
- **Ask the user** before anything longer than ~1 h of GPU time, and before Run 3 (§8).

## 4. Environment

Pick one root you own for everything writable, e.g. `SQG_ROOT=/project/rcc/mehta5/sqg_midway3` (**verify** space:
`/project/rcc/mehta5` or scratch; the env copy is ~5 GB, pylibs ~0.7 GB, outputs small).

**Python** (wdenv is the env Polaris's `sqg-env` copies: jax 0.4.28, torch 2.3.1, pyqg-jax 0.8.1, numpy 1.26.4):
- Option A (try first): `PY=/project/pedramh/weidong/miniconda3/envs/wdenv/bin/python` (as
  `SQG/analysis/profile_solver_midway.sh`), read only. Never `pip install` into it.
- Option B (if A is missing, changed, or not readable): a copy you own, built with the same pins:
  `module load python/miniforge-25.3.0`, `conda create -p $SQG_ROOT/envs/sqg-env python=3.10`, then
  `pip install -r SQG/polaris/requirements-sqg.txt -c SQG/polaris/requirements-wdenv.txt`. Write this as
  `SQG/midway3/setup_sqg_env_midway.sh` (a copy of `SQG/polaris/setup_sqg_env.sh` minus the ALCF proxy / pid-cap
  lines; `PIP_CACHE_DIR` under `$SQG_ROOT`), and save the `pip freeze`.

**Warp and nvmath** (not in wdenv; on Polaris they live in `pip --target` dirs passed to every script as
`--warp-path` / `--nvmath-path`, so the env stays untouched):
```bash
$PY -m pip install --target $SQG_ROOT/pylibs/warp-1.17.0 warp-lang==1.17.0
$PY -m pip install --no-deps --target $SQG_ROOT/pylibs/nvmath-1.0.0 nvmath-python==1.0.0 numba-cuda==0.30.4
```
warp-lang must stay 1.17.0 (the version every Polaris number used; 1.18 removed the legacy XLA custom call). The
in-row solver needs only `nvmath.bindings.cufft` (cuFFT plans on raw pointers).

**Run-time env block** for every Midway script (replaces the Polaris bootstrap; values as on Polaris):
```bash
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=8 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=$SQG_ROOT/tmp WANDB_MODE=offline PYTHONUNBUFFERED=1
export SQG_ASQG_PY=$REPO/SQG/fixed_ring/A-SQG_data.py SQG_TRAINING_ROOT=$SQG_ROOT/AllTrainData
JOB=${SLURM_JOB_ID}
```
**Scheduler header** (CLAUDE.md cluster facts; **verify** the partition has the H100s with `sinfo -p pedramh-gpu`):
`--account=pi-pedramh --partition=pedramh-gpu --nodes=1`; one GPU: `--ntasks=1 --gres=gpu:1 --cpus-per-task=8`;
four GPUs for o1.2: `--ntasks-per-node=1 --gres=gpu:4` + `python -m torch.distributed.run --standalone
--nproc_per_node=4` (one task + torchrun, the S2S launcher shape in CLAUDE.md; o1.2 is not a Lightning job).

**Job 1, env check** (`SQG/midway3/sqg_env_probe_midway.sh`, ≤ 15 min, 4 GPUs): run `SQG/polaris/sqg_env_probe.py`
(cluster-agnostic) and add: `nvidia-smi` + `nvidia-smi topo -m`; jax devices (expect 4 × H100), torch CUDA; Warp
init from the pylibs path and a 4096-point `wp.tile_fft` kernel launched with **cuFFTDx active** (`"mathdx": 1`;
without it the tile FFT is ~4.6× slower and every speed number is invalid); `from nvmath.bindings import cufft` and
the cuFFT version it loads. Token: `SQG_MIDWAY_ENV_OK`.

## 5. Data

| item | needed by | source | size |
|---|---|---|---|
| member 1–4 checkpoints `ckpt_t0100.00 … ckpt_t0400.00.npz` | probe (member 1, t = 100), o1.2 (all 16) | Midway: `/project/pedramh/weidong/SQG/SQG_deterForcing/AllTrainData/trainingdata_m0{1..4}<LABEL_TAIL>/checkpoint/` (**verify** they exist) | 6.0 GiB |
| o1.2 val store `theta.npy` + `attrs.json` | o1.2 | Polaris only: `/eagle/projects/lighthouse-uchicago/members/mehta5/sqg_profile/o12tr_7853365/val_store/` | 20 MiB |
| taxonomy production frames (optional) | o1.2's first-segment bitwise check (INFO for the in-row solver) | Polaris only: `.../sqg_profile/taxonomy_7824121/` | 1.1 GiB |

`LABEL_TAIL=_frozen_ring_NU_B_v2_nu0.0002_B1_mu1_kf3.5_dkf0.5_seed1_N4096_dt5e-06_Tstart0_Tspin20_Tprod1100_save0.0025_pair0.02`

- **Training root:** create `$SQG_ROOT/AllTrainData/trainingdata_m0X$LABEL_TAIL/checkpoint/` for X = 1..4 and
  symlink the 16 checkpoint files from Weidong's tree into them (read-only use; the solver's checks hash the
  checkpoint's config, not its path). If any are missing on Midway, stop and ask the user to Globus the 6.0 GiB
  staged set from Polaris `/eagle/projects/lighthouse-uchicago/members/mehta5/sqg_stage/AllTrainData/`.
- **Val store:** the user copies the 20 MiB directory from Polaris (Globus or scp; their credentials) to
  `$SQG_ROOT/val_store/`. Ask for it; don't wait on it for Jobs 1–2. Skip the taxonomy frames unless the user wants
  the bitwise check (leave `O12_REF_DIR` unset: o1.2 then logs `O12_BITWISE ... NO_REFERENCE`, which is expected).

## 6. Job 2: the in-row probe on one H100

Port `SQG/analysis/opt/inrow/run_inrow_polaris.pbs` → `run_inrow_midway.sh` (stages 0–1 and the scorer; drop stage 2,
the Polaris cuFFT 11.4.1 preload). It runs `inrow_probe.py --tag prod --ckpt 1:<member-1 ckpt_t0100.00.npz>
--warp-path ... --nvmath-path ...` after `test_inrow_math.py` and `test_tilerows_math.py`. ≤ 1 h, 1 GPU.

- **Prereg first** (`PREDICTIONS_inrow_midway.md`, beside the Polaris one): the validity checks and G32 / GIF gates
  of `PREDICTIONS_inrow.md`, unchanged; your predictions for H100 ms/step.
- **Reasoning for the predictions (Polaris facts):** the step is memory-bound (kernels at 82–96% of copy speed on
  A100, `warp/WARP_BUILTINS_AUDIT_2026-11-14.md`). A100 40 GB HBM2 ~1.55 TB/s vs H100 NVL HBM3 ~3.9 TB/s (vendor
  specs) suggests up to ~2.5× faster. A100: `inrow32_if` 1.470 ms, `tilerows32_if` 1.592 ms (ratio 0.923).
- **Keep `block_dim=256`** in the deciding run. A block-dim sweep on H100 is report-only: on A100, block 512 picked a
  different cuFFTDx plan (not bit-identical), so a different block size is a different arithmetic and needs the gates.
- Results → `SQG/analysis/results/midway3_inrow_<jobid>/`; end token `IR_JOB_DONE failures=N`.

## 7. Job 3: o1.2 with `tilerows_inrow_if25` on 4 × H100

Port `SQG/analysis/opt/o12run/run_o12_tilerows_polaris.pbs` → `run_o12_tilerows_midway.sh` (tests, val store,
arms, extrapolation; `O12_VAL_STORE=$SQG_ROOT/val_store`; skip `build_val_store.py` when the copied store is used).
Knobs as Polaris job 7853365: `O12_BATCH=16 O12_START_PAIRS=600 O12_N_EPOCH=4 O12_STEPS_PER_EPOCH=30 O12_N_VAL=64
O12_BUF=16384 O12_TR_GATE=1 O12_MEMBERS=1,2,3,4`.

- **Arms:** `tilerows_if25` (same-job baseline) and `tilerows_inrow_if25` at batch 16 per GPU (comparable to Polaris'
  "1 node, 4 GPUs, batch 16": 0.8885 s/step, 169.4 h on 4 A100s). Then both at **batch 32 per GPU**, o1.2's own
  `BATCH_SIZE`, which fits in 94 GB (add `_b32` arm names to the script's `case` with `O12_BATCH=32`, as the existing
  variants pass env).
- **Extrapolation:** `o12_extrapolate.py`'s `CONFIGS` has "1 node, 4 GPUs, batch 16"; add (additively, in the prereg
  first) a "1 node, 4 GPUs, batch 32" configuration. Single node: no cross-node DDP term.
- **What to expect:** o1.2's step ≈ solver (12 frames × 25 IF steps × ms/step) + the ML step (~0.42 s on A100,
  strict IEEE FP32: o1.2 pins it on purpose, `SQG/o1.2-TrainUViT_m16_tendency.py` "Precision" block; do not enable
  TF32/bf16 here).
- **Prereg:** `PREDICTIONS_o12_inrow_midway.md` + a scorer, as `PREDICTIONS_o12_inrow.md` / `score_o12_inrow.py`.
- Results → `SQG/analysis/results/midway3_o12tr_<jobid>/`; end token `O12TR_JOB_DONE failures=N`.

## 8. Optional Run 3 (only if the user says so)

The ML step's precision on H100 (`O12_PREC=tf32|bf16`, arms `tilerows_inrow_if25_tf32` / `_bf16`, timing only):
mixprec probe 2 (`SQG/analysis/opt/mixprec/PREDICTIONS_mixprec.md`). Adopting it would change the trained model: a
science decision, not an optimization.

## 9. Report back

- `SQG/midway3/midway3_sqg_notes.md` (style: `SQG/polaris/polaris_sqg_notes.md`): every Midway3 fact you confirmed
  (partition, GPU, driver, env choice, cuFFT loaded, topology), what failed and why.
- A `CHANGELOG.md` entry per job.
- To the user: env status; H100 ms/step for `inrow32_if` vs `tilerows32_if` and the G32 / GIF verdicts; o1.2 s/step
  and full-schedule hours on 1 node × 4 H100 at batch 16 and 32, next to Polaris (169.4 h on 4 A100s at batch 16;
  48.9 h on 14 A100s for context); the branch and commits.
