# Prompt: run the in-row SQG solver (`tilerows_inrow_if25`) on Midway3's H100s

> Paste this into a fresh Claude Code session started in the Midway3 checkout,
> `/project/rcc/mehta5/pedramh-profiling` (written 2026-11-16 from the Polaris session; Polaris facts are measured,
> Midway3 facts marked **verify** are not yet checked on Midway3).

## 0. What you are doing, and the one hard limit

Bring up the SQG environment on Midway3 and measure the in-row solver on **H100**, **timed exactly the way Polaris
timed it** (§6), so every H100 number sits next to its A100 number in one table:
1. an environment check job;
2. the solver-step timings on one H100: production `ref` / `lean_u1` (as Polaris job 7814300) and the in-row probe
   with its accuracy gates, per-op split and the JAX arms (as Polaris job 7853317);
3. o1.2 itself on one node (4 × H100) with all six solver arms, `ref` to `tilerows_inrow_if25`, timed, with the
   full-schedule extrapolation (as Polaris jobs 7835278, 7840909, 7853365);
4. the H100-vs-A100 table (as `SQG/analysis/opt/o12run/table_o12_solvers.py`).

**Hard limit: one node, 4 GPUs (H100 NVL, ~94 GB each).** No multi-node work. Every o1.2 projection you report is
the 1-node configuration. (Polaris numbers for 14 A100s exist only for context.)

Nothing here changes what the solver computes (CLAUDE.md #1). The solver code is already validated on A100; on H100
the same accuracy gates (G32 / GIF, against float64) decide, not bitwise equality with Polaris (a different GPU can
pick a different cuFFT / cuFFTDx kernel).

## 1. Read first

- `CLAUDE.md` (the "Things NOT to do" list; Midway cluster facts row) and the top of `CHANGELOG.md` (the 2026-11-16
  entries).
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

## 6. The timing protocol: the same measurements, harnesses and formulas as Polaris

Use the Polaris harnesses **unchanged** (only the Midway launch scripts are new), so each H100 number is measured
the same way as the A100 number it is compared with. Three levels, then one table.

**Level 1: ms per solver step, one H100 (GPU 0).**

| arms | harness (unchanged) | method (as on Polaris) |
|---|---|---|
| `ref`, `lean`, `lean_u1` | `SQG/analysis/opt/seed1/run_seed1.py --out <dir> --budget-min 14`, `SQG_INIT=random` (Polaris round 1, job 7814300) | wall clock with `block_until_ready`: median of 3 repeats × 60 calls of 40 steps, ÷ 2,400 steps. `ref` is o1.2's own call (`lax.scan(GD.inner)` of the unmodified solver). Also logs one o1.2 optimizer-step pattern (12 calls + coarse, median of 5; information) |
| `tilerows32`, `tilerows32_if`, `inrow32`, `inrow32_if` (+ `inrow32_wfirst`, `inrow32_b512`, report only) | `SQG/analysis/opt/inrow/inrow_probe.py --tag prod` (job 7853317) | CUDA events on the solver's stream: 2 warm-up calls, then 5 batches of 10 × `run(40)`; median batch ÷ 400 = **eager**. A CUDA graph of `run(48)` replayed 5× = **graph**. **Per-op split**: 40 steps with an event after every op, median over steps 5–39 |
| `lean32ek_fast`, `if32`, `lean_u1` (JAX) | the same probe, same job | 3 warm-up 40-step calls, then median of 3 × 30 calls (`perf_counter` + `block_until_ready`) ÷ 1,200 |

The table's "ms / solver step" takes `ref` from `run_seed1.py` and every other arm from the probe (`lean_u1`, timed by
both, agreed on A100 within 0.1%: 6.4017 vs 6.4063). The IF arms run 25 steps per frame and the others 40, so compare
whole frames or o1.2 steps across schemes, not ms per step.

**Level 2: seconds per o1.2 optimizer step, one node (4 × H100).** `o12run/o12_polaris.py` under
`python -m torch.distributed.run --standalone --nproc_per_node=4`, one arm at a time, in two jobs as on Polaris:
- Job 3A, a port of `run_o12_polaris.pbs` → `run_o12_midway.sh`: `ref`, `lean_u1`, `lean32ek_fast` (Polaris
  per-arm limits 1,440 / 1,020 / 600 s; job 7835278).
- Job 3B, a port of `run_o12_tilerows_polaris.pbs` → `run_o12_tilerows_midway.sh`: `if32_25`, `tilerows_if25`,
  `tilerows_inrow_if25` (480 s each, `O12_TR_GATE=1`; jobs 7840909, 7853365).
- **The same knobs in both:** `O12_BATCH=16 O12_START_PAIRS=600 O12_N_EPOCH=4 O12_STEPS_PER_EPOCH=30 O12_N_VAL=64
  O12_BUF=16384 O12_MEMBERS=1,2,3,4 O12_VAL_STORE=$SQG_ROOT/val_store` (skip `build_val_store.py` when the copied
  store is used).
- **Pieces**, parsed by `o12_extrapolate.py` from each arm's log:
  - s_step: the median over epochs ≥ 1 of each epoch's median step interval (epoch 0 holds the compiles);
  - s_frame: steady fill seconds per ring frame;
  - t_init, t_load, t_val, and t_save_last / t_save_best.
- **Batch 32**, o1.2's own `BATCH_SIZE`, fits in 94 GB. In Job 3C, repeat at least `ref` and
  `tilerows_inrow_if25` with `O12_BATCH=32` (arm names `_b32`, through the scripts' `case` / env mechanism).

**Level 3: the full schedule.** `o12_extrapolate.py`'s fixed formula (its docstring; o1.2's constants), unchanged:
```
steps = N_EPOCH x ((N_OFFLINE_PAIRS // n) // b)
T     = t_init + t_load + (LAG + START_SEGS x PAIRS_PER_SEG) x s_frame + steps x s_step
        + N_EPOCH x (t_val x ceil(ceil(N_VAL / n) / b) + t_save_last + t_save_best)
        + ceil(steps x FRAMES_PER_STEP / FRAMES_PER_SEG) x t_load               (one node: no cross-node DDP term)
```
- Report "1 node, 4 GPUs, batch 16", the direct A100 comparison. Add a "1 node, 4 GPUs, batch 32" entry to
  `CONFIGS` additively, in the prereg first.
- The 14 / 16-GPU rows the script also prints use Polaris' measured cross-node DDP cost (job 7815637). On Midway they
  are hypothetical: never report them as Midway numbers.

**The table.** Add `SQG/analysis/opt/o12run/table_o12_solvers_midway.py` beside `table_o12_solvers.py`, with the
same columns. It reads the Midway3 result dirs and puts H100 and A100 side by side, with the H100 / A100 ratio, plus
the per-op split for both GPUs.

The Polaris rows it is compared with (A100 40 GB, measured; 1 node = 4 A100s at batch 16):

| solver | steps / frame | ms / solver step (1 A100) | o1.2 s / opt step | × ref | 1 node (h) | source jobs |
|---|---|---|---|---|---|---|
| `ref` | 40 | 9.290 | 4.897 | 1.00× | 933.1 | 7814300 / 7835278 |
| `lean_u1` | 40 | 6.406 | 3.493 | 1.40× | 665.1 | 7853317 / 7835278 |
| `lean32ek_fast` | 40 | 2.696 | 1.711 | 2.86× | 326.0 | 7853317 / 7835278 |
| `if32_25` | 25 | 2.746 | 1.240 | 3.95× | 237.0 | 7853317 / 7840909 |
| `tilerows_if25` | 25 | 1.592 | 0.926 | 5.29× | 176.6 | 7853317 / 7840909 |
| `tilerows_inrow_if25` | 25 | 1.470 | 0.8885 | 5.51× | 169.4 | 7853317 / 7853365 |

Per-op split of the `inrow32` step on A100 (job 7853317, ms):
- build_w 0.148, cufft_cols_w 0.288, cufft_cols_th 0.129, halve0 0.008;
- tile_rows_ir 0.418, cufft_cols_z 0.291, update 0.217;
- sum 1.498; eager step 1.482, graph 1.467.

**Timing hygiene** (what made the Polaris numbers comparable):
- **Exclusive hardware.** Give the 4-GPU jobs the whole node (all 4 GPUs; `--exclusive` if the partition allows);
  keep nothing else on the timed GPU; attach no profiler in timed runs.
- **Record per job:**
  - `nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit` (as
    `run_round1_polaris.pbs`'s `gpu.csv`) and `nvidia-smi topo -m`;
  - `LD_LIBRARY_PATH`, and the cuFFT / cudart actually loaded;
  - the jax / torch / warp versions.
- **Same-job baselines** (`ref` in 3A, `tilerows_if25` in 3B), so drift between jobs cannot pose as a speedup.
- **Statistics:** medians, not means; warm-ups and epoch 0 excluded.
- **Fixed arithmetic.** Nothing that changes the arithmetic (block_dim, cuFFT version, XLA / TF32 flags) is tuned in a
  timed run. Such sweeps are report-only.

## 7. Jobs, preregistration and gates

- **Job 2** (1 H100, ≤ 1 h): a port of `SQG/analysis/opt/inrow/run_inrow_polaris.pbs` → `run_inrow_midway.sh`.
  - It runs `test_inrow_math.py`, `test_tilerows_math.py`, then `seed1/run_seed1.py`, then
    `inrow_probe.py --tag prod --ckpt 1:<member-1 ckpt_t0100.00.npz> --warp-path ... --nvmath-path ...`, then
    `score_inrow.py`.
  - Drop stage 2 (the Polaris cuFFT 11.4.1 preload).
  - The validity checks and G32 / GIF gates of `PREDICTIONS_inrow.md` are unchanged; seed 1's G1–G3 likewise.
  - **Keep `block_dim=256`.** On A100, block 512 picked a different cuFFTDx plan (not bit-identical), so a different
    block size is a different arithmetic and would need the gates.
  - End token `IR_JOB_DONE failures=N`.
- **Jobs 3A, 3B** (4 H100, ≤ 1 h each) and **Job 3C** (batch 32), as §6. End tokens: `O12_JOB_DONE failures=N`
  (3A, as `run_o12_polaris.pbs`) and `O12TR_JOB_DONE failures=N` (3B, 3C).
- **Prereg first:** `SQG/midway3/PREDICTIONS_midway3_timing.md` plus a stdlib scorer, committed before Job 2.
  - Predictions per level for H100 against the A100 table.
  - Reasoning (Polaris facts + vendor specs):
    - The solver step is memory-bound (82–96% of copy speed on A100, `warp/WARP_BUILTINS_AUDIT_2026-11-14.md`).
      A100 40 GB is ~1.55 TB/s against H100 NVL ~3.9 TB/s, so up to ~2.5×.
    - The ML step (~0.42 s of A100's 0.8885 s o1.2 step) is strict IEEE FP32: o1.2 pins it on purpose
      (`SQG/o1.2-TrainUViT_m16_tendency.py`, "Precision" block). So it gains from H100's non-tensor FP32 (~3× A100's
      on paper), not from tensor cores. Do not enable TF32 / bf16 in these jobs.
- **Results** (JSON / logs only): `SQG/analysis/results/midway3_inrow_<jobid>/` (includes `seed1.json`),
  `midway3_o12_<jobid>/`, `midway3_o12tr_<jobid>/`.

## 8. Optional Run 3 (only if the user says so)

The ML step's precision on H100 (`O12_PREC=tf32|bf16`, arms `tilerows_inrow_if25_tf32` / `_bf16`, timing only):
mixprec probe 2 (`SQG/analysis/opt/mixprec/PREDICTIONS_mixprec.md`). Adopting it would change the trained model: a
science decision, not an optimization.

## 9. Report back

- `SQG/midway3/midway3_sqg_notes.md` (style: `SQG/polaris/polaris_sqg_notes.md`): every Midway3 fact you confirmed
  (partition, GPU, driver, env choice, cuFFT loaded, topology), what failed and why.
- A `CHANGELOG.md` entry per job.
- To the user:
  - the env status;
  - the gate verdicts (G32 / GIF, seed 1 G1–G3);
  - **the H100-vs-A100 table** (all six solvers: ms / solver step, o1.2 s / opt step, × ref, 1-node hours at
    batch 16, and batch 32 where measured), and the per-op split for both GPUs;
  - which part of o1.2's step shrank more, the solver or the strict-FP32 ML step;
  - the branch and commits.
  For context only: 48.9 h on 14 A100s (Polaris) is a multi-node figure with no Midway3 counterpart.
