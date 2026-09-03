# Using the makani-E3SM 1-node production checkpoint (`prod1n_b32_sgdr`)

Restore instructions for validation and inference, and the **three ways this run
differs from `prod128_alldata_v2`** — one of which will silently score the wrong
checkpoint if you carry the old recipe over unchanged.

Companion to `2026-08-27_prod128_alldata_checkpoint_usage.md`. **Everything in §0-§4
of that document still applies** (model contract, z-scoring, tendency target,
prescribed forcings, raw-weight loading, the `PlasimTrainer` restore path). This
file records only what is different plus the run's own provenance.

⚠ **PROVISIONAL — job 7585080 is still training** (139/243 epochs at time of
writing). Paths and mechanics are final; the "best" epoch and loss will keep
moving, slowly. Re-read §2 before quoting a number.

## 1. Provenance

| | `prod128_alldata_v2` | **`prod1n_b32_sgdr`** (this run) |
|---|---|---|
| job | 7566145 | **7585080** |
| hardware | 128 nodes / 512 A100 | **1 node / 4 A100** |
| node-hours | 216 | **~46** |
| global batch | 512 | **32** |
| weight updates | 8,500 | **332,424** |
| epochs | 100 | **243** |
| LR schedule | — | **2.0e-3 peak, `CosineAnnealingWarmRestarts` `T_0=20` `T_mult=1`, `min_lr` 1e-6, 3-epoch warmup from `lr_start 0.01`** |
| optimizer | — | AdamW, **β₂ 0.95**, `max_grad_norm` 32 |
| best validation | 0.018297 (epoch 100, still improving at the schedule bound) | **0.01295 (epoch 123)** — 29% better |

Model, channels, data pack and normalization are **identical** to prod128:
SFNO embed 384 / 8 layers / scale-factor 3, E3SMv3 SSP245-AMIP 2015-2044, 1°
(180×360), 107 input → 101 output channels, `target: "tendency"`.
Config: `makani_sfno/polaris/e3sm_alldata_full.yaml`.
Pack: `/eagle/projects/lighthouse-uchicago/members/mehta5/data/e3sm_makani_alldata_production`.

⚠ **β₂ 0.95 and LR 2.0e-3 are load-bearing, not defaults.** The LR ceiling for this
model is **(2e-3, 3e-3]** and does not move with batch size; every arm at 3.0e-3
collapsed irreversibly, and β₂ 0.999 collapsed fastest of all. Do not "modernise"
these when retraining. → `makani_bench_report.md` §5j, §7c/§7d.

## 2. The bundle

| piece | path |
|---|---|
| **best weights** | `.../runs/makani_mn_scaling/e3sm_mn_scaling/prod1n_b32_sgdr/training_checkpoints/best_ckpt_mp0.tar` (1.65 GiB) |
| per-epoch weights | same dir, `ckpt_mp0_v{N}.tar` — ⚠ **`v{N}` holds the state after epoch N+1** |
| config / stats / channel contract / code | unchanged from prod128 — see that doc §1 |

Root: `/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_mn_scaling/e3sm_mn_scaling/prod1n_b32_sgdr`

### Snapshot ensemble — new in this run

`CosineAnnealingWarmRestarts` was chosen partly so that **each cycle end is an
ensemble member**. prod128 had one usable checkpoint; this run has one per cycle:

| epoch | checkpoint | validation |
|---|---|---|
| 23 | `ckpt_mp0_v22` | 0.01402 |
| 43 | `ckpt_mp0_v42` | 0.01341 |
| 63 | `ckpt_mp0_v62` | 0.01316 |
| 83 | `ckpt_mp0_v82` | 0.01304 |
| 103 | `ckpt_mp0_v102` | 0.01299 |
| **123** | **`ckpt_mp0_v122`** | **0.01295** ← currently also `best_ckpt` |
| 143 / 163 / 183 / 203 / 223 / 243 | pending | |

The members are highly correlated late (gains halve each cycle: −61, −25, −12, −5,
−4 ×10⁻⁵), so an ensemble of the **later** members is nearly an ensemble of one.
If you want diversity, include the earlier cycles despite their worse single-model
loss. `CKPT_VERSIONS=250` retains every epoch on purpose — a rolling window of 3
would delete every ensemble member long before the run ends (~400 GiB at 243 epochs).

## 3. ⚠ Three differences from the prod128 recipe

### 3a. `SKIP_TRAIN=1` restores the LATEST epoch, not `best_ckpt`

`deterministic_trainer.py:266-271` resumes via `params.checkpoint_path` and
`get_latest_checkpoint_version`, which picks the newest version **by mtime**.
`best_checkpoint_path` is only ever *written* (`:402`), never read on resume.

For prod128 this was harmless: best *was* the last epoch. **Under warm restarts it
is not** — best is a cycle end, the newest is mid-cycle and worse. Carrying the old
command over unchanged scores a checkpoint that is not the best one.

**To score a specific checkpoint** (best, or any snapshot member), seed a scratch
expDir with it named **`ckpt_mp0_v0.tar`**:

```bash
SRC=/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_mn_scaling/e3sm_mn_scaling
DST=${SRC}/score_best            # any NEW name; must not be an existing run
mkdir -p ${DST}/training_checkpoints
cp ${SRC}/prod1n_b32_sgdr/training_checkpoints/best_ckpt_mp0.tar \
   ${DST}/training_checkpoints/ckpt_mp0_v0.tar
cp ${SRC}/prod1n_b32_sgdr/{config.json,metadata.json,global_means.npy,global_stds.npy} ${DST}/ 2>/dev/null
```

The name matters: `train.py:101-105` gates `resuming` on the existence of **exactly**
`ckpt_mp0_v0.tar`. Seed it under any other name and the job reports
`resuming = False` and **trains from scratch with no error** — the loss curve looks
like a fresh run, not a broken restore. The epoch is not read from the filename;
counters live inside the tar. `makani_sfno/polaris/submit_batch_fork_ab.sh` does
exactly this seeding and can be copied.

### 3b. A scratch/forked expDir needs `WANDB=0`

With wandb on **and** `resuming=True`, `Driver._init_wandb` reads
`<expDir>/wandb/makani_restart.yaml` to rejoin the original run
(`driver.py:237-248`). A seeded dir has no such file and **every rank dies at
construction** — the visible symptom is a wall of NCCL/TCPStore teardown traces and
`rank 0 exited with code 1`, which reads as a fabric fault and is not.

⚠ Do **not** fix this by copying `makani_restart.yaml` in: that makes the scoring job
**write into the training run's wandb history**.

### 3c. Do not point any job at the live run directory

prod128 was finished when its doc was written. While 7585080 is training, a second
job pinned to `RUN_NUM=prod1n_b32_sgdr` puts two processes in one expDir and
corrupts the run. Either wait for it to finish, or use the seeded copy in §3a.

## 4. Validation on Polaris

Once 7585080 has finished, the prod128 one-liner works with the name swapped:

```bash
cd /lus/eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling/makani_sfno
qsub -q debug -l select=1:system=polaris -l walltime=01:00:00 \
     -l filesystems=home:eagle \
  -v RUN_NUM=prod1n_b32_sgdr,TARGET_NODES=1,SKIP_TRAIN=1,FULL=1,\
EVAL_SAMPLES=4380,WANDB=0,CONFIG_YAML=e3sm_alldata_full.yaml,\
LOCAL_BATCH=8,\
OFI_PLUGIN=/eagle/projects/lighthouse-uchicago/members/mehta5/sw/aws-ofi-nccl-1.21.1/lib,\
OFI_NCCL_PROGRESS_MODEL=AUTO,\
PACK=/eagle/projects/lighthouse-uchicago/members/mehta5/data/e3sm_makani_alldata_production \
  polaris/polaris_makani_multinode_scaling.pbs
```

…but per §3a that scores the **last** epoch. To score the **best**, point `RUN_NUM`
at the seeded `score_best` dir instead.

Runs the validation pass only, weights untouched, over the 3-year valid split
(2045-2047, 4,380 samples) with the 3-step autoregressive rollout. ~10 min on one
node. `OFI_*` are harmless single-node (NCCL stays on NVLink) and are kept so the
line scales to multi-node unchanged.

## 5. Memory, for sizing anything downstream

Measured with the per-epoch peak instrumentation added 2026-09-02 (makani's own
`memory footprint [GB]` is an epoch-END snapshot and understates by 11-16 GB):

| samples/GPU | global batch | peak torch | non-torch | **total** | on a 39.49 GiB A100 |
|---|---|---|---|---|---|
| 8 | 32 | 19.23 | 8.01 | **27.24 GB** | 69% — this run |
| 12 | 48 | 27.69 | 8.04 | **35.73 GB** | 90% |
| 16 | 64 | — | — | **~44 GB** | **OOM** (job 7580362) |

Fit: `peak_torch ≈ 2.31 GB + 2.12 GB per sample/GPU`; the non-torch term (CUDA
context + cuFFT/cuDNN + NCCL) is a ~8 GB fixed tax. Training ran bf16 autocast;
fp32 inference is fine (~2× activation memory).

## 6. Known limits

Unchanged from prod128 §6: full rollout **inference** tooling
(`src/sfno_inference`, `src/sfno_eval`) is **Stampede3-pathed**, stock makani's
inference entrypoint is hard-gated off in this fork, and on Polaris only the
validation pass above is wired. Copying the bundle to Stampede3 and using the
existing eval pipeline there remains the shortest path to a full scorecard.

The checkpoint is `mp0` = one complete model (pure data parallelism, model-parallel
group of 1); it loads on any GPU count.

**Open for the science owner:** batch 32 was chosen on measurement — at equal data
it beats batch 48 by 9-19% early and the gap closes to ~0.25% late (fork A/B from
epoch 74, jobs 7588118/7588120), while batch 48 costs 12-27% more wall time and 90%
of the card. Whether 0.01295 at batch 32 is *scientifically* better than prod128's
0.018297 at batch 512 — rather than merely lower on this validation metric — is
jesswan's call, and the per-channel lwrmse panels are the vehicle.
