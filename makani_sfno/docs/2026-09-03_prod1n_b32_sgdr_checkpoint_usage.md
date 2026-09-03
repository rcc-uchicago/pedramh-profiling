# Using the makani-E3SM 1-node production checkpoint (`prod1n_b32_sgdr`)

How to load the checkpoint and run it, plus this run's provenance.

Companion to `2026-08-27_prod128_alldata_checkpoint_usage.md`. **§0-§4 of that
document still apply unchanged** (model contract, z-scoring, tendency target,
prescribed forcings). This file records this run's numbers and its own recipe.

⚠ **PROVISIONAL — job 7585080 is still training** (139/243 epochs at time of
writing). Paths and mechanics are final; the best epoch and loss will keep moving,
slowly.

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

⚠ **β₂ 0.95 and LR 2.0e-3 are load-bearing, not defaults.** The LR ceiling for this
model is **(2e-3, 3e-3]** and does not move with batch size; every arm at 3.0e-3
collapsed irreversibly, and β₂ 0.999 collapsed fastest of all. Do not "modernise"
these when retraining. → `makani_bench_report.md` §5j, §7c/§7d.

## 2. Which checkpoint

**Best:** `best_ckpt_mp0.tar` — 1.65 GiB, epoch 123, validation **0.01295**.

```
RUN=/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_mn_scaling/e3sm_mn_scaling/prod1n_b32_sgdr
${RUN}/training_checkpoints/best_ckpt_mp0.tar
```

Per-epoch weights are `ckpt_mp0_v{N}.tar` in the same directory, where
⚠ **`v{N}` holds the state after epoch N+1**.

### Snapshot ensemble — new in this run

`CosineAnnealingWarmRestarts` was chosen partly so each cycle end is an ensemble
member. prod128 had one usable checkpoint; this run has one per 20-epoch cycle:

| epoch | checkpoint | validation |
|---|---|---|
| 23 | `ckpt_mp0_v22` | 0.01402 |
| 43 | `ckpt_mp0_v42` | 0.01341 |
| 63 | `ckpt_mp0_v62` | 0.01316 |
| 83 | `ckpt_mp0_v82` | 0.01304 |
| 103 | `ckpt_mp0_v102` | 0.01299 |
| **123** | **`ckpt_mp0_v122`** | **0.01295** ← currently also `best_ckpt` |
| 143 / 163 / 183 / 203 / 223 / 243 | pending | |

Late members are highly correlated (cycle gains halve: −61, −25, −12, −5, −4 ×10⁻⁵),
so an ensemble of the last few is nearly an ensemble of one. For diversity, include
the earlier cycles despite their worse single-model loss.

## 3. Load the checkpoint and run inference

### 3a. Files you need

| # | file | why |
|---|---|---|
| 1 | `${RUN}/training_checkpoints/best_ckpt_mp0.tar` | weights (+ optimizer/scheduler/counters, unused for inference) |
| 2 | `${RUN}/config.json` | the run's own resolved config — **required** by `load_eval_params` |
| 3 | `${RUN}/global_means.npy`, `${RUN}/global_stds.npy` | z-score stats — **required**, and must be the ones this run trained with |
| 4 | `${PACK}/metadata/data.json` | channel names + order, level table, fill conventions |
| 5 | `${PACK}/` data files | only if you are rolling out against real ICs |
| 6 | `makani_sfno/src/` on `PYTHONPATH` + `makani==0.2.0`, `torch>=2.8`, `torch_harmonics>=0.9` | code |

```bash
RUN=/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_mn_scaling/e3sm_mn_scaling/prod1n_b32_sgdr
PACK=/eagle/projects/lighthouse-uchicago/members/mehta5/data/e3sm_makani_alldata_production
export PYTHONPATH=/lus/eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling/makani_sfno/src:$PYTHONPATH
```

**Items 2-4 are not optional.** The checkpoint is not self-contained: wrong stats or
wrong channel order produce silently wrong physics, not an error.

**No HPC stack is required.** The aws-ofi-nccl / libfabric machinery in this repo
carries NCCL traffic *between* nodes during training; single-GPU inference never
initializes NCCL.

### 3b. Load and roll out — the supported path

`sfno_inference` does the restore, the wrapper construction and the autoregressive
body (whose loop mirrors `validate_one_epoch` exactly, so rollout matches training):

```python
import torch
from sfno_inference.checkpoint_loader import load_eval_params, build_wrapper_from_checkpoint
from sfno_inference.rollout_driver import rollout_one_ic

RUN    = "/eagle/.../e3sm_mn_scaling/prod1n_b32_sgdr"
device = torch.device("cuda:0")

# 1. config + stats from the run dir (needs config.json, global_means/stds.npy)
eval_params = load_eval_params(RUN, K=56)          # K = rollout steps, 6 h each; 56 = 14 days

# 2. restore weights into a runnable wrapper
wrapper = build_wrapper_from_checkpoint(
    eval_params, f"{RUN}/training_checkpoints/best_ckpt_mp0.tar", device)

# 3. one K-step rollout from one initial condition -> physical units
result = rollout_one_ic(wrapper=wrapper, dataset=ds, ic_global_idx=0,
                        eval_params=eval_params, device=device)
```

⚠ **`sfno_inference` is currently pathed for the Stampede3 PlaSim track** — its
dataset helper is `PlasimForcingDataset` and `_load_run_norm_stats` defaults to
`n_out=53`. For this **101-channel E3SM** checkpoint the channel count and the
dataloader must be pointed at the E3SM pack. The restore logic (steps 1-2) is
portable as written; step 3 is what needs the dataset swap. Stock makani's
inference entrypoint is hard-gated off in this fork, so there is no other
turnkey rollout path — copying the bundle to Stampede3 and using the existing eval
pipeline remains the shortest route to a full scorecard.

### 3c. Just the weights (works anywhere, CPU-only)

```python
import torch
ckpt  = torch.load("best_ckpt_mp0.tar", map_location="cpu")
state = ckpt["model_state"]   # other keys: optimizer, scheduler, counters,
                              # loss_state_dict, comm_grid
```

Keys carry a wrapper prefix; makani strips it with
`makani.utils.checkpoint_helpers.get_model_state_dict_prefix`. Going through §3b
handles this automatically — a raw `load_state_dict` needs the prefix stripped first.

The checkpoint is `mp0` = one complete model (pure data parallelism, model-parallel
group of 1), so it loads on any GPU count. Training ran bf16 autocast; fp32
inference is fine — marginally more accurate, ~2× activation memory.

### 3d. Three semantics that silently produce wrong physics

1. **Inputs are z-scored** with this run's `global_means/stds`, in exactly the
   channel order of `metadata/data.json`.
2. **The output is a tendency** — add it to the input state to get the next state.
3. **The 7 forcing channels are prescribed at every rollout step**, not predicted —
   refresh them from data (SST/ICE/solin vary in time; land masks are static).
   `PRECT` is diagnostic-only and never fed back.

## 4. Validation on Polaris (one command)

Once 7585080 has finished:

```bash
cd /lus/eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling/makani_sfno
qsub -q debug -l select=1:system=polaris -l walltime=01:00:00 \
     -l filesystems=home:eagle \
  -v RUN_NUM=prod1n_b32_sgdr,TARGET_NODES=1,SKIP_TRAIN=1,FULL=1,LOCAL_BATCH=8,\
EVAL_SAMPLES=4380,WANDB=0,CONFIG_YAML=e3sm_alldata_full.yaml,\
OFI_PLUGIN=/eagle/projects/lighthouse-uchicago/members/mehta5/sw/aws-ofi-nccl-1.21.1/lib,\
OFI_NCCL_PROGRESS_MODEL=AUTO,\
PACK=/eagle/projects/lighthouse-uchicago/members/mehta5/data/e3sm_makani_alldata_production \
  polaris/polaris_makani_multinode_scaling.pbs
```

Validation pass only, weights untouched, over the 3-year valid split (2045-2047,
4,380 samples) with the 3-step autoregressive rollout. ~10 min on one node.

**Three ways this differs from the prod128 command — all fail quietly:**

- ⚠ **It scores the LAST epoch, not `best_ckpt`.** Resume uses
  `get_latest_checkpoint_version` (newest by mtime, `deterministic_trainer.py:266-271`);
  `best_checkpoint_path` is only ever *written*, never read. Harmless for prod128,
  where best *was* the last epoch — wrong under warm restarts. **To score a specific
  checkpoint**, seed a new dir and point `RUN_NUM` at it:
  ```bash
  DST=${RUN%/*}/score_best
  mkdir -p ${DST}/training_checkpoints
  cp ${RUN}/training_checkpoints/best_ckpt_mp0.tar ${DST}/training_checkpoints/ckpt_mp0_v0.tar
  cp ${RUN}/{config.json,metadata.json,global_means.npy,global_stds.npy} ${DST}/
  ```
  The name **must** be `ckpt_mp0_v0.tar`: `train.py:101-105` gates `resuming` on that
  exact filename, and any other name gives `resuming = False` and **trains from
  scratch with no error**. The epoch is read from counters inside the tar, not the
  filename.
- ⚠ **`WANDB=0` is required** for a seeded dir. With wandb on and resuming,
  `Driver._init_wandb` reads `<expDir>/wandb/makani_restart.yaml` (`driver.py:237-248`);
  a seeded dir has none and every rank dies at construction, presenting as a wall of
  NCCL teardown traces and `rank 0 exited with code 1`. Do **not** copy that file in —
  it would write into the training run's wandb history.
- ⚠ **Never point a job at the live run dir while it trains** — two processes in one
  expDir corrupts the run.

## 5. Memory, for sizing anything downstream

Measured with the per-epoch peak instrumentation added 2026-09-02 (makani's own
`memory footprint [GB]` is an epoch-END snapshot and understates by 11-16 GB):

| samples/GPU | global batch | peak torch | non-torch | **total** | on a 39.49 GiB A100 |
|---|---|---|---|---|---|
| 8 | 32 | 19.23 | 8.01 | **27.24 GB** | 69% — this run |
| 12 | 48 | 27.69 | 8.04 | **35.73 GB** | 90% |
| 16 | 64 | — | — | **~44 GB** | **OOM** (job 7580362) |

Fit: `peak_torch ≈ 2.31 GB + 2.12 GB per sample/GPU`; the non-torch term (CUDA
context + cuFFT/cuDNN + NCCL) is a ~8 GB fixed tax.
