# Using the makani-E3SM production checkpoint (`prod128_alldata_v2`)

Step-by-step restore instructions for inference and validation — written for an
external user with no HPC stack, plus the one-command Polaris validation path.
Provenance: trained 2026-08-27, job 7566145, 128 Polaris nodes / 512 A100, all
100 epochs to `MAKANI_MN_SCALING_OK` (CHANGELOG 2026-08-27 cont. 2).

## 0. What this model is

SFNO (embed 384, 8 layers, scale-factor 3) trained on E3SMv3 SSP245-AMIP,
years 2015–2044, 1° (180×360), global batch 512, 100 epochs. **107 input
channels** (100 prognostic state + 7 prescribed forcings) → **101 output
channels** (100 state + `PRECT` diagnostic) — the same 108-field variable
contract as the PanguWeather and ai-rossby production runs. Final/best
validation loss 0.0183 (epoch 100 — validation was still improving at the
schedule bound). The network predicts the **normalized tendency** (Δstate),
NOT the state itself (`target: "tendency"`, e3sm_alldata_full.yaml:158); the
rollout wrapper adds it to the input state.

## 1. Gather the bundle (all group-readable on eagle)

| piece | path |
|---|---|
| **weights** | `/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_mn_scaling/e3sm_mn_scaling/prod128_alldata_v2/training_checkpoints/best_ckpt_mp0.tar` (1.77 GB; = epoch 100; `ckpt_mp0_v0.tar` is the same epoch, `v2`/`v1` are epochs 99/98) |
| config | `pedramh-profiling/makani_sfno/polaris/e3sm_alldata_full.yaml` |
| normalization stats | `/eagle/projects/lighthouse-uchicago/members/mehta5/data/e3sm_makani_alldata_production/stats/*.npy` (z-score means/stds + time means for the 101 targets and 7 forcings) |
| channel contract | same pack root, `metadata/data.json` (channel names + order, level table, fill conventions) |
| code | `pedramh-profiling/makani_sfno/src/sfno_training/` + `makani==0.2.0`, `torch>=2.8`, `torch_harmonics>=0.9` |

**No HPC-specific software is required.** The aws-ofi-nccl / libfabric fabric
stack this repo documents exists only to carry NCCL traffic BETWEEN nodes
during multi-node runs; single-GPU or single-node inference never touches it
(single-node multi-GPU NCCL rides NVLink; single-GPU never initializes NCCL).

## 2. Environment (any machine with a GPU; CPU suffices for inspection)

```bash
pip install torch torch_harmonics
pip install makani==0.2.0            # or the group's pinned build
export PYTHONPATH=/path/to/pedramh-profiling/makani_sfno/src:$PYTHONPATH
```

## 3. Inspect / load the raw weights (works anywhere, CPU-only)

```python
import torch
ckpt = torch.load("best_ckpt_mp0.tar", map_location="cpu")
state = ckpt["model_state"]     # weights; other keys: optimizer, scheduler,
                                # counters, loss_state_dict, comm_grid
```

State-dict keys carry a wrapper prefix; makani strips it with
`makani.utils.checkpoint_helpers.get_model_state_dict_prefix`. Restoring
through makani's own path (step 4) handles this automatically — raw
`load_state_dict` may need the prefix stripped first.

## 4. Restore into a runnable model (the supported path)

Use the fork's machinery rather than hand-building — importing
`PlasimTrainer` installs the patches that make the 107/101 contract and the
forcing-feedback rollout work (`sfno_training/trainer/plasim_trainer.py`
docstring lists the four rebinds):

```python
from sfno_training.trainer.plasim_trainer import PlasimTrainer  # installs patches
from makani.utils.YParams import YParams
params = YParams("e3sm_alldata_full.rendered.yaml", "e3sm_mn_scaling")
# point the data/stats paths in params at your copy of the pack, set
# params["resuming"] = True, then PlasimTrainer(params, world_rank=0)
# restores via Driver.restore_from_checkpoint (driver.py:348).
# See src/sfno_inference/ for a worked end-to-end example of exactly this
# restore (Stampede3-pathed, but the restore logic is portable).
```

Three semantics you MUST respect if you bypass the wrappers:

1. **Inputs are z-scored** with the pack's `global_means/stds`, in exactly the
   channel order of `metadata/data.json`. The checkpoint alone is not
   self-contained — wrong stats or order produce silently wrong physics.
2. **The output is a tendency** — add it to the input state to get the next
   state (the checkpoint's `residual_transform` layer is part of this
   contract).
3. **At each rollout step the 7 forcing channels are prescribed, not
   predicted** — refresh them from data (SST/ICE/solin vary in time; the land
   masks are static), and `PRECT` is diagnostic-only, never fed back.

## 5. Validation on Polaris (internal users — one command, debug node)

```bash
cd pedramh-profiling/makani_sfno
qsub -l select=2:system=polaris \
  -v RUN_NUM=prod128_alldata_v2,TARGET_NODES=1,SKIP_TRAIN=1,FULL=1,\
EVAL_SAMPLES=4380,WANDB=1,CONFIG_YAML=e3sm_alldata_full.yaml,\
OFI_PLUGIN=/eagle/projects/lighthouse-uchicago/members/mehta5/sw/aws-ofi-nccl-1.21.1/lib,\
OFI_NCCL_PROGRESS_MODEL=AUTO,\
PACK=/eagle/projects/lighthouse-uchicago/members/mehta5/data/e3sm_makani_alldata_production \
  polaris/polaris_makani_multinode_scaling.pbs
```

Pinning `RUN_NUM` to the existing run makes makani auto-detect the checkpoints
and restore; `SKIP_TRAIN=1` (launcher knob added 2026-08-27) runs ONLY the
validation pass — weights untouched — over the full 3-year valid split
(2045–2047, 4,380 samples) with the 3-step autoregressive rollout. ~10 min on
one node. The `OFI_*` variables are harmless single-node (NCCL stays on
NVLink); they are included so the same line scales to multi-node validation
unchanged.

## 6. Known limits

- Full rollout **inference** tooling (long forecasts, scorecards) lives in
  `src/sfno_inference` / `src/sfno_eval` and is currently pathed for
  **Stampede3**; on Polaris only validation (step 5) is wired, and stock
  makani's inference entrypoint is hard-gated off in this fork. Copying the
  bundle to Stampede3 and using the existing eval pipeline there is the
  shortest path to a full scorecard.
- Training ran bf16 autocast; fp32 inference is fine (marginally more
  accurate, ~2× activation memory).
- The checkpoint is `mp0` = one complete model (the run was pure data-parallel
  with a model-parallel group of 1) — it loads on any GPU count. Only a
  makani `legacy`-format resume cares about matching parallel layout, and
  only if you resume TRAINING in a model-parallel configuration.
- Batch-512 convergence quality vs the reference runs is an open science
  question (jesswan's read); the 102 shared wandb panels
  (`pedramh-profiling` project, run `kfodxzto`) are the comparison vehicle.
