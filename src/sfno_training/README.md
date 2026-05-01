# sfno_training — PlaSim → Makani SFNO trainer wrapper

Subclasses + monkey-patches stock Makani so it consumes the asymmetric
PlaSim dataset contract emitted by `src/plasim_makani_packager/`.

## ⚠️ Inference is OUT OF SCOPE

This package ships a runnable **training + validation** path only.
`PlasimTrainer._plasim_get_dataloader` raises `AssertionError` on
`mode == "inference"`. Stock `makani.utils.inference.inferencer.Inferencer`
has no slot for our 6 forcing channels when `add_zenith=False` and
would silently produce physically wrong predictions.

**Do not run any production scoring, long-rollout evaluation, or
downstream emulator runs against checkpoints from this trainer until
the follow-up `src/sfno_inference/` PR ships and passes its own
integration test.**

## What it ships

| Path | Role |
|---|---|
| `data/plasim_forcing_dataset.py` | `PlasimForcingDataset(MultifilesDataset)` — 4-tuple `(inp_state, tar, inp_forcing, tar_forcing)`. |
| `models/preprocessor.py` | `PlasimPreprocessor` — strips diagnostic from `pred` before feedback. |
| `models/stepper.py` | `PlasimSingleStepWrapper` / `PlasimMultiStepWrapper`. |
| `compat.py` | Python 3.12 timedelta shim for Makani's int64-timestamp read. |
| `trainer/plasim_trainer.py` | `PlasimTrainer` + `_plasim_get_dataloader` + `_install_plasim_patches`. |
| `train_plasim.py` | CLI mirror of `makani/train.py::main`. |
| `config/plasim_sim52_baseline.yaml` | Production SFNO config (placeholders for `OUTPUT_ROOT` / `EXP_DIR`). |
| `config/plasim_sim52_smoke.yaml` | Tiny SFNO smoke config (1 file, 1 epoch). |
| `config/plasim_sim52_tiny.yaml` | Tiny training config (MOST.0003 train, MOST.0101 valid). |
| `config/plasim_sim52_short.yaml` | Short training config (MOST.0003-0007 train, MOST.0101-0102 valid). |
| `submit_train.slurm` | Production training SLURM template. |
| `submit_smoke.slurm` | GPU smoke SLURM template — **hard gate before any production run**. |
| `submit_tiny.slurm` | Tiny training SLURM template on `amd-rtx`. |
| `submit_short.slurm` | Short training SLURM template on `amd-rtx`. |

## Quickstart (after packager has produced a dataset)

```bash
# 1. GPU smoke (HARD GATE)
OUTPUT_ROOT=$SCRATCH/AI-RES/data/makani/sim52_astro_64x128 \
EXP_DIR=$SCRATCH/AI-RES/runs/sfno_smoke \
sbatch src/sfno_training/submit_smoke.slurm

# 2. Build tiny/short subsets when needed.
scripts/build_subset_dataset.py \
    --src $SCRATCH/AI-RES/data/makani/sim52_astro_64x128 \
    --dst $SCRATCH/AI-RES/data/makani/sim52_tiny \
    --train-years 3 --valid-years 101

scripts/build_subset_dataset.py \
    --src $SCRATCH/AI-RES/data/makani/sim52_astro_64x128 \
    --dst $SCRATCH/AI-RES/data/makani/sim52_short \
    --train-years 3-7 --valid-years 101-102

# 3. Launch tiny, then short after review.
sbatch src/sfno_training/submit_tiny.slurm
sbatch src/sfno_training/submit_short.slurm
```

## How the patch surface is minimally invasive

Two module-attribute rebindings, installed by `PlasimTrainer.__init__` BEFORE `super().__init__()`:

- `makani.models.model_registry.{SingleStepWrapper, MultiStepWrapper}` → PlaSim subclasses.
- `makani.utils.training.deterministic_trainer.get_dataloader` → `_plasim_get_dataloader`.

Plus:

- Subclass override of `Trainer._set_data_shapes` to set `params.N_in_channels = 58` after stock population, with hard-asserts on every aux-feature flag that would inject extra channels (`n_history==0`, `history_normalization_mode=="none"`, `add_zenith=False`, no `input_noise`, `add_grid=False`, `add_orography=False`, `add_landmask=False`, `add_soiltype=False`).

**Zero edits to `makani/`.**

## Tests

```bash
# Fast CI tests (CPU)
.venv/bin/python -m pytest tests/sfno_training/ -v

# Slow tests (CPU, real SFNO smoke ~2 min)
.venv/bin/python -m pytest tests/sfno_training/ -v -m slow

# Phase 4b smoke must still be green (regression for the stub re-export)
.venv/bin/python -m pytest tests/plasim_makani_packager/test_multifile_loader_smoke.py -v
```

## See also

- `docs/sfno_training_implementation_plan.md` — implementation plan v4 (PR-A and PR-B; round-3 Codex review resolved).
- `docs/plasim_makani_packager_plan.md` (v9) — dataset contract.
- `skills/sfno-training/SKILL.md` — usage skill for Claude Code workflows.

## Makani version pin

Pinned to upstream `main` of `NVIDIA/makani` at commit `c970430` via editable install of `makani-src/`:

```bash
.venv/bin/pip install --no-deps -e /home1/11114/zhixingliu/AI-RES/makani-src
```

Released `makani 0.2.0` (wheel) lacks the `cache_unpredicted_features` clone fix that's on main; without the pin, the in-place `.copy_(utar)` inside `append_history` mutates the caller's xz tensor. The wrapper test keeps a defensive snapshot in case the env drifts back to an unfixed wheel.
