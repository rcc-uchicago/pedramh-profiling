---
name: sfno-training
description: Use for the **architecture, patch surface, and channel contract** of the SFNO trainer that consumes the PlaSim → Makani packaged dataset. Covers PlasimForcingDataset, PlasimPreprocessor, PlasimSingleStepWrapper, PlasimMultiStepWrapper, PlasimTrainer, the two module-attribute monkey-patches on stock Makani, the 58-channel input contract (52 state + 6 forcing), the 53-channel target contract (52 state + 1 diagnostic pr_6h), the `_set_data_shapes` hard-asserts, the Makani version pin, the Python 3.12 timedelta shim, and the GPU-smoke hard gate. Invoke for any task touching `src/sfno_training/*` modules / wrappers / patches or `tests/sfno_training/*`. **For HPO / global batch / LR / microbench / sweep workflow → see sibling skill `train-sfno-hpo`.** For inference / scoring → `eval-sfno-own` or `eval-sfno-5410`. For dataset packaging → `plasim-makani-packager`.
---

# sfno_training — PlaSim → Makani SFNO trainer wrapper (architecture & contract)

`src/sfno_training/` consumes the three-dataset HDF5 contract emitted by `src/plasim_makani_packager/` and trains an SFNO emulator without editing Makani core. Two module-attribute rebindings + one `_set_data_shapes` override are the entire patch surface.

> **Operator workflow** — choosing GB / LR, running the I1 short-config sweep, the I2 production microbench, validating a launch, recording deviations — lives in the sibling skill `skills/train-sfno-hpo/SKILL.md`. This skill covers the architecture and patch surface only.

## When to use this skill

- Running the GPU smoke (`sbatch src/sfno_training/submit_smoke.slurm`).
- Running tiny or short training (`submit_tiny.slurm`, `submit_short.slurm`).
- Running production training (`sbatch src/sfno_training/submit_zgplev_full.slurm`).
- Modifying any file under `src/sfno_training/` (modules, wrappers, patches) or `tests/sfno_training/`.
- Debugging the trainer-CI integration test or the validation-rollout content sentinels.
- Updating the smoke, tiny, short, or production YAML for **non-HPO** reasons (channel contract, paths, scheduler shape, aux-feature flags).
- Triaging Python 3.12 timedelta shim issues or a Makani-version-pin drift.

## When NOT to use this skill

- **HPO / GB / LR / microbench / sweep workflow** — use `skills/train-sfno-hpo/SKILL.md`. This includes any change to `batch_size`, `lr`, `lr_warmup_steps`, `scheduler_T_max`, `weight_decay`, or `optimizer_max_grad_norm` in the production YAML.
- **Inference / scoring / long-rollout evaluation**. This is **out of scope** in PR-A and PR-B. Stock `makani.utils.inference.inferencer.Inferencer` has no slot for the 6 forcing channels when `add_zenith=False` and would silently produce physically wrong predictions. `_plasim_get_dataloader` hard-fails on `mode == "inference"`. A separate `src/sfno_inference/` plan is owed.
- **Editing `makani/` core**. The patch strategy is "subclass + monkey-patch only" — escalate if a Makani API forces a core edit.

## Locked contract (plan v9)

- **Input** (58): 52 state from `/fields_state` + 6 forcing from `/forcing` (concatenated by stock `Preprocessor.append_unpredicted_features`).
- **Output** (53): 52 state + 1 diagnostic `pr_6h` from `/fields_diagnostic`.
- **Rollout feedback**: `pred[:, :52]` only. `pr_6h` is loss-only and never feeds back. Forcing at step `k+1` comes from truth via `cache_unpredicted_features`.

## Module structure

```
src/sfno_training/
├── __init__.py
├── compat.py                       # Python 3.12 get_timedelta_from_timestamp shim
├── data/
│   └── plasim_forcing_dataset.py   # PlasimForcingDataset(MultifilesDataset)
├── models/
│   ├── preprocessor.py             # PlasimPreprocessor(Preprocessor2D)
│   └── stepper.py                  # PlasimSingleStepWrapper, PlasimMultiStepWrapper
├── trainer/
│   └── plasim_trainer.py           # PlasimTrainer + _plasim_get_dataloader + _install_plasim_patches
├── train_plasim.py                 # CLI mirror of makani/train.py::main
├── config/
│   ├── plasim_sim52_smoke.yaml     # tiny SFNO, 1 file, 1 epoch (GPU smoke)
│   ├── plasim_sim52_tiny.yaml      # tiny training, MOST.0003 / MOST.0101
│   ├── plasim_sim52_short.yaml     # medium SFNO, 5 train + 2 valid files
│   └── plasim_sim52_baseline.yaml  # production SFNO
├── submit_train.slurm
├── submit_tiny.slurm
├── submit_short.slurm
├── submit_smoke.slurm              # HARD GATE before any production run
└── README.md

tests/sfno_training/
├── conftest.py                     # sys.path setup + dummy-nettype registration + packaged_dataset fixture
├── helpers.py                      # RecordingDummyModel + packager fixture builder + load/build helpers
├── test_data_loader.py
├── test_preprocessor.py
├── test_wrappers.py                # 58-in / 53-out + content sentinels
├── test_trainer_ci.py              # PlasimTrainer.train_one_epoch() (PR-B)
├── test_validation_rollout.py      # validate_one_epoch() rollout content sentinels (PR-B)
└── test_smoke_sfno_cpu.py          # tiny real SFNO @pytest.mark.slow (PR-B)
```

## Patch surface (zero edits to makani/)

`PlasimTrainer.__init__` calls `_install_plasim_patches()` BEFORE `super().__init__()`:

1. `makani.models.model_registry.SingleStepWrapper = PlasimSingleStepWrapper`
2. `makani.models.model_registry.MultiStepWrapper = PlasimMultiStepWrapper`
3. `makani.utils.training.deterministic_trainer.get_dataloader = _plasim_get_dataloader`

These targets are `from X import Y` module-scope bindings in their importers — mutable Python module attributes. The integration test (`test_trainer_ci.py`) catches any upstream import-form change as a hard failure.

## Hard asserts in `PlasimTrainer._set_data_shapes`

After the stock super-call populates shape attrs, the override hard-asserts every aux-feature flag that would otherwise inject extra channels and break the 58-channel input contract:

- `params.n_history == 0`
- `params.history_normalization_mode == "none"`
- `not params.add_zenith` (rsdt is in `/forcing`)
- `params.input_noise is None`
- `not params.add_grid`
- `not params.add_orography` (sg is in `/forcing`)
- `not params.add_landmask` (lsm is in `/forcing`)
- `not params.add_soiltype`

Then sets `params.N_in_channels = n_state + n_forcing = 58`.

## Smoke / verification commands

```bash
# Fast CI (CPU, ~6s)
.venv/bin/python -m pytest tests/sfno_training/ -v

# Slow CPU SFNO smoke (~2 min)
.venv/bin/python -m pytest tests/sfno_training/ -v -m slow

# Phase 4b regression (must stay green after stub thinning)
.venv/bin/python -m pytest tests/plasim_makani_packager/test_multifile_loader_smoke.py -v

# GPU smoke — HARD GATE before any production run
OUTPUT_ROOT=$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128 \
EXP_DIR=$SCRATCH/SFNO_Climate_Emulator/runs/sfno_smoke \
sbatch src/sfno_training/submit_smoke.slurm

# Tiny subset and launch
scripts/build_subset_dataset.py \
    --src $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128 \
    --dst $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_tiny \
    --train-years 3 --valid-years 101
sbatch src/sfno_training/submit_tiny.slurm

# Short subset and launch
scripts/build_subset_dataset.py \
    --src $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128 \
    --dst $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_short \
    --train-years 3-7 --valid-years 101-102
sbatch src/sfno_training/submit_short.slurm
```

## What NOT to do

- **Do not** edit anything under `makani/`. If a Makani API forces an edit, stop and escalate.
- **Do not** launch GPU-hours on full training before `submit_smoke.slurm` completes cleanly.
- **Do not** add silent slicing to `PlasimPreprocessor.append_history`. Hard-assert `x2.shape[1] in {n_state_channels, n_full_out_channels}`.
- **Do not** reuse a `LossHandler` across different `n_future` values — `multistep_weight` is frozen at construction (plan v9 §"Do NOT").
- **Do not** flip `n_history > 0`. The `_set_data_shapes` assert will trip; flipping it requires re-deriving `N_in_channels = n_state * (n_history + 1) + n_forcing` and auditing `PlasimForcingDataset` forcing-stacking.
- **Do not** wire inference paths. Any `mode == "inference"` call into `_plasim_get_dataloader` must continue to raise `AssertionError`.
- **Do not** flip aux-feature flags (`add_zenith`, `add_grid`, `add_orography`, `add_landmask`, `add_soiltype`, `input_noise`) without revisiting the channel-count override.
- **Do not** point stock Makani training paths at this metadata — channel-count mismatch will crash.

## Where to read more

- **Plan**: `docs/sfno_training_implementation_plan.md` — full spec, 4 rounds of Codex review.
- **Source**: `src/sfno_training/` — modules above.
- **Tests**: `tests/sfno_training/` — content-sentinel test stack.
- **Dataset contract**: `docs/plasim_makani_packager_plan.md` (v9) and `skills/plasim-makani-packager/SKILL.md`.
- **HPO / GB / LR sibling skill**: `skills/train-sfno-hpo/SKILL.md` — owns the I1 sweep, I2 microbench, GB=32 default, sqrt LR rule, and deviation-recording convention.
- **Stock Makani entry points** (do not edit):
  - `makani/models/model_registry.py:30, 186-188` (wrapper bindings + dispatch).
  - `makani/utils/training/deterministic_trainer.py:38, 109, 661` (dataloader import, _set_data_shapes call, validation rollout).

## Makani version pin

The environment is pinned to **upstream `main` of NVIDIA/makani (commit `c970430`)** via editable install of a clone at `makani-src/`. PyPI/wheel `makani 0.2.0` does NOT contain the `cache_unpredicted_features` clone fix on main (`self.unpredicted_inp_train = xz.clone() if xz is not None else None`). Without the pin, the in-place `.copy_(utar)` inside `append_history` mutates the caller's xz tensor. The wrapper test keeps a defensive snapshot in case the env drifts back to an unfixed wheel.

Reinstall after a clone update:
```bash
.venv/bin/pip install --no-deps -e /home1/11114/zhixingliu/projects/SFNO_Climate_Emulator/makani-src
```

Note: the clone directory must NOT be named `makani` at the repo root, because Python's cwd-on-sys.path treats `./makani/` (no top-level `__init__.py`) as a namespace package and shadows the editable finder's mapping.
