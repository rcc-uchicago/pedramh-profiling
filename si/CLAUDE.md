# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Benchmarking

Throughput benchmarks run on the cluster only — no local execution.

**Run a benchmark job:**
```bash
sbatch bench_midway.sh
```

Results are appended to `bench_results.csv` (one row per job). Columns: `step_med`, `step_p90`, `step_mean`, `step_std`, `samples_per_s`, `peak_mem_gb_max_rank`, `n_steps_counted`, plus config/run metadata.

**How it works:** `bench.py` runs the standard training loop with `BenchCallback` (in `common/bench_callback.py`). The callback places `torch.cuda.synchronize()` around each step (at `on_train_batch_start` and `on_train_batch_end`) so timing is GPU-accurate, not CPU-submit time. After 20 warmup steps it collects 80 measurements, writes the CSV row, and signals the trainer to stop.

**Key bench override:** `accumulate_grad_batches` is forced to 1 regardless of the YAML (SI_midway uses 2). Every step therefore includes an optimizer call, making all measurements uniform. Note this when comparing bench throughput to production throughput.

**Ablation knobs** (set in `bench_midway.sh` before `sbatch`):
- `SI_BENCH_WARMUP` / `SI_BENCH_STEPS` — default 20 / 80
- Edit `#SBATCH --precision` or add `export SI_PRECISION=bf16-mixed` to ablate precision
- `SI_NVTX=1` + nsys wrapper — adds NVTX ranges (`step_N`, `preprocess`, `forward_loss` inside `training_step`) and brackets the 80 measured steps with `cudaProfilerStart/Stop`. Raise `SI_BENCH_WARMUP` to 40 when using `torch.compile`.

**Nsys analysis:** After the job, transfer the `.nsys-rep` + `.sqlite` and query the SQLite directly with `sqlite3`. Look for: `forward_loss` vs unlabeled backward time ratio; top CUDA kernels by total GPU time; NCCL all-reduce share.

## Setup

```bash
conda create -n "my_env" python=3.13
conda install pip
pip install torch torchvision
pip install lightning matplotlib wandb h5py timm einops h5pickle
```

Currently running with PyTorch 2.10 and CUDA 12.8. For SFNO support:

```bash
conda install torch-harmonics
pip install -U tensorly tensorly-torch
```

## Commands

**Train a model:**
```bash
python train.py --config configs/SI_midway.yaml
```

**Validate / evaluate:**
```bash
python val.py --config configs/SI_midway.yaml --checkpoint /path/to/model.ckpt
```

**Compute bias metrics:**
```bash
python bias.py --config configs/combined_midway.yaml
```

**Submit on Midway (SLURM):**
```bash
sbatch run_midway.sh
```

**Override config at runtime (common flags):**
```
--devices 0 1 2 3     # GPU IDs to use
--seed 42
--checkpoint /path/to/last.ckpt
--description my_run_name
--wandb_mode offline   # or online / disabled
```

## Architecture

This is an AMIP-era climate forecasting codebase. The goal is medium-range global weather prediction (up to 10+ days) trained on AMIP atmospheric reanalysis data stored as per-timestep HDF5 files.

### Data flow

`data/amip_new.py` (`GetDataset`) reads from `{data_dir}/{year}_{index:04d}.h5` files. Variables are split into four groups:
- **upper-air**: 3-D pressure-level fields (temperature, winds, geopotential, humidity) — shape `(n_vars, n_levels, nlat, nlon)`
- **surface**: 2-D single-level fields (T2m, Ps, etc.)
- **diagnostic**: output-only 2-D fields (radiation fluxes, precipitation, clouds)
- **varying boundary**: time-varying forcings not predicted by the model (SST, sea-ice, solar, CO2)
- **constant boundary**: static fields loaded once (land-sea mask, orography)

`common/utils.py::assemble_input` flattens and concatenates surface + diagnostic + upper-air into a single `(b, c, h, w)` tensor for model input. `disassemble_input` reverses this. `assemble_forcing` concatenates time-varying boundary + constant boundary into the conditioning grid.

`data/normalizer.py` (the `GetDataset` instance itself) holds mean/std stats loaded from `normalize_mean.nc` / `normalize_std.nc` and exposes `surface_transform`, `upper_air_transform`, `diagnostic_transform`, `boundary_transform` (and their `_inv_transform` counterparts).

### Lightning modules

Three `LightningModule` subclasses live in `modules/`:

| Module | File | Purpose |
|---|---|---|
| `TrainModule` | `modules/train_module.py` | Single-model training and rollout validation |
| `AutoencoderModule` | `modules/ae_module.py` | Trains the downscaler (UNet + DDC scheduler) standalone |
| `CombinedModule` | `modules/combined_module.py` | Evaluation-only: chains low-res forecaster → bilinear upsample → downscaler |

All three share the same `predict` / `validation_step` / `log_losses` / `save_predictions` interface. `val.py` selects which to instantiate based on `model_name == "Combined"` or `autoencoder` flag in config.

### Models

- **DiT** (`modules/models/DiT.py`): Diffusion Transformer. Takes assembled input + timestep + conditioning grid; uses AdaLN-Zero blocks with 2D RoPE. Used as the forecaster backbone for `SI_DiT`, `SI_X`, and `FM`.
- **UNet** (`modules/models/Unet.py`): Convolutional U-Net. Used as the downscaler backbone for `x_DDC`.
- **AE** (`modules/models/AE.py`), **Decoder** (`modules/models/Decoder.py`): older autoencoder architectures.

### Diffusion/interpolant schedulers

All live under `modules/diffusion/`. Each scheduler implements `compute_loss(model, x, cond, y)` and `sample(model, x, cond)`.

| Scheduler | Model name | Notes |
|---|---|---|
| `DriftScheduler` | `SI_DiT` | Stochastic interpolant with full-res DiT |
| `DynamicInterpolant` | `SI_X` | Stochastic interpolant with low-res DiT + bilinear downsample |
| `FlowMatching` | `FM` | Flow matching with low-res DiT |
| `DataDependentInterpolant` | `x_DDC` | DDC interpolant, used with UNet downscaler |

`SI_X` and `FM` use `BilinearEncoder` to downsample inputs to ¼ resolution before the DiT, then bilinearly upsample outputs. `x_DDC` in `AutoencoderModule` reconstructs full-resolution from the low-res bottleneck.

### Combined (forecaster + downscaler) pipeline

`CombinedModule` chains: full-res input → `BilinearEncoder` (4×) → DiT forecaster (low-res stochastic rollout) → `BilinearDecoder` (4×) → UNet downscaler → full-res output. `downscaler_input: 'y_last'` feeds the model's x-prediction (cleaner) rather than the Euler-updated state `'y'`. Weights are loaded from two separate Lightning checkpoints specified via `training.forecaster_checkpoint` and `training.downscaler_checkpoint`.

### Config structure

Every YAML has three top-level keys:

```yaml
model:
    model_name: SI_X   # selects architecture + scheduler in the module
    lr: 5e-5
    SI_X:              # nested block matching model_name
        model: {...}
        scheduler: {...}
data:
    data_dir: ...
    upper_air_variables: [...]
    surface_variables: [...]
    diagnostic_variables: [...]
    ...
training:
    devices: [0,1,2,3]
    strategy: ddp
    log_dir: /path/to/logs/
    wandb_mode: online
    ...
```

Checkpoint paths go in `training.checkpoint` (resume) or `training.partial_checkpoint` (load matching weights only, skip shape mismatches). For `CombinedModule`, use `training.forecaster_checkpoint` and `training.downscaler_checkpoint`.

### Optimizer

Two optimizers are supported (set via `training.optimizer`):
- `adam`: standard Adam
- `muon`: `MuonWithAuxAdam` — Muon on high-dimensional weight matrices in the hidden blocks, AdamW on embeddings/biases/IO projections. Muon LR is set to `lr * 10`.

### Validation metrics

Latitude-weighted RMSE (`common/loss.py::latitude_weighted_rmse`) is reported for T2m, Z500, U250, T850, Q850, and precipitation at 1, 3, 5, 10-day lead times. Predictions and targets for the first batch are saved as `.pt` files to the run's `log_dir`.

### Calendar conditioning

When `return_calendar: True` in the config, the dataset returns a `(sod, doy, co2)` scalar tensor per timestep (second-of-day, day-of-year, global mean CO2). CO2 is extracted from `varying_boundary_variables[0]` and removed from the boundary grid. The DiT receives this via `scalar_embedder` / `c_scalar`. `_ModelWithScalar` is a thin wrapper used to bind per-step calendar to the model without modifying scheduler signatures.
