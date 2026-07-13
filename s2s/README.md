# s2s — canonical S2S model

AI model for probabilistic **subseasonal-to-seasonal (S2S)** weather
forecasting: a Pangu/Plasim Earth-Specific 3D Swin Transformer with VAE-style
reparameterization producing ensemble members. The primary loss is
latitude-weighted CRPS plus a KL term. Trained on ERA5 reanalysis stored as
HDF5.

All active code is under [`v2.0/`](v2.0/).

## Entry points

- **`v2.0/train.py`**, **`v2.0/inference.py`** — the actively-maintained,
  benchmark-instrumented entry points. `train.py` carries the S2S_BENCH
  framework (`S2S_BENCH`, `S2S_BENCH_WARMUP`, `S2S_BENCH_STEPS`, `S2S_BENCH_CSV`),
  DDP with `find_unused_parameters=False, static_graph=True`, `TORCH_COMPILE_MODE`
  support, and live NVTX ranges. The `midway_*`/`UCAR_*`/`stampede_*` HPC scripts
  target these files.
- **`v2.0/train_optimized.py`**, **`v2.0/inference_optimized.py`** — *older*
  variants despite the name; they lack the S2S_BENCH framework and static DDP
  graph. The `nvidia_*` NGC-apptainer scripts target this pair.

## Key modules

- `v2.0/networks/pangu.py` — `PanguModel_Plasim` and the transformer blocks.
- `v2.0/utils/data_loader_multifiles.py` — `get_data_loader` / `get_infer_data`
  build the HDF5-backed `GetDataset`.
- `v2.0/utils/losses.py` — `Latitude_weighted_CRPSLoss` (the `weightedCRPS` loss)
  and `Kl_divergence_gaussians`.
- `v2.0/utils/YParams.py` — config loader, `(yaml_path, config_name)`.
- `v2.0/config/*.yaml` — experiment configs (`exp1`/`exp2`/`exp3`). `data_dir`
  and `checkpoint_path` are cluster-specific.

## Running

Launched via `torchrun` inside an HPC submission script — never directly.
`utils/` and `networks/` are imported as top-level packages, so `v2.0/` must be
on `PYTHONPATH`:

```bash
PYTHONPATH=$(pwd)/v2.0 \
torchrun --standalone --nproc_per_node=$NUM_GPUS \
    v2.0/train.py --yaml_config=v2.0/config/exp2.yaml --run_num=0100
```

Pick the HPC script for your cluster (see `v2.0/HPC_scripts/`):

```
sbatch v2.0/HPC_scripts/midway_training.sh    # Midway (SLURM)
sbatch v2.0/HPC_scripts/stampede_train.sh     # Stampede3 (SLURM)
qsub   v2.0/HPC_scripts/UCAR_training.sh       # Derecho (PBS)
sbatch v2.0/HPC_scripts/nvidia_training.sh     # NGC apptainer (needs NGC_API_KEY)
```

The environment spec is `v2.0/environment.yml`. First-time training needs
`wandb login` (or `WANDB_MODE=offline`, which the HPC scripts set).

## NGC credentials

`v2.0/HPC_scripts/nvidia_*.sh` read the NGC API key from `$NGC_API_KEY`
(`export NGC_API_KEY=nvapi-...` before `sbatch`). Do not hardcode it into a
tracked file.
