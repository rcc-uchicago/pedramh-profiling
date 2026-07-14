# s2s-lightning — PyTorch Lightning port of S2S

A PyTorch **Lightning** restructuring of the S2S model, laid out to mirror the
sibling **SI** project. The goal is a codebase whose only material difference
from SI is the model definition (S2S's `PanguModel_Plasim`).

## Reuses `../s2s`, does not copy it

This directory contains **no copy** of the model, losses, or data loaders. The
Lightning components import the single canonical implementation from
`../s2s/v2.0`:

- `modules/train_module.py` → `from networks.pangu import PanguModel_Plasim`,
  `from utils.losses import ...`
- `data/datamodule.py` → `from utils.data_loader_multifiles import ...`

So any change under `../s2s/v2.0/` is picked up here automatically. The
`S2S_BENCH`/NVTX/CSV instrumentation and the DDP (`static_graph=True`,
`find_unused_parameters=False`) / AMP settings are preserved through Lightning
hooks and `Trainer` config.

## Layout

- `modules/` — `LightningModule`s (e.g. `train_module.py`) wrapping the model + CRPS/KL loss.
- `data/` — `LightningDataModule` (`datamodule.py`) wrapping the S2S HDF5 loaders. `data/constant_mask/*.npy` are the land/soil/topography boundary constants.
- `configs/`, `common/` — SI-style config and shared helpers.
- `train.py`, `val.py`, `bench.py`, `verify_bench.py` — entry points (`bench.py` is the benchmark/NVTX-profiling runner).
- `smoke_datamodule.py`, `smoke_train_module.py` — single-GPU smoke tests; each prints `SMOKE_OK` on success.

## Running

The Lightning components need `../s2s/v2.0` (→ `utils`/`networks`) **and** this
directory (→ `data`/`modules`/`common`) on `PYTHONPATH`. The provided scripts
derive both from their own location, so just run them from within a checkout of
this repo:

```bash
sbatch s2s-lightning/midway_smoke_datamodule.sh     # Phase-1 DataModule smoke
sbatch s2s-lightning/midway_smoke_train_module.sh    # Phase-2 2-step fit smoke
sbatch s2s-lightning/midway_bench_nsys_port.sh       # nsys profile, matched 1:1 to the v2.0 baseline
```

Interactively (from the repo root):

```bash
PYTHONPATH=s2s/v2.0:s2s-lightning python s2s-lightning/smoke_datamodule.py
```

## Environment

Needs `pytorch-lightning` in addition to the S2S environment
(`../s2s/v2.0/environment.yml`). The `*.sh` scripts `mamba activate` a
**cluster-specific** venv path — edit that line for your deployment. See
[`LIGHTNING_PORT.md`](LIGHTNING_PORT.md) for the full port write-up.
