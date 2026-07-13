# pedramh-profiling

Benchmarking / GPU-profiling work for the Pedram Hassanzadeh group's
probabilistic **subseasonal-to-seasonal (S2S) weather-forecasting** models.
This is a **monorepo**: three related codebases live side by side so profiling
runs, DDP/AMP settings, and NVTX/CSV instrumentation can be compared 1:1 across
them.

## Layout

| Directory        | What it is |
|------------------|------------|
| [`s2s/`](s2s/)                     | The canonical S2S model — Pangu/Plasim Earth-Specific 3D Swin Transformer with VAE ensembles and latitude-weighted CRPS. The full codebase lives under `s2s/v2.0/` (the actively-maintained, benchmark-instrumented variant). |
| [`s2s-lightning/`](s2s-lightning/) | The PyTorch **Lightning** port of S2S, restructured to mirror SNFO. It **reuses** the S2S model/losses/data loaders — it does not copy them — by importing `s2s/v2.0` at runtime. |
| [`snfo/`](snfo/)                   | The SNFO model (spectral/neural forecasting operator) — the sibling project whose Lightning layout `s2s-lightning/` is modeled on. |

### How `s2s-lightning/` and `s2s/` relate

`s2s-lightning/` contains **no copy** of the model, losses, or data loaders. Its
`LightningModule`/`LightningDataModule` import the single canonical
implementation from `s2s/v2.0`:

```python
from networks.pangu import PanguModel_Plasim   # resolved from s2s/v2.0
from utils.losses import ...                    # resolved from s2s/v2.0
from utils.data_loader_multifiles import ...    # resolved from s2s/v2.0
```

So any change under `s2s/v2.0/` is picked up by the port automatically — there
is nothing to merge or keep in sync. The port's scripts put both directories on
`PYTHONPATH` (`s2s/v2.0` → `utils`/`networks`; the port dir → `data`/`modules`/`common`)
and derive these paths from the script location, so the two directories must be
checked out together (as they always are in this repo).

## Contributing (branch → PR)

`main` is the shared, clean base. Contribute on a branch and open a PR:

```bash
git clone git@github.com:rcc-uchicago/pedramh-profiling.git
cd pedramh-profiling
git checkout -b <name>/<topic>      # e.g. anthony/snfo-flexattention
# ...work inside one of s2s/, s2s-lightning/, snfo/...
git push -u origin <name>/<topic>   # then open a PR into main
```

Each project is its own top-level directory, so work in different subtrees does
not collide. Consider enabling branch protection on `main` (repo Settings →
Branches) so all changes land through PRs.

## Data

The models train on ERA5 reanalysis stored as HDF5. The dataset is **not** in
this repo. Per-cluster paths:

| Cluster | Path |
|---------|------|
| Midway (RCC / UChicago) | `/project/pedramh/h5data/h5data` |
| Derecho (NCAR)          | `/glade/campaign/univ/uchi0014/yqsun/pangu_s2s/h5data` |
| Stampede3 (TACC)        | `/scratch/08198/tg874973/pangu-s2s/h5data` |

Each project's YAML config sets `data_dir`, `checkpoint_path`, and the mean/std
`.nc` filenames per cluster — edit these before launching a run.

## Environments & HPC scripts

There is no `pip install` package layout; scripts are launched via `torchrun` /
`srun` from inside an HPC job. The `*.sh` submission scripts hardcode
**cluster-specific** paths (conda/venv locations, `data_dir`) that you edit per
deployment — see each subdir's README.

## Security note (NGC credentials)

The NVIDIA NGC apptainer scripts under `s2s/v2.0/HPC_scripts/nvidia_*.sh` read
the NGC API key from the environment:

```bash
export NGC_API_KEY=nvapi-...   # your key, from https://ngc.nvidia.com
sbatch s2s/v2.0/HPC_scripts/nvidia_training.sh
```

**Never hardcode the key into a tracked file.** A previously-committed key was
scrubbed from this repo; if you have access to that old key, treat it as
compromised and rotate it at NGC.
