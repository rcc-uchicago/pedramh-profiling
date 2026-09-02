# pedramh-profiling

GPU bring-up, profiling and optimization for the Pedram Hassanzadeh group's
weather-forecasting models, on **RCC Midway (SLURM)** and **ALCF Polaris (PBS)**.

> ## 🧭 Returning after a while? Start here (updated 2026-09-02)
>
> **The focus is now `makani_sfno/` on Polaris.** The repo has grown well beyond the
> three-model S2S monorepo this README used to describe.
>
> **Read in this order:**
> 1. **[`TODO.md`](TODO.md)** — what to do next, priority first. One file, P0 at the top.
> 2. **[`CHANGELOG.md`](CHANGELOG.md)** — what happened, newest first. The shared memory:
>    measured results, failed approaches, and corrections.
> 3. **[`makani_bench_report.md`](makani_bench_report.md)** — the current measured evidence
>    (start at §0a for column units, §5 for the results that set the production config).
> 4. **[`polaris_pbs_notes.md`](polaris_pbs_notes.md)** — **the durable Polaris reference.**
>    Cluster facts, queues, env repair. *Cite this, not a handoff prompt.*
> 5. **[`CLAUDE.md`](CLAUDE.md)** / **[`DESIGN.md`](DESIGN.md)** — how to work here / what & why.
>
> **Where the current work is:** makani-SFNO trains on E3SM on **1 Polaris node**, batch 32
> (`polaris_makani_1node_production_handoff.md`). The headline result is counter-intuitive and
> worth knowing before planning anything: **at a fixed batch size, adding nodes made training
> slower** — so production moved from 128 nodes to 1, for 16× the optimizer updates at ~20% of
> the node-hours.

## Layout

| Directory | What it is |
|---|---|
| [`makani_sfno/`](makani_sfno/) | **NVIDIA makani SFNO — the current focus.** Trains on E3SM; Polaris launchers, converters and the scaling harness live in `polaris/`. A `git subtree`. |
| [`physicsnemo_ai_rossby/`](physicsnemo_ai_rossby/) | The ai-rossby PanguPlasim recipe (`awikner/physicsnemo`). A `git subtree`. |
| [`physicsnemo_sfno/`](physicsnemo_sfno/) | NVIDIA PhysicsNeMo SFNO. ⚠ **Editable-installed** — makani imports it, so an edit here changes what makani runs. A `git subtree`. |
| [`PanguWeather/`](PanguWeather/) | PanguWeather SFNO — a **fork** of `s2s/v2.0` (~95% identical, *no* shared imports; fixes do not propagate). A `git subtree`. |
| [`ACE2_retrain/`](ACE2_retrain/) | ACE2 (`ai2cm/fme`), vendored. Runs on Midway; the Polaris port is planned in `polaris_ace2_multinode_handoff.md`. |
| [`s2s/`](s2s/) | The original S2S model (Pangu/Plasim 3D Swin + VAE ensembles, lat-weighted CRPS). `s2s/v2.0/` is the maintained, instrumented variant. |
| [`s2s-lightning/`](s2s-lightning/) | Lightning port of S2S. **Imports** `s2s/v2.0` at runtime — no copy, so edits to `s2s/v2.0` change both. |
| [`si/`](si/) | The SI model (DiT/SiT-style), whose Lightning layout the port mirrors. |
| [`baselines/`](baselines/) | DESIGN §4 numerical-equivalence baselines. **Text only** (JSON/CSV) — never tensors. |

⚠ **Four directories are `git subtree`s** (`makani_sfno`, `physicsnemo_sfno`,
`physicsnemo_ai_rossby`, `PanguWeather`). Edits there can conflict on a future
`subtree pull` — keep them minimal and contiguous.

## Syncing after 2026-09-02 — branches were deleted

Six merged branches were removed after **[PR #12](https://github.com/rcc-uchicago/pedramh-profiling/pull/12)**
consolidated all Polaris work. If you cloned before then, your local copy still points at
branches that no longer exist. To clean up safely:

```bash
git fetch --prune                     # drop remote-tracking refs for deleted branches
git branch -vv | grep ': gone]'       # local branches whose upstream is gone
git branch -d <name>                  # -d REFUSES anything unmerged. Never use -D here
```

⚠ **The current work is NOT on `main` yet** — it is on `feat/multinode-ddp-port`, awaiting
review in PR #12. To read or run it today:

```bash
git fetch origin
git checkout feat/multinode-ddp-port
```

`main` still points at the July bring-up. Once #12 is approved and merged, `main` becomes
current and that branch can go.

Deleted (all fully merged into PR #12 first): `A2C`, `fix/tsoi-fill-270`,
`polaris-pbs-bringup`, `polaris-profiling`, `polaris-data-prep`,
`profile/pangu-polaris-profiling`. **PRs #10 and #11 were closed as superseded** — their
commits are in #12, nothing was lost.

⚠ **Two branches are deliberately kept because they hold commits that exist nowhere else** —
`feat/multiyear-datapipe` and `worktree-makani-multinode-ddp-profiling`. Do not delete them
without tagging first (`git tag archive/<name> <sha> && git push origin archive/<name>`).

## Documentation that is no longer current

Executed handoff prompts are deleted once their work is done — the results live in
`CHANGELOG.md` and the bench reports, which is where to look for them. Removed 2026-09-02:
`polaris_handoff_prompt.md`, `polaris_profiling_handoff_prompt.md`,
`polaris_pipelines_handoff_prompt.md`, `polaris_ai_rossby_pangu_handoff_prompt.md` (+ `_v2`),
`panguweather_repo_cleanup.md`. Their still-live content was folded into
`polaris_pbs_notes.md` before deletion.

**If a doc disagrees with `polaris_pbs_notes.md` on a cluster fact, the notes win.**
Several docs carry convenience copies of the queue table; they drift, and they say so.

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
git checkout -b <name>/<topic>      # e.g. anthony/si-flexattention
# ...work inside one of s2s/, s2s-lightning/, si/...
git push -u origin <name>/<topic>   # then open a PR into main
```

The main benefit is that each project is its own top-level directory, so there is no confliciting work.

### Bringing your own existing code

Exisiting work in a separate repo or a local folder, add this repo
as a remote and push it as a branch — you do not need to re-clone:

```bash
# from inside your existing local checkout
git remote add pedramh git@github.com:rcc-uchicago/pedramh-profiling.git
git fetch pedramh

# start your branch from the shared main, then add your code under a subdir
git checkout -b <name>/<topic> pedramh/main
#   ...copy your files into s2s/, s2s-lightning/, si/, or a new top-level dir...
git add <paths>    # stage BY PATH -- `git add -A` sweeps up backups, logs and
                   # scratch files that were left untracked on purpose
git commit -m "Add <your work>"

# push and set the upstream so later `git push` / `git pull` need no arguments
git push -u pedramh <name>/<topic>
```

### Branch protection

`main` is **protected**: direct pushes are blocked and merging requires a pull
request . Push your branch and open a PR — do not commit to `main` directly.

## Data

The models train on ERA5 reanalysis stored as HDF5. The dataset is **not** in
this repo. Per-cluster paths:

| Cluster | Dataset | Path |
|---|---|---|
| Midway (RCC) | ERA5 h5 | `/project/pedramh/h5data/h5data` |
| **Polaris (ALCF)** | E3SM h5 archive | `$E3SM_ROOT/h5/plev_data` (via `polaris_env.sh`) |
| **Polaris** | makani ALLDATA pack, 101-ch | `$MEMBER_ROOT/data/e3sm_makani_alldata_production` |
| **Polaris** | ACE2 ERA5 (2.4 TB single NetCDF) | `/eagle/projects/lighthouse-uchicago/ace2/` |

`source polaris_env.sh` exports `$MEMBER_ROOT`, `$E3SM_ROOT` and friends — every Polaris
script does this first.

Each project's YAML config sets `data_dir`, `checkpoint_path`, and the mean/std
`.nc` filenames per cluster — edit these before launching a run.

## Environments & HPC scripts

There is no `pip install` package layout; scripts are launched from inside an HPC job.
⚠ **The launcher differs per cluster and per model** — on Polaris **never `srun`**, and
makani/PhysicsNeMo need `python -m torch.distributed.run` rather than bare `torchrun`
(their venv has none, so the name resolves to the base conda's, which pins the wrong
python). The per-model table is in `polaris_pbs_notes.md` §6. The `*.sh` submission scripts hardcode
**cluster-specific** paths (conda/venv locations, `data_dir`) that you edit per
deployment — see each subdir's README.

**Never hardcode the key into a tracked file.** A previously-committed NVIDIA key was
removed from this repo; if you have access to that old key, treat it as
compromised and rotate it at NGC.
