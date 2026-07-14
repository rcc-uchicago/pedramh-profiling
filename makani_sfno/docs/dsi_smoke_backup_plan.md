# DSI smoke-only backup plan (v3) — phase 1 prerequisite

> **This document is phase 1 only.** It brings up a parallel working copy
> of `AI-RES/` on DSI just far enough to execute one SFNO zgplev **smoke**
> run. Full training on DSI is a separate phase 2, covered in
> `docs/dsi_full_training_plan.md` (v4) — that plan depends on phase 1
> succeeding first.

**Goal.** Stand up a parallel working copy of `AI-RES/` on the UChicago DSI
cluster, **scoped to running an SFNO zgplev *smoke* run only**. This
verifies the env, the data path, and the DSI submit pattern so phase 2
(full training) can build on a known-good base. Stampede3 remains
authoritative.

**Date.** 2026-05-01. Branch at time of inventory: `zgplev-migration`.

> **What changed from v1 → v2.** External review flagged that v1 was too
> optimistic: proto dataset insufficient for full training, SLURM rewrite
> missed three partitions, env uses CUDA-13 wheels DSI cannot run, `.venv`
> not actually gitignored, smoke `OUTPUT_ROOT` default wrong,
> `slurm_helpers.sh` carries Stampede3 SMTP defaults. v2 narrowed scope to
> smoke, made env rebuild the long pole, used a bootstrap branch, cut data
> transfer from 39 → 21 GB.
>
> **What changed from v2 → v3.** Second review flagged: (1) `earth2studio`
> is not a runtime dependency for SFNO training and pulls native libs
> (`pygrib`, `cfgrib`, `eccodes`) — drop it; (2) DSI selects CUDA via
> `/usr/local/cuda-*` + env vars, not `module load`; (3) the DSI submit
> skeleton uses `$DSI_SCRATCH` under `set -u` without a guard;
> (4) post-2026-05-07 preemption needs explicit QoS handling
> (`--qos=protected` for short or `--qos=general --signal=B:USR1@300
> --requeue` for longer); (5) bootstrap branch must be created *before*
> editing `.gitignore`; (6) `.claude/scheduled_tasks.lock` is not ignored;
> (7) the DSI submit script should not copy-paste the smoke body — it
> should `bash` the original with vars exported; (8) email is not a
> success criterion. v3 fixes all eight.
> Full training on DSI remains out of scope.

---

## 0. Scope and non-goals

### In scope (v2)

- Build a runnable Python env on DSI matching DSI's CUDA stack.
- Transfer the **21 GB** `sim52_astro_64x128_zgplev_proto` dataset only.
- Run `submit_zgplev_smoke.slurm` once on DSI to validate the path.
- Use DSI as a **smoke / debug environment** when Stampede3 is queued up.

### Out of scope (v2)

- **Full SFNO training on DSI.** The full run requires `sim52_zgplev_full`
  (years 12–111 train, year 11 valid), which **does not exist anywhere yet**
  — neither on Stampede3 nor in any backup. Producing it requires either
  (a) running the full packager on DSI from `data/postproc/` (490 GB
  transfer + many node-hours), or (b) running it on Stampede3 first and
  then transferring (~hundreds of GB). Until that dataset exists, full
  training stays on Stampede3.
- **Migrating the postprocessor** (burn7 + namelists) to DSI. Out of scope.
- **DSI as primary.** Stampede3 stays authoritative.

### Why not full training on DSI

1. **Dataset gap.** `sim52_zgplev_full` doesn't exist; `sim52_astro_64x128_zgplev_proto`
   contains 6 train years (`MOST.0003`–`MOST.0008`), 1 valid (`MOST.0009`),
   1 test (`MOST.0010`). Verified directly:
   `find data/makani/sim52_astro_64x128_zgplev_proto/{train,valid,test} -name '*.h5'`.
   The full config (`plasim_sim52_zgplev_full.yaml`) wants years 12–111.
2. **Wall time.** DSI's documented default wall time on `general` is 12 h.
   `submit_zgplev_full.slurm` requests 47:30:00. We'd need
   checkpoint/requeue logic that doesn't exist yet.
3. **Preemption.** Per DSI policy, after **2026-05-07** (six days from
   today) `general` jobs are preemptable. A 47-hour SFNO run with no
   requeue logic is not viable there.

---

## 1. Inventory — verified facts

### 1.1 Top-level layout of `/home1/11114/zhixingliu/projects/SFNO_Climate_Emulator/`

| Entry                | Type    | Real size | Notes                                              |
|----------------------|---------|-----------|----------------------------------------------------|
| `.git/`              | dir     | 1.4 MB    | tracked history                                    |
| `.claude/`           | dir     | ~1 KB     | local Claude state — skip                          |
| `.gitignore`         | file    | 512 B     | **bug**: has `.venv/` (slash) but `.venv` is a *symlink* and is not ignored. Verified with `git check-ignore -v .venv` → "not ignored". |
| `.pr_body_zgplev.md` | file    | 3.4 KB    | in-flight PR body draft — local only               |
| `.pytest_cache/`     | dir     | 20 KB     | rebuildable                                        |
| `.venv`              | symlink | → `/work2/.../stampede3/AI-RES/.venv` (6.5 GB) | **do not copy** — Stampede3-binary, links against CUDA 13 |
| `Untitled`           | file    | 20 B      | gitignored leftover                                |
| `checkpoints`        | symlink | → `/scratch/.../AI-RES/checkpoints` (empty) |                              |
| `configs/`           | dir     | 0 B       | placeholder                                        |
| `data`               | symlink | → `/scratch/.../AI-RES/data` (844 GB) | see §1.3                                  |
| `docs/`              | dir     | 496 KB    | tracked plans + audit snapshots                    |
| `external/`          | dir     | 3.8 MB    | NVIDIA/earth2studio @ `25a1e7ae`, **clean**, untracked |
| `logs/`              | dir     | 343 MB    | one 333 MB `.err` dominates — exclude              |
| `makani-src/`        | dir     | 172 MB    | NVIDIA/makani @ `c9704308`, **clean**, untracked   |
| `notebooks/`         | dir     | 0 B       | placeholder                                        |
| `results`            | symlink | → `/work2/.../AI-RES/results` (12 KB) |                                       |
| `scripts/`           | dir     | 161 KB    | tracked + a few new untracked                      |
| `skills/`            | dir     | 38 KB     | mostly tracked                                     |
| `src/`               | dir     | 949 KB    | tracked + significant untracked sfno_training/     |
| `tests/`             | dir     | 715 KB    | tracked + untracked sfno_training/ tests           |

Tracked git content total: 1.09 MB / 115 files.

### 1.2 Symlinks (recursive, excluding `.git/`)

```
.venv        -> /work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/.venv      (6.5 GB)  SKIP
checkpoints  -> /scratch/11114/zhixingliu/SFNO_Climate_Emulator/checkpoints        (empty)
data         -> /scratch/11114/zhixingliu/SFNO_Climate_Emulator/data               (844 GB)  see §1.3
results      -> /work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results    (12 KB)
```

### 1.3 `data/` breakdown (844 GB)

```
data/inputs/                                    0       empty
data/plasim_postprocess/                      137 MB    intermediate, skip
data/postproc_timing_test/                    3.5 GB    benchmarking artifact, skip
data/boundary_astro/sim52/                    18 GB     consumed only by the *packager*, not by training — SKIP for this plan
data/postproc/                                472 GB    raw NetCDF — only read by packager — SKIP
data/makani/sim52_astro_64x128/               331 GB    v9 packaged training set (sigma-level zg) — out of scope (smoke only)
data/makani/sim52_astro_64x128_zgplev_proto/  21 GB     v10 zgplev prototype, 6+1+1 years — TRANSFER (smoke target)
data/makani/sim52_{full,short,tiny}/          0         empty placeholders
```

`zgplev_proto` decomposes as: `train/` 16 GB (6 H5 files, years 3–8),
`valid/` 2.7 GB (1 file, year 9), `test/` 2.7 GB (1 file, year 10),
`stats/` 1.9 MB, plus tiny `config/`, `metadata/`, `validation/`.

**Tier-1 transfer for v2 smoke-only scope: 21 GB.** No `boundary_astro/`,
no v9 baseline. (Reviewer was right: training reads packaged H5
forcing/stats from `OUTPUT_ROOT/stats`, not raw boundary NetCDF.
Verified: `boundary_astro/` is referenced only in
`scripts/package_sim52_astro.sh` and `scripts/build_boundary_dir.py`,
both packager-side.)

### 1.4 Hardcoded Stampede3 paths in source

| File                                                | Lines        | What                                        |
|-----------------------------------------------------|--------------|---------------------------------------------|
| `src/plasim_makani_packager/metadata.py`            | 27,63,117,118| docstring example + default `--exp-dir`, `source_postproc_root`, `source_boundary_root` |
| `tests/sfno_inference/test_checkpoint_loader.py`    | 42           | hardcoded checkpoint path used in unit test |
| `src/sfno_training/slurm_helpers.sh`                | bottom block | Stampede3 SMTP relay `129.114.112.1` and `${USER}@stampede3.tacc.utexas.edu` From: header |

YAML configs use `{{OUTPUT_ROOT}}` / `{{EXP_DIR}}` template substitution and
need **no** rewriting.

### 1.5 SLURM partitions used (verified by recursive grep)

| Script (16 `.slurm` + `slurm_helpers.sh` = **17 files total**)         | Partition |
|-----------------------------------------------------------------------|-----------|
| `scripts/submit_eval_inference.slurm`                                 | `h100`    |
| `scripts/submit_eval_report.slurm`                                    | `skx`     |
| `scripts/submit_eval_score.slurm`                                     | `skx`     |
| `src/emulator_adaptor/submit.slurm`                                   | `skx`     |
| `src/plasim_makani_packager/submit.slurm`                             | `skx`     |
| `src/plasim_postprocessor/submit.slurm`                               | `skx`     |
| `src/sfno_training/submit_full.slurm`                                 | `h100`    |
| `src/sfno_training/submit_short.slurm`                                | `amd-rtx` |
| `src/sfno_training/submit_smoke.slurm`                                | `amd-rtx` |
| `src/sfno_training/submit_tiny.slurm`                                 | `amd-rtx` |
| `src/sfno_training/submit_train.slurm`                                | `gh`      |
| `src/sfno_training/submit_zgplev_baseline.slurm`                      | `gh`      |
| `src/sfno_training/submit_zgplev_full.slurm`                          | `h100`    |
| `src/sfno_training/submit_zgplev_short.slurm`                         | `amd-rtx` |
| `src/sfno_training/submit_zgplev_smoke.slurm`                         | `amd-rtx` |
| `src/sfno_training/submit_zgplev_tiny.slurm`                          | `amd-rtx` |
| `src/sfno_training/slurm_helpers.sh`                                  | (no `#SBATCH`, but has Stampede3 SMTP defaults — see §1.4) |

For v2 we only need to port **one** of these (`submit_zgplev_smoke.slurm`)
to DSI. Others stay Stampede3-only.

### 1.6 Environment (verified `pip freeze` from the live venv)

```
torch==2.11.0
torchvision==0.26.0
torch_harmonics @ git+https://github.com/NVIDIA/torch-harmonics.git@a632ca7…
nvidia-physicsnemo==2.0.0
cuda-toolkit==13.0.2
nvidia-cudnn-cu13==9.19.0.56
nvidia-nccl-cu13==2.28.9
nvidia-cublas==13.1.0.3   …  (full nvidia-*-cu13 stack)
earth2studio==0.13.0
numpy==2.4.4
```

This is a **CUDA-13 build**. DSI publishes CUDA 11.8 / 12.1 / 12.3 / 12.4.
No CUDA-13. A blind `pip install -r requirements-stampede3.txt` will
install CUDA-13 wheels that fail at `import torch` on DSI's drivers.

The env rebuild is the **dominant blocker** for this plan, not data transfer.

### 1.7 Smoke script bug

`src/sfno_training/submit_zgplev_smoke.slurm:35`:

```bash
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev}"
```

The fallback name is `sim52_astro_64x128_zgplev` (no `_proto`), but the
only existing v10 dataset is `sim52_astro_64x128_zgplev_proto`. On DSI
we will **not** edit the script — we'll always pass `OUTPUT_ROOT`
explicitly via the launch wrapper (§4.3). On Stampede3 the script also
needs an `OUTPUT_ROOT` override today; this is a pre-existing bug, not
DSI-specific.

### 1.8 Secrets

None inside `AI-RES/`. SSH keys live in `~/.ssh/` and stay on Stampede3.

### 1.9 External code clones

| Dir                       | Remote                                  | SHA                | Local mods? | v3 action |
|---------------------------|-----------------------------------------|--------------------|-------------|-----------|
| `external/earth2studio/`  | https://github.com/NVIDIA/earth2studio  | `25a1e7ae` on main | no          | **skip** — not imported by `src/sfno_training/`, `scripts/`, or `tests/sfno_training/` (verified by recursive grep). Its `pyproject.toml` requires `pygrib` (line 24), `cfgrib` (line 62), `eccodes` (line 63) — native-library pain we don't need. |
| `makani-src/`             | https://github.com/NVIDIA/makani        | `c9704308` on main | no          | re-clone on DSI at this SHA. |

### 1.10 Git working-tree state on `zgplev-migration`

- 4 modified tracked files; 23 deleted tracked files (`src/plasim_postprocessor/`
  refactor in flight); 40+ untracked items including the entire
  `src/sfno_training/` package, `tests/sfno_training/`, several new
  `docs/*.md` plans, new `scripts/build_*.py` and `scripts/preflight.py`,
  `external/`, `makani-src/`, `.venv`, `.pr_body_zgplev.md`,
  `docs/audit_snapshots/*.txt`. This is significant unmerged work.

---

## 2. Pre-transfer hygiene on Stampede3

### 2.1 Create the bootstrap branch FIRST (no edits to `zgplev-migration`)

The dirty `zgplev-migration` work-in-progress is part of an open PR. Cut
the bootstrap branch *before* any working-tree edits, so the `.gitignore`
and `requirements-stampede3.txt` changes land on the bootstrap branch only:

```bash
cd /home1/11114/zhixingliu/projects/SFNO_Climate_Emulator
git checkout -b zgplev-migration-dsi-bootstrap
```

### 2.2 Fix the `.venv` ignore bug AND `.claude/` residue (on bootstrap branch)

```bash
# .gitignore currently has `.venv/` (matches dirs only) and only ignores
# .claude/settings.local.json explicitly. Two fixes:
printf '.venv\n.claude/scheduled_tasks.lock\n' >> .gitignore
git check-ignore -v .venv                          # must now report a hit
git check-ignore -v .claude/scheduled_tasks.lock   # must now report a hit
```

(If you prefer to ignore the whole `.claude/` directory, replace the two
new lines with `.claude/`. The current explicit `settings.local.json`
ignore stays compatible.)

### 2.3 Capture the env (informational only)

This freeze is **not** safe to `pip install` on DSI verbatim — see §4.C —
but capturing it tells us which packages we need DSI-compatible versions of:

```bash
source .venv/bin/activate
pip freeze > requirements-stampede3.txt
deactivate
```

### 2.4 Commit the bootstrap snapshot

```bash
git add .gitignore requirements-stampede3.txt
git add README.md docs/plasim_makani_packager_plan.md scripts/submit_eval_report.slurm \
        skills/plasim-makani-packager/SKILL.md tests/plasim_makani_packager/stub_forcing_loader.py
git add src/plasim_postprocessor    # picks up the deletions
git add docs/aires_rad_profile_plan.md docs/audit_snapshots/ \
        docs/plasim_postprocessor_refactor_plan.md docs/sfno_*.md \
        docs/dsi_smoke_backup_plan.md docs/dsi_full_training_plan.md
git add scripts/build_boundary_dir.py scripts/build_subset_dataset.py scripts/preflight.py
git add skills/plasim-postprocess/ skills/sfno-training/
git add src/plasim_postprocessor/submit.slurm
git add src/sfno_training/   tests/sfno_training/

git status --short             # only intentional residue (.pr_body_zgplev.md, Untitled, .venv, .pytest_cache)
git commit -m "bootstrap: zgplev WIP snapshot for DSI parallel-copy"
git push -u origin zgplev-migration-dsi-bootstrap
```

`.pr_body_zgplev.md`, `Untitled`, `.venv` (now ignored), `.claude/scheduled_tasks.lock`
(now ignored), and `.pytest_cache/` must remain untracked. Do not touch
the `zgplev-migration` branch.

---

## 3. DSI side — open items to confirm BEFORE transfer

These need answers from DSI docs / staff / a 10-minute interactive shell
**before** we run any of §4. Treat them as gating questions:

```bash
# On a DSI login node:
dsiquota                                          # quotas on home/project/scratch
sinfo -o "%P %G %l %D %t"                         # partitions, GRES, default time
echo "$SCRATCH"; echo "$WORK"                     # do these exist? where do they point?
ls -d /usr/local/cuda*                            # ALL installed CUDA toolkits (login node)

# Then a 10-minute GPU shell to read the live driver + compute-node CUDA:
srun --partition=general --gres=gpu:1 --time=00:10:00 --pty bash -lc \
    'hostname; nvidia-smi; ls -d /usr/local/cuda*; /usr/local/cuda/bin/nvcc --version 2>/dev/null; python3 --version'
```

Specifically need:

1. **DSI scratch path** (`/scratch/<user>` / `/net/scratch/<user>` / etc.).
2. **DSI project path** (`/project/<group>/<user>` / etc.) and quota.
3. **Default and GPU partition names** + GRES syntax (`-p general --gres=gpu:1`?).
4. **Wall-time max** on the GPU partition you'd use.
5. **Preemption / QoS** post-2026-05-07: which QoS values exist
   (e.g. `protected` vs `general`), what time caps each carries, and the
   checkpoint/requeue contract for `general`.
6. **Allocation / `-A` account string**, if required.
7. **`/usr/local/cuda-*` versions on compute nodes** + `nvcc --version` —
   compute nodes are the source of truth, not login. The matching CUDA
   determines the torch wheel index (`cu121`, `cu124`, …).
8. **`nvidia-smi` driver line** from a GPU node (e.g. "CUDA Version: 12.4")
   — caps the maximum CUDA we can run.
9. **Globus endpoint** name (search `uchicago` on globus.org).
10. **Whether `$SCRATCH` and `$WORK` are exported** in interactive + batch shells.

Without 7+8 we can't pin a torch wheel. Without 1+2+9 we can't transfer
data. Without 5 we can't choose between `--qos=protected` (short, won't
preempt) and `--qos=general --signal=B:USR1@300 --requeue` (longer, may
preempt) for the smoke job.

> DSI's documentation describes CUDA selection via `/usr/local/cuda-*`
> + `CUDA_HOME`/`PATH`/`LD_LIBRARY_PATH`, **not** `module load`. Do not
> assume modules. Use whichever the DSI shell shows in §3.7.

---

## 4. Execution plan (after §3 is answered)

### 4.A Source tree → DSI via GitHub

```bash
# On DSI (assuming SSH key already on GitHub):
cd $HOME
git clone git@github.com:feynmanliu214/SFNO_Climate_Emulator-Stampede3.git AI-RES
cd AI-RES
git checkout zgplev-migration-dsi-bootstrap
```

### 4.B External clones → re-clone (makani-src only)

```bash
cd $HOME/AI-RES
git clone https://github.com/NVIDIA/makani.git makani-src
git -C makani-src checkout c97043086e60d44a3adc3bede9a6b3dc71f5005d
```

**Skip `external/earth2studio`** for v2/v3 smoke scope. Verified by
recursive grep: it is not imported by `src/sfno_training/`, `scripts/`,
or `tests/sfno_training/`. Its install pulls native `pygrib` + `cfgrib`
+ `eccodes` — out of scope.

### 4.C Python venv — rebuild against DSI's CUDA (the long pole)

This step will take **2–6 hours** of trial and error, not 15–60 minutes.
The Stampede3 freeze is CUDA-13 and incompatible.

DSI selects CUDA via `/usr/local/cuda-*` + env vars (no `module load`).
Workflow:

```bash
# 1. On a GPU compute node (compute is source of truth — not login):
srun --partition=<DSI_GPU_PARTITION> --gres=gpu:1 --time=01:00:00 --pty bash -l

# 2. Pick the CUDA toolkit. Use the latest /usr/local/cuda-N.M whose major.minor
#    is <= the driver-reported max ("CUDA Version: …" from `nvidia-smi`).
ls -d /usr/local/cuda-*
nvidia-smi | head -3
# Suppose the choice is /usr/local/cuda-12.4:
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
nvcc --version    # confirms toolkit version

# 3. Build the venv with the system Python:
cd $HOME/AI-RES
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel

# 4. Pin torch to the wheel matching the chosen CUDA. Use the index URL
#    that matches CUDA_HOME (cu121 / cu124 / cu123 / cu118 — substitute):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 5. torch_harmonics at the Stampede3-pinned SHA:
pip install "git+https://github.com/NVIDIA/torch-harmonics.git@a632ca748a12bd9f74dbc1e00653317810991f74"

# 6. physicsnemo: nvidia-physicsnemo==2.0.0 requires torch >= 2.7. If the
#    cu124 torch is older, install an older physicsnemo. Treat as gate item:
pip install "nvidia-physicsnemo"   # let pip pick a torch-compatible version

# 7. makani runtime deps that we WANT pip to install (i.e. everything
#    in makani-src/pyproject.toml NOT already pinned in steps 4-6):
#       torch          — pinned in step 4
#       nvidia-physicsnemo — pinned in step 6
#       torch-harmonics    — pinned in step 5
#       numpy          — usually pulled by torch
#    Remaining deps must be installed explicitly so --no-deps works:
pip install \
    "h5py>=3.11.0" \
    "wandb>=0.13.7" \
    "tqdm>=4.60.0" \
    "more-itertools" \
    "Pillow" \
    "ruamel.yaml" \
    "PyYAML" \
    "pytest"   # tests/ + scripts/preflight.py

# 8. makani (editable, --no-deps so it doesn't re-resolve torch / physicsnemo
#    / torch-harmonics and break the pinned stack):
pip install --no-deps -e ./makani-src
pip check     # surfaces unmet requirements; resolve manually if any

# 9. Verify imports load against the chosen CUDA:
python - <<'PY'
import torch, h5py, yaml, pytest, makani, torch_harmonics
assert torch.cuda.is_available(), "torch can't see GPU"
print(torch.__version__, torch.version.cuda, makani.__file__)
PY
```

Risks specific to this step (rank-ordered):

1. **`nvidia-physicsnemo==2.0.0` torch>=2.7 floor.** If the highest cu* wheel
   on DSI's CUDA is torch < 2.7, pin `nvidia-physicsnemo<2.0`.
2. **`torch_harmonics`** from git rebuilds against the chosen torch ABI;
   expect a 5–15 min compile. CUDA env vars from step 2 must be active.
   If the build fails with "ninja not found", "PEP 517 build failed", or
   "cannot import setuptools", retry with build deps installed and
   isolation off:
   ```bash
   pip install ninja "packaging>=24" "setuptools>=68"
   pip install --no-build-isolation \
       "git+https://github.com/NVIDIA/torch-harmonics.git@a632ca748a12bd9f74dbc1e00653317810991f74"
   ```
3. **`numpy==2.4.4`** is unusually new; some deps may pin `numpy<2`. Let
   pip resolve, then re-test.
4. **Conda alternative.** If DSI's docs prefer Conda/Mamba over pip+venv,
   switch — the solver handles `numpy>=2` / torch / CUDA better than pip.

**Do not pursue this step blind.** Get the §3.7+§3.8 answers first, then
build a fresh `requirements-dsi.txt` by trial. Plan stops here if no
torch+physicsnemo combination installs against any DSI CUDA.

### 4.D Symlinks on DSI

After §3 confirms DSI scratch + project paths:

```bash
DSI_SCRATCH=/scratch/<user>           # confirm
DSI_PROJECT=/project/<group>/<user>   # confirm

mkdir -p $DSI_SCRATCH/AI-RES/{data,checkpoints,runs}
mkdir -p $DSI_PROJECT/AI-RES/results

cd $HOME/AI-RES
ln -s $DSI_SCRATCH/AI-RES/data        data
ln -s $DSI_SCRATCH/AI-RES/checkpoints checkpoints
ln -s $DSI_PROJECT/AI-RES/results     results
# .venv stays a real dir on DSI (built in §4.C).
```

### 4.E Tier-1 data transfer — 21 GB only

```
SOURCE: /scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev_proto/
DEST:   $DSI_SCRATCH/AI-RES/data/makani/sim52_astro_64x128_zgplev_proto/
SIZE:   21 GB (16 + 2.7 + 2.7 + small)
```

Boundary, postproc, postproc_timing_test, plasim_postprocess, v9 baseline:
**all skipped** in v2.

#### 4.E.1 Globus (preferred)

1. Activate `tacc#stampede3` and DSI's Globus endpoint (name from §3).
2. Source: `/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev_proto/`.
3. Destination: `$DSI_SCRATCH/AI-RES/data/makani/sim52_astro_64x128_zgplev_proto/`.
4. Enable: preserve mtimes, verify integrity. Submit.

#### 4.E.2 rsync fallback

```bash
# From a Stampede3 login node:
rsync -avh --partial --append-verify --info=progress2 \
    /scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev_proto/ \
    DSI_USER@DSI_HOST:$DSI_SCRATCH/AI-RES/data/makani/sim52_astro_64x128_zgplev_proto/
```

Login node only — never run from a compute node.

### 4.F DSI-only SLURM wrapper (no copy-paste, no in-place mutation)

Create `src/sfno_training/submit_zgplev_smoke.dsi.slurm` as a **thin
wrapper** that exports DSI-specific vars and then `bash`-invokes the
original Stampede3 smoke script. The original's `#SBATCH` directives are
inert when invoked as `bash <path>`, so we don't need to touch it.
Sibling file, not replacement → no Stampede3 breakage, no merge conflict.

Skeleton (substitute placeholders from §3, choose ONE of the two QoS
blocks):

```bash
#!/bin/bash
#SBATCH -J sfno_zgplev_smoke_dsi
#SBATCH --partition=<DSI_GPU_PARTITION>      # confirm in §3.3
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A <DSI_ACCOUNT>                     # confirm in §3.6, omit if not required
#SBATCH -o logs/sfno_zgplev_smoke_dsi_%j.out
#SBATCH -e logs/sfno_zgplev_smoke_dsi_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# === Choose ONE QoS block based on the §3.5 answer ===

# OPTION A (preferred for smoke — short, no preemption):
#SBATCH --qos=protected
#SBATCH --time=02:00:00

# OPTION B (longer wall, may be preempted post-2026-05-07; needs requeue):
# #SBATCH --qos=general
# #SBATCH --time=12:00:00
# #SBATCH --signal=B:USR1@300
# #SBATCH --requeue

set -euo pipefail

REPO_ROOT="$HOME/AI-RES"

# Hard-fail fast if DSI scratch isn't set — `set -u` would otherwise expand
# the unset var to empty mid-pipeline. Define it here from the §3.1 answer:
DSI_SCRATCH="${DSI_SCRATCH:-/scratch/$USER}"           # SUBSTITUTE actual DSI path
: "${DSI_SCRATCH:?Set DSI_SCRATCH from §3.1 (e.g. /scratch/$USER or /net/scratch/$USER)}"

# CUDA env (DSI uses /usr/local/cuda-*, not modules — see §3.7):
export CUDA_HOME=/usr/local/cuda-12.4                  # SUBSTITUTE actual version
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Force the proto dataset (works around the Stampede3 script's wrong default at submit_zgplev_smoke.slurm:35):
export OUTPUT_ROOT="$DSI_SCRATCH/AI-RES/data/makani/sim52_astro_64x128_zgplev_proto"
export EXP_DIR="$DSI_SCRATCH/AI-RES/runs/sfno_zgplev_smoke_dsi"
export MAIL_TO="zhixingliu@uchicago.edu"

# Override Stampede3 SMTP defaults from slurm_helpers.sh (env-driven, no edit needed):
export SFNO_MAIL_FROM="${USER}@uchicago.edu"
export SFNO_SMTP_RELAY="${SFNO_SMTP_RELAY:-smtp.uchicago.edu}"   # confirm with DSI; mail is best-effort

# Stampede3 script expects $SCRATCH; alias it here so the original's path
# expansions still resolve to DSI scratch:
export SCRATCH="$DSI_SCRATCH"

# Run the original Stampede3 smoke script as plain bash. Its `#SBATCH`
# directives are comments to bash; the script body runs normally.
exec bash "$REPO_ROOT/src/sfno_training/submit_zgplev_smoke.slurm"
```

Notes:

- `slurm_helpers.sh` does not need editing. `SFNO_SMTP_RELAY` and
  `SFNO_MAIL_FROM` are already env-overridable (verified at the bottom
  of the file). Mail is best-effort — see §5.
- `--signal=B:USR1@300 --requeue` only helps if `train_plasim.py` actually
  saves a checkpoint on `SIGUSR1`. For the smoke run (30 min target wall),
  Option A (`--qos=protected --time=02:00:00`) avoids the question.
  Use Option B only if §3.5 confirms `protected` isn't available.

### 4.G Hardcoded-path patches (small, scoped)

Make these edits on the DSI bootstrap branch only, not on `zgplev-migration`:

| File:line                                          | Change                                                                 |
|----------------------------------------------------|------------------------------------------------------------------------|
| `src/plasim_makani_packager/metadata.py:27,63,117,118` | Read paths from env (`os.environ.get("AIRES_DATA_ROOT", "<old>")`); keep Stampede3 path as default |
| `tests/sfno_inference/test_checkpoint_loader.py:42`    | Read path from env or fixture; `pytest.mark.skipif` when missing       |

For v2 smoke scope, neither is on the critical path, but pin them so the
DSI branch tests can run without the Stampede3 paths.

---

## 5. Verification on DSI

```bash
cd $HOME/AI-RES
git status                                                  # clean? on DSI bootstrap branch?
ls -la                                                      # symlinks correct?
find -L data/makani/sim52_astro_64x128_zgplev_proto/{train,valid,test} -name '*.h5' | sort
# Expect 6 + 1 + 1 = 8 H5 files matching the Stampede3 listing.
du -sh data/makani/sim52_astro_64x128_zgplev_proto/         # ~21 GB

source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import makani, torch_harmonics; print('OK')"      # earth2studio NOT required

# Optional CPU-only unit tests (no GPU needed):
pytest tests/sfno_training/test_data_loader.py -v
pytest tests/sfno_training/test_preprocessor.py -v

# Smoke launch:
sbatch src/sfno_training/submit_zgplev_smoke.dsi.slurm
squeue -u $USER
# tail -f logs/sfno_zgplev_smoke_dsi_*.err
```

**Success criterion (revised):** the run completes with a finite final
loss, a checkpoint exists under
`$DSI_SCRATCH/AI-RES/runs/sfno_zgplev_smoke_dsi/plasim_sim52_zgplev_smoke/0/training_checkpoints/`,
and the loaded checkpoint can be re-instantiated with `torch.load`. **Mail
delivery is best-effort and is NOT a success criterion** — DSI SMTP may
silently drop mail without affecting the run.

Concretely:

```bash
ls $DSI_SCRATCH/AI-RES/runs/sfno_zgplev_smoke_dsi/plasim_sim52_zgplev_smoke/0/training_checkpoints/
grep -E 'loss|epoch' logs/sfno_zgplev_smoke_dsi_*.err | tail -20
# A checkpoint file present + a finite loss line → smoke passed.
```

---

## 6. Explicit non-actions

- **Not** copying `.venv` (Stampede3 binary, CUDA-13).
- **Not** copying `.claude/` (Stampede3-local).
- **Not** copying or installing `external/earth2studio/` (not imported by
  SFNO training; pulls native `pygrib`/`cfgrib`/`eccodes`).
- **Not** copying `makani-src/` (re-cloned at the same SHA on DSI).
- **Not** copying `data/postproc/` (472 GB, packager input — not on DSI's path).
- **Not** copying `data/postproc_timing_test/`, `data/plasim_postprocess/`.
- **Not** copying `data/boundary_astro/sim52/` (18 GB, packager input only).
- **Not** copying `data/makani/sim52_astro_64x128/` (331 GB v9 baseline — out of scope).
- **Not** copying `logs/`.
- **Not** copying `~/.ssh/`.
- **Not** mutating Stampede3 SLURM scripts in place — DSI submit script is
  a wrapper that exports vars and `bash`-invokes the original.
- **Not** editing `slurm_helpers.sh` — its SMTP defaults are already
  env-overridable.
- **Not** loading CUDA via `module load` — DSI uses `/usr/local/cuda-*`
  + `CUDA_HOME` / `PATH` / `LD_LIBRARY_PATH`.
- **Not** treating mail delivery as a success criterion.
- **Not** pushing the WIP commit to `zgplev-migration` (uses
  `zgplev-migration-dsi-bootstrap` branch instead).
- **Not** running full SFNO training on DSI (dataset doesn't exist;
  wall-time + preemption don't fit).

---

## 7. Realistic time budget

| Step                                                  | Time              |
|-------------------------------------------------------|-------------------|
| §3 DSI-side reconnaissance (quotas, partitions, CUDA) | 30 min – 1 day (depends on DSI staff response) |
| §2 hygiene + bootstrap branch + push                  | 15 min            |
| §4.A clone on DSI                                     | 1 min             |
| §4.B re-clone external/ + makani-src/                 | 5 min             |
| §4.C **venv build, pinning torch/physicsnemo to DSI CUDA** | **2–6 hours, expect failures** |
| §4.D symlinks                                         | 5 min             |
| §4.E.1 Globus tier-1 (21 GB)                          | 30 min – 2 hr     |
| §4.F DSI-only smoke SLURM                             | 30 min            |
| §4.G patches                                          | 30 min            |
| §5 verification + smoke launch + queue wait           | 1–8 hr            |
| **Total to first DSI smoke result**                   | **1–3 working days** |

If §3 reveals the env can't be built (e.g. physicsnemo refuses to install
on any DSI-compatible torch), the plan stops at §4.C and we fall back to
the original "wait for Stampede3" stance.

---

## 8. Reviewer checklist

### v2 addressed v1's review:

- [x] Dropped the false claim that proto = full training; smoke-only scope.
- [x] Identified the smoke script's wrong `OUTPUT_ROOT` default and works around it via env override.
- [x] Catalogs all four partitions (`h100`, `skx`, `amd-rtx`, `gh`); 17 SLURM-related files.
- [x] Calls out 12-hour wall + 2026-05-07 preemption.
- [x] Acknowledges torch 2.11 + cu13 + physicsnemo 2.0 + numpy 2.4 vs DSI's CUDA 11.8/12.x; 2–6 hour env budget.
- [x] Adds `.venv` (no slash) to `.gitignore`.
- [x] Drops `boundary_astro/` from the transfer set.
- [x] Uses `zgplev-migration-dsi-bootstrap` branch, not `zgplev-migration`.
- [x] DSI-side checks (`dsiquota`, `sinfo`, `srun --pty`) gate execution.
- [x] DSI-specific SLURM scripts as siblings; no in-place mutation.
- [x] `slurm_helpers.sh` SMTP defaults are env-overridable; no edit needed.

### v3 addressed v2's review:

- [x] Drops `external/earth2studio` clone + install — not a runtime dep for SFNO training (verified by recursive grep); avoids `pygrib`/`cfgrib`/`eccodes` native-lib problems.
- [x] Replaces `module load <CUDA_MODULE>` with `/usr/local/cuda-*` discovery + `CUDA_HOME`/`PATH`/`LD_LIBRARY_PATH` env vars (DSI's documented method).
- [x] Adds `: "${DSI_SCRATCH:?…}"` guard before any expansion under `set -u`.
- [x] Explicit QoS choice: Option A `--qos=protected --time=02:00:00` (preferred for smoke) or Option B `--qos=general --signal=B:USR1@300 --requeue` (longer, may preempt post-2026-05-07).
- [x] Bootstrap branch created **before** `.gitignore` edit (§2 reordered).
- [x] `.gitignore` now also covers `.claude/scheduled_tasks.lock`.
- [x] DSI submit script no longer copy-pastes the smoke body — it `bash`-invokes the original Stampede3 script with vars exported (`#SBATCH` lines are inert under bash).
- [x] Mail removed from success criterion (§5); criterion is finite loss + reloadable checkpoint.
