# DSI full SFNO training plan (v4) — phase 2

> **Phase relationship.** This is phase 2. Phase 1
> (`docs/dsi_smoke_backup_plan.md` v3) brings up the DSI environment and
> verifies a smoke run; this document **assumes phase 1 has succeeded**
> and reuses its env, branch, repo clone, and DSI infrastructure answers.
> If phase 1 has not been executed, do that first.

**Goal.** Run **full** SFNO zgplev training (years 12–111 train, year 11
valid) on the UChicago DSI cluster as a parallel worker while Stampede3
GPU queues are saturated. Stampede3 remains authoritative; DSI runs
independent training jobs that produce comparable checkpoints.

**Date.** 2026-05-01. Branch at time of inventory: `zgplev-migration`.

---

## 0. Plan executability — what does NOT yet exist in the worktree

This document describes the steps; it has not yet *applied* them. Before
phase 2 can run, the following changes must land (none are made by
reading this doc):

| Item                                                      | Status | Where prescribed |
|-----------------------------------------------------------|--------|------------------|
| `.gitignore` updated to ignore `.venv` (no slash) + `.claude/scheduled_tasks.lock` | **NOT applied** — `.gitignore:24` still has `.venv/` only | smoke plan §2.2 |
| `src/plasim_makani_packager/submit.slurm` env-default refactor (`SIMS=""` → `${SIMS:-}`, etc., lines 14–19 + 23–28) | **NOT applied** | full plan §2.2 |
| `src/sfno_training/submit_zgplev_full.slurm:36-40` stale `--src` comment fix | **NOT applied** | full plan §3.5 |
| `src/sfno_training/submit_zgplev_full.dsi.slurm` (new file) | **NOT created** | full plan §7 |
| `src/sfno_training/submit_zgplev_smoke.dsi.slurm` (new file) | **NOT created** | smoke plan §4.F |
| `requirements-stampede3.txt` from `pip freeze`                              | **NOT created** | smoke plan §2.3 |
| `train_plasim.py` SIGUSR1 handler (Mitigation B)          | **NOT planned** — first launch drops `--signal` | full plan §8.3 |

**Recommendation: use `git worktree`, but commit the bootstrap snapshot
FIRST** — the current `zgplev-migration` working tree has 40+ untracked
files (entire `src/sfno_training/` package, `tests/sfno_training/`,
several `docs/*.md` plans). A worktree checked out from
`zgplev-migration` will *not* contain those files, because
`git worktree add` materializes only what the source branch has
committed.

> **Risk note.** The bootstrap commit converts a wide swath of
> untracked WIP into a single commit on a side branch. Snapshot what's
> being staged before doing it, and label the commit clearly as a WIP
> snapshot — not a clean science commit:
> ```bash
> git status --short > /tmp/dsi_bootstrap_snapshot.txt
> git ls-files --others --exclude-standard >> /tmp/dsi_bootstrap_snapshot.txt
> ```

Workflow:

```bash
# Step 1: in the existing checkout, create the bootstrap branch and
# commit the WIP snapshot per smoke-plan §2.1–§2.4. Use a commit
# message that flags it as a WIP snapshot, not curated work.
cd /home1/11114/zhixingliu/projects/SFNO_Climate_Emulator
# (smoke-plan §2 commands here — branch from zgplev-migration, edit
#  .gitignore, capture pip freeze, git add the WIP, commit with
#  "WIP-SNAPSHOT: …", push.)

# Step 2: NOW create a clean worktree from the bootstrap branch:
git worktree add ../AI-RES-dsi-bootstrap zgplev-migration-dsi-bootstrap
cd ../AI-RES-dsi-bootstrap
git status         # must be clean — all WIP is in the commit

# Step 3: apply full-plan §2.2 (submit.slurm refactor) + §3.5
# (stale --src comment fix) + new submit_zgplev_*.dsi.slurm files
# here. These are coherent edits worth their own commit, separate
# from the WIP snapshot.
```

When done: `git worktree remove ../AI-RES-dsi-bootstrap`.

If you skip step 1 and run `git worktree add … zgplev-migration`
directly, the worktree will **silently** lack `src/sfno_training/`
and the trainer won't import.

---

## 0a. Why phase 2 is non-trivial — five real blockers

External review (2026-05-01, against v3 smoke plan) called out five
blockers that v3 did not address. v4 owns each of them:

1. **The full v10 zgplev packaged root does not exist anywhere yet.**
   `data/makani/sim52_astro_64x128_zgplev_proto/` (21 GB, years 3–8 train,
   9 valid, 10 test) is the only v10 dataset on disk. Full training expects
   years 12–111 train + 11 valid. **No transfer strategy fixes this — we
   have to *build* the dataset first.**
2. **`src/plasim_makani_packager/submit.slurm`** unconditionally sets
   `SIMS=""`, `POSTPROC_ROOT=`, `BOUNDARY_ROOT=`, `OUTPUT_ROOT=` at lines
   14–17. `sbatch --export=POSTPROC_ROOT=foo …` does **not** work — the
   in-script assignment overwrites the env. The script must be edited
   before each run, or refactored to accept env defaults. v4 prefers
   refactor.
3. **`scripts/build_subset_dataset.py:121`** calls `target.resolve()` and
   then `os.symlink(target, link)`, producing **absolute** symlinks. A
   symlink farm built on Stampede3 cannot be rsync'd to DSI as-is — the
   targets resolve to `/scratch/11114/zhixingliu/...` paths that don't
   exist on DSI. v4 chooses between (a) transfer full root + rebuild
   subset on DSI, or (b) materialize/dereference the subset before
   transfer.
4. **DSI `general` QoS has a 12-hour wall**, and post-2026-05-07 jobs are
   preemptable. The Stampede3 full job requests 47:30:00
   (`submit_zgplev_full.slurm:10`). v4 needs chunked submission +
   checkpoint/resume + signal handling, OR a confirmed under-12h epoch
   that fits a single window.
5. **DSI scratch quotas may be too small** for hundreds of GB of data
   plus per-run checkpoints + experiment outputs. May need DSI project
   storage allocation, which is a multi-day request, not a 10-minute
   shell check.

---

## 1. The data problem — building the full v10 zgplev root

### 1.1 Where the input lives

The packager consumes:

| Input                                    | Path on Stampede3                                    | Size   |
|------------------------------------------|-------------------------------------------------------|--------|
| Postprocessed PlaSim NetCDF              | `/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/postproc/sim52/` | 472 GB |
| Boundary forcings (sst, rsdt, sic, …)    | `/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/boundary_astro/sim52/` | 18 GB  |

(Verified by `du -sh` in phase 1 inventory.)

### 1.2 What it produces

A packager-output root with the layout:

```
sim52_astro_64x128_zgplev/
├── train/  MOST.{YYYY}.h5      # one H5 per year
├── valid/  MOST.{YYYY}.h5
├── test/   MOST.{YYYY}.h5
├── stats/  global_means.npy / global_stds.npy / time_means.npy / forcing_*.npy
├── metadata/  data.json
└── config/
```

### 1.3 Estimated size of the full v10 root

`postproc/sim52/` and `boundary_astro/sim52/` each contain **128 files**
(years 1–128). Packager skips warmup years 1–2, so the packaged output
covers years 3–128 = **126 files** split across train/valid/test.

The proto dataset gives a per-year datapoint: 8 H5 files in 21 GB ⇒
~2.7 GB / year. v9 sigma-level full (`sim52_astro_64x128/`, 331 GB)
covers 126 years at ~2.6 GB/year.

The v10 zgplev root needs the same year coverage as v9. At ~2.7 GB/year
× 126 years ≈ **~340 GB**, plus stats (low MB). Plan for **~400 GB** to
be safe — v10 has additional `zg_plev` channels relative to v9.

### 1.4 Where to build it

**On Stampede3, in the `skx` queue.** The packager is CPU + I/O bound,
not GPU. Building on DSI would require transferring 472+18 = 490 GB of
postproc + boundary first, plus DSI compute time — net loss. Build on
Stampede3, transfer the much smaller packaged output.

---

## 2. Fix `src/plasim_makani_packager/submit.slurm` so it accepts env defaults

### 2.1 The bug

```bash
# Lines 14-17 today:
SIMS=""                 # e.g. "52"
POSTPROC_ROOT=          # postprocess output root, e.g. $SCRATCH/SFNO_Climate_Emulator/data/postproc
BOUNDARY_ROOT=          # boundary output root,    e.g. $SCRATCH/SFNO_Climate_Emulator/data/boundary_astro
OUTPUT_ROOT=            # packager output root,    e.g. $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev
```

These unconditional assignments clobber any `--export` from `sbatch`.
The fail-fast checks at lines 54–57 then trip, even though the user
exported the values.

### 2.2 The fix

Replace each unconditional `=""` / `=` with a default-if-unset:

```bash
SIMS="${SIMS:-}"
POSTPROC_ROOT="${POSTPROC_ROOT:-}"
BOUNDARY_ROOT="${BOUNDARY_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
POSTPROC_SOURCE_DIR="${POSTPROC_SOURCE_DIR:-}"
TRAIN_YEARS="${TRAIN_YEARS:-}"
VALID_YEARS="${VALID_YEARS:-}"
TEST_YEARS="${TEST_YEARS:-}"
SST_LAND_FILL_K="${SST_LAND_FILL_K:-}"
```

Now `sbatch --export=ALL,SIMS=52,POSTPROC_ROOT=…,OUTPUT_ROOT=… submit.slurm`
works. The fail-fast `: "${SIMS:?…}"` block at lines 54–57 still catches
truly unset cases.

This is a one-commit cleanup on `zgplev-migration` (or on the bootstrap
branch from phase 1). Open a PR — it benefits Stampede3 users too.

### 2.3 Sizing the array (no change needed)

The existing `--count-tasks` flow is correct:

```bash
N=$(python3 -m plasim_makani_packager.packager \
        --sims 52 \
        --train-years 3 100 \
        --valid-years 101 120 \
        --test-years 121 128 \
        --count-tasks)
echo "N=$N"
```

The full year coverage to feed `build_subset_dataset.py` for years
12–111 train + 11 valid needs source years that span both: **train 3–100
+ valid 101–120** matches the v9 plan and lets the subset builder pull
year 11 (lives in `train/`) and years 12–111 (12–100 from `train/`,
101–111 from `valid/` — the builder searches all source splits).

---

## 3. Run the packager on Stampede3

The packager is **three independent modules**, not one. The array job
runs `packager.py` (per-year H5). Stats and metadata are produced
separately by `stats.py` and `metadata.py`, after the array completes.
**`build_subset_dataset.py:163`** rejects a source root that is missing
any of `train/`, `valid/`, `test/`, `stats/`, `metadata/`, `config/`,
so all three stages must run before the subset is buildable.

> **Where to run §3 from.** All §3 commands must run against a tree
> where the §2.2 `submit.slurm` env-default refactor has landed —
> otherwise the array submit at §3.2 hits the unfixed
> `SIMS=""`/`POSTPROC_ROOT=` clobber. Two options:
>
> 1. Run §3 from the bootstrap worktree (`../AI-RES-dsi-bootstrap`)
>    where §2.2 is already committed. **Preferred.**
> 2. Or, in `$HOME/AI-RES`, fast-forward / cherry-pick the §2.2 commit
>    from `zgplev-migration-dsi-bootstrap` before running §3.
>
> §3 commands below use a `$REPO` variable. Set it once at the top of
> your shell session to whichever tree has the fix:
>
> ```bash
> REPO=/home1/11114/zhixingliu/projects/SFNO_Climate_Emulator-dsi-bootstrap   # or $HOME/AI-RES if cherry-picked
> ```
>
> Then every `cd "$REPO"` below operates on the right tree. This
> avoids the copy-paste-literal trap.

### 3.1 One-year smoke first

Before launching 126 array tasks, validate **three** years end-to-end —
one in each split. Using the same year for train/valid/test would land
all three in `train/` because `packager.py:441` `resolve_split` returns
the **first** matching split (train wins).

Run on a compute node, not a login node — `stats.py` reads whole H5
arrays per file (`stats.py:159`) and is memory-noisy:

```bash
# On Stampede3, in a compute-node interactive shell:
srun -p skx -t 01:00:00 -N 1 -n 1 --pty bash -l
cd "$REPO"     # set REPO once per session — see §3 prologue
source .venv/bin/activate
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"   # required for `-m plasim_makani_packager.*`

# Stage A — three packager tasks: one train, one valid, one test.
# Use distinct years so each file lands in its own split:
for IDX in 0 1 2; do
    python3 -m plasim_makani_packager.packager \
        --sims 52 \
        --postproc-root $SCRATCH/SFNO_Climate_Emulator/data/postproc \
        --boundary-root $SCRATCH/SFNO_Climate_Emulator/data/boundary_astro \
        --output-root $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev_test \
        --train-years 11 11 \
        --valid-years 101 101 \
        --test-years 121 121 \
        --postprocessor-git-sha "$(git -C src/plasim_postprocessor rev-parse HEAD)" \
        --task-index "$IDX"
done

# Stage B — stats (only scans output_root/train per stats.py:128, so
# the train range here must match what's in train/):
python3 -m plasim_makani_packager.stats \
    --output-root $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev_test \
    --train-years 11 11

# Stage C — metadata + config:
python3 -m plasim_makani_packager.metadata \
    --output-root $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev_test \
    --variant zgplev \
    --train-years 11 11 \
    --valid-years 101 101 \
    --test-years 121 121

# Stage D — full validation (checks files + stats + metadata + smoke-load):
python3 -m plasim_makani_packager.validate \
    --mode full \
    --output-root $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev_test
```

`validate --mode full` rejects `unknown` postprocessor SHAs in
production. If the smoke output validates, scale to the full array.

### 3.2 Full array submit

```bash
cd "$REPO"     # set REPO once per session — see §3 prologue
source .venv/bin/activate
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

# Count tasks for the full year coverage:
N=$(python3 -m plasim_makani_packager.packager \
        --sims 52 \
        --train-years 3 100 \
        --valid-years 101 120 \
        --test-years 121 128 \
        --count-tasks)
echo "N=$N"   # expect 126 (years 3-128, sim 52)

# Submit the array. Assumes §2.2 fix has landed; without it, edit
# submit.slurm lines 14-17 in place before sbatch.
sbatch \
    --array=0-$((N-1)) \
    --export=ALL,\
SIMS=52,\
POSTPROC_ROOT=$SCRATCH/SFNO_Climate_Emulator/data/postproc,\
BOUNDARY_ROOT=$SCRATCH/SFNO_Climate_Emulator/data/boundary_astro,\
OUTPUT_ROOT=$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev,\
TRAIN_YEARS="3 100",\
VALID_YEARS="101 120",\
TEST_YEARS="121 128" \
    src/plasim_makani_packager/submit.slurm
```

### 3.3 Wall time for the array

`-t 01:00:00` per array task. With 126 tasks and `skx` queue
concurrency, wall is queue-dominated. Plan for **8–48 hours from submit
to all-tasks-done** depending on `skx` load.

### 3.4 After the array completes — stats + metadata + validate

The array writes only `train/`, `valid/`, `test/`. Run the remaining
stages once after all 126 tasks are SUCCESS — **on a compute node**,
not a login node (`stats.py:159` reads whole H5 arrays per file):

```bash
srun -p skx -t 02:00:00 -N 1 -n 1 --pty bash -l
cd "$REPO"     # set REPO once per session — see §3 prologue
source .venv/bin/activate
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

# Stats from train-only years per group convention (3-100):
python3 -m plasim_makani_packager.stats \
    --output-root $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev \
    --train-years 3 100

# Metadata + config:
python3 -m plasim_makani_packager.metadata \
    --output-root $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev \
    --variant zgplev \
    --train-years 3 100 \
    --valid-years 101 120 \
    --test-years 121 128

# Final validation:
python3 -m plasim_makani_packager.validate \
    --mode full \
    --output-root $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev
```

Verify file counts before declaring done:

```bash
ls $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev/train/ | wc -l   # 98 (years 3-100)
ls $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev/valid/ | wc -l   # 20 (years 101-120)
ls $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev/test/  | wc -l   # 8  (years 121-128)
ls $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev/stats/           # global_means/stds/time_means + forcing_*
ls $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev/metadata/        # data.json
ls $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev/config/          # *.yaml
du -sh $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev/             # ~340 GB
```

### 3.5 Stale comment in `submit_zgplev_full.slurm`

`src/sfno_training/submit_zgplev_full.slurm:36-40` documents:

```
scripts/build_subset_dataset.py \
    --src $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128 \
    --dst $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full \
    --train-years 12-111 \
    --valid-years 11
```

`--src` points at the v9 sigma path; v10 zgplev needs
`sim52_astro_64x128_zgplev`. Fix in the same commit as §2.2.

### 3.6 Science gate — stats normalization year range

The `build_subset_dataset.py` design **shares stats across subsets**:
the full dataset's `stats/` is symlinked into every subset
(`build_subset_dataset.py:169`), so all subsets train against the same
normalization (per the `docs/sfno_tiny_short_training_plan.md`
rationale).

For full v10 zgplev training, this means: stats computed from train
years 3–100 (in §3.4) are used to normalize a training set that covers
years 12–**111**. Years 12–100 overlap; years 101–111 don't and will
be normalized using out-of-distribution stats.

**Two paths:**

- **Path I (DEFAULT, matches v9 / proto convention):** keep stats from
  3–100. Justification: the convention has held in v9 and proto runs
  without observable issue.

- **Path II (NOT a one-line change — `stats.py` only scans `train/`):**
  Recompute stats from the actual training year range 12–111.
  `stats.py:128` only walks `output_root/train` — so just passing
  `--train-years 12 111` against the full root will **silently miss
  years 101–111** (which live under `valid/` in the full root).

  If you really want Path II, the procedure is:

  ```bash
  # 1. Build the subset on Stampede3 first (years 12-111 land in subset's train/):
  scripts/build_subset_dataset.py \
      --src $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev \
      --dst $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full \
      --train-years 12-111 --valid-years 11

  # 2. The subset's stats/ is a symlink to the FULL root's stats/.
  #    Replace it with a real directory before running stats — otherwise
  #    we'd corrupt the shared full-root stats.
  rm $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full/stats
  mkdir $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full/stats

  # 3. Now stats.py sees train/ with years 12-111 inside the SUBSET:
  python3 -m plasim_makani_packager.stats \
      --output-root $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full \
      --train-years 12 111

  # 4. Update subset metadata.json. The subset's test/ is empty
  #    (build_subset_dataset.py creates it as an empty placeholder), so
  #    pass a zero-width range so metadata reflects the actual layout:
  python3 -m plasim_makani_packager.metadata \
      --output-root $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full \
      --variant zgplev \
      --train-years 12 111 --valid-years 11 11 --test-years 0 0
  ```

  This breaks the share-stats-across-subsets convention. May change
  reproducibility against v9. Flag for explicit modeling-lead approval.

**Default to Path I** and flag for review with the modeling lead before
launch.

---

## 4. Subset strategy — choose between A and B

After §3 the full v10 root exists. We do not want to transfer the **full
~340 GB** if we only train on 101 years (subset). Two options. Both
land on **`$DSI_PROJECT`**, not scratch — the 50 GB scratch default
makes scratch unusable here even after a quota request.

### 4.A Transfer the full root → rebuild the subset on DSI

**Transfer:** ~340 GB of `sim52_astro_64x128_zgplev/` to
`$DSI_PROJECT/AI-RES/data/makani/`.
**On DSI:** run `scripts/build_subset_dataset.py` to produce the symlink
farm. Both source and subset live on DSI; symlinks are absolute but
point at DSI paths, so they resolve.

Pros:
- Single transfer.
- Subset is rebuildable on DSI without round-tripping (e.g. for years
  12–111 today, years 1–10 + 112–128 tomorrow).
- DSI gets the same artifact Stampede3 has.

Cons:
- Larger transfer (~340 GB vs ~280 GB).
- More project storage consumed (~340 GB root + 0 GB subset symlinks).

### 4.B Materialize the subset on Stampede3 → transfer just the subset

**On Stampede3:** build the subset, then `rsync -aL` (with `-L` to
**dereference** symlinks, copying real files) into a holding directory:

```bash
# On Stampede3:
scripts/build_subset_dataset.py \
    --src $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev \
    --dst $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full \
    --train-years 12-111 --valid-years 11

# Materialize for transfer (symlink farm → real-file copy):
mkdir -p $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full_materialized
rsync -aLh --info=progress2 \
    $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full/ \
    $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full_materialized/
```

Then transfer `sim52_zgplev_full_materialized/` to DSI (~280 GB:
100 train + 1 valid + stats/metadata/config).

Pros:
- Smaller transfer (~280 GB).
- DSI project storage holds only what training needs.

Cons:
- On Stampede3, materialization doubles the on-disk usage for the subset
  files (~280 GB extra) for the duration of the transfer.
- If the year range changes, you re-do the materialize + transfer.

### 4.C Recommended

**Path A** is preferred unless DSI project quota is tight (see §5.1).
It matches Stampede3's layout, leaves room for re-subsetting on DSI,
and costs only ~60 GB more than B.

Decision gate: §5.1 quota on project storage. If DSI project free space
< 400 GB after the quota increase, pick B.

---

## 5. DSI side — additional questions for full training

These are **on top of** the phase 1 questions (smoke plan §3). Resolve
before §6 transfer.

### 5.1 Storage is a HARD GATE (not a formality)

DSI default quotas (per cluster-policy.ds.uchicago.edu):

| Volume   | Default quota | Backed up? | Purge risk? |
|----------|---------------|------------|-------------|
| `$HOME`  | small (GB)    | yes        | no          |
| `$SCRATCH` (per location) | **50 GB** | no | yes |
| `$PROJECT` | **500 GB**  | yes        | no          |

This plan needs **280–340 GB of data alone**, far above the 50 GB
scratch default. Path A (full-root transfer) at 340 GB also exceeds the
500 GB project default once `EXP_DIR` (~50 GB / run × N runs) and any
sibling experiments are added.

**Therefore:** before any data transfer, the project storage quota
**must be increased** via DSI staff. This is **not** a self-serve
flag — it requires a request and approval that can take **3–10
business days**. Start early; this is the longest-lead-time item in
the plan after the Stampede3 packager run itself.

**Recommended placement (post-quota-increase):**

| Artifact                                                              | Location                  | Why                                        |
|-----------------------------------------------------------------------|---------------------------|--------------------------------------------|
| Packaged data (read-only)                                             | `$DSI_PROJECT/AI-RES/data/` | Too big for scratch default; durable      |
| `EXP_DIR` (rendered yaml, training_checkpoints/, wandb/)              | `$DSI_PROJECT/AI-RES/runs/` | Important, backed up, durable             |
| Logs (`logs/`)                                                        | `$HOME/AI-RES/logs/`      | Small, want them surviving any purge      |

(Note: I previously suggested `$DSI_SCRATCH/AI-RES/data/`. The 50 GB
scratch default makes that impossible without quota work. Project
storage hosts both data and `EXP_DIR`.)

Update §7's submit script to set both
`OUTPUT_ROOT="$DSI_PROJECT/AI-RES/data/makani/sim52_zgplev_full"` and
`EXP_DIR="$DSI_PROJECT/AI-RES/runs/sfno_zgplev_full_dsi"`.

DSI quota check:

```bash
dsiquota                          # current usage and quota across all volumes
df -h $DSI_PROJECT                # confirm free bytes after quota increase
```

Required free space (per §4 choice):

| §4 path | Data on project | EXP_DIR on project | Logs on home |
|---------|-----------------|--------------------|--------------|
| A (full root + subset)   | ~340 GB | ~50 GB / run | ~5 GB / run |
| B (materialized subset)  | ~280 GB | ~50 GB / run | ~5 GB / run |

**Plan does not proceed past §5.1 until DSI confirms a project quota of
≥ 500 GB headroom over current usage.**

### 5.2 GPU class

`--gres=gpu:1` does not pin GPU model. Full SFNO with `batch_size=4` +
`amp_mode=bf16` + `checkpointing_level=2` (per
`submit_zgplev_full.slurm:90-98`) needs:

- **bf16 native** support → Ampere or newer (A100 / A40 / A6000 / H100).
- **≥ 40 GB GPU memory** (group convention; smaller GPUs have OOMed in
  prior runs).

Ask DSI for the GPU inventory and pin the script:

```bash
#SBATCH --gres=gpu:a100:1                    # or whatever DSI exposes
#SBATCH --constraint=<gpu_class_constraint>  # if GRES typing isn't available
```

### 5.3 Preemption / QoS post-2026-05-07

Already in phase 1 plan, but here it bites harder:

- `--qos=protected` — short cap, no preempt. Likely too short for full.
- `--qos=general` — preemptable. Job may die mid-epoch.

Need DSI's documented `--time` cap per QoS, and whether GPU jobs are
preemptable on the same schedule as CPU jobs.

### 5.4 Allocation

Full training is many GPU-hours. Confirm `-A <account>` is set + has
budget. Stampede3 jobs use TACC allocations; DSI uses its own.

---

## 6. Transfer the data (Globus, large) — destination is `$DSI_PROJECT`

Per §5.1, both data and `EXP_DIR` live on **project storage**, not
scratch. All §6 paths use `$DSI_PROJECT`; do not write to scratch.

### 6.1 Pre-transfer DSI checks (storage gate)

```bash
# On DSI:
dsiquota                          # confirm project quota approved + sufficient
df -h "$DSI_PROJECT"              # at least 400 GB free for path A
test -w "$DSI_PROJECT" && echo OK # writable
```

If any of these fails, **do not proceed**. Re-engage DSI staff.

### 6.2 Sizes (depend on §4 choice)

| §4 path | Source                                               | Dest                                                                | Size   |
|---------|------------------------------------------------------|---------------------------------------------------------------------|--------|
| A       | `data/makani/sim52_astro_64x128_zgplev/`             | `$DSI_PROJECT/AI-RES/data/makani/sim52_astro_64x128_zgplev/`        | ~340 GB |
| B       | `data/makani/sim52_zgplev_full_materialized/`        | `$DSI_PROJECT/AI-RES/data/makani/sim52_zgplev_full/`                | ~280 GB |

### 6.3 Globus (the only viable tool here)

rsync-over-ssh on a login node will not survive a 340 GB transfer
without intermittent failures + restart pain. Use Globus:

1. Activate Stampede3 (`tacc#stampede3`) and DSI Globus endpoints.
2. Source path: chosen per §4.
3. Destination path: under `$DSI_PROJECT` per §6.2.
4. Enable: preserve mtimes, **verify file integrity after transfer**,
   **sync modified only** (so a re-transfer skips already-good files).
5. Submit; expect 4–24 hours wall depending on link.
6. After completion, integrity-check on DSI:
   ```bash
   find $DSI_PROJECT/AI-RES/data/makani/sim52_astro_64x128_zgplev -name '*.h5' | wc -l
   # Expect 126 (98 train + 20 valid + 8 test).
   du -sh $DSI_PROJECT/AI-RES/data/makani/sim52_astro_64x128_zgplev
   # Expect ~340 GB ± 5%.
   ```

### 6.4 If Path A: rebuild the subset on DSI

```bash
# On DSI, after transfer:
cd $HOME/AI-RES
source .venv/bin/activate
scripts/build_subset_dataset.py \
    --src $DSI_PROJECT/AI-RES/data/makani/sim52_astro_64x128_zgplev \
    --dst $DSI_PROJECT/AI-RES/data/makani/sim52_zgplev_full \
    --train-years 12-111 \
    --valid-years 11
```

### 6.5 Final pre-submit check (run before any `sbatch`)

Applies to **both Path A** (rebuilt subset) and **Path B** (materialized
subset transferred directly):

```bash
ROOT=$DSI_PROJECT/AI-RES/data/makani/sim52_zgplev_full
find "$ROOT/train" -name '*.h5' | wc -l    # expect 100 (years 12-111)
find "$ROOT/valid" -name '*.h5' | wc -l    # expect 1   (year 11)
find "$ROOT/test"  -name '*.h5' | wc -l    # expect 0   (subset has empty test/)
ls "$ROOT/stats"                            # global_means/stds/time_means + forcing_*
ls "$ROOT/metadata/data.json"               # exists
ls "$ROOT/config"                           # *.yaml present
find "$ROOT" -xtype l -print                # MUST be empty (broken symlinks: only possible from Path A on a partial transfer of the full root)

# Defensive check: any symlink whose target points back at Stampede3
# paths. A correctly-built Path A subset will only have absolute
# symlinks under DSI; Path B has no symlinks at all. Anything matching
# /scratch/11114/* / /work2/* / /home1/* means the wrong artifact got
# transferred:
find "$ROOT" -type l -exec sh -c '
    for p; do
        t=$(readlink "$p")
        case "$t" in
            /scratch/11114/*|/work2/*|/home1/*) echo "STAMPEDE3-LINK: $p -> $t" ;;
        esac
    done' sh {} +
# Empty output = clean.

# Logs dir must exist BEFORE sbatch — SLURM opens stdout/stderr at
# submission time, before the script body runs. Mkdir-inside-script
# is too late for #SBATCH -o / -e.
mkdir -p "$HOME/AI-RES/logs"

# Storage gate (re-check):
dsiquota
df -h "$DSI_PROJECT"
test -w "$DSI_PROJECT"
```

Any failure here means the transfer/subset build is incomplete.

---

## 7. DSI full-training SLURM script (separate from smoke wrapper)

The phase-1 smoke wrapper is **not appropriate** for full training — it
runs `submit_zgplev_smoke.slurm`. Create a sibling for full.

> **Important.** `exec bash <script>` replaces the wrapper's process,
> which **discards any traps installed in the wrapper.** Don't install
> traps before `exec`. SIGUSR1 handling has to live inside
> `train_plasim.py` (it currently doesn't — see §8.3) or in a wrapper
> that does **not** `exec` and instead launches the trainer directly.
> v4 takes the second path below.

`src/sfno_training/submit_zgplev_full.dsi.slurm`:

```bash
#!/bin/bash
#SBATCH -J sfno_zgplev_full_dsi
#SBATCH --partition=<DSI_GPU_PARTITION>
#SBATCH --gres=gpu:a100:1                    # confirm exact syntax in §5.2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A <DSI_ACCOUNT>
#SBATCH --time=12:00:00                      # DSI general cap
#SBATCH --qos=general                        # OR --qos=protected if available + sufficient
# NOTE: --signal=B:USR1@300 is INTENTIONALLY NOT SET. Today, Python's
# default SIGUSR1 disposition terminates immediately, so the signal would
# kill the trainer 5 min before SLURM would have anyway, with no chance
# to flush a checkpoint. Add this directive only after mitigation B in
# §8.3 (a SIGUSR1 handler in train_plasim.py) lands.
#SBATCH --requeue                            # auto-resubmit on preempt
#SBATCH -o logs/sfno_zgplev_full_dsi_%j.out
#SBATCH -e logs/sfno_zgplev_full_dsi_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE

set -euo pipefail

REPO_ROOT="$HOME/AI-RES"

# DSI_PROJECT must be set from outside; no placeholder default — a typo
# in /project/<group>/$USER would silently route writes to a path that
# doesn't exist. Caller is expected to `export DSI_PROJECT=...` before
# `sbatch submit_zgplev_full.dsi.slurm`.
: "${DSI_PROJECT:?export DSI_PROJECT (e.g. /project/<group>/$USER) before sbatch}"
test -d "$DSI_PROJECT" || {
    echo "DSI_PROJECT=$DSI_PROJECT is not a directory" >&2
    exit 1
}
test -w "$DSI_PROJECT" || {
    echo "DSI_PROJECT=$DSI_PROJECT not writable" >&2
    exit 1
}
# DSI_SCRATCH not needed: per §5.1, data + EXP_DIR live on project storage.

# CUDA env (DSI uses /usr/local/cuda-*, not modules):
export CUDA_HOME=/usr/local/cuda-12.4                 # SUBSTITUTE per phase-1 §3.7
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Per §5.1, both data and EXP_DIR live on project storage (scratch's
# 50 GB default is too small; quota increase MUST land before this point):
# Default-if-unset so callers can override at sbatch time via
# `--export=ALL,EXP_DIR=...,OUTPUT_ROOT=...` (used by the §8.4 gate).
# Without `:-`, the unconditional assignment would clobber the export.
export OUTPUT_ROOT="${OUTPUT_ROOT:-$DSI_PROJECT/AI-RES/data/makani/sim52_zgplev_full}"
export EXP_DIR="${EXP_DIR:-$DSI_PROJECT/AI-RES/runs/sfno_zgplev_full_dsi}"
export MAIL_TO="zhixingliu@uchicago.edu"
export SFNO_MAIL_FROM="${USER}@uchicago.edu"
export SFNO_SMTP_RELAY="${SFNO_SMTP_RELAY:-smtp.uchicago.edu}"

# The original Stampede3 script references $SCRATCH only via DEFAULT
# values for OUTPUT_ROOT / EXP_DIR (submit_zgplev_full.slurm:47-48).
# Since this wrapper EXPORTS both, the defaults are unused — we do NOT
# alias $SCRATCH here, to avoid hiding scratch/project mistakes.

# scan_for_nans.py is invoked at the end of submit_zgplev_full.slurm
# with `logs/sfno_zgplev_full_${SLURM_JOB_ID}.{out,err}`, but this
# wrapper writes `logs/sfno_zgplev_full_dsi_${SLURM_JOB_ID}.{out,err}`
# — naming mismatch. The scan call is guarded by `|| true` so it won't
# fail the run, but it also won't actually scan. Either symlink the
# DSI logs to the expected names, or skip the scan on DSI.
# Quick mitigation: skip by overriding the script that ships scan_for_nans.py.

# DO NOT `exec bash …`. We need to keep this shell alive so we can
# (eventually) install a SIGUSR1 forwarder once train_plasim.py supports
# it. For now, run the original Stampede3 full script as a sub-shell —
# its body activates the venv, renders YAML, runs preflight, then trains.
# Trainer auto-resumes from latest checkpoint (load_checkpoint=legacy at
# argument_parser.py:61).
bash "$REPO_ROOT/src/sfno_training/submit_zgplev_full.slurm"
```

`bash <script>` (without `exec`) keeps the wrapper alive as the parent
process. SLURM's `--signal=B:USR1@300` delivers SIGUSR1 to the **batch
script's process group**. When §8.3 lands, the wrapper can install a
trap that forwards the signal into the trainer's PID. Until then,
`--signal` does nothing useful and the kill is hard.

---

## 8. Checkpoint / resume / preemption — the survival contract

DSI gives 12 h. Full training is many epochs × 100 train years. We
survive across runs by **resume from latest checkpoint each submission**.

### 8.1 What the trainer already does

`src/sfno_training/train_plasim.py:163-167` writes:

- `$EXP_DIR/<config>/<run>/training_checkpoints/ckpt_mp{mp_rank}_v{checkpoint_version}.tar`
- `$EXP_DIR/<config>/<run>/training_checkpoints/best_ckpt_mp{mp_rank}.tar`

(`.tar`, **not** `.pt`.) Resume detection at `train_plasim.py:172-176`
sets `params["resuming"] = True` if every expected `ckpt_mp{N}_v0.tar`
exists. So repeated `sbatch submit_zgplev_full.dsi.slurm` invocations
auto-resume.

**Checkpoint cadence:** epoch-end only. `deterministic_trainer.py:388`
saves once at the end of each epoch's train loop. There is no
mid-epoch checkpoint. Implication: a 12 h job killed mid-epoch loses
**everything since the previous epoch boundary**. Before launching full
training, prove from the §8.4 one-epoch full-data DSI gate that one epoch + validation finishes
**well under 12 h** so progress is bounded by ≤1 epoch.

**Atomicity warning.** `makani-src/.../driver.py:570` calls
`torch.save(store_dict, checkpoint_fname)` directly — **not** to a
`.tmp` + rename. If SLURM kills the job during the save, that
checkpoint version is corrupt.

**Resume-detect bug to be aware of:**
`src/sfno_training/train_plasim.py:172` checks specifically for
`ckpt_mp0_v0.tar` (hardcoded `checkpoint_version=0`). If `_v0` is the
corrupt one, simply `mv`-ing it aside makes resume fail entirely —
the trainer concludes there is no checkpoint and starts from epoch 0,
silently dropping all valid `_v1`, `_v2` files. Recovery must
**replace** `_v0` with a known-good version, not just remove it:

```bash
CKPT_DIR=$EXP_DIR/plasim_sim52_zgplev_full/0/training_checkpoints
LATEST=$(ls -t "$CKPT_DIR"/ckpt_mp0_v*.tar | head -1)

# Confirm corruption:
python -c "import torch; torch.load('$LATEST', map_location='cpu', weights_only=False)" || \
    echo "CORRUPT: $LATEST"

# If $LATEST is _v0 and corrupt, find the next-newest valid version
# and copy it over _v0 (don't move — keep the rotation slot intact):
for CAND in $(ls -t "$CKPT_DIR"/ckpt_mp0_v*.tar | grep -v "$(basename $LATEST)"); do
    if python -c "import torch; torch.load('$CAND', map_location='cpu', weights_only=False)" 2>/dev/null; then
        mv "$LATEST" "$LATEST.corrupt"
        cp "$CAND" "$CKPT_DIR/ckpt_mp0_v0.tar"
        echo "recovered from $CAND"
        break
    fi
done
```

This is why `checkpoint_num_versions ≥ 2` matters — confirm in the YAML
config before launch. **Default in `makani-src/.../driver.py:140` is 3**
(not 2), so unless the YAML explicitly overrides downward, recovery
always has 2 fallback versions to try.

**Long-term fix:** patch `train_plasim.py:170-176` to glob for the
newest valid `ckpt_mp{N}_v*.tar` instead of hardcoding `_v0`. Out of
scope here, but flag for code review.

### 8.2 What `--requeue` does

When SLURM preempts the job, it kills the process. With `--requeue`,
SLURM re-queues the same job ID; when it next runs, the trainer finds
the latest checkpoint and continues. **But** — only checkpoints written
to disk **before the kill** survive.

### 8.3 What's missing — and what `--signal` actually does today

`train_plasim.py` does **not** install a SIGUSR1 handler (verified by
`grep -n SIGUSR1 src/sfno_training/train_plasim.py` — empty). The
`--signal=B:USR1@300` SLURM directive will deliver SIGUSR1 to the batch
script's process group 5 min before the kill, but:

1. Python's default SIGUSR1 disposition is to terminate the interpreter
   immediately. So the signal currently *worsens* the situation: the
   trainer dies 5 min before SLURM would have killed it anyway, with
   the same loss of in-flight work.
2. Wrapper-installed traps don't help: `bash` forwards SIGUSR1 to its
   foreground job (the trainer), not to bash itself, unless we trap
   USR1 in the wrapper *and* the trainer is launched with `&` + `wait`.

**Three viable paths, in increasing order of work:**

- **Mitigation A (zero code, accept loss):** drop `--signal` entirely.
  Tune the YAML so end-of-epoch checkpoints are frequent enough that a
  hard kill at the 12 h boundary loses ≤1 epoch of progress. Best for
  first DSI submission.
- **Mitigation B (small code in trainer):** add a SIGUSR1 handler in
  `train_plasim.py` near the train loop that sets a "save & exit after
  this batch" flag, then writes the checkpoint and `sys.exit(0)`. Then
  `--signal=B:USR1@300` becomes useful. **This is the right long-term
  fix — flag for code review.**
- **Mitigation C (wrapper-side):** in the DSI submit script, run the
  trainer in the background with `&`, capture its PID, and trap SIGUSR1
  to forward it to the PID. Doesn't help unless mitigation B is also
  done — Python still terminates by default.

For first DSI full-training submission, default to **Mitigation A**.
Confirm a checkpoint exists in
`$EXP_DIR/.../training_checkpoints/` after each 12 h window (any
`ckpt_mp{N}_v*.tar` file is sufficient). If preemption losses are
unacceptable, schedule Mitigation B as a separate PR.

### 8.4 One-epoch DSI gate (must pass BEFORE multi-window full launch)

Smoke (proto data, smoke YAML) does **not** measure epoch wall over 100
real train years on the full model. Before committing to multi-day
multi-window training, run a single-epoch DSI job using the **full**
config + dataset.

`train_plasim.py` does **not** accept `--max_epochs` as a CLI flag
(verified: `argument_parser.py` has no such argument). The override
must go in the rendered YAML (`max_epochs: 50` at line 110, plus
`scheduler_T_max: 45` at line 121, both in
`src/sfno_training/config/plasim_sim52_zgplev_full.yaml`).

```bash
export DSI_PROJECT=/project/<your-group>/$USER     # NOT a placeholder
export OUTPUT_ROOT="$DSI_PROJECT/AI-RES/data/makani/sim52_zgplev_full"
export EXP_DIR="$DSI_PROJECT/AI-RES/runs/sfno_zgplev_full_dsi_1epoch_gate"

# Pre-render the YAML with max_epochs=1 in place. The wrapper's body
# (the original Stampede3 script it bash-invokes) re-renders the YAML
# from template via sed at submit time, so we need to either:
#   (a) edit the rendered YAML AFTER the wrapper renders it but BEFORE
#       train_plasim.py reads it — fragile, and the wrapper bash-execs
#       the original synchronously, so there's no clean injection point;
#   (b) prepare a one-epoch template YAML and point the original
#       script at it via env override; or
#   (c) a dedicated DSI gate wrapper that renders to a temp YAML and
#       runs train_plasim.py directly.
#
# Option (c) is cleanest. Recommended skeleton:
#   src/sfno_training/submit_zgplev_full.dsi_1epoch.slurm
# — same #SBATCH block as submit_zgplev_full.dsi.slurm, then:
#       sed -e "s|{{OUTPUT_ROOT}}|$OUTPUT_ROOT|g" \
#           -e "s|{{EXP_DIR}}|$EXP_DIR|g" \
#           -e "s|^[[:space:]]*max_epochs:.*|    max_epochs: 1|" \
#           -e "s|^[[:space:]]*scheduler_T_max:.*|    scheduler_T_max: 1|" \
#           src/sfno_training/config/plasim_sim52_zgplev_full.yaml \
#           > "$EXP_DIR/plasim_sim52_zgplev_full.rendered.yaml"
#       python -m sfno_training.train_plasim \
#           --yaml_config "$EXP_DIR/plasim_sim52_zgplev_full.rendered.yaml" \
#           --config plasim_sim52_zgplev_full \
#           --run_num 0 --batch_size 4 --multistep_count 1 \
#           --amp_mode bf16 --checkpointing_level 2 --disable_ddp
#
# Submit (after creating that gate wrapper):
sbatch --export=ALL,DSI_PROJECT="$DSI_PROJECT",OUTPUT_ROOT="$OUTPUT_ROOT",EXP_DIR="$EXP_DIR" \
    src/sfno_training/submit_zgplev_full.dsi_1epoch.slurm
```

**Gate criterion:** one full epoch + validation + checkpoint write
completes in well under 12 h (≤ 8 h leaves ~4 h headroom for jitter
and validation). If it doesn't fit, **do not launch full training on
DSI** until either (a) batch size / amp / checkpointing tuned to fit,
or (b) DSI grants higher walltime QoS.

If it fits cleanly, proceed to §8.5 multi-window.

### 8.5 Multi-window submit cadence (after §8.4 passes)

```bash
# DSI_PROJECT must be exported AND passed via --export so the SLURM
# child shell sees it (DSI's batch env may not propagate unset vars):
export DSI_PROJECT=/project/<your-group>/$USER
sbatch --export=ALL,DSI_PROJECT="$DSI_PROJECT" \
    src/sfno_training/submit_zgplev_full.dsi.slurm

# After it ends (success or preempt), resubmit. With --requeue, SLURM
# does this automatically on preempt; on clean end, you resubmit manually
# until convergence.
```

`--requeue` only helps **after** an epoch checkpoint has been written.
A 12 h wall-time timeout is **not** a preempt and SLURM does NOT
auto-requeue it — you must manually resubmit, and progress past the
last epoch is lost (no SIGUSR1 handler yet, see §8.3).

For automation, chain via `sbatch --dependency=afterany:<JID>` for the
next window. Still manual on the bookkeeping side — monitor
`$EXP_DIR/.../training_checkpoints/` epoch counter to know when to stop.

### 8.5 Expected wall

Unknown without a first DSI run measurement. The Stampede3 full job
budgets 47:30 h on H100 for the whole training. Even on a slower DSI GPU
with 2× wall, expect **~5–10 × 12-hour windows** to convergence —
i.e. **3–7 days of calendar time** assuming continuous queueing.

---

## 9. Verification on DSI (per submission)

Two important details:

- **Epoch summaries live in `$EXP_DIR/.../out.log`**, written by
  `train_plasim.py:197`. The Slurm `.err` is just stderr; structured
  per-epoch lines (`training loss`, `validation loss`) go to `out.log`.
- **Checkpoint version rotates**, per
  `makani-src/makani/utils/training/deterministic_trainer.py:395`:
  `self.checkpoint_version_current = (… + 1) % checkpoint_num_versions`.
  So files are `ckpt_mp0_v0.tar`, `ckpt_mp0_v1.tar`, … cycling. Don't
  pin to `_v0`; pick the newest:

```bash
# 1. Find the latest checkpoint (mp_rank=0 because submit_zgplev_full.slurm:98 sets --disable_ddp):
LATEST=$(ls -t "$EXP_DIR"/plasim_sim52_zgplev_full/0/training_checkpoints/ckpt_mp0_v*.tar | head -1)
echo "$LATEST"

# 2. Load it to confirm well-formed:
python - <<PY
import torch
ckpt = torch.load("$LATEST", map_location="cpu", weights_only=False)
print("keys:", sorted(ckpt.keys()))
for k in ("epoch", "iters", "global_step"):
    if k in ckpt: print(f"{k}: {ckpt[k]}")
PY

# 3. Read the actual training log (NOT the slurm .err):
grep -E 'training loss|validation loss' \
    "$EXP_DIR"/plasim_sim52_zgplev_full/0/out.log | tail -20
# Look for finite, decreasing losses. Inf / NaN = stop and debug.
```

**Success criterion (single-run):** the latest `ckpt_mp0_v*.tar` mtime
is newer than the previous run, it loads via `torch.load`, and
`out.log` shows finite training/validation losses with epoch counter
advanced from the previous run.

**Success criterion (overall):** validation-loss curve in `out.log` on
DSI matches Stampede3 within reasonable noise margin.

> **Mail is not a success criterion** (inherited from phase-1 §5).

---

## 10. Realistic time + cost budget

| Step                                              | Time              |
|---------------------------------------------------|-------------------|
| Phase 1 (smoke plan v3) complete                  | 1–3 working days  |
| §2 fix `submit.slurm` + §3.4 stale comment + PR   | 1 hour            |
| §3.1 three-split-year packager smoke (3 tasks + stats + metadata + validate) | 1–4 h queue+run |
| §3.2 packager array on Stampede3 (skx queue, 126 tasks) | 8–48 h queue+run  |
| §3.4 stats + metadata + validate full root (98 train H5s, whole-array reads) | 1–3 h compute     |
| §4 subset / materialize decision + execute        | 1–4 h             |
| §5 DSI quota check + project-storage request if needed | **3–10 business days if escalation** |
| §6 Globus 280–350 GB                              | 4–24 h            |
| §6.3 rebuild subset on DSI (path A only)          | 5 min             |
| §7 author + commit DSI full SLURM                 | 1 h               |
| §8 verify SIGUSR1 handling, decide A vs B         | 1–4 h             |
| §9 first DSI full submission                      | 12 h GPU + queue  |
| Multi-window training to convergence              | 3–7 calendar days |
| **Total to first converged DSI checkpoint**       | **~2 weeks**      |

This is **not** a faster path than waiting for Stampede3 unless
Stampede3 queue waits exceed the data + env build overhead. v4's value
is keeping DSI as a viable second pipe, not shortcutting Stampede3.

---

## 11. Cost-of-this-plan honest assessment

Before committing:

- **Equivalent cost on Stampede3.** Continuing on Stampede3 is **48 GPU-h
  in queue + 48 GPU-h running** vs DSI's **350 GB transfer + 350 GB DSI
  storage + 5–10 × 12 h DSI windows + the admin cost of dual-cluster
  bookkeeping**. DSI wins only if Stampede3 queue waits stretch beyond
  ~3–5 days repeatedly.
- **Risk of dual-source-of-truth.** Two clusters writing checkpoints to
  separate dirs invites confusion. Mitigation: run **only one** training
  campaign at a time per branch/seed; the other cluster takes a
  different branch or seed.
- **Failure modes that kill the plan early:**
  - §2 fix can't land due to PR review delays. Mitigation: edit
    `submit.slurm` in place on a private branch instead.
  - DSI scratch quota too small + no project-storage approval. Plan
    stops at §5.1.
  - DSI's max CUDA cannot host a torch wheel that physicsnemo accepts.
    Plan stops at phase-1 §4.C.
  - Epoch wall on DSI's GPU > 12 h with `--qos=general`. Need either
    larger checkpointing freq, smaller batch, or QoS upgrade.

---

## 12. Reviewer checklist

### v4 addressed v3's review:

- [x] Plan no longer "smoke-only"; explicit phase-2 full training scope.
- [x] Identifies that the v10 full zgplev root does not exist and gives a build path (§1, §3).
- [x] Calls out `submit.slurm:14-19` clobber bug and prescribes a refactor (§2).
- [x] Calls out `build_subset_dataset.py:121` absolute-symlink risk and gives two transfer strategies (§4).
- [x] DSI 12 h survivability addressed via `--qos=general --requeue` + checkpoint resume + multi-window strategy (§7, §8).
- [x] Separate `submit_zgplev_full.dsi.slurm` (§7), not a smoke wrapper.
- [x] GPU class targeting (`--gres=gpu:a100:1` placeholder + bf16 / 40 GB requirement) (§5.2).
- [x] Storage quota explicitly flagged with DSI project-storage escalation path (§5.1).
- [x] Stale comment in `submit_zgplev_full.slurm:36-40` flagged for fix (§3.5).
- [x] No conflation of mail with success criterion (§9 inherits phase 1's stance).
- [x] Realistic budget is days-to-weeks, not hours (§10).

### v4-rev6 addresses the seventh review:

- [x] §7 wrapper now uses `${OUTPUT_ROOT:-…}` / `${EXP_DIR:-…}` so caller-side `sbatch --export=…,EXP_DIR=…` actually wins. Without `:-`, the wrapper's unconditional assignment overwrote the gate's override.
- [x] §8.4 one-epoch gate rewritten: `train_plasim.py` does NOT accept `--max_epochs` (verified — `argument_parser.py` has no such flag). The override goes via a dedicated `submit_zgplev_full.dsi_1epoch.slurm` wrapper that pre-renders the YAML with `max_epochs: 1` and `scheduler_T_max: 1` via sed before invoking `train_plasim.py` directly (not via the original Stampede3 script).
- [x] §3 commands replaced `cd $HOME/AI-RES` with `cd "$REPO"` everywhere it applies to Stampede3 packaging — paired with a shell-session `REPO=…/AI-RES-dsi-bootstrap` variable defined in the §3 prologue. (§6.4 DSI subset rebuild keeps `$HOME/AI-RES` because that's correct on DSI.)
- [x] §8.1 wording fixed: "prove from a smoke" → "prove from the §8.4 one-epoch full-data DSI gate" (was contradicting §8.4).

### v4-rev5 addressed the sixth review:

- [x] §3 prologue note: §3 commands MUST run from a tree where the §2.2 `submit.slurm` env-default refactor is committed (preferred: bootstrap worktree). Otherwise §3.2 `--export=ALL,POSTPROC_ROOT=…` is silently overwritten by the unfixed `POSTPROC_ROOT=` line at `submit.slurm:14`.
- [x] §8.1 checkpoint recovery rewritten: resume-detect at `train_plasim.py:172` hardcodes `checkpoint_version=0`, so just `mv`-ing a corrupt `_v0` aside makes resume fail. Recovery procedure now copies the next-newest valid `_v*.tar` over `_v0` to keep the rotation slot intact. Long-term fix flagged for code review.
- [x] §8.4 NEW: one-epoch DSI gate using full dataset + full config + bf16 + checkpointing_level=2 must pass with epoch wall ≤ 8 h **before** multi-window full launch. Smoke is not a substitute for a real-data epoch wall measurement.
- [x] §8.5 makes the `--requeue` semantics explicit: it only helps after epoch checkpoints are written, and **does not** auto-resubmit on a 12 h wall-time timeout (only on preempt). Manual chained resubmission required.
- [x] §8 + §8.4 add explicit `sbatch --export=ALL,DSI_PROJECT="$DSI_PROJECT" …` form so DSI batch env actually carries the project path through.
- [x] §6.5 adds Stampede3-absolute-symlink defensive check (catches wrong-artifact-transferred case where `-xtype l` would not detect because the symlink isn't broken on Stampede3).
- [x] `checkpoint_num_versions` default corrected from 2 → 3 per `driver.py:140`.
- [x] Smoke plan §4.C: removed duplicate `# 8. Sanity` block (left only the `# 9. Verify imports` Python block); added `torch_harmonics` build-isolation fallback (`pip install ninja packaging setuptools` + `--no-build-isolation`).

### v4-rev4 addressed the fifth review:

- [x] §4.A wording corrected from "DSI scratch" to `$DSI_PROJECT` so all data references are consistent post-§5.1.
- [x] §8.1 documents (a) epoch-end-only checkpoint cadence per `deterministic_trainer.py:388`, gating launch on a measured epoch wall well under 12 h, and (b) `torch.save` non-atomicity per `driver.py:570`, with explicit recovery procedure (mv `.corrupt`, fall back to next-newest version).
- [x] §6.5 adds `find "$ROOT/test" -name '*.h5' | wc -l   # 0` and a pre-`sbatch` `mkdir -p $HOME/AI-RES/logs` (SLURM opens `-o`/`-e` paths at submission time, before script body runs).
- [x] §6.5 explicitly applies to both Path A and Path B.
- [x] §7 wrapper drops the `/project/<group>/$USER` placeholder default; now hard-fails on unset/non-dir/non-writable `DSI_PROJECT` via `:?` + `test -d` + `test -w`.
- [x] Smoke plan §4.C step 7 now installs the makani runtime deps explicitly under `--no-deps`: h5py, wandb, tqdm, more-itertools, Pillow, ruamel.yaml, PyYAML, pytest. Plus an import sanity Python block.
- [x] §0 worktree workflow adds an explicit risk note + WIP snapshot recording (`git status --short > /tmp/...`) and labels the commit "WIP-SNAPSHOT: …" so it isn't mistaken for curated work.

### v4-rev3 addressed the fourth review:

- [x] §6 + §6.4 paths corrected from `$DSI_SCRATCH` to `$DSI_PROJECT` (removes the contradiction with §5/§7); §6.1 adds quota gate as pre-transfer check; new §6.5 final pre-submit verification with file counts + `-xtype l` broken-symlink check.
- [x] §7 wrapper drops `DSI_SCRATCH` variable and the `SCRATCH="$DSI_PROJECT"` alias (the original script's `$SCRATCH`-based defaults are unused once `OUTPUT_ROOT`/`EXP_DIR` are exported, so aliasing only obscures mistakes).
- [x] §0 worktree procedure rewritten: bootstrap commit MUST happen first, then `git worktree add` from the bootstrap branch. Without this, the worktree is missing all 40+ untracked files (entire `src/sfno_training/`).
- [x] §3.6 Path II metadata uses `--test-years 0 0` for a subset with empty `test/`, instead of misleading `11 11`.
- [x] §3.4 + timing table: stats over 98 train H5s with whole-array reads is 1–3 h on a compute node, not 30 min.
- [x] Timing label "one-year packager smoke" → "three-split-year packager smoke".

### v4-rev2 addressed the third review:

- [x] Removes `#SBATCH --signal=B:USR1@300` from §7's directives — until trainer-side mitigation B lands, the signal kills Python instead of saving (§7 + §8.3).
- [x] Path II stats now correctly accounts for `stats.py:128` only scanning `train/` and `build_subset_dataset.py:169` symlinking `stats/`. Procedure: build subset first, replace its stats symlink with a real dir, then run stats on subset (§3.6).
- [x] One-year smoke uses distinct years per split (train 11, valid 101, test 121) so they don't all collapse into `train/` via `resolve_split` (§3.1).
- [x] `stats.py` invocations now run on compute nodes via `srun -p skx --pty`, not login nodes — `stats.py:159` reads whole H5 arrays per file (§3.1, §3.4).
- [x] Verification reads `$EXP_DIR/.../out.log` (per `train_plasim.py:197`), not Slurm `.err`. Latest checkpoint via `ls -t ckpt_mp0_v*.tar | head -1`, not pinned to `_v0` (versions rotate per `deterministic_trainer.py:395`) (§9).
- [x] Storage placement reflects DSI defaults: 50 GB scratch / 500 GB project. Both data and `EXP_DIR` go to project; quota increase is a HARD GATE before §6 (§5.1, §7).
- [x] Names new files explicitly as "NOT yet created" in §0; no claim that the plan is executable as written.
- [x] Recommends `git worktree` for bootstrap branch instead of mutating dirty `zgplev-migration` checkout (§0).
- [x] Flags `scan_for_nans.py` log-name mismatch in §7 (wrapper writes `_dsi_<jid>` but original looks for `<jid>` without suffix).

### v4 (initial) addressed the second review:

- [x] Adds the missing `stats.py` + `metadata.py` stages — `packager.py` alone produces only `train/valid/test/`; `build_subset_dataset.py:163` requires `stats/`, `metadata/`, `config/` (§3.1, §3.4).
- [x] Adds `export PYTHONPATH=$PWD/src` before every `python3 -m plasim_makani_packager.*` invocation (§3.1, §3.2, §3.4).
- [x] Fixes `validate.py` arg: `--output-root` (not `--root`), and uses `--mode full` (§3.1, §3.4).
- [x] Removes the `trap … USR1; exec bash` anti-pattern from §7 — `exec` discards traps. Wrapper now uses `bash` (without `exec`) and explicitly notes that SIGUSR1 forwarding is not real until trainer-side mitigation B (§7, §8.3).
- [x] Acknowledges that today's SIGUSR1 default is *worse* than no signal (Python terminates immediately), so first submission drops `--signal` and relies on end-of-epoch checkpointing (§8.3 mitigation A).
- [x] Fixes checkpoint filename: `ckpt_mp0_v0.tar` (per `train_plasim.py:163-164`), not `<latest>.pt`. Verification command updated to `torch.load` the right path (§9).
- [x] Splits storage placement: data on scratch (recreatable), `EXP_DIR` on project (durable, backed up) — DSI scratch is purge-prone per cluster docs (§5.1, §7).
- [x] Adds a one-year packager smoke before the full 126-task array (§3.1).
- [x] Fixes year-count arithmetic: 126 packaged years (3–128 minus warmup 1–2), not 128 (§1.3, §3.2).
- [x] Adds explicit science gate on stats normalization year range (§3.6) — flagged for modeling-lead approval before launch.
