# Migrate AI-RES → SFNO_Climate_Emulator

## Context

The directory name `AI-RES` no longer matches what the project actually is. The work here is **PlaSim → SFNO emulator R&D**: building, training, and evaluating Spherical Fourier Neural Operator emulators of the PlaSim simple-climate model, with two parallel tracks (own-track v10 zgplev, group SFNO-5410). The user wants the on-disk layout to reflect that, and to do a complete rename — working tree, `$SCRATCH`, `$WORK2`, GitHub repo — so future readers (and future Claude sessions) don't get confused by the misleading name.

The move is full-scope: working tree from `~/AI-RES` to `~/projects/SFNO_Climate_Emulator`, plus rename of `$SCRATCH/AI-RES → $SCRATCH/SFNO_Climate_Emulator` and `$WORK2/.../stampede3/AI-RES → $WORK2/.../stampede3/SFNO_Climate_Emulator`. Lustre rename of these large dirs (1.9 TB + 220 GB) is metadata-only — instant, no data copy. The expensive piece is rewriting **241 hits of `$SCRATCH/AI-RES`** across 47 files, the venv fix-up, and verifying nothing breaks.

User confirmed: no SLURM jobs running, surgical venv fix-up (not rebuild), DSI references untouched, transitional symlinks for ~2 weeks, Claude memory bridged via symlink. GitHub repo renamed to `SFNO_Climate_Emulator-Stampede3`.

## Out of scope

- Renaming on the DSI cluster (`$DSI_PROJECT/AI-RES` strings stay; that's a parallel filesystem migration the user will do later).
- Rebuilding the venv from scratch.
- Touching anything in `external/` or `makani-src/` beyond what `git ls-files`-driven sed catches (those are vendored upstream code).

## Pre-flight (do before any rename)

1. Confirm `squeue -u $USER` is empty (re-verify; was empty at planning time).
2. Confirm no Python process holds files: `lsof +D /work2/11114/zhixingliu/stampede3/AI-RES/.venv 2>/dev/null | head`.
3. **Snapshot the untracked-files inventory; do NOT require commit and do NOT stash.** Reality: there are ~100 untracked text files in the worktree containing `AI-RES` (scripts/, src/, src/sfno_training/, tests/, skills/, .claude/skills/). Requiring all of them to be committed before migration is unrealistic. Stashing is unsafe (it removes the working-tree copies before Phase 5 pass 2's content-driven sed can find them). Phase 5 pass 2 already rewrites every untracked text file in place via mime/grep discovery; tracking state is orthogonal.

   Procedure:
   - Snapshot for awareness so Phase 11's git add is deliberate:
     ```bash
     git ls-files --others --exclude-standard > /tmp/migration_untracked_pre.txt
     # Subset that Phase 5 pass 2 WILL sed (active code, not historical artifacts):
     git ls-files --others --exclude-standard \
       | grep -v -E '^(analysis_outputs/|docs/2026-|docs/codex_reviews/|docs/hpo_distill/|docs/run_log/|external/|makani-src/)' \
       | xargs grep -l 'AI-RES' 2>/dev/null \
       > /tmp/migration_untracked_active_with_airrs.txt || true
     echo "Untracked active files Phase 5 will rewrite: $(wc -l < /tmp/migration_untracked_active_with_airrs.txt)"
     ```
   - **Eyeball the list**. If any untracked file is local scratch/experiment that should keep its old paths as-is, either `.gitignore` it or move it outside the tree now. Otherwise the migration treats it as in-scope and rewrites it in place (it stays untracked).
   - Exempted dirs (Phase 5 leaves untouched): `analysis_outputs/`, `docs/2026-…`, `docs/codex_reviews/`, `docs/hpo_distill/`, `docs/run_log/`, `external/`, `makani-src/`.
4. Capture the pre-state for rollback:
   - `git rev-parse HEAD > /tmp/migration_pre_head.txt`
   - `ls -la ~/AI-RES > /tmp/migration_pre_ls.txt`
   - `find ~/AI-RES -type l > /tmp/migration_pre_symlinks.txt`
5. **Verify required CLI tools are available** (no point starting if these aren't):
   ```bash
   command -v gh   >/dev/null || { echo "ABORT: gh not on PATH — Phase 7 needs it"; exit 1; }
   command -v git  >/dev/null || { echo "ABORT: git missing"; exit 1; }
   command -v file >/dev/null || { echo "ABORT: file(1) missing — Phase 5 untracked-text filter needs it"; exit 1; }
   gh auth status >/dev/null 2>&1 || { echo "ABORT: gh not authenticated; run 'gh auth login'"; exit 1; }
   ```
6. **Snapshot the baseline eval outputs for the v11 run** so Phase 10b has something to diff against. Two directories match the v11 pattern; select by required-artifact presence (see Phase 10b for full reasoning):
   ```bash
   match_complete() {
     for d in /work2/11114/zhixingliu/stampede3/AI-RES/results/sfno_eval/*v11_gb32_lr8e4_minlr1e5_noise0p020_epochs75*; do
       [ -f "$d/scores/nwp_scorecard_summary.csv" ] && [ -f "$d/report.md" ] && echo "$d" && return
     done
   }
   OLD_EVAL_DIR=$(match_complete)
   test -n "$OLD_EVAL_DIR" || { echo "ABORT: no complete v11 baseline (need scores/nwp_scorecard_summary.csv + report.md)"; exit 1; }
   BASELINE_COPY=/tmp/baseline_eval_$(basename "$OLD_EVAL_DIR")
   cp -a "$OLD_EVAL_DIR" "$BASELINE_COPY"
   OLD_RUN_DIR=$(find /scratch/11114/zhixingliu/AI-RES/runs -maxdepth 4 -type d -name '0' -path '*v11_gb32_lr8e4_minlr1e5_noise0p020_epochs75*' | head -1)
   test -d "$OLD_RUN_DIR" || { echo "ABORT: run dir not found"; exit 1; }
   {
     echo "OLD_EVAL_DIR=$OLD_EVAL_DIR"
     echo "OLD_RUN_DIR=$OLD_RUN_DIR"
     echo "BASELINE_COPY=$BASELINE_COPY"
   } > /tmp/migration_baseline.txt
   ```
   If either is absent, stop and surface — the verification target must exist before we migrate.

## Phase 1 — Rename external Lustre dirs (metadata-only, ~instant)

Guard each `mv` against an existing destination — without the guard, `mv` would nest the source inside the existing destination dir (e.g. `mv A B` when `B/` exists creates `B/A/`).

```bash
test ! -e /scratch/11114/zhixingliu/SFNO_Climate_Emulator || { echo "ABORT: destination exists"; exit 1; }
mv /scratch/11114/zhixingliu/AI-RES /scratch/11114/zhixingliu/SFNO_Climate_Emulator

test ! -e /work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator || { echo "ABORT: destination exists"; exit 1; }
mv /work2/11114/zhixingliu/stampede3/AI-RES /work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator
```

**Verify:** `ls -d /scratch/11114/zhixingliu/SFNO_Climate_Emulator` and `ls -d /work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator` both succeed.

Symlinks inside `~/AI-RES` now dangle — expected; Phase 3 repairs.

## Phase 2 — Move the working tree

```bash
mkdir -p /home1/11114/zhixingliu/projects
test ! -e /home1/11114/zhixingliu/projects/SFNO_Climate_Emulator || { echo "ABORT: destination exists"; exit 1; }
mv /home1/11114/zhixingliu/AI-RES /home1/11114/zhixingliu/projects/SFNO_Climate_Emulator
```

**Verify:** `ls /home1/11114/zhixingliu/projects/SFNO_Climate_Emulator/.git` resolves.

From this point use the new path. Subsequent commands assume `cd ~/projects/SFNO_Climate_Emulator`.

## Phase 3 — Repoint the 4 working-tree symlinks

```bash
cd ~/projects/SFNO_Climate_Emulator
ln -sfn /scratch/11114/zhixingliu/SFNO_Climate_Emulator/checkpoints checkpoints
ln -sfn /scratch/11114/zhixingliu/SFNO_Climate_Emulator/data         data
ln -sfn /work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results results
ln -sfn /work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/.venv   .venv
```

**Verify:** `ls -L data/ && ls -L results/ && ls -L checkpoints/ && ls -L .venv/bin/python` all succeed.

## Phase 4 — Surgical fix-up of `.venv`

The venv now physically lives at `/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/.venv`, but every internal absolute path inside still says `.../AI-RES/.venv/...`. Fix:

```bash
VENV=/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/.venv
OLD_W2=/work2/11114/zhixingliu/stampede3/AI-RES
NEW_W2=/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator
OLD_SC=/scratch/11114/zhixingliu/AI-RES
NEW_SC=/scratch/11114/zhixingliu/SFNO_Climate_Emulator
OLD_HM=/home1/11114/zhixingliu/AI-RES
NEW_HM=/home1/11114/zhixingliu/projects/SFNO_Climate_Emulator

# 1. Shebangs in bin/ (pip, python, pytest, ipython, ...)
find "$VENV/bin" -type f -exec sed -i \
  -e "s|$OLD_W2|$NEW_W2|g" \
  -e "s|$OLD_SC|$NEW_SC|g" \
  -e "s|$OLD_HM|$NEW_HM|g" {} +

# 2. pyvenv.cfg
sed -i \
  -e "s|$OLD_W2|$NEW_W2|g" \
  -e "s|$OLD_SC|$NEW_SC|g" \
  -e "s|$OLD_HM|$NEW_HM|g" "$VENV/pyvenv.cfg"

# 3. .pth and .egg-link files (editable installs: makani-src, possibly earth2studio)
find "$VENV/lib" \( -name "*.pth" -o -name "*.egg-link" \) -exec sed -i \
  -e "s|$OLD_W2|$NEW_W2|g" \
  -e "s|$OLD_SC|$NEW_SC|g" \
  -e "s|$OLD_HM|$NEW_HM|g" {} +
```

Then re-register editable installs to be safe (rewrites metadata atomically):

```bash
source "$VENV/bin/activate"
pip install --no-deps -e /home1/11114/zhixingliu/projects/SFNO_Climate_Emulator/makani-src
# Earth2Studio is editable too (has setup.py and pyproject.toml). Use grouped if/then —
# the chained `[ ] || [ ] && pip` form has wrong precedence: it would skip pip when the first test passes.
if [ -f external/earth2studio/setup.py ] || [ -f external/earth2studio/pyproject.toml ]; then
  pip install --no-deps -e /home1/11114/zhixingliu/projects/SFNO_Climate_Emulator/external/earth2studio
fi
```

**Verify:**
```bash
python -c "import sys; print(sys.executable)"   # must show new path
python -c "import torch; print('torch', torch.__version__)"
python -c "import makani; print('makani', makani.__file__)"  # must point under new SFNO_Climate_Emulator
pip show makani | grep -i location
```

## Phase 4b — Surgical fix-up of the group conda env

The group training stack uses a separate conda env (not the `.venv`) activated by absolute prefix in `src/sfno_training_group/env_activate.sh:13`. After Phase 1's `mv $WORK2/AI-RES $WORK2/SFNO_Climate_Emulator`, the env lives at `$WORK2/.../SFNO_Climate_Emulator/envs/group_pangu_sfno_v2/` but its internal scripts still embed old absolute paths (verified: `bin/pip:1` → `#!/work2/.../AI-RES/envs/.../bin/python3.11`). Without this fix, group-track activation works through the transitional symlink only — same risk as Phase 4 but for a different env.

```bash
# Reuses OLD_W2/NEW_W2/OLD_SC/NEW_SC/OLD_HM/NEW_HM from Phase 4. If running 4b in a
# fresh shell, redefine them here first (same values as Phase 4).
GENV=/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/envs/group_pangu_sfno_v2
test -d "$GENV" || { echo "ABORT: group conda env not found at $GENV"; exit 1; }

# Build a reusable sed expression once.
export GENV_SED='
  s|'"$OLD_W2"'|'"$NEW_W2"'|g;
  s|'"$OLD_SC"'|'"$NEW_SC"'|g;
  s|'"$OLD_HM"'|'"$NEW_HM"'|g
'

# 1. Shebangs in bin/ (text scripts only; skip ELF binaries).
find "$GENV/bin" -type f -print0 | xargs -0 -I {} sh -c '
  f="$2"
  mime=$(file -b --mime-type "$f")
  case "$mime" in text/*|application/x-shellscript) sed -i -e "$1" "$f" ;; esac
' _ "$GENV_SED" {}

# 2. conda-meta JSON + activate.d / deactivate.d shell scripts + conda history.
find "$GENV/conda-meta" "$GENV/etc/conda" -type f \
  \( -name '*.json' -o -name '*.sh' -o -name '*.csh' -o -name '*.fish' \) \
  -print0 | xargs -0 sed -i -e "$GENV_SED"
# conda-meta/history is a plain text log without a `.json`/`.sh` extension — sed it directly.
[ -f "$GENV/conda-meta/history" ] && sed -i -e "$GENV_SED" "$GENV/conda-meta/history"

# 3. .pth files (for any editable installs into the conda env).
find "$GENV/lib" -name "*.pth" -print0 | xargs -0 sed -i -e "$GENV_SED"

# 4. pkg-config / tcl-config / sysconfig — load-bearing for C-extension builds and Python introspection.
find "$GENV/lib/pkgconfig" -name '*.pc' -print0 2>/dev/null | xargs -0 sed -i -e "$GENV_SED"
find "$GENV/lib" -maxdepth 2 -name '*Config.sh' -print0 2>/dev/null | xargs -0 sed -i -e "$GENV_SED"
find "$GENV/lib" -name '_sysconfigdata*.py' -print0 2>/dev/null | xargs -0 sed -i -e "$GENV_SED"
```

**Verify (activate the new env prefix DIRECTLY — do not source `env_activate.sh` yet; it still points at the old `AI-RES` path until Phase 5 rewrites it):**
```bash
# 1. Recursive grep — must find ZERO old-path strings in text files anywhere in the env.
remaining=$(grep -rl '/AI-RES/' "$GENV" 2>/dev/null --include='*.py' --include='*.sh' --include='*.json' --include='*.pc' --include='*.csh' --include='*.fish' --include='*.cfg' --include='history' | wc -l)
[ "$remaining" -eq 0 ] || { echo "STALE: $remaining files in $GENV still contain /AI-RES/"; grep -rl '/AI-RES/' "$GENV" --include='*.py' --include='*.sh' --include='*.json' --include='*.pc' --include='*.csh' --include='*.fish' --include='*.cfg' --include='history' | head; exit 1; }

# 2. Activate the env directly by absolute prefix — bypass the still-stale env_activate.sh.
source /work2/11114/zhixingliu/stampede3/miniforge3/etc/profile.d/conda.sh
conda activate "$GENV"
python -c "import sys; print(sys.executable)"               # must point under SFNO_Climate_Emulator
which pip                                                    # must point under SFNO_Climate_Emulator
python -c "import sysconfig; [print(k,v) for k,v in sysconfig.get_paths().items()]"   # all paths under new prefix
conda deactivate
```
A later end-to-end check (after Phase 5 rewrites `src/sfno_training_group/env_activate.sh`) lives in Phase 10a — that's where the wrapper script is re-tested.

If the conda env still references old paths after the surgical fix, fall back to rebuilding via the env spec (out of scope for this plan; flag and stop).

## Phase 4c — Surgical fix-up of the packed Derecho env (5410 production)

The 5410 production eval (`scripts/submit_eval_inference_5410_packed.slurm:34`) uses a packed conda env at `$WORK2/.../SFNO_Climate_Emulator/artifacts/derecho_runtime/aires_env_20260509/unpacked/`. After Phase 1's `mv`, the env's internal files still embed the old absolute prefix in 601 locations (verified) — most critically in `bin/gdal-config:4` (`CONFIG_PREFIX="…/AI-RES/…"`), pkg-config files, and CMake configs. Without this fix, any C-extension compile or pkg-config lookup against the packed env reads stale paths.

```bash
PENV=/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_runtime/aires_env_20260509/unpacked
test -d "$PENV" || { echo "ABORT: packed Derecho env not found at $PENV"; exit 1; }

# CONTENT-DRIVEN discovery (same termination pattern as Phase 5.5). The packed env has
# stale paths spread across ~700 files in many locations — include/ (H5pubconf.h),
# share/pkgconfig/, lib/libhdf5.settings, lib/esmf.mk, lib/preload.sh, share/doc/, libexec/,
# etc. Enumerating extensions falls behind. grep -rIl finds every text file with the pattern
# regardless of extension and auto-skips binaries.

# Same sed expression as Phase 4b — only the W2 prefix changes for this env.
export PENV_SED='s|'"$OLD_W2"'|'"$NEW_W2"'|g'

# 1. Discover the full mutation set.
grep -rIl '/AI-RES/' "$PENV" > /tmp/penv_mutation_targets.txt
echo "Files in packed env containing /AI-RES/: $(wc -l < /tmp/penv_mutation_targets.txt)"

# 2. Sed-rewrite each one. -I in grep already filtered binaries, so no further mime gate needed.
xargs -d '\n' sed -i -e "$PENV_SED" < /tmp/penv_mutation_targets.txt
```

**Verify:**
```bash
# Recursive grep — must find ZERO old-path text remaining.
remaining=$(grep -rIl '/AI-RES/' "$PENV" 2>/dev/null | wc -l)
[ "$remaining" -eq 0 ] || { echo "STALE: $remaining files in $PENV still contain /AI-RES/"; grep -rIl '/AI-RES/' "$PENV" | head; exit 1; }

# Sanity: gdal-config reports the new prefix.
"$PENV/bin/gdal-config" --prefix | grep -q SFNO_Climate_Emulator || { echo "ABORT: gdal-config still reports old prefix"; exit 1; }
```

## Phase 5 — Bulk path rewrite in working tree (tracked + active-untracked)

Two passes: once over tracked files with DSI pathspec exclusions, once over active-untracked text files (caught via `git ls-files --others --exclude-standard`). DSI scripts that run on the DSI cluster genuinely need `$HOME/AI-RES` because `$HOME` is the DSI home on that machine; rewriting them would break DSI execution.

From `~/projects/SFNO_Climate_Emulator`:

```bash
# Common sed expression — EXPORTED so it survives into the `sh -c` subshell in pass 2.
export SED_EXPR='
  s|/home1/11114/zhixingliu/AI-RES|/home1/11114/zhixingliu/projects/SFNO_Climate_Emulator|g;
  s|/scratch/11114/zhixingliu/AI-RES|/scratch/11114/zhixingliu/SFNO_Climate_Emulator|g;
  s|/work2/11114/zhixingliu/stampede3/AI-RES|/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator|g;
  s|\$SCRATCH/AI-RES|$SCRATCH/SFNO_Climate_Emulator|g;
  s|\${SCRATCH}/AI-RES|${SCRATCH}/SFNO_Climate_Emulator|g;
  s|\$WORK2/AI-RES|$WORK2/SFNO_Climate_Emulator|g;
  s|\${WORK2}/AI-RES|${WORK2}/SFNO_Climate_Emulator|g;
  s|\$WORK/AI-RES|$WORK/SFNO_Climate_Emulator|g;
  s|\${WORK}/AI-RES|${WORK}/SFNO_Climate_Emulator|g;
  s|\$HOME/AI-RES|$HOME/projects/SFNO_Climate_Emulator|g;
  s|\${HOME}/AI-RES|${HOME}/projects/SFNO_Climate_Emulator|g;
  s|~/AI-RES|~/projects/SFNO_Climate_Emulator|g;
  s|"AI-RES"|"SFNO_Climate_Emulator"|g;
  s|'"'"'AI-RES'"'"'|'"'"'SFNO_Climate_Emulator'"'"'|g
'
# Note on the two quote-bounded patterns at the end: Python path-construction idiom
# `Path(...) / "AI-RES" / "runs"` (e.g. scripts/hpo_prune.py:40) embeds the project name
# as a quoted string component. The sed substitution above catches these. The patterns
# are quote-bounded to avoid false positives on unquoted occurrences (e.g. identifiers,
# free-form prose) that should be handled by Phase 6 manual edits instead.

# Pass 1 — tracked files, EXCLUDING DSI scripts/docs, codex review artifacts, and the migration plan itself.
git ls-files -z \
  -- ':!src/sfno_training/*.dsi*.slurm' \
     ':!docs/dsi_*.md' \
     ':!docs/codex_reviews/**' \
     ':!docs/2026-05-23_sfno_climate_emulator_migration_plan.md' \
  | xargs -0 sed -i -e "$SED_EXPR"

# Pass 2 — active untracked text files (NOT gitignored, NOT binary), same exclusions plus vendored + historical dirs.
# Pass SED_EXPR explicitly as a positional arg ($1) to be robust whether or not it was exported.
git ls-files --others --exclude-standard -z \
  -- ':!src/sfno_training/*.dsi*.slurm' \
     ':!docs/dsi_*.md' \
     ':!docs/codex_reviews/**' \
     ':!docs/2026-05-23_sfno_climate_emulator_migration_plan.md' \
     ':!docs/hpo_distill/**' \
     ':!docs/run_log/**' \
     ':!analysis_outputs/**' \
     ':!external/**' \
     ':!makani-src/**' \
  | xargs -0 -I {} sh -c '
      expr="$1"; f="$2"
      mime=$(file -b --mime-type "$f")
      case "$mime" in text/*|application/json|application/x-shellscript|application/x-yaml) : ;; *) exit 0 ;; esac
      sed -i -e "$expr" "$f"
    ' _ "$SED_EXPR" {}
```

**Intentionally not touched:**
- `src/sfno_training/*.dsi*.slurm` (3 files) — `$HOME/AI-RES` is correct on DSI; pathspec excludes them.
- `docs/dsi_*.md` (`dsi_full_training_plan.md`, `dsi_smoke_backup_plan.md`) — same reason.
- `$DSI_PROJECT/AI-RES` strings — sed patterns don't match `$DSI_PROJECT/AI-RES`.
- `AI-RES-dsi-bootstrap` — worktree name; not matched.
- `feynmanliu214/AI-RES-Stampede3` — GitHub URL, handled in Phase 7.
- Bare `AI-RES` tokens in prose (README title, doc headings) — handled by `Edit` in Phase 6.
- `external/**` and `makani-src/**` — vendored upstream code.

**Verify diff makes sense:**
```bash
git diff --stat                                                          # expect ~50 tracked files changed
git diff --stat -- 'src/sfno_training/*.dsi*.slurm' 'docs/dsi_*.md' \
                   'docs/codex_reviews/**' 'docs/2026-05-23_sfno_climate_emulator_migration_plan.md'   # must be EMPTY (excluded by design)
# Untracked-pass sanity: confirm EVERY file in the pre-flight inventory is clean. These files
# stay untracked (Phase 11 decides which to add); the pass just rewrote their content in place.
while read -r f; do
  [ -f "$f" ] && grep -q 'AI-RES' "$f" 2>/dev/null && echo "STALE: $f still contains AI-RES"
done < /tmp/migration_untracked_active_with_airrs.txt > /tmp/migration_untracked_post_grep.txt
[ -s /tmp/migration_untracked_post_grep.txt ] && {
  echo "STALE untracked files (Phase 5 pass 2 missed):"
  cat /tmp/migration_untracked_post_grep.txt
  exit 1
}
# Now nothing in tracked code should contain old paths (DSI scope + audit artifacts + this plan are the only legitimate carriers):
remaining=$(git grep -l '/AI-RES\|\$SCRATCH/AI-RES\|\$WORK2/AI-RES' \
  -- ':!docs/codex_reviews' \
     ':!src/sfno_training/*.dsi*.slurm' \
     ':!docs/dsi_*.md' \
     ':!docs/2026-05-23_sfno_climate_emulator_migration_plan.md' \
  | wc -l)
[ "$remaining" -eq 0 ] || { echo "STALE: $remaining tracked files still have old paths"; \
  git grep -l '/AI-RES\|\$SCRATCH/AI-RES\|\$WORK2/AI-RES' \
    -- ':!docs/codex_reviews' ':!src/sfno_training/*.dsi*.slurm' ':!docs/dsi_*.md' ':!docs/2026-05-23_sfno_climate_emulator_migration_plan.md'; \
  exit 1; }
```

## Phase 5.5 — Rewrite generated artifacts under run dirs

User decision (Codex round 1): complete the migration by sed-rewriting `config.json` and other text artifacts inside `$SCRATCH/SFNO_Climate_Emulator/runs/` and `$WORK2/.../SFNO_Climate_Emulator/results/`. This makes old runs re-evaluable without depending on the transitional symlink, so the symlink can be removed at 2 weeks as planned. Per `submit_eval_prelude.sh:39`-style code, eval reads `RUN_DIR/config.json:train_data_path` to derive datasets — leaving stale paths there would break post-symlink-removal evals against historical runs.

```bash
NEW_SC=/scratch/11114/zhixingliu/SFNO_Climate_Emulator
NEW_W2=/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator

# Mutation set discovered by content (not by extension) — `grep -rl '/AI-RES'` finds
# every text file under runs/results that actually contains an old path. This terminates
# the round-by-round "did you remember .json? .txt? .md?" expansion: the find target IS
# every file that needs the rewrite, regardless of extension.

BACKUP=/tmp/migration_run_artifact_backup_$(date +%Y%m%d_%H%M%S)
mkdir -p "$BACKUP"

# 1. Discover the mutation set. grep -I auto-skips binary files; -r is recursive.
#    Match all three concrete old-path forms (absolute scratch, absolute work2, absolute home).
grep -rIl '/AI-RES/\|\$SCRATCH/AI-RES\|\$WORK2/AI-RES\|\$HOME/AI-RES' \
  "$NEW_SC/runs" "$NEW_W2/results" \
  > "$BACKUP/text_targets.txt" 2>/dev/null
echo "Text files to mutate: $(wc -l < "$BACKUP/text_targets.txt")"

# 2. Back up exactly those files into a tar (preserves paths inside NEW_SC / NEW_W2 roots).
tar -czf "$BACKUP/run_artifact_text.tar.gz" -T "$BACKUP/text_targets.txt"

# 3. Manifest all stale symlinks BEFORE mutation, so rollback can restore them too.
find "$NEW_SC/runs" "$NEW_W2/results" -type l -lname '*AI-RES*' \
  -printf '%p\t%l\n' > "$BACKUP/symlink_manifest.tsv"
echo "Stale symlinks to retarget: $(wc -l < "$BACKUP/symlink_manifest.tsv")"

echo "BACKUP=$BACKUP" > /tmp/migration_phase55_backup.txt

# 4. Sed-rewrite every discovered text file (same SED_EXPR as Phase 5).
xargs -d '\n' sed -i -e "$SED_EXPR" < "$BACKUP/text_targets.txt"

# 5. Retarget stale symlinks. Eval/training scripts create symlinks like
#    `training_checkpoints` and `inference/ic_nc/*.nc` pointing at /AI-RES/...;
#    without retargeting they break after Phase 8's transitional symlink expires.
awk -F'\t' '{ new=$2; gsub(/\/AI-RES\//, "/SFNO_Climate_Emulator/", new); print $1 "\t" new }' \
  "$BACKUP/symlink_manifest.tsv" \
  | while IFS=$'\t' read -r link new; do ln -sfn "$new" "$link"; done

# 6. Verify both rewrites clean.
stale_text=$(grep -rIl '/AI-RES/' "$NEW_SC/runs" "$NEW_W2/results" 2>/dev/null | wc -l)
stale_link=$(find "$NEW_SC/runs" "$NEW_W2/results" -type l -lname '*AI-RES*' | wc -l)
echo "Post-rewrite: $stale_text text files still containing /AI-RES/, $stale_link symlinks still targeting AI-RES"
[ "$stale_text" -eq 0 ] && [ "$stale_link" -eq 0 ] || { echo "STALE remains — investigate"; exit 1; }
```

**Intentionally not touched under run dirs:**
- `*.h5`, `*.tar`, `*.nc`, `*.npy`, `*.pt`, `*.png`, `*.pdf`, `*.csv` — binary/generated data; sed would corrupt or be inert.
- `training_checkpoints/*.tar` — model state, never touch.

## Phase 6 — Targeted edits for narrative references and DSI-doc Stampede3 paths

Use `Edit` (not sed) for these — each has nearby context that should change coherently:

- `README.md:1` — `# AI-RES` → `# SFNO Climate Emulator`, and the project description paragraph below.
- `.claude/skills/eval-sfno-own/SKILL.md` — title line + any narrative `AI-RES` references (paths already covered by Phase 5).
- `.claude/skills/eval-sfno-5410/SKILL.md` — same.
- `skills/*/SKILL.md` (4 files: `plasim-makani-packager`, `plasim-postprocess`, `sfno-training`, `train-sfno-hpo`) — title lines, narrative.

**DSI docs (`docs/dsi_full_training_plan.md`, `docs/dsi_smoke_backup_plan.md`) need a manual audit** because Phase 5 deliberately excluded them — these docs encode BOTH legitimate DSI paths to preserve AND Stampede3 paths that must migrate. Specifically:

| Pattern in DSI doc | Action |
|---|---|
| `$HOME/AI-RES` | **PRESERVE** — on DSI, `$HOME` is the DSI home; this is correct as-is. |
| `$DSI_PROJECT/AI-RES` | **PRESERVE** — DSI filesystem layout; user keeps DSI naming. |
| `$DSI_SCRATCH/AI-RES` | **PRESERVE** — DSI scratch path; same reasoning. |
| `AI-RES-dsi-bootstrap` (worktree name) | **PRESERVE** — DSI worktree identifier. |
| `/scratch/11114/zhixingliu/AI-RES` (Stampede3) | **REWRITE** to `/scratch/11114/zhixingliu/SFNO_Climate_Emulator`. |
| `/work2/11114/zhixingliu/stampede3/AI-RES` | **REWRITE** to `/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator`. |
| `$SCRATCH/AI-RES`, `$WORK2/AI-RES` | **REWRITE** to `$SCRATCH/SFNO_Climate_Emulator`, `$WORK2/SFNO_Climate_Emulator`. |
| `feynmanliu214/AI-RES-Stampede3` (GitHub URL) | **REWRITE** to `feynmanliu214/SFNO_Climate_Emulator-Stampede3` (matches Phase 7). Examples: `dsi_smoke_backup_plan.md:351`. |

Procedure: for each DSI doc, list line-by-line every `AI-RES` occurrence, classify per the table, and Edit only the ones in the REWRITE rows. Cross-check by grepping after:

```bash
# Positive-match audit: list every line in the DSI docs that hits a REWRITE pattern.
# These lines MUST be hand-edited; nothing else in the DSI docs should change.
# Includes Stampede3 absolute home paths (/home1/...) — DSI's $HOME is different but
# Stampede3-absolute home paths inside DSI docs still need to migrate.
#
# Each `/AI-RES` is bounded with `(/|[[:space:]]|$|`|\")` so we do NOT match the
# preserved worktree name `AI-RES-dsi-bootstrap` (which has `-d` after AI-RES, not a path
# separator or word boundary). Same logic for `AI-RES-Stampede3` (GitHub URL) — handled
# by an explicit pattern so the regex stays unambiguous.
RW='(/home1/[0-9]+/[a-z]+/AI-RES(/|[[:space:]]|$|`|")|/scratch/[0-9]+/[a-z]+/AI-RES(/|[[:space:]]|$|`|")|/work2?/[0-9]+/[a-z]+/(stampede3/)?AI-RES(/|[[:space:]]|$|`|")|\$SCRATCH/AI-RES(/|[[:space:]]|$|`|")|\$WORK2?/AI-RES(/|[[:space:]]|$|`|")|feynmanliu214/AI-RES-Stampede3)'
grep -nE "$RW" docs/dsi_full_training_plan.md docs/dsi_smoke_backup_plan.md \
  > /tmp/dsi_doc_audit_to_rewrite.txt
echo "DSI-doc lines requiring manual Edit:"
cat /tmp/dsi_doc_audit_to_rewrite.txt
# Hand-edit each line above per the rewrite table. After edits, re-run and confirm empty.
```

**After Phase 6:** `git grep -n 'AI-RES' -- ':!docs/codex_reviews' ':!external' ':!makani-src' ':!docs/2026-05-23_sfno_climate_emulator_migration_plan.md'` and classify remaining hits. **Expected remainders (intentionally preserved):**
- DSI-specific patterns from the table above (`$HOME/AI-RES`, `$DSI_PROJECT/AI-RES`, `$DSI_SCRATCH/AI-RES`, `AI-RES-dsi-bootstrap`).
- Historical narrative mentions inside `docs/2026-05-04_makani_local_patches.md`, `docs/plasim_expansion_and_adaptor_plan.md`, and other plan/post-mortem docs where bare `AI-RES` refers to the *historical* project name as recorded at the time of writing. These are not path strings; the absolute paths inside those same docs WERE rewritten by Phase 5 sed. Treat the bare-name occurrences as preserved historical record (similar to how a renamed company's name still appears in old press releases).
- Anything in `docs/codex_reviews/` — review artifacts intentionally frozen.

If any remaining hit is a *path* (not a historical narrative reference), it's a sed miss — investigate and Edit it manually.

## Phase 7 — Rename GitHub repo

```bash
gh repo rename SFNO_Climate_Emulator-Stampede3 --repo feynmanliu214/AI-RES-Stampede3
cd ~/projects/SFNO_Climate_Emulator
git remote set-url origin git@github.com:feynmanliu214/SFNO_Climate_Emulator-Stampede3.git
git remote -v   # verify
git fetch origin
```

GitHub auto-redirects `feynmanliu214/AI-RES-Stampede3` → new URL for ~3 months, so anyone with the old URL keeps working. Update the single in-repo reference: `docs/dsi_smoke_backup_plan.md:351`.

## Phase 8 — Transitional symlinks (~2 weeks safety net)

```bash
ln -sfn /home1/11114/zhixingliu/projects/SFNO_Climate_Emulator /home1/11114/zhixingliu/AI-RES
ln -sfn /scratch/11114/zhixingliu/SFNO_Climate_Emulator        /scratch/11114/zhixingliu/AI-RES
ln -sfn /work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator /work2/11114/zhixingliu/stampede3/AI-RES
```

**Verify both paths work:**
```bash
ls /home1/11114/zhixingliu/AI-RES/README.md
ls /home1/11114/zhixingliu/projects/SFNO_Climate_Emulator/README.md
ls $SCRATCH/AI-RES/data/makani
ls $SCRATCH/SFNO_Climate_Emulator/data/makani
```

Set a calendar reminder for ~2 weeks out to remove these.

## Phase 9 — Bridge Claude auto-memory

Existing memory dir: `/home1/11114/zhixingliu/.claude/projects/-home1-11114-zhixingliu-AI-RES/memory/`

Cwd of new project encodes to: `-home1-11114-zhixingliu-projects-SFNO_Climate_Emulator`

```bash
ln -sfn /home1/11114/zhixingliu/.claude/projects/-home1-11114-zhixingliu-AI-RES \
        /home1/11114/zhixingliu/.claude/projects/-home1-11114-zhixingliu-projects-SFNO_Climate_Emulator
ls /home1/11114/zhixingliu/.claude/projects/-home1-11114-zhixingliu-projects-SFNO_Climate_Emulator/memory/MEMORY.md
```

Future Claude sessions from the new cwd will see the existing 24 memories. Writes flow back to the same files.

## Phase 10 — Verification (cheap smoke + bit-identical eval reproduction)

### 10a — Cheap smoke (catches obvious breakage). Runs with `set -euo pipefail` — any failure aborts.

```bash
set -euo pipefail
cd ~/projects/SFNO_Climate_Emulator
source .venv/bin/activate

# Python/torch/makani import + paths.
python -c "import torch, makani; print(torch.__version__, makani.__file__)"

# Symlinks resolve.
ls -L data checkpoints results .venv/bin/python >/dev/null

# SLURM script syntax — own-track, group-track, and the inline eval bodies the SLURM wrappers source.
for f in src/sfno_training/submit_zgplev_*.slurm \
         scripts/submit_eval_*.slurm \
         src/sfno_training_group/slurm/*.slurm \
         scripts/eval_run_inference_inline.sh \
         scripts/eval_run_score_inline.sh \
         scripts/eval_run_report_inline.sh \
         scripts/eval_run_figures_inline.sh; do
  bash -n "$f" || { echo "BROKEN: $f"; exit 1; }
done

# Group env wrapper end-to-end — fails the whole 10a block if activation lands outside SFNO_Climate_Emulator.
(
  set -euo pipefail
  source src/sfno_training_group/env_activate.sh
  python -c "import sys; assert 'SFNO_Climate_Emulator' in sys.executable, sys.executable; print('group env OK:', sys.executable)"
  conda deactivate
)

# Path strings clean inside tracked source — abort if anything stale remains.
if git grep -nE '/AI-RES/|\$SCRATCH/AI-RES|\$WORK2/AI-RES' \
    -- 'src/' 'scripts/' '*.slurm' '*.yaml' 'tests/' \
    ':!src/sfno_training/*.dsi*.slurm'; then
  echo "STALE: tracked source still contains AI-RES paths"
  exit 1
fi

# A small targeted test.
.venv/bin/pytest tests/plasim_makani_packager/test_metadata.py -q
```

### 10b — Bit-identical eval reproduction (the load-bearing test)

**Goal:** Re-run the full eval chain on run `sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_noise0p020_epochs75` against the same test set, and verify the new IC-averaged scorecard matches the recorded one. Any numeric drift = something in the inference/scoring pipeline depends on a path or env that changed silently.

**Eval pipeline writes (per `scripts/score_nwp.py:8-10` and `scripts/render_eval_report.py:6`):**
- `scores/nwp_scorecard.csv` (long format, per-IC × per-lead)
- `scores/nwp_scorecard_summary.csv` (IC-averaged — the primary diff target)
- `report.md` (markdown summary)
- `figures/` (PNGs)
- `inference/` (per-IC NetCDFs)
- `provenance.txt`

There is **no `score.md`** or `score.json`; do not look for those.

**Baseline-dir selection:** done as Pre-flight step 6 (before any rename) — see that section. The baseline path, baseline copy, and run dir are all persisted to `/tmp/migration_baseline.txt`. Phase 10b sources that file directly; no globbing of `/tmp/baseline_eval_*`.

**Post-migration:** re-run the eval chain. After Phase 5.5, the run-dir `config.json` has been rewritten to new paths, so the eval prelude resolves cleanly without depending on the transitional symlink. Use the eval-sfno-own skill's submission chain (`scripts/submit_eval.sh` per `scripts/submit_eval.sh:14`):

```bash
cd ~/projects/SFNO_Climate_Emulator
source .venv/bin/activate

# Recover the baseline handles from Pre-flight step 6. Sourcing makes this robust to running
# Phase 10b in a fresh shell long after pre-flight ran.
. /tmp/migration_baseline.txt
test -d "$BASELINE_COPY" || { echo "ABORT: BASELINE_COPY=$BASELINE_COPY not found"; exit 1; }

NEW_RUN_DIR=${OLD_RUN_DIR//\/AI-RES\//\/SFNO_Climate_Emulator\/}
ls -d "$NEW_RUN_DIR"   # must exist (renamed Lustre path)

# Submit chain with a fresh RUN_TAG/OUT_ROOT to avoid the ALLOW_RERUN collision guard
# (submit_eval_prelude.sh:153). RUN_TAG is timestamped so retries on failure don't collide
# with an earlier partial attempt — the prelude refuses any existing OUT_ROOT.
export RUN_DIR="$NEW_RUN_DIR"
export RUN_TAG="postrename_v11_gb32_lr8e4_minlr1e5_noise0p020_epochs75_$(date +%Y%m%d_%H%M%S)"
export OUT_ROOT="$WORK2/SFNO_Climate_Emulator/results/sfno_eval/$RUN_TAG"
# Belt-and-braces collision check before submission:
test ! -e "$OUT_ROOT" || { echo "ABORT: OUT_ROOT=$OUT_ROOT already exists — pick a new RUN_TAG"; exit 1; }
export MODE=nwp   # baseline was NWP — confirm via grep -l 'NWP scorecard' $OLD_EVAL_DIR/report.md
# K is NOT overridable from submit_eval.sh; eval_inference.py defaults to --nwp-K=56,
# which matches the baseline (per eval-sfno-own skill: v10/v11 own-track NWP = K=56). No SCORE_ONLY_K env var.

# Persist NEW_OUT_ROOT so the diff step can recover it even from a fresh shell after SLURM finishes.
echo "NEW_OUT_ROOT=$OUT_ROOT" >> /tmp/migration_baseline.txt
echo "NEW_RUN_TAG=$RUN_TAG"   >> /tmp/migration_baseline.txt

SUBMIT_LOG=/tmp/migration_submit_eval_$(date +%H%M%S).log
bash scripts/submit_eval.sh 2>&1 | tee "$SUBMIT_LOG"
# submit_eval.sh has a known fail-open bug: `if ! submit_eval_compute_env; then rc=$?` (submit_eval.sh:37)
# captures the inverted exit, so a prelude failure silently exits 0 and queues no jobs.
# Validate the actual submission by counting the four job IDs the script prints (submit_eval.sh:65-77).
n_jobs=$(grep -cE '^\[submit_eval\] (inference|scoring|report|figures)' "$SUBMIT_LOG")
[ "$n_jobs" -eq 4 ] || { echo "ABORT: submit_eval.sh only logged $n_jobs job submissions (need 4) — prelude likely failed silently. See $SUBMIT_LOG"; exit 1; }
```

Wait for all 4 SLURM jobs (`squeue -u $USER`) to finish (~1h15m end-to-end per `project_bundled_eval_tail_timing.md`), then diff against the baseline:

```bash
. /tmp/migration_baseline.txt   # re-export BASELINE_COPY, OLD_EVAL_DIR, OLD_RUN_DIR, NEW_OUT_ROOT, NEW_RUN_TAG
test -d "$NEW_OUT_ROOT" || { echo "ABORT: NEW_OUT_ROOT=$NEW_OUT_ROOT not found — did the SLURM chain finish?"; exit 1; }
export NEW_SUMMARY="$NEW_OUT_ROOT/scores/nwp_scorecard_summary.csv"
export NEW_LONG="$NEW_OUT_ROOT/scores/nwp_scorecard.csv"
export NEW_REPORT="$NEW_OUT_ROOT/report.md"
export OLD_BASE="$BASELINE_COPY"

# Primary: IC-averaged summary CSV — bit-identical expected.
diff -u "$OLD_BASE/scores/nwp_scorecard_summary.csv" "$NEW_SUMMARY"

# Secondary: long-form per-IC × per-lead CSV.
diff -u "$OLD_BASE/scores/nwp_scorecard.csv" "$NEW_LONG"

# Tertiary (ADVISORY ONLY — not load-bearing): report.md. Expected to differ on RUN_TAG/CKPT
# header lines (render_eval_report.py:387,394) AND on pr_6h convention notes that the current
# render writes by default (render_eval_report.py:135,207,253) — those notes may not have
# existed when the baseline report was generated. The numeric body should still match modulo
# rounding; the CSV diff above is the actual migration acceptance gate. Run the report diff
# for the record, but failures here do NOT block Phase 11.
diff_report() {
  python - "$OLD_BASE/report.md" "$NEW_REPORT" <<'PY'
import re, sys, difflib
def normalize(p):
    s = open(p).read()
    # Anything in `backticks` on a Run-tag-style label line — covers `postrename_…`,
    # date-prefixed historical tags like `20260522_eval-…`, and anything else.
    s = re.sub(r'(\*\*Run tag:\*\*\s*`)[^`]+(`)', r'\1<RUN_TAG>\2', s)
    s = re.sub(r'(\*\*Checkpoint:\*\*\s*`)[^`]+(`)', r'\1<CKPT>\2', s)
    # Any absolute path string we can recognize.
    s = re.sub(r'/scratch/\S*', '<PATH>', s)
    s = re.sub(r'/work2?/\S*', '<PATH>', s)
    s = re.sub(r'/home1/\S*', '<PATH>', s)
    return s
old, new = normalize(sys.argv[1]).splitlines(), normalize(sys.argv[2]).splitlines()
diff = list(difflib.unified_diff(old, new, lineterm=''))
print('\n'.join(diff) if diff else 'report.md identical after header/path normalization')
PY
}
diff_report
# If raw diff is also wanted for the record:
diff -u "$OLD_BASE/report.md" "$NEW_REPORT" > /tmp/report_md_raw_diff.txt 2>&1 || true
echo "Raw report.md diff at /tmp/report_md_raw_diff.txt (expected non-empty; the filtered diff above is the load-bearing check)."

# Tolerant numeric diff (use if `diff -u` shows non-numeric noise but numbers match).
python - <<'PY'
import csv, os
old_p = os.path.join(os.environ['OLD_BASE'], 'scores/nwp_scorecard_summary.csv')
new_p = os.environ['NEW_SUMMARY']
old = list(csv.DictReader(open(old_p)))
new = list(csv.DictReader(open(new_p)))
assert len(old) == len(new), f"row count {len(old)} vs {len(new)}"
diffs = 0
for r_o, r_n in zip(old, new):
    for k in r_o:
        try: a, b = float(r_o[k]), float(r_n[k])
        except (TypeError, ValueError):
            if r_o[k] != r_n[k]: diffs += 1; print(f"STR DIFF {k}: {r_o[k]!r} vs {r_n[k]!r}")
            continue
        if a != b:
            diffs += 1; print(f"NUM DIFF {k}: {a} vs {b}  (delta={b-a:.3e})")
print(f"total diffs: {diffs}")
PY
```

**Acceptance gate (load-bearing):** `diff -u` on `scores/nwp_scorecard_summary.csv` is empty (or the tolerant numeric diff reports 0 diffs). The `report.md` differ is advisory; do NOT block Phase 11 on it.

**If they don't match:** do NOT commit Phase 11. Investigate. Likely causes:
- A stale `.pth` or `.egg-link` in `.venv` still pointing at the old path → wrong module gets imported.
- Inference picks up a different dataset path (Phase 5.5 may have missed something — re-grep `config.json` under run dirs).
- A code path reads a hardcoded absolute path that the sed audit missed.
- Non-determinism: CUDA cudnn/atomics. Validate by re-running the baseline once more on the original tree (before applying migration) to confirm determinism; if the original run isn't reproducible to itself, relax the acceptance to "match to N decimal places" rather than bit-identical and document the new tolerance.

## Phase 11 — Commit and push

```bash
git checkout -b rename-to-sfno-climate-emulator

# Stage only tracked-file modifications (the sed + Edit changes from Phase 5/6).
# Do NOT use `git add -A`: it would stage vendored untracked clones (external/, makani-src/)
# and analysis_outputs/ — all out of scope per the plan's Out-of-scope list.
git add -u                                                   # all tracked modifications

# Decide explicitly which previously-untracked files to add. Pre-flight inventoried
# ~100 untracked files containing AI-RES; Phase 5 pass 2 rewrote them in place but they
# remain untracked. The migration commit should include any that are load-bearing for
# future execution (e.g. active eval scripts, skills, configs) and leave true-WIP/scratch
# files untracked. Review the inventory and add deliberately:
echo "Previously-untracked files Phase 5 rewrote (decide per-line whether to include in this commit):"
cat /tmp/migration_untracked_active_with_airrs.txt
# Example — add the production-critical ones from past loop discussions:
git add scripts/submit_eval_prelude.sh \
        scripts/submit_eval_5410.sh \
        scripts/eval_run_inference_inline.sh \
        scripts/eval_run_score_inline.sh \
        scripts/eval_run_report_inline.sh \
        scripts/eval_run_figures_inline.sh \
        src/sfno_training_group/env_activate.sh \
        .claude/skills/eval-sfno-5410/SKILL.md \
        skills/train-sfno-hpo/SKILL.md
# Add others as appropriate; check `git diff --cached --stat` before committing.

git status     # review: tracked modifications + the explicitly-added files; everything else still untracked
git diff --cached --stat
git commit -m "Rename AI-RES → SFNO_Climate_Emulator across paths, configs, skills, docs"
git push -u origin rename-to-sfno-climate-emulator
gh pr create --title "Rename project: AI-RES → SFNO_Climate_Emulator" --body ...
```

Don't merge until at least one full SLURM job (e.g., a smoke eval) succeeds end-to-end on the new paths.

## Phase 12 — Save plan + update memory

- Copy this plan to `~/projects/SFNO_Climate_Emulator/docs/2026-05-23_sfno_climate_emulator_migration_plan.md` (per the project's `YYYY-MM-DD_*_plan.md` convention).
- Update memory:
  - `reference_stampede3_paths.md` — update `$SCRATCH/AI-RES` → `$SCRATCH/SFNO_Climate_Emulator` etc.
  - `reference_github_repo.md` — update repo URL
  - `project_layout.md` — update directory location
  - `MEMORY.md` index lines that mention old paths
- Add a new memory: `project_directory_rename_2026_05_23.md` documenting the cutover and the transitional-symlink expiry date.

## Critical files / patterns touched

The Phase-5 sed pass hits these high-traffic spots (already inventoried — listing for reference, not exhaustive):

- **SLURM training drivers** (~15 files): `src/sfno_training/submit_zgplev_*.slurm`, `src/sfno_training/slurm_helpers.sh`, `src/sfno_training/bundled_eval.sh`
- **SLURM eval drivers** (~12 files): `scripts/submit_eval_*.slurm`, `scripts/submit_eval.sh`, `scripts/eval_run_*.sh`
- **Python eval/inference**: `scripts/score_5410.py`, `scripts/infer_sfno5410_*.py`, `scripts/build_ic_nc_from_h5.py`, `scripts/debug_sfno5410_*.py`, `scripts/hpo_prune.py`, `scripts/compute_climatology.py`, `scripts/eval_inference*.py`
- **Packager metadata**: `src/plasim_makani_packager/metadata.py:67`
- **Group env activation**: `src/sfno_training_group/env_activate.sh` rewrites via Phase 5; the conda env's internal absolute paths (shebangs, conda-meta JSON, activate.d hooks) are fixed by **Phase 4b** — the `mv` alone is not enough.
- **Skills**: `.claude/skills/eval-sfno-own/SKILL.md`, `.claude/skills/eval-sfno-5410/SKILL.md`, `skills/{sfno-training,plasim-makani-packager,plasim-postprocess}/SKILL.md`
- **Docs**: `docs/dsi_*.md`, `docs/plasim_*.md`, `docs/sfno_*.md`, `docs/2026-05-04_makani_local_patches.md`, plus `README.md`

## Rollback plan (if anything goes badly wrong)

Because Phases 1-3 are all `mv`/`ln -s` operations, rollback is a sequence of reverse `mv`s. Capture in a single script before starting:

```bash
# rollback.sh — DO NOT run unless something is broken

# 1. Remove transitional symlinks at the OLD locations FIRST. If left in
#    place, the reverse `mv` would try to overwrite a symlink with a real
#    directory — `mv` refuses, or worse, nests the dir inside the symlink target.
[ -L /home1/11114/zhixingliu/AI-RES ] && unlink /home1/11114/zhixingliu/AI-RES
[ -L /scratch/11114/zhixingliu/AI-RES ] && unlink /scratch/11114/zhixingliu/AI-RES
[ -L /work2/11114/zhixingliu/stampede3/AI-RES ] && unlink /work2/11114/zhixingliu/stampede3/AI-RES

# 2. Reverse the renames (with the same guards as Phase 1/2).
test ! -e /home1/11114/zhixingliu/AI-RES && \
  mv /home1/11114/zhixingliu/projects/SFNO_Climate_Emulator /home1/11114/zhixingliu/AI-RES
test ! -e /scratch/11114/zhixingliu/AI-RES && \
  mv /scratch/11114/zhixingliu/SFNO_Climate_Emulator /scratch/11114/zhixingliu/AI-RES
test ! -e /work2/11114/zhixingliu/stampede3/AI-RES && \
  mv /work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator /work2/11114/zhixingliu/stampede3/AI-RES

# 3. Re-create old in-tree symlinks (or git restore them):
cd ~/AI-RES
ln -sfn /scratch/11114/zhixingliu/AI-RES/checkpoints checkpoints
ln -sfn /scratch/11114/zhixingliu/AI-RES/data data
ln -sfn /work2/11114/zhixingliu/stampede3/AI-RES/results results
ln -sfn /work2/11114/zhixingliu/stampede3/AI-RES/.venv .venv

# 4. Revert in-repo sed: git reset --hard $(cat /tmp/migration_pre_head.txt)
# 5. Restore generated artifacts under run dirs from the Phase 5.5 backup tar + symlink manifest:
#    BACKUP=$(awk -F= '$1=="BACKUP"{print $2}' /tmp/migration_phase55_backup.txt)
#    tar -xzf "$BACKUP/run_artifact_text.tar.gz" -C /    # tar entries are absolute paths
#    # Restore symlinks to their original (pre-rewrite) targets:
#    while IFS=$'\t' read -r link target; do ln -sfn "$target" "$link"; done < "$BACKUP/symlink_manifest.tsv"
# 6. Revert GitHub repo: gh repo rename AI-RES-Stampede3 --repo feynmanliu214/SFNO_Climate_Emulator-Stampede3
# 7. Remove memory-bridge symlink:
#    unlink /home1/11114/zhixingliu/.claude/projects/-home1-11114-zhixingliu-projects-SFNO_Climate_Emulator
```

Phases 4-6 are all in-repo or under the moved dirs — `git reset --hard` recovers the tree state; the venv fix-up sed is reversible with the inverse sed if needed (or just rebuild the venv).

## Estimated time

- Pre-flight + Phases 1-3: ~5 min
- Phase 4 (venv fix-up + reinstall editables): ~5-10 min
- Phase 4b (group conda env fix-up): ~3-5 min
- Phase 4c (packed Derecho env fix-up — 601 stale paths): ~3-5 min
- Phase 5 (bulk sed, tracked + untracked passes): ~1 min, then ~10-15 min reviewing the diff
- Phase 5.5 (rewrite generated artifacts under run dirs): ~5 min (tar backup + find + sed over O(few hundred) small text files)
- Phase 6 (targeted edits): ~10 min
- Phase 7 (GitHub rename): ~2 min
- Phase 8 (transitional symlinks): ~1 min
- Phase 9 (memory bridge): ~1 min
- Phase 10a (cheap smoke): ~10 min
- Phase 10b (bundled eval reproduction): **~1h15m wall-clock** in SLURM queue + 5 min to diff
- Phase 11 (commit + PR): ~5 min (gated on Phase 10b passing)
- Phase 12 (plan-doc + memory update): ~5 min

**Total: ~2.5 hours wall-clock**, dominated by the bundled eval. Active human time ~45 min; the rest is SLURM tail.
