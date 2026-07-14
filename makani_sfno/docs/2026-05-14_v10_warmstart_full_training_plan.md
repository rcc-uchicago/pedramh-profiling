# v10 group-clone warm-start full-training plan

**Status:** DRAFT-v3 — revised 2026-05-14 (round 2) per Codex re-review:
explicit provenance-file pathspec, automatic in-slurm provenance archive,
preflight count corrected. Round-1 fixes (model_state key, verified
source counters, symlink-aware data check, yaml/slurm diff gates,
raw-vs-EMA framing) carried forward.
Awaiting user re-approval.
**Date:** 2026-05-14
**Author:** Claude (via interview with Zhixing).
**Sibling plan:** [`docs/2026-05-14_v11_clip_warmstart_continuation_plan.md`](2026-05-14_v11_clip_warmstart_continuation_plan.md).

## 1. TL;DR

Run a **50-epoch warm-start continuation** of the v10 own-track SFNO emulator,
loading model weights from the existing
`sfno_zgplev_group_clone` raw-best checkpoint and reinitializing the
optimizer + scheduler + counters from scratch. The continuation recipe is
**byte-identical to the original v10 `plasim_sim52_zgplev_group_clone.yaml`**
(clip=0, input_noise σ=0.05, lr_peak=1e-4 cosine, EMA 0.999), so the only
delta vs that prior v10 run is *the starting weights*. Score against the
own-track v10 NWP eval chain.

This is a **single-knob diagnostic on v10**: does a fresh 50-epoch optimizer
trajectory, starting from the v10 raw-best, materially improve own-track
scores vs the original v10 group-clone run? It parallels the in-flight
v11_clip_warmstart experiment (job 3117419) but on the v10 data /
v10 source-ckpt half of the matrix.

**Compute scope.** This adds **one** training run; no HPO, no recipe sweep.
Pre-authorized by user through the 2026-05-14 interview (Q1–Q4 + Q5).

## 2. Scientific motivation

The v11_clip_warmstart experiment tests whether a fresh optimizer
trajectory from the v11_clip raw-best can close the zg500/tas RMSE gap
vs SFNO-5410 on v11 data. That experiment runs the **continuation recipe
with clip=32** because v11_clip's innovation was clip restoration.

For v10, there is **no clip-matched source ckpt**: every v10 run was
trained with `optimizer_max_grad_norm=0.0` (disabled). Per user direction
(2026-05-14 interview Q5), the continuation recipe will therefore also use
clip=0, keeping the recipe byte-identical to its source-ckpt's training.

This makes the v10 experiment a clean test of the **warm-start mechanism
itself** (fresh optimizer/scheduler/counters from a converged model state),
isolated from the v11_clip recipe change. If v10 warmstart improves over
v10 group_clone but v11_clip_warmstart does *not* improve over v11_clip,
the gain is recipe-independent; if both improve, the warm-start trick is
robust; if neither improves, the trick is a no-op for our problem.

## 3. Source checkpoint

| Property                | Value                                                                                                                  |
|-------------------------|------------------------------------------------------------------------------------------------------------------------|
| Path                    | `$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_group_clone/plasim_sim52_zgplev_group_clone/0/training_checkpoints/best_ckpt_mp0.tar` |
| Size                    | 1,710,240,559 bytes (1.71 GB) — has optimizer state baked in but we will skip optimizer restore                        |
| mtime                   | 2026-05-10 14:21:36 -0500                                                                                              |
| md5                     | `d1b82a9636d2fa62b43b870149c42dab`                                                                                     |
| Recipe at save time     | v10 group_clone: clip=0.0, input_noise σ=0.05, lr_peak=1e-4 cosine, EMA 0.999, AdamW betas=(0.9, 0.999), wd=3e-6        |
| Flavor                  | **raw** best (lower val loss than ema-best at this ckpt's save epoch — same convention as v11_clip_warmstart source)   |
| EMA-best counterpart    | `best_ckpt_ema_mp0.tar` (427,554,515 bytes) — NOT used as source; using raw is the standard warm-start convention      |
| Memory cross-references | `feedback_ema_is_canonical_ckpt` (raw is diagnostic for *eval*; source-ckpt convention separate — see v11_clip_warmstart plan §3 for the same call) |

**Raw vs EMA as source — convention, not a safety claim.** Both flavors
share the same `model_state` key contract; loading either would work
mechanically. The trade-off:

- **Raw** (what we picked): matches live `model.state_dict()` at the moment of
  save; the new run's EMA shadow gets reseeded fresh from these weights at
  trainer construction. Mirrors v11_clip_warmstart §3 for cross-comparability.
- **EMA**: aligns closer with the canonical eval flavor (own-track uses
  `best_ckpt_ema_mp0.tar` per `feedback_ema_is_canonical_ckpt`). Starting
  from the EMA-best weights would arguably be a "warmer" start but would
  reseed a new EMA shadow from EMA weights — minor double-averaging quirk.

For this diagnostic, raw is acceptable and matches the sibling experiment.
The new run still produces its own EMA shadow during training, which
becomes the canonical eval ckpt regardless of which source flavor we load.

## 4. Continuation recipe

**Byte-identical to** `src/sfno_training/config/plasim_sim52_zgplev_group_clone.yaml`.
The new yaml `plasim_sim52_zgplev_group_clone_v10_warmstart.yaml` is a copy of
that file with only the top-level config block name renamed
(`plasim_sim52_zgplev_group_clone` → `plasim_sim52_zgplev_group_clone_v10_warmstart`)
and the header comments pointing at the new submit slurm.

Preserved knobs (from the v10 group_clone yaml):
- `optimizer_max_grad_norm: 0.0` (disabled — matches source)
- `input_noise.sigma: 0.05`, `mode: "perturb"`, 52 state channels perturbed
- `lr: 1.0E-4`, `weight_decay: 3.0E-6`, AdamW betas (0.9, 0.999)
- `max_epochs: 50`, `batch_size: 8` (global, per-rank 2 at 4-GPU DDP)
- `scheduler: "CosineAnnealingLR"`, `scheduler_T_max: 45`, `scheduler_min_lr: 1.0E-8`, `lr_warmup_steps: 5`, `lr_start: 1.0E-4`
- `ema.enabled: True`, `ema.decay: 0.999`, `ema.warmup: True`
- All architecture knobs (embed_dim 256, num_layers 12, etc.) — unchanged
- Channel list, normalization, dataset-path placeholders — unchanged
- `pretrained: !!bool False` stays False; we DO NOT route through stock makani's pretrained path.

Implementation note: the warm-start CLI knob and trainer load-order
were added in the v11_clip_warmstart implementation. **They are NOT
yet committed** as of 2026-05-14 — `git status --porcelain` shows
the following warm-start code paths dirty (`M`) or untracked (`??`):

```
 M scripts/render_eval_report.py
 M scripts/submit_eval_report.slurm
 M src/sfno_training/train_plasim.py
 M src/sfno_training/trainer/plasim_trainer.py
?? src/sfno_training/config/plasim_sim52_zgplev_group_clone_v11_clip_warmstart.yaml
?? src/sfno_training/submit_zgplev_group_clone_v11_clip_warmstart.slurm
?? tests/sfno_training/test_pretrained_warmstart.py
```

This matters for provenance: `makani/utils/logging_utils.py:49`'s
`log_versions()` records only `HEAD` and **does not capture the
working-tree diff**. The v11_clip_warmstart job 3117419 is already
running against this dirty tree — that's a separate provenance debt.

The verified-good list (per Codex review) confirms the load semantics
are correct:
- `src/sfno_training/trainer/plasim_trainer.py:284` — only warm-starts when not resuming.
- `src/sfno_training/trainer/plasim_trainer.py:285` — passes `loss=None, optimizer=None, scheduler=None, counters=None`.
- `src/sfno_training/trainer/plasim_trainer.py:335` — EMA constructed after warm-start load → shadow seeds from loaded weights.

**Pre-launch gate (NEW, see §6 #0):** commit the warm-start code paths
before sbatch so `log_versions()` records a clean HEAD that fully
identifies what's running. Belt-and-suspenders: also archive
`git diff HEAD` + untracked-file list to `$EXP_DIR/code_provenance/`
at submit time so even a post-hoc forensic recovery has a frozen
snapshot.

CLI / config surface (preserved across the commit):
- `train_plasim.py --pretrained_checkpoint_path <path>` argparse flag.
- `PlasimTrainer.__init__` warm-start load between `super().__init__()`
  and `EMAModel` construction.
- `params.resuming` takes precedence over warm-start.

No *new* code changes needed for this experiment — only a new yaml and
a new submit slurm. But the v11_clip_warmstart code must be committed
first (or archived) per above.

## 5. Implementation plan

### 5.1. New files to create

1. **`src/sfno_training/config/plasim_sim52_zgplev_group_clone_v10_warmstart.yaml`**
   - Copy of `plasim_sim52_zgplev_group_clone.yaml`, top-level key renamed.
   - Header comments updated to point at the new submit slurm + this plan.
   - **No `pretrained_checkpoint_path` field in the yaml** — CLI-only,
     per v11_clip_warmstart §4.2 (yaml is data-recipe; ckpt path is
     run-instance and lives in env+CLI).

2. **`src/sfno_training/submit_zgplev_group_clone_v10_warmstart.slurm`**
   - Clone of `submit_zgplev_group_clone_v11_clip_warmstart.slurm`.
   - **Defaults switched to v10**:
     - `OUTPUT_ROOT="$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full"` (no `_v11` suffix).
     - `EXP_DIR="$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_group_clone_v10_warmstart"`.
     - `PRETRAINED_CKPT="$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_group_clone/plasim_sim52_zgplev_group_clone/0/training_checkpoints/best_ckpt_mp0.tar"`.
     - All `plasim_sim52_zgplev_group_clone_v11_clip_warmstart` tokens
       → `plasim_sim52_zgplev_group_clone_v10_warmstart`.
   - **Preserved guards** (verbatim from v11_clip_warmstart submit):
     - PRETRAINED_CKPT readable + size > 0.
     - protect_prior_runs: refuse if `$RUN0_DIR/training_checkpoints/ckpt_mp0_v*.tar` exists.
     - sed-render yaml placeholders.
     - preflight call to `scripts/preflight.py`.
     - status trap + slurm_helpers.
   - **NEW: code-provenance archive stanza** (per §6 #0).
     Add this block to the slurm AFTER `mkdir -p "$EXP_DIR" "$EXP_DIR/training_checkpoints"`
     and BEFORE the call to `scripts/preflight.py`:

     ```bash
     # --- Code-provenance archive (plan §6 #0) ---
     # log_versions() in makani records only HEAD and not the working-tree
     # diff. Even with §6 #0 enforcing a clean tree, archive everything
     # frozen-in-time here so post-hoc forensic recovery has the exact
     # code that ran. Runs unconditionally — cheap (<1s, KB-scale).
     PROV_DIR="$RUN0_DIR/code_provenance"
     mkdir -p "$PROV_DIR"
     ( cd "$REPO_ROOT" \
         && git rev-parse HEAD                 > "$PROV_DIR/HEAD.sha" \
         && git status --porcelain             > "$PROV_DIR/status.txt" \
         && git diff HEAD                      > "$PROV_DIR/diff.patch" \
         && git rev-parse --abbrev-ref HEAD    > "$PROV_DIR/branch.txt" )
     echo "[provenance] archived HEAD + diff + status under $PROV_DIR"
     ```

     This is **NOT** a manual step. If sbatch is invoked but the slurm
     doesn't contain this stanza, abort and add it before retrying.
   - Wallclock budget: 18:30:00 on h100 (same as v11_clip_warmstart — same compute per step, same epoch count).

### 5.2. No new tests required

The warm-start logic is already covered by:
- `tests/sfno_training/test_pretrained_warmstart.py` — 2 tests on the load-order + resume-precedence contract.
- `tests/scripts/test_render_eval_report_warmstart.py` — 3 tests on the sidecar→report.md plumbing.

These tests are recipe-agnostic. The new yaml is byte-identical to an
existing one except for the block name, so no behavior the existing
config-loading tests exercise will change. Full `pytest tests/` will
still be re-run as a preflight gate.

### 5.3. Sidecar provenance

`train_plasim.py` already writes `$EXP_DIR/.../0/warmstart_provenance.txt`
with these fields when `--pretrained_checkpoint_path` is set. The v10 run
will populate the same sidecar; the renderer's
`### Warm-start provenance` block will appear in this run's `report.md`
automatically when the eval chain runs with `RUN_DIR=` pointing at the
warmstart `/0` dir.

Expected sidecar field values (for verification):
- `pretrained_checkpoint_path` = absolute path printed above
- `pretrained_checkpoint_flavor` = `best_ckpt_mp0 (raw)`
- `pretrained_checkpoint_size_bytes` = `1710240559`
- `pretrained_checkpoint_sha256` first 16 hex = (computed at run-time;
  cross-check the md5 above as a coarse sanity)
- `lr_peak` = `0.0001`
- `max_epochs` = `50`
- `batch_size_global` = `8`
- `ema_decay` = `0.999`
- `optimizer_max_grad_norm` = `0.0` ← differs from v11_clip_warmstart sidecar (32.0)
- `input_noise_sigma` = `0.05`
- `channel_weights` = `constant`

## 6. Preflight checklist

Mirror the v11_clip_warmstart §4.4 checklist, adapted to v10 paths and
with the corrected makani ckpt key contract.

0. **Warm-start code committed (explicit pathspec).**
   Gate `git status --porcelain` against the **exact** file set that
   the v10 warmstart run depends on. Broad greps (`grep warmstart`)
   pick up plan docs and unrelated v11 artifacts while missing
   `scripts/submit_eval.sh` if its text doesn't happen to match — so
   we list the files explicitly:

   ```bash
   PROV_PATHS=(
       # --- Trainer / CLI changes the warm-start run executes ---
       src/sfno_training/train_plasim.py
       src/sfno_training/trainer/plasim_trainer.py

       # --- Tests covering the warm-start contract ---
       tests/sfno_training/test_pretrained_warmstart.py
       tests/scripts/test_render_eval_report_warmstart.py

       # --- Renderer / eval-chain code the post-training eval will run ---
       scripts/render_eval_report.py
       scripts/submit_eval_report.slurm
       scripts/submit_eval.sh

       # --- Sibling v11_clip_warmstart yaml + slurm (currently untracked) ---
       src/sfno_training/config/plasim_sim52_zgplev_group_clone_v11_clip_warmstart.yaml
       src/sfno_training/submit_zgplev_group_clone_v11_clip_warmstart.slurm

       # --- New v10 warmstart yaml + slurm (will exist after §5 implementation) ---
       src/sfno_training/config/plasim_sim52_zgplev_group_clone_v10_warmstart.yaml
       src/sfno_training/submit_zgplev_group_clone_v10_warmstart.slurm
   )
   # Use pathspec form so missing files (pre-implementation) are not errors;
   # untracked files still show up under ?? .
   git status --porcelain -- "${PROV_PATHS[@]}"
   ```

   Required: this command's output must be **empty** (every listed
   path must be committed, or absent if not yet created — but at sbatch
   time, all paths must exist and be tracked).

   If non-empty, commit those paths first so `log_versions()`
   (`makani-src/makani/utils/logging_utils.py:49`) records a HEAD that
   fully identifies the running code — `log_versions()` does NOT
   capture the working-tree diff.

   (Codex review 2026-05-14 noted v11_clip_warmstart job 3117419 was
   launched against a dirty tree → that run has incomplete provenance;
   do not repeat for this run.)

   **Belt-and-suspenders archive is enforced by the slurm itself** —
   see §5.1 #2 below. Do not rely on a manual archive step.

1. **EXP_DIR absent or empty.** `$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_group_clone_v10_warmstart` must not exist OR must contain no `training_checkpoints/ckpt_mp0_v*.tar`. The submit-script guard already enforces this; the preflight is a belt-and-braces check before sbatch.

2. **PRETRAINED_CKPT readable.** `stat -c '%s' /scratch/.../sfno_zgplev_group_clone/.../best_ckpt_mp0.tar` returns `1710240559`. The submit script's `[[ ! -s "$PRETRAINED_CKPT" ]]` guard catches absence; preflight confirms size.

3. **Source ckpt key inventory.** Use the correct makani key (`model_state`, NOT `model_state_dict` — Driver loads `checkpoint["model_state"]` at `makani-src/makani/utils/driver.py:423`) and pass `weights_only=False` for current PyTorch:

   ```bash
   python -c "
   import torch
   ck = torch.load(
       '/scratch/11114/zhixingliu/SFNO_Climate_Emulator/runs/sfno_zgplev_group_clone/plasim_sim52_zgplev_group_clone/0/training_checkpoints/best_ckpt_mp0.tar',
       map_location='cpu',
       weights_only=False,
   )
   sd = ck['model_state']
   print('n_keys=', len(sd))
   print('first3=', list(sd)[:3])
   print('has_module_prefix=', any(k.startswith('module.') for k in sd))
   "
   ```

   Expected: `n_keys=128`, `has_module_prefix=False` (same shape as the v11_clip source).

4. **Source ckpt provenance.** Same probe, dump the verified counters (per Codex review of the actual file):

   ```bash
   python -c "
   import torch
   ck = torch.load('.../best_ckpt_mp0.tar', map_location='cpu', weights_only=False)
   print('epoch        =', ck.get('epoch'))
   print('iters        =', ck.get('iters'))
   opt = ck['optimizer_state_dict']['param_groups'][0]
   print('initial_lr   =', opt.get('initial_lr'))
   print('current_lr   =', opt.get('lr'))
   "
   ```

   **Expected values (verified):** `epoch=50`, `iters=909950`, `initial_lr=0.0001`, `current_lr=1e-08`. Source: `src/sfno_training/submit_zgplev_group_clone.slurm:16` documents 909,950 optimizer steps for this very ckpt. If counters differ, the ckpt is not the right v10 group_clone artifact and we abort.

5. **v10 dataset paths populated.** `$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full/{train,valid,test,test_holdout,stats,metadata}` all present. Train file count check **must follow symlinks** because `sim52_zgplev_full/train/MOST.*.h5` are symlinks:

   ```bash
   ls -L $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full/train/MOST.*.h5 | wc -l   # expect 100
   find -L $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full/train -name 'MOST.*.h5' -type f | wc -l   # alt; same expected
   ```

   (Plain `find -type f` without `-L` returns 0 here and would mislead.)

6. **Tests green.** `pytest tests/ -q` returns 0 failures (the warm-start tests are already green — confirming nothing else regressed).

7. **Submit-script syntax.** `bash -n src/sfno_training/submit_zgplev_group_clone_v10_warmstart.slurm`.

8. **Yaml validity.** `python -c "import yaml; print(list(yaml.safe_load(open('src/sfno_training/config/plasim_sim52_zgplev_group_clone_v10_warmstart.yaml')).keys()))"` prints exactly `['plasim_sim52_zgplev_group_clone_v10_warmstart']`.

9. **Yaml diff gate.** `diff -u src/sfno_training/config/plasim_sim52_zgplev_group_clone.yaml src/sfno_training/config/plasim_sim52_zgplev_group_clone_v10_warmstart.yaml` shows ONLY: (a) the single config-block key rename on the one yaml-defining-line in `plasim_sim52_zgplev_group_clone.yaml:16`, and (b) header-comment lines. No knob lines may differ. If any non-comment, non-header line changes, abort.

10. **Slurm content gate.** `grep -E "OUTPUT_ROOT|PRETRAINED_CKPT" src/sfno_training/submit_zgplev_group_clone_v10_warmstart.slurm` must show the v10 dataset path (`sim52_zgplev_full` with no `_v11` suffix) and the v10 source ckpt path. Then `grep -E '(v11_clip|sim52_zgplev_full_v11)' src/sfno_training/submit_zgplev_group_clone_v10_warmstart.slurm | wc -l` must return `0` (no leftover v11 tokens from the clone).

## 7. Post-launch verification (within ~5 min of sbatch)

Per the v11_clip_warmstart §4.4 post-launch checks:

1. **Banner shows** `pretrained_checkpoint_path = /scratch/.../sfno_zgplev_group_clone/.../best_ckpt_mp0.tar` and `resuming = False` in `logs/sfno_zgplev_group_clone_v10_warmstart_<jobid>.err`.

2. **Trainer log** `INFO:sfno_training.trainer:warm-start: loaded weights from /scratch/.../best_ckpt_mp0.tar, optimizer/scheduler/counters NOT restored` appears once.

3. **Sidecar written** `$RUN0_DIR/warmstart_provenance.txt` exists, ~650–700 B, with `optimizer_max_grad_norm = 0.0` (v10 value) — visually distinguishing it from the v11_clip_warmstart sidecar which has `32.0`.

4. **Fresh counters.** `epoch 1`, `iters` start at 0.

5. **First-step loss is O(10⁻³).** Random-init would give O(10⁻¹). The v10 raw-best is a converged model so its loaded weights should land in [0.002, 0.004] on the first batch; if loss is O(10⁻¹), the warm-start did not actually load — abort and debug.

## 8. Eval plan

After the 50 epochs complete (~18 h wall on h100), score on the own-track
v10 NWP chain.

```bash
RUN_DIR="$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_group_clone_v10_warmstart/plasim_sim52_zgplev_group_clone_v10_warmstart/0" \
CKPT="$RUN_DIR/training_checkpoints/best_ckpt_ema_mp0.tar" \
MODE=nwp \
RUN_TAG="20260514_v10_warmstart_on_v10_testset_ema_startckpt-best_ckpt_mp0" \
scripts/submit_eval.sh
```

Skill: `eval-sfno-own` (own-track v10 only).
EMA-best is canonical per `feedback_ema_is_canonical_ckpt`.
TEST_HOLDOUT, TRAIN_DIR, PACKAGER_TEST_SRC fall back to the v10 defaults
documented in the skill (no overrides needed).
RUN_DIR must be passed explicitly because it does not match the
default-resolution glob `sfno_zgplev_full/plasim_sim52_zgplev_full/[0-9]+`.

### 8.1. Primary success gates (vs prior v10 group_clone EMA-best eval)

| Metric                 | v10 group_clone baseline    | v10 warmstart target          |
|------------------------|-----------------------------|-------------------------------|
| `zg500@6h` RMSE        | TBD (read from prior report) | ≤ baseline within noise         |
| `tas@6h` RMSE          | TBD                          | ≤ baseline within noise         |
| `tas@6h` ACC           | ~0.987 (from v11_clip yaml comment) | ≥ 0.987                   |
| Any metric degradation | n/a                          | No >5% RMSE worsen at 336 h    |

The baseline numbers will be read from the prior v10 group_clone eval
report at sbatch time (before launching, so the gates are concrete).

### 8.2. Hold-the-line gates

No channel may degrade by more than 5% in RMSE at any of the four
canonical lead times (6 h, 24 h, 120 h, 336 h) vs v10 group_clone. Per
memory `feedback_respect_compute_scope`, if the warmstart makes things
*worse* this is interesting (it tells us the original v10 trajectory was
near-optimal for this recipe) — we'd document and stop, not iterate.

### 8.3. Cross-experiment comparison (after both runs complete)

Once both v10_warmstart and v11_clip_warmstart finish:

| Comparison | Hypothesis                                                                       |
|------------|----------------------------------------------------------------------------------|
| v10_warmstart vs v10 group_clone | Tests warm-start mechanism in isolation (recipe held)         |
| v11_clip_warmstart vs v11_clip  | Tests warm-start mechanism + recipe (clip=32) jointly         |
| v10_warmstart vs v11_clip_warmstart on shared channels | Cross-data warmstart efficacy        |

Write-up doc: `docs/2026-05-15_warmstart_cross_comparison.md` (deferred,
created after both eval chains complete).

## 9. Risks and known unknowns

| Risk                                                    | Likelihood | Mitigation                                                                                                                                                                          |
|--------------------------------------------------------|------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Source ckpt was not actually 50 epochs of v10 group_clone | Low        | §6 preflight #4 probes the ckpt's `epoch`/`iters` headers explicitly; expected values verified (epoch=50, iters=909950, initial_lr=1e-4, current_lr=1e-8) per Codex 2026-05-14 review |
| Compute cost not justified (warmstart is a no-op)      | Low        | This is the *point* of the diagnostic — a null result is informative; budget pre-authorized by user                                                                                |
| Recipe drift between v10 group_clone yaml and this clone | Very low | §6 #9 (yaml diff gate) blocks any non-header/non-key-rename change                                                                                                                  |
| Slurm clone retains v11 tokens accidentally             | Very low | §6 #10 (slurm content gate) grep-asserts v10 paths and zero v11_clip / v11 tokens                                                                                                   |
| **Code provenance debt** — running code not in HEAD     | **Medium** | §6 #0 mandates committing the warm-start code paths before sbatch; belt-and-suspenders archives `git diff` + status to `$EXP_DIR/code_provenance/`. v11_clip_warmstart job 3117419 has this debt already; this run will not |
| Own-track v10 dataset shifted between 2026-05-10 (source ckpt save) and 2026-05-14 (now) | Low | sim52_zgplev_full hasn't been repackaged since 2026-05; preflight #5 confirms paths (with `-L` symlink-follow); if stats-file mtimes differ from expected, alert and pause          |
| EMA shadow seeded incorrectly                           | Low        | Already verified in v11_clip_warmstart; same code path; Codex 2026-05-14 review confirmed `plasim_trainer.py:335` EMA constructed after warm-start load                              |
| Stampede3 h100 queue priority for 18.5h job             | Medium     | If wait > 24 h, consider splitting into 2× 9 h with `--dependency=afterok` (deferred — only if scheduling actually slips)                                                          |

## 10. Open questions (resolve before sbatch)

1. **Should we also overlay SFNO-5410 on this v10 eval?**
   The default `BENCHMARK_5410_OUT_ROOT` in `submit_eval_report.slurm`
   points at the pinned valid 5410 run. The 5410 group convention has
   unit differences vs our v10 data (`pr_6h` rate×6h vs the legacy v10
   convention). The figures code already handles this with separate
   colorbars per row. Default-on is correct per memory `project_5410_eval_track`.
   **Resolution: keep the default overlay on.**

2. **Channel-weight comparison.**
   v10 group_clone uses `channel_weights: "constant"`. We are NOT
   proposing to change this — memory feedback `do not recommend
   per-channel loss weighting; we cannot improve zg500 by sacrificing
   other variables`. Just confirming inheritance.

## 11. Done definition

- [x] Plan drafted.
- [ ] Plan revised after Codex review 2026-05-14 (this revision).
- [ ] Plan re-approved by user.
- [ ] Warm-start code paths committed (§6 #0).
- [ ] yaml + slurm created.
- [ ] All 11 preflight checks (§6 #0–#10) pass.
- [ ] `sbatch` accepted; job ID logged.
- [ ] Post-launch §7 checks all pass within first 5 min.
- [ ] Training completes 50 epochs without OOM / NaN / crash.
- [ ] Eval chain runs end-to-end and `report.md` materializes with the
      `### Warm-start provenance` block.
- [ ] Results compared to v10 group_clone baseline and v11_clip_warmstart;
      write-up in `docs/2026-05-15_warmstart_cross_comparison.md`.

## 12. Cross-references

- v11_clip_warmstart plan: `docs/2026-05-14_v11_clip_warmstart_continuation_plan.md`
- Memory: `feedback_protect_prior_runs`, `feedback_input_noise_is_load_bearing`,
  `feedback_ema_is_canonical_ckpt`, `feedback_respect_compute_scope`,
  `project_own_track_v10_only`, `project_zgplev_gb_decision`
- Eval skill: `.claude/skills/eval-sfno-own/SKILL.md`
- Source-ckpt training plan/config: `src/sfno_training/config/plasim_sim52_zgplev_group_clone.yaml`
