# 2026-05-14 — v11_clip warm-start continuation diagnostic

Author: Claude (with Zhixing)
Status: proposed — single-knob continuation experiment to test whether a
*second full training stage* from the v11_clip best weights closes the
6 h zg500 / tas RMSE gap to SFNO-5410. **No HPO; no recipe sweep; no
training run launched without explicit sign-off.**

---

## 1. Motivation

Per the forensic audit chain (`docs/2026-05-14_zg500_targeted_hpo_plan.md`
extended with the rsdt-offset and normalization-stats follow-ups in the
2026-05-14 session):

- **Architecture / capacity**: ruled out — makani and 5410 have essentially
  identical real DoF (~106.9 M), per `docs/2026-05-10_sfno_param_count_forensic.md`.
- **Vertical level mapping**: ruled out — both tracks use identical 10 σ
  levels for ta/ua/va/hus and identical 10 plev levels (200..1000 hPa) for zg
  (v10.1 contract; matches 5410 yaml).
- **SST handling**: fixed on our side in v11 (`--sst-mode surface`); v11_clip
  was trained on v11 data and Pre-Run 0 (2026-05-13) confirms within-pipeline
  consistency.
- **rsdt time-of-day offset**: per user 2026-05-14, the offset is on the
  *group* side (theirs end-of-window labeled, ours is the correct PlaSim
  diagnostic) — not a contributor to our gap.
- **Normalization stats**: ruled out by the 2026-05-14 channel-by-channel
  comparison (zg500 mean diff = +0.060 σ_5410, std ratio = 0.998; all sigma
  channels within 1 %; pr_6h is a separate unit issue, see
  `docs/2026-05-14_pr_6h_units_mismatch_ticket.md`).

The single remaining material recipe difference is **cumulative training
compute via warm-start**. 5410's ep48 checkpoint is the result of:

1. Inheriting `model_state` from `load_exp_dir =
   /glade/work/marchakitus/PLASIM/PanguWeather/v2.0/results` (a prior
   PanguWeather PLASIM SFNO cycle; not mirrored to Stampede3).
2. Starting a **fresh optimizer** for this continuation (optimizer
   `step ≈ this-run iters`, not iters+precursor — verified from the ckpt's
   `optimizer_state_dict`).
3. Running a **full LinearWarmupCosineAnnealingLR schedule peaking at
   lr ≈ 1 e-04** over 50 epochs (the optimizer's
   `param_groups[0].initial_lr = 1.0 e-04`; current LR at the saved ckpt is
   `9.504 e-07`, deep in the cosine tail).
4. **No EMA** (no `ema_state` key in the 5410 ckpt; raw weights only).

So 5410's continuation is **not** a "low-LR polish" — it is a second full
training stage at the same LR shape as the precursor, starting from already
non-random weights. The yaml's `lr: 1 e-06` metadata is misleading; the
optimizer state shows the schedule actually ran from 1 e-04.

This plan replicates that recipe on our side: continue v11_clip from its
best weights for a second full 50-epoch run with a fresh optimizer and a
fresh LinearWarmupCosineAnnealing schedule peaking at lr = 1 e-04. Single
knob: **warm-start from v11_clip best vs from-scratch random init**.

## 2. Hypothesis

If cumulative training compute (warm-started weights) is the dominant cause
of the 6 h zg500 / tas RMSE gap to 5410, then a second 50-epoch training
stage starting from v11_clip's best weights will:
- Reduce zg500 @ 6h RMSE on the v11 testset substantially (target: closing
  at least half of the v11_clip → 5410 gap, i.e. ≤ 2.87 m vs current
  3.52 m and 5410's 2.22 m).
- Show similar improvements on tas @ 6h RMSE (current 0.433 K, 5410 0.271 K,
  target ≤ 0.35 K).
- NOT degrade long-lead skill (hold-the-line on zg500 @ 336h, tas @ 336h).

If the continuation does **not** close the 6 h gap, cumulative compute is
ruled out and the field narrows to (a) effective batch size (5410 GB=32 vs
ours GB=8) and (b) the unaudited group preprocessing chain on derecho.
Both are deferred questions; no further runs are pre-authorised by this plan.

## 3. Configuration (single-knob delta vs v11_clip)

All knobs identical to `plasim_sim52_zgplev_group_clone_v11_clip.yaml`
except a fresh `exp_dir` and the new `--pretrained_checkpoint_path`
runtime CLI param (passed by the submit slurm; not a YAML field — see §4.2).

| Knob | v11_clip baseline | This run (`v11_clip_warmstart`) | Why |
|---|---|---|---|
| Initial model weights | random init | **`best_ckpt_mp0.tar` from v11_clip /0** | The knob under test. Raw best (not EMA), per the recommendation in this session's interview — EMA shadow starts fresh from raw weights so the new EMA isn't an "EMA-of-EMA" curve. |
| Optimizer state | fresh (from-scratch) | **fresh** (NOT loaded from v11_clip ckpt) | Match 5410's continuation behavior — verified from the 5410 ckpt that optimizer `step ≈ this-run iters` (no precursor optimizer state inherited). |
| Scheduler | LinearLR warmup + CosineAnnealingLR | **same shape, fresh state** | Fresh schedule deterministically rebuilt from epoch 0 of this run. |
| `lr` (peak) | 1.0 e-04 | **1.0 e-04** | Match 5410's effective peak (per ckpt `initial_lr`). |
| `lr_start` (LinearLR start_factor) | 1.0 e-04 (→ warmup start 1 e-08) | 1.0 e-04 | unchanged |
| `lr_warmup_steps` | 5 epochs | 5 epochs | unchanged |
| `scheduler_T_max` | 45 | 45 | unchanged |
| `scheduler_min_lr` | 1.0 e-08 | 1.0 e-08 | unchanged (cosine asymptote) |
| `optimizer_type` | AdamW | AdamW | unchanged |
| `optimizer_beta1, beta2` | 0.9, 0.999 | 0.9, 0.999 | unchanged |
| `weight_decay` | 3.0 e-06 | 3.0 e-06 | unchanged |
| `optimizer_max_grad_norm` | **32.0** | **32.0** | **kept** per user 2026-05-14 |
| `max_epochs` | 50 | **50** | matches v11_clip; symmetric with 5410's 50-epoch continuation |
| `batch_size` (GLOBAL) | 8 | 8 | unchanged |
| `input_noise.sigma` | 0.05 | **0.05** | kept (load-bearing per `feedback_input_noise_is_load_bearing`) |
| `input_noise.mode`, `perturb_channels` | perturb, 52 state | perturb, 52 state | unchanged |
| `losses[0].channel_weights` | "constant" | **"constant"** | kept (no per-channel reweighting per user instruction) |
| `n_history`, `n_future` | 0, 0 | **0, 0** | kept (no multistep / no rollout2 per user instruction) |
| `valid_autoreg_steps` | 3 | 3 | unchanged |
| `ema.enabled`, `decay`, `warmup` | True, 0.999, True | True, 0.999, True | EMA shadow accumulates fresh from raw weights |
| Dataset | v11 (`sim52_zgplev_full_v11`) | **v11** (same paths) | unchanged |
| Test holdout for eval | `sim52_zgplev_full_v11/test_holdout` (matched) | **same** | unchanged |
| Training data | 100 yr (12–111) v11 | 100 yr (12–111) v11 | unchanged |

## 4. Files to create / modify

### 4.1 Trainer-side patch (one-time, required)

Makani's `Driver.restore_from_checkpoint(..., optimizer=None, scheduler=None,
counters=None)` already supports "load weights only" semantics
(`makani-src/makani/utils/driver.py:354-440`). **Do NOT use the stock
`pretrained: True` flag.** Per Codex review 2026-05-14, the stock pretrained
branch (`makani-src/makani/utils/training/deterministic_trainer.py:237`)
passes optimizer/scheduler/counters into `restore_from_checkpoint` because
`load_optimizer`, `load_scheduler`, `load_counters` default to `True` in
`makani-src/makani/utils/driver.py:137`. Accidentally restoring those from
the v11_clip best ckpt (epoch 49, iters 891 751, current lr 1.317 e-7,
`initial_lr = 1e-4`) would run roughly one tail epoch, not a fresh 50-epoch
stage. We bypass that path entirely with our own explicit call.

#### 4.1.1 Plumbing

Trainer lives at `src/sfno_training/trainer/plasim_trainer.py` (NOT
`src/sfno_training/trainer.py`).

| File | Change |
|---|---|
| `src/sfno_training/trainer/plasim_trainer.py` | Add a `pretrained_checkpoint_path` config knob. The warm-start load must happen **after** `super().__init__()` (line 266, which builds `self.model`, `self.optimizer`, `self.scheduler` with fresh state because EXP_DIR is empty) **and BEFORE** `EMAModel(self.model, ...)` is constructed (line 301), so the EMA shadow initialises from the loaded weights — not from random init. Insertion point: between line 266 and line 268 (the EMA cfg block). Logic: `if (not self.params.resuming) and self.params.get("pretrained_checkpoint_path"): Driver.restore_from_checkpoint(checkpoint_path=str(self.params.pretrained_checkpoint_path), model=self.model, loss=None, optimizer=None, scheduler=None, counters=None, checkpoint_mode="legacy", strict=True)`. **Use the existing `self.params.resuming` flag as the authority** — it is already computed by `src/sfno_training/train_plasim.py:232-237` from `params.checkpoint_path` and `params.experiment_dir = params.exp_dir / config / run_num` (line 216, 224). Do **not** invent a new `_resume_ckpt_exists_in_expdir(params.exp_dir)` helper — that would key off the wrong directory (`params.exp_dir` is the per-experiment root above the `/run_num/` subdirectory; the actual checkpoint lives one level deeper). Log a single `INFO` line `"warm-start: loaded weights from <path>, optimizer/scheduler/counters NOT restored"` so SLURM logs make the path unambiguous. |
| `src/sfno_training/train_plasim.py` | Add `--pretrained_checkpoint_path` CLI argument; pipe into `params["pretrained_checkpoint_path"]`. **This is the single source of truth** — no YAML placeholder for the pretrained-ckpt path (see §4.2). Echo the resolved value in the existing launch banner (the same block that prints `amp_mode`, `multistep_count`, etc.) so SLURM logs document what was loaded. |

The patch is small (<50 lines + a smoke test) and additive — it does not
touch the existing resume path. Backout: remove the new knob from the yaml.

#### 4.1.2 Smoke test (strict — must catch accidental optimizer restore)

`tests/sfno_training/test_pretrained_warmstart.py` — synthetic 1-rank
construction (no real training). The smoke test must assert ALL of:

1. **Counters fresh**: `trainer.start_epoch == 0` and `trainer.iters == 0`
   immediately after `__init__` returns. (The v11_clip raw ckpt has
   `epoch=49, iters=891 751` — accidentally restoring counters would set
   these to non-zero. Per Codex risk #1, the previously-proposed
   `initial_lr == 1e-4` check does NOT catch this because the v11_clip ckpt
   also has `initial_lr == 1e-4`.)
2. **Optimizer state empty**: `len(trainer.optimizer.state_dict()["state"]) == 0`
   BEFORE the first `optimizer.step()` call. (AdamW only populates `state`
   after a step; a non-empty `state` here is direct evidence of accidental
   restore.)
3. **Optimizer step counter is 1 after one synthetic update**: build a
   throwaway loss, call `loss.backward()`, `optimizer.step()`, then assert
   `next(iter(trainer.optimizer.state_dict()["state"].values()))["step"] ==
   1`. If we accidentally restored optimizer momenta from the v11_clip ckpt,
   this number would be ~891 752.
4. **Scheduler is fresh**: `trainer.scheduler.last_epoch == 0` (LinearLR's
   default before any step).
5. **Model weights match the pretrained ckpt** at construction time: load
   the same ckpt independently and assert `torch.allclose(
   trainer.model.state_dict()[k], loaded_ckpt["model_state"][k])` for every
   key. (Strict=True in `restore_from_checkpoint` already guarantees key/
   shape compatibility, but byte-equality is a stronger contract.)
6. **EMA shadow equals the loaded model weights at construction time**:
   `for k in ema._shadow: torch.allclose(ema._shadow[k],
   trainer.model.state_dict()[k])`. (This is the load-order check from
   Codex blocker #3. If the warm-start load ran AFTER `EMAModel(...)`, the
   shadow would carry pre-load random weights instead.)
7. **EXP_DIR guard**: a second test variant constructs the trainer with a
   non-empty `EXP_DIR/0/training_checkpoints/` directory and asserts that
   the resume path takes precedence over `pretrained_checkpoint_path`
   (the warm-start INFO log is NOT emitted; the model weights come from the
   resume ckpt, not the pretrained one).

Test uses the existing `tests/sfno_training/helpers.py` fixtures
(the 58-channel synthetic loader / temporary `EXP_DIR` helpers used by
`tests/sfno_training/test_smoke_sfno_cpu.py` and `test_ema.py`). The new
file goes at `tests/sfno_training/test_pretrained_warmstart.py` next to
its sibling tests. **Do NOT** reference `tests/_helpers/synthetic_helpers.py`
or `tests/plasim_makani_packager/synthetic_helpers.py` — the former does
not exist; the latter is a packager-test helper for synthetic HDF5
inputs, not a trainer fixture.

### 4.2 New training config

`src/sfno_training/config/plasim_sim52_zgplev_group_clone_v11_clip_warmstart.yaml`
— byte-identical copy of `plasim_sim52_zgplev_group_clone_v11_clip.yaml`
with only the yaml block name renamed to
`plasim_sim52_zgplev_group_clone_v11_clip_warmstart`.

**No `pretrained_checkpoint_path` field in the YAML.** The pretrained path
flows in via the CLI arg (§4.1.1). Reasons for CLI-only:
- The yaml is hashed for `TRAIN_SHA7` provenance; baking the path in would
  couple two unrelated concerns (recipe identity vs warm-start source).
- A `{{PRETRAINED_CKPT}}` placeholder risks shipping unrendered if anyone
  edits the submit slurm.
- The yaml is `git`-tracked; the warm-start source path is a runtime concern
  better surfaced via SLURM env + the `warmstart_provenance.txt` sidecar
  (§6.1).

### 4.3 New submit slurm

`src/sfno_training/submit_zgplev_group_clone_v11_clip_warmstart.slurm` —
mirror of `submit_zgplev_group_clone_v11_clip.slurm` with:
- `CFG_NAME = "plasim_sim52_zgplev_group_clone_v11_clip_warmstart"`.
- `EXP_DIR =
  $SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_group_clone_v11_clip_warmstart` (fresh,
  must not exist — protect_prior_runs guard kept verbatim).
- New bash variable `PRETRAINED_CKPT =
  $SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_group_clone_v11_clip/plasim_sim52_zgplev_group_clone_v11_clip/0/training_checkpoints/best_ckpt_mp0.tar`.
- `train_plasim.py` invocation gains a single new line:
  `--pretrained_checkpoint_path "$PRETRAINED_CKPT" \` immediately after the
  existing `--amp_mode bf16 \` line. **No YAML rendering, no
  `sed`/`envsubst` step** — the path is a plain CLI argument.
- Wallclock budget: same as v11_clip submit (~17 h on 4×H100 — fresh
  optimizer, so no startup speedup; 50 epochs × ~17 min/epoch ≈ 14 h
  + warmup/checkpoint overhead).

### 4.4 Preflight checks (run before submitting)

1. `ls $SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_group_clone_v11_clip_warmstart`
   must return **No such file or directory**. If it exists, refuse to launch
   (per `feedback_protect_prior_runs`).
2. `$PRETRAINED_CKPT` must exist with size > 0 (~1.7 GB for the raw-best
   form including optimizer state — we ignore the optimizer state).
3. `python -c "import torch; ckpt = torch.load('$PRETRAINED_CKPT',
   map_location='cpu', weights_only=False); assert 'model_state' in ckpt;
   print('keys:', list(ckpt.keys())); print('model_state keys:',
   len(ckpt['model_state']))"` must show 128 keys for the makani SFNO state
   (verified against the v11_clip raw-best in this session: no `module.`
   prefix; safe under strict=True for `mp0` / model-parallel-size 1; this
   plan does NOT cover multi-MP restore).
4. The smoke test in §4.1.2 must pass on a synthetic 1-rank construction
   before the first `sbatch`. The smoke test is the contract; failure
   means the patch is wrong and the run is not authorised.
5. After launch, the first training-log line must contain
   `warm-start: loaded weights from <PRETRAINED_CKPT>, optimizer/scheduler/counters NOT restored`.
   If absent, the trainer-side patch (§4.1) didn't land — kill the job and
   inspect.
6. The first `epoch 1` log line must show counters consistent with a fresh
   start (`start_epoch=0`, `iters=0` before the first batch). The loss at
   step 0 should be small (consistent with starting from already-trained
   weights, not from random init) — empirically O(10⁻³) is expected; if the
   first reported loss is O(10⁻¹) or larger, the model weights weren't
   actually loaded.

## 5. Decision gates

After 50 epochs complete, run the standard own-track NWP eval against the
v11 testset (matched, per the Pre-Run 0 / 2026-05-14 fix):

```bash
RUN_DIR="$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_group_clone_v11_clip_warmstart/plasim_sim52_zgplev_group_clone_v11_clip_warmstart/0" \
  CKPT="$RUN_DIR/training_checkpoints/best_ckpt_ema_mp0.tar" \
  MODE=nwp \
  TEST_HOLDOUT="$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full_v11/test_holdout" \
  TRAIN_DIR="$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full_v11/train" \
  PACKAGER_TEST_SRC="$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev_v11/test" \
  RUN_TAG="20260514_v11_clip_warmstart_on_v11_testset_ema_startckpt-best_ckpt_mp0" \
  scripts/submit_eval.sh
```

**Why `PACKAGER_TEST_SRC` must be set explicitly (Codex blocker #4):**
`scripts/submit_eval.sh:56` defaults `PACKAGER_TEST_SRC` to the v10 path
(`$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev/test`). It is the
source for (a) `DATA_SHA7` (read from `MOST.0121.h5`'s `packager_git_sha`
attr at line 65–73) and (b) the auto-build source for `TEST_HOLDOUT` if
that directory is missing. If we leave the default, `DATA_SHA7` would
record the v10 packager SHA in `provenance.txt`, mis-attributing the eval.
Worse, if the v11 `TEST_HOLDOUT` were ever missing, the auto-build path
would create a v10 split silently. The v11 packager source path
(`sim52_astro_64x128_zgplev_v11/test`) was verified to exist in this
session; the v11 `TEST_HOLDOUT` also exists, so the immediate metric values
would be unaffected — but provenance honesty requires the override.

**Why the `startckpt-best_ckpt_mp0` suffix in RUN_TAG (Codex medium risk #2):**
Raw-best vs EMA-best of the v11_clip baseline are different warm-start
choices. Embedding the start-ckpt flavor in the eval `RUN_TAG` and the
training EXP_DIR provenance sidecar (§6.1) makes the choice queryable
post-hoc without re-reading the SLURM logs.

### 5.1 Primary gates (the question this run was launched to answer)

| Metric | v11_clip (baseline) | 5410 (ceiling) | gate to pass |
|---|---:|---:|---|
| zg500 @ 6h RMSE | 3.525 m | 2.215 m | ≤ 2.87 m (closes ≥ 50 % of the gap) |
| tas @ 6h RMSE | 0.433 K | 0.271 K | ≤ 0.35 K (closes ≥ 50 % of the gap) |
| zg500 @ 6h ACC | 0.9985 | 0.9994 | ≥ 0.9990 |
| tas @ 6h ACC | 0.988 | 0.995 | ≥ 0.992 |

If both RMSE gates pass: **cumulative training compute confirmed** as the
dominant cause of the 6 h gap. Promote `v11_clip_warmstart` to canonical
own-track baseline; update `feedback_ema_is_canonical_ckpt` link target.

If neither gate is met: cumulative compute is **not** the dominant cause.
Field narrows to effective batch size and group-side preprocessing. Open
follow-up tickets; do not auto-launch more runs.

### 5.2 Hold-the-line gates (no regression on what already works)

| Metric | v11_clip baseline (EMA) | hold-the-line (max degradation allowed) |
|---|---:|---|
| zg500 @ 336h ACC | 0.377 | ≥ 0.36 (Δ ≤ −0.02) |
| zg500 @ 336h RMSE | 72.5 m | ≤ 76 m (≤ +5 %) |
| tas @ 336h ACC | 0.416 | ≥ 0.39 (Δ ≤ −0.03) |
| ua5 @ 336h ACC | 0.322 | ≥ 0.30 |
| ta5 @ 336h ACC | 0.332 | ≥ 0.31 |
| pr_6h @ 6h ACC | (baseline value from existing Pre-Run 0 report) | within ±0.02 |

Hold-the-line breaches **don't** automatically reject the run, but a
warm-start that improves 6 h skill while regressing long-lead skill > 5 %
RMSE is not promotable as canonical — it becomes a *parallel* recipe in
the canonical set (e.g., `v11_clip` for long-lead-priority work,
`v11_clip_warmstart` for short-lead-priority work).

## 6. Output paths

- Training run dir:
  `$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_group_clone_v11_clip_warmstart/`
- Checkpoints (auto-saved):
  - `…/training_checkpoints/best_ckpt_mp0.tar` (raw, ≈ 1.7 GB with optimizer)
  - `…/training_checkpoints/best_ckpt_ema_mp0.tar` (EMA shadow, ≈ 0.4 GB)
  - `…/training_checkpoints/ckpt_mp0_v*.tar` (rotating snapshots)
- Eval scorecard:
  `$WORK2/SFNO_Climate_Emulator/results/sfno_eval/20260514_v11_clip_warmstart_on_v11_testset_ema_startckpt-best_ckpt_mp0/`
- Final rollup (manual markdown after eval completes):
  `docs/2026-05-14_v11_clip_warmstart_results.md`

### 6.1 Provenance sidecar (NEW — addresses Codex medium risk #3)

`submit_eval.sh` writes `provenance.txt` with eval-side metadata (EVAL_SHA7,
DATA_SHA7, TRAIN_SHA7, CKPT, RUN_DIR, TEST_HOLDOUT, TRAIN_DIR) per
`scripts/submit_eval.sh:156`. It does NOT currently record any *training-side*
warm-start metadata, because warm-start is a new concept. Add a sidecar at:

`$RUN_DIR/warmstart_provenance.txt` (written by `train_plasim.py` once at
launch, in the `print_params_to_screen`-equivalent block):

```
pretrained_checkpoint_path = <absolute path to v11_clip raw-best ckpt>
pretrained_checkpoint_flavor = best_ckpt_mp0  (raw | ema)
pretrained_checkpoint_size_bytes = <stat result>
pretrained_checkpoint_sha256 = <first 16 hex chars>
warmstart_load_order = "after super().__init__, before EMAModel construction"
lr_peak = 1.0e-4
lr_schedule = "LinearWarmupCosineAnnealingLR(warmup=5 epoch, min=1e-8, T_max=45)"
max_epochs = 50
batch_size_global = 8
ema_decay = 0.999
optimizer_max_grad_norm = 32.0
input_noise_sigma = 0.05
channel_weights = constant
n_history = 0
n_future = 0
multistep_count = 1
```

Then the eval report's `Train-side provenance` section in `report.md`
(`scripts/render_eval_report.py`) reads this sidecar (if present) and
embeds it. Sidecar is optional — non-warmstart runs simply omit it.

**Plumbing for the report-side read:**
`scripts/render_eval_report.py` does not currently accept any RUN_DIR
input — its existing CLI args (`--out-root`, `--run-tag`, `--eval-sha7`,
`--data-sha7`, `--train-sha7`, `--ckpt-path`, `--report-out`,
`--metadata-json`, `--benchmark-5410-out-root`) all operate on the eval
side. `scripts/submit_eval_report.slurm:23` already has `RUN_DIR` as a
shell variable but does not pass it through. Required changes:

| File | Change |
|---|---|
| `scripts/render_eval_report.py` | Add `--run-dir` optional CLI arg (type=`Path`, default=`None`). When set, look for `<run-dir>/warmstart_provenance.txt`; if present, parse the `key = value` lines and embed them in a new `### Warm-start provenance` block under the existing `_render_provenance` section. When absent or `--run-dir` is unset, omit the block silently (it's only meaningful for warm-started runs). |
| `scripts/submit_eval_report.slurm` | Add `--run-dir "$RUN_DIR" \` to the `python scripts/render_eval_report.py` invocation at line 56, gated behind a `-n` test so non-warmstart evals (where `$RUN_DIR` may be unset) don't break. |

~30 lines in `train_plasim.py` (sidecar write), ~15 lines in
`render_eval_report.py` (sidecar read + render), ~2 lines in
`submit_eval_report.slurm`. Test coverage: extend an existing
`render_eval_report` test (or add `tests/scripts/test_render_eval_report_warmstart.py`)
that builds a synthetic `warmstart_provenance.txt`, runs the renderer with
`--run-dir`, and asserts the new block appears in the output markdown.

## 7. Open questions / deferred items

- **No multistep / no rollout2 / no channel_weighting / no noise sweep.**
  Per user 2026-05-14: this experiment is single-axis. The 2026-05-14
  zg500-targeted HPO plan's Run A (`multistep_count=2`) and Run B
  (channel-weighted loss) remain separately scoped and are **not**
  pre-authorised by this plan.
- **pr_6h units mismatch with 5410's stats** — opened as a separate ticket
  at `docs/2026-05-14_pr_6h_units_mismatch_ticket.md`. Not investigated as
  part of this experiment. Even if the units issue is resolved, it affects
  one channel and won't change the zg500 6 h conclusion.
- **Effective batch size confound (GB=32 vs GB=8).** If the warm-start does
  NOT close the gap, raising GB to 32 is the natural next single-knob test.
  Not authorised by this plan.

## 8. Cross-references

- `docs/2026-05-12_v11_clip_restore_plan.md` — defines the v11_clip baseline
  we warm-start from.
- `docs/2026-05-13_v11_clip_next_hpo_plan.md` — Pre-Run 0 (v11_clip on v11
  test set) eval that established the current 6h zg500 RMSE 3.525 m number.
- `docs/2026-05-14_zg500_targeted_hpo_plan.md` — the parent HPO plan
  proposing Run A (`multistep_count=2`) and Run B (channel weighting). This
  warm-start diagnostic is orthogonal; does not consume their compute budget.
- `docs/2026-05-10_sfno_param_count_forensic.md` — ruled out the
  capacity-gap hypothesis (real DoF matches within 0.017 %).
- `docs/2026-05-10_forcing_pipeline_numerical_diff.md` — established that
  the rsdt 6 h offset is the group's side (theirs end-of-window, ours
  correct); SST was fixed via v11 (`--sst-mode surface`).
- `makani-src/makani/utils/driver.py:354-440` — `restore_from_checkpoint`
  with `optimizer=None` is the underlying mechanism for "load weights only".
- 5410 ckpt verification (this session): `optimizer step = 218,943 ≈
  this-run iters 219,024`; `initial_lr = 1.0e-04`; current `lr = 9.504e-07`;
  no `ema_state` key.

## 9. Revision history

- **2026-05-14 (initial)** — first draft after the forensic-audit chain
  (capacity, vertical levels, SST, rsdt, normalization). Proposed
  `pretrained_checkpoint_path` via the stock makani pretrained branch,
  raw-best start, fresh optimizer, 50-epoch cosine peak 1e-4.
- **2026-05-14 (revised — post Codex read-only review)**. Four blockers
  addressed:
  1. Trainer file path corrected (`src/sfno_training/trainer/plasim_trainer.py`,
     not `src/sfno_training/trainer.py`).
  2. **Do NOT use makani's stock `pretrained=True` path.** Per
     `makani-src/makani/utils/training/deterministic_trainer.py:237` it
     passes optimizer/scheduler/counters into restore by default
     (`load_optimizer/load_scheduler/load_counters=True` in
     `makani-src/makani/utils/driver.py:137`). Accidentally restoring those
     from the v11_clip raw-best (epoch 49, iters 891 751, current lr
     1.317e-7, `initial_lr=1e-4`) would run roughly one tail epoch, not a
     fresh 50-epoch stage. Replaced with an explicit
     `Driver.restore_from_checkpoint(..., optimizer=None, scheduler=None,
     counters=None, strict=True)` call.
  3. **Warm-start load order made explicit**: must happen *after*
     `super().__init__()` (line 266) and *before* `EMAModel(self.model, ...)`
     (line 301), so the EMA shadow initialises from the loaded weights and
     not from random init.
  4. Eval command now sets `PACKAGER_TEST_SRC` to the v11 path explicitly so
     `submit_eval.sh:56` doesn't fall back to v10 for `DATA_SHA7` and
     auto-build-source provenance.

  Risks addressed:
  - The smoke test in §4.1.2 was strengthened (Codex risk #1) — no longer
    relies on `initial_lr == 1e-4` (the v11_clip ckpt has the same value).
    Now asserts `start_epoch==0`, `iters==0`, optimizer `state` empty
    pre-step, `step==1` after one synthetic update, scheduler
    `last_epoch==0`, model byte-equal to pretrained ckpt, EMA shadow
    byte-equal to loaded model weights, and EXP_DIR-resume-precedence
    semantics.
  - RUN_TAG now embeds `startckpt-best_ckpt_mp0` to surface the warm-start
    flavor choice (Codex risk #2).
  - Added §6.1 provenance sidecar so `warmstart_provenance.txt` is written
    by `train_plasim.py` and consumed by `render_eval_report.py` to embed
    the warm-start metadata in `report.md` (Codex risk #3).
  - Strict=True / single-MP safety note added to §4.4 preflight #3
    (Codex risk #4).
- **2026-05-14 (revised — post Codex second pass)**. Four small fixes:
  1. Resume guard now keys off `self.params.resuming` (computed by
     `src/sfno_training/train_plasim.py:232-237` from
     `params.checkpoint_path` and `params.experiment_dir = exp_dir /
     config / run_num`), not a custom `_resume_ckpt_exists_in_expdir`
     helper which would have checked the wrong directory.
  2. Single source of truth for the pretrained-ckpt path is the
     `--pretrained_checkpoint_path` CLI arg. The YAML carries no
     `{{PRETRAINED_CKPT}}` placeholder. Eliminates the risk of an
     unrendered placeholder shipping in the recipe.
  3. Test fixture path corrected to `tests/sfno_training/helpers.py`
     (sibling of `test_smoke_sfno_cpu.py` and `test_ema.py`). The
     previously referenced `tests/_helpers/synthetic_helpers.py` does
     not exist; `tests/plasim_makani_packager/synthetic_helpers.py`
     does exist but is a packager-test helper, not a trainer fixture.
  4. Report-side provenance read now spelled out:
     `render_eval_report.py` gets a new optional `--run-dir` CLI arg;
     `submit_eval_report.slurm` passes `--run-dir "$RUN_DIR"` (gated
     on the variable being set). Renderer parses
     `<run-dir>/warmstart_provenance.txt` and emits an optional
     `### Warm-start provenance` block in `report.md`. Tested via a
     new renderer test that builds a synthetic sidecar.
