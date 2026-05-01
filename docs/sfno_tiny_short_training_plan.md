# SFNO tiny + short training plan

> **Plan v5** (2026-04-24). Addresses Codex round-4 review of v4.
> Inherits from `docs/plasim_makani_packager_plan.md` v9 (dataset contract) and `docs/sfno_training_implementation_plan.md` v4 (trainer wrapper).
> No re-opening of locked decisions; this plan only specifies the **first three runs** after the trainer wrapper has shipped: tiny, short, post-training checks.
>
> ## Changelog
>
> **v5 — Codex round 4 fixes:**
> 1. **`one_step_eval.py` constructs its own single-step dataloader.** When evaluating a short-trained checkpoint the YAML's `valid_autoreg_steps=3` would otherwise yield a 4-target `tar` tensor. v5 explicitly overrides `params.valid_autoreg_steps = 0` when `_plasim_get_dataloader` is called from `one_step_eval.py` (see §C.3); the loop iterates once, `tar` shape is `(B, 1, 53, H, W)`, and there is no slicing-after-flatten ambiguity.
> 2. **Preflight forcing-advance assertion narrowed.** The `not torch.equal(forcing_before, forcing_after)` check is brittle: static forcing channels (`lsm`, `sg`, and `z0` over land) produce identical buffers at t=0 and t=1 by physics, not by bug. v5 drops it and keeps only the strong check, `torch.equal(forcing_after, unpredicted_tar_eval[:, 0:1])`, which confirms the buffer was correctly advanced regardless of whether the value changed.
> 3. **Changelog wording fixed.** v4 changelog said "flatten before cache_unpredicted_features"; the actual pseudocode (and the trainer at `deterministic_trainer.py:474-478`) does **cache first, then flatten**. v5 corrects the wording.
> 4. **NaN/Inf scanner reads both `out.log` and slurm stdout/stderr.** Per-epoch summaries are in `out.log`; per-batch tqdm losses go to slurm `*.out` / `*.err`. v5 adds `scripts/scan_for_nans.py` that greps both for `nan|inf|NaN|Inf` and emits a JSON summary per run; called from §C.5 triage and from `submit_*.slurm` post-training step (non-fatal — it surfaces issues, doesn't gate).
>
> **v4 — Codex round 3 fixes:**
> 1. **Preflight dry-run uses eval mode and flattens history.** `cache_unpredicted_features` branches on `self.training` (`makani-src/.../preprocessor.py:375`) — in train mode it writes `unpredicted_inp_train`, in eval mode `unpredicted_inp_eval`. The dataloader returns 5-D tensors `(B, 1, C, H, W)` that the trainer flattens **after** caching (`deterministic_trainer.py:474-478`). v4 preflight: `wrapper.eval()` first, then `cache_unpredicted_features(*batch)` (writes `unpredicted_inp_eval`, returns 5-D `(inp, tar)`), then `inp_state = preprocessor.flatten_history(inp)`, then the wrapper forward pass.
> 2. **Preflight installs a forward pre-hook to prove the wrapped model receives 58 channels.** v3 only checked the wrapper's outer interface (52→53). v4 registers a `register_forward_pre_hook` on `wrapper.model` (the underlying SFNO) that captures the input tensor and asserts `internal.shape == (B, 58, 64, 128)` — the 52 state + 6 forcing concatenation that `append_unpredicted_features` performs inside `wrapper.forward`.
> 3. **RMSE scripts denormalize explicitly.** Dataset `_get_data` z-scores both input and target via `(data − bias) / scale` (`makani-src/.../data_loader_multifiles.py:396-400`). v3 said "physical units" without specifying the conversion. v4 requires `one_step_eval.py` and `rollout_eval.py` to compute `pred_phys = pred_z * out_scale + out_bias` (and the matching transform for target / persistence input), and explicitly note that `time_means.npy` is already in physical units (it is the dataset's per-channel time mean over the train split, written by the packager). Climatology RMSE compares physical-unit predictions against physical-unit `time_means`.
> 4. **Checkpoint-load API name corrected.** No standalone `load_checkpoint` function; the method is `Driver.restore_from_checkpoint` (`makani-src/.../driver.py:348`), inherited by `Trainer`. v4 evaluator scripts instantiate a `PlasimTrainer` with `skip_training=True` (or build the wrapper directly + call `Trainer.restore_from_checkpoint`) — naming corrected throughout.
>
> **v3 — Codex round 2 fixes:**
> 1. **`valid_autoreg_steps` semantics corrected.** `_plasim_get_dataloader` passes `valid_autoreg_steps` straight through as `n_future` (`src/sfno_training/trainer/plasim_trainer.py:85`). `MultifilesDataset` then yields a target tensor with `n_future + 1` time steps, and `validate_one_epoch` (`makani-src/.../deterministic_trainer.py:620`) iterates the rollout `n_future + 1` times. So `valid_autoreg_steps=N` produces **N+1** predictions at lead times `(1..N+1) × dhours`. v3: tiny uses `valid_autoreg_steps=0` for true single-step (1 prediction at +6 h); short uses `valid_autoreg_steps=3` for 4 predictions at lead times `{6, 12, 18, 24} h`.
> 2. **Preflight forcing assertion rewritten.** `append_history(inp, pred, idt=0)` returns `x2 = pred[:, :n_state]` — the next-step **state only**, 52 channels (`makani-src/.../preprocessor.py:237`); the 58-channel input is built later by `append_unpredicted_features` inside the wrapper's forward (`makani-src/.../preprocessor.py:448`). v3 preflight instead snapshots `preprocessor.unpredicted_inp_eval` before and after `append_history`, asserts the buffer advanced by one slot of `unpredicted_tar_eval`, asserts the returned next-state has shape `(B, 52, 64, 128)`, and confirms wrapper-internal `pred.shape == (B, 53, 64, 128)`.
> 3. **Loss-curve log source corrected.** `train_plasim.py:139` redirects logger output to `EXP_DIR/out.log`; there is no `training_logs/` directory created. v3 has `plot_loss_curves.py` regex-parse `out.log` for the per-epoch lines emitted by `log_epoch` (`makani-src/.../deterministic_trainer.py:714`): `training loss: <value>`, `validation loss: <value>`.
> 4. **Rollout diagnostic fully specified.** `MultiStepWrapper.forward()` calls `_forward_eval` when `self.training==False` (`makani-src/.../stepper.py:152`); `_forward_eval` is a single-step forward (no rollout loop). The multi-step rollout only happens inside `_forward_train`. v3 specifies `rollout_eval.py` to **mimic `validate_one_epoch` line-by-line** with the eval wrapper called once per step and `append_history` between steps, with explicit pseudocode in §C.4.
> 5. **Dataset-length table marked illustrative.** `MultifilesDataset.__len__` subtracts the future offset **once globally** from the concatenated file list, not once per file (`makani-src/.../data_loader_multifiles.py:404`). v2's per-file arithmetic was off in places. v3 marks the lengths table as illustrative and **gates on measured `len(dataset)` printed by preflight** before training starts.
> 6. **Move-to-full gate hardened.** Persistence is the meaningful 6-hour baseline for `tas` and `zg5`. v3 makes "model RMSE ≤ persistence RMSE on `tas` AND `zg5`" the **only** gate to full training; "beats climatology only" is now explicitly a reason to run another short/debug run, not green light.
>
> **v2 — Codex round 1 fixes (all blockers + majors):**
> 1. **Sample-count claims dropped.** `n_train_samples_per_epoch` and `n_eval_samples` are Makani metadata only — `MultifilesDataset.__len__` (`makani-src/.../data_loader_multifiles.py:404`) returns the full sample count and `train_one_epoch` (`deterministic_trainer.py:462`) iterates the entire `train_dataloader`. Plan now derives step counts from `len(dataset) / batch_size` and pins `max_epochs` from a measured-throughput value (tiny run sets the throughput, short uses it).
> 2. **Scheduler config fixed.** `lr_warmup_steps=500` + `ReduceLROnPlateau` raises `NotImplementedError` (`makani-src/.../driver.py:701`). v2 sets `lr_warmup_steps=0` for both tiny and short.
> 3. **Rollout diagnostic moved off Makani logs.** Stock `MetricsHandler` defaults to ERA-5 channel names (`metric.py:240`) which are intersected against `params.channel_names` and produce an empty PlaSim list. v2 adds a small standalone diagnostic, `scripts/rollout_eval.py`, that mimics `validate_one_epoch`'s loop using `PlasimPreprocessor.append_history` directly. Documented as a **diagnostic**, not the production inferencer; the inference hard-gate is unchanged.
> 4. **State vs `pr_6h` loss decomposition is post-hoc.** `LossHandler.forward` (`makani-src/.../loss.py:443`) returns a single scalar; live curves can only show total loss. v2 adds `scripts/loss_decompose.py` to compute state-only and `pr_6h`-only loss components from a saved checkpoint on the valid set, run **once after training**.
> 5. **Runtime preflight added.** New `scripts/preflight.py` runs at the start of each slurm job: asserts `makani.__file__` resolves into `makani-src/`, asserts the wrapper-patched 58-channel input / 53-channel output contract on a single batch (`pred[:, :52]` feeds back, forcing concat fires correctly), and re-runs `tests/sfno_training/test_validation_rollout.py` in the live launch environment.
> 6. **Tiny success criteria rewritten with baselines.** Persistence and climatology RMSE on `tas`/`zg5`/`ua5` are computed by `scripts/one_step_eval.py`. Gate is improvement over persistence (or, weaker, climatology), not an absolute z-RMSE threshold.
> 7. **Short epoch count is concrete.** `max_epochs` is derived from tiny-throughput measurement before short launches; default `max_epochs = 8` and the slurm wallclock allows it to finish cleanly with margin. Open-ended "as many as fit in 6h" wording removed.
> 8. **Artifact names + zg5 derivation corrected.** Checkpoints write `ckpt_mp{mp_rank}_v{ckpt_version}.tar` and `best_ckpt_mp{mp_rank}.tar` (`train_plasim.py:103-107`). Channel ordering re-derived: `pl=0, tas=1, ta1..ta10=2..11, ua1..ua10=12..21, va1..va10=22..31, hus1..hus10=32..41, zg1..zg10=42..51, pr_6h=52`, so `zg5=46`, `ua5=16`.

---

## Scope

The packager has produced a structurally-validated dataset at `$SCRATCH/AI-RES/data/makani/sim52_astro_64x128/` (98 train + 20 valid + 8 test, ~331 GB). The trainer wrapper at `src/sfno_training/` is in place and gated by `submit_smoke.slurm` (1 file, 1 epoch, tiny SFNO).

This plan covers the **next three** runs only:

- **Tiny** — small training-shaped run; produces a believable downward loss curve, sets the throughput baseline used to calibrate short, and exercises the post-training-check pipeline end-to-end.
- **Short** — 5 simulated years, medium SFNO, 8 epochs, gate run before committing to a full 98-file production training.
- **Post-training checks** — first-pass loss curves, sample plots, one-step RMSE with persistence + climatology baselines, multi-step rollout sanity (4 steps), failure triage.

Inference is **out of scope**, as locked in `sfno_training_implementation_plan.md`. The rollout diagnostic added in v2 (`scripts/rollout_eval.py`) is documented as a sanity tool, not a production inferencer; the `_plasim_get_dataloader` `mode == "inference"` hard-fail is unchanged.

---

## Locked decisions (interview, 2026-04-24)

| Topic | Decision |
|---|---|
| Tiny config size | Slightly bigger than smoke (tiny SFNO arch + meaningful epoch count) |
| Tiny goal | Speed (≤30 min wallclock) |
| Tiny validation | Loss + minimal RMSE (tas + zg5) with persistence + climatology baselines; **`valid_autoreg_steps=0`** = single-step only (1 prediction at +6 h) |
| Tiny files | Train `MOST.0003.h5`, valid `MOST.0101.h5` |
| Short years | 5 train (`MOST.0003`–`0007`) + 2 valid (`MOST.0101`–`0102`) |
| Short architecture | Medium SFNO (`embed_dim=128`, `num_layers=4`, `scale_factor=3`) |
| Short compute | 1 GPU, 6 h wallclock |
| Short rollout | **`valid_autoreg_steps=3`** = 4 predictions at lead times `{6, 12, 18, 24} h` |
| Move-to-full gates | All four: smooth loss; **`tas` AND `zg5` model RMSE ≤ persistence RMSE** (no climatology fallback); finite monotonic 4-step rollout (`{6,12,18,24}h`); sample plots qualitatively right |
| `pr_6h` plotting | Loss tracked from start (total loss; decomposed post-hoc); RMSE/plots deferred until state is sane |
| Multi-step rollout sanity | Standalone `rollout_eval.py` diagnostic (not stock Inferencer) — replaces v1's "trainer log parsing" plan |
| Visualization channels | `tas`, `zg5`, `ua5` (no `hus5`) |
| Output artifacts | `loss_curves.png`, `per_channel_rmse.json`, `rollout_rmse.json`, sample PNGs |
| Dataset scope | Strictly `sim52_astro_64x128` (no boundary_dir, no auxiliary data) |
| Cluster / partition | Both runs on Stampede3 `amd-rtx` partition |
| Plan-doc location | `docs/sfno_tiny_short_training_plan.md` (this file) |

---

## Channel index reference (53-channel target order)

Re-derived from `src/sfno_training/config/plasim_sim52_baseline.yaml:36`:

| Index | Name | | Index | Name | | Index | Name |
|---:|---|---|---:|---|---|---:|---|
| 0 | pl | | 22 | va1 | | 42 | zg1 |
| 1 | **tas** | | … | … | | 43 | zg2 |
| 2 | ta1 | | 31 | va10 | | 44 | zg3 |
| … | … | | 32 | hus1 | | 45 | zg4 |
| 11 | ta10 | | … | … | | **46** | **zg5** |
| 12 | ua1 | | 41 | hus10 | | … | … |
| … | … | | | | | 51 | zg10 |
| **16** | **ua5** | | | | | 52 | pr_6h |

Used by `one_step_eval.py`, `rollout_eval.py`, and the visualizer.

---

## Part A — Tiny training plan

### A.1 Subset choice

- **Train**: `$SCRATCH/AI-RES/data/makani/sim52_astro_64x128/train/MOST.0003.h5` (1 file, T ≈ 1460 timesteps).
- **Valid**: `$SCRATCH/AI-RES/data/makani/sim52_astro_64x128/valid/MOST.0101.h5` (1 file, T ≈ 1460 timesteps).

`PlasimForcingDataset(MultifilesDataset)` loads every `.h5` under `train_data_path` / `valid_data_path`, so the subset must live in a separate directory. `scripts/build_subset_dataset.py` builds:

```
$SCRATCH/AI-RES/data/makani/sim52_tiny/
├── train/MOST.0003.h5    # symlink → ../sim52_astro_64x128/train/MOST.0003.h5
├── valid/MOST.0101.h5    # symlink → ../sim52_astro_64x128/valid/MOST.0101.h5
├── test/                 # empty (tiny does not exercise test split)
├── stats/                # symlink → ../sim52_astro_64x128/stats
├── metadata/             # symlink → ../sim52_astro_64x128/metadata
└── config/               # symlink → ../sim52_astro_64x128/config
```

Tiny uses the **full-dataset normalization** stats; this keeps tiny / short / future-full results comparable.

### A.2 GPU resources

- Cluster: Stampede3, partition `amd-rtx`.
- 1 H100 GPU, 1 node, `--gpus-per-node=1`.
- Wallclock: 30 min.
- Mixed precision: `--amp_mode bf16`.
- DDP disabled (`--disable_ddp`).

### A.3 Batch size, derived step counts, epochs

`MultifilesDataset.__len__` (`makani-src/.../data_loader_multifiles.py:404`) is `n_samples_total − dt × (n_history + n_future + toff)` — a **single global subtract**, not per-file. The exact length is whatever preflight prints from a live `len(dataset)` call; the table below is **illustrative only** to size the wallclock budget.

| Param | Value (illustrative) | Source |
|---|---|---|
| `batch_size` | 2 | tiny YAML |
| `len(train_dataset)` | ~1459 | `1 × 1460 − 1` (`n_future=0`, `toff=1`) |
| `len(valid_dataset)` | ~1459 | `1 × 1460 − 1` (`valid_autoreg_steps=0` → `n_future=0`) |
| Train batches/epoch | ~729 | `len // batch_size`, `drop_last=True` |
| Valid batches/epoch | ~729 | same |
| `max_epochs` | **3** | conservative; tiny is for behavior verification, not convergence |
| **`valid_autoreg_steps`** | **0** | **true single-step**: 1 prediction at +6 h (v3 fix; `=N` produces `N+1` lead times) |
| `lr` | 1.0e-3 | |
| `lr_warmup_steps` | **0** | (v2 fix: `ReduceLROnPlateau` rejects warmup > 0) |
| Scheduler | `ReduceLROnPlateau`, `patience=2`, `factor=0.5` | |
| `n_history` / `n_future` | 0 / 0 | locked by `_set_data_shapes` assert |

Compute upper bound order-of-magnitude: ~3 epochs × ~1500 batches × ~100 ms ≈ **7 min** on a GPU node for tiny SFNO at 64×128 bf16. Preflight's measured `len(dataset)` and ms/batch reset the estimate; short reuses tiny's measured throughput.

`n_train_samples_per_epoch` / `n_eval_samples` YAML keys are Makani metadata only — they do **not** cap iteration (the trainer iterates the full dataloader at `deterministic_trainer.py:462`).

### A.4 Checkpointing / log frequency

- `save_checkpoint: legacy` (Makani default).
- Checkpoint paths (corrected per `src/sfno_training/train_plasim.py:103-107`):
  - `EXP_DIR/.../training_checkpoints/ckpt_mp{mp_rank}_v{ckpt_version}.tar` (per-epoch / latest)
  - `EXP_DIR/.../training_checkpoints/best_ckpt_mp{mp_rank}.tar`
- `log_to_screen: True`, `verbose: True` → per-batch tqdm postfix.
- Slurm `-o logs/sfno_tiny_%j.out`, `-e logs/sfno_tiny_%j.err`.

### A.5 Output artifacts

Saved under `$SCRATCH/AI-RES/runs/sfno_tiny/plasim_sim52_tiny/0/`:

- `training_checkpoints/ckpt_mp0_v*.tar`, `training_checkpoints/best_ckpt_mp0.tar`.
- `loss_curves.png` — train + valid total loss by epoch (single panel; state vs `pr_6h` decomposition is post-hoc, not in this PNG).
- `per_channel_rmse.json` — one-step RMSE on valid set for **tas, zg5, ua5** with **persistence and climatology baselines** (one row per channel × baseline pair).
- `samples/{tas,zg5,ua5}_+6h.png` — target | prediction | diff triptych for the first 2 valid samples.
- `loss_decomposition.json` — state-only L2 and `pr_6h`-only L2 on valid (single epoch, post-hoc; written by `loss_decompose.py`).
- Run-provenance bundle in `EXP_DIR/`: rendered YAML, `git rev-parse HEAD`, slurm log copy, `preflight_log.txt`.

### A.6 Success vs failure criteria (revised)

**Success** — all must hold:

1. Slurm preflight passes (makani import + 58/53 contract + sentinel test re-run).
2. Run completes within wallclock with no NaN/Inf in train or valid loss at any logged batch.
3. Train loss strictly decreases from epoch 1 → epoch 3 (epoch-mean basis).
4. Valid loss does not increase over the run.
5. **At least one of**: model `tas` one-step RMSE ≤ persistence `tas` RMSE, or model `tas` one-step RMSE clearly below climatology `tas` RMSE (≥ 10% improvement).
6. Final checkpoint reloads cleanly via `scripts/one_step_eval.py`.

The earlier "z-RMSE < 1.5" absolute threshold is dropped (Codex round 1 finding §6).

**Failure modes & first-look diagnostics**: see §C.5.

---

## Part B — Short training plan

### B.1 Subset choice

- **Train** (5 files = 5 simulated years, ~7300 samples, ~13.5 GB raw): `MOST.0003.h5` … `MOST.0007.h5`.
- **Valid** (2 files): `MOST.0101.h5`, `MOST.0102.h5`.

Symlink-farm at `$SCRATCH/AI-RES/data/makani/sim52_short/` built by the same `scripts/build_subset_dataset.py`. Stats / metadata / config symlinked from the full dataset.

### B.2 Architecture (medium SFNO)

Diff vs `plasim_sim52_baseline.yaml` production config:

| Param | Baseline | **Short (medium)** |
|---|---|---|
| `embed_dim` | 384 | **128** |
| `num_layers` | 8 | **4** |
| `scale_factor` | 3 | 3 |
| `mlp_ratio` | 2 | 2 |
| `filter_type` | linear | linear |
| `operator_type` | dhconv | dhconv |
| All aux flags | False | False (asserted by `_set_data_shapes`) |

### B.3 Compute

- Cluster: Stampede3 `amd-rtx`, 1 GPU, 6 h wallclock.
- `batch_size = 4`, `--amp_mode bf16`, `--disable_ddp`, `num_data_workers = 2`.

### B.4 Derived step counts and epochs

The exact lengths come from preflight; the table below is **illustrative** (`__len__` subtracts `dt × (n_history + n_future + toff)` once globally over the concatenated file list, not once per file).

| Param | Value (illustrative) | Source |
|---|---|---|
| `batch_size` | 4 | short YAML |
| `len(train_dataset)` | ~7299 | `5 × 1460 − 1` (`n_future=0`, `toff=1`) |
| `len(valid_dataset)` | ~2916 | `2 × 1460 − 4` (`valid_autoreg_steps=3` → `n_future=3`, `toff=1`) |
| Train batches/epoch | ~1824 | `len // batch_size`, `drop_last=True` |
| Valid batches/epoch | ~729 | same |
| `max_epochs` | **8** (provisional) | **calibrated from tiny ms/batch with user sign-off** before launch |
| **`valid_autoreg_steps`** | **3** | **4 predictions** at lead times `{6, 12, 18, 24} h` (v3 fix: `=N` gives `N+1` outputs) |
| `lr` | 1.0e-3 | |
| `lr_warmup_steps` | **0** | (v2 fix) |
| Scheduler | `ReduceLROnPlateau`, `patience=3`, `factor=0.5` | |

**Throughput-derived `max_epochs`**:

After tiny finishes, read the average ms/batch from the tiny slurm log. Compute:

```
target_epochs = floor((wallclock_budget_seconds × 0.75) / (epoch_seconds × 1.10))
where epoch_seconds = (1823 + 729) × ms_per_batch_short / 1000
and ms_per_batch_short ≈ ms_per_batch_tiny × (medium_FLOPs / tiny_FLOPs)
```

If the resulting `target_epochs` is materially less than 8, drop to 5 or 4; if much larger, leave at 8 (this is a gate run, not a convergence run). The slurm wallclock is a **hard ceiling**; we deliberately leave headroom so the run finishes cleanly rather than being killed.

The throughput-derivation step is part of the post-tiny review — it's not done autonomously; the user signs off on `max_epochs` for short.

### B.5 Metrics / artifacts

Saved under `$SCRATCH/AI-RES/runs/sfno_short/plasim_sim52_short/0/`:

- `training_checkpoints/ckpt_mp0_v*.tar`, `training_checkpoints/best_ckpt_mp0.tar`.
- `loss_curves.png` — train + valid total loss by epoch (single panel). State vs `pr_6h` decomposition lives in `loss_decomposition.json` (post-hoc, see §C.1).
- `per_channel_rmse.json` — one-step RMSE on valid for all 53 channels with persistence + climatology baselines.
- `rollout_rmse.json` — RMSE at lead times `{6, 12, 18, 24} h` for `tas`, `zg5`, `ua5`. **Source**: `scripts/rollout_eval.py`, not the trainer log (v2 fix; see §C.4).
- `samples/{tas,zg5,ua5}_+{6,12,18,24}h.png` — target | prediction | diff triptychs for the first 2 valid samples.
- `loss_decomposition.json` — state-only and `pr_6h`-only L2 on valid (post-hoc).
- Run-provenance bundle: rendered YAML + git SHA + slurm log + `preflight_log.txt`.

### B.6 Move-to-full gates

All four must hold to justify launching a full 98-file production run:

1. **Loss curves**: train loss decreases smoothly across epochs; valid loss decreases or plateaus (does not increase); no NaN/Inf at any logged batch.
2. **One-step RMSE vs persistence** (the meaningful 6-hour baseline):
   - `tas` model RMSE ≤ persistence-baseline `tas` RMSE.
   - `zg5` model RMSE ≤ persistence-baseline `zg5` RMSE.
   - **No climatology-only fallback.** Beating climatology while failing persistence is a useful learning signal but **does not** justify scaling up — it triggers another short / debug run, not a full-training launch.
3. **4-step rollout** (from `rollout_eval.py`, lead times `{6, 12, 18, 24} h`): `tas` and `zg5` RMSE finite and monotonically non-decreasing across the 4 steps; no NaN at any step; the +24 h RMSE not catastrophic vs +6 h (e.g. ≤ 5× the +6 h RMSE on `tas`).
4. **Sample plots**: `tas` and `zg5` predictions visually resemble the target at lead times +6 h and +12 h — synoptic patterns approximately right; no banding artifacts; no constant fields; no obvious NaN holes.

If any gate fails, **do not** scale up; file artifacts and decide next action with the user via §C.5.

---

## Part C — Post-training sanity checks

Applies to **both** tiny and short runs (`pr_6h` RMSE/plots deferred per interview).

### C.1 Loss curves

- **Source**: `train_plasim.py:139` redirects the Python logger to `EXP_DIR/out.log`. Per-epoch loss lines are emitted by `Trainer.log_epoch` (`makani-src/.../deterministic_trainer.py:714`) in the format `    training loss: <value>` and `    validation loss: <value>`.
- `scripts/plot_loss_curves.py`:
  - Regex-parses `EXP_DIR/.../out.log` for those lines, in order, one (train, valid) pair per epoch.
  - Cross-references with the matching `Epoch <n> summary:` block.
  - Produces `loss_curves.png` — single panel, train + valid total loss per epoch, vertical marker on the best-checkpoint epoch.
- Tiny: 3 points each. Short: 8 (or the throughput-resolved `max_epochs`).
- **State vs `pr_6h` decomposition is post-hoc**, not live (Codex round 1 finding §4):
  - `scripts/loss_decompose.py` loads `best_ckpt_mp0.tar`, runs one validation pass, computes per-channel L2 separately for `[0..51]` (state) and `[52]` (`pr_6h`), saves `loss_decomposition.json`.

### C.2 Sample outputs

- `scripts/one_step_eval.py` writes both `per_channel_rmse.json` AND saves single-step (+6 h) sample PNGs.
- `scripts/rollout_eval.py` writes `rollout_rmse.json` AND saves multi-step sample PNGs at `+12 h, +18 h, +24 h`.
- Channels: **`tas` (1), `zg5` (46), `ua5` (16)**. (`pr_6h` deferred.)
- Sample lead times saved (filenames use lead time, not loop-step index, to avoid the off-by-one trap):
  - Tiny: `+6h` only (single-step).
  - Short: `+6h` from `one_step_eval.py`; `+12h`, `+18h`, `+24h` from `rollout_eval.py`.
- PNG layout: 1×3 panels (target | prediction | diff). Colour scale fixed to target's [2, 98] percentile. Latitude–longitude grid annotated.

### C.3 One-step sanity (with baselines)

- `scripts/one_step_eval.py`:
  - Restores the trained model via `Trainer.restore_from_checkpoint` (`makani-src/.../driver.py:348`) — instantiate `PlasimTrainer` with `skip_training=True`, then call `restore_from_checkpoint(best_ckpt_path, ...)`. There is no standalone `load_checkpoint` function.
  - **Forces `params.valid_autoreg_steps = 0` before building its dataloader** (overriding the YAML's `=3` for short). This guarantees the loader yields a single-target tensor `tar.shape == (B, 1, 53, H, W)`; the rollout loop iterates once; there is no risk of slicing a 4-target `tar` and comparing the +6 h prediction against, say, the +24 h slice. The check is mirrored in `test_one_step_eval.py`.
  - Runs the valid loader for one epoch, no autoregression.
  - **Denormalization** (required because the dataset z-scores both inputs and targets at `data_loader_multifiles.py:396-400`): for each batch,
    - `pred_phys   = pred_z   * out_scale + out_bias` (per-channel broadcast).
    - `target_phys = target_z * out_scale + out_bias`.
    - `pers_phys   = inp_z[:, c]   * in_scale[c]  + in_bias[c]` for state channel `c` (persistence "predicts" `t+1` = `t`).
    - `out_scale` / `out_bias` are loaded from `stats/global_stds.npy` / `stats/global_means.npy` (53-channel target order); `in_scale` / `in_bias` are the first 52 entries (state-only).
    - `time_means.npy` is **already in physical units** (the packager writes per-channel time means in physical units).
  - For each channel, computes (in **physical units**):
    - `model_rmse`: `sqrt(mean((pred_phys − target_phys)^2))`.
    - `persistence_rmse`: `sqrt(mean((pers_phys − target_phys[:, c])^2))` for state channels; for `pr_6h` (input has 52 state channels, not 53), persistence is undefined and the JSON entry is `null`.
    - `climatology_rmse`: `sqrt(mean((time_mean[c] − target_phys[:, c])^2))`.
  - Writes `per_channel_rmse.json` as `[{channel, units, model_rmse, persistence_rmse, climatology_rmse}, ...]`.
- Inspect order: `tas` → `zg5` → `ua5` → (eventually) `pr_6h`.
- Gate metric per Part B.6: **`model_rmse <= persistence_rmse` for `tas` AND `zg5`**, no climatology fallback. Climatology is reported as a sanity floor and to detect "model is worse than the time mean" pathology, not as a green-light criterion.

### C.4 Multi-step rollout sanity

- **Mechanism (revised in v3)**: `scripts/rollout_eval.py` — a small standalone diagnostic that **mimics `validate_one_epoch` line-by-line**. Stock `MultiStepWrapper.forward()` calls `_forward_eval` when `self.training==False` (`makani-src/.../stepper.py:152`), and `_forward_eval` is a single-step forward — the rollout loop only exists in `_forward_train`. So the diagnostic must drive the rollout itself.
- Pseudocode (matches `deterministic_trainer.py:597-661`):

  ```python
  # eval mode
  model_eval.eval()
  preprocessor = model_eval.preprocessor       # PlasimPreprocessor instance
  for batch in valid_dataloader:
      data_tuple = tuple(t.to(device) for t in batch)
      inp, tar = preprocessor.cache_unpredicted_features(*data_tuple)
      inp = preprocessor.flatten_history(inp)
      tarlist = torch.split(tar, 1, dim=1)     # length n_future + 1 = 4 for short
      inpt = inp                               # (B, 52, 64, 128)
      for idt, targ in enumerate(tarlist):     # idt = 0, 1, 2, 3
          targ = preprocessor.flatten_history(targ)              # (B, 53, 64, 128)
          pred = model_eval(inpt)                                # (B, 53, 64, 128)
          rmse_acc[idt][channel] += rmse_phys(pred, targ, channel)
          inpt = preprocessor.append_history(inpt, pred, idt)    # state-only feedback,
                                                                 # advances cached forcing
  ```

- `model_eval` is the wrapper restored via `Trainer.restore_from_checkpoint` (the one Makani built — `PlasimSingleStepWrapper`). The wrapper internally calls `append_unpredicted_features` to concatenate the 6 prescribed forcing channels onto the 52 state channels before the model, and strips them out of the 53-channel target (the `pr_6h` slot in `pred[:, 52:53]` is **never** part of `inpt[:, 0:52]` — `PlasimPreprocessor.append_history` enforces this via the existing shape gate).
- Denormalization (same convention as `one_step_eval.py`): `pred_phys = pred_z * out_scale + out_bias`, `target_phys = target_z * out_scale + out_bias`. RMSE values reported in **physical units**; an additional `rmse_normalized` column is reported for cross-checking against the live trainer loss.
- Lead-time labelling: `idt=0` → `+6 h`, `idt=1` → `+12 h`, `idt=2` → `+18 h`, `idt=3` → `+24 h`. Filenames and JSON keys use the lead time (e.g. `tas_+12h.png`), not the loop index.
- Writes `rollout_rmse.json` as `[{lead_time_hours, channel, units, rmse_physical, rmse_normalized}, ...]`.
- Asserts (hard-fail if violated, written into the JSON's `errors` field):
  - All RMSE values finite.
  - Per-channel RMSE monotonically non-decreasing across `+6 h` → `+24 h` (small jitter at the noise floor allowed).
- **Inference-out-of-scope statement**: `rollout_eval.py` carries a docstring header stating it is a sanity-check diagnostic only, is not the production inferencer, and must not be used for downstream emulator runs or scoring. The `_plasim_get_dataloader` `mode == "inference"` AssertionError remains the actual production hard-gate.

### C.5 Failure triage

**If preflight fails (either run):**

Preflight (`scripts/preflight.py`, see §Implementation deliverables) does FIVE things in order. Note that v3 corrects v2's broken forcing assertion: `append_history(inp, pred, 0)` returns the next-step **state only** (52 channels) at `n_history=0` — the 58-channel input is constructed inside the wrapper's forward pass. So the forcing-prescribed check operates on `preprocessor.unpredicted_inp_eval` (the cached forcing buffer), not on the return of `append_history`.

1. **Makani import path.** `python -c 'import makani; print(makani.__file__)'` → assert path contains `makani-src/makani` (catches PyPI wheel drift back to the unfixed `cache_unpredicted_features` clone).
2. **Re-run rollout sentinel tests.** `pytest tests/sfno_training/test_validation_rollout.py tests/sfno_training/test_wrappers.py -v -x` in the launch venv.
3. **Single-batch contract dry-run.** Build the dataloader on the actual subset directory in **eval mode** (so `cache_unpredicted_features` writes the `_eval` buffers, per `preprocessor.py:375`); load one batch; build a freshly-instantiated wrapper (no checkpoint needed for shape/feedback assertions). The dataloader yields 5-D tensors `(B, 1, C, H, W)` — preflight must flatten history before the wrapper, exactly mirroring `train_one_epoch` / `validate_one_epoch`:

   ```python
   wrapper.eval()                                                    # → preprocessor.training = False
   preprocessor = wrapper.preprocessor

   internal_inputs = []
   def _capture(_module, args):                                       # forward pre-hook
       internal_inputs.append(args[0].detach())
   handle = wrapper.model.register_forward_pre_hook(_capture)

   inp5d, tar5d, *rest = batch                                       # 5-D from dataloader
   inp5d, tar5d = preprocessor.cache_unpredicted_features(*batch)    # populates unpredicted_inp_eval
   inp_state = preprocessor.flatten_history(inp5d)                   # → (B, 52, 64, 128)

   forcing_before = preprocessor.unpredicted_inp_eval.clone()
   pred = wrapper(inp_state)                                         # eval-mode single forward
   handle.remove()
   ```

   Assertions:
   - `inp_state.shape == (B, 52, 64, 128)` (post-flatten state-only).
   - `pred.shape == (B, 53, 64, 128)` (52 state + 1 `pr_6h`).
   - `internal_inputs[0].shape == (B, 58, 64, 128)` — **the wrapped SFNO actually received 58 channels** (52 state + 6 forcing concatenated by `append_unpredicted_features`). This is the strong contract proof.
   - `next_state = preprocessor.append_history(inp_state, pred, idt=0)` → `next_state.shape == (B, 52, 64, 128)` (state-only return at `n_history=0`; forcing concat happens later inside the next wrapper forward).
   - `torch.equal(next_state, pred[:, :52])` — `pr_6h` (channel 52) is excluded from feedback.
   - Snapshot `forcing_after = preprocessor.unpredicted_inp_eval.clone()` and assert:
     - `torch.equal(forcing_after, preprocessor.unpredicted_tar_eval[:, 0:1])` — buffer was advanced to the +6 h target's forcing (truth, not prediction). This is the strong check; it holds regardless of whether the forcing value changed (static channels like `lsm`, `sg`, and `z0` over land legitimately produce identical buffers at consecutive steps).
     - The earlier `not torch.equal(forcing_before, forcing_after)` check is dropped (v5; could spuriously fail on a static-forcing-only batch).
4. **Print resolved sizes.** Log `len(train_dataset)`, `len(valid_dataset)`, `len(train_dataloader)`, `len(valid_dataloader)`, `max_epochs`, and resolved YAML keys (so the run record carries the actual numbers, not just the planned ones).
5. **YAML diff vs template.** Diff the rendered YAML against the template; fail on unexpected substitutions.

If any step fails, slurm exits before training is launched and no GPU-hours are burned. Output is captured in `EXP_DIR/preflight_log.txt`.

**If tiny training fails (preflight passed):**

1. **Smoke regression first.** Re-run `submit_smoke.slurm`. If smoke now also fails, the regression is environmental — fix that before debugging tiny.
2. **NaN/Inf scan.** Run `scripts/scan_for_nans.py` on `EXP_DIR/.../out.log` AND the slurm `*.out` / `*.err` files. Per-batch tqdm losses go to slurm stdout, not `out.log` — the scan must cover both. The emitted `nan_scan.json` shows the first batch where divergence enters the loss stream.
3. **Per-batch verbose log.** Tiny is small enough that all batches' losses fit in stdout; cross-reference `nan_scan.json`'s first-occurrence index against the slurm log to find the divergence point.
4. **AMP off.** Re-run with `--amp_mode none` to isolate AMP overflow.
5. **Data integrity.** `h5ls -r` on the two subset symlinks; verify shapes `(T,52,64,128) / (T,1,64,128) / (T,6,64,128)`; verify `/timestamp` strictly monotone.
6. **Stats finiteness.** `numpy.load(...) → isfinite().all()` on every `.npy` in `stats/`.
7. **Aux-feature flags.** Confirm `_set_data_shapes` asserts pass (would crash at startup; if not, Makani version drift — re-pin via `pip install --no-deps -e makani-src`).

**If short training fails (tiny passed):**

Collect, **before changing any code**:

- Slurm stdout/stderr.
- Final checkpoint (or last partial checkpoint).
- `out.log` and per-epoch summary blocks within it.
- `preflight_log.txt`.
- `nvidia-smi` output for OOM (slurm prologue captures this).
- Rendered YAML and git SHA.

Then triage:

1. Compare loss curves vs tiny. If short diverges where tiny converged → capacity, data variance, or scheduler issue.
2. Verify normalization stationarity: compute per-file mean/std on the 5 short train files; compare against full-dataset stats. Large drift would mean stats need to be local for the short subset (unlikely given `0003-0007` is contiguous).
3. Disk IO check: 5 × 2.7 GB ≈ 14 GB. If there is no node-local staging on the selected GPU node, IO contention may explain slow training (does not explain divergence).
4. `valid_autoreg_steps=3` memory (4-output rollout): medium SFNO + 4 stored predictions + bf16 is expected to fit on the target GPU; if OOM, drop `batch_size` to 2 before suspecting the model.
5. **One root-cause-fix-and-rerun cycle, max.** Don't tweak architecture, lr, or batch size in the same iteration as a data fix.

---

## Implementation deliverables

### New files

| Path | Role |
|---|---|
| `src/sfno_training/config/plasim_sim52_tiny.yaml` | Tiny SFNO arch, 2-file split, `max_epochs=3`, `batch_size=2`, `lr_warmup_steps=0`, **`valid_autoreg_steps=0`** (single-step, +6 h only) |
| `src/sfno_training/config/plasim_sim52_short.yaml` | Medium SFNO arch (`embed_dim=128`, `num_layers=4`), 5+2 file split, `max_epochs=8` (overridable from tiny throughput), `batch_size=4`, `lr_warmup_steps=0`, **`valid_autoreg_steps=3`** (4 leads at `{6,12,18,24} h`) |
| `src/sfno_training/submit_tiny.slurm` | `amd-rtx` partition, 30 min, 1 GPU, `--disable_ddp`. Calls preflight → train. Points at `sim52_tiny/`. |
| `src/sfno_training/submit_short.slurm` | `amd-rtx` partition, 6 h, 1 GPU, `--disable_ddp`. Calls preflight → train. Points at `sim52_short/`. |
| `scripts/build_subset_dataset.py` | Symlink-farm builder. Args: `--src`, `--dst`, `--train-years` (year/range), `--valid-years` (year/range). Symlinks files into `train/`, `valid/`; symlinks `stats/`, `metadata/`, `config/` directories from `--src`. Idempotent. |
| `scripts/preflight.py` | Runs four checks before training (makani import path + sentinel test + 58/53 contract dry-run + config diff). Exits non-zero on any failure. Logs to `EXP_DIR/preflight_log.txt`. |
| `scripts/one_step_eval.py` | Loads checkpoint, computes per-channel one-step RMSE on valid with **persistence + climatology baselines**, saves `per_channel_rmse.json` and sample triptych PNGs for `tas`/`zg5`/`ua5`. |
| `scripts/rollout_eval.py` | Standalone 4-step rollout diagnostic. Uses `PlasimForcingDataset` + `PlasimPreprocessor.append_history`. Saves `rollout_rmse.json`. **Documented as diagnostic, not production inferencer**. |
| `scripts/loss_decompose.py` | Post-hoc state-vs-`pr_6h` loss decomposition on valid. Saves `loss_decomposition.json`. |
| `scripts/plot_loss_curves.py` | Regex-parses `EXP_DIR/.../out.log` for `training loss:` / `validation loss:` per-epoch lines, writes `loss_curves.png`. Single panel (total loss); decomposition is in JSON, not in the PNG. |
| `scripts/scan_for_nans.py` | Greps `EXP_DIR/.../out.log` AND `logs/sfno_*_<jobid>.{out,err}` for `nan\|inf\|NaN\|Inf`, emits `nan_scan.json` with line counts and first-occurrence context. Non-fatal — surfaces issues, does not gate. Called from `submit_*.slurm` post-training and from §C.5 triage. |

### Modified files

- `src/sfno_training/trainer/plasim_trainer.py` — **NO changes** to wrapper logic (the locked patch surface). Only changes to consume the new `tiny`/`short` configs (which is just YAML — no Python edits).
- `makani/` core — **NO changes** (zero-edit invariant).

### Tests (CPU-fast, follow `tests/sfno_training/` pattern)

| Path | Coverage |
|---|---|
| `tests/sfno_training/test_build_subset_dataset.py` | Symlink farm has correct structure; resolves to expected source files; idempotent re-run. |
| `tests/sfno_training/test_preflight.py` | With CI's `RecordingDummyModel`, preflight passes on a valid setup; fails on a forced makani-path drift; fails on a forced channel-count breach. |
| `tests/sfno_training/test_one_step_eval.py` | With `RecordingDummyModel` checkpoint, `one_step_eval` produces a finite-valued JSON with persistence and climatology columns and saves three PNGs of expected dimensions. |
| `tests/sfno_training/test_rollout_eval.py` | With `RecordingDummyModel`, 4-step rollout produces finite RMSE per step; `pr_6h` is asserted out of feedback; forcing is asserted prescribed from truth. |
| `tests/sfno_training/test_loss_decompose.py` | With synthetic checkpoint, decomposition JSON has `state_l2` + `pr_6h_l2` keys with finite values. |
| `tests/sfno_training/test_plot_loss_curves.py` | Synthetic `out.log` text fixture → PNG of expected size, no exceptions; regex correctly extracts N (train, valid) pairs for N epochs. |

### CI hooks

- All new tests added to `tests/sfno_training/` are picked up by the existing pytest collection.
- `submit_smoke.slurm` regression remains the **hard gate before any tiny or short launch**.

---

## Execution order (after plan approval)

1. **Implement** symlink builder + preflight + tiny YAML + tiny SLURM + their tests; commit (chunk 1).
2. **Implement** `one_step_eval.py` + `rollout_eval.py` + `loss_decompose.py` + `plot_loss_curves.py` + tests; commit (chunk 2).
3. **Re-run smoke** (`submit_smoke.slurm`) as the existing hard-gate regression. Must pass before tiny.
4. **Build** `$SCRATCH/AI-RES/data/makani/sim52_tiny/` via the symlink builder.
5. **Launch tiny** on `amd-rtx`, 30 min. Slurm runs preflight first; training only starts after preflight passes.
6. **Run §C scripts** on the tiny output: `plot_loss_curves`, `one_step_eval`, `loss_decompose`. (No `rollout_eval` for tiny — `valid_autoreg_steps=0`, single-step only.)
7. **Review tiny artifacts with the user**, including measured ms/batch. If success criteria pass → continue. If not → §C.5 triage.
8. **Calibrate short `max_epochs`** from tiny throughput; user signs off on the resolved number (default 8).
9. **Implement** short YAML + short SLURM; commit (chunk 3).
10. **Build** `$SCRATCH/AI-RES/data/makani/sim52_short/`.
11. **Launch short** on `amd-rtx`, 6 h. Same preflight-then-train pattern.
12. **Run §C scripts** on the short output: `plot_loss_curves`, `one_step_eval`, `rollout_eval`, `loss_decompose`.
13. **Gate decision**: if all four §B.6 gates pass → user approves a full 98-file run (separate plan). If not → §C.5 triage.

No production-arch full run is in scope of this plan.

---

## Out of scope (explicit non-goals for this plan)

- Inference / scoring / long rollouts > 4 steps. Stays blocked at `_plasim_get_dataloader` until `src/sfno_inference/` ships.
- Hyperparameter search.
- Multi-GPU DDP runs (deferred to full production run, separate plan).
- W&B or external logging integration.
- `pr_6h` sample-plot or RMSE evaluation (deferred until state channels look sane).
- Boundary-dir / external forcing source switch.
- Any edits to `src/sfno_training/` Python code beyond YAML configs.
- Any edits to `makani/` core.
- Live state-vs-`pr_6h` loss curves (post-hoc only — `LossHandler` returns one scalar).
- Live PlaSim per-channel RMSE in trainer logs (`MetricsHandler` defaults to ERA-5 names; not patched).
