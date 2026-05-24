---
name: train-sfno-hpo
description: Practical config guide for SFNO emulator training on the v10 zgplev track (`plasim_sim52_zgplev_*.yaml` configs). When you open one of these YAMLs, this skill tells you what each parameter does, which are tunable / derived / fixed-by-convention / contract, what depends on what, and how to validate a change. Includes a parameter reference for `batch_size`, `lr`, `weight_decay`, `lr_warmup_steps`, `lr_start`, `scheduler`, `scheduler_T_max`, `scheduler_min_lr`, `max_epochs`, `n_train_samples_per_epoch`, `n_eval_samples`, `valid_autoreg_steps`, the `ema` block, `num_data_workers`, `prefetch_factor`, `persistent_workers`, and the CLI overlays (`--amp_mode`, `--checkpointing_level`, `--multistep_count`, `--batch_size`). Active decision right now: GB=16 vs GB=32 (GB=32 in the live YAML cleared microbench but showed accuracy degradation; LR is derived from GB via the sqrt rule). Sweep procedure (Tier A/B/C + I2 microbench) and the full Fixed / Fixed-by-tier knob inventory live in `docs/2026-05-08_hpo_knob_inventory.md`. Sibling skill `sfno-training` owns the architecture / patch surface / channel contract.
---

# train-sfno-hpo — config guide for `plasim_sim52_zgplev_*.yaml`

## When to use this skill

- Opening a `plasim_sim52_zgplev_*.yaml` and needing to know what a
  parameter does or whether it's safe to change.
- Designing or running an HPO sweep on any of the parameters in the
  reference below.
- Choosing CLI overlays at launch (`--amp_mode`, `--checkpointing_level`,
  `--multistep_count`, `--batch_size`, `--skip_validation`,
  `--save_checkpoint`).
- Diagnosing whether a regression is quality-bound (val curve / loss /
  EMA-val) or systems-bound (samples/sec / OOM / NCCL).

## When NOT to use this skill

- Architecture / patch surface / channel contract / aux-feature flags /
  Makani version pin / Python 3.12 timedelta shim →
  `skills/sfno-training/SKILL.md`.
- Inference / scoring / long-rollout evaluation → `eval-sfno-own` /
  `eval-sfno-5410`.
- Dataset packaging → `skills/plasim-makani-packager/SKILL.md`.

## Where the configs live

| File | Role |
| --- | --- |
| `src/sfno_training/config/plasim_sim52_zgplev_full.yaml` | **Production**; full architecture, full schedule. Authoritative for production GB / LR. |
| `src/sfno_training/config/plasim_sim52_zgplev_baseline.yaml` | Group-convention baseline (`scale_factor 3, embed_dim 384, num_layers 8`). |
| `src/sfno_training/config/plasim_sim52_zgplev_short.yaml` | Tier-A short architecture (`scale_factor 3, embed_dim 128, num_layers 4`). |
| `src/sfno_training/config/plasim_sim52_zgplev_{tiny,smoke}.yaml` | Single-task / CPU-feasible debug configs. |
| `src/sfno_training/submit_zgplev_full.slurm` | Production launch (DDP); CLI overlays passed here. |
| `src/sfno_training/submit_zgplev_full_microbench.slurm` | I2 systems-feasibility filter (2 epochs, no val, no ckpt). |
| `src/sfno_training/submit_zgplev_short_ddp.slurm` | Tier-A short-config sweep harness. |
| `src/sfno_training/train_plasim.py` | CLI overlay (`:200-202` resolves `--batch_size`); batch resolver `_resolve_batch_sizes` (`:38-57`); launch summary (`:67+`). |
| `makani-src/makani/utils/argument_parser.py` | Full Makani CLI surface (out-of-scope flags too). |

`{{OUTPUT_ROOT}}` and `{{EXP_DIR}}` placeholders in the YAML are
substituted by the submit script at launch.

## Status legend

- **Tunable** — designed for HPO; change is allowed under the validation
  procedure for that param's group.
- **Derived** — computed from other params (`scheduler_T_max =
  max_epochs - lr_warmup_steps`; `lr ∝ sqrt(GB)` against the anchor).
  Recompute; don't edit independently.
- **Fixed-by-convention** — group default or empirical anchor; sweep only
  with a `docs/YYYY-MM-DD_*.md` plan justifying it.
- **Contract** — never change; breaks data shape, channel count, or
  PlasimTrainer asserts. Owned by sibling skill `sfno-training`.

A larger taxonomy (`Active` / `Deferred` / `Ablation-only` / `Fixed by
convention` / `Fixed-by-tier`) is used in the inventory doc when ranking
*which* tunable knob is on the critical path. For day-to-day "is it safe
to touch" the four labels above are enough.

## Parameter reference

All line numbers reference `plasim_sim52_zgplev_full.yaml` unless noted.

### Batch & learning rate (YAML lines 107-131, 133-136)

| Param | Status | Current | What it does · key dependency · how to validate |
| --- | --- | --- | --- |
| `batch_size` | **Tunable** | 32 (line 117 — historical, **not recommended**) | GLOBAL batch (Makani convention). `train_plasim._resolve_batch_sizes` (`:38-57`) divides by `data_parallel_size` to get per-rank. **Must be a multiple of `data_parallel_size`.** Drives `lr` via the sqrt rule. **Validate**: I2 microbench for feasibility (OOM / mem / throughput) → Tier B for the quality decision. |
| `lr` | **Derived** | 2.83e-4 (line 114) = `sqrt(32/4) · 1e-4` | Peak LR. Tracks `batch_size` via the sqrt rule against the provisional anchor `lr=1e-4 @ GB=4`. **Don't edit independently** — recompute when `batch_size` changes; re-anchor only when arch / dataset change. **Validate**: paired with the GB candidate on Tier B. |
| `weight_decay` | Tunable | 3.0e-6 (line 115) | AdamW decoupled weight decay. **Trigger to sweep**: persistent train-val gap or train plateau at high val. Sweep range `{0, 1e-6, 3e-6, 1e-5}`. **Validate**: Tier B; watch over-regularization signs (val<train, train plateau). |
| `lr_warmup_steps` | Tunable | 5 (line 129) | Number of LinearLR warmup epochs before cosine. Drives `scheduler_T_max = max_epochs - lr_warmup_steps`. **Must be 0 for `scheduler: ReduceLROnPlateau`** (`makani-src/.../driver.py:701` raises `NotImplementedError`). **Validate**: Tier B; watch early-epoch loss / grad-norm spikes. |
| `lr_start` | Fixed-by-convention | 1.0e-4 (line 130) | LinearLR `start_factor`. Matches the group's `warmup_start_lr/lr = 1e-8/1e-4` ratio; PyTorch rejects `0.0`. Don't sweep. |
| `optimizer_beta1` / `_beta2` | Fixed-by-convention / Tunable | 0.9 / 0.95 (lines 134-135) | AdamW betas. β2 sweep `{0.95, 0.99}` only on a late-training instability signal. |
| `optimizer_max_grad_norm` | Tunable | 32 (line 136) | Gradient-norm clip. **Trigger to sweep**: clip consistently active in Makani logs (signals LR too high or warmup too short). Sweep `{1, 8, 32}`. **Validate**: Tier B; check Makani-logged clipped-fraction. |

### Scheduler (YAML lines 116, 123-128)

| Param | Status | Current | What it does · key dependency · how to validate |
| --- | --- | --- | --- |
| `scheduler` | Fixed-by-convention | "CosineAnnealingLR" (line 126) | Cosine decay after LinearLR warmup. Switching to ReduceLROnPlateau requires zeroing `lr_warmup_steps` first. |
| `scheduler_T_max` | **Derived** | 45 (line 127) = `max_epochs - lr_warmup_steps` (50 - 5) | Cosine period in epochs. **Recompute on every change to `max_epochs` or `lr_warmup_steps`** — manual edit is a bug-magnet (cosine decays past zero or stops short). |
| `scheduler_min_lr` | Fixed-by-convention | 1.0e-8 (line 128) | Cosine `eta_min`. Group convention; don't sweep. |
| `max_epochs` | Fixed-by-convention | 50 (line 116) | Schedule length. **Changing this is a schedule re-design, not HPO** — re-derive `scheduler_T_max`, re-anchor LR target only if the schedule shape changes meaningfully. |
| `scheduler_factor` / `_patience` / `_step_size` / `_gamma` | Inert | (unset) | ReduceLROnPlateau-only; ignored under `CosineAnnealingLR`. |

### Sampling & validation

| Param | Status | Current | What it does · key dependency · how to validate |
| --- | --- | --- | --- |
| `n_train_samples_per_epoch` | Fixed-by-convention | unset → full train set | Sub-epoch sampling cap. **Production keeps this unset.** **Microbench overlays it to `1500 × GB`** (`submit_zgplev_full_microbench.slurm:101-134`) for clean step-time measurement — don't bake that into the production YAML. |
| `n_eval_samples` | Fixed-by-convention | unset (defaults: 512 / 1024 in tier configs) | Validation sample budget. Affects val cost only, not train cost. |
| `valid_autoreg_steps` | Tunable | 3 (line 120 — 24-h rollout) | Number of autoregressive rollout steps inside validation. **Doesn't affect train cost.** **Trigger to sweep**: rollout-skill question or val-cost budget pressure. Sweep `{0, 1, 3, 5}`. **Validate**: val curve at lead 6/12/18/24h on Tier A or B. |

### EMA (YAML lines 165-176, 185-187)

| Param | Status | Current | What it does · key dependency · how to validate |
| --- | --- | --- | --- |
| `ema.enabled` | Fixed-by-convention | True (line 171) | Toggles Karras-style EMA over model weights. Group convention; flipping is an ablation. Scoped to legacy save/load (PlasimTrainer hard-errors on "flexible"). |
| `ema.decay` | Ablation-only | 0.999 (line 172) | EMA decay (Karras-warmup-clamped: `min(decay, (1+t)/(10+t))`). Sweep `{0.99, 0.999, 0.9999}` as a one-off. **Mid-run change requires `allow_config_change: True` for one launch** (line 176; resume rejects mismatched `ema_config` otherwise). |
| `ema.warmup` | Fixed-by-convention | True (line 173) | Karras warmup on. Resume rejects mismatch — same `allow_config_change` caveat as `ema.decay`. |
| `ema_validation_period` | Tunable | 1 (line 187) | EMA validation cadence in epochs. **Trigger to sweep**: val cost dominates wallclock. Phase-1 plan flagged `5` as a follow-up. **Validate**: Tier B; watch val cost vs val signal density. |

### DataLoader (YAML lines 138, 185-186)

| Param | Status | Current | What it does · key dependency · how to validate |
| --- | --- | --- | --- |
| `num_data_workers` | Fixed-by-convention | 4 (line 138) | Per-rank DataLoader workers. **Trigger to sweep**: CPU-bound symptoms (low GPU util, samples/sec ceiling). Sweep `{0, 2, 4, 8}`. **Validate**: I2 microbench. |
| `prefetch_factor` | Fixed-by-convention | 4 (line 185) | Per-worker prefetch depth. **Only meaningful when `num_data_workers > 0`.** Inert at the current value; sweep `{2, 4, 8}` only on throughput pressure. **Validate**: I2 microbench. |
| `persistent_workers` | Fixed-by-convention | True (line 186) | Keeps workers alive across epoch boundaries. **Only meaningful when `num_data_workers > 0`.** **Validate**: I2 microbench; watch epoch-boundary stalls. |

### CLI overlays — set in the SLURM script, not the YAML

Production passes these in `submit_zgplev_full.slurm:138-140`. Microbench
adds `--skip_validation` and `--save_checkpoint none` for a noise-free
timing window. Full Makani CLI surface in
`makani-src/makani/utils/argument_parser.py`.

| Flag | Status | Production | What it does · key dependency · how to validate |
| --- | --- | --- | --- |
| `--amp_mode` | Fixed-by-convention | bf16 | Mixed-precision mode. fp16 needs a grad-scaler audit; "none" is debug-only. **Validate** any change: I2 microbench. |
| `--checkpointing_level` | Fixed-by-convention | 2 | Activation checkpointing aggressiveness; group convention 2. Lower = faster + more memory; higher = slower + less memory. **Validate**: I2 microbench. |
| `--multistep_count` | Tunable | 1 | Rollout-training step count (single-step or N-step training). **`LossHandler.multistep_weight` is frozen at construction** — re-instantiate per value (plan v9 §"Do NOT"). **Validate**: Tier B (val loss + train cost). |
| `--batch_size` | Override | (unset; YAML wins) | CLI override for GLOBAL batch. `train_plasim.py:200-201` only takes effect when `> 0`. **Production submit deliberately does not pass it** so the YAML is the single source of truth. **Don't re-introduce it** on `submit_zgplev_full.slurm`. |
| `--skip_validation` | Microbench-only | off (production); **on** (microbench) | Skips the validation loop. Microbench-only — never on production. |
| `--save_checkpoint` | Microbench-only | "legacy" (production); "none" (microbench) | Checkpoint write mode. Microbench disables to keep timings clean. |

## Active decision right now: GB=16 vs GB=32

The 2026-05-08 throughput fix shipped `batch_size: 32` on the strength of
the I2 microbench, but subsequent runs at GB=32 showed accuracy
degradation. The actual question is **GB=16 vs GB=32** on a Tier-B
truncated full run. LR is **derived**, not independently tuned: GB=16 →
`lr=2.0e-4`, GB=32 → `lr=2.83e-4`, both via the sqrt rule against the
provisional anchor (`lr=1e-4 @ GB=4`). See `project_zgplev_gb_decision`.

## Validating a change — sweep tiers

**One knob per sweep, fresh `EXP_DIR` per point, sentinel-guarded.**
Quality questions go through Tier A → Tier B → Tier C; systems-feasibility
questions go through the I2 microbench, which runs in parallel and gates
entry to Tier B.

- **Tier A — short-config sanity** (`scripts/run_zgplev_short_ddp_sweep.sh`
  → `submit_zgplev_short_ddp.slurm`). Cheap pathology check; **never
  decision-grade for production** — short arch differs. Pass: ≥ 2 epochs,
  no NCCL hang, no NaN.
- **Tier B — truncated full** (the actual decision). Production arch, real
  schedule shape, ~10 decision-grade epochs in a fresh `EXP_DIR` under
  `$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_hpo/<knob>_<value>/`. Hand-roll the
  YAML overlay (template:
  `submit_zgplev_full_microbench.slurm:101-134`); change exactly one knob;
  match `scheduler_T_max` to the truncated `max_epochs - lr_warmup_steps`
  so the cosine shape mirrors production at the same epoch index. Pick
  the survivor with the best val-loss / val-ACC trajectory; ties favour
  the more conservative knob (smaller GB, lower LR, more EMA decay).
- **Tier C — full-schedule confirmation** of the single Tier-B winner in
  another fresh `EXP_DIR`. Never sweep at Tier C. If Tier C diverges from
  Tier B, return to Tier B with a wider grid or longer truncated window.
- **I2 microbench** (`submit_zgplev_full_microbench.slurm`): production
  arch, 2 epochs (epoch 1 warmup, epoch 2 measured),
  `--skip_validation --save_checkpoint none`,
  `n_train_samples_per_epoch = 1500 × GB`. Pass
  (`submit_zgplev_full_microbench.slurm:28-31`): no OOM / NaN / NCCL hang;
  peak mem ≤ 65 GB on H100; samples/sec ≥ 1.7× the GB=4 baseline. Launch:
  `GB=<n> sbatch src/sfno_training/submit_zgplev_full_microbench.slurm`.
  **A pass tells you a candidate _can_ run, not whether it _should_** —
  GB=32 cleared microbench and failed Tier B.

## Promotion + recording

A candidate goes to live production YAML only after, in order: I2
feasibility (for systems-relevant changes), optional Tier A, **Tier B
(decision)**, **Tier C (single-candidate confirmation)**, and the
recording convention.

Every promotion records the change in three places (pattern from the
2026-05-05 / 2026-05-08 throughput fix):

- **Dated `docs/YYYY-MM-DD_<topic>_plan.md` + `_resolution.md` pair**
  (auto-memory `feedback_plan_to_docs`); plan first, resolution after.
- **Inline YAML comment** above the changed line citing the doc and the
  empirical numbers — inline-comment-only updates are insufficient.
- **Commit chain** referencing the doc — `HPO X (1/N)` … `HPO X (N/N)`.

## Pitfalls

- **The live YAML `batch_size: 32` / `lr: 2.83e-4` is not the
  recommendation** — historical throughput-fix setting; later runs showed
  accuracy degradation (`project_zgplev_gb_decision`).
- **Microbench pass ≠ accuracy decision.** GB=32 cleared microbench and
  failed Tier B.
- **Don't sweep multiple knobs in one job.** Single-knob preserves
  attribution; interacting knobs run sequentially with the prior winner
  pinned.
- **Don't reuse a sweep `EXP_DIR` and don't run a sweep in the live
  production `EXP_DIR`.** Auto-resume (`train_plasim.py:171-176`) silently
  warm-starts and invalidates the measurement. Sentinel guards on the I1
  wrapper and I2 SLURM exist for this — don't disable them.
- **Don't measure samples/sec from epoch 1.** JIT + cudnn-autotune +
  DataLoader prefetch warmup distorts. Epoch 2 is the measurement.
- **Don't change `lr` without recomputing it** against the sqrt rule.
- **Don't change `max_epochs` without re-deriving `scheduler_T_max`**.
- **Don't change `ema.decay` / `ema.warmup` mid-run** without setting
  `allow_config_change: True` for one launch.
- **Don't re-introduce `--batch_size` on `submit_zgplev_full.slurm`** —
  YAML is the single source of truth for production GB.

## Where to read more

- **Full knob inventory + expanded pitfalls** (Fixed / Fixed-by-tier / all
  Quality / all Systems / all CLI rows): `docs/2026-05-08_hpo_knob_inventory.md`.
- **Sibling skill (architecture / contract)**: `skills/sfno-training/SKILL.md`.
- **Throughput fix**: `docs/2026-05-05_ddp_throughput_fix_plan.md`,
  `docs/2026-05-08_ddp_throughput_fix_resolution.md`.
- **EMA spec**: `docs/2026-05-02_ema_implementation_plan.md`.
- **Phase-1 efficiency** (DataLoader + EMA-validation-period knobs):
  `docs/2026-05-04_phase1_efficiency_implementation_plan.md`.
- **Auto-memory**: `project_zgplev_gb_decision`,
  `feedback_protect_prior_runs`, `feedback_plan_to_docs`,
  `feedback_decide_from_codebase`.
