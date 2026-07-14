# 2026-05-08 — HPO knob inventory (full reference)

Companion reference for `skills/train-sfno-hpo/SKILL.md`. The skill itself
carries only the **Active** knob table and the operational tier procedure;
this doc carries the full knob surface (Fixed, Fixed-by-tier, Quality,
Systems, Schedule, CLI), the full status-taxonomy explanation, and the
expanded pitfalls list.

Source values: `src/sfno_training/config/plasim_sim52_zgplev_full.yaml`
(production), the four tier YAMLs (`_smoke`, `_tiny`, `_short`, `_baseline`),
`src/sfno_training/train_plasim.py` (CLI overlay + `_resolve_batch_sizes`),
and `makani-src/makani/utils/argument_parser.py` (Makani CLI surface).

## Status taxonomy (full)

Every YAML / CLI key in the inventory below carries a **Status** that says
when sweeping is justified:

- **Active** — currently being decided. Today: `batch_size`, and `lr` as its
  derived consequence.
- **Deferred** — could become active later if a specific signal appears
  (loss/val pathology, regression, hardware change). Not on the current
  critical path.
- **Ablation-only** — sweep as a one-off ablation to characterise behaviour
  or for a paper-style study; not part of normal production tuning.
- **Fixed by convention** — group convention or empirical default; do not
  sweep without a strong reason and a corresponding `docs/YYYY-MM-DD_*.md`
  plan.
- **Fixed (contract)** — never sweep. Owned by the `sfno-training` sibling
  skill; listed here so HPO sweeps don't accidentally land on them.
- **Fixed-by-tier (architecture)** — differs across tiers (smoke / tiny /
  short / baseline / full) but is *not* swept inside the production tier.
  Treat as project-defining choices, not HPO points.

If a request lands on a non-Active knob, first justify why it is moving to
Active (what signal triggered it, what the success metric is) before
designing a sweep. The default answer for non-Active knobs is "leave it
alone".

## Knob class definitions

- **Quality** — affects val loss / val ACC / EMA-val. Tested against a real
  training run on the production tier.
- **Systems** — affects samples/sec, OOM, NCCL behaviour. Tested against the
  I2 microbench feasibility filter.
- **Schedule** — schedule length and per-epoch sample budget. Tied
  algebraically; usually changed only with the schedule itself.
- **CLI** — set per launch in the SLURM script, not in YAML.

## Fixed (contract) — never sweep

| YAML key(s) | Production value | Why fixed |
| --- | --- | --- |
| `n_state_channels` / `n_diagnostic_channels` / `n_forcing_channels` | 52 / 1 / 6 | Channel contract; 58-in / 53-out hard-asserted by `PlasimTrainer._set_data_shapes`. |
| `channel_names`, `forcing_channel_names`, `*_path` (data + stats) | zgplev v10 channel set | Dataset-defined; changing it requires a new packager output. |
| `add_grid` / `add_zenith` / `add_orography` / `add_landmask` / `add_soiltype` | False | Hard-asserted; flipping any breaks `N_in_channels`. |
| `n_history` / `n_future` / `history_normalization_mode` | 0 / 0 / "none" | Hard-asserted. `n_history > 0` requires a forcing-stack audit. |
| `normalization` | "zscore" | Stats-path semantics. |
| `target` / `normalize_residual` | "tendency" / False | Plan v9 contract. |
| `prediction_type` | "iterative" | Stock Makani semantics. |
| `save_checkpoint` (YAML) | "legacy" | PlasimTrainer hard-errors on "flexible". |
| `wireup_info` / `wireup_store` | "mpi" / "tcp" | Distributed wireup. |
| `nettype` / `filter_type` / `operator_type` / `complex_activation` | "SFNO" / "linear" / "dhconv" / "real" | Group-convention architecture. |
| `losses` block (type=l2, squared=True, channel_weights=constant, temp_diff_normalization=False) | per group | Group-convention loss. Replacing the loss is a new experiment, not HPO. |
| `pretrained` / `perturb` / `add_noise` / `noise_std` | False / False / False / 0.0 | Disabled paths; flipping is a new experiment. |
| `optimizer_type` | "AdamW" | No other AdamW-class option wired up in Makani. |

## Fixed-by-tier (architecture) — out of HPO scope

These differ across tiers but are *not* swept inside the production tier.
Production = `plasim_sim52_zgplev_full`.

| Key | smoke | tiny | short | baseline | **full (prod)** |
| --- | --- | --- | --- | --- | --- |
| `scale_factor` | 4 | 4 | 3 | 3 | **1** |
| `embed_dim` | 16 | 32 | 128 | 384 | **256** |
| `num_layers` | 2 | 2 | 4 | 8 | **12** |
| `pos_embed` | none | none | none | none | **"direct"** |
| `encoder_layers` | — | — | — | — | **1** |
| `spectral_layers` | — | — | — | — | **3** |
| `mlp_ratio` | 2 | 2 | 2 | 2 | 2 |
| `activation_function` | gelu | gelu | gelu | gelu | gelu |
| `normalization_layer` | instance_norm | instance_norm | instance_norm | instance_norm | instance_norm |
| `hard_thresholding_fraction` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| `big_skip` / `rank` | — | — | — | — | True / 1.0 |

Treat any change here as a *new architecture experiment* outside HPO —
re-anchor LR, re-validate weight decay, and re-run the full promotion chain
afterwards.

## Quality knobs — full table

Tested against a real training run on the production architecture. Default
tier is **Tier B (truncated full)**. Tier A is sanity-only; Tier C is final
candidate confirmation.

| YAML key | Status | Current full | Sweep range (when active) | Tier | Metric | Dependencies / triggers |
| --- | --- | --- | --- | --- | --- | --- |
| `batch_size` (GLOBAL) | **Active** | 32 (historical, **not recommended**) | GB=16 vs GB=32 (the open question); future re-decisions: {8, 16, 32}, multiple of `data_parallel_size` | I2 microbench (feasibility) → **Tier B** for the decision | val loss / val ACC trajectory; throughput is tiebreaker | LR via sqrt rule; per-rank=GB/dp |
| `lr` | **Active (derived)** | 2.83e-4 = sqrt(32/4)·1e-4 (provisional anchor) | Recomputed from chosen GB via the sqrt rule, not independently swept right now | Tier B (with the GB candidate it's paired to) | val loss; only re-anchor if the sqrt rule fails | Provisional anchor (lr=1e-4 @ GB=4); re-anchor with `{0.5, 1, 2}×` scan only when arch / dataset change |
| `weight_decay` | Deferred | 3.0e-6 | {0, 1e-6, 3e-6, 1e-5} | Tier B | val loss; over-regularization signs (val<train, train plateau) | Trigger: persistent train-val gap or train plateau at high val |
| `optimizer_beta1` | Fixed by convention | 0.9 | — | — | — | Group convention; do not sweep |
| `optimizer_beta2` | Deferred | 0.95 | {0.95, 0.99} | Tier B | val loss / grad-norm stability | Trigger: late-training instability or grad-norm noise |
| `optimizer_max_grad_norm` | Deferred | 32 | {1, 8, 32} | Tier B | Makani-logged clipped-fraction; val loss | Trigger: clip consistently active in logs (signals LR too high or warmup too short) |
| `lr_warmup_steps` | Deferred | 5 (LinearLR epochs) | {0, 3, 5, 10} for cosine; **must be 0 for ReduceLROnPlateau** | Tier B | early-epoch loss / grad-norm spikes | Requires `scheduler: CosineAnnealingLR`; tied to `scheduler_T_max` |
| `lr_start` | Fixed by convention | 1.0e-4 (LinearLR `start_factor`) | — | — | — | Matches group's `warmup_start_lr/lr = 1e-8/1e-4` ratio |
| `scheduler` | Fixed by convention | "CosineAnnealingLR" | — | — | — | Group convention; switching to ReduceLROnPlateau requires zeroing `lr_warmup_steps` |
| `scheduler_min_lr` | Fixed by convention | 1.0e-8 | — | — | — | Cosine `eta_min`; group convention |
| `valid_autoreg_steps` | Deferred | 3 (24-h rollout) | {0, 1, 3, 5} | Tier A or B | val curve at lead 6/12/18/24 h | Trigger: rollout-skill questions or val-cost budget pressure; doesn't affect train cost |
| `ema.enabled` | Fixed by convention | True | — | — | — | Group convention; flipping is an ablation |
| `ema.decay` | Ablation-only | 0.999 | {0.99, 0.999, 0.9999} | Tier B | val loss EMA trajectory | Mid-run change requires `allow_config_change: True` for one launch |
| `ema.warmup` | Fixed by convention | True (Karras `min(decay, (1+t)/(10+t))`) | — | — | — | Resume rejects mismatch |
| `ema_validation_period` | Deferred | 1 | {1, 5} (Phase-1 plan flagged 5 as follow-up) | Tier B | val cost vs val signal density | Trigger: val cost dominates wallclock |

## Systems knobs — full table

Tested against the I2 microbench (`submit_zgplev_full_microbench.slurm`).
Microbench is **strictly a systems-feasibility filter** — pass = no OOM /
NaN / NCCL hang, peak mem ≤ 65 GB on H100, samples/sec ≥ 1.7× GB=4 baseline.
Microbench passes never decide the production GB on their own.

| Key (YAML or CLI) | Status | Current | Sweep range (when active) | Tier | Metric | Dependencies / triggers |
| --- | --- | --- | --- | --- | --- | --- |
| `batch_size` (GLOBAL) | **Active** (feasibility side) | 32 | {4, 8, 16, 32}, divisible by `data_parallel_size` | I2 microbench | samples/sec, peak GB | Quality side covered above; microbench gates Tier B entry |
| `--amp_mode` | Fixed by convention | bf16 | — | — | — | bf16 is current default; fp16 needs grad-scaler audit; "none" is debug-only |
| `--checkpointing_level` | Fixed by convention | 2 | — | — | — | Group convention 2; revisit only if memory headroom changes meaningfully |
| `--multistep_count` | Deferred | 1 | {1, 2, 3} | Tier B | val loss; train cost | Trigger: rollout-training experiment. LossHandler `multistep_weight` frozen at construction — re-instantiate per value (plan v9 §"Do NOT") |
| `--jit_mode` | Deferred | none | {none, inductor} | I2 + Tier A | samples/sec; JIT compile cost in epoch 1 | Trigger: throughput pressure unaddressed by GB / DataLoader |
| `num_data_workers` | Fixed by convention | 4 | {0, 2, 4, 8} | I2 | samples/sec; CPU saturation | Currently sufficient; revisit if CPU-bound |
| `prefetch_factor` | Fixed by convention | 4 | {2, 4, 8} | I2 | samples/sec | Inert at current value; only meaningful if `num_data_workers > 0` |
| `persistent_workers` | Fixed by convention | True | {True, False} | I2 | samples/sec; epoch-boundary stalls | Only meaningful if `num_data_workers > 0` |
| `--fin/fout/h/w_parallel_size` | Fixed | 1 / 1 / 1 / 1 | — | — | — | Model parallelism unused; leave at 1 unless model > GPU memory |
| `--parameters_reduction_buffer_count` | Fixed | 1 | {1, 2, 4} | I2 | samples/sec at large param counts | Multi-bucket grad reduction; revisit only at much larger param counts |

## Schedule knobs — algebraically tied

| YAML key | Status | Current full | Notes |
| --- | --- | --- | --- |
| `max_epochs` | Fixed by convention | 50 | Schedule length. Changing requires updating `scheduler_T_max`; treat any change as a schedule re-design, not HPO. |
| `scheduler_T_max` | **Derived** | 45 | `max_epochs - lr_warmup_steps`. Recompute on every change to either input — manual edit is a bug-magnet. |
| `n_train_samples_per_epoch` (production) | Fixed by convention | unset → full train set | Sub-epoch sampling for production should stay unset; **microbench overlays it to `1500 × GB`** for clean step-time measurement. |
| `n_eval_samples` | Fixed by convention | unset / 512 / 1024 | Validation cost; doesn't affect train cost. |
| ReduceLROnPlateau-only knobs (`scheduler_factor`, `scheduler_patience`, `scheduler_step_size`, `scheduler_gamma`) | Inert (cosine) | (ignored when `scheduler: CosineAnnealingLR`) | Only consulted by ReduceLROnPlateau; inert in production. |

## CLI flags — full table

| Flag | Production | I2 microbench | I1 short DDP sweep | Notes |
| --- | --- | --- | --- | --- |
| `--amp_mode` | bf16 | bf16 | bf16 | Quality + systems |
| `--checkpointing_level` | 2 | 2 | (default 0) | Activation checkpointing; production = group convention 2 |
| `--multistep_count` | 1 | 1 | 1 | Rollout-training step count |
| `--batch_size` | (unset; YAML wins) | `$GB` env | `$GB` env | GLOBAL batch override (`train_plasim.py:200-202`) |
| `--skip_validation` | off | **on** | off | Microbench timing-window only |
| `--save_checkpoint` | "legacy" | "none" | "legacy" | Microbench disables checkpoint write to keep timings clean |
| `--disable_ddp` | off (DDP) | off | off | Smoke / tiny single-task path uses it; sweeps don't |
| `--enable_grad_anomaly_detection` | off | off | off | Debug-only |

CLI `--batch_size` overrides YAML `batch_size` only when `> 0`
(`train_plasim.py:200-201`); the production submit script does not pass it,
so the YAML is authoritative for GB on production launches
(`submit_zgplev_full.slurm:115-121`).

## Expanded pitfalls (full list)

The skill keeps the high-frequency / non-obvious pitfalls; this is the full
set, including the rarely-active ones.

- **The live YAML `batch_size: 32` / `lr: 2.83e-4` is not the
  recommendation.** It's the historical throughput-fix setting; subsequent
  runs showed accuracy degradation. GB=16 is the active candidate pending
  validation. (See `project_zgplev_gb_decision` in auto-memory.)
- **Don't sweep multiple knobs in one job.** Single-knob sweeps preserve
  attribution. For interacting knobs, run sequentially with the prior
  winner pinned.
- **Don't decide the production GB from microbench alone.** Feasibility ≠
  accuracy. The GB=32 case demonstrates this.
- **Don't reuse a sweep `EXP_DIR`.** Auto-resume
  (`train_plasim.py:171-176`) silently warm-starts from any prior checkpoint
  and invalidates the measurement. Both the I1 wrapper and the I2 SLURM
  have sentinel guards — don't disable them.
- **Don't run a sweep in the live production `EXP_DIR`.** Auto-memory
  `feedback_protect_prior_runs` applies.
- **Don't measure samples/sec from epoch 1.** JIT + cudnn-autotune +
  DataLoader prefetch fill warmup distorts.
- **Don't switch `scheduler` to ReduceLROnPlateau without zeroing
  `lr_warmup_steps`.** `makani-src/.../driver.py:701` raises
  `NotImplementedError`.
- **Don't change `max_epochs` without re-deriving** `scheduler_T_max =
  max_epochs - lr_warmup_steps`. Cosine decays past zero or stops short.
- **Don't change `ema.decay` / `ema.warmup` mid-run.**
  `allow_config_change: False` rejects mismatch on resume. To change, set
  `allow_config_change: True` for one launch, then set back.
- **Don't reuse a `LossHandler` across different `--multistep_count`
  values.** `multistep_weight` is frozen at construction (plan v9 §"Do
  NOT").
- **Don't change `lr` without recomputing it** against the sqrt rule (or
  re-anchoring per the procedure). Decoupled LR edits are how training
  diverges silently.
- **Don't edit Makani core for an HPO knob.** If a knob requires a
  Makani-side change, escalate; that is a `sfno-training` / patch-surface
  concern, not HPO.
- **Don't re-introduce `--batch_size` as a CLI override on
  `submit_zgplev_full.slurm`.** YAML is the single source of truth for the
  production GB so the launch summary, the YAML, and the runtime always
  agree.
- **Don't drop `--skip_validation` or `--save_checkpoint none` from the
  microbench.** They are what make the timing window noise-free. Run a
  quality check via the validation gate, not by extending the microbench.

## Cross-references

- Skill (compressed playbook): `skills/train-sfno-hpo/SKILL.md`.
- Sibling (architecture / contract): `skills/sfno-training/SKILL.md`.
- Configs:
  `src/sfno_training/config/plasim_sim52_zgplev_{smoke,tiny,short,baseline,full}.yaml`.
- Throughput fix: `docs/2026-05-05_ddp_throughput_fix_plan.md`,
  `docs/2026-05-08_ddp_throughput_fix_resolution.md`.
- EMA: `docs/2026-05-02_ema_implementation_plan.md`.
- Phase-1 efficiency: `docs/2026-05-04_phase1_efficiency_implementation_plan.md`.
- Sweep / microbench harnesses: `scripts/run_zgplev_short_ddp_sweep.sh`,
  `src/sfno_training/submit_zgplev_short_ddp.slurm`,
  `src/sfno_training/submit_zgplev_full_microbench.slurm`.
- Production launch: `src/sfno_training/submit_zgplev_full.slurm`.
- Makani CLI args (full surface): `makani-src/makani/utils/argument_parser.py`.
- Batch resolver + launch summary: `src/sfno_training/train_plasim.py:38-57`
  (`_resolve_batch_sizes`), `:67+` (launch summary emitter).
- Preflight runtime guardrail: `scripts/preflight.py`.
- Auto-memory pointers: `project_zgplev_gb_decision`,
  `feedback_protect_prior_runs`, `feedback_plan_to_docs`,
  `feedback_decide_from_codebase`.
