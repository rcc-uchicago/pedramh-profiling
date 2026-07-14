# v11_gb32_lr8e4_minlr1e5 — Two single-knob HPO probes (noise σ=0.07 + epochs=75)

**Date:** 2026-05-21
**Author:** Zhixing Liu
**Status:** Plan (pre-implementation)
**Parent config:** `plasim_sim52_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5` (current own-track NWP-scorecard best per the 2026-05-21 review)
**Related plans:** `docs/2026-05-12_v11_clip_restore_plan.md`, `docs/2026-05-19_hpo_minlr_sweep.md` (if present), `docs/2026-05-14_v11_clip_warmstart_continuation_plan.md`

---

## 1. Motivation

The β₁ ∈ {0.9, 0.95, 0.97} sweep landed null on 2026-05-21 (see memory
`project_v11_beta1_sweep_null`). Combined with the prior null/flat results
on min_lr ({1e-4, 1e-5} vs 1e-8 baseline tied within ~0.01 RMSE on tas24h)
and the peak-LR ladder maxing out at 8e-4 (1.13e-3 diverged: tas24h 2.52,
roughly 4× the best), the Adam-family knob class is exhausted for this
configuration. Every additional cell on those axes has cost ~6 h of H100
time for a sub-1% move.

Two signals remain unexplored:

1. **Input-noise concavity.** Sister cells σ=0.020 and σ=0.035 at the
   lr8e4 (no min-lr lift) parent both *regressed* significantly
   (tas24h 1.96 and 2.53 vs baseline 0.67). No σ above 0.05 has been
   tested on the **active** lr8e4 / minlr1e5 branch; a historical
   σ=0.075 config exists on the retired `lr1p13e3` branch
   (`plasim_sim52_zgplev_group_clone_v11_gb32_lr1p13e3_noise0p075.yaml`)
   but lr1p13e3 itself diverged (tas24h 2.52), so that cell is
   uninterpretable for the noise axis. A single cell at σ=0.07 on the
   active branch either confirms σ=0.05 is at the concavity peak
   (decisive: stop tuning noise) or shows a small win (action: test
   0.10 next).

2. **Training-duration undertraining signal.** The parent YAML's own
   docstring (`plasim_sim52_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5.yaml`
   lines 5–11) flags that at epoch 50 the train loss was still dropping
   −5.9% over the last 10 epochs while val_ema flattened only −1.0%. The
   min_lr=1e-5 lift was the first half of the response (give the cosine
   tail a nonzero LR to learn at); the natural second half is to also give
   the tail more *wall time*. Extending the cosine T_max from 45 → 70
   (max_epochs 50 → 75) tests whether longer time at the small-but-nonzero
   LR closes the residual short-lead gap to SFNO-5410.

These two probes are *orthogonal*: σ tunes the stochastic-regularization
floor, epochs tunes the schedule wall-time. They can run in parallel.

## 2. Probes

### 2.1 Probe A — input_noise σ probe

**Run name:** `sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_noise0p070`

**Single-knob delta vs parent:**
- `input_noise.sigma`: 0.05 → **0.07**

Everything else byte-identical to the parent (lr=8e-4, weight_decay=3e-6,
betas=(0.9, 0.999), max_epochs=50, T_max=45, lr_warmup_steps=5,
lr_start=1e-4, scheduler_min_lr=1e-5, optimizer_max_grad_norm=32.0,
EMA decay 0.999, GB=32). NO warm-start, single continuous schedule
from scratch.

**Hypothesis:** σ=0.05 is at or near the concavity peak of the input-noise
axis. Tested concavity neighbours: 0.020 (severe regress), 0.035 (severe
regress). The asymmetry would be unusual but possible: the upper side of
the concavity hasn't been sampled, so σ=0.07 either (a) lands flat / very
slightly better — confirms peak; (b) lands slightly better — try σ=0.10
next; (c) regresses — peak is confirmed and we stop.

**Methodology caveat (declared):** the sister noise cells (0.020 and 0.035)
were branched off the **lr8e4** parent (min_lr=1e-8), not the
**lr8e4_minlr1e5** parent we're using here. We accept this inconsistency
in exchange for branching off the current best — the min_lr change is in
a known-flat zone (~0.01 RMSE on tas24h), and we care more about
"is σ=0.07 better than today's best" than about a clean 4-point concavity
plot. If σ=0.07 wins decisively, a paired re-run at the lr8e4 parent can
disambiguate.

**Compute:** ~6h 16m wall (50 epochs × ~7m 25s/epoch on 4× H100), same
as parent. SBATCH budget 10 h (60% headroom, matching parent SLURM).

**Decision matrix:**
| Outcome (tas24h RMSE vs parent's 0.6697) | Action |
|---|---|
| > 0.69 (≥4% regress) | Concavity peak confirmed; stop tuning noise; move on to next axis |
| 0.66–0.69 (flat ±2%) | Peak confirmed; stop tuning noise |
| < 0.66 (decisive win) | Test σ=0.10 as next cell |

### 2.2 Probe B — training-duration probe

**Run name:** `sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_epochs75`

**Single-knob delta vs parent (one logical knob, two dependent fields):**
- `max_epochs`: 50 → **75**
- `scheduler_T_max`: 45 → **70** (max_epochs − lr_warmup_steps, preserves
  the parent's identity that T_max = max_epochs − warmup)

Everything else byte-identical (lr=8e-4, scheduler_min_lr=1e-5,
lr_warmup_steps=5 unchanged, lr_start=1e-4, input_noise.sigma=0.05,
weight_decay=3e-6, betas=(0.9, 0.999), clip=32, EMA decay 0.999, GB=32).
NO warm-start (see methodology note below).

**Hypothesis:** the parent's val_ema plateau is wall-time starvation in
the cosine tail, not capacity-limited convergence. Stretching T_max
45 → 70 keeps the schedule shape identical (peak LR=8e-4 unchanged, floor
1e-5 unchanged, warmup 5 epochs unchanged) but gives the model 25 more
epochs of cosine descent, of which roughly the last ~15 are at the
small-but-nonzero floor region. If val_ema reactivates in epochs 55–75,
training duration is the lever; if it stays flat past epoch 55, training
duration is ruled out and we pivot to weight_decay or capacity questions.

**Methodology — why NOT warm-start:** the existing warm-start templates
(`v11_clip_warmstart.yaml:128,142`, `v10_warmstart.yaml:116`) are
*older GB=8 lower-LR* recipes (`lr: 1.0E-4`, `scheduler_T_max: 45`,
`scheduler_min_lr: 1.0E-8`, 50 epochs), so they do NOT answer the
current parent's `lr=8e-4`/`min_lr=1e-5` tail-extension question.
Adapting them to the current LR regime would require both rebuilding
the schedule (peak, floor, T_max) AND coordinating with the prior
checkpoint, becoming a multi-knob change. Extending T_max from-scratch
is the literal single-knob test of the actual hypothesis. If
from-scratch wins, a follow-up second-phase warm-start (built fresh
against the `lr=8e-4`/`min_lr=1e-5` parent, starting at e.g. 1e-5 with
no warmup, decaying further) becomes the obvious next probe.

**Compute:** ~9h 24m wall (75 epochs × ~7m 25s/epoch). SBATCH budget 15 h
(60% headroom, matching parent's safety factor).

**Decision matrix:**
| Outcome (tas24h RMSE vs parent's 0.6697; val_ema curve shape) | Action |
|---|---|
| tas24h ≤ 0.64 AND val_ema visibly drops past epoch 55 | Training duration is the lever; test 100 epochs next |
| 0.64–0.68 AND val_ema flat past ep 55 | Marginal; investigate weight_decay (3e-6 → 1e-4) as the next axis |
| ≥ 0.68 | Training duration ruled out; pivot to weight_decay or capacity questions |

**Inspection source for val_ema curve:** training `out.log` (the run's
stdout/stderr captured at `$EXP_DIR/.../0/out.log`). Makani's
`save_checkpoint: "legacy"` mode keeps only the last 3 versioned
checkpoints (`ckpt_mp0_v*.tar`, see
`makani-src/makani/utils/checkpoint_helpers.py`), so epoch-55 weights
will not be retained — the val_ema time series must be read from the
training log, not reconstructed by re-evaluating mid-training
checkpoints. `log_to_wandb` is `False` for this run (parent default),
so out.log is the sole source.

## 3. Parent config audit (loader contracts)

Per `feedback_plan_loader_contracts.md`: both new YAMLs are byte-identical
clones of the parent except for the single deltas above. The parent has
been training+evaluating successfully (the 0.6697 tas24h number we're
A/B-ing against is from its 2026-05-20 EMA eval bundle), so:

- HDF5 key layout (`fields_diagnostic`, `forcing`): unchanged
- Stats paths (`global_means.npy`, `global_stds.npy`, `time_means.npy`,
  `forcing_*`): unchanged, same `$OUTPUT_ROOT` substitution
- Channel list (53 channels: 52 state + 1 diagnostic pr_6h): unchanged
- Forcing channels (6: lsm, sg, z0, sst, rsdt, sic): unchanged
- Calendar (proleptic, dhours=6, dt=1): unchanged
- Net dispatch (`nettype: SFNO`, embed_dim=256, num_layers=12,
  spectral_layers=3, etc.): unchanged
- EMA config (decay 0.999, warmup True, `allow_config_change: False`):
  unchanged

No third-party loader audits are required; reusing the validated parent
contracts.

## 4. Safety / hygiene

- **Fresh `EXP_DIR` for each run** (per memory `feedback_protect_prior_runs`):
  - Probe A: `$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_noise0p070/`
  - Probe B: `$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_epochs75/`
- **Collision semantics (clarified):** the parent SLURM's guard refuses
  to write if `$EXP_DIR/<config>/0/training_checkpoints/ckpt_mp0_v*.tar`
  exists, unless `RESUME=1`. It does NOT refuse a *partial* dir that
  contains a rendered yaml or preflight log but no checkpoints — those
  will be silently overwritten on launch. For these two fresh-launch
  probes, the user MUST verify both `$EXP_DIR` paths above do not exist
  before sbatch (a simple `ls $SCRATCH/SFNO_Climate_Emulator/runs/ | grep -E
  'noise0p070|epochs75'` should return nothing). If the dir exists from
  a prior aborted attempt, remove it (or rename) before resubmitting.
- The actual resume mechanism (when `RESUME=1`) is mtime-based: makani
  inspects `$RUN0_DIR/training_checkpoints/ckpt_mp0_v0.tar` as a
  sentinel, then resumes from the most-recent `ckpt_mp0_v*.tar` by
  mtime. We do NOT use resume for these probes.
- No warmstart, so `pretrained_checkpoint_path` is left at the default
  (unset).
- `optimizer_max_grad_norm: 32` retained — input_noise.sigma raised to
  0.07 in Probe A is still within the regime the clip was sized for.
- **SLURM hygiene on copied scripts:** the parent SLURM
  (`submit_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5.slurm:95`) carries
  a stale comment `world_size=4, GB=8 from yaml → per-rank=2` from an
  earlier GB=8 era; the actual yaml has `batch_size: 32` → per-rank=8.
  The two copied SLURM scripts MUST fix this comment to read GB=32 →
  per-rank=8, and update the wallclock comment for Probe B (`50 epochs
  → 75 epochs, ~9h 24m expected, 15 h budget`).

## 4.5. Artifacts to create

Exact filenames + top-level config keys for the copied-and-edited
artifacts. Both YAMLs are byte-identical clones of the parent except
for the single deltas in §2.

**Probe A — noise0p070:**
- YAML: `src/sfno_training/config/plasim_sim52_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_noise0p070.yaml`
  - Top-level config key (line 36 in parent): `plasim_sim52_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_noise0p070`
  - Single line change: `input_noise.sigma: 0.05` → `0.07`
- SLURM: `src/sfno_training/submit_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_noise0p070.slurm`
  - `SFNO_JOB_NAME=sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_noise0p070`
  - `EXP_DIR` default: `$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_noise0p070`
  - `FULL_TPL` / `FULL_YAML` / `--config` strings updated to the new name
  - SBATCH `-J`, `-o`, `-e` updated; `-t 10:00:00` retained (same compute as parent)
  - Fix the parent's stale comment: `world_size=4, GB=8 → per-rank=2` → `world_size=4, GB=32 → per-rank=8`

**Probe B — epochs75:**
- YAML: `src/sfno_training/config/plasim_sim52_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_epochs75.yaml`
  - Top-level config key: `plasim_sim52_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_epochs75`
  - Two dependent line changes: `max_epochs: 50` → `75` AND `scheduler_T_max: 45` → `70`
- SLURM: `src/sfno_training/submit_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_epochs75.slurm`
  - `SFNO_JOB_NAME=sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_epochs75`
  - `EXP_DIR` default: `$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_epochs75`
  - `FULL_TPL` / `FULL_YAML` / `--config` strings updated
  - SBATCH `-J`, `-o`, `-e` updated; SBATCH `-t` **raised 10:00:00 → 15:00:00** (75 epochs × ~7m 25s ≈ 9h 24m + 60% headroom)
  - Fix the parent's stale comment: `world_size=4, GB=8 → per-rank=2` → `world_size=4, GB=32 → per-rank=8`
  - Update the wallclock comment block: `50 epochs at ~7m 25s/epoch = 6h 16m` → `75 epochs at ~7m 25s/epoch ≈ 9h 24m, 15h budget`

## 5. Eval plan

Both runs evaluated via the standard own-track NWP scorecard. The default
`submit_eval_prelude.sh` defaults `TEST_HOLDOUT`, `TRAIN_DIR`, and
`PACKAGER_TEST_SRC` to the v10 paths
(`sim52_zgplev_full`, `sim52_astro_64x128_zgplev`); these are **v11** runs
so each invocation MUST export the v11 data overrides (precedent:
`scripts/submit_beta1_chains.sh:16`). The `RUN_DIR` must point at the
inner `/<config>/0` run dir, not the top-level `$EXP_DIR`, because
`submit_eval_prelude.sh:26` and `train_plasim.py:300` write
`config.json` and `training_checkpoints/` inside that inner dir.

**Probe A — noise0p070:**

```bash
# Run from a login node (login1/login2/login3) — TACC blocks sbatch from
# compute nodes incl. idev sessions. Convention per scripts/submit_beta1_chains.sh:7.
export RUN_DIR=$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_noise0p070/plasim_sim52_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_noise0p070/0
export CKPT=$RUN_DIR/training_checkpoints/best_ckpt_ema_mp0.tar
export TEST_HOLDOUT=$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full_v11/test_holdout
export TRAIN_DIR=$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full_v11/train
export PACKAGER_TEST_SRC=$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev_v11/test
export TRACK=own MODE=nwp
test -s "$CKPT" || { echo "FATAL: $CKPT missing or empty" >&2; exit 1; }
scripts/submit_eval.sh
```

**Probe B — epochs75:**

```bash
# Run from a login node (login1/login2/login3).
export RUN_DIR=$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_epochs75/plasim_sim52_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_epochs75/0
export CKPT=$RUN_DIR/training_checkpoints/best_ckpt_ema_mp0.tar
export TEST_HOLDOUT=$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full_v11/test_holdout
export TRAIN_DIR=$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full_v11/train
export PACKAGER_TEST_SRC=$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev_v11/test
export TRACK=own MODE=nwp
test -s "$CKPT" || { echo "FATAL: $CKPT missing or empty" >&2; exit 1; }
scripts/submit_eval.sh
```

Compare against parent (β₁=0.9 EMA eval bundle at
`$WORK/SFNO_Climate_Emulator/results/sfno_eval/20260520_eval-867fead_data-8b395eb_family-sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_ckpt-best_ckpt_ema_mp0`)
and the SFNO-5410 benchmark already overlaid by the eval pipeline.

**`pr_6h` caveat:** per
`docs/2026-05-14_pr_6h_units_mismatch_ticket.md`, the own-track `pr_6h`
output is in m/s while the 5410 benchmark row is in
kg m⁻² per 6 h — a ~3,600–4,400× numeric gap that the eval pipeline
does NOT reconcile. Separately, `scripts/render_eval_figures.py:28,278`
intentionally omits `pr_6h` line plots entirely because precipitation
is intermittent and both RMSE *and* ACC are unstable diagnostics for
it. Therefore: **do not use cross-track `pr_6h` in any form; within-track
`pr_6h` may be inspected as a secondary diagnostic only.** HPO
decisions in this loop are based on **tas, zg500, ua5, ta5** (and their
RMSE/ACC). The sanity gate (own-track `tas` RMSE at 6 h vs persistence,
`zg500` ACC at 24 h vs 0.6 threshold) is internal to own-track and
unaffected.

The `export ... ; test -s "$CKPT" ; scripts/submit_eval.sh` pattern
(applied in both blocks above) catches the common failure mode where
training didn't reach a best EMA checkpoint (or resume bypassed it),
since `submit_eval_prelude.sh:29` only selects the path without
validating existence — the failure otherwise lands inside the queued
inference SLURM job, wasting queue time. `export` (not one-shot env
prefix) is required so `test -s` and `scripts/submit_eval.sh` see the
same `CKPT` value.

## 6. What this plan does NOT touch

- `n_future > 0` (multistep / AR loss). Reason: AR typically *hurts*
  short-lead skill — exactly the regime where we already trail 5410.
- Capacity (embed_dim / num_layers). Reason: per memory
  `project_makani_5410_same_dof`, the 56.5M vs 106.9M apparent gap is a
  complex64-counting artifact, not a real DoF deficit.
- β₁ further tuning. Reason: settled null on 2026-05-21.
- Peak LR (8e-4). Reason: settled winner; 1.13e-3 diverged.
- min_lr. Reason: tested {1e-4, 1e-5} → flat zone, 1e-5 marginally best
  on tas, picked for current parent.

## 7. Out-of-scope (deferred follow-ups)

- If Probe B wins, a second-phase warm-start with no warmup and floor
  decaying from 1e-5 to e.g. 1e-7 over another 25 epochs.
- If both probes are flat, the next single-knob axis is `weight_decay`
  (3e-6 → 1e-4) to attack the "train still dropping, val flat" mild-overfit
  reading.
