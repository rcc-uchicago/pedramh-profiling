# GB16/GB32 vs GB8: Minimal 40-yr LR Probe (No Anchor)

> **Revised 2026-05-11 after submission:** the earlier draft included a GB8
> 40-yr calibration anchor run. That contradicted the original
> compute-savings premise (40 yr was supposed to *avoid* retraining GB8) and
> was dropped on 2026-05-11. The active matrix is **one new training run
> only — GB32 @ 4e-4 @ 40 yr (job 3109220)** — compared directly against
> the existing 100-yr GB8 group_clone scorecard. The 40-yr-vs-100-yr
> training penalty is absorbed into the decision-gate noise floor rather
> than measured by a dedicated anchor.

## Context

The own-track GB sweep was paused on 2026-05-09 (`project_zgplev_gb_decision.md`)
with GB4 (lr=1e-4, β₂=0.95, grad_clip=32) winning 29/32 cells on the 96-IC NWP
scorecard. The previously trained GB16/GB32 candidates used the **production
optimizer knobs** (β₂=0.95, grad_clip=32, no input noise) and were unable to
match GB4 on medium-range geopotential.

The decision now is to **re-open the question at a different anchor**: GB8
with the **group_clone recipe** (β₂=0.999, grad_clip disabled, input_noise
σ=0.05 on 52 state channels, lr=1e-4) is the new accuracy baseline. The
β₂=0.999 setting is empirically more LR-tolerant in the Makani/SFNO
literature, so the prior GB16/GB32 conclusions do not automatically transfer
— in particular, the linear-scaled GB32 LR was never reached at any prior
recipe.

Scientific goal: decide whether GB16 or GB32 can plausibly match GB8-level
skill closely enough to justify a full 100-yr production run, *without*
re-spending compute on settings that have already been characterised.

User decisions (interview, 2026-05-11):
- **Baseline for the decision gate**: existing GB8 group_clone run (100 yr).
- **Skill tolerance**: <5 % RMSE and <0.01 ACC degradation averaged across
  the NWP scorecard cells, no single cell >10 % degraded.
- **Match basis**: 50 epochs per run (same as prior sweeps).
- **40-yr subset**: contiguous years 12–51.
- **Matrix discipline (key constraint)**: do **not** re-run settings whose
  100-yr behavior is already characterised. The matrix exists to generate
  *new* information, not to redo (GB16, lr=1e-4 / 2e-4) or (GB32, lr=2.83e-4)
  under a different recipe.

## Previous Results Summary (recap — what we already know)

| Run | GB | LR | per-rank | recipe | ms/step | samples/s | val_loss_ema | NWP wins/32 | known? |
|---|---|---|---|---|---|---|---|---|---|
| sfno_zgplev_full (GB4 prod) | 4 | 1e-4 | 1 | prod (β₂=0.95, clip=32, no noise) | ~60 | ~67 | (older log) | **29** | ✓ |
| sfno_zgplev_full_gb16_lr1e4_20260508 | 16 | 1e-4 | 4 | prod | 65.3 | 245 | 0.00283 | 0 (undertrained) | ✓ |
| sfno_zgplev_full_gb16_lr2e4_20260509_retry1 | 16 | 2e-4 | 4 | prod | 64.7 | 247 | 0.00220 | 0 | ✓ |
| sfno_zgplev_full_gb32_20260508 | 32 | 2.83e-4 | 8 | prod | 95.4 | 335 | 0.00233 | 3 (all 336 h) | ✓ |
| sfno_zgplev_group_clone (**GB8 BASELINE**) | 8 | 1e-4 | 2 | group_clone (β₂=0.999, no clip, noise σ=0.05) | 59.5 | 135 | 0.00312 | scorecard exists at `/work2/.../sfno_eval/20260510_eval-8b395eb_data-e3c934b/` (96 ICs) | ✓ for val; ✓ for NWP (verify acceptability before reuse) |

The only **genuinely unknown** points in the (GB, LR, recipe) cube that
plausibly matter are:
- GB32 with linear-scaled LR (4.0e-4 from the GB8 anchor) — never trained
  at any recipe.
- The 40-yr-subset training penalty itself at any setting — never measured.

The matrix below tests exactly those two unknowns and nothing else.

## Experiment Matrix

### Default (1 run)

| # | Role | GB | LR | dataset | recipe | wallclock budget | why this point |
|---|---|---|---|---|---|---|---|
| 1 | **Primary probe** | 32 | 4.0e-4 | 40 yr (years 12–51) | group_clone (β₂=0.999, no clip, noise σ=0.05, EMA=0.999) | 6 h | Linear-scaled LR from the GB8 100-yr baseline (1e-4 × 32/8 = 4e-4). No prior run has ever reached the linear endpoint under the stable β₂=0.999 recipe. The single most informative new point in the (GB, LR, recipe) cube. |

**Total default cost**: ~6 h H100-node wall (1 job — SLURM ID 3109220).
**Baseline for comparison**: the existing 100-yr GB8 group_clone scorecard
at `/work2/.../sfno_eval/20260510_eval-8b395eb_data-e3c934b/`. Used directly
as the 40-yr-probe comparison anchor; the 40-yr-vs-100-yr training-dataset
penalty is absorbed into the decision-gate tolerance, not measured.

### Contingent (run only if triggered)

| # | Trigger | Role | GB | LR | why this point |
|---|---|---|---|---|---|
| C1 | Probe (#2) NaNs / diverges before epoch 10 | GB32 fallback | 32 | 3.0e-4 | Conservative midpoint between the prior 100-yr-known-stable 2.83e-4 (under less-stable prod knobs) and the diverged 4.0e-4. Not an exact duplicate of any existing run. Tests whether linear scaling is too aggressive but ~mid-bracket still gains over sqrt. |

Contingent runs are **not** pre-allocated. The decision to launch C1 is
explicit and requires a fresh sign-off after the default runs complete.

GB16 @ 4e-4 was considered as a possible contingent and is **rejected**: it
is 2× linear scaling from the GB8 anchor, more aggressive than the already
rejected GB16 @ 2.83e-4, and contradicts the diminishing-returns logic that
motivates the [sqrt, linear] interval. If a GB16 cross-check is wanted after
GB32 succeeds, the only defensible point is GB16 @ 2e-4 *under the
group_clone recipe* as an explicit recipe-transfer calibration — but that
repeats an already-characterised (GB, LR) pair and so requires fresh
sign-off; it is not pre-authorised by this plan.

### Explicitly NOT in the matrix (rejected by the matrix-discipline rule)

| Setting | Why rejected |
|---|---|
| GB16 @ 1.41e-4 @ 40 yr | Sub-linear; prior 100-yr GB16 @ 1e-4 already characterised as undertrained. Sub-1.41e-4 territory has no plausible upside. |
| GB16 @ 2.0e-4 @ 40 yr | Repeats existing 100-yr setting (recipe change only). Per matrix-discipline rule, not the question we are spending compute on. |
| GB16 @ 2.83e-4 @ 40 yr | Above linear from GB8 anchor; no theoretical motivation; prior GB16 @ 2e-4 already showed further LR did not buy NWP wins. |
| GB32 @ 2.0e-4 @ 40 yr | Sub-sqrt from GB8 anchor; undertrained territory. |
| GB32 @ 2.83e-4 @ 40 yr | Repeats existing 100-yr setting (recipe change only). |
| GB4 @ 40 yr | GB4 already won the 100-yr scorecard; not the new-information question. |

## Engineering Decisions and Justification

### 1. LR choice for the primary probe (GB32 @ 4.0e-4)

`4.0e-4 = 1.0e-4 × (32 / 8)` — exact linear scaling from the GB8 anchor.
Justifications:
- The [sqrt-scaling, linear-scaling] interval is the only interval with a
  theoretical motivation under common large-batch scaling arguments
  (McCandlish et al.; Smith & Le; Goyal et al.). Above linear has no
  motivation; below sqrt is undertrained.
- The sqrt endpoint (2.0e-4) is already characterised at the less-stable
  β₂=0.95 recipe as a 100-yr point that doesn't beat GB4 — testing it again
  under group_clone recipe is the recipe-change confound the user excluded.
- The linear endpoint has *never* been trained at any recipe on this codebase.
  It is the single most informative new point.
- β₂=0.999 + input_noise is a more stable optimizer setting than β₂=0.95,
  so the divergence risk at the higher LR is lower than the prior sweep would
  suggest. If 4e-4 still NaNs, that is itself a useful negative result and
  triggers Contingent C1.
- A linear-scaled GB16 @ 2.0e-4 already exists at 100 yr under prod knobs
  and characterises GB16's NWP-scorecard failure mode; rerunning it under
  group_clone recipe on 40 yr would be a pure recipe-change ablation, not
  the GB-scaling question we are paying compute for.

### 2. No 40-yr calibration anchor (revised 2026-05-11)

An earlier draft proposed a GB8 @ 1e-4 @ 40 yr "calibration anchor" run to
isolate the 40-yr-vs-100-yr training penalty from the GB-scaling answer.
That was dropped: it contradicts the original compute-savings premise of
the 40-yr subset (which was to avoid retraining GB8) and pays ~8 h of H100
time to resolve an ambiguity we are willing to absorb into the decision-gate
tolerance instead.

The trade-off accepted: if the GB32 @ 4e-4 probe lands ambiguously close to
the 100-yr GB8 baseline (e.g. ~5–10 % worse on the scorecard), we cannot
formally attribute that to GB-scaling vs dataset-size. The fallback is to
promote the candidate to a 100-yr training run anyway, which is the same
action we would take after a passing 40-yr result — so the anchor would
not have changed the decision in the marginal case.

### 3. Match basis: same epochs (50)

User-selected. Per-epoch optimizer steps drop ~60 % on 40 yr vs 100 yr; total
samples seen drop ~60 %. Result is conservative — any 40-yr candidate that
matches the 100-yr GB8 baseline is a true win on sample efficiency *and*
throughput.

### 4. Scheduler / warmup: unchanged

`lr_warmup_steps: 5` and `CosineAnnealingLR` over the full 50 epochs.

Concretely, Makani builds `LinearLR(... total_iters=lr_warmup_steps)` and
`CosineAnnealingLR(T_max=...)` and **steps the scheduler once per epoch
after validation** (`makani-src/makani/utils/driver.py:696`,
`…/training/deterministic_trainer.py:351`). Warmup and cosine are therefore
epoch-indexed, not per-step or per-sample. That has two implications:

- The 40-yr probe runs 5 warmup epochs + 45 cosine epochs = 50 epochs total,
  same shape as the 100-yr GB8 baseline.
- The 40-yr probe's 5 warmup epochs cover ~40 % the optimizer updates and
  ~40 % the samples that the 100-yr baseline saw during its warmup window.
  This is part of the 40-yr-vs-100-yr training penalty absorbed into the
  decision-gate tolerance; we do not "correct" for it by changing
  `lr_warmup_steps`, because doing so would confound the comparison with a
  schedule change.

No reason to alter `lr_warmup_steps` or `T_max`.

### 5. EMA and grad clipping: fixed at GB8 group_clone values

All runs (default and contingent) use:
- `ema.decay: 0.999` with Karras warmup (`min(decay, (1+t)/(10+t))`).
- `optimizer_max_grad_norm: 0.0` (disabled — matches group recipe).
- `optimizer_beta2: 0.999`.
- `weight_decay: 3.0e-6`.
- `input_noise.perturb`: σ=0.05 on the 52 state channels.

These knobs **define** the GB8 baseline; varying any of them turns the
experiment into a multi-axis ablation that the user did not authorise.

### 6. Metrics, checkpoints, evaluation gates

Per-epoch metrics (already logged by `PlasimTrainer`): `train_loss`,
`val_loss`, `val_loss_ema`, `lr`, `grad_norm` (informational, clip disabled),
`timing/training step time [ms]`, derived `samples/sec`.

Checkpoints saved per run (Makani defaults, do not change):
- `best_ckpt_mp0.tar` — best on **base validation loss** (not EMA).
- `best_ckpt_ema_mp0.tar` — best on EMA validation loss.
- rotating `ckpt_mp0_v*.tar` — periodic snapshots during training.

**Which checkpoint gets NWP-scored:** `best_ckpt_mp0.tar` for both the 40-yr
anchor and the GB32 40-yr probe (and any contingent run). This is explicit
alignment with the reused GB8 100-yr baseline scorecard, which was produced
from `best_ckpt_mp0.tar` (verified — see §7). Scoring the EMA-best
checkpoint instead would introduce a checkpoint-selection confound with the
baseline and invalidate the decision gate.

Two-tier gating (revised 2026-05-11, no anchor):
- **Cheap gate (val-only)**: a candidate is eligible for full NWP scoring iff
  `min(val_loss_ema) ≤ 1.10 × 0.00312 = 0.00343`. The 10 % envelope (vs the
  earlier 5 % cheap gate) absorbs the 40-yr-vs-100-yr training penalty that
  the dropped anchor would have measured. Candidates that clearly fail the
  cheap gate are not scored; candidates marginally above it (≤ 1.15× the
  baseline) may still be scored at user discretion since the cheap gate is
  not the decision gate.
- **Decision gate (NWP scorecard, 96 ICs, K=56)** vs the GB8 100-yr baseline
  scorecard across `{tas, zg500, ua5, ta5, pr_6h}` × `{6, 24, 72, 120, 240,
  336 h}` (30 cells): pass iff mean RMSE degradation < 5 %, mean ACC
  degradation < 0.01, no single cell > 10 % RMSE degraded.

### 7. Precondition: verify the existing GB8 group_clone scorecard

A GB8 group_clone NWP scorecard already exists at
`/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results/sfno_eval/20260510_eval-8b395eb_data-e3c934b/`
(verified by Codex review on 2026-05-11; report points to the GB8
group_clone checkpoint and `scores/nwp_scorecard_summary.csv` exists with
96 ICs).

Before treating it as the decision-gate baseline, verify:
- `report.md` lines 3–10 resolve to the sfno_zgplev_group_clone run dir
  (correct checkpoint).
- The scored checkpoint is `best_ckpt_mp0.tar` (base-val-loss-best), not
  `best_ckpt_ema_mp0.tar` (EMA-best) — this is the alignment the anchor
  and probe NWP evals must match.
- `scores/nwp_scorecard_summary.csv` exists and contains rows for all 5
  channels `{tas, zg500, ua5, ta5, pr_6h}` at all 6 canonical leads
  `{6, 24, 72, 120, 240, 336 h}` (= 30 cells), with 96 ICs.
- K (forecast lead count in inference) matches what the new candidates will
  be evaluated against (K=56).

Codex review 2026-05-11 confirmed: the 5 channels × 6 leads × 2 metrics are
present with `n_ics=96`.

If all three hold, reuse this scorecard as the decision-gate baseline and
skip Step 1 below. Only rerun the GB8 baseline eval if a structural
mismatch is found (e.g. wrong K, missing channel, fewer ICs).

### 8. Promotion criterion to full 100-yr training

- If the GB32 @ 4e-4 Probe passes both gates: promote it to a full 100-yr
  training run with the same recipe. This is the win path.
- If the Probe diverges or NaNs: consider C1 (GB32 @ 3e-4), with explicit
  sign-off.
- If the Probe trains stably but misses either gate by a small margin
  (e.g. ≤ 10 % RMSE degraded): the 40-yr-vs-100-yr training penalty is the
  most likely culprit; promote to a 100-yr run anyway and re-judge on the
  100-yr scorecard.
- If the Probe trains stably but misses the gate by a large margin
  (> 10–15 %): no promotion. The GB-scaling answer is "GB32 with linear-
  scaled LR does not match GB8 even under the more stable recipe."

## Files to create / modify

All paths under `/home1/11114/zhixingliu/projects/SFNO_Climate_Emulator/` unless noted.

### New (1 training config + 1 SLURM script)

- `src/sfno_training/config/plasim_sim52_zgplev_gbhpo40_gb32_lr4_0e-4.yaml`
  — copy of `plasim_sim52_zgplev_group_clone.yaml` with `batch_size: 32`,
  `lr: 4.0e-4`, and dataset paths pointing at the 40-yr symlink farm. All
  other knobs identical to the group_clone baseline.

- `src/sfno_training/submit_zgplev_gbhpo40_gb32_lr4_0e-4.slurm`
  — copy of `submit_zgplev_group_clone.slurm` with the SBATCH wallclock
  trimmed to `06:00:00` and `CFG` / `EXP_DIR` pointing at the per-run YAML
  and a fresh run directory. `RESUME` guard kept verbatim — never resume
  into a populated run dir (`feedback_protect_prior_runs`).

### Removed in the 2026-05-11 revision

- `…_gb8_anchor.yaml` and `…submit_…gb8_anchor.slurm` were written and
  deleted in the same session. The GB8 40-yr anchor was dropped because it
  contradicted the compute-savings premise of the 40-yr subset (which was
  meant to avoid retraining GB8 in the first place).
- `scripts/run_zgplev_gbhpo40_default.sh` (sweep driver) was deleted in the
  same revision — a single-job sweep does not need an orchestrator;
  `sbatch <slurm>` is sufficient.

### Contingent (created only when triggered)

- `…_gb32_lr3_0e-4.yaml` + `…submit_…gb32_lr3_0e-4.slurm` (if C1 triggers).

### Reused (no edit, just invoked)

- `scripts/build_subset_dataset.py` — invoked once on the login node to
  build the 40-yr symlink farm. No code change.
- `scripts/submit_eval.sh` and the `eval-sfno-own` skill — invoked once per
  trained candidate. The GB8 100-yr baseline scorecard already exists
  (verified — see §7) and is reused. No code change.

### Not edited

- `plasim_sim52_zgplev_full.yaml`, `plasim_sim52_zgplev_group_clone.yaml`,
  `plasim_sim52_zgplev_full_v11.yaml` — left as-is. The HPO doc already
  notes the live GB=32 entry in `_full.yaml` is "historical, not the
  recommendation"; this sweep does not change that until promotion.

## Run names and output paths

40-yr dataset symlink farm — **use a fresh, date-versioned destination
path** because `build_subset_dataset.py` does not aggressively delete stale
links if the destination already exists:
- `$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_40yr_y12-51_20260511/{train,valid,stats,metadata,config,test}`
  (built once via `build_subset_dataset.py --train-years 12-51 --valid-years 11`).

Training run directories (fresh, no resume permitted):
- `$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_gbhpo40_gb32_lr4_0e-4_20260511/`
- (contingent) `…_gb32_lr3_0e-4_20260511/`

NWP scorecard outputs:
- **GB8 100-yr baseline (reused, already exists)**:
  `/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results/sfno_eval/20260510_eval-8b395eb_data-e3c934b/`
- `/work2/.../sfno_eval/20260511_gbhpo40_gb32_lr4_0e-4_40yr/`

Final rollup report (manual markdown, written after evals complete):
- `docs/2026-05-11_zgplev_gbhpo_40yr_results.md`

## Commands

Step 0 — build the 40-yr dataset symlink farm at a fresh, date-versioned
destination (login node). Confirm the destination does not already exist
before running:

```
test ! -e $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_40yr_y12-51_20260511 \
  && python scripts/build_subset_dataset.py \
       --src $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full \
       --dst $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_40yr_y12-51_20260511 \
       --train-years 12-51 \
       --valid-years 11
```

Step 1 — verify the existing GB8 group_clone 100-yr NWP scorecard at
`/work2/.../sfno_eval/20260510_eval-8b395eb_data-e3c934b/` per §7 of
Engineering Decisions. Reuse if valid; only rerun the GB8 baseline eval if
a structural mismatch is found.

Step 2 — submit the 1-run probe (already done 2026-05-11 as JobID 3109220):

```
sbatch src/sfno_training/submit_zgplev_gbhpo40_gb32_lr4_0e-4.slurm
```

Step 3 — when the training job completes, run `eval-sfno-own MODE=nwp`
against `best_ckpt_mp0.tar` if the cheap val gate passes.

Step 4 — decide:
- Probe passes both gates → promote to 100-yr run (separate sign-off).
- Probe diverges / NaNs → consider triggering C1 (GB32 @ 3e-4), explicit
  sign-off.
- Probe trains stably but misses the gate by a small margin (≤ 10 % RMSE
  degraded) → promote to 100-yr run anyway; the 40-yr-vs-100-yr training
  penalty is the most likely culprit.
- Probe trains stably but misses by a large margin (> 10–15 %) → no
  promotion; conclude GB32 + linear-scaled LR does not match GB8 under
  the group_clone recipe.

Step 5 — write the rollup markdown using the comparison table format below.

## Comparison table format (for the rollup report)

| Run | GB | LR | dataset | recipe | ms/step | samples/s | best val_loss_ema | Δ vs GB8 100-yr | passes cheap gate | NWP mean RMSE Δ% | NWP mean ACC Δ | NWP cells passed (/30) | passes decision gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GB8 100-yr baseline | 8 | 1.0e-4 | 100 yr | group_clone | 59.5 | 135 | 0.00312 | — | — | 0 | 0 | — | — |
| **GB32 40-yr probe** | 32 | 4.0e-4 | 40 yr | group_clone | tbd | tbd | tbd | tbd | tbd | tbd | tbd | tbd | tbd |
| (C1) GB32 40-yr fallback | 32 | 3.0e-4 | 40 yr | group_clone | tbd | tbd | tbd | tbd | tbd | tbd | tbd | tbd | tbd |

Per-channel × per-lead RMSE/ACC delta heatmaps (the existing
`scripts/render_eval_figures.py` already supports overlay comparison) are
attached per-candidate.

## Verification

End-to-end checks before any sweep result is treated as authoritative:

1. `ls $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_40yr_y12-51_20260511/train | wc -l`
   must be 40 (years 12–51 inclusive). `ls .../valid | wc -l` must be 1
   (year 11). If the destination dir pre-existed, abort and pick a different
   date-versioned destination — `build_subset_dataset.py` does not aggressively
   delete stale links.
2. For the 1 new YAML:
   - `diff` against `plasim_sim52_zgplev_group_clone.yaml` to confirm only
     `batch_size` (8→32), `lr` (1e-4→4e-4), and the dataset paths differ
     (verified pre-submit on 2026-05-11).
3. The first training job's epoch-1 log must show `batch_size=32`,
   `data_parallel_size=4`, `per_rank_batch=8`, `optimizer_beta2=0.999`,
   `optimizer_max_grad_norm=0.0`, `ema_decay=0.999`. If any value disagrees,
   kill the job and re-inspect the YAML.
5. The reused GB8 100-yr baseline NWP scorecard
   (`/work2/.../sfno_eval/20260510_eval-8b395eb_data-e3c934b/scores/nwp_scorecard_summary.csv`)
   must contain rows for all 5 channels `{tas, zg500, ua5, ta5, pr_6h}` at
   all 6 canonical leads `{6, 24, 72, 120, 240, 336 h}` (= 30 cells), the
   report must resolve to the sfno_zgplev_group_clone checkpoint at K=56,
   96 ICs, and the scored checkpoint must be `best_ckpt_mp0.tar` (not the
   EMA-best). If all of these hold, the decision gate is defined and the
   sweep proceeds; if any fail, the existing scorecard is invalid for this
   sweep and the GB8 baseline must be re-evaluated before the decision gate
   can be applied.
6. After the probe completes, its val_loss_ema curve must be monotonically
   non-increasing after epoch 5 (post-warmup). A non-converging probe is
   evidence of training instability, not a GB-scaling conclusion; consider
   C1 (GB32 @ 3e-4) before recommending against linear-scaled LR.

## Open items deferred until after the probe completes

- Whether to launch contingent run C1 (GB32 @ 3e-4) — requires explicit
  sign-off.
- Whether to launch a GB16 @ 2e-4 @ 40 yr *recipe-transfer calibration* (the
  defensible GB16 cross-check Codex review suggested if a GB16 confirmation
  is desired post-hoc). This repeats an already-characterised (GB, LR) pair
  under a different recipe and is therefore not pre-authorised by this plan
  — requires fresh sign-off.
- Whether the GB32 @ 4e-4 Probe is promoted to a full 100-yr training run,
  and the wallclock budget for that promotion (likely the same 17 h envelope
  as `submit_zgplev_group_clone.slurm`).
- Whether to update the live `_full.yaml` GB=32 setting once the decision is
  made (or leave the history note as-is).

## Revision history

- **2026-05-11 (initial)**: 2 default runs (GB8 40-yr anchor + GB32 @ 4e-4
  40-yr probe), 1 contingent (GB32 @ 3e-4). Approved by user; submitted as
  JobIDs 3109185 (anchor) + 3109189 (probe, afterok-dependent).
- **2026-05-11 (revised, same day)**: anchor dropped after user pointed out
  it contradicted the compute-savings premise of the 40-yr subset (which
  was scoped to GB16/GB32 only in the original prompt). Anchor + dependent
  probe cancelled; probe resubmitted standalone as JobID 3109220. Plan and
  matrix reduced to 1 default run + 1 contingent.
