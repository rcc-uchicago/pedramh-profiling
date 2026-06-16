# 2026-05-13 — Next-HPO plan after v11_clip lands

Status: **DRAFT — open for review**. Pre-Run 0 submitted (jobs 3115560–63);
Run 1 and Run 2 are gated on Pre-Run 0 evidence.

## Goal

Decide the next 1–3 own-track training runs, given the just-completed v11_clip
result and the four group-clone family runs already evaluated. Avoid replaying
prior experiments. Target a Pareto-improving recipe for the AI-RES score-function
emulator: reasonable short-lead sanity (tas@6h ≥ 0.99) AND better long-lead
stability/skill than the v10 nonoise ceiling.

## 1. Evidence

### 1.1 Six-way ACC trajectory

Per-channel ACC (own-track, 96 ICs, v10 test holdout for all rows below
except SFNO-5410 which is on its own 96-IC eval). EMA columns are
`best_ckpt_ema_mp0.tar`; raw columns are `best_ckpt_mp0.tar`.

#### tas ACC

| lead | clone v10 (raw) | nonoise v10 (raw) | v11 raw | v11 EMA | v11_clip raw | v11_clip EMA | 5410 |
|---|---|---|---|---|---|---|---|
| 6h | 0.9873 | **0.9954** | 0.9186 | 0.9193 | 0.9577 | 0.9581 | **0.9952** |
| 24h | 0.9533 | 0.9515 | 0.6679 | 0.6694 | 0.7646 | 0.7667 | 0.9783 |
| 72h | 0.9014 | 0.8708 | 0.5121 | 0.5148 | 0.6007 | 0.6043 | 0.9467 |
| 120h | 0.8429 | 0.7979 | 0.4502 | 0.4526 | 0.5411 | 0.5454 | 0.9032 |
| 240h | 0.5447 | 0.4703 | 0.2597 | 0.2616 | 0.3290 | 0.3310 | 0.6249 |
| 336h | 0.3188 | 0.2508 | 0.1018 | 0.1003 | 0.1391 | 0.1410 | 0.3757 |

#### zg500 ACC

| lead | clone v10 | nonoise v10 | v11_clip EMA | 5410 |
|---|---|---|---|---|
| 6h | 0.9985 | 0.9993 | 0.9984 | 0.9994 |
| 72h | 0.9834 | 0.9785 | 0.9783 | 0.9902 |
| 336h | 0.3995 | 0.3054 | 0.3522 | 0.4612 |

### 1.2 Recipe deltas across the four trained models

| run | dataset | input_noise.sigma | max_grad_norm | target | EMA | tas@6h |
|---|---|---|---|---|---|---|
| group_clone | v10 | 0.05 | 0.0 (off) | state | 0.999 | 0.987 |
| group_clone_nonoise | v10 | 0.0 | 32.0 | state | 0.999 | **0.995** |
| group_clone_v11 | **v11** | 0.05 | 0.0 (off) | state | 0.999 | 0.919 |
| group_clone_v11_clip | **v11** | 0.05 | **32.0** | state | 0.999 | 0.958 |

All four runs share `lr=1e-4`, `batch_size=8 GLOBAL`, `betas=(0.9, 0.999)`,
`weight_decay=3e-6`, 50 epochs, cosine + 5-epoch warmup, EMA decay 0.999.

### 1.3 Three core findings

**(a) The dominant regressor is the v10 → v11 dataset switch, not the recipe.**
Holding the recipe fixed (input_noise=0.05, no clip), the dataset change alone
dropped tas@6h from 0.987 to 0.919 (Δ = −0.068) and tas@72h from 0.901 to 0.512
(Δ = −0.39). Clip restoration on the same v11 dataset only recovers ~0.04 of
the 0.07 tas@6h gap.

**(b) Removing input noise improved short-lead tas on v10 (0.987 → 0.995 at
6h, essentially matching SFNO-5410) but cost ~0.07 ACC at 336h.** This refines
[[feedback-input-noise-is-load-bearing]]: noise stabilizes long leads but
hurts short leads — the dial sits on a Pareto frontier, not a fixed optimum.

**(c) The tas regression is channel-asymmetric.** Other channels regress only
by 0.01–0.04 ACC at long leads; tas drops by 0.30 at mid-leads. The asymmetry
implicates the SST → tas boundary pathway, because tas is the field most
directly coupled to SST in PlaSim physics.

### 1.4 Critical confound flagged

**v11-trained models have been evaluated against v10 test data.** All four
v11/v11_clip eval provenance files show:

```
TEST_HOLDOUT=/scratch/.../sim52_zgplev_full/test_holdout       ← v10 (SST clamped at 271.35 K)
TRAIN_DIR  =/scratch/.../sim52_zgplev_full/train               ← v10 (climatology source)
```

The v11 fix removed the 271.35 K sea-ice clamp on SST
(`docs/2026-05-10_sst_sea_ice_handling_fix_plan.md`). v11-trained models
learned to map unclamped-SST boundary inputs → tas outputs. At eval time
they see clamped-SST boundary inputs (OOD over polar oceans) → systematic
tas error proportional to SST coupling. Climatology is also computed from
v10 training data, so the ACC anomaly fields are wrong-baseline for the
model's actual operating distribution.

This may explain a substantial fraction — possibly most — of the v11 tas
regression. The v11_clip EMA report at
`$WORK2/.../sfno_eval/20260513_eval-8b395eb_data-e3c934b_family-sfno_zgplev_group_clone_v11_clip_ckpt-best_ckpt_ema_mp0/`
is therefore an unreliable basis for HPO decisions on the v11 dataset.

## 2. Plan

### Pre-Run 0 (SUBMITTED 2026-05-13, jobs 3115560–63): v11_clip on v11 test set

No new training. Re-evaluate the existing v11_clip EMA checkpoint against
v11 test data and v11 climatology, controlling for the dataset confound
above.

```bash
RUN_DIR="$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_group_clone_v11_clip/plasim_sim52_zgplev_group_clone_v11_clip/0"
CKPT="$RUN_DIR/training_checkpoints/best_ckpt_ema_mp0.tar"
TEST_HOLDOUT="$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full_v11/test_holdout"
TRAIN_DIR="$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full_v11/train"
RUN_TAG="20260513_v11_clip_on_v11_testset_ema"
MODE=nwp scripts/submit_eval.sh
```

OUT_ROOT: `$WORK2/SFNO_Climate_Emulator/results/sfno_eval/20260513_v11_clip_on_v11_testset_ema/`

**Decision gate:**
- **CASE A (likely): tas@6h ≥ 0.985 AND tas@72h ≥ 0.85 AND zg500@336h ≥ 0.35.**
  The v11 tas "regression" was largely eval-data mismatch. v11_clip is the
  new own-track production candidate. Adopt; cancel HPO sweep; document.
- **CASE B: tas@6h ∈ [0.92, 0.985] OR tas@72h ∈ [0.60, 0.85].** Partial
  confound. Proceed to Run 1.
- **CASE C: tas@6h < 0.92 AND tas@72h < 0.60 (regression unchanged).**
  Recipe is genuinely broken on v11. Proceed to Run 1.

In all cases, future v11 evals MUST use the v11 test holdout going forward.

### Run 1 (gated on Pre-Run 0 = CASE B/C): v11_noise025_clip

Single-knob delta vs v11_clip: halve input noise.

| knob | value | rationale |
|---|---|---|
| `input_noise.sigma` | **0.025** | half of v11_clip's 0.05; rough match for the v10-physical-noise scale on the v11 unclamped-SST dataset |
| `optimizer_max_grad_norm` | 32.0 | keep |
| `target` | `state` | keep (state was fine on v10) |
| `ema.enabled, decay, warmup` | true, 0.999, true | keep |
| `lr, batch_size, betas` | 1e-4, 8, (0.9, 0.999) | keep |
| `max_epochs`, scheduler | 50, cosine + 5-epoch warmup | keep |
| dataset | v11 | keep |

Files to create (under fresh, non-reused EXP_DIR per
[[feedback-protect-prior-runs]]):

- `src/sfno_training/config/plasim_sim52_zgplev_group_clone_v11_noise025_clip.yaml`
- `src/sfno_training/submit_zgplev_group_clone_v11_noise025_clip.slurm`
- EXP_DIR: `$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_group_clone_v11_noise025_clip/` (must not exist)

Submitter mirrors `submit_zgplev_group_clone_v11_clip.slurm` with the new
config name and a fresh EXP_DIR. Carries the
`OUTPUT_ROOT=$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full_v11` default.

Eval must use the v11 test set:

```bash
RUN_TAG=20260514_eval_v11_noise025_clip_ema MODE=nwp \
TEST_HOLDOUT=$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full_v11/test_holdout \
TRAIN_DIR=$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full_v11/train \
scripts/submit_eval.sh
```

### Run 2 (held in reserve, gated on Run 1 outcome): v11_noise_zero_clip

Single-knob delta from v11_clip: zero noise. Anchors the sigma=0 floor on
the v11 dataset. Worth running only if Run 1 shows tas@6h still < 0.99 but
sigma direction looks right.

```yaml
input_noise.sigma: 0.0
optimizer_max_grad_norm: 32.0       # keep
# all else identical to v11_clip
```

With Run 1, this gives 3 sigma points {0.0, 0.025, 0.05} on the v11 dataset
— enough to choose a production value without further sweeping.

## 3. Why these runs are maximally informative

- **Pre-Run 0** eliminates the eval-data mismatch confound at near-zero cost
  (~1 h of compute; no training). It is the highest-information cheapest
  action available; doing it first prevents a multi-run sweep on a
  misleading signal.
- **Run 1 (sigma=0.025)** interpolates between v10-clone (0.05, good long-
  lead) and v10-nonoise (0.0, best short-lead) on the v11 dataset. Single
  knob change from a known operating point. Highest expected information
  per training run.
- **Run 2 (sigma=0)** completes the sigma response curve at 3 points. With
  Run 1, fully characterizes sigma on v11 — no further sweep needed.

Deliberately **not** included in this plan:

- Target=residual swap: orthogonal axis; defer until sigma is pinned.
- EMA decay sweep: raw ≈ EMA already shown on v11_clip
  (Δ ACC ≤ 0.009, Δ RMSE ≤ 1%); not a productive lever.
- LR / batch_size sweep: already optimized on v10
  (`docs/2026-05-08_hpo_knob_inventory.md`, GB decision).
- Per-channel input_noise (exclude SST/tas): too many degrees of freedom
  to learn from a single run; revisit only if sigma sweep is inconclusive.

## 4. Comparison set per run

| New run | Direct A/B vs | Pareto reference |
|---|---|---|
| Pre-Run 0 | v11_clip EMA on v10 test (existing report) | clone v10 on v10 test; SFNO-5410 96-IC |
| Run 1 (sigma=0.025) | v11_clip EMA on **v11** test (Pre-Run 0) | clone v10 (sigma=0.05); nonoise v10 (sigma=0) |
| Run 2 (sigma=0.0) | Run 1 (sigma=0.025) on **v11** test | nonoise v10 |

All comparisons on the v11 test set going forward.

## 5. Adoption / rejection criteria

### Pre-Run 0
- **Adopt v11_clip as canonical, cancel HPO:** tas@6h ACC ≥ 0.985 AND
  tas@72h ACC ≥ 0.85 AND zg500@336h ACC ≥ 0.35.
- **Proceed to Run 1:** any of the three thresholds missed.

### Run 1 (v11_noise025_clip)
- **Adopt:** tas@6h ≥ 0.99 AND tas@336h ≥ 0.20 AND zg500@336h ≥ 0.38.
  Write up as new own-track production default for v11.
- **Run Run 2:** tas@6h ∈ [0.96, 0.99) AND improved over v11_clip's 0.958
  by ≥ 0.015 (sigma direction is right; need lower).
- **Reject and pivot axis:** tas@6h ≤ 0.958 (no improvement from halving
  sigma). Sigma is not the lever; next axis = `target=residual` or revisit
  v11 normalization stats.

### Run 2 (v11_noise_zero_clip)
- **Adopt sigma=0:** tas@6h ≥ 0.99 AND tas@336h ≥ 0.20 (matching v10
  nonoise short-lead AND beating its long-lead — only likely if v11
  normalization fundamentally helps long-lead too).
- **Reject:** tas@336h < 0.18 (worse than v10 nonoise; confirms noise is
  needed for long-lead on v11). Final sigma choice = the better of Runs 1
  vs v11_clip on long-lead.

## 6. Skill / config changes required

1. **`scripts/submit_eval.sh` RUN_TAG suffix asymmetry.** Currently the
   script appends `_ckpt-best_ckpt_ema_mp0` for EMA but suppresses the
   suffix for raw `best_ckpt_mp0` (treated as legacy default). Now that
   EMA is canonical per [[feedback-ema-is-canonical-ckpt]], this is
   inverted. Fix: always append `_ckpt-<basename>` regardless of flavor.
   Eliminates the silent collision risk when re-evaluating same RUN_DIR
   with different ckpt flavors. No effect on existing OUT_ROOTs.

2. **`.claude/skills/eval-sfno-own/SKILL.md` — add dataset-version
   guidance.** Insert a paragraph under §Inputs noting that `TEST_HOLDOUT`
   and `TRAIN_DIR` defaults are v10; v11-trained models MUST override to
   their matching v11 paths, or evaluation is invalid (per the confound
   documented above). Better still: auto-derive test/train from
   `RUN_DIR`'s family suffix or read it from the run's `config.json`
   `OUTPUT_ROOT`.

3. **`.claude/skills/eval-sfno-own/SKILL.md` — re-eval workflow note.**
   Document that re-evaluating an existing checkpoint with a different
   `TEST_HOLDOUT` or different ckpt flavor on the same RUN_DIR requires
   passing `RUN_TAG=<purpose>` to avoid the RUN_TAG-collision guard. Cite
   this Pre-Run 0 invocation as the canonical example.

4. **Memory update — `feedback-input-noise-is-load-bearing`.** Soften the
   wording from "don't propose removing input_noise.sigma=0.05" to "reducing
   or removing input noise can recover short-lead skill but at a documented
   long-lead cost. Both directions are evidence-supported (v10 nonoise:
   tas@6h ≈ 0.995, tas@336h ≈ 0.25 vs clone tas@6h ≈ 0.987, tas@336h ≈ 0.32).
   Treat sigma as a Pareto-frontier dial tunable per dataset, not a fixed
   constant."

5. **Memory keep — `feedback-ema-is-canonical-ckpt`.** No change. v11_clip
   raw vs EMA differ by ≤ 0.009 ACC across all (channel, lead) cells —
   EMA stays canonical.

6. **Memory keep — `feedback-protect-prior-runs`.** Runs 1 and 2 each get
   a fresh EXP_DIR; do NOT reuse `sfno_zgplev_group_clone_v11_clip/` or
   any other existing family dir.

## Cross-references

- Plan history: `docs/2026-05-12_v11_clip_restore_plan.md` (the previous
  iteration; this plan supersedes it for what to run NEXT).
- v11 dataset rationale: `docs/2026-05-10_sst_sea_ice_handling_fix_plan.md`.
- HPO knob inventory: `docs/2026-05-08_hpo_knob_inventory.md`.
- GB decision: [[project-zgplev-gb-decision]] (GB=8 retained for own-track).
- Skill: `.claude/skills/eval-sfno-own/SKILL.md`.
- v11_clip experiment context: [[project-v11-clip-experiment]].
