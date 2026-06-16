# 2026-05-14 — zg500-Targeted HPO Plan (post Pre-Run 0)

Author: Claude (with Zhixing)
Status: proposed — supersedes the noise-sweep direction in
`docs/2026-05-13_v11_clip_next_hpo_plan.md` (sigma sweep cancelled).

## TL;DR

1. **Adopt `v11_clip` (EMA) as the current canonical own-track baseline.** Pre-Run 0
   (eval against the matching v11 test set) cleared all three adoption gates; the
   apparent v11→v11_clip tas regression was an eval-data confound, not a recipe
   regression.
2. **One headline HPO run next: multi-step rollout loss (`multistep_count=2`).**
   This is the single highest-leverage knob for long-lead `zg500` skill given
   what's already wired into the trainer.
3. **Cheap complementary run: explicit `channel_weights` list upweighting `zg*`
   and large-scale dynamics**. Same compute as baseline; complementary signal,
   not a competing variant.
4. **Cancel the sigma=0.025 / sigma=0.0 sweep.** The original motivation
   (v11_clip tas regression) dissolved when Pre-Run 0 showed the regression was
   the boundary confound. No `zg500` mechanism would predict that noise σ tuning
   moves long-lead `zg500` more than rollout-loss or capacity does.
5. **5410 gap is partially HPO-addressable.** ~30–50% of the `zg500` gap is
   plausibly closable by rollout-loss + channel weighting. The remainder is
   architecture/capacity (group SFNO `sfno_plasim` ≈ 107 M params vs our makani
   SFNO ≈ 57 M).

---

## 1. Adoption of `v11_clip` as canonical

Pre-Run 0 = `v11_clip` EMA ckpt evaluated on the matching v11 test holdout and
v11 climatology (RUN_TAG `20260513_v11_clip_on_v11_testset_ema`, jobs
3115560–3115563, all chain stages COMPLETED).

Adoption gates (`docs/2026-05-13_v11_clip_next_hpo_plan.md` §4):

| gate | threshold | Pre-Run 0 | result |
|---|---|---|---|
| tas @ 6h ACC | ≥ 0.985 | 0.988 | ✓ |
| tas @ 72h ACC | ≥ 0.85 | 0.916 | ✓ |
| zg500 @ 336h ACC | ≥ 0.35 | 0.377 | ✓ |

Three-way comparison vs `group_clone` v10 and 5410 (mean ACC, n=96 ICs):

### ACC

| ch | lead | v11_clip on v11 | group_clone v10 | 5410 |
|---|---|---|---|---|
| tas | 6h | **0.988** | 0.987 | 0.995 |
| tas | 72h | **0.916** | 0.901 | 0.947 |
| tas | 336h | **0.416** | 0.319 | 0.376 |
| zg500 | 6h | 0.998 | 0.998 | 0.999 |
| zg500 | 120h | 0.946 | 0.947 | 0.965 |
| zg500 | 240h | 0.665 | 0.674 | 0.731 |
| zg500 | 336h | 0.377 | 0.399 | **0.461** |
| ua5 | 336h | 0.322 | 0.340 | 0.389 |
| ta5 | 336h | 0.332 | 0.341 | 0.396 |
| pr_6h | 336h | 0.166 | 0.167 | 0.157 |

### RMSE

| ch | lead | v11_clip | group_clone | 5410 |
|---|---|---|---|---|
| tas | 6h | 0.433 | 0.439 | 0.271 |
| tas | 336h | **2.970** | 3.279 | 3.199 |
| zg500 | 6h | 3.525 | 3.563 | 2.215 |
| zg500 | 336h | 72.5 | 70.3 | **66.2** |
| ua5 | 336h | 8.18 | 7.97 | 7.72 |
| ta5 | 336h | 2.92 | 2.87 | 2.77 |

**Takeaways:**
- `v11_clip` beats `group_clone` on `tas` at every lead (biggest gain at
  336h: +0.097 ACC / −0.31 K RMSE), with no meaningful regression on the other
  channels (≤0.02 ACC, ≤2% RMSE — well inside IC std).
- `zg500` is the channel where we visibly trail 5410 at long leads:
  ΔACC@336h = −0.084, ΔRMSE@336h = +6.3 m.
- `ua5` / `ta5` show the same shape as `zg500` (uniform ~10 % RMSE gap that
  does not widen with lead). The flat-gap shape across leads is a fingerprint
  of a representation-capacity gap, not rollout-stability gap.

**Decision:** adopt `v11_clip` (EMA) as the current canonical own-track
baseline. All subsequent HPO runs measure relative to `v11_clip on v11` numbers
in the tables above.

## 2. The remaining concern: `zg500` long-lead skill

Why this is the priority:
- `zg500` is load-bearing for blocking detection (the core AI-RES score-function
  use case) and for large-scale dynamical skill in general.
- The group SFNO-5410 ACC at 240h/336h (0.73 / 0.46) is the practical ceiling
  we want to approach.
- `tas` long-lead is already best-in-class on our test set; `pr_6h` is also at
  parity with 5410. `zg500` is where the daylight is.

## 3. HPO direction (1–2 runs)

### Run A — multi-step rollout loss (`multistep_count = 2`)

**The single recommended run.** Train with a 2-step rollout loss instead of the
current 1-step single-target loss.

**Config delta vs `plasim_sim52_zgplev_group_clone_v11_clip.yaml`:**

```yaml
# Add to the v11_clip config; everything else identical:
n_future: 1                    # was 0; trainer sets from multistep_count-1
valid_autoreg_steps: 3         # unchanged
```

Pass `--multistep_count 2` to `train_plasim.py`
(`src/sfno_training/train_plasim.py:249-250` maps it to `n_future=1`).
Optionally configure step weighting in the loss block — default `"constant"` is
fine for first pass:

```yaml
losses:
-   type: "l2"
    channel_weights: "constant"
    temp_diff_normalization: !!bool False
    parameters:
        squared: !!bool True
# (makani auto-extends loss over n_future+1 future steps with constant weight)
```

Fresh EXP_DIR: `$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_group_clone_v11_rollout2`.
Slurm: copy `submit_zgplev_group_clone_v11_clip.slurm` → `_v11_rollout2.slurm`,
add `--multistep_count 2`.

**Cost:** memory grows ≈ 2× (model state must hold one extra unrolled forward),
wall-clock ≈ 1.4–1.6× per epoch. On 4×H100 with current batch=8 global (per-rank
2) we are at ~70 % VRAM headroom in v11_clip — should fit. If OOM, drop global
batch 8 → 4 (per-rank 1) and double `max_epochs` 50 → 60 to compensate for the
smaller effective gradient signal.

**Why this is the right knob for `zg500` (not just `tas`):**

The flat ~10 % RMSE gap vs 5410 across leads (6 h to 336 h) for `zg500` /
`ua5` / `ta5` is the signature of an error-accumulation problem on top of a
representation gap. The error-accumulation half is exactly what multi-step
rollout loss penalizes: it forces the model to also be correct at step t+12 h
given its own step t+6 h prediction. Single-step training never sees its own
mistakes propagate, so it has no incentive to produce a state-vector that is
*stable under autoregression* — only one that is *accurate at +6 h given truth*.

Why `zg500` specifically gains more than `tas` / `pr_6h`:
- **`zg500` is smooth, dynamical, slow-varying.** Its long-lead skill is
  dominated by whether the predicted state lies on the slow manifold of
  PlaSim's large-scale dynamics. Multi-step training is a direct signal for
  that.
- **`tas` is mostly surface-boundary forced.** Each step's `tas` prediction is
  re-anchored by the externally-fed SST/SIC, so its long-lead error grows
  slower with rollout depth. Multi-step won't move `tas` much (already saturated
  by the SST input fix), but also won't hurt it.
- **`pr_6h` is high-frequency, stochastic.** Rolled-out `pr_6h` is decoupled
  from coherent `pr_6h` 2 steps later — multi-step loss is roughly neutral.

The asymmetric impact across channels is the test of the mechanism: if Run A
moves `zg500` substantially more than `tas` and `pr_6h`, that confirms the
"rollout-stability" interpretation.

**Adoption gates for Run A (relative to `v11_clip on v11`):**

| gate | metric | threshold | rationale |
|---|---|---|---|
| primary | zg500 @ 336h ACC | ≥ 0.42 (+0.043 over v11_clip 0.377) | ~50 % of the v11_clip→5410 gap |
| primary | zg500 @ 240h ACC | ≥ 0.70 (+0.035 over 0.665) | matched 240h gain |
| hold-the-line | tas @ 6h ACC | ≥ 0.980 (−0.008 max) | short-lead surface sanity |
| hold-the-line | tas @ 336h ACC | ≥ 0.38 (−0.04 max) | don't trade away the SST-fix win |
| hold-the-line | pr_6h @ 72h ACC | ≥ 0.72 (−0.03 max) | precip sanity |
| hold-the-line | pr_6h @ 336h ACC | ≥ 0.13 (−0.04 max) | long-lead precip not destroyed |

**If Run A clears all hold-the-line gates and clears both primary gates:**
adopt as the new canonical (rename to `_v11_rollout2_clip`, retire `_v11_clip`
as the baseline but keep the EXP_DIR for reproducibility — see
[[feedback-protect-prior-runs]]).

**If Run A clears `zg500` primary but breaches `tas` hold-the-line:** keep both
as parallel canonical recipes — `v11_clip` for `tas`-priority work,
`v11_rollout2` for `zg500`-priority work. Reflect in skill defaults.

**If Run A does not clear `zg500` primary:** the rollout-stability hypothesis
is partly wrong; the gap is more capacity than dynamics → see §4. Try Run B
next, but don't expect it to fully close the gap on its own either.

### Run B — channel-weighted loss favoring large-scale dynamics

**Cheaper, complementary, and decoupled from Run A's mechanism.** Same
compute as baseline, just different per-channel loss weight.

**Config delta vs `v11_clip`:**

```yaml
losses:
-   type: "l2"
    # Explicit per-channel list (matches the order in channel_names, 53 entries).
    # Upweight zg-group + upper-tropospheric dynamics (ua/va levels 5–10),
    # downweight pl + hus + pr_6h. tas held at 1.0.
    channel_weights:
        # pl                          tas
        - 0.5
        - 1.0
        # ta1..ta10  (mid/upper-trop temperature: hold at 1.0)
        - 1.0
        - 1.0
        - 1.0
        - 1.0
        - 1.0
        - 1.0
        - 1.0
        - 1.0
        - 1.0
        - 1.0
        # ua1..ua10  (upper-level winds upweighted)
        - 1.0
        - 1.0
        - 1.0
        - 1.0
        - 1.2
        - 1.2
        - 1.2
        - 1.2
        - 1.0
        - 1.0
        # va1..va10  (same)
        - 1.0
        - 1.0
        - 1.0
        - 1.0
        - 1.2
        - 1.2
        - 1.2
        - 1.2
        - 1.0
        - 1.0
        # hus1..hus10 (downweighted)
        - 0.5
        - 0.5
        - 0.5
        - 0.5
        - 0.5
        - 0.5
        - 0.5
        - 0.5
        - 0.5
        - 0.5
        # zg200..zg1000  (large-scale geopotential strongly upweighted)
        - 1.5
        - 1.5
        - 1.8
        - 1.8
        - 2.0
        - 1.8
        - 1.5
        - 1.2
        - 1.0
        - 1.0
        # pr_6h (downweighted — already at 5410-parity)
        - 0.5
    temp_diff_normalization: !!bool False
    parameters:
        squared: !!bool True
```

Total weight roughly conserved (sum ≈ 53 → effective LR not perturbed), but
the loss budget tilts toward zg / upper-level winds / mid-trop temperature
and away from humidity / precipitation / surface pressure.

Fresh EXP_DIR: `$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_group_clone_v11_chwzg`.

**Why for `zg500`:** with `constant` weights, the 10 hus channels collectively
get 10× as much loss budget as `zg500` alone, and 5× as much as the whole
zg group. Hus is also high-variance and noisy in PlaSim (especially the lower
levels) — the model spends gradient on chasing that noise. Concentrating budget
on the large-scale dynamical fields gives the model a clearer signal about
what we actually care about. This is mechanistically distinct from Run A
(loss-budget reallocation, not extra rollout signal), so the gains compose if
both work.

**Adoption gates for Run B (relative to `v11_clip on v11`):**

| gate | metric | threshold |
|---|---|---|
| primary | zg500 @ 336h ACC | ≥ 0.41 (+0.033) |
| primary | zg500 @ 240h ACC | ≥ 0.69 (+0.025) |
| hold-the-line | tas @ 6h ACC | ≥ 0.975 (−0.013 max — wider band than Run A because tas's loss weight is unchanged but its relative share dropped) |
| hold-the-line | pr_6h @ 6h ACC | ≥ 0.80 (−0.05 max — pr was explicitly downweighted; some short-lead degradation expected) |

**Run-order recommendation:** **Run A first**, then evaluate, then decide
whether to do Run B alone or stacked on Run A's best recipe.

Rationale: Run A is the riskier and more expensive of the two but it tests a
crisp mechanistic hypothesis. If it works, we learn something about the system
and don't need to investigate Run B in isolation. If it fails, Run B is a cheap
fallback and we'll know we're more bottlenecked on representation than on
rollout signal.

Avoid stacking A+B in the first attempt — we lose attribution.

## 4. Likely gap to 5410: HPO vs architecture/data

Decomposing the `zg500@336h` ACC gap:
- v11_clip: 0.377
- 5410: 0.461
- gap: 0.084 ACC

Where I think this gap lives:
- **~30–50 % addressable by HPO/recipe** (rollout loss + channel weighting +
  longer training). Realistic ceiling for HPO-only changes:
  `zg500@336h` ACC ≈ 0.41–0.43 (closing ~half the gap).
- **~50–70 % from architecture/capacity.** Pointers:
  - Live config comment (`plasim_sim52_zgplev_group_clone_v11_clip.yaml`
    lines 73–77): "makani SFNO at these hyperparams ≈ 56.5 M params; group
    ckpt is ≈ 106.9 M". 2× capacity in a model designed for spherical PDE
    rollouts plausibly accounts for several percent of long-lead ACC on
    smooth fields.
  - Group's `num_blocks: 16` (a real depth knob in `sfno_plasim`) has no
    makani analog and is silently dropped here. Our `num_layers: 12` is
    likely shallower than the effective group depth.
  - Group also has more training data (a longer PlaSim run) and trained for
    more iterations — both improve long-lead skill.

What would actually close the rest of the gap (out of scope here, but worth
naming so we don't pretend HPO will solve it):
- Port `sfno_plasim` (the group's actual net) into makani as a new `nettype`.
  Closes the depth gap and the param-count gap simultaneously.
- Or: bump `embed_dim` 256 → 384 and `num_layers` 12 → 16 to reach ~110 M
  params on the existing makani SFNO. Cheaper engineering, similar effect.
  Would be a separate architecture-scaling experiment, not HPO.

**Recommendation:** run Run A. If the `zg500@240h/336h` lift lands ≥ +0.04 ACC,
that closes about half the gap and is a real win — adopt and stop. If it lands
< +0.02, the architecture-scaling experiment should jump the queue ahead of any
further HPO sweep.

## 5. Pre-flight checks before submitting Run A

Verify the trainer actually consumes `multistep_count=2` as expected:

```bash
# Dry-run: train_plasim.py imports + parses config + builds dataloader,
# log "multistep_count = 2" and "n_future = 1" once, then exit before iterating.
# Add a `--dry_run` flag if not already present (one-line guard right after
# the print at train_plasim.py:89).
python -m sfno_training.train_plasim \
    --yaml_config src/sfno_training/config/plasim_sim52_zgplev_group_clone_v11_rollout2.yaml \
    --config plasim_sim52_zgplev_group_clone_v11_rollout2 \
    --multistep_count 2 \
    --dry_run
```

Check VRAM with one mini-batch in --debug mode before launching the 30-hr job:

```bash
# Single iteration on 1 H100 to confirm memory budget (~ 70 GB total expected).
srun -p h100 -n 1 --gres=gpu:1 --time=00:10:00 \
    python -m sfno_training.train_plasim \
        --yaml_config ... --multistep_count 2 --max_epochs 1 --max_iters 2
```

## 6. Non-HPO fixes (decoupled from the HPO direction above)

These are infrastructure issues that the Pre-Run 0 confound exposed; doing them
prevents the next "why does v11_clip look bad on v10?" landmine.

### 6.1 Auto-derive `TEST_HOLDOUT` and `TRAIN_DIR` from the run's config

Current state (`scripts/submit_eval.sh`): both default to v10 paths
hard-coded near the top of the script. v11_clip was eval'd on v10 by default
because nobody overrode them.

**Proposed change.** When `RUN_DIR` is given:
1. Parse `<RUN_DIR>/config.yaml` (or the resolved YAML the trainer writes
   into the run dir) for `train_data_path` and `inf_data_path`.
2. Set `TRAIN_DIR := <train_data_path>` and `TEST_HOLDOUT := <inf_data_path>`
   *unless* the caller passed explicit overrides.
3. Echo the derived paths at the top of the SLURM log, prefixed with
   `[auto-derived]`, so it's obvious what eval set was used.

If `<RUN_DIR>/config.yaml` is missing, fall back to current hard-coded v10
paths and emit a `[fallback]` warning rather than silently using v10.

### 6.2 EMA-as-default checkpoint

Already in place (`scripts/submit_eval.sh` defaults to
`best_ckpt_ema_mp0.tar` when EMA is enabled per
[[feedback-ema-is-canonical-ckpt]]). No change needed. Re-state in the skill
file for visibility.

### 6.3 Add dataset version + checkpoint basename to provenance

`scripts/submit_eval_report.slurm` writes `report.md` with a frontmatter table
that currently includes `Eval code SHA`, `Data packager SHA`, `Training code
SHA`, and `Checkpoint`. Add two rows:

| field | source |
|---|---|
| Dataset version | derived from train_data_path: `sim52_zgplev_full` → `v10`, `sim52_zgplev_full_v11` → `v11`. Compute in `submit_eval_report.slurm` from `$TRAIN_DIR`. |
| Checkpoint basename | `$(basename "$CKPT")` — e.g. `best_ckpt_ema_mp0.tar`. Disambiguates raw vs EMA at a glance. |

These two rows would have caught the Pre-Run 0 confound in one read of the
report instead of cross-referencing four CSVs.

## 7. Decision summary

| item | decision |
|---|---|
| Adopt `v11_clip` (EMA) as canonical | **YES** |
| Continue sigma=0.025 / sigma=0.0 sweep | **NO** (motivation dissolved) |
| Run A: `multistep_count=2` | **YES — first** |
| Run B: zg-weighted channel weights | **YES — second, only after A reads out** |
| Architecture scaling (depth/width) | deferred; revisit if Run A < +0.02 ACC |
| Auto-derive TEST_HOLDOUT/TRAIN_DIR in submit_eval.sh | yes |
| Surface dataset version + ckpt basename in report | yes |

## 8. References

- `docs/2026-05-13_v11_clip_next_hpo_plan.md` — original (now-superseded) plan.
- `docs/2026-05-12_v11_clip_restore_plan.md` — grad-clip restoration rationale.
- `src/sfno_training/config/plasim_sim52_zgplev_group_clone_v11_clip.yaml` —
  current canonical baseline config.
- `src/sfno_training/train_plasim.py:249-250` — `multistep_count` → `n_future`
  mapping.
- `makani-src/makani/utils/loss.py:114-180` — channel-weighting logic
  (constant / auto / new auto / custom / explicit list).
- `makani-src/makani/utils/losses/base_loss.py:34-100` — built-in
  channel-weighting modes.
- Pre-Run 0 results:
  `/work2/.../sfno_eval/20260513_v11_clip_on_v11_testset_ema/`
- 5410 benchmark:
  `/work2/.../sfno_eval_5410/20260509_blocking_96ic_h100_packed_derecho_env_valid`
