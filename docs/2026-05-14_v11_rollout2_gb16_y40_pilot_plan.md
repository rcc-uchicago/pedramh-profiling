# 2026-05-14 — v11 rollout2 GB16 40y Screening Pilot Plan

Author: Claude (with Zhixing)
Status: proposed; awaits launch confirmation after preflight.
Supersedes: §3 Run A in `docs/2026-05-14_zg500_targeted_hpo_plan.md` (the
full 100-yr `multistep_count=2` run is now gated on this pilot).

## TL;DR

Cheap screening run that asks **one question**: does `multistep_count=2` improve
zg500 enough at v11/GB16/40y to justify a full v11 100-yr rollout2 training run?

- **Variant name:** `plasim_sim52_zgplev_v11_rollout2_gb16_y40_pilot`
- **Branch recipe from:** `group_clone_v11` (no-clip), via the `gbhpo40_gb16_lr2_0e-4`
  GB16 40y baseline (recipe-identical to v11 at GB16 scaling).
- **Dataset axis:** v11 throughout (per user decision 2026-05-14).
  Requires building a v11 40-yr subset first, since none exists.
- **Single config delta vs `gbhpo40_gb16_lr2_0e-4.yaml`:** `n_future: 1`.
  Plus `--multistep_count 2` in the slurm.
- **Preflight first:** confirms (a) memory at GB16 + multistep=2 + bf16 +
  checkpointing_level=2, (b) launch banner shows global_batch_size=16 and
  per-rank=4.
- **No HPO promotion off this pilot — only a go/no-go on a full v11 100y rollout2.**

## 1. Dependency: v11 40-yr subset (not built yet)

Only a v10 40-yr subset exists today:
`/scratch/.../data/makani/sim52_zgplev_40yr_y12-51_20260511`.

To run the pilot v11-throughout (per user decision), build:
`/scratch/.../data/makani/sim52_zgplev_40yr_y12-51_v11_20260514`.

Good news on prerequisites:
- The v11 boundary (`/scratch/.../data/boundary_astro_v11/sim52`) **already
  exists** — produced for the v11 100-yr farm in 2026-05-12. **The emulator
  adaptor does not need to re-run.** Only the packager runs.
- The v11 100y farm's directory layout
  (`sim52_zgplev_full_v11/{train,valid,test,test_holdout,stats,metadata,config}`)
  is the template for the 40-yr output.

### Data-prep work (Phase 0)

Steps, mirroring how `sim52_zgplev_40yr_y12-51_20260511` was built:

1. Submit packager with year range 12–51:
   ```bash
   YEARS_START=12 YEARS_END=51 \
   BOUNDARY_ROOT=$SCRATCH/SFNO_Climate_Emulator/data/boundary_astro_v11 \
   POSTPROC_SOURCE_DIR=<same as v11 100y> \
   OUTPUT_ROOT=$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_40yr_y12-51_v11_20260514 \
   SST_MODE=surface \
   sbatch src/plasim_makani_packager/submit_loop.slurm
   ```
   Override `--sst-mode surface` is the v11 convention (no clamp). Override
   any `SST_LAND_FILL_K` to match the v11 100y `sst_land_fill_K` attr if set.
2. After packager completes, build per-channel stats
   (`stats/{global_means,global_stds,time_means,forcing_global_means,
   forcing_global_stds,forcing_time_means}.npy`) and `metadata/data.json`
   from the resulting H5 files. Match v11 100y procedure.
3. Verify by inspecting any train H5 file's `attrs.sst_mode == "surface"`.

Estimated wall: 2–4 hours on skx (parallel year-level packager). No GPU.

If skipped (i.e. user reverts to "v10 throughout"), pilot can launch
immediately on the existing v10 40y subset. Plan §2 below stays identical
in recipe; only the dataset paths change.

## 2. Pilot config — `plasim_sim52_zgplev_v11_rollout2_gb16_y40_pilot.yaml`

**Branched from `plasim_sim52_zgplev_gbhpo40_gb16_lr2_0e-4.yaml`**. The
gbhpo40 baseline already inherits the v11 recipe (no-clip, σ=0.05 input
noise, β2=0.999, EMA decay 0.999 with warmup, weight_decay 3e-6,
CosineAnnealingLR with 5-epoch warmup, channel_weights="constant").

**Single delta from gbhpo40_gb16_lr2_0e-4:**

| field | gbhpo40 baseline | pilot |
|---|---|---|
| `n_future` | 0 | **1** |

All other fields **unchanged from gbhpo40_gb16_lr2_0e-4**, including:

| field | value | note |
|---|---|---|
| `lr` | 2.0e-4 | linear scaling 1e-4×(16/8) from v11 GB8 |
| `batch_size` (global) | 16 | per-rank 4 at 4-GPU DDP |
| `weight_decay` | 3.0e-6 | unchanged |
| `optimizer_beta2` | 0.999 | unchanged |
| `optimizer_max_grad_norm` | 0.0 | unchanged (matches 5410 and v11) |
| `max_epochs` | 50 | unchanged |
| `scheduler` | CosineAnnealingLR | unchanged |
| `lr_warmup_steps` | 5 | unchanged |
| `lr_start` (LinearLR factor) | 1e-4 | unchanged |
| `scheduler_min_lr` | 1e-8 | unchanged |
| `input_noise.sigma` | 0.05 | unchanged |
| EMA `decay` | 0.999 | unchanged |
| EMA `warmup` | True | unchanged |
| `channel_weights` | "constant" | unchanged (no per-channel weighting) |
| `losses[0].type` | l2 (squared=True) | unchanged |
| `prediction_type` | iterative | unchanged |
| `valid_autoreg_steps` | 3 | unchanged |
| `prefetch_factor` | 4 | unchanged |
| `persistent_workers` | True | unchanged |
| `ema_validation_period` | 1 | unchanged |

**Dataset path change (templated, resolved by slurm):**

| `{{OUTPUT_ROOT}}` | `$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_40yr_y12-51_v11_20260514` |

## 3. Slurm — `submit_zgplev_v11_rollout2_gb16_y40_pilot.slurm`

**Branched from `submit_zgplev_gbhpo40_gb16_lr2_0e-4.slurm`** (h100, 4 GPUs,
bf16, checkpointing_level=2, protect-prior-runs guard).

**Single delta:**

| field | gbhpo40 slurm | pilot slurm |
|---|---|---|
| `--multistep_count` | 1 | **2** |
| `-t` (wall time) | 06:00:00 | **08:00:00** (multistep~1.4× per-epoch) |
| preflight `--multistep-count` *(new flag — see §5)* | (absent) | **2** |

Default env overrides:
```
OUTPUT_ROOT=$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_40yr_y12-51_v11_20260514
EXP_DIR=$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_v11_rollout2_gb16_y40_pilot_20260514
```

**Fresh EXP_DIR** — never resume into prior gbhpo40 or v11 dirs
(per [[feedback_protect_prior_runs]]).

## 4. Preflight & memory budget

GB16 multistep=1 fits cleanly on H100 with bf16 + checkpointing_level=2
(the existing gbhpo40 baselines proved this). Multistep=2 ≈ doubles
forward activations. With checkpointing_level=2 enabled, the activation
memory delta should land in the 50–80% range (not full 2×), keeping us
within 80 GB.

### Memory contingency

If preflight OOMs at GB16 + multistep=2:

- **Fallback A:** drop to global batch 8 (per-rank 2), lr scales back to
  1e-4 (square-root-of-LR-scale-down from GB16). Same dataset, same
  recipe, multistep=2 stays. Pilot still valid; comparability to GB16 40y
  rollout1 baseline gets one extra confound (GB).
- **Fallback B:** keep GB16, drop `multistep_count` to 2 → 1.5 by using
  step-weighting (n_future=1 with the second step weight=0.5). Not
  supported by makani's current loss config out of the box; needs a
  one-line addition under `losses[0].parameters`. Skip unless A also fails.
- **Fallback C:** keep GB16 + multistep=2, increase `checkpointing_level`
  2 → 3. Slower (more recompute) but tighter on memory.

## 5. Preflight script change (small CLI patch)

`scripts/preflight.py:151` currently hardcodes `params["multistep_count"] = 1`
so the memory probe never sees the actual multi-step forward graph. Patch:

```python
# In preflight.py, replace line ~151:
#   params["multistep_count"] = 1
# with:
   params["multistep_count"] = multistep_count  # parameterized
```

Plus add the matching CLI flag and parameter to the `_run_preflight` entry
point. The patch is mechanical and ≤10 lines.

Without this patch the preflight passes trivially (it tests a different
problem than what we'll actually train), which defeats the purpose.

## 6. Eval setup (post-training)

| field | value |
|---|---|
| `RUN_DIR` | `$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_v11_rollout2_gb16_y40_pilot_20260514/plasim_sim52_zgplev_v11_rollout2_gb16_y40_pilot/0` |
| `CKPT` | `$RUN_DIR/training_checkpoints/best_ckpt_ema_mp0.tar` (EMA, per [[feedback_ema_is_canonical_ckpt]]) |
| `TEST_HOLDOUT` | `$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full_v11/test_holdout` (the v11 100-yr test_holdout; matched dataset) |
| `TRAIN_DIR` (climatology source) | `$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full_v11/train` (v11 100-yr, **same as Pre-Run 0/0b**) |
| `RUN_TAG` | `20260514_v11_rollout2_gb16_y40_pilot_on_v11_testset_ema` (explicit override) |

**Why these choices:**
- TEST_HOLDOUT and climatology source **exactly match Pre-Run 0/0b**, so the
  pilot's scorecard is directly comparable to v11 100y GB8 (Pre-Run 0b)
  without any test-side or climatology-side adjustment. Same ACC denominator.
- This means the **primary comparison** in the user's decision rule is
  cross-dataset-size (40y vs 100y) but on identical v11 SST and identical
  climatology — only the multistep axis (1 vs 2) and dataset size (40y vs 100y)
  differ.

### Comparison baselines

| baseline | role | source |
|---|---|---|
| v11 GB8 multistep=1 100y (EMA) | **primary v11 reference** | Pre-Run 0b: `…sfno_eval/20260514_v11_noclip_on_v11_testset_ema/` |
| v10 40y GB16 multistep=1 (EMA), eval'd on v10 | **secondary, "rollout1 at GB16+40y" anchor** | `…sfno_eval/20260514_*gbhpo40_gb16_lr2_0e-4_20260511_ckpt-best_ckpt_ema_mp0/` |
| 5410 benchmark | distant target | embedded in pilot's report |

**Caveat on the v10 GB16 40y comparison:** v11 vs v10 SST shift was measured
in Pre-Run 0 to move zg500 by ≈ −0.02 ACC at 240h (v11 worse). So when
comparing pilot's v11 score to v10 GB16 40y baselines, **subtract ~0.02 ACC
from the v10 baseline** before judging the multistep effect at long-lead zg500.
The dataset effect is small but not zero.

## 7. Decision rule — promote to full v11 100y rollout2?

Adopt all gates; promotion requires every one to pass.

| gate | metric | threshold | source |
|---|---|---|---|
| **A (primary, zg500)** | mean ACC over zg500 @120h/240h/336h | **≥ pilot baseline + 0.02** | vs Pre-Run 0b (v11 100y GB8 multistep=1) |
| **B (primary, zg500@240h)** | zg500 @240h ACC | clear gain over Pre-Run 0b (≥ +0.025 ACC) | vs Pre-Run 0b |
| **C (hold-the-line, tas)** | tas @6h ACC | ≥ 0.980; tas @336h ACC ≥ 0.38 | vs Pre-Run 0b (0.988 / 0.411) |
| **D (hold-the-line, pr_6h)** | pr_6h @72h ACC | ≥ 0.72 | vs Pre-Run 0b (0.747) |
| **E (hold-the-line, ua5/ta5)** | ua5/ta5 @240h ACC | each ≥ Pre-Run 0b − 0.02 | vs Pre-Run 0b (0.586 / 0.597) |
| **F (53-channel sanity)** | no channel × lead cell regresses by > 0.05 ACC vs Pre-Run 0b | full scorecard scan, not just the 5 headline channels |

**If all gates pass → promote.** Build the matching v11 100y rollout2 full
config + slurm; same recipe, same multistep_count=2, 100-yr training data,
wall time ~30 hours. Treat as canonical-baseline candidate, with its own
adoption gate vs Pre-Run 0b after training completes.

**If gates A or B fail (multistep doesn't move zg500) →** the medium-lead
gap to 5410 on zg500 is more representation than rollout-stability. Pivot
the next investment to architecture scaling (embed_dim 256→384, or port
group's `sfno_plasim`) instead of more recipe knobs.

**If hold-the-line gates fail with primaries passing →** keep pilot
checkpoint on disk; do not promote. Investigate the regression (likely a
per-channel loss reweighting question, which can be the *next* HPO axis).

## 8. Compute budget summary

| phase | wall | partition | gate |
|---|---|---|---|
| Phase 0: build v11 40y subset (packager only; boundary exists) | 2–4 h | skx | none — auto |
| Phase 0.5: preflight | ~10 min | h100 (single-node) | bundled into pilot slurm; must pass before training starts |
| Phase 1: pilot training | ~6–8 h | h100 4-GPU | preflight passes |
| Phase 2: pilot eval chain | ~30 min total (h100 inf + skx-dev score+report+figures) | h100 + skx-dev | training completes |
| Phase 3: review against decision rule | manual | n/a | eval lands |

**Total pilot cost:** ~9–13 h wall, of which ~6–8 h is the actual training
(~25–30% the cost of a full v11 100y rollout2 run estimated at ~30 h).

## 9. Provenance — what must land in `provenance.txt`

The pilot's eval `provenance.txt` (auto-produced by `submit_eval.sh`)
already records: `RUN_TAG, EVAL_SHA7, DATA_SHA7, TRAIN_SHA7, TRAIN_FAMILY,
CKPT, CKPT_BASENAME, RUN_DIR, MODE, TEST_HOLDOUT, TRAIN_DIR,
PACKAGER_TEST_SRC, DATE_UTC`.

For this pilot specifically, verify the eval log shows:
- `TRAIN_FAMILY=sfno_zgplev_v11_rollout2_gb16_y40_pilot_20260514`
- `CKPT_BASENAME=best_ckpt_ema_mp0`
- `TEST_HOLDOUT=...sim52_zgplev_full_v11/test_holdout` (NOT v10)
- `TRAIN_DIR=...sim52_zgplev_full_v11/train` (NOT v10, NOT the 40y subset)

The plan-doc note in §6.3 of the prior zg500 plan (add dataset version +
checkpoint basename rows to `report.md`) should also be implemented in a
companion infra commit if time permits — it would surface the test set
version on the face of the report and prevent any future repeat of the
Pre-Run 0 confound.

## 10. What this pilot deliberately does NOT do

- It does **not** test channel-weighted loss (`channel_weights`) — that's a
  separate axis, queued as Run B in the prior plan if rollout2 needs
  supplementing.
- It does **not** explore `multistep_count=3` or higher.
- It does **not** change input noise σ, LR, weight decay, EMA, scheduler.
- It does **not** add a v11 40y rollout1 reference run. The
  cross-dataset-size comparison against Pre-Run 0b (v11 100y rollout1) is
  the user's chosen primary baseline. If results are ambiguous, a 40y
  rollout1 reference can be added afterward.
- It does **not** promote to canonical on success. Promotion requires the
  full v11 100y rollout2 run, evaluated under its own adoption gate.

## 11. Step order — what to do, in what order

1. **Phase 0:** submit packager for v11 40y subset (skx, 2–4 h). Background.
2. **Preflight patch:** small edit to `scripts/preflight.py` to thread
   `--multistep-count` through to the memory probe.
3. **Write config + slurm files** (templates already specified in §2 and §3).
4. **(Phase 0 completes)** verify metadata, stats, sample H5 attrs.
5. **Launch pilot slurm.** Preflight runs first; if it passes, training
   starts automatically.
6. **(Training completes)** submit eval chain with the explicit RUN_TAG and
   v11 TEST_HOLDOUT/TRAIN_DIR overrides from §6.
7. **Apply decision rule (§7);** record outcome in this doc.

User confirmation needed at:
- Phase 0 packager submission (because it consumes ~4h skx wall, even if
  unattended).
- Pilot slurm submission (Phase 1).
- Eval chain submission (Phase 2).
