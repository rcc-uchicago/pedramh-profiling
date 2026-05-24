# SFNO PLASIM full-emulator training plan **v1.1**

**Status:** v1.1 incorporates a 2nd Codex pass — fixes `lr_start=0.0` (PyTorch invalid), `scheduler_T_max` warmup arithmetic, and preflight signature plumbing.
**Author:** Zhixing Liu (with Claude Code)
**Date:** 2026-04-25
**Depends on:** `docs/sfno_tiny_short_training_plan.md` (chunk 1 shipped)

## Goal

Train a full-scale SFNO emulator on PLASIM sim52 at 64×128, drawing **architecture and training schedule** from the group convention `/work2/09979/awikner/stampede3/PanguWeather/v2.0/config/SFNO_PLASIM_H5_DERECHO_5410.yaml`, but translating every field into the **Makani-supported vocabulary** that our `train_plasim.py` path uses. The tiny gate (`plasim_sim52_tiny`) stays untouched and remains the small-model regression check.

### Year ranges (half-open, group convention)
- **Train:** `[12, 112)` → years 12–111 inclusive (100 yrs).
- **Valid:** `[11, 12)` → year 11 only.
- **No test split** (group-convention; user-confirmed).

---

## v1.1 changes (2nd Codex pass)

| # | v1 (still wrong) | v1.1 (fixed) | Source ref |
|---|---|---|---|
| 1 | `lr_start: 0.0` | `lr_start: 1.0E-4` | PyTorch's `LinearLR` raises `ValueError("Starting multiplicative factor expected to be greater than 0...")` on `start_factor=0.0`. Verified locally on torch 2.11. Makani passes `lr_start` directly to `LinearLR(start_factor=...)` at `driver.py:704`. |
| 2 | `scheduler_T_max: 50` (mis-comment "after warmup") | `scheduler_T_max: 45` | Makani wraps cosine in `SequentialLR` after a 5-epoch warmup (`driver.py:706`). T_max=45 = 50-5 reaches `eta_min` at epoch 50. |
| 3 | §A.2 sketched flag injection but `_build_loader_and_wrapper(yaml_config, config_name)` at `preflight.py:85` has no place for the new args | Updated §A.2 to thread `amp_mode` and `checkpointing_level` as kwargs into the helper, with a concrete signature change | `preflight.py:85,382` |
| 4 | `--checkpointing-level choices=[0,1,2]` over-restricted | choices removed; pass-through | defensive |

## Codex review — what changed in v1

v0 mechanically copied PanguWeather field names. Codex traced each into Makani source and flagged 9 issues (3 blocking, 2 high, 4 medium/low). All confirmed against source. Translation table — every change in v1 is justified by a concrete Makani code reference:

| # | v0 (broken) | v1 (Makani-correct) | Source ref |
|---|---|---|---|
| 1 | `scheduler: "LinearWarmupCosineAnnealingLR"` | `scheduler: "CosineAnnealingLR"` + `lr_warmup_steps: 5` (Makani auto-wraps in `SequentialLR(LinearLR_warmup, CosineAnnealingLR)`) | `makani-src/makani/utils/driver.py:681-706` (registry has only 5 schedulers; warmup wrapper is implicit) |
| 2 | `losses: [{type: "raw_l2"}]` | `losses: [{type: "l2", parameters.squared: true}]` | `makani-src/makani/utils/loss.py:34-54` (`_LOSS_REGISTRY` has `l2`, no `raw_l2`); raises at `loss.py:242` |
| 3 | `pos_embed: True` | `pos_embed: "direct"` | `makani-src/makani/models/networks/sfnonet.py:475` (string; `"direct"` = learned, also `"frequency"` / `"none"`) |
| 4 | `checkpointing: 2` (silently swallowed) | YAML field removed; pass `--checkpointing_level 2` in SLURM CLI | `train_plasim.py:123` overrides params from CLI; default `argument_parser.py:41` is `0` |
| 5a | `num_blocks: 16` | **removed** — silent no-op | not in `SphericalFourierNeuralOperatorNet.__init__` (`sfnonet.py:258-292`) |
| 5b | `sparsity_threshold: 0.0` | **removed** — silent no-op | not in constructor |
| 5c | `use_complex_kernels: True` | **removed** — silent no-op | not in constructor |
| 5d | `factorization: None` | **removed** — silent no-op | not in constructor |
| 5e | `complex_network: True` | **removed** — silent no-op | not in constructor |
| 5f | `sync_norm: True` | **removed** — silent no-op | not in constructor |
| 5g | `drop_rate: 0.0` | `pos_drop_rate: 0.0` (or omit; default 0.0) | `sfnonet.py:279` |
| 5h | `drop_path_rate: 0.0` | `path_drop_rate: 0.0` (or omit; default 0.0) | `sfnonet.py:280` |
| 6 | preflight runs forced fp32 + `checkpointing_level=0` (`preflight.py:127,135`) → memory probe doesn't represent training | preflight gains `--amp-mode` / `--checkpointing-level` CLI flags, defaulting to current behavior; SLURM passes the same values training will use | `preflight.py:127,135` |
| 7 | Stats reuse silently defaulted | Stats reuse promoted to **explicit §F approval item** (years 101–111 train data not in stats compute) | n/a |
| 8 | `valid_autoreg_steps: 0` (single-step only) | `valid_autoreg_steps: 3` (24h rollout signal) | n/a |
| 9 | `n_train_samples_per_epoch: -1`, `n_eval_samples: -1` (cosmetic — not consumed) | both removed; document that PlasimForcingDataset uses all files in train/ and valid/ | `plasim_trainer.py:97-124` (constructor doesn't accept these) |

In addition: extra non-translatable group fields (PanguWeather-only) were already excluded in v0; those decisions stand.

---

## Source dataset state (verified 2026-04-25)

```
$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128/
├── train/   MOST.0003.h5 .. MOST.0100.h5     (98 files; years 3–100)
├── valid/   MOST.0101.h5 .. MOST.0120.h5     (20 files; years 101–120)
├── stats/   global_means.npy, global_stds.npy, time_means.npy,
│            forcing_global_{means,stds}.npy, forcing_time_means.npy
├── metadata/
└── config/
```

**Cross-split mapping required:**

| New split | Years | Source location |
|---|---|---|
| `dst/train/` | 12–111 | `src/train/` (12–100) + `src/valid/` (101–111) |
| `dst/valid/` | 11 | `src/train/` (1 file) |

Builder must look across `src/{train,valid,test}/` regardless of dst split. See §A.

---

## Stats coverage — flagged trade-off (now §F.1)

`stats/*.npy` were computed over the original train range (years 3–100).
- New train years 12–100 (89 yrs) → in stats compute ✓
- New train years 101–111 (11 yrs) → **not in stats compute**
- New valid year 11 → in stats compute ✓

Two options, **explicit approval needed**:

- **A.** Reuse existing stats. Default in this plan. ~11% of train data is normalized using stats that didn't see those years. Likely fine for a stationary ~12-yr block but flagged scientifically.
- **B.** Recompute stats over years 12–111. Adds a one-shot script (~2 hr to read 100 H5 files and accumulate per-channel means/stds), and the symlink farm points at the new stats dir. **Not in v1.1 scope** unless approved.

---

## Deliverables (v1.1: 7 files — 3 new, 2 edits, 2 tests)

1. **NEW** `src/sfno_training/config/plasim_sim52_full.yaml` — full SFNO config, Makani-correct (§B).
2. **NEW** `src/sfno_training/submit_full.slurm` — SLURM launcher (§C).
3. **EDIT** `scripts/build_subset_dataset.py` — cross-split year lookup (§A.1).
4. **EDIT** `scripts/preflight.py` — add `--amp-mode` / `--checkpointing-level` flags so memory probe matches training (§A.2).
5. **EDIT** `tests/sfno_training/test_build_subset_dataset.py` — cross-split tests (§D.1).
6. **EDIT** `tests/sfno_training/test_preflight.py` — coverage for new flags (§D.2).
7. **NEW** this plan file.

No changes to PlasimTrainer, dataset wrappers, preprocessor, multistep, or train_plasim.py. The "full" run reuses every line of code that the tiny gate exercised; only YAML, SLURM, and the two scripts change.

---

## §A — Script changes

### §A.1 Cross-split year lookup in `build_subset_dataset.py`

Same proposal as v0 — replace `_link_split` with a year-search version.

```python
SPLITS_AS_DIRS = ("train", "valid", "test")

def _find_year_file(src: Path, year: int) -> Path:
    """Search src/train, src/valid, src/test (in order) for MOST.{year:04d}.h5.
    Precedence: train > valid > test (defensive, since our packager output
    has no duplicates). Raises FileNotFoundError if not found anywhere."""
    for split in SPLITS_AS_DIRS:
        candidate = src / split / f"MOST.{year:04d}.h5"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"year {year}: MOST.{year:04d}.h5 not found in src/{{train,valid,test}}/"
    )

def _link_split(src: Path, dst_split: Path, years: Iterable[int], split_name: str) -> int:
    dst_split.mkdir(parents=True, exist_ok=True)
    n = 0
    for year in years:
        src_file = _find_year_file(src, year)
        _replace_symlink(src_file, _file_for_year(dst_split, year))
        n += 1
    return n
```

### §A.2 Preflight: surface `--amp-mode` and `--checkpointing-level`

Currently `preflight.py:127,135` forces `params["amp_mode"]="none"` and `params["checkpointing_level"]=0`. For the full run, the preflight memory probe must mirror training (bf16 + activation checkpointing) or it tests the wrong path. Two options of how to surface:

- **(a) Required passthrough** — add `--amp-mode {none,fp16,bf16}` and `--checkpointing-level <int>` (pass-through, no enum restriction) to preflight's argparse (defaults `none` / `0` to preserve current tiny-gate behavior). Pass the matching values from `submit_full.slurm`.
- **(b) Auto-mirror** — preflight reads same env vars as submit script. More magic; rejected.

Choose (a). Compatibility: tiny-gate `submit_tiny.slurm` doesn't pass these flags, so it stays at `none`/`0` (unchanged).

```python
# preflight.py argparse additions
p.add_argument("--amp-mode", default="none", choices=["none", "fp16", "bf16"])
p.add_argument("--checkpointing-level", default=0, type=int,
               help="0/1/2 per Makani; not enum-restricted — pass-through")

# Plumb through the build helper signature (currently
# _build_loader_and_wrapper(yaml_config, config_name) at preflight.py:85).
def _build_loader_and_wrapper(
    yaml_config: Path,
    config_name: str,
    *,
    amp_mode: str = "none",
    checkpointing_level: int = 0,
):
    ...
    params["amp_mode"] = amp_mode                    # was hardcoded "none"
    params["checkpointing_level"] = checkpointing_level  # was hardcoded 0

# Caller in main() passes the new args:
trainer, params = _build_loader_and_wrapper(
    args.yaml_config, args.config,
    amp_mode=args.amp_mode,
    checkpointing_level=args.checkpointing_level,
)
```

`--checkpointing-level` is intentionally not enum-restricted (Makani may grow ckpt levels; if they do, an enum here would silently break).

---

## §B — `src/sfno_training/config/plasim_sim52_full.yaml`

**Top-level config name:** `plasim_sim52_full`. Same dataset wiring shape as `plasim_sim52_tiny`, with corrected SFNO sizing and schedule.

### Dataset paths, image/time/grids, channels, PlaSim plumbing, normalization, aux features

Identical to `plasim_sim52_tiny`. Aux features all `False` (asserted by `_set_data_shapes`).

### SFNO architecture — Makani-supported subset of group convention

```yaml
nettype: "SFNO"                       # makani's class (NOT group's "sfno_plasim")
filter_type: "linear"
operator_type: "dhconv"
spectral_transform: "sht"             # makani default; explicit
scale_factor: 1                       # native res (was 4 in tiny)
embed_dim: 256
num_layers: 12
encoder_layers: 1
mlp_ratio: 2.0
use_mlp: !!bool True
# mlp_mode removed — not consumed by Makani SFNO __init__ (sfnonet.py:258-292);
# would silently no-op via **kwargs.
activation_function: "gelu"
pos_embed: "direct"                   # FIXED: string, not boolean
normalization_layer: "instance_norm"
hard_thresholding_fraction: 1.0
big_skip: !!bool True
rank: 1.0
separable: !!bool False
complex_activation: "real"
spectral_layers: 3
# checkpointing_level NOT set in YAML — controlled via CLI --checkpointing_level 2
# All Pangu-only kwargs removed (would silently no-op via **kwargs):
#   num_blocks, sparsity_threshold, use_complex_kernels,
#   factorization, complex_network, sync_norm, drop_rate, drop_path_rate
```

### Loss — Makani-correct l2

```yaml
losses:
-   type: "l2"
    channel_weights: "constant"
    temp_diff_normalization: !!bool False
    parameters:
        squared: !!bool True
```

### Training schedule — Makani-correct

```yaml
lr: 1.0E-4
weight_decay: 3.0E-6
max_epochs: 50
batch_size: 4                          # start at 4 (memory probe); ramp later
n_history: 0
n_future: 0
valid_autoreg_steps: 3                 # FIXED: 3 = 24h rollout signal
prediction_type: "iterative"

# Scheduler — CosineAnnealingLR + LinearLR warmup auto-wrap
# (driver.py:701-706: lr_warmup_steps>0 wraps the chosen scheduler in
#  SequentialLR with a LinearLR warmup using start_factor=lr_start)
scheduler: "CosineAnnealingLR"
scheduler_T_max: 45                    # max_epochs (50) - lr_warmup_steps (5)
scheduler_min_lr: 1.0E-8               # eta_min
lr_warmup_steps: 5                     # epochs of LinearLR warmup
lr_start: 1.0E-4                       # LinearLR start_factor (PyTorch rejects 0.0;
                                       # 1e-4 = group's warmup_start_lr/lr = 1e-8/1e-4)

optimizer_type: "AdamW"
optimizer_beta1: 0.9
optimizer_beta2: 0.95
optimizer_max_grad_norm: 32

num_data_workers: 4
num_visualization_workers: 0
crop_size_x: None
crop_size_y: None

ics_type: "specify_number"
save_raw_forecasts: !!bool True
save_channel:       !!bool False
masked_acc:         !!bool False
maskpath: None
perturb:    !!bool False
add_noise:  !!bool False
noise_std:  0.0
pretrained: !!bool False

target: "tendency"
normalize_residual: !!bool False

log_to_screen: !!bool True
log_to_wandb:  !!bool False            # OFF first launch; flip on after sane curve
log_video: 0
verbose: !!bool True

wireup_info: "mpi"
wireup_store: "tcp"

# Cosmetic — NOT consumed by PlasimForcingDataset; full data is the default.
# Removed: n_train_samples_per_epoch, n_eval_samples
```

> **Note on `lr_start`:** Group's `warmup_start_lr: 1e-8` is an absolute LR. Makani's `LinearLR` uses a `start_factor` (relative). Translation: `lr_start = warmup_start_lr / lr = 1e-8 / 1e-4 = 1e-4`. We set `lr_start: 1.0E-4` for two reasons: (a) PyTorch's `LinearLR` rejects `start_factor=0.0` with "expected to be greater than 0 and less or equal to 1" (verified torch 2.11); (b) it exactly matches the group convention.

> **Note on `scheduler_T_max`:** Makani wraps `CosineAnnealingLR` in `SequentialLR` after the LinearLR warmup (`driver.py:706`); the cosine scheduler ticks for `max_epochs - lr_warmup_steps = 50 - 5 = 45` epochs. Setting `scheduler_T_max: 50` would leave the cosine schedule unfinished (LR not at `eta_min`) at epoch 50.

---

## §C — `src/sfno_training/submit_full.slurm`

Three differences from v0: pass `--checkpointing_level 2`, run preflight with matching flags, add resume-friendly behavior.

```bash
#!/bin/bash
#SBATCH -J sfno_full
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 47:30:00
#SBATCH -o logs/sfno_full_%j.out
#SBATCH -e logs/sfno_full_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Full SFNO emulator training (docs/sfno_full_training_plan.md v1.1).
# Resubmitting picks up from the latest checkpoint (load_checkpoint=legacy
# default in argument_parser.py:61 + train_plasim.py:110-115 resume detection).
#
# Required env vars:
#   OUTPUT_ROOT — packager-output (or symlink subset) root.
#                 Default: $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_full
#   EXP_DIR     — experiment output dir.
#                 Default: $SCRATCH/SFNO_Climate_Emulator/runs/sfno_full
#
# Build the subset (one-time) before launching:
#   scripts/build_subset_dataset.py \
#       --src $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128 \
#       --dst $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_full \
#       --train-years 12-111 \
#       --valid-years 11
# ============================================================================

set -euo pipefail
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_full}"
EXP_DIR="${EXP_DIR:-$SCRATCH/SFNO_Climate_Emulator/runs/sfno_full}"

source "$HOME/projects/SFNO_Climate_Emulator/.venv/bin/activate"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the dataset (subset) root}"
: "${EXP_DIR:?Set EXP_DIR to the experiment output dir}"

FULL_TPL="$HOME/projects/SFNO_Climate_Emulator/src/sfno_training/config/plasim_sim52_full.yaml"
FULL_YAML="$EXP_DIR/plasim_sim52_full.rendered.yaml"
mkdir -p "$EXP_DIR" "$EXP_DIR/training_checkpoints"
sed -e "s|{{OUTPUT_ROOT}}|$OUTPUT_ROOT|g" \
    -e "s|{{EXP_DIR}}|$EXP_DIR|g" \
    "$FULL_TPL" > "$FULL_YAML"

cd "$HOME/projects/SFNO_Climate_Emulator"
mkdir -p logs
set -x

# 1) Preflight — same bf16 + checkpointing_level=2 as training.
python scripts/preflight.py \
    --yaml_config "$FULL_YAML" \
    --config plasim_sim52_full \
    --template "$FULL_TPL" \
    --amp-mode bf16 \
    --checkpointing-level 2 \
    --log "$EXP_DIR/preflight_log.txt"

# 2) Train. --checkpointing_level 2 enables activation checkpointing
# (driver default 0; group convention is 2).
python -m sfno_training.train_plasim \
    --yaml_config "$FULL_YAML" \
    --config plasim_sim52_full \
    --run_num 0 \
    --batch_size 4 \
    --multistep_count 1 \
    --amp_mode bf16 \
    --checkpointing_level 2 \
    --disable_ddp

# 3) Post-training NaN/Inf scan (chunk 2; harmless if missing).
if [ -f scripts/scan_for_nans.py ]; then
    python scripts/scan_for_nans.py \
        --exp-dir "$EXP_DIR/plasim_sim52_full/0" \
        --slurm-logs "logs/sfno_full_${SLURM_JOB_ID}.out" "logs/sfno_full_${SLURM_JOB_ID}.err" \
        || true
fi

echo "full complete: $EXP_DIR/plasim_sim52_full/0/"
```

### Walltime / restart strategy

50 epochs × 100 train years × ~1460 samples/yr ÷ batch 4 ≈ 1.83 M optimizer steps. At bf16 + activation checkpointing on a single H100 with `embed_dim=256, num_layers=12, scale_factor=1`, very rough back-of-envelope is ~5–15 step/sec → 35–100 hr wallclock.

**v1 strategy:** single long job (47:30:00, presumed below Stampede3 h100 cap), per-epoch checkpoints (`save_checkpoint=legacy` default). If the job dies mid-run, resubmit the same `submit_full.slurm` — `train_plasim.py:110-115` detects the existing checkpoint via `load_checkpoint=legacy` (default) and sets `params.resuming=True`.

**Verify before launch:** confirm Stampede3 h100 actual walltime cap. If < ~30 hr, switch to multi-job auto-resume via SLURM `--dependency=afterany`.

---

## §D — Tests

### §D.1 Extend `tests/sfno_training/test_build_subset_dataset.py`

Add 4 cross-split tests:

1. **`test_train_year_from_valid_split`** — fake src has year 105 only in `src/valid/`; `build_subset(src, dst, [105], [11])` produces `dst/train/MOST.0105.h5 → src/valid/MOST.0105.h5`.
2. **`test_valid_year_from_train_split`** — fake src has year 11 only in `src/train/`; `build_subset(src, dst, [12], [11])` produces `dst/valid/MOST.0011.h5 → src/train/MOST.0011.h5`.
3. **`test_year_not_in_any_split`** — year 999 in no split; raises `FileNotFoundError` mentioning `train,valid,test`.
4. **`test_precedence_train_over_valid`** — same year exists in both `src/train/` and `src/valid/`; the train one wins.

Existing tests should continue to pass (year 3 in src/train, year 101 in src/valid — both resolved unchanged).

### §D.2 Extend `tests/sfno_training/test_preflight.py`

Add 2 tests for the new flags:

1. **`test_amp_mode_passthrough`** — invoke preflight programmatically with `args.amp_mode = "bf16"`; assert that the constructed trainer's `params["amp_mode"] == "bf16"`.
2. **`test_checkpointing_level_passthrough`** — same shape with `args.checkpointing_level = 2`; assert `params["checkpointing_level"] == 2`.

Existing tests pass without change (defaults `none`/`0`).

---

## §E — Execution order

1. Code changes (§A.1, §A.2, §B, §C, §D.1, §D.2). Single commit.
2. Run unit tests: `pytest tests/sfno_training/test_build_subset_dataset.py tests/sfno_training/test_preflight.py -q`.
3. Build the symlink farm: `scripts/build_subset_dataset.py --src $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128 --dst $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_full --train-years 12-111 --valid-years 11`. Spot-check counts: train=100, valid=1.
4. **Gate:** confirm tiny SFNO smoke + tiny training are green. If tiny gate hasn't passed, full launch is premature.
5. **Real memory probe** — submit a 1-job 5-min interactive (or 0:30:00 batch) test that runs preflight with `--amp-mode bf16 --checkpointing-level 2` and one train step at `batch_size 4`. Confirm no OOM. Output committed to `logs/sfno_full_memprobe_*.out`.
6. Submit full: `! sbatch src/sfno_training/submit_full.slurm`. Email on BEGIN/END/FAIL.
7. Triage: monitor first 1–2 epochs. If validation loss decreases monotonically and no NaN/Inf, let it run. If it dies on walltime, resubmit (resume is automatic).
8. Post-training: run `scripts/scan_for_nans.py`, `scripts/loss_decompose.py`, `scripts/plot_loss_curves.py` (chunk 2 deliverables).

---

## §F — Open decisions to resolve at review

| # | Decision | Default in plan v1.1 | Alternative |
|---|---|---|---|
| 1 | **Stats coverage gap on 101–111** | Reuse existing stats (3–100). Acknowledged science risk. | Recompute stats over 12–111. Adds a packaging step; stricter for production baseline. |
| 2 | Walltime / restart | Single long job 47:30:00 + resume on resubmit | Multi-job dependency chain if cap < ~30 h |
| 3 | `valid_autoreg_steps` | 3 (24h rollout signal) per Codex finding #8 | 0 (single-step only — matches tiny but provides no rollout signal) |
| 4 | `batch_size` | 4 (after real bf16 + ckpt-level 2 memory probe in §E.5) | Ramp to 8 only after measured headroom |
| 5 | `wandb` | Off for first launch | On after first sane loss curve |
| 6 | `lr_start` (warmup start factor) | `1.0E-4` — exact translation of group's `warmup_start_lr=1e-8` relative to `lr=1e-4`. (PyTorch rejects `0.0`.) | Larger value (e.g. `0.01`) for a faster warmup ramp. |
| 7 | Stampede3 h100 walltime cap | Assumed 48h | Verify before launch via `sinfo -p h100 -h -o "%l"` |

---

## §G — Out of scope (explicitly)

- **Inference / rollout evaluation** — separate plan.
- **Stats recomputation** — flagged §F.1; if approved, separate work.
- **Multi-step training (`n_future > 0`)** — not in group convention; not v1.
- **Multi-GPU / DDP** — single H100 with `--disable_ddp`. Multi-GPU is a separate plan.
- **Hyperparameter tuning** — copying group convention (translated into Makani vocabulary). Tuning happens after a baseline.
- **Test-set evaluation** — no test split per group convention.

---

**Reviewer ask:**
(a) approve year-range mapping (12–111 train / 11 valid, half-open),
(b) approve cross-split builder extension (§A.1),
(c) approve preflight flags (§A.2),
(d) approve translated full-YAML (§B) — verify each translation row in the v0→v1 table is acceptable,
(e) decide on §F items 1–7 (or accept defaults),
(f) approve §E execution order including the explicit memory probe step (§E.5).

Once approved, I implement §A + §B + §C + §D in a single commit.
