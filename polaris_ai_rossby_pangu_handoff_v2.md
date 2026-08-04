# HANDOFF v2 — PanguPlasim-on-E3SM (ai-rossby) on Polaris, with CORRECTED normalization

Supersedes `polaris_ai_rossby_pangu_handoff_prompt.md` (v1). Read this whole file
first. Work on branch `fix/tsoi-fill-270` in
`/eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling`.

*(The branch name comes from a **physicsnemo**-path fill change on 2026-08-03. On
the **PanguWeather** path the TSOI fill was never wrong — only its statistics.)*

**Goal:** train ai-rossby's **`PanguPlasimLegacy`** on the E3SM archive with the
**PanguWeather 108-field variable set**, on Polaris, logging to wandb.

**What changed since v1:** v1 was *parity-first* — it deliberately reproduced
PanguWeather's fills, defects included. That is **abandoned**. Two normalization
defects were found and measured. **ai-rossby will be trained with the fixes;
PanguWeather's own retrain is jesswan's call and out of scope here** — §2's
severity analysis exists to inform that call, not to schedule it.

*This document was adversarially audited (4 independent reviewers + a verify
pass) on 2026-08-04; every correction they confirmed is folded in.*

---

## 1. The one distinction this whole document rests on

> **Parity is asserted on the VARIABLE SET. It is deliberately NOT asserted on
> the fill values or the normalization statistics.**

* **Variable set — IDENTICAL, machine-checked.** Same 108 fields, same seven
  groups, same order within each group, same 18 levels. `VARIABLE_PARITY_OK
  10/10` against jesswan's trained config. Unchanged, and must stay so.
* **Fills + statistics — deliberately CORRECTED.** Reproducing them would
  reproduce two real defects.

Anywhere this document says "parity", it means the first bullet only.

*Channel-count glossary, used throughout:* **108 fields** = 105 encoder inputs
(6 surface + 2 land + 4 constant-boundary + 3 varying-boundary + 5×18 upper-air)
**+ 3 diagnostic outputs**. "The 15 named channels" = the non-upper-air inputs.

---

## 2. The two defects, measured

### 2a. `TSOI_10CM` — statistics computed under the wrong fill ⛔ the serious one

The **fill (270 K) is correct and must not change** — soil temperature is Kelvin,
valid-land mean 272 K, so 270 is a near mean-fill. The defect is that the shipped
`.nc` statistics were computed under a **0-fill**:

| statistics computed over | mean | std | resulting normalized spread |
|---|---:|---:|---:|
| 0-filled data *(recomputed, n=8)* | 104.98 | 133.49 | **0.122** |
| land only *(the near-miss)* | 272.20 | 26.25 | 0.622 |
| **270-filled data (correct)** | **270.85** | **16.34** | **1.0000** |

The **shipped `.nc` itself reads 105.229 / 133.802** — same regime, computed over
the full 35 years. The row above is a small-sample recomputation that reproduces it.

**Cause:** `compute_normalization_e3sm.py:144-145` — **jesswan's, and NOT in this
repo** — builds a fill dict defaulting every field to `0.` and never sets
`mask_fill['TSOI_10CM']`. The same pattern is visible in the in-repo
`compute_normalization.py:116-119`, which is the PLASIM-named sibling (it contains
no E3SM names at all, so the bug cannot literally live there).

**Consequence.** Verified that the scaling does *not* cancel:
`data_loader_multifiles.py:792` normalizes, the fill is applied before it
(`:672`), and `train.py:1573` takes `loss_obj_sfc` on the normalized tensors,
targets included. Measured normalized spread of the 8-wide surface tensor:

```
TREFHT 1.006  U10 0.998  RHREFHT 0.975  PS 0.993
PSL 0.950  TMQ 1.000  SOILWATER_10CM 1.008
TSOI_10CM 0.122   <-- the ONLY outlier
```

**⚠ The size of the effect depends on the loss, and the two runs differ:**

| run | loss | under-weighting |
|---|---|---|
| jesswan's PanguWeather SFNO | `raw_l2` → MSELoss | **~67×** (squared) |
| **the ai-rossby run §5 launches** | **`loss=mae` → `loss_type: l1`** | **~8.2×** (linear) |

Override with `loss=raw_l2` if you want the squared regime. *(This reconciles the
repo's three figures: 8.2× is the amplitude/L1 ratio, 67× is its square, and
`data_for_training.md` R3's "26×" is a different gradient-based comparison.)*

Stated without assumptions: **matched statistics multiply TSOI's loss term by
~67× (MSE) or ~8.2× (L1) relative to today.** Avoid "1/67th of every sibling
channel's" — that would equate field spread with error variance.

**Two honest caveats:**
* **AdamW blunts this.** Per-parameter second-moment normalization cancels much
  of a uniform channel-wise gradient scaling for parameters exclusive to that
  channel. The residual lives in the shared trunk and has not been measured.
* **Rollout feedback is validation-only.** Training is single-step
  (`train.py:1049`); the multi-step machinery is in `validate_one_epoch`. So
  "predicted TSOI re-enters as input" compounds at validation/inference, not
  during training.

Corroboration: TSOI has the **highest** encoder weight norm of the 15 named
channels (1.91× median) — the model straining to amplify a nearly-flat input.

### 2b. `SST` — a Kelvin constant on a Celsius field ⚠ real, but minor

E3SM's `SST` is **degC** — measured `[-1.80, 33.57]`. The floor is exactly −1.8 at
~6% of all grid cells (~9.7% of ocean cells; it varies seasonally with ice
extent), **99.7% of them poleward of 60°** with mean sea-ice fraction **0.97** —
the seawater freezing clamp. It was filled with **270**, i.e. 270 °C.

Unlike TSOI, its fill and statistics **agreed** (both built on 270), so the channel
normalized to spread 1.0 and was correctly weighted. The cost was input precision:
the ocean's real variation occupied 0.093σ, ~**75 distinct bf16 values** (~0.47 °C
each) versus ~2670 with a sane fill. Training ran bf16.

**Why this alone would NOT justify a retrain:**

* **Input-only** — a prescribed boundary, `ocean_variables: []`, never in the loss.
* **Bit-identical across all 35 years**, and only **~12 distinct states per year**
  — a prescribed monthly climatology. Across 43,800 training samples it supplies
  twelve distinct fields.
* `sol_in` uniquely identifies the timestep-of-year (46/46 sampled), so `SST` is
  derivable from an input the model already has.
* The trained model reads it at **0.89× the median** weight norm, rank 70 of 105.
  No pathology.

So: fix it, but **in the same pass as TSOI**. It is not the reason.

---

## 3. The corrected values — use exactly these

### Fills

| field | masked | fill | status |
|---|---:|---:|---|
| `TSOI_10CM` | 61.43% | **270.0** K | **unchanged** — the fill was never wrong |
| `SST` | 37.35% | **−1.8** degC | **CORRECTED** from 270 |
| `SOILWATER_10CM` | 61.43% | 0.0 | unchanged, in-range |
| `TOPO` | 62.65% | 0.0 m | unchanged (sea level) |
| `PCT_GLACIER` | 62.65% | 0.0 | unchanged |
| `PCT_NATVEG` | 62.65% | 0.0 | unchanged |
| `PFTDATA_MASK` | 62.65% | 0.0 | unchanged — see note |
| `ICE` | 37.35% | 0.0 | unchanged |

Every other field has **0% NaN** and correctly has no fill. Eight NaN-carrying
fields, eight `mask_fill` entries — complete, no dead entries.

*`PFTDATA_MASK` note:* the range check flags fill 0 as "outside valid range"
because the field's valid values are only ever `1`. That is a **false positive on
a degenerate field** — filling 0 is what makes it a real 0/1 land–sea mask.

**Why −1.8 for SST — stated honestly, because it is NOT the metric winner:**

| fill | mean | std | ocean signal |
|---|---:|---:|---:|
| ocean-mean (14.54) | 14.538 | 9.112 | **1.263σ** ← best |
| 0.0 degC | 9.108 | 11.511 | 1.000σ |
| **−1.8 degC (chosen)** | **8.436** | **12.062** | 0.954σ ← last |

−1.8 is a **cross-pipeline-consistency convention**, chosen over the numerically
better ocean-mean fill; the signal cost is 1.263σ → 0.954σ, which all reviewers
judged within noise for an input-only, 12-state channel. It is a value the data
itself takes, it matches physicsnemo and makani so all three pipelines agree, and
it applies the same in-distribution principle that fixed TSOI. Ocean-mean was
rejected because it makes land impersonate temperate ocean and is a
dataset-coupled magic number. Confusability (−1.8 collides with ~10% of ocean
cells) is neutralized: the model separately receives a land mask,
glacier/vegetation fraction, topography, and `ICE` on SST's exact mask.

*Recorded, not adopted:* a Poisson/diffusion fill would uniquely remove the
coastline discontinuity a spectral model feels, and SST has only 12 states so it
is cheap to precompute. Judgment: the right **fallback**, only if coastal ringing
is demonstrated. Also considered and not adopted: `mask_output: True` +
land-only statistics (`train.py:839-846` selects `Masked_MSELoss` when
`mask_output` is set; both E3SM configs set it `False`). Cheaper, but it fixes
only the loss side and leaves the *input* flatness untouched.

### Expected normalization constants

| field | mean | std | basis |
|---|---:|---:|---|
| `SST` (fill −1.8) | **8.44** | **12.06** | 24-file test pass |
| `TSOI_10CM` (fill 270) | **271.09** | **16.39** | 24-file test pass |

Old/defective values for contrast: `SST` 109.963 / 123.908, `TSOI_10CM`
105.229 / 133.802.

Expect **±1%** between sampling passes. `SST` should match tightly on the full
35-year run — the field is bit-identical across years, so 35-year statistics equal
any single year's. `TSOI_10CM` genuinely varies year to year.

**The rule, which is the whole fix:** compute the statistics over **exactly the
array the model receives — i.e. after the fill** (given `mask_output: False`,
which both E3SM configs set). Land-only statistics are the plausible near-miss and
still wrong (0.622 spread).

---

## 4. State: what is DONE and verified

| item | status |
|---|---|
| Variable-parity gate | ✅ `VARIABLE_PARITY_OK 10/10` + 16/16 artifacts |
| `physicsnemo_ai_rossby/` subtree | ✅ vendored unsquashed, `87002adb` a real ancestor |
| ai-rossby venv | ✅ `AI_ROSSBY_VENV_OK` — torch 2.10.0+cu129, zarr 3.2.1 |
| Code edits (`sol_in`, land/ocean lists, channels) | ✅ committed |
| Model + dataset configs | ✅ written, contract-checked (paths below) |
| **E3SM → zarr conversion (smoke split)** | ✅ **`CONVERT_ALL_OK`, job 7337122** |
| Store verification | ✅ `PANGU_STORE_VERIFIED 13/13` ×3, bitwise `max\|diff\| = 0` |
| Training launcher + preflight | ✅ written, preflight negative-tested |
| **Corrected normalization** | 🟡 **job 7337234 — see §5 Step 1** |
| **Training smoke** | ⬜ **not yet run — the next real gate** |

Configs: `physicsnemo_ai_rossby/examples/weather/ai_rossby/conf/` →
`model/pangu_plasim_e3sm.yaml`, `dataset/e3sm_pangu_parity.yaml`,
`training/pangu_plasim_legacy.yaml`, `loss/mae.yaml`.

Stores on disk (~60 GB): `$AI_ROSSBY_DATA/e3sm/{train/2015.zarr, val/2045.zarr,
val/2046.zarr}` (the 1-sample 2046 is the validation tail store, §7.7).

**The DATA stores are fill- and stats-agnostic — they preserve raw NaN**, so the
normalization fix needs **no reconversion**. The separate **normalization zarr**
*does* need rebuilding — §5 Step 2. Do not read this row as license to skip it.

---

## 5. Do this next, in order

```bash
cd /eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling
source polaris_env.sh
```

### Step 1 — regenerate the normalization statistics

```bash
# ⚠ FIRST: the script overwrites $PANGU_AUX/data_2015-2050_{mean,std_corr}.nc,
# which is the only local copy of the PRE-FIX statistics — the ones §2 is derived
# from and the 85-epoch checkpoint was trained under. Keep them:
mkdir -p $PANGU_AUX/pre_fix && cp $PANGU_AUX/data_2015-2050_{mean,std_corr}.nc $PANGU_AUX/pre_fix/

cd PanguWeather/v2.0 && qsub HPC_scripts/polaris_compute_e3sm_norm.pbs
qstat -x -u $USER                        # status; job 7337234 was the first submission
tail -f PanguWeather/v2.0/e3sm_norm_moments.o*   # -j oe, no -o: log lands in the submit dir
```

PASS = `MOMENTS_OK` → `NORM_NC_OK` → `NORMALIZATION_OK` → `E3SM_NORM_REGEN_OK`.
jesswan's originals in `$E3SM_ROOT/h5/plev_data/` are never touched.

Stage 1 (the ~2 TB pass) is **skipped** if `$PANGU_AUX/moments_2015-2050.json`
already exists; set `FORCE_MOMENTS=1` to recompute. That JSON stores
fill-independent moments, so regenerating for a *different* fill costs seconds:

```bash
# EXAMPLE ONLY — not part of Step 1; 0.0 is the value §3 rejected.
$AI_ROSSBY_VENV/bin/python PanguWeather/v2.0/compute_e3sm_normalization.py \
  --stage nc --moments $PANGU_AUX/moments_2015-2050.json \
  --config PanguWeather/v2.0/config/E3SM_SFNO_H5_POLARIS.yaml \
  --fill SST=0.0 --out-dir $MEMBER_ROOT/tmp/alt --tag ALT
```

### Step 2 — rebuild the ai-rossby normalization zarr

⚠ **Required before the smoke, and it is the doc's one silent-failure step.** The
zarr on disk was built from the OLD statistics; the dataset config now specifies
`SST: -1.8`. Pairing them recreates the exact mismatch this work removes, and the
preflight will **not** catch it (it checks channel names and order, not statistics).

```bash
mv $AI_ROSSBY_DATA/e3sm/norm/normalization_2015-2050.zarr{,.prefix}   # rollback path

$AI_ROSSBY_VENV/bin/python \
  physicsnemo_ai_rossby/tools/data/e3sm/build_normalization_zarr.py \
  --source-dir $PANGU_AUX --std data_2015-2050_std_corr.nc \
  --output $AI_ROSSBY_DATA/e3sm/norm/normalization_2015-2050.zarr --overwrite
# (--mean defaults to data_2015-2050_mean.nc; --std must be passed because ITS
#  default is data_2015-2050_std.nc, without the _corr.)

# The tool prints NO pass token. Verify explicitly:
$AI_ROSSBY_VENV/bin/python -c "
import xarray as xr, os
d = xr.open_zarr(os.environ['AI_ROSSBY_DATA']+'/e3sm/norm/normalization_2015-2050.zarr')
for v, m, s in (('SST', 8.44, 12.06), ('TSOI_10CM', 271.09, 16.39)):
    gm, gs = float(d[v].sel(stat='mean')), float(d[v].sel(stat='std'))
    ok = abs(gm-m) < 0.5 and abs(gs-s) < 0.5
    print(f'{v:10s} {gm:10.4f} {gs:10.4f}  {\"OK\" if ok else \"*** WRONG ***\"}')"
```

### Step 3 — the training smoke (the next real gate)

```bash
cd physicsnemo_ai_rossby && qsub polaris/polaris_pangu_plasim.pbs
# faster scheduling: qsub -q debug polaris/polaris_pangu_plasim.pbs
```

PASS = `PREFLIGHT_OK`, then **`PANGU_PLASIM_RUN_OK`** in
`physicsnemo_ai_rossby/ai_rossby_pangu_plasim.o<jobid>`, with advancing `train/*`
metrics, finite and decreasing. Defaults: 1 epoch, 1 h walltime, `preemptable`,
run dir `$MEMBER_ROOT/runs/ai_rossby_pangu_plasim/<run>`. **wandb is offline** —
`wandb sync $WANDB_DIR/<run>` from a login node, or `qsub -v WANDB_MODE=online`.

**Four things have never executed** — expect the failure here:

1. Does `PanguPlasimLegacy` construct? The `sol_in` patch should clear the
   `ValueError`, untested on a real config.
2. Do the tensors line up? 8-channel surface vs `[surface|land|ocean]` slicing.
3. **Does it fit in 40 GB?** The most likely failure. Levers, in
   `conf/model/pangu_plasim_e3sm.yaml`: `checkpointing: 3` (`:94`), then
   `embed_dim: 240` (`:72`).
4. Does `window_size: [2,6,10]` (`:66`) divide 18 levels? An assumption carried
   from the S2S config.

### Step 4 — production

```bash
# conversion: ~11 h, ~1.08 TB, ~1.14M inodes.
# NOTE --overwrite is unconditional (convert pbs:143), so the existing
# 2015/2045 stores are rebuilt from scratch; the ~11 h includes them.
cd physicsnemo_ai_rossby && qsub \
  -v TRAIN_YEARS="$(seq -s' ' 2015 2044)",VAL_YEARS="2045 2046 2047 2048" \
  polaris/polaris_e3sm_pangu_convert.pbs

# then training, longer walltime + more epochs:
qsub -v MAX_EPOCHS=100 -l walltime=72:00:00 polaris/polaris_pangu_plasim.pbs
# 72 h is the preemptable max; chain with ../polaris_submit_chain.sh for longer.
```

Bytes are fine (16.24 / 50 TB as of 2026-08-04). **`myquota` reports no inode
limit** — the full conversion is ~1.14 M inodes on this store shape, and sibling
clusters have hit inode caps on it; check with the ALCF helpdesk before the full run.

---

## 6. Tools built — use these rather than re-deriving

| tool | what it does |
|---|---|
| `ai_rossby_variable_contract.py` | The variable contract in ONE place. `--check-ground-truth`, `--check-artifacts`. stdlib-only — **needs `python3.12`**, the login node's bare `python3` is 3.6. |
| `PanguWeather/v2.0/check_normalization.py` | **The gate.** Every channel must normalize to mean ≈0, spread ≈1; separately warns when a fill sits outside the data range. Proven to catch the real bug. |
| `PanguWeather/v2.0/compute_e3sm_normalization.py` | Two-stage regeneration; moments are fill-independent. |
| `PanguWeather/v2.0/inspect_encoder_channels.py` | Per-channel encoder weight norms + cosines from a checkpoint. No GPU; mmap for the 19 GB file. |
| `physicsnemo_ai_rossby/polaris/verify_pangu_store.py` | 13 checks per store (its docstring says "six" — stale). |

---

## 7. Traps confirmed the hard way — do not rediscover these

1. **Channel order is a silent failure.** `ClimateZarrDataset` stacks tensors in
   **store-attrs** order while fills and loss build from **model-config** lists
   (`datapipes/climate/dataset.py:533` vs `examples/weather/ai_rossby/train.py:657`
   for the fill and `:735`/`:770` for the losses, all via `_surface_channel_names()`
   at `:92`). A permutation is correctly-shaped and `torch.cat` raises nothing.
   The preflight exists for this and is negative-tested.
2. **The venv needs `--extra datapipes-extras`.** The fork promoted
   `xarray`/`zarr`/`netCDF4` to core deps but **not `dask`**, which the converter
   imports. Without it conversion dies on the compute node *after queueing*.
3. **`uv` needs `conda activate base`.** `module load conda` alone puts neither
   `python` nor `uv` on PATH. (Not needed for §5, which calls the venv python by
   absolute path.)
4. **Never bare `torchrun`** — it can resolve to the base conda's launcher with the
   wrong python in its shebang. Use `python -m torch.distributed.run`.
5. **Both venvs install physicsnemo EDITABLE**, so each imports from whichever
   checkout built it. Both PBS scripts hard-fail `AI_ROSSBY_WRONG_CHECKOUT`.
6. **The level list trap.** PanguWeather carries `levels` (nominal hPa labels) AND
   `sigma_levels` (the values actually in the H5 keys). `use_sigma_levels: True`
   makes the latter authoritative. Comparing `levels` is a **false PASS**.
7. **Validation tail store.** PanguWeather's final validation IC is Dec 31 18:00
   with its target on Jan 1 of the next year — via the `linspace` endpoint at
   `data_loader_multifiles.py:507` (only on the `num_inferences > 0` path, which the
   E3SM configs use), **not** via the exclusive `val_year_end`. ai-rossby's sampler
   is arange-like, so a one-sample tail store restores that final IC; built
   automatically (`VAL_TAIL_YEAR`). `train/` deliberately gets none.
8. **Don't raise the zero-std floor.** `PRECT`'s real std is ~8.3e-8 (m/s). A
   `1e-6` floor would clobber it to 1.0 and destroy precipitation normalization.
   Only *exactly* zero std is degenerate — what `_std_corr` corrected (16 cloud levels).
9. **`Z` vs `Z_2`.** jesswan's originals use coord `Z`; the loader needs `Z_2`. Our
   regeneration writes `Z_2` directly, so this is retired for files we produce —
   any file taken from jesswan's archive still needs the rename.

---

## 8. Open items

* **jesswan's sign-off on both fills** — required before any resulting model's
  numbers are reported (DESIGN §1). The decisions (TSOI statistics regenerated at
  the unchanged 270 fill; SST 270 → −1.8) are ours pending that.
* **jesswan's own regeneration** had not started as of 2026-08-04 20:44 UTC. We are
  no longer blocked — we generate our own into `$PANGU_AUX`. **Any `.nc`/`.npz`
  later taken from jesswan's archive must pass `check_normalization.py` before
  use**; the current gating covers only files we produce.
* **PanguWeather's `_jsw` configs still carry `SST: 270`** — deliberately not
  edited, they are jesswan's. Only `_POLARIS*` and `tiny_baseline` (ours) were
  corrected.
* **The existing 85-epoch checkpoint is not salvageable by swapping statistics** —
  its weights were learned under the old scaling.
* **Unproven, does not change the plan:** whether SST's ocean signal was actually
  *learned*. The weight norm cannot tell (one vector serves both the land step and
  the ocean variation). A GPU perturbation probe would settle it, ~1 h.
* **Known-stale elsewhere, worth fixing:** `verify_pangu_store.py`'s docstring
  ("six checks", runs 13); `physicsnemo_sfno/polaris/e3sm_h5_to_seqzarr.py:90,:106`
  says TSOI land mean 268 K (measured 272); the CHANGELOG's 2026-08-04 entry has
  the stale `train.py:636,739` anchors and a "NOTHING SUBMITTED YET" line.

---

## 9. Out of scope

* **Retraining PanguWeather itself** — jesswan's call (§2a informs it).
* `embed_dim` tuning **for throughput/accuracy**; lowering it to fit 40 GB is
  in scope (§5 Step 3).
* The Poisson-fill and `mask_output: True` alternatives (§3).
* The `physicsnemo_sfno/` and `makani_sfno/` paths — **audited and clean**. Both
  already use `SST = -1.8` documented as degC, and both are immune to the
  fill/stats **disagreement** class (neither can ship stale statistics: one writes
  none and normalizes online, the other computes them in-stream from packed data).
  The fill value itself still matters to them — that is why physicsnemo's TSOI
  fill was changed 0.0 → 270 — and both already use the corrected values.
