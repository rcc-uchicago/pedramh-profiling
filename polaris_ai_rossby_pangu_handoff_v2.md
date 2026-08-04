# HANDOFF v2 — PanguPlasim-on-E3SM (ai-rossby) on Polaris, with CORRECTED normalization

Supersedes `polaris_ai_rossby_pangu_handoff_prompt.md` (v1). Read this whole file
first. Work on branch `fix/tsoi-fill-270` in
`/eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling`.

**Goal:** train ai-rossby's **`PanguPlasimLegacy`** on the E3SM archive with the
**PanguWeather 108-field variable set**, on Polaris, logging to wandb.

**What changed since v1, and it is the headline:** v1 was *parity-first* — it
deliberately reproduced PanguWeather's fills, defects included. That is
**abandoned**. Two normalization defects were found and measured; both
PanguWeather and ai-rossby are being retrained with the fixes.

---

## 1. The one distinction this whole document rests on

> **Parity is asserted on the VARIABLE SET. It is deliberately NOT asserted on
> the fill values or the normalization statistics.**

* **Variable set — IDENTICAL, and machine-checked.** Same 108 fields, same seven
  groups, same order within each group, same 18 levels. `VARIABLE_PARITY_OK
  10/10` against jesswan's trained config. This has not changed and must not.
* **Fills + statistics — deliberately CORRECTED on both sides.** Reproducing
  them would reproduce two real defects.

Anywhere this document says "parity", it means the first bullet only.

---

## 2. The two defects, measured

### 2a. `TSOI_10CM` — statistics computed under the wrong fill ⛔ the serious one

The **fill (270 K) is correct and must not change** — soil temperature is Kelvin
and the valid-land mean is 272 K, so 270 is a near mean-fill. The defect is that
the shipped `.nc` statistics were computed under a **0-fill**:

| statistics computed over | mean | std | resulting normalized spread |
|---|---:|---:|---:|
| 0-filled data | 104.98 | 133.49 | **0.122** ← what shipped |
| land only (the near-miss) | 272.20 | 26.25 | 0.622 |
| **270-filled data (correct)** | **270.85** | **16.34** | **1.0000** |

Cause: `compute_normalization.py:116-119` builds a fill dict defaulting every
field to `0.`, then overrides only `sst` and `ts`. The soil-temperature entry
falls through to zero.

**Consequence — and this is why it justifies a retrain.** Verified that the
scaling does *not* cancel: `data_loader_multifiles.py:792` normalizes and
`train.py:1573` takes `loss_obj_sfc` (MSELoss, `raw_l2`) on the normalized
tensors, including targets. Measured normalized spread of the 8-wide surface
tensor:

```
TREFHT 1.006  U10 0.998  RHREFHT 0.975  PS 0.993
PSL 0.950  TMQ 1.000  SOILWATER_10CM 1.008
TSOI_10CM 0.122   <-- the ONLY outlier
```

MSELoss is an element-mean, so soil temperature's squared error counted **~1/67th**
of every sibling channel's. It is a **forecast, scored** field that also re-enters
as input each rollout step, and it is land-surface memory at S2S range.
Corroboration: it has the **highest** encoder weight norm of the 15 named channels
(1.91× median) — the model straining to amplify a nearly-flat input.

### 2b. `SST` — a Kelvin constant on a Celsius field ⚠ real, but minor

E3SM's `SST` is **degC** (measured `[-1.80, 32.92]`; the floor is exactly −1.8 at
6.01% of ocean cells, all poleward of 60° with mean sea-ice fraction 0.93 — the
seawater freezing clamp). It was filled with **270**, which is 270 °C.

Unlike TSOI, its fill and statistics **agreed** (both built on 270), so the channel
normalized to spread 1.0 and was correctly weighted. The cost was input precision:
the ocean's real variation occupied 0.093σ, ~**75 distinct bf16 values** (~0.47 °C
each) versus ~2670 with a sane fill. Training ran bf16 (confirmed in the run log).

**Why this alone would NOT justify a retrain** — measured, and worth knowing so
nobody overstates it:

* `SST` is **input-only**: a prescribed boundary, `ocean_variables: []`, never in
  the loss.
* `SST` is **bit-identical across all 35 years** and takes only **~12 distinct
  states per year** — a prescribed monthly climatology. Across 43,800 training
  samples it supplies twelve distinct fields.
* `sol_in` uniquely identifies the timestep-of-year (46/46 sampled), so `SST` is
  derivable from an input the model already has.
* The trained model reads it at **0.89× the median** weight norm, rank 70 of 105 —
  neither ignored nor amplified. No pathology.

So: fix it, but fix it **in the same retrain as TSOI**. It is not the reason.

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
a degenerate field** — filling 0 is what turns it into a real 0/1 land–sea mask.
Leave it.

**Why −1.8 for SST** (adversarial + cold review, 2 of 3 for it; the dissenter
conceded ocean-mean beat 0 °C on every metric, leaving 0 stranded): the candidates
are within noise on signal (0.954σ vs 1.000σ) and bf16 levels (2668 vs 2683).
−1.8 is a value the data itself takes, it matches physicsnemo and makani so all
three pipelines agree, and it applies the same principle that fixed TSOI —
in-distribution physical states, not convenient zeros. Confusability (−1.8 collides
with 10.08% of ocean cells) is neutralized: the model separately receives a land
mask, glacier/vegetation fraction, topography, and `ICE` on SST's exact mask.

*Recorded, not adopted:* a Poisson/diffusion fill would uniquely remove the
coastline discontinuity a spectral model feels, and SST has only 12 states so it
is cheap to precompute. The cold reviewer's judgment: the right **fallback**, only
if coastal ringing is actually demonstrated.

### Expected normalization constants

Cross-check targets, from a 24-file test pass:

| field | mean | std |
|---|---:|---:|
| `SST` (fill −1.8) | **8.44** | **12.06** |
| `TSOI_10CM` (fill 270) | **271.09** | **16.39** |

For reference, the old/defective values were `SST` 109.963 / 123.908 and
`TSOI_10CM` 105.229 / 133.802.

`SST` should match tightly on the full 35-year run — the field is bit-identical
across years, so 35-year statistics equal any single year's. `TSOI_10CM` genuinely
varies year to year, so expect ~1% drift from the test-pass numbers.

**The rule, which is the whole fix:** compute the statistics over **exactly the
array the model receives — i.e. after the fill**. Land-only statistics are the
plausible near-miss and still wrong (0.622 spread).

---

## 4. State: what is DONE and verified

| item | status |
|---|---|
| Variable-parity gate | ✅ `VARIABLE_PARITY_OK 10/10` + 16/16 artifacts |
| `physicsnemo_ai_rossby/` subtree | ✅ vendored unsquashed, `87002adb` a real ancestor |
| ai-rossby venv | ✅ `AI_ROSSBY_VENV_OK` — torch 2.10.0+cu129, zarr 3.2.1 |
| Code edits (`sol_in`, land/ocean lists, channels) | ✅ committed |
| Model + dataset configs | ✅ written, contract-checked |
| **E3SM → zarr conversion (smoke split)** | ✅ **`CONVERT_ALL_OK`, job 7337122** |
| Store verification | ✅ `PANGU_STORE_VERIFIED 13/13` ×3, bitwise `max\|diff\| = 0` |
| Training launcher + preflight | ✅ written, preflight negative-tested |
| **Corrected normalization** | 🟡 **job 7337234 queued** |
| **Training smoke** | ⬜ **not yet run — the next real gate** |

Stores on disk (~60 GB): `$AI_ROSSBY_DATA/e3sm/{train/2015.zarr, val/2045.zarr,
val/2046.zarr}` (the 1-sample 2046 is the validation tail store, §7).

**The stores are fill- and stats-agnostic — they preserve raw NaN.** Filling
happens at training time from the dataset config. So the normalization fix does
**not** require reconversion.

---

## 5. Do this next, in order

```bash
cd /eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling
source polaris_env.sh
```

### Step 1 — wait on the normalization job (7337234)

PASS = `MOMENTS_OK` → `NORM_NC_OK` → `NORMALIZATION_OK` → `E3SM_NORM_REGEN_OK`.
Writes `$PANGU_AUX/data_2015-2050_{mean,std_corr}.nc` (**never** touches jesswan's
originals) plus `moments_2015-2050.json`.

That JSON is the reusable artifact: it stores fill-independent moments, so
regenerating for a *different* fill is seconds and needs no re-read of 2 TB:

```bash
$AI_ROSSBY_VENV/bin/python PanguWeather/v2.0/compute_e3sm_normalization.py \
  --stage nc --moments $PANGU_AUX/moments_2015-2050.json \
  --config PanguWeather/v2.0/config/E3SM_SFNO_H5_POLARIS.yaml \
  --fill SST=0.0 --out-dir /tmp/alt --tag ALT
```

### Step 2 — rebuild the ai-rossby normalization zarr

⚠ **Required before the smoke.** The zarr currently on disk was built from the
OLD defective statistics; the dataset config now specifies `SST: -1.8`. Pairing
them would recreate the exact mismatch this work removes, and the preflight will
**not** catch it (it checks channel names and order, not statistics).

```bash
$AI_ROSSBY_VENV/bin/python \
  physicsnemo_ai_rossby/tools/data/e3sm/build_normalization_zarr.py \
  --source-dir $PANGU_AUX --std data_2015-2050_std_corr.nc \
  --output $AI_ROSSBY_DATA/e3sm/norm/normalization_2015-2050.zarr --overwrite
```

### Step 3 — the training smoke (the next real gate)

```bash
cd physicsnemo_ai_rossby && qsub polaris/polaris_pangu_plasim.pbs
# faster scheduling: qsub -q debug polaris/polaris_pangu_plasim.pbs
```

PASS = `PREFLIGHT_OK` + advancing `train/*` metrics, finite and decreasing.

**Four things have never actually executed** — expect the failure here:

1. Does `PanguPlasimLegacy` construct? The `sol_in` patch should clear the
   `ValueError`, untested on a real config.
2. Do the tensors line up? 8-channel surface vs the model's `[surface|land|ocean]`
   slicing, and the fill/loss name lists.
3. **Does it fit in 40 GB?** `embed_dim 240`, 108 channels, 180×360, bf16,
   batch 1. Untested at this geometry — the most likely failure. Levers in order:
   `checkpointing`, then `embed_dim`.
4. Does `window_size [2,6,10]` divide 18 levels? An assumption carried from the
   S2S config; `window_size` is the knob.

### Step 4 — production

```bash
# conversion: ~11 h, ~1.08 TB, ~1.14M inodes
cd physicsnemo_ai_rossby && qsub \
  -v TRAIN_YEARS="$(seq -s' ' 2015 2044)",VAL_YEARS="2045 2046 2047 2048" \
  polaris/polaris_e3sm_pangu_convert.pbs
```

Then the full training run. Quota is fine on bytes (16.17 of 50 TB) — **inodes
are the constraint to watch**; sibling clusters have hit caps on this store shape.

---

## 6. Tools built — use these rather than re-deriving

| tool | what it does |
|---|---|
| `ai_rossby_variable_contract.py` | The variable contract in ONE place. `--check-ground-truth` (vs jesswan's YAML), `--check-artifacts` (vs configs/converter/store attrs). stdlib-only — **needs `python3.12`**, the login node's bare `python3` is 3.6. |
| `PanguWeather/v2.0/check_normalization.py` | **The gate.** Asserts every channel normalizes to mean ≈0, spread ≈1, and separately warns when a fill sits outside the data's range. Proven to catch the real bug. |
| `PanguWeather/v2.0/compute_e3sm_normalization.py` | Two-stage regeneration. Moments are fill-independent, so any fill is instant afterwards. |
| `PanguWeather/v2.0/inspect_encoder_channels.py` | Per-channel encoder weight norms + cosines from a checkpoint. No GPU; mmap for the 19 GB file. |
| `physicsnemo_ai_rossby/polaris/verify_pangu_store.py` | 13 checks per store: attrs vs contract, 108-field count, levels, time axis, bitwise vs h5, NaN positions. |

---

## 7. Traps confirmed the hard way — do not rediscover these

1. **Channel order is a silent failure.** `ClimateZarrDataset` stacks tensors in
   **store-attrs** order while fills and loss build from **model-config** lists
   (`dataset.py:533` vs `train.py:636,739`). A permutation is correctly-shaped and
   `torch.cat` raises nothing. The preflight exists for this — it is
   negative-tested (pointing `--store` at the norm store makes 5 checks FAIL).
2. **The venv needs `--extra datapipes-extras`.** The fork promoted
   `xarray`/`zarr`/`netCDF4` to core deps but **not `dask`**, which the converter
   imports. Without it conversion dies on the compute node *after queueing*.
3. **`uv` needs `conda activate base`.** `module load conda` alone puts neither
   `python` nor `uv` on PATH.
4. **Never bare `torchrun`** — it can resolve to the base conda's launcher with the
   wrong python pinned in its shebang. Use `python -m torch.distributed.run`.
5. **Both venvs install physicsnemo EDITABLE**, so each imports from whichever
   checkout built it. Both PBS scripts hard-fail `AI_ROSSBY_WRONG_CHECKOUT` rather
   than silently run another tree.
6. **The level list trap.** PanguWeather carries `levels` (nominal hPa labels) AND
   `sigma_levels` (the values actually in the H5 keys). `use_sigma_levels: True`
   makes the latter authoritative. Comparing `levels` is a **false PASS**.
7. **Validation tail store.** PanguWeather's final validation IC is Dec 31 18:00
   with its target on Jan 1 of the next year — via the `linspace` endpoint at
   `data_loader_multifiles.py:507` (only on the `num_inferences > 0` path, which
   the E3SM configs use), **not** via the exclusive `val_year_end`. ai-rossby's
   sampler is arange-like, so a one-sample tail store restores that final IC. Built
   automatically (`VAL_TAIL_YEAR`). `train/` deliberately gets none — PanguWeather's
   `_compute_train_inference_idxs` already drops the same final step.
8. **Don't raise the zero-std floor.** `PRECT`'s real std is ~8.3e-8 (m/s). A
   `1e-6` floor would clobber it to 1.0 and destroy precipitation normalization.
   Only *exactly* zero std is degenerate — that is what `_std_corr` corrected
   (16 cloud levels).
9. **`Z` vs `Z_2`.** jesswan's originals use coord `Z`; the loader needs `Z_2`.
   Our regeneration writes `Z_2` directly, so this trap is retired for files we
   produce — but any file taken from jesswan's archive still needs the rename.

---

## 8. Open items

* **jesswan's sign-off on both fills** — still required before any resulting
  model's numbers are reported (DESIGN §1). The evidence is in this document and
  in `ai_rossby_panguweather_variable_parity.md`; the decisions (TSOI stats
  regenerated at the unchanged 270 fill; SST 270 → −1.8) are ours pending that.
* **jesswan's own regeneration** had not started as of 2026-08-04 20:44 UTC (files
  still Jul 8, contents still the old values). We are no longer blocked on it —
  we generate our own into `$PANGU_AUX`. When theirs lands, reconcile.
* **PanguWeather's `_jsw` configs still carry `SST: 270`** — deliberately not
  edited, they are jesswan's. Only the `_POLARIS*` and `tiny_baseline` configs
  (ours) were corrected.
* **The existing 85-epoch checkpoint is not salvageable by swapping statistics** —
  its weights were learned under the old scaling, so corrected stats are
  out-of-distribution for it. New statistics mean a fresh run.
* **Unproven, and it does not change the plan:** whether SST's ocean signal was
  actually *learned*. The weight norm cannot tell (one vector serves both the land
  step and the ocean variation). A GPU perturbation probe would settle it, ~1 h.

---

## 9. Out of scope

* Retraining PanguWeather itself (jesswan's).
* The `physicsnemo_sfno/` and `makani_sfno/` paths — **audited and clean**. Both
  already use `SST = -1.8` documented as degC, and both are **structurally immune**
  to the fill/stats class: one writes no statistics (online BatchNorm), the other
  computes them in-stream from the packed data. Nothing to change there.
* `embed_dim` tuning; the Poisson-fill alternative (§3).
