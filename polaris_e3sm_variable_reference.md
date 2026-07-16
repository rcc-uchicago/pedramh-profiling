# E3SM archive — variable reference, measured

Every number here was **measured from the archive** on 2026-07-16, or read from a cited
`file:line`. Nothing is inferred from a variable name. Where a claim is inferred, it says so.

- **Archive:** `/eagle/lighthouse-uchicago/members/jesswan/AI4SRM/data/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/h5/plev_data/`
- **Shape:** 51,100 files (`{year}_{idx:04d}.h5`, 2015–2049 × 1460, noleap), one 6-h snapshot each,
  **162 channels + `time`** per file, every field `(180, 360)`. **2.15 TB** (51,100 × 42,051,856 B).
- **Units provenance:** netCDF attributes from `sigma_data/2020_Combined_EAM_ELM.nc` (atmospheric)
  and `boundary_data/{var}_masked.nc` (boundary). The `.h5` files carry **no** unit attributes —
  units cannot be recovered from the archive alone.
- **Ranges / σ / NaN%:** measured over 8 samples spanning 2015–2049 (`(2015,0/365/730/1095)`,
  `(2020,200)`, `(2030,800)`, `(2040,500)`, `(2049,1200)`). σ is over the filled field.

> Companion docs: `polaris_data_prep_decisions.md` (the open decisions),
> `polaris_data_prep_handoff_prompt.md` (converter state + how to verify a store).
> **Where those conflict with this file, this file was measured later and more directly.**

---

## Variable table

162 channels = **8 upper-air names × 18 levels (144)** + **18 surface names**. 26 distinct names.

| Variable | Lev | Units (from metadata) | dtype | Measured range | NaN % | σ (raw) | Used by | Risks |
|---|---:|---|---|---|---:|---:|---|---|
| `T` | 18 | `K` | float32 | 178.6 … 320.7 | 0 | 12.15 | P N M | R6 |
| `U` | 18 | `m/s` | float32 | −56.57 … 112.5 | 0 | 14.27 | P N M | R6 |
| `V` | 18 | `m/s` | float32 | −76.06 … 73.48 | 0 | 8.81 | P N M | R6 |
| `Z3` | 18 | `m` | float32 | −13.3 … 37,250 | 0 | 613.8 | P N M | R6 |
| `RELHUM` | 18 | `percent` | float32 | 6.4e-5 … **168.1** | 0 | 22.11 | P N M | R6, R8 |
| `CLDICE` | 18 | `kg/kg` | float32 | 0 … 8.8e-4 | 0 | 1.31e-5 | · N · | **R1**, R5, R6 |
| `CLDLIQ` | 18 | `kg/kg` | float32 | 0 … 7.9e-4 | 0 | 7.32e-6 | · N · | **R1**, R5, R6 |
| `CLOUD` | 18 | `fraction` | float32 | 0 … 1 | 0 | 0.219 | · N · | **R1** (top 5), R5, R6 |
| `PS` | 1 | `Pa` | float32 | 52,170 … 105,300 | 0 | 9,282 | P N M | — |
| `PSL` | 1 | `Pa` | float32 | 94,410 … 105,600 | 0 | 1,415 | P N · | — |
| `TREFHT` | 1 | `K` | float32 | 201.3 … 321.9 | 0 | 21.16 | P N M | R4b |
| `U10` | 1 | `m/s` | float32 | 6.6e-4 … 27.20 | 0 | 3.65 | P N · | — |
| `RHREFHT` | 1 | `1` ⚠ | float32 | 4.50 … **201.6** | 0 | 17.57 | P N · | R8, R11 |
| `TMQ` | 1 | `kg/m2` | float32 | 0.097 … 82.51 | 0 | 17.45 | P N · | — |
| `PRECT` | 1 | `m/s` | float32 | 0 … 1.87e-6 | 0 | **7.73e-8** | P N M | **R1 (worst)** |
| `FSNT` | 1 | `W/m2` | float32 | 0 … 1,324 | 0 | 309.1 | P N · | R7 |
| `FSNTOA` | 1 | `W/m2` | float32 | 0 … 1,324 | 0 | 309.1 | P N · | R7 |
| `SOILWATER_10CM` | 1 | `kg/m2` | float32 | 0 … 101.5 | 61.43 | 16.73 | P N · | R9 |
| `TSOI_10CM` | 1 | `K` | float32 | **0.0** … 326.9 | 61.43 | 25.95 | P N · | **R3**, R8, R9 |
| `SST` | 1 | `degC` ⚠ | float32 | −1.80 … 32.92 | 37.35 | 11.51 | P N M | **R2**, **R4**, R10, R11 |
| `ICE` | 1 | `Fraction` | float32 | 0 … 1 | 37.35 | 0.310 | P N M | **R2**, R10 |
| `sol_in` | 1 | `W/m2` | float32 | 0 … 1,406 | 0 | 399.9 | P N M | **R2** |
| `TOPO` | 1 | `m` | float32 | −66.86 … 5,601 | 62.65 | 1,132 | P N M | R9 |
| `PFTDATA_MASK` | 1 | `unitless` | float32 | **1 … 1** | 62.65 | 0 | P N M | R12 |
| `PCT_GLACIER` | 1 | `unitless` ⚠ | float32 | 0 … 100 | 62.65 | 45.29 | P N M | R11 |
| `PCT_NATVEG` | 1 | `unitless` ⚠ | float32 | 0 … 100 | 62.65 | 44.58 | P N · | R11 |

**Used by:** `P` = PanguWeather (108 ch), `N` = PhysicsNeMo (162 ch), `M` = makani (59 ch, 10/18 levels).
`·` = not read. Counts verified independently; makani keeps only the **lowest 10** of 18 levels.

### Roles and NaN fills, per pipeline

Cited: Pangu `PanguWeather/v2.0/config/E3SM_SFNO_H5_POLARIS.yaml:52-59,69-76`;
PhysicsNeMo `physicsnemo_sfno/polaris/e3sm_h5_to_seqzarr.py:38,63-71`;
makani `makani_sfno/polaris/convert_e3sm_to_makani.py:96-113`.

| Variable | Pangu role / fill | PhysicsNeMo role / fill | makani role / fill |
|---|---|---|---|
| `T U V Z3 RELHUM` | prognostic | prognostic | prognostic (10 lev) |
| `CLDICE CLDLIQ CLOUD` | **not read** | prognostic | **not read** |
| `PS TREFHT` | prognostic | prognostic | prognostic |
| `PSL RHREFHT TMQ U10` | prognostic | prognostic | **not read** |
| `FSNT FSNTOA` | diagnostic | prognostic | **not read** |
| `PRECT` | diagnostic | prognostic | diagnostic |
| `SOILWATER_10CM` | prognostic / `0.` | prognostic / `0.0` | **not read** |
| `TSOI_10CM` | prognostic / **`270.`** | prognostic / `0.0` | **not read** |
| `SST` | prescribed / **`270.`** | **prognostic** / `-1.8` | prescribed→`sst` / `-1.8` |
| `ICE` | prescribed / `0.` | **prognostic** / `0.0` | prescribed→`ice` / `0.0` |
| `sol_in` | prescribed | prescribed (UNPREDICTED) | prescribed→`solin` |
| `TOPO` | prescribed / `0.` | prescribed / `0.0` | prescribed→`topo` / `0.0` |
| `PFTDATA_MASK` | prescribed / `0.` | prescribed / `0.0` | prescribed→`lsm` / `0.0` |
| `PCT_GLACIER` | prescribed / `0.` | prescribed / `0.0` | prescribed→`glacier` / `0.0` |
| `PCT_NATVEG` | prescribed / `0.` | prescribed / `0.0` | **not read** |

PhysicsNeMo's split is only two-way: `UNPREDICTED` is exactly the 5 constants, so everything
else — including `SST`, `ICE` — is forecast **and scored**. Pangu has five roles, makani three.

---

## Risk register

### R1 — PhysicsNeMo's BatchNorm erases 42 channels, including precipitation ⛔ SEVERE

`train.py:120-126` normalizes with `nn.BatchNorm2d(momentum=None, affine=False)` on **raw
physical units**; default `eps=1e-5`. BatchNorm yields amplitude `σ/√(σ²+eps)`. With
`√eps ≈ 3.16e-3`, any channel whose σ is far below that is crushed toward zero.

**Measured: 42 of 162 channels land below amplitude 0.013** — `CLDICE` ×18, `CLDLIQ` ×18,
`CLOUD` ×5, and **`PRECT`**.

| channel | σ (raw) | post-norm amplitude |
|---|---:|---:|
| `PRECT` | 7.73e-8 m/s | **2.46e-5** |
| `CLDLIQ_256.7` | 7.84e-9 | 2.48e-6 |
| `T_998.5` (healthy, for contrast) | 21.06 | 1.0000 |

The loss compounds it: `batch_normalized_mse` (`train.py:51-61`) is a **global** L2 ratio over
all channels flattened, so a channel's gradient share scales as amplitude². PRECT's share is
**~6e-10** — roughly **1.7e9×** under-represented against a normalized channel. The model has no
gradient reason to learn precipitation, nothing errors, and the BatchNorm state is exported into
the inference package (`train.py:444-445`), making it permanent.

**This is a units bug in the training path, not a data defect, and the fix belongs there.**
The archive is fine: in mm/day, PRECT's σ is **6.72** → amplitude **1.0000**, perfectly healthy.
Fix by **supplying precomputed per-channel stats instead of normalizing online** (equivalently:
rescale after load, or lower `eps`). A converter-side rescale is *possible* but is the worse
option — it makes the store diverge in units from the archive it was built from, for no gain.
**R1 therefore does NOT gate the conversion**: convert first, fix training after.
Of the 42, ~19 are dead anyway (R5); the **~23 that carry real signal** are the loss.

### R2 — Boundary forcings are bit-identical every year ⛔ NEEDS jesswan

`SST`, `ICE`, `sol_in` are **bitwise identical across all 35 years** at the same index-in-year.
Verified: 1,224 md5 comparisons (12 indices × 34 years × 3 vars) + 480 random — **0 mismatches**;
distinct inodes (not hardlinks); valid cells compared by value (not a NaN artifact). Control:
atmospheric fields **never** matched (1,632 comparisons; `TREFHT` differs by up to 30.6 K).
Global SST mean is `14.574015 °C` in 2015, 2020, 2030, 2040 **and** 2049.

**Cause:** `boundary_data/{SST,ICE,sol_in}_masked.nc` each hold exactly **1460 steps = one year**
(`days since 2015-01-01`, calendar `365_day`). `netcdf-to-h5_e3sm.py::create_one_step_dataset`
re-slices from `chunk_id=0` for every year. Proven at data level: `nc[idx].astype(float32)` is
bitwise equal to the h5 field, 543/543 across 2016/2033/2038/2049.
The **frozen 2015 timestamp** in every file is a symptom of the same code path
(`list_time_stamps` is taken from the boundary vars, `:170-171`), not a separate bug.

**Intent is unresolved and this is not ours to decide.** A repeating annual SST cycle is a
standard *fixed-SST control* design, and the run is named `CTL_SST0051`. But `SSP245AMIP` is a
warming-scenario name, and scenario runs normally use an evolving ocean. **Two independent
analyses (one with no prior conclusions) both read this as probably deliberate.** Either way the
archive contains **no interannual ocean signal**.
Consequence if deliberate: PhysicsNeMo forecasting `SST`/`ICE` (R10) is predicting an exact
function of day-of-year.

### R3 — Pangu's `TSOI_10CM` fill contradicts its own stats ⚠ HIGH, and it is upstream's

- Pangu's loader fills `TSOI_10CM` with **270** (`E3SM_SFNO_H5_POLARIS.yaml:70`).
- `compute_normalization_e3sm.py` never sets `mask_fill['TSOI_10CM']`, so the dict comprehension
  (`:144-145`) defaults it to **0.0** — the shipped stats encode a **0-fill**.
- Measured: npz mean **105.229** / std **133.802**; predicted 0-fill **105.266 / 133.857**
  (0.03%), predicted 270-fill **271.13 / 16.43** (not close). Alternatives eliminated:
  `norm/old/*.npz` is md5-identical to the shipped file, and the norm job log shows **35 years**,
  killing the "stats used 2020–2049" explanation.

**Effect:** a *predicted* channel is ~**26× under-weighted** in the loss (0.197σ of signal
against a +1.25σ target offset). With matching stats it would carry **1.604σ**.

**Attribution:** the same `'TSOI_10CM': 270.` is in jesswan's own `E3SM_SFNO_H5_DERECHO_jsw.yaml:66`
and `_STAMPEDE_jsw.yaml:66`, and the loaders are byte-identical. **This is live in the group's
existing training runs, not introduced by the Polaris port.**

**Note the fill value itself is good** — 270 sits 0.02σ from the valid mean, effectively a
mean-fill. The defect is the stats not matching it. (`polaris_data_prep_decisions.md`'s
"✅ Kelvin, in-distribution" is correct *on that point*; this is a separate, additional defect.)
Curiously the knowledge existed: `e3sm_h5_to_seqzarr.py:56-63` documents that "the E3SM npz stats
were themselves computed under a 0-fill convention for that field" — it was never cross-checked
against Pangu's config.

### R4 — Pangu's `SST` fill of 270 is 8× outside a degC field ⚠ HIGH

`SST` is **degC** (metadata `units: degC`; measured [−1.80, 32.92]). Pangu fills land with **270**
(`:75`). Unlike R3 this is **self-consistent** — the npz was computed the same way: 270-filling
`SST_masked.nc` reproduces the shipped constants **exactly** (109.9630 / 123.9083 vs
109.962986 / 123.908279, ~1e-7).

**Provenance — inherited, not chosen.** The ancestors `compute_normalization.py:118-119` and
`compute_normalization_plasim.py:130-131` both carry `mask_fill['sst'] = 270.` and
`mask_fill['ts'] = 270.` — ERA5/PlaSim names for **Kelvin** fields, where 270 K is in-distribution.
The E3SM copy renamed them mechanically to `SST` (now degC, value never re-derived) and `TREFHT`.

**Effect:** measured signal **0.093σ** against a 2.06σ land-mask step; the std inflates 10.8×,
leaving the channel ~99.5% static land-mask variance. `SST` is **input-only** in Pangu, so there
is no direct loss impact — but at bf16 a 0.09σ signal retains only ~30 quantization levels.
The −1.8 fill used by PhysicsNeMo/makani yields **0.954σ** — the physical minimum, in-distribution.

> **Caveat on framing:** normalization is affine and invertible and the masks are static
> (verified bit-identical), so a first-layer weight can absorb this. Read R3/R4 as
> **"miscalibrated normalization"**, not "corrupted data". No empirical skill loss was
> demonstrated. The R3 loss-weighting harm is the concrete one.

#### R4b — `mask_fill['TREFHT'] = 270.` is dead code
`TREFHT` has **exactly zero NaN** in all 280 files checked. Proven twice: the npz TREFHT mean
(279.6635) matches the *unfilled* full-grid mean (279.6808) to 0.006%; and Pangu's `_fill_mask`
only fills `land_variables + ocean_variables`, which excludes TREFHT. Harmless; it is the
fingerprint of the mechanical `ts`→`TREFHT` rename that produced R4.

### R5 — 16 cloud channels carry no information ℹ️
`normalize_std.npz` has σ **exactly 0.0** for **16** channels: `CLDICE` ×4, `CLDLIQ` ×8,
`CLOUD` ×4 (verified over all 35 years — the norm log shows `years: 35/35`). Only PhysicsNeMo
trains on them: ~19 of its 157 predicted channels and ~12% of the store.

Two precisions: it is **not** "the top 4 levels" uniformly — `CLDLIQ` runs 8 deep, to ~145 hPa.
And "constant" is **strictly false** for `CLDLIQ_96.46` and `CLDLIQ_145.04`, which hold sparse
nonzero values (≤1.7e-21 on ≤0.3% of pixels); their variance (~1e-42) merely **underflows float32**.
"No usable information" stands.

**Open question, never asked:** are these zero in the *source* model output, or lost in the
netCDF→HDF5 step? `Gridded_EAM_Lev_Subset/` (1.9 TB) would settle it and has never been opened.

Not a normalization hazard for Pangu: `data_2015-2050_std_corr.nc` — the file its config actually
loads — already replaces those 16 zeros with 1.0.

### R6 — The 18 "plev" levels are terrain-following, not isobaric ⚠ EVALUATION HAZARD
Measured: `corr(Z3_998.5, TOPO) = 0.979` over land, `Z3_998.5 − TOPO` averaging **+15.4 m**;
`Z3_492.47` reaches 9,266 m where a true 500 hPa surface tops near 6,000 m. Top levels do behave
isobarically. **These are sigma (hybrid) levels despite the `plev_data/` directory name.**

Pangu knows (`use_sigma_levels: True`, and its config says so outright). **makani names them
`T850`, `Z500`, `PLEV_NOMINAL = [200, 250, …]`** (`convert_e3sm_to_makani.py:93`) — asserting
isobaric semantics the data lacks. Training stays self-consistent; **evaluation against ERA5 Z500
or standard climatologies will be badly biased over terrain.** Given makani exists for
PlaSim-baseline comparability, this matters for any scorecard.

### R7 — `FSNT` and `FSNTOA` are near-duplicates ℹ️
Measured `corr = 0.99999997` across 2015/2030/2049, max |diff| **0.27 W/m²** on a field reaching
1,324. Not bitwise identical (physically expected: model top vs TOA). PhysicsNeMo forecasts
**both**; Pangu takes both as diagnostics; makani takes neither. ~1 redundant predicted channel.

### R8 — Unmasked out-of-range values ℹ️ minor
- `TSOI_10CM`: exactly **1 stuck pixel = 0 K** inside the valid mask, at (47.5°N, 273.5°E), in
  every file checked; next-lowest valid value 65.4 K. Both physically impossible for soil.
- `RELHUM` reaches **168%**, `RHREFHT` **201.6%** — supersaturation genuinely retained by E3SM
  output. **Real values, not errors**; do not "fix".

### R9 — There is no single land/sea mask ⚠️
The masks are **not** interchangeable: `TSOI`/`SOILWATER` are valid on 1,221 px where `TOPO` is
NaN, and NaN on 433 px where `TOPO` is valid. `SST`-NaN and `TOPO`-NaN *are* exact complements
(intersection 0, union 64,800). **Per-field fills are required** — all three pipelines do this.

### R10 — PhysicsNeMo forecasts prescribed boundary conditions ⚠ DECISION
`SST` and `ICE` are absent from `UNPREDICTED` (`:38`), so they are forecast and scored. Pangu and
makani prescribe them. `SSP245AMIP` is an Atmospheric Model Intercomparison Project design, where
the ocean surface is imposed by definition. Combined with **R2** (those fields are an exact
function of day-of-year), 2 of 157 output channels forecast a lookup table.
One line: `UNPREDICTED += ["SST", "ICE"]`. Left alone because it changes what the model learns
(DESIGN §1). See `polaris_data_prep_decisions.md` Q1.

### R11 — Unit metadata is wrong or misleading on 4 fields ℹ️
- `RHREFHT`: attribute says `1` (dimensionless); values run 4.5–201.6 → it is **percent**.
- `PCT_GLACIER` / `PCT_NATVEG`: attribute says `unitless`; values run 0–100 → **percent**.
- `SST`: `long_name` is **"potential temperature"**, not sea-surface temperature. Units `degC`.
**A variable name is not evidence, and here neither is the attribute.** Measure.

### R12 — `PFTDATA_MASK` is `1`-or-`NaN`, not `0`/`1` ℹ️
Measured range **1 … 1**, σ = 0, NaN over 62.65%. It only becomes a binary land-sea mask *after*
the 0-fill. All three pipelines fill it with 0, so all three get a valid mask — but a consumer
that skipped the fill would get a constant.

---

## Training impact

**See [`data_for_training.md`](data_for_training.md)** — an adversarial assessment of which of
R1–R12 actually affect *training*, as opposed to evaluation, storage or interpretation. Several
are milder there than their entry above implies; only R1 clearly matters, and it is a
**training-code** fix, not a data or conversion fix.


## What is verified, and what is not

**Verified** — measured directly this session, and independently re-measured by an adversarial
agent over a disjoint 280-file sample: units, dtypes, ranges, NaN fractions and masks, σ per
channel, the frozen boundary (R2), both fill/stats findings (R3, R4), the 270 provenance (R4),
the channel counts 108/162/59, the role split, the 16 dead channels (R5), the BatchNorm
arithmetic (R1).

**Not verified:**
- Whether R2 is intentional. **Only jesswan can answer.**
- Whether the 19 dead cloud channels are zero in the *source* E3SM output (R5).
- R1's batch-σ ≈ global-σ step is inferred (the amplitude factor uses measured global σ);
  `momentum=None` makes the running estimate converge to global, but this was not run end-to-end.
- No empirical training-skill impact was demonstrated for **any** finding here. All are
  static-analysis + archive measurements.
- The exact converter *invocation* that built the archive: `netcdf-to-h5_e3sm.py`'s mtime
  (Jul 15) postdates the archive (Jul 8). The signature matches exactly, and the converter
  would `IndexError` on a >1460-step boundary file — so a completed archive is itself evidence
  of a one-year source — but this is inference, not proof.

## Volume — the budget in circulation is ~2× wrong

Measured **2.15 TB** source (51,100 × 42,051,856 B). A full 162-channel float32 uncompressed
PhysicsNeMo store is **also ~2.15 TB**; makani's 59-channel pack ~0.78 TB.
`polaris_data_prep_handoff_prompt.md` repeats **"~1 TB"** throughout. It also records the project
at **15.18 TB against a 15 TB quota** (2026-07-15). **Check `myquota` before any conversion** —
dropping the 42 R1/R5 channels would cut ~26% of the store.
