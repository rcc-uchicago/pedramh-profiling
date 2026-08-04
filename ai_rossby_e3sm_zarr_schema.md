# ai-rossby E3SM per-year Zarr — schema, layout, and the production split

Companion to `ai_rossby_panguweather_variable_parity.md` (which proves *which* variables) —
this is *how they are stored*, and *which years* go where.

Everything below is **measured**, not specified: the geometry comes from a real 8-timestep
store built by `tools/data/e3sm/pangu_h5_to_zarr.py` and checked by
`polaris/verify_pangu_store.py` (`PANGU_STORE_VERIFIED 13/13`, bitwise `max|diff| = 0`).

---

## 1. Directory layout

```
$AI_ROSSBY_DATA/e3sm/
├── train/                              <- cfg dataset.zarr_path
│   ├── 2015.zarr
│   ├── 2016.zarr
│   └── …                               one store per year
├── val/                                <- cfg dataset.val_zarr_path
│   ├── 2045.zarr
│   └── …
└── norm/
    └── normalization_2015-2050.zarr    <- cfg mean_path / std_path
```

> ⚠ **The normalization store MUST NOT live in `train/` or `val/`.**
> `ClimateZarrMultiYearDataset` does `sorted(root_path.glob("*.zarr"))`
> (`datapipes/climate/multiyear.py:98`) and builds a `ClimateZarrDataset` from **every**
> match. A co-located norm store would be loaded as a year of data and fail the layout
> assertion — or worse, be silently sorted into the time axis. Hence the three sibling dirs.

`train/` and `val/` are **directories**, not files: the config points at the directory and
the loader globs it. Adding a year is dropping a `.zarr` in; there is no year-range key.

---

## 2. Store schema (one year)

Zarr **v3**, `consolidated=True`, codecs `bytes(little) → zstd(level=0)`, `fill_value: NaN`.

### Arrays — 23 variables, 108 fields

| Array | Shape | Chunk | dtype | Group |
|---|---|---|---|---|
| `T`, `U`, `V`, `Z3`, `RELHUM` | `(time, 18, 180, 360)` | `(1, 18, 180, 360)` | float32 | `pressure_upper_air_variables` |
| `TREFHT`, `U10`, `RHREFHT`, `PS`, `PSL`, `TMQ` | `(time, 180, 360)` | `(1, 180, 360)` | float32 | `surface_variables` (surface) |
| `SOILWATER_10CM`, `TSOI_10CM` | `(time, 180, 360)` | `(1, 180, 360)` | float32 | `surface_variables` (**land, folded**) |
| `FSNTOA`, `FSNT`, `PRECT` | `(time, 180, 360)` | `(1, 180, 360)` | float32 | `diagnostic_variables` |
| `SST`, `ICE`, `sol_in` | `(time, 180, 360)` | `(1, 180, 360)` | float32 | `varying_boundary_variables` |
| `PCT_GLACIER`, `PFTDATA_MASK`, `PCT_NATVEG`, `TOPO` | `(180, 360)` | `(90, 360)` | float32 | `constant_boundary_variables` |

**Field count:** 5 × 18 + 6 + 2 + 3 + 3 + 4 = **108**.

The four constant-boundary fields carry **no time axis** — they are written once from the
first file of the year and read once at dataset init (`_eager_load_constants`), not per
sample. (They are also bit-identical across all years — R2.)

Chunking is **one timestep per chunk**, matching the access pattern: the sampler draws
`(start_t, lead_t)` pairs, so a sample touches exactly two time indices. An upper-air chunk
holds all 18 levels because a sample always wants the full column.

### Coords

| Coord | Shape | dtype | Notes |
|---|---|---|---|
| `time` | `(1460,)` | int64 (cftime) | **`noleap`** calendar, 6-hourly ⇒ 365 × 4 = 1460/yr |
| `pressure_level` | `(18,)` | float32 | E3SM hybrid levels in hPa, **terrain-following, not isobaric** |
| `lat` | `(180,)` | float32 | −89.5 … +89.5, cell-centred (**no pole row**) |
| `lon` | `(360,)` | float32 | 0.5 … 359.5 |

`pressure_level` is float32, so it differs from the archive's float64 literals at ~1e-7
relative. Both consumers tolerate it: train.py's level-subset guard only fires for a strict
subset, and the normalizer matches nearest-value at `atol=1e-3`.

### Group attrs — the channel-order contract

```
calendar                       "noleap"
data_timedelta_hours           6
e3sm_zarr_schema_version       "1.0"
surface_variables              [TREFHT, U10, RHREFHT, PS, PSL, TMQ, SOILWATER_10CM, TSOI_10CM]
constant_boundary_variables    [PCT_GLACIER, PFTDATA_MASK, PCT_NATVEG, TOPO]
varying_boundary_variables     [SST, ICE, sol_in]
diagnostic_variables           [FSNTOA, FSNT, PRECT]
pressure_upper_air_variables   [T, U, V, Z3, RELHUM]
sigma_upper_air_variables      []
year_index                     2015
sample_range                   [0, 1460]
source_input_dir               "<E3SM_ROOT>/h5/plev_data"
```

> ⚠ **These attrs are load-bearing and the framework does not check them.**
> `ClimateZarrDataset._build_sample` stacks each tensor in **attrs order**
> (`dataset.py:533`), while the NaN fill and the loss are built from the **model config's**
> lists (`train.py:636`, `:739`). A permutation yields correctly-shaped tensors with
> transposed channels, and `torch.cat` raises nothing. `ai_rossby_variable_contract.py
> --check-artifacts --store <s>` asserts attrs == contract, and the PBS launcher runs it as
> a preflight before spending node-hours.

### Why land is folded into `surface_variables`

The store schema has **no** `land_variables` / `ocean_variables` groups — `ClimateZarrStoreLayout`
defines only surface / constant-boundary / varying-boundary / diagnostic / pressure-upper-air /
sigma-upper-air (`dataset.py:57-75`). `PanguPlasimLegacy` expects its surface tensor to carry
`num_surface + num_land + num_ocean` channels sliced **in that order**
(`pangu_plasim_legacy.py:567, 673-678`). So the store's `surface_variables` is
`[6 surface | 2 land | 0 ocean]` while the model config keeps the three lists separate and
keeps its distinct land head. **No variable is added, removed, or re-roled.**

### NaN is stored raw

Filling is the training pipeline's job (`NanFillTransform`, from `conf/dataset/*.yaml`).
A pre-filled store would silently double-apply or contradict the config. Measured NaN
fractions, preserved cell-for-cell from the source:

| Fields | NaN | Fill applied downstream |
|---|---:|---|
| `SOILWATER_10CM`, `TSOI_10CM` | 61.43% (ocean) | `0.0` / **`270.0` K** |
| `SST`, `ICE` | 37.35% (land) | **`270.0`** ⚠ degC field / `0.0` |
| `PCT_GLACIER`, `PFTDATA_MASK`, `PCT_NATVEG`, `TOPO` | 62.65% | `0.0` |
| everything else | 0% | — |

See `ai_rossby_panguweather_variable_parity.md` §4.8 for why `SST = 270` on a degC field is
reproduced deliberately (R4, inherited, self-consistent with the shipped stats) and why
`TSOI_10CM = 270` is simply correct (Kelvin, land mean 268 K).

---

## 3. Measured size and inode cost

From the 8-timestep store (192 files, 173.4 MB; raw float32 payload 216.7 MB ⇒ **1.25×**
compression — climate fields at float32 barely compress):

| Scope | Size | Files |
|---|---:|---:|
| One timestep | ~21.7 MB | 23 |
| **One year** (1460 steps) | **~31.7 GB** | **~33,600** |
| Smoke (1 train + 1 val year) | ~63 GB | ~67,000 |
| **Production** (34 stores, §4) | **~1.08 TB** | **~1,142,000** |

> ⚠ **Inodes are the binding constraint, not bytes.** ~1.14 M files for the production set,
> from one-timestep chunking. Space is comfortable (16.17 of 50 TB used project-wide), but
> check the file count before the full conversion — sibling clusters have hit inode caps on
> exactly this store shape (Derecho's ~26.2 M-file cap, per the fork's own CLAUDE.md).
> If it becomes a problem the lever is `--write-batch` plus a multi-timestep chunk, at the
> cost of read amplification per sample.

---

## 4. Production split

Reproducing `E3SM_SFNO_H5_POLARIS_ALLDATA.yaml`'s all-data split:

```yaml
train_year_start: 2015
train_year_end:   2045   # exclusive -> trains on 2015..2044 (30 years)
val_year_start:   2045
val_year_end:     2049   # exclusive -> validates on 2045..2048
```

**ai-rossby has no year-range keys.** The split is expressed as *which stores are in which
directory*:

| | PanguWeather config | ai-rossby directory | Stores | Timesteps |
|---|---|---|---:|---:|
| Train | `2015 → 2045` (excl.) | `e3sm/train/` = `2015.zarr … 2044.zarr` | 30 | 43,800 |
| Val | `2045 → 2049` (excl.) | `e3sm/val/` = `2045.zarr … 2048.zarr` **+ `2049.zarr` (1 sample)** | 4 + tail | 5,840 **+ 1** |

Submit as:

```bash
qsub -v TRAIN_YEARS="$(seq -s' ' 2015 2044)",VAL_YEARS="2045 2046 2047 2048" \
     polaris/polaris_e3sm_pangu_convert.pbs
```

The **one-sample 2049 tail store is built automatically** — `VAL_TAIL_YEAR` defaults to
`max(VAL_YEARS) + 1` and is converted with `--sample-range 0 1`. Set `VAL_TAIL_YEAR=none`
to skip it. §4.1 explains why it is required for parity.

At ~20 min/year that is ~11 h of single-node time — past the 72 h queue max is not a
concern, but split it across a few submissions or use `polaris_submit_chain.sh` so a
preemption doesn't lose the lot. Each year is independent, so re-running one year is cheap.

### 4.1 The tail store — why validation needs a one-sample 2049

**Cross-year forecast pairs work — inside a directory.** The composite dataset glues the
per-year stores into one contiguous global time index and dispatches a pair's start and
target to whichever year covers each (`multiyear.py:71-74`). So a start at 2020-12-31 18:00
correctly reads its target from `2021.zarr`. That reproduces PanguWeather's continuous
date-range behaviour across the 30 training years.

**But the directory tail loses `max(forecast_lead_times)` starts.** `LeadTimePairSampler`
uses `valid_starts = dataset_length - max_lead` (`samplers.py:139-140`). At
`forecast_lead_times: [1]` that is **one** dropped start per directory: the final timestep,
whose target would fall outside the directory.

Whether that matters depends on a PanguWeather detail worth getting exactly right, because
the reference config's own comment states it imprecisely. Traced through the source and
verified numerically:

1. `partition_date_range` builds the date axis with **`np.arange(start_h, end_h, step)`**
   (`data_loader_multifiles.py`), so `end_date` is **exclusive**. `val 2045 → 2049` is
   therefore **5,840** timesteps ending **2048-12-31 18:00** — the date array contains **no
   2049 entry at all**. (Confirmed against jesswan's job log, which prints
   `Hours: 43794.0` for a 5-year 2045→2050 range: 43,800 − 6.)
2. `max_inference_idx = len(dates) - lead` = **5,839** (`:505`).
3. The last IC then depends on `num_inferences` (`:507` vs `:509`):

| `num_inferences` | index selection | ICs | last IC | its target |
|---|---|---:|---|---|
| `0` | `arange(0, 5839)` — excludes endpoint | 5,839 | 2048-12-31 12:00 | 2048-12-31 18:00 (in range) |
| **`128`** (the full-run value) | `linspace(0, 5839, 129)` — **includes** endpoint | 129 | **2048-12-31 18:00** | **2049-01-01 00:00** → reads `2049_0000.h5` |

So it is the **`linspace` endpoint**, not the exclusive `val_year_end`, that reaches into
2049 — and only on the `num_inferences > 0` path, which is what jesswan's E3SM configs use
(`num_inferences: 128` full, `8` smoke).

**Consequence for us:** ai-rossby's sampler is `arange`-like, so a bare `val/2045..2048`
stops at 2048-12-31 12:00 — one IC short of the reference. A **one-sample 2049 store**
restores it, and the converter job builds it automatically.

**Verified, on real stores** (an 8-step 2015 + a 1-step 2016 in one directory):

```
sub lengths [8, 1] -> composite 9;  valid starts = N - max_lead = 8
  last IC     global 7 -> 2015.zarr local 7  time 2015-01-02 18:00:00
  its target  global 8 -> 2016.zarr local 0  time 2016-01-01 00:00:00

WITHOUT the tail store: composite 8, valid starts 7 -> last IC would be local 6
```

Every real timestep of the full years is a usable IC, and the last one's target resolves
into the tail store. Scaled to production: `5840 + 1 = 5841` composite, `valid_starts =
5840` — ICs spanning **2045-01-01 00:00 … 2048-12-31 18:00**, final target
**2049-01-01 00:00**. That is PanguWeather's `val_year_end: 2049` semantics exactly.

Two things that make this safe, both checked rather than assumed:

* **The loader accepts the short store.** `_assert_layouts_match` compares variable groups,
  calendar, `data_timedelta_hours` and the level lists — **not** time length
  (`multiyear.py`). Confirmed identical across a full and a 1-sample store.
* **The short store passes verification.** `PANGU_STORE_VERIFIED 13/13` on the 1-sample
  store — 108 fields bitwise vs h5, 290,408 NaN cells in the same positions.

Cost: **~22 MB**, one file-read. `VAL_TAIL_YEAR=none` skips it and gives back the one-IC
shortfall.

**What parity does and does not mean here.** With the tail store, the two runs agree on the
*span* of reachable validation ICs — 2045-01-01 00:00 through 2048-12-31 18:00, final target
2049-01-01 00:00. They do **not** agree on IC *density*: PanguWeather subsamples 129 ICs via
`linspace`, while ai-rossby's sampler walks all 5,840. That is a validation-protocol
difference, not a data one, and it is out of scope for this store.

> **`train/` needs no tail store, and adding one would diverge.** PanguWeather's
> `_compute_train_inference_idxs` "excludes the final `timedelta_hours / data_timedelta_hours`
> samples of every contiguous range so that `dates[idx + lead]` always lives in the same
> range" — which is *exactly* ai-rossby's `valid_starts = N - max_lead`. Both drop
> 2044-12-31 18:00. Verified against the job log: a 25-year training range printed
> `Len(inference_idxs): 36499` = 36,500 − 1. Already identical; leave it alone.

### An alternative, if parity is not the goal

The archive is **2015–2049, all 35 years complete** (51,100 files = 35 × 1460 — verified).
PanguWeather leaves 2049 almost entirely unused purely because its `val_year_end` is
exclusive. A non-parity run could use `val/` = `2045.zarr … 2049.zarr` (5 full years, 7,300
timesteps, +25% validation) at no data cost. **Not** what the parity run does — noted so the
choice is deliberate rather than inherited.

---

## 5. Build and verify

```bash
# 1. Normalization store — LOGIN node, seconds. Note --std ...std_corr.nc:
#    the uncorrected std has zero-std channels that would divide to inf.
$AI_ROSSBY_VENV/bin/python tools/data/e3sm/build_normalization_zarr.py \
  --source-dir $PANGU_AUX --std data_2015-2050_std_corr.nc \
  --output $AI_ROSSBY_DATA/e3sm/norm/normalization_2015-2050.zarr

# 2. Per-year stores — COMPUTE node. Converts, then verifies each store.
qsub polaris/polaris_e3sm_pangu_convert.pbs          # defaults: train 2015, val 2045

# 3. Verify any store by hand
$AI_ROSSBY_VENV/bin/python polaris/verify_pangu_store.py \
  --store $AI_ROSSBY_DATA/e3sm/train/2015.zarr \
  --h5-dir $E3SM_ROOT/h5/plev_data \
  --norm  $AI_ROSSBY_DATA/e3sm/norm/normalization_2015-2050.zarr
```

`verify_pangu_store.py` runs 13 checks: the six attr groups + calendar, 108-field count,
levels vs contract **and** vs the norm store, time-axis monotonicity, bitwise equality vs
the source h5 over N random timesteps (**exact**, not `allclose` — the converter only
reshapes), and NaN in the *same cells* as the source. PASS = `PANGU_STORE_VERIFIED`.

**Environment note:** the venv needs `--extra datapipes-extras`. The ai-rossby fork promoted
`xarray`/`zarr`/`netCDF4` to core deps but **not `dask`**, which the converter imports to
allocate the Zarr template. Without it the conversion dies `ModuleNotFoundError` — on the
compute node, after queueing. `polaris_setup_ai_rossby_venv.sh` includes it and verifies it.

---

## 6. Source

| | |
|---|---|
| Archive | `$E3SM_ROOT/h5/plev_data` — `E3SMv3_SSP245AMIP_CTL_SST0051_REST0101` |
| Files | 51,100 = 35 years (2015–2049) × 1460, `{year}_{idx:04d}.h5` |
| Per file | 163 keys = 8 upper-air × 18 levels (144) + 18 flat + `time`; every field `float32 (180, 360)` |
| Selected | 108 of 162 channels — `CLDLIQ`/`CLDICE`/`CLOUD` excluded (3 × 18 = 54), per the science owner |
| Level keys | hPa floats at full precision, e.g. `T_998.4964394917621` |
