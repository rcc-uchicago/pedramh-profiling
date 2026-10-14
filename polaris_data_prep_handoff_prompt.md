# E3SM data prep on Polaris — decisions needed, and the measurements behind them

**Scope: the S2S family — `PanguWeather` (the focus), `makani_sfno`, `physicsnemo_sfno`.
SI is deliberately out of scope here.**

This exists because `physicsnemo_sfno/polaris/polaris_zarr_e3sm_full.pbs` is about to convert
**51,100 samples → ~1 TB over ~10 h**, and the smoke that was supposed to justify it *cannot*.
Everything below was **measured** on the archive, not read off a config or a comment. Where a
claim is inferred rather than measured, it says so.

Read **CLAUDE.md** for house rules, **DESIGN.md** §1 (the science is frozen) and §2c (the
forks share code by copy), **CHANGELOG.md** for status, and `polaris_pbs_notes.md` §4/§8 for
the staging and converter recipes.

---

## 0. What needs a decision

**Five open decisions live in `polaris_data_prep_decisions.md`** — they change what the models
learn, so DESIGN §1 puts them out of my hands. In short: whether PhysicsNeMo should forecast
sea-surface temperature and sea ice (it does; PanguWeather and makani prescribe them, and this
is an Atmospheric Model Intercomparison Project run where they are prescribed by definition);
whether PanguWeather's 270 fill and PhysicsNeMo's zero fill are intended; whether PhysicsNeMo
should train on nineteen constant cloud channels; and whether the three models reading 108 /
162 / 59 channels is deliberate. The measurements behind each are below.

## 1. The dataset (measured)

`/eagle/lighthouse-uchicago/members/jesswan/AI4SRM/data/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/h5/plev_data/`

**51,100 files**, `{year}_{idx:04d}.h5`, years **2015–2049**, exactly **1460/year** (noleap:
365 d × 4/day, 6-hourly — 2016/2020 included, no leap years anywhere). Perfectly regular:

```
2015_0000.h5
└── input/
    ├── {CLDICE,CLDLIQ,CLOUD,RELHUM,T,U,V,Z3}_<level>   8 vars × 18 levels = 144
    ├── FSNT FSNTOA ICE PCT_GLACIER PCT_NATVEG PFTDATA_MASK PRECT PS PSL
    │   RHREFHT SOILWATER_10CM SST TMQ TOPO TREFHT TSOI_10CM U10 sol_in   = 18
    └── time                                            b'2015-01-01 00:00:00'
```

**Every field is `(180, 360) float32`.** One timestep per file. No ragged shapes, no nesting.
Grid confirmed against `boundary_data/TOPO.nc`: **lat −89.5…89.5 ascending** (row 0 = Antarctic),
**lon 0.5…359.5** (cell centres).

> ### ⚠ The archive's `time` is broken: every file is stamped **2015**
> Measured: `2016_0000.h5` → `'2015-01-01 00:00:00'`; `2030_0000.h5` → `'2015-01-01 00:00:00'`;
> `2049_1459.h5` → `'2015-12-31 18:00:00'`. Month/day/hour **do** track the sample index
> correctly (`2020_0800` → day 201 ✓, checked across 14 years). **Only the year is frozen.**
> Any pipeline that trusts the in-file label gets a time axis that resets to 0 every year.
> This is a defect in the source archive, not in our code, and worth raising upstream.

**Units — measured, not assumed** (a variable name is not evidence):

| field | measured | implication |
|---|---|---|
| `SST` | **°C**, `[−1.80, 32.21]`, NaN over 37.4% (land) | min is *exactly* −1.8 = freezing seawater |
| `TSOI_10CM` | **Kelvin**, land mean 268 K, NaN over 61.4% (ocean) | a 0.0 fill is 0 K |
| `TREFHT` | Kelvin, `[224, 310]`, no NaN | |
| `ICE` | fraction `[0, 1]` | 0 is genuinely "none" |
| `PRECT` | m/s (~2e-8) | |
| `sol_in` | W/m², `[0, 1406]`, time-varying | |

**The store mixes units**: SST in °C, every other temperature in K.

---

## 2. What each model actually reads

Built from the configs directly (`E3SM_SFNO_H5_POLARIS.yaml`, `e3sm_h5_to_seqzarr.py`,
`convert_e3sm_to_makani.py`), not by hand.

```
variable          lev   Pangu   nemo  makani     NaN-fill  Pangu / nemo / makani
──────────────────────────────────────────────────────────────────────────────────
CLDICE             18       -   prog       -
CLDLIQ             18       -   prog       -
CLOUD              18       -   prog       -
RELHUM/T/U/V/Z3    18    prog   prog    prog
FSNT / FSNTOA       1    diag   prog       -
ICE                 1   force   prog   force        0 /  0   / 0
PCT_GLACIER         1   force  force   force        0 /  0   / 0
PCT_NATVEG          1   force  force       -        0 /  0   / –
PFTDATA_MASK        1   force  force   force        0 /  0   / 0
PRECT               1    diag   prog    diag
PS / TREFHT         1    prog   prog    prog
PSL/RHREFHT/TMQ/U10 1    prog   prog       -
SOILWATER_10CM      1    prog   prog       -        0 /  0   / –
SST                 1   force   prog   force      270 / −1.8 / −1.8   ← see the decisions file
TOPO                1   force  force   force        0 /  0   / 0
TSOI_10CM           1    prog   prog       -      270 /  0   / –      ← see the decisions file
sol_in              1   force  force   force
──────────────────────────────────────────────────────────────────────────────────
  Pangu  reads 108/162   not read: CLDICE, CLDLIQ, CLOUD          (54 channels)
  nemo   reads 162/162   not read: nothing
  makani reads  59/162   not read: clouds, FSNT/FSNTOA, PSL, RHREFHT, TMQ, U10,
                                   PCT_NATVEG, SOILWATER_10CM, TSOI_10CM,
                                   + only 10 of 18 levels
```

`prog` = prognostic (fed in, forecast forward, **scored by the loss**) ·
`diag` = diagnostic · `force` = prescribed input (fed in, never forecast, never scored) ·
`-` = not read.

**PhysicsNeMo's split is only two-way.** `train.py` does
`unroll(model, predicted, unpredicted, ...)` then
`loss = batch_normalized_mse(net_predicted, predicted[:, nr_input_steps:])` — so `predicted`
is everything the model forecasts *and* is scored on; `unpredicted` is everything it is simply
told. Pangu has five roles (upper-air / surface / diagnostic / land / boundary×2); makani has
three (state / diag / forcing). The two-way split is why FSNT, FSNTOA and PRECT — diagnostics
everywhere else — end up prognostic under PhysicsNeMo.

---

## 3. The paradigm for missing data: there isn't one

E3SM masks ocean-only fields over land and land-only fields over ocean with NaN. You cannot
train on NaN and none of the datapipes mask, so every pipeline fills — and **each invented its
own constants, independently, without reconciling**:

- **`SST`: three pipelines, two different fills** (270 / −1.8 / −1.8), for a field measured to
  be °C in `[−1.80, 32.21]`.
- **`TSOI_10CM`: two pipelines, two different fills** (270 / 0), for a field measured to be K
  around 268.
- Everything else agrees on `0` — but that is **agreement by luck**: those are non-negative
  quantities (`ICE`, `TOPO`, `PCT_*`, `PFTDATA_MASK`) where 0 genuinely means "none".

**Nobody uses a masked loss**, which is the alternative. Every pipeline already feeds
`PFTDATA_MASK` — the land-sea mask — to the model, so the network is *told* where the mask is.
That means the fill value's only real job is: **stay in-distribution so the normalizer isn't
dominated by the mask**. Judged against that:

| fill | verdict |
|---|---|
| `SST → −1.8` (nemo, makani) | ✅ the physical minimum, in-distribution |
| `SST → 270` (Pangu) | ❌ ~8× outside the data. **Measured**: the SST channel becomes ~88% land-mask — real signal **0.27σ** against a **2.06σ** mask step |
| `TSOI → 270` (Pangu) | ✅ Kelvin, in-distribution |
| `TSOI → 0` (nemo) | ❌ 0 K over 61% of the globe |

Each pipeline gets one right and one wrong, in opposite directions.

> **A trap worth recording.** `normalize_mean.npz` says `SST` mean = **109.963**. That looks
> like Kelvin data and `CHANGELOG` recorded it as a "unit mismatch, inferred not measured".
> **It is not Kelvin.** The arithmetic closes exactly on a **270 land-fill of °C data**:
> `0.374 × 270 + 0.626 × 14.70 = 110.06` vs the npz's 109.963. So the npz is self-consistent
> **with Pangu's 270 fill** — it is not wrong, it encodes the 270-fill choice. Anyone "fixing" the npz to
> Kelvin would be fixing a bug that doesn't exist.

---

## 4. Converter defects — what was fixed, and what is still open

`physicsnemo_sfno/polaris/e3sm_h5_to_seqzarr.py`. **Fixed 2026-07-15** (see the PR):

| defect | why the smoke never saw it |
|---|---|
| `--max-samples` defaulted to **64**, so the "full" run converted 64 of 51,100 and printed `CONVERT_OK` | the smoke *asks* for 64 explicitly |
| the time axis trusted the archive's **frozen 2015 year** | the smoke is 2015-only, where the frozen year is *correct* |
| a preempted run leaves a full-shape store whose unwritten samples read back as **silent zero slabs** (zarr pre-allocates with `fill_value: 0.0`) | a completed smoke never hits it |
| `longitude` was `0..359`, not the true cell centres `0.5..359.5` — and `train.py:447-451` reads it straight into the **inference model package** | nothing checked it |
| four `means_/stds_` arrays copied from the npz — **dead** (nothing reads them: the datapipe asks only for `time/predicted/unpredicted`, `train.py` normalizes with `BatchNorm2d`) **and wrong** (npz SST assumes a 270 fill; the store fills −1.8) | nothing read them, so nothing complained |

**That last one is the cautionary tale**: the dead metadata fooled *two independent auditors*
(me and a cold review agent) into concluding the model couldn't see SST. It can — the arrays
are never read. Dead-and-wrong metadata is worse than none; they are deleted, not corrected.

### Still open (found by adversarial review, NOT yet fixed)

1. **A non-contiguous `--years` store passes every gate.** `TRAIN_YEARS="2015 2020 2030"` is a
   documented env knob; the resulting store gets `sampling_mode=contiguous`, a valid
   `conversion_complete`, and **8766-hour seams** the trainer silently learns as t→t+1yr.
   *Fix:* require `unique(diff(time)) == [6]` for contiguous mode before writing the sentinel.
2. **`nchunks_initialized` is defeatable.** zarr 2.18.7 counts chunk keys by *prefix* regex, so
   a `.partial` file left by a kill mid-chunk **counts as written** — measured 6/6 while a
   sample was an all-zero slab. *Fix:* compare the exact expected key set.
3. **Preemption during the *val* conversion destroys the completed *train* store.**
   `polaris_zarr_e3sm_full.pbs` converts train (~9 h) then val in a 12 h `preemptable` window
   with `#PBS -r y`; the requeue re-runs from the top and `zarr.open_group(mode="w")` wipes it.
   Repeated preemption thrashes forever. *Fix:* per-year stores, or skip completed ones.
4. **`--random-sample` ignores `--start-sample` but records it** — lying provenance in the attr
   that exists to record provenance.
5. **`-v SEQZARR_DATA=…` does not work, so the converter changes are UNGATED by the smoke.**
   `polaris_env.sh:155` does `export SEQZARR_DATA="$(_pick SEQZARR_DATA e3sm_seqzarr)"`
   unconditionally, and `_pick` **never reads its first argument** — so a pre-set value cannot
   win. Job **7257791** was submitted with `-v SEQZARR_DATA=…_fresh` specifically to force
   `polaris_sfno_smoke.pbs` to rebuild its store with the new converter; the log shows it used
   the OLD cached store at `…/e3sm_seqzarr` and passed (rc=0, loss 1.049). **It exercised none
   of the changes.** Combined with the smoke's `[ ! -d ]` skip-if-exists, there is currently
   **no way to make the smoke rebuild** short of moving the store aside.
   *Fix:* honour a pre-set value in `_pick` (`echo "${!1:-…}"`), or have the smoke accept an
   explicit override. Until then the converter changes are proven only by
   `polaris_verify_data_prep.pbs` (which does exercise them, exhaustively) — **not** by a
   training run.

---

## 5. How to verify before the ~1 TB run

**The smoke is not the gate.** `polaris_sfno_smoke.pbs` builds 64 train + 16 val samples from
**2015 only** — 16 days of January — and *all three* of the worst defects above were invisible
at exactly that scale. It also `[ ! -d ]`-skips conversion when the store exists, so a re-run
reuses the old store and certifies nothing.

Use instead:

```bash
cd physicsnemo_sfno
qsub polaris/polaris_verify_data_prep.pbs                  # 40 random samples spanning 2015-2049
qsub -v N_SAMPLES=200 polaris/polaris_verify_data_prep.pbs # deeper
```

It converts a **random draw across every year** (`--random-sample`, seeded) and verifies it
**exhaustively** against the source — every sample × every channel, bitwise, plus NaN-fill
placement, the channel map, and the time axis. PASS = **`SEQZARR_VERIFIED (EXHAUSTIVE: …)`**.

**Not `CONVERT_OK`** — that is the converter's own 1-channel-of-1-sample gate, and trusting it
is how we got here.

⚠ A random store is a **correctness fixture, not training data** (its samples aren't
consecutive, so SFNO's t→t+6h target is meaningless). It records `sampling_mode=random` and
`polaris_sfno_full.pbs` refuses it.

Only then:

```bash
qsub -v CONFIRM_FULL=1 polaris/polaris_zarr_e3sm_full.pbs   # ~48,180 samples, ~1 TB, ~10 h
```

The `CONFIRM_FULL` interlock exists because fixing the `--max-samples` default **armed** this
script: it used to write a harmless 64 samples. **Check `myquota` first** — the project shares
a 15 TB eagle quota and was at **15.18 TB** on 2026-07-15. A conversion that dies on quota at
hour 8 leaves exactly the zero-filled store the sentinel exists to catch.

---

## 6. What a green verification still cannot prove

- **That the fill *values* are right.** The verifier imports `NAN_FILL` from the converter, so
  it checks the store against the converter's *own intent*. Change SST's fill to 0.0 in the
  dict and it still passes — by construction. **the fill values are settled by review or not at all**,
  which is why the converter now records `nan_fill` into the store's attrs.
- **That the channel *roles* are right.** Whether sea-surface temperature should be forecast
  or prescribed is a semantic choice; no bitwise check sees it.
- **That a green smoke generalises.** Absence is only evidence where execution happened.

## 7. Suggested order

1. **Answer the sea-surface-temperature/sea-ice role question** — it's ours, it's one line
   (`UNPREDICTED += ["SST", "ICE"]`), and it aligns all three pipelines with how the source
   experiment was actually run.
2. **Take the fill and cloud-channel questions to jesswan** with the measurements in §1/§3.
3. **Close the four open defects** in §4 — 1 and 3 are the ones that silently poison a 1 TB run.
4. **Run `polaris_verify_data_prep.pbs`** and require `SEQZARR_VERIFIED`.
5. **Then** `CONFIRM_FULL=1`.

---

## 8. CONTINUE HERE — the analysis is one-third done

Only **one** of the three data-prep scripts has been audited. Do not read this document as
"the data prep is understood": it means "the PhysicsNeMo converter is understood, and it had
seven defects".

### 8a. Audited ✅ — `physicsnemo_sfno/polaris/e3sm_h5_to_seqzarr.py` (172 → ~290 lines)

Cold review + adversarial review + exhaustive bitwise verification against source.
**7 defects found**, 5 fixed, 4 open (§4). Green:
`SEQZARR_VERIFIED (EXHAUSTIVE: 40 samples × 162 channels = 6480 channel-samples, bitwise)`,
job **7257786**, on a fixture spanning **24 distinct years**.

### 8b. Audited ✅ — `PanguWeather/v2.0/polaris_prepare_e3sm_stats.py` (64 lines)

Read in full. **Genuinely just transformations, and metadata-only**: renames the level dim
`Z` → `Z_2` (values untouched) and re-encodes `climatology.nc` CDF-5 → NETCDF4. No value is
modified; both changes exist to satisfy a *reader*, not the science. **No defects found.**
This is the only prep PanguWeather's full training needs — it reads `h5/plev_data` directly.

### 8c. NOT AUDITED ⬜ — `makani_sfno/polaris/convert_e3sm_to_makani.py` (367 lines)

**The biggest converter, and the least examined.** Everything below is read off its own
comments — none of it is verified:

- **flips latitude** ascending → descending (`:134`) *and* writes a descending lat axis. The
  data and the axis must flip together; PhysicsNeMo does neither. A half-applied flip is
  silent and catastrophic.
- **truncates to 10 of 18 levels** and to 52 state + 1 diag + 6 forcing channels, to fit the
  PlaSim contract. Which 8 levels are dropped, and is that deliberate for E3SM or inherited
  from PlaSim?
- **computes its own stats** from the packed split (float64 sum/sumsq, `1e-12` floor) rather
  than copying the npz — the *right* call, and worth confirming it is done correctly, since
  unlike PhysicsNeMo's these stats are **live**.
- **renames channels** (`PFTDATA_MASK`→`lsm`, `TOPO`→`sg`, `PCT_GLACIER`→`z0`, `SST`→`sst`,
  `sol_in`→`rsdt`, `ICE`→`sic`). `z0` is a roughness slot being fed a glacier percentage —
  check whether the trainer treats slots semantically.
- **synthesizes timestamps from file order** (`offset + arange*21600`), sidestepping the
  frozen-year bug entirely. Confirm it lands on the same axis the PhysicsNeMo fix now produces.
- its pack is **sample-limited to 400/year** (`--max-samples-per-year`), i.e. the makani data
  on disk is *also* smoke-scoped — 18 GB of a ~770 GB full pack.

**Do to it exactly what was done to the PhysicsNeMo converter**, in this order, because the
order is what worked:

1. **Read it against the archive, not its comments.** Measure units and ranges yourself; a
   variable name is not evidence. (`SST` looked like Kelvin to two auditors and is °C.)
2. **Ask of every check: what would it do if the thing were broken?** PhysicsNeMo's
   `CONVERT_OK` verified 1 channel of 1 sample — 0.01% — and was trusted for weeks.
3. **Write an exhaustive verifier** (`verify_seqzarr.py` is the template) and **mutation-test
   it**: corrupt a copy under `/tmp` and confirm each check fails. A test not seen to fail is
   not a test — mine passed nothing on a *correct* store when first written.
4. **Never verify a store against the code that built it.** Re-deriving the file list from the
   same function is circular; that is why the converter now records `source_files` in attrs.
5. **Spawn a cold Fable 5 agent** (CLAUDE.md model policy) with no conclusions in the prompt.
   Two of the three worst findings here came from cold agents, and one cold agent independently
   reproduced my own wrong conclusion — which is itself the evidence that the trap was real.

### 8c-bis. FOUND: how the .h5 archive itself was built — and it is not in this repo

Nothing in this repository documents the step that made `h5/plev_data` out of the raw E3SM
model output. Everything we have audited so far is *downstream* of it. The code that built it
lives in jesswan's own checkout, outside this repo and outside its history:

```
/eagle/lighthouse-uchicago/members/jesswan/PanguWeather/data_utils/
    netcdf_to_h5.py  netcdf_to_h5_2.py  netcdf_to_h5_better.py
    get_stats.py            <- almost certainly what produced normalize_{mean,std}.npz
    combine_nc_files.py     get_data_pl_short_length.py  get_data_sfc_short_length.py
    parallel_copy.py        submit_convert.sh  submit_convert_2.sh
```

`get_stats.py` opens "### From FourCastNet repo" — so the statistics path is adapted from
FourCastNet, which matters because FourCastNet's conventions are not E3SM's.

**Read these before answering the fill questions.** They are the ground truth for three things
this session could only infer:
- **What the 270 sea-surface-temperature fill actually is.** We proved arithmetically that
  `normalize_mean.npz` was computed on 270-filled °C data. `get_stats.py` should say so
  outright — and whether 270 was deliberate or a Kelvin default that leaked into a °C field.
- **Whether the 19 constant cloud channels are zero in the source model output**, or were lost
  in the netCDF → HDF5 step. `Gridded_EAM_Lev_Subset/` (1.9 TB) is the input that would settle
  it, and it has never been opened.
- **Why `input/time` is frozen at 2015 in every file.** The bug is introduced here, not by us.

Note the three `netcdf_to_h5*.py` variants (`_2`, `_better`) — the archive was built by *one*
of them and nothing records which. Ask jesswan rather than guess.

### 8d. Cross-cutting questions still unanswered

- **Do the three pipelines agree on the physics they claim to share?** Measured so far: they
  do **not** agree on SST's role (2 prescribe, 1 forecasts), on SST's fill (270 / −1.8 / −1.8),
  on TSOI's fill (270 / 0), or on which channels to read (108 / 162 / 59). Nobody reconciled
  them. Is any of that intended? (See `polaris_data_prep_decisions.md`.)
- **The archive's frozen `time` year is an upstream defect.** PhysicsNeMo now reconstructs from
  the filename and makani sidesteps it; **Pangu reads `h5/plev_data` directly** — does *its*
  loader use the in-file date? Not checked. If it does, the same bug is live there.
- **19 dead cloud channels** (`CLDICE`/`CLDLIQ`/`CLOUD` at 4.7–200 hPa, identically zero or
  float32 subnormals ~1e-26 across the whole archive). Pangu and makani exclude clouds; only
  PhysicsNeMo trains on them. Are they zero in the *source model output*, or was something
  lost in the E3SM → h5 step? That question has not been asked of `Gridded_EAM_Lev_Subset/`.
- **`Gridded_EAM_Lev_Subset/` (1.9 TB) and `Gridded_ELM/` (210 GB) are read by nothing** in
  this repo. They are the raw EAM/ELM provenance the h5 was derived from. Confirmed by
  enumerating every AI4SRM path literal: only `h5/plev_data` and `bias/` are referenced.
