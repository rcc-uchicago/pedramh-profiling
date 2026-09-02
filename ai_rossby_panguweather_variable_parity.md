# Variable parity — ai-rossby `PanguPlasimLegacy` vs jesswan's trained PanguWeather E3SM run

**Status: ✅ PASS — full variable parity. Coding is unblocked.**

This is the Step-0 gate for the ai-rossby PanguPlasim bring-up
(the ai-rossby/Pangu bring-up handoff, deleted 2026-09-02 once executed — results in
CHANGELOG 2026-08-04/05). It proves the variable set we are about to
train is **identical** to the one jesswan's PanguWeather E3SM run actually trained on, so
that any difference in results is attributable to the architecture, not to the data.

**Companion:** `ai_rossby_e3sm_zarr_schema.md` — *how* these variables are stored
(measured geometry, chunking, and the production year split).

Every table below is **machine-generated evidence**, not a reading. The contract lives in
one place — `ai_rossby_variable_contract.py::PLANNED` — and that module both (a) asserts
`PLANNED` against jesswan's YAML, and (b) later asserts the produced configs, converter and
Zarr store attrs against `PLANNED`. Reproduce §4 with:

```bash
python3.12 ai_rossby_variable_contract.py --check-ground-truth
```

(`python3.12` because the Polaris login node's bare `python3` is 3.6; the module is
stdlib-only by design so it runs before any venv exists.)

---

## 1. The ground truth — which run this is measured against

**Config: `PanguWeather/v2.0/config/E3SM_SFNO_H5_STAMPEDE_jsw.yaml`**, run by jesswan on
**Stampede3** (TACC), May 2026.

Evidence, in the repo:

| Evidence | Where | What it shows |
|---|---|---|
| `INFO - Configuration file: /work2/11095/jwan4/PanguWeather/v2.0/config/E3SM_SFNO_H5_STAMPEDE_jsw.yaml` | `PanguWeather/v2.0/e3sm_train_3167436.e` | The trainer logged **this exact config file** at startup, from jesswan's own Stampede3 `$WORK` (`jwan4`). Also present in 7 of the 10 `e3sm_train_*.e` logs (one under `/work/`, the rest `/work2/`). |
| `Starting Training Loop... / Starting epoch 1/100 / Expected total batches: 2281` then `Epoch [1/100], Year 2015, Loss: 0.8920 → 0.0635` over 7 steps | same file | Training **actually ran** and the loss descended — not a config that merely exists. |
| `slurmstepd: error: *** JOB 3167436 ON c561-010 CANCELLED AT 2026-05-30T04:53:08 DUE TO TIME LIMIT ***`, and `host: c562-005.stampede3.tacc.utexas.edu` in `e3sm_train_3166999.e` | same / sibling | Confirms the cluster is **Stampede3**, matching the `_STAMPEDE_jsw` config name. |
| `4× NVIDIA H100`, `WORLD_SIZE= 4` | `e3sm_train_3167436.o` | 4-GPU DDP run. |
| `Start date: 2045-01-01_00:00:00 / End date: 2050-01-01_00:00:00` | `e3sm_train_3167436.o` | Matches the config's `val_year_start: 2045 / val_year_end: 2050`. |

**One honest wrinkle, and why it does not matter.** The log's progress bar reads
`Year 2015`, while the config on disk today reads `train_year_start: 2020 #2015` — the
trailing comment is the fingerprint of an edit made *after* that job. So the run used the
same file with `train_year_start: 2015`. That is a **train/val split** difference, not a
variable difference, and this document is about variables.

**And the stronger result: the choice of config is moot.** All four PanguWeather E3SM
configs in the repo carry a **byte-equivalent variable contract** — same names, same order,
same levels, same fills. Verified by running the same check against each:

| Config | Result |
|---|---|
| `E3SM_SFNO_H5_STAMPEDE_jsw.yaml` (**the trained one**) | `VARIABLE_PARITY_OK 10/10` |
| `E3SM_SFNO_H5_DERECHO_jsw.yaml` (jesswan's Derecho config, pinned by `HPC_scripts/derecho_training_bing.sh`) | `VARIABLE_PARITY_OK 10/10` |
| `E3SM_SFNO_H5_POLARIS.yaml` (our Polaris smoke port) | `VARIABLE_PARITY_OK 10/10` |
| `E3SM_SFNO_H5_POLARIS_ALLDATA.yaml` (our 35-year port) | `VARIABLE_PARITY_OK 10/10` |

So whichever of jesswan's runs is taken as the reference, the variable contract is the same
one. Parity is not contingent on the identification in this section.

---

## 2. The ground-truth variable contract

Extracted from `E3SM_SFNO_H5_STAMPEDE_jsw.yaml:48-55, 66-73, 92-108` (`:52-59, 69-76` in the
`_POLARIS` copy). **108 fields.**

| Group | Fields | Count |
|---|---|---:|
| `upper_air_variables` | `T`, `U`, `V`, `Z3`, `RELHUM` — × 18 levels | 90 |
| `surface_variables` | `TREFHT`, `U10`, `RHREFHT`, `PS`, `PSL`, `TMQ` | 6 |
| `diagnostic_variables` | `FSNTOA`, `FSNT`, `PRECT` | 3 |
| `land_variables` | `SOILWATER_10CM`, `TSOI_10CM` | 2 |
| `ocean_variables` | *(empty)* | 0 |
| `constant_boundary_variables` | `PCT_GLACIER`, `PFTDATA_MASK`, `PCT_NATVEG`, `TOPO` | 4 |
| `varying_boundary_variables` | `SST`, `ICE`, `sol_in` | 3 |
| | **total** | **108** |

`CLDLIQ`, `CLDICE`, `CLOUD` are **excluded** — commented out on the `upper_air_variables`
line in every config. That is 3 × 18 = 54 of the archive's 162 channels dropped, which is
how 162 → 108. Confirmed as the science owner's decision on 2026-07-16
(`polaris_e3sm_variable_reference.md`, "Variable table" note; R5).

### Levels — 18, and the subtlety worth stating

The config carries **two** level lists and only one of them is real:

* `levels: [5, 10, 20, ..., 1000]` — nominal hPa **labels**.
* `sigma_levels: [4.714998332947841, ..., 998.4964394917621]` — the values actually
  embedded in the E3SM HDF5 keys (e.g. `T_998.4964394917621`).

Because the config sets `use_sigma_levels: True`, the **`sigma_levels` list is the level
identity that reaches the data loader**. The parity check compares against `sigma_levels`
for exactly this reason; comparing the nominal `levels` would have been a false PASS.

These are E3SM **hybrid** levels, i.e. terrain-following, not isobaric — despite living in a
directory called `plev_data` and being labelled in hPa
(`polaris_e3sm_variable_reference.md` R6). Both sides inherit this identically, so it is not
a parity issue; it is an evaluation hazard already on the books.

### Fills (`mask_fill`)

```yaml
mask_fill: {'SOILWATER_10CM': 0., 'TSOI_10CM': 270., 'PCT_GLACIER': 0., 'PFTDATA_MASK': 0.,
            'PCT_NATVEG': 0., 'TOPO': 0., 'SST': 270., 'ICE': 0.}
```

Two fields at **270**, the rest at **0**.

---

## 3. The planned ai-rossby contract

Source of truth: `ai_rossby_variable_contract.py::PLANNED`. It is what
`conf/model/pangu_plasim_e3sm.yaml`, `conf/dataset/e3sm.yaml` and the converter's
`PANGU_E3SM_CHANNELS` will each be written to, and what the preflight will assert them
against (`--check-artifacts`). Dump it with `--dump`.

It is transcribed from §2 field-for-field, **including order within every group**, and the
fills map to:

```yaml
nan_fill_values: {SST: 270.0, TSOI_10CM: 270.0}
nan_fill_default: 0.0
```

which expands to exactly the eight-entry `mask_fill` above (the four constant-boundary
fields, `ICE` and `SOILWATER_10CM` all take the 0.0 default).

### ⚠ These are **deltas from ai-rossby's shipped defaults**, not from parity

ai-rossby today ships an E3SM channel set that is *not* the PanguWeather one. Writing the
contract means changing all of the following — this is the work list for step 3(c), and each
line is a place a silent mismatch could have hidden:

| ai-rossby default (`tools/data/e3sm/pangu_h5_to_zarr.py`) | Contract | Delta |
|---|---|---|
| `surface_variables: [TREFHT, U10, PSL]` | `[TREFHT, U10, RHREFHT, PS, PSL, TMQ]` + land folded last | **+3 vars, +2 land, order changed** |
| `diagnostic_variables: [PRECT]` | `[FSNTOA, FSNT, PRECT]` | **+2 vars** |
| `pressure_upper_air_variables: [T, U, V, RELHUM, Z3]` | `[T, U, V, Z3, RELHUM]` | **order changed** (`Z3`/`RELHUM` swapped) |
| `constant_boundary_variables: [TOPO, PCT_GLACIER, PCT_NATVEG, PFTDATA_MASK]` | `[PCT_GLACIER, PFTDATA_MASK, PCT_NATVEG, TOPO]` | **order changed** |
| `varying_boundary_variables: [SST, ICE, sol_in]` | same | — |
| `pressure_levels` (18 hybrid hPa) | same | — ✅ **bit-identical**, verified: `max|diff| = 0.0` |
| `conf/dataset/e3sm.yaml: nan_fill_values: {SST: 270.0}` | `{SST: 270.0, TSOI_10CM: 270.0}` | **TSOI fill missing** → would have silently filled 0 K |

The two order changes are the dangerous ones. Nothing in ai-rossby cross-checks them:
`ClimateZarrDataset._build_sample` stacks `surface_in` in **store-attrs order**
(`physicsnemo/experimental/datapipes/climate/dataset.py:533`) while the NaN fills and the
loss are built from the **model-config lists** (`examples/weather/ai_rossby/train.py:636`,
`:739`). A `torch.cat` of the right shape in the wrong order raises nothing. Hence the
`--check-artifacts` preflight.

### Why land folds into `surface_variables` in the store

The ai-rossby store schema has no `land_variables` / `ocean_variables` groups —
`ClimateZarrStoreLayout` defines only surface / constant-boundary / varying-boundary /
diagnostic / pressure-upper-air / sigma-upper-air
(`datapipes/climate/dataset.py:57-75`). `PanguPlasimLegacy` meanwhile expects its surface
tensor to carry `num_surface_vars + num_land_vars + num_ocean_vars` channels **sliced in
that order** (`models/pangu_plasim/pangu_plasim_legacy.py:567, 673-678`).

So the store's `surface_variables` attr is written as
`[TREFHT, U10, RHREFHT, PS, PSL, TMQ, SOILWATER_10CM, TSOI_10CM]` — surface, then land, then
ocean (empty). The model config keeps `surface_variables` and `land_variables` separate, so
the architecture keeps its distinct land head. **No variable is added, removed or
re-roled** — this is a container-layout mechanic, not a science change.

---

## 4. The assertion — field by field

`python3.12 ai_rossby_variable_contract.py --check-ground-truth` →

```
=== PLANNED vs ground truth (E3SM_SFNO_H5_STAMPEDE_jsw.yaml) ===
[PASS] upper_air_variables (names + order)
[PASS] surface_variables (names + order)
[PASS] diagnostic_variables (names + order)
[PASS] land_variables (names + order)
[PASS] ocean_variables (names + order)
[PASS] constant_boundary_variables (names + order)
[PASS] varying_boundary_variables (names + order)
[PASS] levels (values + order)
[PASS] mask fills
[PASS] total field count
VARIABLE_PARITY_OK 10/10 checks passed
```

Expanded, per group:

### 4.1 `upper_air_variables` — PASS

| # | PanguWeather | ai-rossby (planned) | Names | Order |
|---:|---|---|---|---|
| 0 | `T` | `T` | ✅ | ✅ |
| 1 | `U` | `U` | ✅ | ✅ |
| 2 | `V` | `V` | ✅ | ✅ |
| 3 | `Z3` | `Z3` | ✅ | ✅ |
| 4 | `RELHUM` | `RELHUM` | ✅ | ✅ |

Excluded on both sides: `CLDLIQ`, `CLDICE`, `CLOUD`. **PASS.**

### 4.2 Level list — PASS

18 vs 18, compared as floats at full precision and again after the float32 cast the
converter applies to the `pressure_level` coord:

```
PanguWeather sigma_levels n = 18
ai-rossby     pressure_levels n = 18
EXACT EQUAL (value + order): True
max |diff| = 0.0
float32-rounded equal: True
```

| # | hPa (both sides) | | # | hPa (both sides) |
|---:|---|---|---:|---|
| 0 | 4.714998332947841 | | 9 | 256.72368590525895 |
| 1 | 10.655023096474308 | | 10 | 302.21364012188303 |
| 2 | 19.235455601758737 | | 11 | 385.999023919911 |
| 3 | 28.79458853709195 | | 12 | 492.46857402252755 |
| 4 | 50.11779996521295 | | 13 | 608.6437744215842 |
| 5 | 69.59908688413749 | | 14 | 713.7046383204334 |
| 6 | 96.46377266572703 | | 15 | 849.6612491105952 |
| 7 | 145.04282239200347 | | 16 | 925.5197481473349 |
| 8 | 200.99889546355382 | | 17 | 998.4964394917621 |

**PASS.** Upper-air channel count: 5 × 18 = **90** on both sides.

### 4.3 `surface_variables` — PASS

| # | PanguWeather | ai-rossby (planned) | Role on both sides |
|---:|---|---|---|
| 0 | `TREFHT` | `TREFHT` | prognostic |
| 1 | `U10` | `U10` | prognostic |
| 2 | `RHREFHT` | `RHREFHT` | prognostic |
| 3 | `PS` | `PS` | prognostic |
| 4 | `PSL` | `PSL` | prognostic |
| 5 | `TMQ` | `TMQ` | prognostic |

**PASS** (6/6, names + order + role).

### 4.4 `diagnostic_variables` — PASS

| # | PanguWeather | ai-rossby (planned) |
|---:|---|---|
| 0 | `FSNTOA` | `FSNTOA` |
| 1 | `FSNT` | `FSNT` |
| 2 | `PRECT` | `PRECT` |

**PASS** (3/3). Both sides keep these in the *diagnostic* role, not promoted to prognostic —
unlike the PhysicsNeMo SFNO path, which forecasts them
(`polaris_e3sm_variable_reference.md`, "Roles and NaN fills").

### 4.5 `land_variables` / `ocean_variables` — PASS

| # | PanguWeather | ai-rossby (planned) |
|---:|---|---|
| land 0 | `SOILWATER_10CM` | `SOILWATER_10CM` |
| land 1 | `TSOI_10CM` | `TSOI_10CM` |
| ocean | *(empty)* | *(empty)* |

**PASS** (2/2 land, 0/0 ocean). `ocean_variables` being empty is what makes `SST` a
*prescribed boundary* rather than a forecast field on both sides — the single most
consequential role assignment in the whole contract, and it matches.

### 4.6 `constant_boundary_variables` — PASS

| # | PanguWeather | ai-rossby (planned) |
|---:|---|---|
| 0 | `PCT_GLACIER` | `PCT_GLACIER` |
| 1 | `PFTDATA_MASK` | `PFTDATA_MASK` |
| 2 | `PCT_NATVEG` | `PCT_NATVEG` |
| 3 | `TOPO` | `TOPO` |

**PASS** (4/4). Note ai-rossby's shipped default had these in a different order — see §3.

### 4.7 `varying_boundary_variables` — PASS

| # | PanguWeather | ai-rossby (planned) |
|---:|---|---|
| 0 | `SST` | `SST` |
| 1 | `ICE` | `ICE` |
| 2 | `sol_in` | `sol_in` |

**PASS** (3/3). `SST` and `ICE` are **prescribed** (input-only, never scored) on both sides.

### 4.8 Fills — PASS

| Variable | PanguWeather `mask_fill` | ai-rossby (planned) | Units | In-distribution? |
|---|---:|---:|---|---|
| `SOILWATER_10CM` | 0. | 0.0 (default) | kg/m² | ✅ physical minimum |
| `TSOI_10CM` | **270.** | **270.0** (explicit) | **K** | ✅ 0.02σ from the valid mean (land mean 268 K) |
| `PCT_GLACIER` | 0. | 0.0 (default) | % | ✅ |
| `PFTDATA_MASK` | 0. | 0.0 (default) | unitless | ✅ |
| `PCT_NATVEG` | 0. | 0.0 (default) | % | ✅ |
| `TOPO` | 0. | 0.0 (default) | m | ✅ sea level |
| `SST` | **270.** | **270.0** (explicit) | **degC** ⚠ | ❌ **8× outside the field's range** |
| `ICE` | 0. | 0.0 (default) | fraction | ✅ |

**PASS** — the fills are identical, which is what parity requires. Two notes on the values
themselves, neither of which is a parity failure:

* **`TSOI_10CM` = 270 K is correct**, and confirmed with the user 2026-08-04: soil
  temperature — like every temperature field — is in **Kelvin**, so 270 is a near mean-fill
  (`polaris_e3sm_variable_reference.md` R3: "the fill value itself is good — 270 sits 0.02σ
  from the valid mean"). This is also what our own `aa43824a` changed the *PhysicsNeMo* path
  to, for the same reason.
* **`SST` = 270 applied to a degC field is a known inherited defect** (R4). `SST` measures
  [−1.80, 32.92] degC; 270 is ~8× the maximum, leaving the channel ~99.5% static land-mask
  variance. It is not our choice — it is upstream's, and it came from a mechanical rename of
  ERA5/PlaSim's `mask_fill['sst'] = 270.`, where the field *was* Kelvin. **We reproduce it
  deliberately, because parity is the point of this run**, and because the shipped
  normalization stats were computed under the same 270-fill, so fill and stats agree
  (unlike `TSOI_10CM`, where they do not — R3). Correcting it to `SST = -1.8` with a
  masked-stats recompute is the **documented fast-follow**, out of scope here.

  This is the one place where "every temperature variable is in Kelvin" does *not* hold in
  the E3SM archive itself: `SST` is degC while `T`, `TREFHT` and `TSOI_10CM` are Kelvin.

### 4.9 Total field count — PASS

`90 + 6 + 3 + 2 + 0 + 4 + 3 = 108` on both sides. **PASS.**

---

## 5. What is *not* asserted identical — and is not meant to be

**The architecture is the intended difference.** This must not be read as a full-run
reproduction.

| | jesswan's E3SM run | This run |
|---|---|---|
| Architecture | **SFNO** (`nettype: sfno_plasim`, `embed_dim: 512`, 12 spectral layers) | **Pangu** (`name: PanguPlasimLegacy`, `embed_dim: 240`, Swin-style 3D transformer) |
| Codebase | `PanguWeather/v2.0` (YParams + custom trainer) | `physicsnemo_ai_rossby` (Hydra + ai-rossby recipe) |
| Variable set | **108 fields, as above** | **108 fields, identical** ✅ |

PanguWeather has **no Pangu-on-E3SM config at all** — every `PANGU_*` config in
`PanguWeather/v2.0/config/` is PLASIM or ERA5-S2S, and every `E3SM_*` config is SFNO. So
"PanguWeather parity" here means *the PanguWeather group's E3SM variable contract*, run
through the Pangu architecture. That comparison is the point of the run.

### Non-variable mechanics that change no variable

Three implementation details are required to make the ai-rossby recipe consume this
contract. None adds, removes or re-roles a field:

1. **`sol_in` solar-name patch.** `PanguPlasimLegacy.__init__` routes one varying-boundary
   channel into the 3D stream, identified by name from
   `_solar_names = ("rsdt", "toa_incident_solar_radiation")`, and raises `ValueError` if
   none matches (`pangu_plasim_legacy.py:262-276`; same at `pangu_plasim.py:364-378`).
   E3SM's solar field is `sol_in`, so the tuple gains that third alias. It selects an
   already-present channel; it does not introduce one.
2. **Land folded into the store's `surface` group.** §3 — a container-layout mechanic
   demanded by the store schema, with the model config keeping the groups separate.
3. **`train.py` name lists.** `NanFillTransform` and `PanguPlasimLoss` are handed
   `model.surface_variables` (`train.py:636` and `:739`), which is 6 names while the
   surface tensor is 8 channels. Appending `land + ocean` makes the lists describe the
   tensor. It changes which *names* the fill and loss iterate, not which *fields* exist —
   and without it the fill would broadcast-mismatch rather than fail silently.
   Two further sites had the same latent mismatch and were fixed with the same helper:
   the `ArchesWeatherLoss` branch (off our `loss=mae` path, but a landmine for any later
   loss switch) and `channel_equal_weight`'s `n_surf`, which would otherwise under-weight
   the surface term against the channels it actually scores. All four now derive from one
   `_surface_channel_names()` helper rather than restating the rule.

### Also not asserted

* **Train/val split** — jesswan's run used 2015→2040 train / 2045→2050 val; the first smoke
  here is 1 train year + 1 val year. Deliberate, and orthogonal to the variable set.
* **Normalization statistics** — reused verbatim from
  `mehta5/pangu_polaris_data/data_2015-2050_{mean,std_corr}.nc`, i.e. the *same* constants
  jesswan's run used, converted (not recomputed) into the ai-rossby zarr schema. This
  carries R3 and R4 across unchanged, which is precisely what parity means here.

---

## 6. Verdict

> **ai-rossby `PanguPlasimLegacy` trains the *identical* variable set as jesswan's
> PanguWeather E3SM run** — 108 fields, group for group, name for name, order for order,
> level for level, fill for fill. 10/10 machine checks PASS, against the trained config and
> against all three of its siblings. The *architecture* (Pangu vs SFNO) is the intended and
> only difference.

Backing config paths:

* Ground truth: `PanguWeather/v2.0/config/E3SM_SFNO_H5_STAMPEDE_jsw.yaml`
  (equivalently `_DERECHO_jsw`, `_POLARIS`, `_POLARIS_ALLDATA` — all four verified)
* Contract: `ai_rossby_variable_contract.py::PLANNED`
* To be written and asserted against it:
  `physicsnemo_ai_rossby/examples/weather/ai_rossby/conf/model/pangu_plasim_e3sm.yaml`,
  `.../conf/dataset/e3sm.yaml`,
  `physicsnemo_ai_rossby/tools/data/e3sm/pangu_h5_to_zarr.py::PANGU_E3SM_CHANNELS`

**Open, carried forward (neither blocks this gate):** the `TSOI_10CM` fill-vs-stats
inconsistency (R3) and the `SST = 270` degC fill (R4) are inherited from upstream and
reproduced deliberately. Both still want **jesswan's sign-off** before any resulting
model's numbers are reported, per DESIGN §1.
