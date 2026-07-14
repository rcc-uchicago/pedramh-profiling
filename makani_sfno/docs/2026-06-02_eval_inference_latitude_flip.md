# Latitude-flip in emulator inference output (`eval_inference.py`)

**Date:** 2026-06-02
**Status:** Confirmed. Fixed in a *downstream* repo (`ensemble-perturbation-study`); the
root cause in **this repo** (`SFNO_Climate_Emulator`) is still present and needs a decision.
**Severity:** Silent corruption of any coordinate-aware scoring of NWP-mode inference NetCDFs.
**NOT affected:** model training, in-training validation / checkpoint selection, the forward
pass, and the forecast *data values* themselves.

> This document is written so an independent agent can re-verify every claim. Reproduction
> commands are inline. Please confirm or refute each numbered claim in the "Verification" section.

---

## 1. One-paragraph summary

`scripts/eval_inference.py` writes NWP-mode inference NetCDFs whose `lat` **coordinate** is in
the **reverse order** of the **data array** it labels. The data rows come from the source h5 in
**descending** latitude order (row 0 = +87.9°N), but the `lat` coordinate is taken from
`_read_lat_lon_from_run()` → `metadata/data.json`, which is **ascending** (−87.9 → +87.9). The
writer (`src/sfno_inference/nc_writer.py`) only length-checks `lat`, never its orientation, so every
written field (`prediction`/`control`/`truth`/`init_state`) ends up **hemisphere-flipped**: row 0
holds Northern-Hemisphere data but is labeled −87.9°S. Anything that later reads these files and
joins them to an **external, coordinate-referenced** field (e.g. a day-of-year climatology, by lat
label) silently scores the wrong hemisphere.

---

## 2. How it was discovered

A downstream study (`~/projects/ensemble-perturbation-study`) ran the emulator and PFS (PlaSim free
run) from the *same* matched initial conditions and scored both with the AI-RES blocking scorer
(`GridpointIntensityScorer` / `Blocking_gridpoint_intensity`, z500 anomaly vs. a sim52 day-of-year
climatology). The emulator blocking A-values came out **strongly negative** (e.g. IC0 ≈ −418) while
PFS was **positive** (≈ +134) for the same IC. Units, dates, and ICs all matched; the giveaway was
that selecting zg500 **by coordinate value** (orientation-proof) showed the emulator field ~500 m
*below* PFS/climatology at high latitude, growing toward the pole — and `OUTPUT(+70°) == raw_h5(−70°)`.

---

## 3. Root cause (exact)

> **Correction (2026-06-02, post-fix verification).** The mechanism below is slightly off and is
> superseded by this note. `metadata/data.json` does **not** carry a reversed top-level `lat`; in
> fact it has **no** top-level `lat` key at all — the array lives under `coords/lat` and is
> **descending (correct)** in every dataset in the repo. The real flip source is that
> `_read_lat_lon_from_run` checked for a *top-level* `lat` key, always missed, and fell through to
> the **torch_harmonics fallback**, which returns Gauss–Legendre nodes **ascending** (−87.86 →
> +87.86). So the packager artifact was fine; the buggy reader + ascending fallback produced the
> flip. Fixed in `scripts/eval_inference.py` (now reads lat/lon from the source h5 with a metadata
> `coords/lat` cross-check) + an orientation guard in `src/sfno_inference/nc_writer.py`. See
> `docs/2026-06-02_eval_inference_latitude_flip_fix_plan.md`.

- **Data order (authoritative):** the source h5 (`fields_state`) is **descending** lat
  (`lat[0] = +87.86`, `lat[-1] = −87.86`). Verified that `fields_state[0, 46(zg500), row=+70°, lon=330] = 5498.1 m`
  — the physically-correct NH-summer 500 hPa height (matches PFS 5489 and the sim52 climatology 5476).
- **Coordinate written:** `scripts/eval_inference.py` does `lat, lon = _read_lat_lon_from_run(args.run_dir)`
  (lines 237, 317) and passes it to `write_rollout_nc(..., lat=lat, ...)` (line ~278).
- `_read_lat_lon_from_run` (eval_inference.py:102) reads `metadata/data.json`'s `lat`, which is
  **ascending** (−87.9 → +87.9). Confirmed by calling it directly (returns `[-87.9, -85.1, ... 85.1, 87.9]`).
- The writer `src/sfno_inference/nc_writer.py` only checks `len(lat) == H` (line ~100) and assigns
  `lat=("lat", np.asarray(lat))` (line ~141) — **no orientation check**, so descending data gets an
  ascending label.

**The inconsistency is in the packager artifact:** `metadata/data.json` `lat` is reversed relative
to the actual data. Two other lat sources in the ecosystem are **correct/descending** and agree with
the data: the h5's own `lat`, and AI-RES's `forecast_modules/PanguPlasimFS/yaml_config/SFNO_V11_STAMPEDE3.yaml`
(`PLASIM.lat`, descending). AI-RES's canonical scorer uses the yaml and is therefore **unaffected**.

---

## 4. Proof it is a label-only flip (data is correct)

For inference file `MOST.0102_ic000.nc` (before the downstream patch):

| Query | Value | Interpretation |
|---|---|---|
| OUTPUT `truth` zg500 @ label `lat=+70` | 4993.9 | SH-winter value (wrong) |
| OUTPUT `truth` zg500 @ label `lat=−70` | 5497.7 | NH-summer value (wrong place) |
| RAW h5 `zg500` @ +70 (NH) | 5498.1 | correct NH |
| RAW h5 `zg500` @ −70 (SH) | 4984.7 | correct SH |

`OUTPUT(+70) ≈ RAW(−70)` and `OUTPUT(−70) ≈ RAW(+70)` ⇒ the **data array is intact**; only the `lat`
coordinate is reversed. Fix = relabel `lat` (reverse the 64-value coordinate), no re-forecast needed.

---

## 5. Scope: what is and is NOT affected

| Stage | Aligns to an external coordinate-referenced field? | Affected? | Why |
|---|---|---|---|
| **Training loss** | No | **No** | Lat/area weights come from quadrature (`utils/grids.py` → `torch_harmonics` `legendre_gauss_weights`, from `grid_type`+`nlat`), applied **positionally** to data tensors. `metadata/data.json` lat array is never used. |
| **In-training validation / ACC / checkpoint selection** | No | **No** | Validation L2 and ACC compare pred vs. target (and a *normalized clim tensor*, `utils/metric.py:382` — positional, channel-masked, in data order) element-wise in the same descending order. No lat-label join. Quadrature/cos weights are equator-symmetric ⇒ flip-invariant anyway. |
| **Forward pass / forecast values** | No | **No** | SFNO SHT grid is internal (`img_shape`,`grid_type`); operates on the data tensor order. |
| **Output coordinate label written by `eval_inference.py`** | — | **Yes** | The bug itself: ascending label on descending data. |
| **Post-hoc scoring that joins output to an external clim by lat** | **Yes** | **Yes** | The asymmetric N≠S join lands on the wrong hemisphere. This is the only place it manifests. |

**Empirical backstop that training is sound:** the resulting checkpoint is highly skillful — emulator
`prediction` vs `truth` zg500 global RMSE = **2.7 m at the IC, 5.2 m at day 1, 18.9 m at day 5**, with
prediction tracking truth at the *correct* hemispheres (NH ~5500, SH ~4920). A model trained or
selected under a latitude inconsistency could not achieve this against correctly-oriented truth.

`metadata/data.json` is read in the training pipeline **only** for `coords.grid_type` (a string,
`utils/parse_dataset_metada.py:37`) and channel names — never the lat array.

---

## 6. Blast radius beyond this repo

- **`ensemble-perturbation-study`** (downstream): its `study/ensemble_eval_driver.py` had the *same*
  pattern (lat from `_read_lat_lon_from_run`). **Already fixed** there: it now reads lat/lon from the
  source h5 with a fail-loud cross-check against the run metadata, and the 10 already-written output
  files were patched in place (reversed only the `lat` variable via netCDF4 `r+`; data untouched).
- **This repo (`SFNO_Climate_Emulator`)**: `scripts/eval_inference.py` NWP-mode outputs are still
  written flipped. Any existing NWP inference NetCDFs from this script are hemisphere-flipped in their
  `lat` coordinate. Whether this ever produced *wrong published numbers* depends on whether those files
  were scored by a coordinate-aware (lat-label-joining) scorer; pure visualization or self-consistent
  pred-vs-truth tensor diffs would not reveal it.
- **AI-RES (`PanguPlasimFS`) canonical scoring**: **not affected** — sources lat from the yaml
  (descending), which matches the data. The PFS path is also unaffected (PlaSim postproc, descending).

---

## 7. Recommended fix (this repo) — pick one, in order of preference

1. **Fix `_read_lat_lon_from_run`** (`scripts/eval_inference.py:102`) to return lat in **data order** —
   read it from the source h5 (`<file>['lat']`) or from the descending yaml, and assert the value-set
   matches `metadata/data.json` (fail loud on a real grid mismatch). This repairs **every** caller at
   once (lines 237, 317).
2. **Regenerate the packager's `metadata/data.json`** with descending `lat` to match the data (fixes
   the source artifact; also makes any other metadata-lat consumer correct).
3. **Add an orientation guard** in `src/sfno_inference/nc_writer.py`: require `lat` to be monotonic and
   (optionally) descending, or cross-check the sign of `lat[0]` against a known data sample. Defense-in-depth.

Reference implementation already shipped downstream (`ensemble-perturbation-study/study/ensemble_eval_driver.py`):
read lat/lon from `test_files[0]` (h5) and assert `sorted(round(h5_lat)) == sorted(round(meta_lat))`.

---

## 8. Verification (for the checking agent)

Environment: `conda activate /work2/11114/zhixingliu/stampede3/conda-envs/aires`.

**Claim A — `_read_lat_lon_from_run` returns ascending:**
```python
import sys; from pathlib import Path
sys.path.insert(0, "scripts")               # run from SFNO_Climate_Emulator/
from eval_inference import _read_lat_lon_from_run
rd = Path("/scratch/11114/zhixingliu/SFNO_Climate_Emulator/runs/"
          "sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_noise0p020_epochs75/"
          "plasim_sim52_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_noise0p020_epochs75/0")
lat, lon = _read_lat_lon_from_run(rd)
print(lat[0], lat[-1])      # expect -87.9 ... 87.9  (ASCENDING)
```

**Claim B — source h5 data is descending and physically correct:**
```python
import h5py
f = h5py.File("/home1/11114/zhixingliu/projects/ensemble-perturbation-study/"
              "data/eval_holdout_restart/MOST.0102.h5", "r")
lat = f["lat"][:]                            # expect 87.9 ... -87.9 (DESCENDING)
cs = [x.decode() for x in f["channel_state"][:]]   # index 46 == 'zg500'
import numpy as np
ila = int(np.argmin(abs(lat-70))); ilo = int(np.argmin(abs(f["lon"][:]-330)))
print(lat[0], lat[-1], f["fields_state"][0, 46, ila, ilo])   # ~5498 at +70N
```

**Claim C — written output is hemisphere-flipped (use any *un-patched* NWP output of this repo):**
```python
import xarray as xr
de = xr.open_dataset("<eval_inference NWP output>.nc", decode_timedelta=True)
ci = [str(x) for x in de["channel"].values].index("zg500")
t = de["truth"].isel(init_time=0, channel=ci, lead_time=0)
print(float(t.sel(lat=70,  lon=330, method="nearest")))   # ~4994 (SH value at +70 label) => flipped
print(float(t.sel(lat=-70, lon=330, method="nearest")))   # ~5498 (NH value at -70 label)
```
(Note: the `ensemble-perturbation-study` outputs under
`/scratch/.../results/emu_eps0_N50_seed0_perlin_sst/inference/` have ALREADY been patched to
descending — they will now read correctly and are NOT a reproduction of the bug.)

**Claim D — AI-RES yaml lat is descending (so AI-RES is unaffected):**
```python
import yaml
y = yaml.safe_load(open("/home1/11114/zhixingliu/projects/AI-RES-clean/"
                        "forecast_modules/PanguPlasimFS/yaml_config/SFNO_V11_STAMPEDE3.yaml"))
print(y["PLASIM"]["lat"][:2], y["PLASIM"]["lat"][-2:])   # 87.9.. .. ..-87.9  (DESCENDING)
```

**Claim E — metadata lat is NOT used for loss/metric weighting (training/validation safe):**
- `makani-src/makani/utils/grids.py:27` `grid_to_quadrature_rule` + `torch_harmonics.quadrature`
  (`legendre_gauss_weights`) ⇒ weights from `grid_type`+`nlat`, positional.
- `makani-src/makani/utils/parse_dataset_metada.py:37` reads only `metadata["coords"]["grid_type"]`.
- `makani-src/makani/utils/metric.py:382` ACC climatology is a normalized **tensor** (channel-masked,
  data order), not a coordinate-joined external field.
- Grep to confirm no `metadata`/`data.json`/lat-array use in `utils/loss.py`, `utils/metric.py`,
  `trainer*`.

---

## 9. Questions for the reviewer

1. Do any existing **NWP-mode** inference NetCDFs from this repo's `eval_inference.py` feed a
   coordinate-aware scorer anywhere (so their results would be wrong)? Or only visualization /
   tensor-space diffs?
2. Preferred fix (Section 7): patch `_read_lat_lon_from_run`, regenerate `metadata/data.json`, or both
   + a writer guard?
3. Should `nc_writer.py` gain a permanent orientation assertion to prevent recurrence?
