# Forcing-pipeline numerical comparison: upstream 5410 vs Makani own-track

Date: 2026-05-10. Companion to `docs/2026-05-10_sfno_param_count_forensic.md`. Raw
numbers in `analysis_outputs/forcing_pipeline_diff_v2_2026-05-10.txt`. Source
scripts in `analysis_outputs/compare_forcing_pipelines{,_v2}.py`.

## Headline corrections vs the previous draft

The previous summary claimed "5410 uses year-11/12 templated boundary regardless of
target year; ours uses actual-year boundary." **This was wrong on the second half.**
Direct comparison of our `boundary.{Y}.nc` files at calendar-matched datetimes
across multiple years shows:

- `rsdt`: byte-identical across years (year-11 Jan 1 vs year-50 Jan 1 vs year-110 Jan 1: max|d| = 0). Astronomical computation depends only on datetime.
- `sic`: byte-identical across years (max|d| = 0).
- `sst`: tiny interannual variation only (RMSE ≤ 0.18 K across years; max|d| = 1.81 K).

**Conclusion: our boundary pipeline is effectively climatological too** —
not "per-target-year" as I asserted. So the "templated vs per-year" framing was
not a real cross-pipeline difference. Both tracks feed essentially climatological
boundary forcing during training; the real differences are in the *physics
definition* of each forcing channel.

## Per-variable numerical comparison

Probe design: 84 (calendar_year, datetime) tuples — first of each month at 00 UTC
across calendar years {11, 15, 25, 50, 75, 100, 110} (all non-leap in
proleptic-Gregorian with year-zero). For each probe, look up:

- **A** = upstream year-11 H5 at the matching day-of-year (what the 5410 loader serves)
- **B** = our `boundary.{Y-5 or Y-6}.nc` (or `MOST.YYYY.nc` for static fields) at the exact datetime

Stats are computed across `(probes × cells)` after masking out cells where either
pipeline is NaN. Per-calendar-year aggregates are essentially the same number
across all 7 years (since neither pipeline depends on year) — the table below
reports one representative row.

| Variable | raw max\|d\| | raw RMSE | raw mean bias (ours − up) | raw σₐ/σᵦ | raw corr | norm-as-fed RMSE | norm-as-fed corr | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `sst` (all common cells) | 71 K | 11.7 K | +3.93 K | 0.604 | 0.865 | 0.612 | 0.865 | **Definitional gap, dominated by sea-ice cells** — see below |
| `sst` (water-only, upstream ≥273 K) | — | **0.36 K** | **−0.073 K** | — | **0.9995** | — | — | **Effectively identical** over open ocean |
| `sst` (ice-surface only, upstream <273 K) | 64 K | 40 K | +34 K | — | low | — | — | Different physical quantity |
| `rsdt` | 1080 W/m² | 458 W/m² | −0.05 W/m² | 0.999 | **0.190** | 1.27 | 0.190 | **Large** — 90° longitude (6 h time-of-day) offset; see below |
| `rsdt` (after +12-cell ≈ +33° east shift on ours) | — | — | — | — | 0.67 | — | — | Residual mismatch persists even after time-alignment |
| `sic` (ocean cells, upstream-valid) | 0 | 0 | 0 | 0.000 | NaN | 0.000161 | NaN | **Pipeline-degenerate**: upstream `sic` is binary `{1, NaN}` (sea-ice indicator), ours is fractional `[0, 1]`. Both reduce to the same value at upstream's valid cells (all = 1.0). |
| `z0` | 2.0e-3 | 2.2e-4 | +4.3e-6 | 1.000 | **1.0000** | 1.3e-4 (norm) | 1.0000 | **Essentially identical** as a snapshot |
| `lsm` | 0 | 0 | 0 | 0.000 | NaN | 1.19e-7 | NaN | Same binary land-sea field; both pipelines encode it as `{1, NaN_or_0}` — verified equal at upstream's valid cells |
| `sg` (orography) | 4e-3 m² s⁻² | 1e-3 | −2e-6 | 1.000 | **1.0000** | 1.2e-7 (norm) | 1.0000 | **Essentially identical** |

## SST: water vs ice-surface definition (the dominant SST discrepancy)

Same datetime (Jan 1 year 11, 00 UTC), commonly-valid cells = 5527:

| Cell class | n cells | mean(ours − upstream) | RMSE | upstream range |
|---|---:|---:|---:|---|
| upstream ≥ 273 K (water) | 4442 | −0.073 K | **0.36 K** | 273.0–303.6 K |
| upstream < 273 K (likely ice surface) | 1085 | +33.7 K | **39.9 K** | 208.9–273.0 K |

**Interpretation.** Upstream's SST field stores actual *surface temperature* —
including sea-ice surface temperatures down to 209 K (≈ −64 °C, plausible polar
winter ice surface). Our pipeline stores *water* SST only and the
`boundary_astro` adaptor pins anywhere-without-water (land, possibly thick ice)
to the 271.35 K land-fill value (`packager.py:181`, `metadata.json[attrs].sst_land_fill_K`).

So:
- **Over open ocean**, the two pipelines agree to ~0.4 K with correlation 0.9995 — numerically a non-difference.
- **Over sea-ice cells**, they differ by ~40 K RMSE because they encode different physical quantities. The model sees this as a feature mismatch wherever sea ice is present.

This is *not* an "interannual" difference and not a "templating" difference — it's a definitional gap in what the field represents.

## RSDT: 6-hour time-of-day offset (the largest single divergence)

Equatorial longitude profile of `rsdt` at year-11 Jan 1 00 UTC (every 8th cell, 16 samples across lon):

```
lon:        0°    22°   45°   67°   90°   112°  135°  157°  180°  202°  225°  247°  270°  292°  315°  337°
upstream:   0     0     0     0     0     73    260   531   845   1081  1150  1042  774   466   210   47
ours:       0     60    236   501   815   1069  1158  1069  815   501   236   60    0     0     0     0
```

- **Upstream** peaks at lon ≈ 225°. That is roughly the expected sub-solar longitude at 00 UTC (180° + small post-noon lag from PlaSim's diagnostic averaging).
- **Ours** peaks at lon ≈ 135°. That is the rsdt pattern expected ~6 h *earlier* in the day (sub-solar at 135° ↔ local solar noon at lon 135° ↔ UTC ≈ 12 − 135/15 ≈ 3 h after midnight UTC).

The two fields are ~90° = 6 hours offset on the longitude axis. A longitude-roll sweep confirms it: rolling our rsdt east by +33° lifts cross-pipeline correlation from 0.19 to 0.67. Even at the optimal roll, residual correlation is only 0.67 — so beyond the time-of-day offset there's additional spatial mismatch (atmospheric attenuation? labeling convention for "time = start vs middle vs end of 6 h"?).

**Numerical impact:** Raw RMSE 458 W/m² out of a 0–1300 W/m² field with mean 301 W/m². Normalized RMSE 1.27 vs the variable's own std of 360 — i.e., **the difference is comparable in magnitude to the signal itself**. This is one of the strongest cross-pipeline divergences in the boundary forcing.

The likely culprit, code-level: our `boundary_astro` adaptor computes rsdt from solar geometry at the exact datetime stamp written in the file. Upstream's `rsdt_masked_6h.nc` carries PlaSim's diagnostic, which under PlaSim's convention is a 6 h average (or instantaneous at a +/-3 h offset to the file's time stamp). I have not audited PlaSim's exact rsdt-diagnostic convention; the time-of-day offset is empirically visible regardless of which side is "correct."

## SIC: binary indicator vs fractional concentration

Upstream year-11 Jan 1 SIC: 951 valid cells (all exactly 1.0); 7241 NaN cells. Same pattern at year-11 day 180 (1095 cells, all = 1.0). Confirmed: upstream stores `sic` as a binary "sea-ice present here = 1, else NaN" indicator. Ours stores fractional `[0, 1]` from PlaSim's continuous output.

At the cells where upstream is valid, our values are 1.0 too within 1.6e-4 (RMSE in normalized space): so within the *upstream-valid subset*, the two pipelines agree. But the *out-of-subset behavior* differs: upstream has NaN where there's no ice; ours has 0.0 (or small fractional) at the same cells. The upstream data loader then z-scores using its global mean/std and feeds the result; the NaN handling at training time is **not audited** here — without access to the trainer's NaN-masking logic, it is uncertain whether NaN-valued sic cells propagate into the model input or are filled with 0 / mean.

**Verdict**: structurally a real difference, but numerically *moderate* because the ice-present cells agree, and the fractional information our pipeline carries is lost only in the sub-1.0 transition zone.

## z0: temporal variance dropped by upstream (small in practice)

Per-pixel temporal std of our z0 over a year, vs upstream's static treatment:

| Calendar year | mean σ_t(z0) over grid | median | max | ocean frac with σ_t > 1e-6 |
|---:|---:|---:|---:|---:|
| 11  | 1.02e-4 | 5.4e-5 | 4.7e-4 | 0.89 |
| 15  | 1.03e-4 | 5.4e-5 | 4.7e-4 | 0.89 |
| 25  | 1.03e-4 | 5.3e-5 | 4.7e-4 | 0.89 |
| 50  | 1.03e-4 | 5.3e-5 | 4.7e-4 | 0.89 |
| 75  | 1.03e-4 | 5.3e-5 | 4.7e-4 | 0.89 |
| 100 | 1.03e-4 | 5.5e-5 | 4.7e-4 | 0.89 |
| 110 | 1.02e-4 | 5.5e-5 | 4.8e-4 | 0.89 |

z0 itself has spatial range 1.5e-5 to 36.7 m (Charnock ocean roughness vs forest canopy). Per-pixel temporal σ ~ 1e-4 m is **~6 orders of magnitude below the field max** and ~5e-5 relative to the field's own global std (1.80). In normalized model-input space, σ_t/σ_global ≈ 6e-5 — well below any noise floor.

**Verdict**: numerically negligible. Dropping z0's temporal dimension (upstream's choice) erases ~0 information that the model could use.

## Constants normalization scheme

Upstream's training loader spatially z-scores the constant boundary fields per
instance (`data_loader_multifiles.py:660-668`). Our packager pre-computes
global means/stds (`forcing_global_means.npy`, `forcing_global_stds.npy`).

For time-constant fields the two schemes are mathematically equivalent up to
rounding:

| Field | upstream-mean − global-mean | upstream-std / global-std | max\|up_z − our_z\| | RMSE after both z-scores |
|---|---:|---:|---:|---:|
| `sg` | 0.0 | 1.0000 | 7.2e-7 | 1.2e-7 |
| `z0` (snapshot) | 0.0 | 1.0000 | 0.0 | 0.0 |
| `lsm` (all 1.0 in upstream valid cells, all 0.0 elsewhere) | +0.6747 | 0.0000 | **1.44** | **1.44** |

The `lsm` case breaks the equivalence because upstream's mask leaves `lsm = 1`
on a small subset (land cells) and NaN elsewhere → spatial std = 0 → z-score is
undefined / NaN. Our global stats include all cells (including the NaN-region as
0 after `_fill_mask`), giving a finite global std and a well-defined z-score. So
**upstream's lsm z-score is degenerate by construction**; our version isn't.
Whether the 5410 trainer actually uses lsm in the model after this NaN
normalization is uncertain — `data_loader_multifiles.py:660-668` returns
constant_boundary_data as a tensor, and `_fill_mask` may fill the NaN before
return, but we don't have the full propagation audit.

## Updated headline for the forensic doc

Replace the earlier "boundary-year templating" and "rsdt method" items with
this short list:

| Difference | Numerical magnitude | Verdict |
|---|---|---|
| `rsdt` time-of-day convention (~6 h offset) | normalized RMSE 1.27, raw 458 W/m² | **Large** — biggest single boundary-forcing divergence |
| `sst` definition (water-only vs surface incl. sea-ice) | open-ocean RMSE 0.36 K (corr 0.9995); ice-cell RMSE 40 K (≈1000 cells) | Moderate — confined to high-latitude sea-ice region |
| `sic` encoding (binary vs fractional) | matches at upstream-valid cells; differs in transition zone | Moderate; numerical impact bounded |
| `z0` static vs time-varying | σ_t / σ_global ≈ 6e-5 | **Negligible** |
| `lsm` normalization (upstream is degenerate) | upstream z = NaN over land subset | Pipeline-level concern; numerical impact depends on NaN handling we didn't audit |
| `sg` normalization | identical | Non-difference |
| "Templated vs per-target-year" boundary | both pipelines are climatological | **Non-difference** (previous claim retracted) |

## Implications for the 5410-skill-advantage hypothesis

Of the boundary-forcing differences, the only one likely to influence model
skill at *any* meaningful magnitude is the **rsdt time-of-day offset**. A model
trained on rsdt with a 6 h labeling offset is learning a fundamentally
different solar-cycle phase relative to atmospheric state. Whether the 5410
ckpt's training-time rsdt is "correct" (and ours wrong) or vice versa depends
on PlaSim's diagnostic convention — but they cannot both be right at the
same datetime.

If we want to remove this as a confounder for the GB8-clone vs 5410
comparison, the cheap next step is to **realign our rsdt by 6 hours in the
packager** (or, equivalently, replace astronomical rsdt with the 6h-averaged
form PlaSim emits) and retrain. The other boundary-forcing differences are
unlikely to produce ≥1 % skill swings on the metrics we care about.

## Verification status table

| Finding | Verification source |
|---|---|
| Upstream H5 preprocessor lists lsm/sg/z0 as CONSTANTS and sst/rsdt/sic as VARYING_BOUNDARY | `netcdf-to-h5-new.py:30-49` |
| Upstream loader pulls boundary from year 11/12 templates regardless of target year | `data_loader_multifiles.py:941-944` |
| Our packager reads `boundary.YYYY.nc[sst/rsdt/sic]` and `MOST.YYYY.nc[lsm/sg/z0]` | `packager.py:155-186` |
| Our boundary.YYYY.nc time range is Aug 1 (Y+5) to Jul 30 (Y+6) | Verified via `xr.open_dataset` on multiple files |
| Our boundary is climatological (rsdt/sic byte-identical across years, sst RMSE ≤0.18 K across years) | Direct value-by-value comparison of `boundary.{5,44,104}.nc[Jan 1]` |
| Upstream SIC is binary {1.0, NaN} at year 11 Jan 1 and day 180 | Direct `h5py` inspection; all valid values = 1.0 |
| Upstream SST stores surface temp incl. sea-ice surfaces (min 209 K) | Direct `h5py` inspection |
| Ours SST water-only with 271.35 K land/ice fill | Direct `xr` inspection + packager docstring |
| RSDT 6h time-of-day offset between pipelines | Equatorial longitude profile comparison + longitude-roll cross-correlation sweep |
| Open-ocean SST agreement 0.36 K RMSE corr 0.9995 | Stratified comparison at upstream ≥ 273 K |
| z0 σ_t ≈ 1e-4 across 7 years | Direct std-over-time on our MOST.YYYY.nc files |
| lsm spatial z-score is degenerate (std=0) | Direct numpy on upstream year-11 lsm snapshot |
| sg / z0 normalization schemes give identical normalized values | Direct numpy comparison |
