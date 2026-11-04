# G4D after 63 epochs: read-out, maps and confidence intervals (2026-11-02)

**What was tested.** G4D was resumed from epoch 23 to epoch 63 (job 7771394), giving four new checkpoints. G4D is
the 5-step-loss, fp32, trained-in-dry-air run whose epoch-16 EMA weights (G4D16) are the current pick. The four new
checkpoints are compared with G4D16 and with A243:
- **G4D36** and **G4D56**: the same point in the learning-rate cycle as G4D16, after 20 and 40 more epochs.
- **G4D43** and **G4D63**: cycle ends.

Each was evaluated on two protocols, described in `../2026-11-02_makani_configurations/`:
- the 14-day forecast sweep (24 forecasts started across the test years 2048–2049);
- the five-year climate protocol: 8 runs per checkpoint.
  - Each run starts in October 2044 and runs to 31 December 2049. "Five-year" is the length of the run, not the
    averaging period.
  - Oct–Dec 2044 is spin-up.
  - Every score, gate and climate-bias map uses only **2045–2047**, the selection window (years 1–3 of the runs).
  - **2048–2049** is the reporting window, held back from selection.

The decision rules were written down before any result existed: `prereg.md`, and `addendum1_extend_rule.md` for
whether to train G4D further.

## Results

All numbers below are checked against their result files with emitall (`claims_final.json`: 30/30 match;
`figures/claims_figures.json`: 19/19 match; `fiveyear/claims_5yr.json`: 32/32 match).

**14-day forecasts: longer training keeps helping.** Median RMSE ratio over the 99 shared channels (below 1 =
more accurate):

| comparison | 6 h | 24 h |
|---|---|---|
| G4D56 / G4D16 | 0.908 (better on 97 of 99 channels) | 0.931 |
| G4D63 / G4D16 | 0.886 | — |
| G4D16 / A243 | 1.385 | — |
| G4D56 / A243 | 1.245 | 1.010 |
| G4D63 / A243 | 1.221 | 0.993 |

- Of G4D16's 6-hour gap to A243, G4D56 closes 0.30 of the log gap (pre-registered: PARTLY, as predicted).
- At 24 h it closes 0.75 (MOSTLY). The prediction was CLOSED, so it missed.
- 14-day anomaly correlation (ACC at 336 h): A243 0.542, G4D16 0.574, G4D56 0.594, G4D43 0.595.

**Five-year climate: it improves in the second learning-rate cycle, then worsens.** The pick stays G4D16 (decided
by the pre-registered tie-break). Score A and gates are in `results.txt`, part 2.
- G4D36 and G4D43 have the lowest score A of any checkpoint; G4D56 and G4D63 are worse than G4D16.
- G4D56 and G4D63 exceed the 3-drifting-group limit.
- The pre-written extend rule therefore says **do not train G4D further**; a second training seed is the better
  use of compute.

**Reading.** More epochs keep improving weather-scale skill, from 6 hours to 14 days. The five-year climate peaks
around epochs 36–43 and degrades after that. All of this rests on one training seed, and every five-year score
is on the 2045–2047 selection window. The five-year figures in `fiveyear/` also show 2048–2049, but descriptively:
nothing was chosen on them.

## Figures (`figures/`)

Every panel names the model and lead time (or averaging window) in its title, labels its axes, and gives its
colorbar's quantity and unit.

| file | what it shows | uncertainty shown |
|---|---|---|
| `rmse_maps_<VAR>.png` | RMSE by grid cell for A243, G4D16, G4D56 and G4D63 at 6, 30, 60 and 270 h | 95 % bootstrap CI of the global RMSE in each title (start dates resampled) |
| `rmse_vs_A243_<VAR>.png` | each G4D checkpoint minus A243, at the same leads (blue = G4D more accurate) | dots where the difference is significant (paired t-test over 24 start dates, Benjamini–Hochberg FDR 10 %) |
| `rmse_vs_G4D16_<VAR>.png` | G4D36, G4D56 and G4D63 minus G4D16: the effect of training longer | same dots |
| `rmse_curves.png` | global RMSE vs lead time for A243, G4D16 and G4D56 | 95 % band over start dates |
| `rmse_ratio_to_A243.png` | the same curves divided by A243's (below 1 = better than A243) | 95 % paired bootstrap band |
| `bias_maps.png` | climate bias averaged over 2045–2047 only (years 1–3 of the five-year runs, the selection window; 2048–2049 not included): mean of the 8 runs minus E3SM, for all five G4D checkpoints | dots where the bias exceeds internal variability (FDR 10 %); RMS-bias CI in each title, bootstrap over the runs re-centred on the estimate (since 11-04) |
| `summary_table.csv` | every number plotted, with its CI | — |

**Variables.**
- TREFHT: 2 m air temperature, K.
- PRECT: precipitation, converted from m/s to mm/day.
- U10: E3SM's 10 m wind *speed*, m/s.
- U_l17: zonal wind on the lowest model level, about 13 m, m/s.
- Z3_l12: geopotential height on model level 12, reference pressure 492 hPa, m.

**Z500 is not shown.** A true 500 hPa height needs E3SM's hybrid coefficients (`hyam`, `hybm`), which the source
files do not carry. Z3_l12 stands in, and it departs from a 500 hPa map over high terrain. A243 has no bias map
because all its five-year runs blow up within 0.6 years.

**What the figures show** (descriptive):
- G4D's short-range deficit against A243 is largest for Z3_l12 and TREFHT. G4D is ahead of A243 from about 1–2
  days, and ahead for precipitation at every lead.
- Five-year TREFHT bias sits mostly at high latitudes and on the rows nearest the poles. Z3_l12 RMS bias is 52.3 m
  for G4D43 and 66.3 m for G4D16.
- Precipitation biases are mostly within internal variability: 3–7 % of the area is significant.
- Near-surface Southern-Ocean westerlies are too strong in every G4D checkpoint.

**Re-centred bias CIs (2026-11-04).** The RMS-bias CI in `bias_maps.png` was first a raw bootstrap percentile
interval. Resampling runs with repeats gives a noisier ensemble mean, which pushes every resampled RMS up, so
that interval sat above its estimate: for PRECT G4D43 and G4D63 the estimate was in its bottom tenth. The figure was
re-rendered with each interval shifted by the bootstrap bias, the same correction as the five-year annual maps.
- The bootstrap draws are the same, and so are the interval widths.
- The maps, estimates, dots and every other number in `summary_table.csv` are unchanged.
- `maps/check_recentre.py` checks this against the first rendering.

## Five-year figures (`fiveyear/`, 2026-11-03)

Four more views of the same five-year runs, covering the whole run, 2045–2049, rather than the 2045–2047 mean.
`fiveyear/README.md` has the method, the gates and the numbers.
- **Views.** Global-mean time series, bias by year, zonal-mean Hovmöller diagrams and annual bias maps.
- **Source.** The NetCDFs had been deleted for quota, so the runs were re-made. The re-run is bit-identical to the
  scored run in its 2045–2047 means and reproduces the protocol scores (gates in `fiveyear/README.md`).
- **No global-mean drift** in any checkpoint.
- **TREFHT.** The spatial bias grows over the first ~15 months, then stays flat. The growth is mostly a
  high-latitude winter warm bias, largest south of 60°S.
- **PRECT** zonal and annual biases are within internal variability.
- **Caveat.** These figures plot 2048–2049, the window the protocol keeps sealed until finalists are frozen. That
  happened after the pick (G4D16) and the stop decision, so no choice was made on it. For these five checkpoints,
  though, it is no longer an unseen hold-out.

## How the numbers were checked

- **Extraction gates.**
  - The 14-day maps reproduce the scorer's own per-lead RMSE to ≤ 4.9e-8 relative, its 8-digit rounding.
  - The five-year maps reproduce the protocol's per-member pattern RMSE to ~2e-16.
  - The A243 and G4D16 re-runs reproduce their original scores.
- **Code review.** The map scripts were code-reviewed (5 findings, all fixed) and smoke-tested on a compute node
  before any raw output was deleted (`maps/smoke2.log`).
- **emitall audits.**
  - `claims_final.json`: the read-out numbers.
  - `figures/claims_figures.json`: the figure numbers. It caught a double-rounding bug in the first figure table,
    which is now fixed.
  - `emitall_audit/`: a retraction guard (`battery.py`) over KNOWLEDGE.md, recent CHANGELOG and the write-ups, and
    a lint of the summary scripts (`lint.py`). Both are clean.

## Regenerating

The working copies of every script live in `$MEMBER_ROOT/runs/makani_eval/g4d_e63_2026-11-02/`, and the scripts
use absolute paths there. `scripts/` and `maps/` here are snapshots of them.

1. Run the chain with `scripts/chain_driver_v3.sh`. It submits each stage as a rerunnable preemptable PBS job, and
   the jobs used are listed in `jobs.txt` and `driver.log`.
2. The extraction jobs write `k56_<ARM>.npz` / `clim_<ARM>.npz`. These are not committed (`*.npz` is gitignored;
   1.4 GB).
3. `maps/polaris_maps_plot.pbs` renders the figures from them. Then run `maps/build_summary_index.py` to rebuild
   the emitall receipt `figures/summary_index.json`.
4. `fiveyear/polaris_fiveyear.pbs` makes the five-year figures, after `fiveyear/polaris_truth_series.pbs`. Its
   npz and the layout-smoke outputs are not committed.

The raw forecast NetCDFs were deleted after extraction.
