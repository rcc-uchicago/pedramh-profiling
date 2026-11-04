# Five-year figures for G4D16/36/43/56/63 (2026-11-03)

**Why.** The G4D63 read-out had only one five-year figure: the 2045–2047 climatological bias maps. The operator
asked for four more five-year views:
- global-mean time series;
- bias by year;
- zonal-mean Hovmöller diagrams;
- annual bias maps.

The five-year NetCDFs had been deleted on 11-02 (quota), so the runs were re-made.

## How

1. **Re-run (7809283, preemptable).**
   - Same code (eval-protocol @ 5848643de503, clean), same EMA weights (`ckpt_mp0_v{15,35,42,55,62}.tar`), same
     8 start dates. `--zonal-mean` was added; it stores the zonal means the global means are already computed
     from (`climate_driver.py:523-530`), so no rollout value changes.
   - Gate: the re-run must *be* the original run. Checked in `extract_5yr.py`, per member:
     - its 2045–2047 window mean must equal the one kept from the original in `maps/clim_<ARM>.npz`;
     - its pattern RMSE must equal the protocol score `e` in `protocol_scores/<tag>_channels.csv` (relative 1e-9).
2. **E3SM truth (7809287).** `truth_series.py` works per year for 2044–2049. It goes through the truth builder's own
   dataset and normalisation path and computes:
   - per-frame global means, weighted exactly as the rollouts weight theirs;
   - per-frame zonal means;
   - the annual time mean.

   Gate: for 2045–2047 the annual mean must equal `window_truth/truth_<year>.npz` (relative 1e-12).
3. **Alignment.** The truth files hold frames 1..1459 of each year: frame 0, Jan 1 00:00, is never a target in the
   builder. Every series here therefore drops frame 0 from the runs too, so runs and truth are compared at
   identical times.
   - Annual maps use the protocol's own `window_mean`, as the 2045–2047 bias maps did.
   - October 2044 is not plotted: it is a partial month for every run, since starts fall on Oct 1–29.
4. **Statistics.** E3SM is a single realisation, and the eight runs' spread stands for internal variability. The
   test is t = bias / (s·√(1/8 + 1)), the one used for the 2045–2047 bias maps.
   - Bias by year: 95 % interval, bias ± t₀.₉₇₅,₇·s·√(1/8 + 1).
   - Maps and Hovmöller diagrams: Benjamini–Hochberg FDR 10 % dots.
   - Annual maps: bootstrap CI over the runs of the global RMS bias (2,000 resamples).
5. `polaris_fiveyear.pbs` runs in two modes:
   - **real** (7809303, after both jobs): lib test → extraction and gates → deletes the re-run's NetCDFs → plots.
   - **smoke**: fake inputs → plots, for checking the layout only.

## Results

All numbers are emitall-checked (`claims_5yr.json` → 32/32 MATCH against `summary_5yr_index.json`, built by
`index_5yr.py` from the figure table, `trefht_band_bias.csv` and the gate logs).

**Gates passed.**
- The re-run is the original run: all 2,592,000 float32 window-mean cells are identical per checkpoint, and the
  protocol scores reproduce to ≤ 2.98e-16 relative.
- The truth series reproduces `window_truth` for 2045–2047 exactly (max relative difference 0).

**Findings.**
- **No global-mean drift in any checkpoint.**
  - Annual global-mean biases, 2045–2049, are at most 0.08 K (TREFHT), 0.015 hPa (PS) and 0.15 kg m⁻² (TMQ).
  - No TREFHT trend interval excludes zero. Over all seven variables × five checkpoints, 1 of 35 trend
    intervals excludes zero (Z3_l12, G4D43), about what chance gives at 95 %.
  - Month to month, all five checkpoints' biases move together (`ts_bias_all.png`). That co-movement is E3SM's own
    single-realisation weather, not model drift.
- **The spatial TREFHT bias grows fast during the first ~15 months, then more slowly.**
  - Annual pattern RMS: 0.54–0.59 K in 2045, then 0.75–1.01 K in 2046–2049, for every checkpoint.
  - *Corrected 11-04.* This bullet first said "then stays flat". The per-year values keep rising after 2046:
    G4D16 goes from 0.833 K in 2046 to 0.948 K in 2048, and G4D43 from 0.755 K to 0.931 K by 2049. In every
    checkpoint, 2048 or 2049 is above 2046 (`claims_window.json`).
  - The rise sits at high latitudes. The mean bias south of 60°S goes from −0.18…+0.31 K in 2045 to
    +0.50…+1.38 K in 2046, mostly in austral winter (`hovmoller_TREFHT.png`). The northern high latitudes warm in
    winter too.
  - 60°S–60°N RMS rises less, from about 0.5 K to about 0.7 K. A part of that is the runs losing the initial
    state's memory, which also makes them track E3SM's own year-to-year weather less closely.
  - Share of TREFHT zonal-month cells beyond internal variability: G4D16 11 %, G4D36 13 %, G4D43 4 %,
    G4D56 10 %, G4D63 30 %.
- **PRECT zonal and annual biases are within internal variability.** At most 0.2 % of zonal-month cells are
  significant, and 0–3 % of the annual-map area.

**Statistics note.** The annual maps' RMS-bias CIs are bootstrap intervals re-centred on the estimate. The raw
percentile interval is shifted upward by resampling (a resample's repeated runs give a noisier mean), and for
single-year PRECT it excluded its own estimate (e.g. 0.545 [0.546, 0.591]). The 2045–2047 `bias_maps.png` first
used the raw interval: its estimates sat inside, but in the bottom tenth for PRECT G4D43 and G4D63. On 11-04 it
was re-rendered with the same re-centring (`maps/check_recentre.py`: same draws, same widths, every other number
unchanged).

**2048–2049 are shown here.** The protocol keeps 2048–2049 as a sealed reporting window until the finalists are
frozen. These figures plot it for G4D16/36/43/56/63, after the pick (G4D16) and the stop decision were made. No
choice was made on it, but for these five checkpoints it is no longer an unseen hold-out.

## 2048–2049 bias maps (2026-11-04)

**Why.** The operator asked for the reporting window as a map: `bias_maps.png` shows 2045–2047 only, the years the
checkpoints were ranked on. This is descriptive and selects nothing.

**How** (`plot_window_bias.py`, debug job 7813261, `WINDOW_BIAS_OK`).
- **Method.** `bias_maps.png`'s method, with the same tests, re-centred CI and colour limits per variable, run
  through the same code (`plot_maps.fig_bias`). The refactor that made it reusable re-renders all 18 published
  PNGs and `summary_table.csv` byte-identically (checked in the same job).
- **The 2048–2049 mean** is the equal-weight mean of the 2048 and 2049 annual means. This holds because every
  year has the same frame count.
  - Gate 1: the 2045–2047 annual means rebuild the scorer's 2045–2047 window means to ≤ 8.9e-8 relative (float32
    storage), for E3SM and every member.
  - Gate 2: 1459 frames in every year, for every member and E3SM.
- **Like-for-like row.** `summary_window_bias.csv` also has 2046–2047. That window is two years long like
  2048–2049, inside the selection window, and after the fast early growth. Comparing 2048–2049 with the 3-year
  2045–2047 instead would mix in a shorter, noisier average and that early growth.

**Results** (`claims_window.json`: 43/43 MATCH).
- **The bias is larger in 2048–2049.** It exceeds 2046–2047 in 23 of 25 variable × checkpoint cases. The ratio,
  2048–2049 over 2046–2047, by variable:
  - TREFHT 1.093–1.178
  - PRECT 0.985–1.159
  - U10 0.995–1.163
  - U_l17 1.022–1.242
  - Z3_l12 1.141–1.218
- **This is mostly growth with time, not the selection effect.**
  - G4D56 and G4D63, which were not chosen, rise as much as the others.
  - Z3_l12's annual pattern RMS rises every year in every checkpoint (20 of 20 year steps): G4D16 goes from
    44.6 m in 2045 to 95.5 m in 2049, and G4D43 from 35.1 m to 77.8 m.
- **G4D43 has the largest rise in all five variables.** That fits its 2045–2047 numbers being somewhat flattered,
  but with one seed and one E3SM realisation it is a hint, not a result.
- **The pick's Z3_l12 weakness holds on unseen years.** G4D16 is 92.9 m against 74.0–81.5 m for the others.

**Superseded figure sets.** These are kept for comparison only.
- `figures_v1/`: coarse colour scale.
- `figures_v2_uncentred/`: raw bootstrap CI.
- `figures_v3/`: trend CI not inflated.

## Files

- **Re-run and truth.**
  - `truth_series.py`, `polaris_truth_series.pbs`, `truth/`: E3SM series.
  - `extract_5yr.py`: the re-run gates, plus `fiveyear_<ARM>.npz` and `fiveyear_truth.npz`.
- **Plotting.**
  - `fiveyear_lib.py` and `test_fiveyear_lib.py` (`TEST_FIVEYEAR_LIB_OK`): calendar, alignment, binning and
    statistics. The test runs on the login node's python3.
  - `plot_5yr.py`: figures and `figures/summary_5yr.csv`, which holds every plotted number with its interval.
  - `make_fake_5yr.py`, `smoke/`: the layout smoke (7809296, `FIVEYEAR_SMOKE_OK`).
  - `plot_window_bias.py`, `polaris_window_bias.pbs`, `window_bias.log`: the 2048–2049 bias maps and the window
    table. `build_claims_window.py` writes `claims_window.json`.

## Figures (`figures/`)

| file | what it shows |
|---|---|
| `ts_global_<ARM>.png` | monthly global means, Nov 2044 – Dec 2049: the 8 runs, their mean, and E3SM. Variables: TREFHT, PRECT, U10, U_l17, Z3_l12, PS, TMQ |
| `ts_bias_all.png` | monthly global-mean bias of all five checkpoints on one axis per variable |
| `bias_by_year.png` | annual global-mean bias, 2045–2049, with 95 % intervals. Filled marker = outside internal variability |
| `hovmoller_TREFHT.png`, `hovmoller_PRECT.png` | zonal-mean bias, latitude × month, one row per checkpoint |
| `annual_bias_<VAR>.png` | annual-mean bias maps: rows are 2045–2049, columns are the checkpoints |
| `bias_maps_2048_2049.png` | bias averaged over 2048–2049 (the reporting window), laid out like `bias_maps.png` and on its colour scale |
| `summary_window_bias.csv` / `.json` | RMS bias with CI and significant area for 2045–2047, 2046–2047 and 2048–2049 |

Shading or dashed lines mark the selection window, 2045–2047: the years used to rank checkpoints. 2048–2049 were
never used for selection.
