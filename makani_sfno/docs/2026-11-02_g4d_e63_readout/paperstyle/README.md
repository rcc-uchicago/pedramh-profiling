# G4D scored the way Duncan et al. (2024) score ACE (2026-11-04)

**Why.** The operator asked how our five-year results compare with the E3SM ACE paper: Duncan et al. (2024),
*Application of the AI2 Climate Emulator to E3SMv2's Global Atmosphere Model*, JGR MLC. Its numbers can't be set
next to our score A directly. The averaging window, the yardstick and the setup all differ. So G4D was re-scored with
the paper's own metric. The plan was pre-registered first: `prereg.md`.

## How the comparison works

**What the paper measures** (Table S5; `members/mehta5/papers/duncan2024_readout.md`). For ACE v1 emulating EAMv2:
1. **Model error.** One continuous 10-year ACE run's time-mean map minus EAMv2's 10-year time-mean map, summarised
   as the global (area-weighted) RMS of that difference.
2. **Natural variability ("noise floor").** The same RMS between two different 10-year periods of EAMv2 itself. That
   is how far apart the target model's own decades are, with no emulator involved.
3. **Their ratio.** About 1 means the emulator's climate is as close to EAMv2 as EAMv2 is to itself.
4. **Reported values:**
   - precipitation: 0.37 / 0.21 mm/day = **1.76**;
   - reflected sunlight at the top of the atmosphere: 4.17 / 1.63 W/m² = **2.56**;
   - outgoing longwave: 2.26 (we have no OLR channel).

**What we did, step for step:**
1. **Model error.** One G4D run's 2045–2049 mean map minus E3SM's 2045–2049 mean map, global RMS.
   - Each of the 8 runs per checkpoint is scored on its own, like the paper's single run.
   - We report the median and the range.
   - Checkpoints: G4D16, G4D36 and G4D43.
2. **Natural variability.** E3SM's own 2040–2044 mean against its 2045–2049 mean.
   - A second pair, 2035–2039 against 2040–2044, checks that this yardstick is stable. It agrees within 3–16 % on
     every reported variable.
3. **Ratio R = (1) ÷ (2).** It is the paper's quantity, on our data.
4. **Reflected sunlight.**
   - We have net sunlight at the top of the atmosphere (FSNTOA), which is incoming minus reflected.
   - Incoming sunlight is prescribed identically for the runs and for E3SM, so FSNTOA's error map is minus the
     reflected-sunlight error map. The RMS is the same.

**Why the yardstick is fair.** Our archive's sea-surface temperature, sea ice and sunlight repeat exactly every year,
as in the paper's fixed-2010 run. So the window-to-window difference is natural variability, not a forced trend. The
check: E3SM's global-mean temperature differs by only 0.03–0.04 K between windows, well under half the yardstick.

**What still differs, and in which direction:**
- **5-year windows here, 10-year in the paper.** Natural variability shrinks with longer averages, but a fixed bias
  doesn't, so the same model scores a higher ratio over 10 years. Our ratios are flattered. A model-based 10-year
  equivalent, `R10 = sqrt(1 + 2·max(R² − 1, 0))`, is given as an approximation.
- **Different data.** Our E3SM archive vs their EAMv2 run.
- **Different emulator.** ACE v1 vs our SFNO.
- **The paper reports no surface pressure, temperature or height numbers.** On those channels we can say how G4D
  compares with E3SM's variability, not how it compares with ACE.

## Results (`results.txt`; emitall `claims_paperstyle.json` 57/57)

**Checks.**
- The re-run is bit-identical to the scored runs (gates 0.0).
- E3SM truth for 2035–2049 is complete.
- No forced trend between windows.

**On the paper's two variables, G4D is below ACE-EAMv2's ratio** (median over 8 runs):

| | G4D16 | G4D36 | G4D43 | ACE-EAMv2 (paper, 10-yr) |
|---|---|---|---|---|
| precipitation | 1.50 | 1.39 | 1.47 | 1.76 |
| ≈ 10-yr equivalent | 1.87 | 1.69 | 1.82 | — |
| reflected sunlight | 1.80 | 1.67 | 1.77 | 2.56 |
| ≈ 10-yr equivalent | 2.35 | 2.14 | 2.29 | — |

**But surface pressure and geopotential height are far outside E3SM's natural variability:**

| | G4D16 | G4D36 | G4D43 | E3SM's variability |
|---|---|---|---|---|
| surface pressure, R | 23.8 | 19.3 | 19.4 | 0.52 hPa |
| (error) | 12.3 hPa | 10.0 hPa | 10.1 hPa | |
| Z3_l12 height, R | 18.0 | 14.5 | 14.4 | 4.3 m |
| (error) | 77 m | 62 m | 61 m | |
| 2 m temperature, R | 3.42 | 3.26 | 3.20 | |

**All 99 channels** (`figures/paperstyle_channels.png`, `figures/paperstyle_figures_summary.json`):
- The typical channel sits near the paper's precipitation ratio: the **median over channels is 1.68 / 1.66 / 1.74**.
- Sea-level pressure is 1.9–2.2.
- The outliers are **surface pressure** and **geopotential height**, rising toward the surface. The lowest level,
  Z3_l17, reaches 9,124–11,422. It follows the terrain, so E3SM barely varies there and the ratio explodes.
- The mean over channels (122 / 99 / 98) is therefore not a useful summary. Without Z3 levels 15–17 it is 2.8–3.2.

**Reading.**
- On the paper's own terms, G4D's precipitation and sunlight climate is at least as close to its target as
  ACE-EAMv2's was to EAMv2's. So is the typical channel.
- G4D's clear weakness is **surface pressure and the height of the lower and middle levels**. These are off by 14–24×
  at PS and Z3_l12, and by more nearer the surface.
- The maps put that error over **Antarctica, Greenland and the Tibetan Plateau, and on the polar rows**. Sea-level
  pressure, the large-scale circulation, is near the typical channel. So this is not a global circulation error.
- Two candidate causes, not yet tested:
  - our training loss gives the two polar rows exactly zero weight (code audit §4.2);
  - surface pressure over high terrain.
- This is the same weakness the five-year bias maps show for Z3_l12, now on the paper's yardstick.
- **Score A understated it.** Score A's yardstick is the spread between our emulator's own runs. For surface pressure
  that spread is about 10× E3SM's natural variability, so "score A ≈ 1–2" there does not mean close to E3SM. (To be
  confirmed against `window_truth/floor_2045_2047.npz`.)
- Surface pressure is exactly what ACE2's tendency-scaled loss weights most heavily (up to ~1,400× ours). Our loss
  weights it as little as everything else.

## Figures (`figures/`)

| file | what it shows |
|---|---|
| `paperstyle_ratios.png` | R per variable for G4D16/36/43 (dot = median of 8 runs, bar = range, open circle = ≈10-yr equivalent), with ACE-EAMv2's ratios from the paper; log scale, line at 1 |
| `paperstyle_maps.png` | one representative run's 2045–2049 error map per checkpoint, next to E3SM's own window-to-window difference, on the same colour scale per variable (precipitation, net sunlight, surface pressure, Z3_l12) |
| `paperstyle_channels.png` | median R for all 99 channels, grouped by variable and level |
| `paperstyle_figures.csv` | every plotted number |
| `paperstyle_figures_summary.json` | per checkpoint: R for every channel, plus its median and mean over channels |

## Files

- `prereg.md`: written before any number existed.
- `paperstyle.py`: gates, metric and `results.json` / `results.txt`.
- `polaris_paperstyle_all.pbs`: one debug job, 7813936. It builds E3SM truth for 2035–2043 on the CPUs while
  re-rolling the 24 runs on the GPUs, then runs the analysis. The re-roll's NetCDFs were deleted after the gates
  (`freed_nc.txt`, 38.7 GB).
- `plot_paperstyle.py`, `polaris_paperstyle_plot.pbs`: the figures. They gate R against `results.json`.
- `claims_paperstyle.json`: the emitall spec for the numbers quoted here and in chat.
- Not committed: `paperstyle_<ARM>.npz` (each run's 2045–2049 mean maps) and `truth_windows.npz`. They stay in
  `runs/makani_eval/g4d_paperstyle_2026-11-04/`.
