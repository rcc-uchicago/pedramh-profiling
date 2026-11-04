# Pre-registration: G4D scored the way Duncan et al. (2024) score ACE (2026-11-04)

Written before any of the numbers below exist. The operator asked how well our results do against the E3SM ACE
paper. Of the two options offered, the operator chose "recompute G4D the paper's way".

## The paper's metric (Duncan et al. 2024, JGR MLC, Table S5; `members/mehta5/papers/duncan2024_readout.md`)

- **Setup.** ACE (v1) is emulating EAMv2 under fixed 2010 forcing, with an annually repeating SST cycle.
- **Model error.** One continuous 10-year ACE run against EAMv2's 10-year validation period. RMSE is the
  global RMS of the time-mean difference map; bias is its global mean.
- **Natural variability ("noise floor").** EAMv2's own reference decade against its validation decade, i.e.
  the same RMSE between two windows of the target model.
- **Reported:**

  | variable | ACE RMSE | EAMv2 floor | ratio |
  |---|---|---|---|
  | precipitation | 0.37 mm/day | 0.21 | 1.76 |
  | reflected SW at TOA (RSW) | 4.17 W/m² | 1.63 | 2.56 |
  | OLR | 2.83 W/m² | 1.25 | 2.26 |

## Our version (fixed now)

**Data.**
- The archive's SST, sea ice and insolation repeat exactly every year (CHANGELOG, "no interannual SST variability
  and no SSP245 warming trend"). That is the same kind of boundary as the paper's.
- **Window: 2045–2049.** These are the five years every five-year run covers after its Oct–Dec 2044 spin-up. The
  paper uses 10 years; we have 5.

**Model.**
- G4D16, G4D36 and G4D43: the frozen-finalist candidates of the pick review (`g4d_e63_2026-11-02/pick_review_2026-11-04/`).
- 8 runs each: the protocol's starts, re-rolled with the identical code (eval-protocol @ 5848643de503), the same
  EMA weights (v15 / v35 / v42) and fp32.
- **Each run is scored on its own**, like the paper's single run. We report the median and the range over the 8.

**Per channel c, area-weighted with `score_window_protocol.area_weights`:**
- `RMSE_c` = global RMS of (run's 2045–2049 mean − E3SM's 2045–2049 mean); `bias_c` = its global mean.
  - The run's mean is the scorer's `window_mean`.
  - E3SM's is the equal-weight mean of its five annual means (`truth_series.py`).
- **Floor:** `F_c` = the same RMSE between E3SM's 2040–2044 mean and its 2045–2049 mean.
- **Floor check:** `F'_c`, the same between 2035–2039 and 2040–2044.
- **Ratio:** `R_c = RMSE_c / F_c`. About 1 means indistinguishable from E3SM's own 5-year variability.

**Comparison with the paper.**
- Precipitation: PRECT, in mm/day.
- Reflected SW: FSNTOA. Insolation is prescribed identically in the runs and in E3SM, so the RSW error equals
  minus the FSNTOA error, and the RMSE is the same.
- OLR has no channel here and is not compared.

**Secondary, model-based "10-year equivalent".** `R10 = sqrt(1 + 2·max(R² − 1, 0))`.
- Assumptions: noise variance falls as 1/years; a run's variability equals E3SM's; the bias is fixed.
- Under those, RMSE² ≈ B² + F² at 5 years and B² + F²/2 at 10 years, with B² estimated as max(RMSE² − F², 0).
- It is reported as an approximation only. The primary number is R at 5 years.

**Also reported.**
- R for PS, TREFHT, U10, Z3_l12, U_l17 and T_l17.
- The mean of R over all 99 channels. It is an analogue of the paper's "all-channel mean"; its channel sets differ
  from ours, so it is not compared numerically.

## Gates (the job stops and keeps the NetCDFs if any fails)

1. **The re-run is the scored run.** Each member's 2045–2047 window mean of the 5 map variables equals
   `maps/clim_<ARM>.npz` (relative ≤ 1e-6). Its 2045–2047 pattern RMSE equals the protocol score `e` in
   `protocol_scores/{g4d,g4d63}_channels.csv` (relative ≤ 1e-9), as in `fiveyear/extract_5yr.py`.
2. **E3SM truth.** Every year 2035–2049 has 1459 frames and the same 99 channels. The 2045–2047 years were already
   gated against `window_truth` (7809287); 2035–2043 have no reference and are built by the identical code.
3. A self-test of the metric code on synthetic fields runs first (`PAPERSTYLE_SELFTEST_OK`).

## Read-out rules (descriptive; this selects nothing and does not change the pick)

- **Per paper variable (precipitation, RSW).** Report the median R over the 8 runs for each checkpoint, next to the
  paper's ratio:
  - **below the paper** if the median R and the max R are both below it;
  - **above the paper** if the median R is above it;
  - **mixed** otherwise.
- **Floor stability.** For each reported channel, compare `F'_c` with `F_c`. If they differ by more than 25 %, that
  channel's ratio is flagged **floor-uncertain**.
- **Trend check.** Report the E3SM global-mean difference between windows for TREFHT and PS. A difference above
  half the floor says there is forced change between windows, and the floor overstates noise.
- **Caveats stated with every number.**
  - 5 years vs 10.
  - Ours is E3SM (our archive) vs the paper's EAMv2.
  - ACE v1 vs our SFNO.
  - Window 2045–2049 includes 2048–2049, already looked at for these checkpoints.

## Expectations (low confidence, written before data)

- PRECT R: between 1.0 and 1.8. G4D's 3-year score-A ratio against the run-to-run floor was 0.90–0.93, and E3SM's
  own floor should be similar or a bit smaller.
- FSNTOA R: between 1.0 and 2.5.
- PS will have the largest R of the surface channels. It is G4D's worst group in score A (r 1.63–2.01).

## Cost (debug, pre-authorized)

- E3SM truth for 2035–2043 (CPU, one process per year, ~30 min).
- Re-roll of 3 checkpoints × 8 runs (1 node, ~20 min).
- Analysis + gates (CPU, ~10 min). The re-roll's NetCDFs (~50 GB) are deleted after the gates pass, as for
  `fiveyear/`.

PASS = `PAPERSTYLE_OK` in `results.txt`.
