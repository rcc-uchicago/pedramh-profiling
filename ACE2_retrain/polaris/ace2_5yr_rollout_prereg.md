# Pre-registration — ACE2 5-year rollout test (2026-10-01)

Written and committed **before** the prep job (`polaris_ace2_rollout_prep.pbs`) is submitted.

## Question

Is ACE2 stable over a 5-year free rollout **on our hardware**, and does it stay stable when
driven by **our E3SM-SRM archive**? Our makani checkpoints are not: in jesswan's 5-year
protocol (job 7649597, 8 start dates Oct 2044) A went non-finite at day 79–212 in 8/8 runs,
and B in 7/8 within 2.5–5 yr, with global-mean PS falling steadily (−226 to −316 hPa by lead
6208; the survivor −269 hPa by lead 7555). ACE2's multi-year stability is so far ai2's
claim, not our measurement.

## Arms (ai2's Hugging Face checkpoints, unmodified)

| arm | checkpoint | data | starts |
|---|---|---|---|
| `era5_ai2` | `allenai/ACE2-ERA5` (`members/mehta5/ace2_era5/ace2_era5_ckpt.tar`) | our staged ERA5 (`/eagle/.../ace2/ace_training`) | 2001-10-01, 05, …, 29 00Z (8) |
| `eamv3_ai2` | `allenai/ACE2-EAMv3` (`members/mehta5/ace2_eamv3/ace2_EAMv3_ckpt.tar`) | our E3SM-SRM archive via `build_eamv3_inputs.py` | 2044 frames 1092+16i (8) = the makani protocol's starts |

**7300 steps = 5 years** each, if the prep job's measured rate fits the 1 h walltime;
otherwise whole years, flagged `TRUNCATED` in `probe.log` and reported as such.

## The EAMv3 inputs are approximate — stated in advance

Forcing maps onto the archive (SST→TS over ocean, ICE, sol_in, land mask, TOPO·g); the
**initial state** is built: 18 terrain-following levels → ACE's 8 layers (checkpoint ak/bk,
sigma assumption for our levels), total water = q(RELHUM, T) + CLDLIQ + CLDICE (no rain
water), land TS from TREFHT. Land fraction is binary. ACE2-EAMv3 was trained on historical
AMIP; 2044–2049 SSP245 SSTs are partly out of its range. The IC jolt shows in the first days;
it does not decide 5-year stability.

Row orientation is **tested, not assumed**: 8-step PS RMSE vs archive truth, south-first vs
north-first rows; the lower one wins (`ORIENTATION=` in `probe.log`). If `flip` wins, the
rollout job stops (`ERROR ORIENTATION_flip`) and the inputs are rebuilt.

## Readout (`ace2_rollout.py readout`, monthly means)

Per run: status (`FINITE` / `NONFINITE` / `NO_OUTPUT`), first non-finite month, global-mean
surface-pressure drift (last finite month − first month, hPa).

| verdict per arm | rule |
|---|---|
| **stable** | 8/8 runs finite to the end **and** median \|ΔPS\| ≤ 5 hPa |
| **survives, drifts** | all finite, median \|ΔPS\| > 5 hPa |
| **unstable** | any run non-finite (report the month) |

The 5 hPa line is ~2% of makani B's drift and well above a seasonal-cycle artefact in the
global-mean PS. `NO_OUTPUT` runs (e.g. a sick debug-scaling GPU) are reported, not imputed;
the verdict uses the runs that exist and says how many.

## What each outcome means

- `eamv3_ai2` stable → ACE2's recipe survives our E3SM forcing; the makani failure is the
  makani recipe, not our data. Strengthens the case for the fme route (jesswan's call).
- `eamv3_ai2` unstable, `era5_ai2` stable → our data/inputs (or the approximate IC, or the
  out-of-range SSP SSTs) break it; not a verdict on the recipe.
- both unstable → our stack (fme version, hardware, dtype) — investigate before any claim.

PASS tokens: `ACE2_ROLLOUT_PREP_OK` (prep), `ACE2_5YR_OK` (rollout + readout).
