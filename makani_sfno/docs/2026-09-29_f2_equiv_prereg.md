# F2 pre-registration — full-width paths unchanged on the merged tree

Written 2026-09-29, **committed before the F2 job was submitted**. The job's `stime`
(`qstat -xf`) must be later than this file's commit time.

Gate source: `polaris_makani_f_finetune_handoff.md` §4 (F2), on `worktree-monitor-ace2`;
`polaris_makani_g_spatial_handoff.md` O2.

## Why

The merge `c51a90be` (ports line + dry-air-negativity) and T4 (`ec5ccfb7`, subset-aware
stats; dry-air stats by `out_channels`) touched the shared inference/training path. Since
the last green run of each gate (`6c689c21`, job 7650461):

| file | Δ |
|---|---|
| `src/sfno_inference/climate_driver.py` | +38 / −7 (subset-aware stats) |
| `src/sfno_inference/rollout_driver.py` | +40 / −0 |
| `src/sfno_training/models/mass_fix.py` | +38 / −8 (dry-air stats by name) |
| `src/sfno_training/models/preprocessor.py` | +3 / −0 |
| `src/sfno_training/trainer/plasim_trainer.py` | +44 / −2 |

Every change is meant to be inert when no channel subset is configured. A and B are
full-width (101 / 101), so their rollouts must not move by one bit.

## Configuration (fixed, unchanged from the original gates)

| gate | script | what | reference |
|---|---|---|---|
| G1 | `polaris_climate_equiv.pbs` | `test_climate_driver.py` on CPU | its own seeded faults |
| G2 | `polaris_climate_equiv.pbs` | streaming driver vs `rollout_one_ic`, ckpt **A** `best_ckpt_mp0`, 2048 f1092, **K = 56**, chunks 40/7/1 | same tree (`docs/2026-09-24_climate_driver_g2_prereg.md`) |
| D0 | `polaris_dryair_equiv.pbs` | `climate_rollout.py`, **B e22** and **A e243**, 2044 f1092, 1460 leads, `--dry-air-fix off` | pre-fix tree `finetune-stability` @ **`873ecd37`** (verified unchanged 2026-09-29) |

One `debug` node; the two scripts run sequentially from `polaris/polaris_f2_equiv.pbs`
(`debug` holds one queued job per user).

## Tolerance — BITWISE, both gates

- G2: `CLIMATE_DRIVER_EQUIV_OK … tolerance=bitwise` — every lead `torch.equal`, as in the
  2026-09-24 prereg.
- D0: `MEMBER_NC_EQUIV_OK vars=20` for **both** pairs (every numeric NetCDF variable
  bitwise), then `DRYAIR_OFF_EQUIV_OK`. A e243 is expected to truncate at **step 595 on PS on
  both sides** (as in 7650461); the truncated files must still compare bitwise.

PASS = `F2_EQUIV_OK` (the wrapper prints it iff `CLIMATE_DRIVER_TEST_OK`,
`CLIMATE_DRIVER_EQUIV_OK` and `DRYAIR_OFF_EQUIV_OK` are all present).

## If it is not bitwise

Not a pass, and the tolerance is **not** loosened (CLAUDE.md #1, #6). Bisect the five files
above against `6c689c21`; the subset-inert claim is the thing under test. No F arm or G run
is queued on this tree until F2 is green.
