# G2 pre-registration — streaming climate driver vs `rollout_one_ic`

Written 2026-09-24, **committed before the G2 job was submitted**. The job's
`stime` (`qstat -xf`) must be later than this file's commit time.

Gate source: `polaris_makani_streaming_driver_handoff.md` §4 (G2).

## Configuration (fixed)

| item | value |
|---|---|
| checkpoint | **A** `$MEMBER_ROOT/runs/makani_mn_scaling/e3sm_mn_scaling/prod1n_b32_sgdr/training_checkpoints/best_ckpt_mp0.tar` |
| run dir / stats | `…/prod1n_b32_sgdr/` (`config.json`, run-dir `global_means/stds.npy`) |
| data | `$MEMBER_ROOT/data/e3sm_makani_alldata_production/test/` (year **2048**) |
| IC | frame **1092** (Oct 1 00:00, noleap) of 2048 |
| horizon | **K = 56** leads |
| chunk lengths | **40, 7, 1** |
| device / precision | one A100, bf16 autocast as in the run config (`amp_mode: bf16`), same process, same wrapper instance for both paths |
| script | `makani_sfno/scripts/climate_driver_equiv.py` (`TOLERANCE = "bitwise"`) |

## Pass rule

`CLIMATE_DRIVER_EQUIV_OK` is printed **iff both**:

1. for every chunk length, **each of the 56 leads**' physical-unit prediction
   (`pred_z · out_scale + out_bias`, the same ops `rollout_one_ic` applies) is
   **bitwise equal** (`torch.equal`) to `rollout_one_ic`'s;
2. chunk lengths 40, 7 and 1 are bitwise equal to each other.

**Expected: bitwise.** Same ops in the same order, same autocast context, same
wrapper and preprocessor instance, and forcing built by the same three operations
(`_read_forcing` → normalise → `as_tensor` + `grid_converter`) as the dataset's
`get_sample_at_index`.

## If it is not bitwise

Not a pass, and the tolerance is **not** loosened (CLAUDE.md #1, #6). The script
prints the first mismatching lead, the max abs / rel error and where
(lead, channel, lat, lon). A control runs `rollout_one_ic` twice:

- control **not** bitwise → GPU nondeterminism in the model itself; the driver
  cannot be judged bitwise on this hardware. Record it; any further criterion is a
  new, separately committed pre-registration, written before re-running.
- control bitwise but the driver differs → a driver bug. Locate it before G3.

## Blind spots (what G2 cannot see)

G2 runs inside one file, so it cannot see cross-file bugs (covered by unit test
§5.7 and G3's hand-off provenance), and it never touches the reductions (unit tests
§5.4–§5.6). Passing G2 does not "prove the driver".
