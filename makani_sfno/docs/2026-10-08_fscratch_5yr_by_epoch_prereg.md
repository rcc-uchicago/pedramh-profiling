# Prereg: 5-yr protocol on F-scratch raw checkpoints by epoch (2026-10-08)

Written before submission. Operator, 2026-10-08: "for F-scratch, can we do the 5 year rollout on the debug
queue to see if it has the same pattern as A that it fails at random checkpoints".

## Why

A's stability by epoch is erratic (1-yr, 7721629): e83 and e143 survive, while e43/63/103/123/163/183 go
non-finite. No maturity trend. F-scratch so far:
- 1-yr: raw e21–e23 are stable at both starts; raw e41–e43 are stable in 1 of 6 runs (7720637/44, 7725618/85).
- 5-yr (7720705): raw e23 is 0/8 (non-finite at 1766–4830) and EMA e23 is 8/8.

One raw checkpoint at 5 yr cannot say whether raw F-scratch fails by epoch at random (A-like) or fails
consistently, with EMA the only thing that survives.

## The run

One `debug` job, `polaris_climate_run.pbs`, from the `b-continuation-dryair` tree. Same protocol as 7720705:
8 members, 2044 f1092 + 16i, to 2049-12-31, no flags (no dry-air fix). Raw checkpoints of
`f_nosoil_2n_b32_e23_scratch` (`ckpt_mp0_v{N-1}.tar` = epoch N, as FS23 = v22 = `ckpt_epoch` 23):

| arm | epoch | file | where in the schedule |
|---|---|---|---|
| FS20 | 20 | v19 | end of SGDR cycle 1 |
| FS21 | 21 | v20 | end of cycle 1 |
| FS22 | 22 | v21 | end of cycle 1 |
| FS41 | 41 | v40 | end of cycle 2 |
| FS42 | 42 | v41 | end of cycle 2 |
| FS43 | 43 | v42 | end of cycle 2 |

FS23 (0/8) is not re-run: the protocol is deterministic. The worst case, every member at full length, is about
42 min.

## Reading (survival = members with `CLIMATE_ROLLOUT_OK`, of 8)

Cycle 1 is FS20, FS21, FS22 plus the existing FS23. Cycle 2 is FS41, FS42 and FS43.
- **LOTTERY (A-like):** within a cycle, one checkpoint has ≥ 6/8 and another has ≤ 2/8.
- **CONSISTENT-FAIL:** every cycle-1 checkpoint has ≤ 2/8. Raw single-step F-scratch then does not hold 5 yr
  at any cycle-end epoch, and EMA is what made e23 survive.
- **CONSISTENT-PASS:** every cycle-1 checkpoint has ≥ 6/8 apart from e23.
- Anything else: mixed. Report the counts and the non-finite step ranges.

Also report, per checkpoint, the median non-finite step, cycle 1 against cycle 2.

**Prediction:** CONSISTENT-FAIL. Cycle-1 checkpoints die at 1–3.5 yr, as FS23 did. Cycle-2 checkpoints die
earlier, mostly within 1.5 yr.

Caveat: A has 5-yr data at e243 only. Its "random by epoch" pattern is a 1-yr result, so the comparison with
A is like-for-like only in kind (the same question), not in protocol length.
