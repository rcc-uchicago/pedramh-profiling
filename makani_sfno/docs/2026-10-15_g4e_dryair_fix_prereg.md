# Depth-4 G1 with the inference dry-air fix (pre-registration, 2026-10-15)

Written before any of these rollouts. Thresholds do not move after results are seen; changes go in dated
amendments at the end. Parents: `docs/2026-10-09_pipeline_comparison_prereg.md` (S set, 2045–2047 window, floor,
gates, §5 decision rule) and `docs/2026-10-13_fp32_reroll_prereg.md` (the fp32 pool). Operator 2026-10-15: "yes
do 1" (roll G4E15–23 with `--dry-air-fix on`).

## Why

The n_future comparison (`runs/makani_eval/nfuture_necessity_2026-10-15/`: its own prereg, `results.txt`) found:
- depth-4 G1 (G4E) against single-step G1 (G1E), e15–23, same code, recipe, precision and batch:
  - healthy members 72/72 vs 41/72;
  - paired ΔA −0.481 ± 0.006;
- every G4E beats the current pick G1D39 on A, by 0.23–0.37;
- but **all nine fail gate 3 and only gate 3**: |dPS| median 2.9–5.2 hPa, all mass loss.

The inference dry-air fix passed gate 3 at small A cost on both depths:
- G1E15/19: +0.09/+0.12;
- B22/B24: +0.07/+0.02;
- FD24: 0/8 → 8/8.

It is not free, though: G1E16fix failed gate 2.

## Arms (9), fp32 rollouts with `--dry-air-fix on`

| arm | weights | checkpoint | flags |
|---|---|---|---|
| G4E15fix … G4E23fix | EMA | `f_nosoil_4n_b32_e23_scratch_fp32_nf4` `ckpt_mp0_v14` … `v22` | `--dry-air-fix on` |

- **Precision:** the run's own (`--amp config` → `amp_mode none` → fp32).
- **Code:** `eval/pipeline-protocol` @ `5848643d`, the same tree as the `g4e` rollouts. Versus `5dd5f4d`, which
  rolled the G1E fix arms, it only adds `--amp bf16`; the dry-air fix and `--amp config` paths are unchanged.
- **Jobs:** `polaris_climate_run.pbs`:
  - debug: G4E19fix–23fix;
  - debug-scaling, 1 node: G4E15fix–18fix.
- **Score:** `polaris_protocol_score.pbs` `TAG=g4fix`, with `TGM=` the stride-1 truth, as for `g4e` and `f32a`.

## Decision rule (the parent §5, unchanged)

- **Pool:** the fp32 pool of the 10-13 re-rank plus `g4e` and `g4fix`:
  - `g1fs` G1E*;
  - `g1fix`;
  - `g1dry`, `g1dry_c`, `g1dry_d`;
  - `pxc` FSEF*;
  - `f32a`.
- **Gates 1–3:**
  1. 8/8 to 2049-12-31;
  2. global TMQ > 0 and ≤ 1 % negative area in the last month;
  3. |dPS| ≤ 1 hPa.
- **Rank** passers by A on 2045–2047:
  - the lowest-A passer wins outright if every other passer is worse by more than 3.3 paired member SE;
  - otherwise it is a tie → fewer drifting groups → lower W.
- **Reported:** whether the winner replaces G1D39.

## Also reported (descriptive, does not change the decision)

- **Fix cost on depth-4 weights:** paired ΔA (G4Efix − G4E) per epoch and pooled over e15–23, over members
  healthy in both. Pooling is per member over epochs, then mean ± SE over members.
- **Fix cost on single-step weights,** for comparison: G1Efix − G1E at e15/16/19, the only fix arms.
- **Per arm:** gates failed, healthy count, median |dPS|, drifting groups.

No prediction is made about the winner. k = 9 new candidate tuples.
