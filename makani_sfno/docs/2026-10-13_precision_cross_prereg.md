# 2 × 2 precision cross: is the G1-vs-F-scratch gap from training or rollout precision? (pre-registration, 2026-10-13)

Written before any rollout of the two new cells. Thresholds do not move after results are seen; changes go in
dated amendments at the end. Parent: `docs/2026-10-09_pipeline_comparison_prereg.md` (same S set, window,
floor, scorer).

## Question

At EMA epochs 15–19, G1 (`f_nosoil_2n_b32_e23_scratch_fp32`) scores lower on protocol score A than F-scratch
(`f_nosoil_2n_b32_e23_scratch`) by 1.46–2.12. Each gap is a paired difference over the 8 S starts, 8/8 members
better (results write-up `makani_results/2026-10-13_makani_results.md`, Result 8). The two runs differ in
training precision: bf16 autocast vs `amp_mode none`. Their rollouts also differ in precision:
- F-scratch members logged `feedback_dtype=torch.bfloat16`;
- G1 members logged `feedback_dtype=torch.float32`.

This is because `climate_rollout.py --amp config` follows each run's `amp_mode`. The cross separates the two.

## Design

Training precision T × rollout precision R. EMA weights at epochs 15, 16, 17, 18 and 19 (`ckpt_mp0_v14`–`v18`,
`WEIGHTS=ema`). The S set is the 8 October-2044 starts, rolled to 2049-12-31, scored on 2045–2047 with the
fixed floor.

| cell | T (training) | R (rollout) | arms | source |
|---|---|---|---|---|
| bf16/bf16 | F-scratch (bf16) | bf16 (`--amp config`) | FSE15–FSE19 | existing, job 7732901 |
| fp32/fp32 | G1 (fp32) | fp32 (`--amp config`, amp_mode none) | G1E15–G1E19 | existing, jobs 7734961 (e15–18) and 7734960 (e19) |
| bf16/fp32 | F-scratch (bf16) | **fp32 (`--amp off`)** | **FSEF15–FSEF19** | new |
| fp32/bf16 | G1 (fp32) | **bf16 (`--amp bf16`, new option)** | **G1EB15–G1EB19** | new |

- **"fp32 rollout"** means autocast off, with TF32 matmuls at PyTorch's defaults. This is the path G1E took
  through `amp_mode none` (both set `amp_enabled=False`, `amp_dtype=float32`).
- **"bf16 rollout"** means bf16 autocast, so the carried state is bf16. This is the path FSE took through
  `amp_mode bf16`.
- The new `--amp bf16` sets exactly `amp_enabled=True` and `amp_dtype=torch.bfloat16`. That is the pair
  `checkpoint_loader.load_eval_params` derives from `amp_mode: bf16`.

## Gate (before any G1EB number is used)

**`AMP_BF16_EQUIV_OK`:** arm FSEQ15 is F-scratch EMA epoch 15 rolled with `--amp bf16` in the same job as
G1EB. It must reproduce `climate_protocol_7732901/member_FSE15<i>.nc` `global_mean` **bitwise over the first 120
steps for all 8 members**.
- On failure the G1EB cell is void and nothing from it is reported.
- Full-length bitwise equality is reported but does not gate.

## Estimands

Notation: A_{T,R}(e, m) is the per-member score. Each channel's r_c is averaged within the 24 groups, then over
groups. This is the existing scorer's per-member value, read from `<tag>_channels.csv`. Effects are paired by
start index m and epoch e:

- R|T=bf16 = A(bf16, fp32) − A(bf16, bf16): the rollout-precision effect on the bf16-trained weights.
- R|T=fp32 = A(fp32, fp32) − A(fp32, bf16): the rollout-precision effect on the fp32-trained weights.
- T|R=bf16 = A(fp32, bf16) − A(bf16, bf16): the training-precision effect under bf16 rollouts.
- T|R=fp32 = A(fp32, fp32) − A(bf16, fp32): the training-precision effect under fp32 rollouts.
- Interaction = R|T=fp32 − R|T=bf16, pooled over the primary epochs using members healthy in all four cells.
- Total = A(fp32, fp32) − A(bf16, bf16) = R|T=bf16 + T|R=fp32 = T|R=bf16 + R|T=fp32.

**Inclusion.** A (member, epoch) enters an effect only if the member is healthy in both of its cells. Healthy
means, from the scorer's budget read-out:
- the rollout ran to 2049-12-31 (n = 7667 − 16 i steps);
- global-mean TMQ stayed > 0;
- |dPS| ≤ 100 hPa.

The scorer keeps truncated members with very large but finite A (about 1e14), so a finiteness test alone is not
enough. Excluded members are counted per cell and reported.

**Pooled estimate (primary):** epochs 15, 16, 18 and 19, the epochs at which the existing bf16/bf16 cell is 8/8
healthy (the same four epochs as Result 8). F-scratch EMA epoch 17 is 4/8 healthy with A ≈ 27, a
learning-rate-cycle collapse checkpoint (Result 5). Pooling it would mix survival with accuracy. Epoch 17 is
still rolled out in both new cells and reported per epoch. This choice was fixed before any new-cell rollout
existed, after a dry run of the analysis on the two existing cells showed epoch 17 dominating the pooled mean.

For each member, average the per-epoch differences over the primary epochs available to it. Then report the
mean ± SE over members, with a paired t-test. df = members − 1. Per-epoch values (all five epochs), healthy
counts and gates 2–3 per cell are reported too.

## Decision rule

"Significant" means |pooled effect| > 3.3 × SE (the house threshold). Negative means fp32 is better.

1. **Rollout precision explains the gap:** both R effects are significant and negative, and neither T effect is
   significant.
2. **Training precision explains the gap:** both T effects are significant and negative, and neither R effect
   is significant.
3. **Both contribute:** at least one R effect and at least one T effect are significant and negative.
4. **Unresolved:** anything else.

In every case we also report:
- shares of the total, R|T=bf16 / Total and T|R=fp32 / Total;
- whether |Interaction| > 3.3 SE.

**Prediction (recorded, not a gate):** outcome 1. The write-up hypothesised that bf16 rounding of the carried
state accumulates over about 7,600 steps.

## What each outcome decides

1. **Rollouts must run in fp32 (`--amp off`).** Every bf16-rolled number in the pool (FSE, FDE, FSEMA, B22,
   B24, DRYAIR24, FD) has to be re-rolled in fp32 before any cross-lineage ranking. Training precision can then
   be chosen on cost.
2. **New training runs use fp32.** bf16-trained lineages stay behind whatever the rollout precision.
3. **Both changes apply,** and the ranking is redone in fp32/fp32.

## Execution

- **Code:** branch `eval/pipeline-protocol` @ `6645c247` (`5bbbf921` + `6645c247` on `5dd5f4d1`, the sha of the two
  existing cells) adds `--amp bf16`, a `TGM` override in `polaris_protocol_score.pbs`
  (default unchanged), the wrapper `polaris/polaris_precision_cross.pbs` and
  `polaris/precision_cross_analysis.py`, with tests `TEST_PRECISION_CROSS_OK` and `TEST_PXC_CHAIN_OK` (both
  passed on a login node). Dry run of the read-out on the two existing cells: Total = −1.852 ± 0.055 (8 members).
- **Jobs:** two 1-node jobs, debug (FSEQ15 + G1EB15–19, 48 members) and debug-scaling (FSEF15–19, 40 members),
  about 40 min each.
- **Scoring:** whichever job finishes second scores both new run directories (`TAG=pxc`). The drift census uses
  the stride-1 E3SM annual-mean file. It then runs the analysis into
  `runs/makani_eval/precision_cross_2026-10-13/results.txt`.
- **k:** 10 new candidate tuples, plus the gate arm.

## Amendment 0 (2026-10-13, before any new-cell result): location of the write-up

No design change. The results write-up cited under **Question** as `makani_results/2026-10-13_makani_results.md`
is committed on this branch at `makani_sfno/docs/2026-10-13_makani_results/2026-10-13_makani_results.md`, with
its figures and regenerating scripts alongside it.
