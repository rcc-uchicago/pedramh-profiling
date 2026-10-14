# Re-rolling the bf16-trained candidates in fp32, and re-ranking (pre-registration, 2026-10-13)

Written before any of these rollouts. Thresholds do not move after results are seen; changes go in dated
amendments at the end. Parents: `docs/2026-10-09_pipeline_comparison_prereg.md` (S set, window, floor, gates,
§5 decision rule) and `docs/2026-10-13_precision_cross_prereg.md`.

## Why

The precision cross came out as outcome 3, "both contribute" (`runs/makani_eval/precision_cross_2026-10-13/results.txt`):
- rolling out in fp32 instead of bf16 lowers score A by 1.03 ± 0.07 for bf16-trained weights;
- for fp32-trained weights the drop is 2.16 ± 0.01;
- fp32-trained weights blew up at 2 of 4 epochs when rolled out in bf16.

Its decision for outcome 3 is that the ranking is redone in fp32. Every bf16-trained candidate in the results
write-up was scored from bf16 rollouts, so its score understates what the same weights do in fp32. The
incumbent pick, G1-dry EMA epoch 39 (write-up Result 14), is already fp32/fp32.

## Arms (17), all rolled with `--amp off`

| arm | weights | run, checkpoint | extra flags | bf16 original (score tag) |
|---|---|---|---|---|
| FSEF23 … FSEF32 | EMA | `f_nosoil_2n_b32_e23_scratch` `ckpt_mp0_v22` … `v31` | — | FSE23 … FSE32 (`g1fs`) |
| FDEF28, FDEF32 | EMA | `fsF_anneal_nf4_b16_rtmax100_scratch` `ckpt_mp0_v27`, `v31` | — | FDE28, FDE32 (`fd1`) |
| BF22, BF24 | raw | `nf4_prod_b16_r1` `ckpt_mp0_e22_stable`, `e24_stable` | — | B22, B24 (`ref1`) |
| BF22fix, BF24fix | raw | same | `--dry-air-fix=on` | B22dryair, B24dryair (`ref2`) |
| DRYAIRF24 | raw | `fs_anneal_dryair_nf4_b16_rb01` `ckpt_mp0_v23` | — (constraint trained in) | DRYAIR24 (`ref1`) |

Checkpoints and flags are those of the original jobs (`makani_climate_run.o<job>` "weights/ckpt" lines). The
F-scratch block covers every epoch from 23 to 32: the cycle-2 epochs whose bf16 rollouts were 8/8 finite, plus 23
(FSEMA). Contiguous coverage, not the bf16 top-k, because the cross showed that bf16 rank does not predict fp32 rank:
- FSE15 improved from 3.62 to 1.90 when rolled in fp32;
- FSE19 improved only from 3.25 to 3.04.

## Ranking rule (the parent §5, applied to the fp32 pool)

- **Pool:** every candidate rolled out in fp32 on the S set:
  - G1 EMA e15–23 (`g1fs`) and G1 EMA e15/16/19 with the prediction-time fix (`g1fix`);
  - G1-dry EMA e15–23, 39–43 and 59–63 (`g1dry`, `g1dry_c`, `g1dry_d`);
  - FSEF15–19 (`pxc`);
  - the 17 arms above.
- **Gates 1–3,** read from the budget read-out:
  1. all 8 members ran to 2049-12-31;
  2. global TMQ > 0 and ≤ 1% negative area in the last month;
  3. |dPS| ≤ 1 hPa.
- **Rank** gate passers by A on 2045–2047. This is the parent §5 rule, with no special status for the incumbent:
  - the lowest-A passer wins outright if its A is lower than every other passer's by more than 3.3 × the paired
    member SE;
  - otherwise every passer within 3.3 SE of it ties, and the tie is broken by fewer drifting groups (gate-4
    census), then by lower W.
- **Reported:** whether the winner differs from the current pick, G1-dry EMA e39.
- **Also reported, as context:** for each re-rolled arm, the paired bf16 → fp32 change in A. Members count only if
  healthy in both cells, as in the cross prereg.
- **Gate 4** is reported. Whether any candidate passes it is not expected to change.

**No prediction is made about the winner.**

## Execution

- **Code:** `eval/pipeline-protocol` @ `5848643d` adds `polaris/polaris_rollout_batch.pbs`, a generic one- or two-node batch
  wrapper that hands off to the last finisher for scoring like the cross wrapper, plus `polaris/fp32_rerank.py`
  and their tests.
- **Jobs:**
  - debug, 1 node: FSEF23–28;
  - debug-scaling, 2 nodes: node 1 runs FSEF29–32 and FDEF28/32, node 2 runs BF22, BF24, BF22fix, BF24fix and
    DRYAIRF24.
- **Scoring:** the last batch to finish scores all three run directories (`TAG=f32a`, stride-1 E3SM reference).
  It writes `runs/makani_eval/fp32_reroll_2026-10-13/results.txt`, ending in `FP32_RERANK_OK`.
- **k:** 17 new candidate tuples.
