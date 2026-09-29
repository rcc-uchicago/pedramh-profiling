# Result — spatial parallelism on the fixed CXI stack, phase 1 (job 7669001)

Scored against `2026-09-29_spatial_cxi_prereg.md` **as written** (that file is unchanged).
Job 7669001, `debug`, 2 nodes (x3001c0s19b0n0, x3004c0s31b1n0), tree `caafdbf7`, started
2026-09-29T20:35Z. Output `makani_sfno/makani_spatial_cxi.o7669001`; per-arm logs
`$MEMBER_ROOT/runs/makani_probe/spatial_cxi/7669001/`; CSV `bench/makani_spatial_cxi.csv`.

## Readouts

Global batch 32, 8 A100-40GB, `STEPS=60 EPOCHS=2`, `step_ms` = epoch 2. Every arm
`FABRIC_CXI_CONFIRMED` (8 cxi lines) + `MAKANI_MN_SCALING_OK`.

| arm | local batch | step ms | vs h1w1 | wall s | ep-1 train / valid | ep-2 train / valid |
|---|---|---|---|---|---|---|
| h1w1 | 4 | **225.2** | — | 134 | 0.3924 / 0.1543 | 0.1112 / 0.0915 |
| h2w2 | 16 | 417.9 | +85.6 % | 132 | 0.3960 / 0.1568 | 0.1098 / 0.0896 |
| h4w1 | 16 | 283.8 | +26.0 % | 110 | 0.3797 / 0.1562 | 0.1099 / 0.0879 |
| h2w4 | 32 | 616.3 | +173.7 % | 170 | **0.4336 / 0.1719** | **0.1394 / 0.1043** |

Post-matrix unit suites: `F_FINETUNE_TESTS_OK 6/6` (`polaris/test_*.py` 69 passed — the
`test_regional_scores` fix of `caafdbf7` verified).

## Predictions

| # | prediction | outcome | verdict |
|---|---|---|---|
| P1 | h1w1 faster than TCP 2-node 627.1 ms | 225.2 ms (−64.1 %) | **held** — the fabric was what made 2 nodes slow |
| P2 | no direction; < 365.4 ms ⇒ 2 nodes beat 1 | 225.2 ms; per-GPU efficiency 365.4 / (2 × 225.2) = **81.1 %** | **2 nodes beat 1** — F/G's 2-node shape is justified |
| P3 | h2w2, h4w1 overhead ≥ +25 % | +85.6 %, +26.0 % | **held** (h4w1 by 1 point); sharding stays a memory tool |
| P4 | h2w4 hangs | trained both epochs, 616.3 ms | **FALSIFIED** — §5b's `w=4` diagnosis was wrong or transport-dependent |
| P5 | arms 1–3 CXI + scaling OK | 4/4 | **held** |

## Not gates, recorded for phase 2

- **h2w4's loss is ~26 % higher at epoch 2** while h1w1/h2w2/h4w1 agree within ~1 %. Consistent
  with §5b's divergent parameter-sync path at `w=4` now completing instead of deadlocking, but
  not shown. **No production use of `w=4`** until phase 2's equivalence check; cheapest first
  test: are the replicated (non-sharded) parameters identical across each `w`-group after N steps.
- Epoch-1 losses differ by ~4 % among the three agreeing arms, so initialisation / data order is
  layout-dependent: phase 2 must restore **one** checkpoint into each layout, not compare fresh runs.
- The prereg's "first logged loss" grep matched the `losses [{'type': 'l2', …}]` config line;
  the table above is read from the epoch summaries.
- **Bug:** h2w4 rank 7 logged `per-lead metric save failed` (h5py "truncated file"):
  `_save_per_lead_metrics` gated on `data_parallel_rank == 0` only, which every model rank of
  data group 0 satisfies. **Fixed** (model rank 0 also required): failing-first test
  `36aa3632` + fix `eb4e0cce`, verified by job 7669129 (`WRITER_C77_ALL_OK 5/5`), cherry-picked
  to this branch as `a413a645` / `8bc7543e` (identical blobs).
- **Checkpoint portability (phase 2a prerequisite, read from makani `driver.py`):** `legacy`
  restore checks the file's `comm_grid` against the live layout, so an h1w1 file cannot load into
  h2w2; `flexible` loads `mp0` and scatters it by the live model's `sharded_dims_mp`, so an
  h1w1 / A checkpoint loads into any layout (a sharded run's legacy `mp0` holds shard 0 only —
  never load that as flexible). The launcher needs a `LOAD_CKPT` → `--load_checkpoint` knob.
