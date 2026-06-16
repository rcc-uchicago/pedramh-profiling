# Round 1 resolution

## Applied
- P1-a test-fixture break — §3.2 + Goal 2: migrate test_nc_writer.py success fixtures to descending lat, add ascending-rejection test.
- P1-b smoke clobber/mix — §4 step 0 + preflight note: require fresh empty OUT_ROOT/RUN_TAG for smoke.
- P1-c EMA-vs-raw ckpt — §4 step 0 + preflight note: pin identical CKPT for pre/post comparison.
- P2-a metadata-path fallback bug — §3.1: prefer metadata_json_path; note train_data_path parent.parent is one level too high; read coords/lat nested.
- P2-b guard-scope overclaim — §3.2 + §6 Q4: narrowed recurrence claim to eval_inference/write_rollout_nc; 5410/group writers noted as h5-sourced/correct-but-unguarded; shared-helper option deferred to Q4.
- P2-c backfill hard-coded zg500 — §3.3: use detect_z500_channel adaptively or reject non-v10 roots loudly.

## Rejected
(none — all findings verified against repo evidence)

## Contested
(none — all are test/safety/clarity refinements; no scope/experiment/path/science change)
