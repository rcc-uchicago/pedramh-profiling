# Pre-registration — spatial parallelism on the fixed CXI stack, phase 1 (2026-09-29)

Written before submission. Task #7 of the 2026-09-29 session; operator: *"makani supports it
hardware spatial parallelism … we also fixed slingshot and nccl"*, then *"when do we do #7"*.

## Why, in one paragraph

Every sharded (`HPAR`/`WPAR` > 1) result in `makani_bench_report.md` §5 predates the fabric
fix: the old v1.6.0 plugin before HPE's rendezvous block (h2w2 IMA/hang, 7554253 / 7563723),
or v1.21.1, which fell back to **TCP** (no `FI_MR_PROV_KEY`, 7629082). The current launcher
default — v1.6.0 + `NCCL_PROTO=Simple` + rendezvous block, `provider=cxi` asserted by the
fabric watchdog — has **never run a sharded shape**, and has never run 2-node pure DDP at
batch 32 either. The second is the number F (7660250, 2 nodes) and G's node choice need.

## What the prior evidence already says (do not re-derive)

- `w=4` hangs were diagnosed from flight-recorder dumps as **application-level divergence**
  (§5b, 7580127): ranks with `rank % 4 == 3` take a different path through parameter sync
  (`all_reduce_barrier` on `default_pg`) while their `w`-group peers wait in a 283 MB
  spectral-weight broadcast. *"Not a transport failure … not the plugin."*
- The **sharding overhead** at equal nodes, GPUs, global batch and sample-equivalents per GPU
  was **+40.8 %** (4 n), **+43.6 %** (8 n) and **+80.8 % at 1 node** (§5c). The 1-node figure
  involves no inter-node fabric at all.
- ⇒ The fabric fix is **not expected to make sharding pay**. Its use here is memory
  (configurations that cannot fit < 1 sample per GPU otherwise), not speed.

## Design — one `debug` job, 2 nodes (8 A100-40GB), global batch 32

Knobs identical to the TCP-era matrix (`submit_when_slot_frees.sh` COMMON): `STEPS=60,
EPOCHS=2` (`step_ms` = final epoch, warmup-free), `e3sm_alldata_full.yaml` (A's 101-channel
config), the production pack; `EVAL_SAMPLES=32` (≥ the global batch, so validation is not
empty; `step_ms` excludes validation). Flight recorder on. **No fabric pins** — the launcher
default is the fixed stack. Every arm: 8 GPUs, **4 sample-equivalents per GPU**.

| order | arm | data groups | `LOCAL_BATCH` | spatial group | crosses nodes? |
|---|---|---|---|---|---|
| 1 | h1w1 (pure DDP) | 8 | 4 | — | all-reduce only |
| 2 | h2w2 | 2 | 16 | 4 (intra-node) | all-reduce only |
| 3 | h4w1 | 2 | 16 | 4 (intra-node) | all-reduce only |
| 4 | h2w4 | 1 | 32 | 8 (**both nodes**) | spatial too |

h2w4 runs last: a hang there must not cost the other three. Each arm has a hard timeout; a
timed-out arm ends the matrix (the GPUs are no longer trusted).

## Predictions and decision rules (reference numbers: `makani_bench_report.md` §5c/§5d)

| # | readout | prediction | falsified by / decision |
|---|---|---|---|
| P1 | h1w1 2-node step vs **TCP 2-node 627.1 ms** (same knobs) | faster on CXI | ≥ 627.1 ms ⇒ the fabric was not what made 2 nodes slow |
| P2 | h1w1 vs the **1-node benchmark 365.4 ms** (same knobs, 7580338) | no directional prediction | **< 365.4 ms ⇒ 2 nodes beat 1 node on wall clock** at batch 32 and the 2-node shape of F/G is justified; per-GPU efficiency = 365.4 / (2 × step) |
| P3 | h2w2, h4w1 sharding overhead vs h1w1 | **≥ +25 %** (1-node +80.8 % had no fabric in it) | ≤ +10 % ⇒ sharding is nearly free on this stack; revisit §5c |
| P4 | h2w4 | **hangs** in setup or first step (§5b, application-level) | trains ⇒ §5b's diagnosis was wrong or transport-dependent |
| P5 | arms 1–3 | `FABRIC_CXI_CONFIRMED` + `MAKANI_MN_SCALING_OK` | any failure ⇒ read the flight recorder before anything else |

Not a gate, recorded for phase 2: the first logged training loss of each arm, side by side.
Equal to ~1e-3 would suggest initialisation is layout-independent; it is **not** an
equivalence check (see below).

## Out of scope for phase 1 — required before any production use of a sharded shape

1. **Equivalence (DESIGN §4).** Same checkpoint, validation loss under each layout; tolerance
   stated in a commit before that job. Needs checkpoint portability across layouts (our
   checkpoints are `legacy`, per model-parallel rank) — to be read in makani, not assumed.
2. **Memory, where sharding actually helps.** At a fixed global batch and GPU count, a spatial
   split redistributes activations but does not shrink them per GPU (h2w2 on 16 GPUs at batch
   16 is still 1 sample-equivalent per GPU, the load that OOMed T-d16 in 7650263). The T-d16
   probe therefore needs **8 nodes × h2w2 at batch 16** (0.5 per GPU) — `debug-scaling`,
   which the ACE2 LR sweep currently holds (operator's call on interleaving).

## PASS

`SPATIAL_CXI_ARM_OK` per arm, `SPATIAL_CXI_ARM_HANG` / `_FAILED` otherwise, and one
`SPATIAL_CXI_MATRIX_DONE <ok>/<run>` line with the step table. Rows go to
`bench/makani_spatial_cxi.csv` — never the TCP-era `makani_spatial.csv`.
