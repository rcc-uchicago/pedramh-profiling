# HANDOFF — makani E3SM emulator: fine-tuning for long-rollout stability

Written 2026-09-24. **Scope:** find out whether further fine-tuning makes the emulator stable
and drift-free over multi-year rollouts, and produce the checkpoint that does it. We test three
levers: **(1) anneal the learning rate properly; (2) deepen the rollout in the loss (`n_future`
8, 16); (3) select checkpoints by rollout behaviour, not validation loss.** Out of scope:
changing the loss definition or adding a mass-conservation constraint (§7; that is jesswan's call).

Read in this order: this file → CHANGELOG `2026-09-24` entries (the climate-driver entry is the
newest) → `makani_sfno/docs/2026-09-15_nfuture_ladder_d1_and_replication.md` →
`makani_sfno/polaris/submit_nfuture_ladder.sh` header (three measured traps) → the result of the
protocol job below. Do not re-derive anything in §1.

---

## 0. Why, in one paragraph

With the streaming driver (`src/sfno_inference/climate_driver.py`, branch
`feat/makani-climate-driver`, draft PR #16), a 600-lead rollout costs about 15 s of GPU time. The
first stability read (job 7649567, n=1 each) found a clear split. Checkpoint **A**, single-step
and 243 epochs, goes non-finite at lead 595, and its channel median crosses 3σ at lead 488
(≈ day 122). Checkpoint **B** is A plus **one** epoch of `n_future=4` fine-tuning. It stays below
3σ for all 600 leads, but its **global-mean surface pressure falls 13.5 hPa**, which means it is
losing atmospheric mass. B's run trained 24 epochs, yet the checkpoint we used is epoch 1,
chosen by validation loss. Nobody has measured stability for epochs 2–24, for other depths, or
for a properly annealed schedule. Those three gaps are this task.

---

## 1. Facts you inherit (verified 2026-09-24; re-open the primary before relying on one)

| fact | value | primary |
|---|---|---|
| A | `e3sm_mn_scaling/prod1n_b32_sgdr/`, `n_future 0`, global batch 32, LR 2e-3, `CosineAnnealingWarmRestarts` T_0=20, 243 epochs; **all 244 checkpoints on disk** | `config.json`; `training_checkpoints/` |
| B | `nf4_prod_b16_r1/`, `n_future 4`, warm-started from A's `best_ckpt_mp0.tar`, global batch 16 (2 nodes × local 2), LR 4e-4, `CosineAnnealingLR` **T_max 100**, warmup 1 epoch, stopped at **24** ⇒ LR never annealed; `best` = epoch 1; only epochs 21–24 kept (`CKPT_VERSIONS=4`) | `config.json`, `warmstart_provenance.txt`, `out.log` |
| B's validation loss | 0.013411 (ep 1) → 0.013449 (2) → 0.013501 (3) → 0.013566 (12) → 0.013612 (24); train loss falls 0.02179 → 0.01770 | `nf4_prod_b16_r1/out.log` |
| **C1**, the existing depth control | `c1_rollout_full_b16/`, **`n_future 1`**, warm-started from A's `best_ckpt_mp0.tar`, global batch 16, LR 4e-4, `lr_start` 0.01, `CosineAnnealingLR` T_max 100, warmup 1, min 1e-6, β₂ 0.95, clip 32, 24 epochs: **identical to B except depth and layout** (1 node × local 4 vs B's 2 nodes × local 2). **All 24 epochs kept** | both `config.json` + `warmstart_provenance.txt`, compared 2026-09-24; 25 files in `training_checkpoints/` |
| other fine-tunes on disk | `nf1_proxy_b8_r{1,2}`, `nf3_proxy_b8_r1`, `nf4_proxy_b8_r{1,2}` (1 epoch, batch 8); `nf1_diag_b8_r2` (24 ep, batch 8, 5 ckpts); `nf4_crps_b4_r1` | `e3sm_mn_scaling/*/config.json` |
| prior depth result (skill, NOT stability) | at matched batch 8, 1 epoch at depth 4 beats 24 epochs at depth 1 by ~3.9 pp RMSE at 126 h; `n_f=4` replicates to 0.03 pp | `docs/2026-09-15_nfuture_ladder_d1_and_replication.md` §1–3 |
| stability, A vs B | A: non-finite at lead 595 (`PS`), first past 3σ `V_l00`@382, median 3σ @488. B: 0 channels past 3σ in 600; global-mean `PS` −0.7/−3.3/−6.6/−9.8/−13.5 hPa at days 30/60/92/120/150 | CHANGELOG climate-driver entry; job 7649567/7649572 |
| driver cost | 0.023–0.026 s/lead, 1.48 GiB peak, per member | `makani_climate_smoke.o7649567` |
| B epoch time | ~1230–1350 s/epoch on 2 nodes ⇒ 24 epochs ≈ 8.8 h, ≈ 18 node-hours | `nf4_prod_b16_r1/out.log` |
| memory | measured 22.29 GiB at depth 4, local 2; refit puts local 3 ≈ 34 GiB, local 4 ≈ 45 GiB (card 39.49). Nothing measured above depth 4 | `submit_nfuture_ladder.sh` header; the 09-15 doc |
| no grad accumulation | makani has none; global batch = ranks × local batch, so deeper ⇒ more nodes at fixed batch | `submit_nfuture_ladder.sh` header |
| scheduler knobs | the harness maps `SCHED`, `SCHED_T0`, `SCHED_TMULT`, `SCHED_MIN_LR`, `WARMUP_EPOCHS`, `LR_START`; **no `scheduler_T_max` knob**: T_max=100 comes from `polaris/e3sm_alldata_full.yaml:132`. The scheduler steps **once per epoch**, so warmup and T_max are in epochs | `polaris_makani_multinode_scaling.pbs:385-397` |
| protocol run | 8 Oct-2044 starts × {A, B-epoch-1} to end-2049, **submitted 2026-09-24, result pending**; outputs `$MEMBER_ROOT/runs/makani_eval/climate_protocol_<jobid>/` | `polaris/polaris_climate_run.pbs` |

Process: same as the driver handoff. Worktree from HEAD of `feat/makani-climate-driver`. No
Python on the login node (operator ruling 2026-09-24; the login node's cgroup caps pids at 256,
**counting threads**). Everything runs as PBS jobs. Use object-only git.

---

## 2. Stage 0: screen what already exists (no training; do this first)

Two open questions can be answered from checkpoints already on disk: whether depth helps
stability, and whether more epochs help or hurt. Build the screening tool, pre-register its
reading, then run it.

### 2a. Build

- `makani_sfno/polaris/polaris_climate_screen.pbs`: a sibling of `polaris_climate_run.pbs`. It
  takes `CKPTS="label=run_dir:ckpt_path ..."` and runs **one fixed rollout per checkpoint**: start
  2044 frame 1092, **1460 leads (one year, crosses into 2045 at lead 368)**, chunk 40. It runs four
  per round, one per GPU, and waits on each PID inline (never through `$(func)`; see the
  `polaris_climate_run.pbs` comment). At ~40 s of stepping each plus ~60 s of setup, about 60
  checkpoints fit in one 1-hour `debug` job.
- **A truth global-mean series** for the same valid times, so drift is measured as model minus
  truth rather than model minus lead 1. The raw model-minus-lead-1 numbers include the seasonal
  cycle; B's −1 to −3 K in `T` over Oct→Feb is uninterpretable without this. Read `PS`, `T_l17`,
  `Z3_l10`, `TREFHT` (by name, from `channel_state`) from `fields_state` for 2044 f1093 … 2045
  f1092, area-weighted with `sfno_ensemble.scores.equiangular_weights`. The files are chunked by
  frame, so read in chunks: 1460 frames × 4 channels is small. Write it once to
  `$MEMBER_ROOT/runs/makani_eval/screen_truth_2044f1092.npz`.
- `makani_sfno/polaris/climate_screen_summary.py`: one CSV row per checkpoint with the §2b
  metrics, plus a markdown table. Unit-test it on synthetic NetCDFs, as with the driver tests.

### 2b. Pre-register the reading (commit before the first screen job)

Per checkpoint, over leads 1–1460:

| metric | definition |
|---|---|
| `survived` | no non-finite value (`truncated_at_step == -1`) |
| `median_cross_3sigma` | lead at which the channel median of `anom_rms_sigma` first exceeds 3 (−1 = never) |
| `n_past_3sigma` | channels whose `anom_rms_sigma` ever exceeds 3 |
| `ps_drift_hpa@600`, `@1460` | (model − truth) global-mean `PS`, hPa |
| `t17_drift_k@1460`, `z10_drift_m@1460` | (model − truth) global mean, physical units |

**Ranking rule** (write it down before viewing any screen output): survivors first, then fewest
`n_past_3sigma`, then smallest `|ps_drift_hpa@1460|`. Report every metric, not only the rank.
State it as n=1 per checkpoint, one start date. Before trusting a small gap between two
checkpoints, re-screen the top three from a second start (2044 frame 1156, Oct 17).

### 2c. Screen list (the order is the priority)

1. **C1 epochs 1–24** and **B epochs 1, 21, 22, 23, 24**. C1 vs B is **depth 1 vs depth 4 at
   matched batch, LR, schedule and epochs**, the cleanest depth comparison that exists. C1's
   epoch series shows whether stability improves or degrades with epochs at a fixed, un-annealed
   schedule.
2. The 1-epoch batch-8 arms (`nf1_proxy`, `nf3_proxy`, `nf4_proxy` r1/r2), a depth ladder at
   1 epoch.
3. A at epochs 200, 220, 243 (snapshot spread of the base model) and `nf4_crps_b4_r1`.

**Stage-0 token:** `CLIMATE_SCREEN_OK n=<checkpoints>`. Record the table in CHANGELOG. **Stop
and report to the operator here.** Stage 1 is conditional on this table.

---

## 3. Stage 1: training arms (only if Stage 0 says depth or epochs matter)

Every arm uses the same warm start (A's `best_ckpt_mp0.tar`), global batch **16**, LR 4e-4,
`CosineAnnealingLR`, 1-epoch warmup, `LR_START=0.01`, `SCHED_MIN_LR=1e-6`, and 24 epochs. Each
varies **one** factor from B:

| arm | varies | settings |
|---|---|---|
| **T-anneal** | schedule | `n_future 4` (MULTISTEP 5), **T_max set so the LR reaches min at epoch 24** |
| **T-d8** | depth | `n_future 8` (MULTISTEP 9), annealed as T-anneal |
| **T-d16** | depth | `n_future 16` (MULTISTEP 17), annealed; **only if its memory probe fits** |

All arms use **`CKPT_VERSIONS=25`**, so every epoch is kept for screening (≈1.8 GB each, ≈43 GB
per arm; check quota first).

### 3a. Things to build or verify first

1. **A `SCHED_TMAX` knob.** Add `"scheduler_T_max": "SCHED_TMAX"` to the `_sched` map in
   `polaris_makani_multinode_scaling.pbs` (an additive line; unset means the config's value, so
   every existing caller is unchanged). **Before choosing the value, read how makani composes
   warmup with the cosine**: whether T_max counts from epoch 0 or from the end of warmup. Get this
   from `deterministic_trainer.py`'s scheduler construction, not by assumption. Confirm it on a
   2-epoch `debug` run by printing the per-epoch LR, and check that it ends at `SCHED_MIN_LR`.
2. **Memory probes** with `submit_nfuture_ladder.sh probe 9` and `probe 17`, **at local batch 1**
   (`-v LOCAL_BATCH=1`). The memory model is extrapolation above depth 4. Depth 8 at local 1 is
   expected to fit. Depth 16 at local 1 may not: the calibrated model predicts ≈46 GiB, though the
   09-15 refit came in lower at depth 4. If depth 16 does not fit, first check that makani's
   `checkpointing_level` (activation recompute) works for this SFNO. Recompute should not change
   the math; prove that with a 1-step bitwise check before relying on it. Spatial parallelism
   (`HPAR/WPAR`) needs the OFI-plugin override documented in the ladder script.
3. **Global batch 16 at local 1 means 4 nodes** (16 ranks). Depth 4 used 2 nodes × local 2.
   Batch is held fixed across arms on purpose (ladder prereg threat T3); do not let it drift with
   depth.

### 3b. Submitting

Use `submit_nfuture_ladder.sh prod <MULTISTEP>` with `-v`-style overrides (`NODES`, `LOCAL_BATCH`,
`SCHED_TMAX`, `CKPT_VERSIONS`). If the script's knobs do not reach the harness, add a **sibling**
script rather than editing it (its own header's rule). Its three traps still apply:
`LOAD_LOSS=0`, `MULTISTEP` is the only `n_future` handle, and a **new `RUN_NUM`** per arm.

**Cost (UNVERIFIED beyond T-anneal):** T-anneal ≈ B ≈ 8.8 h on 2 nodes ≈ 18 node-hours. T-d8 at
4 nodes: per-epoch time unmeasured; if cost scales with frames per sample, expect roughly (9/5) × B
per epoch, i.e. ~16 h and ~64 node-hours. The probe measures it; quote the measured number, not
this one. **Queue: `preemptable` (≤72 h) or `capacity`. Both must be surfaced to the operator
before submitting** (memory `ask-before-submitting-jobs`). Only `debug` probes are pre-authorized.

### 3c. Selection

Screen **every epoch** of every arm with the Stage-0 tool and the Stage-0 ranking rule. Validation
loss is reported beside it and never used to select. The winner goes into the protocol run
(`polaris_climate_run.pbs -v CKPT_B=...`), and only after the protocol's truth references,
aggregator and pre-registration exist.

---

## 4. Gates, in order. Key on the token, never on `rc` (CLAUDE.md #14)

| # | gate | green = |
|---|---|---|
| S0a | screen tool + summary tests | `CLIMATE_SCREEN_TEST_OK`; seeded faults shown red (a truncated member ranks last; drift sign; truth alignment off by one lead is caught) |
| S0b | ranking pre-registered | commit time earlier than the first screen job's `stime` |
| S0c | screen of §2c | `CLIMATE_SCREEN_OK`; CHANGELOG table; **operator decision** on Stage 1 |
| S1a | `SCHED_TMAX` verified | 2-epoch run prints the LR per epoch and ends at the minimum |
| S1b | memory probes | measured peak GiB and s/epoch at depths 8 and 16, recorded |
| S1c | arms trained | each arm's `out.log` shows 24/24 epochs; all 25 checkpoints present |
| S1d | arms screened | per-epoch screen table; the winner named by the pre-registered rule |

---

## 5. Do NOT

- ❌ Select a checkpoint by validation loss. It chose B epoch 1, and nothing says epoch 1 is the
  most stable.
- ❌ Compare arms with different global batch, LR or warm start and attribute the difference to
  depth.
- ❌ Change the loss, add a `PS`/mass constraint, or drop channels in these arms. Each changes what
  the model computes, and each needs jesswan (§7).
- ❌ Edit `submit_nfuture_ladder.sh`, `submit_c1_rollout_finetune.sh`, or any Midway script; add
  siblings instead. The harness gets **only** the additive `SCHED_TMAX` line.
- ❌ Use `strict_restore: false` to get past a loss-state size mismatch; use `LOAD_LOSS=0` (ladder
  trap 1).
- ❌ Submit to `preemptable`/`capacity` without the operator, or resubmit a job stuck on
  `queue_tags` (CLAUDE.md #12).
- ❌ Read the screen's `T`/`Z` drift without the truth series; the seasonal cycle is in the raw
  numbers.
- ❌ Run Python, torch or h5py on the login node.

## 6. Open for the operator (do not block on these; state the default)

1. **Queue for Stage 1.** *Default:* `preemptable`, one arm at a time, after asking.
2. **Warm start** from A (comparable to B and C1) or from B epoch 24 (cheaper, but confounded).
   *Default:* A.
3. **If Stage 0 shows depth does not help stability,** Stage 1 is skipped and the question goes to
   the mass/loss route (§7). *Default:* stop and report.

## 7. Not in scope, but raise it (jesswan)

B's steady `PS` loss suggests the fix may be a **conservation constraint** rather than more
training. ACE2 applies one: `conserve_dry_air: true` in its corrector
(`ACE2_retrain/config_polaris.yaml:170`, `fme/core/corrector/atmosphere.py`). It changes what
the model computes, so it needs jesswan's sign-off. At most, prototype it as a labelled
diagnostic arm, never as a default.

## 8. Definition of done

Stage 0 green, with the screen table in CHANGELOG and the operator's Stage-1 decision recorded.
If Stage 1 runs: every arm trained, every epoch screened, the winning checkpoint named by the
pre-registered rule, and its path handed to the protocol run. Commit and push small on a feature
branch, and open a draft PR against the branch it was cut from.
