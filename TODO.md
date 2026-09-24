# TODO — priority first

**The single prioritised list for this repo.** Status and evidence live in
`CHANGELOG.md`; what/why lives in `DESIGN.md`; how to work here is `CLAUDE.md`.
This file says only **what to do next and in what order**.

Rules: newest state at the top of each item; when an item is done, delete it here and
record the measurement in `CHANGELOG.md`. **PASS is the log token, never `rc`** (#14).

**Focus (2026-09-02): makani.** P0/P1 are makani; P2 is everything else, still live but
not the current push.
✅ **DONE 2026-09-04: job 7585080 completed all 243 epochs**, `Exit_status 0`, best validation
loss **0.01284**, 46.3 node-hours, 12 snapshot-ensemble checkpoints. Items 1 and 2 are closed
by it. 🔵 **NOW RUNNING: job 7591605** — C1 rollout fine-tune on `capacity`, 24 epochs,
warm-started from that checkpoint. → `makani_bench_report.md` §5k, CHANGELOG `2026-09-04`.

---

## P0 — do these first

> 📋 **makani continuation: `polaris_makani_analysis_ensemble_handoff.md`** — written
> 2026-09-04 when the training campaign closed. Covers the lead-time ladder (which decides
> whether the rollout direction is even correct), scoring C1, and the audited 4-item scope
> for building the snapshot ensemble. Includes 9 measured silent-failure traps and 8
> retired claims not to resurrect.

> 📋 **makani climate evaluation (2026-09-24): `polaris_makani_climate_protocol_handoff.md`** —
> the user decided to **"do what ACE2 does"** for jesswan's multi-year protocol (8 ICs Oct 2044,
> score 2045–2049 vs climatology). Nothing is built yet. Order: (1) reference builders
> (2045–49 time mean + monthly; 2015–44 monthly climatology + interannual std), (2) the
> **streaming cross-file driver** with on-the-fly time/monthly means and a per-channel stability
> record, equivalence-checked against `rollout_one_ic` at K=56, (3) aggregator, (4) pre-reg
> **before** the 8-member run. ⚠ Probe 7648967: the checkpoint is physical to day 36 except
> `Z3_l17`/`Z3_l16`/soil drift; the ~120-day blow-up will land in the first scored year — the
> first run is a stability measurement. Six questions for jesswan are in the handoff §5.


1. ✅ **COMPLETE 2026-09-04 — job 7585080, all 243 epochs, `Exit_status 0`**, 46 h 20 min of a
   48 h allocation. Best validation loss **0.01284** at epoch 243; **332,424 weight updates**
   for **46.3 node-hours**. Twelve snapshot-ensemble members on disk (epochs 23 through 243,
   every 20). No overfitting: training and validation descend together throughout. Maximum
   gradient norm over the whole run **0.30**. Full table: `makani_bench_report.md` §5k.
   ⚠ The comparison against the 128-node run (0.018297) is **not converged-versus-converged** —
   that run stopped while still improving. The defensible claim is cost: **11.3× more samples
   per node-hour**. Do not quote the loss ratio without that caveat.
   *Remaining: hand the per-channel lwrmse panels to the science owner.*

2. ✅ **SETTLED, AND NOW BOUNDED ABOVE — LR 2.0e-3, β₂ 0.95, warm restarts.**
   🔴 **Two hyperparameter recommendations changed on measurement (`makani_bench_report.md` §7e,
   9 arms, all collapsed):**
   - **RETIRED: β₂ 0.95 → 0.999.** Measured backwards — 5 of 5 arms at 0.999 collapsed at
     epoch 2, the fastest failure of any configuration. **Keep 0.95.**
   - **CONFIRMED: `optimizer_max_grad_norm` 32 → 1.0.** Delays collapse from epoch 2 to 6 and
     gives the best loss of its batch. Does not prevent collapse; nothing tested does.
   - **The LR ceiling is (2e-3, 3e-3] and does NOT move with batch size.** 2.0e-3 is one rung
     below a hard limit — do not raise it.
   Original sweep (commit `9506ad1f`, `makani_bench_report.md` §5h). Three arms × 3 full-pass epochs:
   4e-4 (upstream's batch-32 value) **came last**; 2e-3 won on both validation loss (0.02352)
   and grad norm (0.01830), by 12.9%. Remaining work is only to *re-test* if the run misbehaves:
   3 epochs cannot catch a tail instability, and 2e-3 was the **top of the range tested**.
   Ship it on a real schedule at the same time — `makani/utils/driver.py:678-708` already
   supports `CosineAnnealingLR` and **`CosineAnnealingWarmRestarts`** (`scheduler_T_0`,
   `scheduler_T_mult`), config-only.
   🐛 **And it fixes a live defect: warmup is impossible under the current scheduler.**
   `lr_warmup_steps > 0` with `ReduceLROnPlateau` raises `NotImplementedError` (line 702), so
   the batch-512 production run had **no warmup at all** and could not have had any.
   Warm restarts also hand us a **free snapshot ensemble** — one checkpoint per restart (item 3).

3. ✅ **CLOSED 2026-09-17 — the per-lead fix landed and six checkpoints have been scored.**
   The empty-intersection defect below is **fixed** by `652e9505` (merged here today from
   `wt-perlead-metrics`; verified job **7602739**, `PERLEAD_METRICS_OK`): `PlasimTrainer` rebuilds
   `MetricsHandler` on the dataset's own channel names and `validate_one_epoch` dumps the full
   `(leads × channels)` curve to `<expDir>/scores/metrics_epN.h5`. It passes **all 101 channels** —
   which is what makani's own `Inferencer` does (`inferencer.py:346`), so it is **not** the science
   choice this item feared. **Picking a headline subset still belongs to jesswan**, and
   all-channels leaves that open.
   **What it measured:** the rollout **blurs and drifts** — RMSE grows **4.65×** from 6 h to 126 h,
   linearly, no saturation; ACC 1.000 → 0.878. C1 is **−3.00 %** at lead 126 h; `n_future=4` is
   **−4.66 %/−4.63 %** across two seeds. ⇒ the rollout/ensemble direction is **not** misaimed.
   ⚠ **Still open:** at ACC 0.878 we are far from climatology, so **exposure bias and
   mode-averaging are not yet separable**. That needs the K=56 / 14-day sweep — see item 4.
   *Superseded text kept below for the traps it records.*

   ~~✅ **SCORING WORKS — and it produces exactly ONE single-step scalar. 2026-09-10.**~~
   The va=3 control reproduced production's **0.01284 exactly** (7598662/3/4, again in 7602599),
   so the `train_plasim.py:382` fix is verified and this item's headline is closed.
   🔴 **But the lead-time ladder came back FLAT, and the cause is not logging.** va=3/10/20 on
   the same checkpoint returned **byte-identical** `0.012838906608521938` while validation time
   scaled 17.3 → 41.9 → 74.4 s. `MetricsHandler` intersects its ERA5 default variable names
   (`u10m, t2m, sp, sst, u500, z500, q500, q50`) with the dataset's `channel_names`
   (`metric.py:269-275`) and builds a handle only if the survivors are non-empty (`:323`). Our
   E3SM channels are `PS, TREFHT, U10, RHREFHT, PSL, TMQ, T_l00…` — **empty intersection, zero
   handles, no per-lead metric ever computed.** The 2026-09-09 patch that assumed they were
   computed-and-discarded is a **no-op** (retracted in CHANGELOG `2026-09-10`).
   ⇒ **Every "validation loss" in this repo — 0.01284, and the 128-node run's 0.018297 — is a
   SINGLE-STEP number**, not the 4-step score the checkpoint-usage docs and
   `submit_rollout_scorecard.sh`'s header claim. Fix those claims.
   **NEXT, and it needs the science owner:** pass E3SM channel names to `MetricsHandler`
   (`l1_/rmse_/acc_var_names`, reachable at the construction site the fork already uses for
   `crps_var_names`). **Which of 101 channels are the headline metrics is jesswan's call** —
   do not pick eight and ship it. Until then **C1 cannot be judged** (see below).
   *Superseded text kept below for the traps it records.*

   **Score the trained model — NOTHING has ever scored a makani checkpoint on Polaris.**
   ⚠⚠ **RE-CORRECTED 2026-09-04. The 2026-09-02 correction below was WRONG, and the original
   claim was right.** `-v SKIP_TRAIN=1` never validated anything: our fork's entrypoint had
   `if params.get("skip_training"): pass` (`train_plasim.py:382`), skipping the whole run —
   restore, print timers, exit 0, no validation, no error. Fixed 2026-09-04 by calling
   `trainer.train()` (makani skips training *inside* the loop at
   `deterministic_trainer.py:362` and still validates at `:370`). Verification queued as
   7598662/3/4. **Until that va=3 arm reproduces 0.01284, treat every "we can score it" claim
   in this repo as unproven.**
   ⚠ *Superseded text, kept to show what was wrong:* "Corrected 2026-09-02: an earlier version
   of this item said 'nothing scores it'. Not true — `makani_sfno/docs/2026-08-27_prod128_alldata_checkpoint_usage.md` documents the
   restore path **and a one-command Polaris validation run**: `-v SKIP_TRAIN=1` restores the
   pinned `RUN_NUM` and runs validation only over the full 3-year split (4,380 samples,
   3-step rollout, ~10 min, 1 node, weights untouched). What is genuinely missing is **long
   rollout forecasts and scorecards**, whose tooling (`src/sfno_inference/`, `src/sfno_eval/`)
   is Stampede3-pathed, and stock makani's inference entrypoint is hard-gated off in this fork.
   **Do not write this from scratch:** `makani_sfno/scripts/` already carries the group's
   4-stage chain (`eval_inference.py` → score → `report.md` → figures, driven by
   `submit_eval.sh`, plus the `eval-sfno-own` / `eval-sfno-5410` skills). It even expects the
   same `best_ckpt_mp0.tar` name. What it does **not** match is our cluster or our data: it is
   **SLURM on Stampede3** against a **PLASIM sim52** test holdout, whereas we have PBS on
   Polaris and the **101-channel E3SM ALLDATA** contract. The job is a port — scheduler,
   test split, channel set — not a design.
   ⚠ It lives inside a `git subtree`; add a Polaris sibling rather than editing in place
   (CLAUDE.md #7's rule, applied to the scheduler axis).
   *Cost: porting, then a short single-node job.*

4. **Ensemble — the track is now unblocked and running. Next steps, in order.**
   🔵 **RUNNING/QUEUED 2026-09-17:** **7630639** = the production candidate the n_future ladder
   named (`n_future=4`, 24 epochs, 2 nodes × local 2 = global batch 16, `preemptable`, ~16
   node-hours), re-run on the cxi stack after **7621853 died over TCP** in epoch 1.
   **7630649** = `polaris_e3sm_port_test.pbs`, PASS = `E3SM_PORT_OK`.
   ✅ **Inference port changes A-D + G landed** (`f857040b`) — two were silent defects
   (`truth_sic` returned `solin`; the scorer used Gauss-Legendre on our equiangular grid,
   over-weighting the polar row **1.50×**). → `docs/2026-09-10_e3sm_inference_port_scope.md`.
   **Remaining, in order:**
   a. **Decision E — step-index vs calendar labelling.** The scope doc §2.3 recommends
      **step-index**: E3SM is packed on a **noleap** calendar with a split-cumulative day count, so
      anchoring to a proleptic-Gregorian `datetime64` drifts one day per leap year crossed. Zero
      effect on any metric; it only changes the label on output files.
   b. **Change F — the Polaris PBS sibling for the eval chain** (`scripts/submit_eval.sh` is SLURM
      on Stampede3; CLAUDE.md #7 ⇒ add a sibling, never edit in place).
   c. 🎯 **Task 10 — the K=56 / 14-day rollout sweep. THE DECISION POINT.** At 126 h ACC is 0.878,
      far from climatology, so exposure bias and mode-averaging cannot yet be told apart. This
      curve decides whether more rollout depth or a **distributional objective** deserves the
      node-hours — i.e. whether the untested CRPS arm is the next run.
   d. **The CRPS/ensemble arm has never executed.** `PlasimEnsembleTrainer` is built with 7 tests
      green and its config root-key defect is fixed; `bash polaris/submit_nfuture_ladder.sh crps 5`.
      Gated on (c) unless run as a cheap smoke.
   e. Stages 1/3/4 of the lagged ensemble (stagger-`d` start generator, member alignment by
      absolute target index, weighted combination `w_k ∝ 1/σ(k)²`) — gated on (c).
      → `docs/2026-09-10_lagged_ensemble_endtoend_plan.md` §5.

   **Snapshot members — two cheap routes, one unavailable.**
   ✅ **2026-09-17: there is nothing to keep — it was never pruned. ALL 243 epoch checkpoints
   of the 1-node production run are on disk**, contiguous `ckpt_mp0_v0…v242`, 1.65 GiB each,
   **403.2 GiB**, under `prod1n_b32_sgdr/training_checkpoints`. `makani_bench_report.md` §5k's
   "twelve members" is that report's every-20th *subset*, not what survived. ⇒ **member spacing
   is a free parameter** and the correlation-vs-skill question can be measured at inference
   cost, training nothing. ⚠ It is also **item 16's non-pruning defect on a second harness** —
   decide deliberately which members to keep before the next long run, and do not let a cleanup
   script pick. → `polaris_makani_128node_decision_prompt.md` §13.
   *Snapshot ensemble*: keep the checkpoint at each `CosineAnnealingWarmRestarts` restart —
   3-4 models from **one** run, free, and a direct payoff of item 2's scheduler change.
   *Multi-seed*: N independent runs; at ~14 node-hours each, 5 seeds ≈ 70 — affordable now,
   impossible at 216/run.
   ❌ **EMA is NOT in makani 0.2.0** (verified: no EMA/SWA/`AveragedModel` anywhere). Live
   consequence: the group's `submit_eval.sh` prefers `best_ckpt_ema_mp0.tar` *when present*,
   and for makani runs it never will be — evaluation silently falls back to raw final weights.
   ⚠ Not the same thing as makani's *ensemble parallelism* (`ensemble_trainer.py`,
   `ensemble_size`, CRPS losses + `input_noise`): that is FCN3's probabilistic objective and a
   different trainer, i.e. a change to what the model computes rather than a post-hoc ensemble.

5. **Settle the ladder: reps, and one warmup-free wandb-off rung set.**
   Every makani number published is **n=1**, and the two ladders disagree on the headline:
   warmup-inclusive says the first Slingshot hop is free (+0.7%) and 67% efficiency at 8 nodes;
   warmup-free says +35% and **47%**. → `makani_bench_report.md` §3c.
   Run §3a's rungs at `EPOCHS=2`, wandb off, ≥3 interleaved reps per rung.
   ✅ **The placement arm is DONE** (2026-09-02, `makani_bench_report.md` §5i): `GPU_ORDER=reverse`
   is **+0.88% slower at 1 node** (3+3 reps, node-matched) and −7.0% faster at 4 nodes sharded —
   **config-dependent, not a free win**; `default` is correct for the 1-node production config.
   Reps for the *ladder rungs* are still outstanding.
   *Cost: ~12 jobs + 2 arms, all ≤10 nodes in `debug-scaling` (≤1 h, 1 job/user).*
   PASS: `MAKANI_MN_SCALING_OK` + a row per rung.
   *Nothing about makani scaling should be published until this exists.*

6. **File the ALCF ticket.** Four independent findings, all with app-free reproducers, none
   reported yet — and one is a **correctness** defect, not a performance one:
   - `fi_domain` returns **ENOSYS** for every aws-ofi-nccl ≥ v1.9 on `/soft` against libfabric
     2.3.1, **and the one-line fix is `OFI_NCCL_PROGRESS_MODEL=AUTO`** (matrix 7563894 and
     7568618 — same answer on NCCL 2.28.3 and 2.27.5, i.e. provider-side, not NCCL-version
     specific). This is almost certainly why ALCF's own 2025-09 plugin rebuilds sit broken.
   - The **tree all-reduce silently corrupts above ~1 GB** — app-free, 2 AND 8 nodes
     (7569805/7569817/7571147): the head of the buffer is reduced, the tail is untouched, and
     it differs by rank. Had the watchdog not fired, training would have continued on
     half-stale gradients. `NCCL_ALGO=Ring` is the workaround.
   - `module load conda` has been **broken since 2026-08-20** (dead `gcc-native/14.2` +
     `cray-hdf5-parallel/1.14.3.5` pins; base-conda torch also has unresolved libs). Every
     script in this repo carries a hand-reconstructed modulefile because of it.
   - **Three nodes with zombie GPU state**: `x3111c0s37b1n0` (3 strikes), `x3201c0s1b1n0`,
     `x3109c0s1b0n0`.
   - ⚠ **NEW 2026-09-02 — the tree defect now has a THIRD harness and a named trigger.**
     ACE2 issues **one all_reduce of its entire 455.8 M-parameter model (1.823 GB =
     1738.86 MiB)** as **collective 14 of the run**, after DDP's parameter broadcast and
     *before the first backward* (job 7586590, flight-recorder dump). That is above the
     measured-failing threshold, so ACE2 would hit the corruption without `NCCL_ALGO=Ring`.
     It also **explains ai-rossby's byte-identical stuck collective under a 200x
     `bucket_cap_mb` change**: the collective is not a gradient bucket, so the cap cannot
     affect it. Two unrelated models, stock DDP, `gradient_as_bucket_view=True`, same
     behaviour — that is a much stronger ticket than "makani and ai-rossby saw a hang".
   *Cost: writing. Unblocks the fastest stack for everyone on the machine.*

7. **cpu-bind / progress-thread sweep on the new plugin — the biggest known lever.**
   At 8 nodes, per-GPU compute is ~115 ms of a 545 ms step ⇒ **~79% is exposed comms**
   (`makani_bench_report.md` §3b). `OFI_NCCL_PROGRESS_MODEL=AUTO` runs libfabric's own progress
   threads on the 8 cores `--cpu-bind depth -d 8` reserves — a bind tuned for the *old* plugin's
   manual progress. Vary `-d`, and settle `OMP_NUM_THREADS` in the same sweep (see the caveat
   below). If it recovers even half the gap, the new plugin dominates the old one everywhere.
   *Cost: 3-4 jobs at 4 nodes.*
   ⚠ **`omp_threads=64` on all 30 existing rows.** PBS exports `OMP_NUM_THREADS=<ncpus>` and the
   launcher's `${OMP_NUM_THREADS:-1}` idiom never overrode it, so every measurement ran at 8×
   CPU oversubscription on exactly the cores the progress engine needs. Comparability is intact
   (constant on every row); the absolute numbers are not clean. **Changing it invalidates
   comparison with all 30 rows — do it as a deliberate, documented re-baseline inside this
   sweep, not as a drive-by fix.**

## P1 — makani, next

8. ✅ **DONE 2026-09-04 — makani's first kernel-level profile** (job 7591822, all 4 ranks,
   steps 30-40, production configuration). Result: **34.9 percent of GPU compute time is spent
   in kernels that compute nothing** (`direct_copy`, `bfloat16_copy`, `nchwToNhwc`,
   `FillFunctor`) against **28.5 percent in GEMM plus FFT** — a ratio of 1.23 to 1, agreeing
   across all four ranks to within 0.4 percentage points. Same pathology as PanguWeather on the
   same A100s (47 percent in `direct_copy`+`conj`). → `makani_bench_report.md` §5m.
   **Follow-on, in order:**
   a. **NVTX phase attribution** — name *which* copies. Harness exists
      (`ACE2_retrain/nvtx_phase_attribution.py`); makani's NVTX ranges need checking against it.
   b. **The §4.1 equivalence baseline (item 9) is now the blocker**, not the profile. No layout
      or hot-path change may be adopted without it (CLAUDE.md #6).
   c. Only then a layout fix, behind that gate.
   ⚠ Not claimed: a speed-up. Kernel time (264.4 ms/step) is not wall time (472.1 ms/step
   production), and the capture is n=1 under profiler overhead.

9. **Capture a DESIGN §4.1 equivalence baseline for makani.** No hot-path change may be
   committed without one (CLAUDE.md #6), and any lever from P0-5/P1-6 is a hot-path change.
   Check first whether makani has a usable seed knob — do **not** port `s2s/v2.0/utils/seeding.py`
   on spec; Pangu already had a stronger one and porting was the wrong call there.

10. **Revive the seven dead launchers** (`makani_bench_report.md` §9). They still open with the
   bare `module load conda` and fail instantly — **including both data packers**, so the green
   results they produced (7253465, `CONVERT_OK` 7252728) are not reproducible today. The fix is
   a two-line swap in a fixed order, already proven on real work by
   `polaris_pack_e3sm_scaling.pbs`. ⚠ Seven files inside a `git subtree` — keep the edits
   minimal and contiguous (notes §6b).

11. **Arm D, never run** (prereg 4, unscored): `-v DATA=synthetic` separates an I/O loss from a
   comms loss. *Cost: 1 job.* (Arm E, `GPU_ORDER=reverse`, has moved up into P0-3 — it is a
   confound on every existing row, not just an unscored prediction. ⚠ ai-rossby queued the
   same test and **both arms were refused** — 7577036 `rc=134`, 7577166 `rc=143` — so the axis
   still has zero measurements anywhere; find out why those died before re-queueing.)

12. **Second-user reproducibility for makani** — Pangu and SI have it, makani and physicsnemo
    do not (DESIGN §8). One run as another user with `PYTHONNOUSERSITE=1`.

13. **Migrate `makani_scaling*.csv` to carry peak memory** (rule #10 — a deliberate
    header change, not a drive-by). `plasim_trainer.log_epoch` now logs
    `peak torch memory [GB]` + `non-torch memory [GB]` per epoch, but they reach only the
    job `.o` log and wandb: `parse_makani_scaling.py` asserts the header matches `FIELDS`
    and **refuses to append** on drift (`:162`, `:172-178`). Adding the two columns means
    rewriting the header of every existing CSV in one commit, plus a parser test.
    **Deliberately deferred** — not to be done while job 7585080 and the batch-48 arms are
    writing to those files. Do it once they land. → `makani_bench_report.md` §5g.

## P2 — other tracks, still live

14. **ACE2 (`fme`) on Polaris — bring-up + ladder DONE; what remains is reps and science.**
    Plan: **`polaris_ace2_multinode_handoff.md`**; prereg + scorecard:
    `ACE2_retrain/polaris/ace2_polaris_prereg.md`; evidence: CHANGELOG `2026-09-02 (cont. 3)`.
    ✅ Venv (`ACE2_VENV_OK`), config, telemetry + bench CSV (ACE2 had neither), one launcher
    for any node count, parser + 46 tests, prereg, and the **full 1/2/4/8-node weak-scaling
    ladder**. Headline results, all measured:
    - **`batch_size: 16` does not fit one node** (local 2 = 34.0 GiB, local 3 OOMs), so ACE2
      always pays the first-hop fabric toll — unlike makani, where 1 node won outright.
    - **Fabric-limited, not I/O-limited.** At 32 ranks the single-OST 2.4 TB `.nc` sustains
      **1.64 GB/s** with `gpu_busy_frac` **0.970** ⇒ **the zarr conversion is not justified.**
    - Shape: cliff at the first hop (−42% per-GPU), then saturating; **87%/82% incremental
      efficiency from the 2-node minimum viable config.**
    - 🔴 **`NCCL_ALGO=Ring` is load-bearing**, not insurance — see P0-6.
    🎯 **PRODUCTION SHAPE SETTLED (2026-09-03): 1 node, global batch 8, ~27 epochs = makani's
    332,424 updates for ~66 node-h.** Smaller batch on one node is **3.4x more
    update-efficient** than 2 nodes at batch 16 and faster in wall-clock too — the ladder's
    ranking inverts once the objective is updates rather than samples/s.
    ✅ Resume gate PASSED (`ACE2_RESUME_GATE_OK`) — warm restarts + snapshot ensemble survive
    preemption. 🔬 LR sweep RUNNING (7589850-53, 4 arms, rule pre-registered in prereg §1a).
    **Next, in order:**
    0. ✅ **LR SWEEP SCORED AND SETTLED 2026-09-10 — the winner is `3e-4`.** All four arms
       completed 3 full epochs (the two preempted ones resumed as 7598647/8): **3e-4 0.19579**
       < 1e-4 0.23332 < 5e-5 0.28380 < 1e-3 0.34571. Ranking identical to the 2-epoch ranking,
       and 3e-4 is an **interior optimum**, so the prereg's endpoint rule does not fire and it
       is adoptable as the warm-restart peak.
       🔵 **REMAINING: launch production.** 1 node, `LOCAL_BATCH=2` (global batch 8), 27 epochs,
       `PRODUCTION=1 LR=3e-4 FULL_VAL=1 T_0=9 T_MULT=1` — `T_0=9` divides 27 exactly, so the
       run ends at an LR minimum and the snapshot ensemble gets 3 members at epochs 9/18/27.
       ⚠ **Queue is an open decision:** `capacity` fits all ~93-124 h in ONE job (never resumes,
       which matters because item 0 below is unmeasured) but takes the project's single slot;
       `preemptable` caps at 72 h so the run *must* resume at least once.
    0aa. ✅ **Metric capture built 2026-09-10** (`ACE2_METRIC_LOG_OK`, 9 tests). fme routed every
       per-channel validation metric through `wandb.log` with `log_to_wandb: false`, so all of
       it was discarded — makani's defect on a second harness. Now `logging.metrics_log_dir`
       (JSONL, resume-safe) + an epoch echo to the screen log, with the dropped-key list named
       so parity is checkable. ✅ **Verified on real fme (7602614): 837 scalars to the screen
       log against 3 before, 305 drops all named and all images, `metrics.jsonl` key count an
       exact match, and `step_med_ms` 716.6 — the timed window is untouched.**
       ✅ **wandb itself verified separately (7602650, new `-v WANDB=1`, default off):** 837
       scalars + 305 wandb media PNGs = the 1142 keys the echo counted, so all three sinks
       agree exactly. Fixed a `/tmp` trap on the way (`wandb_dir_in_experiment_dir` now always
       true, else `wandb.init(dir=...)` overrides `WANDB_DIR` and the offline run dies with the
       job). 🔴 **wandb ignores `resume` when offline** — a resumed run starts a NEW wandb run,
       so wandb cannot be the record for a preempted production run and the JSONL is the one
       that survives. **Second argument for `capacity` over `preemptable` below.**
    0a. ✅ **Determinism answered (2026-09-04):** two independent 1-node jobs at `seed: 3` agree
       **bitwise** on validation loss and to **7.7e-8 (~1 ULP fp32)** on the one cross-rank
       reduced train loss that moved. ⇒ a §4.1 equivalence baseline is achievable; tolerance
       ~1e-7 on reduced scalars. ⚠ 1 node only — multi-node reduction orders are unmeasured.
    0. **Confirm resume is data-deterministic at production scale.** The gate passed on the
       load-bearing criterion (LR trace identical ⇒ `T_cur` survives preemption, so warm
       restarts and the snapshot ensemble are safe), but its loss diverged 23% because
       `sample_with_replacement` selects a bare `RandomSampler` that is not epoch-seeded.
       Production uses `DistributedSampler` + `set_epoch`, which *should* be deterministic —
       **inferred from the source, not measured.** Needs one full-epoch gate.
    0b. **Reps for the placement A/B** — `GPU_ORDER=reverse` is n=1 at both rungs and each
       delta (−0.12% at 1n, +2.80% at 4n) sits *inside* its forward baseline's spread, so
       neither is resolvable. makani's −7.0% is excluded, but ACE2's own effect is not
       measured. Needs node-matched reps, as makani's arm used.
    1. **Reps for 8n** — still **n=1** and must not be published.
       `bash ACE2_retrain/polaris/run_ace2_ladder.sh 3 2` fills the shortest rungs first;
       re-run it each time a wave drains (one job per queue is the hard limit).
    2. **Name the 1.823 GB startup collective.** The flight recorder captured no stack
       frames; one arm with stack capture would identify the call site and close
       ai-rossby's open question properly rather than by analogy.
    3. **Prereg P6 is still untested** — needs an nsys capture to check whether Midway's
       "NCCL is 40-46% of kernel time" transfers to Polaris' NVLink mesh. `ace2_nvtx.py`
       exists; there is no Polaris nsys launcher yet.
    4. **Hand the batch/LR question to jesswan.** Production at global batch 16 on 2 nodes is
       a training-regime change; the LR is flat at 1e-4 and has never been swept here.

15. **ai-rossby: write up the stability sweep.** Jobs through **7578960** have all completed and
    **none of it is in the CHANGELOG** (rows are in `$MEMBER_ROOT/bench/ai_rossby_hpsweep.csv`
    and `ai_rossby_tuning.csv`; the commits are on `feat/multinode-ddp-port`). Also confirm the
    LR-5e-4 restart (**7573280**) cleared **epoch 11**, the point where LR 1.46e-3 diverged.
    The living document is not optional — an unrecorded measurement is a lost one.

16. **Checkpoint retention does not prune — on BOTH harnesses.** ai-rossby:
    `max_checkpoints_to_keep: 5`, 43 epochs kept all 92 files = **870 GB**; at 100 epochs that
    is ~2 TB. **makani, confirmed 2026-09-17:** the 243-epoch production run kept every epoch =
    **403.2 GiB** (item 4). Two independent codebases, same symptom — worth one diagnosis, not
    two. Find out why before the next long run.

17. **PanguWeather: capture the §4.1 baseline, then rung 1 of the §5 ladder.** Nothing blocks
    the baseline any more (all three §4.0 prerequisites met; `tiny_baseline.yaml` runs in ~0.5 s
    of compute). `torch.compile` is measured at 1.40× on ai-rossby but **fails equivalence**
    (4.02e-01 ≫ 1e-2) — it is measured, not adopted. Plan and evidence:
    `PANGU_POLARIS_PROFILING_PLAN.md`, `polaris_bench_report.md`.

18. **Profile SI and physicsnemo on Polaris** (DESIGN §8 Phase 2). Only PanguWeather has a
    kernel-level profile. **SI is cheapest** — it already has `SI_BENCH_*`/`SI_NVTX` and a green
    Polaris bench (7252700 / 7253603); physicsnemo has no comparable harness.

19. **Fix the loader's missing `worker_init_fn`.** Would make `num_data_workers` an
    output-neutral knob worth **+9% wall throughput with 10× less jitter** (1 → 8). Today the
    worker count changes the noise realisation, so the win cannot pass the §4 gate. Ship it with
    a test pinning sample→noise independence from worker count.

20. **Stand up the test harness proper.** Three self-running test files exist (`SEEDING_OK`,
    `BENCH_INSTR_OK`, `VAE_NOISE_OK`); there is no `conftest.py` and no `--fast`.

21. **Merge the open PRs, in order.** `polaris-pbs-bringup` → **#10** `polaris-profiling` →
    **#11** `polaris-data-prep`; then `profile/pangu-polaris-profiling` → `feat/multinode-ddp-port`.
    A solo session cannot self-approve (#9). Every one of these is stacked, so merging out of
    order replays commits.

22. **E3SM data prep: 4 open converter defects + 5 decisions** (jesswan/us) —
    `polaris_data_prep_decisions.md`. The full ~1.43 TB PhysicsNeMo conversion is **not cleared
    to run**. makani's ALLDATA converter was audited clean (`MAKANI_PACK_AUDIT_OK`, 101 channels).

23. **ERA5 Globus stage** → unblocks the s2s and s2s-lightning smokes on Polaris; both scripts
    are written and preflight `ERA5_NOT_STAGED`.

---

### Not on this list on purpose

- **A longer/wider makani production run** — gated on P0-3. More node-hours before a science
  read buys nothing.
  ⚠ **Corrected 2026-09-17: the "85% weak-scaling efficiency" that used to support this line
  was measured over TCP** (the 128-node run never opened a CXI domain) and must not be quoted.
  The conclusion did not depend on it and is now **stronger**: at a hypothetical *zero-cost*
  fabric, 128 nodes at batch 512 would need a 62.7 ms single-node step at 1 sample/GPU, and the
  strictly cheaper 53-channel model measures 65.3 ms at that shape. Realistic gap **3.6-4.6×
  worse per node-hour**, 74-182× worse on **updates** per node-hour. → `polaris_makani_128node_decision_prompt.md`
  §10-§14. Corollary recorded there: `nodes` is already 1, so **step time at batch 32 is the
  only remaining lever** on updates/node-hour — that is P1-8 behind P1-9's equivalence gate.
- **Switching production back to the old (faster) plugin** — ⚠ **REWRITTEN 2026-09-17: the old
  plugin is now the RECOMMENDED one.** The message-size lottery that disqualified it (it wedged
  on the ALLDATA encoder weight, 7565896) is **fixed by HPE's rendezvous block** — 7630227
  against 7629096 swept `all_gather` through the 512 KB size it had wedged at. The "new" plugin
  was only ever correct-everywhere because it had quietly stopped using the fabric.
  → `makani_bench_report.md` §6, `polaris_nccl_metrics.md` §5b.
- **`NCCL_ALGO=Ring` for makani** — probably not needed: makani reduces ~591 MB in one bucket,
  an order of magnitude below the ~1 GB tree-corruption threshold. ai-rossby (4.73 GB) cannot
  run without it. ⚠ **ACE2 cannot either** (measured 2026-09-02: a single 1.823 GB full-model
  all_reduce at startup). It is on by default in `polaris_ace2_train.pbs`; removing it is a
  correctness regression, not a tuning choice.
  ⚠ **Evidence downgraded 2026-09-17: the ~1 GB threshold was measured on TCP.** The margin on
  cxi is unknown. The conclusion may well hold — do not rely on it until re-measured.
  → `polaris_ace2_slingshot_handoff.md` §2a, task T3.
