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

3. **Score the trained model — *rollout* inference is what's missing, not all evaluation.**
   ⚠ **Corrected 2026-09-02:** an earlier version of this item said "nothing scores it". Not
   true — `makani_sfno/docs/2026-08-27_prod128_alldata_checkpoint_usage.md` documents the
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

4. **Ensemble for inference — two cheap routes, one unavailable.**
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
    0. **Score the LR sweep against the pre-registered rule**, then launch production with the
       winner as the warm-restart peak. ⚠ If the winner is an endpoint, extend the range —
       do not adopt. ⚠ fme has no grad-norm and no gradient clipping, so the tie-break is
       batch_loss variance, a weaker proxy.
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

16. **ai-rossby: `max_checkpoints_to_keep: 5` does not prune.** 43 epochs kept all 92 files =
    **870 GB**; at 100 epochs that is ~2 TB. Find out why before the next long run.

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

- **A longer/wider makani production run** — gated on P0-1. We can already run 128 nodes at 85%
  weak-scaling efficiency; more node-hours before a science read buys nothing.
- **Switching production back to the old (faster) plugin** — disqualified, not deprioritised:
  its working regime is a message-size lottery and it wedged on the ALLDATA encoder weight
  (7565896). → `makani_bench_report.md` §6.
- **`NCCL_ALGO=Ring` for makani** — not needed. makani reduces ~591 MB in one bucket, an order
  of magnitude below the ~1 GB tree-corruption threshold. ai-rossby (4.73 GB) cannot run without it.
  ⚠ **ACE2 cannot either** (measured 2026-09-02: a single 1.823 GB full-model all_reduce at
  startup). It is on by default in `polaris_ace2_train.pbs`; removing it is a correctness
  regression, not a tuning choice.
