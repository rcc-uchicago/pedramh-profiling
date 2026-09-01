# TODO — priority first

**The single prioritised list for this repo.** Status and evidence live in
`CHANGELOG.md`; what/why lives in `DESIGN.md`; how to work here is `CLAUDE.md`.
This file says only **what to do next and in what order**.

Rules: newest state at the top of each item; when an item is done, delete it here and
record the measurement in `CHANGELOG.md`. **PASS is the log token, never `rc`** (#14).

**Focus (2026-09-01): makani.** P0/P1 are makani; P2 is everything else, still live but
not the current push. Nothing queued or running as of 2026-09-01.

---

## P0 — do these first

1. **Retrain at the right batch, on one node.** The 128-node run (7566145) is a completed
   training run, not a usable recipe: batch 512 bought only **8,500 weight updates** in 100
   epochs, and its validation minimum sitting at the last epoch says undertrained, not
   converged. Batch 32 on **1 node** gives **136,800 updates in ~16.2 h for ~14 node-hours**
   — 16× the updates for 1/15th the cost of the run already spent (`makani_bench_report.md`
   §5c-d).
   Order: finish the placement reps (in flight) → lock the config → one 100-epoch reference
   run → the LR sweep (item 2).
   ⚠ Needs the **`capacity`** queue: 1-4 nodes, ≤168 h, so it fits unchained — but it is
   `max_run 1` per *project*, so taking it blocks other `lighthouse-uchicago` members for the
   duration. Coordinate before submitting.
   *Note: batch/LR/scheduler changes are training-regime changes; keep the science owner
   informed and hand her the per-channel lwrmse panels — but this does not gate the work.*

2. **Optimize the LR and the schedule — now affordable at full length.**
   The two references disagree by 5-10× at batch 32: upstream FCN3 pretrain2 uses **4e-4**,
   while scaling our shipped (batch 8, 1e-3) gives 2e-3 (sqrt) to 4e-3 (linear). Not derivable
   — measurable. At ~14 node-hours a run, a **3-arm sweep at the full 100 epochs** costs ~42,
   less than the single 128-node run already spent. Full length matters: ai-rossby's short
   sweep would have passed a config that diverged at **epoch 11**.
   Ship it on a real schedule at the same time — `makani/utils/driver.py:678-708` already
   supports `CosineAnnealingLR` and **`CosineAnnealingWarmRestarts`** (`scheduler_T_0`,
   `scheduler_T_mult`), config-only.
   🐛 **And it fixes a live defect: warmup is impossible under the current scheduler.**
   `lr_warmup_steps > 0` with `ReduceLROnPlateau` raises `NotImplementedError` (line 702), so
   the batch-512 production run had **no warmup at all** and could not have had any.
   Warm restarts also hand us a **free snapshot ensemble** — one checkpoint per restart (item 3).

3. **Evaluate the trained model — inference has never been run on makani here.**
   `best_ckpt_mp0.tar` (= epoch 100) exists and nothing scores it. This is the last unchecked
   box of DESIGN §8 Phase 1 for this model, and it is the difference between "a training run
   completed" and "an evaluatable model".
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
   **Carry `-v GPU_ORDER=reverse` as a paired arm at 1 and 4 nodes** — all 30 existing rows
   (production included) ran `default`, which on Polaris' **reversed** GPU↔NUMA map
   (`dev0`→NUMA 3 … `dev3`→NUMA 0, job 7531456) combines with the mandatory
   `--cpu-bind depth -d 8` to place every rank on the NUMA node *farthest* from its own GPU.
   It changes placement, not per-rank arithmetic, so it needs no equivalence gate.
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

8. **First kernel-level profile of makani** — `-v NSYS=1`, per-rank capture. makani has **no**
   profile at all (DESIGN §8 Phase 2); rank-0 step timing cannot say where the step goes, and
   P0-5 is currently reasoning from a subtraction. ⚠ An `NSYS=1` run is truncated by design
   (`exit_on_stop`) — its rows carry the captures and must never be averaged with full arms.

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

## P2 — other tracks, still live

13. **ai-rossby: write up the stability sweep.** Jobs through **7578960** have all completed and
    **none of it is in the CHANGELOG** (rows are in `$MEMBER_ROOT/bench/ai_rossby_hpsweep.csv`
    and `ai_rossby_tuning.csv`; the commits are on `feat/multinode-ddp-port`). Also confirm the
    LR-5e-4 restart (**7573280**) cleared **epoch 11**, the point where LR 1.46e-3 diverged.
    The living document is not optional — an unrecorded measurement is a lost one.

14. **ai-rossby: `max_checkpoints_to_keep: 5` does not prune.** 43 epochs kept all 92 files =
    **870 GB**; at 100 epochs that is ~2 TB. Find out why before the next long run.

15. **PanguWeather: capture the §4.1 baseline, then rung 1 of the §5 ladder.** Nothing blocks
    the baseline any more (all three §4.0 prerequisites met; `tiny_baseline.yaml` runs in ~0.5 s
    of compute). `torch.compile` is measured at 1.40× on ai-rossby but **fails equivalence**
    (4.02e-01 ≫ 1e-2) — it is measured, not adopted. Plan and evidence:
    `PANGU_POLARIS_PROFILING_PLAN.md`, `polaris_bench_report.md`.

16. **Profile SI and physicsnemo on Polaris** (DESIGN §8 Phase 2). Only PanguWeather has a
    kernel-level profile. **SI is cheapest** — it already has `SI_BENCH_*`/`SI_NVTX` and a green
    Polaris bench (7252700 / 7253603); physicsnemo has no comparable harness.

17. **Fix the loader's missing `worker_init_fn`.** Would make `num_data_workers` an
    output-neutral knob worth **+9% wall throughput with 10× less jitter** (1 → 8). Today the
    worker count changes the noise realisation, so the win cannot pass the §4 gate. Ship it with
    a test pinning sample→noise independence from worker count.

18. **Stand up the test harness proper.** Three self-running test files exist (`SEEDING_OK`,
    `BENCH_INSTR_OK`, `VAE_NOISE_OK`); there is no `conftest.py` and no `--fast`.

19. **Merge the open PRs, in order.** `polaris-pbs-bringup` → **#10** `polaris-profiling` →
    **#11** `polaris-data-prep`; then `profile/pangu-polaris-profiling` → `feat/multinode-ddp-port`.
    A solo session cannot self-approve (#9). Every one of these is stacked, so merging out of
    order replays commits.

20. **E3SM data prep: 4 open converter defects + 5 decisions** (jesswan/us) —
    `polaris_data_prep_decisions.md`. The full ~1.43 TB PhysicsNeMo conversion is **not cleared
    to run**. makani's ALLDATA converter was audited clean (`MAKANI_PACK_AUDIT_OK`, 101 channels).

21. **ERA5 Globus stage** → unblocks the s2s and s2s-lightning smokes on Polaris; both scripts
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
