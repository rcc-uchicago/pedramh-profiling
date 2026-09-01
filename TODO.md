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

1. **Get a science read on the 128-node production model.** ⏳ *jesswan, not us.*
   `prod128_alldata_v2` (7566145) trained 100/100 epochs to train 0.01598 / valid 0.018297
   with the validation minimum **at the last epoch**. But it ran at **global batch 512 on the
   shipped LR** — a training-regime change made on operator instruction that the science owner
   has **not signed off**. The 101 per-channel lwrmse panels in wandb (`pedramh-profiling`) are
   the comparison vehicle against Pangu/ai-rossby; absolute losses are not cross-harness
   comparable. → `makani_bench_report.md` §4.
   *Until this lands, do not spend node-hours on a longer or wider production run.*

2. **Evaluate the trained model — inference has never been run on makani here.**
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

3. **Settle the ladder: reps, and one warmup-free wandb-off rung set.**
   Every makani number published is **n=1**, and the two ladders disagree on the headline:
   warmup-inclusive says the first Slingshot hop is free (+0.7%) and 67% efficiency at 8 nodes;
   warmup-free says +35% and **47%**. → `makani_bench_report.md` §3c.
   Run §3a's rungs at `EPOCHS=2`, wandb off, ≥3 interleaved reps per rung.
   *Cost: ~12 jobs, all ≤10 nodes in `debug-scaling` (≤1 h, 1 job/user).*
   PASS: `MAKANI_MN_SCALING_OK` + a row per rung.
   *Nothing about makani scaling should be published until this exists.*

4. **File the ALCF ticket.** Four independent findings, all with app-free reproducers, none
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

5. **cpu-bind / progress-thread sweep on the new plugin — the biggest known lever.**
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

6. **First kernel-level profile of makani** — `-v NSYS=1`, per-rank capture. makani has **no**
   profile at all (DESIGN §8 Phase 2); rank-0 step timing cannot say where the step goes, and
   P0-5 is currently reasoning from a subtraction. ⚠ An `NSYS=1` run is truncated by design
   (`exit_on_stop`) — its rows carry the captures and must never be averaged with full arms.

7. **Capture a DESIGN §4.1 equivalence baseline for makani.** No hot-path change may be
   committed without one (CLAUDE.md #6), and any lever from P0-5/P1-6 is a hot-path change.
   Check first whether makani has a usable seed knob — do **not** port `s2s/v2.0/utils/seeding.py`
   on spec; Pangu already had a stronger one and porting was the wrong call there.

8. **Revive the seven dead launchers** (`makani_bench_report.md` §9). They still open with the
   bare `module load conda` and fail instantly — **including both data packers**, so the green
   results they produced (7253465, `CONVERT_OK` 7252728) are not reproducible today. The fix is
   a two-line swap in a fixed order, already proven on real work by
   `polaris_pack_e3sm_scaling.pbs`. ⚠ Seven files inside a `git subtree` — keep the edits
   minimal and contiguous (notes §6b).

9. **Arms D and E, never run** (prereg 4 and 5, still unscored): `-v DATA=synthetic` separates
   an I/O loss from a comms loss; `-v GPU_ORDER=reverse` tests the measured reversed GPU↔NUMA
   map. *Cost: 2 jobs.*

10. **Second-user reproducibility for makani** — Pangu and SI have it, makani and physicsnemo
    do not (DESIGN §8). One run as another user with `PYTHONNOUSERSITE=1`.

## P2 — other tracks, still live

11. **ai-rossby: write up the stability sweep.** Jobs through **7578960** have all completed and
    **none of it is in the CHANGELOG** (rows are in `$MEMBER_ROOT/bench/ai_rossby_hpsweep.csv`
    and `ai_rossby_tuning.csv`; the commits are on `feat/multinode-ddp-port`). Also confirm the
    LR-5e-4 restart (**7573280**) cleared **epoch 11**, the point where LR 1.46e-3 diverged.
    The living document is not optional — an unrecorded measurement is a lost one.

12. **ai-rossby: `max_checkpoints_to_keep: 5` does not prune.** 43 epochs kept all 92 files =
    **870 GB**; at 100 epochs that is ~2 TB. Find out why before the next long run.

13. **PanguWeather: capture the §4.1 baseline, then rung 1 of the §5 ladder.** Nothing blocks
    the baseline any more (all three §4.0 prerequisites met; `tiny_baseline.yaml` runs in ~0.5 s
    of compute). `torch.compile` is measured at 1.40× on ai-rossby but **fails equivalence**
    (4.02e-01 ≫ 1e-2) — it is measured, not adopted. Plan and evidence:
    `PANGU_POLARIS_PROFILING_PLAN.md`, `polaris_bench_report.md`.

14. **Profile SI and physicsnemo on Polaris** (DESIGN §8 Phase 2). Only PanguWeather has a
    kernel-level profile. **SI is cheapest** — it already has `SI_BENCH_*`/`SI_NVTX` and a green
    Polaris bench (7252700 / 7253603); physicsnemo has no comparable harness.

15. **Fix the loader's missing `worker_init_fn`.** Would make `num_data_workers` an
    output-neutral knob worth **+9% wall throughput with 10× less jitter** (1 → 8). Today the
    worker count changes the noise realisation, so the win cannot pass the §4 gate. Ship it with
    a test pinning sample→noise independence from worker count.

16. **Stand up the test harness proper.** Three self-running test files exist (`SEEDING_OK`,
    `BENCH_INSTR_OK`, `VAE_NOISE_OK`); there is no `conftest.py` and no `--fast`.

17. **Merge the open PRs, in order.** `polaris-pbs-bringup` → **#10** `polaris-profiling` →
    **#11** `polaris-data-prep`; then `profile/pangu-polaris-profiling` → `feat/multinode-ddp-port`.
    A solo session cannot self-approve (#9). Every one of these is stacked, so merging out of
    order replays commits.

18. **E3SM data prep: 4 open converter defects + 5 decisions** (jesswan/us) —
    `polaris_data_prep_decisions.md`. The full ~1.43 TB PhysicsNeMo conversion is **not cleared
    to run**. makani's ALLDATA converter was audited clean (`MAKANI_PACK_AUDIT_OK`, 101 channels).

19. **ERA5 Globus stage** → unblocks the s2s and s2s-lightning smokes on Polaris; both scripts
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
