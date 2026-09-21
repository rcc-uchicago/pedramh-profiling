# HANDOFF — the makani accuracy track, from 2026-09-18

*Paste this into a new session to continue. It is a **discussion** handoff: the
decision is already taken (§1) and two jobs are already in flight (§2). What a new
session has to do is read their results against §3's rule, not re-open §1.*

*Read first: `CHANGELOG.md` top entry (2026-09-18), then
`polaris_makani_128node_decision_prompt.md` **§14-§15** — §1-§9 of that file are
superseded and say so.*

---

## 0. One paragraph of context

makani-SFNO is trained and in production on Polaris (1-node run 7585080: 243 epochs,
332,424 updates, val 0.01284). Bring-up is done; the 128-node question is closed
(don't reopen — §6). The open work is **forecast skill**, and as of 2026-09-18 it has
a decision, a gate, and two running jobs.

## 1. The decision, taken 2026-09-18 — do not re-litigate

**The accuracy lever is ROLLOUT DEPTH.** Measured, on this model, at lead 126 h:

| arm | cost | RMSE @ 126 h |
|---|---|---|
| depth-1 rollout fine-tune, **24 epochs** (C1) | 24 ep | −3.00 % (73/101 channels improved) |
| **`n_future=4`, ONE epoch** | 1 ep | **−4.66 % / −4.63 %** — two seeds, 0.03 pp apart |
| batch 8 vs batch 16, depth 1 (D1) | 24 ep | −0.72 % ⇒ **batch 8 costs 2.3 pp** |

⇒ **One epoch at depth 4 beats twenty-four at depth 1 by ~3.9 pp**, i.e. ~24× the
skill per epoch. Nothing else measured on this model is close. **No fabric,
node-count or batch-shape result touches accuracy at all** — that is why the NCCL
work is parked in §6.

Corollary from D1: **do not shrink the batch to buy depth.** And makani has **no
gradient accumulation**, so `global batch = ranks × LOCAL_BATCH` — depth is bought
with *memory*, which is why the arms below are multi-node without being a throughput
claim.

## 2. What is in flight, and exactly what each one answers

| job | shape | reads out as | answers |
|---|---|---|---|
| **7630639** | depth-4 × 24 ep, 2 nodes × 4 GPU × local 2 = **global batch 16**, warm start from `prod1n_b32_sgdr/best_ckpt_mp0.tar`, fresh optimizer (`LOAD_*=0`), LR 4e-4 `CosineAnnealingLR` → 1e-6, 1 warmup epoch, `MULTISTEP=5`, `preemptable`, ~16 node-hours | `MAKANI_MN_SCALING_OK` + a row in `$MEMBER_ROOT/bench/makani_nfuture_ladder.csv`; per-lead curve at `<expDir>/scores/metrics_epN.h5` | **Does depth-4 COMPOUND over 24 epochs, or was the one-epoch −4.66 % a one-shot?** |
| **7633207** | K=56 (14 days), 24 ICs, 1 node, `preemptable` 2 h | `EVAL_INFERENCE_OK` + NetCDFs under `$MEMBER_ROOT/runs/makani_eval/prod1n_b32_sgdr_K56/inference` | **Task 10 — the decision point.** See §3. |

expDir for 7630639: `$MEMBER_ROOT/runs/makani_mn_scaling/e3sm_mn_scaling/nf4_prod_b16_r1`.
Its predecessor `nf4_prod_b16_r1_tcpfail_7621853` is the **dead tcp run kept on purpose** —
do not delete, do not resume from it.

⚠ Both are `preemptable`. makani auto-resumes (model + optimizer + scheduler), so a
preemption costs wall-clock, not progress. **Do not resubmit a queued job** —
CLAUDE.md #12; a long `eligible_time` means the queue has no nodes, and resubmitting
destroys it.

## 3. The rule for reading 7633207 — write the answer down before you look

> ✅ **DONE 2026-09-20 (job 7639537).** The rule below was superseded by a numeric
> pre-registration — `makani_sfno/docs/2026-09-20_k56_readout_prereg.md`, written
> before scoring — and the curve has been read. **Result: mode-averaging is ruled
> out** (predicted anomaly amplitude at 336 h is 1.019 × truth's; blurring would
> need ≤ 0.60), so **the CRPS arm is not indicated and stays parked**. K=56 did
> *not* reach saturation either (median NRMSE 0.970 of the 1.414 decorrelation
> level, decelerating), so `n_future=8` is not selected by the evidence either.
> The verdict that fired is `DRIFT_FIRST`, on **one** channel — `Z3_l17`, a
> near-constant near-surface level into which the model injects 43× its natural
> variance. Full numbers: that doc's RESULT section and the 2026-09-20 CHANGELOG
> entry. Rows below are kept as the record of what was decided in advance.

Today's curve stops at 126 h, where RMSE has grown **×4.65 from 6 h, linearly, with
no saturation**, and ACC is **0.878** — nowhere near climatology. At that horizon
**exposure bias and mode-averaging are not separable**, and they point at opposite
fixes. K=56 separates them:

| if the 14-day curve shows… | the dominant failure is | so the next arm is |
|---|---|---|
| RMSE **saturating**, ACC decaying toward 0 | **mode-averaging / blurring** — MSE training produces the conditional mean | the **CRPS arm** (`PlasimEnsembleTrainer`, built, 7 tests green, **never run**) ⚠ changes the loss ⇒ **jesswan's sign-off**, CLAUDE.md division of labor |
| RMSE **still climbing linearly**, ACC still well above 0 | **exposure bias** — the model still hasn't seen enough of its own output | **`n_future=8`** ⚠ memory audit FIRST: depth 4 at global 16 already measures **22.29 GiB/GPU at local 2** (local 3 extrapolates to ~34 of 39.49), so depth 8 likely needs 4 nodes at local 1 |

Raise the CRPS question with jesswan **now, in parallel**, so sign-off isn't the
blocker if the curve points that way.

## 4. The next engineering, and it does not depend on the curve

Tasks 11-13 of `makani_sfno/docs/2026-09-10_lagged_ensemble_endtoend_plan.md:286-289`
are `to do` and **sequentially dependent**. Both branches of §3 need them, so they are
the productive work while the jobs run:

| task | stage | note |
|---|---|---|
| 11 | stagger-`d` start generator | `nwp_ic_offsets` deliberately does the opposite — read §1 of the plan before writing it |
| 12 | member alignment by **absolute target index** `s+k` → table `(target, depth)` | |
| 13 | weighted combination, `w_k ∝ 1/σ(k)²`, plus weighted spread | largely shared with a snapshot ensemble |
| 14 | 5-D tensor → `MetricsHandler` with CRPS / spread / SSR / RH | **this one's metric choice depends on §3** |

⚠ **That plan's task table is STALE in two places** — tasks 4-9 are done and verified
(`f857040b` + change F, `E3SM_PORT_OK` 7630654, `EVAL_INFERENCE_OK` 7632679), and task
15's Gauss-Legendre defect was fixed as **change G**. Fix the table when you touch it.

⚠ **Two different ensembles are in play and the docs blur them.** The *plan's* lagged
ensemble has members from **staggered start times** valid at a common target. The
**243 on-disk snapshots** (`ckpt_mp0_v0 … v242`, contiguous, 403.2 GiB) are a
**checkpoint** ensemble — a different construction, from the 128-node prompt §13. The
members are free either way; **the combination machinery does not exist for either.**
Saying "the ensemble is just inference" is wrong until tasks 11-13 land — that error
was made in this session and corrected.

## 5. Two port gaps found on 2026-09-18 — the pattern will repeat

Both were invisible on the Stampede3/SLURM path by construction, and both were fixed
in **our** sibling rather than in shared code:

1. **Holdout glob** (`65f5e405`): the literal `MOST.*.h5` is PLaSim's packer naming;
   the E3SM holdout is `2048.h5` / `2049.h5`. Default is now `*.h5`, with
   `--test-file-glob` to narrow.
2. **`AssertionError: torch.distributed is unavailable. Check pytorch build ...`** —
   **the message is misleading.** torch.distributed is fine. makani's `comm.init`
   reaches physicsnemo's `create_process_subgroup`, which raises whenever
   `manager.distributed` is False (`physicsnemo_sfno/physicsnemo/distributed/manager.py:638`);
   it wants a real process group, **world size 1 included**. `DistributedManager`
   initialises from **ENV, SLURM or OPENMPI** — under PBS there is nothing to read, so
   it logs `Assuming this is a single process job` and everything after asserts. Fixed
   by launching under `python -m torch.distributed.run --standalone --nproc_per_node=1`.
   🔴 **Do NOT fix this class of thing in `src/sfno_inference/`** — that tree is a
   subtree shared with the Stampede3 `eval-sfno-own` path, where the SLURM initialiser
   already works. CLAUDE.md #7.

⇒ **Expect more of these, and read them as port gaps, not as science failures.**

## 6. Parked on purpose — do not spend the session here

* **128-node re-run.** Closed. It cannot beat one node even with a **zero-cost**
  fabric, from fabric-free measurements (decision prompt §10). 3.6-4.6× worse per
  node-hour, 74-182× on updates per node-hour.
* **`NCCL_ALGO` / Tree-vs-Ring.** Correctness question answered (no corruption on cxi
  at 2 **or** 8 nodes). The performance question is worth **~1.7 % of step time at
  ACE2's production shape, against ±3.8 % rep spread** — below its own noise floor, and
  unpinning is a hot-path change needing an equivalence baseline that does not exist.
  ⚠ If you do quote those numbers: **`busbw` is rank-count normalised**
  (`busbw = algbw × 2(N−1)/N`, ×1.75 at 8 ranks, ×1.9375 at 32), so it is **not
  comparable across node counts** — Ring reads +4.8 % from 2n→8n and is actually
  −5.3 % in `algbw`. And `-c 1` checks **one** collective per (size, placement), not
  the ten timed iterations.
* **Batch-shape experiments.** Withdrawn (decision prompt §12): batch 48 already lost
  to batch 32, and the LR ceiling is **(2e-3, 3e-3] and does not move with batch size**
  (9 of 9 arms above it collapsed; `optimizer_max_grad_norm: 1.0` delays collapse from
  epoch 2 to epoch **6**, so a 6-epoch pass proves nothing).
* **Step-time optimisation** (P1-8: **34.9 %** of GPU time in kernels that compute
  nothing — `direct_copy`, `bfloat16_copy`, `nchwToNhwc`, `FillFunctor` — vs 28.5 % in
  GEMM+FFT). Real, and the only speed item with an accuracy consequence (more epochs
  per node-hour), but it sits behind P1-9's equivalence gate and behind everything
  above. n=1 under profiler overhead; kernel time ≠ wall time.

## 7. Open, small, worth confirming rather than assuming

* **100 vs 101 channels.** The loader reports the pack as `180 × 360 × **100**` while
  the contract is quoted elsewhere as 101. The wrapper loads and rolls out, and the
  port tests compare `inp_chans`/`out_chans` against the run's own config, so this is
  almost certainly on-disk-vs-model bookkeeping. **Confirm it when the full sweep
  lands** — a channel-count error would silently misalign a scorecard.
* **403 GiB held by one run.** `max_checkpoints_to_keep` does **not** prune (TODO.md
  item 16, recorded for ai-rossby, live on makani too). Check it **before** the next
  long run and decide deliberately which members to keep — do not let a cleanup script
  choose, the 243 are the ensemble's raw material.
* **Reps.** Every cxi row in the repo is **n=1**.

## 8. Where things are

```
$MEMBER_ROOT = /eagle/projects/lighthouse-uchicago/members/mehta5
  runs/makani_mn_scaling/e3sm_mn_scaling/prod1n_b32_sgdr/   # the 1-node production run
    training_checkpoints/ckpt_mp0_v0 … v242                 # 243 members, 403.2 GiB
    training_checkpoints/best_ckpt_mp0.tar                  # what both live jobs start from
  runs/makani_mn_scaling/e3sm_mn_scaling/nf4_prod_b16_r1/   # 7630639
  runs/makani_eval/prod1n_b32_sgdr_K56/inference/           # 7633207
  bench/makani_nfuture_ladder.csv                           # the depth ladder
  data/e3sm_makani_alldata_production/{train,test}          # test = 2048.h5, 2049.h5
```

Submitting (from `makani_sfno/`, never a login node):

```bash
qsub -v SMOKE=1 polaris/polaris_eval_inference.pbs                       # ~1 min, debug
qsub -q preemptable -l walltime=02:00:00 polaris/polaris_eval_inference.pbs   # K=56, 24 ICs
```

Evidence lives in `makani_bench_report.md` (read §0 first — `step_ms` is the *final
epoch's* mean, and the fabric stack is part of the config but is not a CSV column),
`polaris_makani_128node_decision_prompt.md` §10-§15, and the 2026-09-18 CHANGELOG
entries.

## 9. The first three things to do in the new session

1. `qstat -u $USER` **once**, then read `makani_sfno/makani_eval_inference.o7633207`
   and 7630639's `.o` — key on the PASS token, not `rc` (CLAUDE.md #14).
2. Score 7633207's curve against §3's table and **write the branch down** before
   starting the arm it selects.
3. If both jobs are still queued: start **tasks 11-13** (§4). They are needed either
   way and need no allocation.
