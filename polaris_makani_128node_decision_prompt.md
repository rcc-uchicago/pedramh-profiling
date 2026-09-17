# DECISION PROMPT — is the 128-node makani production run worth re-running?

*Written 2026-09-17, after the Slingshot finding. This is a decision for the
owner, not a recommendation I can make alone — it spends allocation and it
touches the science read. Read `makani_bench_report.md` §5k and §4, `TODO.md`
P0-3, and `polaris_nccl_metrics.md` §5d before deciding.*

---

## 0. The question

The 128-node production run (7566145, 216 node-hours) has just been shown to
have trained entirely **over TCP** — `provider=tcp`, GPUDirect RDMA off. The
fabric is now fixed and measured: 4-node makani step time went **460.5 ms → 186.4
ms**, a 2.47× speedup, with no model change.

So: **does the fabric fix make the 128-node configuration worth re-running?**

And if not — **is continuing the lagged / snapshot ensemble the better use of the
same allocation?**

My reading is **no to the re-run, yes to the ensemble**, and §1-§3 are the
argument. But §7 lists what would change my answer, and two of those are cheap to
check.

## 1. What the fabric fix is actually worth here

Measured (`polaris_nccl_metrics.md` §5d), 53-channel pack, `DATA=real STEPS=60`:

| nodes | tcp (§3b) | **cxi (new)** | speedup |
|---|---|---|---|
| 1 | 115.3 | 118.6 | — (no fabric at 1 node) |
| 2 | 490.7 | **146.9** | 3.34× |
| 4 | 460.5 | **186.4** | 2.47× |

Extrapolated to the production shape — **and this is an extrapolation, flag it as
one wherever it is quoted**:

| | 128-node as run (tcp) | 128-node, optimistic cxi |
|---|---|---|
| step_ms | 576.5 | ~233 (576.5 ÷ 2.47) |
| node-hours | 216 | **~87** |
| samples / node-hour | 20,267 | **~50,000** |

⚠ Two reasons that number is soft. The ladder above is the **53-channel** model;
production is **101-channel ALLDATA**, a different model with a different
compute-to-comms ratio. And ring/tree depth grows with node count, so a 4-node
ratio need not hold at 128. Treat ~87 node-hours as an **optimistic bound**, not
an estimate.

## 2. Why that still does not rescue the 128-node configuration

Because the fabric was never the main problem. §5k's finding was **arithmetic**:

> batch 512 across 512 ranks is **1 sample per GPU**, which starves the hardware.

Compare per node-hour, using the optimistic cxi number:

| | samples / node-hour | vs 1-node |
|---|---|---|
| 1-node production (7585080) | **229,753** | 1.0× |
| 128-node as run (tcp) | 20,267 | 11.3× worse |
| 128-node, optimistic cxi | ~50,000 | **~4.6× worse** |

⇒ **A perfect fabric would close less than half the gap.** The remaining 4.6×
is 1 sample/GPU versus 8, and no network change touches it.

And the update count does not move at all:

| | weight updates | epochs | best val loss |
|---|---|---|---|
| 128-node | 8,500 | 100 (still improving at the bound) | 0.018297 |
| 1-node | **332,424** | 243 | **0.01284** |

The usual large-batch escape — raise the learning rate to compensate for 16×
fewer updates — is **closed for this model**. The LR ceiling is **(2e-3, 3e-3] and
does not move with batch size** (§7e, 9 of 9 arms above it collapsed). Linear
scaling from batch 32 to 512 would want ~32e-3, an order of magnitude past a
limit that collapses irreversibly.

## 3. The blocker that dominates *both* options

🔴 **We cannot currently tell whether any of these checkpoints is good.**

Per `TODO.md` P0-3: `MetricsHandler` intersects its ERA5 default variable names
(`u10m, t2m, sp, sst, u500, z500, q500, q50`) with our E3SM channel names
(`PS, TREFHT, U10, RHREFHT, PSL, TMQ, T_l00…`). The intersection is **empty**, so
zero per-lead handles are built and no per-lead metric is ever computed. The
va=3/10/20 ladder returned **byte-identical** `0.012838906608521938`.

⇒ **Every validation loss in this repo — 0.01284 and the 128-node 0.018297 — is a
SINGLE-STEP number.** Neither model has ever been scored at forecast lead time,
which is the thing a weather emulator is for.

This is why `TODO.md` already lists "a longer/wider makani production run" under
**Not on this list on purpose** — *"more node-hours before a science read buys
nothing."* That rationale was stated with a now-suspect number attached (the "85%
weak-scaling efficiency", measured on TCP), but **the rationale itself survives
the fabric finding intact.** Spending 87 node-hours to produce a checkpoint we
cannot score is the same mistake as spending 216.

## 4. Option A — re-run 128-node production

**For:** a converged large-batch run would settle §5k's open question honestly —
the report is explicit that *"whether batch 512 would ever reach 0.01284 at a
learning rate it can survive is untested"*, and that the 128-node run's minimum
sat at its **last** epoch, still improving.

**Against:** ~87 node-hours (optimistic) for a model we cannot score at lead time;
~4.6× worse per node-hour than 1 node even with a perfect fabric; 16× fewer
updates with no LR headroom to compensate; and the 1-node run already produced a
better single-step loss.

**If chosen, do not run it as before.** At minimum fix the scoring first (§3), and
decide whether the question is "does batch 512 converge" (which needs *more*
epochs, not the same 100) or "how fast is 128 nodes on cxi" (which is a 1-hour
scaling job, not a production run).

## 5. Option B — continue the lagged / snapshot ensemble

**Twelve snapshot members already exist on disk** — epochs 23 through 243, every
20, from the completed 1-node run (7585080). The 128-node run produced **one**.

**For:**
* The members are **already paid for**. Building the ensemble costs inference, not
  training.
* Warm restarts (`CosineAnnealingWarmRestarts`, already supported in
  `makani/utils/driver.py:678-708`, config-only) hand us **more members for free**
  — one checkpoint per restart — on the cheap 1-node track.
* An ensemble is the direction the science is going anyway: FCN3/ACE2 are
  probabilistic models scored with ensemble metrics. A deterministic single-step
  loss is not the deliverable.
* `polaris_makani_analysis_ensemble_handoff.md` already carries the audited
  4-item scope, 9 measured silent-failure traps, and 8 retired claims.

**Against:** it does not answer the large-batch question, and a lagged ensemble of
snapshots from **one** trajectory is correlated in a way an ensemble of
independent runs is not — state that limitation in any skill claim.

## 6. Option C — the middle nobody has costed

Neither the 1-node nor the 128-node shape is obviously optimal, and the fabric
fix makes the middle newly interesting. The constraint is the LR ceiling, not the
node count:

* batch 32 at 8 samples/GPU = 4 GPUs = **1 node** (what we ran)
* batch 64 at 8 samples/GPU = **2 nodes**, wanting LR ~2.8e-3 by sqrt scaling —
  **just under the (2e-3, 3e-3] ceiling**
* batch 128 = 4 nodes, wanting ~4e-3 — **over the ceiling, expect collapse**

⇒ **2 nodes at batch 64 is the largest shape the LR ceiling plausibly allows**, and
on cxi a 2-node step is 146.9 ms against 118.6 at 1 node (+23.9%) for 2× the
samples. That is a real efficiency gain and it has **never been tested**. One
6-epoch job settles whether LR ~2.8e-3 survives at batch 64.

This is the cheapest experiment on this page and the only one that could change
the production shape.

## 7. What would change my answer

| if… | then |
|---|---|
| the lead-time scoring is fixed and the 1-node model scores **badly** at long leads | the case for a differently-shaped run reopens — but shape, not necessarily 128 nodes |
| batch 64 at LR 2.8e-3 **survives** 6 epochs (§6) | the production shape should move to 2 nodes; re-cost everything |
| a 128-node cxi **scaling** job shows better than 2.47× at scale | the ~87 node-hour bound improves; still does not fix 1 sample/GPU |
| the science owner wants a large-batch result **for its own sake** | that is a legitimate reason this analysis cannot weigh — say so explicitly rather than inferring it |

## 8. Recommendation, stated plainly

1. **Do not re-run 128-node production now.** The fabric fix halves its cost and
   leaves it ~4.6× worse per node-hour than 1 node, with 16× fewer updates and no
   LR headroom. And we cannot score the result.
2. **Fix lead-time scoring first** (`TODO.md` P0-3). It gates the value of every
   training run, including any re-run.
3. **Continue the snapshot ensemble** — the members are already on disk and the
   handoff scope is written.
4. **Run the one cheap experiment in §6** (batch 64, 2 nodes, LR ~2.8e-3, 6
   epochs). It is the only thing here that could change the production shape, and
   it costs roughly one debug-queue job.
5. If a large-batch answer is wanted for its own sake, run it as an explicit
   **science** decision with the cost stated — not as a benchmark.

## 9. Repo lines that need correcting regardless of the decision

These are wrong *now*, independent of what is decided:

* `TODO.md` "Not on this list on purpose" — *"We can already run 128 nodes at 85%
  weak-scaling efficiency"*. That figure was measured over TCP. The conclusion
  stands; the number supporting it does not.
* `TODO.md` same section — *"Switching production back to the old (faster)
  plugin — disqualified… its working regime is a message-size lottery"*. The
  lottery is **fixed by HPE's rendezvous block** (7630227 vs 7629096: `all_gather`
  swept through the 512 KB size it had wedged at). The old plugin is now the
  *recommended* one. Rewrite this entry.
* `TODO.md` — *"`NCCL_ALGO=Ring` for makani — not needed. makani reduces ~591 MB
  in one bucket, an order of magnitude below the ~1 GB tree-corruption
  threshold."* ⚠ That threshold was measured **on TCP**. The margin on cxi is
  unknown. The conclusion may well hold, but the evidence for it no longer
  applies — see `polaris_ace2_slingshot_handoff.md` §2a, task T3.
