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

> 📌 **Read §10-§14 (2026-09-17) before acting on §6-§8.** The continued analysis
> firms the "no" into an argument from fabric-free measurements, **withdraws §6**,
> and finds Option B holds **243** members rather than 12. The decision in front
> of the owner is now narrower: it is only §14 items 2 and 3, and both are
> already in `TODO.md`.

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

> ✅ **Firmed up 2026-09-17 → §10.** A second route using only *fabric-free*
> measurements lands at ~229 ms against this table's ~233, so **~87 node-hours is
> a floor**, not merely optimistic. The ÷2.47 is still the wrong *operation*
> (§10a) — do not reuse it elsewhere as if it had a mechanism.

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

> ✅ **Strengthened 2026-09-17 → §10.** "Perfect fabric" can now be taken
> literally: at a **zero-cost** fabric the 128-node shape would need a 62.7 ms
> single-node step at 1 sample/GPU, and the strictly *cheaper* 53-channel model
> measures 65.3 ms at that shape. Two independent routes bracket the gap at
> **3.6-4.6× worse per node-hour**. The fabric cannot reverse the ranking, so a
> 128-node cxi scaling job is not a decision input.

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

> 🔴 **TRUE OF THIS BRANCH ONLY — corrected 2026-09-17 → §15.** Scoring was fixed
> and verified on **`wt-perlead-metrics`** (commit `652e9505`, job 7602739), and
> six checkpoints have since been scored at 21 leads × 101 channels. The rollout
> **blurs and drifts**; C1 is **−3.00%** at lead 126 h and `n_future=4` is
> **−4.66%**. The conclusions in §2-§4 about *cost* are unaffected — none of them
> depend on the loss being lead-resolved — but "we cannot score it" is no longer
> the blocker, and the ensemble track is much further along than §5 knows.

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

> ✅ **Undercounted — corrected 2026-09-17 → §13.** **All 243** epoch checkpoints
> are on disk (contiguous `v0…v242`, 403.2 GiB). The twelve are `makani_bench_report.md`
> §5k's chosen every-20th subset, not what survived. Member spacing is therefore a
> free parameter, and the correlation objection below becomes measurable.

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

> 🔴 **WITHDRAWN 2026-09-17 → §12.** Three of this section's claims are wrong on
> measurements already in the repo: weak-scaling efficiency below 100% *is* a
> node-hour penalty (2 nodes is **19% worse**, not a gain); the exact shape is
> already a row (**7580449**, 2n × 8 samples/GPU × batch 64); and batch 48 was
> tested against batch 32 and lost (§5l). Kept below unedited — the reasoning is
> what §12 has to answer.

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

> ⚠ **Two of these four rows are retired — read the revised table at the end of
> §14.** Kept here as written.

| if… | then |
|---|---|
| the lead-time scoring is fixed and the 1-node model scores **badly** at long leads | the case for a differently-shaped run reopens — but shape, not necessarily 128 nodes |
| batch 64 at LR 2.8e-3 **survives** 6 epochs (§6) | the production shape should move to 2 nodes; re-cost everything |
| a 128-node cxi **scaling** job shows better than 2.47× at scale | the ~87 node-hour bound improves; still does not fix 1 sample/GPU |
| the science owner wants a large-batch result **for its own sake** | that is a legitimate reason this analysis cannot weigh — say so explicitly rather than inferring it |

## 8. Recommendation, stated plainly

> ⚠ **Item 4 is withdrawn and items 1/3 are strengthened — the operative version
> is §14.** Kept here as written.

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

✅ **All three applied to `TODO.md` 2026-09-17.**

---

# CONTINUED ANALYSIS — 2026-09-17

*Written against the repo's own rows rather than against §1-§9's summaries. Three
things change: the fabric question stops being a judgement call (§10), §6 is
withdrawn (§12), and Option B turns out to be better resourced than §5 knew
(§13). The §8 recommendation survives, on firmer evidence.*

## 10. The fabric question is decidable now — no 128-node job needed

§1 divides the production step by 2.47 and flags the result as soft. The division
is the wrong operation — a step is `compute + exposed comms`, and a fabric fix
touches only the second term — but there is a second route to the same place that
uses **only fabric-free measurements**, and at one node there is no fabric, so no
provider can be blamed or credited.

Ask what the 128-node shape would need in order to beat one node. Its shape is
**1 sample per GPU**, so with a hypothetical **zero-cost fabric** its throughput
is exactly the single-node, 1-sample-per-GPU step time:

```
samples / node-hour  =  4 GPUs × 3600 s ÷ step_s
to beat 229,753      ⇒  step  <  62.7 ms
```

The anchor we have is the **53-channel** model at exactly that shape, warmup-free,
on one node: **65.3 ms** (7564492, §3c). The 101-channel ALLDATA model runs the
same trunk with more input and output channels, so it cannot be faster at the same
shape — 65.3 ms is a floor for it, and it is already above the 62.7 ms required.

Using §4a's ~3.5× compute ratio between the two contracts, the realistic figure is
**~229 ms → ~63,000 samples/node-hour.** Two independent routes now bracket it:

| route | 128-node cxi step | samples/node-hour | vs 1 node |
|---|---|---|---|
| §1, 576.5 ÷ 2.47 | ~233 ms | ~50,000 | 4.6× worse |
| §10, 3.5 × 65.3 ms fabric-free | ~229 ms | ~63,000 | **3.6× worse** |
| *1-node production, measured* | *472.1 ms @ 8/GPU* | ***229,753*** | *1.0×* |

They agree to 2% on the step. §1's **~87 node-hours is a floor**, not just an
optimistic bound — 216 ÷ (576.5/229) = 85.8.

⇒ **No fabric result can reverse the ranking**, because the quantity that decides
it is measured at one node, where there is no fabric. A 128-node cxi scaling job
is still worth running to characterise the machine; it is **not** worth running to
inform this decision, and §7's third row should be struck as a decision input.

### 10a. Where the additive model breaks — and why 2-node TCP rows are unusable

Stated so the ÷2.47 arithmetic is not reused elsewhere as if it had a mechanism.
The repo's rows refuse a clean `compute + comms` decomposition: on TCP the 2-node
rung is **slower than the 4-node one** (490.7 vs 460.5 ms, §3b), and the two
2-node ALLDATA reps at global batch 32 read **627.1 and 2386.9 ms** — a 3.8×
spread (7580324 / 7580949, `makani_occupancy.csv`). DDP overlaps the all-reduce
with the backward pass, so the two terms are not additive, and the TCP 2-node rung
is not a measurement of anything. ÷2.47 lands close only because at 128 nodes on
TCP the comms term dominated (~79% of the 8-node step was exposed comms, §3b), so
scaling the whole step ≈ scaling the comms term.

## 11. The objective, restated — and it has exactly one lever left

Everything above is samples per node-hour. The quantity that actually bought the
1-node run its result is **optimizer updates** per node-hour:

| | updates | node-hours | **updates / node-hour** | vs 1 node |
|---|---|---|---|---|
| 1-node (7585080) | 332,424 | 46.3 | **7,180** | 1.0× |
| 128-node (7566145), as run | 8,500 | 216 | **39.4** | **182× worse** |
| 128-node at the full measured 2.47× fabric gain | 8,500 | ~87 | ~97 | 74× worse |

And the identity that closes the shape question:

```
updates / node-hour  =  3600 ÷ (step_s × nodes)
```

`nodes` is already at its minimum of 1. **Step time at batch 32 on one node is the
only lever left** — every other axis on this page is a reshuffle. Its target is
already measured and already in the plan: `TODO.md` P1-8 found **34.9% of GPU
compute time in kernels that compute nothing** (`direct_copy`, `bfloat16_copy`,
`nchwToNhwc`, `FillFunctor`) against 28.5% in GEMM+FFT, at the exact production
configuration. A 20% step-time cut is 20% more updates per node-hour on **every**
future run, compounding across the rest of the campaign in a way no shape change
does — and it is gated on the §4.1 equivalence baseline (P1-9), per CLAUDE.md #6.

⚠ Not a claim of a speed-up: P1-8 is n=1 under profiler overhead, and kernel time
(264.4 ms) is not wall time (472.1 ms).

## 12. §6 is withdrawn — the arithmetic runs the other way

§6 called 2 nodes at batch 64 "the cheapest experiment on this page and the only
one that could change the production shape". Four measured objections, in order of
how decisive they are.

**1. Weak-scaling efficiency below 100% *is* the node-hour penalty.** §6 read
"+23.9% step time for 2× the samples" as an efficiency gain. It is 2× the samples
on **2× the nodes**. The same §5d cxi ladder, restated per node:

| nodes | step_ms | samples/s | **samples/s per node** | weak eff |
|---|---|---|---|---|
| 1 | 118.6 | 33.7 | **33.7** | 100% |
| 2 | 146.9 | 54.5 | **27.2** | 80.7% |
| 4 | 186.4 | 85.8 | **21.5** | 63.6% |

2 nodes is **19% worse per node-hour**, and the 80.7% in the table *is* that
penalty. More nodes is never cheaper per node-hour when efficiency is under 100%.

**2. The shape is not unrun — it is a row.** Job **7580449** is 2 nodes, 8
samples/GPU, global batch 64, pure DDP, ALLDATA: **1018.6 ms** against the 1-node
batch-32 baseline's **365.4 ms** (3 reps, 0.2% spread). That row is TCP, so per
§10a it is not usable as a number — but the configuration was already tried. On
cxi the honest estimate is `365.4 + ~25 ms` (the 1→2-node fabric cost measured at
+23.1 ms warmup-free in §3c and +28.3 ms in §5d; the all-reduce payload is ~591 MB
regardless of batch or contract) ≈ **389-394 ms** ⇒ 81-82 samples/s/node against
1-node's 87.6 ⇒ **6-7% worse per node-hour.**

**3. A bigger batch has already been tested on this model, and lost.** §5l, at
matched LR and matched data: batch 48 is **11.7% worse early**, 0.25-0.57% better
late (inside batch 32's own 0.52%/epoch improvement), and costs **766-859 s/epoch
against 675-682**. The verdict recorded there is "batch 32 stays".

**4. The LR escape is narrower than §6 assumed, and the test is under-powered.**
§5j measured the ceiling at batch 48 as **(2e-3, 3e-3)** — 3.0e-3 collapsed at a
batch *smaller* than §6's 64 — and §7e confirmed the ceiling does not move with
batch (9 of 9 arms at 3.0e-3 collapsed). §6's 2.8e-3 sits in the window §5j calls
"squeezed, not excluded". Worse, §7e measured that `optimizer_max_grad_norm: 1.0`
**delays collapse from epoch 2 to epoch 6** — so a 6-epoch pass is exactly the
horizon where a delayed collapse has already been seen to land. "Survives 6
epochs" would not be evidence of survival.

⇒ Best case, §6's experiment reports that you may buy **half the wall-clock for
+6-7% node-hours and half the updates**. That is a scheduling convenience, not a
production shape, and wall-clock is not binding: `capacity` takes 1-4 nodes for
≤168 h and the 243-epoch run fit in 46. **Withdraw §6 and §7's second row.**

## 13. Option B is better resourced than §5 says — 243 members, not 12

Verified on disk at
`$MEMBER_ROOT/runs/makani_mn_scaling/e3sm_mn_scaling/prod1n_b32_sgdr/training_checkpoints`:
**`ckpt_mp0_v0` … `ckpt_mp0_v242`, contiguous, all 243 present**, 1.65 GiB each,
**403.2 GiB** total, plus `best_ckpt_mp0.tar`. §5k's twelve are the *report's*
every-20th subset, not what survived.

Three consequences for the ensemble scope:

* **Member spacing is a free parameter.** The correlation objection §5 raises
  against a single-trajectory ensemble can now be *measured* — spacing vs skill,
  at inference cost only, training nothing.
* The SGDR restart epochs can be selected **exactly** rather than approximated on
  the every-20 grid.
* ⚠ **403 GiB is being held by one run**, which is the same non-pruning defect
  `TODO.md` item 16 records for ai-rossby (`max_checkpoints_to_keep` does not
  prune). It is live on makani too. Check it **before** the next long run, and
  decide deliberately which members to keep — do not let a cleanup script pick.

## 14. §8, revised

1. **Do not re-run 128-node production.** Strengthened: not "an extrapolation says
   4.6× worse" but "cannot beat one node even with a **zero-cost** fabric", from
   fabric-free measurements (§10). Realistic gap 3.6-4.6× per node-hour, 74-182×
   on updates per node-hour.
2. ~~**Fix lead-time scoring first** (`TODO.md` P0-3).~~ 🔴 **STALE — it is already
   fixed, on another branch.** Commit **`652e9505`** on `wt-perlead-metrics`
   rebuilds `MetricsHandler` on the dataset's own channel names and dumps the full
   `(leads × channels)` curve to `<expDir>/scores/metrics_epN.h5`; verified on a
   compute node, job **7602739**, `PERLEAD_METRICS_OK`. It passes **all 101
   channels**, which is what makani's own `Inferencer` does (`inferencer.py:346`),
   so it is **not** the science choice P0-3 feared — choosing a *headline subset*
   still belongs to jesswan, and all-channels leaves that open.
   ⇒ The action is **merge, not re-fix**. §3 of this document, and `TODO.md` P0-3,
   describe the state of `feat/multinode-ddp-port` only. See §15.
3. **Continue the snapshot ensemble** — with **243** members available, not 12
   (§13), so spacing becomes an experiment rather than an assumption.
4. ~~**Run §6's batch-64 experiment.**~~ **Withdrawn** (§12). The node-hour lever
   is step time at batch 32 on one node — `TODO.md` P1-8, behind P1-9's
   equivalence gate (§11).
5. If a large-batch answer is wanted **for its own sake**, run it as an explicit
   science decision with the cost stated — unchanged.

### §7, revised

| if… | then |
|---|---|
| lead-time scoring is fixed and the 1-node model scores **badly** at long leads | the case for a differently-shaped run reopens — shape, not necessarily 128 nodes. **Unchanged, and still the only open input that could.** |
| ~~batch 64 at LR 2.8e-3 survives 6 epochs~~ | **withdrawn** (§12) — best case buys wall-clock at +6-7% node-hours and half the updates |
| ~~a 128-node cxi scaling job beats 2.47×~~ | **struck as a decision input** (§10) — the deciding quantity is measured at 1 node, where there is no fabric |
| a step-time optimisation passes the §4.1 equivalence gate | **new.** Every future run gets proportionally cheaper in updates/node-hour; nothing else needs re-costing |
| the science owner wants a large-batch result **for its own sake** | unchanged — a legitimate reason this analysis cannot weigh |

## 15. ⚠ The ensemble track is on another branch — and it is blocked by the fix on this one

Found 2026-09-17 while answering "how do we continue the ensemble work". This
section exists because §3 and §5 above describe `feat/multinode-ddp-port` and are
**not** the state of the project.

### 15a. What exists on `wt-perlead-metrics` (21 commits, not on this branch)

| | |
|---|---|
| per-lead scoring | ✅ `652e9505`, verified job **7602739** (`PERLEAD_METRICS_OK`) — all 101 channels, full `(leads × channels)` curve to `<expDir>/scores/metrics_epN.h5` |
| what the rollout does | ✅ **blurs and drifts** — RMSE grows **4.65×** from 6 h to 126 h, *linearly, no saturation*; ACC 1.000 → 0.878 |
| C1 (depth-1 rollout FT, 24 ep, batch 16) | ✅ **−3.00 %** RMSE at lead 126 h, 73/101 channels improved on both metrics |
| `n_future=4`, **one** epoch | ✅ **−4.66 % / −4.63 %** at 126 h, **replicated across two seeds** to 0.03 pp |
| D1 (batch 8, depth 1, 24 ep) | ✅ −0.72 % — batch 8 costs 2.3 pp of long-lead skill |
| CRPS / ensemble training | ⚙️ `PlasimEnsembleTrainer` built, 7 tests green — **never ran** (config root-key defect, since fixed) |
| lagged-ensemble design + E3SM inference port scope | ✅ written (`makani_sfno/docs/2026-09-10_*`) |

⇒ **At matched batch, one epoch at depth 4 beats twenty-four epochs at depth 1 by
~3.9 pp.** The production candidate is `n_future=4` at 24 epochs.

### 15b. Why it stalled — it ran into exactly the bug this branch fixed

The production candidate was submitted as **7621853** (2 nodes × local 2 = global
batch 16, `MULTISTEP=5`, 24 epochs, `preemptable`, ~16 node-hours) and **died in
epoch 1**: all 8 ranks timed out on a **6,306,103-element `ALLREDUCE`** at
`SeqNum=216` after 600 s, then aborted. Its log header reads

```
NET/OFI No eligible providers were found
NET/OFI Selected provider is tcp, fabric is 10.201.0.0/16
NET/OFI Need to force simple protocol: GDR not supported
```

— the **self-built v1.21.1 plugin over TCP**, because `wt-perlead-metrics`
branched at `ddd9bb5e` (2026-09-09), *before* `9c30e304` put HPE's rendezvous
block into `polaris_makani_multinode_scaling.pbs` — the very launcher this arm
uses.

⚠ And 2 nodes is **not optional** for this arm, so it cannot be dodged by running
on one node: **makani has no gradient accumulation**, so global batch = ranks ×
`LOCAL_BATCH`, and depth 4 at global 16 measures 22.29 GiB/GPU at local 2 (local 3
extrapolates to ~34 GiB of 39.49). This is the memory-driven multi-node case §12
explicitly carves out — it is not a throughput claim.

### 15c. The unblocking step is a merge, and it is clean

Measured: the two branches changed **17** and **23** files since `ddd9bb5e`, with
**zero overlap** — no conflicting file, let alone a conflicting hunk.

⚠ Two operational cautions:
* `git merge` writes the working tree across ~4,700 files, which is the class of
  git operation that wedges on this Lustre login node. Run it on a compute node
  or expect it to hang.
* The working tree of `feat/multinode-ddp-port` carries an **uncommitted, retracted**
  version of the per-lead patch (`plasim_trainer.py`, +42 lines, the 2026-09-04
  no-op that assumed the metrics were computed-and-discarded). It is **superseded
  by `652e9505`** — discard it, do not commit it.

### 15d. Order of work after the merge

1. **Resubmit the production candidate** (7621853's exact arm) on the cxi stack —
   ~16 node-hours, 2 nodes, `preemptable`. It is the one arm the ladder named.
2. **Run the CRPS/ensemble arm** — built and tested, never executed, and it is the
   only thing that addresses blurring rather than exposure bias.
3. **The E3SM inference port** (tasks 4-9 of `2026-09-10_lagged_ensemble_endtoend_plan.md`)
   — four generalising fixes in `sfno_inference/` plus a Polaris PBS sibling. Needed
   for *any* inference, lagged or not, and costs engineering rather than allocation.
4. **Task 10, the K=56 (14-day) sweep — the real decision point.** At 126 h the ACC
   is still 0.878, far from climatology, so **exposure bias and mode-averaging are
   not yet separable**. That curve is what says whether further rollout depth or a
   distributional objective is the right spend. Ordering 3-4 before 1-2 is
   defensible on exactly that argument.
