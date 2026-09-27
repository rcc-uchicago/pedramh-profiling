# ACE2 on Polaris — pre-registration

**Written 2026-09-02, before the first ACE2 job on Polaris has run.** Scored
afterwards in `CHANGELOG.md`, **including the misses**. The point of writing it
first is that "prereg beats authority" is a measured result here, not a slogan:
in makani's LR sweep the value taken from upstream's own config came **last** of
three.

Predictions are numbered, falsifiable, and each states the condition under which
it is falsified. Where the basis is weak, that is said, so a hit is not
over-read.

---

## 0. The question this campaign is actually asking

Not "how well does ACE2 scale out". Three independent facts point away from that
framing before a single job runs:

1. makani measured that **at a fixed global batch, more nodes made training
   slower** — 1 node was both cheapest and fastest, because touching the fabric
   costs a fixed ~234 ms/step (`makani_bench_report.md` §5).
2. ai-rossby reproduced the same 2-node trough on an unrelated model, which makes
   it a **machine characteristic**, not a per-model quirk.
3. ACE2's own Midway profile puts **NCCL at 40–46% of GPU kernel time**, and
   `bench_midway_notes.md` already measured that raising the batch moves it
   52.1% → 18.6%. The lever on communication is samples per rank.

So the question is **"what is the fewest GPUs that hold the batch"**, and the
1-node arm is the experiment, not the formality.

---

## 1. Predictions

### P1 — `gpu_busy_frac` on the 1-node arm is **below 0.90** — ❌ **FALSIFIED (job 7586496: 0.9325)**

> **Scored 2026-09-02.** Measured **0.9325** at `LOCAL_BATCH=1`, 1 node, on the
> unconverted `.nc`. The *prediction* is falsified.
>
> ⚠ **But "the loader is not the bottleneck" is a STRONGER claim than this
> measurement supports, and an earlier version of this note made it.** What is
> established is narrow: at 4–8 ranks and 60 steps, ≤6.75% of wall sat outside
> the step window. What is NOT established:
>
> * **The OST's ceiling is unmeasured.** The arms demanded 357 / 406 / 512 MB/s
>   (computed exactly from the file's real dtypes and shapes: 42.77 MB per
>   3-timestep sample × samples ÷ `epoch_wall_s`). Whether that is 15% or 90% of
>   what one OST delivers decides the conversion, and nobody has probed it.
>   → `polaris_ace2_io_probe.pbs`, job 7586630.
> * **The 2-node arm is a WEAKER loader test than the 1-node one, not a stronger
>   one.** Per-rank demand *fell* 101.5 → 64.0 MB/s because NCCL stretched the
>   step 68% and handed the loader more time; total demand rose only 406 → 512.
>   So `gpu_busy_frac` rising to 0.9625 at 2 nodes is **not** evidence that the
>   I/O scales.
> * **The 4–7% gap is unattributed** — loader, CPU, or aggregator, unknown.
> * **A marginal deficit can hide behind prefetch.** `num_data_workers=4` ×
>   `prefetch_factor=2` is an 8-batch cushion; a ~5% shortfall would not drain it
>   inside 60 steps. A sustained large one would.
> * **4–8 nodes is pure extrapolation** — ~1.6–3.2 GB/s from one OST, untested.
>
> One thing does point the right way, and it is worth stating because it is the
> opposite of the makani precedent: `sample_with_replacement` uses
> `RandomSampler(replacement=True)` over 121,262 timesteps, so **these arms were
> not cache-hot** — ~2% hit probability against 2.4 TB of file and ~512 GB of RAM.
> makani's 30% benchmark optimism came from re-reading a warm window; that
> mechanism is absent here. Production uses a shuffled `DistributedSampler`, i.e.
> the same random pattern.
>
> ⇒ **The conversion is not justified *yet*, at ≤8 ranks. It is not closed.**

*Because* every rank reads the same 2,388.77 GB NetCDF and `lfs getstripe` says
that file has `lmm_stripe_count: 1` — it lives on **one Lustre OST**. ai-rossby
sharded across 30 separate zarr stores and held ≥0.976 on every arm.

**Falsified if** it comes in ≥0.90. That would be the good outcome: the loader
is not the bottleneck, the 2.4 TB → zarr conversion (handoff §2a) is unjustified
speculative spend, and the comms analysis is the right lens after all.

⚠ **Basis: moderate.** The stripe count is measured; the loader's access pattern
is not. `RandomSampler(replacement=True)` over ~230k samples with 4 workers per
rank may hide the latency behind prefetch. This prediction is as likely to be
wrong as right, and it is written down precisely so that cannot be re-narrated
afterwards.

### P2 — ACE2's largest single gradient collective is **150–250 MB**, not one ~2.7 GB collective — 🔴 **FALSIFIED (job 7586590: 1738.86 MiB)**

> **Scored 2026-09-02, and it is the result that matters.** The 2-node dump shows
> **one `nccl:all_reduce` of `numel=455,831,040` — the entire model, 1.823 GB =
> 1738.86 MiB.** That is not in the untested gap; it is **above 1000 MiB, inside
> the range where Tree was measured to fail**. ⇒ **ACE2 is exposed, and
> `NCCL_ALGO=Ring` is load-bearing rather than insurance.**
>
> It fires **once, as collective 14 of the run**, immediately after DDP's
> parameter broadcast and before the first backward. Per-*step* traffic is the
> benign shape P2 predicted: ~11 buckets, largest ≈215 MB.
>
> This is the same full-model coalescing ai-rossby showed (`numel=1182108160`),
> and it explains why its `bucket_cap_mb` sweep made no difference: **the
> collective is not a gradient bucket.** Mechanism still unnamed — the dump
> carried no stack frames; one arm with stack capture would name the call site.
>
> ⚠ **P2's own premise also has to be retracted.** It cited "~2.7 GB" as the
> alternative, from the handoff's complex64 correction. Both the source
> (`s2convolutions.py:148` declares a **float32** tensor with a trailing size-2
> dim; `view_as_complex` is applied at use time) and the dump (`dtype=['Float']`,
> no complex dtype in 5,520 records) say the gradient volume is **1.823 GB**.

Predicted ~165 MB (the bucket size `PROFILING_PLAN.md:171` already measured:
"11.4 buckets/step, ~165 MB each") and ~212 MB for the standalone dhconv weight
(384×384×180 complex64 = 212.34 MB; DDP never splits one parameter across
buckets).

**Falsified if** the flight-recorder dump of a 2-node run shows one full-model
collective — which is what ai-rossby inexplicably did (`numel=1182108160` in
*both* the 25 MB-bucket run and the forced-one-bucket run, a byte-identical stuck
collective under a 200× difference in `bucket_cap_mb`). That would make ACE2
exposed to the tree defect exactly as ai-rossby was, and why DDP coalesces is
still open in the CHANGELOG.

⚠ This **replaces** an earlier draft prediction ("ACE2 hits the tree defect,
falsified if the default works"), which was misconceived: it compared *total*
gradient volume against a *per-collective* threshold, so it would have been
falsified for a reason that teaches nothing about the fabric.

### P2b — at ~165–212 MB, the **default** algorithm does not hang at 2 nodes — ⚪ **MOOT, and its premise is refuted**

> **2026-09-02.** The premise ("~165–212 MB") is false: the largest collective is
> **1738.86 MiB**, above the measured-failing threshold rather than below it. So
> the interesting question is no longer "does the gap fail" but the settled one:
> ACE2 is in the range that already fails. A `-v NCCL_ALGO=` (empty) arm would now
> be **deliberate fault injection**, not a control — worth one job as ALCF ticket
> evidence, not as a candidate configuration.

**Falsified if** it does — which would pull the Tree corruption threshold below
212 MiB, is a genuinely new fabric result, and belongs in the ALCF ticket
alongside makani and ai-rossby as a **third independent harness**.

⚠ Not tested by the ladder, which runs `NCCL_ALGO=Ring` throughout as insurance.
Testing it needs a deliberate `-v NCCL_ALGO=` (empty) arm at 2 nodes, and that
arm's row goes in its own file.

### P3 — the **1-node arm is the fastest per-GPU point**, and 2 nodes is a trough — ✅ **HIT, and the full ladder now confirms the shape**

> **Scored 2026-09-02 on the complete ladder** (LOCAL_BATCH=2, 60 steps, Ring):
>
> | nodes | ranks | global | step_med_ms | s/s/rank | s/s total | gpu_busy | reps |
> |---|---|---|---|---|---|---|---|
> | 1 | 4 | 8 | 716.0 | **2.7932** | 11.17 | 0.9288 | 3, ±0.1% |
> | 2 | 8 | 16 | 1204.4 | 1.6606 | 13.29 | 0.9625 | 3, ±3.9% |
> | 4 | 16 | 32 | 1426.1 | 1.4024 | 22.44 | 0.9691 | 2, ±3.4% |
> | 8 | 32 | 64 | 1498.5 | 1.3346 | 42.71 | 0.9703 | 1 |
>
> **The cliff is the first hop, then it saturates** — −42% per-GPU at 1→2 nodes,
> then only −12% and −6.5% for the next two doublings. That is ai-rossby's shape
> reproduced on a third harness, and it is the *opposite* of an I/O wall.
>
> ⚠ **Weak-scaling efficiency against 1 node (100/58/51/48%) is the wrong headline
> for this model**, because 1 node cannot hold the production batch at all
> (local 3 OOMs). Measured from the 2-node minimum viable config, ACE2 scales
> **13.04 → 22.83 → 42.71**, i.e. 1.75× and 3.28× for 2× and 4× the hardware —
> **87% and 82% incremental efficiency.** ACE2 scales well *once past the toll it
> cannot avoid paying*.

Concretely: `samples_s_rank` at 2 nodes is **below** the 1-node value, and 4
nodes recovers only partially. Both prior harnesses show this shape.

**Falsified if** 2-node `samples_s_rank` ≥ the 1-node value.

### P4 — first-hop penalty lands in **390–1560 ms** — ✅ **HIT, but do not bank it**

> **Scored 2026-09-02, updated as reps landed:** median 1204.4 − 716.0 =
> **+488.4 ms/step** at fixed local batch 2 (n=3 at both rungs). Inside the window, below
> the ~780 ms centre.
> ⚠ Two reasons to treat this as weak: the prediction's basis was already stated
> as weak, and **the gradient volume it scaled from (2.67 GB) has since been
> corrected to 1.823 GB** — so the arithmetic that produced the window was wrong
> even though the window contained the answer. n=1.

ACE2's 2.67 GB of gradients is ~0.56× ai-rossby's 4.73 GB, so if the penalty
tracks gradient volume, arm(2n) − arm(1n) ≈ 0.56 × 1384 ms ≈ **780 ms**.

**Falsified outside 390–1560 ms.**

⚠ **Basis: weak, and stated as such.** ai-rossby's own +1384 ms carries ~±11%
(its 2-node arm spread 15% over 5 reps), and "volume-linear" was itself inferred
from a single makani comparison. A miss here is uninformative about the fabric;
a hit is weak evidence.

### P5 — the AUTO pin carries again — ✅ **HIT (weakly: not a knob matrix)**

> **Partially scored 2026-09-02.** Every arm reported `Using network AWS
> Libfabric` under `OFI_NCCL_PROGRESS_MODEL=AUTO`, including the 2-node arm, so
> the pin carries on torch 2.10.0+cu129 / NCCL 2.27.5. ⚠ **No other progress
> model was tried**, so this confirms AUTO works, not that it is uniquely the one
> that works. The full six-way matrix was deliberately not re-run because this
> venv's NCCL is byte-identical to ai-rossby's, where it was.

`OFI_NCCL_PROGRESS_MODEL=AUTO` is the only working combination, as on makani's
torch 2.8.0/NCCL 2.28.3 and ai-rossby's 2.10.0/NCCL 2.27.5.

**Falsified if** any other progress model opens a domain, or if AUTO does not.
Inherited rather than re-measured because this venv pins **the same torch 2.10.0
+cu129 as ai-rossby** — that is the reason for the pin in
`polaris_setup_ace2_venv.sh`, and it is what makes inheriting legitimate instead
of lazy. If the torch version ever moves, this prediction reverts to unmeasured.

### P6 — the Midway NCCL share does **not** transfer: on 1 Polaris node, NCCL is materially **below 40%** of GPU kernel time

The Midway profile was taken on **A100-PCIE with no NVLink**, where an fp32
gradient all-reduce is expensive — that is *why* it read 40–46%. Polaris A100-SXM4
is a full NV4 mesh at **82.9–83.1 GB/s uniform on every pair** (job 7533457).

**Falsified if** it is still ≥40%, which would mean the Midway diagnosis was
never about the interconnect at all.

⚠ Needs a kernel-level capture (`ace2_nvtx.py` + nsys), not the scaling ladder.
Not scheduled yet; recorded here so it is not quietly dropped.

### P7 — memory: `LOCAL_BATCH=1` fits; there is a **cliff, not a curve**, at some batch ≤8

> **Scored 2026-09-02 — half hit, half ❌ FALSIFIED.**
>
> | local batch | peak GiB allocated | job |
> |---|---|---|
> | 1 | 21.316 (reserved 21.764, `alloc_retries=0`) | 7586496 |
> | 2 | 33.959 (reserved 34.371) | 7586506 |
> | 3 | **OOM** — 38.20 allocated, 39.39 in use, died asking for 286 MiB | 7586526 |
>
> ✅ `LOCAL_BATCH=1` fits. ❌ **There is no cliff.** +12.643 GiB per added sample
> from 1→2, and 3 fails exactly where that increment predicts (46.6 > 39.49).
> **makani's discrete cliff did not reproduce here**; the shape is the boring one.
> ⚠ Two points define a line trivially, so no model is being claimed — the result
> is the measured boundary: **local batch 2 is the maximum on a 40 GB A100**.
>
> ⇒ **`batch_size: 16` does not fit one node.** It needs 8 GPUs at local 2, so
> §0's "fewest GPUs that hold the batch" answers **two nodes** — the opposite of
> makani, and it means ACE2 *must* pay the fabric toll makani could avoid. The
> live comparison is now 2 nodes × local 2 vs 4 nodes × local 1 at equal global
> batch, which is a strong-scaling question, not the weak-scaling ladder.

Fixed state alone is ~10.7 GB of a 39.49 GiB card (2.67 GB parameters + 2.67 GB
gradients + AdamW's two moments). makani, with 2.37 GB of fixed state, went from
18.97 GB at 12 samples/GPU straight past 39.5 GB at 16.

**Falsified if** peak memory is linear in `LOCAL_BATCH` across every value that
fits, i.e. the last fitting value and the first OOM differ by roughly one
increment's worth of memory.

⚠ **This is measured one arm per value and never fitted.** makani's curve was
fitted twice and refuted twice; both refuted models are recorded in
`makani_bench_report.md` §5g so neither gets refitted. If this prereg's own
"cliff" framing turns out to be a third bad model, say so.

### P8 — rep spread < 5% per arm — ✅ **HIT so far (1n and 2n)**

> **Scored 2026-09-02.** 1 node: **±0.1%** over 3 reps (716.147 / 715.404 /
> 716.0 ms) — near-identical. 2 nodes: **±3.8%** over 2 reps (1203.5 / 1250.6).
> Both inside 5%, and the 2-node figure is far tighter than ai-rossby's 15% miss
> at the same rung.
> ⚠ **4n and 8n are still n=1** and must not be published as ladder points until
> they have reps; the driver (`run_ace2_ladder.sh`) fills the shortest rungs
> first.

**Falsified if** any arm's 3 interleaved reps spread wider. ai-rossby missed this
at 2 nodes (15% over 5 reps), so a miss at 2 nodes specifically would be a
reproduction, not a surprise.

### P9 — the ladder saturates on **I/O**, not only on the fabric — 🔴 **FALSIFIED ON ALL THREE COUNTS**

> **Scored 2026-09-02, and it is the cleanest miss in this file** — registered
> before the 4n/8n arms ran, refuted by them within the hour.
>
> | | predicted | measured |
> |---|---|---|
> | P9a `gpu_busy_frac` | <0.90 at 4n, <0.80 at 8n | **0.9688 / 0.9703** — it *rose* monotonically |
> | P9b `samples_s_total` at 8n | < 26.6 (under 2× the 2n rung) | **42.71** — 3.3× the 2n rung |
> | P9c implied read rate at 8n | 500–800 MB/s | **1643.6 MB/s** |
>
> **The single OST sustains ~1.64 GB/s under the real loader at 128 concurrent
> readers, with the GPUs 97% busy.** The I/O is not the bottleneck anywhere on
> this ladder, and the 2.4 TB → zarr conversion is not justified — now on much
> stronger evidence than the 1-node arm that first suggested it.
>
> ⚠ **The app-free probe that motivated P9 UNDERSTATES the OST by ~4.8×, and its
> absolute numbers should not be quoted.** It divides total bytes by a wall clock
> that includes `mp.Pool` startup and 32–128 opens of a 2.4 TB HDF5 file, none of
> which scale with reader count. Its *latency* series (median window 2.0 → 2.7 →
> 3.7 s from 8 → 16 → 32 readers) is real and does show contention appearing; its
> *aggregate* is a floor, not a ceiling. The training arms are the better
> instrument and they supersede it.
>
> ⚠ **And `gpu_busy_frac` rising with node count is not the good news it looks
> like.** The step gets longer (NCCL), so a fixed loader gap becomes a smaller
> fraction of it. The column measures loader idle, not communication cost — which
> is exactly the caveat recorded under P1, now confirmed in the other direction.

The app-free probe (job 7587664, clean seeds) measured the single OST's
per-reader throughput **collapsing past 8 concurrent readers**:

| concurrent readers | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| aggregate MB/s | 21.4 | 42.9 | 82.2 | 161.8 | 220.3 | 343.2 |
| per reader | 21.4 | 21.5 | 20.5 | **20.2** | **13.8** | **10.7** |
| scaling efficiency | 100% | 100% | 96% | 94% | **64%** | **50%** |

Each rung runs `4 × nodes` ranks × 4 loader workers, so concurrency is 16 / 32 /
64 / 128 readers at 1 / 2 / 4 / 8 nodes. **The 1-node arm was already past the
knee.**

**Predictions, at fixed `LOCAL_BATCH=2`:**

* **P9a — `gpu_busy_frac` falls below 0.90 at 4 nodes and below 0.80 at 8**,
  reversing the 0.9357 → 0.9625 rise seen from 1 → 2 nodes. That rise was a
  fabric artifact (NCCL stretched the step and handed the loader more time); once
  the OST binds, the loader gap should reopen faster than NCCL can hide it.
* **P9b — `samples_s_total` saturates: the 8-node rung is under 2× the 2-node
  rung** (i.e. < 26.6 samples/s), against the 4× a linear ladder would give.
* **P9c — the 8-node rung's implied read rate lands in 500–800 MB/s**, i.e. the
  OST is delivering 1.5–2.3× its measured 32-reader figure and no more.

**Falsified if** `gpu_busy_frac` stays ≥0.90 at 8 nodes, which would mean the OST
scales well past where the probe says it stops and the whole I/O concern is
misplaced.

⚠ **Basis: moderate, and the confound is named.** The probe uses h5py directly
while fme reads through xarray/h5netcdf, and the 1-node training arm achieved
~348 MB/s where the probe's 16-reader point gave 220 — so the real loader is
**~58% faster than the probe at equal concurrency**, for reasons not yet
identified. The *shape* (per-reader decay past 8 readers) is the prediction; the
absolute numbers are not.

⚠ This prediction cannot be scored off `gpu_busy_frac` alone, because that column
counts exposed NCCL time as *busy*. Score it with `samples_s_total` and the
implied MB/s together.

### P10 — ACE2 does **not** reproduce makani's "more nodes = slower" at FIXED global batch — ✅ **HIT on direction, ❌ MISSED on magnitude**

> **Scored 2026-09-02, job 7588972.** At global batch 16:
>
> | config | step_med_ms | s/s total | `gpu_busy_frac` |
> |---|---|---|---|
> | 2 nodes × local 2 | 1204.4 | 13.29 (n=3) | 0.9625 |
> | **4 nodes × local 1** | **1156.6** | **13.83** | 0.9723 |
>
> **Direction: hit.** More nodes at fixed global batch is **faster, +4.1%** —
> makani's ladder never recovered its 1-node throughput at any larger node count.
> **Magnitude: missed.** Predicted 15.0 samples/s; got 13.83. The derivation
> assumed the 1→4-node toll measured at local batch 2 (+685.7 ms) carries
> unchanged to local batch 1; the real step was 1156.6 ms, not the predicted
> 1066 — so the toll is **not** batch-independent, and that assumption (flagged as
> unmeasured when registered) is now refuted.
>
> ⚠ **Read the size honestly: +4.1% for 2× the hardware is 2% efficiency on the
> added nodes.** The correct claim is "**ACE2 does not get WORSE with more nodes
> at fixed batch**", not "ACE2 scales at fixed batch". And makani's headline
> question — *is 1 node fastest?* — **cannot even be posed for ACE2**, because
> 1 node cannot hold global batch 16 at all.

**The comparison so far has been unfair, and this prediction exists to fix it.**
makani's §1f table holds the **global** batch at 32 and shrinks samples/GPU
8→4→2→1: that is **STRONG** scaling. My ladder holds the **local** batch at 2 and
grows the global batch 8→16→64: that is **WEAK** scaling. Under weak scaling the
added ranks bring added work, so a roughly fixed per-step collective is amortised
over more compute; under strong scaling they do not. **So "ACE2 does not suffer
like makani" is currently a statement about two ladder designs, not about ACE2.**

The apples-to-apples point is **global batch 16 on 2 nodes (local 2) vs 4 nodes
(local 1)** — same work, twice the hardware, which is exactly makani's axis.

**Prediction: the 4-node/local-1 arm is FASTER, ~15.0 samples/s against the
2-node/local-2 arm's measured 13.04** (n=3). Derived by adding the ladder's own
1→4 node toll at local batch 2 (1401.7 − 716.0 = +685.7 ms) to the measured
1-node/local-1 step of 380.6 ms ⇒ ~1066 ms for 16 samples.

**Falsified if** it comes in **below 13.04 samples/s**, which would mean ACE2
behaves exactly like makani and the difference was purely the experiment design.

**Mechanism, if the prediction holds — ACE2 is ~7.3× heavier per sample:**

| | ms of compute per sample | at 1 sample/GPU, vs a ~0.5 s fabric toll |
|---|---|---|
| ACE2 | **335.4** (marginal, from 380.6 → 716.0 at local 1 → 2) | compute still ~40% of the step |
| makani | **45.7** (average at 8 samples/GPU) | compute ~16% of the step |

A per-step toll that is roughly independent of node count is amortised by
whatever compute remains on each GPU. makani's fixed-batch ladder could slice
down to 1 sample/GPU, where 46 ms of compute sits against a 234 ms toll and the
fabric necessarily dominates. **ACE2 cannot get there**: it is 7.3× heavier per
sample, and it physically cannot go below 1 sample/GPU — so it never enters the
regime makani's table is measuring.

⚠ **Basis: moderate.** The 335.4 ms/sample marginal cost is a two-point fit
(local 1 and local 2 at one node) and nothing rules out non-linearity between
them. The toll is assumed batch-independent because it is the gradient
all-reduce, whose volume is the model size — that is sound in principle and
unmeasured here.

⚠ Its row goes in `ace2_polaris_strongscale.csv`, **not** the weak-scaling table:
a strong-scaling point in a weak-scaling table is the exact mislabelling §3.1 of
the handoff warns about.

### P11 — `GPU_ORDER=reverse` is **faster** for ACE2 at 1 node — ❌ **FALSIFIED at 4n; UNRESOLVABLE at 1n**

> **Scored 2026-09-02** (jobs 7588998, 7588999):
>
> | rung | forward | reverse | delta | forward's own spread |
> |---|---|---|---|---|
> | 1 node | 716.0 ms (n=3) | 715.2 ms (n=1) | **−0.12%** | ±0.1% |
> | 4 nodes | 1426.1 ms (n=2) | 1466.0 ms (n=1) | **+2.80% SLOWER** | ±3.4% |
>
> ⚠ **NEITHER DELTA IS RESOLVABLE.** Each reverse arm is n=1 and each delta sits
> inside its own forward baseline's rep spread. The prediction is scored
> **falsified on direction at 4 nodes** — I bet faster, it came out slower — but
> the honest statement is that **ACE2 shows no placement effect either rung can
> resolve at n=1.**
>
> **What IS established, and it is the useful part: makani's −7.0% does NOT
> reproduce.** A 7% gain at 4 nodes would sit far outside the ±3.4% spread, so an
> effect of that size is excluded even at n=1.
>
> ⚠ **And the comparison was never apples-to-apples in the first place** — a point
> I should have made before betting. makani's −7.0% was measured at 4 nodes
> **SHARDED** (model-parallel). **ACE2 has no model-parallel path**; it is pure
> DDP. The configuration in which makani found the win does not exist here, so
> there was never a reason to expect its sign, in either direction.
>
> ⇒ The repo's existing verdict stands and is reinforced by a third harness:
> **placement is config-dependent; do not port a sign** — including a sign
> inferred from first principles, as this prediction's NUMA/H2D reasoning was.
> `forward` remains correct for every ACE2 configuration measured.

**Not previously tested: all 12 ACE2 rows so far are `gpu_order=forward`.**
`polaris_pbs_notes.md` §1 measured the GPU↔NUMA map to be REVERSED (dev0→NUMA3
… dev3→NUMA0, job 7531456), so under `--cpu-bind depth -d 8` local rank 0 gets
cores 0–7 = NUMA 0, whose GPU is dev3 — the default pairing puts every rank
maximally far from its own GPU.

**Prediction: reverse is 0–3% FASTER at 1 node.** ⚠ This is a deliberate bet
**against** the neighbouring harness: makani measured reverse **+0.88% *worse*** at
1 node (3+3 node-matched reps) and −7.0% better at 4 nodes sharded, and its
verdict was explicitly "config-dependent, do not port a sign". The reason to
expect a different sign here is that ACE2's loader pushes **41.73 MB per sample**
through pinned host memory on the H2D path, so NUMA locality of the staging
buffers should matter more than it did for makani's smaller per-sample payload.

**Falsified if** reverse is slower at 1 node, i.e. makani's sign carries after
all.

The 1-node rung is the right place to detect it: its forward baseline is
**n=3 at ±0.1% spread**, so even a 1% effect is resolvable. A 4-node arm runs
behind it because that is where makani's sign flipped.

⚠ **Not node-matched.** makani's placement arm ran 3+3 reps on the *same* nodes
because node-to-node variation can swamp the effect. These arms take whatever
PBS gives them — justified only by that ±0.1% forward spread across three
separate allocations, which bounds node variation for ACE2 as small. If the
measured effect is under ~1%, it is not resolvable this way and needs the
node-matched design.

⚠ `gpu_order` **is a column** in the scaling CSV (shared with ai-rossby's
schema), so reverse rows belong in the same table — but every ladder summary must
then filter `gpu_order == forward` or it will average two configurations.

## 1a. LR SWEEP — SELECTION RULE, WRITTEN BEFORE THE ARMS EXIST

> ### 🔬 SCREEN SCORED 2026-09-04 (not the registered sweep — 8% of its updates)
> | LR | final valid @ 3,000 updates |
> |---|---|
> | 1e-4 *(config's own)* | 0.9362 — **last** |
> | 3e-4 | 0.7249 |
> | **1e-3** | **0.6417 — winner** |
> | 3e-3 *(added by the endpoint rule)* | worse than all at every epoch |
>
> ✅ **Rule applied as written.** 1e-3 won at the range TOP, so the endpoint clause fired and
> **3e-3 was run before scoring**; it is worse everywhere, so 1e-3 is an **interior optimum**
> and the result is adoptable in principle. ✅ No arm was disqualified — no NaN, no rising
> validation loss, and no divergence even at 3e-3 with clipping unavailable. The
> batch_loss-variance tie-break was never needed (the gap is 11.5%, far outside a tie).
>
> 🎯 **The config's own 1e-4 came LAST — makani's finding reproduced on a second harness.**
>
> ⚠ **Not the registered experiment.** 3,000 updates/arm vs the registered 36,702, n=1, and
> early rank order need not survive to 332k updates. The four full arms (7589850-53) are still
> queued. Treat 1e-3 as the current best estimate, not a settled value.

**Registered 2026-09-03, before jobs are submitted.** "Prereg beats authority" is a
measured result here: in makani's sweep the value taken from **upstream's own
config came LAST of three**. ACE2's `optimization.lr: 1e-4` is inherited from the
ai2cm/Delta config at global batch 16 and **has never been validated on this data
at any batch**.

**Configuration under test:** 1 node, global batch **8** (local 2), **3 full
epochs** (12,234 updates/epoch ⇒ 36,702 updates/arm), **flat LR**
(`-v NO_SCHEDULER=1`) — a schedule would confound the arms by testing each at a
different effective LR by the time it is scored. Production then uses the winner
as the **peak** of `CosineAnnealingWarmRestarts`, which is makani's exact shape.

**Arms (4):** `5e-5, 1e-4, 3e-4, 1e-3` — spanning 20×. This deliberately brackets
both the linear-scaled value for the halved batch (5e-5) and the config's own
value (1e-4), and extends upward because makani's winner was **5× upstream's
value**.

### The rule

1. **Disqualify** any arm that raises `Loss is NaN-valued during training`
   (`Optimization._validate_loss` raises, so this is loud), or whose validation
   loss *increases* from epoch 2 → 3.
2. **Winner = lowest validation loss after epoch 3.**
3. **Tie-break** (within 2%): lower variance of the per-step `batch_loss` across
   the final epoch.

⚠ **The tie-break is NOT gradient norm, and that is a forced change from makani's
rule.** `grep` over `ace_exp/fme` finds **no gradient-norm logging and no
gradient clipping anywhere** — fme neither computes nor exposes it. Anyone
expecting makani's `pick_lr.py` criterion here will not find it. The batch_loss
variance is a *proxy* for gradient noise and is weaker; say so when scoring.

⚠ **fme also does no gradient clipping at all**, and it is not configurable. That
is a real risk at the top of this range: ai-rossby's divergence investigation
listed `grad_clip_norm: 0.0` as a leading suspect, and here it is not 0 by
choice — the capability is absent. An arm that diverges at 1e-3 may be telling us
about the missing clip rather than about the LR.

⚠ **If the winner is 5e-5 or 1e-3 — i.e. an ENDPOINT — the range was
insufficient and the result must NOT be adopted as-is.** Extend and re-run.
makani's CHANGELOG carries exactly this caveat ("2e-3 was the top of the range
tested") and repeating it knowingly would be worse than the first time.

🔵 **OPERATOR DECISION 2026-09-18 (rmehta1987): these are ADOPTED. Production is
not gated on external sign-off.** What follows is therefore a *record of the
deviations*, not a request for approval — the distinction matters because every
item below is still a real departure from ai2's published ACE2-ERA5 recipe, and a
reader comparing our numbers against the paper's needs the list either way.

Batch 8 is a deviation: the config says 16, and 8 is chosen as *the largest batch
that fits one node* — a hardware fact — which also happens to be **3.4× more
update-efficient** (5,028 vs 1,495 updates/node-hour) because it avoids the
fabric entirely.

⚠ **RESTATED 2026-09-18: that 3.4× is a TCP-era figure. On cxi it is 2.3×**
(5,044 vs 2,208 updates/node-hour, from the measured 713.7 / 815.3 ms steps —
jobs 7631544 / 7631529). **The conclusion survives and the magnitude does not.**
1 node still wins on *both* axes — updates per node-hour **and** updates per hour
of wall-clock (5,044 vs 4,415) — so the production shape is unchanged by the
fabric fix. ⚠ Do not read §1f's "a second node costs only +14% node-hours" as
licensing a 2-node production run: that +14% is **per sample**, and per *update*
2 nodes costs 2.3× the allocation to train more slowly.

⚠ **And "the largest batch that fits one node" is now only true at
`use_gradient_accumulation=false`.** Job 7608867 measured the config's own
global 16 fitting on **one** node at 37.095 GiB with accumulation **on** — but
that is truncated BPTT, i.e. a **different gradient**, and it invalidates the LR
sweep that was run without it.

**It is the paper-fidelity alternative, and it is NOT what the recommended run
uses — on evidence, not on permission.** Three measured reasons: it is
**−6.5% samples/s**; it leaves **2.4 GiB of headroom** (37.095 of 39.49 = 94%),
which is thin enough that the inline-inference path is unproven at that footprint;
and the **LR 3e-4 winner was measured at `false`**, so adopting `true` re-opens
the one hyperparameter this campaign actually settled. Running it would mean
re-running the LR sweep first. Worth doing as its own arm; not worth bundling
into the production launch.

---

## 1b. Scorecard as of 2026-09-02 (11 arms: full 1/2/4/8-node ladder + the batch search)

| # | prediction | outcome |
|---|---|---|
| P1 | `gpu_busy_frac` < 0.90 | ❌ **falsified** — 0.9325 at 1n, and it *rises* to 0.9703 at 8n |
| P2 | largest collective 150–250 MB | 🔴 **falsified** — **1738.86 MiB**, one full-model all_reduce ⇒ **ACE2 is exposed to the tree defect** |
| P2b | default algo safe at 165–212 MB | ⚪ moot — premise refuted |
| P3 | 2 nodes is a per-GPU trough | ✅ hit — −42% samples/s/rank, then it saturates (−12%, −6.5%) |
| P4 | first hop 390–1560 ms | ✅ hit — **+488.4 ms** (n=3 both rungs) |
| P5 | the AUTO pin carries | ✅ hit, weakly — confirmed working, not shown unique |
| P6 | NCCL < 40% of kernel time on Polaris | ⚪ untested — needs an nsys capture |
| P7 | batch fits at 1; cliff, not curve | half ✅ / ❌ — batch **2 is the max**, and there is **no cliff** |
| P8 | rep spread < 5% | ✅ hit at 1n (±0.1%, n=3) and 2n (±3.8%, n=2); 4n/8n still n=1 |
| P9 | the ladder saturates on I/O | 🔴 **falsified 3/3** — OST sustains **1.64 GB/s** at 32 ranks, `gpu_busy` **0.970** |
| P10 | more nodes at fixed batch is not slower | ✅ direction hit (+4.1%), ❌ magnitude missed (13.83 vs 15.0 predicted) |
| P11 | `GPU_ORDER=reverse` faster at 1n | ❌ falsified at 4n (+2.80%), unresolvable at 1n (−0.12%); makani's −7.0% excluded |

**Four of seven scored predictions were wrong**, and the misses are where the
value is: P1 and P7 removed speculative work; P2 found a correctness hazard that
was live in every multi-node arm and is only mitigated because `NCCL_ALGO=Ring`
shipped on by default from job one; and **P9 was registered specifically to test
the I/O worry and refuted it decisively** — the single OST sustains 1.64 GB/s
under the real loader while the GPUs sit at 97% busy.

⇒ **ACE2 on Polaris is fabric-limited, not I/O-limited.** The 2.4 TB → zarr
conversion is not justified. The one unavoidable cost is that the production
batch needs 2 nodes, so ACE2 always pays the first-hop toll; past that it scales
at 82–87% incremental efficiency.

---

## 1c. 🔴 SLINGSHOT RE-MEASUREMENT — written 2026-09-17, before any cxi arm exists

**Everything in §1b was measured over TCP.** aws-ofi-nccl 1.21.1 cannot negotiate
the CXI provider (it omits `FI_MR_PROV_KEY`, which CXI mandates) and silently
falls back to `tcp` with GPUDirect RDMA off — while still printing
`Using network AWS Libfabric`, the string this campaign's `transport` column
recorded. Six ACE2 training logs confirmed: `provider=tcp` on every one.
→ `polaris_ace2_slingshot_handoff.md`.

⚠ What that does and does not invalidate is in §1d. These four predictions are
registered **before** the re-measured ladder, on the same rule as the rest of this
file: written first, scored after, misses kept.

### P12 — the 2-node step time improves by **>1.5×** on cxi vs its tcp row — ❌ **MISSED BY 1.5% (job 7631529: 815.3 ms, 1.477×)**

> **Scored 2026-09-18, n=1.** `ACE2_POLARIS_TRAIN_OK nodes=2`, **`provider=cxi`** —
> the first ACE2 row on Slingshot, in `ace2_polaris_scaling_cxi.csv`.
>
> | quantity | tcp (n=3) | cxi (n=1) | change |
> |---|---|---|---|
> | `step_med_ms` | 1204.4 | **815.275** | **−32.3%** |
> | `step_p90_ms` | — | 818.411 | p90 within **0.4%** of the median |
> | `samples_s_rank` | 1.6605 | **2.4532** | **1.477×** |
> | `samples_s_total` | 13.29 | **19.6253** | 1.477× |
> | `gpu_busy_frac` | 0.9288 | 0.9469 | +1.9 pp |
> | `epoch_wall_s` | — | 55.181 | — |
> | `peak_mem_gb` | 33.959 | **33.959** | **identical** — memory is set by training, as expected |
>
> **1.477× against a 1.5× threshold.** It needed < 803 ms and delivered 815.3, so
> the prediction is **wrong by 1.5%** — recorded as a miss, not rounded into a hit.
> The honest reading is that the *direction and rough size were right* and the
> threshold was set on makani's 2-node number (~3.3×) without accounting for the
> protocol pin the cxi stack requires.
>
> ⚠ **This is a three-variable change, not one.** plugin 1.21.1→v1.6.0 **and**
> `NCCL_PROTO=Simple` **and** HPE's rendezvous block, all at once. The fabric's own
> contribution is therefore **bounded below** by 1.477×: the two accompanying
> changes are both *costs* (LL/LL128 disabled; ~12–18% of all_reduce bandwidth).
> P15's 1-node arm is what separates them — it has the same two costs and **no**
> fabric — and until it lands, "cxi is worth 1.48× to ACE2" is a statement about
> the whole stack.
> ⚠ **n=1.** The tcp 2-node rung's own rep spread was **±3.8%** over 3 reps, so a
> single cxi row cannot be quoted to three digits. Reps are the next cheap thing.

tcp 2-node: **1204.4 ms** (median of n=3, local_batch 2) ⇒ P12 asserts **< 803 ms**.

makani got **2.47×** at 4 nodes (460.5 → 186.4 ms) and ~3.3× at 2. ACE2's exposed
inter-node cost is +488.4 ms of a 1204.4 ms step (§1b P4), and cxi measured
**5.2× tcp** app-free (7629082, 18.03 vs 3.46 GB/s), so the toll should fall by
roughly that factor: 716.0 + 488.4/5.2 ≈ **810 ms**, i.e. the prediction sits
deliberately just *below* the naive estimate.

⚠ The estimate is not clean, and this is the interesting part: the cxi
configuration also pins `NCCL_PROTO=Simple`, which **disables LL/LL128** —
a slower *intra-node* path that the fabric win has to pay for before it shows.
Its size is unsettled: the quoted −26% on makani's 1-node arm (114.9 →
144.7 ms/step) is flagged in CHANGELOG 2026-09-17 as conflating the pin with the
tcp fallback, and HPE's rendezvous block costs a further 12–18% of all_reduce
bandwidth. If P12 misses, check the 1-node arm (P15) before blaming the fabric.

### P13 — the Tree all-reduce **still fails** at 2000 MB on cxi — ❌ **FALSIFIED (job 7631550: Tree is clean on cxi, and 21% FASTER than Ring)**

> **Scored 2026-09-18.** `ACE2_NCCL_TESTS_OK arms=6/6`, all six on `provider=cxi`,
> **every row `#wrong = 0`**, every arm carrying nccl-tests' own
> `Out of bounds values : 0 OK` terminator — i.e. no corruption *and* no hang, at
> 256 MB / 512 MB / 1 GiB / 2 GiB **and** at ACE2's exact 1,823,324,160 B
> collective, in-place and out-of-place.
>
> | arm | avg busbw (GB/s) | rows | `#wrong` | finished |
> |---|---|---|---|---|
> | `intra_Tree` (1 node, no fabric) | 138.43 | 5 | all 0 | ✅ |
> | `intra_Ring` (1 node, no fabric) | **187.04** | 5 | all 0 | ✅ |
> | `inter_Tree` (2 nodes) | **42.79** | 5 | all 0 | ✅ |
> | `inter_Ring` (2 nodes) | 35.37 | 5 | all 0 | ✅ |
> | `ace2_startup_Tree` (1.74 GiB exact) | **44.03** | 1 | 0 | ✅ |
> | `ace2_startup_Ring` (1.74 GiB exact) | 35.78 | 1 | 0 | ✅ |
>
> ⇒ **The tcp-era failure was possibility (1): a defect in the path ACE2 was
> actually using (aws-ofi-nccl 1.21.1 over tcp), not an NCCL Tree bug.** The
> intra-node control arms say the same thing from the other side — Tree is correct
> with no fabric in the path at all.
>
> 🟢 **CONFIRMED AT 8 NODES TOO — job 7631624, 32 ranks, `arms=6/6`, split 0,
> `Exit_status 0`, 53 s.** Every row `#wrong = 0`, every arm terminated. The
> original tcp defect showed at 2 nodes *and* 8; **both node counts are now clean
> on cxi**, so the correctness case for the pin is gone.
>
> ⚠ **AND THE PERFORMANCE ANSWER REVERSES WITH SCALE — which is the finding.**
>
> | inter-node avg busbw (GB/s) | Tree | Ring | winner |
> |---|---|---|---|
> | **2 nodes** / 8 ranks (7631550) | **42.79** | 35.37 | **Tree +21%** |
> | **8 nodes** / 32 ranks (7631624) | 28.90 | **37.00** | **Ring +28%** |
> | ACE2's 1.74 GiB, 2 nodes | **44.03** | 35.78 | Tree +23% |
> | ACE2's 1.74 GiB, 8 nodes | 29.48 | **37.55** | Ring +27% |
>
> At 8 nodes Ring wins by **25–29% at every size** in the sweep (128 MiB → 2 GiB),
> not just on the average. The crossover therefore sits **between 2 and 8 nodes**,
> and the mechanism is the one §1a already inferred for why Ring escaped the defect:
> Tree's per-link message does not shrink with N, while Ring's reduce-scatter hands
> each rank `S/N`. Inside a node Ring leads throughout (193 vs 141 GB/s).
>
> ⇒ **The pin is not simply wrong — it is right at 8 nodes and costs ~21% of
> collective bandwidth at 2**, which is ACE2's production shape at global batch 16.
> Translated through the 2-node toll (101.6 ms of an 815 ms step), Tree there is
> worth roughly **2% of step time** — real, modest, and not worth a correctness
> risk taken carelessly.
> ⇒ So the move is **not "pin Tree"** but **unpin** (`-v NCCL_ALGO=`) and let NCCL's
> own tuner pick per size and per scale — it is the thing that knows this
> crossover, and forcing one algorithm everywhere is what guarantees being wrong at
> one end.
>
> ⚠ **STILL GATED, and two of the three original reasons stand:**
> 1. ~~2 nodes only~~ — **satisfied** by 7631624 (2 and 8 both clean).
> 2. **nccl-tests is not the trainer.** ACE2's exposure is a DDP all_reduce inside
>    fme with ~10 other collectives in flight; the probe drives one communicator
>    with nothing else running. The makani wedge that started all of this was a
>    *broadcast* inside DDP setup that no app-free probe ever reproduced.
> 3. It is a **hot-path change**, so DESIGN §4 applies: a captured equivalence
>    baseline, not a bandwidth table. Changing the algorithm changes the reduction
>    order, and ACE2 has **no equivalence baseline yet** (TODO item 14).
> ⚠ Each arm is **n=1**.

#### P13 as registered, 2026-09-17 (kept verbatim — the reasoning is what was wrong)

A MISS here is the more useful outcome: it would mean the defect was
1.21.1's tcp path and `NCCL_ALGO=Ring` can be dropped, giving ACE2 back Tree's
small-message latency. A HIT means the defect is NCCL-side and §1a of the
multi-node handoff must be re-titled rather than deleted.

Stated as a HIT because the corruption was reproduced at **two different node
counts** (2 and 8) and at three sizes, which reads more like an algorithm bug
than a transport bug — but this is a genuine guess, which is why the probe runs
an **intra-node control arm** (`polaris_ace2_tree_probe.pbs`): Tree corrupting
with no fabric in the path settles it without any inter-node byte at all.

### P14 — `wireup_s` / time-to-first-step **increases** vs the tcp rows — ⚪ **UNSCORED (the quantity is not recorded)**

> **2026-09-18.** As registered, this is **UNSCORABLE, not a hit**: ACE2's CSV has
> no wireup column and nothing in the harness times "up to the first step".
>
> The one adjacent quantity that *is* recorded moved the **opposite** way — within
> the timed window, `epoch_wall_s − n_steps × step_med` (loader/other idle, **not**
> wireup) went 7.31 → 4.57 s at 1 node and 7.88 → 6.26 s at 2. That is a different
> quantity measured after wireup has already happened, so it neither scores P14 nor
> contradicts it; it is recorded so the next person does not mistake it for an
> answer. Closing P14 properly needs a timestamp at `init_process_group` return,
> which is a telemetry change, not a measurement.

#### P14 as registered, 2026-09-17

It did on makani (8.54 → 14.18 s at 2 nodes). Mechanism unknown — CXI memory
registration is the suspect — so this is a guess, registered because a startup
regression hiding inside a step-time win is exactly what a ladder table does not
show. ⚠ ACE2's CSV has **no wireup column**; this is scored off `epoch_wall_s`
minus Σ step time, or the log timestamps, and if it cannot be read that way the
prediction is **UNSCORABLE**, not a hit.

### P15 — the **1-node** step time gets *worse*, by 5–30% — ❌ **FALSIFIED (job 7631544: 713.693 ms, −0.3%)**

> **Scored 2026-09-18, n=1.** `provider=cxi`, `NCCL_PROTO=Simple`, rendezvous block
> on. **713.693 ms against the tcp anchor's 716.0** — a 0.3% *improvement*, i.e. no
> change at all, against a predicted 752–931 ms. `gpu_busy_frac` 0.9357 → 0.9660;
> `peak_mem_gb` identical at 33.959.
>
> **The protocol pin costs ACE2 nothing measurable**, and that is the useful part:
> makani's quoted −26% for `Simple` does **not** transfer. The mechanism is
> consistent — LL/LL128 are small-message protocols, and ACE2's intra-node traffic
> is a 1.74 GiB full-model all_reduce plus ~11 buckets of ~165 MB, all far above
> the sizes where those protocols pay. It is also further evidence for CHANGELOG
> 2026-09-17's reading that the "26%" conflated the pin with the tcp fallback.
>
> ⇒ **P12's caveat tightens: the 2-node 1.477× is essentially all fabric.** The two
> accompanying changes were bounded as "costs of unknown size"; measured at the one
> node count where they act alone, their size is **≈0**.

tcp 1-node: **716.0 ms** ⇒ P15 asserts **752–931 ms**.

1 node never leaves NVLink, so the plugin change is inert there — but
`NCCL_PROTO=Simple` is not, and neither is HPE's rendezvous block. If the anchor
moves, every "cost of the fabric" figure in §1b Table 9 is re-based, and the
comparison that matters (2n cxi vs 2n tcp) is *not* affected. Registered
separately from P12 so a compute-side regression cannot be absorbed silently into
a fabric-side win.

### 1c-bis. The 16-node comparison — unregistered, and it reframes §1c

Run 2026-09-18 on operator request (7633410 vs 7633560), **without a prediction
registered first** — recorded as such, because the repo's method is to predict
before measuring and this arm did not.

| | 1 node / 4 GPUs | **16 nodes / 64 GPUs** |
|---|---|---|
| global batch | 8 | **128** |
| `step_med_ms` | 716.195 | **746.469 (+4.2%)** |
| samples/s total | 11.17 | **171.47** = **15.35×** for 16× the GPUs |
| weak-scaling efficiency | — | **95.9%** |
| node·s per sample | 0.08952 | 0.09331 (**+4.2%**) |
| `gpu_busy_frac` | 0.9431 | 0.9412 |

**What it changes:** P12's 2-node arm turns out to be the *worst* multi-node rung,
not a representative one — 2 nodes (815.3 ms) is **9.2% slower than 16 nodes**
(746.5). The ladder is **non-monotonic**, which is §1e's 2-node trough on a third
unrelated model, and the margin exceeds that rung's own ±3.8% spread. So the
"first-hop toll" framing that this whole file inherited from the tcp era describes
the trough, not the fabric: per-rank message size falls as `S/N` under Ring while
inter-node busbw holds (35.37 → 37.00 GB/s, 2 → 8 nodes), so the penalty is worst
at the smallest multi-node world and nearly gone by 16.

⇒ **The case for 1-node production is now purely about BATCH SIZE.** The fabric
costs 4.2% at 64 GPUs; the batch costs 16× the updates (updates/node-hour
5,027 → 301). Whether a 16× larger batch with a rescaled LR reaches the same loss
per sample is a **critical-batch-size question this campaign has never measured** —
and it is now the single highest-value open experiment for ACE2, worth more than
any remaining fabric work.

⚠ n=1 at 16 nodes; the 4- and 8-node cxi rungs do not exist, so the shape between
2 and 16 is unmeasured and known to be non-monotonic; and the 16-node arm ran on a
**different rack** (`x3201c0s*`) from the 1- and 2-node arms (`x3001c0s*`) — an
uncontrolled confound that a node-matched rep would remove.

## 1e. CRITICAL BATCH SIZE — predictions written 2026-09-19, BEFORE the arms ran

§1c-bis left exactly one question standing between ACE2 and a **14× wall-clock**
speedup: the fabric costs 4.2% at 64 GPUs, so the only penalty for scaling out is
that the global batch scales with it. **Does global batch 128 with a rescaled LR
reach the same loss per sample as global batch 8?**

**Design — matched SAMPLES, not matched steps.** 3 full epochs each (the same
horizon the LR sweep used), flat LR (`NO_SCHEDULER=1`, as the sweep did),
`FULL_VAL=1` so validation loss means the same thing in every arm. Batch 8 sees
36,702 updates; batch 128 sees 2,292 — that 16× gap **is** the thing under test.

| arm | nodes | global batch | LR | rule |
|---|---|---|---|---|
| A | 1 | 8 | 3e-4 | the measured winner at this batch (0.19579) |
| B1 | 16 | 128 | 3e-4 | control: batch changed, LR held |
| B2 | 16 | 128 | 1.2e-3 | **√-scaling** (×4) |
| B3 | 16 | 128 | 4.8e-3 | **linear scaling** (×16) |

⚠ **A SHORT ARM CANNOT ANSWER THIS, and that is measured, not assumed.** The
3,000-update LR screen picked 1e-3, which the full 3-epoch sweep ranked **worst of
four**. So the cheap version of this experiment is known to invert. 3 full epochs
is the minimum defensible horizon, which is why arm A costs ~9 h.

### 1e SCORED, 2026-09-19 — ✅ P17 HIT, ✅ P18 HIT, ✅ P19 HIT. **The batch question is closed: 1 node stands.**

| arm | ep 1 | ep 2 | ep 3 | job |
|---|---|---|---|---|
| batch 8, LR 3e-4 (reference) | 0.2846 | 0.2205 | **0.19579** | 7589850 → 7598647 |
| batch 128, LR 3e-4 | 0.8717 | 0.6052 | **0.5017** | 7633624 → resumed 7633879 |
| batch 128, LR 1.2e-3 (√-scaling) | 0.7406 | **0.5126** | — | 7633846 |
| batch 128, LR 4.8e-3 (linear) | 2.1755 | **1.7993** | — | 7633847 |

**The selection rule fires unambiguously.** Best batch-128 arm is ~0.50 against the
reference's 0.19579 — **156% worse**, against a ">10% worse ⇒ 1 node stands"
threshold. **Production stays at 1 node, global batch 8, LR 3e-4.**

* **P17 (>10% worse) — HIT, and by a margin that makes the caveats irrelevant.** The
  gap is **2.6×**, so it cannot be explained by the reference having been measured
  in an earlier env: 1-node runs have no fabric, ACE2 is bitwise deterministic at
  fixed seed, and no plausible environmental effect is worth 160%.
* **P18 (1.2e-3 wins among the B arms) — HIT.** At matched epochs, 1.2e-3 (0.5126)
  beats 3e-4 (0.6052) and demolishes 4.8e-3 (1.7993). √-scaling was the right rule
  and linear was not, as makani's batch-independent LR ceiling suggested.
* **P19 (4.8e-3 worse than 0.34571) — HIT on the criterion, with the mechanism
  wrong.** It is 1.7993, far past the threshold — but it did **not** diverge or
  NaN; it descends steadily (2.1755 → 1.7993), i.e. it is *badly suboptimal*
  rather than *collapsed*. Recorded as a hit whose stated mechanism failed.

⇒ **THE 16-NODE SPEEDUP IS REAL AND UNUSABLE FOR THIS MODEL.** 95.9% weak-scaling
efficiency and 15.35× throughput are correctly measured — but the batch that comes
with them costs **2.6× in validation loss at matched samples**, which no throughput
factor recovers. The fabric was never the thing standing between ACE2 and a faster
good model; the batch was, and it still is.

⚠ **Two arms stopped at 2 of the registered 3 epochs** (walltime, even at `small`'s
3 h cap — the epoch-1 I/O penalty again). Their third epochs were **not** bought,
because no third epoch closes a 2.6× gap and the decision was already determined.
The horizon is therefore complete for one B arm and short for two — stated rather
than smoothed over.
⚠ The matched-conditions reference arm (7633848) never started, so the comparison
is against the **historical** 0.19579. See P17 for why that does not threaten the
conclusion.
⚠ Every arm is n=1 at one init. This ranks configurations; it does not size the gap
to better than ACE2's unmeasured init noise.

### P17 — batch 128 at its best LR is **>10% worse** than batch 8 at 3e-4

Batch 8's reference is **0.19579** ⇒ P17 asserts the best B arm lands **above
0.2154**. Mechanism: 16× fewer updates, and ACE2's own sweep says the LR optimum
is narrow — **1e-3 was already the worst of four arms at batch 8** (0.34571), so
the critical batch size is plausibly *below* 128. If P17 misses, ACE2 production
should move to 16 nodes and finish 27 epochs in ~4.6 h instead of 65.5.

### P18 — the best batch-128 LR is **1.2e-3** (√-scaling), not 3e-4 or 4.8e-3

√-scaling is the better-supported rule for Adam-family optimizers; linear scaling
was derived for SGD+momentum. A genuine guess: the repo has no ACE2 LR-vs-batch
data at all.

### P19 — 4.8e-3 (linear) **diverges or collapses** — worse than 0.34571

makani's LR ceiling is (2e-3, 3e-3] and **did not move with batch size** across
9 arms, 5 of which collapsed. 4.8e-3 is above that ceiling. Registered because a
collapse *bounds* the usable LR at batch 128, which a null result cannot.

**Selection rule, fixed now:** winner = **lowest final EMA validation loss at 3
full epochs**. If batch 128's best is **within 5%** of 0.19579, production moves
to 16 nodes (14× wall-clock at +4% node-hours). If it is **>10% worse**, 1 node
stands and the batch question is closed. Between 5% and 10% is **no decision** —
it would need the 4-node and 8-node intermediate batches, not a coin flip.
⚠ Every arm is **n=1 at one init**, so this ranks configurations; it does not
measure the size of the gap to better than init noise, which for ACE2 is
**unmeasured** (makani's two-seed calibration suggests O(0.1 pp)).

### §1c scorecard as of 2026-09-18 (6 cxi arms: 1n ×2, 2n, 16n, and the 6-arm Tree probe)

| # | prediction | outcome |
|---|---|---|
| P12 | 2-node > 1.5× faster on cxi | ❌ **missed by 1.5%** — 1.477× (815.3 ms vs a 803 ms bar) |
| P13 | Tree still fails at 2000 MB on cxi | ❌ **falsified** — 6/6 arms clean, `#wrong = 0` everywhere; it was 1.21.1's **tcp** path |
| P14 | `wireup_s` increases | ⚪ **unscored** — the quantity is not recorded anywhere |
| P15 | the 1-node arm gets 5–30% worse | ❌ **falsified** — −0.3%; `NCCL_PROTO=Simple` costs ACE2 nothing |
| P16 | the LR-3e-4 conclusion is unchanged | ✅ **holds** — untouched, and nothing here bears on it |

**Three of three scored predictions were wrong, and two of the three were wrong in
the direction that removes work:** the protocol pin is free (P15), so the 1.477×
belongs to the fabric; and the Tree defect is a tcp artefact (P13), so the
`NCCL_ALGO=Ring` question is now a live 21% rather than a closed safety matter.
P12's miss is the least interesting — the threshold was borrowed from makani
without allowing for ACE2's different collective shape.

⚠ **What the scorecard must not be read as saying.** Every cxi row is **n=1**, only
the 1- and 2-node rungs exist, and the Tree probe ran at **2 nodes** where the
original defect showed at 2 *and* 8. Nothing here licenses dropping the Ring pin.

⚠ **P16 is not a prediction, it is a gate**: the LR-3e-4 conclusion (§1a) must be
**unchanged**. Jobs 7598647/7598648 were 1-node runs with no inter-node traffic,
and loss is a numerical result that does not move with the wire. Scoring it exists
to stop a settled result being quietly re-opened by a fabric finding.

## 1d. What the TCP finding does NOT invalidate

Recorded before the re-measurement so that nothing is re-litigated after it:

* **The LR sweep (3e-4).** 1-node runs, no fabric. Untouched.
* **P9 — I/O is not the bottleneck.** `gpu_busy_frac` 0.970 and 1.64 GB/s off the
  single OST are properties of the loader and the store; a slower wire makes the
  GPU *less* starved, not more, so the conclusion is if anything conservative.
  ⚠ The corollary — the zarr conversion is not justified — should be re-checked
  once steps get faster: the same I/O rate against a shorter step is a larger
  fraction of it.
* **P7 — the batch ceiling (local 2, no cliff).** Memory, not comms.
* **P1's direction and P11.** Both intra-node.
* **§1b's shape argument** (2 nodes is a per-GPU trough; "fewest GPUs that hold
  the batch" is the first question) — that is a *batch-size* argument. Its
  magnitudes were computed against tcp step times and every one of them moves.

Everything else in §1b that involves more than one node is a **tcp measurement**
and must be labelled as such wherever it is quoted, not deleted: it is a real
measurement of a configuration we ran for three weeks.

---

## 2. What would make the whole table invalid

Recorded so that a green-looking CSV cannot be tabled past any of these:

* `world_sizes_seen` ≠ ranks — the rank shim did not apply and fme silently built
  `NonDistributed`, i.e. N independent single-GPU trainers.
* `ranks_reporting` ≠ ranks — a rank died before the banner.
* `provider` ≠ `cxi` on a multi-node arm — the job did not cross Slingshot.
  ⚠ **This bullet used to read `transport` not `AWS Libfabric`, and that check
  could not fail.** The plugin prints that string whether it bound the fabric or
  fell back to tcp, so every multi-node arm in §1b satisfied it while running on
  Ethernet, for three weeks (CLAUDE.md #10; handoff §8). `transport` is retained
  as a label; `provider` is the guard, gated on nodes > 1 because a 1-node run
  never initialises the net plugin.
* `n_steps` ≠ requested — a walltime-truncated arm is not comparable to a full one.
* Arms not interleaved. Two runs of an identical config once measured 42.2% vs
  37.4% for the same quantity (CHANGELOG §4.4c).
* Arms run against the live makani production job on the same filesystem. Three
  concurrent 1-node arms cost that job **+2.3% median epoch wall**; at ACE2's I/O
  shape the effect could be much larger, in both directions.

All four of the first are enforced by `parse_ace2_scaling.py` and tested by
`test_parse_ace2_scaling.py` (23 tests); the last two are operator discipline.

⚠ **The test to apply to any future check: what value of this field would make
the job fail?** If there is no such value it is a label, not a guard — which is
what `transport` was, and why a pre-registered prediction once scored a HIT
against it.

---

## 3. Selection rule, written before the arms exist

If an LR arm is needed (a larger global batch is a numerics change and the LR
moves with it), the winner is the arm with the **lowest validation loss**, ties
broken by lower gradient norm.

⚠ **Gradient norm is not a health signal on its own.** Measured on makani
2026-09-02: in a 4-arm LR sweep the **worst** arm (4e-3, validation 0.10703 —
4.5× worse than the winner) had the **lowest** gradient norm of the four
(0.00913). A falling grad norm is equally consistent with healthy convergence and
with a collapsed optimizer. Read it *with* the loss, never instead of it.

⚠ ACE2 needs no `FLAT_LR` equivalent. fme's default `SchedulerConfig.type` is
`None` and `config_polaris.yaml` sets no scheduler, so the LR is **flat at 1e-4**
for the whole run and `-v LR=` alone is a valid arm. This corrects the multi-node
handoff §3.1, which says "ACE2 anneals its own LR, so pin the schedule flat for
the sweep".

⚠ The batch and LR used by the *ladder* arms are not a proposal for production:
those arms are 60 timed steps and are thrown away. The production values are
settled separately in §1a, and as of **2026-09-18 they are an operator decision
(rmehta1987), adopted and not gated on external sign-off** — global batch 8,
LR 3e-4, 27 epochs, `use_gradient_accumulation=false`, warm restarts T_0=9.
They remain departures from ai2's published recipe and are listed as such.
