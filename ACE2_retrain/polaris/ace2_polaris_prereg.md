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

## 2. What would make the whole table invalid

Recorded so that a green-looking CSV cannot be tabled past any of these:

* `world_sizes_seen` ≠ ranks — the rank shim did not apply and fme silently built
  `NonDistributed`, i.e. N independent single-GPU trainers.
* `ranks_reporting` ≠ ranks — a rank died before the banner.
* `transport` not `AWS Libfabric` — a silent fallback reads exactly like
  "Slingshot is slow".
* `n_steps` ≠ requested — a walltime-truncated arm is not comparable to a full one.
* Arms not interleaved. Two runs of an identical config once measured 42.2% vs
  37.4% for the same quantity (CHANGELOG §4.4c).
* Arms run against the live makani production job on the same filesystem. Three
  concurrent 1-node arms cost that job **+2.3% median epoch wall**; at ACE2's I/O
  shape the effect could be much larger, in both directions.

All five of the first are enforced by `parse_ace2_scaling.py` and tested by
`test_parse_ace2_scaling.py`; the last two are operator discipline.

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

⚠ Changing the batch or the LR for **production** is a training-regime change and
is jesswan's call. These arms are 60 timed steps and are thrown away.
