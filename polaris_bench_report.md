# Polaris (A100) profiling report — PanguWeather SFNO

The Polaris analog of `s2s/v2.0/bench_report.md` and `si/bench_midway_notes.md`: what the
profiling phase measured on **4× A100-SXM4-40GB**, how it was measured, and which of it is
trustworthy. Style follows `si/bench_midway_notes.md` — narrative + a dated decisions log.

See **CLAUDE.md** for how to work here, **DESIGN.md** for what/why, **CHANGELOG.md** for
cross-cutting status, and **`polaris_pbs_notes.md`** for the cluster facts and bring-up traps.

> ## ⚠ These numbers are NOT comparable with Midway's
> Midway's `bench_results.csv` and `bench_report.md` are **H100 NVL (~94 GB, PCIe Gen4,
> NVLink within socket-pairs)**. This is **A100-SXM4-40GB (AMD Milan host)**. A different
> node class. Per DESIGN §1 non-goals, a slower A100 step is *expected*, not a regression —
> and the two must never share a table. Nothing below is compared against Midway. The
> *method* (`S2S_BENCH`, warmup 20 / steps 80, the same NVTX range names) is deliberately
> identical so that future within-cluster comparisons are valid.

---

## 1. What was profiled, and why only this

**PanguWeather SFNO on E3SM** (`nettype: sfno_plasim`, `--config=SFNO`), the model the
2026-07-15 focus change points at (DESIGN §2c) and the only one of the S2S family that runs
on Polaris today — `s2s/v2.0` and the Lightning port are still blocked on the ERA5 Globus
stage.

Profiling it **began by building the instrumentation**: PanguWeather carried **zero** NVTX
ranges and no `S2S_BENCH` harness, so there was nothing to measure with. That port is the
bulk of this session's work; the range names and CSV columns are byte-identical to s2s's on
purpose (CLAUDE.md #10).

**SI / makani / physicsnemo are NOT profiled here yet.** The handoff asks for all four. This
report covers one, honestly, rather than four superficially — the instrumentation port,
its regression proof, and the two DESIGN §4.0 prerequisites consumed the session. SI already
has its own `SI_BENCH_*`/`SI_NVTX` harness and a green Polaris bench (7252700/7253603), so it
is the cheapest next one; makani and physicsnemo have no comparable harness at all.

### Configuration under test

| | |
|---|---|
| Model | PanguWeather `sfno_plasim`, **1,182,108,160 params** (measured, job 7255410) |
| Data | staged E3SM v3 SSP245-AMIP, 1460 samples/yr, 180×360, 18 sigma levels |
| Shape | 4 ranks × batch 1, bf16 autocast (no GradScaler), DDP `find_unused_parameters=False` |
| Activation checkpointing | `checkpointing: 3` (**on** — recompute in backward) |
| EMA | `use_ema: True` but `ema_warmup_epochs: 6`, so **inactive** in the 1-epoch runs below |
| Window | warmup 20 / measured 80 (the Midway convention), eager — **no `torch.compile`** |

> **1.18 billion parameters, not ~79M.** DESIGN and CLAUDE.md both describe the S2S family as
> "~79M-param" (the `test.yaml` trap, rule #12). That figure is about the *Pangu/Swin* model.
> The E3SM **SFNO** at `embed_dim: 512, num_layers: 12, num_blocks: 16` is **15× larger**.
> Any resource intuition carried over from the 79M figure is wrong for this path.

---

## 2. The headline: GPU-bound, and bound by *elementwise* work

The handoff asks for the cheap, high-value question first — **is the hot path GPU-bound or
input-bound?** If it were input-bound, every rung of the DESIGN §5 kernel ladder would be
premature.

**It is GPU-bound.** At the shipped `num_data_workers: 1`, **0.7%** of loop wall time is
spent waiting on the data loader. The kernel ladder is not premature.

And the profile says *which* kernels: **61% of GPU time is elementwise/pointwise**, spread
over **~1506 launches per step**. Only **15%** is tensor-core matmul. That is the signature of
a **memory-bandwidth-bound, fusion-starved** model — which is precisely what `torch.compile`
(DESIGN §5 rung 1) exists to fix. The ladder's existing ordering is now **measured**, not
assumed.

---

## 3. Data-loader sweep — and the metric that had to be built to answer it

`num_data_workers` sweep, everything else identical (4×A100, bf16, batch 1/GPU, eager,
warmup 20 / steps 80). Full rows: `$MEMBER_ROOT/bench/pangu_sfno_polaris_bench.csv`.

| workers | job | step_med | step_p90 | step_std | loader_wait_med | **loader_wait_frac** | `samples_per_s` (step rate) | **WALL samples/s** | peak mem |
|---|---|---|---|---|---|---|---|---|---|
| **1** (shipped) | 7255410 | 0.652 s | 0.826 s | 0.110 | 0.0002 s | **0.7%** | 6.13 | **6.09** | 26.98 GB |
| 0 | 7255434 | 0.615 s | 0.631 s | 0.011 | 0.1042 s | **14.8%** | 6.50 | **5.53** | 26.98 GB |
| 8 | 7255480 | 0.602 s | 0.603 s | 0.0003 | 0.0002 s | **0.0%** | 6.64 | **6.64** | 26.98 GB |

### `samples_per_s` is a step rate, not throughput — read this before quoting it

`samples_per_s = global_batch / step_med`, and `step_med` **excludes the between-step loader
fetch**. So it overstates throughput exactly when the loader is the problem. Convert:

```
wall throughput = samples_per_s × (1 − loader_wait_frac)
```

At `workers=0` the CSV says **6.50** while the real rate is **5.53**. Comparing that 6.50
against `workers=1`'s 6.13 would have ranked the **slower** configuration first. The column
name is s2s's and is kept (CLAUDE.md #10); the conversion is the fix.

### Why a new metric was needed at all

`cpu_prep_frac` — the closest thing the inherited harness had — is **not** the data-loader
idle fraction, and reading it as one is a trap:

* `cpu_prep_med` times `_prepare_inputs_batch`, which runs on a batch the loader has
  **already produced** (H2D + reshape). It is 0.002–0.004 s, i.e. **0.3–0.6%** of the step,
  in all three runs — *including the deliberately starved one*.
* The blocking fetch happens inside the loader's `__next__`, **between** steps, inside no
  step window at all. It is invisible to every column s2s writes.

Worse, it was **fatal**: `_bench_finalize` reconciles `elapsed` against `sum(step_times)`,
which holds only while the loader keeps ahead of the GPU. On an input-bound run the gap
lands in `elapsed`, the 10% self-check fires, and the row is **refused** — the harness
aborts precisely when the loader is the finding. Now measured explicitly
(`loader_wait_med` / `loader_wait_frac`, appended after s2s's 19 columns) and folded into
the self-check, which makes it *tighter*: any residual >10% is a genuine timer bug again.

### The metric was falsified before it was believed

A metric that cannot move proves nothing. `workers=0` forces the fetch to happen
synchronously inside `__next__`, so the gap **must** appear:

> **0.7% → 14.8%, a 21× move**, with `cpu_prep_frac` flat at 0.3→0.4% throughout.

So the "GPU-bound" verdict is a measurement, not a dead metric reading zero.

### What the sweep actually shows

* **The shipped `num_data_workers: 1` is not the bottleneck, but it is not free either.**
  It costs ~5% of step time (0.652 vs 0.602 s) and **10× the jitter** (step_std 0.110 vs
  0.0003; p90 0.826 vs 0.603). The single worker's HDF5 read runs *concurrently* with the
  step and contends with the rank's main thread for CPU. With 8 workers each batch is
  prepared 8 step-times ahead, so the read is fully hidden and the step is almost perfectly
  regular (p90 − med = **0.3 ms**).
* **`workers=0` is the worst of both**: the cleanest step (no contention → std 0.011) but
  0.104 s of dead time per step. This is the documented "`num_data_workers=0` fakes a
  GPU-idle bottleneck" trap (CHANGELOG) showing up quantitatively.
* **`workers: 1 → 8` is +9% wall throughput (6.09 → 6.64 samples/s) and removes the jitter.**

> ### ⚠ …but `num_data_workers` is NOT an output-neutral knob here. Do not just bump it.
> `utils/data_loader_multifiles.py:1031/1102` draws `torch.randn(*surface_t.shape)` per
> sample whenever `epsilon_factor > 0` — and the config sets `epsilon_factor: 0.1`, so the
> **loader adds noise inside the worker processes**. There is **no `worker_init_fn`**, so
> workers are seeded by PyTorch's default `base_seed + worker_id` and the sample→worker
> assignment depends on `num_workers`. Changing it therefore changes *which* noise each
> sample gets and **moves the loss trajectory**.
>
> The change is statistically benign (the noise is iid gaussian either way) but it is **not**
> bit-identical, so it cannot be validated by the DESIGN §4 equivalence gate — it needs a
> distributional argument. Recorded as a **finding, not a recommendation**. The clean fix is
> a seeded `worker_init_fn`, which would make the noise depend on the sample rather than on
> the worker count, and would make the knob genuinely free.

---

## 4. Where the step goes (nsys, job 7255503)

`nsys` on Polaris is **not** a module — it ships with the CUDA toolkit at
`/soft/compilers/cudatoolkit/cuda-12.9.1/bin/nsys` (Nsight Systems 2025.1.3).
Trace + sqlite: `$MEMBER_ROOT/bench/nsys_pangu_sfno_7255503.{nsys-rep,sqlite}`.
40 measured steps × 4 ranks = 160 rank-steps, eager, `--capture-range=cudaProfilerApi`.

### 4.1 NVTX ranges — **CPU-side; they do not sum to the step**

| range | n | median | mean | min | max |
|---|---|---|---|---|---|
| `data_prep` | 160 | 0.2 ms | 0.2 | 0.1 | 0.4 |
| `forward_loss` | 160 | 36.3 ms | 39.4 | 35.3 | 268.4 |
| `backward` | 160 | **280.8 ms** | 280.5 | 250.5 | 297.2 |
| `optimizer` | 160 | 18.0 ms | 18.0 | 17.1 | 18.7 |
| **step total** | 156 | **603.5 ms** | 608.7 | — | std 31.9 |

**Read this carefully.** The sub-ranges sum to **335 ms of a 603 ms step — 55%.** The other
45% is **not** missing work: these ranges are pushed/popped on the **CPU thread**, so they
measure *enqueue*, not GPU execution. CUDA is asynchronous; the trailing
`cuda.synchronize()` that closes the bench window drains whatever the GPU still owes, and
that drain sits inside `step_N` but inside none of the sub-ranges.

So: **do not read `backward = 280 ms` as "backward is 47% of GPU time."** It is 280 ms of
CPU *launch* work. Attributing GPU time requires the kernel table (§4.2), which is why both
are reported.

**§4.3 now does that attribution, and it confirms this paragraph while correcting its
number in the other direction:** the ~268 ms gap contains **zero kernel launches** (pure
drain, measured, not inferred — the CPU is inside `cudaDeviceSynchronize`), and the
CPU-side reading *understates* `backward`. Like-for-like against the 603.5 ms step:
**46.5% read CPU-side vs 67.7% read GPU-side** (`backward`'s 408.6 ms/rank-step **union**;
its summed kernel time of 467.0 ms is 72.6% of kernel time but overlaps itself 12.5% on
NCCL's stream, so it must not be divided into a step time — see §4.3b/§4.3d). Both shares
carry NCCL **wait**, so their magnitude is run-dependent: on the n=2 `backward`'s union
share moves 69.9% → 67.4% with self-overlap 12.5% → 18.5% (§4.4b). The *direction* — the
CPU-side reading understates `backward` — is unaffected. The CPU
issues forward work **3.8×** faster than the GPU retires it.

`ema` does not appear: `ema_warmup_epochs: 6` and these are 1-epoch runs, so EMA never
fired. **A full training run will pay it** — an every-step sweep over 1.18 B parameters —
and it is instrumented and waiting.

### 4.2 GPU kernel time — the actual finding

Aggregated from `CUPTI_ACTIVITY_KIND_KERNEL`, all 64 distinct kernels, normalised to one
rank-step (**102.9 s of kernel time over 354,720 launches / 160 rank-steps**):

| category | ms / rank-step | % GPU | launches / step |
|---|---|---|---|
| **elementwise / pointwise** | **392.0** | **61.0%** | **1506** |
| GEMM / tensor-core matmul | 97.0 | 15.1% | 351 |
| NCCL all-reduce (DDP grad sync) | 67.8 | 10.5% | 16 |
| normalization | 28.9 | 4.5% | 72 |
| optimizer (fused multi-tensor) | 25.9 | 4.0% | 59 |
| cuFFT (spherical harmonic transform) | 21.0 | 3.3% | 78 |
| other | 5.6 | 0.9% | 42 |
| reductions | 4.9 | 0.8% | 93 |
| **total** | **643.2** | 100% | **2217** |

643 ms of kernel time against a 603 ms step ⇒ **the GPU is saturated** (>100% because NCCL
overlaps compute on its own stream). Consistent with the loader verdict from the other
direction.

> ### ⚠ The `% GPU` column is not reproducible — quote `ms / rank-step`
> Every share above is taken against the full kernel total, **which contains NCCL wait**.
> §4.4c measures that denominator moving **+12.7%** between two runs of an identical config
> (on different nodes), which rescales all eight compute rows: the copies row alone reads
> **42.2% and 37.4%**. The `ms / rank-step` column is the durable one. Re-normalised to
> compute-only (575.4 ms/rank-step): elementwise **68.1%**, GEMM **16.9%**, copies
> **47.1%** — and note NCCL's own row is *mostly wait*, not transfer.

**The elementwise fraction is the story.** 61% of GPU time in pointwise kernels, 4× the
matmul time, at ~260 µs average — these are not launch-overhead-bound micro-kernels, they
are **large memory-bound passes**. The lever is **fewer round-trips to HBM**, not
faster matmul. ⚠ **But only half of it is a fusion target.** For the activation half,
fusion is the instrument (`torch.compile`, §5 rung 1); for the weight half — **133
ms/rank-step, ~21% of the step, §4.5c** — fusion does not apply and the lever is *hoisting a
layout* (DESIGN §5 rung 1b). ACE2 also measured `InductorError: KeyError: 'complex64'` on the
whole SFNO, so Inductor cannot reach this model's complex64 hot path, which is exactly where
the dominant copies live.

Also measured:
* **NCCL = 67.8 ms/step (10.5%)** over 16 calls — DDP gradient sync on a 1.18 B model.
  §4.3 splits it: **all-reduce 67.1 ms, 100% inside `backward`**, plus the 0.7 ms
  `broadcast_buffers` broadcast in `forward_loss`. A comm hook touches only the all-reduce.
  **Corrected sizing:** this 10.5% is a share of *kernel* time on NCCL's own stream, and
  plan §0b measures that stream as **88.7% overlapped with compute / 1.2% exposed** — so
  §5 rung 3's ceiling on one node is **~1.2% of wall-clock, not 5%**. It becomes interesting
  multi-node (plan item 12), not here.
* **cuFFT = 21.0 ms (3.3%)** — the spherical-harmonic transform is *not* a hotspot. Note
  `si/bench_midway_notes.md` §3–4's standing warning: the fp32 island around the SHT is
  deliberate and must not be "optimized" to bf16.
* **H2D: 962 transfers, 348.8 ms total, 8.38 GB** across the window ≈ 2.2 ms/rank-step,
  i.e. **~0.4% of the step**. Input transfer is not a problem.

---

### 4.3 Which phase owns the GPU time — the §4.1↔§4.2 join, done correctly

§4.1 gives CPU-side range times, §4.2 gives GPU kernel time, and until now nothing
connected them: the report could say "61% elementwise" and "backward is 280 ms of
*launch*" without being able to say **how much of the GPU time backward owns.** This
closes that, on the same capture, with no new GPU time
(`ACE2_retrain/nvtx_phase_attribution.py`, plan item 1).

Every number below was re-derived independently by an adversarial review pass, which
landed 15 strikes on the first draft of this section; the corrections are folded in and
the two it could not break are marked. **Read §4.3d before quoting any share against a
step time** — the first draft of this section made the sum-vs-union error that
`PANGU_POLARIS_PROFILING_PLAN.md` §0 lists as already-refuted, and got caught.

#### 4.3a Two things make this join wrong if done naively, and both had to be fixed

1. **`correlationId` is unique per PROCESS, not per capture.** One sqlite holds all four
   ranks, so the bare `RUNTIME.correlationId = KERNEL.correlationId` join cross-products
   them: **459,088 rows for 354,720 kernels, +29.4% phantom.** The guard is
   `KERNEL.globalPid = RUNTIME.globalTid & ~0xFFFFFF` (an nsys `globalTid` is
   `globalPid | tid` with the tid in the low 24 bits; verified — masking every RUNTIME
   `globalTid` reproduces the four KERNEL `globalPid`s exactly). With the guard the join
   returns **exactly 354,720 rows = one per kernel**, `COUNT(DISTINCT k.rowid)` = 354,720
   (nothing duplicated), and nothing is orphaned: all 354,720 rows are
   `launchType = REGULAR` with `graphNodeId IS NULL`, so there are no graph-launched or
   CDP kernels to fall out. Summed duration over the join equals `SUM(end-start)` over the
   whole KERNEL table to the nanosecond (102,910,943,542 ns). This independently
   reproduces the **+30.8%** measured on the Midway ACE2 capture (handoff §5), and it is
   the bug still live in `ACE2_retrain/kernel_census.py:58`.
2. **The launching thread is not the thread the NVTX range is on.** PyTorch's autograd
   engine launches from its own worker thread, so a range pushed on the main thread never
   contains the backward launches by thread identity: on rank 0, **62,680 of 88,680
   launches** come from `pt_autograd_*`. Thread-scoped attribution therefore credits
   `(outside)` for *the whole of backward* — which is exactly the origin of the "81% of GPU
   time lands outside any range" figure in `ACE2_retrain/bench_midway_notes.md`, and it was
   never an NVTX limitation. Attribution is scoped to the **process** instead, sound here
   because the four phase windows are non-overlapping per rank (asserted at load; nested
   ranges are refused rather than silently double-counted).

**Also settled: the NVTX text path.** The house ranges live in the inline
`NVTX_EVENTS.text` column (`domainId = 0`, `eventType = 59`), at 160 rows each = 40 steps ×
4 ranks. The `textId → StringIds` path holds **only** NCCL's registered strings
(`ncclAllReduce` 2402, `ncclBroadcast` 160, `domainId = 1`). Nothing of ours was ever
missing from the capture, and `parse_nsys.py`'s `WHERE text IN (…)` was already on the
right path. (Precisely: rank 0 carries **841** NVTX events in total; the **201** with
non-NULL `text` are all on the main thread, and 600 of the remainder are NCCL's own ranges
on the autograd worker. The conclusion — the *house* ranges carry no backward launches by
thread identity — is what matters and is unaffected.)

#### 4.3b GPU kernel time by phase — with sum AND union, because they differ

Launch-time attribution, all 4 ranks, normalised over 160 rank-steps. **`sum` counts a
kernel once per stream it runs on; `union` is wall-clock occupancy**, computed per rank and
then added across ranks (never unioned across devices):

| phase | launches/step | sum ms/rs | % of sum | **union ms/rs** | % of union | self-overlap |
|---|---|---|---|---|---|---|
| `backward` | 1568 | 466.95 | 72.6% | **408.63** | **69.9%** | **12.5%** |
| `forward_loss` | 590 | 150.32 | 23.4% | 150.32 | 25.7% | 0.0% |
| `optimizer` | 59 | 25.93 | 4.0% | 25.93 | 4.4% | 0.0% |
| `data_prep` | **0** | 0.00 | 0.0% | 0.00 | 0.0% | — |
| `(outside)` | **0** | 0.00 | 0.0% | 0.00 | 0.0% | — |
| total | 2217 | 643.19 | 100% | 584.88 | 100% | 9.1% |

The `sum` column reconciles with §4.2 exactly (2217 launches/step, 643.2 ms, 102.911 s), so
this is a **partition** of §4.2, not a second estimate of it.

**Only `backward` overlaps itself**, and the reason is measured: it carries 67.1
ms/rank-step of `ncclAllReduce` on `streamId 19` while compute runs on `streamId 7`. So
`forward_loss` and `optimizer` are union-safe and their numbers may be quoted directly;
**`backward`'s may not.**

**`(outside)` is 0.0% — every one of the 354,720 launches falls inside one of the four
phases.** That closes §4.1's open question about the "missing" 45% of the step: the ~268 ms
between `optimizer` ending and `step_N` ending (median 265.8 ms) contains **zero kernel
launches**, and on rank 0 the CPU spends 10.72 s of the capture inside
`cudaDeviceSynchronize` (119 calls) — it is a blocking drain, exactly as §4.1 argued, now
measured rather than inferred.

#### 4.3c The deliverable: the 42.2% copy time, split by phase

`direct_copy` + `conj` = **43.390 s = 42.2% of all GPU kernel time = 271.19 ms/rank-step**,
reproducing §0d of `PANGU_POLARIS_PROFILING_PLAN.md` exactly through an independent query
path (§0d's 121 / 111 / 38 ms rows re-derive as 121.40 / 111.50 / 38.30). **All 90,240 of
these kernels are on `streamId 7`, so their summed time IS their union** — this split is
union-safe and is not damaged by §4.3d:

| phase | kernel | launches/step | ms/rank-step | % of the copy time |
|---|---|---|---|---|
| `backward` | `direct_copy`⟨float, nocast⟩ | 246 | 82.28 | 30.3% |
| `backward` | `direct_copy`⟨complex64, nocast⟩ | 60 | 72.05 | 26.6% |
| `backward` | `conj`⟨complex64⟩ *(nocast + vectorized)* | 24 | 38.30 | 14.1% |
| `backward` | `direct_copy`⟨float, unrolled⟩ | 103 | 4.95 | 1.8% |
| `forward_loss` | `direct_copy`⟨complex64, nocast⟩ | 24 | 39.44 | 14.5% |
| `forward_loss` | `direct_copy`⟨float, nocast⟩ | 104 | 34.12 | 12.6% |
| `forward_loss` | `direct_copy`⟨float, unrolled⟩ | 3 | 0.05 | 0.0% |
| **`backward` total** | | **433** | **197.58** | **72.9%** |
| **`forward_loss` total** | | **131** | **73.61** | **27.1%** |
| `optimizer` / `data_prep` | | 0 | 0.00 | 0.0% |

**`backward` owns 72.9% of the copy time.** Two facts fall out:

* **`conj` fires only in `backward`** — 24/rank-step, 38.30 ms. **The warrant for calling it
  the adjoint is the source, not this table** (a kernel that only ran during *recompute*
  would look identical here): `grep -rn conj PanguWeather/v2.0/networks/modulus_sfno/`
  returns **nothing**, the spectral contraction is `torch.einsum` over `view_as_complex`
  operands (`contractions.py:29-31,59`; `s2convolutions.py:159,197`), and the backward of a
  complex einsum needs `x.conj()`/`w.conj()`. Corroborating: 24/rank-step = **2 ×
  `num_layers` (12)**. ⚠ **Not for the reason first given here:** §4.5b splits those 24 into
  **12 conjugates of the 377 MB spectral *weight*** and 12 of a small 16.68 MB activation, so
  it is *not* "one conjugate per operand per contraction" — that rationale is retracted
  (§4.5d). The conclusion holds: no checkpointing level removes it — and note a
  recompute-only kernel would merely be *relocated* by lowering checkpointing, not removed.
* **NCCL confirms the DDP model:** `ncclAllReduce` **100%** in `backward`, `ncclBroadcast`
  **100%** in `forward_loss` (0.7 ms/rank-step, the 0.11% `broadcast_buffers` cost of plan
  §0c). Cross-tabulated, there is **zero** cross-boundary leakage: `pt_autograd_*` →
  `backward` 250,720 of 250,720 launches; main thread → `forward_loss` 94,400, `optimizer`
  9,440, `backward` 160 (1/rank-step). No stream-callback or other launcher exists.

#### 4.3d What this bounds — ESTIMATED throughout, and the buckets are not the phases

At `checkpointing: 3` every SFNO block is wrapped
(`PanguWeather/v2.0/networks/modulus_sfno/sfnonet.py:692`); the levels are **cumulative**,
so `>= 2` *also* keeps the MLP wrapped (`…/layers.py:137`) and `>= 1` the encoder and
decoder (`sfnonet.py:704,731`). `backward`'s GPU time is therefore *recompute + adjoint*,
with no NVTX range between them — separating them is plan item **16** (SFNO-internal ranges
re-fire inside `backward` when `torch.utils.checkpoint` re-executes the wrapped module's
Python forward), **not** this item, and not item 17 (`--python-sampling` samples CPU stacks
and cannot partition GPU time at all).

**The recompute lives inside `backward`, so `backward` is the bucket that shrinks.** Stated
carefully, because the first draft of this section had it backwards:

| bucket | ms/rank-step of copy time | removable by lowering `checkpointing`? |
|---|---|---|
| `forward_loss` copies | 73.61 | **No** — the forward always runs |
| `backward`: recompute copies | **≈ 74.6 (est.)** | **Yes** — this is the only removable bucket |
| `backward`: adjoint copies | ≈ 123.0 (est.) | No |

So **≈27% of the 271.19 ms is removable and it is all inside `backward`'s 197.58 ms**; the
27% headline is unchanged from the first draft but the phase it names is now the right one.

> **⚠ What "recompute" contains — from §4.5, and it changes the class, not the number.**
> **35.61** of those 74.6 ms/rank-step (**47.7%**, measured — §4.5c) is the `ckpt3` pass
> **re-permuting the invariant 377 MB spectral weight**, not recomputing activations. The
> ≈27% stands
> arithmetically. But the converse is the more useful reading: **lowering `checkpointing`
> removes one of the two invariant weight permutes, whereas storing the parameter
> pre-permuted would remove both plus the gradient-side copy** — ≈97 ms/rank-step, ~16% of
> the step — at **no memory cost**, and the two compose. That candidate is not on the
> DESIGN §5 ladder; it is a hot-path *and* checkpoint-format change, so it is fully §4-gated
> and blocked on item 18.

**And "recompute ≤ the forward's GPU time" is NOT a bound — recompute is measurably more
expensive.** Isolating the kernels whose `forward_loss` and `backward` launch counts are
*equal* (the pure-recompute signature, no adjoint counterpart — `cudnn::bn_fw_tr`,
`regular_fft_c2r`, `GeluCUDAKernelImpl`, 7 kernels, 16.54 → 16.77 ms) gives a
backward/forward ratio of **1.0136 mean, 1.0105 median, min 1.0076 — never below 1.0**.
Recompute does not even run the same kernels: `cutlass_80_tensorop_bf16_s16816gemm_relu…`
runs 12/step in `forward_loss` and **0** in `backward`, while `ampere_s16816gemm_bf16_128x256…`
runs 0 in `forward_loss` and 12/step in `backward` at **+15% per call**. (A nested
double-recompute of the MLP was tested for and refuted: `GeluCUDAKernelImpl` runs 26/step
in each phase, so `use_reentrant=False` recomputes once.)

⇒ **Recompute at `ckpt3` ≈ 148–152 ms/rank-step (ESTIMATED, ≈150.3 × 1.014).** Removing it
entirely would be worth **≈25% of the 603.5 ms step (≈1.34×)** — an estimate, not a
measured bound.

**Three qualifiers, all of which shrink the actionable number:**

1. **`ckpt0` is almost certainly not reachable on a 40 GB A100.** ai-rossby's `ckpt0` peaked
   at **36.11 GB** (CHANGELOG 2026-08-06, job 7365119) and Pangu runs **+5.58 GB** higher at
   `ckpt3` (26.98 vs 21.40 GB, §5), projecting Pangu's `ckpt0` to **~41.7 GB > 40 GB**. The
   ≈25% is the size of a prize that cannot be collected in full.
2. **Production already banks most of it.** Pangu ships `checkpointing: 2` (plan §0e,
   CHANGELOG 2026-08-07, jobs 7366939→7366940), and ai-rossby's `ckpt3 → ckpt2` is 1.274×.
   Residual `ckpt2 → ckpt0` is therefore **≈1.045×** — so measured against **the config we
   actually run**, checkpointing headroom is **≈4%, not ≈25%.** Plan §0e's warning that
   "every percentage in §0d is a `ckpt3` percentage" applies to this number too.
3. **A `ckpt3 → ckpt2` delta does not measure "blocks".** Because the levels are cumulative,
   it measures **block-minus-MLP** recompute; the MLP, encoder and decoder are still
   checkpointed at `ckpt2`.

**Consistency check (not a measurement), and it nearly saturates the estimate.** ai-rossby's
ladder on the identical model shape is `ckpt3 → ckpt2` **1.274×** and `ckpt3 → ckpt1/ckpt0`
**1.307×** — converted to step share, 21.5% and **23.5%**, against this capture's
independently derived **≈25%**. Two completely different measurements (an nsys phase
attribution on Pangu at `ckpt3` vs an A/B timing sweep on ai-rossby) agree to **1.8
points**, which says essentially the *whole* forward is recomputed at `ckpt3`. Different
harness and a cross-job ratio, so it stays a check — plan item 10 measures it in Pangu's own
harness, one job, interleaved. **Pre-registerable consequences:** a Pangu `ckpt3 → ckpt0`
materially above **1.34×** falsifies either this estimate or the phase attribution; and if
Pangu's `ckpt3 → ckpt2` is ~1.274×, `ckpt2 → ckpt0` must be **≈1.045×**.

#### 4.3e The bandwidth reference nobody had — 82% of peak, above L2 only

The same capture contains a **measured** answer to "can this node reach HBM peak at all",
which is what makes §0d's *estimated* 17–27% interesting rather than possibly-an-artifact.
The device's own peak and L2 size are read **from the capture** (`TARGET_INFO_GPU`:
`memoryBandwidth = 1,555,200,000,000` B/s, `l2CacheSize = 40 MiB`, `NVIDIA A100-SXM4-40GB`
×4), not assumed.

A D2D memcpy's DRAM traffic is `2 × bytes` — read the source, write the destination — **only
if the transfer misses L2.** Bucketing the 33,920 D2D copies by size shows the rule failing
where it must:

| transfer size | n | GB | GB/s (2×) | % of peak | |
|---|---|---|---|---|---|
| 127.27 MB | 8,320 | 1110.28 | 1193.0 | 76.7% | above L2 |
| 126.56 MB | 7,680 | 1019.22 | 1386.8 | 89.2% | above L2 |
| 22.37 MB | 640 | 15.01 | 1187.5 | 76.4% | sub-L2, rule invalid |
| **12.48 MB** | 480 | 6.28 | 1939.5 | **124.7%** | sub-L2 — **>100% proves the rule fails** |
| **11.12 MB** | 480 | 5.60 | 1710.8 | **110.0%** | sub-L2 — same |
| 1012.50 KB | 480 | 0.50 | 638.3 | 41.0% | sub-L2, rule invalid |
| 379.69 KB | 480 | 0.19 | 337.1 | 21.7% | sub-L2, rule invalid |
| 2.00 KB | 15,360 | 0.03 | 1.7 | 0.1% | sub-L2, rule invalid |

**Quote only the population above L2: 16,000 copies, 2129.50 GB = 98.7% of all D2D bytes,
1279 GB/s = 82% of peak** — per device 81.3 / 82.6 / 82.6 / 82.4%, so within 1.3 points and
not noise. Every row is `copyKind = 8` with `srcKind = dstKind = Device`; there is **zero**
`copyKind = 10` (peer-to-peer), so no NVLink traffic is masquerading as local HBM.

| path | ms/rank-step | bytes/rank-step | achieved DRAM bandwidth |
|---|---|---|---|
| D2D memcpy, **above L2** | 20.82 | 13.31 GB | **1279 GB/s = 82% of peak** |
| `direct_copy`/`conj` kernels | 271.19 | — | **17–27% (estimated, §0d)** |
| H2D (loader) | 2.18 | 49.9 MB | 24 GB/s (host link, not HBM) |

**The A100s in this run demonstrably sustain ~82% of HBM peak**, so the copy *kernels* are
slow because of the path they take, not because the hardware cannot deliver. This **narrows
but does not replace** plan item 7: ncu still has to say whether the kernels are at ~25% of
peak (⇒ fix contiguity) or near peak on inflated traffic (⇒ move fewer bytes). It removes
the third possibility, "82% is unreachable here."

> **⚠ Two things this is NOT.** (i) It is **intra-device HBM** — `cudaMemcpyAsync` D2D
> within one GPU's own memory. It says nothing about NVLink/PCIe *between* the four devices,
> so it does **not** close the OPEN topology cell in `POLARIS_PROFILING_HANDOFF.md` §4 and
> does **not** substitute for plan item 6 (`gpu_topology_check.py`). (ii) It is **not** a
> concurrency-inflated figure: all 33,920 D2D copies and all 962 H2D copies are on
> `streamId 7`, the same stream as the compute kernels, so they are serialized with compute
> and with each other. It is nonetheless a **lower** bound, for a different reason than the
> first draft claimed — **30.2% of rank 0's D2D copy time (0.2604 of 0.8622 s) overlaps
> `ncclDevKernel` on `streamId 19`**, which is consuming HBM at the same time.

#### 4.3f The data-movement bill is larger than the kernel table shows, and §0a's idle is smaller

Memcpy and memset are GPU work that **is not in §4.2's 643.2 ms kernel total at all**, yet
they sit on the same `streamId 7` as compute. Adding them:

| | ms/rank-step |
|---|---|
| `direct_copy` + `conj` kernels | 271.19 |
| D2D memcpy | 21.31 |
| H2D memcpy | 2.18 |
| memset (7,840 ops, 302.6 MB/rank-step) | 0.29 |
| **total data movement** | **294.97** |

And the consequence for plan §0a, which reported GPU-busy from kernels alone:

| device | kernel union / kernel span | **all GPU work / work span** | idle |
|---|---|---|---|
| dev0 | 95.7% | **98.6%** | **1.4%** |
| dev1 | 95.6% | **98.5%** | **1.5%** |
| dev2 | 96.5% | **98.5%** | **1.5%** |
| dev3 | 96.5% | **98.5%** | **1.5%** |

The kernel-only column reproduces plan §0a exactly (95.7 / 95.6 / 96.5 / 96.5 — the
per-device differences are a span definition: on dev2/dev3 memcpy extends the work span
0.22 s beyond the kernel span). **§0a's "3.5–4.4% idle" is therefore an overstatement;
counting all GPU work, idle is 1.4–1.5%.** This makes §0a's conclusion *stronger*: there is
even less idle to reclaim, and still nothing for CUDA Graphs.

#### 4.3g Method notes — three ways to get this wrong

**1. Do not bucket by kernel execution time.** Attribution here is by **launch** time (the
phase that *requested* the work — causal, and unaffected by CUDA being async). Bucketing by
**execution** time instead puts **42.4% of launches and 46.1% of GPU time in `(outside)`**
and drops `forward_loss` from 23.4% to 5.4%. That is not a competing answer, it is §4.1's
run-ahead showing up as an artifact: the CPU has left the phase window long before the GPU
reaches those kernels. `--by-exec` exists to expose that, not to be quoted. The same applies
to the memcpy rows: by execution time only 0.96 of the 2.18 ms/rank-step of H2D lands in
`data_prep` and 7.76 ms of D2D lands in `(outside)`.

**2. One of the 40 steps is a comms stall, and it moves the NCCL number by 12%.** The
per-rank-step series (`--per-step`) is otherwise flat — median **634.36 ms**, min 631.28 —
but **step index 30** reaches 1222.34 ms on two ranks, of which **614 ms is NCCL** against a
~59 ms norm at identical launch counts. That is a straggler *wait*, not work. Excluding it:
total 643.19 → **634.57** ms/rank-step (−1.3%), NCCL 67.82 → **59.67** ms (**−12.0%**). The
phase shares move ≤0.4 points, so §4.3b is unaffected — but §4.2's "NCCL = 67.8 ms/step
(10.5%)" carries the stall, and any sizing derived from it should use the ~59.7 ms figure.

**3. Warmup 20 was enough — plan item 5's question. Judge it on COMPUTE, not the total.**
The first measured step is +0.9% on the total and **+0.1% on compute** (compute median
**574.89 ms**), and `forward_loss` GPU time spans only 150.20–150.52 ms across all 40 steps.
No compute warmup regime — **confirmed on n=2** (§4.4b: 7255557's step 0 is +6.7% on the
total but only **+0.5% on compute**, the excess being NCCL). Judging this on the total is
exactly the tool bug §4.4e records. §4.1's `forward_loss` max of 268.4 ms is a **CPU-side**
outlier — §4.4e identifies it as a CPython gen-2 GC pause, not a warmup effect.

#### 4.3h Reproducing every table above

```bash
CAP=$MEMBER_ROOT/bench/nsys_pangu_sfno_7255503.sqlite
python3 ACE2_retrain/nvtx_phase_attribution.py $CAP                                # 4.3b
python3 ACE2_retrain/nvtx_phase_attribution.py $CAP --kernel-regex 'direct_copy|conj'  # 4.3c
python3 ACE2_retrain/nvtx_phase_attribution.py $CAP --memcpy                       # 4.3e/f
python3 ACE2_retrain/nvtx_phase_attribution.py $CAP --per-step                     # 4.3g
python3 ACE2_retrain/nvtx_phase_attribution.py $CAP --by-exec                      # 4.3g note 1
python3 ACE2_retrain/test_nvtx_phase_attribution.py                                # PASS, no GPU
```

Runs on a Polaris **login node** — pure sqlite, `mode=ro`, no torch import, no allocation.
(`parse_nsys.py` could not, before the same commit fixed it: `sqlite3.connect(PosixPath)`
needs Python ≥ 3.7 and the login default is 3.6.15.)

---

### 4.4 The n=2 (job 7255557) — which numbers are stable, and which are not

Plan item 2. Job **7255557** is the clean re-run of 7255503 after a clock-placement fix:
7255503 sampled `elapsed` *after* `cudaProfilerStop()` and its bench row was refused by the
self-check (CHANGELOG 2026-07-15). Ten minutes apart, and — as it turns out — **on
different nodes**: `x3001c0s19b0n0` vs `x3001c0s1b1n0`, with four disjoint GPU UUIDs. So
this is a **node-to-node** n=2, which makes it more useful than the gate asked for.

**Preregistered before any query against the second capture** (commit `952fcb8d`): that the
*absolute* copy time would reproduce within ±5% while its *share* would fall, because
7255557's CSV shows `step_std` = 39% of the median against 7255503's 5.3%. **5/5 hit.**

An adversarial pass then landed 11 strikes, four of them FATAL, and **found the actual
cause of the stall that §4.4d originally mis-diagnosed**. Everything below is the
post-review text; §4.4f records what was withdrawn.

#### 4.4a Config identity — established structurally, not by sha

**There is no env file for job 7255503** (only `bench_env_polaris_nsys_7255557.txt`
exists), so the two runs **cannot** be compared by `git_sha` or yaml hash, and no claim
here rests on one. What establishes identity:

| check | result |
|---|---|
| total kernel launches | **354,720 in both** |
| per-name launch counts | identical for **62 of 64** names (see below) |
| all D2D memcpy bytes | **2,157,110,906,880 B — byte-identical** |
| NVTX phase rows | 160 each of `data_prep`/`forward_loss`/`backward`/`optimizer`, both |
| `(outside)` launches | **0 in both**, so total = compute + NCCL exactly |
| provenance | CHANGELOG 2026-07-15: only the `elapsed` sample *site* changed |

**Kernel selection is not bit-reproducible, and that is worth knowing.** 7255503 has **64**
distinct kernel names, 7255557 has **63**. One `cutlass__5x_cudnn` implicit-GEMM
`Conv2dWgrad` variant (40 launches) appears only in 7255503, and its sibling runs 120 vs
160 launches — cuDNN picked a `GemmShape<128,128,64>` tile on one rank of 7255503 where
7255557 used `<128,128,32>` everywhere. H2D count differs by 2 (962 vs 960; 16 bytes).
Total impact: **3.2 ms of 102.9 s ≈ 2e-5%**. So the denominators *are* like-for-like — but
"every launch count identical" would be false, and an autotune-sensitive config could
plausibly show a warmup effect this pair does not.

#### 4.4b The comparison

| quantity | 7255503 | 7255557 | delta |
|---|---|---|---|
| **COMPUTE-only kernel time** (ms/rank-step, mean) | 575.37 | 579.00 | +0.63% |
| — median | 574.89 | 575.67 | +0.14% |
| — **mean over non-stalled steps** | **574.90** | **575.35** | **+0.08%** |
| **NCCL** (ms/rank-step) | 67.82 | 145.65 | **+114.8%** |
| total kernel time (ms/rank-step) | 643.19 | 724.64 | +12.7% |
| **`direct_copy`+`conj`, absolute** | 271.19 | 270.94 | **−0.09%** |
| copies as % of **all** kernel time | 42.16% | 37.39% | **−4.77 pt** |
| copies as % of compute | 47.13% | 46.79% | −0.34 pt |
| **copy split, `backward` share** | 72.86% | 72.83% | **−0.03 pt** |
| D2D above L2 | 1279 GB/s (82.2%) | 1281 GB/s (82.4%) | +0.2% |
| memset | 317.21 MB/rs | 317.16 MB/rs | −0.02% |

**Quote the quiet-step figure for compute: +0.08%.** The headline +0.63% is itself
contaminated — 79% of that rise is the widening mean−median gap, and dev0 alone (mean
587.87 vs its own median 576.10) accounts for 88% of 7255557's gap. Compute-only
*union* is even tighter: **22.997–23.059 s on 7 of 8 devices across the two nodes, 0.27%**.

**The copy row is the strongest in the section** and survives every estimator: −0.09% on
the mean, −0.07% on the median, −0.05% on quiet steps; the `backward` share −0.03 pt.

#### 4.4c The finding: a share of the full kernel total is not a reproducible quantity

**The same measurement, on two runs of one config, moved 4.77 points** (42.16% → 37.39%)
while its numerator moved 0.09%. All of it is denominator: NCCL more than doubled, and NCCL
time is largely *waiting*.

⇒ **Quote `direct_copy`+`conj` as 271 ms/rank-step. Do not quote "42.2% of GPU kernel
time".** The same caution applies to every row of §4.2's category table and to plan §0d.

**Share-of-compute is better, but not for the reason the first draft gave.** Its −0.34 pt
is **algebraically forced**, not independent evidence: Δ(C/T)/(C/T) = ΔC/C − ΔT/T =
−0.09% − 0.63%. The numerator is 47% of its own denominator, which damps sensitivity
**1.88×**; against an external denominator (copies ÷ *non-copy* compute) the same pair
moves **−1.20 pt**. And had non-copy compute moved 10%, share-of-compute would move
−2.37 pt. **So share-of-compute is stable exactly to the extent the compute total is
stable — the same conditional the share-of-total failed, just with a much better-behaved
denominator.** The absolute ms/rank-step is the only form with no denominator to drift.

**The compute/NCCL split is also not perfectly clean.** NCCL wait leaks into the *compute*
column through SM contention: a spinning ring kernel shares SMs with the compute it
overlaps, and on the stalled step the **waiting** ranks' non-NCCL time inflates ~+5%
(7255503 dev2: 607.27 vs a 575.02 median). This is why dev0 — which waits most — shows the
highest compute mean, and it means dev0's +2.1% is a *consequence* of waiting, not a cause.

#### 4.4d Where the extra NCCL went, and the `broadcast_buffers` question reopens slightly

> ⚠ **SUPERSEDED by §4.7 — read that before using any NCCL number here.** A warm traced
> capture (job 7550368) gives NCCL **82.34 ms/rank-step** at **1.6×** spread with no step above
> 1.44× the median, against **8.4×** on the cold captures. **Every NCCL figure in this
> subsection came from a cold single-arm run**, so the ×41 broadcast blow-up and the 17-of-40
> stalled steps are first-run variance, not workload properties. The *mechanism* argument — a
> big broadcast number is ranks waiting, not broadcast cost — survives; its magnitudes do not.

| | 7255503 | 7255557 | delta |
|---|---|---|---|
| `ncclAllReduce` (in `backward`) | 67.12 ms/rs | 116.75 ms/rs | +73.9% |
| `ncclBroadcast` (in `forward_loss`) | **0.70 ms/rs** | **28.90 ms/rs** | **×41.3** |
| non-broadcast `forward_loss` compute | 149.62 | 149.67 | **+0.03%** |
| steps with non-rooted NCCL > 1.5× median | **1 of 40** | **17 of 40** |  |

`forward_loss`'s entire rise (+28.25 ms/rank-step) **is** the broadcast (+28.20), at an
unchanged **160 launches** in both. The broadcast's kernel cost did not change; it absorbed
skew. Per-rank, dev0 is the **root** at 0.23 ms/step in both captures while the non-roots
absorb 26.1 / 42.2 / 47.1 ms in 7255557 against 0.58 / 1.00 / 1.00 in 7255503.

**The attribution retraction stands: a large broadcast number is never a broadcast cost.**
Midway's 33.14% was ranks waiting (plan §0c, handoff §1), and this is the same mechanism
caught on Polaris at 0.11% → 4.0% with nothing broadcast changing.

**But the first draft went one step too far.** It claimed `broadcast_buffers=False` "would
only move the wait to the next collective." §4.3b's own union column refutes that:
`forward_loss` self-overlap is **0.0% in both captures** — the broadcast wait is **100%
exposed** — whereas `backward`'s NCCL is **83–87% overlapped** with compute. Skew that
lands at a backward all-reduce is mostly *hidden*; skew at the forward broadcast is not.
⇒ **On this capture's own statistics, removing the broadcast could hide the majority of
that 28.9 ms/rank-step rather than conserve it.** That is an argument about where skew
lands, not a measurement — and the change is still **jesswan-gated** (BN running
statistics). It does not revive the Midway framing; it means the *sizing* is open where the
first draft called it closed.

#### 4.4e The reproducible stall is CPython garbage collection — not NUMA, not affinity

> ⚠ **TWICE SUPERSEDED — this heading is now wrong on both halves. Do not build on it.**
> (1) The **causal** GC claim was refuted by an interleaved A/B (jobs 7549941, 7550007): with a
> warm-up arm absorbing the first-run effect, disabling the collector changes nothing —
> `step_std` B/A = **1.0102**, peak memory identical to three decimals. (2) The **stall itself**
> is a cold-start artifact per **§4.7**: warm, NCCL spread is 1.6× and no step exceeds 1.44× the
> median. What survives is only the narrow observation that on torch 2.8 the CPU *was* inside
> `gc_collect_main` during that particular window. Retained as the record of what a cold
> capture shows.

**One event reproduces across the two nodes, and it is the section's best finding.** At
step index **30** — the same training iteration (`step_50`) in both captures — a rank's
`forward_loss` window blows out to ~7× the median. CPU sampling names the cause outright
(`--stall-cause`):

| capture | rank | `forward_loss` | top CPU leaf symbols in the window |
|---|---|---|---|
| 7255503 | dev1 | **268.4 ms** (7.4×) | `gc_collect_main` **116**, `visit_reachable` 37, `dict_traverse` 21, `subtype_traverse` 15, `func_traverse` 14 |
| 7255503 | dev0 | **247.5 ms** (6.8×) | `gc_collect_main` **88**, `visit_reachable` 39, `dict_traverse` 24 |
| 7255557 | dev1 | **259.3 ms** (7.0×) | `gc_collect_main` **88**, `visit_reachable` 35, `dict_traverse` 19 |

Nothing else in either capture exceeds 5 GC samples in a window. Thread state is `Running`
throughout; blocking-syscall time in the 247 ms window is **0.6 ms** and CUDA API time is
normal. **The rank is not descheduled and not waiting on memory or I/O — it is burning CPU
in the collector.**

**This is why it recurs at the same iteration on different hardware.** A CPython gen-2
collection triggers on a *deterministic function of allocation count*, and its pause scales
with the tracked object graph — so the same training loop reaches it at the same step
regardless of node. **No NUMA or affinity hypothesis can predict an iteration index.**

⇒ ⚠ **RETRACTED 2026-08-21 — the `gc.freeze()` recommendation is WITHDRAWN, and the
"recurring" framing with it.** An interleaved A/B on torch 2.10 (jobs 7549941, 7550007)
tested it directly: with a throwaway warm-up arm absorbing the first-run effect, **turning
the collector off changes nothing measurable** — `step_std` B/A = **1.0102**, `step_p90` B/A
= **0.9981**, `step_med` B/A = 0.9987, and peak memory **identical to three decimals** (so
the collector was not freeing anything significant either). Both GC-ON arms were as clean as
both GC-OFF arms.

**What the stall actually is, on this evidence: a FIRST-RUN effect.** It appeared in the
first arm of job 7549941 (`step_std` 0.0770) and, when a throwaway arm was placed first in
7550007, **it moved there** (0.0610, **65×** the later arms' 0.00094). It follows the *first
arm*, not the GC setting. Partially cold-cache I/O — `loader_wait_frac` 0.0056 vs
0.0001–0.0002 — though that quantifies to only ~9% of the excess, so the remainder is
unattributed.

**The material correction:** this section said the cost was "a recurring multi-hundred-ms
hit" on a 100-epoch run. If it is a first-run effect it is paid **once per job**, not per
epoch — so it is a cold-start cost, not a recurring one, and far less interesting than
written.

**What still stands, and what does not.** The torch-2.8 sampling evidence below is an
*observation* and is not retracted: the CPU demonstrably was inside `gc_collect_main` for
~180 of ~300 samples in that window. What is refuted is the **causal** reading — GC time
was present in that pause, but disabling GC does not prevent the pause class. ⚠ **And note
this A/B addresses only the FIRST-ARM stall. Job 7255557's pattern was different — 19 of 40
steps stalling mid-run, with dev0 out of phase — which is not a first-arm effect and remains
unexplained** (plan item 6b(B)).

**A second, distinct mechanism accounts for 7255557's other stalls, and it is not
diagnosed.** On 16 of its 17 stalled steps the pattern is different: **dev0 alone waits
~600 ms** at the all-reduce while the other three sit at 60–70 ms. That is dev0 out of
phase with the group, not one rank straggling — and the argmin margins between the other
three are 1–2 ms, i.e. noise. The late work is in the **inter-step gap** (`optimizer` end →
next `data_prep`), median 266 ms → 335–579 ms, with `_PyEval_EvalFrameDefault` /
`__libc_malloc` / `_int_free` frames: host-CPU-bound work at the step boundary. Candidate,
**unestablished**: the Pangu nsys script sets `OMP_NUM_THREADS=8`
(`polaris_bench_nsys_e3sm_sfno.pbs:51`) and **no CPU binding**, so 4 unbound ranks put ~32
OpenMP threads plus 4 main threads plus loader workers on 32 physical cores; the ai-rossby
single-node script leaves torchrun to pin `OMP_NUM_THREADS=1`. → plan item **6b**.

**Consequence for plan item 12 (multi-node).** The single-node "comms are essentially free"
result (§0b: 88.7% overlapped, 1.2% exposed) was measured on the *quiet* capture. On
7255557, identical config, exposed NCCL is **4.9–8.8% of span** and 17 of 40 steps stall.
Comms are free **when the ranks are balanced**; balance is a per-run property. Note the GC
mechanism is **node-count independent** — it will neither hide inside NVLink nor grow with
nodes — so multi-node work must separate it from genuine interconnect effects.

#### 4.4f What was withdrawn from the first draft of this section

Recorded so it is not re-derived, and because two of these were confidently wrong:

1. **"`deviceId 1` is the straggler in both captures." WITHDRAWN.** It is one event
   (step 30) in each, not a rank property. Over 7255503's other 39 steps no straggler
   exists (spread 1.0–1.1×); over 7255557's 17 stalls the late rank rotates. The original
   per-rank ranking also **summed the rooted broadcast**, where the root's time is ~0 by
   construction — that handed dev0 a spurious "late" credit on every step. `--per-rank` now
   excludes rooted collectives and `_is_rooted()` parses the collective name, because a
   regex on `Reduce` misclassifies `AllReduce`.
2. **"The delay is CPU-side and invisible in the kernel table." HALF WRONG.** CPU-side is
   right; invisible is not — the *waiting* ranks' compute inflates ~5% via SM contention.
3. **"The other three burn ~610 ms."** In 7255503 **dev0 was itself stalled** (247.5 ms of
   GC), which its own 251 ms cell in the first draft's table already contradicted.
4. **The NUMA/affinity hypothesis as the explanation for step 30. WITHDRAWN** — superseded
   by the GC measurement. It survives only as a candidate for the *other* stall pattern.
5. **"+20% warmup on the total" for 7255557.** The real figure is **+6.7%** vs the median
   total (685.09 vs 641.95 ms/rank-step); the +20% came from comparing one rank's step 0
   against the all-rank median. The verdict is unaffected — +6.7% clears a 3% threshold, so
   the old total-based test really would have misfired — and the tool's docstring is fixed.
6. **"Same `git_sha`."** No env file exists for 7255503; identity is structural (§4.4a).
7. **"D2D bytes 2129.50 GB."** That is the **above-L2 subset** (98.7%); all D2D is
   **2157.11 GB**, byte-identical in both. And `memset 302.57 MB` was really MiB — the
   tool's unit helper was 1024-based while its bandwidth figures divide by 1e9. Fixed.

#### 4.4g Scope — what these two captures do NOT establish

Both are **1 node, 4 ranks, `checkpointing: 3`, batch 1/rank, bf16, `num_data_workers: 1`,
warmup 20, eager, `use_ema: True` with EMA never firing** (`ema` appears 0 times in both).

* **The 72.86 / 27.14 copy split is a `checkpointing: 3` property, not a model property.**
  Activation checkpointing is precisely what relocates work from forward into backward, so
  this row cannot be quoted at another level. Plan item 9 re-captures at `ckpt2`.
* "Warmup 20 was long enough" is established for **this** config only — and §4.4a shows
  cuDNN's kernel choice was *not* stable across the pair, so an autotune-sensitive config
  can still warm up.
* Node-to-node variation is n=2 on **two** nodes; the repo's 10.5% figure is for wall time.
  What this pair adds: **compute is reproducible node-to-node (0.08–0.27%); anything
  containing NCCL is not (+114.8%).** That refines the standing rule "a cross-job ratio on
  Polaris is not a measurement" into: *cross-job **compute** comparisons are sound;
  cross-job comparisons of any quantity containing NCCL are not.*

```bash
CAP=$MEMBER_ROOT/bench/nsys_pangu_sfno_7255557.sqlite
python3 ACE2_retrain/nvtx_phase_attribution.py $CAP --per-step      # compute vs NCCL, warmup verdict
python3 ACE2_retrain/nvtx_phase_attribution.py $CAP --per-rank      # straggler test, rooted excluded
python3 ACE2_retrain/nvtx_phase_attribution.py $CAP --stall-cause   # the GC diagnosis
python3 ACE2_retrain/nvtx_phase_attribution.py $CAP --memcpy        # bandwidth, L2-aware
```

---

### 4.5 What the copies are actually moving — a weight in the wrong layout

Plan item 3: build an analytic bytes-per-step model from the config and check it against
§0d's launch-geometry estimate. `ACE2_retrain/sfno_bytes_model.py` (+ test, no capture
needed). n=2 on both captures; no GPU time.

**Preregistered at `45cbd7de`, and all four size predictions were wrong — which is the
deliverable.** The prereg enumerated every *activation* the step touches and predicted the
dominant copies would land on the spectral activation (133.45 MB) and the fp32 latent
(132.71 MB). **The inventory omitted the weights**, and for this model at batch 1 that
omission is the whole story.

**Headline, after an adversarial pass that landed two FATAL strikes on the first draft:**
**133.15 ms/rank-step — 20.7–23.1% of the step depending on the basis — is spent moving a
377 MB spectral weight, in four places, all traceable to one layout mismatch.** Of that,
**107.14 ms (17.8% of a 603.5 ms step) moves invariant values** and **26.01 ms is the
gradient**, which is not invariant — the first draft counted it as such and claimed 22.0%.
§4.5c has the decomposition; §4.5d records every withdrawal.

#### 4.5a The tensor inventory, weights included

Shapes grounded in source, not assumed: `modes_lat = int(h·thf) = 180`,
`modes_lon = int((w//2+1)·thf) = 181` (`sfnonet.py:481-482`); for `operator_type: dhconv`
the spectral weight is `[in_channels, out_channels, modes_lat]` complex
(`s2convolutions.py:107-136`). **Channel counts come from the loader, not from re-counting
the YAML variable lists** — that is how the first version of this table got 108:
`surface_variables` absorbs `land_variables` (6+2 = 8), the 3 diagnostics are *output*-only,
and `varying_boundary` (3) is input-only alongside `constant_boundary` (4), giving
`in_chans = 90+8+3+4 = 105` and `out_chans = 90+8+3 = 101`
(`data_loader_multifiles.py:457,641-655`). **The parameter total pins it independently:**
reconstructing the count forces `2·in + out = 311 = 2(105) + 101` against the logged
1,182,108,160, and the test asserts that.

| tensor | shape | elements | dtype | MB |
|---|---|---|---|---|
| act input | 105×180×360 | 6,804,000 | fp32 | 27.22 |
| act output | 101×180×360 | 6,544,800 | fp32 | 26.18 |
| act latent | 512×180×360 | 33,177,600 | fp32 | 132.71 |
| act big_skip cat | 617×180×360 | 39,981,600 | fp32 | 159.93 |
| **act MLP hidden** | 1024×180×360 | 66,355,200 | fp32 | **265.42** |
| act spectral | 512×180×181 | 16,680,960 | complex64 | 133.45 |
| act spectral, one part | 512×180×181 | 16,680,960 | fp32 | 66.72 |
| **WGT spectral, per layer** | **512×512×180** | **47,185,920** | **complex64** | **377.49** |
| WGT MLP fc1/fc2 | 1024×512 | 524,288 | fp32 | 2.10 |

**The largest tensor in this model is a weight: 377.49 MB, which is 1.42× the largest
activation (the MLP hidden, 265.42 MB) and 2.83× the spectral activation.** Twelve of them
are **1,132,462,080 of the model's 1,182,108,160 parameters — 95.8%.** (⚠ At batch 1. The
weight:activation ratio is `E/(B·mmax)` and it inverts at batch 4 — §4.5c-scope.)

**The weight is dense, not block-diagonal — `num_blocks: 16` does not apply here.** That is
the obvious objection (16 blocks of 32×32×180 would be 1/16th the size), so it is checked
rather than assumed: `weight_shape` is built as `[in_channels]` → `+= [out_channels]`
(`separable: False`) → `+= [modes_lat_local]` (`operator_type: dhconv`) at
`s2convolutions.py:107-136`, and **`num_blocks` appears nowhere in it.** It is read into
`self.num_blocks` at `sfnonet.py:415-416` and never read again — a vestigial knob. Three
independent lines agree on the dense shape: the source, the launch geometry (exactly
47,185,920 elements), and the parameter total.

#### 4.5b Every dominant copy lands on a tensor exactly

Launch geometry → elements. **Elements per block depends on the launch path**, and getting
that wrong rescales everything: `elementwise_kernel<(int)nt,(int)vt,…>` is
`launch_legacy_kernel` with `grid = ceil(N/(nt·vt))`, so `nt·vt = 256` per block; but
`unrolled_elementwise_kernel<…,(int)4,…>` and `vectorized_elementwise_kernel<(int)vec,…>`
both come from `launch_vectorized_kernel` with `blockX × thread_work_size = 512` per block —
and the vectorized name's leading `(int)` is `vec_size`, **not** `nt`. Reading it as `nt`
under-counts those rows by exactly 4×, which is what the first version of this table did.
Matched against the inventory **at the same dtype**, so a float copy is never credited to a
complex tensor. Job 7255503; 7255557 is geometry-identical and within **0.19 points** on
every share.

| kernel | calls/rank-step | MB/call | µs/call | % copy | tensor |
|---|---|---|---|---|---|
| `direct_copy`⟨float,nocast⟩ | **348** | 66.72 | 332.7 | **42.7%** | **= act spectral, one part** |
| `direct_copy`⟨complex64,nocast⟩ | **36** | **377.49** | 2700.5 | **35.8%** | **= WGT spectral** |
| `conj`⟨complex64,nocast⟩ | **12** | **377.49** | 2994.3 | **13.2%** | **= WGT spectral** |
| `direct_copy`⟨complex64,nocast⟩ | 48 | 133.45 | 297.4 | 5.3% | = act spectral |
| `direct_copy`⟨float,unrolled⟩ | 24 | 132.71 | 179.6 | 1.6% | = act latent |
| `conj`⟨complex64,vec⟩ | 12 | 66.72 | 197.4 | 0.9% | **0.5000×** act spectral — unidentified |
| `direct_copy`⟨float,nocast⟩ | 2 | 132.71 | 304.7 | 0.2% | = act latent |
| `direct_copy`⟨float,unrolled⟩ | 1 | 159.93 | 216.0 | 0.1% | = act big_skip cat |
| `direct_copy`⟨float,unrolled⟩ | 24 | 2.10 | 6.8 | 0.1% | = WGT MLP fc1/fc2 |

**Nine kernels covering ~99.6% of copy time match an analytic tensor**, eight of them
exactly. There are 18 distinct (kernel, geometry) groups in total; the 9 not shown carry
**0.25%** of copy time, so this is not a top-N crop hiding a mismatch.

**Three caveats on "exactly", all real:**

1. **"×1.0000" is a threshold, not a printed measurement.** The tool prints `= <tensor>`
   when `|ratio−1| < 0.001`. The underlying products *are* exact
   (`184320×256 = 47,185,920`, `65160×256 = 16,680,960`, `64800×512 = 33,177,600`), but
   geometry only pins `N` to the interval `(elems − per_block, elems]` — up to 512 elements
   of block rounding. The big_skip row is visibly that case: 78,090 blocks × 512 =
   39,982,080 against a true 39,981,600.
2. **The match is not injective.** `w`, `conj(w)` and `grad_w` all have 47,185,920 complex
   elements, so element count alone cannot tell a parameter from its gradient — §4.5c
   separates them by duration, phase and NCCL adjacency instead. Likewise at `thf = 1.0` the
   pre-truncation rfft field and the post-truncation spectral field both have 16,680,960
   elements because `lmax = nlat`, so the 42.7% row cannot be distinguished from geometry
   alone (its identification as one *part* of the complex field rests on the stride-2
   signature of `xout[..., 0] = …` / `torch.stack((rl, im), -1)` in torch-harmonics).
3. **The two sides are less independent than "completely independent" suggests.** Both read
   the same `gridX`/`blockX` from the same CUPTI table; only the *interpretation* is
   independent. And the model covers `direct_copy|conj` **kernels** only — §4.3f's
   13.48 GB/rank-step of D2D **memcpy** is a further ~21% of total movement and is not joined
   to this inventory at all.

⚠ **What this does and does not do to §0d.** It corroborates §0d's byte estimate for the
legacy path (97.06% of copy time) and **re-introduced then fixed** an error on the other two
paths (the 4× above). It does **not** remove §0d's coalescing caveat: these are still
*useful* bytes, and whether real DRAM traffic is higher is item 7 (ncu), untouched.

⚠ **This is a post-hoc consistency check, not the preregistered confirmation, and the
prereg's own decision rule said so.** `45cbd7de` pre-committed that §0d's byte caveat is
removed "only in case (a)" — if P1–P3 held. They did not; case (b) obtained. **The weight
row was added to the inventory *after* the mismatch was seen.** What remains is still strong
— many independent kernels landing on analytic tensors at matched dtype on two captures —
but it is evidence, not a fulfilled prediction, and the house standard (§4.4f) is to say so.

⚠ **Units.** §4.3e reports these tensors in **MiB** (127.27, 126.56) and §4.5a/b in decimal
**MB** (133.45, 132.71). Same capture, same tensors. Per CLAUDE.md #10 the columns are a
contract: **§4.5 uses decimal MB throughout** (bytes ÷ 1e6), matching the GB/s figures.

#### 4.5c The finding: one layout decision costs 133 ms/rank-step, in four places

The 36 weight-sized copies are **not** one operation repeated. Splitting the
`gridX = 184320` population by duration and by NVTX phase separates them cleanly, and both
captures agree to 0.04 ms:

| population | calls/rank-step | µs/call | ms/rank-step | phase | invariant? |
|---|---|---|---|---|---|
| weight permute, forward | **12** = 1 × `num_layers` | 2967 | **35.60** | `forward_loss` | yes |
| weight permute, **checkpoint recompute** | **12** | 2967 | **35.61** | `backward` | yes |
| `conj(w)` — the adjoint | **12** | 2994 | **35.93** | `backward` | yes |
| **`grad_w` → DDP bucket** | **12** | **2168** | **26.01** | `backward` | **NO** |
| total | 48 | | **133.15** | | |

**So `36 = 2 × num_layers + 1 × num_layers`, not `3 × num_layers`** — two forward
*executions* (because `checkpointing: 3` re-runs the block) plus a gradient-side copy. With
checkpointing off it would be 24.

**The fourth row is the gradient, and it is not invariant.** It is a distinct operation, not
the same copy: identical grid and identical 377.49 MB, but **27% faster** (348 vs 254 GB/s),
**`backward` only**, and each one is followed by an `ncclDevKernel` at a median gap of
**0.871 ms with p10–p90 = 0.867–0.876 ms** — a 9 µs spread, i.e. deterministic adjacency.
The slow population's next-NCCL gap is 20.5 ms and scattered. There are **15 all-reduces + 1
broadcast = 16 NCCL kernels/rank-step** (§4.2), and a 377 MB parameter far exceeds DDP's
25 MB default bucket cap, so **each spectral weight gets its own bucket**. This is DDP's
`Reducer::mark_variable_ready_dense` copying `grad_w` into the flat bucket immediately
before its all-reduce.

⇒ **Invariant-valued movement is 107.14 ms/rank-step = 17.8% of the 603.5 ms step, not
22.0%; 13.59 GB, not 18.12 GB.** The earlier figure counted the gradient as invariant.

#### The four are one root cause, and the obvious fix is already applied

The parameter is stored `(in, out, modes_lat)` (`s2convolutions.py:107-136`) but
`_contract_dhconv` is `torch.einsum("bixy,iox->boxy", ac, bc)` (`contractions.py:185-196`).
The weight is indexed `iox` with `x` shared by both operands and the output, so einsum must
move `x` to the batch position for the underlying `bmm`: the weight is permuted
`(i,o,x) → (x,i,o)`, which is non-contiguous, and `bmm` materialises it. That single
mismatch produces all four rows — the forward permute, its recompute, the conjugate
(materialised in the same permuted layout), and the gradient (which emerges in that layout
and therefore does **not** match its DDP bucket view).

**And `gradient_as_bucket_view=True` is already set** (`train.py:302`). That is the flag
that would normally eliminate the bucket copy, so its presence is what makes the layout the
culprit rather than the DDP configuration: with the flag on, `param.grad` *is* a view into
the bucket, and a copy is still needed only because the gradient's layout differs from the
view's. **Inference, not measurement** — labelled as such; item 8 names the call site.

**⚠ Retracted from the first draft of this subsection.** It attributed the 3 copies/layer to
`spectral_layers: 3`. **`spectral_layers` never reaches this module:** `filter_type:
'linear'` builds `SpectralConvS2` (`sfnonet.py:100-118`), which is not passed it and does
not accept it (`s2convolutions.py:57-72`) — it goes only to the `non-linear` branch's
`SpectralAttentionS2`. It is inert in this config, and `SpectralConvS2` holds exactly **one**
weight per layer, which §4.5a's `12 × 94,371,840 = 95.8%` already implies. The agreement
with 3 was numerology. Also retracted: the claim that `view_as_complex` on the parameter
forces the copy — a freshly-allocated `nn.Parameter(…, 2)` is born contiguous with last-dim
stride 1, which is exactly what `view_as_complex` wants, so that view is **free**. The
strided operand is the *permutation*, not the view.

**✅ RESOLVED 2026-08-21 by direct construction — job 7541613, and the layout argument above
stands.** Built the real net on a compute node and inspected the object:

```
weight type    : torch.nn.parameter.Parameter        (NOT a FactorizedTensor)
shape          : (512, 512, 180, 2) = 47,185,920 complex elements
contiguous     : True    stride = (184320, 360, 2, 1)
```

So the weight **is** the `nn.Parameter(…, 2)` this section assumed, it **is** contiguous, and
its element count matches §4.5a exactly. ⇒ `view_as_complex` on it really is free, and the
strided operand really is the einsum's **permutation**, not the view. Every mechanism claim
in §4.5c is on solid ground.

⚠ **What is still not understood — flagged, but it no longer blocks anything.** With
`factorization: None` (and `YParams.py:20` converts the YAML string `'None'` to Python `None`
for every key), `use_tensorly = False if factorization is None else True` is **False**, which
should land on `s2convolutions.py:150-152`'s `assert factorization == "ComplexDense"` and
raise. The probe confirmed `factorization` is `NoneType` **and** that asserts are active in
that interpreter — yet construction succeeded and produced exactly the `else`-branch object.
So `factorization` must arrive as the string `"ComplexDense"` at the conv, by some route
five separate source readings failed to find (checked and eliminated: `PYTHONOPTIMIZE`/`-O`;
a `train.py` overwrite; the `params=grid_type` indirection at `sfnonet.py:771`, where `Params`
holds only `data_grid` so `hasattr` is False; a stale subtree file — `s2convolutions.py` has
one commit, predating every capture; and an alternative filter branch — there are only four
and `linear`+`RealSHT` reaches `SpectralConvS2` alone). **The consequence is settled even
though the route is not**, so this is a curiosity rather than a gate.

**Two provenance gaps surfaced on the way, both worth knowing:** `has_diagnostic` is read
**unguarded** at `sfnonet.py:767` but appears in neither the rendered config nor
`config/E3SM_SFNO_H5_POLARIS.yaml`, and `YParams` has no `__getattr__` fallback (it ends at
line 49) — so the profiled runs obtained it from somewhere this repo does not record. And
`sfnonet.py:566` calls `.item()` on a linspace, so the net **cannot** be constructed on the
meta device.

#### What this changes about the levers

* **~71.5 ms/rank-step** (forward permute + adjoint conj) is addressable by **storing the
  parameter pre-permuted as `(modes_lat, in, out)`** — plausibly bit-identical (same `bmm`
  operands, same reduction order) but it is a hot-path **and checkpoint-format** change, so
  fully DESIGN §4-gated and blocked on the missing baseline (plan item 18).
* **35.61 ms/rank-step exists only because `checkpointing: 3` re-runs the block.** That is
  ≈43% of the ≈74.6 ms §4.3d calls "removable recompute" — so a *part* of the ckpt prize is
  weight re-permutation, not activation recompute. The two levers compose: the layout fix
  removes all three invariant permutes, ckpt removes one.
* **26.01 ms/rank-step is gradient movement** and is not touched by either — it needs the
  gradient's layout to match the bucket, which follows from the same storage change.
* **Fusing pointwise activation ops (DESIGN §5 rung 1) touches none of these four rows.**
  That is the substantive correction to the standing "61% elementwise ⇒ fuse" read, and it is
  reinforced by evidence already in the repo: ACE2 measured
  `InductorError: KeyError: 'complex64'` on the whole SFNO, so Inductor cannot reach this
  model's complex64 hot path at all (`ACE2_retrain/PROFILING_PLAN.md:238`).

#### 4.5c-scope This is a batch-1, 180×360 result and inverts at batch 4

Weight bytes are `8·E²·lmax`; spectral-activation bytes are `8·B·E·lmax·mmax`. The ratio is
therefore **`E / (B · mmax)`** — here `512 / (1 × 181) = 2.83`. Weight copies are
**batch-independent**; activation copies scale with `B`. Consequences the headline must carry:

* At **`batch_size: 4`** — the base config's own commented-out default (`batch_size: 1 #4`)
  — the ratio is `512/(4×181) = 0.71`: the spectral activation and MLP hidden become the
  largest tensors and the weight share of copy time falls to roughly **~19%**. **"This model
  *is* its spectral weights" is a batch-1 statement.**
* At a 721×1440 grid the ratio is 0.71 even at `B = 1`.
* `hard_thresholding_fraction < 1` makes it **worse**: weight ∝ `thf`, activation ∝ `thf²`,
  so at `thf = 0.5` the ratio doubles to ~5.7×. §4.5a's 377.49 MB is a `thf = 1.0` figure.
* 12 of the 48 exist only because of `checkpointing: 3`; 12 only because DDP is on 4 ranks.
  Single-GPU or checkpointing-off changes the count.

#### 4.5d What the prereg got wrong, and why it is recorded

| prediction | outcome |
|---|---|
| P1 complex64 copy ≈ 133.45 MB (act spectral) | **MISS on the dominant kernel.** That copy exists and matches exactly — but it is 5.3% of copy time. The dominant complex64 copy is **377.49 MB**, the weight. |
| P2 `conj` ≈ 133.45 MB, 24 calls/rank-step = 2 × `num_layers` | **count HIT** (24 across both variants), **size MISS** (377.49 MB weight + 16.68 MB activation). The stated rationale — "one conjugate per operand per contraction" — was also wrong: it is 12 weight conj + 12 small activation conj. |
| P3 float copy ≈ 132.71 MB (act latent) | **MISS.** The dominant float copy is **66.72 MB** = *one part* (real or imaginary) of the complex spectral activation. The 132.71 MB latent copy exists at 0.2%. |
| P4 no copy exceeds 265.42 MB | **MISS**, and the bound itself was wrong — 265.42 MB was the largest tensor *I had enumerated*. With weights included the largest is 377.49 MB and nothing exceeds it (this survives the 4× launch-path correction). |

**Also withdrawn after review, beyond the four predictions:** the `spectral_layers`
mechanism (refuted from source — it never reaches this module); the `view_as_complex`
explanation for the strided operand (a fresh `nn.Parameter(…, 2)` is contiguous, so that
view is free; the permutation is the strided thing); "18.12 GB of invariant payload" and
"22.0% of the step" (26.01 ms of it is the gradient — 13.59 GB and 17.8%); "2.8× the largest
activation" (that is 2.83× the *spectral* activation; vs the MLP hidden it is **1.42×**);
`C_in = 108` (it is **105**, and the parameter total pins it); and a **1.185× "no clean
tensor"** row that was a tool bug, not a mismatch — reading the vectorized kernel's leading
`(int)` as `nt` under-counted the non-legacy launch paths by 4×, and with it fixed that row
is the fp32 latent exactly. All are folded in above.

The prereg's decision rule anticipated this branch ("if they miss high, the excess is the
finding: name the multiple and say what could produce it"). The multiple was **2.83× the
spectral activation**, and what produced it was a tensor class the model omitted. Worth
noting the prereg also recorded, before measuring, that §0d's published `GB/s × µs/call`
implied ~238 MB and ~55 MB rather than the predicted clean tensors — so the mismatch was
visible in advance and is not being re-narrated as a hit.

```bash
python3 ACE2_retrain/sfno_bytes_model.py                                   # inventory
python3 ACE2_retrain/sfno_bytes_model.py --match $MEMBER_ROOT/bench/nsys_pangu_sfno_7255503.sqlite
python3 ACE2_retrain/test_sfno_bytes_model.py                              # PASS, no capture
```

---

### 4.6 Re-baseline on torch 2.10 — what survived a major version bump, and what did not

The base conda that produced every number above (**torch 2.8.0**) is orphaned by a Cray PE
migration and is not returning in that form: its torch and h5py link `*_gnu_123.*` sonames
the PE roll removed, and hdf5 moved soversion **200 → 310** — an ABI break, not a rename
(`polaris_pbs_notes.md` §1). Work now runs in the ai-rossby venv, **torch 2.10.0+cu129**.
Job **7545291**, `REBASE_OK`, same knobs as 7255503 (warmup 20, 40 steps, 4 ranks, eager,
`checkpointing: 3`). Preregistered before submission.

**⚠ Do not table a 2.8.0 row next to a 2.10 row.** Captures are named `..._t210_*` and the
CSV is separate for exactly this reason.

#### 4.6a The comparison

| quantity (ms/rank-step) | torch 2.8.0 (7255503) | torch 2.10 (7545291) | delta |
|---|---|---|---|
| **COMPUTE-only, mean** | 575.37 | 623.08 | **+8.29%** |
| — median | 574.89 | 620.74 | **+7.98%** |
| — spread across steps | 1.033× | **1.015×** | tighter |
| `direct_copy`+`conj` | 271.19 | **277.49** | **+2.32%** |
| — as % of compute | 47.1% | 44.5% | −2.6 pt |
| **weight copies, calls/rank-step** | **36** | **36** | **identical** |
| **weight `conj`, calls/rank-step** | **12** | **12** | **identical** |
| weight bytes/call | 377.49 MB | **377.49 MB** | identical |
| **weight share of copy time** | **49.0%** | **49.5%** | **+0.5 pt** |
| NCCL | 67.82 (quiet) / 145.65 | 217.22 | *not comparable* |
| `(outside)` | 0.0% | **0.0%** | identical |

#### 4.6b What survived: the weight finding, exactly

**§4.5's headline is version-robust.** The spectral weight is still copied **36 times** and
conjugated **12 times** per rank-step, still **377.49 MB** per call (47,185,920 complex
elements), and still **≈49% of all copy time**. Those are not approximate matches — the call
counts and the per-call geometry are *identical* across a major torch version. That is what
you would expect if the mechanism is what §4.5c says it is: a **layout mismatch in the source**
(`weight` stored `(in, out, lmax)` vs `einsum("bixy,iox->boxy")` wanting `(x, i, o)`), not an
artefact of a particular kernel library.

`(outside)` is again **0.0%**, and the phase partition again reconciles to the kernel total
(271,520 + 103,040 + 9,440 = 384,000), so §4.3's attribution method holds unchanged.

#### 4.6c What did not: the activation-side copies restructured

| | torch 2.8.0 | torch 2.10 |
|---|---|---|
| activation spectral copies | 48/rank-step @ 16,680,960 elem (512×180×**181**) | **146**/rank-step @ 16,588,800 (512×180×**180**) + 12 @ the old shape |
| `direct_copy`⟨float,nocast⟩ | 350/rank-step | 236/rank-step, and 66.36 MB/call not 66.72 |
| total copy **bytes** | — | **within 2.3%** |

So torch 2.10 moves *the same bytes* through *differently shaped and differently counted*
kernels — one longitude mode narrower, and roughly three times as many calls on the
activation side. ⇒ **The per-call sizes and call counts in §4.5b/§4.5c's activation rows are
torch-2.8.0 facts. The byte totals, and everything about the weight, are not.** Quote
accordingly.

#### 4.6d torch 2.10 is ~8% slower in compute on this model

**623.08 vs 575.37 ms/rank-step of compute, +8.29% (medians +7.98%)**, with step spreads of
1.015× and 1.033× — i.e. both tight, so this is a real difference and not a noisy draw. Copy
time grew only 2.32%, so the regression is in the **non-copy** compute: 345.6 → 345.6 ms is
what it would be if flat, and instead the non-copy remainder went 304.2 → 345.6 ms
(**+13.6%**). Worth knowing before reading any 2.10 number as an improvement, and worth an
n=2 before treating the 8% as settled.

#### 4.6e A prereg miss that was my own methodology, not a surprise

Five of six predictions hit: the gate, `(outside)` = 0.0%, the weight counts (**exactly**),
the copy time within ±10% (+2.32%), and "some kernel geometries will differ".

The miss: I preregistered total kernel time within **±15%** of 643.19 and it came in at
**840.3 (+30.6%)**. But **that band was the wrong thing to predict, and §4.4c says so in this
same document** — a total containing NCCL *wait* is not a reproducible quantity, and here NCCL
was a 217 ms draw against 67.82 on the quiet 2.8 capture. The compute-only figure I should
have banded moved **+8.3%**. Recording it as a methodology error rather than a model finding:
the lesson §4.4c exists to teach was available to me and I still predicted against a
contaminated denominator.

**Status:** this capture is the reference for plan items **9** and **10**. It is **n=1** on
this env, so per §4.4 anything comms-containing stays provisional until there is a second.

---

### 4.7 ⚠ Every capture above is a COLD single-arm run — and that is what §4.4's NCCL story was measuring

Job **7550368** ran an untraced warm-up arm and *then* a traced arm, which no previous capture
in this project had done. The GPU-side tail changes completely:

| | cold traced (7545291) | **warm traced (7550368)** |
|---|---|---|
| compute spread across steps | 1.015× | **1.003×** |
| NCCL median (ms/rank-step) | 156.77 | **82.34** |
| **NCCL spread** | **8.4×** | **1.6×** |
| worst step's NCCL | 530.48 (3.4× median) | **118.36 (1.44× median)** |
| effect of excluding the worst step | NCCL −3.7% | **−1.0%** |

**Every NCCL figure in §4.4 came from a cold, single-arm capture:**

| capture | NCCL ms/rank-step | state |
|---|---|---|
| 7255503 (torch 2.8) | 67.82 | cold, traced |
| 7255557 (torch 2.8) | 145.65 | cold, traced |
| 7545291 (torch 2.10) | 217.22 | cold, traced |
| **7550368 (torch 2.10)** | **84.43** (median 82.34) | **warm, traced** |

⇒ **The >2× NCCL swing §4.4 attributed to "rank balance is a per-run draw" is cold-start
variance.** Warm, within a single capture, NCCL is stable at 1.6× spread with no step above
1.44× the median. The 3.20× spread across the cold captures is a property of *first-run
measurement*, not of the workload.

#### What this retracts

* **§4.4d's straggler analysis** — "dev1 is the modal straggler", the per-rank NCCL tables,
  the 19-of-40 stalled steps on 7255557 — is **downgraded to a cold-start artifact.** Warm,
  there is no straggler to find. (§4.4f had already withdrawn the stronger "dev1 in both
  captures" claim on other grounds; this removes the residue.)
* **§4.4e's "undiagnosed host-CPU stall pattern"** — the dev0-out-of-phase pattern that plan
  item **6b(B)** exists to explain — is the same artifact. **Item 6b(B) closes:** the
  phenomenon is not reproducible on a warm capture, so there is nothing for a CPU-binding A/B
  to fix. **The reversed GPU→NUMA map remains a real, measured cluster fact
  (`polaris_pbs_notes.md` §1) with no measured symptom.**
* **The "+3.7% median step from tracing"** I attributed to nsys in §4.6 is wrong: warm traced
  is 0.6547 s against warm untraced 0.6515, i.e. **nsys costs +0.5% on the median**. The
  +3.7% was cold start.

#### What survives, and it is most of the substance

Everything §4.4 established about **compute** is untouched, and this strengthens it:

* Compute reproducibility — **0.08–0.63%** across captures, and **1.003×** step spread warm.
* The copy time — **0.09%** across the torch-2.8 pair, **+2.32%** across the version bump.
* The **weight counts** — 36 copies + 12 conj, identical on 2.8 and 2.10 (§4.6b).
* **§4.4c's headline is still correct, and now better explained.** "A share of the full
  kernel total is not reproducible" holds — the denominator moves because NCCL moves. What
  changes is *why* NCCL moves: cold start, not rank placement. The rule to quote
  ms/rank-step or share-of-compute is unaffected.

#### The methodological rule this establishes

**Every capture from here runs an untraced warm-up arm first.** Not a preference — a
measured requirement:

| capture class | `step_std / step_med` | GPU NCCL spread |
|---|---|---|
| single-arm + nsys (7255557, 7545291) | 39–41% | 8.4× |
| first arm in a job, untraced | 9–15% | — |
| warm arms, untraced | **0.09–0.19%** | — |
| **warm + nsys (7550368)** | 34% *(host-side, see below)* | **1.6×** |

**One caveat, because the warm traced row looks bad and is not:** its CSV `step_std` is 34%
while its `step_p90` is 0.6569 — p90/median **1.003**. Low median, low p90, huge std means a
*few* enormous outliers, and the GPU series shows compute at 1.003× and NCCL at 1.6× — so
those outliers are **host-side, not GPU-side**, and the obvious candidate is nsys flushing its
trace buffer. Consequence: **a traced run's wall-clock `step_std` is not comparable to an
untraced one's, but its GPU-kernel analysis is sound.** Use the CSV for untraced A/Bs and the
capture for kernel attribution; do not mix them.

---

## 5. Memory

**26.98 GB peak of 40 GB**, identical across all three sweep runs (loader workers don't move
it). ~13 GB headroom on a 1.18 B-param model at batch 1 + bf16 + `checkpointing: 3`.

Two things follow, both **unmeasured hypotheses, flagged as such**:

1. **`batch_size` may have room** (2/GPU). Do not assume it: `bench_report.md` §II.4 records
   that on Midway "batch ≥3/card (bf16) is a trap — throughput collapses near allocator
   saturation". Measure with the sweep, watch `peak_mem_gb_max_rank`.
2. **`checkpointing: 3` is buying memory we appear not to need**, and paying recompute for
   it. Turning it down would trade the 13 GB of headroom for step time. This is a real
   candidate lever — but it is a **hot-path change** and therefore gated on DESIGN §4,
   which is not yet executable. **Not attempted.**
   **§4.3 now sizes the prize, and it is smaller than "61% elementwise" suggests.**
   Recompute at `ckpt3` is **≈148–152 ms/rank-step (estimated** — measurably ~1.4% *more*
   than the forward it replaces, not bounded by it), so removing it entirely would be worth
   **≈25% of a stall-free step (≈1.34×)** — a run carrying comms stalls realises less,
   and §4.4b's method note applies: measure this on the compute-only median, not on mean
   total step time. But `ckpt0` projects to **~41.7 GB > 40 GB** for Pangu and
   is likely unreachable, and production already ships `checkpointing: 2`, which banks most
   of it — **residual headroom against the config we actually run is ≈4%, not ≈25%.** And
   `conj` (14.1% of the copy time) **and ~2/3 of the 35.8% weight-copy row** are immune to
   every checkpointing level (§4.5c) — the adjoint is warranted from source (§4.3c), and two
   of the three weight materialisations per layer are not recompute.

---

## 6. Optimizing is still blocked — and one §4.0 prerequisite turned out to already exist

Per the handoff: **profiling is unblocked, optimizing is not.** Nothing in this report
changed the hot path. `TORCH_COMPILE_MODE` is now wired and left **unset**; both PBS scripts
say why in a comment.

> **Correction (2026-07-15).** An earlier draft of this report said `TORCH_COMPILE_MODE` was
> "already plumbed in the ported harness". **It was not.** The harness port brought the
> `S2S_BENCH`/NVTX plumbing across but not the compile knob — PanguWeather had only a
> commented-out `torch.compile(self.model, mode='default')` (`train.py:639`) and no env read,
> exactly as DESIGN §2c's table says (`TORCH_COMPILE_MODE`: s2s **2**, PanguWeather **0**).
> The commented-out `export TORCH_COMPILE_MODE=…` in both bench scripts was therefore a live
> trap: uncomment it, get no compile, no error, and conclude "torch.compile doesn't help this
> model". Now genuinely wired (`get_model()`, gated, unset ⇒ legacy) with a test that fails if
> the knob is ever disconnected again.

### 6b. Fork drift vs `s2s/v2.0` — which `bench_report.md` optimizations actually reached here

DESIGN §2c's warning is that the forks share code by **copy**, so "nothing tells you the other
copy drifted". Checked rather than assumed. **The drift is bidirectional** — each fork has
something the other lacks:

| `bench_report.md` finding | `s2s/v2.0` | `PanguWeather` | on the Polaris path? |
|---|---|---|---|
| §4 **bf16** (+5.3% on H100) | env `S2S_AMP_DTYPE`, defaults **fp16** | ✅ YAML `amp_dtype`, **defaults `bfloat16`** — already on, and a better design than an env knob | ✅ yes — the green runs are bf16 |
| §4 `find_unused_parameters=False` | ✅ | ✅ | ✅ |
| §4 **`static_graph=True`** | ✅ | ❌ **missing** | ✅ would apply — **candidate** |
| `gradient_as_bucket_view=True` | ❌ **missing** | ✅ | drift the *other* way |
| §5 **`TORCH_COMPILE_MODE`** | ✅ | ❌ → ✅ **fixed here** | rung 1 |
| §4 batch **2**/card (+11.4% on H100) | config | ❌ config is batch 1 | candidate — but see §5 memory |
| per-iteration `empty_cache()` removal | ✅ | ✅ (independently) | ✅ |
| grad-norm without per-param `.item()` | ✅ (fused `grad_norm_and_max`) | ✅ (separate on-device `grad_norm`/`grad_max`) | ✅ |
| §7 checkpoint `os.path.isfile` guard | ✅ | ✅ (more call sites) | ✅ |
| §7 `--async_save` | ✅ (inference only) | ✅ (**more** files than s2s) | inference |
| NVTX `vae_encoder1/2` ranges (+ the backward-bracketing autograd trick) | ✅ | ❌ **missing** | ViT only — not on the SFNO path |

**The two real gaps are `static_graph=True` and (until now) the compile knob.** Everything
else either landed independently or is *ahead* in PanguWeather.

> `static_graph=True` is a **candidate, not a known win.** `bench_report.md` §4 changed bf16
> and `find_unused_parameters=False`+`static_graph=True` **together** and attributes +5.3% to
> the pair, so `static_graph`'s isolated contribution was never measured — and PanguWeather
> already has the expensive half (`find_unused_parameters=False`). Also: s2s needs a
> **dead-module freeze** (`layer_perturbation2`, `layer_purturbation_e2`, `train.py:437-444`)
> for `static_graph` to be legal; PanguWeather has **no such freeze**, so this cannot simply
> be copied across. Measure it, don't assume it.

### 6c. The ViT/Swin optimizations: there are none to port, and they are not on this path

Asked directly: have `bench_report.md`'s transformer/ViT optimizations been done in
PanguWeather? **Two independent reasons the answer is "the question doesn't bite yet":**

1. **They were never implemented in *either* fork.** `bench_report.md` §3's ViT findings —
   LayerNorm-backward is the 2nd-largest GPU consumer (17.3 s), memory-layout conversions
   (4.4 s), `roll` for shifted-window attention (7.7 s), matmul only 6th/18th — are
   **profiler observations, explicitly deferred to `torch.compile`** (§3: "Fusing LayerNorm
   via torch.compile is the single biggest optimisation the profile suggests"; §5: "in
   progress"). §5-ladder rung 2 (FlexAttention) is unstarted. A `diff` of the two
   `networks/pangu.py` files shows **the only perf-relevant divergence is s2s's NVTX
   instrumentation** — `F.scaled_dot_product_attention` is already in `EarthAttention3D` in
   **both** (s2s:1091/1099, PanguWeather:1079/1087), both have the same 2 `torch.roll`s and
   13 `LayerNorm`s, and both have transformer-block checkpointing commented out identically.
   So there is nothing to port: the ViT cores agree.
2. **The ViT does not run on Polaris.** The green E3SM path is `nettype: sfno_plasim`, which
   builds `networks/modulus_sfno/sfnonet.py` and **never touches `networks/pangu.py`**. The
   Swin/ViT (and its VAE, and this report's `vae_noise` hook) belong to `pangu_plasim`,
   blocked on PLASIM h5 that is not staged.

**But the two profiles agree on the lever, which is the interesting part.** The H100 ViT
(`bench_report.md` §3/§5: element-wise ops "the single largest GPU-time consumer … launched
as hundreds of small kernels", matmul ranks 6th) and this A100 SFNO (§4.2: **61%** elementwise
over ~1506 launches/step, GEMM 15%) are **different architectures that profile the same way** —
memory-bandwidth bound. ⚠ **They agree on *memory-bound*, not on *mechanism* — and the
"one conclusion" framing was overstated.** The ViT/Swin finding is *activation* layout (the
model flipping between channel-first and channel-last between layers); §4.5 shows this SFNO's
largest single item is a **weight** in the wrong layout, 133 ms/rank-step. Different
mechanisms reaching the same adjective. `torch.compile` is still rung 1 — cheap and already
wired, and now reachable on PanguWeather — but it is not the instrument for the weight half,
and ACE2 measured Inductor failing outright on the complex64 SFNO.

Status of the three DESIGN §4.0 prerequisites **for PanguWeather** (they were tracked for
`s2s/v2.0`; the trees are forks and share nothing, DESIGN §2c):

| prerequisite | state on PanguWeather |
|---|---|
| seed mechanism | ✅ **already existed — do not port `s2s`'s `seeding.py` here.** `train.py:3825` has `--global_seed` (default 0) feeding `seed_torch()` (`:3742`), called at `:3785`. It seeds `PYTHONHASHSEED`/numpy/torch/CUDA and sets `cudnn.benchmark=False` + `cudnn.deterministic=True`. It is **stronger than s2s's legacy path was** — the numpy gap that made s2s's baselines irreproducible does not exist here. Two competing seed mechanisms would be a regression. Gaps: Python's `random` is unseeded, and `torch.use_deterministic_algorithms(True)` is never set. |
| `tiny_baseline.yaml` | ✅ **written AND run — measured, not asserted** (job **7255583**, rc=0, `--config=TINY`, 1 GPU). See the table below. |
| VAE noise-fixing hook | ✅ **built** — `utils/vae_noise.py` + 16 tests (`VAE_NOISE_OK`). **But INERT on this path**: `sfno_plasim` is deterministic and has no VAE (DESIGN §2c). It gates `pangu_plasim`, which is blocked on PLASIM h5 that is not staged. |

### `tiny_baseline.yaml` is genuinely small — the measured delta

"Tiny" is a measurement, not a name. That is the whole `test.yaml` lesson (CLAUDE.md #12: a
config called *test* that is really the full model), so this was measured rather than sized
by arithmetic:

| | real SFNO (7255410) | **TINY** (7255583) | ratio |
|---|---|---|---|
| trainable params | 1,182,108,160 | **7,166,656** | **165× smaller** |
| step_med | 0.652 s | **0.023 s** | 28× faster |
| peak memory | 26.98 GB | **1.00 GB** | 27× less |
| step_std | 0.110 | 0.00015 | — |

A K=20-step §4.1 baseline is **~0.5 s of compute** and fits in 1 GB. It reports
`loader_wait_frac = 76.7%` — expected and harmless: the model is now so fast it outruns the
`num_data_workers: 0` loader §4.1 asks for. Irrelevant for an equivalence baseline (which
measures numbers, not speed), and a third independent confirmation that `loader_wait_frac`
tracks reality — it has now read 0.0%, 0.7%, 14.8% and 76.7% in the four situations where
each was the right answer.

**So all three §4.0 prerequisites are now met on PanguWeather.** What remains is to *capture*
the baseline (§4.1: fixed seed, world size 1, K=20 per-step loss trajectory + output summary
stats) — no longer blocked on building anything.

> **`cudnn.benchmark=False` + `deterministic=True` are already on** (via `seed_torch`, which
> always runs). That is a *performance* fact hiding in a reproducibility mechanism: this model
> is benchmarked in cuDNN's deterministic mode. Turning benchmark on might buy time, and would
> cost reproducibility. Not attempted — it is a hot-path change. Recorded so it is not
> mistaken for an oversight.

---

## 7. How to reproduce

```bash
cd PanguWeather/v2.0
qsub HPC_scripts/polaris_bench_e3sm_sfno.pbs                    # CSV bench, 4-GPU
qsub -v NUM_DATA_WORKERS=8 HPC_scripts/polaris_bench_e3sm_sfno.pbs
qsub HPC_scripts/polaris_bench_nsys_e3sm_sfno.pbs               # nsys trace + sqlite
qsub -v CONFIG_NAME=tiny_baseline,CONFIG_SECTION=TINY,NPROC=1 \
     HPC_scripts/polaris_bench_e3sm_sfno.pbs                    # the §4.0 small config

# tests — no GPU/data/cluster needed
python PanguWeather/v2.0/test/bench_instrumentation_test.py     # BENCH_INSTR_OK (9)
python PanguWeather/v2.0/test/vae_noise_test.py                 # VAE_NOISE_OK  (16)
```

**PASS is the work token, never `rc=0`** (the makani lesson: a resumable trainer exited 0
having trained zero steps). The CSV bench gates on the CSV **gaining a row**
(`ERROR NO_BENCH_ROW`); the nsys script gates on the trace containing **bench NVTX ranges**
(`ERROR NO_NVTX_TABLE` / `ERROR NO_BENCH_RANGES`) — nsys writes a `.nsys-rep` even when it
captured nothing, so file existence proves nothing.

Analysis: `python s2s/v2.0/HPC_scripts/parse_nsys.py <trace>.sqlite` (the nsys script runs it
automatically).

---

## 8. What is NOT established here

Stated plainly so nothing below reads as done:

* **SI, makani and physicsnemo are unprofiled on Polaris.** Only PanguWeather SFNO.
* **No baseline is captured.** `tiny_baseline.yaml` now runs (job 7255583), but the §4.1
  capture itself — fixed seed, world size 1, K=20 loss trajectory + output stats written to
  `baselines/` — has not been done. That is the next job, and it is no longer blocked.
* **No optimization was attempted or measured.** No `torch.compile`, no precision change, no
  DDP tuning, no `checkpointing` change. The §4 gate is not executable yet.
* **`TORCH_COMPILE_MODE` is wired and statically tested, but has NOT been exercised at
  runtime.** Deliberate: actually compiling the model *is* §5 rung 1, and the handoff's rule
  is that optimizing stays blocked until §4 is executable — so setting it once "just to see"
  would be starting the ladder without the gate. The test proves the env value reaches
  `torch.compile`; it does **not** prove `torch.compile` succeeds on this model. It may hit
  graph breaks or fail outright against `checkpointing: 3` / the SFNO's custom ops. Finding
  that out is rung 1's first task, after the baseline. Do not read "wired" as "works".
* **`static_graph=True` was not tried** — see the §6b box for why it is not a free copy from
  s2s (unmeasured in isolation, and it needs a dead-module freeze PanguWeather lacks).
* **The `workers=8` +9% is a bench result, not an endorsement** — see the `epsilon_factor`
  box in §3. It changes the loss trajectory.
* **The mechanism producing the weight copies is NOT established.** §4.5c's candidate (the
  einsum permuting `(i,o,x) → (x,i,o)` for `bmm`) is source-plausible but untested, its first
  two rivals were refuted from source (`spectral_layers`, `view_as_complex`), and the
  `assert factorization == "ComplexDense"` contradiction means we do not even know whether
  the weight is an `nn.Parameter` or a `FactorizedTensor`. Plan item 8, and it is free.
* **Single-run numbers.** Each sweep point is one job. The step distributions are tight
  (`workers=8` std = 0.3 ms) so within-run precision is high. **Partly superseded:** the
  nsys capture is now **n=2 across two nodes** (§4.4b) — **compute** reproduces to
  0.08–0.27%, **NCCL does not (+114.8%)**, and total step time moves **12.7%**. So a
  cross-job compute comparison is sound and a cross-job comparison of anything containing
  NCCL is not. The *sweep* points remain n=1 each, and their deltas (notably
  `num_data_workers` +9%) are inside that 12.7% spread.
* **The nsys numbers are eager and 40 steps**, vs the CSV bench's 80. The step medians agree
  (603.5 ms NVTX vs 602–652 ms CSV), which is the cross-check.
* **`ema` never fired** (`ema_warmup_epochs: 6`). Full training pays a per-step sweep over
  1.18 B params that these runs did not.

---

## Decisions / changes log

* **2026-07-15** — **Ported the `S2S_BENCH` + NVTX harness into PanguWeather** (it had zero).
  Range names and CSV columns byte-identical to s2s's (CLAUDE.md #10); `ema` added as a new
  range and to `parse_nsys.py`'s list (which skips absent ranges, so Midway traces are
  unaffected). Gated so unset knobs ⇒ legacy path byte-for-byte.
  **Proven, not asserted:** job **7255505** (no `S2S_BENCH`) reproduced the GREEN reference
  **7253591** exactly — train loss **0.3411**, valid_loss **0.7049359679222107**,
  bit-identical. Static gating tests are not a substitute for that smoke.
* **2026-07-15** — **`cpu_prep_frac` is not loader idle; built `loader_wait_*`.** It measures
  `_prepare_inputs_batch` on an already-fetched batch (0.3–0.6% of the step even at
  `workers=0`). The real fetch is between steps, in no step window — and it drove the
  elapsed-vs-sum self-check into *refusing the row* on exactly the runs where the loader is
  the finding. Now measured and reconciled. **Falsified before being believed**: `workers=0`
  moved it 0.7% → 14.8% (21×) while `cpu_prep_frac` stayed flat.
* **2026-07-15** — **`elapsed` was being sampled after `cudaProfilerStop()`.** The profiler's
  buffer flush landed inside the measured wall time; under nsys job **7255503** that read
  `elapsed=51.8 s` against `sum=25.7 s` (50% "disagreement") and discarded a perfectly good
  bench row — on **every** profiled run. The timers were fine; the clock was stopped in the
  wrong place. Fixed (`loop_end` sampled first) + an AST test pinning the ordering.
* **2026-07-15** — **VERDICT: GPU-bound.** `loader_wait_frac` = **0.7%** at the shipped
  `num_data_workers: 1` (job 7255410). The §5 kernel ladder is **not** premature.
* **2026-07-15** — **VERDICT: elementwise-bound, not matmul-bound.** 61% of GPU time in
  pointwise kernels over ~1506 launches/step vs 15% in GEMM (job 7255503). Memory-bandwidth
  bound and fusion-starved ⇒ `torch.compile` (§5 rung 1) is the right first lever, now on
  evidence rather than assumption. NCCL is 10.5% of kernel time but only **1.2% exposed** (plan §0b), so §5 rung 3 is
  worth ~1.2% on one node, not ≈5%; cuFFT/SHT is only 3.3%.
* **2026-07-15** — **The model is 1,182,108,160 params**, not the ~79M the docs assume — that
  figure is the Pangu/Swin model, not the E3SM SFNO. 26.98 GB peak of 40 GB.
* **2026-07-15** — **PanguWeather already has a seed knob** (`--global_seed` → `seed_torch`,
  seeding numpy + torch + CUDA and forcing `cudnn.deterministic`). The handoff implied it
  needed porting from `s2s/v2.0/utils/seeding.py`; it does **not**, and porting would create
  two competing mechanisms. This also explains why the 0.3411 green is bit-reproducible.
* **2026-07-15** — **`num_data_workers` is not output-neutral**: the loader draws per-sample
  gaussian noise inside the workers (`epsilon_factor: 0.1`) with no `worker_init_fn`, so the
  worker count changes the noise realization and moves the loss. The `1 → 8` (+9%, and 10×
  less jitter) is recorded as a **finding, not a recommendation**; the clean fix is a seeded
  `worker_init_fn`. ⚠ **The +9% is NOT established** — it is a **cross-job** delta (7255410
  vs 7255480), n=1 per point, and §4.4b measures two identical-config jobs differing 12.7%
  in total per-rank-step GPU time. Plan §5 measures **0.991×** for the same knob in a
  *within-job* sweep. Treat +9% as unmeasured until it is re-run interleaved.
* **2026-08-20** — **§4.3 added: the §4.1↔§4.2 join, done correctly.** The 42.2% copy time
  splits `backward` **72.9%** / `forward_loss` **27.1%**; `(outside)` is **0.0%**, so the
  step's "missing" ~268 ms holds zero launches and is pure drain. Two join bugs fixed first
  (a missing `globalPid` guard, **+29.4%** phantom; and thread- rather than process-scoped
  attribution, which had been credited to an NVTX limitation for months). D2D memcpy above
  L2 sustains **82% of HBM peak**, so §0d's estimated 17–27% for the copy kernels is about
  the path they take. Plan items 1 and 5. Preregistered `985214b5`, 3/3 hit.
* **2026-08-20** — **§4.4 added: a share of the GPU-kernel total is not a reproducible
  quantity.** On two identical-config runs (**different nodes**) the copies read 42.2% and
  37.4% — **4.77 points** — while the numerator moved **0.09%**. Compute reproduces
  (+0.08% on quiet steps); NCCL does not (**+114.8%**). ⇒ quote **271 ms/rank-step**, not a
  share. **VERDICT: cross-job *compute* comparisons on Polaris are sound; cross-job
  comparisons of anything containing NCCL are not.** The reproducible stall at the same
  training iteration on both nodes is **CPython gen-2 GC** (`gc_collect_main`), not NUMA —
  output-neutral to fix. Plan item 2. Preregistered `952fcb8d`, 5/5 hit. An adversarial pass
  landed 4 FATAL strikes on the first draft; §4.4f records all seven withdrawals.
* **2026-08-20** — **§4.5 added: 133 ms/rank-step is one weight in the wrong layout.** The
  `dhconv` spectral weight is `[in, out, lmax]` complex = **377.49 MB/layer**, and 12 layers
  are **95.8%** of the 1.18 B params — the largest tensor in the model. It is moved in
  **four** places, all from one mismatch (stored `(in,out,lmax)`, contracted by
  `einsum("bixy,iox->boxy")` which permutes to `(x,i,o)` for `bmm`): forward permute 35.60,
  `ckpt3` recompute 35.61, adjoint `conj` 35.93, and **`grad_w` → DDP bucket 26.01** — the
  last of which is **not invariant**, so invariant movement is **107.14 ms = 17.8% of the
  step**, not the 22.0% first published. **VERDICT: fusing pointwise activation ops touches
  none of these four rows**, so the standing "61% elementwise ⇒ fuse" read covers only half
  the copy time; DESIGN §5 gains rung **1b**. Scoped: weight:activation is `E/(B·mmax)`, so
  the finding **inverts at batch 4**. Plan item **3**. Preregistered `45cbd7de`, **0/4 size
  predictions hit — and the miss is the deliverable.** An adversarial pass landed 2 FATAL
  strikes and two tool bugs (a 4× launch-path under-count, and `C_in`); §4.5d records all
  seven withdrawals.
