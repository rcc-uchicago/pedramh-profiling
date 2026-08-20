# Polaris profiling handoff — SFNO family (ACE2 / PanguWeather / ai-rossby)

Written 2026-08-19, **revised 2026-08-20** after re-deriving every figure from the
captures on disk. Rows marked OPEN are not settled — do not quote them as results.
Companion to `polaris_bench_report.md` (the existing A100
profile), `ACE2_retrain/bench_midway_notes.md` (ACE2 detail),
`PanguWeather/v2.0/bench_midway_notes.md`, and `ACE2_retrain/PROFILING_PLAN.md`
(the full ranked plan this summarises).

**All three models share the NVIDIA Modulus SFNO backbone**, so most findings
transfer. Three axes do NOT transfer, and every number below is scoped by them:

- **Interconnect** — `midway3-0423` is a pair-bridge node (NV12 in-pair, `SYS`
  cross-pair, 261 vs 18.3 GB/s measured, `topocheck_53539369.out`).
- **Batch size** — the Midway ACE2 captures run batch 4; production is 16.
  Exposed all-reduce is 44% of GPU-busy time at batch 4 on H100 and **7%** at
  batch 16 on H200 (jobs 53534648 / 53483668). Batch moves these numbers more
  than the interconnect does.
- **Software stack** — ACE2 ran NCCL 2.26.2 / torch 2.7.1+cu126, Pangu NCCL
  2.27.5 / torch 2.9.1+cu129. An env var honoured by one was not honoured by
  the other (§2).

---

## 1. Corrections to earlier claims — read before quoting anything

Several conclusions recorded during the Midway work do not survive scrutiny.
They were re-derived from captures already on disk, at no GPU cost.

| earlier claim | status |
|---|---|
| "the ~9% training idle is **launch latency**" | **REFUTED, on the capture that produced it and on two others.** Job 53479120 is 91.4% GPU-busy / 8.6% idle as recorded — and its median launch→execute queue depth is **45.0 ms**. On the NVTX reference job 53534648 it is **8.87 ms** (p90 20.4 ms, 97.1% positive), and it is deep in *every* phase, not just around the all-reduce: `forward_loss` 8.0 ms, `backward` 11.0 ms. The CPU spends only **7.5%** of that window in `cudaLaunchKernel` (0.820 s of 10.91 s, 7.9 µs/call). Both figures require the **`globalPid`-guarded** join; the naive join gives 8.62 ms / p90 21.8 ms and is wrong. The CPU-side cost is instead `cudaStreamSynchronize`: 7.0 calls/step, 163 ms of a 347 ms step — **but do not add that to the row below; ~75% of it is the same exposed all-reduce (see the note after this table).** |
| "NCCL kernel time is an **upper bound** on comm cost" | **RESOLVED, and worse than the caveat.** Per-device kernel totals differ by <2.2% (10.232/10.453/10.446/10.444 s), so it is not straggler wait. On **4× H100 NVL at batch 4**, **35.7% of wall-clock is *exposed* all-reduce** and only 26.3% of NCCL overlaps compute (job 53534648; 33.2–37.7% and 25.4–30.6% across the four ranks). **Per-node, per-batch, and — newly — PER-RUN.** `polaris_bench_report.md` §4.4b: two
   identical-config Polaris runs give NCCL **67.82 vs 145.65 ms/rank-step (+114.8%)** with
   the same launch counts, entirely wait. Balance is a per-run draw, so one capture's
   `<2.2%` per-device spread does **not** establish "not straggler wait" as a property —
   re-check per-rank balance (`--per-rank`) in every capture before quoting an exposed
   fraction. The range across captures on disk is 44% → 30% → 7% of GPU-busy time; see §2. |
| "GPU occupancy 91%" | **Mislabelled.** That is *GPU-busy fraction* (union of kernel intervals), not warp occupancy. Quote the job with it: **91.4% on job 53479120** (dev0, A100), **81.0% on the reference job 53534648** (dev0, H100 NVL). Real occupancy needs ncu. |
| "GEMM 7.6%" | **Low by ~32%, but not for the reason recorded.** The classifier keyed on `demangledName` already, matching the substring `gemm` — which *does* catch cutlass `Kernel2<…c1688gemm…>`. The only family it misses is cuBLAS `nvjet_*`, whose names contain no "gemm". Correct ACE2 H100 figure: 7.55% → **11.22%** (job 53534648). `nvjet_*` is **Hopper-only (sm90)**: same env and `cuda/12.6`, job 53534648 (H100) = 3.67% nvjet, job 53479120 (A100) = **0.000%**. **So Polaris is unaffected — see §3 Tier A.** The tables that *are* low are the Hopper ones: Pangu Midway H100 5.71% → **6.71%**, ACE2 8×H200 11.76% → **17.45%** (job 53483668 node0). Fix the pattern list, not the column. |
| "`log_snapshots=false` halves validation" | **Half the lever was missed.** `log_mean_maps` is the sibling flag; images are **82.6%** of warm validation and arm E still burned 10.81 s. Also the render is **O(1) per epoch**, so the *percentage* does not transfer to production's ~2900-sample window — the absolute seconds do. **This row is right and `ACE2_retrain/bench_midway_notes.md` (validation section) is wrong** where it says "the absolute seconds scale but the *proportions* are what transfer" — `snapshot.py:49-61` only rebinds `_target_data`/`_gen_data` per batch, so the render is genuinely O(1)/epoch. Fix the note, not this row. |
| "`torch.compile` corrector = −2.2%, ±0.5%" | **Upheld on replication; only the harness caveat survives.** Two independent interleaved round-robin series (n=4 per arm) reproduce it: jobs **53586889-900** give none 0.3427 s / corrector 0.3357 s (**−2.04%**) / mlp 0.3540 s (+3.28%), and jobs **53707930-941** give 0.3418 / 0.3350 (**−1.98%**) / 0.3528 (+3.22%). In both series the three ranges do not overlap and run-to-run spread is ≤1.3%. The second series **reverses the arm order** (`mlp, corrector, none` instead of `none, corrector, mlp`), which de-confounds arm from slot — and the effects follow the arm, not the slot: `none` 0.3427 (pos 1) → 0.3417 (pos 3), `mlp` 0.3540 (pos 3) → 0.3528 (pos 1), while per-**position** medians swing (position 1: 0.3427 → 0.3528). Effect sizes computed independently within each block agree to 0.06 pp. DVFS and warm-up order are therefore both excluded. **The corrector is a real ~2% win and is the one compile target worth carrying to Polaris.** What still stands: `step_med_s` is a delta between log lines emitted after `metrics_aggregator.get_metrics()` (a `dist.reduce_mean` + D2H) and `wandb.log` (`trainer.py:573`), so it is not a pure GPU step time. That contamination is common-mode across arms — it bounds the *absolute* step, not the *difference*. Use CUDA-event timing before quoting any absolute step figure. |
| "A100 → H100 was a clean hardware swap" | **~81% of the epoch delta is one single-threaded matplotlib call** on a different node (snapshot logging: 164.27 s vs 17.86 s = 146.41 s of a 179.87 s epoch delta), not GPU speed. It is ~97% of the *validation* delta, which is probably the denominator that was meant. |

> **The sync row and the exposed-all-reduce row are the same time. Do not sum them.**
> On dev0 of job 53534648 over 30 steps (span 10.9137 s): `cudaStreamSynchronize`
> union = 5.110 s (46.8%), exposed all-reduce = 3.899 s (35.7%), and their
> **intersection is 3.820 s — 98.0% of all exposed NCCL and 74.8% of all sync
> time.** Sync with no NCCL kernel resident at all is **0.604 s = 20 ms/step =
> 5.5% of the window**. That 20 ms/step is the entire *sync-elimination*
> headroom; everything else labelled "sync" is the §2 all-reduce seen from the
> CPU side. `ACE2_retrain/PROFILING_PLAN.md:46` states this; the table above
> must not drop the qualifier.

**Method notes that caused these**, worth carrying to any future capture:

- A correlationId join across a multi-rank single-report capture **must be
  guarded on `globalPid`**, or kernels are attributed to the wrong rank.
- Bucket kernels by **`demangledName`** with an explicit pattern list —
  `gemm|cutlass|nvjet|xmma` — and keep an `(unclassified)` bucket that fails a
  gate above ~2%. The defect was never the column: matching only `gemm` silently
  drops cuBLAS `nvjet_*` on Hopper (3.7 pp on ACE2 H100, 5.7 pp on H200, 1.0 pp
  on Pangu H100; **0.0 pp on A100**).
- Our NVTX SQL **discarded 1,392 rows, 1,368 of them NCCL's own `ncclAllReduce`
  ranges** (domainId=1, registered strings). NCCL instruments itself; we were
  throwing it away.

---

## 2. The one measurement that reorders everything

**Exposed vs overlapped communication.** On 4× H100 NVL (`midway3-0423`) **at
batch 4**, ACE2 spends ~130 ms of every ~347 ms step with the GPU running a
**6-SM ring kernel** (of 132 SMs) and nothing else — 35.7% of wall-clock, 44% of
GPU-busy time (job 53534648, all four ranks 33.2–37.7%).

**Scope, before you quote that number.** It is the top of a wide range, and
batch size moves it further than the interconnect does. Exposed all-reduce as a
fraction of GPU-busy time (span-independent, so windowed and full captures
compare):

| capture | exposed / GPU-busy |
|---|---|
| 4× H100 NVL, batch 4 (53534648, 53524918) | **44–47%** |
| 4× A100, batch 4 (53479120) | **30%** |
| 8× H200, 2 nodes, batch 16 (53483668) | **7%** (1.6% of span; 71% of NCCL overlaps) |

So 35.7% is a `midway3-0423`-at-batch-4 property, not an SFNO-family one.
**Measure it on Polaris rather than assuming it** (§3 Tier A item 3).

And the protocol is the surprise: NCCL selected **`RING_LL`** for ~160 MB
buckets. LL interleaves a 4-byte flag per 4 bytes of payload — **2× the wire
bytes** — and exists for latency-bound *small* messages. That much is confirmed
by kernel name on every capture in this repo, for both models.

**OPEN — the bandwidth arithmetic does not close, so "the link is saturated" is
not established.** Per rank per step ACE2 moves 1.5 × 1.823 GB = 2.735 GB of
ring egress in 176.3 ms of NCCL kernel time = **15.5 GB/s of user bytes**.
Doubling that for LL's encoding gives **31.0 GB/s of wire on a link measured at
18.3 GB/s** — impossible by 1.7×. The repo now carries three incompatible values
for this link (18.3 measured, ~26–27 from the 2-rank probe, 35 asserted in
`bench_midway_notes.md`). Until that is reconciled, **do not quote a saturation
fraction and do not claim a 2× headroom for `Simple`.** Settle it with
`NCCL_DEBUG=INFO` + `NCCL_DEBUG_SUBSYS=INIT,TUNING`, which prints the ring
topology and the channel count NCCL actually chose.

⇒ **Run an `NCCL_PROTO` / `NCCL_ALGO` sweep before any code optimisation.**
One job, one env var per arm, zero code change, targeting 35.7% of wall-clock.
Arms: default, `NCCL_PROTO=Simple`, `NCCL_PROTO=LL128`, `NCCL_ALGO=Tree`.

**MEASURED SINCE, on 4× H100 NVL — single-run, and the arms are NOT
protocol-verified. Treat every cell as provisional:**

| arm | ACE2 (1.82 GB grads) | PanguWeather (4.73 GB grads) |
|---|---|---|
| `NCCL_PROTO=Simple` | **-0.15% — real, and expected** (`Simple` and the default are within 1% on this node: 23.76 vs 23.55 ms on a standalone 256 MB all-reduce, jobs 53724297/300) | -7.8% *(n=1)* |
| `NCCL_PROTO=LL128` | +27% slower *(n=1, arm unverified)* | -12.9% faster *(n=1, arm unverified)* |
| `NCCL_ALGO=Tree` | fails (no AllGather) | fails (no AllGather) |

**The two mechanisms previously offered for the split are both withdrawn.**

- *Not topology.* Both sweeps ran on **the same node**, `midway3-0423`, with
  byte-identical GPU UUIDs. A property of a link cannot make LL128 +27% slower
  for one model and 12.9% faster for the other.
- *Not message size.* The messages are 160 MB (ACE2) and 315 MB (Pangu) — six
  orders of magnitude above LL128's 128-byte granularity and only 2× apart.
  There is no threshold between them.

The one **confirmed** asymmetry is the software stack: ACE2 ran NCCL **2.26.2**
(torch 2.7.1+cu126), Pangu NCCL **2.27.5** (torch 2.9.1+cu129). Correct
`PanguWeather/v2.0/bench_midway_notes.md:129`, which says "same NCCL build".

⇒ **Sweep per software stack, and verify each arm by kernel name.** After each
arm, run `SELECT s.value, COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN
StringIds s ON k.demangledName = s.id WHERE s.value LIKE '%nccl%' GROUP BY 1` —
the kernel name is ground truth, an env var being set is not — or
`NCCL_DEBUG=INFO` **with** `NCCL_DEBUG_SUBSYS=INIT,TUNING`. Note that
`NCCL_DEBUG_SUBSYS` alone prints nothing: job 53707483 set it without
`NCCL_DEBUG` and produced zero NCCL lines. Then replicate ≥3 interleaved reps
per arm and require non-overlapping ranges, the pattern that settled the compile
sweep (jobs 53586889-900).

This applies to **Polaris too, and has never been checked there.** Pangu's
Polaris profile reports NCCL at 10.5% — low enough that protocol choice was
never questioned, but the check is nearly free and the gradient volume is
2.6× ACE2's (4.728 GB vs 1.823 GB).

**But first, read Pangu's Midway "NCCL 74.2%" correctly — it is not all
gradient exchange.** On job 53539872 it splits into `AllReduce_Sum_f32_RING_LL`
**41.05%** and `ncclDevKernel_Broadcast_RING_LL` **33.14%**. The Broadcast fires
exactly once per rank-step (160 launches = 40 steps × 4 ranks), inside
`forward_loss`, with per-device medians of **534 / 509 / 4 / 498 ms** — three
ranks waiting on one. That is DDP's per-forward buffer sync, not gradients (see
§3 Tier B item 5). On a like-for-like all-reduce basis Pangu is **41.1%**
against ACE2's **51.4%** — the *opposite* ordering to the "NCCL share vs
gradient volume" story in `PROFILING_TABLES.md`.

---

## 3. What to do on Polaris, in order

**Tier A — no GPU time. Re-analysis of captures you already have.**

1. ~~Re-bucket `polaris_bench_report.md`'s kernel table by `demangledName`.~~
   **Checked — NOT needed. The blind spot cannot reach Polaris.** It is
   `nvjet_*`, an sm90-only cuBLAS family that is **0.000%** on the A100 capture
   53479120, where a bare `gemm` match already equals the complete GEMM family
   (11.21%). Polaris is A100-SXM4. The Polaris table is already
   `demangledName`-keyed — it counts "64 distinct kernels" where the same model
   on Midway has 63 distinct `demangledName` against 34 distinct `shortName` —
   and its residual `other` bucket is 0.9%, capping any misclassification at
   ≤0.9 pp against the ~4.5 pp the claim needs. **"61% elementwise / 15.1%
   GEMM" stands; rung 1 of the ladder keeps its justification.** The tables that
   do need re-bucketing are the Hopper ones (§1).
2. Re-run any correlationId attribution with a `globalPid` guard.
3. Compute **exposed vs overlapped NCCL** on the existing Pangu capture — union
   of NCCL intervals minus their intersection with compute. **Correction: the
   `backward` spread of 112–993 ms is NOT the signature ACE2 showed.** ACE2 is
   balanced (per-device kernel totals 10.232/10.453/10.446/10.444 s, <2.2%);
   Pangu 53539872 is a **straggler** — its per-step Broadcast is 534/509/4/498 ms
   across the four devices, i.e. three ranks wait ~500 ms every step for one.
   Expect a different fix (see Tier B item 5), and note the capture used
   `PANGU_BENCH_WARMUP=20`, which its own notes call contaminated — it straddles
   the 560 ms → 1090 ms regime change, so re-capture at warmup ≥40 before
   trusting any percentage from it.
4. Stop discarding NCCL's own NVTX ranges in the parse SQL.

**Tier B — cheap jobs.**

5. **`broadcast_buffers=False` — Midway only; it DOES NOT TRANSFER to Polaris.**
   ⚠ **Superseded 2026-08-20 — do not spend a job on this.** The check this item asked
   for has now been run twice: `PANGU_POLARIS_PROFILING_PLAN.md` §0c measures the
   broadcast at **112 ms of 102.9 s = 0.11%** of GPU kernel time on Polaris, and
   `polaris_bench_report.md` §4.3c confirms it is present exactly once per rank-step and
   **100% inside `forward_loss`** at 0.7 ms/rank-step. Midway's 33.14% was three ranks
   *waiting* on a straggler, not the broadcast itself. Opening a jesswan science gate (BN
   running statistics) for 0.1% is not worth it. The code pointers below stay accurate.
   `PanguWeather/v2.0/train.py:298-303`
   constructs `DistributedDataParallel` without `broadcast_buffers`, so it
   defaults to `True` and DDP broadcasts every buffer on every forward; ACE2
   sets it `False` (`ace_exp/fme/core/distributed/torch_distributed.py:192`,
   comment "no per-step buffer broadcast"). That broadcast is **33.14% of all
   GPU kernel time** on Midway job 53539872 — but **0.11% on Polaris** (see above), so
   the "larger than every row in §4 combined" framing is Midway-specific and must not be
   carried across. **Gate it:** buffers are BN running statistics, so this changes
   what the model computes across ranks — it needs an equivalence check and
   jesswan's sign-off (DESIGN §4.1, CLAUDE.md "Division of labor"), not a
   silent flip.
6. `NCCL_PROTO`/`NCCL_ALGO` sweep (§2), for Pangu and ai-rossby — **verify each
   arm by kernel name before recording it** (§2), and run ≥3 interleaved reps.
7. Add the **SFNO-internal NVTX ranges** both models lack —
   `spectral_filter`, `sfno_block`, `sfno_mlp`, `sht_fwd`/`sht_inv`. Pangu
   already emits the shared phase names (`data_prep`, `forward_loss`,
   `backward`, `optimizer`, `ema`, `to_ensemble_batch`), so this is purely the
   layer below. Use an injector like `ACE2_retrain/ace2_nvtx.py` — **never edit
   the trees**: PanguWeather is a fork (no propagation) and
   `physicsnemo_ai_rossby` is a **git subtree** (edits conflict on pull).
8. `nsys --python-sampling=true` on any CPU-heavy phase. Supported on Arm SBSA,
   so it works on Delta GH200 as well.

**Tier C — needs care.**

9. ncu for real occupancy and achieved bandwidth, **single-rank only**
   (`--nproc_per_node=1`). Kernel replay re-executes `ncclDevKernel`, which
   spins on peer flags → deadlock or corruption, and a stalled rank trips the
   collective watchdog. Use an explicit metric list, not `--set full`.

---

## 4. Code-level findings that transfer to Pangu and ai-rossby

| finding | transfers |
|---|---|
| **Spectral no-op copy** — `s2convolutions.py:196` does `zeros_like` + slice-assign with no guard; ACE2 guards it (`ace_exp/fme/ace/models/modulus/s2convolutions.py:176`), both other trees do not. **The guard fires unconditionally — not because of `hard_thresholding_fraction: 1.0`.** `sfnonet.py:481-503` builds every SHT with `lmax=modes_lat, mmax=modes_lon`, so the transform truncates internally and the spectral tensor is *already* (modes_lat, modes_lon); the inverse transform asserts `x.shape[-2]==lmax`. The slice is full-width at **any** thresholding fraction — do not re-derive this from the YAML. **Measured size (job 53539872):** ⚠ **REVISED 2026-08-20 — not sub-1%, and the 0.18% was doubly wrong.** It counted only the
   zero-*fills*, not the full-width slice-assign **copy** the same guard removes — and
   `polaris_bench_report.md` §4.5b sizes that copy on Polaris: `direct_copy`⟨complex64,nocast⟩,
   **48 calls/rank-step at 133.45 MB = 5.3% of copy time = 14.4 ms/rank-step ≈ 2.4% of the
   step**, plus the fills. It was also a share of a Midway kernel total that is **74.2%
   NCCL**, which §4.4c forbids quoting. Corroborated independently by
   `ACE2_retrain/bench_midway_notes.md:123-126` ("Pangu performs 12 × (allocate + zero-fill
   + full copy) of a complex64 tensor per forward, for no result"). ⇒ **re-rank this: it is
   among the cheapest levers in §4, not a shelved one.** Original figure, for the record:
   the complex64 fills the guard removes are 308.7 ms of 174,873 ms = **0.18% of kernel time**, 1.93 ms of a 1093 ms rank-step (36 launches/step = 12 layers × 3). *All* zero-fills of every dtype = 0.59%. | **yes — but sub-1% class.** Compare Tier B item 5 (33%) and `NCCL_PROTO` (§2). |
| **`FourierNeuralOperatorBlock.forward`** (`sfnonet.py:225,241` Pangu / `:234,251` ai-rossby) does `zeros(...)` + full-width slice-assign twice per block. Removal *looks* bitwise identical there — the buffer is fp32 and the capture shows every norm running fp32 — **but this has never been run, and `baselines/` holds only `ai_rossby_pangu_plasim/` and `ai_rossby_sfno/`: there is no PanguWeather equivalence baseline, so the DESIGN §4.1 gate cannot be closed for Pangu.** **OPEN, not a finding.** Tolerance floors for whichever gate you build: 2.5e-7 same GPU/node, ~1e-5 cross-architecture. | yes, and cleaner there — **after a baseline exists** |
| **cutlass `align1` complex GEMM** — bigger in Pangu (12 layers × 512) than ACE2 (8 × 384). **Note the exemplar caveat:** `67242e348` is not an upstream ai2cm/fme commit — it is one of 4 local commits ahead of `main` in the vendored clone, and `CHANGELOG.md:253-256` records it as a hot-path change that "arrives already applied and has **never been equivalence-checked**". ACE2's guards are a pattern worth copying, not a validated one. | **yes** |
| AMP/SHT boundary copies (`x.float()`, `.contiguous()`, `.to(dtype)`) | **yes** — identical code |
| Interconnect / two-level all-reduce | **no** — `midway3-0423` (H100 NVL) only: NV12 in-pair, `SYS` cross-pair, **measured** 261 GB/s vs **18.3 GB/s** (`topocheck_53539369.out`). Delta `gh121` is **measured** NV6 full mesh, 126–132 GB/s all pairs (`PROFILING_TABLES.md:158-176`). **Polaris is inferred, not measured — OPEN.** No `nvidia-smi topo -m` or pairwise matrix exists for any Polaris node in this repo. The inference: **⚠ Use the MINIMUM observed NCCL time, not the mean: NCCL kernel time is an upper bound
on transfer time because it includes wait, so the implied bandwidth is a lower bound. On
7255503 excluding its one stalled step that is 59.67 ms ⇒ ≥79 GB/s; the identical config's
other capture reads 145.65 ms ⇒ ~32 GB/s, which is wait, not link, and would spuriously
fail to exclude a 25 GB/s hop. Still OPEN — run `gpu_topology_check.py`.**
Pangu moves 4.73 GB of fp32 gradients in **67.8 ms** of NCCL per 652 ms rank-step (job 7255410) ⇒ **≥69.7 GB/s** algorithm bandwidth, where a ring all-reduce over an 18.3 / 25 / 64 GB/s slowest hop would cost 388 / 284 / 111 ms. A PCIe-class cross-pair hop is excluded by 1.6–5.7×, so the operative advice holds — but **run `gpu_topology_check.py` in the first Polaris allocation and paste the matrix here.** It costs one minute and closes the only unmeasured cell in this table. |
| Validation / `log_snapshots` / `log_mean_maps` | **no** — fme aggregators, ACE2 only |

---

## 5. Tools ready to use

| tool | what it does | needs |
|---|---|---|
| `gpu_topology_check.py` | measures real pairwise GPU bandwidth; distinguishes full mesh from pair bridges. **Run this first — Polaris topology is the one unmeasured cell in §4, and "A100" is a device name, not a topology** (this repo's own `beagle3-0012` was A100-**PCIE**, no NVLink; Polaris reports A100-**SXM4**, which is the NVLink form factor, but that has never been confirmed by a bandwidth measurement here). | torch, a GPU allocation |
| `ACE2_retrain/nvtx_phase_attribution.py` | GPU kernel **time** per NVTX phase — pid-guarded **and process-scoped** join, launch-time bucketing, `--by-exec` sensitivity check, `--memcpy` (L2-aware bandwidth), `--per-step` (stall/warmup series), sum **and** union columns. Refuses nested ranges rather than double-counting. Ships a passing test needing no capture. This is the tool to reach for; it produced `polaris_bench_report.md` §4.3. | nsys sqlite (runs on a login node) |
| `ACE2_retrain/kernel_census.py` | attributes kernel **count** per NVTX range — ✅ **FIXED 2026-08-20** (plan item 4): it now *delegates* the join to `nvtx_phase_attribution.py` and agrees with `polaris_bench_report.md` §4.3b row for row (354,720 launches / 102.911 s / `backward` 250,880 / no `(outside)`). It also now tests its own thesis instead of asserting it, and reports **no batching target** on either Pangu capture (largest count-minus-time skew **+3.2 pt** / **+2.0 pt** vs a +10 pt bar). **For the record, what it used to print, because three docs cited those numbers as an NVTX limitation:** (a) Line **58** joined `ON k.correlationId = r.correlationId` with no `globalPid` guard, which §1 says is mandatory: on `n.sqlite` that yields **630,428 rows against 481,841 kernels (+30.8%)**, and the guarded join returns exactly 481,841. The published copy-attribution table has never been re-derived with the guard — with it, the `sfno_block` and `sfno_mlp` rows collapse to 0.0%; they are cross-rank phantoms. (b) `enclosing(rs, tid)` (lines 37-49) looks the NVTX range up on the **launching thread**, but PyTorch's autograd engine launches from its own worker: on Pangu job 7255503 rank 0, **62,680 of 88,680 launches** come from `pt_autograd_*` while all the house NVTX ranges are on the main thread — so it credits `(outside)` for the whole of `backward` *even after the guard lands*. This is the real origin of the "57% / 81% outside any range" rows, and it was never an NVTX limitation. Both are fixed by the delegation above, as is a third bug nobody had noticed: the normaliser counted distinct `step_%` starts (**156**) instead of **160 rank-steps**. The docstring's refuted "launches wreck the pipeline / ~9% idle" story is retired too. ⚠ **The ACE2 copy-attribution table still needs re-deriving** — the fix makes that possible but nobody has run it on an ACE2 capture. | any nsys sqlite with NVTX |
| `ACE2_retrain/ace2_nvtx.py` | injects NVTX without editing the tree — the pattern to copy for Pangu/ai-rossby | — |
| `ACE2_retrain/parse_nsys.py` | house-format NVTX summary, ACE2 range names added | nsys sqlite |
| `ACE2_retrain/PROFILING_PLAN.md` | the full ranked plan, with dead ends called out | — |

**Structural hazard, already hit once:** `parse_nsys.py` keeps its range-name
list **twice** — the SQL `WHERE text IN (…)` at line 83 and the print loop at
line 100. Extending only the query fetches rows and silently never prints them,
indistinguishable from "the instrumentation did not fire". The bug was diagnosed
and the lists reconciled, and the file now carries the warning comment at line
102 — but the drift is **still live**: `unstack` is in the SQL and not in the
print loop (benign only because nothing calls `unstack`). **When you add
`spectral_filter`/`sht_fwd` ranges for Pangu, edit both lists**, or better,
hoist the tuple to a module constant and use it in both places.

---

## 6. Dead ends — do not spend time here

1. **CUDA Graphs** for launch overhead. Median launch→execute queue depth is
   8.87 ms on job 53534648 and **45 ms** on job 53479120; a `cudaLaunchKernel`
   call costs 7.9 µs and all of them together are 7.5% of the window. There is
   no CPU launch bottleneck to remove. *Caveat, so nobody re-opens this on the
   wrong grounds:* dev0 **is** GPU-idle 19.0% of that window, and only 0.13 s of
   the idle sits inside a sync — 1.94 s is idle with the CPU unblocked, in
   22,967 gaps of 10–100 µs (median 35 µs), 1.03 s of it inside `backward`.
   With an 11 ms-deep queue during `backward` that is stream / DDP-bucket
   **dependency**, not launch starvation, and a graph does not remove a
   dependency. Still fix the false comment at `midway_bench_nsys.sh:56-58`; do
   not fund the work.
2. **Ranking by kernel count.** The premise (launches wreck the pipeline) is
   refuted. Keep count attribution, drop the ranking.
3. **Grid-size tuning on compute kernels.** Non-NCCL mean SM fraction is 0.988.
   The half-empty-kernel story is 100% NCCL.
4. **`channels_last`.** `torch_harmonics` wants NCHW; NHWC would turn two cheap
   `.contiguous()` calls into real transposes for ~2.8% of GPU time.
5. **ncu under real DDP** (deadlocks on `ncclDevKernel` replay).
6. **Raising `num_data_workers` — ACE2-on-Midway only. NOT a dead end for
   PanguWeather.** ACE2's `config_midway.yaml:66,82` already ships **8**, where
   8→16 gains nothing (34.20 s vs 34.43 s) and dev0 idle *between* NVTX steps is
   2.5–3.9% of span. But that is *kernel* idle, not loader wait —
   `bench_midway_notes.md:802` explicitly forbids comparing it to a
   `loader_wait_frac`, and ACE2's loader wait is **unmeasured**.
   **All three shipped SFNO configs use `num_data_workers: 1`**
   (`E3SM_SFNO_H5_{POLARIS,POLARIS_ALLDATA,MIDWAY}.yaml`), and there 1 → 8 is a
   measured **+9% wall throughput** (6.09 → 6.64 samples/s; jobs 7255410 /
   7255480) with p90−median collapsing 174 ms → 1 ms. Pangu's Midway
   `loader_wait_frac` is **8.8%**. Even ACE2's own arm C measures 1 worker as
   **+28%**. It is withheld only because the loader draws per-sample
   `torch.randn` (`utils/data_loader_multifiles.py:1031,1102`,
   `epsilon_factor: 0.01`) with **no `worker_init_fn`**, so worker count changes
   the noise realization and the win cannot pass the DESIGN §4 bitwise gate.
   That is a **correctness blocker with a known fix** (CHANGELOG "Next actions"
   #4), not a dead end.
7. **`GradScaler` removal as a throughput win.** It is 0.35% of GPU time. Worth
   doing for correctness (bf16 needs no loss scaling) — note
   `optimization.py:117` ties `enabled` to `gscaler is not None`, so naive
   removal silently disables AMP.
