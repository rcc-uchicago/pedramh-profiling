# Polaris profiling handoff — SFNO family (ACE2 / PanguWeather / ai-rossby)

Written 2026-08-19. Companion to `polaris_bench_report.md` (the existing A100
profile), `ACE2_retrain/bench_midway_notes.md` (ACE2 detail),
`PanguWeather/v2.0/bench_midway_notes.md`, and `ACE2_retrain/PROFILING_PLAN.md`
(the full ranked plan this summarises).

**All three models share the NVIDIA Modulus SFNO backbone**, so most findings
transfer. What does NOT transfer is anything about `midway3-0423`'s interconnect.

---

## 1. Corrections to earlier claims — read before quoting anything

Several conclusions recorded during the Midway work do not survive scrutiny.
They were re-derived from captures already on disk, at no GPU cost.

| earlier claim | status |
|---|---|
| "the ~9% training idle is **launch latency**" | **REFUTED.** Median launch→execute queue depth is **8.87 ms** (p90 20.4 ms); the CPU runs thousands of kernels ahead and spends only 7.5% of the window in `cudaLaunchKernel`. The real cost is **`cudaStreamSynchronize`: 7.0 calls per step, 163 ms of a 346 ms step.** It is syncs, not launches. |
| "NCCL kernel time is an **upper bound** on comm cost" | **RESOLVED, and worse than the caveat.** Per-device kernel totals differ by <2%, so it is not straggler wait. **35.7% of wall-clock is *exposed* all-reduce** — only 26.3% of NCCL overlaps compute. |
| "GPU occupancy 91%" | **Mislabelled.** That is *GPU-busy fraction* (union of kernel intervals), not warp occupancy — and it is 81.0% on the reference job. Real occupancy needs ncu. |
| "GEMM 7.6%" | **Low by ~30%.** The bucket classifier keyed on `shortName` and missed cutlass `Kernel2` and cuBLAS `nvjet_*`. Correct figure ≈ **10.9%**. Same blind spot applies to Pangu's "15.1% GEMM" on Polaris. |
| "`log_snapshots=false` halves validation" | **Half the lever was missed.** `log_mean_maps` is the sibling flag; images are **82.6%** of warm validation and arm E still burned 10.81 s. Also the render is **O(1) per epoch**, so the *percentage* does not transfer to production's ~2900-sample window — the absolute seconds do. |
| "`torch.compile` corrector = −2.2%, ±0.5%" | **Not resolvable by that harness.** Step deltas were taken from log lines emitted after a `dist.reduce_mean`, a `wandb.log`→`barrier()` and a D2H. One job per arm, no clock logging. 2.2% is inside H100 NVL DVFS drift. |
| "A100 → H100 was a clean hardware swap" | **~90% of the epoch delta is one single-threaded matplotlib call** on a different node (snapshot logging: 164 s vs 18 s), not GPU speed. |

**Method notes that caused these**, worth carrying to any future capture:

- A correlationId join across a multi-rank single-report capture **must be
  guarded on `globalPid`**, or kernels are attributed to the wrong rank.
- Bucket kernels by **`demangledName`**, never `shortName`, and keep an
  `(unclassified)` bucket that fails a gate above ~2%.
- Our NVTX SQL **discarded 1,392 rows, 1,368 of them NCCL's own `ncclAllReduce`
  ranges** (domainId=1, registered strings). NCCL instruments itself; we were
  throwing it away.

---

## 2. The one measurement that reorders everything

**Exposed vs overlapped communication.** On 4× H100 NVL, ACE2 spends ~130 ms of
every 346 ms step with the GPU running a **6-SM ring kernel** (of 132 SMs) and
nothing else.

And the protocol is the surprise: NCCL selected **`RING_LL`** for ~165 MB
buckets. LL interleaves a 4-byte flag per 4 bytes of payload — **2× the wire
bytes** — and exists for latency-bound *small* messages. Measured effective
bandwidth is 15.5 GB/s of user bytes on an 18.3 GB/s link, i.e. the link is
~85% saturated *in the LL encoding*.

⇒ **Run an `NCCL_PROTO` / `NCCL_ALGO` sweep before any code optimisation.**
One job, one env var per arm, zero code change, targeting 35.7% of wall-clock.
Arms: default, `NCCL_PROTO=Simple`, `NCCL_PROTO=LL128`, `NCCL_ALGO=Tree`.

**MEASURED SINCE, on 4x H100 NVL — and it is model-dependent:**

| arm | ACE2 (1.82 GB grads) | PanguWeather (4.73 GB grads) |
|---|---|---|
| `NCCL_PROTO=Simple` | -0.15% (noise) | **-7.8%** |
| `NCCL_PROTO=LL128` | **+27% SLOWER** | **-12.9% FASTER** |
| `NCCL_ALGO=Tree` | fails (no AllGather) | fails (no AllGather) |

So the sweep is worth running **per model**, and neither result generalises. For
ACE2 the protocol is not the lever and the link is simply saturated; for Pangu
LL128 is a free 12.9%. LL128's dependence on 128-byte store atomicity (an NVLink
property this node's cross-pair PCIe hop lacks) plausibly explains the ACE2
regression, and message size the Pangu gain.

This applies to **Polaris too, and has never been checked there.** Pangu's
Polaris profile reports NCCL at 10.5% — low enough that protocol choice was
never questioned, but the check is nearly free and the gradient volume is
2.6× ACE2's.

---

## 3. What to do on Polaris, in order

**Tier A — no GPU time. Re-analysis of captures you already have.**

1. Re-bucket `polaris_bench_report.md`'s kernel table by `demangledName`. The
   "61% elementwise / 15.1% GEMM" split has the `Kernel2`/`nvjet` blind spot.
2. Re-run any correlationId attribution with a `globalPid` guard.
3. Compute **exposed vs overlapped NCCL** on the existing Pangu capture — union
   of NCCL intervals minus their intersection with compute. Pangu's `backward`
   spans 112–993 ms; that spread is the same signature ACE2 showed.
4. Stop discarding NCCL's own NVTX ranges in the parse SQL.

**Tier B — cheap jobs.**

5. `NCCL_PROTO`/`NCCL_ALGO` sweep (§2), for Pangu and ai-rossby.
6. Add the **SFNO-internal NVTX ranges** both models lack —
   `spectral_filter`, `sfno_block`, `sfno_mlp`, `sht_fwd`/`sht_inv`. Pangu
   already emits the shared phase names (`data_prep`, `forward_loss`,
   `backward`, `optimizer`, `ema`, `to_ensemble_batch`), so this is purely the
   layer below. Use an injector like `ACE2_retrain/ace2_nvtx.py` — **never edit
   the trees**: PanguWeather is a fork (no propagation) and
   `physicsnemo_ai_rossby` is a **git subtree** (edits conflict on pull).
7. `nsys --python-sampling=true` on any CPU-heavy phase. Supported on Arm SBSA,
   so it works on Delta GH200 as well.

**Tier C — needs care.**

8. ncu for real occupancy and achieved bandwidth, **single-rank only**
   (`--nproc_per_node=1`). Kernel replay re-executes `ncclDevKernel`, which
   spins on peer flags → deadlock or corruption, and a stalled rank trips the
   collective watchdog. Use an explicit metric list, not `--set full`.

---

## 4. Code-level findings that transfer to Pangu and ai-rossby

| finding | transfers |
|---|---|
| **Spectral no-op copy** — `s2convolutions.py:196` does `zeros_like` + slice-assign with no guard. ACE2's vendored perf commit `67242e348` guards it; **both other trees lack it**, and every Pangu config sets `hard_thresholding_fraction: 1.0`, the exact condition where the guard always fires. Complex64, 12 layers at embed_dim 512. | **yes** |
| **`FourierNeuralOperatorBlock.forward`** does `zeros(...)` + full-width slice-assign twice per block. Bitwise identical to remove in Pangu/ai-rossby (their buffer and `norm0` are both fp32); in ACE2 the cast must be kept. | **yes, and cleaner there** |
| **cutlass `align1` complex GEMM** — bigger in Pangu (12 layers × 512) than ACE2 (8 × 384). | **yes** |
| AMP/SHT boundary copies (`x.float()`, `.contiguous()`, `.to(dtype)`) | **yes** — identical code |
| Interconnect / two-level all-reduce | **no** — Midway H100 NVL only. Polaris A100 and Delta GH200 (NV6 full mesh, measured 126–132 GB/s all pairs) do not have the pair-bridge split. |
| Validation / `log_snapshots` / `log_mean_maps` | **no** — fme aggregators, ACE2 only |

---

## 5. Tools ready to use

| tool | what it does | needs |
|---|---|---|
| `gpu_topology_check.py` | measures real pairwise GPU bandwidth; distinguishes full mesh from pair bridges | torch, a GPU allocation |
| `ACE2_retrain/kernel_census.py` | attributes kernel **count** per NVTX range | any nsys sqlite with NVTX |
| `ACE2_retrain/ace2_nvtx.py` | injects NVTX without editing the tree — the pattern to copy for Pangu/ai-rossby | — |
| `ACE2_retrain/parse_nsys.py` | house-format NVTX summary, ACE2 range names added | nsys sqlite |
| `ACE2_retrain/PROFILING_PLAN.md` | the full ranked plan, with dead ends called out | — |

**Known bug to fix first:** `parse_nsys.py` keeps its range-name list **twice**
(SQL + print loop). Extending only the query fetches rows and silently never
prints them — indistinguishable from "the instrumentation did not fire".

---

## 6. Dead ends — do not spend time here

1. **CUDA Graphs** for launch overhead. Queue depth is already 8.87 ms; there is
   no launch bottleneck to fix.
2. **Ranking by kernel count.** The premise (launches wreck the pipeline) is
   refuted. Keep count attribution, drop the ranking.
3. **Grid-size tuning on compute kernels.** Non-NCCL mean SM fraction is 0.988.
   The half-empty-kernel story is 100% NCCL.
4. **`channels_last`.** `torch_harmonics` wants NCHW; NHWC would turn two cheap
   `.contiguous()` calls into real transposes for ~2.8% of GPU time.
5. **ncu under real DDP** (deadlocks on `ncclDevKernel` replay).
6. **Raising `num_data_workers`.** 8 saturates; inter-step idle is 3.0%.
7. **`GradScaler` removal as a throughput win.** It is 0.35% of GPU time. Worth
   doing for correctness (bf16 needs no loss scaling) — note
   `optimization.py:117` ties `enabled` to `gscaler is not None`, so naive
   removal silently disables AMP.
