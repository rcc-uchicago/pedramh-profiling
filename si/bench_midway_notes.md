# SI Benchmark & Performance Report

**Dates:** 2026-05-20 – 2026-05-21
**Hardware under test:** Midway `pedramh-gpu` — 4 × NVIDIA H100 NVL (93.0 GiB each), verified from the nsys profiles; plus single-GPU smoke tests on the same H100 NVL node and on a shared-partition A100.
**Model:** `SI_X` — a stochastic-interpolant (DynamicInterpolant) generative weather model with a low-resolution Diffusion Transformer backbone (`dim=1536`, 24 blocks, 8 cross-attention blocks) operating on a 45×90 downsampled grid; trained in bfloat16-mixed with DDP across 4 GPUs.

---

## Summary

The `pedramh-gpu` node is four NVIDIA H100 NVL cards with 93.0 GiB of memory each, read directly from the profile's device table; it is neither the "80 GB H100" assumed in the earlier notes nor the "H200" recorded in a commit message. Because the card's true capacity is 93.0 GiB, the full benchmark's 64.7 GiB peak corresponds to 69.5 % of memory rather than the "81 % of 80 GB" stated previously, which means the available headroom, about 28 GiB, is considerably larger than was reported.

The benchmark is compute-bound: across the measured steps the GPUs are busy roughly 94 % of the time. Enlarging the DDP gradient buckets and compressing them to bfloat16 reduced communication from 622 to 407 ms per step and the step time from 928 to 735 ms, an improvement of about 21 %, while leaving the compute itself unchanged at 637 ms per step in both runs. The optimized step is therefore now about 87 % compute-bound, and any further communication tuning can recover at most the 98 ms of NCCL time that is not yet overlapped with the backward pass.

Because the step is now dominated by computation, the remaining opportunities lie there. The largest contributors are roughly 3,500 small element-wise kernels per step, which together account for 211 ms, and the attention and MLP matrix multiplications, which account for a further 174 ms; both are well suited to the kernel fusion that `torch.compile` provides and that is already available behind a configuration flag. The spherical-harmonic transform, which we had expected to be costly, turns out to be negligible — a single kernel contributing essentially no time per step — and can be set aside.

---

# Part I — Hardware identity and run environment

Before any throughput figure can be trusted, the hardware that produced it has to be identified correctly, and on this point three sources disagreed: the earlier notes described the node as "80 GB H100s," a commit message tagged it `pedramh-gpu (h200)`, and the recorded memory peak was 64.7 GB. The profile resolves the question. Nsight Systems records the device inventory in its `TARGET_INFO_GPU` table, and the entry is identical across both captured jobs and all four ranks.

| Field | Value |
|---|---|
| `name` | NVIDIA H100 NVL |
| `chipName` / compute capability | GH100 / 9.0 |
| `totalMemory` | 99,875,094,528 B = 93.0 GiB |
| `memoryBandwidth` | 4.02 TB/s |
| `smCount` / L2 | 132 / 60 MiB |

*Source: `sqlite3 si_nvtx_49898288_rank0.sqlite "SELECT * FROM TARGET_INFO_GPU"`, confirmed identical in job 49887289 and across ranks 0–3.*

Three of these properties identify the card unambiguously. The device name is reported literally as NVIDIA H100 NVL; the framebuffer is 93.0 GiB, whereas an H200 would report roughly 141 GB and an 80 GB SXM card about 79.6 GiB; and the HBM bandwidth of 4.02 TB/s matches the NVL specification of approximately 3.9 TB/s rather than the 3.35 TB/s of the 80 GB SXM part. The compute capability of 9.0 further excludes the A100, which reports 8.0. The commit message's "h200" is therefore incorrect, and the "80 GB" figure that recurs throughout the earlier notes refers to the wrong memory class; the correct denominator for any statement about the fraction of GPU memory in use is 93.0 GiB. The job log confirms this independently, as its opening lines print `GPU 0: NVIDIA H100 NVL`.¹

The single-GPU H100 smoke test ran on this same node, since `bench_gpu_test_h100.sh` requests the same `-p pedramh-gpu` partition as the full benchmark, so its "80 GB" label likewise refers to the NVL card and should read 93 GiB. The A100 smoke test is the only piece of hardware that could not be verified from the raw data: it ran on a different partition (`-p gpu --constraint=a100`) and produced no nsys profile, so its 40 GB size and node identity rest on the run prose rather than measurement.² A compute capability of 8.0 holds for any A100, and 40 GB is plausible for Midway3's standard cards, but a single `nvidia-smi -q` line in that job would put the matter beyond doubt.

---

# Part II — Throughput, memory, and the profile

## 1. What the benchmark measures, and how it differs from production

`bench.py` runs the ordinary training loop under `BenchCallback`, which brackets every step with `torch.cuda.synchronize()` so that the recorded time reflects GPU execution rather than CPU submission. It discards a warm-up of 20 steps (40 when `torch.compile` is active) to absorb driver initialisation and cuDNN autotuning, then times the measured window and writes a single row to a CSV. Each training step comprises the familiar four phases: loading a batch and moving it to the GPU, the forward pass to the interpolant loss, the backward pass for gradients, and the optimizer update.

The benchmark is deliberately not the production configuration, and three differences are worth keeping in mind when reading the throughput figures. First, `accumulate_grad_batches` is fixed at 1 so that every step contains exactly one optimizer call and one gradient all-reduce, whereas production uses 2; because DDP suppresses the all-reduce on accumulation sub-steps, the per-sample communication cost is roughly halved in production, and the benchmark consequently overstates the share of the step spent in NCCL. Second, the benchmark uses Adam while production uses Muon, whose Newton–Schulz iterations add matrix-multiplication work that the benchmark does not exercise. Third, `multistep_rollout` is set to 1, taking the single-`compute_loss` path, so the activation-memory and compute multiplication that a multi-step rollout would introduce is absent. None of these choices invalidate the benchmark; they simply make it a clean and uniform measurement of compute, and production can be expected to be somewhat less communication-bound and somewhat more optimizer-heavy.

## 2. Memory footprint

The `peak_mem_gb_max_rank` column is computed as `max_memory_allocated() / 1024³` (`bench_callback.py:170`), so despite the `_gb_` in its name it is expressed in GiB, and it counts allocated bytes, which means the true occupancy reported by `nvidia-smi` is somewhat higher. With the correct denominator, the full benchmark's 64.7 GiB peak corresponds to 69.5 % of the card's 93.0 GiB,³ leaving roughly 28 GiB of nominal headroom rather than the approximately 15 GB implied by the earlier "81 % of 80 GB" framing.

The single-GPU runs explain the shape of this figure. At fp32 with a batch size of 1 the model peaks at 38.5 GiB on the NVL, which is why the NVL is the only single-GPU target that accommodates fp32 at all. The A100 bf16 run peaks at 31.0 GiB, higher than a naive "half of fp32" estimate would suggest, because under bfloat16-mixed only the activations are cast to bf16; the weights, the Adam moments, the gradients, and the fp32 master copy all remain in fp32. The roughly 19 GiB of model state therefore does not shrink, and only the roughly 12 GiB activation slice does. The same fact explains why four samples per card reach 64.7 GiB rather than four times the single-sample peak: only the activations scale with batch size. In practical terms, at 69.5 % occupancy there is genuine room to raise the batch size to 5 or 6 on the NVL for additional throughput before activation checkpointing would become necessary.

## 3. Throughput

| Metric | Smoke (1× NVL, fp32, bs=1) | Full (4× NVL, bf16-mixed, bs=4) |
|---|---|---|
| Median step time | 0.659 s | 0.884 s |
| Throughput | 1.52 samples/s | 18.10 samples/s |
| Wall-clock throughput | — | 17.72 samples/s |
| Data-idle fraction | — | 0.021 |
| Peak memory | 38.5 GiB | 64.7 GiB |

*Source: `bench_test_h100_results.csv` (git 55a07f6); `bench_pedramh_node_results.csv` (git 2d12ae9, config 237cae5c318de8e4).*

The most important figure in this table is the data-idle fraction of 0.021: across the measured window the GPUs sat idle waiting on the dataloader for only about 2 % of the time, which establishes that this is a compute-bound benchmark. Reaching that point required some effort, because the first multi-GPU attempt failed its own sanity check with the GPU idle for roughly 46 % of every step while waiting on HDF5 reads,⁴ and the fix — eight dataloader workers, eight CPUs per task, and `OMP_NUM_THREADS=1` — is what closed the gap. Scaling from a single fp32 sample to sixteen bf16 samples increased the per-step time by only a factor of 1.34, so the four-GPU configuration delivers about 11.9 times the single-GPU throughput, a strong-scaling efficiency of roughly 74 %, with the remaining quarter attributable to the change from fp32 to bf16 and to NCCL. A rough projection places an epoch near 45 minutes, though this depends on the actual dataset length and should be checked before being used for estimates.

One correctness patch underlies all of these measurements. The `RealSHT` and `InverseRealSHT` operations in `torch_harmonics` call `torch.view_as_complex`, which does not accept bfloat16, so `SphereNoiseGenerator.forward` wraps its inverse transform in `autocast(enabled=False)` to keep spherical-noise generation in fp32. The cost is negligible (§4 below), and the companion `SphericalSpectralProjector` would require the same treatment only if `spectral_weight` were nonzero, which the benchmark leaves at zero.

## 4. Where the step time goes

The two NVTX-instrumented captures — job 49887289 before the DDP changes and job 49898288 afterward — each contain 19 fully captured measured steps.⁵ Utilisation is the first quantity to recompute carefully, because the simple sum of kernel durations is misleading when communication overlaps compute on a separate stream; the meaningful figure is the union of the busy intervals.

| Per-rank-0 metric | Baseline (49887289) | DDP-optimized (49898288) |
|---|---|---|
| GPU busy (union of all kernels) | 94.4 % of wall | 93.6 % |
| …including memcpy and memset | 96.6 % | 96.7 % |
| Compute-only occupancy (NCCL excluded) | 68.0 % (637 ms/step) | 85.4 % (637 ms/step) |
| NCCL-only occupancy | 66.4 % (622 ms/step) | 54.5 % (407 ms/step) |
| Inter-kernel idle gaps | 51 ms/step (5.5 %) | 47 ms/step (6.3 %) |

*Source: union of kernel `(end − start)` intervals over `CUPTI_ACTIVITY_KIND_KERNEL` within the measured step window; queries in the Appendix.*

The most informative result here is the quantity that did not change: the compute-only occupancy is 637 ms per step in both runs. The DDP optimization reduced the step time without altering the compute at all, which is the expected behaviour, and in doing so it raised compute occupancy from 68 % to 85 % and reduced the un-overlapped communication from roughly a quarter of the wall-clock time to under a tenth. This implies a firm ceiling on any further communication work: with the step at 735 ms and compute at 637 ms, only about 98 ms remains as non-overlapped NCCL, so even ideal communication could recover no more than about 13 %, while the larger share of the step lies in the 637 ms of compute. The 94 % union figure is also consistent with the CSV's data-idle fraction of 0.021, so two independent measurements agree that the workload keeps the GPU occupied.

The same captures allow the compute to be broken down by kernel category, shown here for the optimized run on rank 0.

| Category | ms/step | Launches/step | Identity |
|---|---|---|---|
| NCCL (communication) | 407 | ~29 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` |
| Element-wise / pointwise | 211 | ~3,500 | AdaLN-Zero scale/shift/gate/add chains |
| GEMM (cuBLAS/cutlass) | 174 | ~500 | `nvjet_*` attention and MLP matmuls |
| Flash attention, backward | 109 | 34 | `cudnn_…_sdpa_sm90_flash_bprop_wgmma_f16` |
| `cat` (`CatArrayBatchedCopy`) | 42 | ~370 | `assemble_input` / `assemble_forcing` |
| Flash attention, forward | 40 | 34 | `…_flash_fprop_wgmma_f16` |
| Reduce / norm | 29 | ~530 | LayerNorm and reductions |
| Adam (`multi_tensor_apply`) | 28 | ~420 | optimizer |
| FFT / SHT | ~0 | 1 | spherical-noise inverse transform (fp32) |

*Source: grouped `SUM(end − start)` over `CUPTI_ACTIVITY_KIND_KERNEL` joined to `StringIds`.*

Two categories deserve particular attention. The first is element-wise work, which at 211 ms per step spread across roughly 3,500 individual kernel launches is the largest single component of compute, exceeding even the matrix multiplications, and which did not appear in the earlier breakdown at all. This is characteristic of the AdaLN-Zero block, in which every transformer block applies a chain of small scale, shift, gate, and residual-add operations, each launched as its own small kernel, and it is exactly the kind of work that kernel fusion addresses. The matrix multiplications themselves account for about 174 ms per step, somewhat more than the earlier figure of "~120 ms," which had counted only the `nvjet_*` subset. Flash attention is already in use, evident from the `flash_*_wgmma_f16` cuDNN kernels, so no work remains there, and the host-to-device transfer is not a concern at about 20.5 ms per step, moving roughly 311 MB through pinned memory each step.⁶

The second category worth noting is the spherical-harmonic transform, which we had expected might be expensive. The data is clear on this point: there is one FFT kernel per step, contributing essentially no measurable time. With `spectral_weight` set to zero there is no spectral-loss transform, and the noise generator's single fp32 inverse transform on the 45×90 grid is too small to register. The transform is therefore not a useful optimization target, and the recommendations below state as much explicitly.

## 5. The DDP optimization, before and after

This communication change is the most substantial improvement in the record, and its mechanism is worth describing. DDP does not all-reduce each gradient tensor individually, since that would issue thousands of small NCCL calls per step and waste the interconnect on per-call overhead. Instead it packs gradients into fixed-byte buckets that fill in reverse layer order, firing one all-reduce as each bucket becomes full while the backward pass continues to produce the next bucket's gradients. Small buckets imply many calls and more overhead but little stalling, whereas large buckets imply fewer calls and better bandwidth utilisation at the risk that the final bucket delays the optimizer step. The default cap of 25 MB placed this model at 118 buckets per step, each paying its own launch and synchronisation cost.

| Per-rank-0 metric | Baseline | DDP-optimized | Change |
|---|---|---|---|
| Median step time | 928.4 ms | 735.0 ms | −20.8 % |
| NCCL calls per step | 118 | 25 | 4.7× fewer |
| Average per call | 5.27 ms | 16.27 ms | larger payload |
| NCCL total per step | 622 ms | 407 ms | −34.6 % |
| `forward_loss` median | 111.9 ms | 111.9 ms | unchanged |
| `backward` / `optimizer` (NVTX) | — | 317.0 / 171.4 ms | new ranges |

*Source: AllReduce kernels in `CUPTI_ACTIVITY_KIND_KERNEL` together with `NVTX_EVENTS`; these reproduce the earlier notes to the decimal.*

Raising the bucket cap to 200 MB, registering the bfloat16 gradient-compression hook, and switching to bucket-view gradients reduced the step time from 928 to 735 ms. The kernel name itself confirms that the hook is active, since `…AllReduce_Sum_bf16_RING_LL` carries the bfloat16 payload, and the call count fell from 118 to 25 in line with the bucket model. The projected throughput is about 21.8 samples per second, an increase of roughly 20 % over 18.1, but this figure is derived from the profile, which carries about 5 % overhead, and has not yet been confirmed by a clean production-shape benchmark.

A note of caution applies to the deeper communication analysis attempted in the earlier notes. Statements of the form "an ideal 17 MB all-reduce takes about 20 µs, so we are 270 times off" and "5 GB of gradients at 900 GB/s implies a 5.5 ms floor" rest on quantities that were not in fact measured here. The parameter count was never logged, so the gradient volume is known only approximately, somewhere in the range of 3 to 5 GB in fp32 and roughly half that on the bfloat16 wire. The duration of an NCCL kernel includes the cross-rank synchronisation wait as well as launch overhead, so interpreting it as pure per-call overhead overstates that component. And 900 GB/s is the SXM NVLink figure, whereas the NVL bridge runs slower. The remedy is inexpensive — logging `sum(p.numel())` and capturing the NCCL byte volume with a communication-domain trace — and until that is done the "floor" is better regarded as an open question than as a target.

## 6. Comparison across GPUs at matched precision

A direct comparison of step times between the A100 and the H100 NVL is misleading, because the H100 smoke test ran in fp32 — the only precision that fit when it was taken — while the A100 ran in bf16. The appropriate comparison is the per-sample time at a common precision, bfloat16: the A100 requires 0.410 s per sample, while the NVL requires 0.221 s per sample in the full benchmark and 0.184 s after the DDP fix. The H100 NVL is therefore about 1.85 times faster than the A100 in bf16, which is consistent with NVIDIA's published BF16 tensor-core ratios and reassuring, since it indicates that nothing in the SI code path is leaving A100-specific performance unrealised. The cost trade-off is the conventional one: the NVL is about 1.85 times faster but typically two to three times more expensive per GPU-hour, so it is preferable for raw throughput while the A100 remains competitive for development iteration. The A100's own memory-headroom figures depend on the unverified 40 GB size noted in Part I.

---

# Part III — Optimization opportunities

With the step now dominated by computation, the remaining opportunities are concentrated there, and the most promising of them is already implemented. The recommendations below are ordered by expected return, and each is tied to the evidence that motivates it and to the run that would confirm it.

1. The first recommendation is to enable `torch.compile` for the DiT. The mechanism is already present behind the `SI_TORCH_COMPILE` flag and disabled by default. Because the step is about 87 % compute-bound and roughly 211 ms of that compute is spent in some 3,500 small element-wise kernels, with a further 47 ms per step lost to inter-kernel idle, fusing these operations is the natural target, and a reduction of 10 to 18 % in step time is plausible. The effect can be confirmed with `bench_nvtx_compile.sh`, with the warm-up raised to 40 steps, by observing whether the `forward_loss` and `backward` ranges and the total kernel count fall. This is low-effort work.

2. The training dataloader should be given `persistent_workers=True` and a larger `prefetch_factor`. The loader in `amip_new.py:182` currently sets neither, whereas the loader in `bias.py:186` already sets both alongside pinned memory, so the training path is the inconsistent one; respawning eight workers at each epoch boundary is wasted work, and avoiding it protects the data-idle fraction over long runs. This change is trivial.

3. The batch size can be raised to 5 or 6. At 69.5 % memory occupancy there is roughly 28 GiB of headroom on the NVL, more than the earlier accounting suggested, and larger batches improve the efficiency of the matrix multiplications and attention. The peak-memory column should be watched as the batch grows. This is low-effort work.

4. The concatenations in `assemble_input` and `assemble_forcing` can be fused. They appear as roughly 370 `CatArrayBatchedCopy` launches and 42 ms per step, and preallocating the output and copying slices into it — or simply allowing the `torch.compile` change above to absorb them — would remove most of that cost. This is a medium-effort change.

5. A bucket cap of 400 MB is worth an ablation. The optimized run still issues 25 buckets at an average of about 117 MB, rather than the roughly 15 it could in principle pack into 200 MB, so a little more overhead could be amortised, though with diminishing returns given the 98 ms ceiling on communication savings. This is trivial to test.

6. The parameter count and NCCL byte volume should be logged, and the NVL's NVLink bandwidth confirmed. Doing so is what would make the communication-floor analysis in §5 rigorous rather than speculative. This is trivial.

7. The bfloat16 gradient-compression hook should be validated numerically before it reaches production, by comparing the loss trajectory over about 1,000 steps against full-precision gradients. This is low-effort work.

Three further candidates can be ruled out. The spherical-harmonic transform is negligible, at a single kernel and essentially no time per step. Flash attention is already enabled. And `channels_last` would offer little here, because the workload consists of attention and MLP operations — matrix multiplications, element-wise kernels, and flash-attention kernels — rather than convolutions, the patchify step being the only convolution present.

---

## Appendix — reproducing the numbers

```bash
# Hardware identity (Part I)
sqlite3 si_nvtx_49898288_rank0.sqlite \
  "SELECT name, totalMemory, memoryBandwidth, smCount, computeMajor, computeMinor FROM TARGET_INFO_GPU;"

# NVTX phase medians (Part II §4–§5)
sqlite3 F.sqlite "SELECT text,(end-start)/1e6 ms FROM NVTX_EVENTS \
  WHERE text IN ('forward_loss','backward','optimizer','preprocess') AND end>start;"

# NCCL AllReduce kernels
sqlite3 F.sqlite "SELECT count(*), SUM((k.end-k.start)/1e3) us FROM CUPTI_ACTIVITY_KIND_KERNEL k \
  JOIN StringIds s ON s.id=k.shortName WHERE s.value LIKE '%AllReduce%';"

# Utilisation = union of kernel intervals / step window; category breakdown = GROUP BY
# kernel name on CUPTI_ACTIVITY_KIND_KERNEL; H2D bandwidth from CUPTI_ACTIVITY_KIND_MEMCPY copyKind=1.
```

---

### Footnotes

¹ `si_bench_bench_midway.sh_49874162.out`, line 3: `GPU 0: NVIDIA H100 NVL (UUID: …)`.

² The A100 ran via `bench_gpu_test_a100.sh` on `-p gpu --constraint=a100`, which left no nsys profile. A compute capability of 8.0 holds for any A100; the specific 40 GB size and node identity are from the run prose only. A single `nvidia-smi -q` in that job would verify it.

³ `64.685 GiB / (99,875,094,528 / 1024³ = 93.0 GiB) = 69.5 %`.

⁴ `[BenchCallback] WARN sanity check failed (elapsed=129.194s expected~70.281s ratio=0.46)` — per-step compute of about 0.88 s against about 1.62 s of wall-clock, that is, roughly 46 % idle on HDF5 reads. Resolved by raising `num_data_workers` from 4 to 8 and `--cpus-per-task` from 4 to 8 with `OMP_NUM_THREADS=1`.

⁵ Both captures contain the NVTX ranges `step_21..step_39`, which is 19 complete steps. `SI_BENCH_STEPS=20` was requested, but one step's range fell outside the capture window. The earlier notes' diagnosis correctly divides by 19 even where the prose says "20."

⁶ About 206 host-to-device transfers across 19 steps, totalling roughly 5.9 GB at an aggregate rate of about 15 GB/s. The transfers are pinned (`amip_new.py:188`), and at about 2 % of the step they are well off the critical path; the sub-PCIe-Gen5 rate is a curiosity rather than a bottleneck.
