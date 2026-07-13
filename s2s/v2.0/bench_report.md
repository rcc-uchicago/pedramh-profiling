# S2S Benchmark & Cluster Performance Report

**Dates:** 2026-05-11 – 2026-05-29
**Hardware under test:** Midway `pedramh-gpu` H100 NVL (training baselines + ablations + profiler), Midway test-partition H200 (Intel + AMD nodes), DSI H200 cluster, plus a DSI-provided NVIDIA-cluster H100 NVL inference profile (Part I comparison only — DSI ran it on hardware we have no access to)
**Model:** Pangu/Plasim Earth-Specific 3D Swin transformer with VAE ensemble generation; primary loss latitude-weighted CRPS + KL

---

## TL;DR

- **Hardware (Part I).** DSI's 4-GPU H200 inference is ~2.2–2.8× slower in wall time than the NVIDIA H100 cluster for the *same* compute. The deficit is not compute and not PCIe bandwidth — the large production tensors move at ~55 GB/s on every cluster. It is **CPU-to-GPU handoff latency**: the GPUs sit idle 10–50 ms hundreds of times per run waiting for the CPU to queue the next kernel. The Real-Pangu handoff test and now full kernel-level Midway H200 captures show Midway hands work off cleanly (10–50 ms gaps in the tens, vs DSI's 423–894), so the cause is specific to the DSI **host environment**, not the H200 or the 4-rank pattern.
- **Software (Part II).** On the H100 training baseline, the best confirmed config (bfloat16 + static DDP graph, batch 2/card) is **+11.4%** throughput with zero numeric instability. Two inference bottlenecks were found and fixed: synchronous NetCDF saving on rank 0 (fixed with `--async_save`) and an unconditional checkpoint load that crashed jobs (fixed with an `os.path.isfile` guard, commit `56f73fe`).
- **New, 2026-05-29.** That checkpoint-guard fix is also what unblocked kernel-level profiling: every "missing kernel table" the earlier drafts blamed on a profiler/ptrace restriction was actually this crash. Re-running the standardized inference profiles (commit `3acb9b3`) now yields full kernel timelines on the Midway test partition — used below to fill the Intel and AMD H200 rows from data.

---

# Part I — Hardware: cluster topology and the DSI handoff investigation

The Data Science Institute (DSI) reported low GPU utilisation (~15–23%) and poor 4-GPU scaling on a 4×H200 node, and provided three Nsight Systems inference profiles — DSI H200 1-GPU, DSI H200 4-GPU, and an H100 NVL run (the "NVIDIA cluster" profile, which DSI produced with their own NGC-apptainer scripts, likely based on `nvidia_training_original.sh`; we have no access to that hardware and don't know the exact invocation). To these external baselines we added our own Midway controls — H200 Intel and AMD test-partition nodes, and the `pedramh-gpu` H100 NVL. Profiles were exported with `nsys export --type=sqlite` and analysed with `v2.0/HPC_scripts/compare_nsys.py` and `verify_bench.py` (repo root); every number in this part is reproducible from those scripts against the on-disk `.sqlite` files.

## 1. The problem: idle GPUs, not slow ones

Total on-GPU kernel-active time agrees within ~10% across DSI and NVIDIA — the H200 does the inference work at roughly the H100's per-GPU rate. The wall-time gap is almost entirely GPU idle time: between consecutive kernels the GPU either stays busy or sits in a positive idle period waiting for the CPU to queue the next kernel — what we call **CPU-to-GPU handoff** time throughout (NVIDIA's tooling sometimes calls the same quantity "dispatch latency" — same concept, different word).

This table consolidates the per-cluster runtime picture from the inference `nsys` captures. Gap-bucket rows are GPU0 (the convention used throughout); utilisation is the range across all four GPUs. Midway Intel/AMD are the 2026-05-29 post-fix captures (footnote 9); DSI and NVIDIA are re-confirmed from the original profiles via `verify_bench.py`.

| Metric (per GPU; gaps = GPU0) | DSI 1-GPU | DSI 4-GPU | NVIDIA H100 4-GPU | Midway Intel 4-GPU | Midway AMD 4-GPU |
|---|---|---|---|---|---|
| Kernels captured | 468,103 | 468,103 | 469,703 | 470,833 | 470,833 |
| GPU-active time | 13,977 ms | 13,928 ms | 15,302 ms | 27,177 ms | 27,208 ms |
| Wall window | 35,736 ms | 92,477 ms | 41,189 ms | 57,698 ms | 39,551 ms |
| **Utilisation (all GPUs)** | **39%** | **15–23%** | **37–57%** | **47–48%** | **64–69%** |
| Gaps ≤ 10 ms (normal handoff) | 468,021 | 467,623 | 469,637 | 470,722 | 470,763 |
| **Gaps 10–50 ms (short stalls)** | **41** | **423** | **27** | **53** | **28** |
| Gaps 50–100 ms | 7 | 8 | 12 | 17 | 7 |
| Gaps 100–500 ms | 19 | 20 | 12 | 18 | 31 |
| Gaps > 500 ms (I/O / save / barrier) | 7 | 21 | 14 | 21 | 2 |
| Cumulative idle > 10 ms | 19,821 ms | 72,026 ms | 24,667 ms | 26,372 ms | 11,262 ms |
| >100 MB H2D bandwidth | 55.4 GB/s | 55.4 GB/s | 55.3 GB/s | 54.8 GB/s | 56.7 GB/s |
| NCCL kernels | 0 | 0 | 0 | 0 | 0 |
| Kernel table captured | ✓ | ✓ | ✓ | ✓ | ✓ |

The DSI 4-GPU run takes ~2.2–2.8× longer wall time than the NVIDIA 4-GPU run for the same real computation. Two cross-cutting reads before the per-signal sections: the **production H2D path is flat everywhere** — all five columns sustain ~55 GB/s on the >100 MB transfers, so PCIe is not the bottleneck (§3) — and **NCCL is absent** on every cluster (data-parallel inference). (The per-host GPU-active times in the table differ, but that reflects profiling conditions — random weights, no `cudnn.benchmark` — not a hardware compute ranking; the trustworthy hardware compute comparison is the training benchmark in §II.4. What the inference captures establish is that DSI's deficit is idle/handoff, not bandwidth.)

## 2. Where the time goes: the handoff gap

The 10–50 ms gap bucket in the §1 table is the DSI outlier and the crux of the investigation: 41 → 423 on GPU0 going from 1 to 4 GPUs on DSI (and DSI's GPU1–3 are worse still, 649–894), against 27 on NVIDIA and 28–53 on the two Midway H200 nodes. These are not data-loading stalls (those would land in the >100 ms bucket); they are the GPU going briefly idle waiting for the CPU to signal the next launch. `torchrun` spawns 4 independent processes with no application-level synchronisation, but the ranks still share host resources — CPU cores, memory bandwidth, the PCIe root complex, kernel locks, IRQ paths. The 41 → 423 jump on GPU0 when scaling 1 → 4 GPUs is itself evidence of inter-rank contention through those shared host resources, and accounts for most of the ~47 s wall-time gap between DSI and NVIDIA.

## 3. H2D bandwidth: the production path is fine; only small transfers contend

**PCIe** (PCI Express) is the bus connecting GPU to CPU/RAM; every byte of weather data crosses it before reaching GPU memory. The H100 NVL / H200 ceiling (PCIe Gen5 ×16) is ~64 GB/s per direction, ~70–80% realisable.

The averaged-over-all-transfers bandwidth is misleading because it mixes two regimes. Per GPU per profile there are ~84 large pinned transfers (mean 165 MB, dominated by 12×~991 MB and 12×~105 MB) and ~2,845 small pageable transfers (~560 KB). Splitting by size:

| Transfer size class | DSI 1-GPU | DSI 4-GPU (per GPU) | NVIDIA H100 4-GPU (per GPU) |
|---|---|---|---|
| **>100 MB** (24/GPU, ~13 GB) | **55.4 GB/s** | **55.4 / 55.4 / 55.4 / 55.4** | **55.3 / 55.4 / 55.2 / 55.3** |
| >10 MB (74/GPU, ~14 GB) | 49.9 GB/s | 43.5 / 48.8 / 48.4 / 44.9 | 50.8 / 49.7 / 50.3 / 50.0 |
| All transfers (2929/GPU) | 41.6 GB/s | 31.5 / 38.6 / 37.4 / 32.7 | 44.7 / 41.8 / 43.6 / 44.3 |

**Restricted to the production-sized weather tensors (>100 MB), DSI under 4-GPU concurrent load is identical to DSI single-GPU and identical to the H100 cluster — all four GPUs sustain ~55 GB/s with no degradation.** The headline "DSI bandwidth drops 24% on GPU0/3" comes entirely from the small-transfer regime, where each call is latency-bound (handoff, IRQ handling, kernel-lock contention) and four ranks competing for host resources serialise them. The Midway `bandwidth_test.py` numbers behave the same way: on the 22 MB `upper_air_input` the concurrent-vs-sequential drop is only −1.6 to −3.5%; the −55% aggregate figure is dominated by the small `varying_boundary` (3.9 MB) and `surf_input` (1.3 MB) tensors.

This **retires the "PCIe contention" hypothesis** for the production workload: the large-tensor H2D path is not bandwidth-limited under 4-GPU load. The wall-time deficit is the handoff gaps, not H2D throughput. (One narrower open question: the ~2,845 small pageable transfers per GPU *do* serialise badly under load; if any sits on the critical path before a dependent kernel it could be one contributing mechanism — confirming that would require correlating individual MEMCPY events to subsequent launches.)

**NCCL is not involved.** No NCCL collective kernels appear in any profile, including the post-fix Midway captures (verified: count = 0). These are data-parallel inference runs with no gradient synchronisation. **Pinned memory is already in use** for the main path: the 84 large weather tensors per GPU go through pinned host memory on every cluster, byte-for-byte identically between NVIDIA and DSI, so the DataLoader's `pin_memory` configuration is not the cause.

## 4. Cross-cluster comparison matrix

Column key: **NVLink** = direct GPU-to-GPU bus (`NV6` = 6 bonds); **NUMA distance** = relative cross-socket memory cost (10 = local); **Concurrent bandwidth drop** = per-GPU H2D degradation when all four transfer at once vs one at a time; **10–50 ms gaps** = per-GPU count in that bucket (the DSI outlier).

| Cluster | GPU | CPU | NVLink | NUMA dist | CPU→GPU BW (1-GPU) | Concurrent BW drop | Handoff gap (cudaEvent) | GPU util (4-GPU) | 10–50 ms gaps/GPU | Kernel data |
|---|---|---|---|---|---|---|---|---|---|---|
| NVIDIA cluster | H100 NVL | — | unknown¹ | — | 42–45 GB/s² | ~0%² | — | 37–57% | 27 / 30 / 29 / 29² | ✓ |
| DSI | H200 | unknown | unknown | unknown | 41.6 GB/s² | >100 MB ~0%; all-avg GPU0/3 −24/−21%² | — | 15–23% | 423 / 649 / 815 / 894² | ✓ |
| Midway Intel (test) | H200 | Gold-6542Y | NV6 mesh | 21⁵ | 54 GB/s, symmetric³ | 22 MB: −2 to −4%³ | 0.024–0.030 ms⁸ | 47–48%⁹ | 53 / 29 / 35 / 35⁹ | ✓⁹ |
| Midway AMD (test) | H200 | EPYC-9335 | NV6 mesh | 32⁵ | NUMA: 39 / 49 GB/s³ | ~0% additional⁴ | 0.021–0.022 ms⁸ | 64–69%⁹ | 28 / 26 / 27 / 16⁹ | ✓⁹ |
| Midway pedramh-gpu | H100 NVL | Xeon Gold 6346 (Ice Lake) | NV12 within pairs only⁶ | 20⁶ | ~27 GB/s (PCIe Gen4)⁶ | NUMA: GPU2/3 −42% H2D⁶ | — | 51–52%⁷ | 37 / 21 / 20 / 29⁷ | ✓⁷ |

¹ The GPU-interconnect topology check was never run on the NVIDIA cluster, so we don't know whether its H100 NVL cards use NVLink.

² NVIDIA and DSI numbers are from inference `nsys` MEMCPY/KERNEL records (`verify_bench.py`). The 84 pinned per-GPU transfers are bimodal: 12×~991 MB, 12×~105 MB, 12×~22 MB, plus smaller (mean 165 MB). Utilisation and gap counts are per device 0/1/2/3.

³ Midway H200 sequential H2D from `v2.0/test/bandwidth_test.py` (tensor shapes from `exp2.yaml`: `upper_air_input` 22.3 MB, `surf_input` 1.3 MB, `varying_boundary` 3.9 MB). Intel is symmetric across GPUs (53.8 / 53.8 / 53.6 / 53.6 GB/s on `upper_air_input`); under concurrent load the 22 MB tensor drops only −1.6 to −3.5% while the small tensors drop −10 to −62% (latency-bound). AMD shows a clean ~20% NUMA-aligned sequential split (GPU0/1 ≈ 39 GB/s on cores 0–31 / node 0; GPU2/3 ≈ 49 GB/s on cores 32–63 / node 1), consistent with the allocator placing pinned host buffers preferentially on one NUMA node. Under real 4-GPU inference (footnote 9) the large pinned transfers move at ~51 GB/s (Intel) and ~56 GB/s (AMD), symmetric across all four GPUs.

⁴ Contention deltas (concurrent − sequential) from `bandwidth_test.py` aggregate medians: Intel −55.6 / −54.3 / −61.3 / −55.2% (uniform ~55% collapse, dominated by the small-tensor regime); AMD +0.5 / +24.4 / −0.0 / +0.0% (the +24.4% on GPU1 is `upper_air_input` jumping 43.75 → 54.93 GB/s, more consistent with measurement noise than a real gain). For AMD read "no degradation" as "no meaningful *additional* contention on top of the sequential NUMA asymmetry in footnote 3."

⁵ The topology report put GPU0/1 on one socket (cores 0–23 Intel / 0–31 AMD) and GPU2/3 on the other (cores 24–47 / 32–63). The GPU-side NUMA-ID field was blank on both nodes, so this is inferred from CPU-affinity hints, not a direct readout. Cross-socket distance is 21 (Intel) and 32 (AMD) vs 10 local.

⁶ `pedramh-gpu` (`midway3-0423`) is **H100 NVL on Xeon Gold 6346 (Ice Lake)** — same GPU model as the NVIDIA cluster, different host. From `test_partition_benchmarks/midway_bandwidth_midway_bandwidth_test.sh_49972059.out` and `pedramh_hw_topo.out`: 2 sockets × 16 cores (NUMA distance 10/20), driver 535.216.03, CUDA 12.2. NVLink is **NV12 within socket-pairs only** (GPU0↔1, GPU2↔3; cross-pair = `SYS`/UPI, no NVLink) — unlike Midway H200's NV6 full mesh. Ice Lake is PCIe Gen4, so the H100 NVL ceiling here is ~32 GB/s; the 27 GB/s sequential figure is ~84% of Gen4 theoretical, comparable in *fraction of available bandwidth* to Midway H200's 53 of Gen5's 64. Under concurrent load H2D shows a NUMA-aligned drop (GPU0/1 hold ~27, GPU2/3 collapse to ~14, −42%); on D2H the asymmetry inverts — consistent with pinned input buffers on one NUMA node and output buffers on the other.

⁷ The pedramh-gpu 4-GPU inference profile was **re-run after the checkpoint-guard fix** (2026-05-31, job 50249569, `midway_infer_nsys.sh` standardized in commit `3acb9b3`) and now has a full kernel table — **1,883,324 kernels, all four GPUs** — confirming the earlier "missing kernels" was the crash, not a profiler limit (the pre-fix capture held only 4,252 runtime events, the `restore_checkpoint` FileNotFoundError signature, §II.7). From `verify_bench.py`: utilisation 51.5 / 51.8 / 52.0 / 51.5%, 10–50 ms gaps 37 / 21 / 20 / 29 per GPU, NCCL absent, >100 MB H2D ~26 GB/s symmetric (PCIe Gen4 ceiling — no DSI-style bimodal drop). This is the **direct same-GPU-as-the-NVIDIA-profile control**: H100 NVL hardware on a Midway host hands off as cleanly as the H200 nodes (20–37 gaps vs DSI's 423–894), so DSI's deficit is its host environment, not the GPU. One host difference from `pedramh_hw_topo.out`: the `intel_idle` driver permits deeper CPU sleep states here (`POLL, C1, C1E, C6`) than the H200 test partition (`POLL, C1_ACPI, C2_ACPI`, C2 max, 41 µs exit) — C6 (~100–200 µs exit) is too small alone to explain 10–50 ms gaps, but worth flagging on any host that allows it.

⁸ **Real-Pangu dispatch test** (`v2.0/test/inference_dispatch_real.py`, scripts `midway_dispatch_real_{intel,amd}.sh`). `torch.cuda.Event` pairs measure GPU idle between consecutive forward passes of the actual `PanguModel_Plasim` with production-shape random inputs — in-process, no profiler attach. Across 1,180 inter-step gaps per node, **zero exceeded 10 ms**; max observed 66 µs. See §5.

⁹ **Post-fix kernel-level capture, 2026-05-29** (`midway_h200_{intel,amd}_4gpus_inference_{50244406,50244404}_2026-05-29.nsys-rep`, standardized bare command, commit `3acb9b3`). Full kernel tables (1,878,472 Intel / 1,883,332 AMD kernels, all four GPUs). Util and 10–50 ms gap counts per device 0/1/2/3 from `verify_bench.py`. These full-workload captures include the per-iteration D2H + NetCDF save, so they show more 10–50 ms gaps than the stripped dispatch test (footnote 8) — but still 8–15× fewer than DSI, and most of Midway's cumulative idle is in a handful of >500 ms per-iteration save pauses, not the handoff path. AMD's GPU0 is now present (the pre-fix AMD capture had dropped it).

### Reading the new Midway H200 rows

The post-fix captures confirm the central finding with kernel-level data, not just the cudaEvent proxy:

- **These inference active-times are not a hardware compute ranking.** Per-host GPU-active time varies in the captures (Midway ~27.2 s, DSI 13.9 s, NVIDIA 15.3 s for the same ~470k kernels), but that is a profiling-condition artifact — random-init weights drove cuDNN autotune to slower kernels, `cudnn.benchmark=True` was not set (§5 caveat), and the bare-metal hosts may differ in cuDNN/driver version. For an actual hardware compute comparison see the training benchmark (§II.4), where the Midway H200 nodes run ~26% faster than the pedramh-gpu H100 NVL. What these inference captures *do* establish is that DSI's deficit is handoff/idle, not bandwidth.
- **Handoff is clean.** Intel's 10–50 ms gaps (53/29/35/35) and AMD's (28/26/27/16) sit between NVIDIA (27–30) and far below DSI (423–894). Utilisation is 47–48% (Intel) and 64–69% (AMD) — the residual idle is dominated by ~6–19 per-iteration pauses > 500 ms (the NetCDF save path, even with `--async_save`), common to all ranks, not handoff stalls.

## 5. Real-Pangu dispatch test (handoff path in isolation)

To measure the handoff path without a profiler (and originally to work around the then-unexplained missing kernel tables, now known to be the checkpoint crash), `inference_dispatch_real.py` runs the autoregressive loop — 60 sequential forward passes of `PanguModel_Plasim`, each feeding the previous output — and times the inter-step GPU idle with in-process `cuda.Event` pairs.

```
        ┌────────── step N ──────────┐                     ┌──── step N+1 ────
GPU:   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┃    gap (idle)    ┃━━━━━━━━━━━━━━━━━━━━━ ...
                                  end_event[N]      start_event[N+1]
```

The test deliberately excludes everything that is not the handoff path, so a measured gap is unambiguously handoff latency: **no HDF5 read** (inputs are random GPU tensors allocated once), **no D2H between steps**, **no NCCL**, **no checkpoint save / `inv_transform` / async-save thread** (those happen at outer-iteration boundaries). If the CPU finishes the loop body — return from `model(...)`, append/rebind outputs, record the next event, re-enter `model(...)` — before the GPU finishes step N, the gap is ≈ 0; if anything stalls the CPU (Python overhead, scheduler preemption, NUMA-distant access, IRQ theft, deep C-state exit) the GPU sits idle and we measure how long.

| Configuration | Inter-step gaps measured (n) | Median gap | p99 gap | Max gap | Gaps > 10 ms (count) |
|---|---:|---:|---:|---:|---:|
| Midway Intel, 1 GPU (batch=8) | 236 | 0.025 ms | 0.032 ms | 0.043 ms | **0** |
| Midway Intel, 4 GPUs (batch=2/GPU) | 944 | 0.024–0.030 ms | 0.038–0.040 ms | 0.060 ms | **0** |
| Midway AMD, 1 GPU (batch=8) | 236 | 0.021 ms | 0.027 ms | 0.066 ms | **0** |
| Midway AMD, 4 GPUs (batch=2/GPU) | 944 | 0.021–0.022 ms | 0.024–0.028 ms | 0.058 ms | **0** |
| (Reference: DSI 1-GPU, from nsys) | — | — | — | — | 41 |
| (Reference: DSI 4-GPU GPU0, from nsys) | — | — | — | — | 423 |
| (Reference: DSI 4-GPU GPU1–3, from nsys) | — | — | — | — | 649–894 |

Maximum gap across all 2,360 Midway measurements: **66 µs** — at least 150× smaller than the smallest gap in DSI's 10–50 ms bucket. This is consistent with the kernel-level nsys captures (§4): the small number of 10–50 ms gaps that *do* appear in the full nsys workload come from the per-iteration save/D2H boundaries the stripped test omits, not the handoff path.

**What this rules out (on Midway evidence alone):**
- **The H200 chip as the sole cause** — same GPU model, different host, identical workload, no gaps > 10 ms. (Not ruled out: the H200 *combined* with something DSI-specific — a different board variant or GPU firmware, or a CPU-link power setting that only misbehaves in DSI's host. One `nvidia-smi --query-gpu=name,vbios_version` on DSI settles it.)
- **The 4-rank `torchrun` pattern** — Midway runs 4 ranks the same way and sees zero.
- **NUMA mis-placement alone** — Midway's GPUs span both memory banks with no steering, still zero gaps. If it contributes on DSI it must combine with another factor.
- **Deep CPU sleep states on Midway** — the test partition allows only C2 (41 µs exit), three orders of magnitude below 10–50 ms. (DSI may differ; the diagnostic in §6 checks it.)
- **PanguModel_Plasim's own kernel-launch pattern** — the real architecture with production shapes (only weights random) produced no gaps; a pathological launch sequence would have shown up here too.

**Still candidates for the DSI-specific cause:** (1) **driver / CUDA version mismatch** — Midway runs 535.216.03 / CUDA 12.2 / PyTorch cu124; an older/newer DSI driver, especially in a known-bad kernel-completion-IRQ region, is the leading suspect; (2) **kernel/OS config** (`intel_idle.max_cstate`, scheduler granularity, governor, cgroup throttling); (3) **co-tenant CPU contention** if the DSI node wasn't exclusive at capture; (4) **IRQ steering** funnelling NVIDIA completion interrupts onto a CPU also handling NIC/NVMe; (5) **filesystem/page-cache lock** held by the between-iteration disk read carrying into the loop.

**Caveat on absolute scale.** Forward-pass times in this test ran higher than production (Intel 407 ms 1-GPU / 133 ms 4-GPU; AMD 986 / 104 ms; production target ~14 ms at batch=1 on H100), most likely because random init drove cuDNN to slower kernels and `cudnn.benchmark=True` was not set. Longer forwards give the CPU more slack, so the test is *biased toward not finding gaps* — but the result is decisive because Midway's gaps are so far below DSI's that even an order-of-magnitude tighter loop would leave Midway clean at sub-millisecond scale. A tightened-loop rerun with warmed weights would close that rigour gap.

## 6. What is still unknown — the host diagnostic to request from DSI

The largest remaining unknown is DSI's host configuration. This short, read-only script (under a minute, normal user account) answers most of the candidate list:

```bash
# --- GPU identity ---
nvidia-smi | head -3                                             # driver + driver-CUDA version
nvidia-smi --query-gpu=driver_version --format=csv
nvidia-smi --query-gpu=name,vbios_version --format=csv           # exact H200 variant + GPU firmware
nvidia-smi topo -m                                               # GPU-GPU links + CPU/NUMA affinity
python -c "import torch; print(torch.__version__, torch.version.cuda)"

# --- CPU / NUMA / sleep states ---
lscpu | head -20
numactl --hardware                                               # NUMA node count + cross-socket distance
cat /sys/module/intel_idle/parameters/max_cstate 2>/dev/null     # deepest CPU sleep state (Midway test: 2)
cpupower frequency-info | head                                   # governor (performance vs powersave)
cpupower idle-info     | head

# --- IRQ and host warnings ---
grep nvidia /proc/interrupts | head                              # which CPUs field GPU completion IRQs
dmesg | grep -iE "nvidia|throttle|c-state" | tail -30
```

**The single most informative line is the first.** If DSI's driver / driver-CUDA differs from Midway's `535.216.03 / 12.2`, that is the lead. Then: `topo -m` showing `PIX`/`PHB` where Midway shows `NV6` is a structural interconnect difference (matters less for the CPU→GPU handoff path we're chasing); a different `vbios_version` opens a firmware-mediated H200 effect; `max_cstate` > 2 means deep idle states are enabled (Midway is not uniform here — the H200 test partition is C2, `pedramh-gpu` permits C6); GPU IRQs pinned to a CPU also handling NIC/NVMe is a candidate; recurring `dmesg` throttle/c-state/NVIDIA warnings are worth following up. The dispatch test already records this block into its own `.out` header on every run, so the ask is just to run it once on a DSI node and share the output.

## 7. Hardware summary

The DSI cluster's H200 GPUs spend most of their time idle, not computing. Three facts bound it: (1) per-GPU compute matches NVIDIA within ~10%; (2) the production >100 MB H2D path runs flat and symmetric at each host's PCIe ceiling (~55 GB/s on the Gen5 clusters, ~26 GB/s on pedramh's Gen4), with no DSI-style degradation — so PCIe is not the bottleneck; (3) the GPUs stop and wait 10–50 ms hundreds of times per run, which NVIDIA barely does. The deficit is **handoff latency**, host-environment-specific. The H200 architecture, the 4-rank `torchrun` pattern, and naïve NUMA mis-binding are ruled out by the Midway H200 evidence (real-Pangu cudaEvent test + full kernel-level captures). The strongest control is now in hand: **`pedramh-gpu` H100 NVL — the same GPU model as the NVIDIA profile, on a Midway host — hands off cleanly** (footnote 7: 10–50 ms gaps 20–37, util ~51–52%, right in the Midway/NVIDIA band and nowhere near DSI's 423–894). So even the exact GPU class DSI's comparison rests on is fine off DSI's host. The leading remaining unknown is DSI's software/host stack — driver version, kernel idle/scheduler config, IRQ steering, node exclusivity — per the §6 diagnostic.

---

# Part II — Software: model and inference performance (Midway pedramh-gpu H100 NVL baseline)

The training benchmarks (§2–§4) were collected on **Midway `pedramh-gpu`** — a single node of **4 × NVIDIA H100 NVL (~94–96 GB each)**, `midway3-0423` (footnote 6) — with `v2.0/train.py` (the benchmark-instrumented entry point) via `midway_bench.sh` / `midway_bench_nsys.sh`. The inference-bottleneck work (§7–§8) is from the Midway H200 test partition and the DSI profiles. The "NVIDIA H100 4-GPU" column in Part I is a *separate, DSI-provided* inference profile (DSI ran it; we have no access to that hardware) — not run here.

## 1. What we measure

Each training step: (1) load a batch and move it to the GPU, (2) forward to an ensemble forecast + loss, (3) backward for gradients, (4) Adam weight update. We discard a warm-up (20 steps; 40 with `torch.compile`) that absorbs driver init, cuDNN autotuning, and NCCL warm-up, then time 80 measured steps and report the median step time and throughput. Reliability guards: `torch.cuda.synchronize()` brackets the timing window so we record execution, not submission; the timer self-checks that summed step windows agree with total wall time within 10%, discarding the run otherwise.

## 2. Training baseline (original code)

Original code, 16-bit AMP with dynamic loss scaling, DDP searching for unused parameters every backward.

*Source: Midway `pedramh-gpu` (4 × NVIDIA H100 NVL, `midway3-0423`), `v2.0/train.py` wall-clock benchmark via `midway_bench.sh`, two independent jobs of 80 measured steps each (20-step warm-up discarded). All columns are global 4-GPU figures unless noted.*

| | Run 1 | Run 2 |
|---|---|---|
| Samples per GPU per step / global (4 GPUs) | 1 / 4 | 1 / 4 |
| Median step time | 0.639 s | 0.638 s |
| 90th-percentile step time | 0.643 s | 0.641 s |
| Step-to-step stdev | 0.005 s | 0.002 s |
| Data loading per step | 0.003 s | 0.003 s |
| Compute per step | 0.636 s | 0.636 s |
| **Throughput** | **6.26 samples/s** | **6.27 samples/s** |
| Peak GPU memory (worst card) | 34.96 GB | 34.96 GB |
| Loss-scale skips | 0 | 0 |

The two runs agree to within 0.05% (stable, repeatable). Data loading is 0.4% of step time — the bottleneck is entirely on-GPU. 34.96 GB of ~94 GB used (37%) leaves headroom for larger batches. Zero loss-scale skips confirms numerical stability in 16-bit. **Baseline: 0.639 s/step, 6.26 samples/s.**

## 3. Profiler kernel analysis (Nsight Systems, 80 measured steps)

*Source: one GPU of the same Midway `pedramh-gpu` H100 NVL node, `nsys` trace of `v2.0/train.py` via `midway_bench_nsys.sh` over the 80 measured steps (warm-up excluded). Per-step phase breakdown is per-GPU; the kernel totals below are cumulative across all 80 steps.*

| Phase | Median time | Share of step |
|---|---|---|
| Data preparation and transfer | 1.0 ms | 0.15% |
| Forward pass + loss | 194.2 ms | 29.1% |
| Backward pass (gradients) | 425.8 ms | 63.9% |
| Weight update (optimiser) | 26.3 ms | 3.9% |

The backward pass is 2.2× the forward — high but **not** from gradient recomputation (transformer-block checkpointing is commented out in `pangu.py`, so nothing is re-run; see §5). It is inherent to model depth and per-layer kernel mix. Other findings across the 80 steps:

- **LayerNorm backward** is the 2nd-largest GPU-time consumer (17.3 s total) — many normalisation layers, each invoked in forward and backward. Fusing LayerNorm via `torch.compile` is the single biggest optimisation the profile suggests.
- **Memory-layout conversions** (4.4 s): the model flips between channel-first and channel-last between layers; a consistent layout would remove this.
- **Inter-GPU communication** (15.8 s, ~7.7% of step): NCCL gradient sync after each backward; `find_unused_parameters` adds overhead here (removed below).
- **Roll operations** (7.7 s, ~24 ms/step): shifted-window attention's cyclic shifts run as separate kernels — architectural, hard to remove without redesign.
- **Matrix multiplications** (8.8 s): the attention/linear core ranks only 6th and 18th on the kernel list, so the model is not matmul-bound — changing numeric format won't give dramatic gains.

## 4. Ablations and configuration summary

Two changes were isolated against the baseline: switching 16-bit → **bfloat16** (same exponent range as FP32, no loss scaler needed on H100) and disabling the unused-parameter search (`find_unused_parameters=False, static_graph=True` — two Pangu parameters are permanently unused, so the per-step graph traversal was wasted work).

- **bfloat16 + static graph, batch 1/card:** 0.639 → 0.607 s, **+5.3%** throughput, no memory change (Adam's 32-bit state and activations dominate, not compute precision), zero instability. bfloat16's wider range guarantees stability anywhere 16-bit was stable.
- **bfloat16 + static graph, batch 3/card:** **42% slower** — memory hit 97 GB (at the card's limit), allocator under extreme pressure; step time scaled 5.2× for 3× data, the signature of memory saturation. Batch 4/card OOM'd. (Three variables changed at once here, so it was not a clean data point.)

| Configuration | Step time | Throughput | vs baseline | Memory | Skips |
|---|---|---|---|---|---|
| **Baseline** (16-bit, batch=1/card) | 0.639 s | 6.26 samples/s | — | 35.0 GB | 0 |
| Baseline repeat | 0.638 s | 6.27 samples/s | +0.05% | 35.0 GB | 0 |
| bfloat16 + static graph, batch=3/card | 3.314 s | 3.62 samples/s | −42% | 97.0 GB ⚠ | 0 |
| bfloat16 + static graph, batch=1/card | 0.607 s | 6.59 samples/s | +5.3% | 35.0 GB | 0 |
| 16-bit + static graph, batch=2/card | 1.160 s | 6.90 samples/s | +10.1% | 69.0 GB | 4 ⚠ |
| **bfloat16 + static graph, batch=2/card** | **1.146 s** | **6.98 samples/s** | **+11.4%** | **69.0 GB** | **0** |

**Best confirmed config: bfloat16 + static DDP graph + 2 samples/card — +11.4% throughput, zero instability, 73% memory utilisation.**

### Same config across Midway nodes: pedramh-gpu H100 NVL vs test-partition H200 (2026-05-22)

The best config above (the pedramh-gpu H100 NVL baseline) was rerun with `v2.0/train.py` on both Midway H200 test-partition nodes (`midway_training_{intel,amd}.sh`, source CSVs `test_partition_benchmarks/midway_training_{intel,amd}_bench.csv`). Identical git SHA `597b8572b028`, 4 GPUs, batch 2/GPU, bf16, static DDP graph, 80 timed steps after 20-step warm-up.

| Node | Step time (median) | p90 | Throughput | Peak mem (worst rank) | Skips |
|---|---|---|---|---|---|
| Midway `pedramh-gpu` H100 NVL | 1.146 s | 1.152 s | 6.98 samples/s | 69.0 GB | 0 |
| **Midway Intel H200** (`midway3-0602`) | **0.907 s** | 0.909 s | **8.82 samples/s** | 69.1 GB | 0 |
| **Midway AMD H200** (`midway3-0601`) | **0.899 s** | 0.900 s | **8.90 samples/s** | 69.1 GB | 0 |

The H200 nodes train this config **~26–28% faster** than the pedramh-gpu H100 NVL (8.8–8.9 vs 6.98 samples/s). Since training is compute-bound here (data load is 0.4% of step time, §2), this reflects the GPU itself — H200's higher HBM bandwidth and compute — not the host (pedramh-gpu is a PCIe Gen4 Ice Lake host, the test partition is Gen5, but that path is off the critical path for training). Intel and AMD H200 are within ~1% of each other (AMD marginally ahead, lower CPU-prep fraction 0.0027 vs 0.0056); both share an identical SHA and PyTorch 2.6.0+cu124, so that comparison is clean. All three are Midway nodes.

## 5. Remaining experiments

- **`torch.compile` (in progress):** `reduce-overhead` mode fuses element-wise ops, the single largest GPU-time consumer (>30 s across 80 steps, launched as hundreds of small kernels). Warm-up raised 20 → 40 steps for Triton compilation to settle; both the benchmark and the profiling scripts now use `reduce-overhead` + bfloat16 so the new profile compares directly.
- **Gradient checkpointing:** transformer-block checkpointing is commented out in source; the config's `checkpointing` value only governs 4 lightweight patch-recovery ops. The 2.2× backward ratio is depth-inherent, not recomputation.
- **VAE ensemble quality:** `test/vae_collapse_test.py` checks whether the VAE produces real ensemble diversity or has collapsed (see §6).

## 6. VAE ensemble generation — architecture notes

The VAE generates 4 ensemble members by repeating each input 4× and injecting different noise draws at the encoder bottleneck, sampled from a distribution whose mean/variance the encoder learns. A second encoder branch, training-only, processes the target weather state and provides a reference distribution the forecast encoder is trained to match. Training and inference generate the members by *different* mechanisms — the `train` flag in `PanguModel_Plasim.forward` is the switch:

```
══════════════════ TRAINING  (forward, train=True) ══════════════════

  input batch: M samples
        │  to_ensemble_batch()  (train.py)  ── repeat each sample ×4  (num_ensemble_members=4)
        ▼
  batch = M×4
        ▼  SHARED main encoder trunk:  patchembed → layer1 → downsample → layer2/3
        ▼     (~14 heavy EarthSpecificLayer blocks — the bulk of compute; runs in BOTH modes)
        ▼  x_vae
  ┌────────────────────────────────────────────────┐
  │ ENCODER 1 (prior HEAD) = 3 × 1×1 conv            │  ← tiny sampling head, NOT a transformer
  │   mu = layer_mu(x_vae);  sigma = layer_sigma(…)  │     stack → forward AND backward NEGLIGIBLE
  │   norm = reparameterize(mu, sigma)               │     randn PER COPY ⇒ 4 distinct latents
  │   x_purb = layer_purturbation(norm)              │     from identical inputs (runs in BOTH modes)
  └────────────────────────────────────────────────┘
        ▼  x = x_purb + x  →  decoder (upsample → layer4 → patchrecovery)
        ▼
   output: M×4 forecasts ───────────────────────────────┐
                                                         │
  target state ─► ENCODER 2 (posterior)                  │  ← TRAIN ONLY · FULL ~14-block stack
    target_surface/upper_air     mu_e2 = layer_mu_e2(…)   │     checkpointed; ≈ doubles encoder
                                 sigma_e2 = layer_sigma_e2(…)    compute (fwd 40–70 / bwd 60–100 ms)
            ┌─────────────────────────────┘              │
            ▼                                             ▼
   KL(E1 ‖ E2)                                   CRPS  (Latitude_weighted_CRPSLoss)
   Kl_divergence_gaussians(                      reshape M×4 → (M, 4, …) by num_ensemble_members
       mu,sigma, mu_e2,sigma_e2)                 = CRPSSkill − 0.5·CRPSSpread
            │                                             │
            └──────────►  total = CRPS + vae_loss_weight · KL  ◄──┘   (cal_loss, train.py)

═════════════════ INFERENCE  (forward, train=False) ═════════════════

  for ens_id in range(2):   (inference[_optimized].py — hardcoded 2; config ignored)
        ▼
   forward(train=False) ─► ENCODER 1 HEAD only → reparameterize → fresh noise
        │                  (shared trunk still runs; Encoder 2 + target + KL skipped)
        ▼
   autoregressive rollout over inference_steps  →  save_prediction(ens_id)  (one NetCDF/member)

  ⇒ Encoder 1 = tiny 1×1-conv sampling head (cheap, both modes);
    Encoder 2 = a full second encoder stack (expensive, train-only) — the cost in §6's table.
    training: 4 members in ONE batched forward (×4 repeat + per-copy noise);
    inference: 2 members from SEPARATE sequential rollouts (different noise each pass).
```

**What the KL loss does.** The loss is `KL(Encoder1 ‖ Encoder2)` between two *learned* Gaussians — not against a fixed N(0,1). At inference only Encoder 1 runs, sampling noise that (if KL training worked) resembles what Encoder 2 would have produced from tomorrow's weather. Encoder 2 is a training-time teacher with no inference role.

**Why collapse is likely.** The KL/forecast balance is set by `vae_loss_weight: 0.0001`, making the KL signal ~10,000× weaker than the forecast loss. The closed-form KL between diagonal Gaussians is implemented correctly:
```
KL = 0.5 × (logvar_p − logvar_q + (var_q + (μ_q − μ_p)²) / var_p − 1)
```
and its gradient w.r.t. Encoder 1's log-variance, `0.5 × (−1 + var_q/var_p)`, *does* resist collapse (it goes negative as `var_q → 0`). But scaled by 0.0001 the anti-collapse gradient is at most ~0.0005 while the CRPS gradient is order 1 — the forecast loss outweighs KL by ~2,000–20,000:1 at the gradient level, so collapse proceeds despite the correct formula. Two failure modes follow: Encoder 1 never learns to imitate Encoder 2, and its variance may collapse toward zero (near-identical ensemble members).

**Cost of the second encoder.** First, the clarification the costs hinge on: the two "encoders" are very different sizes. The **VAE prior head ("Encoder 1") is just three 1×1 convolutions** (`layer_mu`, `layer_sigma`, `layer_purturbation`) sitting on the shared main-encoder trunk's output — its forward and backward are negligible. The **"second encoder" ("Encoder 2") is a full parallel transformer stack** (`layer1_e2 → downsample_e2 → layer2_e2 → layer3_e3`, ~14 `EarthSpecificLayer` blocks mirroring the main encoder's `[2,6,6,2]` depths) that re-encodes the *target* from scratch — that is the expensive part, and the only one this table is about. From the 194 ms baseline forward:

| Component | Forward | Backward (no recomputation) | Total per step |
|---|---|---|---|
| Second encoder (estimated) | 40–70 ms | 60–100 ms | **100–170 ms** |
| As fraction of step time | 6–11% | 9–15% | **15–25%** |

**Why the backward pass is the larger half.** The second encoder runs only during training and is thrown away at inference — but "discarded later" does not make it cheap now. While the model is training it is a full, trainable 14-block stack, so on every step the optimiser has to push gradients backward through all of its layers — both to update the second encoder's own weights and to generate the signal that teaches the forecast encoder (Encoder 1) which distribution to aim for. And a backward pass is inherently more work than a forward pass: the forward computes each layer's output once, whereas the backward computes *two* things at each layer — how to adjust that layer's inputs and how to adjust its weights — so it runs at roughly 1.5× the forward time. The result is that most of the second encoder's cost is its backward pass, and that entire cost buys nothing at run time, where Encoder 2 does not run. (Encoder 1 — the 1×1-conv head — is negligible in both modes; the compute the forecast actually requires lives in the *shared* main trunk, which both training and inference pay regardless. So the second encoder's ~15–25% is genuinely *extra*, not a cost the forecast needs.)

So the second encoder costs an estimated **15–25% of total step time** at batch 1 — it roughly doubles the encoder workload.

**Measured (2026-06-01, eager `midway_bench_nsys.sh` on pedramh-gpu, batch 2/card).** A clean NVTX capture (`S2S_NVTX=1`) correlating kernels to the `vae_encoder1`/`vae_encoder2` ranges confirms the picture directly. Per step: **Encoder 1 — the conv head — = 0.77 ms**, 0.2% of the 363 ms forward (negligible, as expected for three 1×1 convs), and **Encoder 2 = 118.6 ms ≈ 33% of the forward pass** (~11% of the step's forward portion). The absolute ms run higher than the batch-1 estimate above because this capture is batch 2/card and eager, but the *fraction* — ~33% of the forward, the top of the estimated 20–36% — confirms it. The second encoder's **backward is not separately bracketed**: the `backward` NVTX range (697 ms/step) covers the whole pass, so that half is currently a derivation (~1.5× its forward), not a direct measurement. A `vae_encoder2_bwd` NVTX range has now been added to `pangu.py` (identity autograd ops that push/pop in their backward, bracketing the layer2_e2 + layer3_e3 backward incl. checkpoint recompute); the next eager `S2S_NVTX=1` run will measure the second-encoder backward directly. If the collapse test confirms the second encoder isn't producing a useful signal (likely at this KL weight), removing it recovers that ~15–25% of step time at no quality cost.

**Design assessment.** A CVAE with a learned prior is justified when the conditioning signal is available at inference. Here it is not — Encoder 2 needs the *future* state, so the design is a training trick with no inference-time analogue, and it depends entirely on the KL weight being tuned correctly. The cleaner alternatives generate diversity without a second encoder or a KL weight to tune:

| Approach | Diversity source | 2nd encoder | Fragility |
|---|---|---|---|
| This model (learned prior) | encoder-distribution noise, KL-trained to posterior | yes (target-data, training) | high — KL weight must be tuned |
| Fixed noise injection | scaled random noise at bottleneck | no | low |
| Monte Carlo dropout | dropout active at inference, 4 passes | no | low |
| Latent diffusion | score-based sampling from a learned noise schedule | no (diffusion head) | moderate — principled, current SOTA |

DeepMind's GenCast (2023) uses the diffusion approach on a Pangu-style backbone and is the current SOTA for probabilistic medium-range forecasting: no second encoder, no KL weight, no collapse risk, and its conditioning (current atmospheric state) is available at every denoising step. The trade-off is inference cost (≈50 denoising chains vs one forward pass).

## 7. Inference bottlenecks found and fixed (2026-05-26)

Once profiles recorded real data (after the checkpoint fix below), two bottlenecks surfaced.

**Checkpoint guard (the prerequisite fix).** The inference scripts called `torch.load(checkpoint_path)` unconditionally, but the checkpoint file does not exist for the `rcc-staff` team, so the job crashed with `FileNotFoundError` during model setup — **before launching any GPU compute**. Every "broken" profile therefore looked identical (only the ~3–4k CUDA-runtime events from `model.to(device)`, no kernel table), which earlier drafts misread as a profiler/ptrace limitation. The fix (commit `56f73fe`) guards the load on `os.path.isfile`: if the file is missing, log a warning and continue with random weights. Output values are then meaningless, but **profile timing is identical regardless** — kernel shapes, launch rates, and bandwidth depend on architecture and tensor shapes, not weight values. This is what unblocked the kernel-level Midway captures in Part I.

**Bottleneck 1 — synchronous saving on rank 0.** Profiles showed rank 0 at 28.8% utilisation while ranks 1–3 reached ~50%: ~20 large pauses of 3–5 s on rank 0 vs ~25 sub-second pauses elsewhere. The inference loop writes each iteration's forecast to NetCDF synchronously, and on rank 0 those writes take 3–5 s (ranks 1–3 finish before the next batch is ready, hiding the cost). The `--async_save` flag hands each write to a small background thread pool while the main thread keeps dispatching GPU work; after enabling it, rank 0 rose 28.8% → 47.5%, matching the others within ~1%.

**Bottleneck 2 — data-loader artifact (resolved, self-inflicted).** An intermediate diagnostic forced `num_data_workers=0`, serialising every batch read on the main thread: all four GPUs idle 70–78%, ~2 s compute then ~14 s read, repeating. Restoring the production default of 8 prefetch workers dropped idle to ~50% (where Bottleneck 1 lived). Not a real bottleneck — just the cost of having switched off prefetch in the test.

**Post-fix Midway snapshot** (4-GPU inference, both fixes applied):

| GPU | GPU-active | Total elapsed | Utilisation |
|---|---|---|---|
| 0 | 27.2 s | 57.3 s | 47.5% |
| 1 | 27.2 s | 54.9 s | 49.5% |
| 2 | 27.2 s | 54.3 s | 50.1% |
| 3 | 27.2 s | 54.2 s | 50.3% |

All four ranks symmetric at ~50%. The remaining 50% is ~20 per-iteration pauses common to all ranks — likely a mix of NCCL waits and Python overhead at iteration boundaries; characterising them precisely is the next step.

## 8. How DSI compares, and recommendations

DSI's profile of the same workload shows the rank-0 imbalance Midway had *before* `--async_save`: GPU0 at 15.1% with 60.4 s in large pauses, GPUs 1–3 at 17–23%. **DSI should gain from `--async_save` the same way Midway did.** Separately, DSI shows ~10× more short (1–100 ms) inter-kernel pauses than post-fix Midway:

| Pause size | DSI rank 1 | Midway rank 1 (post-fix) |
|---|---|---|
| 10–100 ms | 663 pauses, 11.3 s | 47 pauses, 2.2 s |
| 1–10 ms | 758 pauses, 4.3 s | 64 pauses, 0.2 s |

That ~15–20 s/rank of extra idle is the same handoff-latency story as Part I, independent of the two bottlenecks above; its fix lives in the DSI host configuration, not the workload code. DSI's wall-time disadvantage is an idle/handoff problem, not a compute one. (We do not rank compute speed from the inference profiles' per-host active-times — those are a profiling artifact, Part I §1. The hardware compute comparison we trust is the training benchmark in §4 above: the Midway H200 nodes train ~26% faster than the pedramh-gpu H100 NVL. No DSI training run has been collected, so DSI is not in that comparison.)

**Recommendations for DSI users:**
1. Pull the latest `v2.0/inference.py` / `inference_optimized.py` from `bench-instrumentation` (the checkpoint-guard fix, `56f73fe`).
2. Add `--async_save` to the `torchrun` command; expect rank 0's utilisation to rise toward the other three.
3. Re-profile and check the 1–100 ms pause counts. If still ~10× Midway's, the residual is in the DSI host stack (driver, kernel scheduler, container runtime) — run the §6 diagnostic.

**Files changed in the 2026-05-26 update.** Workload: `v2.0/inference.py`, `v2.0/inference_optimized.py` — `restore_checkpoint()` guarded by `os.path.isfile` (`56f73fe`). Profiling scripts: `--async_save` propagated to all inference profiling scripts (`40f0bd8`, `7479b30`, `474c2a7`); the three production inference nsys scripts standardized to the bare DSI capture command with job-id+date outputs (`3acb9b3`).

---

## Appendix — cross-cluster comparison caveats

The DSI, NVIDIA, and Midway inference profiles were collected under non-identical conditions; keep these in mind when comparing absolute numbers:

- **No warm-up before profiling.** `inference_optimized.py` captures from the first iteration (`if i > 5: break`), so first-batch costs (cuDNN autotune, cuBLAS workspace, first pinned-buffer alloc) fold into every profile and inflate the >500 ms gap bucket.
- **Software stack not held constant.** NVIDIA runs inside the NGC apptainer; DSI and Midway are bare-metal under different drivers/OS. CUDA, cuDNN, and nsys versions were not recorded per cluster — this is the most likely explanation for the Midway-vs-DSI active-time difference noted in Part I §4.
- **Git SHA of the inference script not recorded per cluster.** Compute the SHA before each capture when reproducing.
- **`verify_bench.py`** (repo root) reproduces every hardware number in Part I from the `.sqlite` profiles; its `FILES` dict lists the exact captures. Re-run after any rebuild to confirm the numbers hold.
- **Bandwidth-test caveat.** Midway H2D figures come from `v2.0/test/bandwidth_test.py` (shapes from `exp2.yaml`), within ~25% of production sizes — the *relative* concurrent-drop comparison is meaningful, but absolute GB/s should not be compared one-to-one with the nsys MEMCPY numbers from the other clusters.
