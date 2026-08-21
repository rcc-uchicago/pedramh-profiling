# Profiling tables — ACE2 & PanguWeather (2026-08-18/19)

Tables only. Detail: `ACE2_retrain/bench_midway_notes.md`,
`PanguWeather/v2.0/bench_midway_notes.md`, `polaris_bench_report.md`.

> ### ⚠ Read this before quoting any percentage in this file
> **Almost every table here is a table of shares, one capture per column** — and a share
> of a GPU-kernel total is **not a reproducible quantity**, because NCCL *wait* sits in the
> denominator. `polaris_bench_report.md` §4.4b measures one such share moving **4.77 points**
> between two runs of an **identical config** (42.2% → 37.4%) while its numerator moved
> **0.09%**. Prefer **ms/rank-step**, or **share-of-compute** (non-NCCL), which held to 0.34
> points across the same pair. Compute itself is reproducible to **0.27% across 8 devices on
> two different nodes**; anything containing NCCL is not.

---

**ACE2 — GPU kernel time by bucket, three hardware/batch configurations.**
Jobs 53479120 (A100), 53524918 (H100), 53483668 (H200).

| bucket | 4× A100-PCIe, batch 4 | 4× H100 NVL, batch 4 | 8× H200, batch 16 |
|---|---|---|---|
| NCCL (comm + wait) | 40.6% | 52.1% | 18.6% |
| elementwise / copy | 35.4% | 26.8% | 47.9% |
| GEMM | 11.3% | 7.6% | 11.9% |
| norm / cudnn | 4.7% | 3.1% | 5.3% |
| optimizer | 4.2% | 3.2% | 3.0% |
| FFT / SHT | 2.4% | 2.4% | 4.2% |
| reduction | 1.0% | 1.0% | 2.8% |

---

**ACE2 — composition of the elementwise/copy bucket, 8× H200.** Job 53483668.

| | share of bucket | share of all GPU time | launches |
|---|---|---|---|
| copies (`direct_copy`, `bfloat16_copy`) | 58.2% | 28% | 400,712 |
| add | 20.1% | 9.6% | 194,856 |
| other pointwise math | 10.0% | 4.8% | 214,376 |
| unary (scale/cast-like) | 7.7% | 3.7% | 89,528 |
| fill | 4.1% | 2.0% | 96,408 |

---

**ACE2 — GPU occupancy by phase, union of kernel intervals per device.** Job 53479120.

| phase | GPU occupancy |
|---|---|
| training (steady) | 91% |
| validation tail | 3.3% |

---

**ACE2 — copy-kernel GPU time by innermost NVTX range.** Job 53534648.

| range | share of copy time |
|---|---|
| `(outside)` — backward, autograd threads | 57.1% |
| `spectral_filter` | 35.6% |
| `sfno_block` | 3.1% |
| `sfno_mlp` | 3.0% |
| `stack` | 0.2% |

> ⚠ **SUPERSEDED — do not quote this table.** It was derived with (a) an unguarded
> `correlationId` join (**+30.8% phantom rows**; the `sfno_block` and `sfno_mlp` rows are
> cross-rank phantoms and collapse to 0.0% with the guard) **and** (b) a thread-scoped
> range lookup, which is the *entire* reason `(outside)` reads 57.1% — backward launches
> come from autograd worker threads while the ranges sit on the main thread. It is not an
> NVTX limitation. With attribution scoped to the **process**, `(outside)` → **0.0%**
> (measured on Pangu job 7255503). Re-derive with **either**
> `ACE2_retrain/nvtx_phase_attribution.py` **or** `ACE2_retrain/kernel_census.py` — the
> census was fixed 2026-08-20 (plan item 4) and now delegates to the same join, so the two
> agree row for row. **Nobody has run it on an ACE2 capture yet**, so this table is
> superseded but not yet replaced. See `polaris_bench_report.md` §4.3a.

---

**ACE2 — copy-kernel time by causing operation, autograd-annotated.** Job 53535415.
Timings void by design (`emit_nvtx`); shares only.

| cause | share of copy time |
|---|---|
| `aten::clone` | 45.3% |
| `aten::select_backward` | 6.5% |
| `AddBackward0` | 2.7% |
| `aten::bmm` / `fill_` / `nccl:all_reduce` / `convolution_backward` | ~1% each |

---

**ACE2 — `torch.compile` regional sweep, 4× H100, torch 2.7.1.** 64 steps, median of last 30.

| arm | step_med | vs control | loss drift | verdict |
|---|---|---|---|---|
| `none` (control) | 0.3425 s | — | — | baseline |
| `corrector` | 0.3350 s | −2.2% | 2.0e-5 | only winner |
| `all` (norm+corr) | 0.3345 s | −2.3% | 8.9e-3 | — |
| `normalizer` | 0.3420 s | −0.15% | 8.4e-3 | reject |
| `safe` (mlp+corr) | 0.3450 s | +0.7% | 1.2e-5 | reject |
| `mlp` | 0.3540 s | +3.4% | 3.6e-6 | reject |
| `network` (whole SFNO) | FAILED | — | — | `InductorError: KeyError: 'complex64'` |

---

**ACE2 — `torch.compile` on torch 2.8.0 vs 2.7.1.** Jobs 53531456/457/459.

| arm | torch 2.7.1 | torch 2.8.0 |
|---|---|---|
| control | 0.3425 s | 0.3510 s |
| `corrector` | 0.3350 s (−2.2%) | 0.3420 s (−2.6%) |
| `network` | hard fail (`complex64`) | runs, +3.4% slower (eager fallback) |

---

**ACE2 — validation probe, 64-sample window, epoch-2 (warm) comparison.**
Jobs 53524580/581/674/675/752.

| arm | change | warm validation | vs baseline |
|---|---|---|---|
| A | baseline (batch 4, 8 workers) | 34.20 s | — |
| B | batch 16 (4× fewer batches) | 33.86 s | −1% |
| C | 1 data worker | 43.88 s | +28% |
| D | 16 data workers | 34.43 s | +1% |
| E | `log_snapshots=false` | 16.45 s | −52% |

---

**ACE2 — hardware baseline, identical config and seed.** Jobs 53478978 (A100), 53524865 (H100).

| | 4× A100-PCIe | 4× H100 NVL |
|---|---|---|
| samples/s/rank | 1.84 | 2.97 |
| training, 16 steps | 48 s | 19.2 s |
| validation | ~183 s | 32.2 s |
| epoch | 231 s | 51.2 s |
| wall | 5:55 | 2:03 |

---

**ACE2 — 8-GPU H200 throughput vs 4-GPU A100.** Job 53483666.

| | 4× A100-PCIe (batch 4) | 8× H200, 2 nodes (batch 16) |
|---|---|---|
| samples/s/rank | 1.84 | 6.82 |
| aggregate samples/s | 7.4 | 54.2 |
| per-rank batch | 1 | 2 |

---

**ACE2 — reproducibility floors, `train_loss` relative difference.**

| comparison | relative |
|---|---|
| same GPU, same node (53483666 vs 53483667) | 2.5e-7 |
| A100 vs H100, same config and seed | 3.3e-6 (train), 1.1e-5 (valid) |

---

**ACE2 — same-pair vs cross-pair NVLink, 2 ranks, 1 sample/rank.**
Jobs 53538838/53538839, `midway3-0423`.

| arm | GPUs | link | step_med | vs NVLink |
|---|---|---|---|---|
| A | 0,1 | NVLink (NV12) | 0.2250 s | — |
| B | 0,2 | SYS (PCIe + NUMA) | 0.2900 s | +28.9% |

---

**`midway3-0423` — measured pairwise GPU copy bandwidth (GB/s), 256 MiB transfers.**
`gpu_topology_check.py`, job 53539369.

| | cuda:0 | cuda:1 | cuda:2 | cuda:3 |
|---|---|---|---|---|
| **cuda:0** | — | 261.2 | 18.3 | 18.3 |
| **cuda:1** | 261.8 | — | 18.3 | 18.3 |
| **cuda:2** | 18.4 | 18.4 | — | 262.3 |
| **cuda:3** | 18.3 | 18.4 | 261.6 | — |

---

**Delta (DeltaAI `ghx4`, gh121) — measured pairwise GPU copy bandwidth (GB/s), 256 MiB.**
4x GH200 120GB, `nvidia-smi topo -m` reports NV6 between all pairs.

| | cuda:0 | cuda:1 | cuda:2 | cuda:3 |
|---|---|---|---|---|
| **cuda:0** | — | 132.0 | 131.9 | 126.3 |
| **cuda:1** | 132.1 | — | 132.1 | 126.3 |
| **cuda:2** | 132.1 | 132.0 | — | 126.0 |
| **cuda:3** | 132.1 | 132.1 | 126.4 | — |

---

**GPU interconnect by node, slowest hop in a 4-GPU ring.**

| node | GPUs | topology | fastest link | slowest hop |
|---|---|---|---|---|
| `midway3-0423` | 4x H100 NVL | NV12 pairs only | 261 GB/s | **18 GB/s** |
| Delta `gh121` | 4x GH200 120GB | NV6 full mesh | 132 GB/s | **126 GB/s** |

---

**PanguWeather — Midway vs Polaris, same model and shape.**
Jobs 53539872 (Midway), 7255410 (Polaris).

| | 4× H100 NVL (Midway) | 4× A100 (Polaris) |
|---|---|---|
| step_med | 1.100 s | 0.652 s |
| samples_per_s | 3.64 | 6.13 |
| peak mem | 26.97 GB | 26.98 GB |
| loader_wait_frac | 8.8% | 0.7% |

---

**PanguWeather — GPU kernel time by bucket, Midway vs Polaris.**

| bucket | 4× H100 NVL (Midway) | 4× A100 (Polaris) |
|---|---|---|
| NCCL (comm + wait) | 74.2% | 10.5% |
| elementwise / copy | 13.5% | 61.0% |
| GEMM | 5.7% | 15.1% |
| norm / cudnn | 2.9% | — |
| FFT / SHT | 1.3% | — |
| optimizer | 1.2% | — |
| reduction | 0.2% | — |

---

**PanguWeather — GPU kernel time by NVTX phase, 4× A100 (Polaris).** Job **7255503**,
launch-time attributed, pid-guarded + process-scoped, /160 rank-steps. `sum` counts a
kernel per stream; **`union` is wall-clock occupancy and is the one to divide into a step
time.** → `polaris_bench_report.md` §4.3b.

| phase | launches/step | sum ms/rs | union ms/rs | % of union | self-overlap |
|---|---|---|---|---|---|
| `backward` | 1568 | 466.95 | **408.63** | 69.9% | **12.5%** (NCCL on its own stream) |
| `forward_loss` | 590 | 150.32 | 150.32 | 25.7% | 0.0% |
| `optimizer` | 59 | 25.93 | 25.93 | 4.4% | 0.0% |
| `data_prep` | **0** | 0.00 | 0.00 | 0.0% | — |
| `(outside)` | **0** | 0.00 | 0.00 | 0.0% | — |

n=2 (job 7255557, identical config, different node): `launches/step` and the compute
`ms/rs` are **identical/stable**, but `% of union` and `self-overlap` move with rank
balance — `backward` reads **69.9% / 12.5%** (7255503) and **67.4% / 18.5%** (7255557).
Quote the ms, not those two columns.

---

**PanguWeather — the 42.2% copy time, split by phase.** Job **7255503**. Union-safe (all
90,240 kernels on one stream). → §4.3c.

| phase | ms/rank-step | % of copy time |
|---|---|---|
| `backward` | **197.58** | **72.9%** |
| `forward_loss` | 73.61 | 27.1% |
| total | 271.19 | 100% (= 42.2% of all GPU kernel time) |

`conj` (38.30 ms/rs, 14.1%) fires **only** in `backward` — the adjoint of the complex
einsum, warranted from source. Removable by lowering `checkpointing`: **only** the
recompute inside `backward`, ≈74.6 ms/rs (est.).

---

**Polaris node — measured pairwise GPU bandwidth.** Job **7533457** on `x3204c0s13b1n0`,
`gpu_topology_check.py`, 256 MiB per transfer, unidirectional `copy_`. → plan item 6.

| | GPU0 | GPU1 | GPU2 | GPU3 |
|---|---|---|---|---|
| **cuda:0** | — | 83.0 | 83.0 | 83.0 |
| **cuda:1** | 83.0 | — | 83.0 | 83.1 |
| **cuda:2** | 83.0 | 83.0 | — | 83.0 |
| **cuda:3** | 82.9 | 83.0 | 83.1 | — |

**Full NVLink mesh — `NV4` on every pair, 82.9–83.1 GB/s, spread 0.24%.** No 2×2 block
structure (that is the H100-NVL pair-bridge pattern) and no PCIe-class pair (~25 GB/s).
`nvidia-smi topo -m` agrees: `NV4` in all 12 cells. ⇒ this is **intra-node device-to-device**;
it is not the §4.3e figure (1279 GB/s, *intra-device* HBM) and says nothing about multi-node.

**GPU↔NUMA is REVERSED** — from `nvidia-smi`'s CPU-Affinity column, independently matching
sysfs on a different node (job 7531456): GPU0→NUMA **3** (cores 24-31,56-63), GPU1→**2**,
GPU2→**1**, GPU3→**0**. A naive `--cpu-bind depth -d 8` therefore puts local rank 0 on cores
0-7 = NUMA 0, whose GPU is GPU3 — **every rank maximally far from its own GPU** (plan item 6b).

---

**PanguWeather — achieved DRAM bandwidth, 4× A100 (Polaris).** Job **7255503**. Peak
(1555.2 GB/s) and L2 (40 MiB) read from `TARGET_INFO_GPU`. → §4.3e.

| path | ms/rank-step | achieved | % of peak |
|---|---|---|---|
| D2D memcpy, **above L2** (98.7% of D2D bytes) | 20.82 | **1279 GB/s** | **82%** |
| `direct_copy`/`conj` kernels | 271.19 | — | **17–27% (estimated)** |
| H2D (loader, 100% in `data_prep`) | 2.18 | 24 GB/s | host link, not HBM |

⚠ Sub-L2 transfers reach **124.7% of peak** under the `2 × bytes` rule, which is how you
know the rule does not apply to them — quote the above-L2 population only. This is
**intra-device HBM**, not the interconnect: it does not close the topology cell.
**n=2:** job 7255557 (different node) gives **1281 GB/s = 82%**, byte-identical volumes —
reproduces to **0.2%**. This is the most reproducible number in the file.

---

**PanguWeather — NVTX phase durations, 4× H100 NVL.** Job 53539872.
⚠ **CPU-side *launch* time, not GPU time.** On the A100 capture the CPU-side reading of
`backward` understates its GPU share by **~21 points** (46.5% of the step vs 67.7%), and
the sub-ranges do not sum to the step. → `polaris_bench_report.md` §4.1/§4.3.

| range | n | median | min | max |
|---|---|---|---|---|
| `data_prep` | 160 | 0.5 ms | 0.2 | 2.3 |
| `forward_loss` | 160 | 46.3 ms | 42.5 | 73.7 |
| `backward` | 160 | 539.8 ms | 112.2 | 993.1 |
| `optimizer` | 160 | 4.5 ms | 3.4 | 6.3 |
| step total | 156 | 995.5 ms (std 278.3) | — | — |

---

**NCCL share vs gradient volume, identical node (`midway3-0423`).**

> ⚠ **SUPERSEDED — do not quote. The "share ∝ gradient volume" thesis is refuted twice.**
> (1) `POLARIS_PROFILING_HANDOFF.md` §2 already records that on a like-for-like all-reduce
> basis Pangu is **41.1%** against ACE2's **51.4%** — the *opposite* ordering. That
> correction never reached this table. (2) The Pangu cell is a **straggler capture**
> (per-device broadcast 534/509/4/498 ms), so it is largely rank-imbalance wait, not
> transfer. §4.4c: the same collective reads 0.11% and 4.0% on two runs of one config.

| model | params | gradients (fp32) | NCCL share |
|---|---|---|---|
| ACE2 | 456 M | 1.82 GB | 52.1% |
| PanguWeather | 1.18 B | 4.73 GB | 74.2% |

---

**Three-model step times.** Different clusters, batch sizes and model sizes.

| | ACE2 | PanguWeather | ai-rossby |
|---|---|---|---|
| params | 455.8 M | 1,182.1 M | — |
| step_med | 0.343 s (H100) | 0.652 s (A100) | 0.450 s (A100) |
| NCCL share | 52.1% (H100) | 10.5% (A100) | — |
| elementwise | 26.8% (H100) | 61.0% (A100) | — |
| GEMM | 7.6% (H100) | 15.1% (A100) | — |

---

**SFNO backbone copies in the repo.** All Modulus-lineage, none identical.

| file | lines |
|---|---|
| `physicsnemo_ai_rossby/.../modulus_sfno/sfnonet.py` | 859 |
| `PanguWeather/v2.0/networks/modulus_sfno/sfnonet.py` | 838 |
| `si/modules/models/old/SFNO/sfnonet.py` | 768 |
| `ACE2_retrain/ace_exp/fme/ace/models/modulus/sfnonet.py` | 749 |
| `ACE2_retrain/.../fme/core/models/conditional_sfno/sfnonet.py` | 736 |
| `ACE2_retrain/.../fme/ace/models/makani/sfnonet.py` | 588 |

---

**Spectral no-op copy guard, by tree.**

| tree | `SpectralConvS2` guard | block-norm buffer | `hard_thresholding_fraction` |
|---|---|---|---|
| ACE2 | present | `zeros_like(x)` → bf16 | 1.0 |
| PanguWeather | absent | `zeros(..., float32)` | 1.0 |
| ai-rossby | absent | `zeros(..., float32)` | — |
