# Profiling tables — ACE2 & PanguWeather (2026-08-18/19)

Tables only. Detail: `ACE2_retrain/bench_midway_notes.md`,
`PanguWeather/v2.0/bench_midway_notes.md`, `polaris_bench_report.md`.

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

**PanguWeather — NVTX phase durations, 4× H100 NVL.** Job 53539872.

| range | n | median | min | max |
|---|---|---|---|---|
| `data_prep` | 160 | 0.5 ms | 0.2 | 2.3 |
| `forward_loss` | 160 | 46.3 ms | 42.5 | 73.7 |
| `backward` | 160 | 539.8 ms | 112.2 | 993.1 |
| `optimizer` | 160 | 4.5 ms | 3.4 | 6.3 |
| step total | 156 | 995.5 ms (std 278.3) | — | — |

---

**NCCL share vs gradient volume, identical node (`midway3-0423`).**

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
