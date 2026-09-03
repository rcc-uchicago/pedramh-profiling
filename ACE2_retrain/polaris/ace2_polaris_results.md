# ACE2 on Polaris — complete measured results

Generated from the CSVs the launchers wrote; **do not hand-edit** — regenerate.
Sources: `$MEMBER_ROOT/bench/ace2_polaris_scaling.csv` (weak ladder + placement +
failed arms), `ace2_polaris_strongscale.csv` (strong scaling), and
`epoch_telemetry_ace2_polaris.csv` (the per-epoch rows those are derived from).

Common to EVERY row below unless stated: Polaris 4×A100-40GB/node, `fme` @ our
`ace_exp` checkout, torch 2.10.0+cu129 / NCCL 2.27.5, `NCCL_ALGO=Ring`,
transport `AWS Libfabric` (aws-ofi-nccl 1.21.1 + libfabric 2.3.1,
`OFI_NCCL_PROGRESS_MODEL=AUTO`), `--cpu-bind depth -d 8`, `OMP_NUM_THREADS=2`,
4 loader workers/rank, 60 timed steps, 1 epoch, AMP bf16, LR 1e-4 flat,
`env_source=manual-reconstruction`, store = the single 2.4 TB NetCDF (`data=nc`).

## Column dictionary — with units

| column | unit | meaning |
|---|---|---|
| `jobid` | — | PBS job id (short form) |
| `nodes` | count | allocation size actually trained on (after any GPU-health pruning) |
| `ranks` | count | = nodes × 4; one rank per GPU |
| `local_batch` | samples/GPU | the knob. Per-GPU work |
| `global_batch` | samples/step | fme's `batch_size`, which is **GLOBAL** — it divides by world size internally |
| `data` | — | store identity; `nc` = the unconverted 2,388.77 GB NetCDF |
| `rep` | count | repetition label; reps are interleaved across rungs, never batched |
| `gpu_order` | forward\|reverse | `reverse` sets `CUDA_VISIBLE_DEVICES=3,2,1,0` (NUMA-local pairing) |
| `steps` | count | timed steps **requested** |
| `n_steps` | count | timed steps **actually recorded**; a mismatch fails the parse |
| `step_med_ms` | **ms** | **median** GPU step time. train_on_batch → optimizer → EMA → scheduler. The headline |
| `step_p90_ms` | **ms** | 90th percentile step |
| `step_mean_ms` | **ms** | mean step — ⚠ contaminated by 1–2 multi-second warmup steps; do not quote |
| `step_std_ms` | **ms** | population stdev — ⚠ same contamination; std > mean is warmup, not spread |
| `samples_s_rank` | samples/s/rank | **derived**: local_batch ÷ (step_med/1000). GPU-time view |
| `samples_s_total` | samples/s | **derived**: samples_s_rank × ranks. GPU-time view |
| `samples_s_wall` | samples/s | **measured**: n_steps × local_batch × ranks ÷ epoch_wall_s. Includes loader idle |
| `gpu_busy_frac` | fraction 0–1 | Σ step_ms ÷ epoch_wall_s. ⚠ **loader idle, NOT comms cost** — exposed NCCL counts as *busy* |
| `epoch_wall_s` | **s** | wall clock for the timed window only (excludes validation + first-batch probe) |
| `peak_mem_gb` | **GiB** | ⚠ **GiB (bytes/1024³), despite the column name.** Run-to-date max over ranks. Not comparable to PanguWeather's decimal-GB column of the same name |
| `transport` | — | network NCCL selected; anything but `AWS Libfabric` invalidates the row |
| `world_sizes_seen` | count | world size off the trainer's own banner — guards against silent NonDistributed fallback |
| `ranks_reporting` | count | distinct PALS rank labels that emitted the banner — independent second view |
| `torch` | — | interpreter's torch build |
| `env_source` | — | `module-conda` or `manual-reconstruction` (the modulefile is broken) |
| `omp_threads` | count | OMP_NUM_THREADS actually in effect |
| `log` | — | per-arm log filename |
| `*(derived below)* read_MB_s` | **MB/s** | samples_s_wall × 41.73 MB/sample — implied demand on the single OST |

⚠ **`samples_s_rank`/`_total` (GPU time) and `samples_s_wall` (wall clock) have
different denominators.** `gpu_busy_frac` is the ratio between the two views.

## Table 1 — every arm, every column

| experiment | `jobid` | `nodes` | `ranks` | `local_batch` | `global_batch` | `data` | `rep` | `gpu_order` | `steps` | `n_steps` | `step_med_ms` | `step_p90_ms` | `step_mean_ms` | `step_std_ms` | `samples_s_rank` | `samples_s_total` | `samples_s_wall` | `gpu_busy_frac` | `epoch_wall_s` | `peak_mem_gb` | `transport` | `world_sizes_seen` | `ranks_reporting` | `omp_threads` | read MB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BATCH-SEARCH | 7586496 | 1 | 4 | 1 | 4 | nc | 1 | forward | 60 | 60 | 380.606 | 383.216 | 446.858 | 481.426 | 2.6274 | 10.5096 | 8.3474 | 0.9325 | 28.751 | 21.316 | AWS Libfabric | 4 | 4 | 2 | 348 |
| WEAK-LADDER | 7586506 | 1 | 4 | 2 | 8 | nc | 1 | forward | 60 | 60 | 715.404 | 718.28 | 788.576 | 529.182 | 2.7956 | 11.1825 | 9.4927 | 0.9357 | 50.565 | 33.959 | AWS Libfabric | 4 | 4 | 2 | 396 |
| FAILED | 7586526 | 1 | 4 | 3 | 12 | nc | 1 | forward | 60 | — | — | — | — | — | — | — | — | — | — | — | AWS Libfabric | 4 | 4 | 2 | — |
| WEAK-LADDER | 7586590 | 2 | 8 | 2 | 16 | nc | 1 | forward | 60 | 60 | 1203.473 | 1247.781 | 1285.536 | 502.385 | 1.6619 | 13.2949 | 11.9791 | 0.9625 | 80.139 | 33.959 | AWS Libfabric | 8 | 8 | 2 | 500 |
| WEAK-LADDER | 7588696 | 1 | 4 | 2 | 8 | nc | 1 | forward | 60 | 60 | 716.147 | 719.409 | 771.233 | 394.513 | 2.7927 | 11.1709 | 9.6346 | 0.9288 | 49.821 | 33.959 | AWS Libfabric | 4 | 4 | 2 | 402 |
| WEAK-LADDER | 7588702 | 4 | 16 | 2 | 32 | nc | 1 | forward | 60 | 60 | 1401.679 | 1495.657 | 1472.913 | 488.462 | 1.4269 | 22.8298 | 21.0485 | 0.9688 | 91.218 | 33.959 | AWS Libfabric | 16 | 16 | 2 | 878 |
| FAILED | 7588719 | 8 | 32 | 2 | 64 | nc | 1 | forward | 60 | — | — | — | — | — | — | — | — | — | — | — | AWS Libfabric | — | 0 | 2 | — |
| WEAK-LADDER | 7588721 | 2 | 8 | 2 | 16 | nc | 2 | forward | 60 | 60 | 1250.638 | 1355.009 | 1332.534 | 491.099 | 1.5992 | 12.7935 | 11.5596 | 0.9627 | 83.048 | 33.959 | AWS Libfabric | 8 | 8 | 2 | 482 |
| WEAK-LADDER | 7588734 | 8 | 32 | 2 | 64 | nc | 1 | forward | 60 | 60 | 1498.546 | 1659.304 | 1576.622 | 511.97 | 1.3346 | 42.7081 | 39.3859 | 0.9703 | 97.497 | 33.959 | AWS Libfabric | 32 | 32 | 2 | 1644 |
| WEAK-LADDER | 7588735 | 1 | 4 | 2 | 8 | nc | 3 | forward | 60 | 60 | 716.02 | 730.986 | 780.584 | 462.831 | 2.7932 | 11.1729 | 9.5183 | 0.9287 | 50.429 | 33.959 | AWS Libfabric | 4 | 4 | 2 | 397 |
| WEAK-LADDER | 7588758 | 4 | 16 | 2 | 32 | nc | 2 | forward | 60 | 60 | 1450.487 | 1550.563 | 1532.753 | 508.559 | 1.3788 | 22.0616 | 20.2398 | 0.9695 | 94.863 | 33.959 | AWS Libfabric | 16 | 16 | 2 | 845 |
| WEAK-LADDER | 7588759 | 2 | 8 | 2 | 16 | nc | 3 | forward | 60 | 60 | 1204.354 | 1239.938 | 1277.206 | 501.639 | 1.6606 | 13.2851 | 12.0314 | 0.9604 | 79.791 | 33.959 | AWS Libfabric | 8 | 8 | 2 | 502 |
| STRONG-SCALE | 7588972 | 4 | 16 | 1 | 16 | nc | 1 | forward | 60 | 60 | 1156.587 | 1200.601 | 1204.841 | 316.743 | 0.8646 | 13.8338 | 12.9124 | 0.9723 | 74.347 | 21.316 | AWS Libfabric | 16 | 16 | 2 | 539 |
| PLACEMENT | 7588998 | 1 | 4 | 2 | 8 | nc | 1 | reverse | 60 | 60 | 715.185 | 718.314 | 777.888 | 451.305 | 2.7965 | 11.1859 | 9.5264 | 0.9263 | 50.386 | 33.959 | AWS Libfabric | 4 | 4 | 2 | 398 |
| PLACEMENT | 7588999 | 4 | 16 | 2 | 32 | nc | 1 | reverse | 60 | 60 | 1466.02 | 1811.477 | 1564.36 | 491.747 | 1.3642 | 21.8278 | 19.7832 | 0.9671 | 97.052 | 33.959 | AWS Libfabric | 16 | 16 | 2 | 826 |

`torch`=2.10.0+cu129 and `env_source`=manual-reconstruction on every row (omitted above for width).

## Table 2 — WEAK-SCALING LADDER (the headline experiment)

`local_batch` held at **2 samples/GPU**; `global_batch` grows with the allocation.
Per-GPU work is constant, so every change is communication + imbalance.

| nodes | ranks | global_batch (samples/step) | step_med (ms) | rep spread | n | samples/s/rank | samples/s total | speedup vs 1n | weak-scaling eff. | eff. vs **2n** (min. viable) | gpu_busy_frac | read (MB/s) | peak mem (GiB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 8 | **716.0** | ±0.1% | 3 | 2.7932 | 11.17 | 1.00× | 100% | — | 0.9288 | 397 | 33.959 |
| 2 | 8 | 16 | **1204.4** | ±3.9% | 3 | 1.6606 | 13.29 | 1.19× | 59% | 100% | 0.9625 | 500 | 33.959 |
| 4 | 16 | 32 | **1426.1** | ±3.4% | 2 | 1.4024 | 22.44 | 2.01× | 50% | 84% | 0.9691 | 861 | 33.959 |
| 8 | 32 | 64 | **1498.5** | — | 1 | 1.3346 | 42.71 | 3.82× | 48% | 80% | 0.9703 | 1644 | 33.959 |

**Shape: the cliff is the first hop, then it saturates** — −42% per-GPU at 1→2 nodes,
then −12% and −6.5% for the next two doublings. First-hop penalty **+488 ms/step**.
⚠ Efficiency-vs-1-node is the **wrong headline**: 1 node cannot hold the production
global batch of 16 at all (see Table 3). The `eff. vs 2n` column is the usable one.
⚠ **8n is n=1** and must not be published as a ladder point.

## Table 3 — BATCH-SIZE SEARCH (1 node, one arm per value, never fitted)

| local_batch (samples/GPU) | global_batch | peak mem (GiB) | of 39.49 GiB | step_med (ms) | ms/sample | samples/s/rank | result |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 21.316 | 54% | 380.6 | 380.6 | 2.6274 | ✅ fits (n=1) |
| 2 | 8 | 33.959 | 86% | 716.0 | 358.0 | 2.7932 | ✅ fits (n=3) |
| 3 | 12 | **≥38.20 at OOM** | **≥97%** | — | — | — | ❌ **OOM** — died in `sfnonet.py:250` asking for 286 MiB with 93 MiB free |

**+12.643 GiB per added sample; NO CLIFF** — arm 3 fails exactly where linear predicts
(21.316 + 2×12.643 = 46.6 > 39.49). makani's discrete 12→16 cliff did **not** reproduce.
⚠ Two points define a line trivially; the claim is the **measured boundary**
(local 2 fits, 3 does not), not the model.
⇒ **the config's `batch_size: 16` needs 8 GPUs = 2 nodes.**

## Table 4 — STRONG SCALING (fixed global_batch = 16 — makani's axis)

| nodes | ranks | local_batch | global_batch | step_med (ms) | samples/s total | gpu_busy_frac | peak mem (GiB) | n |
|---|---|---|---|---|---|---|---|---|
| 2 | 8 | 2 | 16 | 1204.4 | 13.29 | 0.9625 | 33.959 | 3 |
| **4** | 16 | **1** | 16 | **1156.6** | **13.83** | 0.9723 | 21.316 | 1 |
| 1 | 4 | 4 | 16 | — | — | — | — | ❌ **impossible — OOM** |

**+4.1% for 2× the hardware** — faster, not slower. makani's ladder never recovered its
1-node throughput at any larger node count; ACE2 does not get worse.
⚠ +4.1% for 2× hardware is **2% efficiency on the added nodes** — 'does not get worse',
not 'scales'. And makani's headline question *is 1 node fastest?* **cannot be posed here.**
⚠ This row lives in a **separate CSV**: a strong-scaling point in a weak-scaling table is
the exact mislabelling the handoff §3.1 warns about.

## Table 5 — PLACEMENT A/B (`GPU_ORDER`)

| nodes | forward step_med (ms) | n | reverse step_med (ms) | n | delta | forward's own spread | resolvable? |
|---|---|---|---|---|---|---|---|
| 1 | 716.0 | 3 | 715.2 | 1 | **-0.12%** | ±0.1% | **no** — inside the baseline spread |
| 4 | 1426.1 | 2 | 1466.0 | 1 | **+2.80%** | ±3.4% | **no** — inside the baseline spread |

⚠ **Neither delta is resolvable** (reverse is n=1; each delta sits inside its baseline's
spread). ✅ What *is* established: **makani's −7.0% does not reproduce** — that would sit
far outside ±3.4%. ⚠ And makani's −7.0% was measured at 4 nodes **SHARDED**
(model-parallel); **ACE2 has no model-parallel path**, so that configuration does not
exist here. `forward` remains correct for every ACE2 config measured.

## Table 6 — FAILED / DIAGNOSTIC ARMS (these are measurements, not lost jobs)

| jobid | config | outcome |
|---|---|---|
| 7586526 | 1 node, local_batch **3**, global 12 | ❌ `torch.OutOfMemoryError` in `sfnonet.py:250` (`x + self.outer_skip(residual)`): 38.20 GiB allocated, 39.39 in use, asking 286 MiB with 93.25 MiB free. **This is the batch-boundary measurement.** |
| 7588719 | 8 nodes, 32 ranks, local 2 | ❌ `ValueError: No batches in dataloader: 0 samples, batch size is 2` — a *fixed* 4-day validation window (~14 samples) is under one local batch per rank at 32 ranks. **Launcher bug, only visible at ≥32 ranks**; fixed by sizing the window from `NRANKS`. |
| 7586630 | I/O probe v1 | ❌ `IndexError` — probe assumed `ndim==3`; `global_mean_co2` is 1-D in time |
| 7586642 | I/O probe v2 | ⚠ ran, but **self-contaminated**: overlapping seeds across repeats (0/12/56/78% as readers grew) + taking the *best* repeat ⇒ fabricated linear scaling to a false 636.7 MB/s peak |

## Table 7 — NON-TIMING measurements (no CSV; these settled the design questions)

| # | measurement | job | value | unit | what it decided |
|---|---|---|---|---|---|
| 1 | Lustre stripe of the store | — (`lfs getstripe`) | `lmm_stripe_count: 1`, 1 MiB stripe, OST idx 46 | — | the whole 2,388.77 GB `.nc` is on **one OST**, read by every rank |
| 2 | Store shape | — (h5py) | 121,262 × 180 × 360, **contiguous, uncompressed** (`chunks=None`, `compression=None`) | timesteps × lat × lon | a 3-timestep window = one ~778 KB contiguous read/variable; **no chunk amplification** |
| 3 | Bytes per sample | — (derived from 2) | **41.73** | MB/sample | 56 config variables × 3 timesteps; the basis of every `read MB/s` figure |
| 4 | Largest single collective | 7586590 (2n, `-v FR_DUMP=1`) | `all_reduce`, `numel=455,831,040`, `dtype=Float` = **1,823,324,160 B = 1738.86 MiB** | bytes | 🔴 **above the ~1000 MiB where Tree was measured to silently corrupt ⇒ `NCCL_ALGO=Ring` is load-bearing, not insurance** |
| 5 | When that collective fires | 7586590 | `record_id=13`, `collective_seq_id=14`, **once** per run | — | after DDP's parameter broadcast, **before the first backward** ⇒ it is **not a gradient bucket**, which explains ai-rossby's `bucket_cap_mb` null result |
| 6 | Per-*step* collectives | 7586590 | ~11 buckets, largest `numel=53,824,896` ≈ **215** | MB | the benign shape; the exposure is the startup collective only |
| 7 | Gradient volume | 7586590 + source | **1.823** (not 2.67) | GB | `s2convolutions.py:148` declares **float32** with a trailing size-2 dim; `view_as_complex` at use time. The handoff's complex64 'correction' **double-counts** |
| 8 | Single-OST read ceiling | 7587664 (app-free) | 21.4 / 42.9 / 82.2 / **161.8** / 220.3 / 343.2 at 1/2/4/8/16/32 readers | MB/s | ⚠ **understates the OST by ~4.8×** — divides by a wall clock including `mp.Pool` startup and 32–128 opens of a 2.4 TB file. Its **latency** series (2.0→2.7→3.7 s/window at 8→16→32) is real; its **aggregate is a floor** |
| 9 | Achieved read rate, real loader | 7588734 (8n) | **1,644** | MB/s | at 128 concurrent readers with `gpu_busy_frac` 0.9703 ⇒ **I/O is not the bottleneck; the zarr conversion is not justified** |
| 10 | Model size | — | **455,831,040** | float32 params | = the `numel` of #4, confirming the collective is the whole model |

## Environment — constant across every arm

| item | value |
|---|---|
| hardware | Polaris, 4 × A100-SXM4-40GB per node (**39.49 GiB** usable), NV4 mesh |
| venv | `$MEMBER_ROOT/conda-envs/fme-venv` — python 3.12.11, **torch 2.10.0+cu129, NCCL 2.27.5**, torch_harmonics 0.8.0, zarr 3.3.0; `fme` **editable** from `ACE2_retrain/ace_exp` |
| fabric | self-built aws-ofi-nccl **1.21.1** + cray libfabric **2.3.1** + `OFI_NCCL_PROGRESS_MODEL=AUTO`; transport reported `AWS Libfabric` on **every** arm |
| launcher | PALS `mpiexec --ppn 4 --cpu-bind depth -d 8 --label --line-buffer` + `polaris_rank_env.sh` (PMI_* → `env://`) → `ace2_telemetry.py` |
| knobs | `NCCL_ALGO=Ring`, `OMP_NUM_THREADS=2`, `MASTER_PORT=20000+jobid%20000`, `TMPDIR=/tmp`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (verified honoured — both env-var spellings are in `libc10_cuda.so`) |
| model/config | `config_polaris.yaml` = `config_midway.yaml` with 11 paths repointed, nothing else. SFNO, embed_dim 384, 8 layers, dhconv; AMP **bf16**; AdamW(fused) — `FusedAdam` needs **no apex** here; **LR 1e-4 and FLAT** (fme's default scheduler is `None`) |
| per arm | 60 timed steps, 1 epoch, `sample_with_replacement = 60 × global_batch` so every rank runs exactly 60 steps; `inference=null`; `save_checkpoint=false` |

## What is NOT measured

| gap | why it matters |
|---|---|
| **8n has n=1**; 4n reverse and 1n reverse have n=1 | nothing at those points should be published; `run_ace2_ladder.sh` fills the shortest rungs first |
| The 1.823 GB startup collective has **no named call site** | the flight-recorder dump carried no stack frames; one arm with stack capture would close ai-rossby's open question properly rather than by analogy |
| **No kernel-level profile on Polaris** (prereg P6) | Midway's 'NCCL is 40–46% of GPU kernel time' was taken on A100-**PCIE with no NVLink**; whether it transfers to this NV4 mesh is untested. `ace2_nvtx.py` exists; there is no Polaris nsys launcher |
| **No LR sweep**, and batch 16 on 2 nodes is a training-regime change | jesswan's call, not ours |
| **No equivalence baseline** (DESIGN §4) | required before any hot-path change is committed |
| `GPU_ORDER` A/B is **not node-matched** | makani's arm used 3+3 reps on the same nodes precisely because node variation can swamp a small effect |
## Table 8 — ACE2 vs makani: SAME SFNO, ONE CONFIG KNOB APART

Both are the Modulus/makani SFNO on a **180×360 equiangular grid** with **identical**
`embed_dim=384`, `num_layers=8`, `operator_type=dhconv`, `filter_type=linear`,
`normalization_layer=instance_norm`, `hard_thresholding_fraction=1.0`, `use_mlp`,
`separable=False`. makani numbers from `makani_bench_report.md` + `polaris_ace2_multinode_handoff.md` §1f;
makani config = `makani_sfno/polaris/e3sm_alldata_full.yaml`.

| | ACE2 | makani | note |
|---|---|---|---|
| **`scale_factor`** | **1** | **3** | ⭐ the only architectural difference that matters |
| internal grid in the 8 blocks | 180×360 | 60×120 | **9× fewer positions** for makani |
| autoregressive steps / sample | **2** (`n_forward_steps=2`) | **1** (`n_future=0`) | ACE2 backprops through 2 applications |
| `pos_embed` | true | none | minor |
| channels | 56 (43 in / 40 out) | 101 + 7 forcing | makani's encoder is wider |
| dhconv weight / layer | 384×384×**180**×2 = **212.34 MB** | 384×384×**60**×2 = **70.78 MB** | = internal lat |
| params | **455,831,040** | **147,860,000** | ratio **3.08×** = the scale_factor ratio |
| gradient volume | **1.823 GB** | **0.591 GB** | derived 0.566 GB dhconv + rest ⇒ reproduces makani's recorded 591 MB |
| dhconv share of params | 93.2% | 95.7% | both are almost entirely spectral weights |
| **largest single collective** | **1738.86 MiB** (whole model, once, at startup) | ~591 MB (one bucket) | |
| **tree-defect exposure** | 🔴 **YES — `NCCL_ALGO=Ring` is load-bearing** | ✅ no — below the ~1 GB threshold | the single biggest operational difference |
| ms / sample | **335.4** | 45.7 | **7.3×**; per *model application* 167.7 vs 45.7 = 3.7× |
| **GiB per added sample** | **12.64** | **0.73** | **17.3×** — vs 18× predicted by 9× spatial × 2× rollout |
| max samples/GPU on 40 GB | **2** | **12** (16 OOMs) | |
| memory at that max | 33.96 GiB | 18.97 GiB | |
| production global batch | 16 | 32 | |
| **fewest GPUs that hold it** | **8 (2 nodes)** | **4 (1 node)** | ⇐ this is why the scaling verdicts differ |
| best measured config | 2+ nodes | **1 node** | makani: 1 node cheapest AND fastest |

**The decomposition closes.** `scale_factor` 1 vs 3 gives 3× the spectral modes ⇒ **3.08×**
the parameters (measured 455.83 M / 147.86 M), and 9× the internal spatial positions;
times ACE2's 2-step rollout that is **18× predicted activation memory per sample against
17.3× measured**. Nothing else needs to be invoked.

⇒ **ACE2 and makani do not disagree about Polaris; they are different-sized models.**
makani at `scale_factor=3` is small enough that one node holds its whole production batch,
so it never has to touch the fabric — and its ladder found that touching it costs ~234 ms
and is never worth it. ACE2 at `scale_factor=1` is 17× heavier per sample in activations,
cannot fit its production batch below 8 GPUs, and therefore **has no fabric-free option**.
Given it must pay the toll, it then amortises it well (82–84% incremental efficiency from
2 nodes), because it is also 7.3× heavier per sample in compute.

⚠ **`scale_factor` is a SCIENCE knob, not a performance knob.** ACE2 running at 1 rather
than 3 is what the ai2cm config specifies; changing it changes what the model computes and
is jesswan's call, not a tuning decision. It is listed here to explain the measurements,
**not** as a proposal.

⚠ Two caveats on the compute row: makani's 45.7 ms/sample is an *average* at 8 samples/GPU
(it includes fixed per-step overhead), while ACE2's 335.4 is a *marginal* two-point fit;
and the 3.7×-per-application residual is not derived from first principles — the encoder
runs at full resolution in both and makani has 2.4× the channels there, which partly
offsets its 9× spatial advantage.
## Table 9 — WHAT THE FABRIC COSTS ACE2

Weak scaling holds per-GPU work identical on every row, so **everything above the
1-node step is exposed inter-node cost** (fabric + load imbalance).

| nodes | step_med (ms) | over 1n (ms) | **% of the step** | node·s / sample | vs 1 node |
|---|---|---|---|---|---|
| 1 | 716.0 | — (baseline) | — | 0.0895 | 1.00× |
| 2 | 1204.4 | **+488.4** | **40.6%** | 0.1505 | 1.68× |
| 4 | 1426.1 | +710.1 | 49.8% | 0.1783 | 1.99× |
| 8 | 1498.5 | +782.5 | **52.2%** | 0.1873 | **2.09×** |

**Over half of the 8-node step is fabric**, and 8 nodes costs **2.09× the node-seconds
per sample** that 1 node would — except 1 node cannot run the production batch, so the
usable comparison is **2 → 8 nodes: +24% node-hours for 3.2× the wall-clock throughput.**

### ACE2 pays a BIGGER toll than makani, not a smaller one

| | toll (ms/step) | gradient volume | effective bandwidth |
|---|---|---|---|
| **ACE2** | **488.4** | 1.823 GB | 3.73 GB/s |
| makani | 234.0 | 0.591 GB | 2.53 GB/s |
| ratio | **2.09×** | 3.08× | — |

Both land in the 2.5–4.4 GB/s this stack has measured elsewhere, so the toll is
**bandwidth-bound and scales with gradient volume** — ACE2 pays more because
`scale_factor=1` gives it 3.08× the gradients (Table 8).

### So why does ACE2's ladder still look better? Because of what the toll hides behind

| config | toll | per-GPU compute | toll as % of compute |
|---|---|---|---|
| ACE2 @ 2 samples/GPU | 488.4 ms | 716.0 ms | **68%** |
| ACE2 @ 1 sample/GPU | 488.4 ms | 380.6 ms | 128% |
| makani @ 8 samples/GPU | 234.0 ms | 365.4 ms | 64% |
| makani @ 1 sample/GPU | 234.0 ms | 45.7 ms | **512%** ← the regime ACE2 cannot reach |

At their respective 1-node production configs the toll is a **similar fraction** (68% vs
64%). The divergence is entirely in the *range of configurations reachable*: makani's
fixed-batch ladder could slice to 1 sample/GPU where the fabric is 5× the compute, while
ACE2 caps at 2 samples/GPU and bottoms out at 128%.

⚠ **`gpu_busy_frac` cannot see any of this** — it reads 0.9288 → 0.9703 *rising* across
these same rows, because exposed NCCL runs as GPU kernels and counts as *busy*. It
measures loader idle. Use the `over 1n` column for communication cost.

⚠ **`over 1n` is not purely fabric**: it also contains load imbalance and straggler
effects, which weak scaling does not separate. And the 716.0 ms baseline already
contains **intra-node** NCCL over NVLink, so the figures above are the *inter-node
increment*, not ACE2's total communication cost. Separating those needs the kernel-level
capture that prereg P6 still lacks.
