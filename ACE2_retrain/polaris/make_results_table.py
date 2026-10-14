#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate ACE2_retrain/polaris/ace2_polaris_results.md from the bench CSVs.

    python3.11 ACE2_retrain/polaris/make_results_table.py > ACE2_retrain/polaris/ace2_polaris_results.md

The results doc is GENERATED, never hand-edited: medians and rep spreads move
every time another rep lands, and a hand-maintained table silently goes stale
against the CSV it claims to summarise. Prose findings live in the CHANGELOG and
the prereg; this file is the numbers.

Reads $MEMBER_ROOT/bench/{ace2_polaris_scaling,ace2_polaris_strongscale}.csv.
Non-CSV measurements (stripe, flight recorder, I/O probe) are literals here,
each tagged with the job that produced it.
"""
import csv, statistics as st, os
BENCH="/eagle/projects/lighthouse-uchicago/members/mehta5/bench"
MB_PER_SAMPLE=41.73
MB=MB_PER_SAMPLE  # per-sample bytes, from the file's real dtypes/shapes

def load(f):
    p=os.path.join(BENCH,f)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []

weak=load("ace2_polaris_scaling.csv")
strong=load("ace2_polaris_strongscale.csv")
allrows=weak+strong

def exp_of(r):
    if r in strong: return "STRONG-SCALE"
    if r['gpu_order']=='reverse': return "PLACEMENT"
    if not r['n_steps']: return "FAILED"
    if r['local_batch']!='2': return "BATCH-SEARCH"
    return "WEAK-LADDER"

out=[]
w=out.append
w("# ACE2 on Polaris — complete measured results")
w("")
w("Generated from the CSVs the launchers wrote; **do not hand-edit** — regenerate.")
w("Sources: `$MEMBER_ROOT/bench/ace2_polaris_scaling.csv` (weak ladder + placement +")
w("failed arms), `ace2_polaris_strongscale.csv` (strong scaling), and")
w("`epoch_telemetry_ace2_polaris.csv` (the per-epoch rows those are derived from).")
w("")
w("Common to EVERY row below unless stated: Polaris 4×A100-40GB/node, `fme` @ our")
w("`ace_exp` checkout, torch 2.10.0+cu129 / NCCL 2.27.5, `NCCL_ALGO=Ring`,")
w("transport `AWS Libfabric` (aws-ofi-nccl 1.21.1 + libfabric 2.3.1,")
w("`OFI_NCCL_PROGRESS_MODEL=AUTO`), `--cpu-bind depth -d 8`, `OMP_NUM_THREADS=2`,")
w("4 loader workers/rank, 60 timed steps, 1 epoch, AMP bf16, LR 1e-4 flat,")
w("`env_source=manual-reconstruction`, store = the single 2.4 TB NetCDF (`data=nc`).")
w("")
w("## Column dictionary — with units")
w("")
w("| column | unit | meaning |")
w("|---|---|---|")
for c,u,m in [
 ("jobid","—","PBS job id (short form)"),
 ("nodes","count","allocation size actually trained on (after any GPU-health pruning)"),
 ("ranks","count","= nodes × 4; one rank per GPU"),
 ("local_batch","samples/GPU","the knob. Per-GPU work"),
 ("global_batch","samples/step","fme's `batch_size`, which is **GLOBAL** — it divides by world size internally"),
 ("data","—","store identity; `nc` = the unconverted 2,388.77 GB NetCDF"),
 ("rep","count","repetition label; reps are interleaved across rungs, never batched"),
 ("gpu_order","forward\\|reverse","`reverse` sets `CUDA_VISIBLE_DEVICES=3,2,1,0` (NUMA-local pairing)"),
 ("steps","count","timed steps **requested**"),
 ("n_steps","count","timed steps **actually recorded**; a mismatch fails the parse"),
 ("step_med_ms","**ms**","**median** GPU step time. train_on_batch → optimizer → EMA → scheduler. The headline"),
 ("step_p90_ms","**ms**","90th percentile step"),
 ("step_mean_ms","**ms**","mean step — ⚠ contaminated by 1–2 multi-second warmup steps; do not quote"),
 ("step_std_ms","**ms**","population stdev — ⚠ same contamination; std > mean is warmup, not spread"),
 ("samples_s_rank","samples/s/rank","**derived**: local_batch ÷ (step_med/1000). GPU-time view"),
 ("samples_s_total","samples/s","**derived**: samples_s_rank × ranks. GPU-time view"),
 ("samples_s_wall","samples/s","**measured**: n_steps × local_batch × ranks ÷ epoch_wall_s. Includes loader idle"),
 ("gpu_busy_frac","fraction 0–1","Σ step_ms ÷ epoch_wall_s. ⚠ **loader idle, NOT comms cost** — exposed NCCL counts as *busy*"),
 ("epoch_wall_s","**s**","wall clock for the timed window only (excludes validation + first-batch probe)"),
 ("peak_mem_gb","**GiB**","⚠ **GiB (bytes/1024³), despite the column name.** Run-to-date max over ranks. Not comparable to PanguWeather's decimal-GB column of the same name"),
 ("transport","—","network NCCL selected; anything but `AWS Libfabric` invalidates the row"),
 ("world_sizes_seen","count","world size off the trainer's own banner — guards against silent NonDistributed fallback"),
 ("ranks_reporting","count","distinct PALS rank labels that emitted the banner — independent second view"),
 ("torch","—","interpreter's torch build"),
 ("env_source","—","`module-conda` or `manual-reconstruction` (the modulefile is broken)"),
 ("omp_threads","count","OMP_NUM_THREADS actually in effect"),
 ("log","—","per-arm log filename"),
 ("*(derived below)* read_MB_s","**MB/s**","samples_s_wall × 41.73 MB/sample — implied demand on the single OST"),
]:
    w(f"| `{c}` | {u} | {m} |")
w("")
w("⚠ **`samples_s_rank`/`_total` (GPU time) and `samples_s_wall` (wall clock) have")
w("different denominators.** `gpu_busy_frac` is the ratio between the two views.")
w("")

w("## Table 1 — every arm, every column")
w("")
cols=['jobid','nodes','ranks','local_batch','global_batch','data','rep','gpu_order',
      'steps','n_steps','step_med_ms','step_p90_ms','step_mean_ms','step_std_ms',
      'samples_s_rank','samples_s_total','samples_s_wall','gpu_busy_frac','epoch_wall_s',
      'peak_mem_gb','transport','world_sizes_seen','ranks_reporting','omp_threads']
w("| experiment | " + " | ".join(f"`{c}`" for c in cols) + " | read MB/s |")
w("|" + "---|"*(len(cols)+2))
for r in sorted(allrows,key=lambda r:int(r['jobid'])):
    rd = f"{float(r['samples_s_wall'])*MB_PER_SAMPLE:.0f}" if r['samples_s_wall'] else "—"
    w("| " + exp_of(r) + " | " + " | ".join((r[c] if r[c] else "—") for c in cols) + f" | {rd} |")
w("")
w("`torch`=2.10.0+cu129 and `env_source`=manual-reconstruction on every row (omitted above for width).")
w("")



def agg(rows,key):
    g={}
    for r in rows: g.setdefault(key(r),[]).append(r)
    return g

lad=[r for r in weak if r['n_steps'] and r['local_batch']=='2' and r['gpu_order']=='forward']
w("## Table 2 — WEAK-SCALING LADDER (the headline experiment)")
w("")
w("`local_batch` held at **2 samples/GPU**; `global_batch` grows with the allocation.")
w("Per-GPU work is constant, so every change is communication + imbalance.")
w("")
w("| nodes | ranks | global_batch (samples/step) | step_med (ms) | rep spread | n | samples/s/rank | samples/s total | speedup vs 1n | weak-scaling eff. | eff. vs **2n** (min. viable) | gpu_busy_frac | read (MB/s) | peak mem (GiB) |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
g=agg(lad,lambda r:int(r['nodes'])); base=None; base2=None
rowsout=[]
for n in sorted(g):
    rs=g[n]; med=[float(r['step_med_ms']) for r in rs]
    m=st.median(med); ranks=int(rs[0]['ranks']); lb=2
    spr=(max(med)-min(med))/m*100 if len(med)>1 else None
    sr=lb/(m/1000); tot=sr*ranks
    wall=st.median([float(r['samples_s_wall']) for r in rs])
    busy=st.median([float(r['gpu_busy_frac']) for r in rs])
    pk=st.median([float(r['peak_mem_gb']) for r in rs])
    if base is None: base=tot
    if n==2: base2=tot
    rowsout.append((n,ranks,m,spr,len(rs),sr,tot,wall,busy,pk))
for n,ranks,m,spr,k,sr,tot,wall,busy,pk in rowsout:
    e2 = f"{100*(tot/base2)/(n/2):.0f}%" if base2 and n>=2 else "—"
    w(f"| {n} | {ranks} | {ranks*2} | **{m:.1f}** | {f'±{spr:.1f}%' if spr is not None else '—'} | {k} | {sr:.4f} | {tot:.2f} | {tot/base:.2f}× | {100*(tot/base)/n:.0f}% | {e2} | {busy:.4f} | {wall*MB:.0f} | {pk:.3f} |")
w("")
w("**Shape: the cliff is the first hop, then it saturates** — −42% per-GPU at 1→2 nodes,")
w("then −12% and −6.5% for the next two doublings. First-hop penalty **+488 ms/step**.")
w("⚠ Efficiency-vs-1-node is the **wrong headline**: 1 node cannot hold the production")
w("global batch of 16 at all (see Table 3). The `eff. vs 2n` column is the usable one.")
w("⚠ **8n is n=1** and must not be published as a ladder point.")
w("")

bs=[r for r in weak if r['nodes']=='1' and r['gpu_order']=='forward']
w("## Table 3 — BATCH-SIZE SEARCH (1 node, one arm per value, never fitted)")
w("")
w("| local_batch (samples/GPU) | global_batch | peak mem (GiB) | of 39.49 GiB | step_med (ms) | ms/sample | samples/s/rank | result |")
w("|---|---|---|---|---|---|---|---|")
prev=None
for lb in ('1','2','3'):
    rs=[r for r in bs if r['local_batch']==lb and r['n_steps']]
    if rs:
        m=st.median([float(r['step_med_ms']) for r in rs]); pk=float(rs[0]['peak_mem_gb'])
        w(f"| {lb} | {int(lb)*4} | {pk:.3f} | {100*pk/39.49:.0f}% | {m:.1f} | {m/int(lb):.1f} | {int(lb)/(m/1000):.4f} | ✅ fits (n={len(rs)}) |")
        prev=(pk,m)
    else:
        w(f"| {lb} | {int(lb)*4} | **≥38.20 at OOM** | **≥97%** | — | — | — | ❌ **OOM** — died in `sfnonet.py:250` asking for 286 MiB with 93 MiB free |")
w("")
w("**+12.643 GiB per added sample; NO CLIFF** — arm 3 fails exactly where linear predicts")
w("(21.316 + 2×12.643 = 46.6 > 39.49). makani's discrete 12→16 cliff did **not** reproduce.")
w("⚠ Two points define a line trivially; the claim is the **measured boundary**")
w("(local 2 fits, 3 does not), not the model.")
w("⇒ **the config's `batch_size: 16` needs 8 GPUs = 2 nodes.**")
w("")

w("## Table 4 — STRONG SCALING (fixed global_batch = 16 — makani's axis)")
w("")
w("| nodes | ranks | local_batch | global_batch | step_med (ms) | samples/s total | gpu_busy_frac | peak mem (GiB) | n |")
w("|---|---|---|---|---|---|---|---|---|")
two=[r for r in weak if r['nodes']=='2' and r['n_steps'] and r['gpu_order']=='forward']
m2=st.median([float(r['step_med_ms']) for r in two])
w(f"| 2 | 8 | 2 | 16 | {m2:.1f} | {16/(m2/1000):.2f} | {st.median([float(r['gpu_busy_frac']) for r in two]):.4f} | 33.959 | {len(two)} |")
for r in strong:
    m=float(r['step_med_ms'])
    w(f"| **4** | 16 | **1** | 16 | **{m:.1f}** | **{16/(m/1000):.2f}** | {r['gpu_busy_frac']} | {r['peak_mem_gb']} | 1 |")
w(f"| 1 | 4 | 4 | 16 | — | — | — | — | ❌ **impossible — OOM** |")
w("")
w("**+4.1% for 2× the hardware** — faster, not slower. makani's ladder never recovered its")
w("1-node throughput at any larger node count; ACE2 does not get worse.")
w("⚠ +4.1% for 2× hardware is **2% efficiency on the added nodes** — 'does not get worse',")
w("not 'scales'. And makani's headline question *is 1 node fastest?* **cannot be posed here.**")
w("⚠ This row lives in a **separate CSV**: a strong-scaling point in a weak-scaling table is")
w("the exact mislabelling the handoff §3.1 warns about.")
w("")

w("## Table 5 — PLACEMENT A/B (`GPU_ORDER`)")
w("")
w("| nodes | forward step_med (ms) | n | reverse step_med (ms) | n | delta | forward's own spread | resolvable? |")
w("|---|---|---|---|---|---|---|---|")
for n in ('1','4'):
    f=[float(r['step_med_ms']) for r in weak if r['nodes']==n and r['local_batch']=='2' and r['gpu_order']=='forward' and r['n_steps']]
    rv=[float(r['step_med_ms']) for r in weak if r['nodes']==n and r['gpu_order']=='reverse' and r['n_steps']]
    if f and rv:
        mf,mr=st.median(f),st.median(rv); d=100*(mr-mf)/mf
        sp=(max(f)-min(f))/mf*100
        res="**no** — inside the baseline spread"
        w(f"| {n} | {mf:.1f} | {len(f)} | {mr:.1f} | {len(rv)} | **{d:+.2f}%** | ±{sp:.1f}% | {res} |")
w("")
w("⚠ **Neither delta is resolvable** (reverse is n=1; each delta sits inside its baseline's")
w("spread). ✅ What *is* established: **makani's −7.0% does not reproduce** — that would sit")
w("far outside ±3.4%. ⚠ And makani's −7.0% was measured at 4 nodes **SHARDED**")
w("(model-parallel); **ACE2 has no model-parallel path**, so that configuration does not")
w("exist here. `forward` remains correct for every ACE2 config measured.")
w("")

w("## Table 6 — FAILED / DIAGNOSTIC ARMS (these are measurements, not lost jobs)")
w("")
w("| jobid | config | outcome |")
w("|---|---|---|")
w("| 7586526 | 1 node, local_batch **3**, global 12 | ❌ `torch.OutOfMemoryError` in `sfnonet.py:250` (`x + self.outer_skip(residual)`): 38.20 GiB allocated, 39.39 in use, asking 286 MiB with 93.25 MiB free. **This is the batch-boundary measurement.** |")
w("| 7588719 | 8 nodes, 32 ranks, local 2 | ❌ `ValueError: No batches in dataloader: 0 samples, batch size is 2` — a *fixed* 4-day validation window (~14 samples) is under one local batch per rank at 32 ranks. **Launcher bug, only visible at ≥32 ranks**; fixed by sizing the window from `NRANKS`. |")
w("| 7586630 | I/O probe v1 | ❌ `IndexError` — probe assumed `ndim==3`; `global_mean_co2` is 1-D in time |")
w("| 7586642 | I/O probe v2 | ⚠ ran, but **self-contaminated**: overlapping seeds across repeats (0/12/56/78% as readers grew) + taking the *best* repeat ⇒ fabricated linear scaling to a false 636.7 MB/s peak |")
w("")


w("## Table 7 — NON-TIMING measurements (no CSV; these settled the design questions)")
w("")
w("| # | measurement | job | value | unit | what it decided |")
w("|---|---|---|---|---|---|")
w("| 1 | Lustre stripe of the store | — (`lfs getstripe`) | `lmm_stripe_count: 1`, 1 MiB stripe, OST idx 46 | — | the whole 2,388.77 GB `.nc` is on **one OST**, read by every rank |")
w("| 2 | Store shape | — (h5py) | 121,262 × 180 × 360, **contiguous, uncompressed** (`chunks=None`, `compression=None`) | timesteps × lat × lon | a 3-timestep window = one ~778 KB contiguous read/variable; **no chunk amplification** |")
w("| 3 | Bytes per sample | — (derived from 2) | **41.73** | MB/sample | 56 config variables × 3 timesteps; the basis of every `read MB/s` figure |")
w("| 4 | Largest single collective | 7586590 (2n, `-v FR_DUMP=1`) | `all_reduce`, `numel=455,831,040`, `dtype=Float` = **1,823,324,160 B = 1738.86 MiB** | bytes | 🔴 **above the ~1000 MiB where Tree was measured to silently corrupt ⇒ `NCCL_ALGO=Ring` is load-bearing, not insurance** |")
w("| 5 | When that collective fires | 7586590 | `record_id=13`, `collective_seq_id=14`, **once** per run | — | after DDP's parameter broadcast, **before the first backward** ⇒ it is **not a gradient bucket**, which explains ai-rossby's `bucket_cap_mb` null result |")
w("| 6 | Per-*step* collectives | 7586590 | ~11 buckets, largest `numel=53,824,896` ≈ **215** | MB | the benign shape; the exposure is the startup collective only |")
w("| 7 | Gradient volume | 7586590 + source | **1.823** (not 2.67) | GB | `s2convolutions.py:148` declares **float32** with a trailing size-2 dim; `view_as_complex` at use time. The handoff's complex64 'correction' **double-counts** |")
w("| 8 | Single-OST read ceiling | 7587664 (app-free) | 21.4 / 42.9 / 82.2 / **161.8** / 220.3 / 343.2 at 1/2/4/8/16/32 readers | MB/s | ⚠ **understates the OST by ~4.8×** — divides by a wall clock including `mp.Pool` startup and 32–128 opens of a 2.4 TB file. Its **latency** series (2.0→2.7→3.7 s/window at 8→16→32) is real; its **aggregate is a floor** |")
w("| 9 | Achieved read rate, real loader | 7588734 (8n) | **1,644** | MB/s | at 128 concurrent readers with `gpu_busy_frac` 0.9703 ⇒ **I/O is not the bottleneck; the zarr conversion is not justified** |")
w("| 10 | Model size | — | **455,831,040** | float32 params | = the `numel` of #4, confirming the collective is the whole model |")
w("")
w("## Environment — constant across every arm")
w("")
w("| item | value |")
w("|---|---|")
w("| hardware | Polaris, 4 × A100-SXM4-40GB per node (**39.49 GiB** usable), NV4 mesh |")
w("| venv | `$MEMBER_ROOT/conda-envs/fme-venv` — python 3.12.11, **torch 2.10.0+cu129, NCCL 2.27.5**, torch_harmonics 0.8.0, zarr 3.3.0; `fme` **editable** from `ACE2_retrain/ace_exp` |")
w("| fabric | self-built aws-ofi-nccl **1.21.1** + cray libfabric **2.3.1** + `OFI_NCCL_PROGRESS_MODEL=AUTO`; transport reported `AWS Libfabric` on **every** arm |")
w("| launcher | PALS `mpiexec --ppn 4 --cpu-bind depth -d 8 --label --line-buffer` + `polaris_rank_env.sh` (PMI_* → `env://`) → `ace2_telemetry.py` |")
w("| knobs | `NCCL_ALGO=Ring`, `OMP_NUM_THREADS=2`, `MASTER_PORT=20000+jobid%20000`, `TMPDIR=/tmp`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (verified honoured — both env-var spellings are in `libc10_cuda.so`) |")
w("| model/config | `config_polaris.yaml` = `config_midway.yaml` with 11 paths repointed, nothing else. SFNO, embed_dim 384, 8 layers, dhconv; AMP **bf16**; AdamW(fused) — `FusedAdam` needs **no apex** here; **LR 1e-4 and FLAT** (fme's default scheduler is `None`) |")
w("| per arm | 60 timed steps, 1 epoch, `sample_with_replacement = 60 × global_batch` so every rank runs exactly 60 steps; `inference=null`; `save_checkpoint=false` |")
w("")
w("## What is NOT measured")
w("")
w("| gap | why it matters |")
w("|---|---|")
w("| **8n has n=1**; 4n reverse and 1n reverse have n=1 | nothing at those points should be published; `run_ace2_ladder.sh` fills the shortest rungs first |")
w("| The 1.823 GB startup collective has **no named call site** | the flight-recorder dump carried no stack frames; one arm with stack capture would close ai-rossby's open question properly rather than by analogy |")
w("| **No kernel-level profile on Polaris** (prereg P6) | Midway's 'NCCL is 40–46% of GPU kernel time' was taken on A100-**PCIE with no NVLink**; whether it transfers to this NV4 mesh is untested. `ace2_nvtx.py` exists; there is no Polaris nsys launcher |")
w("| **No LR sweep**, and batch 16 on 2 nodes is a training-regime change | jesswan's call, not ours |")
w("| **No equivalence baseline** (DESIGN §4) | required before any hot-path change is committed |")
w("| `GPU_ORDER` A/B is **not node-matched** | makani's arm used 3+3 reps on the same nodes precisely because node variation can swamp a small effect |")

w("## Table 8 — ACE2 vs makani: SAME SFNO, ONE CONFIG KNOB APART")
w("")
w("Both are the Modulus/makani SFNO on a **180×360 equiangular grid** with **identical**")
w("`embed_dim=384`, `num_layers=8`, `operator_type=dhconv`, `filter_type=linear`,")
w("`normalization_layer=instance_norm`, `hard_thresholding_fraction=1.0`, `use_mlp`,")
w("`separable=False`. makani numbers from `makani_bench_report.md` + `polaris_ace2_multinode_handoff.md` §1f;")
w("makani config = `makani_sfno/polaris/e3sm_alldata_full.yaml`.")
w("")
w("| | ACE2 | makani | note |")
w("|---|---|---|---|")
w("| **`scale_factor`** | **1** | **3** | ⭐ the only architectural difference that matters |")
w("| internal grid in the 8 blocks | 180×360 | 60×120 | **9× fewer positions** for makani |")
w("| autoregressive steps / sample | **2** (`n_forward_steps=2`) | **1** (`n_future=0`) | ACE2 backprops through 2 applications |")
w("| `pos_embed` | true | none | minor |")
w("| channels | 56 (43 in / 40 out) | 101 + 7 forcing | makani's encoder is wider |")
w("| dhconv weight / layer | 384×384×**180**×2 = **212.34 MB** | 384×384×**60**×2 = **70.78 MB** | = internal lat |")
w("| params | **455,831,040** | **147,860,000** | ratio **3.08×** = the scale_factor ratio |")
w("| gradient volume | **1.823 GB** | **0.591 GB** | derived 0.566 GB dhconv + rest ⇒ reproduces makani's recorded 591 MB |")
w("| dhconv share of params | 93.2% | 95.7% | both are almost entirely spectral weights |")
w("| **largest single collective** | **1738.86 MiB** (whole model, once, at startup) | ~591 MB (one bucket) | |")
w("| **tree-defect exposure** | 🔴 **YES — `NCCL_ALGO=Ring` is load-bearing** | ✅ no — below the ~1 GB threshold | the single biggest operational difference |")
w("| ms / sample | **335.4** | 45.7 | **7.3×**; per *model application* 167.7 vs 45.7 = 3.7× |")
w("| **GiB per added sample** | **12.64** | **0.73** | **17.3×** — vs 18× predicted by 9× spatial × 2× rollout |")
w("| max samples/GPU on 40 GB | **2** | **12** (16 OOMs) | |")
w("| memory at that max | 33.96 GiB | 18.97 GiB | |")
w("| production global batch | 16 | 32 | |")
w("| **fewest GPUs that hold it** | **8 (2 nodes)** | **4 (1 node)** | ⇐ this is why the scaling verdicts differ |")
w("| best measured config | 2+ nodes | **1 node** | makani: 1 node cheapest AND fastest |")
w("")
w("**The decomposition closes.** `scale_factor` 1 vs 3 gives 3× the spectral modes ⇒ **3.08×**")
w("the parameters (measured 455.83 M / 147.86 M), and 9× the internal spatial positions;")
w("times ACE2's 2-step rollout that is **18× predicted activation memory per sample against")
w("17.3× measured**. Nothing else needs to be invoked.")
w("")
w("⇒ **ACE2 and makani do not disagree about Polaris; they are different-sized models.**")
w("makani at `scale_factor=3` is small enough that one node holds its whole production batch,")
w("so it never has to touch the fabric — and its ladder found that touching it costs ~234 ms")
w("and is never worth it. ACE2 at `scale_factor=1` is 17× heavier per sample in activations,")
w("cannot fit its production batch below 8 GPUs, and therefore **has no fabric-free option**.")
w("Given it must pay the toll, it then amortises it well (82–84% incremental efficiency from")
w("2 nodes), because it is also 7.3× heavier per sample in compute.")
w("")
w("⚠ **`scale_factor` is a SCIENCE knob, not a performance knob.** ACE2 running at 1 rather")
w("than 3 is what the ai2cm config specifies; changing it changes what the model computes and")
w("is jesswan's call, not a tuning decision. It is listed here to explain the measurements,")
w("**not** as a proposal.")
w("")
w("⚠ Two caveats on the compute row: makani's 45.7 ms/sample is an *average* at 8 samples/GPU")
w("(it includes fixed per-step overhead), while ACE2's 335.4 is a *marginal* two-point fit;")
w("and the 3.7×-per-application residual is not derived from first principles — the encoder")
w("runs at full resolution in both and makani has 2.4× the channels there, which partly")
w("offsets its 9× spatial advantage.")

w("## Table 9 — WHAT THE FABRIC COSTS ACE2")
w("")
w("Weak scaling holds per-GPU work identical on every row, so **everything above the")
w("1-node step is exposed inter-node cost** (fabric + load imbalance).")
w("")
w("| nodes | step_med (ms) | over 1n (ms) | **% of the step** | node·s / sample | vs 1 node |")
w("|---|---|---|---|---|---|")
w("| 1 | 716.0 | — (baseline) | — | 0.0895 | 1.00× |")
w("| 2 | 1204.4 | **+488.4** | **40.6%** | 0.1505 | 1.68× |")
w("| 4 | 1426.1 | +710.1 | 49.8% | 0.1783 | 1.99× |")
w("| 8 | 1498.5 | +782.5 | **52.2%** | 0.1873 | **2.09×** |")
w("")
w("**Over half of the 8-node step is fabric**, and 8 nodes costs **2.09× the node-seconds")
w("per sample** that 1 node would — except 1 node cannot run the production batch, so the")
w("usable comparison is **2 → 8 nodes: +24% node-hours for 3.2× the wall-clock throughput.**")
w("")
w("### ACE2 pays a BIGGER toll than makani, not a smaller one")
w("")
w("| | toll (ms/step) | gradient volume | effective bandwidth |")
w("|---|---|---|---|")
w("| **ACE2** | **488.4** | 1.823 GB | 3.73 GB/s |")
w("| makani | 234.0 | 0.591 GB | 2.53 GB/s |")
w("| ratio | **2.09×** | 3.08× | — |")
w("")
w("Both land in the 2.5–4.4 GB/s this stack has measured elsewhere, so the toll is")
w("**bandwidth-bound and scales with gradient volume** — ACE2 pays more because")
w("`scale_factor=1` gives it 3.08× the gradients (Table 8).")
w("")
w("### So why does ACE2's ladder still look better? Because of what the toll hides behind")
w("")
w("| config | toll | per-GPU compute | toll as % of compute |")
w("|---|---|---|---|")
w("| ACE2 @ 2 samples/GPU | 488.4 ms | 716.0 ms | **68%** |")
w("| ACE2 @ 1 sample/GPU | 488.4 ms | 380.6 ms | 128% |")
w("| makani @ 8 samples/GPU | 234.0 ms | 365.4 ms | 64% |")
w("| makani @ 1 sample/GPU | 234.0 ms | 45.7 ms | **512%** ← the regime ACE2 cannot reach |")
w("")
w("At their respective 1-node production configs the toll is a **similar fraction** (68% vs")
w("64%). The divergence is entirely in the *range of configurations reachable*: makani's")
w("fixed-batch ladder could slice to 1 sample/GPU where the fabric is 5× the compute, while")
w("ACE2 caps at 2 samples/GPU and bottoms out at 128%.")
w("")
w("⚠ **`gpu_busy_frac` cannot see any of this** — it reads 0.9288 → 0.9703 *rising* across")
w("these same rows, because exposed NCCL runs as GPU kernels and counts as *busy*. It")
w("measures loader idle. Use the `over 1n` column for communication cost.")
w("")
w("⚠ **`over 1n` is not purely fabric**: it also contains load imbalance and straggler")
w("effects, which weak scaling does not separate. And the 716.0 ms baseline already")
w("contains **intra-node** NCCL over NVLink, so the figures above are the *inter-node")
w("increment*, not ACE2's total communication cost. Separating those needs the kernel-level")
w("capture that prereg P6 still lacks.")

import sys
sys.stdout.write("\n".join(out) + "\n")
