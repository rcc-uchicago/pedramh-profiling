# PanguWeather on Polaris — profiling plan and to-do list

Written 2026-08-20. Companion to `POLARIS_PROFILING_HANDOFF.md` (the Midway/Delta
handoff this re-scopes for Polaris), `polaris_bench_report.md` (the existing A100
profile), and `PROFILING_TABLES.md`.

**The question this plan answers:** the Polaris profile reads as "saturated — 643 ms
of kernel time in a 603 ms step, 61% elementwise, loader 0.7%", which looks like
there is nothing left to find. That reading is **half right, and the half that is
wrong is where all the remaining headroom is.**

---

## 0. What is now measured, from captures already on disk

Derived today from `$MEMBER_ROOT/bench/nsys_pangu_sfno_7255503.sqlite` (job 7255503:
4× A100-SXM4-40GB, `sfno_plasim` E3SM, 1,182,108,160 params, batch 1/rank, bf16
autocast, `checkpointing: 3`, eager, 40 measured steps × 4 ranks). **No GPU time
spent.** This closes handoff Tier A items 3 and 5-adjacent for Polaris.

### 0a. Time occupancy really is maxed — this part of "we're at the maximum" is true

| device | kernel sum / span | **kernel union / span (GPU-busy)** |
|---|---|---|
| dev0 | 105.5% | **95.7%** |
| dev1 | 104.0% | **95.6%** |
| dev2 | 106.4% | **96.5%** |
| dev3 | 106.5% | **96.5%** |

`polaris_bench_report.md` §4.2 inferred saturation from `sum/span > 100%`, which is
an artifact of NCCL running on its own stream. The **union** confirms it anyway:
3.5–4.4% idle — and that is **kernels only**; counting memcpy and memset (GPU work on the
same stream, absent from the kernel table) idle is **1.4–1.5%**
(`polaris_bench_report.md` §4.3f), which makes this conclusion stronger. There is no idle to reclaim, no launch-latency story, and nothing for
CUDA Graphs — consistent with handoff §6 dead end 1, now on Polaris evidence.

### 0b. Communication is not the problem on Polaris **when the ranks are balanced** — and balance is not guaranteed

⚠ **REVISED 2026-08-20 by the item-2 n=2.** The original form of this section read
"comms are essentially free, deprioritize" off **one** capture. The identical config on a
**different node** swings the number 7×, so the deprioritization is conditional.

| quantity | job **7255503** (dev0…dev3) | job **7255557** (dev0…dev3) |
|---|---|---|
| NCCL kernel union | 2.67 / 2.34 / 2.87 / 2.98 s | **10.14 / 3.46 / 4.64 / 5.08 s** |
| overlapped with compute | **88.7 / 88.3 / 83.7 / 84.0%** | **79.0 / 62.4 / 54.4 / 54.0%** |
| **exposed (GPU running only NCCL)** | **1.2 / 1.1 / 1.9 / 2.0% of span** | **8.0 / 4.9 / 7.9 / 8.8% of span** |
| stalled steps (NCCL > 1.5× median) | **1 of 40** | **19 of 40** |
| compute union (the control) | 23.01 / 23.00 / 23.03 / 23.02 s | 23.52 / 23.02 / 23.06 / 23.04 s |

**Compute is invariant to 0.27% across 7 of 8 devices on two different nodes; NCCL is
not.** So: exposed comms is **1.1–2.0% of span on a balanced run and 4.9–8.8% on an
unbalanced one**, at identical config. Compare Midway's 35.7% exposed on `midway3-0423`.

⇒ **The `NCCL_PROTO`/`NCCL_ALGO` agenda (handoff §2, Tier B item 6) is still deprioritized
— but for a sharper reason: what varies is not the protocol, it is rank placement.** A
protocol change cannot recover time the GPU spends *waiting* for a late rank. The lever for
the unbalanced case is item 6b; the protocol only becomes interesting multi-node (item 12).
Reproduce: `nvtx_phase_attribution.py --per-rank`.

### 0c. The handoff's "largest single item on the project" does not transfer

`ncclDevKernel_Broadcast_RING_LL` **is present** — 160 launches = exactly 1 per
rank-step, confirming `broadcast_buffers=True` at `PanguWeather/v2.0/train.py:298-303`
— but it costs **112 ms of 102.9 s = 0.11% of GPU kernel time**, 0.7 ms/rank-step.
Midway's 33.14% was three ranks *waiting* on a straggler, not the broadcast itself.
**`broadcast_buffers=False` is a ~0.1% change on Polaris.** It still needs jesswan's
sign-off (BN buffers), so the cost/benefit no longer justifies opening that gate.

### 0d. Where the step actually goes — and the finding that reopens the ceiling

Top kernels by GPU time, all four ranks, full `demangledName`, 102.9 s total:

| kernel | % all GPU time | ms/rank-step | µs/call | est. GB/s | % of 1555 GB/s |
|---|---|---|---|---|---|
| `direct_copy_kernel_cuda` ⟨float⟩ | **18.9%** | 121 | 266 | 415 | **27%** |
| `direct_copy_kernel_cuda` ⟨complex64⟩ | **17.3%** | 111 | 1327 | 359 | **23%** |
| `conj_kernel_cuda` ⟨complex64⟩ | **6.0%** | 38 | 1596 | 257 | **17%** |
| `MulFunctor` ⟨complex64⟩ | 3.1% | 20 | 235 | 461 | 30% |
| `CUDAFunctor_add` ⟨bf16⟩ *(vectorized path)* | 2.2% | 14 | 219 | **810** | **52%** |
| `FusedAdamMathFunctor` | 4.0% | 26 | 455 | — | — |
| `cutlass_80_tensorop_c1688gemm_64x64_16x4_nt_align1` | 3.9% | 25 | 696 | — | — |
| `cudnn::bn_fw_tr_1C11_kernel_NCHW` ⟨fp32⟩ | 2.5% | 16 | 329 | — | — |

**271 ms/rank-step — 47% of compute time — is `direct_copy` + `conj`: kernels that
perform zero arithmetic.** They move bytes and nothing else.

⚠ **Quote it that way, not as "42.2% of all GPU kernel time."** Item 2 measured the same
quantity on an identical config and got **42.2% / 37.4%** — a **4.77-point** move with a
numerator that moved **0.09%**. A share of the full kernel total is not reproducible
because NCCL *wait* sits in the denominator. The absolute (271 ms/rank-step, reproduced to
0.09%) and the share-of-compute (47.1% / 46.8%) are the durable forms.
→ `polaris_bench_report.md` §4.4b.

And they move them badly. **≈97%** of that time runs on the **non-vectorized**
`elementwise_kernel<128,2>` / `gpu_kernel_impl_nocast` path (the TensorIterator
fallback taken for non-contiguous or awkwardly-strided operands) at **17–27% of A100
HBM2e peak**, while a *vectorized* bf16 add in the same capture reaches **52%**. (Not
100%: a vectorized `conj`⟨complex64⟩ and an unrolled/cast `direct_copy`⟨float⟩ account for
the other **2.7%** — §4.3c. Better still, item 1 found a **measured** ceiling in the same
capture: D2D memcpy above L2 sustains **1279 GB/s = 82% of peak**, so the hardware plainly
delivers and the 17–27% is about the path these kernels take. That narrows item 7 without
replacing it.)

> **⇒ We are at maximum GPU *time* occupancy and nowhere near maximum *bandwidth*.**
> The ceiling we are hitting is our own data movement, not the A100. That is why the
> profile "looks maxed" and why it is unclear — 96% busy and 25% of peak bandwidth are
> both true at once.

**Method, and its OPEN caveat.** Bytes are estimated as
`grid × block × elements_per_thread × sizeof(dtype) × 2` (read+write) from the CUPTI
launch geometry and the kernel's template parameters. That is *useful* bytes. If the
access pattern is uncoalesced, real DRAM traffic is **higher**, so achieved DRAM
bandwidth would be closer to peak and the defect would be *wasted* traffic rather than
unused bandwidth. **Both diagnoses point at the same class of fix (move fewer, better
laid-out bytes) but at different mechanisms**, so do not quote "25% of peak" as a
measurement until item 7 runs. It is an estimate with a stated method.

### 0e. The capture does not match what production runs

Job 7255503 is `checkpointing: 3`, EMA never fired (`ema_warmup_epochs: 6`), warmup
20, no ZeRO. **Pangu production launched at `checkpointing: 2`, ZeRO OFF** (CHANGELOG
2026-08-07, jobs 7366939→7366940). On the same-model ai-rossby sweep (job 7365119, one
node, back-to-back) `ckpt3 → ckpt2` is **1.274×** — so the profiled config is ~27%
slower than the one we ship, and the recompute traffic that dominates §0d is exactly
what `checkpointing` changes. **Every percentage in §0d is a `ckpt3` percentage.**

---

## 1. Tier 0 — free. No GPU, no queue. Do these first.

- [x] **1. DONE (2026-08-20)** — `polaris_bench_report.md` **§4.3**,
      `ACE2_retrain/nvtx_phase_attribution.py` (+ passing test, no GPU needed).
      **Text path:** the house ranges are in the inline `NVTX_EVENTS.text` column
      (`domainId 0`, `eventType 59`), 160 rows each = 40 steps × 4 ranks; the
      `textId → StringIds` path holds **only** NCCL's registered strings (all-rank
      `ncclAllReduce` 2402, `ncclBroadcast` 160 — the ×600/×40 above were per-rank).
      Nothing of ours was ever missing. **Join:** the `globalPid` guard is
      `k.globalPid = (r.globalTid & ~0xFFFFFF)`; unguarded it inflates **+29.4%** here.
      A second bug mattered as much — attribution must be scoped to the **process**, not
      the launching thread, or `backward` reads as `(outside)`.
      **Result:** copy time is `backward` **72.9%** (197.58 ms/rank-step) /
      `forward_loss` **27.1%** (73.61); `(outside)` **0.0%**.
      **⚠ It does NOT isolate recompute** — the split cannot see inside `backward`; that
      needs item **16**. What it gives is an estimate: recompute ≈ 148–152 ms/rank-step
      (measurably ~1.4% *more* than the forward, so not a bound), ⇒ ckpt-off ≈ 1.34×,
      of which production at `ckpt2` has already banked most (residual ≈ 4%).
- [x] **2. DONE (2026-08-20)** — `polaris_bench_report.md` **§4.4**. Prereg `952fcb8d`,
      **5/5 hit**. The config does **not** differ (identical per-kernel launch counts,
      byte-identical D2D volumes), and the two jobs ran on **different nodes**
      (`x3001c0s19b0n0` vs `x3001c0s1b1n0`, disjoint GPU UUIDs) — so this is a
      node-to-node n=2, not just run-to-run.
      **The finding is bigger than the gate asked for:** compute is reproducible
      (**+0.63%**; copy time **−0.09%**; the backward/forward split **−0.03 pt**) but
      **NCCL is not (+114.8%)** — so **a share of the full GPU-kernel total is not a
      reproducible quantity** (copies: 42.2% → 37.4%, **−4.77 pt**, numerator −0.09%).
      Quote absolute ms/rank-step or share-of-compute. §0b and §0d revised above.
      ⇒ This also resolves the "a cross-JOB ratio on Polaris is not a measurement" rule
      into something sharper: **cross-job *compute* comparisons are fine (0.27% across 8
      devices, 2 nodes); cross-job comparisons of anything containing NCCL are not.**
- [x] **3. DONE (2026-08-20)** — `polaris_bench_report.md` **§4.5**,
      `ACE2_retrain/sfno_bytes_model.py` (+ test). Prereg `45cbd7de`: **0/4 size
      predictions hit**, and per this item's own wording the mismatch *is* the finding.
      **What the copies move:** the `dhconv` spectral weight is `[in, out, modes_lat]`
      complex = **512×512×180 = 377.49 MB/layer**, and 12 layers are **95.8%** of the
      1,182,108,160 params — the largest tensor in the model, 1.42× the largest
      activation. Nine kernels covering **~99.6%** of copy time match an analytic tensor.
      **The finding:** **133.15 ms/rank-step** moves that weight in **four** places —
      forward permute (35.60), its `ckpt3` recompute (35.61), the adjoint `conj` (35.93),
      and **`grad_w` → DDP bucket (26.01, not invariant)** — all traceable to one layout
      mismatch: the parameter is stored `(in, out, lmax)` but `_contract_dhconv` is
      `einsum("bixy,iox->boxy")`, which must permute it to `(x, i, o)` for `bmm`.
      **Invariant movement is 107.14 ms = 17.8% of the step.** `num_blocks: 16` is
      vestigial and does **not** block-diagonalise this weight.
      **⚠ Scope:** weight:activation is `E/(B·mmax)` — 2.83 at batch 1, **0.71 at batch 4**
      (the base config's commented default), where the weight share of copy time falls to
      ~19%. **⚠ OPEN, blocks every mechanism claim:** with `factorization: None`,
      `use_tensorly=False` lands on `assert factorization == "ComplexDense"`
      (`s2convolutions.py:151`), which must fail — yet both jobs ran. Until that is
      resolved we do not know whether the weight is an `nn.Parameter` or a
      `FactorizedTensor`. No *measured* number depends on it.
- [x] **4. DONE (2026-08-20)** — `ACE2_retrain/kernel_census.py` rewritten to **delegate**
      attribution to `nvtx_phase_attribution.py` instead of re-deriving it (+ its own test).
      Prereg `a336c6fc`, **4/4 hit**. It had **three** bugs, not two: the missing `globalPid`
      guard (**+29.4%** phantom rows, and **+31%** on time), the thread-scoped range lookup —
      which reported **`backward` as 203 launches and 0.0% of GPU time** where the truth is
      250,880 and 72.6%, and is the true origin of the `(outside) 69.6%` row several docs had
      recorded as an NVTX limitation — and a normaliser that counted distinct `step_%` starts
      (**156**) instead of **160 rank-steps**. It now agrees with §4.3b row for row.
      **⇒ Batching is retired — but NOT for the reason first published, and the correction
      matters.** The first version said the phase-level skew was small (+3.2 pt / +2.0 pt vs a
      +10 pt bar) and concluded "no launch-pipeline headroom". **That sentence was false about
      the capture**, and an adversarial pass caught it: there ARE **73,251 kernels under 10 µs
      = 20.7% of all launches** (458/rank-step, none of them NCCL), skewing **+20.4 pt** —
      twice the bar. The phase metric could never have seen them: `skew_r = pc_r·(1 −
      avg_r/avg_all)` is a *count-weighted relative* mean-duration test, so a range holding
      2.7% of launches cannot skew past 2.7 pt however tiny its kernels are, and this harness
      emits only **3** non-empty phase ranges. **The right reason to retire batching is the
      prize and the queue:** fusing **all** 73,251 recovers at most **0.27% of GPU time**
      (0.280 s of 102.9 s) against a launch queue **220 ms deep** (median launch→execute; n=2
      227 ms), i.e. the CPU runs ~⅓ of a step ahead and an ~8 µs launch call cannot starve it.
      Also corrected: the "~9% idle is launch latency" reading was **already** retired on its
      own turf by handoff §6 dead ends 1–2 (stream/DDP-bucket dependency) — this capture is
      *consistent with* that and does **not** refute the ACE2/Midway number, which is a
      different model on different hardware.
- [x] **5. DONE (2026-08-20)** — `polaris_bench_report.md` §4.3g, from
      `nvtx_phase_attribution.py --per-step`. **Warmup 20 was enough: there is no warmup
      regime.** First measured step **640.26 ms** vs a **634.36 ms** median (+0.9%), and
      `forward_loss` GPU time spans only 150.20–150.52 ms across all 40 steps.
      **Confirmed on n=2** by item 2 (7255557's step 0 is +6.7% on the total but only
      **+0.5% on compute** — judge warmup on **compute**, never the total).
      **But one step is contaminated for a different reason:** step index **30** — the same
      training iteration in *both* captures — costs ~600 ms of NCCL against a ~59 ms norm at
      identical launch counts. Item 2 identified the cause from CPU sampling: **a CPython
      generation-2 garbage collection** (`gc_collect_main` 116/88/88 samples; nothing else
      in either capture exceeds 5), which is why it lands on the same iteration on two
      different nodes. Excluding it, NCCL **67.82 → 59.67 ms/rank-step (−12.0%)** while
      phase shares move ≤0.4 points; **§4.2's NCCL row carries the stall.**

> ### ✅ UNBLOCKED 2026-08-21 — but on a DIFFERENT torch, and that is load-bearing
> The base conda that produced every number in §0 (torch **2.8.0**) is orphaned by a Cray PE
> migration: its torch and h5py link `*_gnu_123.*` sonames the roll removed, and hdf5 moved
> soversion **200 → 310** — an ABI break, not a rename. It is not coming back in that form.
> **Work now runs in the ai-rossby venv (torch 2.10.0+cu129)** plus `$PANGU_SHIM`, which holds
> only the five packages that venv lacks (`cartopy`, `natsort`, `pyproj`, `shapely`, `pyshp`) —
> a shim with only those cannot shadow the venv's `torch`/`torch_harmonics`, which putting the
> whole top-ups on `PYTHONPATH` would. Job **7541487** confirmed Pangu's full import set green.
> **Recipe:** `polaris_rebaseline_nsys.pbs`.
>
> **Consequence for these items, and do not blur it:** items **6b, 7, 8, 12** ask mechanism
> questions that are not torch-version-sensitive (host-side stalls, per-kernel access patterns,
> source call sites, comms scaling) and can proceed directly. Items **9, 10** produce *ratios*
> against §0d and therefore need a **re-baseline capture on this env first**. **Any table must
> state which torch it used** — §4.4c already showed how fast incomparable numbers get tabled
> together.
>
> ⚠ Submission authority changed 2026-08-20: single-node `debug` jobs are **auto**; `capacity`,
> `preemptable` and multi-node still stop and ask. The heading below predates that.

## 2. Tier 1 — cheap `debug`-queue jobs (≤1 h). Ask before submitting.

- [x] **6. DONE (2026-08-20) — job 7533457, `TOPO_OK`.** Prereg `5063d221`: **5/5 hit.**
      **Polaris is a full NVLink mesh: `NV4` between every pair, measured 82.9–83.1 GB/s
      uniform (spread 0.24%)** on 4× A100-SXM4-40GB. No 2×2 block structure, no PCIe-class
      pair. ⇒ handoff §4's OPEN topology cell is **closed with a measurement**, and §0b's
      "comms are free inside a node when balanced" now has a mechanism rather than an
      inference. Sharp corroboration of the §4.4 method fix: the *minimum*-NCCL anchor
      implied ≥79 GB/s against a measured **83.0** (within **5%**), so on the balanced
      capture the all-reduce runs at essentially link speed — whereas the stall-carrying
      *mean* would have implied ~32 GB/s and "found" a PCIe hop that does not exist.
      **Bonus (item 6b's key input):** `nvidia-smi`'s own CPU-Affinity column independently
      confirms the **reversed** GPU→NUMA map — GPU0→NUMA 3 (cores 24-31,56-63), GPU1→2,
      GPU2→1, GPU3→0 — matching the sysfs reading from job 7531456. See
      `polaris_pbs_notes.md` §1.
      **Cost: 4 attempts, 3 of them infra/self-inflicted** — see the journal; the blocker
      was a cluster-side conda breakage, not this item.
- [ ] **6b. Two host-CPU stalls, one of which is already diagnosed. ADDED 2026-08-20 by
      item 2 (§4.4e); not in the frozen list.**
      **(A) ✅ CLOSED 2026-08-21 with a RECORDED NULL — GC is not the cause.** Tested
      directly on torch 2.10 (jobs 7549941, 7550007): interleaved A/B/A/B with a throwaway
      warm-up arm, `gc.disable()` injected via `sitecustomize.py` on `PYTHONPATH` so both
      arms ran byte-identical code. **Turning the collector off changes nothing measurable**
      — `step_std` B/A = **1.0102**, `step_p90` B/A = 0.9981, peak memory identical to three
      decimals. **The stall is a FIRST-RUN effect:** it sat in the first arm of 7549941
      (`step_std` 0.0770) and *moved to the throwaway arm* in 7550007 (0.0610, **65×** the
      later arms), i.e. it follows the first arm, not the GC setting. ⇒ **the `gc.freeze()`
      lever is withdrawn** (§4.4e retracted), and the "recurring multi-hundred-ms hit on a
      100-epoch run" framing is wrong — a first-run cost is paid **once per job**.
      **The first A/B nearly produced a false positive** and only the interleaving caught it:
      on aggregates it read as −12% p90 / −97% std, all of which was the first arm.
      ⚠ **This closes only the first-arm stall.** Job 7255557's pattern — **19 of 40** steps
      stalling mid-run with dev0 out of phase — is *not* a first-arm effect and stays open
      under (B).
      **(B) NOT diagnosed — the other stall pattern.** On 16 of 7255557's 17 stalled steps
      **dev0 alone waits ~600 ms** while the other three sit at 60–70 ms (dev0 out of phase
      with the group, not one rank straggling), with the late work in the **inter-step gap**
      and `_PyEval_EvalFrameDefault`/`__libc_malloc` frames. **Candidate, unestablished** —
      an A/B, one `debug` job, interleaved. Arms: (i) as-shipped — `OMP_NUM_THREADS=8`
      (`PanguWeather/v2.0/HPC_scripts/polaris_bench_nsys_e3sm_sfno.pbs:51`), bare
      `torchrun`, **no CPU binding at all**; (ii) `OMP_NUM_THREADS=1` (one line, zero
      risk — what torchrun pins when the var is unset); (iii) `mpiexec --cpu-bind depth
      -d 8`. Rationale: 4 unbound ranks × 8 OpenMP threads = 32 threads on 32 physical
      cores, *plus* 4 main threads and the loader workers.
      **Verdict on per-rank NCCL spread and stalled-step count (`--per-rank`), NOT on step
      time** — step time is too noisy to resolve this (item 10's method note).
      **Output-neutral:** a launcher change, so it is outside the DESIGN §4 equivalence
      gate and needs no jesswan sign-off.
      **Cheaper first:** `ACE2_retrain/PROFILING_PLAN.md:224` already scopes the *direct*
      straggler test — `TORCH_NCCL_TRACE_BUFFER_SIZE=20000 TORCH_NCCL_ENABLE_TIMING=1` +
      `_dump_nccl_trace_json`, which times the same collective on all four ranks without
      an nsys capture at all.
- [ ] **7. ⭐ ncu on the top six kernels, single rank (`--nproc_per_node=1`), explicit
      metric list — this is the measurement that settles "are we at the maximum".**
      Metrics: `dram__bytes_read.sum`, `dram__bytes_write.sum`,
      `dram__throughput.avg.pct_of_peak_sustained_elapsed`,
      `sm__throughput.avg.pct_of_peak_sustained_elapsed`,
      `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` (sectors/request ⇒ coalescing),
      **Name the kernels rather than "top six" (§4.5b):** `direct_copy`⟨float,nocast⟩
      @66.72 MB (42.7%), `direct_copy`⟨complex64,nocast⟩ @377.49 MB (35.8%),
      `conj`⟨complex64,nocast⟩ @377.49 MB (13.2%), `direct_copy`⟨complex64,nocast⟩
      @133.45 MB (5.3%). **The two 377 MB rows are the same tensor**, so one ncu pass
      answers both — and `sectors/request` on a permuted `(E,E,lmax)` operand is the direct
      test of §4.5c's mechanism.
      `launch__occupancy_limit_*`. **Never under real DDP** — kernel replay re-executes
      `ncclDevKernel`, which spins on peer flags → deadlock (handoff Tier C / dead end
      5). Not `--set full`. **Decision it unblocks:** if the copies are at ~90% of DRAM
      peak, the only lever is *fewer* bytes (fusion, dtype, not materialising); if they
      are at ~25% with poor sectors/request, the lever is *contiguity* — a layout fix,
      possibly with no change to what is computed.
- [ ] **8. Attribute `direct_copy`/`conj` to source lines.** `torch.profiler` with
      `with_stack=True`, or `emit_nvtx` for autograd-op names (timings void by design —
      shares only, per handoff's ACE2 precedent, where `aten::clone` was 45.3% of copy
      time). **Deliverable:** the three or four call sites responsible for 42% of GPU
      time. Without this, "fuse the elementwise work" has no target.
      **Scope, from item 1 (§4.3c):** only **27.1%** of the copy time is in
      `forward_loss` — **72.9% is in `backward`**, and `conj` (14.1%) has **zero** forward
      launches. So a forward-only `with_stack` census can reach at most a quarter of the
      target; budget for autograd-node attribution (`emit_nvtx`) or the recompute path.
      **Targets, from item 3 (§4.5), in priority order:** (1) the **42.7%** row —
      `direct_copy`⟨float,nocast⟩, 348 calls/rank-step at 66.72 MB = one part of the
      complex spectral field — is the **largest single kernel and the only dominant one
      with no mechanism at all** (348 does not factor cleanly over 12 layers); (2) the
      weight permutation behind the 35.8% + 13.2% rows, where the question is *which* call
      permutes and whether the parameter can be stored `(lmax, in, out)`; (3) the
      **`assert factorization == "ComplexDense"`** contradiction above — a source question,
      free to answer, and it gates the mechanism. **Note the target for the weight rows is
      not fusion, it is hoisting.**
- [ ] **9. Re-capture nsys at the production config** — `checkpointing: 2`, ZeRO as
      shipped, **warmup ≥ 40**, and long enough that **EMA is active**. Everything in
      §0 is `ckpt3` with EMA never fired; production pays an every-step sweep over
      1.18 B params that no capture in this repo contains. `ema` is already
      instrumented and waiting.
- [ ] **10. Confirm the `ckpt` ladder in Pangu's own harness, one job, interleaved
      A/B/A/B.** The 1.274×/1.307× ladder is ai-rossby's trainer on the identical model
      shape, not Pangu's. Same model ≠ same harness, and cross-job Polaris ratios are
      not measurements (CHANGELOG 2026-08-06). Watch `peak_mem_gb_max_rank`: Pangu
      benched **26.98 GB** at `ckpt3` where ai-rossby benched 21.40 GB, so Pangu's
      `ckpt2` headroom is ~5.6 GB tighter than the sweep suggests, and the ZeRO sweep
      showed these estimates run 1–2 GB optimistic.
      **Prereg, from item 1 (§4.3d) — write these into the script before it runs:**
      Pangu's `ckpt3 → ckpt0` should be **≈1.34×**, and materially above that falsifies
      either the recompute estimate or the phase attribution; if `ckpt3 → ckpt2` is
      ~1.274× then `ckpt2 → ckpt0` must be **≈1.045×**. Note ai-rossby's *full* ladder
      (1.307× = 23.5% of the step) already sits within **1.8 points** of the estimate, so
      there is very little slack. Also: because the levels are **cumulative**, a
      `ckpt3 → ckpt2` delta measures **block-minus-MLP** recompute, not "blocks" — the
      MLP, encoder and decoder stay checkpointed at `ckpt2`. And `ckpt0` may be
      **unreachable**: it projects to ~41.7 GB > 40 GB for Pangu.
      **⚠ METHOD, from item 2 — read before writing the script.** The predicted effect
      (`ckpt2 → ckpt0` ≈ **1.045×**, i.e. 4.5%) is **smaller than the same-config
      run-to-run spread of total step time**, which item 2 measured at **+12.7%**
      (7255557's own CSV has `step_mean/step_med` = 1.149×). Interleaving A/B/A/B does not
      fix this on its own, because a comms stall lands in whichever arm draws it.
      ⇒ **Measure on the compute-only median** (`nvtx_phase_attribution.py --per-step`
      reports it), report per-arm NCCL ms/rank-step separately, and **reject any arm whose
      per-rank NCCL is imbalanced > 1.5×** (`--per-rank`). A ratio taken on mean total
      step time cannot resolve 4.5% on this cluster.
- [ ] **11. Verify the EMA/`ckpt2` memory margin for the live production config.**
      `ckpt2` no-ZeRO OOM'd for ai-rossby, EMA adds ~4.4 GB, and Pangu's EMA switches
      on at epoch 6. Read `peak_mem_gb_max_rank` from the production logs across the
      epoch-6 boundary. Free if the logs exist; a smoke if they do not.

## 3. Tier 2 — the axes nobody has profiled at all. This is where "the maximum" is most likely false.

- [ ] **12. ⭐ Multi-node scaling. ⚠ Fix rank placement FIRST (item 6b) or this measures
      the wrong thing.** Item 2 found a same-local-rank straggler already present **on one
      node** and on **two different nodes** (§4.4d), invisible in the kernel table except
      as the other ranks' wait. An unbound straggler at 1 node becomes a 4-node scaling
      loss that reads as a Slingshot problem. **Record per-rank NCCL at every node count**
      (`--per-rank`), not just the aggregate. Every Polaris number in this repo is single-node
      4× A100. §0b's "comms are free" holds *inside* a node; across Slingshot 11 it
      will not, and NCCL's 11% becomes the term that decides whether 100 epochs is
      reachable at all (current arithmetic: 275–325 h single-node ⇒ 4–5 chained
      `preemptable` links). `physicsnemo_ai_rossby/polaris/polaris_sfno_e3sm_multinode.pbs`
      is sitting untracked in the working tree — the scaffolding exists. Profile
      1→2→4 nodes: exposed/overlapped NCCL, samples/s/rank, and *then* the
      `NCCL_PROTO`/`NCCL_ALGO` sweep (§0b defers it to exactly here), verified by
      kernel name, ≥3 interleaved reps.
- [ ] **13. ⭐ A whole-epoch profile, not an 80-step window.** The bench window is
      known to understate production by ~9% (EMA + metric reduction + logging), and
      the bench-vs-production gap was already traced to **cold page cache**
      (CHANGELOG 2026-08-07). Nothing profiled covers: first-epoch I/O, validation
      (129 ICs × 60-step rollouts), checkpoint write, or the epoch boundary. At
      2.55 h/epoch these are the terms that actually set time-to-model.
- [ ] **14. ⭐ Memory profile — `torch.cuda.memory._record_memory_history` + snapshot.**
      Memory is the *binding constraint on the only lever that pays*: `batch_size` 2
      OOMs at every fast `ckpt` setting, `ddp_static_graph` OOMs, and ZeRO-1 was
      required to reach `ckpt2` at all. Nobody has asked **what holds the 36 GB**. If a
      few GB are recoverable, batch 2 unlocks and the whole §0d picture changes (the
      handoff's own lesson: batch size moves these numbers more than the interconnect
      does). The OOMs are genuine capacity, not fragmentation — 38.29 GiB allocated
      vs 57.76 MiB reserved-unallocated — so this is an accounting question, not an
      allocator-flag question.
- [ ] **15. Quantify the fp32-complex64 spectral island.** `direct_copy`⟨complex64⟩ +
      `conj`⟨complex64⟩ + `MulFunctor`⟨complex64⟩ + `c1688gemm_..._align1` = **30.3%**
      of GPU time, all in the complex64 SHT/spectral path, and the cutlass kernel is
      `align1` — unvectorized loads on a 64×64 tile. **Alignment is not precision:**
      making that GEMM `align4`/`align8` changes no arithmetic and needs no science
      sign-off, whereas the fp32 island around the SHT is deliberate and must not be
      pushed to bf16 (`si/bench_midway_notes.md` §3–4). Separate the two before
      proposing anything.
- [ ] **16. Add SFNO-internal NVTX ranges** — `spectral_filter`, `sfno_block`,
      `sfno_mlp`, `sht_fwd`/`sht_inv`. Pangu emits the shared phase names only, so
      §0d's 42% cannot be attributed to a layer. Use an injector on the
      `ACE2_retrain/ace2_nvtx.py` pattern — **do not edit the trees**: PanguWeather is
      a fork (no propagation) and `physicsnemo_ai_rossby` is a subtree (conflicts on
      pull). When extending `parse_nsys.py`, **edit both range lists** (SQL line ~83
      and the print loop ~line 100) or hoist them to one constant; the drift is still
      live (`unstack`).
- [ ] **17. `nsys --python-sampling=true`** on the step. NVTX `backward` is 280 ms of
      *CPU enqueue* in a 603 ms step; with the GPU 96% busy that is not a bottleneck,
      but it is also completely unattributed, and it is one flag.

## 4. Tier 3 — gated. List, do not run yet.

- [ ] **18. Capture the missing PanguWeather equivalence baseline.** `baselines/` holds
      only `ai_rossby_pangu_plasim/` and `ai_rossby_sfno/`. **There is no Pangu
      baseline, so the DESIGN §4.1 gate cannot be closed for any Pangu hot-path
      change** — every optimization above is blocked behind this one item, including
      `torch.compile` (§5 rung 1) and the `FourierNeuralOperatorBlock` fill removal.
      Fixed seed, world size 1, K=20 loss trajectory + output stats. Tolerance floors:
      2.5e-7 same GPU/node, ~1e-5 cross-architecture. **Highest-leverage single job in
      this plan** — it is the gate, not an optimization.
- [ ] **19. `torch.compile` (rung 1)** — the wired-but-never-exercised knob
      (`TORCH_COMPILE_MODE`). Right lever for §0d's fusion-starved copies, and ACE2's
      evidence says expect the *regional* win (corrector ≈ −2%) not a whole-network
      one (`InductorError: KeyError: 'complex64'` on the full SFNO — likely to bite
      here too, this model's hot path *is* complex64). After item 18.
- [ ] **20. `broadcast_buffers=False`** — **deprioritized on Polaris evidence**: 0.11%
      here (§0c), not the 33% the handoff estimated from Midway. Needs jesswan's
      sign-off for BN buffers; not worth opening that gate for 0.1%.

## 5. Confirmed dead on Polaris — do not re-spend time

From the same-model ai-rossby sweep (job 7365119, one node, back-to-back) and §0:

| knob | verdict |
|---|---|
| `num_data_workers` 1→8 | **0.991× — slightly negative.** `data_idle_frac` is 0.0068. The old "+9%" was PanguPlasim at 449 ms/step and does **not** transfer to a 1.18 B SFNO. |
| `ddp_static_graph` | **CUDA OOM** at `ckpt1`/`ckpt0`; the `sfno_plasim.yaml` comment calling it "safe to enable here" is wrong at low checkpointing. |
| `batch_size` > 1 | **CUDA OOM** at every fast setting — reopens only via item 14. |
| CUDA Graphs / launch-latency work | GPU-busy is 95.7% (§0a). No launch bottleneck exists. |
| Grid-size tuning, `channels_last` | handoff §6 dead ends 3–4; `torch_harmonics` wants NCHW. |
| single-node `NCCL_PROTO`/`NCCL_ALGO` | ≤1.2% available (§0b). Defer to item 12. |

---

## Method rules this plan inherits

1. **One node, one job, interleaved A/B/A/B** — a cross-job Polaris ratio is not a
   measurement (node-to-node spread is 10.5%, the same order as the effects we chase).
2. **`globalPid`-guard every correlationId join**, or kernels land on the wrong rank.
3. **Bucket by `demangledName`** with an explicit pattern list and an
   `(unclassified)` bucket gated at ~2%. (`nvjet_*` is sm90-only ⇒ 0.000% on A100, so
   the Hopper blind spot cannot reach Polaris — but the discipline still applies.)
4. **Verify env-var arms by kernel name**, not by the variable being set.
5. **No optimization lands without a passing smoke and an equivalence check** — which
   for Pangu means item 18 first.
