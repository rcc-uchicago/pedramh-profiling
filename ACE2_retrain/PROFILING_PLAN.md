# ACE2 instrumentation plan — ranked, with what the existing captures already prove

*Written 2026-08-19 against `outs/midway_nsys_53534648/n.sqlite` (4× H100 NVL, batch 4, 30 NVTX-bounded steps, median step 346.5 ms) and the val-probe logs. Everything in §0 was measured while writing this, at zero GPU cost — no new job. Ranked plan starts at §4.*

---

## §0 — Measured now from data already on disk

Nine numbers, none of which required a job. Several contradict entries above.

| quantity | value | source |
|---|---|---|
| dev0 span / union-busy (`gpu_busy_frac`) | 10.914 s / **81.0%** | union of kernel intervals, dev0 |
| NCCL busy on dev0 | 5.290 s = 48.5% of span | |
| non-NCCL compute busy | 4.942 s = 45.3% | |
| NCCL∩compute | 1.391 s → **only 26.3% of NCCL overlaps compute** | |
| **exposed NCCL** | **3.899 s = 35.7% of wall = ~130 ms/step** | |
| `cudaStreamSynchronize` | 7,549 calls / **20.34 s CPU** = #1 runtime API (vs 3.22 s in `cudaLaunchKernel`) | `CUPTI_ACTIVITY_KIND_RUNTIME` |
| ...inside `step_N` | 840 = **7.0/step, 163.4 ms of a 346 ms step** | |
| **kernel queue time**, dev0 | **median 8.87 ms**, p90 20.4 ms, 77% of launches wait >1 ms | RUNTIME→KERNEL, `globalPid`-guarded |
| `cudaLaunchKernel` CPU, dev0 | 103,497 calls, **0.820 s total = 7.5% of window** | |
| non-contiguous `elementwise_kernel<128,N>` | **7.273 s = 17.5%** of 41.575 s, n=136,560 | `demangledName` |
| contiguous `vectorized_elementwise_kernel` | 3.193 s = 7.7% | |
| NCCL kernel, 4×H100 | `ncclDevKernel_AllReduce_Sum_f32_**RING_LL**`, **gridX=6, blockX=288** (6 of 132 SMs) | |
| all-reduces per step per rank | **11.4** (342 `ncclAllReduce` NVTX / 30 steps), not ~70 | |
| effective all-reduce bandwidth | 2.735 GB egress / 176.3 ms = **15.5 GB/s** (vs 18.3 GB/s measured P2P) | |
| NVTX rows our SQL discards | **1,392**, of which **1,368 are NCCL's own `ncclAllReduce`** (domainId=1, registered string) | |
| validation, warm epoch-2 arm A | loop 5.25 s + **snapshot 17.13 s + mean_map 11.13 s** = images are **82.6%** | `outs/midway_valprobe_b4_53524580/out.log:141-148` |
| validation, arm E (`log_snapshots=false`) | still pays **10.81 s** of mean_map render | `..._53524752/out.log:143-146` |

### The one-step timeline (step 20, rank on dev0), everything ≥1 ms

```
   0.00 ms  step_20                       346.9 ms
   3.12 ms  forward_loss                   26.7 ms
  34.21 ms  cudaStreamSynchronize           3.51 ms   <- torch.isnan(loss)
  39.40 ms  forward_loss                   28.5 ms
  72.30 ms  cudaStreamSynchronize           3.39 ms   <- torch.isnan(loss)
  77.11 ms  cudaStreamSynchronize x3       ~0.008 ms  <- torch.any(regularizer_loss>0)
  77.26 ms  backward                      102.1 ms
 179.43 ms  optimizer                       2.61 ms
 182.28 ms  cudaMemcpyAsync (D2H)           0.019 ms
 182.30 ms  cudaStreamSynchronize         157.65 ms   <- 45% of the step
```

During that 157.6 ms sync the GPU is **not idle**: union-busy = 157.9 / 158 ms, of which **151 ms is 10 `ncclDevKernel_AllReduce` kernels**. The CPU is blocked waiting for an all-reduce that could not fit inside backward (11 × ~16 ms = 176 ms of comm vs 102 ms of backward compute).

---

## §1 — (a) Existing conclusions NOT supported by the evidence

| claim, as recorded | status | what settles it |
|---|---|---|
| *"The 9% idle during training is **launch latency**, not a stall… ~2,900 tiny kernels per step cannot keep the launch queue ahead of the GPU"* (notes §"Windowed re-capture"; `kernel_census.py:1-14` docstring) | **REFUTED, from data on disk.** Median launch→execute queue time is **8.87 ms**, p90 20.4 ms; 97% of launches have positive queue time. The CPU runs thousands of kernels ahead. Total CPU time in `cudaLaunchKernel` on dev0 is 0.820 s = 7.5% of a 10.9 s window, against 5.11 s/rank blocked in `cudaStreamSynchronize`. | Already settled. Cross-check for free with `nsys stats --report cuda_kern_exec_sum:nvtx-name` on the existing `.nsys-rep` (report exists: `.../nsight-systems-2024.5.1/target-linux-x64/reports/cuda_kern_exec_sum.py`). |
| *"NCCL kernel time is not comm cost… treat it as an **upper bound**"* — caveat never resolved | **RESOLVED and it is worse than the caveat implied.** 74% of NCCL time is *exposed*, not spin-wait: per-device total kernel time is 10.232/10.453/10.446/10.444 s (<2% rank imbalance), so it is not straggler wait. 35.7% of wall-clock is all-reduce with an otherwise-idle SM array. | Already settled. Confirm by a second method with `ddp._get_ddp_logging_data()` (`avg_backward_comm_time` − `avg_backward_compute_comm_overlap_time`). |
| *"GPU occupancy: training **91%**"* (`PROFILING_TABLES.md:35-40`) | **Mislabelled, and the number is from a different job.** That is union-of-kernel-intervals = *GPU busy fraction*, not occupancy; on 53534648 it is **81.0%**. Rename the column `gpu_busy_frac` (CLAUDE.md #10 — it is a cross-project column name). | Rename now. Real warp occupancy needs ncu (`sm__warps_active.avg.pct_of_peak_sustained_active`). |
| *"the average kernel uses 50% of the machine"* (SM-weighted 47%) | **Half right, and the half that matters is wrong.** Excluding NCCL, the mean SM fraction of a compute kernel is **0.988** — compute grids are fine. The entire deficit is NCCL: 5.05 of its 5.29 s are "wasted SM-seconds" at 6/132 SMs. Do **not** chase grid sizes on compute kernels. | Settled. |
| *"a 4-GPU ring… ~26 GB/s on the cross-pair path; two methods, same answer"* | **Arithmetically inconsistent** with the same file's own P2P matrix (18.3 GB/s, `PROFILING_TABLES.md:151`). Measured directly: 2.735 GB ring egress in 176.3 ms = **15.5 GB/s**, i.e. 85% of the 18.3 GB/s link. The "+65 ms ⇒ 26 GB/s" inference also assumed zero overlap. Arm B additionally moves NUMA node *and* PCIe root complex — `midway_topology_probe.sh:68` sets only `CUDA_VISIBLE_DEVICES`, no `numactl`. | Add a `numactl --cpunodebind/--membind` arm and a `world_size=1 on GPU0 vs GPU2` arm. |
| *"GEMM 7.6%"* (H100) | **Low by ~30%.** The bucket classifier keys on `shortName`; the cutlass complex GEMM's shortName is `Kernel2` and cuBLAS SM90 GEMMs are `nvjet_*`. Measured from `demangledName`: cutlass complex 4.98% + nvjet 3.67% + xmma 2.25% = **10.9%**. | Commit the classifier as `ACE2_retrain/kernel_buckets.py` keyed on `demangledName`, with an `(unclassified)` bucket that fails the gate above 2%. |
| *"`log_snapshots=false` halves validation (−52%)… ~30% faster epochs"* | **Half the lever was missed.** `log_mean_maps` is the sibling field (`fme/ace/aggregator/one_step/main.py:124`, gated at `deterministic.py:108`) and is never mentioned anywhere in the repo. Images are **82.6%** of warm validation; arm E still burns 10.81 s. Also the render cost is **O(1) per epoch** (`SnapshotAggregator.record_batch` only rebinds), so the *percentage* does **not** transfer to the production 128-batch / ~2900-sample window — the absolute ~28 s does. | Sixth arm: `validation_aggregator.log_mean_maps=false`. And restate the epoch claim as seconds, not a percentage. |
| *"`corrector` is the only torch.compile winner (−2.2%), measurements good to ±0.5%"* | **Not resolvable by that harness.** `midway_compile_probe.sh:117-142` medians deltas between `Step N:` log lines, which are emitted after `dist.reduce_mean`, a `wandb.log`→`dist.barrier()` and a tensor→string D2H (`trainer.py:566-571`). One job per arm, no clock logging. A 2.2% effect is inside H100 NVL DVFS drift. | CUDA-event timing inside the `ace2_nvtx.py` wrapper + `log_train_every_n_batches=0` + interleaved arms in one job + background `nvidia-smi --query-gpu=clocks.sm,power.draw -l 1`. |
| *"A100 → H100 is a clean hardware swap: validation 183 s → 32.2 s, epoch 231 s → 51.2 s"* | **~90% of that delta is a single-threaded matplotlib call on a different node** (`Getting logs for snapshot`: 164.27 s on beagle3-0012 vs 17.86 s on midway3-0423). The validation *loop* is 3.59 s vs 3.67 s — identical. Only the step-time row (0.542 → 0.337 s, 1.61×) is a defensible hardware number. Most likely a cold `MPLCONFIGDIR`/NFS font cache — the exact cold/warm boundary the notes warn about and then crossed. | Re-run the A100 smoke twice on the same node; or settle it with no GPU at all via a warm-vs-cold `MPLCONFIGDIR` micro-test. Set `MPLCONFIGDIR` to node-local in every ACE2 script regardless. |
| *"NVTX ranges applied: … `corrector`, `ema`"* | **Never fired in any capture.** Job 53534648's banner has no `corrector`/`ema`; `NVTX_EVENTS` confirms zero rows. `ace2_nvtx.py:171-188` swallows `ImportError/AttributeError`. `sht_fwd`/`sht_inv` = 0 rows too. Meanwhile `ema` is precisely the range that would have named the 157.6 ms sync. | Make the `except` fatal unless `ACE2_NVTX_OPTIONAL=1`, and assert in `parse_nsys.py` that every name in `applied` has non-zero rows. |
| *"CUDA-graph kernels do not appear in `CUPTI_ACTIVITY_KIND_KERNEL`"* (`midway_bench_nsys.sh:56-58`) | **False.** `--cuda-graph-trace=node` exists exactly so they do. The comment is a false blocker on the one lever the compile sweep could not test. | Fix the comment. (Graphs are still a *low* priority — see §3.) |

### New bug: the correlationId join double-counts across ranks

`kernel_census.py:56-59` joins `CUPTI_ACTIVITY_KIND_RUNTIME` to `CUPTI_ACTIVITY_KIND_KERNEL` on `correlationId` alone. `correlationId` is **per-process**; with 4 ranks in one report the naive join returns **630,428 rows for 481,841 kernels — +30.8% phantom launches**, attributed to whatever range happened to enclose another rank's launch timestamp. Every count and every time in `kernel_census.py` output, and the `spectral_filter` 35.6% / `(outside)` 57.1% split in `PROFILING_TABLES.md:44-52`, are affected.

Fix (one clause, verified to yield exactly 481,841 rows):

```sql
JOIN CUPTI_ACTIVITY_KIND_KERNEL k
  ON k.correlationId = r.correlationId
 AND k.globalPid = (r.globalTid & -16777216)   -- globalPid = globalTid with low 24 bits cleared
```

Same defect class as the already-recorded `parse_nsys.py` silent-drop. Re-run the copy attribution after fixing before quoting those shares again.

---

## §2 — (b) The single measurement that most changes the ranking

**Already made, above: exposed vs overlapped NCCL.**

`NCCL 52.1%` was a share of summed kernel time and was disclaimed as an upper bound. It is in fact **35.7% of wall-clock exposed**, with the GPU running a 6-SM ring kernel and nothing else, for ~130 ms of every 346 ms step. That single number reorders everything:

- Copies (17.5% non-contiguous elementwise) drop to **second**.
- `torch.compile` (2.2% ceiling) and CUDA Graphs (attacking a launch queue that is already 8.9 ms deep) drop to **noise**.
- The three levers that matter are all on the comm path and none of them has ever been tried.

**And the protocol is the surprise.** NCCL selected **`RING_LL`** for ~165 MB buckets. LL interleaves a 4-byte flag with every 4 bytes of payload — it costs **2× the wire bytes** and is meant for latency-bound small messages. Measured effective bandwidth is 15.5 GB/s of *user* bytes against an 18.3 GB/s link, i.e. the link is ~85% saturated *in the LL encoding*. If `NCCL_PROTO=Simple` is selectable on this topology it is worth up to ~2× on the wire.

> **If you do one thing: run the six-arm `NCCL_PROTO` / `NCCL_ALGO` sweep. It is one job, one env var per arm, zero code change, and it targets 35.7% of wall-clock.**

---

## §3 — (c) Dead ends — drop these so nobody spends time on them

1. **CUDA Graphs as a launch-overhead fix.** Median queue depth 8.87 ms; launch CPU time is 7.5% of the window and never on the critical path. Graphs would attack a bottleneck that does not exist, at the cost of static shapes, a private memory pool that fights `expandable_segments`, and DDP hook ordering. *Still fix the false comment at `midway_bench_nsys.sh:56-58`; do not fund the work.*
2. **`kernel_census.py`'s ranking premise.** "Rank by launch count because launches wreck the pipeline" is refuted. Keep the tool for *attribution*, retire the *ranking*, and fix the join first.
3. **Chasing grid size on compute kernels.** Non-NCCL mean SM fraction is 0.988. The "half-empty kernels" story is 100% NCCL.
4. **`channels_last`.** `torch_harmonics` contracts over latitude with spatial dims innermost and `s2convolutions.py:171,185` call `.contiguous()` (NCHW). NHWC would turn two currently-cheap calls into real transposes, and `sfnonet.py:218` rebuilds a contiguous buffer every block anyway. Only the 1×1 conv skips and the norms benefit — together 2.84% of GPU time. Decide from the dispatch census's stride data (§4.6), never by running an arm.
5. **`--set roofline` / `--set full` under ncu.** 2,841 and 619 metrics respectively on a workload issuing ~3,300 kernels/step. Use the explicit 15-metric list with `--kernel-name` + `--launch-count`.
6. **ncu under real DDP.** Kernel replay re-executes `ncclDevKernel`, which spins on peer flags → deadlock or corruption, and rank 0 stalled in replay trips the collective watchdog. Everything in the ncu lens is a per-kernel property; `--nproc_per_node=1 --override train_loader.batch_size=1` is bit-identical in kernel geometry (gridX=48600 and 48870 are both exactly one sample) and 4× cheaper.
7. **Raising `num_data_workers`.** Arms C/D already settled it (8 saturates), and dev0 idle *between* steps is 0.330 s = **3.0% of span**. The loader is hidden.
8. **`--sample=system-wide`, `--cpuctxsw=system-wide`, `--event-sample`, `--os-events`, `--ftrace`.** `perf_event_paranoid=2` on Midway; all need root.
9. **Faking a 2-node topology with `NCCL_HOSTID` to force hierarchical collectives.** NCCL would route cross-pair traffic over the *network* transport instead of P2P/SHM. Strictly slower.
10. **The `GradScaler` removal as a *throughput* win.** Measured: `_amp_foreach_non_finite_check_and_unscale_` is **0.35% of GPU time** (0.144 s), and the `optimizer` range is 2.6 ms. Worth doing for correctness-of-intent (bf16 needs no loss scaling) and to remove one sync, not for the time. Note `optimization.py:117` makes `enabled = self.gscaler is not None`, so a naive removal silently disables AMP — a new `use_grad_scaler` field is required.

---

## §4 — Ranked plan

Sorted by (evidence gained) / (effort + risk). **Tier A costs no GPU time at all.**

### TIER A — zero GPU cost, do today

**A1. Fix the correlationId join and the NVTX registered-string blindness.** *(1 h)*
- Blind to: 1,392 NVTX rows, incl. **all 1,368 of NCCL's own `ncclAllReduce` ranges** (`domainId=1`, registered string, `text IS NULL`); and 30.8% phantom launches.
- Where: `kernel_census.py:29-32` (`WHERE text IS NOT NULL`) and `:56-59` (join); `parse_nsys.py:75-86` (`WHERE text IN (...)`).
- Change: `LEFT JOIN StringIds s ON s.id = e.textId` + `COALESCE(e.text, s.value)`, select `e.domainId`; add the `globalPid` guard above.
- Changes a decision: the `spectral_filter` 35.6% / `(outside)` 57.1% copy split must be re-derived; and every "this range recorded zero events" conclusion is suspect until re-checked. **GH200: yes.**
- Payload columns are all NULL for NCCL on this build — do **not** plan on reading message size from NVTX.

**A2. Re-derive the validation decomposition and add the sixth arm.** *(30 min analysis + 1 short job)*
- Blind to: `log_mean_maps` entirely. Images are 82.6% of warm validation, not 52%.
- Where: `grep 'Getting logs for' outs/midway_valprobe_*/out.log`; then `ACE2_VAL_EXTRA="validation_aggregator.log_snapshots=false validation_aggregator.log_mean_maps=false"` in `midway_validation_probe.sh`.
- Changes a decision: takes warm validation 34.20 s → ~5.6 s (−84%). **But** run it at two window sizes (`stop_time=1996-01-17` vs `1996-04-01`) and restate the epoch claim in seconds — the render is O(1)/epoch and the "~30% faster epochs" percentage does not survive the production 2900-sample window, where the *loop* dominates.
- Same "no consumer when wandb is off" argument applies (`fme/core/wandb.py:164` discards the Images). `flush_diagnostics` also drops `mean_map_diagnostics.nc`, inert while `save_per_epoch_diagnostics=false`. **GH200: yes.**

**A3. Commit `kernel_buckets.py` keyed on `demangledName`.** *(2 h)*
- Blind to: `Kernel2` (cutlass complex GEMM, **4.98%**), `nvjet_*` (3.67%), `conj_kernel_cuda` (**1.51%**), `cudnn::bn_*` (2.84%). Published `GEMM 7.6%` should be **10.9%**.
- Explicit regex table + an `(unclassified)` bucket that fails the gate above 2% — this *is* the CLAUDE.md #10 contract. Restate old rows with the mapping recorded; do not silently replace them. **GH200: yes.**

**A4. Add `runtime_api_summary()` and `overlap_summary()` to `parse_nsys.py`.** *(2 h)*
- The two queries in §0. Apply retroactively to 53479120 (A100), 53524918 (H100), 53483668 (H200) → an exposed-comm comparison across all three hardware points, free.
- Preview: on the 8×H200 2-node capture, NCCL is **71.4% overlapped, 1.6% of span exposed** — the exposed-comm problem is specific to the pair-split H100 node. *(Caveat: that capture predates `ace2_nvtx.py`, so its span includes startup and validation; bound it before quoting.)* **GH200: yes.**

**A5. Run the shipped recipes and expert rules on existing reports.** *(half a day, mostly venv setup)*
- All present at `.../nsight-systems-2024.5.1/target-linux-x64/python/packages/nsys_recipe/recipes/`: `nccl_gpu_overlap_trace`, `nccl_gpu_proj_sum`, `nccl_sum`, `cuda_api_sync`, `gpu_gaps`, `gpu_time_util`, `cuda_gpu_kern_pace`, `nvtx_gpu_proj_pace`, `diff`. Reports: `cuda_kern_exec_sum` (queue time), `nvtx_kern_sum`, `nvtx_gpu_proj_sum`, `cuda_gpu_kern_gb_sum` (grid/block).
- Deps `pyarrow`+`psutil` are absent and the fme env has pandas **3.0.5** which 2024.5's recipe code predates → **separate venv with `pandas<3`**; installing into `/project/rcc/mehta5/envs/fme` would perturb the benchmarked env.
- Value: an independent implementation to check A1/A4 against — the exact thing missing when `parse_nsys.py` shipped its silent-drop bug. `nvtx_gpu_proj_sum` also does the RUNTIME→KERNEL projection correctly, for free. **GH200: yes** (but check `nsys --version` there first — recipe names differ across versions, and `delta_bench_nsys.sh:45-46` echoes host/arch but not the tool version. **Add `"${NSYS_BIN}" --version` to that banner.**)

### TIER B — one job each, no numerics, largest expected payoff

**B1. `NCCL_ALGO` × `NCCL_PROTO` sweep + `NCCL_DEBUG=INFO` topology dump.** *(1 job, 6 arms)*
- Blind to: which algorithm/protocol NCCL chose and why. We now know it is `RING_LL`; we do not know what it would pick otherwise, nor whether the cross-pair hop is P2P-over-PCIe or a host-SHM bounce.
- Where: env block `midway_bench_nsys.sh:82-92`. Replace `unset NCCL_DEBUG` (`:83`) for the diagnostic arm only:
  ```
  export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING,ENV
  export NCCL_DEBUG_FILE=${EXP_DIR}/nccl.%h.%p.log     # NOT stderr — the PASS gate greps .err
  export NCCL_TOPO_DUMP_FILE=${EXP_DIR}/topo.xml NCCL_GRAPH_DUMP_FILE=${EXP_DIR}/graph.xml
  ```
  Then one arm each: `NCCL_PROTO=Simple|LL|LL128` × `NCCL_ALGO=Ring|Tree`. Reachable space is exactly six — NVLS needs NVSwitch (H100 NVL is PCIe + bridge, none), CollNet needs SHARP, PAT is not AllReduce.
- **Verify each arm actually took effect from the TUNING lines** — NCCL silently falls back, and an arm that "shows no change" may never have run.
- Changes a decision: if `Simple` is selectable, LL's 2× wire cost disappears from 35.7% of wall-clock. This is the highest expected value in the plan.
- Risk: reduction order changes → sits at the 2.5e-7 same-hardware floor, i.e. DESIGN §4-gated, not free. **GH200: yes** (expect a null result — NV6 mesh, comm already 71% overlapped).

**B2. `NCCL_MIN/MAX_NCHANNELS`, `NCCL_NTHREADS`, `NCCL_CGA_CLUSTER_SIZE`, `TORCH_NCCL_HIGH_PRIORITY`.** *(same job as B1, ~6 more arms)*
- Blind to: SM contention. The ring runs **gridX=6 × 288 threads** and burns 5.05 "wasted SM-seconds" of 5.29. On a link-limited hop, fewer channels usually costs nothing on the wire and returns SMs.
- Zero numerical risk (channel count does not change the tree). Bracket the default; do not guess one value. **GH200: yes.**

**B3. `bucket_cap_mb` × `static_graph` sweep + `_get_ddp_logging_data()`.** *(1 job)*
- Blind to: DDP's own timers. `avg_backward_comm_time`, `avg_backward_compute_comm_overlap_time` (both confirmed in this torch build) measure exposed comm by a completely independent method — if it agrees with A4's 26.3%, the number is real.
- Where: DDP is built at `ace_exp/fme/core/distributed/torch_distributed.py:181-193` with **no `bucket_cap_mb`, no `static_graph`, no comm hook** (`grep -rn 'no_sync\|register_comm_hook\|bucket_cap\|static_graph' fme/` → zero hits). Patch `TorchDistributed.wrap_module` from `ace2_nvtx.py::install()` so the subtree stays clean. `ddp._set_ddp_runtime_logging_sample_rate(1)`; log all four ranks.
- **Correct the search range**: it is **11.4 buckets/step**, ~165 MB each — not the ~70 that a 25 MiB cap predicts. Sweep *around* 165 MB, and find the message-size knee first with `nccl-tests` (`all_reduce_perf -b 1M -e 2G -f 2 -t 4 -g 1`, built `MAP=0 NCCL_HOME=/software/nccl-2.29.7-el8-x86_64` — torch's bundled copy has no dev symlink).
- Use `TORCH_DISTRIBUTED_DEBUG=INFO`, never `DETAIL` (DETAIL inserts a per-collective sync and falsifies the numbers it reports). `static_graph=True` is safe today (`n_forward_steps=2` fixed, `drop_last=True`, checkpointing off) and would become unsafe if `CheckpointConfig` is ever enabled. **GH200: yes.**

**B4. GPU Metrics — `--gpu-metrics-set=gh100`.** *(1 job, gated on a 2-min permission test)*
- Blind to: every hardware counter. DRAM read/write % of peak, NVLink/PCIe/CTC bytes, `tpc__warps_active` (true warp residency), Tensor Active.
- Where: `midway_bench_nsys.sh:153-160` and `delta_bench_nsys.sh:119-126`, add `--gpu-metrics-devices=all --gpu-metrics-set=gh100 --gpu-metrics-frequency=20000`. Sets confirmed on disk: `.../nsight-systems-2024.5.1/target-linux-x64/GpuMetrics/{gh100,gh100-ct}.config`.
- Read back with a **timestamp-window** join to `NVTX_EVENTS` (not correlationId — counters have none). Only attribute to windows ≥ a few hundred µs: `step_N`, `forward_loss`, `backward`, `amp_region`. Not `sfno_mlp` (0.3 ms median).
- **GATE:** needs `NVreg_RestrictProfilingToAdminUsers=0`. Test first, in a job, in 2 minutes:
  ```
  cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly     # 0 = open
  nsys profile --gpu-metrics-devices=all --gpu-metrics-set=gh100 -d 10 \
       python -c 'import torch;torch.zeros(1,device=0)'
  ```
  If `ERR_NVGPUCTRPERM`, the RCC ask is one line in `/etc/modprobe.d` plus a node drain — plan a 1–2 week lead time. Also mutually exclusive with DCGM: if `dcgm-exporter` runs on midway3-0423, `dcgmi profile --pause` first.
- Known artifact: inactive NVLink rows report as fully utilised — read only the pairs that physically exist. NVLink/PCIe/CTC metrics are `schedulingRule: optional` and can be silently dropped. **GH200: yes**, and it is the only source for `ctc__rx/tx_bytes` (the Grace↔Hopper C2C link, which carries all host↔device traffic on Delta and has no analogue in the 29.9 GiB HtoD figure from Midway).

**B5. Restore `osrt` + add `--python-sampling` + extend the capture window over validation.** *(1 job)*
- Blind to: the largest unexplained block of wall-clock in the project. Validation is 61% of an epoch at 3.3% GPU, and the capture window **structurally excludes it** (`ace2_nvtx.py:224-227` calls `cudaProfilerStop` when training steps end; `midway_bench_nsys.sh:145` passes `--capture-range-end=stop`).
- Where: `midway_bench_nsys.sh:154` → `--trace=cuda,nvtx,cudnn,cublas,osrt --osrt-threshold=10000 --osrt-backtrace-threshold=100000`, plus `--python-sampling=true --python-sampling-frequency=1000`, plus `PYTHONUNBUFFERED=1`. The size objection at `:48-49` no longer applies — the capture is cudaProfilerApi-bounded.
- To reach validation: `--capture-range-end=repeat:2` and a second start/stop pair wrapping `fme.core.generics.trainer.Trainer.validate_one_epoch` (`ace_exp/fme/core/generics/trainer.py:653`), installed beside the `train_on_batch` wrapper at `ace2_nvtx.py:210-229`.
- CPU sampling is **already on** by default in every ACE2 capture and has never been opened; python-sampling is what makes those samples legible (they currently resolve to `_PyEval_EvalFrameDefault`). Python 3.11.15 vs the shipped `libToolsInjectionPythonBacktrace64.so` (3.10/3.11/3.12) — compatible. No root needed.
- Also **set `-s process-tree --backtrace=dwarf --samples-per-backtrace=4` explicitly on both scripts**: today Midway silently gets LBR (x86) and Delta will silently get frame pointers (aarch64, no LBR) on a conda torch built without `-fno-omit-frame-pointer` — so the CPU half of the cross-cluster comparison the Delta script exists for is not comparable. **GH200: yes** (SBSA supported; run `nsys status -e` in the job to check paranoid level there).

### TIER C — targeted, higher effort, needs a permission or a subtree edit

**C1. Name the four syncs and kill the two free ones.** *(1 short job)*
- Blind to: which Python line owns the 157.6 ms sync. It sits 0.25 ms after the `optimizer` range pops and is preceded by a D2H `cudaMemcpyAsync` — an `.item()`. Candidates: `torch/amp/grad_scaler.py:355` (`found_inf` `.item()`) or `TrainOutput` construction. **Do not guess — name it.**
- Two ways, both cheap: `torch.cuda.set_sync_debug_mode("warn")` in `ace2_nvtx.py::install()` (3 steps only — it floods stderr); or `--cudabacktrace=sync:20000 --python-backtrace=cuda` at `midway_bench_nsys.sh:153-160` (treat that run's timings as shape-only).
- Located in source already, all confirmed:
  | # | site | per step | measured |
  |---|---|---|---|
  | 1 | `optimization.py:267` `if torch.isnan(loss)` | 2 | **3.51 + 3.39 ms** |
  | 2 | `single_module.py:1574` `if torch.any(regularizer_loss > 0)` | 1 | ~0.008 ms ×3 |
  | 3 | after `optimizer` pops (unidentified `.item()`) | 1 | **157.6 ms** |
  | 4 | `device.py:59` `value.to(device)` — no `non_blocking=True`, loader sets `pin_memory=True` (`getters.py:126`) | ~50 | 0.727 s / 4 ranks / 30 steps = **6.1 ms/step**, between steps |
  | 5 | `ema.py:134` `min()` on two CUDA tensors | 1 | never measured — the `ema` range has never fired |
- **Honest ceiling.** #3 is a *symptom*: the GPU is 100% busy during it (151 ms of NCCL). Removing it does not remove GPU work; it only lets the CPU run ahead, and the next forward cannot start before the optimizer anyway. The real wins here are **#1 (~6.9 ms/step, 2%)** — defer the NaN check to a device-side accumulator read once per log interval, keep the exception — and **#4 (~6.1 ms/step, 1.8%)**, which is one keyword and numerically identical (source is pinned; add `record_stream` only if a consumer reads on another stream). ~4% total, free, no gate beyond the smoke. **GH200: yes.**

**C2. ncu on the strided copies — the layout-vs-fusion decision.** *(1 debug-queue job, ~5 min)*
- Blind to: bytes. 7.273 s (17.5%) is on ATen's **non-contiguous** iterator path (`elementwise_kernel<128,2>` ⇔ non-contiguous, elem ≥4 B — `torch/include/ATen/native/cuda/CUDALoops.cuh:455-462`), at an inferred ~26% of HBM3 peak (gridX=48870, 12,510,720 elem = 384×180×181, avg 101 µs). The contiguous path (`vectorized_elementwise_kernel`) hits ~89% of peak on the same machine. We cannot tell "slow" from "fast but over-fetching 32-B sectors", and those have **opposite fixes**.
- Decision rule: `dram__bytes.sum` ≫ N×(in+out) **and** `gpu__dram_throughput` >80% ⇒ over-fetch ⇒ **layout** is the only lever, fusion cannot help. `dram__bytes ≈` useful **and** throughput ~25% ⇒ latency/offset-calculator bound ⇒ fusion and bigger kernels both help. The tell is `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio` (32 = perfect fp32 coalescing, ≤8 = ≥4× over-fetch).
- Third possibility nothing can currently see: high `sm__inst_executed_pipe_alu` vs `pipe_lsu` ⇒ the `OffsetCalculator`'s per-element integer divmod is the bottleneck and the kernel is **compute**-bound on integer math — entirely plausible at 26% of peak.
- Invocation: `--nproc_per_node=1 --override train_loader.batch_size=1` (kernel geometry is per-rank and provably identical), `--profile-from-start off` (pairs with the existing `cudaProfilerStart` at `ace2_nvtx.py:215`), `--nvtx --nvtx-include "spectral_filter/"`, `--kernel-name regex:"elementwise_kernel|vectorized_elementwise_kernel|unrolled_elementwise_kernel"`, `--launch-count 120`. Add as a **sibling** `ACE2_retrain/midway_bench_ncu.sh`; do not touch the working nsys path.
- **Record `--clock-control` and `--cache-control` in the CSV header.** Defaults are `base`/`all`; ncu durations at base clock are not comparable to nsys durations at boost, and flushing L2 before every pass misrepresents a workload whose 50–100 MB tensors sit against a 50 MB L2 (evidence: the fastest observed 149 MB vectorized copy runs at an apparent 5.7 TB/s, above HBM3 peak — only possible with L2 hits). Collect both settings; make it a recorded convention (CLAUDE.md #10).
- Same **GATE ZERO** as B4. `ncu 2024.3.2` is on PATH at `/software/cuda-12.6-el8-x86_64/bin/ncu`, matching torch's cu126. **GH200: yes**, ncu 2025.2.0 on default PATH per `physicsnemo_ai_rossby/hpc/deltaai.md:99`; add `--section C2CLink --section NumaAffinity` there — `dram__bytes` is blind to Grace LPDDR traffic and a kernel at 26% of "DRAM peak" on Midway could be at 100% of C2C peak on Delta for a completely different reason.

**C3. `TorchDispatchMode` + `ModuleTracker` copy census.** *(~30 lines, 1 short job)*
- Blind to: whether `.contiguous()` at `s2convolutions.py:171,185` actually copies. `contiguous()` on an already-contiguous tensor returns `self` for free; that `aten::clone` is 45.3% of copy time means the inputs are **not** contiguous — and *which* stride pattern arrives there is the difference between "structural" and "one einsum output layout away from free".
- `torch.utils.module_tracker.ModuleTracker` installs **backward** hooks, so it names **modules** during backward, which phase-level NVTX ranges cannot resolve. ⚠ **Corrected 2026-08-20:** the 57%/81% `(outside)` was **not** an NVTX limitation — it was thread-scoped attribution (autograd launches from its own worker; the ranges are on the main thread). Scope the join to the *process* and `(outside)` goes to **0.0%** (`ACE2_retrain/nvtx_phase_attribution.py`, `polaris_bench_report.md` §4.3a). What survives is the narrower claim: NVTX phases cannot name *which module* runs in backward.
- Record `(tracker.parents, tracker.is_bw, shape, stride(), dtype, nbytes)` for `aten::{clone,copy_,_to_copy,contiguous}`; sum bytes per key. Cross-check the two `torch.zeros_like` + slice-assign sites at `sfnonet.py:218,234` (16 no-op copies per forward, currently an estimate).
- 2–5× slowdown: attribution only, never quote seconds. Also feeds the channels_last decision (§3.4) for free. **GH200: yes.**

**C4. `FlopCounterMode` — one number the project has never had.** *(1 step, rank 0, negligible)*
- `torch.utils.flop_counter.FlopCounterMode(mods=stepper.modules, depth=3)` around one `train_on_batch`; divide by `step_med`=0.3425 s and 4 ranks; compare to H100 NVL bf16 dense peak.
- Every table in `PROFILING_TABLES.md` is a share; a share cannot say whether the GPU is at 3% or 40% of peak, and the stated goal is efficiency. Report explicitly as a **lower bound** — `torch.fft` has no formula (einsum contractions do, they lower to matmul). **GH200: yes.**

**C5. NCCL flight recorder — a standing per-collective record.** *(env only)*
- `TORCH_NCCL_TRACE_BUFFER_SIZE=20000 TORCH_NCCL_ENABLE_TIMING=1`, dumped via `torch._C._distributed_c10d._dump_nccl_trace_json(...)` from the `ace2_nvtx.py` stop path. Gives per-collective duration and size, and the same collective's duration across all four ranks — the direct straggler test.
- `ENABLE_TIMING` inserts CUDA events per collective; run it as a **separate arm** and never quote its `step_med` against the 0.3425 s baseline (CLAUDE.md #10). Confirm the keyword names with `help()` on a compute node before a script depends on them. **GH200: yes.**

### TIER D — real levers, but gated and later

| item | target | gate |
|---|---|---|
| **Two-level all-reduce hook** (reduce_scatter in-pair → all_reduce cross-pair → all_gather in-pair) at `torch_distributed.py:193`. Arithmetic: 2.73 GB → 0.91 GB across the 18.3 GB/s boundary. | up to ~92 ms of a 346 ms step | reduction-order change → 2.5e-7 floor, DESIGN §4. **GH200: NO** — Delta is NV6 full mesh, there is no slow hop to route around. Midway-H100-only. |
| **`bf16_compress_hook`** — halves 1.82 GB of fp32 gradients on a model whose forward is bf16 (`optimization.py:118`). Composes with the above → 6× fewer cross-boundary bytes. | largest single comm lever | **real numerics change** (8 mantissa bits), well above both floors. jesswan's call. Run it as a *measurement* arm to size the bandwidth component, adoption left open. |
| **`use_gradient_accumulation`** as the memory-free batch-size lever | would make things **worse** as implemented | `optimization.py:164-168` calls `_backward` per accumulated loss with **no `no_sync()`** anywhere in fme (`grep no_sync fme/` → zero hits). At `n_forward_steps=2` that doubles all-reduces on a workload already 35.7% exposed. Confirm with one smoke counting nccl kernels per `step_N`; the fix is an upstream `module.no_sync()` wrap. |
| **cutlass `_80_..._align1` complex GEMM** — `contractions.py:184-195` `_contract_dhconv`, Ampere tile schedule on Hopper, 1-element alignment (no 128-bit loads), 4.98% of GPU time | ~2.5% of GPU time | pad the contracted dim to 8 complex elements, or reshape to an explicit `bmm`. Changes summation order. The one GEMM-level fix Inductor can never reach (complex64) — orthogonal to the torch.compile verdict. Measure with ncu first. |
| **`conj_kernel_cuda`** 1.51% — PyTorch has had a lazy conjugate bit since 1.10; something in the `view_as_complex`/`view_as_real` round trip is materialising it | 1.5% | plausibly **bit-exact** (a conj view and its materialisation hold identical values) → the cheapest gated item. May live in vendored `torch_harmonics 0.8.0`. |
| **`expandable_segments:True`** — set in both bench scripts, never A/B'd | unknown | it is an ungated hot-path allocator change under CLAUDE.md #6, and it switches `cudaMalloc`→VMM, which changes what `--cuda-memory-usage=true` records — so the memory side of every ACE2 report is silently non-comparable to a run without it. Two smokes. Pin it before touching anything graph- or NCCL-registration-related. |

---

## §5 — Cross-cutting hygiene (each is a one-liner, all block comparability)

- `midway_bench_nsys.sh:56-58` — the CUDA-graph comment is factually wrong (`--cuda-graph-trace=node`). Fix the text even though the lever is deprioritised.
- `delta_bench_nsys.sh:45-46` — add `"${NSYS_BIN}" --version` and `nsys status -e` to the banner. Midway is 2024.5.1.113; Delta is unknown, and recipe names and `--gpu-metrics-*` spelling move between versions.
- Both scripts — set `-s`/`--backtrace` explicitly (see B5); the defaults differ by architecture and silently break the cross-cluster CPU comparison the Delta script exists for.
- `ace2_nvtx.py:162,176,187,206` — make the `except (ImportError, AttributeError)` fatal unless `ACE2_NVTX_OPTIONAL=1`; add a post-capture assertion in `parse_nsys.py` that every name printed in `applied` has non-zero `NVTX_EVENTS` rows.
- ACE2 still has **no bench CSV and no equivalence baseline**, unlike S2S/SI/PanguWeather. Add `torch.cuda.Event(enable_timing=True)` phase timers at the four sites `ace2_nvtx.py` already wraps (`optimization.py:173/179`, `single_module.py:335`, `ema.py:119`), buffered so `elapsed_time()` never touches a fresh event (that would add a *fifth* sync), emitting the house column names. Log `memory_stats()['num_alloc_retries','reserved_bytes.all.peak']` alongside — ACE2 has **zero** memory data on any hardware, while its largest recommended lever is batch size.
- Keep the NVTX *message strings* byte-identical if `_range` (`ace2_nvtx.py:66-79`) is ever moved to the `nvtx` package for domains/categories/payloads. Add fields; never rename. (`pip install nvtx` is required and is **not** currently in `/project/rcc/mehta5/envs/fme` — install into a copy, not the benchmarked env.)

---

## §6 — Transfer to PanguWeather / ai-rossby

| finding | transfers? |
|---|---|
| Exposed-comm measurement (A4) | **Yes, and harder.** Pangu's NCCL share is 74.2% on the same node with 4.73 GB of gradients. Its NVTX `backward` spans 112–993 ms (8.8×) with no per-rank breakdown — run `cuda_gpu_kern_pace` on it. |
| `RING_LL` protocol check + `NCCL_PROTO` sweep (B1) | **Yes** — same node, same NCCL, larger messages. Highest-value transfer. |
| correlationId `globalPid` guard (A1) | **Yes** — any multi-rank single-report capture. |
| `demangledName` bucketing (A3) | **Yes** — Pangu's `61% elementwise / 15% GEMM` on Polaris has the same `Kernel2`/`nvjet` blind spot. |
| cutlass `align1` complex GEMM | **Yes and bigger** — 12 SFNO layers at embed_dim 512 vs ACE2's 8 at 384. |
| Spectral no-op-copy guard (`67242e348`) | **Yes** — already documented; both trees lack it and `hard_thresholding_fraction: 1.0` means the guard always fires. |
| Two-level all-reduce hook | Midway-H100 only; **not** on Polaris A100 or Delta NV6. |
| Validation / `log_mean_maps` | ACE2-specific (fme aggregators). |