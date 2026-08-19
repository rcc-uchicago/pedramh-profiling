# ACE2 (ai2cm `fme`) on Midway — bring-up and first profile

Living notes for ACE2 on Midway, in the style of `si/bench_midway_notes.md`:
narrative first, then a dated decisions log. Cross-cutting summary lives in
`CHANGELOG.md`; this file carries the measured detail.

## What exists

| Piece | Path |
|---|---|
| Vendored model | `ACE2_retrain/ace_exp/` (ai2cm/ace @ `1c3ebad80`, `fme` 2026.5.1) |
| Midway config | `ACE2_retrain/config_midway.yaml` — port of the Delta `config_nsight.yaml` |
| Smoke | `ACE2_retrain/midway_smoke_train.sh` → `ACE2_SMOKE_OK` |
| nsys profile | `ACE2_retrain/midway_bench_nsys.sh` → `ACE2_NSYS_OK` |
| Env | `/project/rcc/mehta5/envs/fme` — torch 2.7.1+cu126, fme 2026.5.1 |

The original `train.sh` is the **Delta/NCSA** launcher and is left untouched
(rule #7). Its env (`/scratch/midway3/krucker01/envs/fme`) is not readable by
us, which is why a Midway env had to be built from
`ace_exp/Makefile::create_environment` (minus `[docs,graphcast]` and the
healpix/analysis extras — not needed for ERA5 lat-lon).

## Why ACE2's NCCL share is 52% here — ALREADY DOCUMENTED in s2s/v2.0/bench_report.md

`nvidia-smi topo -m` on midway3-0423 (job 53537121):

    GPU0  X    NV12  SYS   SYS       GPU0<->GPU1 NVLink (12 links)
    GPU1  NV12  X    SYS   SYS       GPU2<->GPU3 NVLink
    GPU2  SYS  SYS    X    NV12      across the pairs: SYS = PCIe + NUMA hop
    GPU3  SYS  SYS   NV12   X        GPU0/1 on NUMA 0, GPU2/3 on NUMA 1

This **reproduces** `s2s/v2.0/bench_report.md` footnote 6, which already recorded
it: *"NVLink is NV12 within socket-pairs only (GPU0<->1, GPU2<->3; cross-pair =
SYS/UPI, no NVLink) -- unlike Midway H200's NV6 full mesh."* Treat the above as
independent confirmation on current hardware, not a new finding.

**Midway H200 nodes ARE fully meshed (NV6).** It is specifically this H100 NVL
node that is pair-only -- which is why the same ACE2 code measured NCCL at
**18.6% on 8x H200** versus **52.1% here**.

**Why S2S never surfaced this**: the S2S profiles contain **no NCCL at all**
(bench_report.md: *"No NCCL collective kernels appear in any profile ... verified:
count = 0. These are data-parallel inference runs with no gradient
synchronisation."*). S2S was profiled doing inference, so it never used the
cross-pair link for collectives; its analysis targets CPU->GPU handoff latency.
**ACE2 is the first training workload on this node to do heavy gradient
all-reduce, and so the first to stress that path.**

The transfer-side weakness was already measured there too: under concurrent load
GPU0/1 sustain ~27 GB/s while **GPU2/3 collapse to ~14 GB/s (-42%, NUMA-aligned)**,
against an Ice Lake PCIe Gen4 ceiling of ~32 GB/s. Our ~15 GB/s effective
all-reduce bandwidth is consistent with that.

⇒ ACE2's 52% NCCL share is a property of **this node's topology**, not of the
model. On the H200 nodes the same code spends 18.6%. Any ACE2 scaling number
taken on pedramh-gpu should say so.

### PROVEN on-node: the split NVLink costs +29% per step

`midway_topology_probe.sh`, jobs **53538838 / 53538839**. Identical 2-rank job
run twice, changing only which two GPUs are used. `batch_size=2` on 2 ranks
keeps 1 sample/rank, matching the 4-GPU baseline, so per-rank compute is
unchanged and the only variable is the gradient path.

| arm | GPUs | link | step_med | vs NVLink |
|---|---|---|---|---|
| A | 0,1 | **NVLink (NV12)** | **0.2250 s** | — |
| B | 0,2 | **SYS** (PCIe + NUMA hop) | **0.2900 s** | **+28.9%** |

**+65 ms per step for crossing the pair boundary**, on an otherwise identical
workload. Sanity check: a 2-GPU ring all-reduce moves 1.82 GB per rank, so 65 ms
of extra time implies ~**26 GB/s** on the cross-pair path -- essentially the
~27 GB/s PCIe Gen4 sequential figure `s2s/v2.0/bench_report.md` measured
independently on this host. Two different methods, same answer.

It also explains the 4-GPU baseline (0.3425 s/step): a 4-GPU ring crosses that
boundary **twice**, so it is slower than either 2-GPU arm.

⇒ ACE2's 52% NCCL share on this node is the interconnect, demonstrated rather
than argued. No Polaris run was needed -- and ACE2's 2.39 TB dataset is not
staged on eagle anyway (per `polaris_pbs_notes.md` even plain ERA5 is not), so a
Polaris comparison would have cost a Globus stage and still changed GPU
generation, CPU, filesystem and data all at once.

**Practical consequence**: for 2-GPU work on this node, pin to one NVLink pair
(`CUDA_VISIBLE_DEVICES=0,1` or `2,3`) and get 29% for free. For 4-GPU work the
penalty is unavoidable here; the H200 nodes (NV6 full mesh) are the fix, at the
cost of the `test` partition and its queue.

## ⭐ REVERSE TRANSFER: ACE2 has a spectral fix PanguWeather and ai-rossby lack

ACE2's vendored perf commit `67242e348` guards the no-op copy in
`SpectralConvS2.forward`:

```python
if self.modes_lat_local >= x.shape[-2] and self.modes_lon_local >= x.shape[-1]:
    # the slices below cover the whole tensor (e.g. when
    # hard_thresholding_fraction == 1.0), so the zeros_like plus
    # slice-assign is a full no-op copy; contract directly instead.
    x = self._contract(...).contiguous()
else:
    xp = torch.zeros_like(x)   # only when the slices are partial
```

**PanguWeather and ai-rossby have the unguarded version** — both at
`.../modulus_sfno/s2convolutions.py:196`, straight to `zeros_like` + slice-assign
with no fast path.

**And the guard would always fire for them**: every PanguWeather config sets
`hard_thresholding_fraction: 1.0` (`E3SM_SFNO_H5_POLARIS.yaml`,
`..._DERECHO_jsw`, `..._STAMPEDE_jsw`, `..._ALLDATA`, `tiny_baseline`,
`SFNO_PLASIM_..._5411`), which is exactly the condition that makes the slices
cover the whole tensor.

Scale of the waste, per forward pass:

| | layers | embed_dim | guarded? |
|---|---|---|---|
| PanguWeather (`E3SM_SFNO_H5_POLARIS`) | **12** | 512 | ❌ no |
| ai-rossby | same tree | — | ❌ no |
| ACE2 | 8 | 384 | ✅ yes |

So Pangu performs **12 × (allocate + zero-fill + full copy) of a complex64
tensor** (8 bytes/element) per forward, per unroll step, for no result — the
commit message for the ACE2 fix calls it "a full no-op copy of a complex tensor
per block per unroll step". ACE2 pays none of it.

This is consistent with PanguWeather profiling at **61% pointwise vs 15% GEMM**
(`polaris_bench_report.md`) — a higher elementwise share than ACE2's 47.9%.

**Adoption note**: unlike the `FourierNeuralOperatorBlock` norm sites (where
ACE2's dtype differs and a cast must be kept), this one is a straight lift — the
guard is the author's own, already running in ACE2. It is still a hot-path
change and needs the DESIGN 4 gate. PanguWeather is a **fork** (no propagation)
and `physicsnemo_ai_rossby` is a **subtree** (edits can conflict on a future
pull), so each needs its own patch.

## NCCL protocol sweep — the RING_LL hypothesis is REFUTED (ACE2)

The adversarial profiling pass found NCCL had selected `RING_LL` for ~165 MB
buckets and recommended an `NCCL_PROTO`/`NCCL_ALGO` sweep as the single highest
lever, "worth up to ~2x" against 35.7% of wall-clock. Measured, jobs
53546801/802/804/805, 4x H100 NVL:

| arm | step_med | vs control |
|---|---|---|
| control (`RING_LL`) | 0.3420 s | — |
| `NCCL_PROTO=Simple` | 0.3415 s | **-0.15% (noise)** |
| `NCCL_PROTO=LL128` | 0.4350 s | **+27% SLOWER** |
| `NCCL_ALGO=Tree` | FAILED | not applicable |

**The protocol is not the lever.** Switching off LL's inline-flag encoding
changes nothing measurable.

**The env vars WERE honoured** -- LL128's +27% is the control that proves it. A
null result from `Simple` alone would have been ambiguous ("did the setting even
apply?"); LL128 moving the number removes that reading. LL128 slowing things is
itself consistent: it relies on 128-byte store atomicity that NVLink provides and
this node's cross-pair PCIe hop does not.

`NCCL_ALGO=Tree` errors outright: *"no algorithm/protocol available for function
AllGather with datatype ncclInt8"*. Tree is a reduction tree with no AllGather,
which DDP requires.

⇒ The 35.7% exposed all-reduce is **real but not addressable by protocol
selection**. The link is saturated in whatever encoding is used. That leaves the
comm levers as: fewer/larger buckets, bf16 gradient compression (changes
numerics -- jesswan's call), or **larger batch** (fewer all-reduces per sample),
which the H200 data already showed is the effective one.

**Method note.** The workflow's recommendation carried an internal arithmetic
tension -- LL cannot both double the wire bytes and deliver the measured
15.5 GB/s of user data on an 18.3 GB/s link. That tension was the tell, and one
job settled it. Detailed reasoning is not evidence.

## Partition policy — pedramh-gpu ONLY (from 2026-08-19)

All ACE2 runs go to **`pedramh-gpu`** and no other nodes. Every script's SBATCH
block now defaults to it:

    --account=pi-pedramh  -p pedramh-gpu  --qos=pedramh-gpu  --constraint=H100

`--qos=pedramh-gpu` is mandatory (the partition's `AllowQos`), and `DefaultTime`
is NONE so `--time` must be explicit. The partition is **one node**,
`midway3-0423`, 4x H100, 32 cores — shared with nobody, so `--exclusive` costs
no one else.

**Consequence: multi-node work is over.** 8 GPUs needs 2 nodes and this
partition has one, so `midway_{smoke_train,bench_nsys}_2node.sh` are **parked**.
They are pointed at pedramh-gpu deliberately, which makes `sbatch` reject them
outright ("More processors requested than permitted") rather than silently
taking nodes from another partition. The 8-GPU H200 results already captured
(53483666/667/668) stand; re-running needs a conscious override back to `test`.

**Consequence: the hardware baseline moves A100 -> H100.** The profile tables
below were taken on A100-PCIE in `test`. Step times are not comparable across
that boundary — only shapes are. The `ACE2_NSYS_DELAY`/`DURATION` defaults were
also measured on A100 and will sit in the wrong place on a faster node; check
them against a fresh smoke timeline before trusting a windowed capture.

## H100 baseline — job 53524865, pedramh-gpu (midway3-0423, 4x H100 NVL)

First run under the pedramh-gpu-only policy. Same config, same seed (3), same
`batch_size=4` (1/rank) as the A100 baseline, so this is a clean hardware swap.

| | 4x A100-PCIE (53478978) | 4x H100 NVL (53524865) |
|---|---|---|
| samples/s/rank | 1.84 | **2.97** (1.61x) |
| training, 16 steps | 48 s | **19.2 s** |
| validation | ~183 s | **32.2 s** |
| epoch | 231 s | **51.2 s** |
| wall | 5:55 | **2:03** |

**Validation is 63% of the epoch here** (32.2 s of 51.2 s) — training sped up
1.6x and validation still dominates, exactly as on A100 and H200. The
`log_snapshots=false` result applies directly: halving 32.2 s takes this epoch
to ~35 s, **-31% for a config flag with no numerical risk**.

### Cross-hardware reproducibility: ~1e-5

Same config and seed, A100 vs H100:

| | A100 | H100 | relative |
|---|---|---|---|
| train_loss | 35.78267288208008 | 35.782554626464844 | **3.3e-6** |
| valid_loss | 36.21318817138672 | 36.212799072265625 | **1.1e-5** |

Together with the same-hardware floor of 2.5e-7 (jobs 53483666/667), that gives
two tolerance floors for DESIGN §4:

- **same GPU, same node: 2.5e-7** — the run-to-run nondeterminism floor
- **across GPU architectures: ~1e-5** — a baseline captured on one arch and
  checked on another cannot be tightened past this

An ACE2 equivalence gate must sit above whichever applies, and must record which
hardware the baseline was captured on. Note for contrast that the ai-rossby
`torch.compile` failure was 4.02e-01 — four orders of magnitude above even the
cross-arch floor, so that verdict is nowhere near these limits and stands.

## H100 windowed profile — job 53524918, and a CORRECTION

Window `ACE2_NSYS_DELAY=18 ACE2_NSYS_DURATION=55`, re-derived from the H100
smoke (the A100-era 45/110 would have opened after training ended). 52.9 s span,
130.8 s kernel time over 4 ranks, batch 4 (1/rank).

| bucket | A100 batch 4 | **H100 batch 4** | H200 batch 16 (8 GPU) |
|---|---|---|---|
| NCCL | 40.6% | **52.1%** | 18.6% |
| elementwise / copy | 35.4% | **26.8%** | 47.9% |
| GEMM | 11.3% | 7.6% | 11.9% |

**CORRECTION to the 8x H200 entry below.** That entry reads the -22pp NCCL drop
as confirmation that "the NCCL share really was the PCIe interconnect". This
H100 point refutes it: at the **same batch size**, moving A100 -> H100 pushed
NCCL's share *up* (40.6% -> 52.1%), because compute got faster while the
per-step gradient all-reduce did not. So the H200 improvement came mostly from
**batch 16 vs 4** -- 4x more compute per all-reduce -- not from NVLink/IB. The
original entry flagged that hardware and batch moved together and could not be
separated; this run separates them, and batch is the dominant term.

⇒ **The lever on communication is batch size, not interconnect.** Raising the
per-rank batch does more for the comm share than better hardware does, and it
costs nothing numerically (it changes optimisation dynamics, which is jesswan's
call, but not the arithmetic of a step).

Copies alone are 14.5% of GPU kernel time here, versus 28% on H200-at-batch-16 —
consistent, since NCCL's dominance at batch 4 compresses every other share.

### Script bug found and fixed: windowed captures were marked FAILED

Job 53524918 wrote a valid 217 MB report and SLURM still recorded **FAILED**.
When `--duration` expires, nsys **SIGTERMs the target**, so torchrun's elastic
agent dies with `SignalException: got signal: 15` and exits non-zero; with
`set -eo pipefail` the script aborted before its own PASS gate could run. That
is exactly the trap CLAUDE.md #14 warns about — gate on the artifact, not the
exit code. `midway_bench_nsys.sh` now captures the return code, treats non-zero
as expected **only** when a duration window is set (and still a hard error when
it is not), and tolerates a missing epoch-end loss since a windowed run is cut
short by design.

## NVTX instrumentation, and a CORRECTION it forced

`ace2_nvtx.py` injects the shared range names without editing the vendored
subtree, and `midway_bench_nsys.sh` with `ACE2_NVTX=1` now uses the house
`--capture-range=cudaProfilerApi --capture-range-end=stop` flags -- the
hand-derived `--delay/--duration` window is no longer needed. First capture:
job **53533290**, 127 MB.

### ⚠ The "copies come from the dict<->tensor round trip" claim was WRONG

Earlier entries attribute the copy traffic (28% of GPU kernel time on H200) to
fme's `dict[str, Tensor]` state being stacked and unstacked around the network,
and recommend ending that round trip as the largest available lever. **That was
inferred from kernel names plus code reading. Measured, it does not hold:**

| | measured |
|---|---|
| GPU time launched from inside `stack` | **0.03%** |
| `normalize` + `denormalize` | 0.3% |
| `unstack` | **never called** -- no callers in the training path |
| copy kernels overall (this H100 capture) | 18.5% of GPU kernel time |
| ...launched from autograd/backward threads | **58%** |
| ...launched from the forward path | 42% |

So the copies are real and large, but they are **spread across forward and
backward**, not concentrated at a gather site. That pattern fits AMP fp32<->bf16
autocast conversions and internal `.contiguous()` calls throughout the model.
`Stacker._stack_levels` is 10 calls per step at 0.07 ms of launch cost -- it is
not the problem, and refactoring it would buy approximately nothing.

**Do not act on the stacker recommendation.** The remaining copy sources are
unattributed; finding them needs ranges inside the SFNO blocks and around the
autocast boundaries, which is the next instrumentation step, not a refactor.

### WHERE THE COPIES ACTUALLY ARE: the spectral filter — job 53534648

Ranges added around the autocast boundaries and inside the SFNO, then GPU time
attributed by joining RUNTIME -> KERNEL on correlationId and taking the
**innermost** enclosing range.

**Copy-kernel GPU time (7.91 s = 13.4% of GPU in this capture):**

| range | share of copy time |
|---|---|
| `(outside)` — backward, on autograd threads | 57.1% |
| **`spectral_filter`** | **35.6%** |
| `sfno_block` | 3.1% |
| `sfno_mlp` | 3.0% |
| `stack` | **0.2%** |

**Of everything attributable to the forward path, `SpectralFilterLayer` owns
83% of the copies.** Same ordering for GPU time overall: `spectral_filter` 9.1%,
`sfno_block` 3.3%, `sfno_mlp` 2.6%.

That is a coherent story rather than a coincidence: the spectral path is where
complex tensors live, where FFT/SHT impose layout (contiguity) requirements, and
where AMP is switched **off** and back on around the transform — so every entry
and exit is a candidate fp32<->bf16 conversion. It is also **the same path
Inductor refuses to compile** (`KeyError: 'complex64'`). One region accounts for
the largest copy source, the compile blocker, and the FFT/SHT work.

Two ranges recorded **zero** events, both informative:

- `sht_fwd`/`sht_inv` — patched onto `fme.sht_fix.RealSHT`/`InverseRealSHT`,
  which are **not the classes in use**. The startup banner says
  `RealSHT from torch_harmonics.sht`: the vendored perf commit switched to the
  native 0.8.0 transform, so `sht_fix`'s versions are dead code on this path.
  The transform still runs, inside `spectral_filter`, via torch_harmonics.
- `amp_region` — nonzero as a *range* (median 28.1 ms) but ~zero under innermost
  attribution, because everything inside it is covered by a deeper range. That
  is the attribution working as intended, not a miss.

**Still unattributed: the 57% launched from autograd threads.** `ACE2_NVTX=1
ACE2_NVTX_AUTOGRAD=1` turns on `emit_nvtx` and will name those, at the cost of
timings that must not be quoted.

### parse_nsys.py had a silent-drop bug

The name list exists **twice** — once in the SQL and once in the print loop.
Extending only the query fetched the new ranges and then never printed them,
which looked exactly like "the ranges did not fire". Both lists now carry the
same names, with a comment saying they must track each other. Corrected output:

    forward_loss     240   27.9 ms      amp_region       240   28.1 ms
    backward         120  102.6 ms      sfno_net         240   19.7 ms
    optimizer        120    5.1 ms      sfno_block      1920    2.3 ms
    stack           1200    0.1 ms      spectral_filter 1920    1.3 ms
    normalize        720    1.3 ms      sfno_mlp        1920    0.3 ms
    denormalize      240    1.7 ms      step total       120  346.2 ms

The `backward` (102.6 ms) and `optimizer` (5.1 ms) figures also confirm the
mapping fix from the previous capture, where they read 0.12 ms and 263 ms.

### The backward side, and what actually creates the copies — job 53535415

`ACE2_NVTX_AUTOGRAD=1` (emit_nvtx) names the autograd ops, so the 57% of copy
time that previously showed as `(outside)` is now attributed. **Timings from
this run are void** -- emit_nvtx wraps every autograd op -- so read the shares,
not the seconds, and take magnitudes from the clean capture (53534648).

By the nearest *informative* enclosing op (leaf `aten::copy_` stripped, since
"the copy was caused by a copy" says nothing):

| cause | share of copy time |
|---|---|
| **`aten::clone`** | **45.3%** |
| `aten::select_backward` | 6.5% |
| `AddBackward0` | 2.7% |
| `aten::bmm` / `fill_` / `nccl:all_reduce` / `convolution_backward` | ~1% each |

**One operation, `aten::clone`, causes nearly half the copy traffic.**

### The code behind it

`SpectralConvS2.forward` (`fme/ace/models/modulus/s2convolutions.py`), under AMP:

```python
x = x.float()                        # 165: bf16 -> fp32, full-tensor copy
with torch.amp.autocast("cuda", enabled=False):
    x = self.forward_transform(x)    # the SHT must run in fp32
    x = x.contiguous()               # 171: layout materialisation
    residual = residual.to(dtype)    # 173: fp32 -> bf16 back
x = self._contract(...).contiguous() # 185: our path (hard_thresholding_fraction 1.0)
```

Every AMP boundary crossing around the transform is a full-tensor dtype
conversion, and the SHT's layout requirement forces `.contiguous()`. This is the
autocast-boundary hypothesis, confirmed in code rather than assumed.

### ⭐ An actionable fix the perf commit already validated elsewhere

`FourierNeuralOperatorBlock.forward` (`sfnonet.py:217`) does this **twice per
block**, and there are 8 blocks:

```python
x_norm = torch.zeros_like(x)                       # full alloc + zero fill
x_norm[..., :H, :W] = self.norm0(x[..., :H, :W])   # slice-assign it all back
```

When the slice covers the whole tensor -- which it does without spatial
parallelism, i.e. our configuration -- that is a **full no-op copy**, 16 per
forward pass. The vendored perf commit `67242e348` fixed **exactly this pattern**
in `s2convolutions.py`, and even left the comment explaining it ("the slices
below cover the whole tensor ... so the zeros_like plus slice-assign is a full
no-op copy; contract directly instead") -- but the two instances in
`FourierNeuralOperatorBlock.forward` were not touched.

**CORRECTION to an earlier draft of this entry**, which called the fix "bitwise
identical" and "numerically free". Removing the buffer outright is NOT
equivalent in ACE2: `zeros_like(x)` inherits x's dtype, which under ACE2's AMP
is **bfloat16** (`Optimization.autocast` sets `dtype=torch.bfloat16`), while
`norm0` returns fp32 under autocast. The slice-assign therefore performs an
fp32 -> bf16 **downcast**, and dropping the buffer would leave the result in
fp32 -- more precision, different numbers.

The bitwise-identical formulation keeps the cast and drops only the allocation
and zero-fill:

```python
x_norm = self.norm0(x).to(x.dtype)   # same rounding, no zeros_like, no fill
```

That still removes an allocation and a fill kernel per site (16 per forward),
just not the copy itself. Adoption remains gated on a DESIGN 4 baseline, which
ACE2 does not have.

### The same pattern is in PanguWeather and ai-rossby -- and they differ

All three Modulus-lineage SFNOs carry it. A grep for `zeros_like` finds only
ACE2 because the other two spell it differently:

| tree | buffer | effect under AMP |
|---|---|---|
| ACE2 | `torch.zeros_like(x)` -> **bf16** | norm output is **downcast** fp32 -> bf16 |
| PanguWeather | `torch.zeros(x.shape, dtype=torch.float32)` | no downcast |
| ai-rossby | identical to PanguWeather | no downcast |

Both non-ACE2 trees carry the comment: *"Use float32 for zero tensor to avoid
implicit float16 downcast from norm layers (norm layers return float32 under AMP
autocast; assigning into float16 can overflow)."* Someone hit that and fixed it
in those two trees only.

**So the three are not numerically equivalent at this point in the network.**
ACE2 rounds normalised activations to bf16; the others keep fp32. The overflow
motivation in the comment concerns fp16, and ACE2 uses **bf16**, which has
fp32's exponent range -- so ACE2 is unlikely to overflow. But it does lose
mantissa (8 bits vs 24) where the other two do not. Worth raising with jesswan:
it is a real precision difference between our SFNO implementations, not a
performance detail.

**The fix is cleaner for the other two than for ACE2.** Their buffer is already
fp32 and `norm0` already returns fp32, so `x_norm = self.norm0(x)` is bitwise
identical and removes the allocation, the fill AND the copy. In ACE2 the cast
must be kept.

### Method note: an NVTX range bounds CPU time, not GPU time

This is why the first read was misleading. CUDA is async, so a range around a
launch site does not contain the GPU time of the kernels it launches -- `stack`
shows 0.07 ms of launch cost while its kernels execute later. GPU attribution
requires joining `CUPTI_ACTIVITY_KIND_RUNTIME` to `CUPTI_ACTIVITY_KIND_KERNEL`
on `correlationId` and finding the range enclosing the **launch**. And backward
kernels are launched from autograd worker threads, so they sit outside any range
pushed on the main thread -- 81% of GPU time lands "outside any range" for that
reason alone, not because it is unaccounted for.

### Two instrumentation bugs the first capture exposed

1. **`backward` and `optimizer` were mapped by method name, not behaviour.**
   With `use_gradient_accumulation: false`, fme's `accumulate_loss` only does
   `_accumulated_loss += loss`; the real backward runs inside `step_weights`.
   The capture showed `backward` = 0.12 ms and `optimizer` = 263 ms (76% of the
   step) -- `optimizer` was swallowing the backward. Under CLAUDE.md #10 that is
   worse than having no range: the same name would mean something different here
   than in every other project. Now wrapped on `_backward` and `_step_weights`.
2. **`unstack` never fired**, because nothing calls it. Range dropped.

Step totals from the capture, for scale: **median 348 ms/step**, `forward_loss`
27.8 ms x2 per step (16.7% of GPU time by correlation).

## torch.compile: where it can go, and why it is not the lever

Seven arms on 4x H100 (`midway_compile_probe.sh` + `ace2_compile_probe.py`),
batch 4, 64 steps, median of the last 30 inter-step gaps so compile warmup is
excluded rather than averaged in.

| arm | step_med | vs control | loss drift | verdict |
|---|---|---|---|---|
| `none` (control) | 0.3425 s | — | — | baseline |
| `corrector` | 0.3350 s | **-2.2%** | 2.0e-5 | **the only winner** |
| `all` (norm+corr) | 0.3345 s | -2.3% | 8.9e-3 | gain is corrector's, drift is normalizer's |
| `normalizer` | 0.3420 s | -0.15% | **8.4e-3** | reject |
| `safe` (mlp+corr) | 0.3450 s | +0.7% | 1.2e-5 | MLP cancels the corrector |
| `mlp` | 0.3540 s | **+3.4% slower** | 3.6e-6 | reject |
| `network` (whole SFNO) | FAILED | — | — | **impossible** |

Self-consistency: predicting `safe` from the individual effects gives 0.3465 s
vs 0.3450 s measured, and `all` (0.3345) reproduces `corrector` (0.3350). The
measurements are good to about +/-0.5%, so the 2-3% differences are real.

### The three findings that matter

1. **The SFNO cannot be compiled at all.**
   `torch._inductor.exc.InductorError: KeyError: 'complex64'`. The spherical
   harmonic transform path is complex-valued and Inductor has no complex64
   lowering. This is a backend limitation, not a tuning problem — 92 dynamo
   mentions confirm compilation ran and then failed in codegen. Note also that
   fme hands DDP to the compiler here (`self.module._module` is a
   `DistributedDataParallel`), which is the wrong order for a real adoption.
2. **Compiling the normalizer is actively bad.** Zero speedup (0.15%, inside
   noise) for **8.4e-3** relative loss drift — 4 orders above the 2.5e-7
   same-hardware floor and ~400x the run-to-run drift over the same 64 steps.
   It was the top-ranked candidate on kernel-count reasoning (~100 tiny kernels
   per call); that reasoning was right about launches and wrong about payoff.
3. **Compiling the MLP blocks makes it slower** (+3.4%). The blocks are small
   and called many times, so guard and wrapper overhead exceeds whatever fusion
   buys. Combining it with the corrector nets out worse than the corrector alone.

### torch 2.8.0 re-test: the complex64 wall is still there, it just stopped shouting

Built `/project/rcc/mehta5/envs/fme-torch28` as a single-variable copy of the
2.7.1 env — same cu126 wheels, same `torch_harmonics==0.8.0`, same fme; only
torch moves. Jobs 53531456/457/459.

| arm | torch 2.7.1 | torch 2.8.0 |
|---|---|---|
| control | 0.3425 s | 0.3510 s (**baseline 2.5% slower**) |
| `corrector` | 0.3350 s (-2.2%) | 0.3420 s (**-2.6%**) |
| `network` | **hard fail**: `InductorError: KeyError: 'complex64'` | **runs, +3.4% slower** |

The network arm no longer crashes on 2.8 — but it did not compile either:

    torch/_inductor/lowering.py:1890: UserWarning: Torchinductor does not
    support code generation for complex operators. Performance may be worse
    than eager.

**torch 2.8 downgraded the hard error to a warning and falls back to eager.**
Corroborated three ways: zero graph breaks (a lowering fallback, not a break),
warmup only +0.6 s over control (nothing substantial was compiled), and a 3.4%
slowdown from dynamo guard overhead with no fusion in return — precisely what
torch's own warning predicts.

⇒ **Inductor still cannot fuse the SFNO's spectral path on either torch.** The
2.7.1-vs-2.8 difference is error-vs-warning, not capability. This also resolves
the ai-rossby tension recorded earlier: its working `torch.compile` is a torch
version difference in *error handling*, and whatever 1.40x it measured did not
come from fusing the complex-valued transform.

Incidental but worth knowing: **the torch 2.8 baseline is 2.5% slower** than
2.7.1 for this workload. An upgrade taken for other reasons costs about as much
as the corrector arm gains.

### Verdict

**torch.compile is not the lever for ACE2** — confirmed on two torch versions.
The entire reachable gain is the corrector's **2.2%** (2.6% on torch 2.8),
against an estimated ~20% ceiling for fusion. The estimate
was too optimistic because it assumed the fusable pointwise work was contiguous
enough to fuse; in practice it is spread across many small call sites, and the
one big contiguous region (the SFNO) is complex-valued and off-limits.

The measured alternatives are far larger and carry less numerical risk:

| lever | measured | numerics |
|---|---|---|
| `log_snapshots=false` | **~30% off the epoch** | none (output is discarded when wandb is off) |
| raise batch size | NCCL 52.1% -> 18.6% | none in the step arithmetic |
| stop dict<->tensor round trips | targets 28% of GPU time | none if done correctly |
| `torch.compile` corrector | 2.2% | 2.0e-5 |

⇒ Pursue the copies (`stacker.py:121`) and the config-level wins. If the
corrector's 2.2% is wanted anyway it is cheap, but it needs a DESIGN 4 baseline
first, and 2.0e-5 drift puts it above the same-hardware floor — so it is a
gated change, not a free one.

## Cluster facts confirmed on-node 2026-08-18

Run on `--account=rcc-staff -p test`, as requested — **not** the project's usual
`pi-pedramh`/`pedramh-gpu`.

| Item | Value |
|---|---|
| Partition | `test` — `Hidden=YES`, `AllowAccounts=rcc-staff`, **`AllowQos=test`** (so `--qos=test` is mandatory) |
| Default walltime | **`00:05:00`** — omit `--time` and you silently get 5 minutes |
| Hardware | **mixed**: `beagle3-*` = A100, `midway3-02xx` = V100, `midway3-0320` = A30 ⇒ **`--constraint=a100` is load-bearing** |
| GPU actually allocated | **A100-PCIE-40GB** ×4 on `beagle3-0012` — PCIe, *not* SXM/NVLink. This dominates the profile below. |
| nsys | `module load cuda/12.6` (matches torch's cu126 build) |

## Measured — jobs 53478978 (smoke) and 53478979 (nsys)

Both at `batch_size=4` global (1/rank), `stepper_training.n_forward_steps=2`,
AMP on, eager. **Not production shape**: the Delta config uses `batch_size 16`,
sized for 96 GB GH200s; 40 GB A100 headroom is still unmeasured.

- **455,831,040 trainable parameters** (455.8 M).
- **`tf32=True`** is logged at startup — i.e. the vendored `67242e348` perf
  commit is *active*. It has never been equivalence-checked (DESIGN §4).
- **Step time 0.54 s** (smoke, cold) / **0.56 s** (under nsys, warm) at
  1.84 training samples/s/rank ⇒ nsys overhead ≈ 4%.
- **Page cache dominates wall-clock**, exactly as recorded for PanguWeather:

  | | job 53478978 (cold) | job 53478979 (warm, same node) |
  |---|---|---|
  | launch → "Starting Training Loop" | 80 s | 50 s |
  | epoch total | 231 s (64 samples) | 117 s (**512** samples) |

  The second run trained **8× more samples in half the wall-clock**. Any
  timing compared across a cold/warm boundary is meaningless.
- **The epoch is dominated by validation, not training.** In the cold smoke,
  16 training steps took ~48 s and the remaining **~184 s (80%)** went to
  validation + train-evaluation aggregators.

## First profile — job 53478979, 275 MB report, whole-run capture

Bucketed from `CUPTI_ACTIVITY_KIND_KERNEL` over all 4 ranks
(352.3 s of kernel time, 2,380,116 launches, 148.6 s wall):

| bucket | % GPU kernel time | seconds | launches |
|---|---|---|---|
| NCCL (comm **+ wait**) | **45.8%** | 161.2 | 8,648 |
| elementwise / copy | **32.2%** | 113.5 | 1,702,640 |
| GEMM | 10.4% | 36.6 | 178,840 |
| norm / cudnn | 4.4% | 15.4 | 80,072 |
| optimizer | 3.8% | 13.3 | 57,856 |
| FFT / SHT | 2.2% | 7.7 | 38,304 |
| reduction | 0.9% | 3.2 | 110,336 |
| other | 0.4% | 1.4 | 203,420 |

Memory traffic: HtoD **29.9 GiB / 2.08 s**, DtoH 0.31 GiB / 0.03 s,
DtoD **5,665 GiB / 10.14 s**.

Occupancy: **do not read one off this capture** — it spans startup, training and
validation, and a summed-kernel-time average across those phases (~59%) is not a
quantity that means anything. See the per-phase occupancy under the windowed
capture below: 91% during training, 3.3% during validation.

### How to read this — three caveats that change the conclusion

1. **NCCL kernel time is not comm cost.** Ring kernels spin while waiting for
   peers, so that 45.8% conflates real transfer with load imbalance and
   straggler wait. The single largest AllReduce instance is **4.16 s** against a
   median of 11.2 ms — that is waiting, not bandwidth. Treat 45.8% as an upper
   bound on "time not spent computing", not as "time spent on the wire".
2. **This capture includes startup and validation**, not just the training hot
   path — and validation is ~80% of an epoch (above). Job 53479120 re-runs it
   windowed (`ACE2_NSYS_DELAY=45 ACE2_NSYS_DURATION=110`) to isolate training.
3. **`batch_size=4`, not 16.** Smaller batches make the per-step gradient
   all-reduce a larger share of the step. Expect the NCCL fraction to fall at
   production batch size.

### Windowed re-capture — job 53479120 (`ACE2_NSYS_DELAY=45 ACE2_NSYS_DURATION=110`)

205 MB report, 100.5 s trace span. The window covers ~71 s of training plus the
trailing validation, so it is the training-dominated view the unbounded capture
could not give:

| bucket | whole window | **first 71 s (training)** | unbounded (53478979) |
|---|---|---|---|
| NCCL (comm + wait) | 40.9% | **40.6%** | 45.8% |
| elementwise / copy | 35.2% | **35.4%** | 32.2% |
| GEMM | 11.2% | **11.3%** | 10.4% |
| norm / cudnn | 4.7% | 4.7% | 4.4% |
| optimizer | 4.1% | 4.2% | 3.8% |
| FFT / SHT | 2.4% | 2.4% | 2.2% |

The shape is stable across all three views, so it is not an artifact of where
the capture window fell.

**Occupancy — do NOT quote a whole-window average.** Summing kernel time over
the whole window gives "70% busy", which is meaningless: it averages a busy
training phase with an idle validation tail. Measured properly (union of kernel
intervals per device, so multi-stream overlap is not double-counted, in 5 s
bins):

| phase | GPU occupancy |
|---|---|
| training (steady, ~0–58 s) | **91%** |
| validation tail | **3.3%** |

The 9% idle during training is **launch latency, not a stall**: 326,176 idle
gaps totalling 4.74 s over 55 s on device 0, largest single gap **9.76 ms**, no
sync bubble. Device 0 issues **397,207 kernels in 55 s = 7,222 launches/s**, one
every ~138 µs. 38% of the idle sits in 0.1–1 ms gaps and 32% in 1–10 ms gaps.
Same root cause as the 35% elementwise share — ~2,900 tiny elementwise kernels
per step per rank cannot keep the launch queue ahead of the GPU — so fusion
(`torch.compile`, CUDA graphs) would attack both at once.

**This is NOT comparable to PanguWeather's "0.7% loader idle"**
(`polaris_bench_report.md`). That is `loader_wait_frac` — the fraction of
*training-loop* wall time blocked on the data loader — not kernel occupancy over
a capture window. ACE2 has no instrumentation, so its loader-wait equivalent is
unmeasured. `polaris_bench_report.md` records this exact trap already
("`cpu_prep_frac` is not loader idle; built `loader_wait_*`").

**New finding — validation is CPU-bound, not GPU-bound.** 280.2 s of the
window's 282.5 s of kernel time falls in the first 71 s. Validation therefore
contributes **~1% of GPU kernel time while consuming ~40% of the window's
wall-clock** (and ~80% of a cold epoch, above). The aggregators, not the GPU,
are what make an ACE2 epoch long. Any "speed up ACE2" work that only touches
the training step is optimizing the smaller half of the epoch.

Even discounted, two things look real: this is an **elementwise-bound** model
(32% of kernel time, 1.7 M launches, versus 10% GEMM) — the same shape the
PanguWeather profile found — and **fp32 gradient all-reduce over PCIe** is
expensive for a 455.8 M-parameter model on non-NVLink A100s.

## No instrumentation exists in fme

There is **no** `cudaProfilerApi`, `torch.profiler`, or NVTX anywhere in the
SFNO lat-lon training path — the only NVTX in the tree is in the HEALPix layers
and the downscaling module, neither of which this config touches. Consequences:

- The house `--capture-range=cudaProfilerApi --capture-range-end=stop` flags
  would capture **nothing** here, so `midway_bench_nsys.sh` uses a time window
  instead. This is the one place it deliberately departs from the s2s/SI/port
  scripts.
- `parse_nsys.py` produces no useful NVTX summary for ACE2 — it keys on
  `data_prep`/`forward_loss`/`backward`/`optimizer`, which this model never
  emits. The tables above came from querying the sqlite directly.
- `GlobalTimer`'s category breakdown only reaches wandb, which the house rule
  disables. **Set `logging.metrics_log_dir`** to get those scalars on disk.

Adding `ACE2_*` bench knobs + NVTX that emits the **shared** range names is the
follow-up that makes ACE2 comparable to the other models. Per CLAUDE.md #10 the
names must match the existing contract, not invent new ones.

## 8 GPUs (2 nodes x 4 H200) — **GREEN**, job 53483666

`ACE2_SMOKE_2NODE_OK train_loss=60.243 valid_loss=57.124 world=8 batch=16`, 2:26
wall, on `midway3-[0603,0604]` — both `gold-6542Y`, so homogeneous by luck
despite the smoke's loose `--constraint=H200`.

| | 4x A100-PCIE (batch 4) | 8x H200, 2 nodes (batch 16) |
|---|---|---|
| samples/s/rank | 1.84 | **6.82** (3.7x) |
| aggregate samples/s | 7.4 | **54.2** (7.4x) |
| per-rank batch | 1 | 2 |

The 3.7x is **not** a pure per-GPU comparison: H200 ran 2 samples/rank against
A100's 1, so part of it is better utilisation at larger per-rank batch.
`ACE2_BATCH_SIZE=8` gives the like-for-like 1/rank number.

Three things this settles:

- **Multi-node NCCL works on Midway.** `NCCL_SOCKET_IFNAME=^lo,docker0`, flagged
  as inherited-and-unverified, is now confirmed — no hang, no fallback.
- **The production `batch_size=16` fits**, exercised for the first time in this
  project. The A100s had to drop to 4 for 40 GB.
- **Validation gets *worse* on faster hardware, as predicted.** Training ended
  21:46:21 and the epoch ended 21:46:52: **30 s of a 49 s epoch (61%) is
  validation**, up from ~40% on A100. Speeding up the training step raises
  validation's share of the epoch — it does not shrink it.

`nproc` reads "2 cores" in that job's banner. That is an artifact of the
node-info probe (an `srun` overriding `--ntasks` without `--cpus-per-task`,
which binds the step to one core), NOT what training ran with: `sacct` shows
step `.2` with AllocCPUS=96 over 2 nodes = 48/node, and the throughput confirms
it. Fixed in both scripts.

## 8-GPU H200 profile — job 53483668, 2 reports / 166 MB

Both node reports combined (83.4 s kernel time, 41.9 s span each), batch 16.
**This is the profile that matters** — it is the target hardware, at the
production batch size.

| bucket | 8x H200 | 4x A100-PCIe | delta |
|---|---|---|---|
| **elementwise / copy** | **47.9%** | 35.4% | **+12.5pp** |
| NCCL (comm + wait) | **18.6%** | 40.6% | **-22.0pp** |
| GEMM | 11.9% | 11.3% | +0.6 |
| other | 6.4% | 0.4% | +6.0 |
| norm / cudnn | 5.3% | 4.7% | +0.6 |
| FFT / SHT | 4.2% | 2.4% | +1.8 |
| optimizer | 3.0% | 4.2% | -1.2 |
| reduction | 2.8% | 1.0% | +1.8 |

**Both A100-era hypotheses are confirmed.** The 40.6% NCCL share *was* largely
the PCIe interconnect: NVLink inside a node plus IB between them, together with
4x the batch (fewer all-reduces per sample), more than halved it. And the
elementwise share is what survives better hardware — it is now the largest
bucket by a wide margin.

Caveat: two variables moved at once (hardware **and** batch 4 -> 16), so the
-22pp on NCCL cannot be attributed to interconnect alone. `ACE2_BATCH_SIZE=8`
on H200 would separate them.

### Inside the 47.9%: it is mostly COPIES, not math

| | share of bucket | share of all GPU time | launches |
|---|---|---|---|
| **copies** (`direct_copy`, `bfloat16_copy`) | **58.2%** | **28%** | 400,712 |
| add | 20.1% | 9.6% | 194,856 |
| other pointwise math | 10.0% | 4.8% | 214,376 |
| unary (scale/cast-like) | 7.7% | 3.7% | 89,528 |
| fill | 4.1% | 2.0% | 96,408 |

**ACE2 spends 2.4x more GPU time copying tensors (28%) than doing matrix
multiplies (11.9%).** Copies are the single largest identifiable cost on the
target hardware — larger than NCCL.

This sharpens the `torch.compile` question decisively:

- **Fusion reaches ~20% of GPU time** (add + unary + fill + other pointwise),
  plus some of the launch-latency idle. Real, worth doing, individually gateable
  region by region.
- **Fusion does NOT reach the 28%.** Those copies are structural: fme stores
  state as `dict[str, Tensor]` and round-trips it through
  `stacker.py:121 torch.stack([data[name] for name in names], dim=-1)` for ~43
  inputs and `unstack()` for ~50 outputs, every step, for each of the 3
  timesteps in the `n_forward_steps=2` window — plus AMP casts. `torch.compile`
  cannot make a gather of 50 separate tensors free.
- ⇒ **Keeping state stacked is the bigger lever**, and it is pure data movement:
  no numerics change if done correctly, so it is not gated on jesswan's sign-off
  the way the corrector or TF32 are. It is an upstream-shaped change to fme.

Still unmeasured: *which* module emits those 400k copies. There is no NVTX in
the SFNO path, so the attribution above is from kernel names plus code reading.
The `ACE2_*` NVTX follow-up is what would prove it.

### Run-to-run reproducibility floor

Jobs 53483666 and 53483667 ran the **same config, same seed (3), same two
nodes** and returned `train_loss` 60.24342346191406 vs 60.243438720703125 —
a **2.5e-7 relative** difference. Not bitwise reproducible (TF32, atomics, NCCL
reduction order). **Any future ACE2 equivalence baseline must set its tolerance
above this floor**, and the floor should be re-measured on the hardware the
baseline is captured on. → DESIGN §4.

## Validation: NOT a dataloading problem — it is snapshot rendering

Jobs 53524580/581/674/675/752 on `midway3-0423` (4x H100, `pedramh-gpu`).
64-sample validation window, 2 epochs each; the cross-arm comparison uses
**epoch 2**, warm in every arm. Everything else held identical.

| arm | change | warm validation | vs baseline |
|---|---|---|---|
| A | baseline (batch 4, 8 workers) | 34.20 s | — |
| B | batch 16 (4x fewer batches) | 33.86 s | **-1%** |
| C | 1 data worker | 43.88 s | +28% |
| D | 16 data workers | 34.43 s | +1% |
| **E** | **`log_snapshots=false`** | **16.45 s** | **-52%** |

Read in order, this is conclusive:

1. **Not per-batch overhead.** 4x fewer batches changed nothing (arm B). Rules
   out aggregator call overhead, python per-batch cost, launch counts.
2. **Not loader-bound.** Loader parallelism helps only from 1 -> 8 workers
   (~10 s) and then **saturates**: 16 workers gains nothing (arm D). The config's
   existing `num_data_workers: 8` is already the right value; raising it is
   pointless. So data loading contributes ~10 s of a 44 s serial-loader case and
   is fully hidden at the default.
3. **It is snapshot image rendering.** Turning off `log_snapshots` halves
   validation outright (arm E).

### The images have no consumer in our configuration

`fme/ace/aggregator/one_step/snapshot.py:91 get_logs()` calls
`plot_paneled_data(...)` to build wandb `Image` objects. The `_enabled` guard
lives *inside* `WandB.log()` (`fme/core/wandb.py:144/164/174`), so the panels are
**rendered first and discarded afterwards** whenever wandb is off. With the house
`log_to_wandb: false` / `WANDB_MODE=offline`, plus `save_per_epoch_diagnostics`
at its default `false`, nothing reads them.

⇒ For offline runs, `validation_aggregator.log_snapshots=false` removes work
whose output is thrown away. **It changes no numerics**, and with wandb disabled
it changes no observable output either. It is NOT free if you turn wandb back on
or enable `save_per_epoch_diagnostics` — then it is a real reporting change and
jesswan's call.

### Epoch-level impact

On 8x H200, validation was 30 s of a 49 s epoch (61%). Halving it takes the
epoch to roughly 34 s — about **30% faster epochs for a config-flag change with
no numerical risk**. That is larger than anything `torch.compile` offers on the
training step (~20% of GPU time, gated on an equivalence baseline that does not
yet exist), and it is available today.

Caveat: measured on a 64-sample validation window. The production config
validates over 1996-1997 (~2900 samples), so the absolute seconds scale but the
*proportions* are what transfer.

## Still queued




`midway_smoke_train_2node.sh` (→ `ACE2_SMOKE_2NODE_OK`) and
`midway_bench_nsys_2node.sh` (→ `ACE2_NSYS_2NODE_OK`), jobs **53483263** and
**53483265** (the latter chained `afterok`). Both **queued as of 2026-08-18**;
no result yet. They are siblings — the single-node scripts are untouched.

**The launcher had to change, which is why these are separate scripts.**
`torchrun --standalone` binds rendezvous to localhost and *cannot* span nodes.
Multi-node needs one launcher per node sharing a c10d rendezvous — the shape the
Delta `train.sh` already proved for this codebase:

```
--ntasks-per-node=1   # ONE launcher per node; torch.distributed.run forks the 4 local ranks
srun python -m torch.distributed.run --nnodes 2 --nproc_per_node 4 \
     --rdzv_id $SLURM_JOB_ID --rdzv_backend c10d --rdzv_endpoint <head_ip>:29500
```

`--ntasks-per-node=4` here would start 4 launchers per node = 16 ranks, not 8.

**H100, not H200** — measured with `sbatch --test-only` on 2026-08-18:

| constraint | est. start | nodes |
|---|---|---|
| **H100** | **08-18 09:44** | `midway3-[0372,0423]` |
| H200 | 08-18 20:45 | mixed flavours |
| `H200&gold-6542Y` | 08-19 08:31 | homogeneous |
| `H200&epyc-9335` | 08-20 03:34 | homogeneous |

H100 was both soonest *and* automatically homogeneous: `--constraint=H100` with
`--gres=gpu:4` can only match `gold-6346,512g` 32-core nodes, because the other
H100 box (`midway3-0432`, Gold-6448Y/1TB) has `gpu:2` and is excluded by the
4-GPU request. Homogeneity is not cosmetic here — the single-node profile showed
NCCL ring kernels spin while waiting for peers, so a slower partner node gets
recorded as communication cost that does not exist. Retarget without editing:
`sbatch --constraint=H200 ...` (then prefer `"H200&gold-6542Y"` to measure).

**Batch size 16 — the production value — runs for the first time here.** fme
requires `batch_size % world_size == 0`; world size is 8, so 16 gives 2/rank.
The A100 runs had to drop to 4 for 40 GB. `ACE2_BATCH_SIZE=8` gives 1/rank, the
like-for-like weak-scaling comparison against the 4-GPU A100 runs.

**What this is meant to answer.** The single-node profile put NCCL at 40.6% of
GPU kernel time on **A100-PCIE, which has no NVLink**. Two nodes of H100 change
both variables at once — NVLink within a node, and an inter-node hop across
InfiniBand (`ib0`). So expect the split to move; the useful question is whether
the elementwise 35.4% share holds, since that is the part no interconnect can
fix. nsys writes **one report per node** (`_node0`/`_node1`); read them together
or inter-node imbalance is invisible.

Open risk: `NCCL_SOCKET_IFNAME=^lo,docker0` is inherited from the repo's legacy
`midway_training.sh`, not confirmed against a working Midway multi-node NCCL
run. If job 53483263 hangs at startup, that is the first thing to suspect —
re-run with `ACE2_NCCL_DEBUG=INFO`.

## Decisions / changes log

- **2026-08-18** — First ACE2 bring-up and profile on Midway. Env built at
  `/project/rcc/mehta5/envs/fme`; `config_midway.yaml` ported from the Delta
  config (paths + wandb only, model/loss/optimizer/variables byte-identical);
  `midway_smoke_train.sh` and `midway_bench_nsys.sh` added beside the untouched
  `train.sh`. Jobs **53478978** `ACE2_SMOKE_OK` (train 35.783 / valid 36.213,
  5:55) and **53478979** `ACE2_NSYS_OK` (275 MB report, 4:24). Numbers above.
  Job **53479120** windowed capture `ACE2_NSYS_OK` (205 MB, 4:00) — confirms the
  bucket shape is stable and shows **validation is CPU-bound** (~1% of GPU kernel
  time for ~40% of the window's wall-clock).
  - Smoke/bench shorten the production config with `--override` rather than
    forking a second config, so every deviation is visible in the script.
  - A pre-flight `python -m fme.ace.validate_config` runs before the GPU work:
    fme parses strict (dacite `strict=True`), so one stale key aborts the run —
    catching that in seconds beats catching it after a 4-GPU allocation opens a
    2.39 TB file.
  - **Open**: production `batch_size=16` is unvalidated on 40 GB A100; the
    `tf32=True` hot-path change is unvalidated; no equivalence baseline exists
    for ACE2 at all.
