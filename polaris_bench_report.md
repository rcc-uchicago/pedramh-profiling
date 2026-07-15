# Polaris (A100) profiling report — PanguWeather SFNO

The Polaris analog of `s2s/v2.0/bench_report.md` and `si/bench_midway_notes.md`: what the
profiling phase measured on **4× A100-SXM4-40GB**, how it was measured, and which of it is
trustworthy. Style follows `si/bench_midway_notes.md` — narrative + a dated decisions log.

See **CLAUDE.md** for how to work here, **DESIGN.md** for what/why, **CHANGELOG.md** for
cross-cutting status, and **`polaris_pbs_notes.md`** for the cluster facts and bring-up traps.

> ## ⚠ These numbers are NOT comparable with Midway's
> Midway's `bench_results.csv` and `bench_report.md` are **H100 NVL (~94 GB, PCIe Gen4,
> NVLink within socket-pairs)**. This is **A100-SXM4-40GB (AMD Milan host)**. A different
> node class. Per DESIGN §1 non-goals, a slower A100 step is *expected*, not a regression —
> and the two must never share a table. Nothing below is compared against Midway. The
> *method* (`S2S_BENCH`, warmup 20 / steps 80, the same NVTX range names) is deliberately
> identical so that future within-cluster comparisons are valid.

---

## 1. What was profiled, and why only this

**PanguWeather SFNO on E3SM** (`nettype: sfno_plasim`, `--config=SFNO`), the model the
2026-07-15 focus change points at (DESIGN §2c) and the only one of the S2S family that runs
on Polaris today — `s2s/v2.0` and the Lightning port are still blocked on the ERA5 Globus
stage.

Profiling it **began by building the instrumentation**: PanguWeather carried **zero** NVTX
ranges and no `S2S_BENCH` harness, so there was nothing to measure with. That port is the
bulk of this session's work; the range names and CSV columns are byte-identical to s2s's on
purpose (CLAUDE.md #10).

**SI / makani / physicsnemo are NOT profiled here yet.** The handoff asks for all four. This
report covers one, honestly, rather than four superficially — the instrumentation port,
its regression proof, and the two DESIGN §4.0 prerequisites consumed the session. SI already
has its own `SI_BENCH_*`/`SI_NVTX` harness and a green Polaris bench (7252700/7253603), so it
is the cheapest next one; makani and physicsnemo have no comparable harness at all.

### Configuration under test

| | |
|---|---|
| Model | PanguWeather `sfno_plasim`, **1,182,108,160 params** (measured, job 7255410) |
| Data | staged E3SM v3 SSP245-AMIP, 1460 samples/yr, 180×360, 18 sigma levels |
| Shape | 4 ranks × batch 1, bf16 autocast (no GradScaler), DDP `find_unused_parameters=False` |
| Activation checkpointing | `checkpointing: 3` (**on** — recompute in backward) |
| EMA | `use_ema: True` but `ema_warmup_epochs: 6`, so **inactive** in the 1-epoch runs below |
| Window | warmup 20 / measured 80 (the Midway convention), eager — **no `torch.compile`** |

> **1.18 billion parameters, not ~79M.** DESIGN and CLAUDE.md both describe the S2S family as
> "~79M-param" (the `test.yaml` trap, rule #12). That figure is about the *Pangu/Swin* model.
> The E3SM **SFNO** at `embed_dim: 512, num_layers: 12, num_blocks: 16` is **15× larger**.
> Any resource intuition carried over from the 79M figure is wrong for this path.

---

## 2. The headline: GPU-bound, and bound by *elementwise* work

The handoff asks for the cheap, high-value question first — **is the hot path GPU-bound or
input-bound?** If it were input-bound, every rung of the DESIGN §5 kernel ladder would be
premature.

**It is GPU-bound.** At the shipped `num_data_workers: 1`, **0.7%** of loop wall time is
spent waiting on the data loader. The kernel ladder is not premature.

And the profile says *which* kernels: **61% of GPU time is elementwise/pointwise**, spread
over **~1506 launches per step**. Only **15%** is tensor-core matmul. That is the signature of
a **memory-bandwidth-bound, fusion-starved** model — which is precisely what `torch.compile`
(DESIGN §5 rung 1) exists to fix. The ladder's existing ordering is now **measured**, not
assumed.

---

## 3. Data-loader sweep — and the metric that had to be built to answer it

`num_data_workers` sweep, everything else identical (4×A100, bf16, batch 1/GPU, eager,
warmup 20 / steps 80). Full rows: `$MEMBER_ROOT/bench/pangu_sfno_polaris_bench.csv`.

| workers | job | step_med | step_p90 | step_std | loader_wait_med | **loader_wait_frac** | `samples_per_s` (step rate) | **WALL samples/s** | peak mem |
|---|---|---|---|---|---|---|---|---|---|
| **1** (shipped) | 7255410 | 0.652 s | 0.826 s | 0.110 | 0.0002 s | **0.7%** | 6.13 | **6.09** | 26.98 GB |
| 0 | 7255434 | 0.615 s | 0.631 s | 0.011 | 0.1042 s | **14.8%** | 6.50 | **5.53** | 26.98 GB |
| 8 | 7255480 | 0.602 s | 0.603 s | 0.0003 | 0.0002 s | **0.0%** | 6.64 | **6.64** | 26.98 GB |

### `samples_per_s` is a step rate, not throughput — read this before quoting it

`samples_per_s = global_batch / step_med`, and `step_med` **excludes the between-step loader
fetch**. So it overstates throughput exactly when the loader is the problem. Convert:

```
wall throughput = samples_per_s × (1 − loader_wait_frac)
```

At `workers=0` the CSV says **6.50** while the real rate is **5.53**. Comparing that 6.50
against `workers=1`'s 6.13 would have ranked the **slower** configuration first. The column
name is s2s's and is kept (CLAUDE.md #10); the conversion is the fix.

### Why a new metric was needed at all

`cpu_prep_frac` — the closest thing the inherited harness had — is **not** the data-loader
idle fraction, and reading it as one is a trap:

* `cpu_prep_med` times `_prepare_inputs_batch`, which runs on a batch the loader has
  **already produced** (H2D + reshape). It is 0.002–0.004 s, i.e. **0.3–0.6%** of the step,
  in all three runs — *including the deliberately starved one*.
* The blocking fetch happens inside the loader's `__next__`, **between** steps, inside no
  step window at all. It is invisible to every column s2s writes.

Worse, it was **fatal**: `_bench_finalize` reconciles `elapsed` against `sum(step_times)`,
which holds only while the loader keeps ahead of the GPU. On an input-bound run the gap
lands in `elapsed`, the 10% self-check fires, and the row is **refused** — the harness
aborts precisely when the loader is the finding. Now measured explicitly
(`loader_wait_med` / `loader_wait_frac`, appended after s2s's 19 columns) and folded into
the self-check, which makes it *tighter*: any residual >10% is a genuine timer bug again.

### The metric was falsified before it was believed

A metric that cannot move proves nothing. `workers=0` forces the fetch to happen
synchronously inside `__next__`, so the gap **must** appear:

> **0.7% → 14.8%, a 21× move**, with `cpu_prep_frac` flat at 0.3→0.4% throughout.

So the "GPU-bound" verdict is a measurement, not a dead metric reading zero.

### What the sweep actually shows

* **The shipped `num_data_workers: 1` is not the bottleneck, but it is not free either.**
  It costs ~5% of step time (0.652 vs 0.602 s) and **10× the jitter** (step_std 0.110 vs
  0.0003; p90 0.826 vs 0.603). The single worker's HDF5 read runs *concurrently* with the
  step and contends with the rank's main thread for CPU. With 8 workers each batch is
  prepared 8 step-times ahead, so the read is fully hidden and the step is almost perfectly
  regular (p90 − med = **0.3 ms**).
* **`workers=0` is the worst of both**: the cleanest step (no contention → std 0.011) but
  0.104 s of dead time per step. This is the documented "`num_data_workers=0` fakes a
  GPU-idle bottleneck" trap (CHANGELOG) showing up quantitatively.
* **`workers: 1 → 8` is +9% wall throughput (6.09 → 6.64 samples/s) and removes the jitter.**

> ### ⚠ …but `num_data_workers` is NOT an output-neutral knob here. Do not just bump it.
> `utils/data_loader_multifiles.py:1031/1102` draws `torch.randn(*surface_t.shape)` per
> sample whenever `epsilon_factor > 0` — and the config sets `epsilon_factor: 0.1`, so the
> **loader adds noise inside the worker processes**. There is **no `worker_init_fn`**, so
> workers are seeded by PyTorch's default `base_seed + worker_id` and the sample→worker
> assignment depends on `num_workers`. Changing it therefore changes *which* noise each
> sample gets and **moves the loss trajectory**.
>
> The change is statistically benign (the noise is iid gaussian either way) but it is **not**
> bit-identical, so it cannot be validated by the DESIGN §4 equivalence gate — it needs a
> distributional argument. Recorded as a **finding, not a recommendation**. The clean fix is
> a seeded `worker_init_fn`, which would make the noise depend on the sample rather than on
> the worker count, and would make the knob genuinely free.

---

## 4. Where the step goes (nsys, job 7255503)

`nsys` on Polaris is **not** a module — it ships with the CUDA toolkit at
`/soft/compilers/cudatoolkit/cuda-12.9.1/bin/nsys` (Nsight Systems 2025.1.3).
Trace + sqlite: `$MEMBER_ROOT/bench/nsys_pangu_sfno_7255503.{nsys-rep,sqlite}`.
40 measured steps × 4 ranks = 160 rank-steps, eager, `--capture-range=cudaProfilerApi`.

### 4.1 NVTX ranges — **CPU-side; they do not sum to the step**

| range | n | median | mean | min | max |
|---|---|---|---|---|---|
| `data_prep` | 160 | 0.2 ms | 0.2 | 0.1 | 0.4 |
| `forward_loss` | 160 | 36.3 ms | 39.4 | 35.3 | 268.4 |
| `backward` | 160 | **280.8 ms** | 280.5 | 250.5 | 297.2 |
| `optimizer` | 160 | 18.0 ms | 18.0 | 17.1 | 18.7 |
| **step total** | 156 | **603.5 ms** | 608.7 | — | std 31.9 |

**Read this carefully.** The sub-ranges sum to **335 ms of a 603 ms step — 55%.** The other
45% is **not** missing work: these ranges are pushed/popped on the **CPU thread**, so they
measure *enqueue*, not GPU execution. CUDA is asynchronous; the trailing
`cuda.synchronize()` that closes the bench window drains whatever the GPU still owes, and
that drain sits inside `step_N` but inside none of the sub-ranges.

So: **do not read `backward = 280 ms` as "backward is 47% of GPU time."** It is 280 ms of
CPU *launch* work. Attributing GPU time requires the kernel table (§4.2), which is why both
are reported.

`ema` does not appear: `ema_warmup_epochs: 6` and these are 1-epoch runs, so EMA never
fired. **A full training run will pay it** — an every-step sweep over 1.18 B parameters —
and it is instrumented and waiting.

### 4.2 GPU kernel time — the actual finding

Aggregated from `CUPTI_ACTIVITY_KIND_KERNEL`, all 64 distinct kernels, normalised to one
rank-step (**102.9 s of kernel time over 354,720 launches / 160 rank-steps**):

| category | ms / rank-step | % GPU | launches / step |
|---|---|---|---|
| **elementwise / pointwise** | **392.0** | **61.0%** | **1506** |
| GEMM / tensor-core matmul | 97.0 | 15.1% | 351 |
| NCCL all-reduce (DDP grad sync) | 67.8 | 10.5% | 16 |
| normalization | 28.9 | 4.5% | 72 |
| optimizer (fused multi-tensor) | 25.9 | 4.0% | 59 |
| cuFFT (spherical harmonic transform) | 21.0 | 3.3% | 78 |
| other | 5.6 | 0.9% | 42 |
| reductions | 4.9 | 0.8% | 93 |
| **total** | **643.2** | 100% | **2217** |

643 ms of kernel time against a 603 ms step ⇒ **the GPU is saturated** (>100% because NCCL
overlaps compute on its own stream). Consistent with the loader verdict from the other
direction.

**The elementwise fraction is the story.** 61% of GPU time in pointwise kernels, 4× the
matmul time, at ~260 µs average — these are not launch-overhead-bound micro-kernels, they
are **large memory-bound passes**. The lever is **fusion** (fewer round-trips to HBM), not
faster matmul. That is `torch.compile`'s core competency and it is §5 rung 1.

Also measured:
* **NCCL all-reduce = 67.8 ms/step (10.5%)** over 16 calls — DDP gradient sync on a 1.18 B
  model. §5 rung 3 (bf16 comm hook) targets roughly half of this ≈ 5% of the step.
* **cuFFT = 21.0 ms (3.3%)** — the spherical-harmonic transform is *not* a hotspot. Note
  `si/bench_midway_notes.md` §3–4's standing warning: the fp32 island around the SHT is
  deliberate and must not be "optimized" to bf16.
* **H2D: 962 transfers, 348.8 ms total, 8.38 GB** across the window ≈ 2.2 ms/rank-step,
  i.e. **~0.4% of the step**. Input transfer is not a problem.

---

## 5. Memory

**26.98 GB peak of 40 GB**, identical across all three sweep runs (loader workers don't move
it). ~13 GB headroom on a 1.18 B-param model at batch 1 + bf16 + `checkpointing: 3`.

Two things follow, both **unmeasured hypotheses, flagged as such**:

1. **`batch_size` may have room** (2/GPU). Do not assume it: `bench_report.md` §II.4 records
   that on Midway "batch ≥3/card (bf16) is a trap — throughput collapses near allocator
   saturation". Measure with the sweep, watch `peak_mem_gb_max_rank`.
2. **`checkpointing: 3` is buying memory we appear not to need**, and paying recompute for
   it. Turning it down would trade the 13 GB of headroom for step time. This is a real
   candidate lever — but it is a **hot-path change** and therefore gated on DESIGN §4,
   which is not yet executable. **Not attempted.**

---

## 6. Optimizing is still blocked — and one §4.0 prerequisite turned out to already exist

Per the handoff: **profiling is unblocked, optimizing is not.** Nothing in this report
changed the hot path. `TORCH_COMPILE_MODE` is now wired and left **unset**; both PBS scripts
say why in a comment.

> **Correction (2026-07-15).** An earlier draft of this report said `TORCH_COMPILE_MODE` was
> "already plumbed in the ported harness". **It was not.** The harness port brought the
> `S2S_BENCH`/NVTX plumbing across but not the compile knob — PanguWeather had only a
> commented-out `torch.compile(self.model, mode='default')` (`train.py:639`) and no env read,
> exactly as DESIGN §2c's table says (`TORCH_COMPILE_MODE`: s2s **2**, PanguWeather **0**).
> The commented-out `export TORCH_COMPILE_MODE=…` in both bench scripts was therefore a live
> trap: uncomment it, get no compile, no error, and conclude "torch.compile doesn't help this
> model". Now genuinely wired (`get_model()`, gated, unset ⇒ legacy) with a test that fails if
> the knob is ever disconnected again.

### 6b. Fork drift vs `s2s/v2.0` — which `bench_report.md` optimizations actually reached here

DESIGN §2c's warning is that the forks share code by **copy**, so "nothing tells you the other
copy drifted". Checked rather than assumed. **The drift is bidirectional** — each fork has
something the other lacks:

| `bench_report.md` finding | `s2s/v2.0` | `PanguWeather` | on the Polaris path? |
|---|---|---|---|
| §4 **bf16** (+5.3% on H100) | env `S2S_AMP_DTYPE`, defaults **fp16** | ✅ YAML `amp_dtype`, **defaults `bfloat16`** — already on, and a better design than an env knob | ✅ yes — the green runs are bf16 |
| §4 `find_unused_parameters=False` | ✅ | ✅ | ✅ |
| §4 **`static_graph=True`** | ✅ | ❌ **missing** | ✅ would apply — **candidate** |
| `gradient_as_bucket_view=True` | ❌ **missing** | ✅ | drift the *other* way |
| §5 **`TORCH_COMPILE_MODE`** | ✅ | ❌ → ✅ **fixed here** | rung 1 |
| §4 batch **2**/card (+11.4% on H100) | config | ❌ config is batch 1 | candidate — but see §5 memory |
| per-iteration `empty_cache()` removal | ✅ | ✅ (independently) | ✅ |
| grad-norm without per-param `.item()` | ✅ (fused `grad_norm_and_max`) | ✅ (separate on-device `grad_norm`/`grad_max`) | ✅ |
| §7 checkpoint `os.path.isfile` guard | ✅ | ✅ (more call sites) | ✅ |
| §7 `--async_save` | ✅ (inference only) | ✅ (**more** files than s2s) | inference |
| NVTX `vae_encoder1/2` ranges (+ the backward-bracketing autograd trick) | ✅ | ❌ **missing** | ViT only — not on the SFNO path |

**The two real gaps are `static_graph=True` and (until now) the compile knob.** Everything
else either landed independently or is *ahead* in PanguWeather.

> `static_graph=True` is a **candidate, not a known win.** `bench_report.md` §4 changed bf16
> and `find_unused_parameters=False`+`static_graph=True` **together** and attributes +5.3% to
> the pair, so `static_graph`'s isolated contribution was never measured — and PanguWeather
> already has the expensive half (`find_unused_parameters=False`). Also: s2s needs a
> **dead-module freeze** (`layer_perturbation2`, `layer_purturbation_e2`, `train.py:437-444`)
> for `static_graph` to be legal; PanguWeather has **no such freeze**, so this cannot simply
> be copied across. Measure it, don't assume it.

### 6c. The ViT/Swin optimizations: there are none to port, and they are not on this path

Asked directly: have `bench_report.md`'s transformer/ViT optimizations been done in
PanguWeather? **Two independent reasons the answer is "the question doesn't bite yet":**

1. **They were never implemented in *either* fork.** `bench_report.md` §3's ViT findings —
   LayerNorm-backward is the 2nd-largest GPU consumer (17.3 s), memory-layout conversions
   (4.4 s), `roll` for shifted-window attention (7.7 s), matmul only 6th/18th — are
   **profiler observations, explicitly deferred to `torch.compile`** (§3: "Fusing LayerNorm
   via torch.compile is the single biggest optimisation the profile suggests"; §5: "in
   progress"). §5-ladder rung 2 (FlexAttention) is unstarted. A `diff` of the two
   `networks/pangu.py` files shows **the only perf-relevant divergence is s2s's NVTX
   instrumentation** — `F.scaled_dot_product_attention` is already in `EarthAttention3D` in
   **both** (s2s:1091/1099, PanguWeather:1079/1087), both have the same 2 `torch.roll`s and
   13 `LayerNorm`s, and both have transformer-block checkpointing commented out identically.
   So there is nothing to port: the ViT cores agree.
2. **The ViT does not run on Polaris.** The green E3SM path is `nettype: sfno_plasim`, which
   builds `networks/modulus_sfno/sfnonet.py` and **never touches `networks/pangu.py`**. The
   Swin/ViT (and its VAE, and this report's `vae_noise` hook) belong to `pangu_plasim`,
   blocked on PLASIM h5 that is not staged.

**But the two profiles agree on the lever, which is the interesting part.** The H100 ViT
(`bench_report.md` §3/§5: element-wise ops "the single largest GPU-time consumer … launched
as hundreds of small kernels", matmul ranks 6th) and this A100 SFNO (§4.2: **61%** elementwise
over ~1506 launches/step, GEMM 15%) are **different architectures that profile the same way** —
memory-bandwidth bound and fusion-starved. Two independent measurements, one conclusion:
`torch.compile` is rung 1. That is now reachable on PanguWeather.

Status of the three DESIGN §4.0 prerequisites **for PanguWeather** (they were tracked for
`s2s/v2.0`; the trees are forks and share nothing, DESIGN §2c):

| prerequisite | state on PanguWeather |
|---|---|
| seed mechanism | ✅ **already existed — do not port `s2s`'s `seeding.py` here.** `train.py:3825` has `--global_seed` (default 0) feeding `seed_torch()` (`:3742`), called at `:3785`. It seeds `PYTHONHASHSEED`/numpy/torch/CUDA and sets `cudnn.benchmark=False` + `cudnn.deterministic=True`. It is **stronger than s2s's legacy path was** — the numpy gap that made s2s's baselines irreproducible does not exist here. Two competing seed mechanisms would be a regression. Gaps: Python's `random` is unseeded, and `torch.use_deterministic_algorithms(True)` is never set. |
| `tiny_baseline.yaml` | ✅ **written AND run — measured, not asserted** (job **7255583**, rc=0, `--config=TINY`, 1 GPU). See the table below. |
| VAE noise-fixing hook | ✅ **built** — `utils/vae_noise.py` + 16 tests (`VAE_NOISE_OK`). **But INERT on this path**: `sfno_plasim` is deterministic and has no VAE (DESIGN §2c). It gates `pangu_plasim`, which is blocked on PLASIM h5 that is not staged. |

### `tiny_baseline.yaml` is genuinely small — the measured delta

"Tiny" is a measurement, not a name. That is the whole `test.yaml` lesson (CLAUDE.md #12: a
config called *test* that is really the full model), so this was measured rather than sized
by arithmetic:

| | real SFNO (7255410) | **TINY** (7255583) | ratio |
|---|---|---|---|
| trainable params | 1,182,108,160 | **7,166,656** | **165× smaller** |
| step_med | 0.652 s | **0.023 s** | 28× faster |
| peak memory | 26.98 GB | **1.00 GB** | 27× less |
| step_std | 0.110 | 0.00015 | — |

A K=20-step §4.1 baseline is **~0.5 s of compute** and fits in 1 GB. It reports
`loader_wait_frac = 76.7%` — expected and harmless: the model is now so fast it outruns the
`num_data_workers: 0` loader §4.1 asks for. Irrelevant for an equivalence baseline (which
measures numbers, not speed), and a third independent confirmation that `loader_wait_frac`
tracks reality — it has now read 0.0%, 0.7%, 14.8% and 76.7% in the four situations where
each was the right answer.

**So all three §4.0 prerequisites are now met on PanguWeather.** What remains is to *capture*
the baseline (§4.1: fixed seed, world size 1, K=20 per-step loss trajectory + output summary
stats) — no longer blocked on building anything.

> **`cudnn.benchmark=False` + `deterministic=True` are already on** (via `seed_torch`, which
> always runs). That is a *performance* fact hiding in a reproducibility mechanism: this model
> is benchmarked in cuDNN's deterministic mode. Turning benchmark on might buy time, and would
> cost reproducibility. Not attempted — it is a hot-path change. Recorded so it is not
> mistaken for an oversight.

---

## 7. How to reproduce

```bash
cd PanguWeather/v2.0
qsub HPC_scripts/polaris_bench_e3sm_sfno.pbs                    # CSV bench, 4-GPU
qsub -v NUM_DATA_WORKERS=8 HPC_scripts/polaris_bench_e3sm_sfno.pbs
qsub HPC_scripts/polaris_bench_nsys_e3sm_sfno.pbs               # nsys trace + sqlite
qsub -v CONFIG_NAME=tiny_baseline,CONFIG_SECTION=TINY,NPROC=1 \
     HPC_scripts/polaris_bench_e3sm_sfno.pbs                    # the §4.0 small config

# tests — no GPU/data/cluster needed
python PanguWeather/v2.0/test/bench_instrumentation_test.py     # BENCH_INSTR_OK (9)
python PanguWeather/v2.0/test/vae_noise_test.py                 # VAE_NOISE_OK  (16)
```

**PASS is the work token, never `rc=0`** (the makani lesson: a resumable trainer exited 0
having trained zero steps). The CSV bench gates on the CSV **gaining a row**
(`ERROR NO_BENCH_ROW`); the nsys script gates on the trace containing **bench NVTX ranges**
(`ERROR NO_NVTX_TABLE` / `ERROR NO_BENCH_RANGES`) — nsys writes a `.nsys-rep` even when it
captured nothing, so file existence proves nothing.

Analysis: `python s2s/v2.0/HPC_scripts/parse_nsys.py <trace>.sqlite` (the nsys script runs it
automatically).

---

## 8. What is NOT established here

Stated plainly so nothing below reads as done:

* **SI, makani and physicsnemo are unprofiled on Polaris.** Only PanguWeather SFNO.
* **No baseline is captured.** `tiny_baseline.yaml` now runs (job 7255583), but the §4.1
  capture itself — fixed seed, world size 1, K=20 loss trajectory + output stats written to
  `baselines/` — has not been done. That is the next job, and it is no longer blocked.
* **No optimization was attempted or measured.** No `torch.compile`, no precision change, no
  DDP tuning, no `checkpointing` change. The §4 gate is not executable yet.
* **The `workers=8` +9% is a bench result, not an endorsement** — see the `epsilon_factor`
  box in §3. It changes the loss trajectory.
* **Single-run numbers.** Each sweep point is one job. The step distributions are tight
  (`workers=8` std = 0.3 ms) so within-run precision is high, but run-to-run variance across
  *nodes* was not measured.
* **The nsys numbers are eager and 40 steps**, vs the CSV bench's 80. The step medians agree
  (603.5 ms NVTX vs 602–652 ms CSV), which is the cross-check.
* **`ema` never fired** (`ema_warmup_epochs: 6`). Full training pays a per-step sweep over
  1.18 B params that these runs did not.

---

## Decisions / changes log

* **2026-07-15** — **Ported the `S2S_BENCH` + NVTX harness into PanguWeather** (it had zero).
  Range names and CSV columns byte-identical to s2s's (CLAUDE.md #10); `ema` added as a new
  range and to `parse_nsys.py`'s list (which skips absent ranges, so Midway traces are
  unaffected). Gated so unset knobs ⇒ legacy path byte-for-byte.
  **Proven, not asserted:** job **7255505** (no `S2S_BENCH`) reproduced the GREEN reference
  **7253591** exactly — train loss **0.3411**, valid_loss **0.7049359679222107**,
  bit-identical. Static gating tests are not a substitute for that smoke.
* **2026-07-15** — **`cpu_prep_frac` is not loader idle; built `loader_wait_*`.** It measures
  `_prepare_inputs_batch` on an already-fetched batch (0.3–0.6% of the step even at
  `workers=0`). The real fetch is between steps, in no step window — and it drove the
  elapsed-vs-sum self-check into *refusing the row* on exactly the runs where the loader is
  the finding. Now measured and reconciled. **Falsified before being believed**: `workers=0`
  moved it 0.7% → 14.8% (21×) while `cpu_prep_frac` stayed flat.
* **2026-07-15** — **`elapsed` was being sampled after `cudaProfilerStop()`.** The profiler's
  buffer flush landed inside the measured wall time; under nsys job **7255503** that read
  `elapsed=51.8 s` against `sum=25.7 s` (50% "disagreement") and discarded a perfectly good
  bench row — on **every** profiled run. The timers were fine; the clock was stopped in the
  wrong place. Fixed (`loop_end` sampled first) + an AST test pinning the ordering.
* **2026-07-15** — **VERDICT: GPU-bound.** `loader_wait_frac` = **0.7%** at the shipped
  `num_data_workers: 1` (job 7255410). The §5 kernel ladder is **not** premature.
* **2026-07-15** — **VERDICT: elementwise-bound, not matmul-bound.** 61% of GPU time in
  pointwise kernels over ~1506 launches/step vs 15% in GEMM (job 7255503). Memory-bandwidth
  bound and fusion-starved ⇒ `torch.compile` (§5 rung 1) is the right first lever, now on
  evidence rather than assumption. NCCL is 10.5% (⇒ §5 rung 3 ≈ 5%); cuFFT/SHT is only 3.3%.
* **2026-07-15** — **The model is 1,182,108,160 params**, not the ~79M the docs assume — that
  figure is the Pangu/Swin model, not the E3SM SFNO. 26.98 GB peak of 40 GB.
* **2026-07-15** — **PanguWeather already has a seed knob** (`--global_seed` → `seed_torch`,
  seeding numpy + torch + CUDA and forcing `cudnn.deterministic`). The handoff implied it
  needed porting from `s2s/v2.0/utils/seeding.py`; it does **not**, and porting would create
  two competing mechanisms. This also explains why the 0.3411 green is bit-reproducible.
* **2026-07-15** — **`num_data_workers` is not output-neutral**: the loader draws per-sample
  gaussian noise inside the workers (`epsilon_factor: 0.1`) with no `worker_init_fn`, so the
  worker count changes the noise realization and moves the loss. The `1 → 8` (+9%, and 10×
  less jitter) is recorded as a **finding, not a recommendation**; the clean fix is a seeded
  `worker_init_fn`.
