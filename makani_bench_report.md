# makani-SFNO on Polaris — benchmark results

*Measured evidence for makani-SFNO on E3SM, 4× A100-SXM4-40GB nodes.* Companion to
`polaris_bench_report.md` (PanguWeather) and `si/bench_midway_notes.md` (SI).
Written 2026-09-01 from the CSVs and job logs, replacing `makani_multinode_ddp_plan.md`
(deleted — its live content is §7-§9 here).

Everything below is **one rep per configuration**. Nothing here is a rep-averaged
result, and §8 says what that costs.

Raw data:
`$MEMBER_ROOT/bench/makani_multinode_scaling.csv` (30 rows, 15 with timings),
`makani_production.csv` (1 row), `makani_wandb_check.csv` (7 rows);
logs `makani_sfno/makani_mn_scaling.o<jobid>` (launcher header) and
`$MEMBER_ROOT/runs/makani_mn_scaling/*.log` (trainer).

---

## 0. How to read a row — three traps, all of which have already bitten

1. **`step_ms` is the FINAL EPOCH's mean step time**, not the run's. The parser
   (`polaris/parse_makani_scaling.py`) takes the *last* `Average step time after step N`
   line. For a 60-step single-epoch arm that average **includes warmup**; for a
   multi-epoch run it does not. The difference is 60-100 ms/step — larger than most of
   the effects being measured (§3c).
2. **The fabric stack is part of the configuration, and it is not a CSV column.** It is
   printed in the launcher header as `fabric pin:`. Old-plugin and new-plugin rows must
   never share a table (§3a vs §3b differ by up to 3.4× at the same node count).
3. **PASS is `MAKANI_MN_SCALING_OK` plus a CSV row with a timing**, never `rc`. Half the
   rows in the scaling CSV carry no timing on purpose (§6).

## 1. What was measured

| item | value |
|---|---|
| hardware | Polaris, 4× A100-SXM4-40GB/node, HPE Slingshot 11 |
| software | torch **2.8.0** / NCCL **2.28.3+cuda12.9** / torch_harmonics 0.9.2a / makani 0.2.0 / h5py 3.16.0 (overlay) |
| env | `polaris_makani_env.sh`, `MAKANI_ENV_SOURCE=manual-reconstruction` on every row (the sanctioned `module load conda` has never worked since 2026-08-20) |
| launcher | `makani_sfno/polaris/polaris_makani_multinode_scaling.pbs`, one script for every node count |
| model | SFNO `embed_dim 384`, `num_layers 8`, E3SM 180×360 — **147,818,882 params** (53-ch) / **147,863,863** (ALLDATA) |
| DDP shape | `parameters_reduction_buffer_count 1` — makani reduces in **one bucket**, ≈**591 MB** fp32 per step |
| scaling mode | **weak** — local batch pinned at 1 sample/GPU, so global batch = 4 × nodes and per-GPU arithmetic is constant |
| rank↔GPU placement | ⚠ **`GPU_ORDER=default` on all 30 rows, production included.** Polaris' GPU↔NUMA map is **reversed** (`dev0`→NUMA 3 … `dev3`→NUMA 0, job 7531456), and the **mandatory** `--cpu-bind depth -d 8` puts local rank 0 on NUMA 0 — whose GPU is `dev3`. The two required settings therefore combine to place **every rank maximally far from its own GPU**. `-v GPU_ORDER=reverse` (`CUDA_VISIBLE_DEVICES=3,2,1,0`) fixes it and **has never been run** (prereg 5, §8) |

**Three packs, and they are not interchangeable** (a pack change alone moved the
single-node step time 41%, §2):

| pack | contract | train samples |
|---|---|---|
| `e3sm_makani` | 53-ch smoke | 400 |
| `e3sm_makani_scaling` | 53-ch (58 in / 53 out), 2 years | 2,920 |
| `e3sm_makani_alldata_production` | **ALLDATA 101-ch**, Pangu/ai-rossby parity, 30 years | 43,800 |

**Two fabric stacks:**

* **OLD** — `/soft/libraries/aws-ofi-nccl/v1.6.0-libfabric-1.22.0` + cray libfabric 2.3.1.
  The only `/soft` pairing that initialises at all (probe 7553823, 1 of 6).
* **NEW** — self-built `$MEMBER_ROOT/sw/aws-ofi-nccl-1.21.1` + `OFI_NCCL_PROGRESS_MODEL=AUTO`
  + cray 2.3.1. The env var is the whole unlock: without it libfabric 2.3.1's CXI provider
  refuses `fi_domain` with ENOSYS (matrix 7563894, 1 of 6 combos).

## 2. Single node — one arm, six numbers, a 3.0× spread

Every row is 1 node / 4 ranks / global batch 4 on the same model and the same node class.

| job | step_ms | plugin | `NCCL_PROTO` | pack | epochs × steps | wandb |
|---|---|---|---|---|---|---|
| 7553836 | **196.5** | old | default | smoke 400 | 1 × 60 | off |
| 7553890 | **114.9** | old | default | scaling | 1 × 60 | off |
| 7554222 | **144.7** | old | Simple | scaling | 1 × 60 | off |
| 7564184 | **115.3** | new | Simple | scaling | 1 × 60 | off |
| 7564401 | **147.7** | old | Simple | scaling | 1 × 60 | **on** |
| 7564492 | **65.3** | old | Simple | scaling | **2** × 60 | on |

What each delta is worth, and which have a mechanism:

* **Pack: −41.5%** (196.5 → 114.9). `io_gbs` 0.52 → 0.88 — the 400-sample pack starves the
  loader. Mechanism established; this is why every arm in a table must name its pack.
* **Warmup: −55%** (144.7 → 65.3, ≈147.7 → 65.3 with wandb held on). The second epoch is the
  steady state; the first 60-step epoch is dominated by warmup. Mechanism established, and
  it is the largest single effect on this page.
* **`NCCL_PROTO=Simple` on the old plugin: +25.9%** (114.9 → 144.7). LL/LL128 genuinely help
  intra-node, so the pin that makes ≥3 nodes work costs a quarter of the single-node step.
* ⚠ **Old vs new plugin at one node: −20.3%** (144.7 → 115.3) — **no mechanism.** At one node
  NCCL never leaves NVLink and the network plugin is not on the path. Both rows are n=1 and
  ran on different hosts. Until it is reproduced this gap is the honest **noise floor for
  single-node makani rows**, and it is bigger than several effects in §3.

## 3. The ladder — 1 / 2 / 4 / 8 nodes, 53-channel pack

### 3a. OLD plugin + `NCCL_PROTO=Simple` — the ladder of record

60 steps, one epoch, no wandb. Every row `AWS Libfabric`, `world_size` verified.

| nodes | ranks | job | step_ms | vs 1n | samples/s total | weak eff | wireup s | io GB/s |
|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 7554222 | 144.7 | — | 27.6 | 100% | 2.76 | 0.70 |
| 2 | 8 | 7554241 | 145.7 | +0.7% | 54.9 | 99.3% | 8.54 | 1.39 |
| 4 | 16 | 7554216 | 199.5 | +37.9% | 80.2 | 72.5% | 21.38 | 2.03 |
| 8 | 32 | 7564288 | 215.3 | +48.8% | 148.6 | 67.2% | 25.14 | 3.77 |

The cliff is at 4 nodes and it **saturates** (4→8 costs a further +7.9%, not another ×). Wireup
grows sublinearly, 2.8 → 25.1 s. Throughput never goes backwards.

⚠ **Provenance footnote:** 7554216 is the run that *discovered* the Simple pin, so it was
passed by hand (`-v NCCL_PROTO=Simple`) before the launcher echoed the pin in its header.
Its header therefore does not carry the `NCCL_PROTO=Simple` line that 7554222 onward do; the
pin is attested by the submission and the CHANGELOG, not by the log. Re-run it when reps are
taken.

### 3b. NEW plugin (v1.21.1 + AUTO) + Simple — everything works, inter-node is 2.3× worse

| nodes | ranks | job | step_ms | vs 1n | samples/s total | weak eff | wireup s | io GB/s |
|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 7564184 | 115.3 | — | 34.7 | 100% | 3.17 | 0.88 |
| 2 | 8 | 7564264 | 490.7 | +326% | 16.3 | 23.5% | 14.03 | 0.41 |
| 4 | 16 | 7564185 | 460.5 | +299% | 34.7 | 25.0% | 12.50 | 0.88 |
| 8 | 32 | 7564137 | 545.0 | +373% | 58.7 | 21.2% | 13.41 | 1.49 |

**Fastest single node, worst everything else.** The collapse is at the *first hop* and then
flat (490.7 → 460.5 → 545.0): a large, roughly constant cost for touching the fabric at all,
not a bandwidth wall. 16 GPUs at 4 nodes deliver exactly 4 GPUs' throughput. Per-GPU compute
is ~115 ms, so **~430 ms ≈ 79% of the 8-node step is exposed comms.**

Signature matches progress-engine CPU starvation: `OFI_NCCL_PROGRESS_MODEL=AUTO` runs
libfabric's own progress threads on the 8 cores `--cpu-bind depth -d 8` reserves — a bind
tuned for the old plugin's *manual* progress. Untested; it is the single highest-value
tuning experiment left (§8).

**Trade-off, stated plainly:** old plugin = fast pure DDP, spatial parallelism broken, and its
"working" regime is a message-size lottery (§6). New plugin = correct everywhere, 2.3× slower
inter-node at ≤8 nodes. **Production runs on the new plugin** and pays for it — but §4 shows
the penalty is amortised at production step lengths.

### 3c. The warmup-free read — and the correction it forces

Same plugin, proto, pack and node counts as §3a; **2 epochs instead of 1**, so `step_ms` is
epoch 2 and carries no warmup. wandb is on (adds per-iteration diagnostics), so these are
*upper bounds* on the steady-state step time.

| nodes | job | step_ms (ep 2) | vs 1n | weak eff | §3a − §3c |
|---|---|---|---|---|---|
| 1 | 7564492 | 65.3 | — | 100% | +79.4 ms |
| 2 | 7564493 | 88.4 | **+35.4%** | 73.9% | +57.3 ms |
| 4 | 7564555 | 101.5 | **+55.4%** | 64.3% | +98.0 ms |
| 8 | 7564566 | 138.3 | **+111.8%** | **47.2%** | +77.0 ms |

⚠ **This overturns the headline of §3a.** The warmup-inclusive ladder adds 57-98 ms/step to
*every* rung, which is a near-constant that **compresses the relative penalty** and flatters
efficiency. Read warmup-free, the same stack scales to **47% at 8 nodes, not 67%**, and the
much-quoted **"the first Slingshot hop is free (+0.7%)" does not survive: it is +35%.**

Not decisive, and it must not be re-quoted as if it were: §3c also turns wandb on, so it is a
two-variable comparison. **One cheap job settles it** — §3a's rungs at `EPOCHS=2`, wandb off
(≤10 nodes, `debug-scaling`, ~30 min). Until then, neither efficiency curve should be
published; the *shape* (cliff then saturation) is what both agree on.

## 4. The 128-node production run — the deliverable

**Job 7566145, `prod128_alldata_v2`: 128 nodes / 512 ranks, 100/100 epochs, rc=0,
`MAKANI_MN_SCALING_OK`.** The first full makani-E3SM production training in this project,
on the 101-channel ALLDATA contract that matches Pangu/ai-rossby.

| | |
|---|---|
| shape | 512 ranks × 1 sample/GPU = **global batch 512**, `h_par=w_par=1` (pure DDP) |
| stack | **new plugin** v1.21.1+AUTO + Simple, `AWS Libfabric`, `world_sizes_seen=512` |
| data | `e3sm_makani_alldata_production`, 43,800 samples; FULL epochs of 43,520 = **85 steps** |
| wall | train **5,749.2 s** + wireup **315.7 s** ⇒ **≈216 node-hours** (cap was 774) |
| cost/epoch | 57.5 s ⇒ **2.04 node-hours per epoch** |
| step time | CSV row **576.5 ms** (= epoch 100). Across 100 epochs: **median 603.6**, mean 604.4, min 552.5, max 666.8 (epoch 1) |
| throughput | **888 samples/s** at epoch 100; median 848 |
| I/O | 43.1 GB/s at epoch 100; median 41.2, max 45.0 aggregate |
| memory | **8.36 GB/GPU of 40** — 4.8× headroom |
| artifacts | `best_ckpt_mp0.tar` (= epoch 100) + 3 versioned checkpoints @ 1.77 GB; wandb run with the 102-key Pangu-parity contract |

**Training result — validation minimum is the LAST epoch:**

| epoch | 1 | 2 | 10 | 25 | 50 | 75 | 100 |
|---|---|---|---|---|---|---|---|
| train loss | 0.3072 | 0.0834 | 0.0336 | 0.0244 | 0.0186 | 0.0172 | **0.01598** |
| valid loss | 0.1080 | 0.0678 | — | — | — | — | **0.018297** |
| grad norm | 0.432 | — | 0.0485 | — | 0.0195 | — | 0.0117 |

Monotone descent to the schedule bound with **no overfit onset** — the opposite shape from
the Pangu/ai-rossby runs' plateau-then-drift at ~epoch 25. ⚠ Absolute loss values are **not**
cross-harness comparable; the shared per-channel lwrmse panels are the comparison vehicle.
⚠ **Global batch 512 at the shipped LR is a training-regime change made on operator
instruction and has NOT been signed off by the science owner.**

### 4a. 8 → 128 nodes: the penalty amortises

The only like-for-like multi-node pair on this contract (same pack, same stack, wandb on
both, `step_ms` warmup-free on both):

| nodes | ranks | job | step_ms | samples/s total |
|---|---|---|---|---|
| 8 | 32 | 7565972 | 492.7 | 64.9 |
| 128 | 512 | 7566145 | 576.5 (median 603.6) | 888.1 |

**Weak-scaling efficiency 8→128 = 85.5%** on the CSV metric (81.6% against the 100-epoch
median), for a **16× rank increase** and ×13.7 throughput.

**Why this is so much better than §3b's 21%:** the ALLDATA model has essentially the *same*
parameter count as the 53-channel one (147.86 M vs 147.82 M ⇒ the same ~591 MB reduction) but
~3.5× the per-step compute. The multi-node cost is roughly fixed, so **efficiency rises with
step length.** Corollary for planning: makani scales well *because the production step is
long*, not because the fabric is fast — and any optimisation that shortens the step will
give back some of this efficiency.

The ~591 MB reduction is also why makani never hit the tree-all-reduce corruption that
blocked ai-rossby (which reduces 4.73 GB, above the ~1 GB threshold where NCCL switches
Ring→Tree). makani needs **no `NCCL_ALGO=Ring` pin**; ai-rossby cannot run without it.

## 5. Spatial (model) parallelism — works, barely exercised

| job | nodes | h×w | global batch | plugin | step_ms |
|---|---|---|---|---|---|
| 7554351 | 1 | 2×2 | 1 | old + Simple | 371.5 |
| 7563780 | 1 | 2×2 | 1 | old + Simple (+ `_serialized_sync_params`) | 219.5 |
| 7564035 | 4 | 2×2 | 4 | **new** + Simple | 569.9 |

* Multi-node spatial is **impossible on the old plugin** — it loses progress under
  small-message storms on subgroup communicators at ≥3 nodes (IMA 7554253, hang 7554367,
  serialized-and-still-hung 7563723 with `enqueued 84, completed 83`). The new plugin fixes it.
* `_serialized_sync_params` (in `plasim_trainer.py`, setup-path only, output-neutral) is kept
  as NCCL-contract hygiene — upstream's own call site carries `# DEBUG: this also needs to be
  fixed in NCCL`. It is **not** what unblocked 4 nodes.
* ⚠ **371.5 vs 219.5 is a 1.69× spread between two runs whose recorded configuration is
  identical** (same plugin, proto, pack, batch, GPU order). Both n=1. No spatial number here
  should be quoted until this is reproduced.

## 6. What the refused rows say — 15 of 30 rows carry no timing, by design

The parser refuses a row rather than writing a plausible number (`NO_STEP_TIMING` → `csv_rc=4`).
The refusals are the failure catalogue:

| class | jobs | cause |
|---|---|---|
| stale fabric pin | 7553811 (rc=139), 7553824 | `/opt/cray/libfabric/2.2.0rc1` vanished; **a nonexistent dir on `LD_LIBRARY_PATH` is ignored, not an error**, so the pin no-opped → `fi_domain` ENOSYS |
| 4-node LL deadlock | 7553891, 7554143, 7554185 | old plugin's LL/LL128 paths wedge vs NCCL 2.28.3 at ≥3 nodes → `NCCL_PROTO=Simple` |
| spatial hang/IMA | 7554129, 7554253, 7554367, 7563723 | old plugin, subgroup small-message storms (§5) |
| sick node (zombie GPU) | 7563960, 7563991, 7564075, 7564123, 7564227 | `CUDA-capable device(s) busy` at init → `-v TARGET_NODES=N` with `select=N+1` prunes it |
| harness | 7564377 | `NO_STEP_TIMING`, rc=143 |
| message-size lottery | 7565896 (in `makani_wandb_check.csv`) | old plugin wedged on a **41,088-element = 384×107** broadcast, the ALLDATA encoder weight — the size the 53-ch model (384×58) happened to dodge. **This disqualified the old plugin for production**, and it is why the faster stack is not the production stack |

Two failures were only catchable because the guards exist: `world_size` is read from the
trainer's own banner (N independent `world_size=1` trainers would otherwise time plausibly),
and the epoch-wrap guard refuses a run that re-serves cached samples.

## 7. Prereg, scored

Recorded before the first multi-node job (in the deleted plan, §4). Three scored, two never run.

| # | prediction | verdict |
|---|---|---|
| 1 | multi-node is not free: 4-node step ≥ +10% vs 1 node | ✅ **HIT** — +37.9% (and +48.8% at 8). At **2** nodes the answer depends on the metric: +0.7% warmup-inclusive, +35.4% warmup-free (§3c) — the disagreement is itself the finding |
| 2 | `transport` reads `AWS Libfabric` on every multi-node row | ✅ **HIT** — every row except 7553811, which is the ENOSYS failure the column exists to expose |
| 3 | 4-node `wireup_s` > 2× 1-node | ✅ **HIT, large** — 21.38 vs 2.76 = **7.7×** |
| 4 | synthetic-data arm degrades less than real (separates I/O from comms) | ⛔ **never run** |
| 5 | `GPU_ORDER=reverse` (NUMA-local pairing) within 5% of default | ⛔ **never run** |

### 7a. Prereg — the `h4w4` / batch-32 arm, recorded before submission (2026-09-01)

Upstream reaches 512-1024 GPUs at data-parallel batch **16-32** by spending ranks on
ensemble × spatial parallelism (`fourcastnet3.yaml`: pretrain1 `e16 × b16 × h2w2` = 1024;
pretrain2 `e2 × b32 × h2w4` = 512, *"to fit into memory on 80GB GPUs"*). Our production run
spent all 512 ranks on data parallelism instead ⇒ batch 512, 16× pretrain2, and 8,500
optimizer updates. Since `GLOBAL_BATCH = LOCAL_BATCH × NRANKS/(HPAR·WPAR)`, `h4w4` +
`LOCAL_BATCH=32` reproduces the batch-32 regime on **4 nodes**. 16-fold spatial has never
been run here — only `h2w2` (§5).

1. **It initialises and trains** — 60/60 steps, `MAKANI_MN_SCALING_OK`. *Falsified by* a hang
   or IMA, which would kill batch-32-by-sharding and force the batch down on pure DDP instead.
2. **Memory fits** — peak < 30 GB/GPU. Estimate ~17 GB: production held 8.36 GB at batch 1
   unsharded, and each rank here holds 1/16 of the domain for 32 samples ≈ 2 sample-equivalents.
   *Falsified by* OOM ⇒ retry at `LOCAL_BATCH=16` (pretrain1's batch).
3. **Step time lands in 1.5-6 s** — arm F's `h2w2` at 4 nodes was 569.9 ms for batch 4; this
   carries 8× the samples on 4× the split. *Falsified outside the band*, meaning the cost model
   is wrong, not the run.
4. **The loss is sane** — first steps O(1-3) and descending within the epoch, matching the
   production run's opening (2.14 → 0.89 over its first six steps). *Falsified by* NaN or a flat
   loss ⇒ the spatial split changes **what** is computed, not just where, and that is a
   correctness bug rather than a placement question.
5. **`gpu_order=default`, deliberately** — this arm does not touch placement, so it stays
   comparable to arm F. The placement axis is a separate, still-unmeasured arm (§8).

**Paper context, kept from the deleted plan:** FourCastNet 3 (arXiv:2507.12144 §E.2) trains
pretrain-2 on **512 A100** as `ensemble 2 × batch 32 × h2w4`. The production run above is also
512 A100 — but as **pure data parallelism** on a different model and dataset. It reproduces the
paper's *rank budget*, not its decomposition, and no table from it should be captioned as an
FCN3 reproduction.

## 8. Not measured — and what each would cost

| gap | why it matters | cost |
|---|---|---|
| **Reps 2-3 of any ladder** | every number here is n=1, and §2 shows a 20% unexplained single-node gap | ~3 jobs/rung, ≤10 nodes, `debug-scaling` |
| **A warmup-free, wandb-off ladder** | decides §3a vs §3c, i.e. whether the first hop is free or +35% | 4 jobs, ≤1 h |
| **cpu-bind / progress-thread sweep on the new plugin** | ~79% of the 8-node step is exposed comms (§3b); the bind was tuned for the old plugin's manual progress | 3-4 jobs at 4 nodes |
| **`nsys` per-rank capture (`-v NSYS=1`)** | makani has **no kernel-level profile at all**; rank-0 logging cannot answer where the step goes | 1 job (truncated by design — `exit_on_stop`) |
| **arm D (synthetic)** | separates an I/O loss from a comms loss | 1 job |
| **arm E (`GPU_ORDER=reverse`)** | every row here — and the production run — placed each rank on the NUMA node *farthest* from its GPU (§1). It is a placement change, not an arithmetic one, so it needs no equivalence gate, and it is the standing candidate for the host-CPU stall in `polaris_bench_report.md` §4.4e. **It matters more under spatial parallelism**, where intra-node halo traffic rides that same distance every step. ai-rossby queued the identical test and both arms were refused (7577036 `rc=134`, 7577166 `rc=143`), so the axis has **zero measurements in either harness** | 1 job + 1 control |
| **A DESIGN §4 equivalence baseline** | no hot-path change may be committed without one; makani has none | 1 short job |
| **Inference / evaluation on `best_ckpt_mp0.tar`** | a trained model nobody has scored | not yet scripted |
| **`omp_threads=64`** | all 30 rows ran at **8× CPU oversubscription** on the same cores the progress engine uses (PBS exports `OMP_NUM_THREADS=<ncpus>`; the launcher's `${OMP_NUM_THREADS:-1}` idiom never overrode it). Comparability is intact — the value is constant on every row — but the absolute numbers were taken under it | deliberately **not** changed: flipping it makes future rows incomparable with all 30. Owner's call, and it interacts with the cpu-bind sweep above |

## 9. Operational facts that outlive the plan

**Seven launchers are still dead.** They open with the bare `module load conda` /
`conda activate base` pair and fail with `conda: command not found`:

```
makani_sfno/polaris/polaris_sfno_smoke.pbs          polaris_sfno_alldata_smoke.pbs
makani_sfno/polaris/polaris_sfno_full.pbs           polaris_sfno_alldata_full.pbs
makani_sfno/polaris/polaris_sfno_full_probe.pbs     polaris_pack_e3sm_full.pbs
makani_sfno/polaris/polaris_pack_e3sm_alldata_full.pbs
```

Verified still true 2026-09-01. This includes **both data packers**, so the green results they
produced (7253465, `CONVERT_OK` 7252728) are not reproducible today. The revival is mechanical
— replace the module pair with, **in this order** (the flip is load-bearing):

```bash
source "${MAKANI_ROOT}/../polaris_env.sh" || exit 2          # FIRST: defines MEMBER_ROOT
source "${MAKANI_ROOT}/polaris/polaris_makani_env.sh" || exit 2
```

`polaris_pack_e3sm_scaling.pbs` and `polaris_pack_alldata_production.pbs` already demonstrate
the swap on real work. Each of the seven also needs the h5py overlay, since they all import
makani → h5py.

**Queue geography** (verified `qstat -Qf`): `debug` ≤2 nodes, `debug-scaling` ≤10 nodes / 1 h /
1 job per user; `prod` routes ~16→`small`, 25-99→`medium`, **100-496→`large`**. Settle tuning
at ≤10 nodes; climb with the settled config only.

**Spare-node preflight is mandatory at scale.** `-v TARGET_NODES=N` with `-l select=N+1` runs a
per-node 4-GPU touch, names sick hosts, prunes them and runs on the first N healthy nodes. At
128 nodes a bad draw is near-certain; the production run passed 129/129.

**Pack arithmetic** for future rungs, at 60 steps and 1 sample/GPU (samples needed = 240 ×
nodes): 12 nodes → 2,880 (the 2-year scaling pack covers it); 32 → 7,680; 100 → 24,000. The
30-year ALLDATA production pack (43,800) covers **182 nodes** at 60 steps and is the pack to
use for anything above 12.

## 10. Provenance

| what | job | log |
|---|---|---|
| §2 single-node arms | 7553836, 7553890, 7554222, 7564184, 7564401, 7564492 | `makani_sfno/makani_mn_scaling.o<jobid>` |
| §3a old-plugin ladder | 7554222, 7554241, 7554216, 7564288 | ” |
| §3b new-plugin ladder | 7564184, 7564264, 7564185, 7564137 | ” |
| §3c warmup-free ladder | 7564492, 7564493, 7564555, 7564566 | ” |
| §4 production | **7566145** | `.o7566145` + `$MEMBER_ROOT/runs/makani_mn_scaling/prod128_alldata_v2.log` |
| §4a 8-node ALLDATA | 7565972 | ” |
| §5 spatial | 7554351, 7563780, 7564035 | ” |
| fabric probes | 7553823 (6 pairings), 7563854 (build), 7563894 (knob matrix), 7554170 / 7563925 (app-free NCCL) | `makani_sfno/*.o<jobid>` |

Narrative and the diagnosis chains behind each of these live in **CHANGELOG.md**
(2026-08-23 → 2026-08-27). Priorities live in **TODO.md**.
