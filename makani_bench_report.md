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
4. **`samples_s_total` was wrong on model-parallel rows until 2026-09-01.** The parser used
   `local_batch × ranks`, but makani splits `--batch_size` across the **data** group, so under
   spatial parallelism `local_batch` is per data group and there are only `ranks/(h_par·w_par)`
   of them — every spatial row overstated throughput by exactly `h_par·w_par`. Fixed to derive
   from `global_batch`, which is an identity on `h1w1` rows: the 30-row scaling table and the
   production row are bit-unchanged, and the 5 spatial rows were corrected in place. If you
   have a copy of a spatial row taken before that date, **divide its throughput by `h_par·w_par`.**


### 0a. Column units and definitions

Report tables use these names. Where a name matches a CSV column it is kept **verbatim** —
CSV column names are a cross-project contract (CLAUDE.md #10) and must not be renamed — so the
unit is given here rather than in the header.

| column | unit | definition |
|---|---|---|
| `step_ms` | **milliseconds per optimizer step** | the *final epoch's* mean step time (§0 trap 1). CSV column |
| `samples/s (total)` | **samples per second, all ranks** | `global_batch ÷ step_ms`. CSV `samples_s_total` |
| `samples/s per GPU` | **samples per second per GPU** | `samples/s (total) ÷ ranks`. CSV `samples_s_rank` |
| `samples/GPU` | **samples per GPU per step** | = `local_batch`; how many samples one GPU holds in a step |
| `sample-equiv/GPU` | **sample-equivalents per GPU per step** | `global_batch ÷ ranks` — one full sample through the whole model = 1. A GPU holding 8 samples on a ¼ domain carries 2. Invariant to the decomposition |
| `weak eff (%)` | **percent** | `step_ms(1 node) ÷ step_ms(N nodes)` at 1 sample/GPU |
| `vs 1n (%)` | **percent change in step time** | positive = slower than the 1-node arm |
| `wireup (s)` | **seconds, once per job** | process-group setup before step 1 |
| `io (GB/s)` | **gigabytes per second, aggregate** | makani's `minimal IO rate` |
| `epoch wall (s)` | **seconds per epoch** | includes validation + checkpoint I/O, unlike `step_ms` |
| `wall (h)` | **hours** | end-to-end for the stated epoch count |
| `node-hours` | **nodes × wall (h)** | at 1 node this equals `wall (h)` |
| `updates` | **count of optimizer steps** | `steps/epoch × epochs`; *not* samples |
| `memory (GB)` | **gigabytes per GPU, epoch-END snapshot — a LOWER BOUND, not a peak** | makani's `memory footprint`: `(total − free)` sampled after the step's transients are freed, so it understates by an amount that grows with batch. → §5g. For the real peak use `peak torch memory [GB]` + `non-torch memory [GB]`, logged from 2026-09-02 |
| `trunk tile` | **grid points per rank** | after `scale_factor`, i.e. of the 60×120 trunk |

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
| rank↔GPU placement | ⚠ **`GPU_ORDER=default` on all 30 rows, production included.** Polaris' GPU↔NUMA map is **reversed** (`dev0`→NUMA 3 … `dev3`→NUMA 0, job 7531456), and the **mandatory** `--cpu-bind depth -d 8` puts local rank 0 on NUMA 0 — whose GPU is `dev3`. The two required settings therefore combine to place **every rank maximally far from its own GPU**. `-v GPU_ORDER=reverse` (`CUDA_VISIBLE_DEVICES=3,2,1,0`) inverts it. ✅ **MEASURED 2026-09-02 (§5i): reverse is +0.88% SLOWER at 1 node** (3+3 reps, node-matched pair confirms) and −7.0% *faster* at 4 nodes sharded — placement is **config-dependent, not a free win**, and `default` is correct for the production config |

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

| nodes | ranks | job | step_ms | vs 1n (%) | samples/s (total) | weak eff (%) | wireup (s) | io (GB/s) |
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

| nodes | ranks | job | step_ms | vs 1n (%) | samples/s (total) | weak eff (%) | wireup (s) | io (GB/s) |
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

| nodes | job | step_ms (ep 2) | vs 1n (%) | weak eff (%) | §3a − §3c (ms) |
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

| nodes | ranks | job | step_ms | samples/s (total) |
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

## 5. Spatial (model) parallelism — `w ≤ 2` works, and it BEATS pure DDP

The headline reverses the assumption this section used to carry. Sharding was treated as a
cost paid for memory; measured at equal global batch, it is **faster per GPU than pure data
parallelism** on this model, because it is what lets a GPU hold more than one sample.

### 5a. Which shapes work — the boundary is `w`, not the total split

All four at global batch 32 on the ALLDATA pack, new plugin + Simple, `EPOCHS=2`.

| job | nodes | h×w | spatial factor | trunk tile (grid pts/rank) | result |
|---|---|---|---|---|---|
| 7580122 | 4 | **4×1** | 4 | 15×120 | ✅ **trains**, 626.8 ms |
| 7580162 | 4 | **2×2** | 4 | 30×60 | ✅ **trains**, 576.6 ms |
| 7580127 | 4 | 2×**4** | 8 | 30×30 | ❌ hangs in **setup** |
| 7580084 | 4 | 4×**4** | 16 | 15×30 | ❌ hangs in the **first step** |

**Every arm with `w_parallel_size = 4` failed; every arm with `w ≤ 2` worked — including
`h=4`.** So the broken axis is `w`, and the earlier guess that "h=4 on a 60-row trunk" was the
problem is refuted by 7580122. Note also that upstream never ships `w=4` outside the
*stochastic* trainer (FCN3 pretrain2); the only `w` value in its deterministic recipe is 1.

### 5b. Why `w=4` fails — the ranks are not running the same program

7580127's flight-recorder dumps (16/16 ranks, the instrumentation 7580084 lacked):

| ranks | stuck on | group |
|---|---|---|
| 0, 1, 2, 4, 5, 6 | `nccl:broadcast` of `[35389440, 2]` — the **complex spectral weight**, 283 MB | 2-rank h-subgroup |
| **3, 7, 11, 15** | `nccl:all_reduce_barrier`, seq 2 | **`default_pg`** (all 16) |
| 8, 9, 10, 12, 13, 14 | nothing — every collective `completed` | — |

Ranks 3, 7, 11, 15 are exactly the **last rank of each 4-member `w`-group** (`rank % 4 == 3`).
One rank per `w`-group takes a different path through parameter sync than its peers, so the
collectives can never pair up. **This is an application-level divergence, not a transport
failure** — it is not size (283 MB broadcasts succeed elsewhere), not the plugin, not a
desync of the usual "one rank never arrives" kind. 191 of 192 collectives on rank 0 completed.

### 5c. Throughput at equal global batch — the win is WORK PER GPU, not sharding

⚠️ **CORRECTED 2026-09-01 by job 7580297.** The first version of this section claimed sharding
beats pure DDP by 1.71× per GPU. **That was a confound**, and the controlled arm refutes it:
the 4-node `h2w2` row carries *both* sharding *and* twice the per-GPU work, and only the second
is doing the work. Recorded as a correction rather than edited away.

All rows global batch 32, ALLDATA, new plugin, warmup-free (`step_ms` = final epoch).
⚠ Totals are **post-fix**: `samples_s_total` previously multiplied by rank count instead of
data-group count and overstated every spatial row by exactly `h_par·w_par` (§0 trap 4).

| config | job | nodes | GPUs | samples/GPU | sample-equiv/GPU | step_ms | samples/s (total) | samples/s per GPU |
|---|---|---|---|---|---|---|---|---|
| **pure DDP** | 7565972 | 8 | 32 | 1 | **1** | **492.7** | 64.9 | 2.03 |
| **h2w2** | 7580297 | 8 | 32 | 4 | **1** | **707.7** | 45.2 | 1.41 |
| h2w2 | 7580162 | 4 | 16 | 8 | 2 | 576.6 | 55.5 | 3.47 |
| h4w1 | 7580122 | 4 | 16 | 8 | 2 | 626.8 | 51.1 | 3.19 |

**Rows 1-2 are the controlled pair** — same nodes, same GPUs, same global batch, same
sample-equivalents per GPU, differing only in whether each GPU takes 4 samples over a quarter
domain or 1 sample over the whole one. **Sharding costs +43.6%.**

> **"Sharding overhead" (called the "tax" elsewhere in this repo), defined:**
> `overhead % = 100 × (step_ms[sharded] ÷ step_ms[pure DDP] − 1)`, measured at **identical**
> nodes, GPUs, global batch and sample-equivalents per GPU. It is a **relative slowdown of the
> step**, not a fixed cost and not a fee — `+43.6%` means a sharded step takes **1.436×** as
> long as a pure-DDP step doing the same work. Unit: percent.

§5a's success only means `w ≤ 2` *works*, not that it pays.

What actually pays is **work per GPU**. Compare rows 2 and 3, both sharded: halving the GPUs
and doubling the per-GPU work made the step *faster* in absolute terms (707.7 → 576.6), i.e.
**2× the work in 0.82× the time — 2.46× better per sample-equivalent.** Fixed per-step cost
dominates this model at batch 32, so the lever is fewer GPUs each doing more, not more GPUs.

⇒ **This predicted a configuration nobody had run: plain DDP at 4 nodes with `LOCAL_BATCH=2`**
(16 GPUs × 2 samples = batch 32, 2 sample-equivalents per GPU, *no* sharding and so no tax).
Extrapolating 576.6 / 1.436 ⇒ **~400 ms**.

✅ **MEASURED: 409.5 ms** (job **7580312**) — the prediction landed within **2.4%**. The
4-node sharding tax computed from that pair is **576.6 / 409.5 = +40.8%**, against +43.6% at
8 nodes (§5c table) and +80.8% at 1 node (7580404) — three independent node counts, all
between +40% and +81%.

⚠ And the trend did not stop at 4 nodes: **1 node, `LOCAL_BATCH=8`, is faster still at
365.4 ms** (job 7580338, 3 reps, 0.2% spread) — see §5d. Fewer GPUs each doing more kept
winning until there was only one node left.

Memory is not the constraint anywhere here: 6.0-6.2 GB on the sharded arms against the
production run's 8.36 GB, on 40 GB cards. ⚠ But see §5g — that figure is a *benchmark* peak
(`EVAL_SAMPLES=8`); at production settings it is **16.0 GB**.

### 5d. What it costs to train 100 epochs at batch 32 (**1,368** steps/epoch)

**One methodology for every row**, because mixing two is how the earlier version of this table
went wrong: `wall = steps/epoch × step_ms × epochs × 1.17`, `node-hours = nodes × wall`. The
**1.17** is the measured per-epoch overhead of the 128-node production run (57.47 s epochs
against 49.0 s of stepping) — it covers validation and checkpoint I/O, which `step_ms` excludes.

⚠ **An earlier version quoted node-hours computed from `step_ms` alone while quoting wall clock
with the overhead included.** At 1 node those two numbers are the *same quantity*, so "13.9
node-hours, 16.2 h wall" was self-contradictory; the figure is **16.3**. Corrected throughout.
⚠ The 1.17 factor is borrowed from a 512-GPU run where validation was amortised 128× more
widely than it is at 1 node. Job **7582088** measures it directly — prefer that number.

| plan | nodes | step_ms | **wall (h)** | **node-hours** | updates |
|---|---|---|---|---|---|
| **1n pure DDP** | **1** | **365.4** | **16.3 h** | **16.3** | **136,800** |
| 1n h2w2 | 1 | 660.6 | 29.4 h | 29.4 | 136,800 |
| 2n pure DDP | 2 | 627.1 | 27.9 h | 55.8 | 136,800 |
| 4n pure DDP | 4 | 409.5 | 18.2 h | 72.8 | 136,800 |
| 4n h2w2 (`reverse`) | 4 | 536.3 | 23.8 h | 95.4 | 136,800 |
| 4n h2w2 | 4 | 576.6 | 25.6 h | 102.5 | 136,800 |
| 4n h4w1 | 4 | 626.8 | 27.9 h | 111.5 | 136,800 |
| 8n pure DDP | 8 | 492.7 | 21.9 h | 175.2 | 136,800 |
| 8n h2w2 | 8 | 707.7 | 31.5 h | 251.7 | 136,800 |
| *what we actually ran — 128n, batch **512*** | *128* | *576.5* | *1.7 h* | ***216*** | ***8,500*** |

**16× the optimizer updates for 7.5% of the node-hours of the run we already did.** The
128-node run spent its parallelism inflating the batch, which buys fewer updates rather than
more work. Note the ordering tracks **work per GPU**, not decomposition, and that 1-node pure
DDP wins on *both* axes — it is simultaneously the cheapest and the fastest.

Wall clock is the only cost: 16.3 h exceeds `debug` (1 h) and `prod` won't take a single node,
so this needs the **`capacity`** queue (1-4 nodes, ≤168 h — it fits **unchained**, no
`-W depend` chain required). ⚠ `capacity` is `max_run 1` per *project*.

### 5e. Older spatial rows, kept for the record

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
  fixed in NCCL`. It is **not** what unblocked 4 nodes, and it does **not** cover §5b's
  divergence, which is in the same family but survives it.
* ⚠ **371.5 vs 219.5 is a 1.69× spread between two runs whose recorded configuration is
  identical** (same plugin, proto, pack, batch, GPU order). Both n=1, both at global batch 1,
  and neither should be quoted.

### 5f. Prereg §7b scored — 3 hits, 1 partial, 1 miss

| # | prediction | verdict |
|---|---|---|
| 1 | `h4w1` trains | ✅ **HIT** — 120/120 steps, loss 0.414 → 0.116 |
| 2 | `h2w2` trains | ✅ **HIT** — 120/120 steps, loss 0.398 → 0.111 |
| 3 | at least one of the three trains | ✅ **HIT** — two did |
| 4 | cost ordering `h2w2` ≲ `h4w1` < `h2w4` | ◐ **PARTIAL** — 576.6 < 626.8 as predicted; `h2w4` unscoreable |
| 5 | all three exceed the 985 ms zero-overhead reference | ❌ **MISS** — both successes are far **below** it |

**P5 is the miss that carries the result.** The 985 ms reference (2 × the 8-node pure-DDP
step) assumed per-GPU cost scales with sample count independently of *local batch*. It does
not: **8 samples per GPU is far more efficient than 1**, and that — not the sharding — is why
the 4-node arm beat the reference. The reference was built on the wrong invariant, and the
prediction failing is how that surfaced.

⚠ **An earlier version of this paragraph ended "…which is precisely why sharding wins in §5c".
That was wrong and contradicted §5c's own conclusion.** Job **7580297** (8 nodes, `h2w2`,
1 sample-equivalent/GPU — matched against 7565972) measured sharding as a **+43.6% tax**, and
7580404 confirmed it is not a fabric effect (**+80.8% at 1 node, no fabric at all**). The
4-node arm carried *two* changes — sharding *and* 2× the per-GPU work — and only the second
was doing anything.

## 5g. The production baseline — and why the benchmark absolutes are ~30% optimistic

⚠ **Everything in §2-§5f is a 60-step arm. None of those step times may be used for planning.**
Measured at real production settings (full-pass epochs, `EVAL_SAMPLES=512`, checkpointing):

| | step_ms | epoch wall (s) | memory (GB) |
|---|---|---|---|
| 60-step benchmark arm (7580338) | 365.4 | — | 10.33 GB |
| **production settings** (7582088 / 7585080) | **475 / 665 s per epoch** | **665-676 s** | **16.0 GB** |

**The cause is I/O, not instrumentation.** A controlled twin — same settings, wandb off — lands
at **474.9 ms** against 475.6 with wandb on (0.15%). So the **101-key per-iteration wandb
contract costs nothing**, and the gap is full-pass streaming: a real epoch reads 43,776 samples
from 30 files, while a 60-step arm re-reads a 1,920-sample window that was cache-hot.

⇒ **The occupancy *ranking* survives** — every arm carried the same bias — **but the absolutes
do not.** Plan on 475 ms.

**The two memory figures measure different things** and must not be conflated: 10.33 GB is a
*training*-dominated peak (`EVAL_SAMPLES=8`), 16.03 GB is a *validation*-dominated peak
(`EVAL_SAMPLES=512`, 3-step autoregressive rollout).

⚠⚠ **MEMORY vs samples/GPU: THREE MEASURED POINTS AND NO WORKING MODEL.** Two successive
models were fitted here and *both were refuted by the next measurement*. What is left is the
data:

| samples/GPU | global batch | memory | job |
|---|---|---|---|
| 8 | 32 | **16.04 GB** | 7585080 / 7582088 |
| **12** | **48** | **18.97 GB** | **7585983** |
| 16 | 64 | **≥39.5 GB → OOM** on a 39.49 GiB card | 7580362 |

**8→12 adds only 0.73 GB per sample; 12→16 would predict ~22 GB and instead exceeds 39.5.**

⚠⚠ **DIAGNOSED 2026-09-02 — the "cliff" is largely an ARTIFACT OF THE METRIC.
`memory footprint [GB]` is not a peak.**

```python
# makani/utils/training/training_helpers.py:22-27
free_mem, total_mem = torch.cuda.mem_get_info(device)
allocated_mem_gb = (total_mem - free_mem)/GB          # INSTANTANEOUS device usage
torch_mem_gb      = torch.cuda.max_memory_allocated()  # the real high-water mark
# deterministic_trainer.py:703 -- logs the first, DISCARDS the second:
all_mem_gb, _ = get_memory_usage(self.device)
```
It is sampled **at epoch end**, after the step's transients are freed. So 16.04 and 18.97 are
*post-epoch snapshots* that understate the true requirement by an amount that **grows with
batch** — which manufactures an apparent discontinuity where the underlying curve may be smooth.

What the batch-64 OOM actually shows (job 7580362, full message):
* **Not fragmentation** — `276.76 MiB reserved by PyTorch but unallocated`. PyTorch's own
  message says fragmentation would make that number large.
* **~7.7 GB is NON-PyTorch overhead** — at the OOM, 39.41 GiB in use but only **31.46 GiB
  allocated by PyTorch** (CUDA context + cuFFT/cuDNN workspaces + NCCL buffers). Corroborated
  by the scaffolding line: `9.66 GB (1.99 GB for pytorch)`. **That ~7.7 GB is a fixed tax on
  every configuration** and it is inside the reported footprint.
* **It failed in the LOSS, not the forward** — `makani/utils/loss.py:402`
  (`loss_vals.append(lfn(...))`), i.e. at the transient peak where the whole forward graph is
  still alive for backward *plus* the loss intermediates. That transient is exactly what an
  end-of-epoch sample cannot see.

⇒ **FIX SHIPPED 2026-09-02 — the peak is now logged.** `plasim_trainer.log_epoch` emits two
new per-epoch keys to the screen log and to wandb, and calls `reset_peak_memory_stats` so each
epoch reports its own peak rather than a running max:

| key | what it is | why both are needed |
|---|---|---|
| `peak torch memory [GB]` | gigabytes, `torch.cuda.max_memory_reserved` per GPU | the allocator high-water mark — **the part that scales** with batch and `n_future` |
| `non-torch memory [GB]` | gigabytes per GPU, `(total − free) − torch reserved` at epoch end | CUDA context + cuFFT/cuDNN workspaces + NCCL buffers — the **~7.7 GB fixed tax** above |

Card occupancy to compare against **39.49 GiB** is the **sum**. Makani's own
`memory footprint [GB]` is left untouched and still logged, so no prior row is invalidated —
the new keys are additive.

Two deliberate limits, so this isn't over-read in turn:
* **Not in `makani_scaling*.csv`.** `parse_makani_scaling.py` asserts the header matches
  `FIELDS` and *refuses to append* on drift (`:162`, `:172-178`) — the rule-#10 guard doing its
  job. Adding a column means migrating every existing CSV, which is a deliberate change, not
  something to do mid-campaign with a production job running. → TODO. Read the values from the
  job's `.o` log or wandb until then.
* **`max_memory_reserved` is still PyTorch-only.** It is a high-water mark of the *allocator*,
  not of the device; the non-torch term is sampled at epoch end and assumed near-constant. That
  assumption is itself untested — it is just far more defensible than a fit to post-epoch
  snapshots.

First arms carrying it: the batch-48 LR sweep **7586382 / 7586383 / 7586384** (patched while
still queued, so all three land with it) and any capacity job submitted after this date.

Retired models, kept so neither is refitted:
* ~~"≈0.99 GB per sample-step, linear"~~ — from the benchmark arms; predicted 18.2 GB at 16
  samples. **Refuted by the OOM.**
* ~~"≈2.5× per doubling, superlinear"~~ — fitted to the 8 and 16 points. Predicted ~21 GB at
  12; the truth is 18.97, and the model cannot produce a cliff at all. **Refuted by 7585983.**

⇒ **Do not extrapolate memory in this regime. Measure one arm per configuration.** In
particular the apparent ~20 GB of headroom at 8 samples/GPU is *not* usable capacity: batch 48
fits with room, batch 64 does not fit at all, and nothing in between has been probed.

### 5h. LR sweep — the prereg beat the prior

Three arms, 3 full-pass epochs each (1,368 updates/epoch), identical warm-restart schedule, one
variable. Selection rule committed **before** arms 2-3 existed (`9506ad1f`).

| arm | valid loss @ ep3 | grad norm @ ep3 | verdict |
|---|---|---|---|
| 4.0e-4 — upstream FCN3 pretrain-2's batch-32 value | 0.03237 | 0.04966 | worst of the three tested first |
| 1.0e-3 — our shipped config, unscaled | 0.02655 | 0.02616 | |
| **2.0e-3 — sqrt-scaled from batch 8** | **0.02352** | **0.01830** | **WINNER — minimum** |
| 4.0e-3 *(added 2026-09-02, job 7585996)* | **0.10703** | 0.00913 | **4.5× worse** |

✅ **CLOSED 2026-09-02 — the optimum is bracketed on both sides, and it is 2.0e-3.** The first
three arms had 2e-3 winning at the *top* of the range, so the optimum was only bounded below.
The 4e-3 arm settles it: validation is **4.5× worse**, giving a clean U-shape with the minimum
at 2e-3. The pre-registered rule picked the right value from an open-ended range.
⚠ Note 4e-3's grad norm is the *lowest* of the four (0.00913) while its loss is the worst —
a low gradient norm is **not** a health signal on its own; read it with the loss.

**4e-4 was my stated prior on upstream's authority and it came third of four** — the
pre-registered rule is the only reason the choice wasn't rationalised after the fact.

⚠ Three epochs still cannot catch a tail instability (ai-rossby diverged at **epoch 11**).
Warm restarts bound the risk: 2e-3 is the peak of every cycle, so trouble surfaces in cycle 1
with checkpoints intact — and it did not (production cleared cycle 1 with grad norm monotone).

### 5i. Rank↔GPU placement — measured, and it is config-dependent

`GPU_ORDER=reverse` sets `CUDA_VISIBLE_DEVICES=3,2,1,0`, undoing Polaris' reversed GPU↔NUMA
map (`dev0`→NUMA 3 … `dev3`→NUMA 0, job 7531456) so each rank sits on its own GPU's NUMA node.

| config | order | reps (step_ms) | step_ms median | delta (%) |
|---|---|---|---|---|
| **1 node**, batch 32, pure DDP | default | 365.1 / 365.9 / 365.4 | **365.4** | — |
| **1 node**, batch 32, pure DDP | **reverse** | 366.7 / 368.6 / 369.7 | 368.6 | **+0.88% SLOWER** |
| 4 nodes, batch 32, `h2w2` | default | n=1 | 576.6 | — |
| 4 nodes, batch 32, `h2w2` | **reverse** | n=1 | 536.3 | **−7.0% faster** |

**Opposite signs, so placement cannot be ported as a rule.** It helps when sharded traffic
crosses the fabric and costs slightly at 1 node where there is no fabric to help.

⚠ The 1-node default arm has a **0.2% rep spread** — the tightest measurement in the campaign —
so +0.88% is real, not noise. The confound was checked: the six arms landed on six hosts, but
**7580482 (default) and 7580507 (reverse) ran on the same node** `x3001c0s19b1n0` → 365.58 vs
369.85 = **+1.17%**. Node-matched, reverse is still slower.
⚠ Residual: the reverse reps drift monotonically (366.8 → 368.8 → 369.9) while default does not.
Don't quote 0.88% as precise. **The decision — `default` for the 1-node production config — is
unaffected.**

### 5j. The LR ceiling — batch-independent, and the failure is irreversible

**Scored 2026-09-03 from the batch-48 sweep (7586382/3/4) + the batch-32 sweep.** The batch-48
arms did not fail *because of* batch 48. The same failure happens at batch 32.

| arm | job | ep 1 valid | ep 2 valid | ep 3+ valid | ep-2 grad norm | verdict |
|---|---|---|---|---|---|---|
| b32 4.0e-4 | 7584861 | 0.2302 | 0.0382 | 0.0324 | 0.1855 | ok |
| b32 1.0e-3 | 7584961 | 0.1312 | 0.0319 | 0.0265 | 0.1142 | ok |
| **b32 2.0e-3** | 7585026 | 0.0944 | 0.0299 | **0.0235** | 0.1013 | **ok — best** |
| **b32 4.0e-3** | 7585996 | 0.0724 | **0.1072** | 0.1070 | **1.05e7** | **💥 collapsed** |
| b48 2.0e-3 | 7586382 | 0.1120 | 0.0327 | 0.0263 → 0.0211 | 0.1399 | ok |
| **b48 3.0e-3** | 7586383 | 0.0923 | **0.1071** | 0.1069 | **6.30e7** | **💥 collapsed** |
| **b48 4.5e-3** | 7586384 | 0.0816 | **0.1071** | 0.1070 | **3.22e11** | **💥 collapsed** |

**Three facts the traces establish.**

1. **Epoch 1 is fine, and higher LR is strictly *better*** (0.1120 → 0.0923 → 0.0816 as LR
   rises). There is no gradient-noise penalty anywhere before the failure. `lr_warmup_steps: 1`
   means an arm first meets full LR in **epoch 2** — it dies on first sustained exposure.
2. **Death is one epoch wide**, train loss 0.24 → 6.05e5 → 0.1056, and **severity tracks LR**
   (1.05e7 → 6.30e7 → 3.22e11). That is geometric divergence past a stability threshold, not a
   noise problem.
3. **It never recovers.** All three land on train 0.1055–0.1056, valid 0.1067–0.1072, grad norm
   0.007–0.011 — *the same absorbing state to three decimals*, whatever the batch or LR.
   Post-collapse grad norm is **30× below** a healthy arm's, so cosine annealing back down
   cannot rescue it. **Divergence here is irreversible, not a setback.**

**Reproducible:** the 4.5e-3 collapse occurred in **both** the v1 and v2 batch-48 runs — epoch 2
both times, terminal valid 0.1071 both times, epoch-1 valid 0.0814 vs 0.0816. The blow-up
*magnitude* differs (3.0e3 vs 1.16e10), so it is not bitwise deterministic; the *failure* is.

**The ceiling did not move with batch.**

| batch | survives | dies | ceiling |
|---|---|---|---|
| 32 | 2.0e-3 | 4.0e-3 | (2e-3, 4e-3) |
| 48 | 2.0e-3 | **3.0e-3** | **(2e-3, 3e-3)** |

Linear scaling predicts ceiling₄₈ = 1.5 × ceiling₃₂ > 3e-3, i.e. **3e-3 should have been safe at
batch 48. It was not** — refuted without needing ceiling₃₂ exactly. Sqrt scaling predicts
> 2.45e-3 and survives only in the narrow window (2.45e-3, 3e-3): squeezed, not excluded.

⇒ **Hypothesis: the ceiling is curvature-limited, not noise-limited.** Linear scaling assumes
gradient *variance* binds. If instead the stability bound η < 2/λ_max binds, the ceiling is a
property of the loss surface and averaging more samples cannot move it — which predicts all of
the above. **Under test as prereg §7c.**

**Two config choices that plausibly lower that ceiling** (both are what §7c isolates):

* **`optimizer_max_grad_norm: 32` is decorative.** `clip_grads`
  (`training_helpers.py:100-115`) clamps correctly and returns the *pre*-clip norm, so clipping
  did fire — but production's grad norm over 64 epochs has an **all-time max of 0.2995**. The
  clip sits **107× above the operating point** and has never engaged in a healthy step.
  **makani's own default is 1.0** (`train.py:74-75`); the 32 is a fork convention from
  `polaris/e3sm_alldata_full.yaml:141`, and sibling configs in the same family use `0.0`.
* **`optimizer_beta2: 0.95`** — ~20-step second-moment memory vs ~1000 at 0.999; a *large-batch*
  setting inherited from the batch-512 run. ⚠ But see §7c P3: for a **spike** the sign flips,
  and 0.999 may be worse. Do not treat this as a known fix.

**Production is safe, with a thinner margin than it looks.** 7585080 (batch 32, LR 2.0e-3) has a
textbook trace — each cycle anneals 0.0296 → 0.0043, restart bumps back, three restarts
survived, cycle minima improving 0.01402 → 0.01341 → 0.01316. But it runs **one rung below a
cliff that is irreversible**, and SGDR returns it to peak LR every 20 epochs with **no
re-warmup**. Empirically fine; the failure mode is worth knowing.

⚠ **n = 1 per arm, 3–6 epochs.** The blow-up reproduces across three arms and two batch sizes, so
the *mechanism* is solid; the exact ceiling location is not.

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
| 5 | `GPU_ORDER=reverse` (NUMA-local pairing) within 5% of default | ✅ **SCORED 2026-09-02 — HIT at 1 node** (+0.88%, inside 5%), **MISS at 4 nodes sharded** (−7.0%). The prediction was written as if placement had one sign; it does not (§5i) |

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

### 7b. Prereg — the three upstream-sanctioned shapes at batch 32 (2026-09-01, after 7580084)

`h4w4` (7580084) **failed** — prereg 7a-P1 falsified, zero steps, wedged on a 20.5 MB
all-reduce at `SeqNum=137` with all 16 ranks enqueued. Checking upstream afterwards (which
should have come first): **`h4w4` appears nowhere in makani.** Documented shapes are `h4w1`
(README, 256 GPUs, the *deterministic* trainer = ours), `h2w2` (FCN3 pretrain1) and `h2w4`
(FCN3 pretrain2) — maximum spatial factor **8**, and we asked for 16. Worse, relative to grid:
FCN3 splits 721×1440 at `scale_factor 2` by h2w4 ⇒ ~180×180 per rank; we split 180×360 at
`scale_factor 3` (trunk **60×120**) by h4w4 ⇒ **15×30 per rank**, ~70× smaller tiles than the
most aggressive split upstream ever uses.

This set runs the three documented shapes, all at **global batch 32 on 4 nodes**, where the
launcher's `GLOBAL_BATCH = LOCAL_BATCH × NRANKS/(HPAR·WPAR)` gives **identical per-GPU work
(2 sample-equivalents)** in every arm ⇒ the only variable is communication topology.

| arm | h×w | data ranks | LOCAL_BATCH (samples/data group) | trunk tile (grid pts/rank) |
|---|---|---|---|---|
| A | 4×1 | 4 | 8 | 15×120 |
| B | 2×4 | 2 | 16 | 30×30 |
| C | 2×2 | 4 | 8 | 30×60 |

1. **`h4w1` trains** (120/120 steps over 2 epochs). It is upstream's own deterministic recipe.
   *Falsified by* a hang — which would mean the `h` axis at 4 is the problem on a 60-row trunk,
   i.e. **spatial parallelism is unusable at this resolution**, and 7580084 was not about `w`.
2. **`h2w2` trains.** It is the one shape already proven here (arm F 7564035, 4 nodes, new
   plugin) — at batch 4 on the 53-ch pack, so this re-tests it at batch 32 on ALLDATA.
   *Falsified by* a hang ⇒ the batch-32/ALLDATA change broke a previously working shape.
3. **At least one of the three trains.** *Falsified by* three hangs ⇒ drop spatial parallelism
   for this model at this resolution and revisit only if the rollout stage forces it.
4. **Cost ordering `h2w2` ≲ `h4w1` < `h2w4`** — same per-GPU work, so larger tiles and smaller
   groups should win, and B pays an 8-way split for it. *Falsified* by any other ordering.
5. **All three exceed the 985 ms zero-overhead reference** (2 × the 8-node pure-DDP batch-32
   step, 492.7 ms, 7565972 — warmup-free on both sides now, since these run `EPOCHS=2`).
   The number that decides whether sharding is worth using is the **ratio** to 985 ms, not the
   raw step time; arm F's `h2w2` ratio on the 53-ch pack was ~5×.

Flight recorder is on for all three (`TORCH_NCCL_TRACE_BUFFER_SIZE`, dumps to a persistent
dir, `HEARTBEAT_TIMEOUT_SEC=1800` so the informative 600 s collective-timeout message wins the
race) — its absence is why 7580084 cost a job and still could not name what else was in flight.
Rows route to `bench/makani_spatial.csv`: these are 2-epoch rows and must not sit in the
single-epoch scaling table (§0 trap 1).

**Paper context, kept from the deleted plan:** FourCastNet 3 (arXiv:2507.12144 §E.2) trains
pretrain-2 on **512 A100** as `ensemble 2 × batch 32 × h2w4`. The production run above is also
512 A100 — but as **pure data parallelism** on a different model and dataset. It reproduces the
paper's *rank budget*, not its decomposition, and no table from it should be captioned as an
FCN3 reproduction.

### 7c. Prereg — can the LR ceiling be raised at batch 48? (2026-09-03, before submission)

**Question.** §5j established an LR ceiling in **(2e-3, 3e-3]** that did **not** move when batch
went 32 → 48, and identified two config suspects. This is the single-variable test of both.

**Probe point: batch 48, LR 3.0e-3** — a *known-dead* arm (7586383). All arms 1 node,
`e3sm_alldata_full.yaml`, `EPOCHS=6`, `EVAL_SAMPLES=512`, warm restarts `T_0=2`, `WARMUP=1`,
identical to the sweep they are compared against.

| arm | β₂ | clip | isolates |
|---|---|---|---|
| **E** `b48fix_base` | 0.95 | 32 | **CONTROL — repeat of the dead arm.** Without it, A–D are uninterpretable |
| **A** `b48fix_beta2` | **0.999** | 32 | β₂ alone |
| **B** `b48fix_clip` | 0.95 | **1.0** | clip alone (= makani's own default, `train.py:75`) |
| **C** `b48fix_both` | **0.999** | **1.0** | both |
| **D** `b48fix_push` | **0.999** | **1.0** | both, at **LR 4.5e-3** — does the ceiling move a little or a lot? |

**Scoring rule, fixed now.** *Survives* = epoch-3 validation < 0.05 **and** train loss finite
(< 1.0) through epoch 3. *Collapsed* = validation ≥ 0.10 **and** grad norm < 0.02 (the dead
attractor: train 0.1055–0.1056, valid 0.1067–0.1072, grad norm 0.007–0.011).

| # | prediction | conf. |
|---|---|---|
| P1 | **E collapses again at epoch 2.** If it survives, the collapse is stochastic and A–D say nothing | 0.85 |
| P2 | **B (clip alone) survives.** Clipping bounds what enters Adam's `m` and `v`, capping the runaway. *Counter:* Adam's update is ~invariant to a uniform gradient rescale, so it may only delay | 0.55 |
| P3 | **A (β₂ alone) survives** | **0.40 — and the mechanism cuts BOTH ways** |
| P4 | **C (both) survives**, and if any single-factor arm survives, C does | 0.65 |
| P5 | **D (4.5e-3) survives** — 2.25× the known-good LR, collapsed 2/2 at shipped settings | 0.25 |
| P6 | **If C survives, its epoch-3 validation beats b48 2.0e-3's 0.0263**, because higher LR was strictly better pre-collapse (0.1120 → 0.0923 → 0.0816 at epoch 1) | 0.70 |

⚠ **P3 is deliberately below 0.5, against the standing recommendation.** The handoff lists
β₂ 0.95 → 0.999 as recommendation #3 on the "0.95 is a large-batch setting" argument, and that
argument is real: short memory makes `v̂` a noisy estimate, so some parameters transiently get a
small `v̂` and an oversized update. **But for a *spike* the sign flips.** A single outlier
gradient enters `v` with weight `1 − β₂` — **0.05 at β₂=0.95 versus 0.001 at 0.999**. The short
memory therefore reacts ~50× more strongly and damps the *next* step much harder; the long
memory barely notices the excursion. So 0.999 may be **worse** for the exact failure we
measured, even though it is better for steady-state noise. Recorded before the result precisely
because the honest prior is two-sided, and a post-hoc story could be told either way.

**What each outcome licenses.** A or B alone surviving ⇒ the ceiling is an artefact of optimizer
config, not the loss surface, and the whole LR sweep must be re-run at corrected settings. Only
C surviving ⇒ the two interact. **Nothing surviving ⇒ the curvature-limited hypothesis (§5j)
stands and 2e-3 is near a genuine architectural limit** — which is the outcome that most
strongly validates the production run's current setting.

### 7d. Prereg — the batch-32 mirror (2026-09-03, before submission)

**Why LR 3.0e-3 and not batch 32's own dead point (4.0e-3).** Matching the *probe LR* to §7c is
what makes the two sets differ in **batch alone**; matching the *dead point* instead would
confound batch with LR. It also fills a real gap: **3.0e-3 at batch 32 has never been run** — it
sits inside the untested (2e-3, 4e-3) bracket — so the control arm is a new measurement, not a
repeat. Everything else is identical to §7c (1 node, `LOCAL_BATCH=8`, `EPOCHS=6`,
`EVAL_SAMPLES=512`, warm restarts `T_0=2`, `WARMUP=1`).

| arm | job tag | β₂ | clip | mirrors |
|---|---|---|---|---|
| **E32** | `b32fix_base` | 0.95 | 32 | §7c E — **control, and it narrows the batch-32 ceiling** |
| **A32** | `b32fix_beta2` | **0.999** | 32 | §7c A |
| **B32** | `b32fix_clip` | 0.95 | **1.0** | §7c B |
| **C32** | `b32fix_both` | **0.999** | **1.0** | §7c C |

No `push` arm: §7c D already probes how far the ceiling moves, and at batch 32 the ceiling
location is what E32 is measuring. Four arms keeps the project under `preemptable`'s
`max_run 10` with a slot spare (5 from §7c + 4 = 9).

**Same scoring rule as §7c** (survives = epoch-3 valid < 0.05 and train < 1.0; collapsed =
valid ≥ 0.10 and grad norm < 0.02).

| # | prediction | conf. |
|---|---|---|
| Q1 | **E32 collapses** (⇒ ceiling (2e-3, 3e-3] at *both* batches — identical, batch-independence confirmed sharply) | **0.55** |
| Q2 | **The fix arms behave the SAME way at batch 32 as at batch 48** — i.e. whichever of A/B/C rescues 3.0e-3 at batch 48 also rescues it at batch 32, and vice versa | 0.75 |
| Q3 | If E32 *survives*, then ceiling₃₂ ∈ (3e-3, 4e-3) while ceiling₄₈ ∈ (2e-3, 3e-3] ⇒ **the ceiling DECREASES with batch** — anti-scaling, and a stronger refutation than §5j's | 0.45 (= 1 − Q1) |

⚠ **Q1 is near a coin-flip on purpose.** Two considerations point to collapse: batch 32 runs
**1368 optimizer steps/epoch vs 912** (50% more chances to meet an outlier), and its gradients
are noisier per step. One points the other way: the existing brackets — b32 dies at 4.0e-3, b48
dies at 3.0e-3 — are *already consistent* with a ceiling that falls as batch rises, which would
put 3.0e-3 on the safe side at batch 32. **Both outcomes are informative, which is why the arm
is worth running**; neither is the "hoped-for" result.

**Interaction with §7c.** The pair of controls (7587738 at batch 48, E32 at batch 32) is the
actual batch-independence test — a matched-LR, matched-optimizer, batch-only contrast. That
contrast did not exist before: §5j inferred batch-independence from two *different* LRs.

## 8. Not measured — and what each would cost

| gap | why it matters | cost |
|---|---|---|
| **Reps 2-3 of any ladder** | every number here is n=1, and §2 shows a 20% unexplained single-node gap | ~3 jobs/rung, ≤10 nodes, `debug-scaling` |
| **A warmup-free, wandb-off ladder** | decides §3a vs §3c, i.e. whether the first hop is free or +35% | 4 jobs, ≤1 h |
| **cpu-bind / progress-thread sweep on the new plugin** | ~79% of the 8-node step is exposed comms (§3b); the bind was tuned for the old plugin's manual progress | 3-4 jobs at 4 nodes |
| **`nsys` per-rank capture (`-v NSYS=1`)** | makani has **no kernel-level profile at all**; rank-0 logging cannot answer where the step goes | 1 job (truncated by design — `exit_on_stop`) |
| **arm D (synthetic)** | separates an I/O loss from a comms loss | 1 job |
| ~~**arm E (`GPU_ORDER=reverse`)**~~ ✅ **DONE 2026-09-02, §5i** | every row before it — and the production run — placed each rank on the NUMA node *farthest* from its GPU (§1). It is a placement change, not an arithmetic one, so it needs no equivalence gate, and it is the standing candidate for the host-CPU stall in `polaris_bench_report.md` §4.4e. **It matters more under spatial parallelism**, where intra-node halo traffic rides that same distance every step — and that is exactly what the measurement showed: **−7.0% at 4 nodes sharded, but +0.88% SLOWER at 1 node** (§5i). ai-rossby's two attempts at the same test were both refused (7577036 `rc=134`, 7577166 `rc=143`); makani's 3+3 reps are the first measurements of this axis in the project | ✅ done, 6 arms |
| **A DESIGN §4 equivalence baseline** | no hot-path change may be committed without one; makani has none | 1 short job |
| **Inference / evaluation on `best_ckpt_mp0.tar`** | a trained model nobody has scored | not yet scripted |
| **`omp_threads=64`** | all 30 rows ran at **8× CPU oversubscription** on the same cores the progress engine uses (PBS exports `OMP_NUM_THREADS=<ncpus>`; the launcher's `${OMP_NUM_THREADS:-1}` idiom never overrode it). Comparability is intact — the value is constant on every row — but the absolute numbers were taken under it | deliberately **not** changed: flipping it makes future rows incomparable with all 30. Owner's call, and it interacts with the cpu-bind sweep above |

## 9. Operational facts that outlive the plan

🐛 **Forking a run from a checkpoint fails SILENTLY unless the seeded file is named `v0`.**
makani gates resume on the existence of *exactly* `ckpt_mp0_v0.tar` —
`makani/train.py:101-105` formats the path with `checkpoint_version=0` **hardcoded**, and does
**not** look for the newest file. Seeding a fresh expDir with `ckpt_mp0_v71.tar` gives
`resuming = False` and the job **begins training from scratch with no error and no warning**;
the loss curve reads as a fresh run, not a broken fork (measured 2026-09-03, jobs
7588049/7588050, killed at 2 min). The epoch is *not* taken from the filename —
`restore_from_checkpoint` reads counters stored inside the tar, and
`get_latest_checkpoint_version` (`checkpoint_helpers.py:33-42`) selects by **mtime** over the
glob — so a lone `ckpt_mp0_v0.tar` resolves correctly and the run continues at the checkpoint's
true epoch. ⇒ **Always assert `resuming True` in the log before trusting a warm start.**
*(Production 7585080 is unaffected: it holds `v0`…`v72`, so its own resume path is valid.)*

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


> ⚠ **Authoritative source: `polaris_pbs_notes.md` §1b.** This copy is a convenience and
> drifts — when the two disagree, **the notes win**. (2026-09-02: a stale `preemptable`
> claim had to be corrected in five files at once; that is what this line exists to stop.)

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
