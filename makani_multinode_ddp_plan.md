# Makani multi-node DDP scaling — plan and prereg

*Target:* profile the **makani** codebase on Polaris across node counts, mirroring
the parallel decomposition of **FourCastNet 3** (arXiv:2507.12144) §E.2, which is
makani's own paper. Opened 2026-08-21.

Companion to `PANGU_POLARIS_PROFILING_PLAN.md`, whose **item 12** this executes for
a different model. Read that file's §0b first: every Polaris number in this repo is
single-node 4× A100, and this is the axis that has never been measured.

---

## 1. What the paper actually specifies, and what of it we can run

§E.2 (pre-training) and §E.3 (fine-tuning) of arXiv:2507.12144:

| stage | hardware | wall | steps | shape |
|---|---|---|---|---|
| pre-train 1 (6-hourly) | **1024 H100** | 78 h | 208,320 | batch **16**, ensemble **16**, spatial **4-fold** |
| pre-train 2 (rollout) | 512 A100 | 15 h | 5,040 | 4 autoregressive steps, LR cut every 840 |
| fine-tune | 256 H100 | 8 h | — | spatial **16-fold**, 6-hourly 2012–2016 |

**16 × 16 × 4 = 1024 exactly.** The stage-1 rank budget is a three-way product of
batch-, ensemble- and spatial-parallelism, and all three are the same makani
machinery — `makani/utils/comm.py::init(model_parallel_sizes=[h,w,fin,fout],
data_parallel_sizes=[ensemble,batch])`.

> **CONFIRMED 2026-08-24 — no longer an inference.** The official repo (cloned to
> `${MEMBER_ROOT}/external/makani-upstream`; the pip wheel ships no configs) carries
> `config/fourcastnet3.yaml`, whose comments state the layouts: pretrain1
> `ensemble 16 × batch 16` + *"requires a model-parallelism of h=2, w=2"* = 1024;
> **pretrain2 (512 A100)** `ensemble 2 × batch 32` + *"h=2, w=4 to fit into memory
> on 80GB GPUs"* = 512; finetune `ensemble 4 × batch 4` × 16-fold = 256. The spatial
> split is CLI-only (`--h_parallel_size/--w_parallel_size`) — there is no "512-GPU
> config file" to copy, which is why this plan's `-v HPAR=/WPAR=` shape is already
> the right one. Upstream's canonical *deterministic* SFNO scale-out (README): 256
> GPUs = `--h_parallel_size=4 --w_parallel_size=1 --batch_size=64` bf16, i.e. 1
> sample per data group split 4-way — spatial parallelism at small per-GPU batch is
> NVIDIA's own lever for our trainer class. The paper still does not give
> optimizer/LR values or the AMP dtype for its runs.

What transfers to us, stated so nobody later reads more into the result than is there:

| paper element | here | why |
|---|---|---|
| batch / data parallelism | ✅ **yes** | 4 nodes × 4 A100 = 16 ranks at 1 sample/GPU = **global batch 16**, numerically the paper's stage-1 batch |
| spatial (domain) parallelism | ⚠️ **mechanism yes, untested** | `HPAR`/`WPAR` knobs; 4-fold is the paper's stage-1 value. Never exercised in this repo, so **default OFF** — an untested model-parallel path must not be able to fail the DDP measurement |
| ensemble parallelism | ❌ **no** | FCN3 is probabilistic, trained with an ensemble CRPS loss (`train_stochastic` / `ensemble_trainer`). Our fork runs `sfno_training.train_plasim → PlasimTrainer → deterministic_trainer`; the ensemble group is size 1 |
| model + data | ❌ **no** | FCN3: its own architecture on ERA5 0.25° (721×1440). Ours: SFNO `embed_dim 384` / `num_layers 8` on E3SM 180×360, 58-in / 53-out |
| scale | ❌ 16 ranks vs 1024 | 1/64 |

⇒ **This reproduces the paper's parallel decomposition at 1/64 scale on our model.
It is not an FCN3 reproduction, and no table from it should be captioned as one.**
Scaling *behaviour* is what transfers; absolute numbers do not.

## 2. The blocker that had to be cleared first, and how

`module load conda` has been broken cluster-side since ~2026-08-20 and was
**re-confirmed broken 2026-08-21**: `/soft/modulefiles/conda/2025-09-25.lua`
depends on `cray-hdf5-parallel/1.14.3.5` and `gcc-native/14.2`, and the current PE
ships only `1.14.3.9` and `14`. Every makani PBS script in this repo opens with
that module, and `$SFNO_VENV` is a `--system-site-packages` venv with **no torch of
its own** — it inherits the base conda's 2.8.0. So makani was as blocked as Pangu
items 7–17.

Cleared without waiting on the ALCF ticket. Three separate problems:

1. **The modulefile.** Reconstructed by hand in `polaris_makani_env.sh`: the same
   five `depends_on` modules with the two dead pins replaced by the only installed
   versions, plus every `setenv`/`prepend_path` line copied value-for-value, then
   `conda.sh` sourced from the install directory — which is intact; only the
   modulefile pointing at it is broken. It **tries `module load conda` first** and
   reports which path it took (`MAKANI_ENV_SOURCE`), so it reverts to the
   sanctioned env by itself once ALCF fixes it.
2. **`libcudart.so.13`**, dangling in torch 2.8.0's `libtorch_global_deps.so` —
   present in the `cuda-13.0.1` toolkit. A path.
3. **`libmpi_gnu_123.so.12`**, which `polaris_pbs_notes.md` §1 recorded as *"not
   present anywhere"*. **That is true of the filename and false of the library.**
   `_123` was the old cray-mpich's spelling of the gcc-12.3 build, and `mpich/9.1.0`
   ships exactly that build as `libmpi_gnu.so.12` under `.../ofi/gnu/12.3/lib`.
   SONAME and soversion both stay `12`, so a symlink under the former name is a
   *rename*, not an ABI substitution. (The modulefile's own last line names this
   soname in a commented-out PyTorch hotfix.)

   With 2 and 3 applied, `ldd libtorch_global_deps.so` reports **0 unresolved**
   (verified on a login node — a link check, not an import).

4. **h5py** is a fourth problem and is *not* shimmable: hdf5 1.14.3.5's
   `libhdf5_parallel_gnu_123.so.200` became 1.14.3.9's `...gnu.so.310`, and
   200 → 310 is a real soversion bump. `makani/utils/metric.py:19` is a bare
   `import h5py`, so it sits on the import path of **every** makani entrypoint —
   including `--enable_synthetic_data` runs, which therefore do not dodge it.
   Fixed with a minimal PyPI overlay (`h5py 3.16.0`, vendored `libhdf5-*.so.320`),
   the same `$POLARIS_TOPUPS` pattern, holding **h5py only** because PYTHONPATH
   outranks site-packages and a fatter overlay would shadow the venv's
   torch/torch_harmonics. Built and green: `MAKANI_H5PY_OVERLAY_OK`.

**Why this env, and not the ai-rossby venv's torch 2.10.** For *Pangu*, substituting
that env is disqualified — every number in `polaris_bench_report.md` was measured on
2.8.0 and §4.4a showed kernel selection is not bit-reproducible even within one torch
version. **That argument does not bind makani, which has no prior profile at all.**
The choice here is therefore free, and 2.8.0 was taken for a different reason: it is
the version the green makani runs (7253465) used, so the 1-node arm stays comparable
to the only makani evidence that exists. Every row records its `torch` and
`env_source`.

## 3. Measurement design

**Weak scaling.** Local batch fixed at 1 sample/GPU; global batch grows 4 → 8 → 16.
Per-GPU arithmetic is therefore constant and every change in step time is
communication plus load imbalance — the actual question. Strong scaling would move
the per-GPU work and confound the two.

| arm | nodes | ranks | global batch | h×w | data | what it isolates |
|---|---|---|---|---|---|---|
| A | 1 | 4 | 4 | 1×1 | real | baseline; NCCL never leaves NVLink |
| B | 2 | 8 | 8 | 1×1 | real | first Slingshot hop |
| C | 4 | 16 | **16** | 1×1 | real | **the paper's stage-1 batch size, pure DDP** |
| D | 4 | 16 | 16 | 1×1 | synthetic | same as C with the shared filesystem removed |
| E | 1 | 4 | 4 | 1×1 | real | A with `GPU_ORDER=reverse` (NUMA-local pairing) |
| F | 4 | 16 | 4 | 2×2 | real | *optional* — the paper's 4-fold spatial split |

Arms A–C are the deliverable. D separates a comms loss from an I/O loss. E tests
`polaris_pbs_notes.md` §1's measured **reversed GPU↔NUMA map** as the candidate
mechanism for `polaris_bench_report.md` §4.4e's undiagnosed host stall. F is
optional because the spatial path is unexercised here.

**≥3 interleaved reps per arm** (`-v REP=`), interleaved and never batched —
§4.4c's lesson: two identical-config runs read 42.2% and 37.4% for the same
quantity, so a single rep of each arm cannot support a scaling claim.

**Per-rank, not aggregate.** `train_plasim.py` sets `log_to_screen = (world_rank == 0)`,
so makani's own step timing is rank-0 only and **cannot** answer item 12's actual
question. `-v NSYS=1` captures every rank (`--output=rank_%q{PMI_RANK}`, makani's
`--capture_type cupti` driving `cudaProfilerStart/Stop` over a 10-step mid-run
window) using the flag set proven in `polaris_rebaseline_nsys.pbs`.

⚠️ **An `NSYS=1` run is truncated by design.** makani's `CUDAProfiler` calls
`sys.exit(0)` at `capture_range_stop` (`exit_on_stop` defaults `True`), so the run
ends mid-epoch, `total_train_s` is empty, and the step average covers fewer steps —
under profiler overhead besides. Hence the **`nsys` column**: those rows carry the
per-rank captures but must never be averaged in with the full-length arms.

**Mandatory, and silent if dropped:** `mpiexec --cpu-bind depth -d 8`. Without it
aws-ofi-nccl's progress engine starves and inter-node all-reduce drops **9.1×**
(4.08 vs 36.93 GB/s busbw, colleague's job 7368993). On a scaling study that failure
would masquerade as the result.

## 4. Prereg — predictions recorded before the first job

Written before any makani multi-node run exists. Scored honestly afterwards,
including the misses.

1. **Multi-node is not free.** Arm C's `step_ms` will exceed arm A's by **≥10%**.
   *Falsified if* C is within 5% of A — which would extend §0b's "comms are free"
   across Slingshot and make the 100-epoch arithmetic much friendlier.
2. **The ported fabric block works on torch 2.8.0's NCCL.** `transport` will read
   `AWS Libfabric` on every B/C/D row. *Falsified by* `UNKNOWN` or a TCP/socket
   fallback — the block's values were measured on torch 2.10.0+cu129 with a
   different bundled NCCL, so this is a genuine open question, not a formality.
3. **Wireup grows.** Arm C's `wireup_s` will be **> 2×** arm A's. Communicator
   setup across 16 ranks on 4 nodes builds strictly more process groups over a
   slower fabric. *Falsified if* C ≤ 1.5× A.
4. **Part of any real-data loss is I/O, not comms.** Arm D's A→C-equivalent
   degradation will be **smaller** than arm C's. *Falsified if* D degrades within
   2% of C — which would mean the loss is NCCL and the shared filesystem is
   innocent, and would kill the "pack more years" line of attack.
5. **NUMA pairing is second-order for this model.** Arm E will differ from arm A by
   **< 5%**. *Falsified if* reversing the GPU order moves step time by more — which
   would give §4.4e's host stall a concrete, output-neutral cause worth fixing
   everywhere, Pangu included.

Prediction 1 is the one that matters; 2 is the one most likely to fail for a boring
reason.

## 5. How to run

```bash
# once, login node
bash makani_sfno/polaris/polaris_setup_makani_h5py_overlay.sh   # MAKANI_H5PY_OVERLAY_OK  [DONE]
python makani_sfno/polaris/test_parse_makani_scaling.py         # MAKANI_SCALING_PARSE_OK  [DONE]

cd makani_sfno
qsub polaris/polaris_makani_env_probe.pbs                       # MAKANI_ENV_OK   <- gate, do not skip

# then the sweep (same script, three node counts)
qsub                  -l select=1:system=polaris polaris/polaris_makani_multinode_scaling.pbs
qsub                  -l select=2:system=polaris polaris/polaris_makani_multinode_scaling.pbs
qsub -q debug-scaling -l select=4:system=polaris polaris/polaris_makani_multinode_scaling.pbs

# arms D / E / F
qsub -q debug-scaling -l select=4:system=polaris -v DATA=synthetic  polaris/polaris_makani_multinode_scaling.pbs
qsub                  -l select=1:system=polaris -v GPU_ORDER=reverse polaris/polaris_makani_multinode_scaling.pbs
qsub -q debug-scaling -l select=4:system=polaris -v HPAR=2,WPAR=2   polaris/polaris_makani_multinode_scaling.pbs

# per-rank NCCL, once the arms above are green
qsub -q debug-scaling -l select=4:system=polaris -v NSYS=1 polaris/polaris_makani_multinode_scaling.pbs
```

`NNODES` is derived from `$PBS_NODEFILE`, so one script serves every node count and
a command-line `-l select` overrides the directive. `debug` allows 1–2 nodes; 4
nodes needs `debug-scaling` (≤1 h, one running job per user).

PASS is **`MAKANI_MN_SCALING_OK`** plus a new row in
`$MEMBER_ROOT/bench/makani_multinode_scaling.csv` — not `rc=0` (CLAUDE.md #14).

## 6. What would invalidate a row, and is checked automatically

* **N independent `world_size=1` trainers.** The launcher's worst failure: without
  the PALS rank shim, physicsnemo's `DistributedManager` warns *"Assuming this is a
  single process job"* and every rank trains alone at a plausible step time.
  `PHYSICSNEMO_DISTRIBUTED_INITIALIZATION_METHOD=ENV` makes it a hard error, and the
  parser rejects any row whose logged `world_size` ≠ the launched rank count.
* **An epoch that wraps.** Re-serving cached samples makes a rank look fast, and the
  wrap point moves with the global batch — i.e. *along the scaling axis*. Gated on
  the real sample count read from the pack, not the file count.
* **An unnamed transport.** A row that cannot say which network carried it is not
  evidence about an interconnect. Warned on, and recorded as `UNKNOWN`.
* **Schema drift in the CSV.** Columns are a cross-run contract (CLAUDE.md #10);
  appending under a changed header is refused rather than silently done.

## 7. Consequence not in scope here: every *existing* makani launcher is dead

Deliberately **not fixed in this change** — flagging it beats silently touching
seven files inside a `git subtree` (CLAUDE.md: keep subtree edits minimal and
contiguous), and none of them is on the multi-node path. But nobody should
discover it by submitting one:

```
makani_sfno/polaris/polaris_sfno_smoke.pbs
makani_sfno/polaris/polaris_sfno_full.pbs
makani_sfno/polaris/polaris_sfno_full_probe.pbs
makani_sfno/polaris/polaris_sfno_alldata_smoke.pbs
makani_sfno/polaris/polaris_sfno_alldata_full.pbs
makani_sfno/polaris/polaris_pack_e3sm_full.pbs
makani_sfno/polaris/polaris_pack_e3sm_alldata_full.pbs
```

All seven open with the bare `module load conda` / `conda activate base` pair and
therefore fail immediately with `conda: command not found` → `python: command not
found`. **This includes the two data packers**, so the full E3SM pack that
`polaris_sfno_full.pbs` needs cannot currently be built either — worth knowing
before planning around it. It also means the green results they produced
(**7253465**, `CONVERT_OK` 7252728) are not reproducible today without this fix.

The revival is mechanical once `MAKANI_ENV_OK` is recorded — replace

```bash
module use /soft/modulefiles
module load conda
conda activate base
source "${MAKANI_ROOT}/../polaris_env.sh" || exit 2
```

with

```bash
source "${MAKANI_ROOT}/../polaris_env.sh" || exit 2          # FIRST: defines MEMBER_ROOT
source "${MAKANI_ROOT}/polaris/polaris_makani_env.sh" || exit 2
```

— note the **order flip**, which is load-bearing (§2, and the header of
`polaris_makani_env.sh`). Each of the seven also needs the h5py-overlay
precondition, since they all import makani and therefore h5py.

Do this **after** the probe is green, not before: patching seven launchers against
an environment that has not yet been shown to import torch would just multiply one
unknown by seven.

## 8. Status

- [x] env blocker cleared — `polaris_makani_env.sh`, 0 dangling libs (static check)
- [x] h5py overlay built and green — `MAKANI_H5PY_OVERLAY_OK`
- [x] launcher written — `polaris_makani_multinode_scaling.pbs`, 1/2/4 nodes, one file
- [x] result parser + 11 tests green — `MAKANI_SCALING_PARSE_OK`
- [x] prereg recorded (§4) — **before** any job
- [x] **`MAKANI_ENV_OK` — job 7551240, 84 s.** torch 2.8.0 / CUDA 12.9 / NCCL 2.28.3 imports,
      `device_count=4`, cuBLAS + `RealSHT(180,360)` run, h5py from the overlay, `torch_harmonics`
      from the venv, and **4/4 ranks up through the PALS shim** (`world_size=4`, all-reduce
      correct) ⇒ makani is no longer confined to `--standalone`. Single-node, so prediction 2 is
      untested.
- [x] **2026-08-23 — `module load conda` re-confirmed BROKEN a third time** (`gcc-native/14.2`,
      `cray-hdf5-parallel/1.14.3.5` still unknown; `which python` → none). The ALCF ticket is
      unfixed 3 days on, so `polaris_makani_env.sh`'s reconstruction is not a stopgap — it is
      the path. Prerequisites re-verified on disk: `sfno-venv`, `makani-h5py-overlay`,
      `data/e3sm_makani` (train/valid/test/stats/metadata). Parser re-run green,
      `MAKANI_SCALING_PARSE_OK 11 tests` — note it needs `/usr/bin/python3.11`; the login
      node's bare `python3` is **3.6.15** and dies on `from __future__ import annotations`.
- [x] **fabric pin repaired — `FABRIC_PROBE_OK`, job 7553823.** The ported block pinned
      `/opt/cray/libfabric/2.2.0rc1` (gone; ALCF ships 2.3.1) and `aws-ofi-nccl v1.9.1-aws`,
      which segfaults against 2.3.1. Of six installed pairings **only
      `v1.6.0-libfabric-1.22.0` + cray 2.3.1** works. New `polaris_makani_fabric_probe.pbs`;
      every fabric dir is now existence-checked with a hard `exit 3`, because the original
      defect was that a load-bearing pin could vanish *silently*.
- [x] **pack rebuilt — `CONVERT_OK` + `PACK_SCALING_OK`, job 7553851.** The old pack was a
      **400-sample smoke** (`--max-samples-per-year 400`), which is why arms B/C were refused.
      `polaris_pack_e3sm_scaling.pbs` (new; applies §7's two-line swap instead of editing seven
      subtree files) packed 4 full years in **23m36s** → train 2015–2016 = **2,920 samples**,
      good for STEPS=60 to **12 nodes**. Select with `-v PACK=`.
- [ ] arms A/B/C × 3 reps — **rep 1: A ✅ 7553890 (114.9 ms), B ✅ 7553897 (174.6 ms, +52%),
      C queued 7553891.** ⚠ `debug` is `max_queued 1` per *user*, so A and B cannot be queued
      together; `debug-scaling` (arm C) is a separate queue and can overlap. Interleave reps
      rather than batching them — §3. ⚠ **All arms in a comparison must share one pack:** arm A
      read 196.5 ms on the smoke pack and 114.9 ms on the new one — 41% from data alone.
- [x] **arm F driven to a definitive verdict (2026-08-24/26): spatial parallelism works
      single-node (7563780, 219.5 ms — first spatial measurement here) and is BLOCKED
      multi-node by the plugin**, which loses progress on subgroup-communicator small-message
      storms at ≥3 nodes (evidence chain: IMA 7554253 / hang 7554367 / serialized-and-still-hung
      7563723 `enqueued 84, completed 83` / single-communicator probe passes 7554170).
      Application-side fixes exhausted — `_serialized_sync_params` (kept, hygiene) was the last
      one available. Unblock = modern aws-ofi-nccl vs libfabric 2.3.1: ALCF ticket, or
      self-build into `$MEMBER_ROOT`.
- [ ] arms D/E
- [ ] per-rank NCCL (`NSYS=1`), then the `NCCL_PROTO`/`NCCL_ALGO` sweep §0b defers to here
      (note the sweep's LL arms will re-hit the deadlock until the plugin is rebuilt)
- [ ] score the prereg, write the results section

## 9. The 100-node goal, and what actually caps it

The stated destination is **100 nodes**. Recording now, before any arm reports, what stands
between here and there — because the binding constraint is *not* the one the plan is built to
measure.

**⚠️ CORRECTED 2026-08-23.** An earlier draft of this section called the data itself the
constraint and put the requirement at "≈14 training years" as if that were out of reach. **That
was wrong, and the operator was right to push back: it is the same E3SM data, just in a
different format.** The corrected version:

**All 35 years are already staged.** `${E3SM_ROOT}/h5/plev_data/` holds **51,104 files covering
2015–2049**, 1460 samples/year, which `convert_e3sm_to_makani.py` documents as its input
contract. What is only 1 year is the **packed** form: `data/e3sm_makani/train/` contains exactly
`2015.h5`. So the gap is a **format repack, not a data gap** — per-sample E3SM h5 → makani's
"multifiles" `{split}/{year}.h5` contract, and the year range is a CLI flag
(`--train-years START END`).

**And it is cheap.** The converter's own header: **~5–10 min/year, ~22 GB/year**. Thirty
training years is therefore ≈**3–5 h of packing and ~660 GB** — against **32 T free** on eagle.
Not a research problem; an afternoon of I/O.

**And the starvation was worse than "not enough for 100 nodes" — it blocked the deliverable.**
Measured 2026-08-23: the original pack holds **400 train samples**, not 1460, because it was
built `--max-samples-per-year 400`. Arm A (60 × 4 = 240) fit; **arms B (480) and C (960) were
refused outright** by the wrap guard. So the pack was the binding constraint on 2 and 4 nodes,
never mind 100.

**Fixed, and the fix is now proven end-to-end.** `polaris_pack_e3sm_scaling.pbs` packed 4 full
years in **23m36s** — train 2015–2016 = **2,920 samples**, enough for STEPS=60 at **12 nodes**.
It also demonstrates §7's two-line bootstrap swap on real work, so reviving the seven dead
launchers is no longer a hypothesis.

⚠️ **A pack is not interchangeable with another pack.** Arm A measured **196.5 ms** on the
400-sample pack and **114.9 ms** on the new one — **41% from the data alone** (`io_gbs` 0.52 →
0.88). Every arm in one comparison must therefore be run against one pack, and any table must
name it.

Arithmetic for the climb, at STEPS=60 and 1 sample/GPU (global batch = 4 × nodes, samples
needed = 240 × nodes):

| nodes | global batch | samples needed | train years |
|---|---|---|---|
| 4 | 16 | 960 | <1 |
| 12 | 48 | 2,880 | 2 (**have it**) |
| 32 | 128 | 7,680 | 6 |
| 100 | 400 | **24,000** | **~17** |

Seventeen of the 35 staged years, at ~5–10 min and ~22 GB each ⇒ ≈**2–3 h and ~370 GB**. Against
32 T free, that is scheduling, not a research problem.

⇒ **Ladder: arms A–C × 3 reps on the 4-year pack → extend the pack to ~17 years → climb toward
100.** Arms A–C remain the right next measurement: extrapolating needs the *shape* of the curve,
and rep 1 already shows it is not flat.
