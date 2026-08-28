# HANDOFF — port multi-node DDP to ACE2 on Polaris

Third harness of `polaris_multinode_ddp_port_handoff.md` (makani → ai-rossby →
**ACE2**). Read this, then CHANGELOG `2026-08-28` (the ai-rossby campaign this
generalises), then `ai_rossby_multinode_ddp_plan.md` for the prereg method.

**The one thing to read first (§1): a fabric defect on Polaris silently corrupts
and then hangs `all_reduce` above ~1 GB. ACE2's gradient all-reduce is 1.82 GB.
It WILL hit this. `NCCL_ALGO=Ring` is the fix and it is not optional.**

Definitions: `$MEMBER_ROOT` = `/eagle/projects/lighthouse-uchicago/members/mehta5`
(exported by `polaris_env.sh`). "Ladder" = weak-scaling arms at 1/2/4/8 nodes,
60 timed steps each, interleaved reps.

---

## 1. What transfers from ai-rossby — read before writing any launcher

### 1a. ⚠ THE RING/TREE DEFECT — the finding that matters most

**Measured on Polaris, app-free (no PyTorch model, no DDP, just NCCL):**

| traffic | size | result |
|---|---|---|
| `broadcast` | 4.7–4.9 GB | ✅ completes, 2 and 8 nodes |
| `all_reduce`, 190 × 25 MB | 25 MB each | ✅ completes |
| `all_reduce`, default algo (**Tree**) | 1000 / 2000 / 4000 / 4700 MB | ❌ **fails, 2 nodes AND 8 nodes** |
| `all_reduce`, **`NCCL_ALGO=Ring`** | 2000 / 4700 MB | ✅ completes correctly |

It is **not a clean hang — it silently corrupts first.** With the correctness
check repaired, a 2000 MB tree all-reduce returned
`first=8.000 last=1.000 want=8 WRONG`: the start of the buffer correctly
reduced, the end untouched at its input value, and *different ranks disagreeing*
(`ranks OK: 1/8`, `3/8`). Had the watchdog not fired, training would have
continued on gradients correct at one end of the tensor and stale at the other.

* **Failure threshold: between 25 MiB and 1000 MiB, FOR THE TREE ALGORITHM.**
  Not 2³¹ or 2³² bytes — both were tested and are not the boundary.
  ⚠ State it as a *Tree* threshold, not a size threshold: in the table above
  size and algorithm covary (every failing row is Tree; the passing all-reduce
  rows are either small or Ring). A 4.7 GB *broadcast* passes, which a
  size-only story cannot explain. Probe sizes are MiB (`mb*1024*1024/4`
  elements); ACE2's "1.82 GB" is decimal = 1.70 GiB. At a claimed boundary that
  ~7% ambiguity matters.
* **Node-count independent** (jobs 7569817 vs 7571147): fails at 2 nodes and at
  8. Assume it fails at 16/48/64 too.
* **Why Ring escapes:** ring's reduce-scatter gives each rank `S/N`, which falls
  under the threshold and *keeps shrinking with scale*; tree's per-link message
  does not shrink with N. (Mechanism is inference; the failures are measured.)
* **Why makani never hit it:** ⚠ an earlier draft said "its ~150 M-param model
  reduces ~0.6 GB, below the threshold". That is the SAME total-volume error
  corrected below — makani buckets too, so its largest *message* was never
  0.6 GB. The conclusion (makani was unaffected) holds; the stated mechanism did
  not, and it is left here as a worked example of the trap.

### ⚠ CORRECTED 2026-08-28 after adversarial review — READ THIS, the first
### version of this section was WRONG

An earlier draft said: *"ACE2 is 455,831,040 parameters = 1.82 GB fp32
gradients. Above the threshold. It WILL hit this."* **That inference does not
follow, and the error is instructive enough to keep.**

The defect triggers on the size of an **individual collective**, not on total
gradient volume. DDP buckets gradients and issues many all-reduces per step, so
total volume is the wrong quantity entirely.

What is actually established:

* **fme wraps with stock DDP and never sets `bucket_cap_mb`** —
  `ace_exp/fme/core/distributed/torch_distributed.py:182-193`:
  `DistributedDataParallel(..., gradient_as_bucket_view=True,
  broadcast_buffers=False)`. No `static_graph`, no comm hook, no FSDP/ZeRO, no
  gradient accumulation. `bucket_cap_mb` appears **zero times** in all of
  `ACE2_retrain/`.
* **This repo already measured the answer** — `ACE2_retrain/PROFILING_PLAN.md:171`:
  *"it is **11.4 buckets/step, ~165 MB each** — not the ~70 that a 25 MiB cap
  predicts."*
* DDP never splits a single parameter across buckets, so the dhconv weight
  (384×384×180 = **212.34 MB**) gets its own bucket. **~212 MB is therefore the
  largest single gradient collective in an ACE2 step.**

**⇒ ACE2's exposure is UNKNOWN, not certain.** ~165–212 MB sits in the gap
between our largest *passing* all-reduce probe (25 MB) and our smallest
*failing* one (1000 MiB). Nobody has tested that range.

**⚠ And the obvious counter-argument — "bucketing keeps every message small, so
ACE2 is safe" — is ALSO not established.** ai-rossby's flight recorder showed a
**single all-reduce of the entire 1.18 B-parameter model** (`numel=1182108160`),
in BOTH the default-25 MB-bucket run (7569744) and the forced-one-bucket run
(7569690) — byte-identical stuck collectives under a 200× difference in
`bucket_cap_mb`. ai-rossby also uses `gradient_as_bucket_view=True`. Why DDP
coalesced there is recorded in CHANGELOG as **still open**. Until that is
understood, do not assume ACE2 buckets the way the config implies.

**What to do about it (cheap, decisive, no new allocation):**
1. Run the existing 2-node smoke with the flight recorder on and **read the
   actual per-collective sizes** — `PROFILING_PLAN.md:240` already specifies
   this dump. That single artifact settles exposure before any Polaris
   multi-node allocation is spent.
2. **Set `NCCL_ALGO=Ring` anyway.** It is one env var, it costs nothing
   measurable at ≤32 ranks (ai-rossby's whole ladder ran on it), and it removes
   the failure mode regardless of which way (1) lands. Insurance, not a
   diagnosis.
3. Probe the **25 MB → 1000 MiB gap** app-free with
   `physicsnemo_ai_rossby/polaris/polaris_ai_rossby_nccl_mn_probe.pbs`
   (`-v BUCKET_MB=165 -v N_BUCKETS=12`, then 212) if you want the threshold
   pinned. Cheap, and it is the missing row in §1a's table.

If it does hang anyway, the symptom is a `ProcessGroupNCCL watchdog got stuck`
message that points nowhere — use the diagnosis kit below.

**Diagnosis kit, if it happens anyway.** The flight recorder is what named the
collective, and **three settings are needed together** — the buffer alone
captures nothing readable:
```
TORCH_NCCL_TRACE_BUFFER_SIZE=2000   TORCH_FR_BUFFER_SIZE=2000
TORCH_NCCL_DUMP_ON_TIMEOUT=1
TORCH_NCCL_DEBUG_INFO_TEMP_FILE=<a path on EAGLE>   # default /tmp is node-local, dies with the job
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300                # SHORT: see below
```
⚠ Do **not** raise the heartbeat hoping the collective timeout will name the op.
Evaluating that timeout is the *watchdog's* job and the watchdog is what is
stuck — only the monitor thread can report this class of hang. Raising it to
1800 cost two jobs' diagnostics to walltime here.
Read the dump with `pickle.load`; `state=scheduled` on all ranks with an
identical `collective_seq_id` means enqueued-and-never-launched.

### 1b. The fabric stack (unchanged, still mandatory)

Self-built **aws-ofi-nccl v1.21.1** (`$MEMBER_ROOT/sw/aws-ofi-nccl-1.21.1`) +
cray libfabric 2.3.1 + **`OFI_NCCL_PROGRESS_MODEL=AUTO`**. Re-measured under
ai-rossby's torch 2.10/NCCL 2.27.5: `C_progress_auto` is the *only* working
combo of six, exactly as on makani's 2.8.0/2.28.3 — so the pin is CXI-side and
NCCL-version-independent. Everything else fails `fi_domain` with ENOSYS.
Existence-check every pinned dir and hard-exit: a missing dir on
`LD_LIBRARY_PATH` is **ignored, not honoured**, and the run then measures a
different transport before crashing somewhere unrelated.

### 1c. Traps that cost real time here — do not pay twice

1. **`MASTER_PORT` must be unique per job.** A hard-coded 29500 hung **four**
   jobs with every rank blocked in the TCPStore rendezvous — imports complete,
   then `init_process_group` never emits a line. A green run prints
   `NCCL INFO NCCL version` *immediately* after the last import warning; a
   stalled one stops dead at that boundary. Cause: an orphaned rank from a
   `qdel`'d job holding the port on a re-allocated node. Use
   `20000 + jobid%20000`.
2. **`export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"` DOES NOTHING.** PBS Pro
   exports `OMP_NUM_THREADS=<ncpus>`=64 into every job, so the `:-` default is
   dead code. All 30 makani rows and the early ai-rossby rows ran at 64 threads
   on 8 cores. Take the override from a name you own (`OMP_THREADS`). (Measured
   impact at 1 node: none — but the comment claimed a pin that never applied.)
3. **Invoking a venv python directly under PALS `mpiexec` loses the venv.**
   `bin/python` is a symlink into the base conda; PALS execs the resolved
   target, so `pyvenv.cfg` is never found and base-conda packages get imported.
   A GPU-health preflight written this way reported **5 healthy nodes as sick**
   and read exactly like an ALCF hardware fault. Always go through an
   intermediate shell: `bash -c "exec '${PY}' '${SCRIPT}'"`.
4. **Never let a correctness check compare a value against itself.** An
   app-free probe did `all_reduce(buf); buf.fill_(1.0)` then asserted
   `buf[0]==1.0` — vacuous, and it reported `ranks OK: 8/8` while the watchdog
   was simultaneously reporting that same collective timing out. Check *before*
   overwriting, and sample **both ends** of a large buffer: the real failure is
   a partial reduction that index 0 alone cannot see.
5. **Read the raw log tail before theorising.** Two separate detours here were
   spent explaining why a knob might be slow when the job had never reached the
   knob's code at all (34 min inside `import`, and a rendezvous stall). One
   `tail` settled both.
6. **A nonexistent path on `LD_LIBRARY_PATH` is silently ignored** (§1b).
7. **GPU-health preflight + one spare node is mandatory at ≥8 nodes.** It earned
   its keep on its first prod run: `48 healthy of 49 allocated` — one sick node
   pruned instead of killing a 48-node allocation.

### 1d. Queue geography (verified `qstat -Qf` 2026-08-28)

| queue | nodes | max walltime |
|---|---|---|
| `debug` | 1–2 | 1 h |
| `debug-scaling` | 1–10 | 1 h |
| `prod` → `small` | 10–24 | **3 h** |
| `prod` → `medium` | 25–99 | **6 h** |

⚠ **`prod`'s 24 h is the ROUTING queue's limit, not the execution queue's.** A
48-node job lands in `medium` and gets **6 h**. Any long run at ≥25 nodes is a
*chain* of ≤6 h links, so **checkpoint + resume is load-bearing, not a safety
net.** Also: one job per queue in Q/H state, but `-W depend=afterany:` jobs sit
in `H` and a `debug` H + `debug-scaling` Q pair is accepted.

### 1e. What ai-rossby measured, for calibration

1.18 B params, 4.73 GB gradients, `NCCL_ALGO=Ring`, 60 steps, one store:

| nodes | ranks | step_med | samples/s | speedup | efficiency |
|---|---|---|---|---|---|
| 1 | 4 | 698.7 ms | 5.73 | 1.0× | 100% |
| 2 | 8 | 2082.6 ms | 3.84 | 0.7× | 33.5% |
| 4 | 16 | 2484.0 ms | 6.44 | 1.1× | 28.1% |
| 8 | 32 | 2867.2 ms | 11.16 | 1.9× | 24.4% |
| 16 | 64 | 2881.5 ms | 22.21 | 3.9× | 24.2% |
| 48 | 192 | 3390.9 ms | 56.62 | 9.9× | 20.6% |

Shape: **the cliff is the first hop (×2.97), then it saturates** — 8→16 nodes
costs 0.5% of step time for double the ranks. 2 nodes is a throughput *trough*
(0.7×); it only pays from 4 nodes up. Do not generalise from a 2-node point —
that error is recorded in CHANGELOG because I made it.
⚠ Per-arm rep spread: 1n 0.1%, 4n 2.5%, 8n 4.0%, but **2n 15%** across 5 reps.
Anchor claims on the tight arms.

---

## 2. ACE2 survey — what is already here (established 2026-08-28)

**This harness is further along than the parent handoff's §3c assumed.** It is
not a green-field bring-up.

* **Code:** `ACE2_retrain/` — profiling harness (`ace2_nvtx.py`,
  `kernel_census.py`, `nvtx_phase_attribution.py`, `parse_nsys.py`, all with
  tests), configs, and Midway scripts. `ACE2_retrain/ace_exp/` is the vendored
  ACE/`fme` tree (has its own `CLAUDE.md` and `AGENTS.md` — read them).
* **Entrypoint:** `python -m fme.ace.train <config> --override ...`, with
  `python -m fme.ace.validate_config` as a cheap pre-check. **Use both.**
* **Existing multi-node anchor:** `ACE2_retrain/midway_smoke_train_2node.sh`,
  PASS token **`ACE2_SMOKE_2NODE_OK`** — a *SLURM* 2-node smoke using
  `srun python -m torch.distributed.run --nnodes N --rdzv_backend c10d`
  with `--ntasks-per-node=1`. So ACE2 already works multi-node **on Midway**;
  what is missing is the **Polaris/PBS** path.
* **Model: 455,831,040 trainable parameters (455.8 M)** → **1.82 GB** fp32
  gradient all-reduce. §1a applies.
* **Data on Polaris (given by the operator):**
  ```
  /eagle/projects/lighthouse-uchicago/ace2/ace_training/merged_ACE2_ERA5_final.nc
  /eagle/projects/lighthouse-uchicago/ace2/normalization/{centering,scaling-full-field,scaling-residual,time-mean}.nc
  ```
  ⚠ **`merged_ACE2_ERA5_final.nc` is 2,388.77 GB — a single 2.4 TB NetCDF**,
  not a per-year store. The normalization files are trivial (≤20 MB). This is a
  fundamentally different I/O shape from ai-rossby's 30 zarr stores and it is
  the biggest unknown in this port:
  * **every rank reads the same file** — at 4 nodes that is 16 ranks against one
    Lustre object set; ai-rossby's loader sharded across 30 separate stores.
    Check the striping (`lfs getstripe`) before blaming NCCL for a slow step.
  * `HDF5_USE_FILE_LOCKING=FALSE` is already exported by `polaris_env.sh` and is
    **required** on Lustre — a bare netCDF/h5py open without it can hang.
  * The wrap-guard arithmetic still applies but must be derived from this file's
    real sample count, not from a file count.
  * ⚠ It also means **`gpu_busy_frac` is the metric to watch first.** ai-rossby
    ran ≥0.976 on every arm, so its penalty was provably inside the collective.
    If ACE2 comes in materially lower, the loss is I/O and the whole
    ring/tree/comms analysis is the wrong lens for it.

### 2a. ✅ CONVERT THE NetCDF TO ZARR — operator call, and fme supports it natively

**Verified 2026-08-28: `fme` reads zarr for TRAINING data, not just inference
output.** `XarrayDataConfig` takes `engine` and `file_pattern`, and the tests
exercise exactly this:
```python
XarrayDataConfig(data_path="foo", file_pattern=".zarr", engine="zarr")
```
⇒ **no fme code change is needed** — conversion is a config change
(`engine: "zarr"`, `file_pattern: "*.zarr"`) plus the conversion job.

Why it is worth doing here specifically:
* one 2.4 TB NetCDF read concurrently by every rank is the worst shape for
  Lustre; per-year/per-chunk zarr stores let ranks touch disjoint objects, which
  is what ai-rossby's loader does and part of why it held `gpu_busy_frac` ≥0.976;
* zarr chunking can be matched to the sample/time access pattern, which a
  monolithic `.nc` cannot;
* it removes the `HDF5_USE_FILE_LOCKING` hazard class entirely.

⚠ **One fme-specific gotcha already handled upstream, do not undo it:**
`fme/ace/data_loading/getters.py:100` forces the **forkserver** multiprocessing
start method whenever the zarr engine is used with `num_data_workers > 0`. Leave
that alone, and keep `TMPDIR=/tmp` in the launcher — Polaris' default TMPDIR on
eagle exceeds the 108-byte AF_UNIX `sun_path` limit and kills DataLoader workers
(the exact failure recorded in the parent handoff's ledger #10).

**Conversion precedent in this repo** — reuse rather than reinvent:
* `physicsnemo_sfno/polaris/e3sm_h5_to_seqzarr.py` + `polaris_zarr_e3sm_*.pbs`
  (chunked writes, PBS-shaped, PASS-token'd)
* `physicsnemo_ai_rossby/tools/data/e3sm/pangu_h5_to_zarr.py` (per-year stores,
  writes the variable metadata into each store's `attrs`)
* `physicsnemo_sfno/polaris/verify_seqzarr.py` / `polaris_verify_store.pbs` —
  **write a verifier and run it before training on the output.** A conversion
  that silently permutes channel order produces correctly-shaped tensors that
  nothing downstream objects to; that class of bug cost this project a
  ~110-node-hour restart on the wandb-key equivalent.

**Budget it honestly:** 2.4 TB read + write is a multi-hour, I/O-bound job and
roughly doubles the storage footprint until the `.nc` is retired. Check quota
first, and confirm with the operator before deleting the original.
* **Prior profiling context:** `ACE2_retrain/bench_midway_notes.md` already
  records that ACE2 is the first *training* workload profiled on those nodes
  with heavy gradient traffic, that its dhconv weight is 384×384×180 =
  212.34 MB/layer (~93% of params), and that `torch.compile` is not the lever
  (`InductorError: KeyError: 'complex64'` on the complex64 SFNO hot path).

## 3. The three ACE2-specific blockers (none of which ai-rossby had)

1. **`fme` requires `batch_size % world_size == 0`.** The Midway script enforces
   it and aborts with `ACE2_BATCH_NOT_DIVISIBLE`. The production config uses
   `batch_size: 16`, so at batch 16 world_size cannot exceed 16 (4 nodes).
   ✅ **OPERATOR DECISION (2026-08-28): raising the batch is approved**, so the
   ladder is not capped at 4 nodes. Two things still follow from the constraint
   and must not be lost:
   * the global batch must stay divisible by world size at **every** arm, so
     pick a batch that divides 4/8/16/32/… (e.g. 32 or 64), not an arbitrary one
     tuned per arm — a table whose arms use different batches is not a weak
     scaling ladder;
   * a larger batch is still a **numerics** change, so the LR moves with it.
     ai-rossby's lesson: the config's linear rule and the sqrt rule differed
     **7×** at large batch, and neither had been measured. Run a short flat-LR
     sweep (see `-v FLAT_LR=1` in
     `physicsnemo_ai_rossby/polaris/polaris_ai_rossby_multinode_scaling.pbs`)
     before committing a long run — ~5% of the run's cost to find the
     divergence boundary. Note ACE2 anneals its own LR, so pin the schedule flat
     for the sweep or every arm will look identical.
2. **Memory:** the config's own header says batch_size 16 is sized for Delta
   **GH200s** and is "NOT known to fit" a 40 GB A100. Polaris is 40 GB A100.
   Expect to need a smaller per-rank batch — which interacts with (1), since
   the divisibility constraint is on the *global* batch.
3. **Distributed init differs from ai-rossby.** ACE2 uses
   `torch.distributed.run` with `--rdzv_backend c10d`; ai-rossby uses the PALS
   rank shim + `env://`. On Polaris **never `srun`** (CLAUDE.md). Two options,
   pick deliberately and record why:
   * `mpiexec --ppn 4 bash polaris_rank_env.sh python -m fme.ace.train …`
     (the shim exports RANK/WORLD_SIZE/LOCAL_RANK; needs `fme` to honour
     `env://`), **or**
   * `mpiexec --ppn 1 python -m torch.distributed.run --nnodes N
     --nproc_per_node 4 --rdzv_backend c10d …` (one launcher per node, closest
     to the working Midway recipe).
   Find `fme`'s `init_process_group` call in `ACE2_retrain/ace_exp/` and let
   that decide, rather than assuming.

## 4. Plan

- [ ] **Convert the 2.4 TB NetCDF to zarr (§2a)** — operator-approved, fme reads
      it natively. Reuse an existing converter, ship a verifier, and check the
      channel order against the config before training on the output.
- [ ] **Survey, ~1 h, no allocation.** Locate `fme`'s `init_process_group`;
      confirm which venv/torch on Polaris (any torch ≠ 2.8.0 ⇒ re-run the three
      fabric probes under it, per the parent handoff §1); confirm the `.nc`
      files are readable and what `fme` expects `data_path` to contain.
- [ ] **Env bootstrap.** `module load conda` is **broken cluster-side** since
      the 2026-08 PE roll. Copy the pattern of
      `physicsnemo_ai_rossby/polaris/polaris_ai_rossby_env.sh`: try the module,
      fall back to a hand reconstruction, and report which path ran.
- [ ] **1-node smoke** on Polaris → its own PASS token. This is the anchor;
      multi-node work starts from a green single-node run.
- [ ] **2-node smoke with `NCCL_ALGO=Ring` from the outset** (§1a).
- [ ] **Parser + tests** adapted from
      `physicsnemo_ai_rossby/polaris/parse_ai_rossby_scaling.py`. Keep all five
      guards: world_size read only off the trainer's own banner (never off
      anything the launcher echoed), ranks-reporting from PALS labels, telemetry
      cross-check, step-count, transport. Absent banner = ERROR, not a pass.
- [ ] **Prereg committed BEFORE the ladder**, scored honestly afterwards.
- [ ] **Ladder** 1/2/4/8 nodes (the batch raise in §3.1 removes the old 4-node
      cap), ≥3 interleaved reps, one config, one store, one batch value that
      divides every arm's world size.
- [ ] **Ticket material:** ACE2 hitting the same tree defect at 1.82 GB would be
      a *third independent harness* confirming it — worth adding to the ALCF
      ticket alongside makani and ai-rossby.

## 5. Prereg — write these before the first job, score them after

Suggested predictions (falsifiable, with the condition stated):

1. **Largest gradient collective.** The flight-recorder dump of a 2-node run
   shows ACE2's largest `all_reduce` at **150-250 MB** (predicted ~165 MB
   bucket, ~212 MB for the standalone dhconv weight), NOT a single ~1.8 GB one.
   *Falsified if* it shows one full-model collective — which is what ai-rossby
   inexplicably did, and would make ACE2 exposed exactly as ai-rossby was.
   ⚠ This REPLACES an earlier prediction ("ACE2 hits the tree defect, falsified
   if the default works") which was misconceived: it conflated total gradient
   volume with per-collective size, so it would have been falsified for a reason
   that teaches nothing about the fabric, and — worse — a non-hang would have
   been mis-scored as evidence the threshold is above 1.82 GB.
1b. **Given ~165-212 MB collectives, the default algorithm does NOT hang** at
   2 nodes. *Falsified if* it does — which would pull the Tree threshold below
   212 MiB and is a genuinely new fabric result worth the ALCF ticket.
2. **The AUTO pin carries again.** `C_progress_auto` alone in the knob matrix.
3. **First-hop penalty.** ACE2's 1.82 GB is ~0.39× ai-rossby's 4.73 GB, so if
   the penalty tracks gradient volume, expect arm B − arm A ≈ 0.39 × 1384 ms
   ≈ **540 ms**, i.e. in 270–1080 ms. *Falsified outside that* — and a miss
   high would say the cost is not volume-linear after all.
4. **The cliff saturates:** 2→4 node growth ≤25%.
5. `gpu_busy_frac` ≥ 0.85 on every arm.
6. Rep spread < 5% per arm (ai-rossby missed this at 2 nodes: 15%).

## 6. Definition of done

A CHANGELOG entry containing: the fabric-probe result (or a logged waiver), a
ladder table with ≥3 interleaved reps per arm, the scored prereg **including
misses**, and an explicit statement of the batch/world-size constraint from
§3.1 and what was decided about it. Plus, if ACE2 reproduces the tree defect,
the app-free evidence added to the ALCF ticket.
