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

* **Failure threshold:** between 25 MB and 1000 MB. Not 2³¹ or 2³² bytes — both
  were tested and are not the boundary.
* **Node-count independent** (jobs 7569817 vs 7571147): fails at 2 nodes and at
  8. Assume it fails at 16/48/64 too.
* **Why Ring escapes:** ring's reduce-scatter gives each rank `S/N`, which falls
  under the threshold and *keeps shrinking with scale*; tree's per-link message
  does not shrink with N. (Mechanism is inference; the failures are measured.)
* **Why makani never hit it:** its ~150 M-param model reduces ~0.6 GB, below the
  threshold. Same cluster, same plugin, same NCCL.

**⇒ ACE2 is 455,831,040 parameters = 1.82 GB fp32 gradients. Above the
threshold. Set `NCCL_ALGO=Ring` from the first multi-node job.** If you skip
this you will spend a day rediscovering it, and the intermediate symptom is a
`ProcessGroupNCCL watchdog got stuck` message that points nowhere.

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
  Note this is a **single merged NetCDF**, not a per-year store — a different
  I/O shape from ai-rossby's 30 zarr stores, and every rank reads the same file.
* **Prior profiling context:** `ACE2_retrain/bench_midway_notes.md` already
  records that ACE2 is the first *training* workload profiled on those nodes
  with heavy gradient traffic, that its dhconv weight is 384×384×180 =
  212.34 MB/layer (~93% of params), and that `torch.compile` is not the lever
  (`InductorError: KeyError: 'complex64'` on the complex64 SFNO hot path).

## 3. The three ACE2-specific blockers (none of which ai-rossby had)

1. **⚠ `fme` requires `batch_size % world_size == 0`.** The Midway script
   enforces it and aborts with `ACE2_BATCH_NOT_DIVISIBLE`. The production config
   uses `batch_size: 16`, so **world_size cannot exceed 16 (4 nodes) without
   changing the batch** — and changing it is a *science* change (jesswan).
   This CAPS the ladder at 4 nodes on the shipped config. Decide early:
   either ladder to 4 nodes only, or get sign-off for a larger batch. Do not
   quietly bump it to make a scaling table look better.
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
- [ ] **Ladder** 1/2/4 nodes (see §3.1 on the 4-node cap), ≥3 interleaved reps,
      one config, one store.
- [ ] **Ticket material:** ACE2 hitting the same tree defect at 1.82 GB would be
      a *third independent harness* confirming it — worth adding to the ALCF
      ticket alongside makani and ai-rossby.

## 5. Prereg — write these before the first job, score them after

Suggested predictions (falsifiable, with the condition stated):

1. **ACE2 hits the tree defect.** With the default algorithm a 2-node run hangs
   in the first gradient all-reduce; with `NCCL_ALGO=Ring` it trains.
   *Falsified if* the default works — which would mean the threshold is above
   1.82 GB and would usefully narrow it (we only know it is between 25 MB and
   1000 MB… and ACE2 at 1.82 GB would then contradict that, so this is a real
   test of our threshold, not a formality).
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
