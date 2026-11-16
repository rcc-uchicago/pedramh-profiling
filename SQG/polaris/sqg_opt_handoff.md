# SQG solver optimization + multi-node on Polaris: handoff (written 2026-11-05)

> **Superseded for current work (2026-11-06 evening).** The live task is item 3 of the 2026-11-06 team plan, the
> generator-error taxonomy: archive run job 7824121. Start from `SQG/analysis/opt/taxonomy/HANDOFF_taxonomy.md`
> and its paste prompt. This file stays as the record of rounds 1–1c.

**Prompt for the fresh session (paste as-is):**

> Continue the SQG Polaris optimization work. Call EnterWorktree with
> `path=/lus/eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling/.claude/worktrees/sqg-polaris-profiling`,
> then read `SQG/polaris/sqg_opt_handoff.md` (this file) and follow §3. Don't re-profile; jobs A–F are done.

This file is a starting point, not the record. The record is `SQG/polaris/polaris_sqg_notes.md` (Parts I–VII) and
the 2026-11-04/05 SQG entries in `CHANGELOG.md`. Where this file and those disagree, they win.

> **Status update (2026-11-05, later the same day). §2–§5 below are superseded by notes Parts VIII–IX.**
> - 7815096 has been scored (Part VIII).
> - Round 1c = job **7815637** is DONE with `failures=0` (Part IX).
>
> Results:
> - **Seed 1:** `lean_u1` is **bitwise** and 1.451× (optimizer step 4.47 → 3.09 s).
> - **NCCL over CXI is fixed:** a dir holding `libcudart.so` → the env's own `libcudart.so.12` on
>   `LD_LIBRARY_PATH` (`run_round1c_polaris.pbs`, `arm_env B_shim`) gives 31.5 GB/s with GDRDMA.
> - **Multi-node DDP over CXI works:** about 5 ms per step.
> - **Replicate vs decouple:** equal GPU-seconds measured (0.94×). Decoupling gives 3.8× shorter steps at a smaller
>   global batch.
>
> Open for the user:
> - ship `lean_u1` to o1.2 (no sign-off needed), with `lean32` (the user's pick) after validation;
> - round 2 (the u1 trigger + C2C packing);
> - sizing a decoupled run.
>
> The decouple harness now reports the paced phase by its mean.

---

## 0. Orientation

| item | value |
|---|---|
| Worktree | `.claude/worktrees/sqg-polaris-profiling` (sparse: cone `SQG` + root files). Branch **`sqg-polaris-profiling`**, pushed. |
| PR | **draft #25** → base `sqg` (https://github.com/rcc-uchicago/pedramh-profiling/pull/25). Leave it draft; the maintainer reviews. |
| Head at handoff | `f12ce3aa` (round 1 results + CHANGELOG), on top of `bc8ab106` (round 1b code + 2-node smoke). |
| Env | `/eagle/projects/lighthouse-uchicago/members/mehta5/conda-envs/sqg-env` (py3.10, jax 0.4.28, torch 2.3.1; CUDA from the nvidia-* wheels, no module at run time). Built by `SQG/polaris/setup_sqg_env.sh`. |
| Model code | `SQG/o1.2-TrainUViT_m16_tendency.py` (online training), `SQG/fixed_ring/A-SQG_data.py` (solver; **hash-pinned** by `SQG/fixed_ring/SHA256SUMS`; never edit it). Optimizations go in o1.2's own step function. |
| Job logs | `/eagle/projects/lighthouse-uchicago/members/mehta5/sqg_profile/logs/<jobid>.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov.OU`; raw outputs `…/mehta5/sqg_profile/<arm>_<jobid>/`; committed text summaries `SQG/analysis/results/polaris_*_<jobid>/` (gitignored dir: `git add -f`). |
| Science owners | jesswan / Weidong. Anything that changes what is computed (STRIDE, steps per sample, precision in production, replay ratio) is their call, not ours. |

**Working rules learned the hard way in this worktree:**
- **Commands.** Use one plain command per call; the worktree guard refuses chained git, `bash script.sh` and big heredocs.
  - Commit with `git commit -q -F $CLAUDE_JOB_DIR/tmp/msgN.txt`.
  - Run analysis as `python3 <file>` or a short `python3 -c`.
  - Grep/find are denied, so search with python3.
  - Use literal `rmehta1987` in `qstat -u`.
- **Login node.** Never run GPU work on it, and watch the pid cap (`/sys/fs/cgroup/users/$USER/pids.current`, max 256).
- **Submitting.** 1-node `debug` may be submitted without asking. Ask before multi-node or `capacity`/`preemptable`:
  2-node debug job 7815096 was submitted because the user asked for it. One debug job per user at a time, including held ones.
- **Job completion.** The user runs `qstat` and tells you when a job is done; don't poll. A PBS `.OU` file appears when
  the job *starts*, so key on its end token, not its existence.
- **Queued jobs freeze files.** A queued or running job reads the worktree at start, so don't edit what it imports.
  7815096 imports:
  - `SQG/analysis/*.py`
  - `SQG/analysis/opt/**`
  - `SQG/fixed_ring/**`
  - `SQG/model_UViT.py`
- **Gate policy (user, 2026-11-04).** "don't be too much of a blocker if it does not match bitwise".
  - A tolerance written *before* the run decides; bitwise is reported as information only.
  - Never loosen a threshold after seeing results. Round 1b's new gates were set *before* its rerun, at the user's request.
- **Predictions before every job.** Write `PREDICTIONS_<round>.md` before submitting, and never edit it afterwards.

## 1. Goal and the real wall-clock

Make o1.2 online training faster without changing what it computes.
- **Today (user's numbers, H200, 7 GPUs, 195,250 optimizer steps):** online = **7 days**; offline (same steps, stored
  frames) = **21 h**.
- **Per optimizer step:** ≈ 3.1 s online vs ≈ 0.39 s offline. The solver is ≈ 87% (≈ 147 h of 168 h).
- **What one step does:** each GPU runs its own pyqg_jax AB3 solver (4096², float64, dt 5e-6) for 480 solver steps,
  yielding 12 fresh pairs. Separately, the UViT trains with DDP. Offline 21 h is the floor once the solver is hidden or
  replaced.
- **STRIDE:** keep it at 40. The user reported that STRIDE 400 overfits, so don't recommend thinning it (CHANGELOG 11-04).

## 2. Where things stand

**Done (all n = 2 or better, A100-40GB; details in notes):**

| job | id | what it found |
|---|---|---|
| A/B/C | 7809356/412, 7809385/433, 7809341/408 | Solver is 91% of the step and keeps the GPU 99% busy. A background solver thread gains only −1.4%. UViT batch 32 OOMs at 40 GB, so batch 16. Notes Part III–IV. |
| D | 7813332 | Per-phase sync audit. Only the `lax.switch` sync costs time, and nothing is left on the host side. Part V. |
| E | 7813816 | A float32 step is 3.61 ms vs 9.30 ms in float64 (2.575×). Part VI. |
| F | 7814044 | ncu on the AB3 fusion: 3.99 GB DRAM read for ≈1.0 GB unique, 52% excess sectors (uncoalesced; 4 c128 elements per thread). Not FP64-bound. No XLA unroll flag in jaxlib 0.4.28. Part VI. |
| lit | — | Fusing pointwise work into cuFFT callbacks (gyaradax) gives a ≈3.1–3.5 ms f64 / ≈1.5–1.8 ms f32 roofline. That route is not bitwise and needs statistical validation. Part VII. |
| round 1 | 7814300 | Six solver variants; **all FAILED their pre-set gates; the verdicts stand.** Results in `SQG/analysis/results/polaris_opt_round1_7814300/ranking_round1.txt`. |

**Round 1 ms per solver step (ref 9.29):**

| variant | seed | precision | ms/step | gate failure |
|---|---|---|---|---|
| `lean` | 1 | f64 | 8.00 | `t` counter only (fields bitwise) |
| `lean_u1` | 1 | f64 | **6.40** (1.45×) | `t` counter only (fields bitwise) |
| `lean_split` | 1 | f64 | 7.07 | `t` counter only (fields bitwise) |
| `lean64` | 2 | f64 | 7.07 | `t` counter only (fields ≤ 2.9e-14) |
| `mixed` | 2 | f32 nonlinear, f64 state | **4.68** (1.98×) | 2.30e-5 > 1e-5; `t` |
| `lean32` | 2 | f32 | **3.20** (2.90×) | 2.30e-5 > 1e-5; `t` |

Why they failed:
- **The `t` counter.** `t` is a float32 time counter the step never reads. The likely cause is XLA folding the
  unrolled `(t + dt) + dt`.
- **The float32 limit.** 1e-5 sits below float32's own floor: the *original* step run in float32 scores 2.48e-5 from
  the same state.

**H200 online projections** (these assume the A100 speed-ups carry over, which is not measured):
- `lean_u1` ≈ 5.1 days
- `mixed` ≈ 4.0 days
- `lean32` ≈ 3.0 days

**In flight: job 7815096** (`debug`, 2 nodes / 8 A100s, walltime 0:55, script
`SQG/analysis/opt/run_round1b_multisim_polaris.pbs`). Predictions are in `SQG/analysis/opt/PREDICTIONS_round1b.md`.
- **Phase 1, round 1b (node 0, GPU 0).** Reruns `lean64` / `mixed` / `lean32` (seed 2 only, as the user asked) under
  gates fixed before the rerun:
  - float32 variants: ≤ min(1e-4, 2 × the original step's own float32 error from the same state, same run);
  - `lean64`: ≤ 1e-12;
  - all: `tc` / `_ablevel` exact, |t − t_ref| ≤ 1e-6 relative. Exact `t` is reported as information.
  - `t` now advances by a run-time `dt` (`tables["tdt"]`).
- **Phase 2 (`SQG/analysis/opt/multisim/`).** Each rank sees one GPU via `rank_env.sh`.
  - `ncclprobe`: a 64 MB NCCL all_reduce with the aws-ofi-nccl v1.6.0 plugin under torch 2.3.1's bundled NCCL
    (2.20.x; this pairing is untried). PASS needs "Selected Provider is cxi" on every node.
  - `independent ref` / `independent lean32`: 8 trajectories, scaling vs 1 GPU.
  - `stream ref` / `stream lean32`: 7 simulators send o1.2-style pairs over gloo on hsn0 to 1 trainer GPU running
    o1.2's ConvNextUViT (batch 16, strict fp32) from an in-RAM FIFO. **Nothing is written to disk**, as in online
    training. Key output: `fresh_pairs_per_step_mean` vs o1.2's 12.

## 3. What to do next, in order

1. **When the user says 7815096 is done**, read its `.OU`. Check the final line `OPT_R1B_MS_DONE failures=N`, then:
   - **phase 1:** `SEED2_DONE`, `OPT_RANK_OK`, the `SEED2_F32BASE` / `SEED2_GATE_OK` / `SEED2_BENCH` / `SEED2_BANDS`
     lines, and `ranking_round1b.txt`;
   - **phase 2:** `NCCL_CXI_OK providers=N`, plus `MULTISIM_OK <mode> <solver>` × 5. Their JSON is
     `multisim_<mode>_<solver>.json`.

   On any failure, read the `ERROR …` line and the matching `multisim_*.log` first. Summaries land in
   `SQG/analysis/results/polaris_opt_round1b_ms_7815096/`; commit it with `git add -f`.
2. **Score `PREDICTIONS_round1b.md`** item by item (confirmed / refuted / half), without editing the file.
3. **Send combine_handoffs what they asked for** (a peer Claude session; `ListAgents`, then `SendMessage
   to:"combine_handoffs"`). Take it from `multisim_ncclprobe_ref.log` (full `NCCL_DEBUG=INFO`, subsystems INIT,NET)
   and the probe JSON:
   - (1) the exact `Loaded net plugin AWS Libfabric (vN)` and `Selected Provider is …` lines, with a per-node count,
     and the `GDR` value on the `Connected all rings` line;
   - (2) intra-node `via P2P/…` vs `via SHM/…` counts, and the `NCCL version` line;
   - (3) `busbw_GBps` (min over ranks; 10 timed rounds).

   Also say that this job had **no arm without `NCCL_PROTO=Simple`**. If the session is gone, put the answers in the
   CHANGELOG entry and tell the user.
4. **Write notes Part VIII** (round 1 + round 1b + the multisim smoke) in `polaris_sqg_notes.md`, in Parts V–VI's
   style: predictions, results table, scorecard, what it means. Update `CHANGELOG.md` (dated entry) and the memory
   file `sqg-polaris-profiling-state.md`.
5. **Ask the user** (don't decide) about the open items in §4, then plan the multi-node DDP job (§5).

## 4. Open decisions (the user's)

- **Seed 1 was not rerun.** `lean_u1` (bitwise fields, 1.45×) failed only on `t`, and its harness
  (`SQG/analysis/opt/seed1/`) still has round 1's exact-`t` check. Should it get round 1b's `t` rule and a rerun?
  It's the best bitwise-field candidate.
- **Which variant to carry into o1.2.** f64 (`lean_u1` / `lean64`, near-bitwise) or f32 (`mixed` / `lean32`, ≈2–3×).
  Precision in production is a science change and needs statistical validation (gyaradax-style) plus jesswan /
  Weidong's sign-off. Round 1b only proves the 1-step error is at float32's own level.
- **Round 2 targets** (notes Part VI–VII):
  - make the AB3 fusion one element per thread / coalesced. `lean_u1`'s cos(±0) trigger took `lean` from 8.00 to
    6.40 ms; it is unverified that this came from the per-thread layout, so check its HLO (`hlo_seed1_lean_u1.txt`)
    or run ncu;
  - drop the 2 history-rotation copies and the û/v̂ C2R input copies;
  - cuFFT LTO callbacks via the XLA FFI (the biggest lever, not bitwise).
- **How to scale out** (§5): replicate o1.2 across nodes, or decouple simulators from trainers.

## 5. Multi-node: the recipe and the design question

**Launch recipe from combine_handoffs** (their makani / ACE2 Polaris work; `run_round1b_multisim_polaris.pbs` encodes
it, so copy from there):
- **Launcher:** `mpiexec -n $NRANKS --ppn 4 --hostfile $PBS_NODEFILE --cpu-bind depth -d 8 --label --line-buffer`
  through a PMI→RANK shim (`multisim/rank_env.sh`). `--cpu-bind depth -d 8` is mandatory.
- **Rendezvous:** `MASTER_ADDR=$(head -1 $PBS_NODEFILE)`. `MASTER_PORT` must be unique per job *and* per run
  (`20000 + jobid % 20000 + i`); a fixed port hung 4 makani jobs.
- **Environment:**
  - `TMPDIR=/tmp` (AF_UNIX path limit);
  - set `OMP_NUM_THREADS` explicitly, since PBS exports 64;
  - a short process-group timeout.
- **NCCL over CXI:**
  - `LD_LIBRARY_PATH` = `/opt/cray/libfabric/2.3.1/lib64` + `/soft/libraries/aws-ofi-nccl/v1.6.0-libfabric-1.22.0/lib`
    + `/soft/libraries/hwloc/lib`;
  - `NCCL_NET="AWS Libfabric" NCCL_CROSS_NIC=1 NCCL_NET_GDR_LEVEL=PHB NCCL_PROTO=Simple`;
  - `FI_CXI_DISABLE_HOST_REGISTER=1 FI_MR_CACHE_MONITOR=userfaultfd FI_CXI_DEFAULT_CQ_SIZE=131072`;
  - the `FI_CXI_RDZV_*` block (`alt_read`, eager / threshold / get_min 0), `FI_CXI_DEFAULT_TX_SIZE=2048`,
    `FI_CXI_RX_MATCH_MODE=hybrid`.
  - Their 2-node CXI all_reduce measured 18 GB/s.
- **Their open flag:** `OMP_NUM_THREADS=8` shares the 8 bound cores with aws-ofi-nccl's progress thread. It's
  untested across nodes; at 1 node, 64 vs 1 made no difference. Use 1, or A/B 1 vs 8, in the DDP job.
- **GPU order:** in makani, reversed order (`GPU_ORDER=reverse`, NUMA-local) was 0.88% *slower* for 1-node DDP and 7%
  *faster* for 4-node h2w2. Keep the default unless an SQG A/B says otherwise.

**The design question for SQG.** 7815096's `stream` arms are its first data point.
- **(a) Replicate o1.2.** Every GPU runs its own solver + a DDP UViT rank, and the all_reduce goes over CXI. This is
  the same computation as today, just with more ranks. Per-step time stays solver-bound (≈ 3.1 s on the H200), and the
  global batch / pair count grows with ranks, which changes the training recipe unless batch-per-rank is reduced.
- **(b) Decouple.** S simulator GPUs stream pairs to T trainer GPUs (DDP among trainers). This hides the solver, so
  the 21 h floor is reachable if S/T is large enough: ≈ 2.4–7 simulators per trainer depending on the solver variant.
  It changes freshness and the replay ratio unless matched to 12 fresh pairs per step. That's a science call.

**Before planning:** read 7815096's `fresh_pairs_per_step_mean`, `train_s_per_step_median` and `send_ms_*`, then
propose a 2-node DDP debug job (prereg first; ask before submitting).

## 6. File map (this branch)

| path | what |
|---|---|
| `SQG/polaris/polaris_sqg_notes.md` | The write-up: Parts I–VII; Part VIII is next. |
| `SQG/polaris/PROFILING_PROMPT.md` | The original task brief (jobs A/B/C). |
| `SQG/analysis/profile_{solver,solver_fp32,overlap,uvit,ab3_ncu}.py` + `*_polaris.pbs` | Jobs A–F harnesses. `summarize_nsys_phases.py` = the nsys SQLite → per-phase summary. |
| `SQG/analysis/opt/seed1/` | Seed 1's f64 `lean` / `lean_u1` / `lean_split` (`lean_solver.py`, `run_seed1.py`, `test_lean_ring.py`). |
| `SQG/analysis/opt/seed2/` | Seed 2's `lean64` / `mixed` / `lean32` (`seed2_opt.py`; the ROUND 1b block holds the new gates). |
| `SQG/analysis/opt/rank_round1.py` (+ test) | Ranking: tolerance-gated; bitwise is information. |
| `SQG/analysis/opt/PREDICTIONS_round1{,b}.md` | Pre-registered predictions. |
| `SQG/analysis/opt/run_round1_polaris.pbs`, `run_round1b_multisim_polaris.pbs` | The two optimization jobs. |
| `SQG/analysis/opt/multisim/{sqg_multisim.py,rank_env.sh}` | 8-GPU smoke: `--mode ncclprobe\|independent\|stream`, `--solver ref\|lean32`. |
| `SQG/analysis/results/polaris_*_<jobid>/` | Committed text summaries per job. |
