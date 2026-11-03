# Prompt — profile SQG online training on Polaris (`debug` queue)

> Paste everything below the line into a fresh Claude Code session started in a
> clone of this repo (branch `sqg`) **on a Polaris login node**. It is
> self-contained: it needs no other branch, no Midway access, no training data,
> and no checkpoint. Its PBS style follows the makani-SFNO Polaris line
> (`origin/worktree-makani-multinode-ddp-profiling`). Written 2026-11-02.

---

You are profiling the **code** of the SQG online-training loop: the JAX A-SQG solver,
the PyTorch ConvNextUViT, and how the two are sequenced. You run on **Polaris (ALCF), 1
node, `debug` queue, one GPU per job**. This is a code-performance measurement only:

- **Not a hardware study.** Do not compare GPUs and do not report cross-GPU ratios. Every
  finding must be a statement about the code that would hold on any GPU (e.g. "`ref`
  blocks the Python thread for the whole 40-step call; `fixed` does not").
- **Not a data-loading study.** No job reads training data. A and B start from a
  random-IC solver state (§2b), and C uses random input. The one in-memory data step that
  is part of the loop (B's sample gather + H2D copy) is timed separately and kept out of
  the compute numbers.
- **Do not change what the model or solver computes** (CLAUDE.md rules 1, 6, 11). Main
  session is Opus at xhigh; subagents use Fable 5.

## 0. Read first

1. `CLAUDE.md`: rules, and the Polaris column of the cluster-facts table.
2. `CHANGELOG.md`: the "SQG online-training profiling (jobs A/B/C)" entry.
3. `SQG/analysis/profile_solver.py`, `profile_overlap.py`, `profile_uvit.py`,
   `summarize_nsys.py`, and their `*_midway.sh` launchers. **These are what you port.**
   Each file's docstring says what it measures and what PASS means.
4. *Optional background, not required:* `git fetch origin
   worktree-makani-multinode-ddp-profiling:refs/remotes/origin/worktree-makani-multinode-ddp-profiling`,
   then `git show <that ref>:polaris_pbs_notes.md` (queue table, error decoder, NUMA map)
   and `:makani_sfno/polaris/polaris_sfno_smoke.pbs` (PBS style). Every fact this task
   needs is already restated below, so do not block on this fetch.

## 1. Jobs, the question each one answers, and its PASS token

| job | harness | code question | PASS token |
|---|---|---|---|
| **A** | `profile_solver.py` (`--mode bench`; then `--mode nsys` for `ref` and `fixed`) | Does a jitted 40-step solver call block the Python thread? HLO op counts (fft / conditional / while / copy / fusion); `donate` and `fixed`-AB3-level variants bitwise == `ref` | `PROFILE_OK bench` (+ `NSYS_SUMMARY_OK` ×2 if nsys exists) |
| **B** | `profile_overlap.py` | Does the solver overlap the UViT step? (serial vs background thread; pyqg AB3 vs fixed level) | `OVERLAP_OK` |
| **C** | `profile_uvit.py` | torch.profiler breakdown of one UViT training step (random input) | `UVIT_OK` |

Hypothesis under test: pyqg_jax `AB3Stepper.apply_updates` does a runtime `lax.switch`
on `_ablevel` (steppers.py:478). This may sync the host every step and serialise solver
and network. Production's per-step time is roughly the *sum* of the two, not the max
(o1.2 log: ~2.6 s ≈ 2.2 + 0.4).

Deliverables, on branch `sqg-polaris-profiling` off `sqg`:
- `SQG/analysis/profile_{solver,overlap,uvit}_polaris.pbs`, named after each `_midway.sh`
  (rule #7). **Do not edit the `_midway.sh` files.**
- The env-var edits to the `.py` harnesses in §3. **With every new env var unset, the
  Midway behaviour must be byte-for-byte the same code path.**
- JSON/CSV summaries in `SQG/analysis/results/polaris_<job>_<jobid>/`. Heavy files
  (`.nsys-rep`, `.sqlite`) stay in `$MEMBER_ROOT/sqg_profile/`; `.gitignore` blocks them.
- `SQG/polaris/polaris_sqg_notes.md`, in the style of `si/bench_midway_notes.md`, plus
  a dated CHANGELOG entry.

## 2. One-time setup (login node; compute nodes have NO network)

**2a. Confirm the inputs are in your clone.** Run `ls SQG/fixed_ring SQG/polaris`. You
need `SQG/fixed_ring/{A-SQG_data.py, sqg.py, SHA256SUMS}` and
`SQG/polaris/{requirements-sqg.txt, requirements-wdenv.txt}`. If they are missing, run
`git pull`. If they are still missing, stop with `needs input: push SQG/fixed_ring and
SQG/polaris to origin/sqg`. Then run `(cd SQG/fixed_ring && sha256sum -c SHA256SUMS)`
and expect two `OK`s.

These are byte-identical copies of Weidong's Midway `fixed_ring/` solver. It is NOT the
older `SQG/A-SQG_data.py`. **Never edit these two files** (§3.1).

**2b. Solver start state: random IC, no checkpoint.** On Midway, A and B load an
archived AB3 checkpoint. For code profiling that is unnecessary:
- `DT` is fixed, so every step is the same set of 4096² float64 FFTs whatever the field
  values.
- The CFL checks are host-side diagnostics in the data-generation loop, not in the
  jitted step.
- The only state that changes what code runs is the AB3 level (0 → 1 → 2 over the first
  two steps).

So add `SQG_INIT=random|checkpoint` to `profile_solver.py` and `profile_overlap.py`.
**When unset, the default is `checkpoint`**, i.e. Midway behaviour; the Polaris PBS
scripts set `random`. Under `random`, replace only the `load_checkpoint` line with
`st0 = GD.advance(template, 2)`, where `template` is the
`GD.sqg.init(GD.p, jax.random.key(GD.SEED_INITIAL_CONDITION))` the harness already
builds (the production spin-up IC). Keep the existing `assert int(st0._ablevel) == 2`.
In the JSON, write `"init": "random"`, `"checkpoint": null`, and `"completed_blocks": null`
(`profile_solver.py:160`).

The bitwise checks stay valid because all variants start from the same `st0`. B's fill
loop (`while online.n_pairs < 2*BATCH`) runs the solver itself, so it needs nothing
else.

**2c. Build the env** (a pip copy of Midway's `wdenv`; the CUDA libraries come from pip
wheels, so **no Cray CUDA or conda modules are needed at run time**).
```bash
export MEMBER_ROOT=/eagle/lighthouse-uchicago/members/$USER      # must be writable
export PIP_CACHE_DIR=$MEMBER_ROOT/pip-cache                       # NOT $HOME (small quota; ~5 GB of wheels)
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
unset PYTHONPATH; export PYTHONNOUSERSITE=1                       # no user-site / topups leaking in
```
Get an interpreter. Try these in order and record which one worked:
1. `module use /soft/modulefiles && module load conda` (this was broken in 2026-08; it
   may be fixed now). If it fails, `source
   /soft/applications/conda/2025-09-25/mconda3/etc/profile.d/conda.sh`. Then run
   `conda create -y -p $MEMBER_ROOT/conda-envs/sqg-env python=3.10`.
2. If `conda create` fails, use the base conda's own Python:
   `python3 -m venv $MEMBER_ROOT/conda-envs/sqg-env`. That is Python 3.12; every pin has a
   cp312 wheel.
3. If there is no conda at all, use any `python3` ≥ 3.10 on the login node
   (`ls /usr/bin/python3*`), with `-m venv`.

**Never use `--system-site-packages`.** Then:
```bash
$MEMBER_ROOT/conda-envs/sqg-env/bin/python -m pip install -U pip
$MEMBER_ROOT/conda-envs/sqg-env/bin/python -m pip install \
    -r SQG/polaris/requirements-sqg.txt -c SQG/polaris/requirements-wdenv.txt
```
`requirements-sqg.txt` lists the top-level imports: jax[cuda12] 0.4.28, pyqg-jax 0.8.1,
torch 2.3.1, timm 0.9.12, netCDF4, h5py, numpy 1.26.4, nvcc wheel. The wdenv freeze is
passed as *constraints*, so transitive versions match Midway. **This exact command was
dry-run against PyPI on 2026-11-02 (cp310) and resolved cleanly**, including cuDNN
8.9.2.26, cuFFT 11.0.2.54, and nvcc 12.9.86.

If resolution fails on 3.12, drop `-c` and keep `-r`; the top-level pins are the ones
that matter. **Never change a top-level pin**: the hypothesis is about this jax/pyqg_jax
version.

Login-node check (import only, rule #3):
`.../bin/python -c "import jax, pyqg_jax, torch, timm, netCDF4, h5py; print(jax.__version__, torch.__version__)"`.
The first `debug` job is a ≤5-min env probe. It prints `jax.devices()`,
`torch.cuda.get_device_name(0)`, and the result of a 4096² complex128 `jnp.fft.fft2`, then
`SQG_ENV_OK`.

## 3. Harness edits and traps (each one would block a job or silently invalidate a number)

1. **Solver import side effects.** `A-SQG_data.py` hard-codes
   `TRAINING_ROOT = Path("/project/pedramh/...")` (line 76) and `mkdir`s run dirs under it
   at import, which dies on Polaris. It also hashes its own bytes into `CONFIG`, so
   **do not edit the file**. Instead, in `import_asqg()` of both solver harnesses:
   - Make `ASQG_PY` overridable by `SQG_ASQG_PY`.
   - If `SQG_TRAINING_ROOT` is set, read the source, replace **only** the
     `TRAINING_ROOT = …` line in memory with `TRAINING_ROOT = Path(r"<value>")`, then
     `exec(compile(src, ASQG_PY, "exec"), mod.__dict__)` with `mod.__file__ = ASQG_PY`
     (the unmodified file). Its self-hash stays correct.
   - Keep the existing argv (`--member 1`), stdout silencing, and signal-handler restore.

   **The solver cannot be imported on a CPU.** It raises `JAX is not using the allocated
   GPU` at import (A-SQG_data.py:272), so do not try a login-node import test. Two
   tests instead:
   - (a) Login node, pure string test: the substitution changes exactly one line, and
     that line is the `TRAINING_ROOT` line (it occurs once, at line 76).
   - (b) Inside the `SQG_ENV_OK` probe job (GPU): the patched import succeeds,
     `GD.TRAINING_ROOT` is the override, `GD.CONFIG["solver_sha256"] ==
     sha256(SQG/fixed_ring/A-SQG_data.py)`, and `GD.advance(template, 2)._ablevel == 2`.
     Print `SQG_IMPORT_OK`.
2. **cuFFT/cuDNN come from the wheels.** In the PBS scripts, do not load CUDA modules and
   do not put `/soft/compilers/cudatoolkit/*/lib64` on `LD_LIBRARY_PATH`. The harness
   records which libraries were actually loaded (`profile_solver_bench.json`). Copy that
   line into the notes.
3. **nsys is optional; it does not block the run.** Find it with `NSYS=$(command -v nsys
   || ls -1 /soft/compilers/cudatoolkit/cuda-*/bin/nsys 2>/dev/null | sort -V | tail -1)`.
   (On 2026-08 it was `/soft/compilers/cudatoolkit/cuda-12.9.1/bin/nsys`; there is no
   nsight module.) If none is found, print `WARN NSYS_NOT_FOUND` and run A's bench phase
   only. The core questions (dispatch vs completion, HLO `conditional`, bitwise) come
   from bench mode. Use the nsys flags from `_midway.sh` verbatim. nsys writes a report
   even when it captured nothing, so require `NSYS_SUMMARY_OK` and a non-empty kernel
   table.
4. **Batch size is a knob.** Add `SQG_BATCH` (B, C) and `SQG_BUF` (B) overrides; unset
   keeps 32 / 8192. Polaris A100s have 40 GB, and B holds the solver state, the GPU ring,
   and the UViT at once. **If anything OOMs, halve the batch and resubmit (32 → 16 →
   8). Do not stop.** Use the same batch for B and C, and state it next to every ML
   number. The code questions do not depend on it. Keep
   `XLA_PYTHON_CLIENT_PREALLOCATE=false` and
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (JAX and torch share the GPU, as in
   production).
5. **Keep the data path separate.** In B, wrap `OnlinePairs.sample()` in a timer + NVTX
   range and add it as a new JSON field. This is additive, so existing keys keep their
   names (rule #10). Headline ML time = the CUDA-event network time the harness already
   measures on torch's stream.
6. **One harness per job, on GPU 0, unbound.** Running A/B/C concurrently on GPUs 0–3
   would share host CPU and contaminate B. *Optional:* one extra B arm with `numactl
   --cpunodebind=3 --membind=3` (the GPU↔NUMA map is reversed: dev0 → NUMA 3). Skip it if
   `numactl` is absent. Label it and keep it in a separate row.
7. **Queue limits.** `debug`: ≤1 h, 1–2 nodes, **1 running + 1 queued per user**, and
   `-W depend` is rejected. So submit the next job only after `qstat -u $USER` is empty.
   If `debug` rejects you or is unavailable (`qstat -Q`), use `debug-scaling` (≤1 h)
   with the same script and `-q debug-scaling`. Add `-v PHASE=bench|nsys|all` to A
   (default `all`), and split it into two jobs if `all` approaches the walltime.

## 4. PBS shape (self-contained; no other repo script is sourced)

```bash
#!/bin/bash -l
#PBS -N sqg_prof_solver            # _overlap / _uvit
#PBS -A lighthouse-uchicago
#PBS -q debug
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle     # REQUIRED: jobs are rejected without it
#PBS -l walltime=00:55:00          # A 00:55:00; B 00:55:00; C 00:30:00
#PBS -j oe
cd "${PBS_O_WORKDIR}"                                   # qsub from the repo root
MEMBER_ROOT="${MEMBER_ROOT:-/eagle/lighthouse-uchicago/members/${USER}}"
[ -w "$MEMBER_ROOT" ] || { echo "ERROR SQG_NO_MEMBER_ROOT: $MEMBER_ROOT"; exit 2; }
PY="${SQG_PY:-${MEMBER_ROOT}/conda-envs/sqg-env/bin/python}"
[ -x "$PY" ] || { echo "ERROR SQG_ENV_MISSING: $PY"; exit 3; }
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false HDF5_USE_FILE_LOCKING=FALSE \
       OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
       TMPDIR="${MEMBER_ROOT}/tmp" WANDB_MODE=offline
export SQG_ASQG_PY="${PBS_O_WORKDIR}/SQG/fixed_ring/A-SQG_data.py"
export SQG_TRAINING_ROOT="${SQG_TRAINING_ROOT:-${MEMBER_ROOT}/sqg_stage/AllTrainData}"   # import mkdirs here
export SQG_INIT="${SQG_INIT:-random}"
JOB="${PBS_JOBID%%.*}"
OUT="${MEMBER_ROOT}/sqg_profile/profile_solver_${JOB}"; mkdir -p "$OUT" "$TMPDIR" "$SQG_TRAINING_ROOT"
(cd SQG/fixed_ring && sha256sum -c --quiet SHA256SUMS) || { echo "ERROR SQG_SOLVER_SHA_MISMATCH"; exit 4; }
echo "host=$(hostname) job=$JOB py=$PY init=$SQG_INIT batch=${SQG_BATCH:-32} buf=${SQG_BUF:-8192}"
# + nvidia-smi query / dmon (same flags as the _midway.sh), the NSYS lookup (§3.3), the harness
#   calls with the SAME arguments as the _midway.sh (--reps 5; --steps 20 --warmup 4), then copy
#   *.json/*.csv/*.txt summaries into SQG/analysis/results/polaris_<job>_${JOB}/
```

## 5. Order of work (one gated step at a time; commit after each)

1. §2 setup (env build, login-node import check).
2. Harness edits (§2b, §3.1, §3.4, §3.5) + the login-node string test (§3.1a). Check with
   `git diff` that every new branch is gated behind an env var that defaults to Midway's
   behaviour.
3. Probe job (~5 min, `debug`) → `SQG_ENV_OK` + `SQG_IMPORT_OK` (§2c, §3.1b).
4. **C** (~30 min) → `UVIT_OK`.
5. **A** → `PROFILE_OK bench` (+ nsys). The bitwise checks (`donate`/`fixed`/`fixed_donate`
   == `ref`) must hold. A mismatch is a **finding**: record it, keep it out of any speed
   claim, finish B and C, and trace it after. Do not loosen the check.
6. **B** → `OVERLAP_OK` (+ optional NUMA arm).
7. If the queue allows, run a second rep of A-bench and B, interleaved rather than
   back-to-back, so headline numbers have n ≥ 2.
8. Notes + CHANGELOG, push (`git -c pack.threads=1 push`: the ALCF login node caps
   processes), open a **draft PR against `sqg`**. Do not merge, and do not push to main.

## 6. If something fails: what to do (do not stop unless the row says so)

| failure | action |
|---|---|
| `SQG/fixed_ring` or `SQG/polaris` missing after `git pull` | **stop** — `needs input:` push them to `origin/sqg` |
| `module load conda` / `conda create` fails | next interpreter option in §2c |
| pip resolution fails | drop `-c` constraints and keep the top-level pins; record it |
| `ERROR SQG_NO_MEMBER_ROOT` | `export MEMBER_ROOT=<a writable /eagle dir>` (and pass it with `qsub -v MEMBER_ROOT=…`) |
| solver import: `FileNotFoundError` / `PermissionError` on mkdir | `SQG_TRAINING_ROOT` override not applied, so fix §3.1 |
| `JAX is not using the allocated GPU` | JAX found no GPU: you imported on a login node, or the cuda12 plugin is not installed (check `jax.devices()` in the probe; reinstall from §2c) |
| `Checkpoint ... does not match` | you are in `SQG_INIT=checkpoint` mode by mistake; the Polaris default is `random` |
| CUDA OOM | halve `SQG_BATCH` (then `SQG_BUF`) and resubmit; record the value |
| nsys absent or capture empty | bench-only for A; note it |
| `numactl` absent | skip the optional NUMA arm |
| `qsub` rejected (`max_queued`) | wait until `qstat -u $USER` is empty |
| `debug` unavailable or long wait | `-q debug-scaling` |
| walltime kill on A | `-v PHASE=bench` and `-v PHASE=nsys` as two jobs |
| bitwise mismatch | finding: report, continue other jobs, trace after |
| no GPU allocation possible at all | **stop** — `needs input:` with `qstat -Q` output |

## 7. Pre-registered predictions (write them in the notes BEFORE job A; score after)

Each one is about the code. Score each as confirmed or refuted, citing the number.

1. **Host sync (A):** for `ref`, the dispatch time of a jitted 40-step call ≈ its
   completion time (Python is blocked), and the optimized HLO has a `conditional` inside
   the `while` body. For `fixed`, dispatch ≪ completion and there is no `conditional`. If
   `ref` does not block, the `lax.switch` hypothesis is refuted. Say so and look for the
   sync elsewhere (D2H copies, `cudaStreamSynchronize` in the nsys CUDA-API rows).
2. **Bitwise (A):** `donate`, `fixed`, and `fixed_donate` are bitwise == `ref` after 480 steps.
3. **Copies (A):** `donate` removes one ~400 MB device-to-device copy of the AB3 carry per
   call (HLO `copy` count, nsys memcpy rows), and ms/step changes by < 5%.
4. **Sequencing (B):** `serial_ref` ≈ `solver_only_ref` + `ml_only`, and `thread_fixed` ≈
   max(solver, ML). Overlap efficiency for `ref` is < 20%. If `thread_ref` already
   overlaps well, the blocking is in o1.2's call order, not in pyqg_jax.
5. **Data path is not the cost (B):** the timed `sample()` gather+H2D is < 10% of `ml_only`.
6. **UViT (C):** a few conv/attention kernels dominate, with no single op > 40% of CUDA
   time. Under strict fp32 (TF32 off, as in production), `cudnn.benchmark` /
   `channels_last` change step time by < 15%.

## 8. Reporting (≤10 lines per job in chat; full tables in the notes)

Per job: host, job id, interpreter path used, jax/torch/cuFFT actually loaded, batch,
the PASS-token line, and the headline numbers with n and spread. Lead with **code
findings**: where Python blocks, which copies or syncs exist, and whether solver and ML
overlap. Absolute ms come second. Keep `sample()` time on its own line. No cross-GPU
ratios. End with a ranked list of code changes the evidence supports, each tagged with
the equivalence check it would need (DESIGN §4). **Do not implement them in this task.**
