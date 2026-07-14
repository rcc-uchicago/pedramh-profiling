# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository. What &
why lives in **DESIGN.md** — read it first. This file is *how to work here*.

## Model: use Fable 5

**Run this project's Claude Code sessions and subagents on Fable 5**
(`claude-fable-5`). Set it with `/model claude-fable-5` at the start of a session
(or your Claude Code config). If you spawn subagents/agent teams, keep them on
Fable 5 unless a task explicitly needs a different tier.

## Why we're here

Make the three S2S-family weather models (**S2S**, **S2S-Lightning**, **SI**)
**faster on HPC GPUs without changing what they compute.** The work is: bring
them up on each cluster, capture correctness baselines, then optimize the hot
path one gated step at a time. Scope is **performance + cluster bring-up**, NOT
retraining, NOT forecast repro, NOT science changes. See DESIGN.md §1.

## Things NOT to do (read before you touch anything)

These are the ways to silently break the project. Do not do them.

1. **Never change model outputs to make a benchmark faster.** Every hot-path
   change is gated on numerical equivalence vs a captured baseline (DESIGN.md §4).
   If a "faster" version's loss/output drifts beyond tolerance, it's a **bug** —
   find the cause; do **not** loosen the tolerance to pass.
2. **Never `find /`, `find /eagle`, `find /project`, or scan outside the repo.**
   HPC filesystems have millions of files and it will hang. Search with `grep`/
   `Grep` inside `.` only.
3. **Never run training/inference directly** — always through the scheduler wrap
   (`sbatch` on Midway, `qsub` on Polaris). Never on a login node.
4. **Never invert the `train.py` vs `train_optimized.py` attribution.** In
   `s2s/v2.0/`, `train.py`/`inference.py` are the bench-instrumented, actively
   maintained files; the `_optimized` ones are older despite the name.
5. **Never edit shared `s2s/v2.0/` code to satisfy one harness.** It's imported by
   S2S *and* the Lightning port — changes must serve both; re-run both smokes.
6. **Never commit an optimization without (a) a passing smoke and (b) an
   equivalence check** against the baseline. No exceptions.
7. **Never break the Midway (SLURM) path when adding Polaris (PBS) scripts.**
   Add `polaris_*` files beside the `midway_*` ones; don't edit in place.
8. **Never commit secrets or big binaries.** NGC key → `$NGC_API_KEY` only (never
   hardcoded). No `*.h5/*.nc/*.pt/*.ckpt/*.npy/*.nsys-rep/*.sqlite` (`.gitignore`
   blocks them; the `s2s-lightning/data/constant_mask/*.npy` carve-out is the only
   allowed exception).
9. **Never push to `main` directly** — it's branch-protected. Branch → PR.
10. **Never let a benchmark's instrumentation drift.** Dropping an NVTX range or a
    CSV column silently invalidates every comparison. Treat `*_BENCH`/NVTX/CSV
    plumbing as part of the contract.
11. **Never add fudge factors or `--skip`/`xfail` a failing correctness test** to
    get green. A wrong number means a wrong term — trace it.

## Orientation (do this when you start a session)

1. Read **CHANGELOG.md** — what's done, what's in progress, what's blocked, and
   the failed approaches not to re-try.
2. Confirm your model is Fable 5 (`/model`), and which cluster you're on
   (`hostname`, `sinfo`/`qstat -Q`).
3. Run the fast checks (once the harness exists): `pytest -q --fast 2>&1 | tail -20`.
4. Pick the next unchecked roadmap item / failing test.
5. **Before you stop, update CHANGELOG.md** with what you did, the measured
   result, and anything you learned or that's now blocked.

## Development principles (small commits, tests pass, living doc)

- **Small, testable commits.** One logical change per commit — one script ported,
  one optimization rung, one bugfix. A refactor is its own commit, separate from
  features.
- **Every commit passes the tests it can run.** Run `pytest -q --fast` (and the
  relevant smoke) before committing. If something regresses, fix it before
  committing — never "fix it later". If a change legitimately alters expected
  behavior, update the test explicitly (don't delete/skip it).
- **Every change ships its test.** New optimization → its equivalence check. New
  bugfix → a test that reproduces the bug first. New cluster script → the smoke
  that proves it.
- **The living document is not optional.** `CHANGELOG.md` (cross-cutting) and the
  per-cluster notes (`polaris_pbs_notes.md`, etc.) are the shared memory across
  sessions. Record: what worked, what didn't (and why — so it's not re-attempted),
  measured speedups, and every cluster fact you confirmed. Style: the narrative +
  dated "Decisions / changes log" of `midway_notes.md`.
- **Read the `.err`/stderr first when a job fails.** Most bring-up failures are
  path/module/OOM and are visible immediately. (On Midway, a whole class of
  "missing kernels" was really an early crash before any GPU work — read the error.)
- **Concise output.** Tests print ≤10 lines on success, ~20 on failure; report max
  relative error + where it occurs, not raw tensors; keep `ERROR <reason>`
  greppable on one line; log verbose diagnostics to files.
- **Never claim a step passed without reading the actual output.** Smoke scripts
  print a success token / write a CSV row — key on that, not on exit code alone.

## Cluster facts

| Item | Midway (RCC) | Polaris (ALCF) |
|---|---|---|
| Scheduler | SLURM — `sbatch`/`squeue`/`scancel` | PBS Pro — `qsub`/`qstat`/`qdel` |
| Account / queue | `--account=pi-pedramh`, `-p pedramh-gpu` | `-A <project>`, `-q debug` (smoke) |
| GPU | H100 NVL ~94 GB (Intel Ice Lake, PCIe Gen4) | **4× A100 40 GB SXM4** (AMD Milan) |
| Node/GPU directive | `--nodes=1 --gres=gpu:4` | `-l select=1:system=polaris -l place=scatter` |
| Filesystems | implicit | `-l filesystems=home:eagle` (**rejected without it**) |
| Env | `module load python/miniforge-25.3.0 && eval "$(mamba shell hook --shell bash)" && mamba activate <env>` | `module use /soft/modulefiles && module load conda && conda activate <env>` |
| Data (ERA5 HDF5) | `/project/pedramh/h5data/h5data` | Globus-stage to `/eagle/<project>/…` |
| Job id in script | `$SLURM_JOB_ID` | `$PBS_JOBID` (use `${PBS_JOBID%%.*}`) |

Configs are **cluster-specific**: fix `data_dir`, `checkpoint_path`, and the
mean/std `.nc` filenames in the YAML before launching (they fail deep in the data
loader, not early). Use `WANDB_MODE=offline`.

## Common commands

```bash
# --- S2S (canonical, torchrun) ---
cd s2s
PYTHONPATH=$(pwd)/v2.0 torchrun --standalone --nproc_per_node=4 \
    v2.0/train.py --yaml_config=v2.0/config/exp2.yaml --run_num=0100

# --- S2S-Lightning (imports ../s2s/v2.0) ---
cd s2s-lightning
PYTHONPATH=../s2s/v2.0:$(pwd) python smoke_train_module.py    # prints SMOKE_OK

# --- SI ---
cd si
python bench.py --yaml_config configs/SI_midway.yaml --config SI --devices 0

# Bench env knobs (S2S): S2S_BENCH=1 S2S_BENCH_WARMUP S2S_BENCH_STEPS S2S_BENCH_CSV S2S_NVTX
# Bench env knobs (SI):  SI_BENCH_*  SI_NVTX  SI_PRECISION
```

Submit real work through the cluster scripts (`s2s/v2.0/HPC_scripts/midway_*.sh`,
`s2s-lightning/midway_*.sh`, `si/bench_midway.sh`; Polaris `*_polaris.pbs` per the
handoff), never directly.

## Repo architecture

See **DESIGN.md §2–3** for the model pipeline and how the three relate. Short
version: `s2s/v2.0/` holds the shared model (`networks/pangu.py::PanguModel_Plasim`),
losses (`utils/losses.py`), and HDF5 loaders (`utils/data_loader_multifiles.py`);
`s2s-lightning/` imports that code and wraps it in Lightning; `si/` is the separate
DiT/SiT model (with its own `si/CLAUDE.md` for SI-specific bench details).
`PYTHONPATH` must include `s2s/v2.0` for any `from utils…`/`from networks…` import.
