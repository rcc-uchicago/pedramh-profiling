# CLAUDE.md

Guidance for Claude Code working in this repository. What &
why lives in **DESIGN.md** — read it first. This file is *how to work here*, and
is the **single source of truth for cluster facts** (§Cluster facts below).

## Model policy

**Main session: Opus 4.8 at xhigh reasoning effort** (confirm with `/model`).
**Subagents / agent teams: Fable 5** (`claude-fable-5`) — set it on every agent you
spawn unless a task explicitly needs a different tier.

## Why we're here

Get the six weather codebases here (**PanguWeather** — the focus — **makani**,
**PhysicsNeMo**, **S2S**, **S2S-Lightning**, **SI**) running on HPC GPUs, then
make them **faster without changing what they compute.** Two phases, in order
(DESIGN.md §1):

1. **Bring-up** — environment, staged/prepared data, scheduler scripts, then
   **training runs that produce evaluatable models**, and **inference**.
2. **Profile, then optimize** — measure first; every hot-path change gated on
   numerical equivalence vs a captured baseline (DESIGN.md §4).

Phase is **per model, per cluster**: s2s is deep into Phase 2 on Midway while
PanguWeather is mid-Phase-1 on Polaris. Check CHANGELOG.md for where each is.

**The division of labor — this is the line that matters:** bring-up and training
are **ours**; the **science is jesswan's** — variable sets, fill values, channel
roles, loss definitions, the physics. Training to get an evaluatable model is our
work; *changing what a model computes* needs jesswan's sign-off. In the hot path
it stays mechanical: an "optimization" that moves outputs beyond tolerance is a
**bug**, not a result.

*(Scope widened 2026-07-16, owner's decision: training moved from out-of-scope to
Phase 1. This file previously said "NOT retraining".)*

## Things NOT to do (read before you touch anything)

These are the ways to silently break the project. Do not do them.

1. **Never change model outputs to make a benchmark faster.** Every hot-path
   change is gated on numerical equivalence vs a captured baseline (DESIGN.md §4).
   If a "faster" version's loss/output drifts beyond tolerance, it's a **bug** —
   find the cause; do **not** loosen the tolerance to pass.
2. **Never `find /`, `find /eagle`, `find /project`, or scan outside the repo.**
   HPC filesystems have millions of files and it will hang. Search with `grep`/
   `Grep` inside `.` only.
3. **Never run training/inference on a login node, and never bypass the
   scheduler for real work.** Submit real jobs with `sbatch` (Midway) / `qsub`
   (Polaris). The bare commands in §Common commands are for **interactive
   compute-node allocations only** (see the preface there).
4. **Never invert the `train.py` vs `train_optimized.py` attribution.** In
   `s2s/v2.0/`, `train.py`/`inference.py` are the bench-instrumented, actively
   maintained files; the `_optimized` ones are older despite the name. The
   evidence, since the name argues the opposite: `train_optimized.py:424` still
   wraps DDP with `find_unused_parameters=True`, while `train.py:450-451` uses
   `find_unused_parameters=False, static_graph=True` — the newer, faster path.
5. **Never edit live-coupled code to satisfy one consumer.** Two pairs are coupled
   (§Repo architecture): (a) `s2s/v2.0/` is imported by S2S *and* the Lightning
   port — changes must serve both; re-run both smokes. (b) `physicsnemo_sfno/` is
   editable-installed into the shared SFNO venv and **makani imports it** — a
   change there must serve makani too; re-run both SFNO smokes.
   **Scope: those two pairs.** This rule does **not** reach `PanguWeather/` or
   `si/`, which are copies — a same-named file there (e.g. `utils/metrics.py`,
   which `s2s/v2.0/` does not even have) has a blast radius of exactly its own
   project. Check the table before assuming either way.
6. **Never commit an optimization without (a) a passing smoke and (b) an
   equivalence check** against the baseline. No exceptions.
7. **Never break the Midway (SLURM) path when adding Polaris (PBS) scripts.**
   Add a Polaris script beside each Midway one, mirroring its name with
   `midway`→`polaris` (e.g. `midway_training.sh` → `polaris_training.pbs`); don't
   edit the Midway script in place.
8. **Never commit secrets or big binaries.** NGC key → `$NGC_API_KEY` only (never
   hardcoded). No `*.h5/*.nc/*.pt/*.ckpt/*.npy/*.nsys-rep/*.sqlite` (`.gitignore`
   blocks them; the `s2s-lightning/data/constant_mask/*.npy` carve-out is the only
   allowed exception). Baselines are committed as JSON/CSV summaries only (DESIGN §4.2).
9. **Never push to `main` directly** — it's branch-protected (PR + 1 approving
   review). Branch → PR. A solo session cannot self-approve; leave the PR open for
   the maintainer to review/merge and note it in CHANGELOG (don't try `--admin`).
10. **Never let a benchmark's instrumentation drift.** Dropping/renaming an NVTX
    range or a CSV column silently invalidates every comparison (and breaks
    `parse_nsys.py`). S2S and SI use *different* range names — don't cross them.
11. **Never add fudge factors or `--skip`/`xfail` a failing correctness test** to
    get green. A wrong number means a wrong term — trace it.
12. **Never launch `test.yaml` bare.** Despite the name it is the full ~79M-param
    model (OOMed a 93 GiB H100 at its defaults); the smokes fit it only via a
    `batch_size=1` override + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Orientation (do this when you start a session)

1. Read **CHANGELOG.md** — what's done, what's in progress, what's blocked, and
   the failed approaches not to re-try.
2. Confirm the model policy (`/model`): main = Opus 4.8 xhigh, subagents = Fable 5.
   Note which cluster you're on (`hostname`, `sinfo`/`qstat -Q`).
3. Run the fast checks **once the harness exists** (`pytest -q --fast`); until then,
   run the relevant smoke (§Common commands).
4. Pick the next unchecked roadmap item (DESIGN §8) / failing check.
5. **Before you stop, update CHANGELOG.md** with what you did, the measured
   result, and anything you learned or that's now blocked.

## Development principles (small commits, tests pass, living doc)

- **Small, testable commits.** One logical change per commit — one script ported,
  one optimization rung, one bugfix. A refactor is its own commit, separate from
  features.
- **Every commit passes the checks it can run.** Once the test harness exists, run
  `pytest -q --fast` before committing; **until then, gate every commit on the
  relevant smoke** (and, for a hot-path change, the §4 equivalence check). If
  something regresses, fix it before committing — never "fix it later". If a change
  legitimately alters expected behavior, update the test explicitly (don't delete/skip it).
- **Every change ships its test.** New optimization → its equivalence check. New
  bugfix → a test that reproduces the bug first. New cluster script → the smoke
  that proves it.
- **The living document is not optional.** `CHANGELOG.md` (cross-cutting) and the
  per-cluster notes (`polaris_pbs_notes.md`, etc.) are the shared memory across
  sessions. Record: what worked, what didn't (and why — so it's not re-attempted),
  measured speedups, and every cluster fact you confirmed. Style: the narrative +
  dated "Decisions / changes log" of `si/bench_midway_notes.md`.
- **Read the `.err`/stderr first when a job fails.** Most bring-up failures are
  path/module/OOM and are visible immediately. (On Midway, a whole class of
  "missing kernels" was really an early crash before any GPU work — read the error.)
- **Concise output** (full hygiene: DESIGN §7). ≤10 lines on success; report max
  relative error + where it occurs (not raw tensors); `ERROR <reason>` greppable.
- **Never claim a step passed without reading the actual output.** Smoke scripts
  print a success token / write a CSV row — key on that, not on exit code alone.
- **Readability standards** live in [`.claude/comments.md`](.claude/comments.md) —
  follow it on any readability pass or new module.

## Cluster facts (single source of truth)

| Item | Midway (RCC) | Polaris (ALCF) — **confirmed on-node 2026-07-14** (detail: `polaris_pbs_notes.md`) |
|---|---|---|
| Scheduler | SLURM — `sbatch`/`squeue`/`scancel` | PBS Pro — `qsub`/`qstat`/`qdel` |
| Account / queue | `--account=pi-pedramh`, `-p pedramh-gpu` | **`-A lighthouse-uchicago`**, `-q debug` (smoke; 1 running job/user, ≤1 h). NOT `-q prod` for 1 node — it needs ≥10; use `preemptable` for long single-node runs |
| GPU | H100 NVL ~94 GB (Intel Ice Lake, PCIe Gen4) | **4× A100 40 GB SXM4** (AMD Milan) |
| Node/GPU directive | `--nodes=1 --gres=gpu:4` | `-l select=1:system=polaris -l place=scatter` |
| Filesystems | implicit | **`-l filesystems=home:eagle`** (confirmed accepted) — jobs are **rejected** if the flag is absent |
| Env (S2S, port) | `module load python/miniforge-25.3.0 && eval "$(mamba shell hook --shell bash)" && mamba activate /project/pedramh/shared/S2S/v2.0/venv && module load cuda/12.6` | `module use /soft/modulefiles && module load conda && conda activate <env>` |
| Env (SI) | same but `conda`: `... && conda activate /project/pedramh/shared/anthonyz/venv` (see `si/bench_midway.sh`) | same as above |
| Data (ERA5 HDF5) | `/project/pedramh/h5data/h5data` | **NOT staged** → S2S + port are blocked; Globus-stage to `/eagle/projects/lighthouse-uchicago/members/mehta5/era5_h5data/h5data`. **E3SM** (AMIP) *is* staged and drives SI/Pangu-SFNO/makani/physicsnemo — see `polaris_pbs_notes.md` §4 |
| Job id in script | `$SLURM_JOB_ID` | `$PBS_JOBID` (use `${PBS_JOBID%%.*}`) |

Configs are **cluster-specific**: fix `data_dir`, `checkpoint_path`, and the
mean/std `.nc` filenames in the YAML before launching (they fail deep in the data
loader, not early). Use `WANDB_MODE=offline`.

## One-time setup (per cluster)

Midway already has the shared envs (filled into the table above). To build fresh:
`conda env create -f <model>/…/environment.yml --prefix <project>/envs/<name>`
(SI's is `name: si`; the port adds `pytorch-lightning wandb`). On Polaris match the
torch build to the cluster CUDA (12.x) — don't reuse a Midway wheel. `wandb login`
once or `WANDB_MODE=offline` (scripts set it). Polaris also needs a one-time Globus
stage of the HDF5 data to `/eagle/<project>/…`, then repoint each config's paths.

## Common commands

> **Run these inside an interactive compute-node allocation**, never on a login
> node (Midway: `sinteractive --account=pi-pedramh -p pedramh-gpu --gres=gpu:4`;
> Polaris: `qsub -I -A <project> -q debug -l select=1:system=polaris -l filesystems=home:eagle -l walltime=1:00:00`).
> Real/long work goes through the submission scripts (`sbatch`/`qsub`), not these.

```bash
# --- S2S (canonical, torchrun) ---
cd s2s
PYTHONPATH=$(pwd)/v2.0 torchrun --standalone --nproc_per_node=4 \
    v2.0/train.py --yaml_config=v2.0/config/exp2.yaml --run_num=0100

# --- S2S-Lightning (imports ../s2s/v2.0; config resolved relative to the script) ---
cd s2s-lightning
PYTHONPATH=../s2s/v2.0:$(pwd) python smoke_train_module.py    # prints SMOKE_OK

# --- SI (bench.py takes --config = the YAML PATH, not a section name) ---
cd si
python bench.py --config configs/SI_midway.yaml --devices 0

# Bench env knobs (S2S): S2S_BENCH=1 S2S_BENCH_WARMUP S2S_BENCH_STEPS S2S_BENCH_CSV
#                        S2S_NVTX  S2S_AMP_DTYPE=bf16|fp16  TORCH_COMPILE_MODE=reduce-overhead|max-autotune
# Bench env knobs (port): S2S_BENCH_* S2S_NVTX S2S_PRECISION S2S_TORCH_COMPILE S2S_DDP_BUCKET_CAP_MB
# Bench env knobs (SI):   SI_BENCH_*  SI_NVTX  SI_PRECISION  SI_DDP_*
# Bench env knobs (PanguWeather): PANGU_BENCH=1 PANGU_BENCH_{WARMUP,STEPS,CSV} PANGU_NVTX
#                        TORCH_COMPILE_MODE. Precision is a YAML knob (amp_dtype), NOT an env
#                        var. Renamed from S2S_* on 2026-07-16 — PanguWeather is its own
#                        project (fork by COPY, not import), and nothing shared read them.
#                        train.py errors LOUDLY (ERROR LEGACY_BENCH_ENV) if an S2S_* bench knob
#                        is set, because "unset" silently means "no benchmarking".
```

Submit real work through the cluster scripts (`s2s/v2.0/HPC_scripts/midway_*.sh`,
`s2s-lightning/midway_*.sh`, `si/bench_midway.sh`; Polaris `*.pbs` per the handoff),
never directly.

## Smokes: what to run, what PASS looks like

Key on the log (token / CSV row), not the exit code. After any `s2s/v2.0/` edit the
**S2S and port** smokes must both pass (rule #5); SI is independent.

| Model | Submit (Midway) | PASS = |
|---|---|---|
| S2S | `sbatch s2s/v2.0/HPC_scripts/midway_bench.sh` | new `bench_results.csv` row + the bench summary line in the `.out` |
| Port | `sbatch s2s-lightning/midway_smoke_train_module.sh` | `SMOKE_OK` in the `.out` (finite per-step loss) |
| SI | `sbatch si/bench_midway.sh` | new `SI_BENCH_CSV` row; sanity via `si/validate_bench.py` |

Writing a **new** submission script? **Launcher shape** — S2S = `--ntasks-per-node=1`
+ `torchrun --nproc_per_node=4`; port/SI = `--ntasks-per-node=4` (== devices) +
`srun python …` (Lightning's SLURM launcher aborts on a mismatch). **Polaris/PBS: never
`srun`** — but the launcher is per-model, not "always single `python`": S2S / Pangu use
`torchrun --standalone --nproc_per_node=$NPROC`; the port and SI use a single `python`
(they self-distribute); makani / physicsnemo must use `python -m torch.distributed.run`
(their venv has no `torchrun` of its own, so the bare name resolves to the BASE conda's,
whose shebang pins the wrong python). Every Polaris script must also
`source polaris_env.sh`. And copy the env-bootstrap block verbatim from the same
model's `midway_*.sh` — module ordering differs on purpose (S2S `module purge`s; the
port must NOT).

## Repo architecture

**TWO pairs are live-coupled; the rest borrow by copy.** Know which you are in before you
edit — "these are separate projects" is true of some pairs and dangerously false of others.
See **DESIGN.md §2** for all six.

| pair | coupling | what that means |
|---|---|---|
| `s2s/v2.0/` ↔ `s2s-lightning/` | **live, by PYTHONPATH import** | the port has no copy — an edit in `s2s/v2.0/` changes both. **Rule #5.** `PYTHONPATH` must include `s2s/v2.0` for the port's `from utils…`/`from networks…` |
| `physicsnemo_sfno/` → `makani_sfno/` | **live, by EDITABLE install** ⚠ | `polaris_setup_sfno_venv.sh:78-79` pip-installs makani from a **GitHub pin** and `physicsnemo_sfno` as `-e` (editable) into **one shared venv**; the `.pth` points into this repo, and makani imports physicsnemo (`makani/utils/comm.py:19`). **An edit to `physicsnemo_sfno/physicsnemo/` changes what makani jobs execute.** Nothing in either directory says so |
| `PanguWeather/` ← `s2s/v2.0/` | **copy** (fork) | ~95% identical, no import. Fixes do **not** propagate — §2c. But see the instrumentation exception below |
| `si/`, and everything else | **copy / unrelated** | borrow freely; no live coupling |

**makani is NOT vendored in this repo** — zero files; it is a pip pin
(`git+https://github.com/NVIDIA/makani.git@<pin>`). Only `physicsnemo_sfno/` is an in-repo tree.
Corollary: PhysicsNeMo's SFNO *is* makani's model (entry point
`[physicsnemo.models] SFNO = makani.models.networks.sfnonet:SFNO`) — the two "separate" SFNO
projects run the same class.

**One rule DOES span the copy boundary, by design:** #10. `PanguWeather/v2.0/train.py:206` pins
its NVTX range names to `s2s/v2.0/train.py:188`, and
`PanguWeather/v2.0/test/bench_instrumentation_test.py` asserts s2s's CSV columns — so the two
forks' benchmarks stay comparable. Bench *knobs* are per-project (`PANGU_*` vs `S2S_*`); range
names and CSV columns are **shared contract**. Do not "fix" that asymmetry.

`s2s/v2.0/` holds the shared model (`networks/pangu.py::PanguModel_Plasim`), losses
(`utils/losses.py`), HDF5 loaders (`utils/data_loader_multifiles.py`); `si/` is the
separate SI model (own `si/CLAUDE.md`).

> ### ⚠ Independence is enforced by `PYTHONPATH` ORDER, not by structure
> `s2s/v2.0/` and `PanguWeather/v2.0/` export the **same top-level module names**
> (`utils`, `networks`, `config`) and both import **unqualified** (`from networks.pangu
> import …`). So `networks.pangu` resolves to **whichever tree is first on `PYTHONPATH`** —
> and the two `networks/pangu.py` differ by **106 lines** (measured 2026-07-16). A leaked or
> mis-ordered `PYTHONPATH` silently runs one project's trainer against the other's model.
> **Set `PYTHONPATH` to exactly one tree.** Never both.
>
> Corollary — **do not carry a rule across the boundary**. `s2s/v2.0/utils/metrics.py` does
> not even exist; PanguWeather's copy is imported only by its own `train.py`. Citing rule #5
> (which exists because the *port imports* `s2s/v2.0`) about a PanguWeather file is a category
> error — it happened in this repo, and it cost real work.

**Where to look:** measured evidence → `s2s/v2.0/bench_report.md`,
`si/bench_midway_notes.md`, `s2s-lightning/LIGHTNING_PORT.md` (+ the port-vs-v2.0
`step_med` caveat in `midway_bench_nsys_port.sh`'s header); SI knobs →
`si/CLAUDE.md` (auto-loads under `si/`); Polaris bring-up → `polaris_handoff_prompt.md`.
