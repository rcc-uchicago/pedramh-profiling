# CLAUDE.md

*How to work here.* What & why → **DESIGN.md** (read it first). Where things stand
→ **CHANGELOG.md**. Cluster detail → **`polaris_pbs_notes.md`**.

This file is deliberately short. When a rule needs evidence, the reference carries
it — don't inline the evidence back here.

## Model policy

Main session **Opus 4.8 @ xhigh** (confirm `/model`); subagents **Fable 5**
(`claude-fable-5`) unless a task needs another tier.

## Why we're here

Get six weather codebases (**PanguWeather** — the focus — makani, PhysicsNeMo,
S2S, S2S-Lightning, SI) running on HPC GPUs, then make them **faster without
changing what they compute**. Two phases, in order: **bring-up** (env → data →
scheduler scripts → training that yields an evaluatable model → inference), then
**profile, then optimize**. Phase is per model, per cluster — see CHANGELOG.
Details: DESIGN §1, roadmap §8.

**Division of labor:** bring-up and training are **ours**; the **science is
jesswan's** — variable sets, fill values, channel roles, loss definitions, physics.
*Changing what a model computes needs her sign-off.*

## Things NOT to do

The ways to silently break this project.

1. **Never change model outputs to make a benchmark faster.** Every hot-path change
   is gated on numerical equivalence vs a captured baseline. A "faster" version that
   drifts beyond tolerance is a **bug** — never loosen the tolerance. → DESIGN §4
2. **Never `find /`, `find /eagle`, `find /project`, or scan outside the repo.**
   Millions of files; it hangs. `grep`/`Grep` inside `.` only.
3. **Never run training/inference on a login node or bypass the scheduler.**
   `sbatch` (Midway) / `qsub` (Polaris). §Common commands are for interactive
   compute-node allocations only. Login nodes are also unreliable for *tests* —
   importing torch there can hang or core-dump. → `polaris_recipe_tests.pbs`
4. **Never invert `train.py` vs `train_optimized.py`.** In `s2s/v2.0/` and
   `PanguWeather/v2.0/`, **`train.py` is the current, bench-instrumented file**;
   `_optimized` is older despite the name. → DESIGN §2c
5. **Never edit live-coupled code to satisfy one consumer.** Exactly two pairs are
   coupled; everything else is a copy. Check §Repo architecture *before* editing,
   and re-run **both** smokes of a coupled pair.
6. **Never commit an optimization without (a) a passing smoke and (b) an equivalence
   check.** No exceptions. → DESIGN §4.1
7. **Never break the Midway (SLURM) path when adding Polaris (PBS) scripts.** Add a
   sibling named `midway`→`polaris`; don't edit the Midway script in place.
8. **Never commit secrets or big binaries.** NGC key → `$NGC_API_KEY`. No
   `*.h5/*.nc/*.pt/*.ckpt/*.npy/*.nsys-rep/*.sqlite` (`.gitignore` blocks them; the
   `s2s-lightning/data/constant_mask/*.npy` carve-out is the only exception).
   Baselines are JSON/CSV text summaries only. → DESIGN §4.2
9. **Never push to `main`** — branch-protected (PR + 1 review). Branch → PR; a solo
   session cannot self-approve, so leave it open and note it in CHANGELOG.
10. **Never let benchmark instrumentation drift.** A renamed NVTX range or CSV column
    silently invalidates every prior comparison and breaks `parse_nsys.py`. Range
    names and CSV columns are a **cross-project contract**; knobs are per-project.
    → DESIGN §2, `PanguWeather/v2.0/test/bench_instrumentation_test.py`
11. **Never add fudge factors or `--skip`/`xfail` a failing correctness test.** A
    wrong number means a wrong term — trace it.
12. **Never resubmit a stuck job without diagnosing first.** `queue_tags` + a large
    `eligible_time` means the queue has **no nodes** — walltime and priority are
    irrelevant and resubmitting destroys accrued eligible time. Cost a day on
    2026-08-05. → `polaris_pbs_notes.md` §1b
13. **Never launch `test.yaml` bare** — despite the name it is the full model and
    OOMs a 93 GiB H100 at its defaults.
14. **Never trust an exit code or a truncated log as a result.** Key on the PASS
    token / CSV row. `rc=0` from a killed run means nothing.

## Orientation (start of session)

1. Read **CHANGELOG.md** — state, blockers, and failed approaches not to re-try.
2. Confirm `/model`; note the cluster (`hostname`, `qstat -Q`).
3. Run the relevant smoke (§Smokes) — or `pytest -q --fast` once that harness exists.
4. Pick the next unchecked roadmap item (DESIGN §8).
5. **Before stopping, update CHANGELOG.md**: what you did, the measured result, what
   you learned, what's now blocked.

## Development principles

- **Small commits**, one logical change each; a refactor is its own commit.
- **Every commit passes the checks it can run** — the relevant smoke, plus the §4
  equivalence check for a hot-path change. Fix regressions before committing.
- **Every change ships its test.** Optimization → its equivalence check. Bugfix → a
  test reproducing the bug first. New cluster script → the smoke proving it.
- **The living document is not optional.** CHANGELOG.md + per-cluster notes are the
  shared memory. Record what worked, what didn't *and why*, and every cluster fact
  confirmed. Style model: `si/bench_midway_notes.md`.
- **Read the `.err`/stderr first.** Most bring-up failures are path/module/OOM and
  visible immediately.
- **Concise output**: ≤10 lines on success; max relative error *and where*, never raw
  tensors; `ERROR <reason>` greppable on one line. → DESIGN §7
- **Readability standards** → [`.claude/comments.md`](.claude/comments.md).

## Cluster facts

| Item | Midway (RCC) | Polaris (ALCF) |
|---|---|---|
| Scheduler | SLURM (`sbatch`/`squeue`) | PBS Pro (`qsub`/`qstat`/`qdel`) |
| Account | `--account=pi-pedramh -p pedramh-gpu` | `-A lighthouse-uchicago` |
| Queue | `pedramh-gpu` | smoke → `debug` (≤1 h); long single-node → **`capacity`** (≤168 h); `preemptable` is the fallback and **may never start**. → notes §1b |
| GPU | 4× H100 NVL ~94 GB | 4× A100 40 GB SXM4 |
| Node directive | `--nodes=1 --gres=gpu:4` | `-l select=1:system=polaris -l place=scatter` |
| Filesystems | implicit | **`-l filesystems=home:eagle`** — jobs are *rejected* without it |
| Job id | `$SLURM_JOB_ID` | `$PBS_JOBID` (`${PBS_JOBID%%.*}`) |
| Env / data paths | — | `source polaris_env.sh`; detail → notes §2, §4, §7 |

Configs are **cluster-specific**: fix `data_dir`, `checkpoint_path` and the mean/std
`.nc` names before launching — they fail deep in the loader, not early. Use
`WANDB_MODE=offline`.

## Common commands

> Interactive **compute-node allocations only** — never a login node. Real work goes
> through the submission scripts.
> Midway `sinteractive --account=pi-pedramh -p pedramh-gpu --gres=gpu:4` ·
> Polaris `qsub -I -A lighthouse-uchicago -q debug -l select=1:system=polaris -l filesystems=home:eagle -l walltime=1:00:00`

```bash
cd s2s && PYTHONPATH=$(pwd)/v2.0 torchrun --standalone --nproc_per_node=4 \
    v2.0/train.py --yaml_config=v2.0/config/exp2.yaml --run_num=0100

cd s2s-lightning && PYTHONPATH=../s2s/v2.0:$(pwd) python smoke_train_module.py  # SMOKE_OK

cd si && python bench.py --config configs/SI_midway.yaml --devices 0   # --config is a PATH
```

**Bench knobs and launcher shape are per model** — the authoritative table is
**DESIGN §2**. Two traps it records: PanguWeather uses `PANGU_*` (a stale `S2S_*`
errors `LEGACY_BENCH_ENV`), and on **Polaris never `srun`** — makani/physicsnemo need
`python -m torch.distributed.run`, not bare `torchrun`
(→ `polaris_pipelines_handoff_prompt.md` §launcher).

## Smokes

Key on the log token, not the exit code. After any `s2s/v2.0/` edit, **both** S2S and
the port must pass (rule #5); SI is independent.

| Model | Submit (Midway) | PASS = |
|---|---|---|
| S2S | `sbatch s2s/v2.0/HPC_scripts/midway_bench.sh` | new `bench_results.csv` row + summary line |
| Port | `sbatch s2s-lightning/midway_smoke_train_module.sh` | `SMOKE_OK` (finite per-step loss) |
| SI | `sbatch si/bench_midway.sh` | new `SI_BENCH_CSV` row; check `si/validate_bench.py` |

Writing a new submission script? Copy the env-bootstrap block **verbatim** from the
same model's existing script — module ordering differs on purpose (S2S `module
purge`s; the port must not) — and on Polaris `source polaris_env.sh`.

## Repo architecture

**Two pairs are live-coupled; the rest are copies.** Know which you're in before you
edit. Full picture: DESIGN §2.

| pair | coupling | consequence |
|---|---|---|
| `s2s/v2.0/` ↔ `s2s-lightning/` | **live, PYTHONPATH import** | the port has no copy — one edit changes both |
| `physicsnemo_sfno/` → `makani_sfno/` | **live, EDITABLE install** ⚠ | one shared venv; makani imports physicsnemo, so an edit there changes what makani jobs execute. Nothing in either directory says so |
| `PanguWeather/` ← `s2s/v2.0/` | **copy** (fork) | ~95% identical, no import — fixes do **not** propagate (→ DESIGN §2c) |
| `si/` + everything else | copy / unrelated | no live coupling |

Four directories are **`git subtree`s** (`PanguWeather/`, `makani_sfno/`,
`physicsnemo_sfno/`, `physicsnemo_ai_rossby/`) — edits there can conflict on a future
`subtree pull`, *including* `physicsnemo_ai_rossby/examples/weather/ai_rossby/`, which
reads as fork-owned but is not. Keep such edits minimal and contiguous. → notes §6b

> ### ⚠ Project independence is enforced by `PYTHONPATH` ORDER, not structure
> `s2s/v2.0/` and `PanguWeather/v2.0/` export the **same** top-level module names
> (`utils`, `networks`, `config`) and import unqualified, so `networks.pangu` resolves
> to whichever tree is first on `PYTHONPATH` — and the two differ by 106 lines.
> **Set `PYTHONPATH` to exactly one tree. Never both.**
>
> Corollary: **don't carry a rule across the copy boundary.** Citing rule #5 (which
> exists because the *port imports* `s2s/v2.0`) about a PanguWeather file is a
> category error — it happened here and cost real work.

**Where to look:** measured evidence → `polaris_bench_report.md`,
`s2s/v2.0/bench_report.md`, `si/bench_midway_notes.md`,
`s2s-lightning/LIGHTNING_PORT.md`; SI knobs → `si/CLAUDE.md` (auto-loads under `si/`);
ai-rossby → `physicsnemo_ai_rossby/CLAUDE.md`; Polaris bring-up →
`polaris_pbs_notes.md`.
