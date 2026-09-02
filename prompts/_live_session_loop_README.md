# Live-session cluster loop — how to stand one up on Polaris

A **cluster loop** runs a milestone semi-autonomously: the same prompt is re-fed each iteration (Ralph-style),
and the driver orients → implements → tests → commits → submits the gate compute → decides the pre-registered
gate → writes the verdict. This is the **Polaris/PBS adaptation** of the RCC/Midway pattern in
`L2LGWAS_DFE:prompts/_TEMPLATE_cluster_autonomous_loop.md` + `scripts/_TEMPLATE_cluster_loop.sbatch`, and it
carries every guardrail from that lineage (see the §R ledger in the loop prompt).

## Why this one is a LIVE session, not a batch orchestrator

The Midway pattern runs a cheap `--partition=build` orchestrator that submits gate compute as **nested
`sbatch`** jobs and rides them with a wait-loop. **That design deadlocks on Polaris.** From
`polaris_pbs_notes.md` §1b, queried from PBS:

| queue | walltime | limit that breaks the pattern |
|---|---|---|
| `debug` | ≤ 1 h | **`max_run 1` AND `max_queued 1` per USER.** A second `qsub` is rejected outright; a `-W depend=` job is rejected too, because a held job still counts. |
| `capacity` | ≤ 168 h | **`max_run 1` / `max_queued 2` per PROJECT** — one slot for all of `lighthouse-uchicago`. |
| `preemptable` | ≤ 72 h | ⚠ load-dependent: 0/9 started in 11.5 h (2026-08-05) vs 22 running (2026-09-02). Runs only on nodes `prod` isn't using. 10 concurrent/project |

So a batch orchestrator on `debug` is its own competitor and cannot submit the very jobs it exists to submit; on
`capacity` it would burn the project's only long slot to sit and wait. There is no `build`-equivalent queue.

**⇒ The live session is the orchestrator.** It runs on a login node (driving only — never compute), and each
measurement is one ordinary `qsub`. This is exactly what the notes prescribe for `debug`: *"it needs a driver
that submits chunk N+1 when chunk N finishes."* It also puts a human in front of every submission, which is the
standing rule here anyway.

## The three files of a loop

| Role | File | Status |
|---|---|---|
| **Frozen plan** (the gate ladder) | `PANGU_POLARIS_PROFILING_PLAN.md` | ✅ written |
| **Loop prompt** (the driver) | `prompts/_live_session_pangu_polaris_loop.md` | ✅ written |
| **Loop journal** (the state) | `prompts/pangu_polaris_loop_journal.md` | created by the loop on tick 1 |

There is no fourth file: the scheduler script *is* whatever PBS script the current stage needs, copied from
`PanguWeather/v2.0/HPC_scripts/polaris_bench_nsys_e3sm_sfno.pbs` and renamed `-N pploop-<stage>`.

## Running it

From a Polaris **login node**, in a live Claude Code session at the repo root:

```bash
# 0. once: be on the loop's branch, cut off the branch carrying the plan doc (NEVER main)
git checkout -b profile/pangu-polaris-profiling fix/tsoi-fill-270

# 1. start the loop — self-paced (the model picks its own cadence around qstat checks)
/loop Continue the PanguWeather-on-Polaris profiling loop per prompts/_live_session_pangu_polaris_loop.md

# ...or on a fixed interval, if you want predictable ticks while a job is in flight
/loop 20m Continue the PanguWeather-on-Polaris profiling loop per prompts/_live_session_pangu_polaris_loop.md
```

Stop it by interrupting, or the loop stops itself on `MILESTONE_COMPLETE` / `BLOCKED` (§10 of the prompt).
Kill an in-flight measurement with `qdel <jobid>` — the job id is in the journal.

**A tick that only polls `qstat` and finds the job still running is a legitimate quiet tick.** That is the
normal steady state while compute is in flight; the loop should say so in one line, not invent work.

### Internet from the driver

The live session runs on a **login node**, which has outbound access, so nothing special is needed for the
Claude API. This matters only if the loop is ever moved into a batch job: **Polaris compute nodes have no direct
route** and everything must go through the ALCF squid proxy — a direct connection returns HTTP 000
(`polaris_env.sh:86`, `polaris_setup_wandb.sh:12-14`, `polaris_pbs_notes.md:44-49`):

```bash
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export ftp_proxy=http://proxy.alcf.anl.gov:3128     # `module load conda` sets the first two, NOT this one
export no_proxy=localhost,127.0.0.1,::1,.alcf.anl.gov
```

The measurement jobs themselves need it only for `wandb` — and this loop runs `WANDB_MODE=offline`, so they do
not.

### Submission policy — there is exactly one, and it is not configurable

**The loop NEVER submits a job without explicit, per-submission approval from the operator.** There is no
standing-approval mode, no "scoped batch" mode, and no cap that converts into blanket permission. Starting the
loop is **not** approval to submit. Approving a stage, a plan, or a prediction is **not** approval to submit.
Approving one `qsub` is **not** approval of the next one, even for the identical script.

Each time a stage is ready for compute, the loop stops and presents:

- the literal `qsub` line it wants run, and the script it points at,
- the queue, the walltime and the node count, with the node-hour arithmetic,
- the pre-registered prediction the job is meant to test,
- what changes if it is *not* run (which stage stalls).

Then it waits. A `debug` job is still a submission — a short walltime does not make it free. This applies
equally to a **resubmit** of a job that failed, and to any dependency chain.

**Why this is absolute:** a submission spends shared allocation and takes queue position that cannot be handed
back. `capacity` allows one running job per **project**, so taking it blocks every other `lighthouse-uchicago`
member for up to 168 h; `debug` is one running / one queued per **user**, so an unwanted job blocks the loop's
own next step. And cancelling to undo a mistake destroys accrued `eligible_time`, which is the priority that
gets a job started at all (`polaris_pbs_notes.md` §1b).

The practical consequence for the loop: **Tier 0 of the plan needs no `qsub` at all** — it is re-analysis of
captures already on disk. The loop can run unattended through the whole of Tier 0 and only then come back for
the first submission.

## What's baked in (don't weaken)

Carried verbatim from the Midway lineage: the **branch guard** (never `main`, never push, never amend), the
**per-change commit ratchet** with reset-to-last-green, the **hashed pre-result prereg** (no back-fitting), the
**explicit staging** rule (never `git add -A` — every capture here is >120 MB), the **infra-failure ≤5
ratchet**, the **stage-scoped reference fence**, the **completion-honesty rule**, and ~30-min-cadence status
reporting. Added for this cluster: **one job in flight**, **diagnose-before-resubmit** (`queue_tags` ⇒ stop —
this cost a day on 2026-08-05), **PASS-token-not-`rc`**, and **science fenced to jesswan**. Dropped, with
reasons, in the prompt's §R: the batch orchestrator, the nested wait-loop, the transient-`squeue` retry.

## What changes per milestone

Only the science: the frozen plan (`PANGU_POLARIS_PROFILING_PLAN.md`) and the §2 stage ladder + §0 refuted
framings in the loop prompt. Everything else is the same skeleton. To start a second loop (SI, ai-rossby,
multi-node), copy the prompt, swap §0/§2, and give it its own journal and its own `-N` prefix so the in-flight
check stays scoped to one loop.

## If ALCF ever offers a cheap long-lived queue

Then the headless variant becomes viable and this becomes a true autonomous loop: port
`_TEMPLATE_cluster_loop.sbatch` to PBS (`#PBS` headers, `qsub`/`qstat -u`/`qdel`, `$PBS_JOBID`), replace
`--signal=B:USR1@600` with a self-timer (PBS has no pre-kill signal — compute a deadline from
`qstat -f $PBS_JOBID` → `Resource_List.walltime` and chain `CHAIN_MARGIN` seconds early, with `trap … TERM` as
the backstop), export the proxy block above, and keep `set -m` so the trap can kill Claude's whole process
group. Two things that are **not** portable and must be re-derived: PBS truncates job names in `qstat` (use
persisted job ids, not a name filter), and `qsub`-from-a-compute-node has never been exercised on Polaris by
this project — verify it before relying on a self-chain, and fall back to
`qsub -W depend=afterany:<jobid>` from a login node.
