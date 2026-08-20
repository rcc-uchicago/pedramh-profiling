# Live-session cluster loop — PanguWeather profiling on Polaris

> **How to run this.** From a Polaris **login node**, in a live Claude Code session:
> `/loop 20m Continue the PanguWeather-on-Polaris profiling loop per prompts/_live_session_pangu_polaris_loop.md`
> (or `/loop` with no interval to let the model self-pace between `qstat` checks).
> Setup and the guardrail ledger: [`_live_session_loop_README.md`](_live_session_loop_README.md).
>
> This is a **Ralph-style loop** (Huntley) adapted to a live session: the same prompt is re-fed each tick, your
> previous work persists in **git history + `CHANGELOG.md` + the loop journal**, and you iterate until a terminal
> state. **The completion-honesty rule is absolute — see §10.** The generic guardrails below are carried from the
> hardened Midway `_TEMPLATE_cluster_autonomous_loop.md`; §R records what each prevents and which three were
> dropped for Polaris (with the reason). **Do not weaken them.**

You are driving the **PanguWeather-SFNO-on-Polaris profiling milestone** from a live session on a login node.
A human is present but is *not* re-deriving your work: they approve `qsub`s and read your verdicts. Your job:
**walk the frozen plan's stage ladder in order, one measurement at a time, and turn each open question into a
recorded number with a stated trust bound** — never into an assumption.

**The frozen plan is [`PANGU_POLARIS_PROFILING_PLAN.md`]** (the tiered to-do list, each item with what it
measures, how, and what decision it unblocks). **Read it from disk each tick; do NOT re-derive or re-order its
gates.** Companions: `POLARIS_PROFILING_HANDOFF.md` (the Midway/Delta findings and which transfer),
`polaris_bench_report.md` (the existing A100 profile), `PROFILING_TABLES.md`, `polaris_pbs_notes.md` (cluster
facts), `CHANGELOG.md` (state, blockers, failed approaches).

> **⚠️ CENTRAL DISCIPLINE: VERIFY AGAINST THE ARTIFACT — NEVER ASSUME.** Every load-bearing number you act on is
> re-derived from a primary you can open: an nsys `.sqlite`, a `bench_results.csv` row, a job log, a `path:line`
> in the source. Distinguish **measured** from **estimated/argued** and label which in every write-up. A number
> that does not trace, or a gate that looks gamed, is a **finding** — surface it, never bury it.
>
> **Refuted framings NOT to re-import** (they are already dead; re-deriving them wastes a tick):
> the ~9% idle is *not* launch latency; "91% occupancy" is *GPU-busy fraction*, not warp occupancy; NCCL kernel
> time is *not* an upper bound on comm cost; `broadcast_buffers=False` is *not* worth 33% here (it is 0.11% on
> Polaris — §0c of the plan); `num_data_workers`, `ddp_static_graph` and `batch_size>1` are measured dead on this
> model (plan §5); CUDA Graphs has no bottleneck to remove (GPU-busy is 95.7%).

---

## 0. Authority, invariants, fences — read before anything destructive

**Authority order:** `CLAUDE.md` (how to work here) → `DESIGN.md` (what/why) → the frozen plan → `CHANGELOG.md`
→ per-cluster notes. Where they disagree about *science*, **jesswan owns the science** and nothing changes
without her sign-off (CLAUDE.md §Division of labor).

**Non-negotiable invariants — do NOT flip, do NOT relax:**

- **Never change what the model computes to make a benchmark faster.** Every hot-path change is gated on
  numerical equivalence vs a captured baseline; a "faster" version that drifts beyond tolerance is a **bug**.
  **Never loosen a tolerance, never add a fudge factor, never `--skip`/`xfail` a failing correctness test.**
  (CLAUDE.md #1, #11; DESIGN §4.)
- **This loop is a PROFILING loop, not an optimization loop.** Measuring is unblocked; *adopting* an
  optimization is not, and stays blocked until a PanguWeather §4.1 baseline exists (plan item 18). If a stage
  tempts you to land a speedup, that is the back-fit trap — record the measurement, leave the code alone.
- **`PYTHONPATH` must name exactly ONE tree.** `s2s/v2.0/` and `PanguWeather/v2.0/` export the same top-level
  module names (`utils`, `networks`, `config`) and import unqualified, so `networks.pangu` resolves to whichever
  is first. Never both. (CLAUDE.md §Repo architecture.)
- **`train.py` is the current, bench-instrumented file; `train_optimized.py` is older despite the name.** Never
  invert them. (CLAUDE.md #4.)
- **Benchmark instrumentation is a cross-project contract.** NVTX range names and CSV columns must not drift — a
  rename silently invalidates every prior comparison and breaks `parse_nsys.py`. Knobs are per-project; names
  are not. When you add a range, **edit BOTH lists in `parse_nsys.py`** (the SQL `WHERE text IN (…)` *and* the
  print loop) or hoist them to one constant. (CLAUDE.md #10; the drift is still live for `unstack`.)
- **Coupling:** `PanguWeather/` is a **copy/fork** of `s2s/v2.0` — fixes do NOT propagate, and rule #5 (the
  live-coupled pair) does **not** apply to a PanguWeather file. Citing it there is a category error that has
  already cost this project real work. `PanguWeather/` is also a **`git subtree`** — keep edits minimal and
  contiguous.
- **Never edit the vendored trees to add instrumentation.** Use an injector (the `ACE2_retrain/ace2_nvtx.py`
  pattern): PanguWeather is a fork and `physicsnemo_ai_rossby` is a subtree whose edits conflict on pull.

**Hard safety fences:**

- **Never `find /`, `/eagle`, `/project`, or scan outside the repo.** Millions of files; it hangs. `grep`/`Grep`
  inside `.` only. (CLAUDE.md #2.)
- **SUBMISSION AUTHORITY — narrowed by the operator on 2026-08-20, after Tier 0 completed.** The original
  rule here was "never submit without explicit per-submission approval, absolute, no standing-approval mode."
  **The operator explicitly replaced it** (this is an authorised change, not drift — do not "restore" it):

  | scope | authority |
  |---|---|
  | **`debug`, 1 node, ≤1 h** | ✅ **auto-submit. No approval needed, per submission or otherwise.** |
  | `capacity` (any size) | ⛔ **stop and ask, every time** |
  | `preemptable` (any size) | ⛔ **stop and ask, every time** |
  | **multi-node**, any queue | ⛔ **stop and ask, every time** |

  **Why the split, so it is not widened by accident.** `debug` is `max_run 1`/`max_queued 1` **per user** and
  capped at 1 h, so a `debug` job cannot block another project member — the worst case is one wasted hour of
  my own slot. `capacity` is `max_run 1` **per PROJECT**: taking it blocks every other member of
  `lighthouse-uchicago` for up to 168 h, and cancelling to undo destroys accrued `eligible_time`, so a
  submission there cannot be handed back. `preemptable` started **0/9** jobs in 11.5 h on 2026-08-05 — cheap
  but it may never run, which is the operator's call, not mine. Multi-node multiplies the allocation per job.

  **What still holds inside the auto-submit scope, unchanged:**
  * **Still write the prereg first, and commit it before submitting** (§2). Auto-submit removes the approval
    gate, not the pre-registration gate.
  * **Still at most ONE job in flight** — `debug`'s per-user limits make a second `qsub` a rejection anyway.
  * **Still never resubmit on `queue_tags`.** That is a queue-has-no-nodes signal; resubmitting resets accrued
    `eligible_time` and is strictly harmful (CLAUDE.md #12). Auto-submit authority is **not** retry authority:
    the ≤5 infra-failure ratchet stands, and a *crashing* job's re-submit still gets diagnosed first.
  * **Still report every submission** with its job id, queue, walltime, node-hour arithmetic and the prediction
    it tests — the operator reads verdicts, so the accounting does not go away just because the gate did.
  * **Still no `qsub` outside the plan.** Auto-submit covers jobs the frozen plan calls for, not exploratory
    ones I invent.

  ⇒ **Tier 0 of the plan needed no `qsub` at all** and is now complete (items 1–5), so from here every item
  costs compute.
- **Never run training/inference on a login node or bypass the scheduler.** The live session drives; every
  measurement is a `qsub`. Importing torch on a login node can hang or core-dump — so **not even a "quick
  check"** goes there. Pure-CPU sqlite/JSON re-analysis of an existing capture is the one exception and is
  explicitly allowed (that is what Tier 0 is). (CLAUDE.md #3.)
- **Never commit secrets or big binaries.** No `*.h5/*.nc/*.pt/*.ckpt/*.npy/*.nsys-rep/*.sqlite`. `.gitignore`
  blocks them — **never `-f` past it.** Baselines are JSON/CSV text summaries only. NGC key → `$NGC_API_KEY`.
  (CLAUDE.md #8.)
- **Stage files explicitly, by path. Never `git add -A`.** This loop generates `.nsys-rep`, `.sqlite` and
  scratch scripts every stage; a blanket add is how gigabytes land in history.
- **Never push, never open a PR, never merge to `main`** (branch-protected; a solo session cannot self-approve).
  Leave the branch for review and note it in `CHANGELOG.md`. (CLAUDE.md #9.)
- **`git reset --hard` ONLY to discard an UNCOMMITTED broken WIP** back to the last green commit — never a green
  commit, never force-push, never amend.
- **Never launch `test.yaml` bare** — despite the name it is the full model and OOMs at its defaults.
  (CLAUDE.md #13.)

## 0.5 Reference fence — verify on disk at orient; do NOT proceed from memory

Each tick, confirm the references the **current stage** needs are present and readable. Do *not* check
everything up front — a Tier-2 artifact missing must not halt a Tier-0 stage.

- The frozen plan `PANGU_POLARIS_PROFILING_PLAN.md`; `CHANGELOG.md`; `polaris_pbs_notes.md`.
- `polaris_env.sh` and `polaris_require_topups`; the Pangu Polaris scripts under
  `PanguWeather/v2.0/HPC_scripts/polaris_*.pbs` (the env-bootstrap block you copy from).
- The existing captures: `${MEMBER_ROOT}/bench/nsys_pangu_sfno_7255503.{sqlite,nsys-rep}` and `…_7255557.*`
  (Tier 0 reads these; **read-only, `mode=ro` URI** — never write to a capture).
- `ACE2_retrain/{parse_nsys.py,kernel_census.py,ace2_nvtx.py}` (the tools; `kernel_census.py` is **known
  broken** — unguarded `correlationId` join, plan item 4).
- `baselines/` — currently only `ai_rossby_pangu_plasim/` and `ai_rossby_sfno/`. **The absence of a
  PanguWeather baseline is itself a load-bearing fact** (plan item 18), not a fence failure.

**The fence (HALT condition):** if an artifact the *current* stage needs is absent or unrecoverable, **STOP that
stage**: write the one-line blocker into the journal, tell the human what to restore, and move to an
independent stage. Do not reconstruct a number from memory.

---

## 1. Each tick — orient first, in parallel

1. `git status --short`; `git log --oneline -8`; `git rev-parse --abbrev-ref HEAD` (must be
   `profile/pangu-polaris-profiling`, cut off the branch carrying the plan — **never `main`**).
2. Read the **loop journal** `prompts/pangu_polaris_loop_journal.md` (create it on the first tick from the
   skeleton in §7): current stage, its pre-registered prediction, the in-flight job id if any, the tick count.
3. **Is one of this loop's jobs in flight?** `qstat -u $USER`. If a `pploop-*` job is `R` or `Q`:
   **do not submit anything and do not start a second stage.** Report its state and end the tick — the loop
   will re-fire. If it is `Q` and has been for a while, run the §4 diagnostic *once*; a `queue_tags` comment
   means the queue has no nodes and **resubmitting is strictly harmful.**
4. If a job just finished: **key on the PASS token / CSV row, never on `rc` or a truncated log** (CLAUDE.md
   #14). Read the `.err`/stderr first — most failures are path/module/OOM and visible immediately.
5. Read the current stage's entry in the frozen plan. The gates are frozen; if a result tempts you to edit one,
   that is the back-fit trap.

---

## 2. The deliverable — the stage ladder (dependency order, from the frozen plan)

Walk the plan's tiers **in order**. Each stage below is one plan item; its **gate** is "the number exists, is
reproducible, and is written up with its trust bound", and its **fail-branch** is a recorded null or a
pre-registered blocker — *never* a skipped stage.

| # | stage | gate = done when | fail-branch |
|---|---|---|---|
| **T0** | plan items **1–5** (no GPU: NVTX join, n=2 re-derivation, analytic bytes model, fix `kernel_census.py`, warmup check) | each number committed + a CHANGELOG bullet; the copy time split forward vs backward exists | if the NVTX text path cannot be resolved from the capture, record *that* as the finding and move on — do not guess a split |
| **T1a** | item **6** — `gpu_topology_check.py`, 1 min | a pairwise bandwidth matrix pasted into `PROFILING_TABLES.md` + the handoff's OPEN cell closed | node has an unexpected topology ⇒ that is the finding, not a failure |
| **T1b** | item **7** ⭐ — ncu on the top six kernels, **single rank**, explicit metric list | achieved DRAM bandwidth + sectors/request per kernel; the plan §0d "~25% of peak" estimate either confirmed or replaced | ncu unavailable/permission-denied on Polaris ⇒ blocker with the exact error; fall back to the analytic model (T0 item 3) and say so |
| **T1c** | item **8** — attribute `direct_copy`/`conj` to source lines | the 3–4 call sites behind 42% of GPU time, cited `path:line` | `emit_nvtx` timings are void by design — report **shares only**, never a time |
| **T1d** | item **9** — re-capture nsys at the **production** config (`checkpointing: 2`, warmup ≥40, EMA active) | a capture whose NVTX table is non-empty + the §0d table re-derived on it | OOM at `ckpt2` (ai-rossby OOM'd there without ZeRO) ⇒ record the OOM as the finding; it is a real production risk |
| **T1e** | items **10–11** — the `ckpt` ladder in Pangu's own harness (one job, interleaved A/B/A/B) + the EMA/`ckpt2` memory margin | ratios from ONE job on ONE node; `peak_mem_gb_max_rank` across the epoch-6 EMA boundary | a cross-job ratio is **not** a measurement — if you cannot get them in one job, record no ratio |
| **T2** | items **12–17** — multi-node scaling ⭐, whole-epoch profile ⭐, memory profile ⭐, complex64 island, SFNO-internal NVTX, python sampling | each its own commit + CHANGELOG bullet | multi-node needs ≥2 nodes: if the queue cannot supply them, that is a queue finding, not a stage failure |
| **T3** | item **18** — capture the missing PanguWeather §4.1 baseline (**the gate for everything else**); then 19–20 remain **listed, not run** | `baselines/pangu_e3sm_sfno/*.json` exists, fixed seed, world size 1, K=20 | any drift beyond 2.5e-7 (same node) / ~1e-5 (cross-arch) is a **finding to trace**, never a tolerance to widen |

**The headline discipline, restated in every stage:** *we are establishing where the time goes and what the
hardware ceiling actually is — not making anything faster.* A stage that produces a speedup instead of a
number has failed even if the number is good.

**Pre-registration — this project's own precedent, and it is mandatory.** Before every measurement job, write
your **predicted** result and the decision rule into the journal and **commit that as its own pre-result
commit** (`prereg: <stage> — predicted …`). Then run the job. Then compare. This is the pattern that made the
ZeRO sweep credible ("predictions were written into the script before it ran; **6 of 8 exact**") and the one
that catches a back-fit. Before recording any verdict, assert the prereg commit is an **ancestor of HEAD**
(`git merge-base --is-ancestor <sha> HEAD`) and **unmodified since** (`git diff --quiet HEAD -- <file>`). If it
is absent or dirty, that is a §9 blocker — **never** write a prereg after seeing the number.

---

## 3. The inner loop — the per-change commit ratchet

1. One stage at a time. Small commits, one logical change each; a refactor is its own commit.
2. Edit, then run the **directly-touched** test/smoke. **GREEN ⇒ COMMIT** (the new last-green floor; stage the
   touched source/test/doc files **by explicit path**). **RED ⇒ fix in place, ≤5 attempts, else
   `git reset --hard <last-green>`** to discard the uncommitted WIP and re-implement simpler.
3. **Every change ships its test** (CLAUDE.md §Development principles): a new analysis script ships the check
   that proves it; a bugfix ships the test reproducing the bug *first*; a new PBS script ships the smoke.
   Analysis code that produces a load-bearing number must be re-runnable by someone else, from the repo.
4. After any `s2s/v2.0/` edit, **both** the S2S smoke and the s2s-lightning port smoke must pass (they are
   live-coupled). This loop should not be touching `s2s/v2.0/` at all — if a stage wants to, stop and ask.
5. **A failed check is a real bug or a real finding.** Fix the cause. Never loosen, never `xfail`.
6. Commit trailer: `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` (match the branch's existing
   trailers — check `git log --format='%(trailers:only=true)'` rather than assuming).

---

## 4. Cluster discipline — ONE job in flight, and the resubmit trap

- **Every measurement is a `qsub`, and every `qsub` needs the operator's explicit approval first** (§0 — no
  standing approval, no exceptions, resubmits included). Prepare fully: write the script, run the static checks,
  do the node-hour arithmetic, write the prereg — **then stop and show the literal command.**
- **The live session never runs GPU work itself.** Copy the env-bootstrap block
  **verbatim** from `PanguWeather/v2.0/HPC_scripts/polaris_bench_nsys_e3sm_sfno.pbs` (module ordering differs
  between models on purpose) and `source polaris_env.sh`. Required PBS header, or the job is **rejected**:
  ```
  #PBS -N pploop-<stage>              # the loop prefix — how the §1 wait-check finds only OUR jobs
  #PBS -A lighthouse-uchicago
  #PBS -q debug                       # ≤1 h; 9/9 started, median 19 s wait. See the queue table below.
  #PBS -l select=1:system=polaris
  #PBS -l place=scatter
  #PBS -l filesystems=home:eagle      # jobs are REJECTED without this
  #PBS -l walltime=00:55:00
  #PBS -j oe
  ```
  On Polaris **never `srun`**. Pangu uses `torchrun`; makani/physicsnemo need
  `python -m torch.distributed.run` (never bare `torchrun`).
- **`qstat` truncates job names**, so `-N pploop-<stage>` is for humans. **Persist every job id to the journal**
  as you submit — the id is authoritative for the §1 in-flight check.
- **AT MOST ONE job in flight.** `debug` is **`max_run 1` AND `max_queued 1` per USER** — a second `qsub` is
  rejected outright, and even a `-W depend=` job is rejected because a held job still counts. This is also why
  this loop lives in a live session and not in a batch orchestrator (§R).
- **Queue selection** (`polaris_pbs_notes.md` §1b — read it before choosing):
  | queue | walltime | note |
  |---|---|---|
  | `debug` | ≤1 h | **the default for this loop.** Proven; cannot chain or pre-load |
  | `capacity` | ≤168 h | 1–4 nodes, but **`max_run 1` per PROJECT** — check the slot is free and coordinate *before* taking it: `qstat -a \| awk 'NR>5 && $3 ~ /capacity/ {print $2, $10}'` |
  | `preemptable` | ≤72 h | **0/9 started in 11.5 h** on 2026-08-05. The fallback that may never start |
- **⚠ NEVER resubmit a stuck job without diagnosing first** (CLAUDE.md #12 — this cost a day on 2026-08-05):
  ```bash
  qstat -x -f <jobid> | grep -E "^ +comment|^ +eligible_time|^ +job_state"   # anchor the grep!
  qstat -Q | grep -E "^Queue|^debug |^capacity"
  ```
  `Insufficient amount of resource: queue_tags` ⇒ **the queue has no nodes.** Not walltime, not priority, not
  node count. Resubmitting resets accrued `eligible_time` and makes it strictly worse. A large-and-growing
  `eligible_time` says the same thing. **Diagnose once; then wait, chunk into `debug`, or tell the human —
  never loop on resubmit.** A bare `grep eligible_time` matches `wfp_eligible_time_exp` first and returns an
  exponent — anchor it.
- **An infra failure (OOM / wall-timeout / crash before a metric) is not a gate FAIL.** Right-size and propose
  a re-submit — **each one separately approved** (§0), **≤5 times total** across the whole milestone (count it
  in the journal); on the 5th, it is a blocker. Never re-submit a crashing job indefinitely, and never treat the
  first approval as covering the retry.
- **Never `Monitor`-block a whole tick waiting on a job.** Submit, record the id, end the tick. The loop's next
  firing checks `qstat`.
- **`WANDB_MODE=offline`.** Configs are cluster-specific: fix `data_dir`, `checkpoint_path` and the mean/std
  `.nc` names before launching — they fail deep in the loader, not early.
- **Caches on `/eagle`, never `/local/scratch`.**

---

## 5. Review gauntlet — reduced, because a human is present

The Midway loop runs a full Fable-5 reviewer/critic gauntlet as the *autonomous substitute for operator
sign-off*. **A live session has the operator**, so the gauntlet shrinks to the two roles a human reviewer
cannot cheaply play. Spawn on **Fable 5** (`claude-fable-5`) per CLAUDE.md §Model policy, **only** when a stage
lands a load-bearing number:

- **An adversary** (`general-purpose`), fed the gate, the measured number, and the artifact it came from, with
  the standing charges: *is this an artifact of the sum-vs-union confusion? of an unguarded `correlationId`
  join? of a cross-job comparison? of a contaminated warmup? of a config that is not the production config? is
  the "estimate" being quoted as a measurement?* An un-refuted strike re-enters §3.
- **A drift auditor**, before each CHANGELOG entry: does the entry match what actually landed, and does any
  *other* doc now contradict it (`polaris_bench_report.md`, `PROFILING_TABLES.md`, the handoff, `DESIGN.md`)?
  This project's docs are its shared memory; a stale table is a future wrong decision.

Anything that changes what the model computes goes to **jesswan**, not to a subagent.

---

## 6. Branch + rollback

- Work on **`profile/pangu-polaris-profiling`**, cut off the branch carrying the plan doc. **Never `main`.**
- The verdict is its own commit. **Do not push, do not amend, do not `--no-verify`, do not merge.**
- **One stage landed ⇒ a clean slate for the next.** In a live session that means: summarize the stage into the
  journal + CHANGELOG so the next tick can orient from *files*, not from context that may have been compacted.
  Assume your context can vanish between ticks; the journal is what survives.

---

## 7. Bookkeeping — the journal, then the CHANGELOG

**The loop journal** `prompts/pangu_polaris_loop_journal.md` is the loop's own state. Committed (it is small
text), newest tick at the top:

```markdown
## tick <N> — <YYYY-MM-DD HH:MM> — stage <T1b> — <one line: what this tick did>
- **in flight:** job <id> (`pploop-<stage>`, queue, submitted <ts>) | none
- **prereg:** <the prediction + decision rule, written BEFORE the job ran> (commit <sha>)
- **result:** <measured value> — <measured | estimated | OPEN> — vs prediction: <hit/miss>
- **next:** <the single next action>
- **infra-failure count:** <n>/5
```

**On a stage landing, update `CHANGELOG.md`** — newest entry at the top of `## Decisions / changes log`, in the
house format, which is `- **YYYY-MM-DD** — **<what happened>** — <result/measurement> — <what it means / next>`
with nested bullets for the detail. Match the existing voice: bold the claim, give the job id, mark **OPEN** /
**REFUTED** / **RETRACTED** explicitly, and say what it means for the next decision. Concretely:

1. The **honest measured numbers**, each with its job id and whether it is measured or estimated.
2. **What it refutes or supersedes**, named — if a number contradicts `polaris_bench_report.md` or
   `PROFILING_TABLES.md`, fix that doc in the same commit and say so ("Fix the note, not this row").
3. **What is now blocked**, and what the next unchecked plan item is.
4. Keep the ≤10-line-on-success discipline in *logs*; the CHANGELOG is the place for the long form.
5. Update `polaris_pbs_notes.md` with any **new cluster fact confirmed** (a queue limit, a module quirk, an
   `nsys`/`ncu` path). Style model: `si/bench_midway_notes.md`.
6. Tick the plan's checkbox in `PANGU_POLARIS_PROFILING_PLAN.md` and update the **Next actions** pointer in
   `CHANGELOG.md`.
7. **Record what did NOT work and why.** A failed approach not written down gets re-tried; that is the single
   most expensive failure mode in this repo's history.

---

## 8. Science-edit protocol

**Variable sets, fill values, channel roles, loss definitions, physics — jesswan's.** Anything that changes
what the model computes: flag it (a `SCIENCE-NOTE` line in the CHANGELOG "what it means" + the journal), never
self-apply. Named instances already on the table: `broadcast_buffers=False` (BN running statistics diverge
across ranks) and any seeded `worker_init_fn` change (it changes the noise realization and therefore the loss
trajectory). The production default path stays byte-identical.

---

## 9. Blocker protocol

A genuine blocker: a required artifact is unrecoverable; a tool is unavailable on Polaris (`ncu` permissions);
an infra failure hits the 5th non-completion; the adversary lands an unrefuted strike; a gate needs jesswan.
**Stop that stage.** Write the blocker into the journal *and* a one-line `ERROR <reason>` the human can grep,
commit only that, tell the human exactly what you need, and **continue to an independent stage** if one
remains. A pre-registered fail-branch (a recorded null, an OOM that is itself the finding) is a **valid
result, not a blocker**. Never fabricate a pass, never skip a gate, never loosen a threshold.

---

## 10. Loop control and terminal states

Re-enter at §1 each tick. Three terminal states, mapped onto the live-session harness:

- **`STAGE_LANDED`** — the stage's number is measured, committed, written up, the gauntlet cleared. Report it,
  clear the journal's "in flight", and continue to the next plan item on the following tick.
- **`MILESTONE_COMPLETE`** — every stage in §2 decided (each a measured result or a recorded fail-branch), the
  plan's checkboxes ticked, `CHANGELOG.md` + `polaris_pbs_notes.md` current, the branch left for review with a
  one-paragraph summary of what is now known that was not before. **Then stop the loop** (`ScheduleWakeup` with
  `stop: true`) and tell the human.
- **`BLOCKED`** — §9 fired and no independent stage remains. Stop the loop and say exactly what unblocks it.

Otherwise: no terminal state — report the tick's one-line status and let the loop re-fire. A tick that only
polled `qstat` and found a job still running is a legitimate quiet tick; say so briefly rather than inventing
work.

> **THE COMPLETION-HONESTY RULE (absolute).** A terminal state is a *promise the human trusts*. Declare
> COMPLETE or BLOCKED **only when it is completely and unequivocally true** — never to escape the loop because
> you feel stuck, over-budget, or that you "should be done." If you are stuck, the honest exit is BLOCKED with
> a real blocker written down. A false completion is the worst failure mode of a loop: it ships an unfinished
> milestone as done. The escape hatches are the tick cap and the human — not a lie. **Corollary for this
> project specifically: never report a number you did not measure, never round an estimate into a
> measurement, and never let `rc=0` stand in for a PASS token.**

---

## R. Robustness ledger — why each guardrail exists

Carried from the Midway `_TEMPLATE_cluster_autonomous_loop.md` §R (hardened across dir-58 → dir-59 →
dfe-info-budget); each line is a bug that bit or would have.

| Guardrail | Failure it prevents |
|---|---|
| **Per-submission operator approval, always, no standing mode** | An unattended loop spending shared allocation the operator did not authorise. `capacity` blocks the whole project for up to 168 h and a cancel-to-undo destroys accrued `eligible_time` — a submission cannot be handed back. |
| **One job in flight; ids persisted to the journal** | `debug`'s `max_run 1`/`max_queued 1` rejecting a second submit — or worse, two jobs writing the same capture path. |
| **Diagnose before ANY resubmit; `queue_tags` ⇒ stop** | Resubmitting into a dry queue, destroying accrued `eligible_time`. Cost this project a day on 2026-08-05 (CLAUDE.md #12). |
| **Anchored `grep` on `eligible_time`** | Matching `wfp_eligible_time_exp` first and reading an exponent as a wait time. |
| **PASS token / CSV row, never `rc`** | `rc=0` from a killed run reading as success; nsys writes a report file even when it captured nothing (CLAUDE.md #14). |
| **Pre-result prereg commit (ancestor + unmodified)** | Back-fitting a prediction after seeing the number. The ZeRO sweep's credibility came from exactly this. |
| **One node / one job / interleaved A/B** | A cross-job Polaris ratio being quoted as a measurement — node-to-node spread is 10.5%, the same order as the effects being chased. |
| **Explicit staging; never `git add -A`** | Committing `.nsys-rep`/`.sqlite`/checkpoints. Each capture here is >120 MB. |
| **Branch guard (never `main`, never push)** | Committing to a protected branch, or shipping unreviewed work. |
| **Per-change commit ratchet + reset-to-last-green** | A red WIP contaminating the last-green floor; an unbounded fix spin. |
| **Infra-failure ≤5 ratchet** | A crashing job re-submitted ~25× before anyone notices. |
| **Stage-scoped reference fence** | Halting the whole milestone at orient over an artifact only the last stage needs. |
| **Journal is the state, not context** | A compacted or lost context silently restarting a stage, or losing an in-flight job id. |
| **Adversary on every load-bearing number** | A plausible-but-wrong finding landing as fact — this repo has already had to retract several (§0 refuted framings). |
| **Drift auditor before each CHANGELOG entry** | A new number landing while the old table still says the opposite; the docs are the shared memory. |
| **Completion-honesty rule** | Faking COMPLETE to escape → shipping an unfinished milestone as done. |
| **Science fenced to jesswan** | A "harmless" flag flip changing what the model computes. |
| **~30-min-cadence status reporting** | A long capture mistaken for a stall. |

**Three Midway guardrails are deliberately DROPPED, with the reason** (do not re-add them without re-reading
this):

| Dropped | Why it does not apply on Polaris |
|---|---|
| The `--partition=build` **batch orchestrator** + `USR1` chain-resubmit | There is no cheap long-lived queue here. `debug` is `max_run 1`/`max_queued 1` **per user**, so a batch orchestrator that submits nested `debug` jobs is its own competitor and deadlocks; `capacity` is `max_run 1` **per project**, so a nested design would consume the project's only long slot twice. The **live session is the orchestrator** instead — which also puts a human in front of every `qsub`, matching the standing rule that jobs are not submitted without approval. |
| The nested-job **wait-loop** with the `⟨SLUG⟩-nest*` name filter | Replaced by the §1 `qstat` in-flight check driven by persisted job ids — more precise, because PBS truncates job names in `qstat` output. |
| The **transient-`squeue` retry** (3× before fail-open) | The live session cannot double-submit while waiting for a human's approval, so a transient `qstat` failure costs a tick rather than causing an overlapping launch. Re-check next tick instead of failing open. |
