# Loop journal — PanguWeather profiling on Polaris

The loop's own state, and the thing that survives a lost or compacted context. **Newest tick at the top.**
Driver: [`_live_session_pangu_polaris_loop.md`](_live_session_pangu_polaris_loop.md) · frozen plan:
[`../PANGU_POLARIS_PROFILING_PLAN.md`](../PANGU_POLARIS_PROFILING_PLAN.md) · setup:
[`_live_session_loop_README.md`](_live_session_loop_README.md).

Entry shape (keep it):

```markdown
## tick <N> — <YYYY-MM-DD HH:MM> — stage <id> — <one line: what this tick did>
- **in flight:** job <id> (`pploop-<stage>`, <queue>, submitted <ts>) | none
- **prereg:** <prediction + decision rule, written BEFORE the job ran> (commit <sha>)
- **result:** <value> — <measured | estimated | OPEN> — vs prediction: <hit | miss + why>
- **next:** <the single next action>
- **infra-failure count:** <n>/5
```

---

## tick 0 — 2026-08-20 — setup — loop machinery written; nothing submitted

- **in flight:** none
- **prereg:** n/a (no measurement this tick)
- **result:** the three loop files exist (driver prompt, README, this journal) and the frozen plan is
  `PANGU_POLARIS_PROFILING_PLAN.md`. Design decision recorded: **live-session driver, not a batch
  orchestrator** — `debug` is `max_run 1`/`max_queued 1` per user and `capacity` is `max_run 1` per project, so
  a PBS orchestrator that submits nested PBS jobs deadlocks against itself (README §"Why this one is a LIVE
  session"). — measured (the queue limits are quoted from `polaris_pbs_notes.md` §1b, queried from PBS
  2026-08-05; not re-queried today)
- **starting state, so tick 1 does not re-derive it:**
  - Branch for the loop: `profile/pangu-polaris-profiling`, to be cut off `fix/tsoi-fill-270` (carries the
    plan). Not yet created.
  - Tier 0 is fully unblocked — both captures are on disk:
    `${MEMBER_ROOT}/bench/nsys_pangu_sfno_{7255503,7255557}.sqlite`.
  - Already derived from 7255503 (do **not** re-measure; see the plan §0): GPU-busy union 95.6–96.5%; NCCL
    88.7% overlapped / 1.2% exposed; `ncclDevKernel_Broadcast_RING_LL` present at 0.11%; `direct_copy` + `conj`
    = 42.2% of GPU kernel time at an **estimated** 17–27% of HBM peak.
  - Known-broken tool: `ACE2_retrain/kernel_census.py` (unguarded `correlationId` join) — plan item 4.
  - Known-open blocker for everything downstream: **no PanguWeather §4.1 baseline exists** — plan item 18.
- **next:** tick 1 = orient, create the branch, then plan item **1** (fix the NVTX↔kernel join so the 42% copy
  time can be split forward vs backward). No GPU, no `qsub`.
- **infra-failure count:** 0/5
