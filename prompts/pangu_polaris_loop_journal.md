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

## tick 1 — 2026-08-20 — stage T0 item 1 — NVTX text path resolved; prereg for the fwd/bwd copy split

- **in flight:** none (Tier 0 needs no `qsub`)
- **prereg (written BEFORE the split was computed):**
  - **P1.** Launches falling **outside** all four house phases will be **< 2%** of all launches on rank 0.
    Rationale: the four phase windows sum to ~335 ms of a 603 ms step, but the missing ~266 ms sits between
    `optimizer` end and `step_N` end, which reads as the CPU blocking on a sync while the GPU drains — a wait,
    not a launch site. If instead >10% of launches land outside, there is an unranged launch site (EMA? logging
    sync? validation?) and the phase attribution is incomplete — that becomes the finding.
  - **P2.** `backward` will hold **>= 65%** of the `direct_copy` + `conj` GPU time. Rationale: `checkpointing: 3`
    recomputes the forward inside backward, so backward pays recompute (~1x forward) plus the adjoint
    (~1.5-2x forward) while forward pays 1x.
  - **P3.** `forward_loss` **15-30%**, `optimizer` **< 5%** (FusedAdam is its own kernel; EMA never fired),
    `data_prep` **< 1%** (0.16 ms/step of CPU window).
  - **Decision rule.** If P2 holds, activation **recompute** is a first-order share of the 271 ms/step and the
    `ckpt3 -> ckpt2 -> ckpt1` ladder (plan item 10) directly deletes a measurable part of §0d — the ladder is
    then the highest-value Tier-1 item after item 7. If instead `backward` < 50%, the copies are intrinsic to
    the SFNO spectral path (SHT transpose/`conj`), checkpointing barely touches them, and the lever is layout,
    not recompute. Either way the number is recorded; neither outcome edits a gate.
  - **Stated limit, pre-committed:** this split cannot separate *recompute* from *adjoint* inside `backward` —
    there is no NVTX range between them. That separation is plan item 17, not this item.
- **result:** OPEN — the split is computed in tick 1's second half, after this commit.
- **measured this tick already (not part of the prereg):**
  - **The NVTX text path is the inline `text` column, `domainId=0`, `eventType=59` (push/pop).** All four house
    ranges are present in `nsys_pangu_sfno_7255503.sqlite`: `data_prep`/`forward_loss`/`backward`/`optimizer`
    at **160 rows each** = 40 steps x 4 ranks, plus `step_20..step_59`. The `textId -> StringIds` path holds
    **only** NCCL's registered strings (`ncclAllReduce` 2402, `ncclBroadcast` 160, `domainId=1`). So the plan
    §1 item-1 symptom is explained, not merely worked around: nothing is missing from the capture.
    `parse_nsys.py`'s `WHERE text IN (...)` was already on the correct path.
  - **The `correlationId` join inflates by +29.4% on this capture** (naive 459,088 rows vs guarded 354,720 =
    exactly one row per kernel). The guard is `k.globalPid = (r.globalTid & -16777216)`; verified that clearing
    the low 24 bits of every RUNTIME `globalTid` reproduces the four KERNEL `globalPid` values exactly.
    Independently confirms the handoff §5 figure of +30.8% on a different capture. — measured
  - **`kernel_census.py` has a second, independent bug** beyond the missing guard: `enclosing(rs, tid)` looks up
    the NVTX range on the *launching* thread, but on rank 0 **62,680 of 88,680 launches come from the autograd
    worker thread** (`...2d149f`) while **all 201 NVTX events are on the main thread** (`...2d139d`). Same-thread
    attribution would therefore credit `(outside)` for the whole of `backward`. Attribution must be scoped to
    the **process** (globalPid), not the thread. — measured, plan item 4
- **next:** compute the split with a process-scoped, pid-guarded, launch-time attribution; then land item 4's
  fix to `kernel_census.py` reusing the same join.
- **infra-failure count:** 0/5

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
