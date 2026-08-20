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

## tick 2 — 2026-08-20 — stage T0 item 2 — prereg for the n=2 re-derivation on job 7255557

- **in flight:** none (Tier 0 needs no `qsub`)
- **preconditions established from FILES, not from the capture** (so they are not part of the prediction):
  - `bench/bench_env_polaris_nsys_7255557.txt` records **the same workload** as 7255503's §0 description:
    `nettype: sfno_plasim`, `checkpointing: 3`, `world_size: 4`, `bench_warmup: 20`, `bench_steps: 40`,
    `num_data_workers: 1`, batch 1/GPU, bf16, `ddp_find_unused: false`, `use_ema: True`,
    `yaml_sha256_16: 47d632f85c84353a`, `git_sha: 9c3122e67d71`, torch 2.8.0 / CUDA 12.9, A100-SXM4-40GB.
    **There is no env file for 7255503**, so the two configs cannot be compared by sha — the structural
    check has to come from the captures themselves (rank count, phase-row count, kernel set).
  - **7255557 is the clean RE-RUN of 7255503, and the fix was to the clock, not the model** (CHANGELOG
    2026-07-15): `elapsed` was sampled *after* `cudaProfilerStop()`, so 7255503 read `elapsed=51.8 s` vs
    `sum=25.7 s` and its bench row was **refused** by the self-check. 7255557 records cleanly at rc=0.
    ⇒ the two should be the same workload; that is exactly what makes this a usable n=2.
  - **7255557 is much noisier at the harness level.** Its CSV row: `step_med 0.606010`, `step_p90 0.812770`,
    `step_mean 0.695946`, **`step_std 0.235865` = 39% of the median** — against 7255503's capture-side step
    std of **31.9 ms on 603.5 ms = 5.3%** (§4.1). `step_mean >> step_med` says there are outlier steps.
    Also `peak_mem_gb_max_rank` **28.762 GB** vs the sweep's 26.98 GB (§5) — the nsys run costs ~1.8 GB more.
- **prereg (written BEFORE any query against `nsys_pangu_sfno_7255557.sqlite`):**
  - **P1 (structure).** 4 ranks; `data_prep`/`forward_loss`/`backward`/`optimizer` at **160 rows each**;
    `(outside)` = **0.0%** again; the pid-guarded join returns exactly one row per kernel.
  - **P2 (the one that matters — ABSOLUTE work reproduces, SHARES may not).** `direct_copy`+`conj` should
    reproduce in **absolute ms/rank-step within ±5% of 271.19**, because that is config-determined compute.
    Its **share** of GPU kernel time may come in **below 42.2%**, because a stall inflates NCCL and therefore
    the denominator. Predicting the share moves *down*, not up.
  - **P3 (the split).** `backward` / `forward_loss` of the copy time reproduces **within ±2 points** of
    **72.9 / 27.1** — this is a ratio of two compute buckets and should be the most stable number here.
  - **P4 (stalls).** Given `step_std` = 39% and `step_mean >> step_med`, 7255557 contains **more or larger
    comms stalls than 7255503's single step-30 event**, and its NCCL ms/rank-step will exceed 7255503's
    67.82. Total GPU kernel time per rank-step will therefore come in **above** 643.19.
  - **P5 (bandwidth).** D2D above L2 within **±3 points of 82%** of peak.
  - **Decision rule.** If P2+P3 hold while P4 also holds, then **§0d's percentages are stall-sensitive and
    §0's numbers should be quoted as absolute ms/rank-step (or stall-excluded), not as shares** — that is a
    methodological finding worth landing in the plan, and it makes the n=2 stronger rather than weaker. If
    the *absolute* copy time does **not** reproduce within ±5%, then the two captures are not the same
    workload despite the matching env, and item 2's answer is "no usable n=2 exists" — a recorded null, not
    a failure. If P3 fails, the split from item 1 is not a property of the model and §4.3c must be
    de-generalised to job 7255503 alone.
  - **Stated limit:** this is n=2 on the **same node type, same day, same git sha**. It tests
    run-to-run reproducibility, **not** node-to-node (spread 10.5%) and not config sensitivity.
- **result:** OPEN — measured after this commit.
- **next:** run the re-derivation.
- **infra-failure count:** 0/5

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
- **result:** **3/3 predictions HIT.** Prereg integrity asserted *before* writing this line:
  `985214b5` is an ancestor of HEAD, and the prereg text above is byte-identical to that commit
  (`git diff --quiet 985214b5 HEAD -- prompts/pangu_polaris_loop_journal.md` was clean at the time
  of measurement; re-check against `985214b5`, not against a later HEAD, since this `result:` line
  was added afterwards by design).
  - **P1** `(outside)` < 2% → **0.0%** (measured). Stronger than predicted: **all** 354,720 launches
    fall inside a house phase, so the step's "missing" 268 ms contains zero launches — it is pure
    GPU drain, which closes an open question in `polaris_bench_report.md` §4.1.
  - **P2** `backward` >= 65% of copy time → **72.9%** (measured; 197.6 ms/rank-step of 271.2).
  - **P3** `forward_loss` 15-30% → **27.1%**; `optimizer` < 5% → **0.0%** of copies; `data_prep`
    < 1% → **0.0%** (it launches zero kernels at all; its GPU cost is 2.18 ms/rank-step of H2D).
  - **Decision rule fired (P2 held):** activation recompute is first-order, so the `ckpt` ladder
    (plan item 10) is the highest-value Tier-1 item after item 7. The split gives a **ceiling**, not
    a decomposition: recompute <= forward's own 150.3 ms/rank-step ⇒ ckpt-off <= **24.9% of the
    step (<= 1.33x)** — and ai-rossby's measured full ladder (1.307x = 23.5%) sits **1.4 points**
    under it, so the ceiling is nearly saturated and essentially the whole forward is recomputed at
    `ckpt3`.
  - **Correction to the prereg's own pointer:** the recompute-vs-adjoint separation is plan item
    **16** (SFNO-internal NVTX ranges re-fire inside `backward` during checkpoint recompute), not
    item 17 (`--python-sampling`, which samples CPU stacks and cannot partition GPU time). The
    prereg text above is left unedited — it is frozen; this is the correction of record.
  - **Extra, not preregistered:** D2D `cudaMemcpyAsync` in the same capture sustains **1265 GB/s =
    81% of the A100's 1555 GB/s**, all four devices within 2 points. Intra-device HBM only — it does
    **not** close the interconnect/topology cell (item 6). It does kill the "81% is unreachable on
    this node" reading of §0d's estimated 17-27%, so it narrows item 7 without replacing it.
- **gauntlet: CLEARED — both roles returned, 42 findings between them, all triaged.** Ran on the
  inherited tier, not Fable 5: **`claude-fable-5` is out of credits on this account** (both first
  attempts died with "Usage credits are required for this model"). Worth knowing before the next tick
  plans a subagent.
  - **Drift auditor:** 27 items. It independently re-derived **every** §4.3 number from the capture and
    reproduced all of them. Three were real defects in my §4.3 (bytes column mixed per-rank-step with
    all-rank totals; a dropped `forward_loss` row so 128 != 131 launches; "26 points" mixed
    denominators). Two were interpretation errors (item 17 -> item **16**; "comfortably inside the
    ceiling" when ai-rossby's ladder actually sits 1.4 points under it). All applied.
  - **Adversary: 2 FATAL + 5 MATERIAL + 8 MINOR, and both FATALs held up.**
    1. **The removable/non-removable buckets were INVERTED.** Recompute happens *inside* `backward`, so
       `backward`'s 197.58 ms is the bucket that shrinks and `forward_loss`'s 73.61 ms is the one no
       checkpointing level can remove. My sentence said the opposite. The 27% magnitude was right only
       by coincidence (recompute ~ forward). **A reader would have concluded "73% is untouchable, skip
       the ckpt ladder" — the exact wrong decision.**
    2. **"Recompute cannot exceed the forward's GPU time" is not a bound.** Kernels with equal fwd/bwd
       launch counts run at **1.0136x, never below 1.0**, and recompute selects *different* GEMMs
       (+15%/call). So it is an estimate of ~150 ms. It also caught that my section contradicted itself
       in six lines: header said ESTIMATED, bullet said "measured bound".
    3. **My "backward is 77.4% of the step" was the sum-vs-union confusion** — the very framing the
       driver's §0 lists as already-refuted. Union is **408.63 ms = 67.7%**; only `backward` self-overlaps
       (12.5%, NCCL on `streamId 19`). I verified this myself by adding a union column to the tool: it
       reproduces the adversary's figure exactly.
    4. Also confirmed by my own re-derivation: the `2 x bytes` rule fails sub-L2 (**two** buckets compute
       to >100% of peak, 124.7% and 110.0% — the proof); the D2D stream claim (all on `streamId 7`, so
       serialized with compute, not concurrent — my hedge was the wrong hedge); 960 of 962 H2D, not 100%;
       step-30 comms stall worth -12.0% on the NCCL mean; and the `conj` warrant must come from **source**
       (no `conj` in `networks/modulus_sfno`, einsum over `view_as_complex`, 24/rs = 2 x `num_layers`),
       not from the phase split, since a recompute-only kernel would look identical.
    5. **Two claims it tried and could not break:** the +29.4% phantom join and the guard dropping
       nothing (it checked for orphaned/graph-launched kernels and found none), and `(outside)` = 0.0%
       (it went further and found the CPU inside `cudaDeviceSynchronize` for 10.72 s on rank 0).
  - **Strike 15 was a process finding against me:** the bandwidth table had **no committed code behind
    it** — it came from an ad-hoc heredoc, violating "analysis code that produces a load-bearing number
    must be re-runnable from the repo". Fixed by adding `--memcpy`, `--per-step` and the union column to
    the tool, so §4.3h now lists a command for every table in the section.
- **also landed this tick (commit 99378811), found by the new tool's test:** `parse_nsys.py` could
  not run on a Polaris login node **at all** — `sqlite3.connect(PosixPath)` needs Python >= 3.7 and
  the login default is **3.6.15**, and `statistics.fmean` is 3.8+. Both fixed with regression tests;
  the NVTX range list hoisted to one `RANGE_NAMES` constant, which also repairs the live CLAUDE.md
  #10 drift (`unstack` was in the SQL but not the print loop, so its rows were fetched and silently
  dropped). Verified behaviour-preserving: the NVTX table now reproduces §4.1 exactly.
- **FLAGGED FOR THE OPERATOR, not fixed:** `s2s/v2.0/HPC_scripts/parse_nsys.py` is a *different*
  copy carrying only **8 of the 19** range names, and it is the one the Polaris PBS scripts and
  `physicsnemo_ai_rossby/polaris/bench_instrumentation_test.py` actually invoke. So when plan item
  16 adds SFNO-internal ranges, the Polaris analysis path will print **nothing** and look like the
  instrumentation never fired. Fixing it touches the **live-coupled** s2s pair, which this loop must
  not do (driver §3.4) — it needs its own change with both S2S and s2s-lightning smokes run.
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
- **STAGE_LANDED — plan items 1 AND 5, both ticked.** Item 5 fell out of the adversary's warmup
  attack: no warmup regime (first step 640.26 vs 634.36 ms median), but step index 30 is a comms stall
  worth -12.0% on the NCCL mean. Commits: `985214b5` (prereg) -> `17d2baf1` (tool+test) -> `99378811`
  (parse_nsys 3.6 fixes) -> `66572846` (verdict) -> `dfb36132` (item 1, §4.3) -> `5b48b176` (drift
  repair across 6 docs).
- **next:** plan item **2** — re-derive §0d on the **second** capture, job **7255557**, as a true n=2.
  The drift auditor was explicit that item 2 is **NOT** closed by this tick: §4.3 is an independent
  *query path* on the *same* capture. Then item **3** (analytic bytes model) and item **4**
  (`kernel_census.py`). Tier 0 remains `qsub`-free; the first submission request will be item **6**.
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
