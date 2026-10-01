# Analysis — what the 2026-09-29 debug jobs mean for G and for fine-tuning, and what to run next

**Moderated three-agent review, job 7669964** (`polaris/polaris_makani_debate.pbs`, `debug`, 1 node,
2026-09-30 05:11–05:30 UTC, tree `9adfc33a`, `MAKANI_DEBATE_OK`). Three `claude -p --model
claude-opus-5-5` processes on a compute node via the ALCF proxy, no tools, answering only from the
evidence pack (brief `docs/2026-09-30_g_finetune_debate_brief.md` + both handoffs + the spatial result
+ CHANGELOG 2026-09-20..29). Analyst A (stability/science lens) and Analyst B (evidence/cost lens)
wrote independently, then rebutted each other; the moderator wrote what follows. Full transcript:
`$MEMBER_ROOT/runs/makani_debate/7669964/{a1,b1,a2,b2,moderator}.md` (+ `evidence_pack.md`, prompts).

Working-session check after the job: the moderator's correction that the 43x worst channel is
**`Z3_l17`** is right — CHANGELOG 2026-09-20 ("The verdict that DID fire is `DRIFT_FIRST`, on ONE
channel of 101: `Z3_l17`", every other channel <= 1.371 NRMSE at 336 h). The brief called it
unconfirmed. Recommendations below are the moderator's, not decisions: queue/node choices are the
operator's; science items are jesswan's.

---

## Bottom line
- The 2026-09-29 jobs showed the machinery works. None of them measured whether F or G stays stable. Evidence: F2 bitwise (7669103), unit suites 6/6 (7669001), and G trains on 2 CXI nodes (7668637).
- 2 nodes beat 1 at equal settings (81.1 % per-GPU efficiency, one 120-step run, 7669001). G ≈ 24.7 h / 49 node-h is a **projection**. F's log will give the first production 2-node step time.
- Every single-step model screened has died. Failure starts at the model top and ends at PS (7649597, 7649647). G is single-step, keeps both, and loses the only mass corrector built (D6). It is a base for fine-tuning, not a fix.
- Run first, today: jesswan's multi-year protocol on B e22 + B e24 (debug). It is the only test of whether the 1-year screen, which will pick every arm's winner, predicts 5-year survival.
- Then F → F3+F4 in one job → T-anneal-F + `anneal_dryair`-F → per-epoch screens → multi-year runs on each arm's top epochs.
- G: scratch, 2 nodes, `SPARE=0`, 48 h, released by hand after F's extension decision. Gate it on jesswan's TMQ answer, not on F3.
- Park G-warm, T-d16, sharding and `w=4`. Run T-d8-F only if a pre-registered trigger fires.

## What the debug jobs mean for G

| job | what it shows for G | strength |
|---|---|---|
| 7668600 | The 2020–44 view and its stats are valid. Means moved ≤1.42 % σ, stds ≤2.9 %, mostly at the model top | measured; effect on stability unknown |
| 7668637 | G trains on 2 nodes (in 76 / out 77, EMA); the arm `lrcheck` settings run on G | machinery only. The step-264 blow-up comes from a 40-step model. The step-368 train→valid handoff (O6) is still unexercised |
| 7669001 | 225.2 vs 365.4 ms per step, so 2 nodes beat 1; sharding is 26–174 % slower | one run per layout, 60 steps × 2 epochs, on A's 101-channel config. Carrying this over to G's production step is a projection |
| 7669103 | Full-width code paths are bitwise unchanged | the 77-channel paths rest on unit tests and the smoke only |
| 7669129 | Sets G's comparison row: A on the common 77 channels | 1 checkpoint, 24 ICs, 2048–49 |

**Nodes and walltime (operator decides).**
- 2 nodes with `SPARE=0`: 49 vs 74 node-h (projected). The pack records that the spare is charged, not what it protects against ([2] §4).
- Walltime 48 h. My arithmetic with the 09-29 sizing formula: 48 h covers 243 epochs at any step up to ≈0.59 s, so it holds even if 2 nodes give no speed-up over A's 473 ms.
- Use F's measured step to choose nodes, not walltime. If it is ≥473 ms, 2 nodes save no wall-clock time and D2's alternative (1 node, 60–72 h) is cheaper.

**Scratch, not warm.**
- No transfer proof exists at G's scale (24 channels dropped, new stats).
- F's own proof is still provisional: its 99-channel denominator (7646696) has no result in the pack.
- A warm start inherits the 2015–19 spin-up years that the split excludes (09-29 "Split").
- The ≈4.4 h vs 24.7 h saving assumes F's 43 epochs would be enough for G. F has not run yet.

**Evaluation bar.**
1. **Stability.** Screen G at 2044 f1092 and f1156 on 77 channels under a pre-registered 77-channel addendum. None exists yet; [3] §3.5 covers only 99 channels. This screen also closes O6. That G dies like A is a prediction, not evidence.
2. **Skill.** Compare only against the common-77 row (NRMSE336 1.0660, ACC336 0.4400, VR336 1.0130, worst channel 1.371) and state the split difference.
   - Compare the numbers, not the verdict class. A's DRIFT_FIRST rested only on the worst-channel clause.
   - The 09-20 entry names `Z3_l17` as the channel that tripped it and puts every other channel at ≤1.371. That matches the common-77 worst channel exactly.
   - So a G verdict other than DRIFT_FIRST reflects the channel set, not the model (7669129).
   - NRMSE and ACC are flattered by the annual-mean climatology, so VR is the cleaner discriminator (09-20).
3. **Admissibility.** Run the multi-year protocol with a PS clause on a *G fine-tune*, not on single-step G.

**What G can fix:** the `Z3` drift where the loss is blind, and the hypsometric inconsistency (7646192, 7646252).

**What G cannot fix:**
- Single-step instability (A died at 8/8 starts; A e200/e220/e243 died, 1 start each).
- Model-top drift in `V_l00`, `RELHUM_l00/l04` and `T_l00/l01`. G keeps all of these.
- PS mass loss. G keeps PS, and without TMQ it cannot run `DryAirFix`.
- Negativity (7650442), and FSNT/FSNTOA staying prognostic (D5).
- G's own PS drift is unmeasured.

**Timing.**
- On `capacity`, G cannot start before F and any extension end (one running job per project).
- Do not chain it with `afterany:7660250`: that fires before F's extension resume (D1).

## What they mean for fine-tuning
- **Readiness.** F0/F1 are green (6/6 in 7669001 and 7669129) and F2 is bitwise (7669103). Launcher dry runs refuse wrong bases (09-29). Still open: F itself, F3, F4, the 99-channel addendum, D7, and the operator's re-confirmation (F5).
- **Speed.** 7669001 measured single-step training at batch 32, local 4. The arms use batch 16, local 2 or 1, and multi-step 5 or 9, which is unmeasured. Keep the 12 h and 16 h walltimes. T-d8's 11–12 h assumes linear scaling from a 1-node probe (7650039). 7669001 shows 81 % efficiency at 2 nodes, not 100 %, so the extra 4 h are needed margin.
- **Gap in the selection rule.** The 1-year screen reproduces the mass-loss rate (B e01: −31/−32 hPa at 1 yr in 7649792, vs −34…−42 at 1.25 yr in 7649597). It does not predict survival: B e01 survived a year at both starts, then died at 7 of 8 starts at 2.5–5 yr. Add multi-year runs on each arm's top epochs.
- **Annealing.** B was never annealed, and its fixed PS bias swings −58 / +0.7 / −68 / +4.6 hPa across epochs 21–24. That annealing narrows the swing is a **hypothesis**. T-anneal-F's per-epoch PS column tests it.
- **Confound.** T-anneal-F vs B e22 changes both the base (A→F) and the schedule. The A-based control was declined ([3] §6.3); say so in the result.
- **Dry-air on F only.** Applied after training, it cut survivors from 16 to 14 (n=1 start, selection-biased). It also cooled the column: `T_l17` got colder in 12 of 14 survivors (7650652). The trained arm is the real test. Any default is jesswan's.
- **D7 (raw vs EMA checkpoint).** Screen both in F3. It is cheap: 38 one-year rollouts took 11 min (7649647).
- **Depth 16: park it.**
  - Activation recompute is not yet proven bitwise.
  - Sharding needs `debug-scaling`, which is held (D3), and runs slower.
  - Sharding cannot carry dry-air, because h/w > 1 is refused.
  - `w=4` gives a 26 % higher loss, unexplained.
  - Depth 8 has never trained.
- **G-based arms:** only T-anneal-G and T-d8-G, after G's screen.

## What to run next

| # | run | queue | nodes | walltime | node-h | gate before it | what it answers | decision owner |
|---|---|---|---|---|---|---|---|---|
| 1 | Multi-year protocol, 8 starts → 2049-12-31, on **B e22 + B e24** (inference). Optional: B e01 with dry-air on | debug | 1–2* | ≤1 h | ≤2 | Stable copies of both files. B e22 sits in the rotating `ckpt_mp0_v1.tar` slot; confirm B e24's epoch from inside its file. Pre-register the PS clause, the three outcomes (see disagreement 2) and a model-top timing readout. Submit before F ends | Does 1-yr drift predict 5-yr survival? Does the model top fail while mass holds? (Mechanism only: 101 channels with soil) | operator (run); jesswan (criteria) |
| 2 | F 7660250 as queued, plus its extension check | capacity | 3 (2 + spare) | 12 h | ≈16 projected; ≤36 | none | First production 2-node step time; the soil-free base | operator |
| 3 | F3 (raw-best, EMA-best and snapshots; f1092, f1156) + F4 `lrcheck`, in one job | debug | 2 | ≤1 h | ≤2 | F ended; 99-channel addendum with a PS clause | Blow-up without soil (confounded); evidence for D7; does the launcher load F? | operator (pre-authorised) |
| 4 | T-anneal-F + `anneal_dryair`-F | preemptable | 2 each | 12 h each | ≤24 each (T-anneal ~18, 09-24 estimate) | #3 tokens green; D7 decided; F5; T-d8 trigger pre-registered | Does annealing settle the per-epoch PS bias? Does trained conservation keep mass without cooling the column? | operator (queue, concurrency); jesswan (dry-air stays diagnostic) |
| 5 | Per-epoch screens (2 starts), then multi-year on each arm's top 2 epochs | debug | 1–2* | ≤1 h each | ≤2 per job | the arm's epochs exist | Is there an admissible soil-free checkpoint, and is the winner an isolated epoch? | operator; jesswan (admissibility) |
| 6 | G scratch, 243 epochs, `SPARE=0` | capacity, released by hand after F and its extension | 2 | 48 h | ≈49 projected; ≤96 | F's step time read; extension decided; #1 recorded; jesswan's TMQ answer | The G base | operator; jesswan (TMQ) |
| 7 | G screen on 77 channels, f1092 and f1156 | debug | 1–2* | ≤1 h | ≤2 | #6 done; 77-channel addendum | G's single-step stability vs F3; closes O6 | operator |
| 8 | T-d8-F | preemptable | 4 | 16 h | 45–50 estimated; ≤64 | Depth-4 winners fail #5, or #1 gives outcome (b) | Does depth 8 buy survival? | operator |

\*Node counts for the screen and protocol jobs are not in the pack.

- **#1 runtime (estimate):** survivors run ≈7,667 leads at 0.023–0.026 s/lead (7649567), about 3.5 min each. 16–24 members fit in 1 h only if they run in parallel, as 7649597's did. If time is tight, drop the dry-air sub-arm first.
- **Parked:** G warm, T-d16, phase 2, `w=4`, and the A-based T-anneal control.

## Where the analysts disagreed, and the ruling
Settled in round 2: G scratch by default, no `afterany` chaining, T-d8-F conditional, a 77-channel addendum, B e24 added to #1, multi-year runs on each arm's winners.

1. **F3 as a hard gate for G.**
   - A (round 1): gate G on F3. B: no gate. A (round 2) conceded, but expects F3 to be read first anyway.
   - **Ruling: not a gate.**
   - Evidence: F3 is confounded. F = A e243 + 43 more single-step epochs with warm restarts ([3] §1). A e200, e220 and e243 all died, which is only a partial control (1 start each). F3 speaks only to soil; G drops 22 more channels.
   - Measurement that would separate the causes: a single-step continuation of A that keeps soil. Not proposed.
2. **How to read B e22's multi-year run.**
   - A (round 1): if it survives, mass is a selection problem, dry-air is optional and G goes ahead unconditionally. If it dies, the model top is the killer. B (round 2): "dies at PS" separates nothing; read the timing instead.
   - **Ruling: B.**
   - Evidence: every non-finite run ended at PS (7649597), yet A had *gained* PS (+156 Pa at lead 368, 7649572) before the model top failed (`V_l00`@382). PS is where runs end, not what causes the failure.
   - Pre-register three outcomes:
     - (a) survives with |ΔPS| near its 1-year value;
     - (b) `V_l00` leaves 3σ, or the run dies, while |ΔPS| stays small, meaning the model top fails independently of mass;
     - (c) |ΔPS| grows past its 1-year value, meaning the screen underestimates drift.
   - "Dry-air optional" is jesswan's call, and one checkpoint does not make a rule.
3. **A dry-air arm in #1 (A, round 2).**
   - **Ruling: optional, and on B e01, not B e22.**
   - B e22 barely drifts: PS +0.7→−0.07 hPa, `T_l17` −0.44→−0.59 K. The corrector has almost nothing to correct.
   - B e01 is where the fix acts: PS −31→−1.0 hPa, `T_l17` −0.83→−10.5 K (7650652). B e01 also has a matched fix-off record at the same 8 starts.
   - The fix is still applied after training, which the pack says "is not the real test". It cannot decide the no-TMQ question for G. The trained `anneal_dryair`-F arm can.
4. **When to ask jesswan about TMQ.**
   - A: ask with #1's result in hand. B: ask today.
   - **Ruling: ask today and forward #1 when it lands.** The decisive evidence, the trained arm, arrives after G would start. She should know she is deciding without it.
5. **"G makes mass drift worse" (A, round 1).**
   - **Ruling: B's reading.** G's PS drift is unmeasured, and the fine-tune itself can set it (B −661 vs A +156 Pa at lead 368, n=1 each). What can be said is that G loses the only corrector built.
6. **Concurrent vs chained arms.**
   - **Ruling: recommend concurrent; the operator decides.**
   - Neither arm's readout depends on the other, and `preemptable` allows 10 jobs per project.
   - The one-at-a-time approval (09-24) predates the re-base on F, and F5 needs re-confirmation anyway.
   - F4 green limits the risk of one F-path defect burning both runs.
7. **G walltime.**
   - D2 gives 36–48 h. Both analysts (round 2) proposed F's step × 1.5–2, capped at 48 h.
   - **Ruling: 48 h.** It already covers the case where 2 nodes give no speed-up.

Questions the pack cannot decide, and the measurement that would:

| question | measurement |
|---|---|
| Does annealing narrow the per-epoch PS swing? | T-anneal-F's per-epoch PS spread |
| Does trained conservation help? | `anneal_dryair`-F's screens plus multi-year runs |
| Does 81 % efficiency hold in production? | F's log |

## Claims to verify before acting

| claim | source | issue |
|---|---|---|
| G 24.7 h, 49/74 node-h; F 5.2 h | [1] §4 | Projection from one short run on A's config. 7668637's G step time is not in the pack |
| 81 % efficiency carries over to the arms | a1, b1 | The arms' batch, local batch and multi-step settings are unmeasured |
| T-d8 needs 45–50 node-h, 11–12 h | 09-24 | Linear scaling assumed; 7669001 contradicts it |
| Warm G ≈4.4 h, saves ≈20 h, costs "one-fifth" | a1, b1 | Assumes 43 epochs suffice, plus the 292 ms projection |
| #1 "costs minutes" | a2, b2 | 7649597's ~8 min included A's early deaths |
| #1's dry-air arm decides the TMQ question | a2 | Post hoc ≠ trained; nearly inert on B e22 |
| The protocol job can switch dry-air on and has a TMQ reference | a2 | Not in the pack |
| B e24's file name and whether it sits in a rotating slot | b2 | Only B e22's file is named (7649792) |
| "If F3 survives, stability is a channel-set problem" | b1 | Confounded; speaks to soil only |
| "Depth 4 helps" | 7649647 | 4 of 5 matched epochs at 1 start; 2 matched epochs at the second |
| Annealing stabilises the mass bias | a1 | Hypothesis |
| G is "the endpoint model" | a1 | Not stated in the pack |
| The 43× channel is "unconfirmed" | [1] §2 | 09-20 names `Z3_l17`. The 43.16 (labelled VR) vs 43.18 mismatch should be checked in `k56_metrics.h5` |
| F starts 2026-09-30 06:51 UTC | [1] §1 | PBS estimate only |

## Decisions needed

| owner | decision | default I would take |
|---|---|---|
| operator | Submit #1 now | Yes: copy the files first, then B e22 + B e24. Add dry-air on B e01 only if the protocol job accepts the flag without a code change |
| operator | Queue for F's extension resume | `capacity`, per F's rule |
| operator | D7: raw vs EMA checkpoint for the arms | Raw best, revisited with F3's EMA rows |
| operator | Arms: queue and concurrency | `preemptable`, concurrent, after F5 |
| operator | T-d8-F | Conditional on the pre-registered trigger |
| operator | G release, queue, nodes, spare, walltime | By hand after F's extension decision; `capacity`; 2 nodes; `SPARE=0`; 48 h. Switch to 1 node / 60–72 h if F's step is ≥473 ms |
| operator | D3 / phase 2; A-based T-anneal control | Wait; no |
| jesswan | G without TMQ or dry-air (D6) | Accept (D6 default). The only conservation evidence so far is negative, and B e22 held PS below 1 hPa/yr with no corrector. Revisit if `anneal_dryair`-F wins; a G with TMQ would cost ≈49 node-h again (projection) |
| jesswan | Is a warm start admissible? | No: scratch |
| jesswan | Multi-year admissibility: PS threshold and the three outcomes | Adopt the outcomes; the threshold is hers |
| jesswan | Dry-air default | Stays diagnostic |
| jesswan | PRECT/RELHUM clamp; `Z3` diagnostic; capped loss weights | No change until the arms are screened |
| jesswan | Model-top question | Raise it if #1 gives outcome (b) |
| jesswan | Ensemble (D4) | Not now |
