# Debate brief — what the 2026-09-29 debug jobs mean for port G, for fine-tuning, and what to run next

Written 2026-09-30 by the working session as the framing for a three-agent review (two analysts,
one moderator; `polaris/polaris_makani_debate.pbs`). It states facts with their primary source and
poses the questions. It deliberately does **not** give the working session's recommendation.
Everything after this file in the evidence pack is primary: the G/spatial handoff, the F fine-tune
handoff, the spatial result doc, and the makani CHANGELOG entries of 2026-09-20 … 2026-09-29.

## 1. The models (letters are used throughout the pack)

| model | what | rollout training | channels (out / in) | train years | status |
|---|---|---|---|---|---|
| **A** `prod1n_b32_sgdr` | makani SFNO, production recipe (batch 32, LR 2e-3, cosine warm restarts T₀=20, 243 epochs, 1 node, 46.3 h) | **single-step** (`n_future 0`) | 101 / 107 (100 state + PRECT; 7 forcings) | 2015–2044 | done |
| **B** `nf4_prod_b16_r1` | fine-tune of A | `n_future 4`, global batch 16, 24 epochs; LR never annealed (epoch 24 at 89 % of peak) | 101 / 107 | same | done; **B e22** = Stage-0 screen winner; B e01 = best by validation loss |
| **C1** | fine-tune of A | `n_future 1`, batch 16, 24 epochs, 8 h (7593272) | 101 / 107 | same | done |
| Stage-1 arms (A-based) | T-anneal (depth 4, LR annealed to 1e-6, `SCHED_TMAX=22`), `anneal_dryair` (+ dry-air fix, diagnostic), T-d8 (depth 8, 4 nodes) | multi-step | 101 | same | **cancelled 2026-09-28 before running**, to be re-based on F |
| **F** `f_nosoil_2n_b32_e43_warm` | A's recipe minus SOILWATER_10CM, TSOI_10CM; warm start from A sliced (surgical transfer val 0.01427 vs A's 0.01284, 7646690) | **single-step** | 99 / 105 | 2015–2044 | job **7660250**, `capacity`, 2 training nodes + 1 spare, 43 epochs, 12 h — **queued since 2026-09-25**, eligible 84.9 h, PBS estimate start 2026-09-30 06:51 UTC (capacity at its node limit) |
| **G** `g_ace2vars_*` | port G: the ACE2-EAMv3 variable set — F minus U10, RHREFHT, PSL, TMQ, Z3_l00…Z3_l17 (keeps RELHUM, TREFHT) | **single-step** (F's recipe) | 77 / 83 (76 state + PRECT) | **2020–2044** (valid 2045–47, test 2048–49), own stats | built, smoked, **not queued** (operator: "G runs after F", "don't queue it to capacity yet"); jesswan approved G + split (relayed) |

G launcher `submit_g_ace2vars.sh <QUEUE> <NODES 1|2|4> <EPOCHS=3+20k> <WALL> [scratch|warm]`:
`scratch` = from initialisation (default; "no checkpoint shares G's widths *and* its
normalization"); `warm` = A sliced to 83/77 with a fresh optimizer ("A learned under the 2015-2044
stats; that is a mismatch the fresh optimizer must absorb"). F-/G-based fine-tune arms:
`submit_subset_finetune_arm.sh <F|G> <arm>` (T-anneal / anneal_dryair / T-d8 / lrcheck; the
launcher refuses `anneal_dryair` for G — G has no TMQ, so `DryAirFix` cannot run on G).

## 2. What the debug jobs of 2026-09-29 measured

| job | what | result (PASS tokens in the job output) |
|---|---|---|
| 7668600 | build the 2020–2044 train view + stats | `TRAINVIEW_OK`; 36,500 samples; dropping 2015–19 moved means ≤ 1.42 % σ (`T_l00`), stds ≤ 2.9 % (`RELHUM_l03`), concentrated at the model top |
| 7668637 | G smoke, 2 nodes | `G_SMOKE_OK 3/3`: trains on CXI, EMA, 36,500 samples, `CHANNEL_SUBSET in=76 out=77`; `lrcheck` vars on G `LR_SCHEDULE_OK`. Rollout from a **40-step** smoke model: non-finite at step 264 — machinery only, says nothing about G's stability; the train→valid file handoff (step 368) is still unexercised on the view |
| 7669001 | spatial-parallel matrix, 2 nodes, batch 32, 60 steps × 2 epochs, A's 101-ch config | all 4 layouts on CXI. Step ms: **h1w1 225.2**, h2w2 417.9 (+85.6 %), h4w1 283.8 (+26.0 %), h2w4 616.3 (+173.7 %). h1w1 vs same-knob 1-node 365.4 ms ⇒ **2 nodes beat 1, 81.1 % per-GPU efficiency**; vs TCP-era 2-node 627.1 ms: −64.1 %. Epoch-2 train/valid loss: h1w1 0.1112/0.0915, h2w2 0.1098/0.0896, h4w1 0.1099/0.0879, **h2w4 0.1394/0.1043** (~26 % worse; prereg predicted h2w4 would hang — it trained). Unit suites `F_FINETUNE_TESTS_OK 6/6` |
| 7669103 | F2 gate: merged tree's full-width paths unchanged | `F2_EQUIV_OK tolerance=bitwise`: climate driver vs `rollout_one_ic` (A, 2048 f1092, K=56, 56/56 leads), and dry-air-off rollouts (B e22, A e243 × 1460 leads) vs the pre-fix tree, 20/20 NetCDF variables bitwise. Gate F2 of the F handoff closed |
| 7669129 | metric-writer fix + A's baseline on G's 77 channels | `WRITER_C77_ALL_OK 5/5`. (a) per-lead metric file was written by every model-parallel rank under HPAR/WPAR>1 (race seen in h2w4) — fixed, test red on pre-fix, 1-node h2w2 writes one file. (b) **A's K=56 read-out restricted to G's 77 channels** (table below) |

A's K=56 read-out (24 ICs, 2048–49; `…/prod1n_b32_sgdr_K56/scores/`):

| readout | 101 (published) | common 99 (F's set) | **common 77 (G's set)** |
|---|---|---|---|
| NRMSE126 / 336 | 0.5189 / 0.9696 | — / 0.9689 | 0.5848 / 1.0660 |
| ACC126 / 336 | 0.8669 / 0.5463 | 0.8665 / 0.5420 | 0.8269 / 0.4400 |
| VR336 | 1.0189 | 1.0189 | 1.0130 |
| worst-channel NRMSE336 | 43.18 | 43.18 | 1.371 |
| verdict (pre-registered bands, 2026-09-20 prereg) | DRIFT_FIRST | DRIFT_FIRST | AMBIGUOUS |

The 101 verdict was DRIFT_FIRST **only** through the worst-channel clause (43.18 > 3.0; NRMSE336
0.97 is below the 1.6 band). The worst channel is not named in the JSON; it is one of G's 22
non-soil drops. The 2026-09-24 analysis (CHANGELOG, "`Z3_l17`'s NRMSE 152 is a denominator
artefact") makes `Z3_l17` the likely candidate: truth anomaly 0.15–0.22 m vs 815.6 m topographic
std, real linear drift −0.35 m/day. Unconfirmed from the h5.

## 3. The stability record (primary: CHANGELOG 2026-09-24 entries)

- **A and every A checkpoint die in long rollouts**: non-finite at leads 490–664 (1 start), and
  8/8 at leads 317–849 across 8 starts (7649597). First channels past 3σ: model top (`V_l00`,
  `RELHUM_l04`).
- **Multi-step fine-tuning is the only thing shown to survive a year.** Stage-0 screen (7649647,
  38 checkpoints, 1 start): all 5 B checkpoints survive the year; C1 (depth 1) 2 of 5 at matched
  epochs; "depth 4 helps", "more epochs help", non-monotone. Second start (7649792): **B e22 rank 1
  at both starts**, PS drift +0.67 / −0.46 hPa/yr; other survivors drift 5–70 hPa/yr, and **PS
  drift is a fixed per-checkpoint property** (reproducible to 1–2 hPa across starts), invisible to
  validation loss. C1's survival is not reproducible across starts.
- **Multi-year** (jesswan's protocol, 7649597, 8 starts to 2049-12-31): **B e01** (best-val
  checkpoint) 7/8 non-finite at 2.5–5 yr, steady mass loss (PS −34 … −42 hPa at 1.25 yr, −226 …
  −316 at 4.25 yr), model top leaves 3σ first at ~1 yr. **B e22 has never been run multi-year.**
- **Dry-air conservation** (inference only, 7650652, n=1 start, 16 survivors): removes PS drift
  exactly but by the pre-registered rule **hurts** (survivors 16→14; the column cools/dries
  instead). "Post hoc is not the real test" — the trained arm `anneal_dryair` is. Diagnostic; any
  default is jesswan's.
- **Negativity** (7650442): PRECT negative over 10–17 % of the globe in every surviving rollout;
  RELHUM negatives at the model top (l00–l02). A clamp changes the model → jesswan.
- **Loss blindness** (tendency probe 7646192): with full-field z-score normalisation the slowest
  channels are `Z3` levels and `PS`; Spearman(r_c, bias²/MSE @336 h) = −0.647 — drift concentrates
  where the loss is blind. `temp_diff_normalization` is not a one-line flip (PRECT weight → 8e-4
  without a cap). Only `Z3_l17`, `Z3_l16` carry material bias² share at 336 h.

## 4. Compute and queue facts

- `capacity`: ≤ 168 h, **max_run 1 per project** (taking it blocks every other project member),
  `nodect` ≤ 4. `preemptable`: ≤ 72 h, 10 concurrent/project, start latency load-dependent.
  `debug`: ≤ 1 h, ≤ 2 nodes, one queued job per user. `debug-scaling` is held by an ACE2 LR sweep.
- Production step at 1 node: 473.2 ms (A). **Projection** (not measured): 473.2 × 225.2/365.4 ≈
  292 ms at 2 nodes ⇒ G scratch 243 epochs ≈ 24.7 h at 2 nodes (≈ 49 node-h with `SPARE=0`, 74 with
  the spare) vs ≈ 38.7 h at 1 node; F ≈ 5.2 h of its 12 h. G has 1,140 updates/epoch (A: 1,368).
- Stage-1 arm estimates (A-based, 2026-09-24): T-anneal 2 nodes × 12 h; T-d8 4 nodes ≈ 11–12 h
  (from a 1-node probe, linear scaling assumed); T-d16 **OOMs** at 1 sample/GPU (7650263) — needs
  activation recompute (unproven bitwise) or spatial sharding (≥ 8 nodes at h2w2 for 0.5 samples
  per GPU; `debug-scaling`).
- Sharding: a memory tool, not a speed tool here (+26 % to +174 % step time at equal work). `w=4`
  trains but its loss is off; no production use before a cross-layout equivalence check. makani's
  `flexible` checkpoint load can put an h1w1/A checkpoint into any layout; the launcher needs a
  `LOAD_CKPT` knob first.

## 5. Governance

Operator (rmehta1987) owns queues, node counts, walltime, and when G is released. **jesswan** owns
the science: variable sets, loss weighting, clamps, conservation defaults, making `Z3` diagnostic,
ensembles. Rules: every hot-path change gated on numerical equivalence; never loosen a tolerance;
G is quotable only against A's common-77 row; F only against common-99. Open decisions D1–D7 are
in the G/spatial handoff §3.

## 6. Questions

1. **What do these debug jobs mean for G?** (node count and walltime; scratch vs warm; what G's
   evaluation bar now is, given the common-77 re-take; what G can and cannot fix given it is
   single-step and has no TMQ; whether G should run at all before F's results.)
2. **What do they mean for fine-tuning?** (F-based Stage-1 arms: readiness after F2, speed after
   the 2-node result, depth-16 via sharding vs recompute, G-based arms later, dry-air on F only.)
3. **What are the best makani runs to do next?** A ranked, costed plan — queue, nodes, walltime,
   node-hours, the gate before each, what each answers, and who must decide. Candidates to weigh
   (not exhaustive): let F run and screen it; T-anneal-F → anneal_dryair-F → T-d8-F; G scratch vs G
   warm, before or after the F arms; a multi-year protocol run on **B e22** (inference, debug);
   the phase-2 cross-layout equivalence / T-d16 memory probe; waiting for jesswan on `Z3`,
   loss weighting, and clamps.
