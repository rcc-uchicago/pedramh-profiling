# Pre-registration — Stage-0 stability screen (2026-09-24)

Handoff: `polaris_makani_finetune_stability_handoff.md` §2. Written and committed **before** the first
screen job; gate S0b = this commit's time is earlier than that job's `stime`. Nothing here may be
edited after a screen has been read; a change is a new, dated section with its reason.

Tool: `polaris/polaris_climate_screen.pbs` + `polaris/climate_screen_summary.py` (S0a green: job
7649642, `CLIMATE_SCREEN_TEST_OK`, 15/15 screen tests + 25/25 driver tests).

## 1. What is run

- One rollout per checkpoint: IC 2044 frame 1092 (Oct 1 00:00), **1460 leads** (to 2045 frame 1092),
  chunk 40, the streaming climate driver, unchanged. Hand-off 2044→2045 at lead 368.
- Checkpoints: `polaris/climate_screen_stage0.list` (38), priority order = handoff §2c:
  C1 epochs 1–24; B epochs 1, 21–24; the 1-epoch batch-8 proxies (nf1 r1/r2, nf3 r1, nf4 r1/r2);
  A epochs 200, 220, 243; `nf4_crps_b4_r1`.
- Truth: `$MEMBER_ROOT/runs/makani_eval/screen_truth_2044f1092.npz` — area-weighted
  (`equiangular_weights`) global mean of `PS`, `T_l17`, `Z3_l10`, `TREFHT` from the pack's
  `fields_state`/`fields_diagnostic` at the valid time of each lead.

## 2. Metrics (per checkpoint, leads 1–1460) and the ranking rule

| metric | definition |
|---|---|
| `survived` | no non-finite value (`truncated_at_step == -1`) and all 1460 leads run |
| `median_cross_3sigma` | first lead at which the channel median of `anom_rms_sigma` exceeds 3 (−1 = never) |
| `n_past_3sigma` | channels whose `anom_rms_sigma` ever exceeds 3 |
| `ps_drift_hpa@600`, `@1460` | (model − truth) global-mean `PS`, hPa |
| `t17_drift_k@1460`, `z10_drift_m@1460`, `trefht_drift_k@1460` | (model − truth) global mean, K / m / K |
| `ps_drift_hpa@last` | the same at the last finite lead (only informative for non-survivors) |

`anom_rms_sigma` = area-weighted RMS of (prediction − `stats/time_means.npy`) in the run's
`global_stds`, exactly as the driver writes it.

**Ranking rule** (`climate_screen_summary.rank_key`, the only implementation):
1. survivors first;
2. then fewest `n_past_3sigma`;
3. then smallest `|ps_drift_hpa@1460|` (NaN, i.e. non-survivors, last);
4. tie-breaks: later `truncated_at_step` first, then label.

Every metric is reported, not only the rank. Validation loss is reported beside it where known and is
**never** used to rank.

## 3. How the table will be read (decided now)

n = 1 per checkpoint, one start date. Every statement below is qualified that way.

- **Depth (C1 vs B at matched epochs 1, 21, 22, 23, 24).** "Depth 4 helps stability" if at **≥ 4 of
  the 5** matched epochs B ranks above C1 by the rule. "Depth 4 hurts" if C1 ranks above B at ≥ 4 of 5.
  Otherwise "not separated at n=1". The proxies (nf1/nf3/nf4, 1 epoch, batch 8) are a secondary
  depth read at one epoch and do not override this.
- **Epochs (C1 epochs 1–24, fixed un-annealed schedule).** Compare the block of epochs 1–6 against
  19–24: survivors per block, median `n_past_3sigma`, median `|ps_drift_hpa@1460|` over survivors.
  "More epochs help" / "hurt" only if survivors differ by ≥ 2 **or**, with equal survivors, both medians
  move in the same direction. Otherwise "no epoch trend at n=1".
- **Mass.** `PS` drift is read as a separate question from blow-up. A checkpoint that survives with
  `|ps_drift_hpa@1460| > 10` is flagged as losing mass whatever its rank; that feeds §7 of the handoff
  (jesswan, conservation), not the ranking.
- **Small gaps.** Before trusting the order of the top three, re-screen them from 2044 frame 1156
  (Oct 17) with the same tool (`-v START_FRAME=1156`, own truth file). A gap is "small" if the top
  three differ neither in `survived` nor in `n_past_3sigma` and their `|ps_drift_hpa@1460|` span is
  < 2 hPa. If the second start reorders them, report both orders and name no single winner.
- **Epoch labels.** A row whose `epoch_check` is not `ok` is reported, and its label is not used for an
  epoch or depth statement until the stored epoch is understood.

## 4. Decision it feeds (operator's, not this document's)

Stage 1 (T-anneal, T-d8, T-d16) runs only if §3 says depth or epochs matter. If the depth read is
"hurts" or "not separated", the default (handoff §6.3) is to stop and report, and the question moves to
the mass/loss route for jesswan.

## A1. Addendum (2026-09-24, after screen 7649647, before the second-start job)

Why: §3's small-gap trigger did not fire (top-3 |PS drift| span 57 hPa), but B's PS drift swung
−58 / +0.7 / −68 / +4.6 hPa across epochs 21–24, so one start cannot rank neighbouring epochs. The
operator asked for a broader re-screen than §3's top three.

- Run: `polaris/climate_screen_rescreen_f1156.list` (B e01, e21–24; C1's 11 one-year survivors) from
  **2044 frame 1156** (Oct 17), 1460 leads, same tool, own truth `screen_truth_2044f1156.npz`.
- **Combined rule** (both starts, per checkpoint), decided now:
  1. survived **both** starts first;
  2. then fewest `n_past_3sigma` summed over the two starts;
  3. then smallest **mean** `|ps_drift_hpa@1460|` over the two starts.
  The per-start ranks from §2 are reported beside it.
- **Winner:** the combined rank-1 checkpoint is named only if it is also in the top three at each
  start separately. Otherwise report "no stable winner at n=2" and hand the protocol run the combined
  rank-1 checkpoint with that label.
- **Depth reread:** §3's depth rule is re-applied at the second start alone, then to the combined
  rank. The two readings are reported side by side, and neither overrides the other.
- Stage 1 goes ahead on the operator's instruction (2026-09-24) whatever this shows; the re-screen
  sets the reference those arms are compared against.

## A2. Addendum (2026-09-24): inference-only dry-air fix — DIAGNOSTIC

Operator asked for ACE2's dry-air conservation (`conserve_dry_air`) in the fine-tune; chose
"test at inference first, then train" and "dry-air mass". It changes what the model computes, so
it is a labelled diagnostic until jesswan signs off, and **no checkpoint is selected from it**.

- Gate first: `polaris_dryair_equiv.pbs` must print `DRYAIR_OFF_EQUIV_OK` (flag off = bitwise the
  pre-fix code) before any fix-on number is read.
- Run: `polaris_climate_screen.pbs -v DRY_AIR_FIX=on,CKPT_LIST=climate_screen_rescreen_f1156.list`
  from **2044 f1092**, i.e. the same 16 checkpoints and start as their fix-off rows in 7649647.
- **Sanity (the fix acts):** every fix-on member has `|dry_drift_hpa@1460| < 1` (or at its last
  lead). E3SM's own global dry-air mass is ~constant, so model − truth dry drift ≈ 0 by
  construction. A violation means the fix is not wired, and nothing else is read.
- **Effect, per checkpoint (fix-on vs its own fix-off row):** survived, `n_past_3sigma`, and
  `|ps_drift_hpa@1460|` (now ≈ g·TMQ drift).
  "Post-hoc fix helps stability" if, over the 16, **survivors increase by ≥ 2 and** median
  `n_past_3sigma` does not rise. "Hurts" if survivors fall by ≥ 2 or median `n_past_3sigma` rises by ≥ 2.
  Otherwise: "no stability effect; mass only". n = 1 start.
- The training arm (`anneal_dryair`) is compared against T-anneal by §2's rule once both exist;
  that comparison, not this one, is the fix's real test.

## A3. Addendum (2026-10-03, after F finished 43/43 epochs, before any F screen): raw vs EMA (D7)

Why: F is the soil-free fine-tune. It uses B's recipe and lineage, with 99 output channels instead of 101. It has finished training and saved a raw `best_ckpt`, an EMA `best_ckpt` and some intermediate snapshots. **D7** is the question of whether this lineage should prefer raw or EMA from here on.

D7 is a question about mass. A1 measured B's PS drift swinging −58 / +0.7 / −68 / +4.6 hPa across neighbouring epochs 21–24 at one start. Averaging weights across epochs is exactly what EMA does. The rule below therefore works in this order:
- survival is a gate, because a run that goes non-finite has no full-year PS number;
- PS drift decides among survivors;
- `n_past_3sigma` breaks ties and acts as a guard.

Gate S0b applies: this section and its list are committed before the `stime` of the first F screen job. Nothing here may be changed once an F row has been read.

### A3.1 Which checkpoints are run

The list `polaris/climate_screen_f_d7.list` is generated mechanically from F's run directory and committed together with this section.

- **Primary pair:** the raw `best_ckpt` and the EMA `best_ckpt`, exactly as training saved them. These are what each path would ship. Validation loss is reported beside them and is never used to rank them (§2).
- **Matched pair:** raw and EMA at **epoch 43**.
  - Purpose: it separates "averaging" from "a different epoch was picked". Without it, a best-epoch pick that happened to land on a low-drift epoch would look like an EMA effect.
  - If both `best_ckpt` files are already epoch 43, the matched pair *is* the primary pair and is not run twice.
  - If there is no EMA file for epoch 43, the matched pair is recorded as unavailable, and every D7 verdict carries the label "best-epoch confound uncontrolled".
- **Snapshots (context and noise floor only; never D7 inputs).** These follow §1's precedent for B, which used the first epoch plus the last kept epochs.
  - If the run kept **≤ 6** raw snapshots, all of them are run.
  - Otherwise, run the earliest kept snapshot, the last four kept before epoch 43, and the kept snapshot nearest the midpoint (if two are equally near, take the earlier).
  - EMA snapshots are run only at those same epochs, and only if they exist.

**Gates.** These are checked in the same job, before the first rollout. No number is read until all of them pass.
- Every row has `epoch_check` = `ok` (§3, "Epoch labels"). If a primary-pair or matched-pair row is not `ok`, D7 is not read until the stored epoch is understood.
- Raw and EMA files at the same epoch both load into the same 99-channel model, and their weights are **not** bitwise equal. If they are equal, the EMA was not saved and D7 is not read.
- F's `stats/time_means.npy` and `global_stds` both have 99 channels, in the order of F's channel map.

### A3.2 Run

- `polaris_climate_screen.pbs` is used unchanged, with `-v CKPT_LIST=climate_screen_f_d7.list` and **`DRY_AIR_FIX` off**. Each rollout uses 1460 leads, chunk 40 and the streaming driver.
- Every row is run from **both** starts: 2044 f1092 and 2044 f1156 (`-v START_FRAME=1156`). This is A1's two-start design from the outset.
- Why the fix is off: D7 asks which checkpoint conserves mass *without help*. A2's fix would erase the very quantity being compared, and it remains diagnostic-only. If the summary writes `dry_drift_hpa` for fix-off rows, it is reported beside PS. It is not a D7 input.

### A3.3 Truth and statistics for 99 channels

- **Truth is reused unchanged, subject to two checks.** `screen_truth_2044f1092.npz` and `screen_truth_2044f1156.npz` hold only pack-derived, area-weighted global means of `PS`, `T_l17`, `Z3_l10` and `TREFHT` (§1). They do not depend on the model's channel set. Before the first rollout, the job checks that:
  - (a) all four variables are present **by name** in F's channel map;
  - (b) the driver pairs truth with model output **by name, not by output index**. If it used the index, the 101→99 shift would silently pair the wrong channel.
- **If either check fails,** the truth is rebuilt from the pack using F's channel map, as `screen_truth_2044f1092_c99.npz` and `..._f1156_c99.npz`. The rebuilt files must match the old ones **bitwise** on all four variables. A mismatch is a bug, and nothing is read until it is found.
- **If `PS` is missing from F,** D7 is not read, and no other variable stands in for it. If one of the other three is missing, its column is NaN.
- **Statistics are F's own.** `anom_rms_sigma` uses F's 99-channel `time_means` and `global_stds`, which is what the driver loads by default. B's stats are never used.
  - Raw-vs-EMA `n_past_3sigma` counts are therefore directly comparable.
  - Any F-vs-B line in the write-up uses B's count recomputed over the 99 shared channels, and is labelled as such. None of those lines feed D7.

### A3.4 D7 decision rule (decided now)

**Definitions,** per candidate and per start *s*:
- `survived_s` is §2's `survived` at that start.
- The mass score is **M_s = max(|`ps_drift_hpa@600`|, |`ps_drift_hpa@1460`|)**, defined for survivors only.
  - Both columns already exist in §2, so no new metric is introduced.
  - Taking the maximum stops an endpoint that happens to pass through zero from scoring as "conserving". B e22's +0.7 hPa, between −58 and −68, is exactly that pattern.
- **Shared starts** are the starts that both candidates survived.

**Step 1: survival gate.**
- If one candidate survived more starts, it **wins D7, whatever its PS drift**. The loser's `ps_drift_hpa@last` and `truncated_at_step` are reported. They are never compared with a survivor's M, because a run that stopped early has had less time to drift.
- If both survived 0 starts, D7 = "**neither; F not shippable as screened**". Both `truncated_at_step` values are reported.
- If both survived exactly one start, but not the same one, D7 = "not separated (no shared start)".
- If both survived the same single start, Steps 2–3 run on that start alone, and the verdict is labelled **n = 1**.
- If both survived both starts, Steps 2–3 run on both.

**Step 2: mass.** Use the mean M over the shared starts. The candidate with the lower mean M wins only if **both** of these hold:
- the gap is **≥ max(2 hPa, 0.2 × the larger of the two mean M values)**. The 2 hPa floor is §3's small-gap span. The 20 % term stops a gap such as 40 vs 42 hPa from counting as separation when neighbouring epochs differ by tens of hPa;
- it has the lower M at **every** shared start.

Otherwise the candidates are mass-tied, and Step 3 applies. The signs of `ps_drift_hpa@1460` are reported but do not decide.

**Step 3: stability tie-break** (only if Step 2 ties). Use `n_past_3sigma` summed over the shared starts. The candidate with fewer channels wins only if:
- the summed gap is **≥ 2** (A2's threshold), **and**
- its count is lower or equal at every shared start.

Otherwise D7 = "tie". `median_cross_3sigma` is reported beside the result and does not decide it.

**Stability guard on a mass win.** Suppose Step 2 names X. If X's summed `n_past_3sigma` is **≥ 2 higher** than the other candidate's, and higher or equal at every shared start, the verdict reads "**D7 = X on mass, stability-split**". This is not adopted until the operator rules. Until then, the default below applies.

**Matched-pair check.** The epoch-43 pair is scored with the same Steps 1–3 and the same guard.
- If it agrees with the primary verdict, or ties, the primary verdict stands.
- If it **reverses** the primary verdict, D7 = "not separated (best-epoch confound)", and both verdicts are reported.
- If the matched pair is unavailable, the label from §A3.1 applies.

**Default when D7 is tie, not separated, or a pending split: raw.** Adopting EMA is the change, so EMA has to show an effect. The default is a convention, not a finding, and that label is carried forward.

**Mass flag (§3, unchanged).** Any survivor with `|ps_drift_hpa@1460| > 10` at either start is flagged as losing mass and goes to handoff §7 (jesswan), whether or not it won D7. The flag does not decide D7; Step 2 does. If the D7 winner is flagged, the verdict reads "D7 = X, mass-flagged".

**Moderator's rulings** (A = stability-first, B = mass-first):
- *What decides among survivors: **B**.* D7 exists because of the PS swing, and the task asks for PS to decide. In A's rule, PS is effectively unreachable: A's per-start check uses §2's `rank_key`, which puts `n_past_3sigma` ahead of PS, so any count difference at either start blocks the PS winner. A's noise objection is valid, but it is answered by requiring two starts, agreement at every start and a noise-scaled gap, not by demoting PS. §3's rule that "mass feeds jesswan, not the ranking" governs §2's stability ranking, not a decision whose subject is mass.
- *B's "flag class" step (unflagged beats flagged): **dropped**.* A hard cut at 10 hPa would let 9.5 vs 10.5 hPa decide D7, a 1 hPa gap well below the noise B itself argues for. Step 2's magnitude rule already handles B's −45 vs −4 hPa example (a gap of 41 against a threshold of 9). The flag stays as §3 defines it: it is reported, and it is not a decider.
- *Split survival on different starts: **B**.* A sends this case to a PS step over shared starts, but there are none, so the step is undefined.
- *Endpoint |@1460| vs M: **B**.* An endpoint alone rewards a lucky zero-crossing. The data already contains one (B e22).
- *Blow-up can't be repaired: **A's concern is kept** through the survival gate (Step 1 decides regardless of PS) and the stability guard.* B's point that `conserve_dry_air` is diagnostic-only and does not correct g·TMQ means "fix mass later" cannot justify demoting PS.

### A3.5 Snapshots: noise floor and context

- **Raw mass noise floor.** At each start, the M values of the raw snapshots give the per-epoch mass spread. Their range and median are reported per start. This is the first measurement of A1's swing in the F lineage.
- **"EMA damps per-epoch mass noise"** holds only if EMA `best_ckpt`'s M is **below the median raw-snapshot M at both starts**. Otherwise the report says "no damping shown at n=2". This is a context read and does not override §A3.4.
- **Ranking and selection.** Snapshots are ranked by A1's combined rule, and every metric is reported. They are **never selected automatically**.
  - If a snapshot beats both D7 candidates under Step 2 (or under Step 1), the report says "snapshot outranks". What to do about it is the operator's call.
  - An epoch-trend statement for F uses §3's "Epochs" thresholds, comparing the earliest kept snapshot with the last four. It is labelled n = 2 starts.

### A3.6 What it feeds (the operator's decision, not this document's)

D7 decides two things:
- which F checkpoint goes to climate-fidelity scoring for jesswan;
- the default checkpoint type (raw or EMA) for later fine-tunes in this lineage: the B continuation, and G when it is released.

A verdict of "tie", "not separated" or "stability-split (pending)" means raw by default, with that label carried forward. It is not a finding that EMA does not help.

### Threats specific to A3

- **Small sample.** There are n = 2 starts of one year each, and a single F training run. The EMA effect is confounded with this run's EMA decay and its best-epoch selection. The matched pair is the only control.
- **PS mixes dry air and water.** With the fix off, PS drift = dry-air drift + g·TMQ drift, so a mass win could come from the water term. This is why `dry_drift_hpa`, if emitted, is reported.
- **M samples only two leads, 600 and 1460.** A drift that peaks between them and recovers is missed.
- **`n_past_3sigma` has unmeasured start-to-start noise.** It is measured against the all-year `time_means` (§5), so the seasonal cycle is not removed. That effect is common to raw and EMA, but it is the reason Step 3 also requires agreement at every start.
- **Cross-lineage comparisons.** F has 99 channels and B has 101. Comparisons across the two need the 99-channel recount, and none of them feed D7.

**Provenance.** Drafted by a two-analyst debate (Debater A: stability-first; Debater B:
mass-conservation-first) and reconciled by a moderator, three independent `claude-opus-5-5`
processes, run directly in-session on the login node per operator override (2026-10-03). Reviewed
and committed by the operator in place of an unassisted human draft — the usual author of a
addendum in this document. Subject to the same rule as every other section here: nothing may be
edited after an F screen has been read; a change is a new, dated section with its reason.

## 5. Threats

- **One start date, one year.** Chaotic sensitivity can reorder close checkpoints; §3's re-screen rule
  is the only mitigation in Stage 0.
- **Layout confound for C1 vs B:** C1 trained on 1 node × local 4, B on 2 nodes × local 2 (same global
  batch). Not expected to matter; not controlled.
- **B's epochs 2–20 do not exist on disk** (only 21–24 kept), so B's epoch trend is not measurable.
- **File → epoch mapping** is from write order; the `epoch_check` column verifies it against the epoch
  stored inside each checkpoint.
- **Oct→Oct window** covers one full seasonal cycle; truth subtraction removes it from drift, not from
  `anom_rms_sigma` (which is vs the all-year `time_means`, as in G4).
