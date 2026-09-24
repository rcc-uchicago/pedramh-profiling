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
