# makani handoff — port the ACE2 mechanisms, cheapest and least-gated first

*Written 2026-09-23.* Acts on `ace2_vs_makani_differences.md` §8 (PR #15). That document
says **what** differs and cites every row; this one says **what to do about it, in what
order, and what would make each answer wrong.**

Read in this order: **`ace2_vs_makani_differences.md`** (the differences, config-verified),
then **`makani_sfno/docs/2026-09-10_ace2_comparison_the_corrector.md`** (the mechanism),
then **`makani_sfno/docs/2026-09-10_rollout_spectra_and_drift.md` §2-§3** (the measurement
this is all aimed at), then this file. `$MEMBER_ROOT` =
`/eagle/projects/lighthouse-uchicago/members/mehta5`. **PASS is the log token, never `rc`**
(CLAUDE.md #14).

---

## 0. The five ports, in the order to do them

| # | port | cost | science gate | what it decides |
|---|---|---|---|---|
| **A** | **positivity clamp at inference** on the physically non-negative channels — ⚠ **demoted below F**: its headline target `SOILWATER_10CM` is being removed from the state, leaving `PRECT`/`RHREFHT`/humidity, none of which has been shown to go unphysical | **one job, no retraining** | none *as a diagnostic* | whether the reservoir-drift chain is the divergence cause — **now best folded into the post-F retrain** |
| **B** | **`time_diff_stds.npy`** → `temp_diff_normalization` — ⚠ **now known to need a weight cap and a `PRECT` decision, so it is code + config, not config** | converter pass (done in probe form) + a small code change + one training arm | **loss change ⇒ sign-off** | ✅ the *mechanism* is measured (job 7646192, §3); what is open is how to enable it without deleting precipitation |
| **D** | **EMA** (`ema.enabled: true`) | config-only, rides along with any arm | lowest — changes which weights are selected, not what the model computes | free variance reduction on the shipped weights |
| **C** | **per-channel loss weights** | one training arm | loss change ⇒ sign-off | whether `constant` is costing us |
| **E** | **move fluxes/precipitation out of the feedback loop** | — | **science decision, not a port** | raise with jesswan; do not implement unasked |
| **F** | 🔵 **DECIDED — drop `SOILWATER_10CM` and `TSOI_10CM` from the prognostic set and retrain** | config change + a retrain (~46 node-h) | **operator decision taken 2026-09-23 (rmehta1987)** | gates the next training run; see §6a |

**Order, revised 2026-09-23 when F was decided: F → B(statistic) → D → C → A → E.**
**F gates everything that involves training**, because it changes the channel contract and
therefore every baseline number: doing B, C or D against the old contract produces results
that have to be redone. B's *statistic* and D are cheap and can be prepared while F's
retrain is queued, so long as the arms themselves run on the post-F contract. A is demoted
because F removes its main target. E is not ours.

---

## 1. What this rests on — do not re-derive any of it

| fact | source |
|---|---|
| `SOILWATER_10CM` global mean falls **6.36 % in 14 days**, decreasing on **76 % of steps**, linear trend extrapolating to **−59 % by step 500** — the step at which the rollout diverges | `2026-09-10_rollout_spectra_and_drift.md` §2 |
| makani has **zero** `corrector` / `conserve_dry_air` / `force_positive` / `moisture_budget` machinery anywhere | `2026-09-10_ace2_comparison_the_corrector.md` §3 |
| ACE2 runs **7300 steps (5 years)** stably at the **same 2-step training window** | `config_polaris.yaml`; same doc §1-§2 |
| the loss weight under `temp_diff_normalization` is **σ_c / δ_c** (climatological std ÷ per-step-change std), denominator clamped at `1e-4` | `makani/utils/loss.py:181-184`, quoted in the drift doc §3 |
| our pack has **no `time_diff_stds.npy`** — the converter never computed it, so the switch was never *available* | drift doc §3 |
| upstream FCN3 runs `temp_diff_normalization: True` **at every stage** and `channel_weights: auto` | `2026-09-10_fcn3_recipe_vs_ours.md` |
| `Z3_l17` has **0.271 m** of temporal variability yet the model makes 10.22 m of anomaly by 336 h, half of it a systematic **−5.15 m** drift | `2026-09-20_k56_readout_prereg.md` |
| the LR ceiling is **(2e-3, 3e-3]** and does not move with batch size; **β₂ 0.95** (0.999 is backwards); grad clip **1.0 > 32** | `makani_bench_report.md` §7e |
| the pre-registered significance bar is **2σ₀ = 1.92 %** — keep quoting the conservative published figure, not the 0.88 % recompute | `2026-09-15_nfuture_ladder_d1_and_replication.md` §1 |
| the lagged ensemble is **closed** — −31.7 % RMSE, 0/101 channels improved, monotone in ensemble size | `2026-09-21_lagged_ensemble_result.md` |
| a distributional loss is **not indicated** — VR is 1.019 at 336 h, never leaves 0.99–1.02 | `2026-09-20_k56_readout_prereg.md` |
| **`r_c = δ_c/σ_c` spans 1.31e4 across the 101 channels** (0.000105 → 1.375) | **job 7646192**, §3 |
| **Spearman(`r_c`, bias²/MSE @336 h) = −0.647** — the drift really is concentrated where the loss is blind | **job 7646192**, §3 |
| the train→test warming shift does **not** explain the 336 h bias — Spearman +0.079 | **job 7646192**, §3 |
| the slow channels are the **`Z3` levels and `PS`**, *not* the land reservoirs as a class | **job 7646192**, §3 |

---

## 2. Port A — the positivity clamp. Do this first.

**The claim to test.** A slow water reservoir walking monotonically toward zero is exactly
what ACE2's `force_positive_names` (16 variables) exists to prevent. Clamp the physically
non-negative channels after each step and re-run to divergence. **If the blow-up step moves
substantially, the causal chain is confirmed and the fix is scoped. If it does not, the
divergence is something else and this whole branch is misaimed** — which is equally worth
knowing for one job.

### 🐛 First, a correction to the doc that proposed this

`2026-09-10_ace2_comparison_the_corrector.md` §6 says to test it "in `longroll.py`".
**There is no `longroll.py` anywhere in this repo** — verified 2026-09-23 by a recursive
glob over the whole tree. Before writing any code, find out what that doc meant: it is
either under `$MEMBER_ROOT` outside the repo, or it was never written. **Do not create a
new long-rollout harness if one already exists**; the ones that certainly do exist are

* `makani_sfno/polaris/eval_inference.py` + `polaris/polaris_eval_inference.pbs` — the K=56 sweep, `EVAL_INFERENCE_OK`
* `makani_sfno/src/sfno_inference/rollout_driver.py` — `rollout_one_ic`, the loop itself

### Where the clamp goes, and the constraint on it

⚠ **`src/sfno_inference/` is a `git subtree` shared with the Stampede3 `eval-sfno-own`
path** (CLAUDE.md #5, #7). Every change must **generalise**, never "make it work for E3SM":

* the clamped channel set must come from **config**, not a hard-coded list;
* it must **default to off**, so every existing config is byte-unchanged;
* keep the edit **minimal and contiguous** — subtree pulls conflict on scattered hunks.

A `force_positive_names`-shaped key (name it after ACE2's, so the provenance is obvious)
read from `eval_params`, applied after each step, is the whole change.

### Candidate channels

`SOILWATER_10CM`, `PRECT`, `RHREFHT`, and the humidity levels. ⚠ **Which channels are
non-negative by physics is science-owned.** Getting this list wrong clamps something that
legitimately goes negative and silently changes the answer. Confirm the list before running
anything that will be quoted.

### What to report

Divergence step with the clamp vs without, on the **same** initial condition, plus the
`SOILWATER_10CM` trajectory in both. One number is the result: does the blow-up move?

⚠ A clamp is a **diagnostic**, not a shipped change. If it works, shipping it is a separate
decision with jesswan — and note ACE2's corrector does two more things (dry-air-mass
conservation, moisture-budget correction) that **may not port at all**: our 101-channel
contract is not obviously budget-closable the way ACE2's `specific_total_water_*` +
`PRATEsfc` + advective-tendency set is.

---

## 3. Port B — `time_diff_stds.npy`. Compute the statistic before deciding anything.

### What it is

One number per channel: the standard deviation of `x(t+6h) − x(t)` over the **training**
split (2015–2044). Same role as `global_stds.npy`, but of the one-step *change* rather than
the field. makani then builds the loss weight `σ_c / δ_c`.

### Why it should help, stated so it can be falsified

Let `r_c = δ_c / σ_c` — the fraction of a channel's variability in play in one step. Under
z-score normalization with `channel_weights: constant`, error is measured in units of `σ_c`,
so a model that predicts *no change at all* for a slow channel earns an error of order
`r_c` — near zero. A per-step bias `b_c = ε·δ_c` accumulates to `N·ε·r_c` climatological
standard deviations after `N` steps. With `r_c ≈ 1` the one-step loss already punished it;
with `r_c ≈ 1e-3` you reach `0.5σ` of pure drift by `N = 500` having never paid for it in
training. `temp_diff_normalization` multiplies the weight by `1/r_c` and removes exactly
that blind spot. It is also the consistent choice given our config already sets
`target: "tendency"` — the network emits a change, so the loss should score the change
against the change's own scale.

### ✅ MEASURED 2026-09-23 — job **7646192**, `TENDENCY_PROBE_OK`, 60 s on one debug node

The argument above was **an argument**, so it was tested before being acted on:
`polaris/tendency_norm_probe.py` + `polaris/polaris_tendency_norm_probe.pbs`, five
hypotheses with thresholds fixed in the script before any number existed.
Outputs: `$MEMBER_ROOT/runs/makani_probe/tendency_norm/7646192/`.

| # | hypothesis | threshold | measured | verdict |
|---|---|---|---|---|
| H1 | the mechanism is in the **installed** makani, shaped `σ_c/δ_c` with a `1e-4` clamp | present & same shape | `makani/utils/loss.py:152`, **verbatim as quoted** | ✅ **SUPPORTED** |
| H2 | `r_c` spans ≥ 1.5 orders of magnitude | max/min ≥ 30 | **1.31e4** (0.000105 → 1.375) | ✅ **SUPPORTED**, by 400× |
| H3 | the **land reservoirs** are the slow channels | both in bottom quartile | `SOILWATER_10CM` 5th pct ✓, **`TSOI_10CM` 39th pct ✗** | 🔴 **FALSIFIED** |
| H4 | the drift is concentrated where the loss is blind | Spearman ≤ −0.30 | **−0.647** | ✅ **SUPPORTED** |
| H5 | the train→test warming shift explains it better | \|ρ₅\| > \|ρ₄\| | **+0.079** vs 0.647 | ✅ refuted — H4 stands |

**The mechanism survives; the framing in this section did not.** The blind spot is
dominated by the **geopotential levels and surface pressure**, not by land reservoirs:
five of the ten slowest channels are `Z3_l17, l16, l15, l14, l13`, with `SOILWATER_10CM`
sixth and `PS` seventh. `TSOI_10CM` is unremarkable (r = 0.135). Rewrite any claim about
"slow reservoir variables as a class" — soil *moisture* qualifies, soil *temperature* does
not.

And the ranking lines up channel by channel with where the drift actually is:

| channel | `r_c` | weight `1/r` if enabled | bias²/MSE @336 h |
|---|---|---|---|
| `Z3_l17` | 0.000105 | **9518** | **0.253** |
| `Z3_l16` | 0.00369 | 271 | **0.139** |
| `Z3_l15` | 0.00711 | 141 | 0.038 |
| `SOILWATER_10CM` | 0.0229 | 43.7 | 0.018 |
| `PS` | 0.0248 | 40.3 | 0.002 |
| `PRECT` | 0.957 | 1.04 | ~0 |

### 🔴 Two traps the probe found, neither of which was anticipated above

**1. The `1e-4` clamp is in PHYSICAL units, and it would delete precipitation.**
`get_time_diff_stds` is a bare `np.load(params.time_diff_stds_path)` — **no internal
normalization** (verified). So the clamp compares a raw physical `δ_c` against `1e-4`.
`PRECT`'s tendency std is **7.95e-08 m/s**, far below it, so the clamp fires and the
weight becomes `σ/1e-4 = 8.3e-04` instead of the correct **1.044** — a factor of
**0.0008**. Precipitation would be effectively removed from the loss, with no error and
no warning. It is the only channel affected, and it is the one channel a weather model is
most often judged on.

**2. Enabling it as-is hands one near-constant channel the whole loss.** The weight range
would be **0.727 → 9518, a 1.3e4 span**, with `Z3_l17` alone at 9518 — a channel with
**0.271 m** of temporal variability. ACE2's hand-picked weights span 40× by comparison.
A loss dominated 9518-to-1 by a terrain-following geopotential level is not the upstream
recipe; it is a different model.

⇒ **`temp_diff_normalization: True` is NOT a one-line flip on this pack.** It needs a cap
on the weight (or equivalently a floor on `r_c`, in *dimensionless* units, not physical
ones) and an explicit decision about `PRECT`. That makes port B a small code change plus a
config change, not a config change — reprice it accordingly.

### ⚠ What the probe did NOT establish

* **`δ_c` is from 4 × 60 samples of `train/2044.h5`**, not the full 30-year split. The
  *ordering* is robust; the values are a sample, and the real `time_diff_stds.npy` must be
  computed over the whole training split.
* ~~The identity of `scale`~~ — **closed 2026-09-23 by reading the installed file.**
  `loss.py:99` is `bias, scale = get_data_normalization(params)` and `:106` indexes it
  `[:, params.out_channels, ...]`, so under `normalization: "zscore"` **`scale` is exactly
  `global_stds.npy`** ⇒ the multiplier is `σ_c / clamp(δ_c, 1e-4)` = `1/r_c`, and the
  weight numbers above stand. `:160-166` then *multiplies* the channel weights by it, so
  `channel_weights: "constant"` and this switch compose rather than conflict.
  ⚠ Both `scale` and `time_diff_scale` are indexed by `params.out_channels`, so
  `time_diff_stds.npy` must be **full-length and in the same channel order as
  `global_stds.npy`** — a positional mismatch mis-weights everything silently.
* The 101 channels are **not independent** (18 levels × a handful of variables), so
  Spearman's effective sample size is far below 101. ρ = −0.647 is large, but do not attach
  an n=101 p-value to it.
* `bias²/MSE` is read at a **single lead** (336 h).

### Step 1 — compute the statistic properly, and **inspect**, do not enable

Add the accumulator to `makani_sfno/polaris/convert_e3sm_to_makani_alldata.py`'s existing
stats pass (it already computes `global_means/stds`, `time_means` and the three forcing
equivalents; this is a fourth over consecutive pairs **within** each year file — never
across a file boundary, since sample index 5 is a different time in `2048.h5` than in
`2049.h5`).

⚠ **Use the converter's `--stats-only` path, and respect its completeness guard.** It
verifies the packed train years are complete first, because subset statistics are the
silent-wrong-normalization bug that guard exists to stop.

Then **publish the table before touching the config**:

| report | why it is the finding, not a side effect |
|---|---|
| `r_c = δ_c / σ_c` for all 101 channels, sorted | the spread across channels *is* the claim that slow channels are under-penalised |
| the implied weight ratio `max(1/r_c) / min(1/r_c)` | if one channel would get 1000× the weight of the rest, the loss **is** that channel |
| any channel with `δ_c` at or under the `1e-4` clamp | the clamp is load-bearing; a channel hitting it gets an arbitrary weight |
| `Z3_l17` explicitly | 0.271 m of temporal variability makes it the prime candidate for a degenerate `δ_c` |

That table is a result on its own and costs one CPU job. It may also *refute* the
hypothesis — if `r_c` turns out to be within an order of magnitude across all 101 channels,
the under-penalisation story is wrong and port C is the better spend.

### Step 2 — verify the wiring before running a training arm

`temp_diff_normalization` lives in **installed makani**, not in this repo (the fork is
`src/sfno_training/`; makani itself is in the conda env). Before the arm:

1. ✅ **the key is `params.time_diff_stds_path`** (verified, job 7646192) and
   `get_time_diff_stds` raises `ValueError` if the attribute is absent — so a missing key
   fails loud, which is the one part of this that cannot go silently wrong;
2. the array must be **full-length and in `global_stds.npy`'s channel order** — both it and
   `scale` are indexed by `params.out_channels`. A positional mismatch mis-weights every
   channel with no error, the same failure class the lagged scorer's
   `CHANNEL_ORDER_MISMATCH` guard exists for. Assert it against `channel_names`;
3. **decide the cap and the `PRECT` handling first** (the two traps above) — enabling
   before that is a known-wrong run, not an experiment;
4. run the smoke (`e3sm_alldata_smoke.yaml`) with the switch on and **print the realised
   per-channel weight vector**, not just the loss. The loss changing proves the key
   arrived; only the weight vector proves it arrived *correctly*.

### Step 3 — the arm

One training arm, warm-started from `prod1n_b32_sgdr`, **identical in every other knob**,
scored by the §7 protocol. ⚠ This changes the loss ⇒ **jesswan's sign-off** (CLAUDE.md,
division of labor). The statistic in step 1 is the argument to bring her.

---

## 4. Port D — EMA. Config-only, and the machinery already exists.

**It is already implemented in the fork** — `src/sfno_training/trainer/ema.py` (Karras-style
warmup `decay_t = min(decay_max, (1+t)/(10+t))`, complex-dtype-aware shadows for the SFNO
spectral weights), wired into `PlasimTrainer` with a second EMA validation pass and a
`best_ckpt_ema_mp{mp_rank}.tar` export. Knobs, with defaults:

| key | default | note |
|---|---|---|
| `ema.enabled` | **`false`** | this is why the production run had none |
| `ema.decay` | `0.999` | e-folding ≈ 1000 updates |
| `ema_validation_period` | `1` | asserted `>= 1` |

⚠ **`ema.enabled: true` is scoped to legacy save AND load** (`plasim_trainer.py:~421`);
flexible-mode EMA is explicitly deferred. Our config is `save_checkpoint: "legacy"`, so this
is satisfied — **assert it in the log rather than assuming**.

**There is no `ema:` block in `e3sm_alldata_full.yaml` and no EMA passthrough in
`polaris_makani_multinode_scaling.pbs`.** Both need adding; that is the whole port.

**Sizing `decay` matters more than enabling it.** At 1368 updates/epoch, `0.999` averages
over ≈ 0.7 epoch — a short window. ACE2 carries `0.999` at ~12,234 updates/epoch, i.e.
**0.08 epoch**, which is nearly the instantaneous weights; do not copy the number without
the update rate. Pick the decay from a target averaging window in *epochs*, and say which.

Why it is worth doing: it is weight-space averaging — the free version of a checkpoint
ensemble, one model at inference instead of M. ⚠ It does **not** carry the
`MSE(M) = b̄² + σ̄²[ρ̄ + (1−ρ̄)/M]` guarantee that averaging *forecasts* has; averaging
weights only behaves like averaging forecasts when the checkpoints lie in one connected
low-loss basin. Treat the gain as empirical and measure it.

---

## 5. Port C — per-channel loss weights

Gated on B's `r_c` table, because that table tells you whether the weights are the problem
and which direction to move them. Upstream FCN3 uses `channel_weights: auto`; **verify what
values makani actually accepts** before assuming `auto` exists on our version — the current
config sets `constant` and nothing in this repo has ever run anything else.

⚠ Do **not** hand-pick weights the way ACE2 does (17 explicit entries spanning 0.25–10).
Those are ai2's science choices for their variable set; ours has 101 different channels and
picking a headline subset is jesswan's call, not the porter's. `auto` (a rule) is portable;
a dictionary (a set of decisions) is not.

---

## 6. Port E — not a port

ACE2 keeps **12** diagnostics out of the feedback loop (precipitation, all six radiative
fluxes, latent/sensible heat, `TMP850`, `h500`, advective water tendency). We keep **1**
(`PRECT`) and feed back 100 channels, including two unforced land reservoirs. Every extra
fed-back channel is an error-amplification path.

**This is a question for the science owner, not a change to make.** Bring it together with
the `Z3_l17` question already queued for her — whether a near-constant near-surface
terrain-following geopotential level should be prognostic at all.

---

## 6a. Port F — DROP `SOILWATER_10CM` AND `TSOI_10CM`, THEN RETRAIN 🔵 DECIDED

**Operator decision, rmehta1987, 2026-09-23.** Taken on the evidence in jobs 7646383 and
7646391, recorded here so the reasoning survives the decision.

### Why

| channel | measured | jobs |
|---|---|---|
| `SOILWATER_10CM` | truth over the filled region ≈ 0; the model predicts mean **−0.856** there — **negative soil water** — and **12.5 %** of its lat-weighted 336 h MSE comes from cells whose answer is a known constant. 71.7 % of the grid is exactly constant in time. Not redundant with `PRECT` (median per-cell corr **0.160**) — it is an integrator, so it cannot be reconstructed from what remains | 7646391 |
| `TSOI_10CM` | truth over ocean is a 270 K constant; the model predicts **std 1.77 K, max 303.7 K** there, and **21.8 %** of its 336 h MSE is outside the valid region. 61.4 % of the grid constant in time | 7646383 |

Both are **fed-back state channels**, so those fabricated values re-enter the network over
60–70 % of the globe at every one of 56 steps. Neither has a drift problem worth chasing
(systematic error share 0.018 and 0.0026, against 0.253 for `Z3_l17`) — **the problem is the
fill region, not the physics.**

⚠ **State the consequence plainly rather than burying it:** this removes the model's
land-surface memory entirely, so it can no longer represent soil-moisture → evaporation →
precipitation feedback. **ACE2's 38 prognostic channels contain no soil variables either**
(`ACE2_retrain/config_polaris.yaml`, verified), so this moves us toward the reference
contract rather than away from it — but it is a science consequence, not a neutral cleanup,
and it should be stated that way to anyone comparing against a land-coupled model.

### How — and it is almost certainly NOT a repack

✅ **Do not regenerate the 1.4 TB pack, and do not delete anything from it.** Keeping the
data makes the decision reversible at zero cost. The subsetting machinery already exists:

* `src/sfno_training/data/plasim_forcing_dataset.py:308` reads with
  `channels=self.in_channels`, and `:320` builds the target as
  `np.asarray(self.out_channels)[: self.n_out_channels - 1]` — so the loader already honours
  an index subset;
* makani indexes the normalization arrays the same way — `loss.py:106` is
  `scale[:, params.out_channels, ...]` — so **`global_means/stds` and `time_means` do not
  need regenerating either.** They get indexed.

⚠ **`:320` assumes the diagnostic channel is the LAST entry of `out_channels`.** `PRECT` is
index 100; keep it last after the subset or the target tensor silently takes the wrong
channel.

**Step 1, before writing anything: find out how `in_channels` / `out_channels` are actually
populated on this path.** They are set neither in `e3sm_alldata_full.yaml` nor in
`train_plasim.py` (checked), so makani is deriving them — read `driver.py` and confirm.
That answer decides the shape of the change:

| if… | then |
|---|---|
| they are settable config keys | **keep `channel_names` at 101** and express the removal as index lists. The pack, the stats and the converter↔trainer gate are all untouched — this is the clean route |
| they are derived from `len(channel_names)` | editing `channel_names` to 99 **will trip the converter↔trainer gate**, which asserts it equals `TARGET_CHANNELS` exactly and aborts on drift (that gate exists to stop same-width name drift mapping channels to the wrong slots). Teach the gate about the subset rather than weakening it |

Then set `n_state_channels: 100` → **98**; leave `n_diagnostic_channels: 1` (the config's
own note says it must stay 1 — `plasim_forcing_dataset.py:320`). Inputs go **107 → 105**,
outputs **101 → 99**; the 7 forcings are unchanged.

### The retrain, and one thing it cannot do

**The existing checkpoint cannot be warm-started as-is.** The encoder's input width and the
decoder's output width both change, so `restore_from_checkpoint(..., strict=True)` will
refuse. Two routes:

1. **From scratch** — the measured cost of the production run is **243 epochs, 332,424
   updates, 46.3 node-hours** on 1 node (`capacity`). This is what "retrain again" means and
   it is the safe option.
2. **Surgical transfer** (cheaper, untested here): the trunk is `embed_dim`-sized and
   unchanged; only the encoder's first layer and decoder's last layer carry per-channel
   rows. Dropping the two rows/columns and warm-starting is possible in principle. **If you
   try it, prove it** — a 1-epoch arm whose loss starts near the base model's rather than at
   initialization.

### 🔴 The comparison trap, which will bite on the first scorecard

**The new model has 99 channels; every existing baseline is a 101-channel number.** Medians
over channels are not comparable across different channel sets — `validation loss 0.01284`,
`NRMSE336 0.970`, and every percentage in the `n_future` ladder would shift from the channel
set alone. Before the first comparison, **recompute the old baselines restricted to the
common 99 channels.** That is free: `k56_metrics.h5` holds per-channel arrays, so it is a
re-take of the medians, not a re-run. Do it *before* the new numbers exist, so the
comparison cannot be tuned after the fact.

### 🔗 What this does to the rest of the list

**It supersedes most of port A.** The positivity clamp's headline target was
`SOILWATER_10CM` — the channel whose −59 %-by-step-500 extrapolation started this whole
line of work. With it gone, port A's remaining candidates are `PRECT`, `RHREFHT` and the
humidity levels, which have never been shown to go unphysical. ⇒ **Re-rank port A below
port F**, and if the retrain is happening anyway, fold the clamp question into it rather
than running it against the old checkpoint.

It also **removes the leading blow-up candidate** for the ~500-step divergence. That makes
the post-retrain long rollout a real experiment: if divergence still happens at ~500 steps
with no soil reservoir in the state, the reservoir-drift chain was never the cause and
`2026-09-10_ace2_comparison_the_corrector.md` §4 needs retracting. **Run it, and say so
either way.**

## 7. Protocol — every training arm follows this, or it is not comparable

1. **Warm-start from `prod1n_b32_sgdr`'s best checkpoint**, single-variable A/B, and write a
   `warmstart_provenance.txt` naming every knob. The D1-vs-C1 comparison only worked because
   that file proved the two differed in exactly one.
2. **Score with `submit_rollout_scorecard.sh` at `valid_autoreg_steps=20`** — 21 leads, 101
   channels, 512 samples — and compare **models at a fixed rollout length**, never across
   lengths.
3. **Report RMSE and ACC together.** RMSE alone cannot distinguish skill from amplitude
   damping; ACC is penalised by lost variance. That pairing is what refuted the blurring
   hypothesis for `n_f=3/4`, and it is the check that makes a result publishable here.
4. **The bar is 1.92 %.** ACC is the noisier metric — quote a range across seeds, not the
   luckier one.
5. **Pre-register the read-out** before looking at numbers, as
   `2026-09-20_k56_readout_prereg.md` did, with the rule implemented as a tested function so
   it cannot move afterwards. That document's own disclosed defect (a max-over-channels
   clause selected by a degenerate denominator) is the model for how to handle it when the
   rule misfires: report as written, fix the *next* prereg.
6. **`validation loss` is a SINGLE-STEP number** and always was. Never use it to judge a
   change aimed at long-lead behaviour.

---

## 8. Traps, each of which has already cost a job

1. **`ckpt_mp0_v0.tar` or nothing** — makani gates `resuming` on exactly that filename. Seed
   a directory with any other name and it **trains from scratch with no error**. Assert
   `resuming True` in the log.
2. **A seeded/forked directory needs `WANDB=0`** — with wandb on *and* resuming, every rank
   dies at construction, presenting as NCCL teardown noise.
3. **`LOAD_COUNTERS=0` for any validation-only run** — restored epoch 243 against `EPOCHS=1`
   makes the loop empty, and validation lives inside it. Exits 0, writes no loss.
4. **`LOAD_LOSS=0` when `n_future` changes** — `LossHandler`'s running statistics are
   shape-dependent on `n_future`. ⚠ **Assume this applies to a loss-weighting change too**
   until shown otherwise; ports B and C both touch `LossHandler` construction.
5. **A config-side `n_future` does nothing** — `train.py:119` overwrites it from
   `--multistep_count`. `MULTISTEP` is the only handle.
6. **`pretrained` and `resuming` are mutually exclusive** — a fine-tune needs a **new**
   `RUN_NUM`, or resuming wins silently.
7. **The config root key must match the filename** — `polaris_makani_multinode_scaling.pbs`
   derives it from the stem and rewrites `^<stem>:`. A copied config that kept its parent's
   root key is how the CRPS arm failed after 45 s (and the guard is why it was 45 s and not
   a silent run of the wrong loss).
8. **`qalter` is refused outright on Polaris** — walltime and dependencies are immutable
   after submission. Size with margin.
9. **Never resubmit a stuck job before diagnosing** — `queue_tags` with a large
   `eligible_time` means the queue has no nodes, and resubmitting destroys accrued eligible
   time (CLAUDE.md #12).

---

## 9. Do not

* **Do not chase depth 1.** Four arms agree it is the weakest rung (C1, D1, both 1-epoch proxies).
* **Do not re-run the lagged ensemble** in any variant — more members, other residue classes,
  or the 685 GB full-year sweep. The failure is structural, not a sample-size problem.
* **Do not run the CRPS / `PlasimEnsembleTrainer` arm** on the strength of blurring. VR is
  1.019 at 336 h; mode-averaging is ruled out. It stays parked.
* **Do not resubmit 7630639** (depth-4 × 24 epochs). Last seen queued with `eligible_time`
  past 72 h and `comment = Insufficient amount of resource: queue_tags` — that is the queue,
  not the job.
* **Do not raise the learning rate.** 2e-3 is one rung below a measured hard ceiling.
* **Do not edit `scripts/score_nwp.py`** for E3SM — it still calls
  `legendre_gauss_lat_weights` directly on what is an **equiangular** grid (GL over-weights
  the polar row 1.50× and the only downstream guard is a shape check that passes). Use
  `polaris/score_rollout_nc.py`, the E3SM sibling.

---

## 10. Assets

| what | where |
|---|---|
| the differences this acts on | `ace2_vs_makani_differences.md` (PR #15) |
| best checkpoint | `$MEMBER_ROOT/runs/makani_mn_scaling/e3sm_mn_scaling/prod1n_b32_sgdr/training_checkpoints/best_ckpt_mp0.tar` (epoch 243) |
| all 243 epoch checkpoints | same directory, `ckpt_mp0_v0…v242`, 403.2 GiB |
| training config | `makani_sfno/polaris/e3sm_alldata_full.yaml` |
| launcher | `makani_sfno/polaris/polaris_makani_multinode_scaling.pbs` |
| converter (port B step 1) | `makani_sfno/polaris/convert_e3sm_to_makani_alldata.py` |
| rollout driver (port A) | `makani_sfno/src/sfno_inference/rollout_driver.py` ⚠ subtree |
| K=56 sweep + scorer | `polaris/polaris_eval_inference.pbs`, `polaris/score_rollout_nc.py` |
| 56-lead σ(k) and bias(k) for all 101 channels | `$MEMBER_ROOT/runs/makani_eval/prod1n_b32_sgdr_K56/scores/k56_metrics.h5` |
| scorecard | `makani_sfno/polaris/submit_rollout_scorecard.sh` |
| EMA implementation | `makani_sfno/src/sfno_training/trainer/ema.py` |

---

## 11. ✅ DONE — the free analysis, and it shrinks the prize

Ran inside job **7646192**. `bias² / MSE` at 336 h, per channel:

| channel | bias² share of 336 h MSE |
|---|---|
| `Z3_l17` | **0.253** |
| `Z3_l16` | **0.139** |
| `T_l00` | 0.041 |
| `Z3_l15` | 0.038 |
| `T_l01` | 0.028 |
| `Z3_l00` | 0.022 |
| `SOILWATER_10CM` | 0.018 |
| everything else (94 channels) | **≤ 0.010** |

🔴 **This is a correction to what I said earlier in the session.** I called removable drift
"possibly the largest single win". It is not: **two channels of 101 carry a material
systematic component, and both are `Z3` levels.** For the other 99, a per-channel drift
correction is worth ≤ 1 % of MSE and is not worth building.

What it *is* is a sharp pointer: **the drift problem is a `Z3`-family problem**, it sits in
exactly the channels the loss is blindest to (§3's `r_c` ranking puts `Z3_l17` slowest of
all 101 by a factor of 35 over the next), and it is the same channel the K=56 read-out
flagged independently on a completely different normalization. Three independent lines now
point at the same place.

⇒ **Take the `Z3` question to the science owner before building anything.** Whether a
near-constant terrain-following geopotential level should be prognostic at all (§6) now has
quantitative backing: `Z3_l17` is simultaneously the slowest channel in the pack, the one
whose 336 h error is most systematic, and the one that would receive 9518× the loss weight
if port B were enabled naively. It may be that the correct fix is to stop predicting it,
not to weight it harder.

**The competing explanation is refuted**, so it needs no further work: train is 2015–2044
and test 2048–2049 under a warming scenario, with all statistics computed on the train
split, but Spearman(`Δmean_c`, `bias_c` @336 h) in σ units is **+0.079** — essentially zero.
The drift is not the model regressing toward a stale climatology.
