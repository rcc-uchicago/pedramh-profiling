# The two open caveats are closed: `n_f=4` replicates, and batch 8 is not innocent

**Measured 2026-09-15** from arms that finished *after*
`2026-09-11_nfuture_ladder_result.md` was written. That doc's §6 listed four
things "not yet established"; items 1 and 2 are now answered. Items 3 and 4
(126 h is short of 500 steps; stability is a different axis) stand unchanged.

The headline survives and gets **stronger**, for a reason that was not
anticipated: the batch-16 reference it was measured against was *helping* the
reference.

---

## 1. The full ladder, one protocol

All arms warm-started from the same base checkpoint (`prod1n_b32_sgdr`) and
scored by `submit_rollout_scorecard.sh` at `valid_autoreg_steps=20` — 21 leads,
101 channels, 512 samples. RMSE is the median over channels of the per-channel
ratio against base (negative is better); ACC is the median absolute difference
(positive is better).

Reproduce with `polaris/analyze_nfuture_ladder.py`, which did not exist when the
first table was made — those numbers were computed ad hoc. It reproduces all
four published rows exactly.

| arm | epochs | batch | depth | lead-6 | best | @ | **lead-126** | ACC@126 | both |
|---|---|---|---|---|---|---|---|---|---|
| **C1** | 24 | 16 | 1 | +1.50 % | -4.65 % | 36 h | **-3.00 %** | +0.00493 | 73/101 |
| **D1** 🆕 | 24 | **8** | 1 | +2.10 % | -3.32 % | 30 h | **-0.72 %** | +0.00103 | 51/101 |
| proxy r1 | 1 | 8 | 1 | +3.26 % | +0.64 % | 18 h | +7.11 % | -0.02331 | 0/101 |
| proxy r2 🆕 | 1 | 8 | 1 | +3.41 % | +0.60 % | 18 h | +7.67 % | -0.02572 | 0/101 |
| `n_f=3` | 1 | 8 | 3 | +5.13 % | -3.16 % | 126 h | **-3.16 %** | +0.00113 | 54/101 |
| `n_f=4` r1 | 1 | 8 | 4 | +6.24 % | -4.66 % | 126 h | **-4.66 %** | +0.00584 | 76/101 |
| `n_f=4` r2 🆕 | 1 | 8 | 4 | +6.25 % | -4.64 % | 114 h | **-4.63 %** | +0.00445 | 66/101 |

**Bar: `2*sigma_0 = 1.77 %`**, recomputed here as the standard deviation over the
three snapshot-ensemble arms of RMSE % at lead 126 h. The published Phase-0
figure was 1.92 % (`sigma_0 = 0.96 %`); this is 0.88 %. Nothing in the table
changes sign or verdict under either bar, and the looser published number is the
conservative one — **keep quoting 1.92 %.**

## 2. `n_f=4` replicates, tightly on RMSE and loosely on ACC

Two independent seeds (7603424 / 7607361), scored identically:

| lead | r1 RMSE | r2 RMSE | r1 dACC | r2 dACC |
|---|---|---|---|---|
| 6 h | +6.24 % | +6.25 % | -0.00088 | -0.00089 |
| 18 h | -0.61 % | -0.57 % | +0.00014 | +0.00015 |
| 126 h | -4.66 % | -4.63 % | +0.00584 | +0.00445 |

✅ **RMSE replicates to within 0.03 pp at the endpoint** and both seeds cross
from worse-than-base to better-than-base at the same lead (18 h). The
single-step cost and the long-lead gain are both reproducible.

⚠ **ACC is the noisier metric, and the published `76/101` was the luckier
seed.** r1's dACC keeps climbing to +0.00584; r2's plateaus near +0.0041 from
lead 96 onward, ending at +0.00445 with 66 of 101 channels improved on both
metrics. **Quote the range 66–76/101, not 76.** The blurring verdict is
unaffected — both seeds improve RMSE *and* ACC across a clear majority of
channels, which is what the check tests.

## 3. D1: batch 8 costs 2.3 pp of long-lead skill at depth 1

D1 (7606726) ran 24 epochs at depth 1 and finished 2026-09-13 11:53. Against C1
it is a **clean single-variable A/B**: `warmstart_provenance.txt` records the two
runs as identical in every knob — same base checkpoint, `lr_peak 4e-4`,
`CosineAnnealingLR(warmup=1 epoch, T_max=100)`, `max_epochs 24`, no EMA,
`optimizer_max_grad_norm 32`, `n_future 1`, `multistep_count 2` — **except
`batch_size_global`: 16 vs 8.**

> **C1 batch 16 → -3.00 % at 126 h. D1 batch 8 → -0.72 %.**
> D1 does not clear the bar. **Batch 8 is not innocent.**

And it is not an under-training story: at the same LR, batch 8 takes **twice the
optimizer steps per epoch**, so D1 did ~2× C1's updates and still ended 2.3 pp
worse at long lead.

**D1's curve has C1's shape, not `n_f=3/4`'s.** It peaks early (-3.32 % at 30 h)
and decays monotonically to -0.72 % by 126 h, while `n_f=3/4` are still improving
at the last lead. That shape is therefore a property of **depth 1**, not of
batch 16 — which is what D1 was run to discriminate.

### What this does to the headline

The published comparison was `n_f=4` at batch 8 (-4.66 %) against C1 at batch 16
(-3.00 %) — a depth difference confounded with a batch difference. D1 supplies
the batch-matched reference:

| at batch 8 | lead-126 |
|---|---|
| depth 1, **24 epochs** (D1) | -0.72 % |
| depth 4, **1 epoch** (`n_f=4`) | **-4.66 % / -4.63 %** |

⇒ **At matched batch size, one epoch at depth 4 beats twenty-four epochs at
depth 1 by ~3.9 pp.** The confound ran *against* the conclusion: batch 16 was
flattering the depth-1 reference, so the original headline understated the depth
effect rather than overstating it.

Residual confound: the 1-epoch arms have `warmup=0`, C1/D1 have `warmup=1 epoch`.
Depth is not isolated from warmup by any arm in this table.

## 4. The depth-1 proxy failure replicates too

`nf1_diag_b8_r1` (7606724) is a second seed of the 1-epoch depth-1 proxy:
+3.41 % at lead 6, +7.67 % at 126 h, dACC -0.02572, **0 of 101 channels improved
on both** — the same total failure as r1 (+3.26 / +7.11 / -0.02331 / 0 of 101).

⇒ The proxy gate did not fire on noise. Depth 1 at one epoch damages the model
on both metrics, reproducibly.

## 5. Defect found: the CRPS arm never ran

`nf4_crps_b4_r1` (**7607600**) exited after 45 s with

```
ERROR CONFIG_KEY_RENAME_FAILED: root key 'e3sm_alldata_crps:' not found in e3sm_alldata_crps.yaml
```

`polaris_makani_multinode_scaling.pbs:321-327` derives the config key from the
**filename** and rewrites `^<stem>:` to `e3sm_mn_scaling:`. `e3sm_alldata_crps.yaml`
was copied from `e3sm_alldata_full.yaml` and kept its parent's root key, so the
`sed` matched nothing and the guard caught it. Fixed by renaming the root key to
`e3sm_alldata_crps:`.

✅ The guard is the reason this is a 45-second failure and not a silent run of
the wrong config — the run would otherwise have trained `e3sm_alldata_full`'s
deterministic loss under a CRPS tag. **The ensemble/CRPS path is still unmeasured.**

## 6. Recommendation, unchanged and now better supported

- **Run `n_future = 4` at 24 epochs as the production candidate.** It is the best
  arm at n=2, and the batch-matched reference makes its margin larger than
  published.
- **Stop chasing depth 1.** Three arms now agree: C1, D1 and both 1-epoch proxies.
- **Do not run the production candidate at batch 8 by default.** D1 is direct
  evidence that batch 8 costs long-lead skill at depth 1; whether it costs the
  same at depth 4 is untested, and depth 4 at batch 16 has never been run.
- The CRPS/ensemble arm needs a resubmit now that its config key is fixed.
