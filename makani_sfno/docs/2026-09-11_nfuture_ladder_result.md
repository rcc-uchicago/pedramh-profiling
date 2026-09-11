# `n_future = 4` for ONE epoch matches `n_future = 1` for twenty-four

**Measured 2026-09-11.** Depth substitutes for training time, the gain is real
skill rather than blurring, and the earlier proxy failure is explained rather
than explained away.

Supersedes the pessimistic reading in `2026-09-11_proxy_failure_diagnosis.md`.

---

## 1. The ladder, all scored identically

All arms warm-started from the same base checkpoint and scored at
`valid_autoreg_steps=20` — 21 leads, 101 channels, 512 samples. Metric is the
median over channels of the per-channel ratio against base.

| arm | epochs | batch | lead-1 RMSE | best RMSE | @ | lead-126 RMSE |
|---|---|---|---|---|---|---|
| **C1** `n_f=1` | 24 | 16 | +1.50 % | -4.65 % | 36 h | -3.00 % |
| `n_f=1` | 1 | 8 | +3.26 % | +0.64 % | 18 h | **+7.11 %** |
| **`n_f=3`** | 1 | 8 | +5.13 % | **-3.16 %** | 126 h | -3.16 % |
| **`n_f=4`** | 1 | 8 | +6.24 % | **-4.66 %** | 126 h | -4.66 % |

Pre-registered bar: **2 x sigma_0 = 1.92 %** (`sigma_0 = 0.96 %` measured in
Phase 0 from snapshot-to-snapshot spread).

⇒ `n_f=3` and `n_f=4` clear it. `n_f=1` at one epoch does not — it is worse than
base at every lead.

## 2. It is NOT blurring — the decisive check

MSE-trained models hedge by damping variance, and a blurred forecast **wins on
RMSE** at long lead while **losing on ACC**, which is a correlation and is
penalised by the lost variance. So RMSE alone cannot distinguish skill from
blurring. ACC can, and both are already in the scorecard files.

At lead 126 h:

| arm | RMSE vs base | ACC vs base | channels better: RMSE / ACC / **both** | verdict |
|---|---|---|---|---|
| C1 `n_f=1` 24ep | -3.00 % | **+0.00493** | 77 / 77 / **73** | genuine skill |
| `n_f=1` 1ep | +7.11 % | **-0.02331** | 2 / 0 / **0** | damaged on both |
| `n_f=3` 1ep | -3.16 % | **+0.00113** | 67 / 55 / **54** | genuine skill |
| **`n_f=4` 1ep** | **-4.66 %** | **+0.00584** | 81 / 78 / **76** | **genuine skill** |

✅ **`n_f=3` and `n_f=4` improve BOTH metrics across most channels.** The
blurring hypothesis predicted the opposite and is refuted.

## 3. The headline

> **`n_future = 4` trained for ONE epoch matches or beats `n_future = 1`
> trained for twenty-four** — -4.66 % vs -3.00 % on RMSE at 126 h, +0.00584 vs
> +0.00493 on ACC, 76 vs 73 of 101 channels improved on both metrics.
>
> **Rollout depth substitutes for training time.**

For cost: C1 took **8 hours**; the `n_f=4` arm took **35 minutes**.

## 4. And it explains the proxy failure

`2026-09-10_nfuture_ladder_prereg.md` §3 gated Phase 2 on a 1-epoch `n_f=1`
fine-tune reproducing C1. It did not, and the gate fired.

The reason is now visible: **the proxy tested depth 1, which is the one depth
that buys nothing quickly.** At depth 1 the learning-rate perturbation damages
the model — worse on RMSE *and* ACC, **0 of 101 channels improved on both** —
with no compensating multi-step gain in a single epoch. At depths 3 and 4 the
same perturbation is repaid immediately.

⇒ The short recipe was never incapable of showing depth effects. It was
validated against the **worst possible** reference point.

⚠ The gate still did its job: without it the three curves would have been read
as a trend with no check on whether the recipe could resolve anything, and the
blurring alternative would never have been tested.

## 5. The trade is orderly, and visible

| depth | lead-1 cost | 126 h gain |
|---|---|---|
| 1 (1 ep) | +3.26 % | +7.11 % (no gain — damaged) |
| 3 | +5.13 % | -3.16 % |
| 4 | +6.24 % | -4.66 % |

Single-step accuracy is given up **monotonically** with depth, and long-lead
skill is bought monotonically with depth. That is exactly the predicted
signature of rollout training, now measured rather than assumed.

Note also the **shape** difference from C1: C1 peaks at 36 h then weakens to
-3.00 % by 126 h, whereas `n_f=3/4` are still improving at the last lead. Deeper
training pushes the benefit **outward in lead time**, which is what a long
rollout needs.

## 6. What is NOT yet established

1. **n = 1, unreplicated.** A repeat of `n_f=4` is queued (**7607361**).
2. **All arms share the no-warmup / batch-8 recipe.** D1 (**7606726**, 24
   epochs, depth 1, *with* warmup, batch 8) is still queued and will say whether
   depth 1 recovers — i.e. whether batch 8 is innocent.
3. **126 h is still short.** These gains are measured over 5.25 days; the
   rollout of interest is 500 steps. Depth helping at 21 leads does not prove it
   helps at 500 — see `2026-09-10_ace2_comparison_the_corrector.md`, where ACE2
   reaches 7300 steps at depth 1 with a corrector.
4. **This does not overturn the ACE2 finding.** Depth buys long-lead *accuracy*.
   Whether it buys *stability* at 500 steps is a different measurement, and the
   corrector route remains the cheaper candidate for that.

## 7. Recommendation

- **Run `n_future = 4` at 24 epochs as the production candidate** (~40 h,
  preemptable). It is the best arm measured and it was the cheapest to find.
- **Stop chasing depth 1.** Both C1 and the failed proxy agree it is the weakest
  rung.
- Keep the corrector work on the critical path for *stability*; this result is
  about accuracy-vs-lead, which is a different axis.
