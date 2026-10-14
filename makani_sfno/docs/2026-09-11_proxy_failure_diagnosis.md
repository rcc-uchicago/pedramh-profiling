# Why the 1-epoch proxy failed to reproduce C1 — hypotheses and the discriminating runs

**2026-09-11.** The Phase 1 gate in `2026-09-10_nfuture_ladder_prereg.md` §3
fired: a 1-epoch `n_future=1` fine-tune did **not** reproduce C1's known result,
so the short-fine-tune proxy is invalid and the `n_future=3/4` arms run under it
cannot be interpreted. This document says why that might be, and what
discriminates the options.

---

## 1. The observation

Both scored identically — `valid_autoreg_steps=20`, 21 leads, 101 channels,
512 samples, median per-channel RMSE ratio against the same base.

| | lead-1 | best | @ | lead-126 |
|---|---|---|---|---|
| **C1** — 24 epochs, global batch 16 | +1.50 % | **-4.65 %** | 36 h | -3.00 % |
| **proxy** — 1 epoch, global batch 8 | +3.26 % | **+0.64 %** | 18 h | **+7.11 %** |

Correlation of the two curves across leads: **-0.001**.

⚠ The proxy is worse than base at **every** lead and degrades monotonically with
lead. C1 is better than base from lead 2 onward. These are not the same
behaviour weakly expressed — they are opposite.

## 2. What the configs actually differ by

Read from each run's own `config.json`:

| key | C1 | proxy | |
|---|---|---|---|
| `lr` | 4.0e-4 | 4.0e-4 | same |
| `scheduler` | CosineAnnealingLR | CosineAnnealingLR | same |
| `scheduler_T_max` | 100 | 100 | same |
| `scheduler_min_lr` | 1e-6 | 1e-6 | same |
| `optimizer_beta2` | 0.95 | 0.95 | same |
| `optimizer_max_grad_norm` | 32 | 32 | same |
| **`lr_warmup_steps`** | **1** | **0** | ⚠ **differs — introduced by me** |
| **`max_epochs`** | **24** | **1** | ⚠ differs |
| **`global_batch_size`** | **16** | **8** | ⚠ differs |

✅ **`scheduler_T_max = 100` in both**, so the cosine is only 1-24 % traversed in
either run and the learning rate is essentially **flat at 4e-4** throughout.
**Schedule decay is refuted** as an explanation.

That leaves exactly three candidates.

## 3. Hypotheses

### H-A — too few optimizer updates

Proxy: 1 epoch at batch 8 = **5,534 updates**.
C1: 24 epochs at batch 16 = **66,412 updates**. A **12x** gap.

Under H-A the proxy is simply early on C1's trajectory and the rollout benefit
has not had time to materialise.

⚠ **Weak point:** H-A predicts "looks like C1 but weaker." It does **not**
naturally predict "worse than base at every lead, monotonically worsening with
lead." The proxy is not a diluted C1; it is somewhere else.

### H-B — no warmup, applied to converged weights. **(mine)**

`submit_nfuture_ladder.sh` set `WARMUP_EPOCHS=0`. C1 used **1**.

With `lr_start: 0.01`, C1's **entire first epoch** was a ramp from
`0.01 x 4e-4 = 4e-6` up to `4e-4`. The proxy's entire single epoch ran at the
**full 4e-4 from step 0**.

⇒ The proxy applied roughly **2x C1's average first-epoch learning rate**,
immediately, to a checkpoint that had converged over 243 epochs — and had one
epoch in which to recover, which is none.

This predicts the observed signature well: an indiscriminate perturbation of
converged weights degrades everything, and the degradation compounds with
rollout depth because errors feed forward.

### H-C — global batch 8 is too noisy

Batch 8 is **4x below** the base run's 32. At fixed `lr`, a 4x smaller batch is
a substantially more aggressive per-sample step with 2x the relative gradient
noise. Compounds H-B rather than competing with it.

⚠ **If H-C is true it breaks the ladder's own design.** Prereg threat T3 holds
global batch at 8 across arms specifically so depth is not confounded with
batch — but if batch 8 is itself damaging, every arm is damaged and the control
is worthless.

## 4. The discriminating runs

Both hold `n_future = 1`, so C1's answer (-4.65 % at best lead) is the target.

| arm | job | epochs | warmup | batch | isolates | cost |
|---|---|---|---|---|---|---|
| **D2** | **7606724** | **1** | **1** | 8 | **H-B** — differs from the failed proxy *only* in warmup | ~20 min |
| **D1** | **7606726** | **24** | **1** | 8 | **H-C** — matches C1 in everything *except* batch | ~6.4 h |

### Pre-registered readings

**D2 (vs the failed proxy, warmup the only difference):**
- large improvement -> **H-B confirmed**; the proxy failure was my launcher, not
  a property of short fine-tunes, and a cheap valid proxy may exist
- little change -> H-B rejected; warmup was not the issue

**D1 (vs C1, batch the only difference):**
- reproduces C1 (best lead within ~1 sigma_0 = 0.96 % of -4.65 %) -> **H-C
  rejected, H-A confirmed.** Batch 8 is fine; the proxy simply needed the
  epochs. ⇒ a valid ladder requires **24-epoch arms**: ~32 h at `n_future=3`,
  ~40 h at `n_future=4`.
- fails to reproduce C1 -> **H-C confirmed.** Batch 8 is damaging, the ladder's
  batch-holding control is broken, and the design needs rethinking before any
  further compute.

⚠ D1 and D2 together also bound H-A: if D2 (1 epoch, warmup) already recovers
much of C1's effect, then 24-epoch arms are unnecessary and the ladder gets
~30x cheaper.

## 5. Why this matters beyond the ladder

The prereg's proxy-validation step cost two short jobs and stopped three tidy
`n_future` curves from being read as a trend. Without it there would have been a
plot of depth 1 / 3 / 4, all run under a recipe now known not to reproduce a
result we already had.

⚠ **The `n_future=3` and `n_future=4` arms already run (7603423 / 7603424) must
not be quoted as evidence about rollout depth.** They were trained under the
invalid recipe — `WARMUP_EPOCHS=0` included.
