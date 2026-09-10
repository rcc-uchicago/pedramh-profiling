# What FourCastNet 3's recipe actually is, and what we ran instead

**Question, 2026-09-10:** "isn't this how the makani paper and codebase trains
it? the pretraining at time-step=0 and then the fine tuning"

**Answer: the two-stage *shape* is correct, and we followed it. But FCN3 is
probabilistic in every stage, and we ran the deterministic ablation of it.**
The components we dropped are precisely the ones that buy long-rollout
stability.

Read from `makani-upstream/config/fourcastnet3.yaml` (the shipped config, not
the paper).

⚠ **This corrects our own documentation.** `polaris_makani_1node_production_handoff.md`
describes upstream's recipe as "FCN3 pretrain-2 exists *to get good
autoregressive rollouts* and is a fine-tune from pretrain-1's checkpoint."
That sentence is accurate — `fourcastnet3.yaml:252` says exactly that — but it
is **incomplete**, and reading it as "stage 1 is deterministic single-step" is
wrong. Stage 1 is a 16-member CRPS ensemble with input noise.

---

## 1. FCN3's three stages, as configured

| | rollout steps | loss | ensemble size | input noise | epochs |
|---|---|---|---|---|---|
| `pretrain1` (`:222`) | **1** | `ensemble_crps` (cdf) **+ `ensemble_spectral_crps`** @ 0.1 | **16** | **diffusion, 8 ch, uncentered** | 130 |
| `pretrain2` (`:254`) | **4** | fair CRPS (`skillspread`) **+ spectral fair CRPS** @ 0.1 | 2 | diffusion, 8 ch, **centered** | 24 |
| `finetune` (`:297`) | 4 | fair CRPS + spectral | 4 | centered | 4 |

Anchors: `ensemble_base: &ENSEMBLE_BASE` (`:144`), `ensemble_finetune:
&ENSEMBLE_FINETUNE` (`:174`).

> `:220-221` — *"Can accommodate up to 2 autoregressive training steps, but
> trained with a single step"*
> `:251-253` — *"second pretraining stage which switches to fair CRPS to get
> better calibration; also uses 4 step rollouts to get good autoregressive
> rollouts"*

So yes: **stage 1 is single-step.** But it is single-step *and probabilistic* —
it is never a plain MSE regressor.

⚠ `n_future` appears nowhere in the YAML. It is set from the command line:
`train.py:119` does `params["n_future"] = args.multistep_count - 1`. So FCN3's
"4 step rollouts" is `--multistep_count 4`, i.e. `n_future = 3`.

## 2. makani also ships a deterministic baseline — and that is what we ran

`deterministic_base: &DETERMINISTIC_BASE` (`:204`) — a single `l2` loss, no
ensemble, no noise. It is consumed by `det_sfno2_sc3_edim384_layers10` (`:356`)
and `det_fcn3_sc3_edim25_layers10` (`:363`).

Our production config, from `prod1n_b32_sgdr/config.json`:

```
losses      = [{'type': 'l2', 'parameters': {'squared': True},
                'channel_weights': 'constant',
                'temp_diff_normalization': False}]
nettype     = SFNO      scale_factor = 3    embed_dim = 384   num_layers = 8
n_future    = 0         multistep_count = 1
ensemble_size, input_noise: ABSENT
```

That is `DETERMINISTIC_BASE`, on essentially the `det_sfno2_sc3_edim384`
architecture (`sc3`, `edim384`; 8 layers against their 10).

⇒ **We ran makani's deterministic baseline, correctly.** It is a supported,
shipped configuration — it is just the *comparison point*, not the
configuration that produces FCN3's published stable rollouts.

## 3. Side-by-side

| | FCN3 flagship | ours |
|---|---|---|
| stage-1 loss | ensemble CRPS + **spectral CRPS** | `l2` (squared) |
| stage-1 ensemble | 16 members | none (deterministic) |
| input noise | **every stage** | none |
| stage-2 rollout | **4 steps** (`n_future=3`) | **2 steps** (`n_future=1`, C1) |
| stage-3 finetune | yes (lr 4e-6, 4 epochs) | none |
| channel weights | `auto` | `constant` |
| `temp_diff_normalization` | `True` | `False` |

The last two rows are deviations from **even the deterministic baseline** and
are loss-weighting choices — which channels the optimizer prioritizes. They
belong to the science owner.

## 4. Why this explains the blow-up

Of the four dropped components, **`ensemble_spectral_crps` is the one that
matters most for a 500-step rollout.** It is carried in *every* FCN3 stage at
relative weight 0.1, and it scores the forecast's **spectrum**. Uncontrolled
accumulation of small-scale energy is the classic autoregressive blow-up mode,
and upstream penalises it directly, throughout training. We have no equivalent
term.

Input noise is the second. Training on noised inputs is a direct remedy for
exposure bias: it teaches the model to map perturbed states back toward the
attractor. FCN3 applies it in all three stages.

This reframes the measured result in `2026-09-10_longroll_blowup_analysis.md`.
C1 (`n_future=1`, deterministic `l2`) lowered error **3-4.7 %** at every lead
past the first, yet left every stability statistic unchanged — crossing of the
climatological ceiling at 94 vs 95 steps, late/early slope ratio 0.774 vs
0.779. That is now unsurprising: **more rollout steps under an MSE loss changes
the error level, not the spectral behaviour that governs stability.** Upstream
never relies on rollout depth alone.

## 5. Consequences

1. **Nothing was done wrong.** The two-stage shape is right and was followed.
   The gap is that our stage 1 is the deterministic ablation.
2. **Going to `n_future = 3` (matching FCN3 stage 2) is worth doing** and is
   affordable — the measured memory model allows up to ~3 at batch 4/GPU
   (`polaris_makani_1node_production_handoff.md` §C2). But §4 predicts it buys
   accuracy, not stability.
3. ⚠ **There is a real tension with the no-noise decision.** The science owner
   ruled out noise injection (`2026-09-10_lagged_ensemble_design.md` §3), and
   the lagged ensemble exists to avoid it. But FCN3 — the recipe being followed —
   uses input noise in **all three** stages. "Follow the makani recipe" and
   "no noise" cannot both hold. Flagged as a decision, not a recommendation:
   the lagged ensemble remains a sound *inference-time* ensemble either way, it
   simply does not substitute for noise during *training*.
4. **A spectral loss term is separable from the noise question.**
   `ensemble_spectral_crps` needs an ensemble, but a deterministic spectral
   penalty is not the same commitment as input noise, and it targets the
   observed failure directly. Worth scoping before committing to a full
   probabilistic port.
5. **Cheapest test of the mechanism, no retraining:** apply a mild spectral
   filter at *inference only* inside `longroll.py`'s loop and see whether the
   divergence step moves. If it moves a lot, accumulated small-scale energy is
   confirmed as the mode, and §4 becomes the priority rather than C2.
