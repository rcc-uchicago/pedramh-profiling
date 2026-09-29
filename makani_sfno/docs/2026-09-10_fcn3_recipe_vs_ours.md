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

## 4a. What the stochastic route would COST — measured, 2026-09-10

`ensemble_size` folds into the batch (`ensemble_trainer.py:502`,
`expand_ensemble(inp, E)` -> `(B*E, C, H, W)`), so both memory and compute scale
**linearly in E**. Combining that with the measured memory model and the
measured throughput law:

    memory:      (n_future+1) * samples_per_GPU * E  <=  13.76
    throughput:  samples/s  ~  67.8 / ((n_future+1) * E)

The throughput law is well validated — predicted vs measured samples/s is
16.95 vs **16.83** at `n_future=3` and 13.56 vs **14.11** at `n_future=4`
(jobs 7603323 / 7603324).

| configuration | `n_f` | E | max b/GPU | global batch | samples/s | relative |
|---|---|---|---|---|---|---|
| **our base today** | 0 | 1 | 8 | 32 | 67.8 | **1.00x** |
| C1 | 1 | 1 | 6 | 24 | 37.1 | 0.55x |
| `n_future=3` (measured) | 3 | 1 | 3 | 12 | 16.8 | 0.25x |
| `n_future=4` (measured) | 4 | 1 | 2 | 8 | 14.1 | 0.21x |
| **FCN3 stage-2 shape** | 3 | **2** | 1 | **4** | ~8.4 | **0.12x** |
| **FCN3 stage-1 shape** | 0 | **16** | 0.86 | — | — | ❌ **does not fit** |

⇒ **FCN3's stage 1 does not fit our hardware.** At `ensemble_size: 16` the
activation term needs 16x the memory and the bound gives **under one sample per
GPU**. FCN3 says as much itself (`:220`): stage 1 needs 80 GB VRAM *and*
model-parallelism `h=2, w=2`. We have 40 GB cards, so we would need sharding
merely to fit one sample — and our own scaling study measured that sharding
costs throughput.

⇒ **A stochastic FINE-TUNE, however, is affordable.** Global batch collapses to
4 and throughput to 12 % of baseline, but FCN3 deliberately shrinks stage 2's
data budget (`n_train_samples_per_epoch: 6720` against stage 1's 26880). At that
budget 24 epochs is **~5 hours**; at C1-style full epochs it is ~35 hours.

⇒ **Redoing the base pretrain stochastically is not affordable**: 46.3
node-hours becomes roughly **350** (7.6x), before counting CRPS's own cost —
pairwise/sorting over members, plus a spherical harmonic transform per member
for the spectral term.

⚠ And the binding cost is not GPU hours. Our fork extends the **deterministic**
`Trainer`; the four contract patches that make our 107->101 channels and forcing
feedback work **do not exist on `ensemble_trainer.py`**. That port is the
bottleneck (handoff §4).

### 4b. And ACE2 says you may not need to pay any of it

**ACE2 trains with `loss: type: MSE`** (`config_polaris.yaml:120-121`) — no
ensemble, no CRPS, no input noise anywhere in its config — and reaches
**7300-step (5-year)** stable rollouts.

So the two working reference points disagree about the route:

| | how it buys stability | cost to us |
|---|---|---|
| **FCN3** | probabilistic objective — CRPS, ensemble, input noise, spectral CRPS | high; stage 1 does not fit our cards |
| **ACE2** | **corrector** (conservation + positivity), per-channel loss weights, residual normalization | near-free; the corrector needs **no training at all** |

⇒ **Treat the stochastic route as the fallback, not the plan.** The cheap
experiments — an inference-time positivity clamp, and `time_diff_stds` +
`temp_diff_normalization` — sit on the ACE2 path and cost hours. The stochastic
port earns its price only if those fail.

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

## 6. Compute — FCN3 as the paper reports it, against our runs (added 2026-09-29)

FCN3 rows are the paper's own numbers (arXiv:2507.12144, training section, quoted
verbatim): *"1024 NVIDIA H100 on the NVIDIA Eos Supercomputer for a total of 78 hours"*,
*"208,320 gradient descent steps … batch size of 16 and an ensemble size of 16"*; *"5,040
steps … 15 hours on 512 NVIDIA A100 GPUs … NERSC Perlmutter"*, *"4 autoregressive rollout
steps"*; *"256 NVIDIA H100 GPUs on the Eos system and took 8 hours"*. Stage-3 steps are the
config's (2920 samples / batch 4 × 4 epochs); node counts assume 8 GPUs per Eos (DGX H100)
node and 4 per Perlmutter GPU node. Our A row is measured; F and G are the estimates of
CHANGELOG 2026-09-29 (range = linear scaling … the 4-node CXI smoke's 521 ms/step).

| run | GPUs | nodes | batch × ensemble | steps | rollout | wall (h) | s / step | GPU-hours |
|---|---|---|---|---|---|---|---|---|
| FCN3 stage 1 | 1024 H100 | 128 | 16 × 16 | 208,320 | 1 | **78** | 1.35 | 79,872 |
| **FCN3 stage 2** | **512 A100** | 128 | 32 × 2 | 5,040 | **4** | **15** | 10.7 | 7,680 |
| FCN3 stage 3 | 256 H100 | 32 | 4 × 4 | 2,920 | 4 | **8** | 9.9 | 2,048 |
| ours A (done) | 4 A100-40GB | 1 | 32 × 1 | 332,424 | 1 | **46.3** | 0.50 | 185 |
| ours F (queued) | 8 A100-40GB (+4 spare) | 2 (+1) | 32 × 1 | 58,824 | 1 | 4.3–8.9 | 0.26–0.54 | 34–71 (+50 %) |
| ours G, 1 node | 4 A100-40GB | 1 | 32 × 1 | 277,020 | 1 | ~38.7 | ~0.50 | ~155 |
| ours G, 2 nodes | 8 A100-40GB (+4 spare) | 2 (+1) | 32 × 1 | 277,020 | 1 | 20.5–42.4 | 0.27–0.55 | 164–339 (+50 %) |

Reading it:

1. **Scale.** FCN3 spent **89,600 GPU-hours**; our finished base model (A) spent **185** —
   484× less. The A100 stage alone (7,680 A100-hours) is 41× A. The difference is mostly the
   problem, not the method: FCN3 trains on 721×1440 ERA5 (16× our 180×360 grid points) with a
   16-member ensemble in stage 1 (256 forecasts per step against our 32).
2. **Steps.** We take *more* optimizer steps: A = **1.60×** FCN3 stage 1, G ≈ 1.33×. Our steps
   are cheap (0.5 s against FCN3's 1.35 s in stage 1 and ~10 s in its rollout stages).
3. **The A100 stage is the analogue of our Stage-1 fine-tune arms**, not of A/F/G: a short,
   multi-step (4-step) continuation of a single-step pretrain. Ours roll 5 steps
   (`MULTISTEP=5`, depth-4) or 9 (T-d8), deterministic, at batch 16.
4. **Hardware fit.** FCN3 needs a spatial split (h2 w4 on 80 GB cards in stage 2, 16-fold in
   stage 3) to hold one ensemble member; each GPU holds 1/8 to 1/16 of a sample. We fit **4–8
   whole samples per 40 GB card** with no split — which is why we can run pure data
   parallelism, and why a stochastic stage 1 does not fit our cards (§4a).
5. **Efficiency.** The paper reports none (no throughput, scaling efficiency or utilization;
   Appendix G describes the decomposition only). Our one measurement: **56 % GPU kernel-busy
   at 1 node** (nsys 7591822), with 35 % of compute in copy/layout kernels. No efficiency
   comparison between the two can be made from published numbers.
