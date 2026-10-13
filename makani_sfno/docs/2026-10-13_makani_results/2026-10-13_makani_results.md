# Results: the makani SFNO emulator of E3SM — the 128-node run, A, B and the F family

*Written 2026-10-13. Every number in this document was recomputed for this analysis from the primary records
(training logs, rendered configs, rollout read-outs, scoring CSVs), unless it carries the tag **UNVERIFIED**.
UNVERIFIED means we copied it from a project document and did not re-derive it. The scripts and the
intermediate JSON that produce every table and figure are in [`scripts/`](scripts/) next to this file.
Reference keys: **[L#]** = published literature, **[R#]** = a primary record in this project. Both are listed at
the end.*

**Status (2026-10-13):** the experiment proposed in the Verdict, the 2 × 2 cross of training precision ×
rollout precision, is running: debug job 7742848 and debug-scaling job 7742849. It was pre-registered before
submission in `makani_sfno/docs/2026-10-13_precision_cross_prereg.md`. Its outcome will be added here as Result 14.

## 1. Question and design

**What we tested.** We asked which training recipe gives a makani spherical Fourier neural operator (SFNO;
[L1]) emulator of the E3SM climate model that does two things. First, it predicts the next 6-hour state
accurately. Second, its own predictions can be fed back as inputs for five years without the run breaking down
or its climate drifting away from E3SM's. The second requirement is the one that matters for climate use,
because a climate emulator is run *autoregressively*: each output becomes the next input, thousands of times.
The project goal is therefore long-rollout climate accuracy. Our evaluation was geared towards that, so we chose
a five-year, eight-start rollout protocol scored against the model's own run-to-run noise [R6]. We did not use
the trainer's single-step validation loss, which measures only one step ahead from a true state.

**Arms.** Table 1 lists the eight training runs and how they relate. All share the SFNO architecture, the E3SM
data pack and the optimizer's β₂ = 0.95 (read from each run's rendered config [R2]). The architecture
size (embedding 384, 8 layers) and the data years (training 2015–2044, validation 2045–2047) are UNVERIFIED
[R9]. *Bold* marks the setting that differs from A.

**Table 1. The arms.** Settings were read from each run's rendered config and log header [R1, R2]. "Depth" is
the number of chained 6-hour predictions inside one training loss: depth 1 is ordinary single-step training;
depth 5 means the model is trained on its own predictions for 4 extra steps [L11]. "EMA" is an exponential
moving average of the weights (decay 0.9995 per update), kept alongside the raw weights [L5, L4].
"Warm-restart cosine" (SGDR [L2]) repeats a cosine learning-rate decay every 20 epochs after a 3-epoch warmup.

| arm | starts from | output channels | depth | peak learning rate, schedule | global batch (GPUs) | EMA | training precision | epochs |
|---|---|---|---|---|---|---|---|---|
| 128-node run | random | 101 | 1 | **1e-3, reduce-on-plateau, no warmup** | **512 (512)** | no | bf16 UNVERIFIED | **100** |
| A | random | 101 | 1 | 2e-3, warm-restart cosine | 32 (4) | no | bf16 | 243 |
| B | **A epoch 243** | 101 | **5** | **4e-4, one cosine over 100 epochs (never reaches its minimum)** | **16 (8)** | no | bf16 | **24** |
| F-warm | **A epoch 243, cut to 99 outputs** | **99** | 1 | 2e-3, warm-restart cosine | 32 (8) | **yes** | bf16 | **43** |
| F-scratch | random | **99** | 1 | 2e-3, warm-restart cosine | 32 (8) | **yes** | bf16 | **43** |
| F-depth4 | **F-scratch epoch 23 (raw)** | **99** | **5** | **4e-4, one cosine over 100 epochs** | **16 (8)** | **yes** | bf16 | **63** |
| G1 | random | **99** | 1 | 2e-3, warm-restart cosine | 32 (8) | **yes** | **fp32** | **23** |
| G1-dry | random | **99** | 1 | 2e-3, warm-restart cosine | 32 (8) | **yes** | **fp32** | **63** |

The F arms drop the two soil channels (`SOILWATER_10CM`, `TSOI_10CM`) from the outputs. G1 is F-scratch rerun
without mixed precision: fp32 weights and activations, with the GPU's TF32 matrix multiplies left on
(`amp_mode = none` in its log). G1-dry is G1 with the dry-air mass constraint trained in
(`conserve_dry_air: True` in its config). That constraint shifts predicted surface pressure (PS) uniformly so
that global dry-air mass (PS minus g × column water vapour) stays equal to the input's, as in ACE [L12, L13].

```
lineage (arrows = initialized from)

  random ──► 128-node run (batch 512, 100 ep)

  random ──► A (batch 32, 243 ep) ──► B (depth 5, 24 ep)
                 │
                 └─(cut to 99 ch)──► F-warm (43 ep)

  random ──► F-scratch (99 ch, EMA, 43 ep) ──(raw e23)──► F-depth4 (depth 5, 63 ep)
  random ──► G1      (F-scratch recipe in fp32, 23 ep)
  random ──► G1-dry  (G1 + trained dry-air constraint, 63 ep)
```

**What each comparison isolates.**
- **128-node run vs A:** global batch, node count, learning-rate schedule and epochs all differ at once. The
  data, outputs, validation set and loss are the same. We therefore compare them at matched optimizer updates
  and at matched compute, not at matched epochs.
- **A vs F-scratch:** soil outputs, EMA and data-parallel width (4 vs 8 GPUs at the same global batch).
- **F-scratch vs G1:** training precision (bf16 vs fp32). Rollout precision also differs (below), so this pair
  measures the two together.
- **G1 vs G1-dry:** the trained-in dry-air constraint, on an otherwise identical recipe.
- **A vs B, and F-scratch vs F-depth4:** depth-5 fine-tuning, together with its different learning rate,
  schedule and batch.

**Replicates.** Each arm is **one training run with one random seed**. Differences between arms therefore carry
no estimate of run-to-run training noise. The replicates in this analysis are the **eight rollout starts**:
they measure how a fixed set of weights behaves from eight different initial states, not how the recipe would
behave if retrained.

**Evaluation.**
- *Validation loss:* the trainer's own loss on the same 512 validation samples for every arm
  (`n_eval_samples: 512`, `valid_autoreg_steps: 3` in every config [R2]), averaged over output channels. F arms
  average over 99 channels and A, B and the 128-node run over 101, so F-vs-A gaps of a few percent may partly
  reflect the two dropped channels.
- *Five-year protocol* [R6]: 8 deterministic rollouts start on 2044 frames 1092 + 16i (1–29 October 2044) and
  run to 2049-12-31, about 7,600 six-hour steps each, with sea-surface temperature prescribed from E3SM.
- *Survival:* a rollout is "finite" if the read-out did not truncate it [R3]. A finite rollout is not
  necessarily sane (Result 3).
- *Score A:* for each channel c and rollout, e_c is the area-weighted pattern root-mean-square error between
  the rollout's 2045–2047 time-mean map and E3SM's. We divide e_c by floor_c, the root-mean-square of the same
  quantity between pairs of rollouts of the reference models (their internal variability), to get r_c. We then
  average r_c within 24 channel groups, average the groups, and take the median over the 8 rollouts. A = 1
  would mean the time-mean error equals rollout-to-rollout noise; lower is better. W is the worst group's median
  r. Our recomputation from the per-channel files reproduces every published A to a relative error of 4e-16.
- *Paired comparisons:* we difference two candidates' per-rollout A at the same start date, report mean ± SE
  (SD/√8), and test with a paired t-test, 7 degrees of freedom [L19]. Survival fractions carry exact
  (Clopper–Pearson) 95% intervals [L16].
- *Gates* [R6]: (1) 8/8 finite; (2) water physical (global-mean column water vapour TMQ > 0 at every step and
  ≤ 1% of the globe negative in the last month); (3) |dPS| ≤ 1 hPa, where dPS is the change in global-mean
  surface pressure from the first to the last step; (4) at most 3 of the 24 groups drift (median channel drift
  between 2045 and 2047 > 2 interannual SDs of E3SM).

## 2. Results

### Result 1. The 128-node run made the same progress per update as A, at about 138 times the compute

As Table 2 and Figure 1 show, A reached the 128-node run's final validation loss (0.018297) at epoch 8, with
0.017797. That cost 1.48 node-hours, against 204.3 node-hours for the 128-node run, 138 times less. Here
node-hours = the sum of logged epoch times × nodes, which excludes job start-up. At matched optimizer updates
the two curves coincide: after 9,576 updates (epoch 7) A stood at 0.018343, against 0.018297 for the 128-node
run after 8,500 updates (0.25% apart). The 128-node run was still improving when it stopped, by 0.13% per
epoch over its last 10 epochs, so its final value is not converged. One seed per run means no SE is available
for these curves. A's final loss (0.012839 at epoch 243) is 29.8% below the 128-node run's (0.018297); it
comes from 39 times more updates (332,424 vs 8,500). The two runs also used different learning rates (1e-3
vs 2e-3) and schedules, so this is not a controlled test of batch size alone. Linear learning-rate scaling [L8]
would normally let a 16-times-larger batch take larger steps. Here it was not available: the learning-rate
ceiling measured at batch 32 and 48 lies between 2e-3 and 3e-3 (UNVERIFIED [R7, R8]), and the 128-node run used
1e-3 with no warmup. Therefore, at the learning rates usable for this model, validation loss tracked the
*number of updates* and not the number of samples per update, the behaviour expected above the critical batch
size, where a larger batch no longer buys a proportionally larger step [L6, L7].

![Figure 1](fig1_batch_and_cost.svg)

**Figure 1.** Validation loss of A (blue) and the 128-node run (orange), against optimizer updates (left) and
node-hours (right), both on log scales. Hover a marker for its value. The underlying values are in Table 2 and
in `scripts/epochs/`.

**Table 2. Training outcomes, recomputed from the per-epoch summaries in each training log [R1].** Global
batch = samples per epoch (config) ÷ logged steps per epoch. Compute = summed epoch times × nodes.

| arm | nodes | steps per epoch | global batch (samples per update) | epochs | updates | compute (node-hours) | median epoch time (s) | final validation loss | best validation loss (epoch) | best EMA validation loss (epoch) |
|---|---|---|---|---|---|---|---|---|---|---|
| 128-node run | 128 | 85 | 512 | 100 | 8,500 | 204.3 | 58 | 0.018297 | 0.018297 (e100) | no EMA kept |
| A | 1 | 1368 | 32 | 243 | 332,424 | 46.3 | 672 | 0.012839 | 0.012839 (e243) | no EMA kept |
| B | 2 | 2737 | 16 | 24 | 65,688 | 17.4 | 1305 | 0.013612 | 0.013411 (e1) | no EMA kept |
| F-warm | 2 | 1368 | 32 | 43 | 58,824 | 21.4 | 894 | 0.013064 | 0.013064 (e43) | 0.012891 (e30) |
| F-scratch | 2 | 1368 | 32 | 43 | 58,824 | 23.0 | 931 | 0.013776 | 0.013776 (e43) | 0.013758 (e41) |
| F-depth4 | 2 | 2737 | 16 | 63 | 172,431 | 95.1 | 2717 | 0.015320 | 0.015320 (e63) | 0.015220 (e63) |
| G1 | 2 | 1368 | 32 | 23 | 31,464 | 11.4 | 886 | 0.014311 | 0.014311 (e23) | 0.014308 (e23) |
| G1-dry | 2 | 1368 | 32 | 63 | 86,184 | 30.3 | 864 | 0.013361 | 0.013361 (e63) | 0.013331 (e59) |

### Result 2. Single-step validation loss separated the single-step arms by only 1–3%

As Table 2 shows, at matched epochs F-scratch's raw validation loss was 2.4% above A's at epoch 14, 2.5% at
epoch 23 and 2.8% at epoch 43. G1 was 2.0% above A at epoch 23. G1-dry was 1.4% above A at epochs 23 and 43 and
1.5% above at epoch 63. F-warm's best EMA value (0.012891, epoch 30) was 0.4% above A's best (0.012839).
Depth-5 fine-tuning raised the loss: B was 4.5–6.0% above A epoch 243 (its own starting point), and F-depth4
epoch 63 (0.015320) was 6.5% above its starting point, F-scratch epoch 23 (0.014380). The two single-step
arms that differ only in precision were 0.5% apart in EMA validation loss at epochs 15 and 19 (G1 0.014827 vs
F-scratch 0.014898; 0.014412 vs 0.014483). As Result 8 shows, the same pair differs 2-fold on the five-year
score. A depth-5 loss raises single-step loss because it trades one-step accuracy for multi-step
accuracy [L11]. In sequence models, the mismatch between training on true inputs and running on predicted inputs
is called exposure bias [L10]. Therefore single-step validation loss measures the error of one step from a true
state, not how errors grow when the model consumes its own predictions, which is the property a climate rollout
depends on.

### Result 3. Five-year survival separated checkpoints that validation loss ranked close together

As Table 3 shows, A's production checkpoint (epoch 243) did not survive a single 5-year rollout: 0 of 8 (95%
interval 0.00–0.37). All 8 went non-finite between steps 317 and 849 (0.22–0.58 years). B, fine-tuned at depth 5
from that same checkpoint, survived 8 of 8 at epochs 22 and 24 (interval 0.63–1.00), but only 1 of 8 at epoch 1.
Epoch 1 is the checkpoint B's trainer saved as "best", because it had the lowest validation loss. F-scratch's
raw weights survived at most 2 of 8 at every one of the 7 epochs tested (20–23 and 41–43). Its EMA weights at
epoch 23 (called FSEMA below) survived 8 of 8. F-depth4's raw weights survived 8 of 8 at epochs 14, 22 and 24.
"Finite" overstates health. G1 epoch 20 was 8/8 finite, but 7 of its 8 rollouts changed global PS by more than
100 hPa. G1-dry epoch 59 was 8/8 finite, but 3 of its 8 rollouts carried negative global water vapour
(budget read-outs [R4]). The rollout read-out's truncation check is thus a lower bound on failure, and gates 2
and 3 have to be read alongside it. Therefore survival is a property of the individual checkpoint, not of the
recipe.

**Table 3. Five-year survival, raw and EMA checkpoints of A, B, F-scratch and F-depth4 [R3].** One step =
6 hours; 1,460 steps = 1 year.

| checkpoint | rollouts finite to 2049-12-31 | 95% interval for the finite fraction | first non-finite step, range (years) |
|---|---|---|---|
| A e243 (raw) | 0/8 | 0.00 to 0.37 | 317 to 849 (0.22 to 0.58) |
| B e01 (raw; its lowest-validation-loss checkpoint) | 1/8 | 0.00 to 0.53 | 3639 to 7339 (2.49 to 5.03) |
| B e22 (raw) | 8/8 | 0.63 to 1.00 | none |
| B e24 (raw) | 8/8 | 0.63 to 1.00 | none |
| F-scratch e20 (raw) | 0/8 | 0.00 to 0.37 | 2007 to 2904 (1.37 to 1.99) |
| F-scratch e21 (raw) | 0/8 | 0.00 to 0.37 | 1250 to 3483 (0.86 to 2.39) |
| F-scratch e22 (raw) | 2/8 | 0.03 to 0.65 | 3668 to 7477 (2.51 to 5.12) |
| F-scratch e23 (raw) | 0/8 | 0.00 to 0.37 | 1766 to 4830 (1.21 to 3.31) |
| F-scratch e41 (raw) | 0/8 | 0.00 to 0.37 | 517 to 2673 (0.35 to 1.83) |
| F-scratch e42 (raw) | 0/8 | 0.00 to 0.37 | 662 to 1098 (0.45 to 0.75) |
| F-scratch e43 (raw) | 0/8 | 0.00 to 0.37 | 496 to 2511 (0.34 to 1.72) |
| F-scratch e23 (EMA) = FSEMA | 8/8 | 0.63 to 1.00 | none |
| F-depth4 e14 (raw) | 8/8 | 0.63 to 1.00 | none |
| F-depth4 e22 (raw) | 8/8 | 0.63 to 1.00 | none |
| F-depth4 e24 (raw) | 8/8 | 0.63 to 1.00 | none |

### Result 4. Within a training run, lower validation loss went with a worse five-year climate

As Table 5 shows, across the 29 F-scratch EMA checkpoints from epochs 15 to 43, EMA validation loss and score A
had a Spearman rank correlation [L18] of −0.41 (p = 0.026): later checkpoints with lower loss had worse
climates. Across the 9 G1 checkpoints (epochs 15–23) the correlation was −0.93 (p = 0.00024). Across 19 G1-dry
checkpoints it was +0.10 (p = 0.69), and across 16 F-depth4 checkpoints +0.00 (p = 0.99). Non-finite scores enter
these correlations as the largest ranks. Selecting the checkpoint with the lowest validation loss, as the
trainer's `best_ckpt` does, picked B epoch 1 (1/8 at 5 years) over B epochs 22 and 24 (8/8). Therefore, within a
run, validation loss either carried no information about the five-year score or pointed the wrong way.

### Result 5. Single-step checkpoints late in each learning-rate cycle failed most often

As Figure 2 shows, the failures of the three warm-restart arms clustered at the end of each 20-epoch cycle. That
is where the learning rate approaches its minimum (cycle position t = (epoch − 4) mod 20, reconstructed from
the config; the learning rate itself is not logged per epoch). Of the 24 EMA checkpoints in the last 4 epochs of
a cycle (t = 16–19), 8 kept 8/8 rollouts finite. Of the 33 earlier in the cycle, 25 did (Fisher exact test
[L17], p = 0.0025, odds ratio 0.16). This test counts finite rollouts, so G1 epochs 20–22 enter as late-cycle
successes even though 7, 8 and 8 of their rollouts changed global PS by more than 100 hPa. The test therefore
understates the effect. It also treats neighbouring epochs as independent, although they share most of their weights, which
makes the p-value optimistic. The pattern repeated in every cycle observed:
- F-scratch EMA epochs 20–22 and 38–43 were 0/8;
- G1 epochs 21–23 had A between 41 and 145;
- G1-dry epoch 43 was 0/8, and epochs 60–63 were at most 2/8.

It is not confined to the cycle end: F-scratch EMA epochs 33 and 34 (t = 9, 10) were also 0/8. The depth-5
F-depth4 run, whose learning rate decays once and never reaches its minimum, had 15 of 16 sampled epochs at
8/8 finite (Table 5). We hypothesize that as the learning rate falls, single-step training settles into
weights that fit the one-step target more tightly but amplify small departures from the training
distribution. Rollouts then leave that distribution within months, the instability mode that
[L14] traced to errors at small spatial scales. Two experiments would test this:
- Measure, for checkpoints across one cycle, the growth rate of a small perturbation over the first 100 rollout
  steps. It should rise towards the cycle end.
- Train one single-step arm with a constant learning rate and compare its rollouts at matched epochs.

Therefore the epoch at which a single-step checkpoint is taken is a hidden variable in every rollout
comparison, and a comparison of two recipes at one epoch each samples this cycle as much as the recipe.

![Figure 2](fig2_cycle_position.svg)

**Figure 2.** For the three single-step arms (F-scratch EMA, G1 EMA, G1-dry EMA): (a) the learning rate
reconstructed from the config; (b) five-year score A by checkpoint, log scale, with filled markers for 8/8
finite rollouts and hollow for fewer (values above 400, or non-finite, are drawn at the top line); (c) EMA
validation loss. Shaded bands are the last 4 epochs of each cycle. Table 5 holds every plotted value.

### Result 6. Ranked on score A, the fp32 runs led and B22 trailed most of the F family

As Table 4 and Table 5 show, the lowest scores among checkpoints with 8/8 finite rollouts came from the two
fp32 runs:
- G1-dry epochs 39 and 59 (1.52 each; epoch 59 minus epoch 39 = +0.005 ± 0.023, p = 0.84, but 3 of epoch 59's
  rollouts blew up in water, Result 3);
- G1 epochs 15–19 (1.59–1.72).

The best depth-5 checkpoint was F-depth4 EMA epoch 32 (1.99), and the best bf16 single-step checkpoint
F-scratch EMA epoch 30 (2.16). B epoch 22 scored 3.47. In paired tests against B22:
- G1-dry epoch 39 was lower by 1.942 ± 0.051 (p = 2.1e-9);
- G1 epoch 15 by 1.874 ± 0.045 (p = 1.3e-9);
- F-depth4 EMA epoch 32 by 1.483 ± 0.055 (p = 2.6e-8);
- FSEMA by 0.348 ± 0.055 (p = 4.0e-4);
- every one of these in 8 of 8 paired rollouts.

B epoch 24 was higher than B22 by 0.807 ± 0.048 (p = 6.1e-7). DRYAIR24 was higher by 0.886 ± 0.070
(p = 4.5e-6); it is B epoch 1 fine-tuned for 24 more epochs at depth 5 with the dry-air constraint trained in
and the learning rate annealed to 1e-6 (UNVERIFIED [R8]). G1-dry epoch 39 is lower than G1 epoch 15 by
0.068 ± 0.021 (p = 0.013), a lead of 3.3 SE. The pre-registered rule requires a lead above 3.3 SE on the sealed
2048–2049 reporting window [R6], which has not been opened. Each checkpoint is one point from one training run
(Result 5), and 96 candidate entries were scored on the same 2045–2047 window. Therefore the score orders these
checkpoints' five-year climates by more than their rollout-to-rollout noise, but the top of the ranking is
subject to selection bias (the "winner's curse") until the reporting window is scored.

**Table 4. Score A, gates and paired differences from B22, on the 2045–2047 selection window [R4].** Paired
mean ± SE over 8 start dates; p from a paired t-test with 7 degrees of freedom. dPS = global-mean PS, last step
minus first step. "Drifting groups" for the B, DRYAIR, FSEMA and F-depth4 rows may have been computed against
an E3SM reference file that failed its own consistency check (Limits). **Names:** B22 = B epoch 22 (raw
weights); FSE26 = F-scratch EMA epoch 26 (FSEMA = FSE23); FD22 = F-depth4 raw epoch 22; FDE32 = F-depth4 EMA
epoch 32; G1E15 = G1 EMA epoch 15; G1D39 = G1-dry EMA epoch 39; DRYAIR24 = epoch 24 of the B fine-tune with the
constraint trained in. Suffix `dryair` or `fix` = the dry-air correction applied at prediction time; `posclamp`
= a humidity positivity clamp applied at prediction time.

| candidate | finite (of 8) | score A | worst group W | A minus B22 (paired mean ± SE; p) | gate 2: water physical | gate 3: \|dPS\| ≤ 1 hPa (range, hPa) | drifting groups (gate 4: ≤ 3) |
|---|---|---|---|---|---|---|---|
| B22 | 8 | 3.47 | 6.35 | reference | pass | fail (-8.49 to -5.97) | 21 |
| B24 | 8 | 4.26 | 6.36 | +0.807 ± 0.048; 6.1e-07 | pass | fail (+12.74 to +17.71) | 23 |
| B22dryair | 8 | 3.52 | 6.29 | +0.067 ± 0.061; 0.311 | pass | pass (-0.17 to -0.08) | 21 |
| B24dryair | 8 | 4.29 | 6.18 | +0.827 ± 0.057; 1.7e-06 | pass | pass (+0.28 to +0.41) | 22 |
| B22posclamp | 8 | 3.46 | 6.36 | -0.002 ± 0.055; 0.966 | pass | fail (-7.09 to -3.59) | 21 |
| DRYAIR24 | 8 | 4.35 | 7.45 | +0.886 ± 0.070; 4.5e-06 | pass | pass (-0.24 to -0.14) | 22 |
| FSEMA | 8 | 3.10 | 4.55 | -0.348 ± 0.055; 4.0e-04 | pass | pass (-0.83 to +0.37) | 18 |
| FSE26 | 8 | 2.20 | 3.16 | -1.261 ± 0.044; 1.7e-08 | pass | fail (+1.74 to +4.01) | 12 |
| FSE30 | 8 | 2.16 | 4.30 | -1.296 ± 0.044; 1.4e-08 | fail (6 of 8 members) | pass (-0.38 to +0.70) | 9 |
| FD22 | 8 | 4.46 | 13.04 | +0.981 ± 0.033; 1.4e-08 | fail (8 of 8 members) | fail (+35.14 to +40.04) | 22 |
| FD24 | 8 | 7.17 | 21.32 | +3.703 ± 0.049; 1.9e-11 | pass | fail (-107.94 to -106.04) | 23 |
| FD24dryair | 8 | 6.86 | 19.38 | +3.421 ± 0.056; 8.3e-11 | pass | pass (-0.69 to -0.56) | 21 |
| FDE28 | 8 | 2.02 | 4.81 | -1.438 ± 0.050; 1.5e-08 | fail (4 of 8 members) | fail (-33.14 to -31.21) | 17 |
| FDE32 | 8 | 1.99 | 4.56 | -1.483 ± 0.055; 2.6e-08 | fail (8 of 8 members) | fail (-34.38 to -32.95) | 18 |
| G1E15 | 8 | 1.59 | 2.63 | -1.874 ± 0.045; 1.3e-09 | pass | fail (-2.40 to -1.36) | 15 |
| G1E15fix | 8 | 1.68 | 2.57 | -1.783 ± 0.039; 6.0e-10 | pass | pass (-0.44 to -0.35) | 17 |
| G1E18 | 8 | 1.71 | 2.91 | -1.755 ± 0.053; 6.2e-09 | pass | fail (+1.04 to +2.45) | 7 |
| G1D20 | 8 | 2.09 | 3.32 | -1.387 ± 0.072; 2.5e-07 | fail (6 of 8 members) | pass (+0.10 to +0.71) | 19 |
| G1D39 | 8 | 1.52 | 2.52 | -1.942 ± 0.051; 2.1e-09 | pass | pass (-0.30 to -0.18) | 14 |
| G1D59 | 8 | 1.52 | 2.34 | -1.937 ± 0.037; 2.4e-10 | fail (7 of 8 members) | fail (-2.3e+20 to +0.88) | 16 |

### Result 7. Only G1-dry epoch 39 passed gates 1–3 without a prediction-time correction, and no checkpoint passed gate 4

As Tables 4 and 5 show, eight candidates pass gates 1–3:
- G1-dry epoch 39 (A 1.52);
- G1 epoch 15 with the dry-air correction applied at prediction time (1.68);
- G1 epoch 19 with it (1.74);
- FSEMA (3.10);
- B22 with the prediction-time correction (3.52);
- B24 with it (4.29);
- DRYAIR24 (4.35);
- F-depth4 raw epoch 24 with it (6.86).

G1-dry epoch 39 scored lower than the best corrected candidate, G1 epoch 15 + correction, by 0.158 ± 0.021
(p = 1.4e-4), and lower than FSEMA by 1.594 ± 0.024 (p = 5.3e-11). No candidate passed gate 4. The fewest drifting groups was 7
(G1 epoch 18), against a limit of 3; G1-dry epoch 39 had 14. The drift census counts groups whose global mean
moves by more than 2 interannual SDs between 2045 and 2047. Therefore no checkpoint satisfies the full
pre-registered definition of a usable climate model, and slow drift, not survival or mass, is now the binding
constraint for every candidate.

### Result 8. Training in fp32 halved score A at matched epochs

As Table 4 and Table 5 show, at matched epochs G1 (fp32) scored lower than F-scratch (bf16):
- epoch 15: by 2.091 ± 0.105 (p = 2.0e-7);
- epoch 16: by 2.120 ± 0.035 (p = 8.6e-11);
- epoch 18: by 1.736 ± 0.020 (p = 8.3e-12);
- epoch 19: by 1.459 ± 0.193 (p = 1.3e-4);
- 8 of 8 paired rollouts at every epoch.

Both runs used the same recipe and, per the pre-registration, makani's default seed (UNVERIFIED [R6]). The fp32
run reached roughly half the score of the bf16 run (1.59 vs 3.62 at epoch 15), while their validation losses
differed by 0.5% (Result 2). The rollouts also differed in precision: G1's member logs record `feedback_dtype=torch.float32` and F-scratch's
`torch.bfloat16` [R3], so this pair cannot separate training precision from rollout precision. bf16 keeps 8
significant bits, about 2–3 decimal digits [L15], so rounding the carried state at each of about 7,600 steps is
a plausible source of accumulated drift. Therefore fp32 training, fp32 rollouts or both reduce long-rollout
error, at no measurable cost in single-step loss.

### Result 9. Training the dry-air constraint in pinned mass but cost accuracy and water positivity at matched epochs

As Table 5 shows, at epochs 15–19 G1-dry scored higher (worse) than G1:
- epoch 15: by 0.625 ± 0.043 (p = 1.6e-6);
- epoch 16: by 0.590 ± 0.060 (p = 2.4e-5);
- epoch 17: by 1.137 ± 0.056 (p = 1.8e-7);
- epoch 18: by 0.577 ± 0.035 (p = 6.9e-7);
- epoch 19: by 1.043 ± 0.042 (p = 4.1e-8);
- 0 of 8 paired rollouts better at any epoch.

It held global PS within 1.16 hPa at every epoch from 15 to 23, against 2.40–8.00 hPa for G1 at epochs 15–19.
It failed the water gate at all 9 epochs from 15 to 23, where G1 passed at all 5 of epochs 15–19. G1-dry epoch
39, in the second cycle, passed both (|dPS| ≤ 0.30 hPa, water gate pass, A 1.52). ACE2 pairs dry-air
conservation with a water-positivity step [L13]; our model has the first and not the second. Therefore the
constraint removes the dry-air mass leak, but with PS pinned any remaining mass error has to appear as a change
in column water, so water positivity becomes the binding gate.

### Result 10. The prediction-time dry-air correction pinned PS on every checkpoint, with a checkpoint-dependent effect on accuracy

As Table 4b shows, applying the dry-air correction only at prediction time reduced the largest |dPS| over 8
rollouts to below 0.7 hPa on all 7 checkpoints tested (for example B22 8.49 → 0.17 hPa, F-depth4 epoch 24
107.94 → 0.69 hPa). Its effect on A varied:
- no detectable change on B22 (+0.067 ± 0.061, p = 0.31) or B24 (+0.020 ± 0.042, p = 0.65);
- worse on G1 epochs 15, 16 and 19 (+0.090 ± 0.021, p = 0.004; +0.221 ± 0.019, p = 7.5e-6; +0.117 ± 0.022,
  p = 0.001);
- worse on F-depth4 raw epoch 22 (+0.187 ± 0.033, p = 7.1e-4);
- better on F-depth4 raw epoch 24 (−0.282 ± 0.019, p = 1.4e-6).

Whether it helps or hurts the climate depends on where the uncorrected checkpoint's error sits, so each
checkpoint has to be scored with and without it. Therefore the correction controls the mass budget by
construction, but it does not repair the circulation that produced the drift.

**Table 4b. Prediction-time dry-air correction, paired with and without, same checkpoint and start dates [R4].**
Names as in Table 4.

| checkpoint | A without fix | A with fix | with minus without (paired ± SE; p) | \|dPS\| without → with (max over members, hPa) |
|---|---|---|---|---|
| B22 | 3.47 | 3.52 | +0.067 ± 0.061; 0.311 | 8.49 → 0.17 |
| B24 | 4.26 | 4.29 | +0.020 ± 0.042; 0.650 | 17.71 → 0.41 |
| FD22 | 4.46 | 4.63 | +0.187 ± 0.033; 7.1e-04 | 40.04 → 0.38 |
| FD24 | 7.17 | 6.86 | -0.282 ± 0.019; 1.4e-06 | 107.94 → 0.69 |
| G1E15 | 1.59 | 1.68 | +0.090 ± 0.021; 0.004 | 2.40 → 0.44 |
| G1E16 | 1.60 | 1.82 | +0.221 ± 0.019; 7.5e-06 | 8.00 → 0.67 |
| G1E19 | 1.63 | 1.74 | +0.117 ± 0.022; 0.001 | 4.55 → 0.34 |

### Result 11. Weight averaging (EMA) carried most of the F family's accuracy

As Table 4 and Table 5 show, for F-depth4 at epoch 24 the raw weights scored higher than the EMA weights by
4.198 ± 0.021 (A 7.17 vs 2.95; p = 2.1e-14, 8 of 8 rollouts). Raw epoch 22 scored above EMA epoch 20 by
1.837 ± 0.029 (p = 6.5e-11), and raw epoch 14 above EMA epochs 12 and 16 by 1.092 ± 0.030 (p = 3.0e-9) and
1.722 ± 0.017 (p = 2.3e-12). For F-scratch
survival, the 7 raw/EMA pairs at matched epochs (20–23, 41–43) gave 2 of 56 raw rollouts and 8 of 56 EMA
rollouts finite. The whole difference came from epoch 23 (raw 0/8, EMA 8/8); at epoch 22 raw did better (2/8 vs
0/8), so averaging does not rescue a checkpoint from the late-cycle failure of Result 5. Therefore averaging
the weights over roughly the last 2,000 updates (1/(1 − 0.9995); 1.5 epochs of F-scratch, 0.7 epochs of
F-depth4) reduces the time-mean climate error, consistent with averaging moving the solution towards a flatter
region of the loss surface [L4, L5], the property linked to better generalization in [L9].

### Result 12. Depth-5 fine-tuning bought survival and cost mass conservation

As Table 3 shows, depth-5 fine-tuning took A's epoch 243 from 0/8 to 8/8 (B epochs 22 and 24), and F-scratch's
raw epochs 22–23 (2/8, 0/8) to 8/8 (F-depth4 raw epochs 14, 22, 24). F-depth4's EMA checkpoints were 8/8 finite
at 15 of 16 sampled epochs. The same fine-tunes let global-mean PS drift over the five years:
- −5.97 to −8.49 hPa for B22, +12.74 to +17.71 hPa for B24, and +35.14 to +40.04 hPa for F-depth4 raw epoch 22;
- 13.5–54.9 hPa (largest rollout) across the F-depth4 EMA sweep, excluding epoch 48, which had non-finite
  rollouts;
- against −0.83 to +0.37 hPa for FSEMA.

On score A, F-depth4's best EMA checkpoint (epoch 32) was 0.187 ± 0.017 below F-scratch's best, EMA epoch 30
(p = 9.4e-6). Its raw epoch 22 was 1.329 ± 0.040 above FSEMA (p = 5.6e-9). The first fine-tune epoch added almost nothing (B epoch 1 was 1/8),
consistent with that epoch running at a constant learning rate of 4e-6 (UNVERIFIED [R8]). Therefore training on
4 extra steps of its own predictions teaches the model to keep rollouts finite, the purpose of multi-step
training [L11], but it does not constrain the slow global-mean quantities that a 30-hour loss window (5 steps of 6 hours) cannot see.

### Result 13. A's checkpoints failed the one-year screen at most epochs, so A could not seed a selection

From the one-year screens [R5], 1 rollout per start:
- non-finite within a year at epochs 43, 63, 103, 123, 163, 183, 200, 220 and 243, between steps 490 and 1,291;
- finite with PS within 15 hPa of E3SM at epochs 21 (+14.8 and +12.1 hPa at two starts), 83 (−6.8 hPa) and
  143 (+8.3 hPa), the last two from one start each;
- finite but with PS 527–1,442 hPa below E3SM at epochs 22 and 23.

Only epoch 243 went through the five-year protocol (0/8, Table 3). Every A checkpoint that was screened sits
at cycle position 16–19: epochs 21–23, the cycle ends 43 to 243 (kept as snapshot-ensemble members [L3]), and
epochs 200 and 220. Therefore A cannot test Result 5's position effect, and its only five-year number comes
from a late-cycle checkpoint.

## 3. Sweeps and ablations: every level tested

Table 5 lists every EMA epoch of the four swept runs that went through the five-year protocol. Per component:
- **Rollout survival:** depth-5 training carried it (Results 3 and 12). Among single-step checkpoints,
  learning-rate cycle position mattered more than any recipe setting (Result 5). EMA mattered at one of seven
  matched epochs (Result 11).
- **Climate accuracy (score A):** precision carried the largest effect (Result 8, about 2-fold), followed by EMA
  for depth-5 weights (Result 11).
- **Mass conservation:** the dry-air constraint carried it, trained in or applied at prediction time
  (Results 9 and 10). Depth-5 training worked against it (Result 12).
- **Validation loss:** no component predicted the five-year outcome (Results 2 and 4).

**Table 5. Every swept checkpoint [R1, R3, R4].** "Finite" counts rollouts the read-out did not truncate.
"Max |dPS|" is the largest absolute PS change over the 8 rollouts; "not finite" means at least one rollout's
budget overflowed. The water gate is gate 2. Cycle position is (epoch − 4) mod 20 for the warm-restart arms.

**F-scratch, EMA weights, bf16 rollouts**

| epoch | cycle position (0–19) | EMA validation loss | finite (of 8) | score A | max \|dPS\| (hPa) | water gate |
|---|---|---|---|---|---|---|
| 15 | 11 | 0.014898 | 8 | 3.62 | 8.23 | pass |
| 16 | 12 | 0.014753 | 8 | 3.70 | 14.75 | fail |
| 17 | 13 | 0.014639 | 4 | 26.92 | not finite | fail |
| 18 | 14 | 0.014550 | 8 | 3.45 | 17.29 | fail |
| 19 | 15 | 0.014483 | 8 | 3.25 | 11.70 | pass |
| 20 | 16 | 0.014436 | 0 | > 400 | not finite | fail |
| 21 | 17 | 0.014407 | 0 | > 400 | not finite | fail |
| 22 | 18 | 0.014389 | 0 | > 400 | not finite | fail |
| 23 | 19 | 0.014381 | 8 | 3.10 | 0.83 | pass |
| 24 | 0 | 0.015244 | 8 | 4.31 | 7.37 | fail |
| 25 | 1 | 0.014858 | 8 | 2.71 | 20.28 | pass |
| 26 | 2 | 0.014697 | 8 | 2.20 | 4.01 | pass |
| 27 | 3 | 0.014587 | 8 | 2.46 | 15.90 | pass |
| 28 | 4 | 0.014483 | 8 | 2.84 | 1.09 | fail |
| 29 | 5 | 0.014379 | 8 | 2.95 | 2.86 | fail |
| 30 | 6 | 0.014278 | 8 | 2.16 | 0.70 | fail |
| 31 | 7 | 0.014188 | 8 | 3.45 | 6.93 | pass |
| 32 | 8 | 0.014101 | 8 | 2.68 | 15.42 | fail |
| 33 | 9 | 0.014028 | 0 | 10.31 | not finite | fail |
| 34 | 10 | 0.013959 | 0 | 11.13 | not finite | fail |
| 35 | 11 | 0.013901 | 7 | 3.24 | not finite | fail |
| 36 | 12 | 0.013854 | 3 | 4.67 | not finite | fail |
| 37 | 13 | 0.013820 | 5 | 3.20 | not finite | fail |
| 38 | 14 | 0.013789 | 0 | > 400 | not finite | fail |
| 39 | 15 | 0.013772 | 0 | 5.61 | not finite | fail |
| 40 | 16 | 0.013761 | 0 | > 400 | not finite | fail |
| 41 | 17 | 0.013758 | 0 | > 400 | not finite | fail |
| 42 | 18 | 0.013761 | 0 | > 400 | not finite | fail |
| 43 | 19 | 0.013763 | 0 | > 400 | not finite | fail |

**G1, EMA weights, fp32 rollouts**

| epoch | cycle position (0–19) | EMA validation loss | finite (of 8) | score A | max \|dPS\| (hPa) | water gate |
|---|---|---|---|---|---|---|
| 15 | 11 | 0.014827 | 8 | 1.59 | 2.40 | pass |
| 16 | 12 | 0.014684 | 8 | 1.60 | 8.00 | pass |
| 17 | 13 | 0.014570 | 8 | 1.72 | 7.11 | pass |
| 18 | 14 | 0.014480 | 8 | 1.71 | 2.45 | pass |
| 19 | 15 | 0.014412 | 8 | 1.63 | 4.55 | pass |
| 20 | 16 | 0.014364 | 8 | 2.25 | 1417.31 | fail |
| 21 | 17 | 0.014334 | 8 | 41.03 | 3144.48 | fail |
| 22 | 18 | 0.014316 | 8 | 99.69 | not finite | fail |
| 23 | 19 | 0.014308 | 6 | 144.97 | not finite | fail |

**G1-dry, EMA weights, fp32 rollouts with the dry-air constraint**

| epoch | cycle position (0–19) | EMA validation loss | finite (of 8) | score A | max \|dPS\| (hPa) | water gate |
|---|---|---|---|---|---|---|
| 15 | 11 | 0.014754 | 8 | 2.22 | 1.08 | fail |
| 16 | 12 | 0.014608 | 8 | 2.11 | 0.48 | fail |
| 17 | 13 | 0.014491 | 8 | 2.83 | 0.57 | fail |
| 18 | 14 | 0.014399 | 8 | 2.29 | 0.92 | fail |
| 19 | 15 | 0.014329 | 8 | 2.65 | 1.16 | fail |
| 20 | 16 | 0.014279 | 8 | 2.09 | 0.71 | fail |
| 21 | 17 | 0.014246 | 8 | 2.79 | 0.25 | fail |
| 22 | 18 | 0.014227 | 8 | 2.76 | 0.23 | fail |
| 23 | 19 | 0.014218 | 8 | 2.26 | 0.45 | fail |
| 39 | 15 | 0.013594 | 8 | 1.52 | 0.30 | pass |
| 40 | 16 | 0.013582 | 7 | 1.77 | not finite | fail |
| 41 | 17 | 0.013577 | 7 | 1.76 | not finite | fail |
| 42 | 18 | 0.013577 | 6 | 1.77 | not finite | fail |
| 43 | 19 | 0.013579 | 0 | > 400 | not finite | fail |
| 59 | 15 | 0.013331 | 8 | 1.52 | not finite | fail |
| 60 | 16 | 0.013331 | 1 | 1.60 | not finite | fail |
| 61 | 17 | 0.013335 | 0 | > 400 | not finite | fail |
| 62 | 18 | 0.013340 | 2 | > 400 | not finite | fail |
| 63 | 19 | 0.013345 | 0 | > 400 | not finite | fail |

**F-depth4, EMA weights, bf16 rollouts (no warm restarts)**

| epoch | EMA validation loss | finite (of 8) | score A | max \|dPS\| (hPa) | water gate |
|---|---|---|---|---|---|
| 4 | 0.015429 | 8 | 4.29 | 54.51 | pass |
| 8 | 0.015484 | 8 | 3.30 | 53.11 | pass |
| 12 | 0.015494 | 8 | 3.74 | 28.19 | pass |
| 16 | 0.015483 | 8 | 3.12 | 38.15 | pass |
| 20 | 0.015462 | 8 | 2.61 | 54.90 | pass |
| 24 | 0.015434 | 8 | 2.95 | 31.94 | pass |
| 28 | 0.015409 | 8 | 2.02 | 33.14 | fail |
| 32 | 0.015382 | 8 | 1.99 | 34.38 | fail |
| 36 | 0.015357 | 8 | 2.71 | 48.13 | pass |
| 40 | 0.015331 | 8 | 2.22 | 43.73 | pass |
| 44 | 0.015308 | 8 | 2.34 | 42.84 | fail |
| 48 | 0.015287 | 6 | 3.53 | not finite | fail |
| 52 | 0.015264 | 8 | 3.22 | 28.56 | pass |
| 56 | 0.015248 | 8 | 2.63 | 19.70 | pass |
| 60 | 0.015233 | 8 | 3.26 | 17.99 | pass |
| 63 | 0.015220 | 8 | 4.07 | 13.50 | pass |

Further arms rolled out over five years, with every level tested:
- B22 with a positivity clamp on 18 humidity channels at prediction time: no change in A (−0.002 ± 0.055,
  p = 0.97).
- B22 and B24 with the dry-air correction at prediction time: Table 4b.
- DRYAIR at epochs 1 and 24: epoch 1 was 0/8 finite, epoch 24 8/8.
- SOILFIX (soil-moisture correction trained in) at epochs 21 and 24: 7/8 and 5/8 finite.
- F-scratch EMA with the prediction-time correction: 2/8 finite.
- The stop-early trained-in dry-air test on F-depth4:
  - EMA epoch 2: 8/8 finite, A 96.6;
  - EMA epoch 4: 8/8, A 15.7;
  - raw epoch 2: 8/8, A 269.1;
  - raw epoch 4: 0/8.

## 4. Interpretation

Taken together, the results describe three separate failure modes. Each responds to a different component.
1. **Rollouts that go non-finite within months.** Training on the model's own predictions (depth 5) removes
   this. Among single-step checkpoints it depends on the learning-rate cycle. Therefore it is an
   error-amplification problem of the autoregressive loop, invisible to a one-step loss and addressed directly
   by multi-step training [L10, L11, L14].
2. **Slow drift of global mass.** A trained-in or prediction-time dry-air constraint removes it, and depth-5
   training worsens it. Therefore it is a conservation problem that a loss computed over 6–30 hours does not
   penalize, and it needs an explicit constraint, as in ACE [L12, L13].
3. **Time-mean climate error among rollouts that survive and conserve mass.** fp32 and weight averaging reduce
   it. We hypothesize that bf16 rounding of the fed-back state adds a small bias at every step, which
   accumulates over 7,600 steps. The test is the 2 × 2 cross of training precision × rollout precision on
   F-scratch and G1 at the same epochs, which needs rollouts only, no training. Therefore this error depends on
   the numerical precision of the carried state and on the flatness of the solution, not on the loss value.

**The 128-node run** belongs to none of these. Its only measured property is that its batch was 16 times larger
for the same per-update progress (Result 1); no rollout of the 128-node checkpoint exists. Therefore what we know
against it is its cost; its stability is unmeasured.

## 5. Limits

- **One training seed per arm.** Every recipe difference above is a difference between two training runs, with
  no estimate of seed-to-seed variation. The paired SEs measure rollout noise for fixed weights, not training
  noise. Claims are about checkpoints, not recipes.
- **The 128-node checkpoint was never rolled out** at any length. Its fabric (TCP rather than the Slingshot
  network, UNVERIFIED [R7]) affects only its speed. We did not check whether its bf16 setting matches A's: the
  log header lacks the line (UNVERIFIED [R9]).
- **Precision is confounded** between training and rollout (Result 8).
- **Selection window.** Every A, W and drift-census number is on 2045–2047, the window used to select. 96
  candidate entries were scored on it. The sealed 2048–2049 window and the April-start set R [R6] have not been run, so
  the top of Table 4 is subject to selection bias.
- **The drift census for the rows scored on 2026-10-10** (score tags `ref1`, `ref2`, `fd1`: B, DRYAIR, FSEMA,
  F-depth4) may have used the E3SM annual-mean file built at a 5-step stride. Its builder job printed
  `ERROR TRUTH_GM_EQUIV max|diff|/interannual_sd=1.205 (tol 0.1)` [R10]; we read that value from the log and did
  not recompute it (UNVERIFIED). The score logs do not record which file each census used (UNVERIFIED).
  Gate 4 fails for every candidate under either file.
- **"Finite" undercounts failure** (Result 3). We did not re-derive the read-out's truncation rule.
- **The learning rate per epoch was reconstructed** from the config (warm-restart cosine, T₀ = 20, 3 warmup
  epochs, minimum 1e-6). It is not logged. The step convention (once per epoch) is UNVERIFIED [R8].
- **Not run:** the variability gate (gate 5) for any candidate, a truth-vs-truth noise floor, a seasonal-cycle
  metric, a single-step control of A trained for as many epochs as B, and a depth-5 fine-tune of G1 or G1-dry.
- **Figures were not rendered for visual inspection.** No SVG renderer exists on the login node. We checked the
  layout geometrically (no text overlaps, nothing outside the canvas, no line crossing a label).

## 6. Verdict

**Decided:** select checkpoints on five-year rollouts, not validation loss, and train at batch 32, not 512; no checkpoint yet passes every gate (lead: G1-dry EMA epoch 39). **Next:** the 2 × 2 training-precision × rollout-precision cross (rollouts only), then freeze finalists and open the 2048–2049 reporting window.

## References

**Literature**
- [L1] Bonev, B., Kurth, T., Hundt, C., Pathak, J., Baust, M., Kashinath, K., Anandkumar, A. (2023). Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere. *ICML*. arXiv:2306.03838.
- [L2] Loshchilov, I., Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. *ICLR*. arXiv:1608.03983.
- [L3] Huang, G., Li, Y., Pleiss, G., Liu, Z., Hopcroft, J. E., Weinberger, K. Q. (2017). Snapshot Ensembles: Train 1, Get M for Free. *ICLR*. arXiv:1704.00109.
- [L4] Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., Wilson, A. G. (2018). Averaging Weights Leads to Wider Optima and Better Generalization. *UAI*. arXiv:1803.05407.
- [L5] Polyak, B. T., Juditsky, A. B. (1992). Acceleration of Stochastic Approximation by Averaging. *SIAM J. Control Optim.* 30(4), 838–855.
- [L6] McCandlish, S., Kaplan, J., Amodei, D., OpenAI Dota Team (2018). An Empirical Model of Large-Batch Training. arXiv:1812.06162.
- [L7] Shallue, C. J., Lee, J., Antognini, J., Sohl-Dickstein, J., Frostig, R., Dahl, G. E. (2019). Measuring the Effects of Data Parallelism on Neural Network Training. *JMLR* 20(112). arXiv:1811.03600.
- [L8] Goyal, P., et al. (2017). Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour. arXiv:1706.02677.
- [L9] Keskar, N. S., Mudigere, D., Nocedal, J., Smelyanskiy, M., Tang, P. T. P. (2017). On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima. *ICLR*. arXiv:1609.04836.
- [L10] Bengio, S., Vinyals, O., Jaitly, N., Shazeer, N. (2015). Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. *NeurIPS*. arXiv:1506.03099.
- [L11] Lam, R., et al. (2023). Learning skillful medium-range global weather forecasting. *Science* 382(6677), 1416–1421. arXiv:2212.12794.
- [L12] Watt-Meyer, O., et al. (2023). ACE: A fast, skillful learned global atmospheric model for climate prediction. arXiv:2310.02074.
- [L13] Watt-Meyer, O., et al. (2024). ACE2: Accurately learning subseasonal to decadal atmospheric variability and forced responses. arXiv:2411.11268.
- [L14] Chattopadhyay, A., Hassanzadeh, P. (2023). Long-term instabilities of deep learning-based digital twins of the climate system: The cause and a solution. arXiv:2304.07029.
- [L15] Kalamkar, D., et al. (2019). A Study of BFLOAT16 for Deep Learning Training. arXiv:1905.12322.
- [L16] Clopper, C. J., Pearson, E. S. (1934). The use of confidence or fiducial limits illustrated in the case of the binomial. *Biometrika* 26(4), 404–413.
- [L17] Fisher, R. A. (1922). On the interpretation of χ² from contingency tables, and the calculation of P. *J. R. Stat. Soc.* 85(1), 87–94.
- [L18] Spearman, C. (1904). The proof and measurement of association between two things. *Am. J. Psychol.* 15(1), 72–101.
- [L19] Student (1908). The probable error of a mean. *Biometrika* 6(1), 1–25.

**Primary records** (`$M` = `/eagle/projects/lighthouse-uchicago/members/mehta5`; `RUNS` = `$M/runs/makani_mn_scaling`; `EVAL` = `$M/runs/makani_eval`)
- [R1] Training logs, `RUNS/<tag>.log`: `prod128_alldata_v2` (job 7566145), `prod1n_b32_sgdr` (7585080), `nf4_prod_b16_r1` (7630639), `f_nosoil_2n_b32_e43_warm` (7660250), `f_nosoil_2n_b32_e23_scratch` (epochs 24–43; epochs 1–23 from job stdout `.claude/worktrees/makani-ace2-ports/makani_sfno/makani_mn_scaling.o7718436`), `fsF_anneal_nf4_b16_rtmax100_scratch` (epochs 17–63; epochs 1–16 from `.claude/worktrees/arm-fsF-nf4-bsched/makani_sfno/makani_mn_scaling.o7719538`), `f_nosoil_2n_b32_e23_scratch_fp32` (7731784), `f_nosoil_2n_b32_e63_scratch_fp32_dryair` (7735732).
- [R2] Rendered configs, `RUNS/e3sm_mn_scaling.<tag>.yaml`, same tags.
- [R3] Five-year rollout read-outs and member logs, `EVAL/climate_protocol_<job>/readout.log`, jobs 7649597, 7707597, 7711720, 7720705, 7721150, 7721233, 7724006, 7727625, 7727731, 7731448, 7731806, 7731807, 7732564, 7732686, 7732901, 7732928, 7733015, 7733110, 7733210, 7733258, 7734370, 7734960, 7734961, 7735162, 7738106, 7738107, 7738185, 7738283.
- [R4] Protocol scores, `EVAL/protocol_scores/<tag>_{channels,scores,budget,census}`, tags `ref1`, `ref2`, `fd1`, `g1fs`, `dry1`, `g1fix`, `g1dry`, `g1dry_c`, `g1dry_d`.
- [R5] One-year screens, `EVAL/climate_screen_<job>/screen_summary.csv`, jobs 7649647, 7720637, 7720644, 7721629, 7725618, 7725885.
- [R6] Pre-registration: `makani_sfno/docs/2026-10-09_pipeline_comparison_prereg.md`, branch `docs/makani-knowledge`.
- [R7] `makani_bench_report.md` §4 (128-node run) and §5j–5k (learning-rate ceiling; the 1-node run).
- [R8] `makani_sfno/KNOWLEDGE.md` §2 and §6, branch `docs/makani-knowledge`.
- [R9] `makani_sfno/docs/2026-08-27_prod128_alldata_checkpoint_usage.md`, branch `docs/makani-knowledge`.
- [R10] E3SM annual-mean reference build, job 7732604: `.claude/worktrees/eval-protocol/makani_sfno/makani_truth_gm.o7732604`. Score jobs: `makani_protocol_score.o{7734901,7735052,7735690,7738179,7738245,7739744}` in the same directory.
