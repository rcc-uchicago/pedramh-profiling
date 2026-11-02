# Configurations: the makani SFNO training runs and rollout variants on E3SM

*Written 2026-11-02. This is a record of **what was run and what was compared**, not an analysis: it reports
settings, sizes and costs, and no scores, rankings or verdicts on the models.

## 1. Question and design

**What the set of runs was built to compare.** The project needs a makani spherical Fourier neural operator
(SFNO) emulator of E3SM whose own predictions can be fed back as inputs for five years without the run breaking
down or its climate drifting. The training campaign was geared towards finding the recipe that does this, so we
chose to vary one recipe component at a time where the queue allowed it, starting from two baselines:

- **Baselines.** **A** is the single-step production run trained from scratch on one node. The **128-node run**
  is the same model and data at a 16-times-larger global batch, kept as the large-batch reference.
- **Depth fine-tunes of the A lineage.** **B** fine-tunes A's best checkpoint with a 5-step (30 h) loss.
  **DRYAIR** and **SOILFIX** fine-tune B's best checkpoint again, with a conservation constraint trained in.
  - **DRYAIR** (`fs_anneal_dryair_nf4_b16_rb01`, job 7709268) trains with the dry-air mass constraint on
    (`conserve_dry_air: True`; `DryAirFix` in `sfno_training/models/mass_fix.py`, makani's adaptation of ACE2's
    `conserve_dry_air`). After every predicted 6-h step, in training, validation and rollout alike, it does three
    things:
    - It computes dry-air surface pressure as PS − g·TMQ. Here PS is the surface pressure in Pa, TMQ the column
      water vapour in kg m⁻², and g = 9.81 m s⁻², so g·TMQ is the water column's share of PS in Pa.
    - It shifts the predicted PS by one globally uniform amount, so that the area-weighted global mean of dry-air
      pressure equals the input state's. TMQ is left unchanged, and the correction is applied at full strength on
      every call, with no ramp.
    - In effect, the global dry-air mass of the atmosphere cannot change from one step to the next. The model
      learns its weights with that constraint inside the 5-step loss.
    - The fix was checked numerically against ACE2's own `_force_conserve_dry_air` (job 7721257 UNVERIFIED).
  - **SOILFIX** (`fs_anneal_soilfix_nf4_b16_rb01`, job 7715005) is the same fine-tune with a soil-moisture
    constraint in place of the dry-air one (`conserve_soil_moisture: True`; `SoilMoistureFix` in
    `sfno_training/models/soil_moisture_fix.py`). It reimplements the intent of ACE2's land
    `soil_moisture_corrector`, whose code is not in our vendored ACE2 tree. Per grid cell:
    - where the step's input 10-cm soil temperature `TSOI_10CM` is below 272.0 K (frozen soil), the predicted
      10-cm soil water `SOILWATER_10CM` is replaced by its input value, so frozen soil water cannot change;
    - predicted soil water is then floored at 0, so it is never negative.

    Unlike the dry-air fix it is purely local: there is no global sum, and it does not touch PS or the atmosphere.
  - **What the two share.** Both start from B e01, B's lowest-validation-loss `best_ckpt`, not from B22: by the
    2026-10-03 (UNVERIFIED) ruling, B22 barely drifts in mass, which leaves a corrector nothing to fix. Both use the same
    fine-tune recipe (Table 1b):
    - 24 epochs on a 5-step loss;
    - peak LR 0.0004 with a cosine schedule of T_max 22 (B itself used T_max 100);
    - global batch 16 on 2 nodes, bf16, no EMA.

    They differ only in which constraint is on, plus the code commit (`43a2476a` vs `d6489f5d`). The "anneal" in
    both tags names that cosine LR schedule, not a ramp of the constraint.
- **The F family (no soil outputs, EMA on).** **F-warm** continues A after its two soil outputs are cut off;
  **F-scratch** trains A's recipe on the 99-output channel set from random weights; **F-depth4** applies B's
  depth recipe to F-scratch epoch 23; **F-depth4-dry** is F-depth4 with dry air trained in, run as a 4-epoch
  stop-early chain.
- **The G family (F-scratch's recipe without mixed precision).** **G1** is F-scratch in fp32; **G1-dry** adds the
  trained-in dry-air constraint; **G4E** trains G1's recipe at depth 5 from scratch; **G4D** is G4E with dry air.

**Replicates.** Each arm is **one training run with one seed**. No recipe was trained twice, so no comparison
between arms carries an estimate of run-to-run training noise. The seed is not printed in the trainer logs and was
not checked. The replicates that do exist are on the evaluation side: the 8 start dates of the five-year protocol
and the 24 initial conditions of the K=56 sweep (Table 3).

**Which settings differ.** Tables 1a and 1b list every setting that differs between the runs. The following
were the same in all 13 runs (read from each trainer log): the SFNO architecture (`embed_dim` 384, `num_layers` 8,
`scale_factor` 3), AdamW with β₁ 0.9 and weight decay 0.0, the 7 forcing channels, `n_eval_samples` 512 and
`valid_autoreg_steps` 3, and the data split of the production pack (training years 2015–2044, 30 files;
validation 2045–2047; test 2048–2049). The 101-output runs have 147863863 trainable parameters and the
99-output runs 147861915.

```
lineage (arrows = initialised from; "e" = epoch)

  random ──► 128-node run
  random ──► A ──(A e243)──► B ──(B e01)──► DRYAIR
              │                    └──(B e01)──► SOILFIX
              └─(A e243, soil outputs sliced off)──► F-warm
  random ──► F-scratch ──(raw e23)──► F-depth4
                    └────(raw e23)──► F-depth4-dry
  random ──► G1      random ──► G1-dry      random ──► G4E      random ──► G4D
```

## 2. Configurations

As Table 1a shows, the 13 runs split into two channel sets, two loss depths, two training precisions and three
conservation settings. **Output channels** = `n_state_channels` + `n_diagnostic_channels` as printed by the
trainer (100 + 1 with soil, 98 + 1 without); the inputs add the 7 forcing channels. **Loss depth** is the
trainer's `multistep_count`, the number of chained 6-hour predictions inside one training loss, followed by
`n_future` = `multistep_count` − 1. Every part of Tables 1a–1c starts with the run's name, which is also its
acronym family; the last column of part 1 lists every checkpoint label derived from the run, and Table 2
defines the suffixes.

**Table 1a, part 1. What each run is.** The last column lists every checkpoint acronym derived from the run (Table 2 defines the suffixes).

| run | role | initialised from | checkpoint acronyms |
|---|---|---|---|
| **128-node** | large-batch baseline on many nodes | random | none (no checkpoint of this run was rolled out) |
| **A** | single-step baseline from scratch | random | A (= A243, `best_ckpt` = e243); K=56: A243, A16, A243_repro, A243rr; lagged: A243; 1-yr screen: A_e21…A_e243 |
| **B** | depth fine-tune of A | A e243 (A's `best_ckpt`), fresh optimizer | B (= B01, `best_ckpt` = e01), B22, B24; B22fp32, B22posclamp; B01dryair, B22dryair, B24dryair; BF22, BF24, BF22fix, BF24fix; 1-yr screen: B_e01, B_e21…B_e24 |
| **DRYAIR** | depth fine-tune of B, dry air trained in | B e01 (B's `best_ckpt`), fresh optimizer | DRYAIR01, DRYAIR24, DRYAIRF24; 1-yr screen: dryair_* |
| **SOILFIX** | as DRYAIR with the soil-moisture fix instead of dry air | B e01 (B's `best_ckpt`), fresh optimizer | SOILFIX21, SOILFIX24; 1-yr screen: soilfix_* |
| **F-warm** | A's recipe without soil outputs, warm-started from A | A e243 with the soil outputs sliced off (surgical file), fresh optimizer | 1-yr screen only: raw_best, ema_best, raw_e01, raw_e22, raw_e39…raw_e42 |
| **F-scratch** | A's recipe without soil outputs, from scratch, EMA on | random | FS20…FS23, FS41…FS43; FSE15…FSE43; FSEMA (= FSE23), FSEMAdryair; FSEF15…FSEF19, FSEF23…FSEF32; FSEQ15; 1-yr screen: fs_*, fs43_* |
| **F-depth4** | B's depth recipe applied to F-scratch | F-scratch e23 raw (`best_ckpt` at the end of its first job) | FD14, FD22, FD24, FD22dryair, FD24dryair; FDE04…FDE60 (every 4th epoch), FDE63; FDEF28, FDEF32; 1-yr screen: fsF_* |
| **F-depth4-dry** | F-depth4 plus trained-in dry air, stop-early chain | F-scratch e23 raw (stable copy) | DAE02, DAE04, DAR02, DAR04 |
| **G1** | F-scratch in full precision | random | G1E15…G1E23; G1E15fix, G1E16fix, G1E19fix; G1EB15…G1EB19 |
| **G1-dry** | G1 plus trained-in dry air | random | G1D15…G1D23, G1D39…G1D43, G1D59…G1D63; K=56: G1D16 |
| **G4E** | G1's recipe at depth from scratch | random | G4E15…G4E23; G4E15fix…G4E23fix |
| **G4D** | G4E plus trained-in dry air | random | G4D15…G4D23, G4D36, G4D43, G4D56, G4D63; K=56: G4D16, G4D16rr, G4D36, G4D43, G4D56, G4D63; lagged: G4D16 |

**Table 1a, part 2. Run tag and PBS jobs.**

| run | run tag | PBS job id(s) |
|---|---|---|
| **128-node** | `prod128_alldata_v2` | 7566145 |
| **A** | `prod1n_b32_sgdr` | 7585080 |
| **B** | `nf4_prod_b16_r1` | 7630639 |
| **DRYAIR** | `fs_anneal_dryair_nf4_b16_rb01` | 7709268 |
| **SOILFIX** | `fs_anneal_soilfix_nf4_b16_rb01` | 7715005 |
| **F-warm** | `f_nosoil_2n_b32_e43_warm` | 7660250 UNVERIFIED |
| **F-scratch** | `f_nosoil_2n_b32_e23_scratch` | 7718436 + 7721577 |
| **F-depth4** | `fsF_anneal_nf4_b16_rtmax100_scratch` | 7719538 + 7725884 |
| **F-depth4-dry** | `fsF_dryair_nf4_b16_rtmax100_scratch` | 7732625 + 7733190 + 7733283 + 7733346 |
| **G1** | `f_nosoil_2n_b32_e23_scratch_fp32` | 7731784 |
| **G1-dry** | `f_nosoil_2n_b32_e63_scratch_fp32_dryair` | 7735732 |
| **G4E** | `f_nosoil_4n_b32_e23_scratch_fp32_nf4` | 7740992 |
| **G4D** | `f_nosoil_4n_b32_e23_scratch_fp32_dryair_nf4` | 7742231 + 7771394 |

**Table 1a, part 3. Data and loss.** Read from `out.log` (trainer-printed values), the PBS stdout and the provenance sidecars.

| run | output channels (count; soil) | loss depth (multistep_count / n_future) | dry-air constraint in training | training precision (amp_mode) | EMA (decay) |
|---|---|---|---|---|---|
| **128-node** | 101; soil yes | 1 / 0 | none | bf16 | none |
| **A** | 101; soil yes | 1 / 0 | none | bf16 | none |
| **B** | 101; soil yes | 5 / 4 | none | bf16 | none |
| **DRYAIR** | 101; soil yes | 5 / 4 | trained in (`conserve_dry_air True`) | bf16 | none |
| **SOILFIX** | 101; soil yes | 5 / 4 | none (soil-moisture fix trained in, `conserve_soil_moisture True`) | bf16 | none |
| **F-warm** | 99; soil no | 1 / 0 | none | bf16 | 0.9995 |
| **F-scratch** | 99; soil no | 1 / 0 | none | bf16 | 0.9995 |
| **F-depth4** | 99; soil no | 5 / 4 | none | bf16 | 0.9995 |
| **F-depth4-dry** | 99; soil no | 5 / 4 | trained in (`conserve_dry_air True`) | bf16 | 0.9995 |
| **G1** | 99; soil no | 1 / 0 | none | none (fp32 weights and activations) | 0.9995 |
| **G1-dry** | 99; soil no | 1 / 0 | trained in (`conserve_dry_air True`) | none (fp32 weights and activations) | 0.9995 |
| **G4E** | 99; soil no | 5 / 4 | none | none (fp32 weights and activations) | 0.9995 |
| **G4D** | 99; soil no | 5 / 4 | trained in (`conserve_dry_air True`) | none (fp32 weights and activations) | 0.9995 |

Every EMA run printed `EMA enabled: decay=0.9995, warmup=True`, i.e. the effective decay ramps up to 0.9995
over the first epochs. The initialisations were read from the trainer's `pretrained_checkpoint_path`. Three of
them point at a `best_ckpt_mp0.tar`, which holds the epoch with the lowest raw validation loss. For A that is
epoch 243 of 243. For B it is epoch 1, so DRYAIR and SOILFIX start from B e01. For F-scratch it is epoch 23 at
the end of its first job; F-depth4 started from that file while F-scratch was between its two jobs.

As Table 1b shows, the runs used three peak learning rates, four schedules, three global batch sizes and four
node counts. **Global batch** = local batch per GPU × GPUs; it equals the printed `global_batch_size` in every
run. **Updates** = the sum over epochs of the logged training steps per epoch (= steps per epoch × epochs; steps
per epoch = `n_train_samples_per_epoch` ÷ global batch). **Wall-clock training time** = the sum over the run's
jobs of the trainer's `Total training time` line, ÷ 3600; it excludes job start-up and any post-training hang
(Table 1c). **Node-hours** = Σ over jobs of (training seconds × nodes) ÷ 3600, counting the nodes the trainer used,
not spare nodes. **Median epoch time** = the median of the logged `epoch time [s]`, which includes validation.
**Warmup epochs** is the printed `lr_warmup_steps`; the warmup starts at `lr_start` 0.01 × peak LR in every run
that has one, and the launcher passes it as `WARMUP_EPOCHS` (provenance sidecars).

**Table 1b, part 1. Settings shared by every run.** Listed per run so each value is checked; all 13 rows are identical.

| run | optimizer | β₂ | max grad norm | spatial parallel (h × w) | scale_factor |
|---|---|---|---|---|---|
| **128-node** | AdamW | 0.95 | 32 | 1 x 1 | 3 |
| **A** | AdamW | 0.95 | 32 | 1 x 1 | 3 |
| **B** | AdamW | 0.95 | 32 | 1 x 1 | 3 |
| **DRYAIR** | AdamW | 0.95 | 32 | 1 x 1 | 3 |
| **SOILFIX** | AdamW | 0.95 | 32 | 1 x 1 | 3 |
| **F-warm** | AdamW | 0.95 | 32 | 1 x 1 | 3 |
| **F-scratch** | AdamW | 0.95 | 32 | 1 x 1 | 3 |
| **F-depth4** | AdamW | 0.95 | 32 | 1 x 1 | 3 |
| **F-depth4-dry** | AdamW | 0.95 | 32 | 1 x 1 | 3 |
| **G1** | AdamW | 0.95 | 32 | 1 x 1 | 3 |
| **G1-dry** | AdamW | 0.95 | 32 | 1 x 1 | 3 |
| **G4E** | AdamW | 0.95 | 32 | 1 x 1 | 3 |
| **G4D** | AdamW | 0.95 | 32 | 1 x 1 | 3 |

**Table 1b, part 2. Learning rate.**

| run | peak LR | LR schedule (type, T₀ / T_mult / T_max, min LR) | warmup epochs |
|---|---|---|---|
| **128-node** | 0.001 | reduce-on-plateau: factor 0.5, patience 1, no min printed | 0 |
| **A** | 0.002 | SGDR: T₀ 20, T_mult 1, min 0.000001 | 3 |
| **B** | 0.0004 | cosine: T_max 100, min 0.000001 | 1 |
| **DRYAIR** | 0.0004 | cosine: T_max 22, min 0.000001 | 1 |
| **SOILFIX** | 0.0004 | cosine: T_max 22, min 0.000001 | 1 |
| **F-warm** | 0.002 | SGDR: T₀ 20, T_mult 1, min 0.000001 | 3 |
| **F-scratch** | 0.002 | SGDR: T₀ 20, T_mult 1, min 0.000001 | 3 |
| **F-depth4** | 0.0004 | cosine: T_max 100, min 0.000001 | 1 |
| **F-depth4-dry** | 0.0004 | cosine: T_max 100, min 0.000001 | 1 |
| **G1** | 0.002 | SGDR: T₀ 20, T_mult 1, min 0.000001 | 3 |
| **G1-dry** | 0.002 | SGDR: T₀ 20, T_mult 1, min 0.000001 | 3 |
| **G4E** | 0.002 | SGDR: T₀ 20, T_mult 1, min 0.000001 | 3 |
| **G4D** | 0.002 | SGDR: T₀ 20, T_mult 1, min 0.000001 | 3 |

**Table 1b, part 3. Batch and hardware.**

| run | local batch per GPU | GPUs | nodes | global batch |
|---|---|---|---|---|
| **128-node** | 1 | 512 | 128 | 512 |
| **A** | 8 | 4 | 1 | 32 |
| **B** | 2 | 8 | 2 | 16 |
| **DRYAIR** | 2 | 8 | 2 | 16 |
| **SOILFIX** | 2 | 8 | 2 | 16 |
| **F-warm** | 4 | 8 | 2 | 32 |
| **F-scratch** | 4 | 8 | 2 | 32 |
| **F-depth4** | 2 | 8 | 2 | 16 |
| **F-depth4-dry** | 1 | 16 | 4 | 16 |
| **G1** | 4 | 8 | 2 | 32 |
| **G1-dry** | 4 | 8 | 2 | 32 |
| **G4E** | 2 | 16 | 4 | 32 |
| **G4D** | 2 | 16 | 4 | 32 |

**Table 1b, part 4. Length and cost.**

| run | epochs trained | updates | wall-clock training time (h) | node-hours | median epoch time (s) |
|---|---|---|---|---|---|
| **128-node** | 100 | 8500 | 1.6 | 204.4 | 58 |
| **A** | 243 | 332424 | 46.3 | 46.3 | 672 |
| **B** | 24 | 65688 | 8.7 | 17.4 | 1305 |
| **DRYAIR** | 24 | 65688 | 7.9 | 15.7 | 1180 |
| **SOILFIX** | 24 | 65688 | 7.6 | 15.3 | 1142 |
| **F-warm** | 43 | 58824 | 10.7 | 21.4 | 894 |
| **F-scratch** | 43 | 58824 | 11.5 | 23.0 | 931 |
| **F-depth4** | 63 | 172431 | 47.5 | 95.1 | 2717 |
| **F-depth4-dry** | 4 | 10948 | 1.4 | 5.6 | 1256 |
| **G1** | 23 | 31464 | 5.7 | 11.4 | 886 |
| **G1-dry** | 63 | 86184 | 15.1 | 30.3 | 864 |
| **G4E** | 23 | 31464 | 8.1 | 32.5 | 1261 |
| **G4D** | 63 | 86184 | 22.8 | 91.0 | 1300 |

As Table 1c shows, four runs were trained in more than one job, and five jobs were ended by the PBS walltime
rather than by the trainer. A job that hung after training (the post-training `mpiexec`/wandb hang) ran to its
wall; its elapsed time is in the `PBS: job killed: walltime … exceeded limit` line, the only record of a job's
elapsed time we read. For F-depth4's first job the trainer never printed `Total training time`: the job was killed
during epoch 17, so its training time is the sum of its 16 logged epoch times, and the partial epoch 17 is not
counted.

**Table 1c, part 1. Jobs and epochs of each training run.** "max_epochs printed" is the target the trainer was given for that job.

| run | job | epochs in this job | nodes | max_epochs printed |
|---|---|---|---|---|
| 128-node | 7566145 | 1–100 | 128 | 100 |
| A | 7585080 | 1–243 | 1 | 243 |
| B | 7630639 | 1–24 | 2 | 24 |
| DRYAIR | 7709268 | 1–24 | 2 | 24 |
| SOILFIX | 7715005 | 1–24 | 2 | 24 |
| F-warm | 7660250 UNVERIFIED | 1–43 | 2 | 43 |
| F-scratch | 7718436 | 1–23 | 2 | 23 |
| F-scratch | 7721577 | 24–43 | 2 | 43 |
| F-depth4 | 7719538 | 1–16 | 2 | 24 |
| F-depth4 | 7725884 | 17–63 | 2 | 63 |
| F-depth4-dry | 7732625 | 1–1 | 4 | 1 |
| F-depth4-dry | 7733190 | 2–2 | 4 | 2 |
| F-depth4-dry | 7733283 | 3–3 | 4 | 3 |
| F-depth4-dry | 7733346 | 4–4 | 4 | 4 |
| G1 | 7731784 | 1–23 | 2 | 23 |
| G1-dry | 7735732 | 1–63 | 2 | 63 |
| G4E | 7740992 | 1–23 | 4 | 23 |
| G4D | 7742231 | 1–23 | 4 | 23 |
| G4D | 7771394 | 24–63 | 4 | 63 |

**Table 1c, part 2. Time and code of each job.**

| run | job | training time (s) | basis | PBS-killed elapsed (s) | git hash (trainer log) |
|---|---|---|---|---|---|
| 128-node | 7566145 | 5749.22 | Total training time | not recorded | `53770be6` |
| A | 7585080 | 166683.54 | Total training time | not recorded | `38c493d7` |
| B | 7630639 | 31291.86 | Total training time | not recorded | `a5b83ae2` |
| DRYAIR | 7709268 | 28336.04 | Total training time | not recorded | `43a2476a` |
| SOILFIX | 7715005 | 27497.24 | Total training time | not recorded | `d6489f5d` |
| F-warm | 7660250 UNVERIFIED | 38549.19 | Total training time | not recorded | `a530bf65` |
| F-scratch | 7718436 | 21436.66 | Total training time | not recorded | `a530bf65` |
| F-scratch | 7721577 | 20032.31 | Total training time | not recorded | `a530bf65` |
| F-depth4 | 7719538 | 43012.38 | epoch-time sum | 43229 | `a1b27d79` |
| F-depth4 | 7725884 | 128166.85 | Total training time | not recorded | `a1b27d79` |
| F-depth4-dry | 7732625 | 1292.51 | Total training time | not recorded | `9018ec79` |
| F-depth4-dry | 7733190 | 1252.84 | Total training time | not recorded | `9018ec79` |
| F-depth4-dry | 7733283 | 1256.65 | Total training time | not recorded | `9018ec79` |
| F-depth4-dry | 7733346 | 1254.96 | Total training time | not recorded | `9018ec79` |
| G1 | 7731784 | 20549.66 | Total training time | 43204 | `9018ec79` |
| G1-dry | 7735732 | 54479.49 | Total training time | 66676 | `faf610d2` |
| G4E | 7740992 | 29221.49 | Total training time | 50441 | `9018ec79` |
| G4D | 7742231 | 29200.77 | Total training time | 50451 | `faf610d2` |
| G4D | 7771394 | 52725.88 | Total training time | not recorded | `faf610d2` |

Each job id was tied to its log segment by the PBS stdout's `jobid=`, `run_num=` and rendered `max_epochs`, and,
where both exist, by an identical `Total training time`. F-scratch's two stdouts were in a worktree that no longer
exists; its job ids come from the launcher's resume sidecar (`identical to 7718436 except EPOCHS 23->43`,
`job: 7721577`), which matches the segments' `max_epochs` 23 and 43. The F-warm and F-scratch first jobs each
allocated 1 spare node on top of the 2 training nodes (provenance sidecars); node-hours count only the 2.

### Table 2. Rollout and evaluation variants that are not training runs

Every five-year arm is a tuple (training run, checkpoint, raw or EMA weights, rollout precision, inference-time
corrections). As Table 2 shows, 166 distinct arm names went through the five-year protocol, in 168 arm-job rows
(A and G4D16 were each rolled in two jobs). Weights and checkpoint files come from the rollout jobs' stdouts
(`weights <arm>: ema from …`, `ckpt <arm>: …`) or, where those print none, from the submit scripts
(`submit_fp32_reroll.sh`, `submit_precision_cross.sh`). Epochs, rollout precision and the dry-air switch come from
each arm's member-00 log (`ckpt_epoch=`, `feedback_dtype=`, `dry_air_fix=`). **Epoch numbering:** an arm's number
is the epoch, and epoch N's checkpoint is `ckpt_mp0_v(N−1)`. For all 127 arms rolled from a `ckpt_mp0_vN.tar`
file the member log's `ckpt_epoch` equals N + 1 (G4D16 = EMA weights of epoch 16 = `ckpt_mp0_v15`). B is the
exception that names its files differently: it kept 4 rotating slots (`checkpoint_num_versions` 4), so B22 and B24
were rolled from the stable copies `ckpt_mp0_e22_stable.tar` and `ckpt_mp0_e24_stable.tar`.

**Table 2a, part 1. Five-year-protocol arm families and their acronyms.** "fp32"/"bf16" = the dtype of the fed-back state in the rollout
(`feedback_dtype`). "Dry-air fix" = the inference-time `DryAirFix` (PS shifted so global dry-air mass is
conserved); "from config" means the run trained with the constraint and the rollout keeps it on.

| arm family | training run | arms | epochs rolled | acronym rule |
|---|---|---|---|---|
| A | A | 1 | e243 | A = A's `best_ckpt` = epoch 243; called A243 elsewhere |
| B | B | 1 | e1 | B = B's `best_ckpt` = epoch 1; called B01 / B_e01 elsewhere |
| B‹e› | B | 2 | e22, e24 | B + epoch |
| B‹e›fp32 | B | 1 | e22 | suffix fp32 = only the fed-back state kept in fp32 |
| B‹e›posclamp | B | 1 | e22 | suffix posclamp = humidity positivity clamp at prediction time |
| B‹e›dryair | B | 3 | e1, e22, e24 | suffix dryair = inference-only dry-air fix; B22dryair is the "B22fix" / "B22 + fix" of later tables |
| BF‹e› | B | 2 | e22, e24 | infix F = bf16-trained weights re-rolled in fp32 (autocast off) |
| BF‹e›fix | B | 2 | e22, e24 | F re-roll + fix |
| DRYAIR‹e› | DRYAIR | 2 | e1, e24 | DRYAIR + epoch of the DRYAIR run (DRYAIR01 is not B e01) |
| DRYAIRF‹e› | DRYAIR | 1 | e24 | F re-roll |
| SOILFIX‹e› | SOILFIX | 2 | e21, e24 | SOILFIX24 was rolled from `best_ckpt_mp0.tar`, whose `ckpt_epoch` is 24 |
| FS‹e› | F-scratch | 7 | e20–e23, e41–e43 | FS = F-scratch raw |
| FSE‹e› | F-scratch | 29 | e15–e43 | FSE = F-scratch EMA |
| FSEMA | F-scratch | 1 | e23 | FSEMA = `best_ckpt_ema_mp0.tar` = EMA e23 = FSE23 |
| FSEMAdryair | F-scratch | 1 | e23 | FSEMA + fix |
| FSEF‹e› | F-scratch | 15 | e15–e19, e23–e32 | FSE re-rolled in fp32 |
| FSEQ‹e› | F-scratch | 1 | e15 | Q = the bf16-path equivalence gate (must equal FSE15 bitwise) |
| FD‹e› | F-depth4 | 3 | e14, e22, e24 | FD = F-depth4 raw |
| FD‹e›dryair | F-depth4 | 2 | e22, e24 | FD + fix |
| FDE‹e› | F-depth4 | 16 | e4–e60 every 4th, e63 | FDE = F-depth4 EMA |
| FDEF‹e› | F-depth4 | 2 | e28, e32 | FDE re-rolled in fp32 |
| DAE‹e› | F-depth4-dry | 2 | e2, e4 | DA = the dry-air chain; E = EMA |
| DAR‹e› | F-depth4-dry | 2 | e2, e4 | R = raw |
| G1E‹e› | G1 | 9 | e15–e23 | G1E = G1 EMA |
| G1E‹e›fix | G1 | 3 | e15, e16, e19 | suffix fix = inference-only dry-air fix |
| G1EB‹e› | G1 | 5 | e15–e19 | suffix B = fp32-trained weights rolled in bf16 |
| G1D‹e› | G1-dry | 19 | e15–e23, e39–e43, e59–e63 | G1D = G1-dry EMA |
| G4E‹e› | G4E | 9 | e15–e23 | G4E = G4E EMA |
| G4E‹e›fix | G4E | 9 | e15–e23 | G4E + fix |
| G4D‹e› | G4D | 13 | e15–e23, e36, e43, e56, e63 | G4D = G4D EMA; G4D16 was re-rolled in 7808258 after the NetCDFs were deleted |

**Table 2a, part 2. How each arm family was rolled out.**

| arm family | weights | rollout precision | dry-air fix in rollout | extra flags | 5-yr job(s) |
|---|---|---|---|---|---|
| A | raw | bf16 | not logged | — | 7649597, 7649650 |
| B | raw | bf16 | not logged | — | 7649597 |
| B‹e› | raw | bf16 | off | — | 7707597 |
| B‹e›fp32 | raw | fp32 | off | `--feedback-fp32` | 7707597 |
| B‹e›posclamp | raw | bf16 | off | `--force-positive=relhum` | 7707597 |
| B‹e›dryair | raw | fp32 | on | `--dry-air-fix=on` | 7707597, 7721150 |
| BF‹e› | raw | fp32 | off | `--amp off` | 7743369 |
| BF‹e›fix | raw | fp32 | on | `--amp off --dry-air-fix on` | 7743369 |
| DRYAIR‹e› | raw | fp32 | on (from config) | — | 7711720 |
| DRYAIRF‹e› | raw | fp32 | on (from config) | `--amp off` | 7743369 |
| SOILFIX‹e› | raw | fp32 | off | — | 7724006 |
| FS‹e› | raw | bf16 | off | — | 7720705, 7727625 |
| FSE‹e› | EMA | bf16 | off | — | 7732901, 7732928, 7733015, 7733110, 7733210, 7733258 |
| FSEMA | EMA (EMA-best file) | bf16 | off | — | 7720705 |
| FSEMAdryair | EMA (EMA-best file) | fp32 | on | `--dry-air-fix=on` | 7721233 |
| FSEF‹e› | EMA | fp32 | off | `--amp off` | 7742849, 7743368, 7743369 |
| FSEQ‹e› | EMA | bf16 | off | `--amp bf16` | 7742848 |
| FD‹e› | raw | bf16 | off | — | 7727731 |
| FD‹e›dryair | raw | fp32 | on | `--dry-air-fix=on` | 7731448 |
| FDE‹e› | EMA | bf16 | off | — | 7731806, 7731807, 7732564, 7732686 |
| FDEF‹e› | EMA | fp32 | off | `--amp off` | 7743369 |
| DAE‹e› | EMA | fp32 | on (from config) | — | 7734370 |
| DAR‹e› | raw | fp32 | on (from config) | — | 7734370 |
| G1E‹e› | EMA | fp32 | off | — | 7734960, 7734961 |
| G1E‹e›fix | EMA | fp32 | on | `--dry-air-fix on` | 7735162 |
| G1EB‹e› | EMA | bf16 | off | `--amp bf16` | 7742848 |
| G1D‹e› | EMA | fp32 | on (from config) | — | 7738106, 7738107, 7738185, 7738283 |
| G4E‹e› | EMA | fp32 | off | — | 7749890, 7749891 |
| G4E‹e›fix | EMA | fp32 | on | `--dry-air-fix on` | 7750375, 7750376 |
| G4D‹e› | EMA | fp32 | on (from config) | — | 7760862, 7760863, 7807790, 7808258 |

**Table 2b. K=56 and lagged-ensemble arms** (14-day forecasts, Table 3). Definitions are quoted from the campaign
files named in `configs.json` (`k56_arms`).

| arm (sweep dir) | definition | acronym rule |
|---|---|---|
| A243 (`prod1n_b32_sgdr_K56`, `prod1n_b32_sgdr_lagged_K56_d4`) | `prod1n_b32_sgdr/best_ckpt_mp0.tar`, epoch 243, raw, bf16 | A + epoch |
| A16 (`A16_K56`) | `prod1n_b32_sgdr/ckpt_mp0_v15.tar`, raw (A has no EMA), bf16 path as A243 | A + epoch |
| A243_repro (`A243_repro_K56`) | A243 re-rolled on one IC to check bitwise reproduction at the evaluation commit | _repro = reproduction gate |
| A243rr, G4D16rr (`A243rr_K56`, `G4D16rr_K56`) | full re-rolls of A243 and G4D16 for maps after the NetCDFs were freed | rr = re-roll |
| G1D16 (`G1D16_K56`) | EMA of `f_nosoil_2n_b32_e63_scratch_fp32_dryair` v15 (G1-dry), fp32, dry-air fix from config | G1D + epoch |
| G4D16 (`G4D16_K56`, `G4D16_lagged_K56_d4`) | EMA of `f_nosoil_4n_b32_e23_scratch_fp32_dryair_nf4` v15, fp32, dry-air fix from config | G4D + epoch |
| G4D36, G4D43, G4D56, G4D63 (`G4D36_K56` … `G4D63_K56`) | EMA of G4D `ckpt_mp0_v35`, `v42`, `v55`, `v62` | G4D + epoch |

### Table 3. Evaluation protocols (configuration only)

As Table 3 shows, four protocols were used; their settings are the defaults of the named scripts, read by the
extractor. The five-year protocol's score and gates are those of the 2026-10-09 pre-registration.

| protocol | members / ICs | start | length | what is scored, and how | source |
|---|---|---|---|---|---|
| **5-yr protocol** | 8 members per arm | year 2044, frame 1092 + 16 × i for member i (October 2044) | 7667 − 16 × i six-hour steps (7667 for the first member, 7555 for the last), to the last frame of 2049-12-31 | Selection window 2045–2047; reporting window 2048–2049, sealed until finalists are frozen, plus a reporting start set R at frames 364 + 16 × i. Score A: per channel, e_c = area-weighted pattern RMSE of the window time-mean map vs truth; floor_c = RMS over the 28 member pairs of the same quantity between members; r_c = e_c / floor_c; A = median over members of the mean over 24 groups (9 surface channels + 5 level families × 3 bands) of the group-mean r_c. Gates: *finite*, every member finite to the end; *water*, global-mean TMQ positive at every lead and at most 1 % negative-TMQ area in the last month; *mass*, \|dPS\| at most 1 hPa; *drift*, at most 3 drifting groups, a group drifting when its median channel drift exceeds 2 interannual SDs. | `polaris_climate_run.pbs`, `score_window_protocol.py`, `2026-10-09_pipeline_comparison_prereg.md` |
| **K=56 sweep** | 12 ICs per test file × 2 files (2048, 2049) = 24 ICs | the ICs' monthly offsets in the test years | 56 six-hour steps = 336 h = 14 days | RMSE / ACC by lead on the common channels | `polaris_eval_inference.pbs`, `polaris_score_k56.pbs` |
| **Lagged ensemble** | 32 targets × 14 members (starts 4 steps apart) | first start 0, test file 2048 | 56 steps; member lags 4 … 56 steps | ensemble mean and CRPS per target; for G4D16, 21 rollouts | `polaris_eval_inference.pbs` (`IC_MODE=lagged`), G4D16 prereg |
| **1-yr screen** | one rollout per checkpoint | year 2044, frame 1092; second start 1156 | 1460 six-hour steps (one year) | survival and drift against the screen truth; retired as a selection gate by the pipeline-comparison prereg. 14 screen jobs, 164 screen labels on disk | `polaris_climate_screen.pbs` |

Five-year members are rolled 4 at a time, one per GPU, with `climate_rollout.py`; a member that goes non-finite
stops and is counted as a measurement (`CLIMATE_ROLLOUT_TRUNCATED`). EMA arms are rebuilt from a checkpoint's
`ema_state` by `extract_ema_ckpt.py` before the rollout.

## 3. Sweeps and levels

Every level tested, per swept setting, with no outcome attached:

- **Epochs trained (per run):** 100 (128-node), 243 (A), 24 (B, DRYAIR, SOILFIX), 43 (F-warm, F-scratch), 63
  (F-depth4, G1-dry, G4D), 4 (F-depth4-dry), 23 (G1, G4E). Extensions: F-scratch 23 → 43 (second job), F-depth4
  planned 24 → 63 (first job killed at the wall during epoch 17, resumed from epoch 16), G4D 23 → 63 (second job),
  F-depth4-dry 1 → 4 (one epoch per job).
- **Checkpoints rolled out under the 5-yr protocol (distinct epochs per run):** A 1 (e243); B 3 (e1, e22, e24);
  DRYAIR 2 (e1, e24); SOILFIX 2 (e21, e24); F-scratch 29 (e15–e43); F-depth4 18 (e4–e60 every 4th, e14, e22, e63);
  F-depth4-dry 2 (e2, e4); G1 9 (e15–e23); G1-dry 19 (e15–e23, e39–e43, e59–e63); G4E 9 (e15–e23); G4D 13
  (e15–e23, e36, e43, e56, e63); 128-node and F-warm 0.
- **Peak learning rate:** 0.001 (128-node), 0.002 (A, F-warm, F-scratch, G1, G1-dry, G4E, G4D), 0.0004 (B, DRYAIR,
  SOILFIX, F-depth4, F-depth4-dry).
- **Schedule:** reduce-on-plateau (128-node); SGDR with T₀ 20, T_mult 1 (A, F-warm, F-scratch, the G family);
  one cosine with T_max 100 (B, F-depth4, F-depth4-dry); one cosine with T_max 22 (DRYAIR, SOILFIX). Minimum LR
  0.000001 wherever printed.
- **Warmup epochs:** 0 (128-node), 1 (the depth fine-tunes), 3 (the SGDR runs).
- **Global batch:** 512, 32, 16. **Local batch per GPU:** 1, 2, 4, 8. **Nodes:** 128, 1, 2, 4. **GPUs:** 512, 4,
  8, 16.
- **Loss depth (`multistep_count`):** 1 and 5. No other depth was trained among these 13 runs.
- **Training precision (`amp_mode`):** bf16 and none (fp32 with TF32 matmuls, per the provenance sidecars).
- **Rollout precision:** bf16 and fp32 fed-back state, crossed with training precision on F-scratch and G1 (FSE,
  FSEF, G1E, G1EB; Table 2a).
- **Dry air:** none; trained in (DRYAIR, F-depth4-dry, G1-dry, G4D); inference-only fix (the ‹e›dryair / ‹e›fix
  arms of B, F-scratch, F-depth4, G1 and G4E).
- **Other conservation:** the soil-moisture fix trained in (SOILFIX only).
- **EMA:** none (128-node, A, B, DRYAIR, SOILFIX) or decay 0.9995 with warm-up (every F and G run); rolled out as
  raw and as EMA weights from the same run for F-scratch (FS / FSE), F-depth4 (FD / FDE) and F-depth4-dry (DAR /
  DAE).
- **Output channels:** 101 (with the two soil channels) and 99.
- **Initialisation:** random; A e243; A e243 with soil outputs sliced off; B e01; F-scratch e23 raw.
- **Spatial parallelism:** none (1 x 1) in every run.

## 4. What each comparison isolates

Only comparisons that were actually made (in the 2026-10-13 write-up, the CHANGELOG entries of 10-13 to 11-02 and
the pre-registrations) are listed. Each names the setting it was meant to isolate and every setting that differs
alongside it in Tables 1a–1c and 2a.

**128-node run vs A.** The data, outputs, loss depth, precision and architecture match. Global batch (512 vs 32),
nodes (128 vs 1), peak LR (0.001 vs 0.002), schedule (reduce-on-plateau without warmup vs SGDR with 3 warmup
epochs) and epochs (100 vs 243) all differ at once. Therefore 128-node vs A isolates the cost of the large-batch
layout at matched updates but not the effect of batch size alone, because LR and schedule changed with it.

**A vs B.** B starts from A e243 and changes the loss depth from 1 to 5. It also changes the peak LR (0.002 →
0.0004), the schedule (SGDR → one cosine with T_max 100), warmup (3 → 1), global batch (32 → 16), nodes (1 → 2)
and the optimizer state (fresh). Therefore A vs B isolates "depth-5 fine-tuning of A e243" as a package but not
depth alone; no single-step continuation of A with B's LR, schedule and batch was run.

**B vs DRYAIR.** DRYAIR adds the trained-in dry-air constraint, but it starts from B e01 rather than A e243, anneals
with T_max 22 instead of 100, and its rollouts carry an fp32 fed-back state (DRYAIR24) where B22/B24 carry bf16.
Therefore B vs DRYAIR isolates the DRYAIR package but not the dry-air constraint, which is confounded with the
initial checkpoint, the annealed schedule and the rollout precision.

**DRYAIR vs SOILFIX.** Same initial checkpoint (B e01), schedule, batch, nodes and rollout precision (fp32); one
trains the dry-air constraint, the other the soil-moisture fix. The code commit differs (`43a2476a` vs `d6489f5d`).
Therefore DRYAIR vs SOILFIX isolates which conservation constraint is trained in but not anything about either
constraint against no constraint, since no plain T_max 22 anneal from B e01 was run.

**A vs F-warm.** F-warm continues A e243 with the soil outputs cut off, EMA on, 8 GPUs instead of 4 at the same
global batch 32, a fresh optimizer and a restarted SGDR schedule for 43 more epochs. Therefore A vs F-warm isolates
nothing singly: dropping soil, EMA, the data-parallel width and continued training all change together.

**A vs F-scratch.** Same recipe (LR, SGDR, warmup, global batch 32, bf16, depth 1, random start); F-scratch drops
the soil outputs, turns EMA on and uses 8 GPUs × 4 instead of 4 GPUs × 8. Therefore A vs F-scratch isolates the
F channel set together with EMA at matched epochs but not either one alone, and its raw-weight comparison also
carries the 4 → 8 GPU change in reduction order.

**F-scratch raw vs EMA (FS vs FSE), F-depth4 raw vs EMA (FD vs FDE), and DAR vs DAE.** Each pair is one training run
at one epoch, rolled at the same precision (bf16, bf16 and fp32 respectively). Therefore each pair isolates the
EMA weight average but not EMA's effect on training, which is identical by construction (the shadow does not feed
back into the raw trajectory).

**F-scratch vs F-depth4 (FD{k} vs FSE{23+k}).** F-depth4 starts from F-scratch e23 raw and changes the loss depth
1 → 5, the peak LR 0.002 → 0.0004, the schedule (SGDR → cosine T_max 100), warmup 3 → 1 and the global batch
32 → 16. Therefore this pair isolates B's depth package applied to the F lineage but not depth alone, and matching
FD{k} to FSE{23+k} matches training length only.

**F-depth4 vs F-depth4-dry (FDE04 vs DAE04, pre-registered).** Both start from F-scratch e23 raw with B's recipe at
global batch 16. F-depth4-dry adds the dry-air constraint, runs on 4 nodes × local batch 1 instead of 2 × 2, uses
a different commit (`9018ec79` vs `a1b27d79`), and its rollouts carry an fp32 state where FDE carries bf16. Therefore
FDE04 vs DAE04 isolates the trained-in dry-air constraint on depth 5 but not the rollout precision, which changed
with it.

**F-scratch vs G1 (FSE{e} vs G1E{e}, matched epochs 15–23).** Same recipe, layout (2 nodes × 4 GPUs × 4), depth,
EMA and channel set; G1 trains without autocast. The commit differs (`a530bf65` vs `9018ec79`) and the default
rollouts differ in precision (bf16 vs fp32). Therefore F-scratch vs G1 isolates training plus rollout precision
together but not either one; the precision cross below separates them.

**The precision cross (FSE / FSEF / G1E / G1EB, epochs 15–19).** Training precision (F-scratch bf16, G1 fp32) is
crossed with rollout precision (bf16, fp32) on the same checkpoints; FSEQ15 gates the bf16 path. Therefore the cross
isolates rollout precision within each trained network but not training precision from the other differences
between two separately trained runs (seed noise and the commit).

**The fp32 re-roll (FSEF / FDEF / BF / DRYAIRF vs their bf16-rolled originals).** Same weights, rolled again with
autocast off. Therefore it isolates rollout precision for the bf16-trained pool but not for DRYAIR24, whose original
rollout already carried an fp32 state (its member log for the original roll does not record `amp`).

**G1 vs G1-dry (G1E{e} vs G1D{e}, matched epochs).** Same recipe, layout, precision, depth, EMA and channel set;
G1-dry trains the dry-air constraint and was given 63 epochs instead of 23 (SGDR's cycle does not depend on the
target, so matched epochs see the same LR). The commit differs (`9018ec79` vs `faf610d2`; the CHANGELOG states the
training code is identical, UNVERIFIED). Therefore G1 vs G1-dry isolates the trained-in dry-air constraint on depth
1 but not code-commit differences, which were not diffed here.

**Inference-only dry-air fix vs none (G1E vs G1Efix, G4E vs G4Efix, BF vs BFfix, B22 vs B22dryair, FD24 vs
FD24dryair).** Same weights with the fix switched on at prediction time. In the G and BF pairs both rollouts carry
an fp32 state; in B22 vs B22dryair and FD vs FDdryair the fixed arm carries fp32 and the plain arm bf16. Therefore
the G and BF pairs isolate the inference-time fix but the B22 and FD pairs do not, since they confound the fix with
rollout precision.

**G1 vs G4E (G1E{e} vs G4E{e}, epochs 15–23).** Same peak LR, SGDR schedule, warmup, global batch 32, fp32, EMA,
channel set and commit (`9018ec79`); G4E trains at depth 5 and therefore runs on 4 nodes × local batch 2 instead
of 2 × 4. Therefore G1 vs G4E isolates the loss depth but not the data-parallel width (8 vs 16 GPUs, a different
reduction order) or the 2.8-fold node-hours that came with it (32.5 vs 11.4).

**G1-dry vs G4D, G4E vs G4D, and G4Efix vs G4D (epochs 15–23).** G4D differs from G1-dry by depth and layout (as
G1 vs G4E), from G4E by the trained-in constraint and commit (`faf610d2` vs `9018ec79`), and from G4Efix by
trained-in versus inference-only dry air. All are fp32 EMA rollouts with the dry-air fix on in the G4D, G1D and
G4Efix arms. Therefore G4E vs G4D isolates the trained-in constraint on depth 5, G1-dry vs G4D isolates depth under
trained-in dry air, and G4Efix vs G4D isolates the route to mass conservation, but none of them isolates code-commit
or seed differences.

**Depth vs training length at K=56 (G4D16, G1D16, A16, A243; pre-registered 2026-10-21).** G4D16/G1D16 differs by
depth and layout; G1D16/A16 by the whole F/G recipe (soil, EMA, fp32 training and rollout, dry air, 8 vs 4 GPUs);
A16/A243 by training length and by SGDR cycle position (A16 mid-cycle, A243 at a cycle end). Therefore the
decomposition isolates depth (with layout) and training length (with cycle position) but not the individual parts
of "recipe".

**G4D16 vs G4D36 / G4D56 (and G4D23 vs G4D43 / G4D63).** One run, same SGDR cycle position (e16 ≡ e36 ≡ e56; e23,
e43, e63 are cycle ends), EMA throughout, but the later epochs came from the second job (7771394) after a resume
from `ckpt_mp0_v22`. Therefore these pairs isolate training length within G4D but not resume effects, which were
not checked.

## 5. Limits

- **One job id is UNVERIFIED.** F-warm's job id 7660250 UNVERIFIED comes from KNOWLEDGE.md's lineage table; its
  PBS stdout and that of F-scratch's two jobs were in the deleted `makani-ace2-ports` worktree. F-scratch's ids are
  tied to its segments by the launcher sidecar, a primary record; F-warm's training settings, times and epochs come
  from its own `out.log` and are not affected.
- **Rendered yaml vs trainer.** The rendered yaml carries the template's `batch_size: 8` (a global batch that the
  launcher overrides) and `n_future: 0`. The trainer printed the per-rank batch (1, 2, 4 or 8) and, for every
  depth-5 run, `n_future 4`. We used the printed values; `configs.json` lists each disagreement under
  `yaml_vs_trainer`.
- **F-depth4's first job** has no `Total training time`; its 43012.38 s is the sum of 16 epoch times, which
  include validation and exclude start-up. The partial epoch 17 it was running when killed is not counted anywhere.
- **Job elapsed time** was read only for the five walltime-killed jobs. Queue waits, start-up, and the elapsed time
  of jobs that ended normally were not recorded here; node-hours are training node-hours, not allocated
  node-hours (the spare nodes and the post-training hangs are excluded).
- **The 2026-10-13 write-up** gives 204.3 node-hours for the 128-node run (UNVERIFIED), from summed epoch times;
  Table 1b's 204.4 uses the `Total training time` line. Every other node-hours and median epoch time that write-up's Table 2 shares
  with Table 1b agrees.
- **Warmup semantics.** That `lr_warmup_steps` counts epochs (the scheduler steps once per epoch, so a 1-epoch
  warmup runs epoch 1 at `lr_start` × peak) is from KNOWLEDGE.md §2 (UNVERIFIED); the launcher variable name
  `WARMUP_EPOCHS` in the provenance sidecars agrees.
- **Code differences between commits** (the git hash column of Table 1c) were not diffed. The CHANGELOG's statement
  that `faf610d2` and `9018ec79` train identically is UNVERIFIED here.
- **Seeds** are not printed in the trainer logs; "one seed per run" rests on the launcher passing none.
- **Not covered:** the C1 depth-2 run (`c1_rollout_full_b16`) and the 1-epoch n_future proxies, which appear in
  the 1-yr screens; the surgical soil-slicing job that produced F-warm's start file; the LR and batch sweeps
  (`lr_*`, `b32fix_*`, `b48*`), the multinode scaling probes and the smokes in `runs/makani_mn_scaling/`.
- **Not run.** **G4D-sf2** (G4D's recipe with `scale_factor` 2, prepared 2026-10-19 in
  `runs/makani_mn_scaling/g4d_sf2_2026-10-19/`) was never submitted: there is no run directory or log. The A-based
  Stage-1 arms **T-anneal**, **`anneal_dryair` from A** and **T-d8** were qdel'd before running (jobs 7650020,
  7650769 and 7650770, 2026-09-28, all UNVERIFIED from KNOWLEDGE.md §1); their F-based replacements never ran either.
- **Rollout records.** Arms rolled by the rollbatch and precision-cross jobs print no checkpoint path; their weights
  and flags are read from the submit scripts, and their epochs from the member logs. The two oldest five-year jobs
  (7649597, 7649650) predate the `dry_air_fix` field in the member log. All rollout NetCDFs were deleted on
  2026-11-02; the logs used here were kept.

## 6. Verdict

This configuration set compares depth, trained-in vs inference-time dry air, EMA, and training and rollout
precision mostly one or two settings at a time on the 99-channel F/G lineage. The most important missing
configuration is a second training seed of any recipe, without which no difference between arms can be separated
from run-to-run training noise.
