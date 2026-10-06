# makani_sfno — what we know, verified (E3SM on Polaris)

One file for what the makani track has established, what broke, and what was believed and then
retracted. Each entry is tagged and cites its **primary** artifact: a job log, a CSV row, or code.
A doc citation alone is not enough. Started 2026-10-06 from
`polaris_makani_docs_consolidation_handoff.md`. The doc inventory, unverified, is
[`docs/2026-10-06_knowledge_inventory.md`](docs/2026-10-06_knowledge_inventory.md).

## Status (update this block before you stop)

| § | section | tier | state (2026-10-06) |
|---|---|---|---|
| 0 | Where the 5-year accuracy path stands | — | **done**. Rewrite it when a new 5-yr result lands |
| 1 | Checkpoint lineages | 1 | **done** (recipes and survival from logs and CSVs; G/H secondary) |
| 2 | Hyperparameters | 1 | **done** (A recipe, LR ceiling, β₂, clip and fine-tune schedule from logs; batch-48 and memory secondary) |
| 3 | Depth / `n_future` | 2 | done at Tier 2 |
| 4 | Mass conservation / dry-air | 2 (numbers Tier 1) | **done** |
| 5 | Multi-node scaling | 2 | done at Tier 2 (brief; `makani_bench_report.md` is the full record) |
| 6 | Climate screening methodology | 2 | done |
| 7 | ACE2 comparison and ports | 2 | done (port-status table checked against code) |
| 8 | Known silent-failure traps | 1 | **done**: 14 from the 09-04 handoff + 6 later |
| 9 | Retired / contradicted claims | 1 | **done** for everything that steers current work. Older scaling and memory rows are `❓ secondary-only` |
| 10 | Dangling citations and contradictions | — | **done** for this pass (17 items) |
| 11 | Adversarial review: open questions and next experiments | — | done (2 critics, 2026-10-06) |

**Not covered in this pass** (inventory lines only): `docs/codex_reviews/` (61 files), `docs/hpo_distill/`,
`docs/run_log/`, `docs/audit_snapshots/`; the 65 upstream PlaSim/AI-RES docs from the subtree import
(inventory §4); CHANGELOG makani entries before 2026-09-01, apart from the retractions grep. Scaling and
memory rows in §9 are carried from their source docs and tagged secondary.

**Resuming after a context clear:** read this block, then §0, then §10. Verification rules: handoff §2.
Tier 1 = checked against the primary artifact. Tier 2 = a secondary citation is acceptable, tagged
`❓ secondary-only`.

## How to read the tags

- `✅ CONFIRMED`: the primary artifact was opened on 2026-10-06 and says this. `✅ (critic)` means a
  2026-10-06 review agent opened it and this session did not re-open it.
- `❌ BROKEN / LIVE TRAP`: the failure is real and still present in the code or system today.
- `🔧 FIXED`: the trap was real. The fix was checked in today's code, and a later job shows it working.
- `🔁 RETIRED`: once believed, now contradicted by later evidence. Do not resurrect it.
- `❓ secondary-only`: carried from a doc that cites it. The primary artifact was not opened.
- `❓ UNVERIFIED`: no primary source found. Listed so it stays visible.

Paths: `$MEMBER_ROOT` = `/eagle/projects/lighthouse-uchicago/members/mehta5`. `RUNS` =
`$MEMBER_ROOT/runs/makani_mn_scaling` (training runs: `<tag>.log`, `e3sm_mn_scaling.<tag>.yaml` = the
rendered config). `EVAL` = `$MEMBER_ROOT/runs/makani_eval`. "Installed makani" means
`$MEMBER_ROOT/conda-envs/sfno-venv/lib/python3.12/site-packages/makani` (0.2.0). "The fork" means
`makani_sfno/src/` at `44a3ea9c`. Line numbers drift (`.claude/comments.md`), so code is cited by file
and symbol. A line number, where given, is as of 2026-10-06.

---

## 0. Where the 5-year accuracy path stands (2026-10-06)

The goal is the most **accurate** 5-year rollout, not just one that stays finite. What is measured:

1. **Survival is solved for the depth-4 lineage. Accuracy has barely been measured.** B22, B24 and
   DRYAIR24 finish the 8-member, 5-year protocol 8/8 (§1). The one accuracy score of any 5-yr run
   (`EVAL/climate_fidelity_7709968/fidelity_b22_b24.csv`, time-mean pattern RMSE vs the true 2045–49
   climate) covers B22, B24 and two B22 inference variants only. Mean over 101 channels, per member:
   **B22 0.246–0.259 σ, B24 0.310–0.327 σ**. DRYAIR24 and soil-fix have **never** been accuracy-scored.
   No pre-registered rule ranks on accuracy. ✅
2. **The PS champion is not the temperature champion.** At 1 yr vs truth, TREFHT drift is DRYAIR24
   −1.44 K, B22 −0.33 K, soil-fix e21 −0.04 K. PS drift is DRYAIR24 −0.15 hPa, B22 +0.25 to +0.67 hPa,
   soil-fix e21 −8.8 hPa. ✅ (§4, §6)
3. **Every surviving arm leaves 3σ at the model top (`V_l00`, then `V_l01`) at 0.9–1.5 yr.** That
   holds with and without dry-air, and no built or queued arm targets it. ✅ (critic; `readout.log`
   "first past 3σ")
4. **Two causal claims steering the plan are confounded** (§9):
   - **"Dropping soil destabilizes the depth-4 recipe" (F vs B).** F was trained **single-step** on
     A's recipe (`.o7660250`: `multistep_count = 1`). Its 1-yr deaths, at 281–1243 steps, match A's
     (490–595). No measurement isolates soil. ✅
   - **"DRYAIR24 vs B22/B24" is not a matched comparison.** B ran `scheduler_T_max 100` and was never
     annealed. The DRYAIR and SOILFIX arms ran `T_max 22`, annealed to 1e-6. A plain-anneal control
     from B_e01 has never run. ✅
5. **The external accuracy bar does not exist yet.** The ai2 ACE2-EAMv3-on-our-archive 5-yr test was
   called "decisive and unread" by the 2026-10-02 decision job. In fact it never ran:
   - its prep job `ace2_5yr_prep.o7704377` ends `ERROR PROBE_INCOMPLETE` / `ERROR PROBE_FAILED`;
   - every probe died on `dacite ... UnexpectedDataError: can not match "log_nino34_index",
     "log_zonal_mean_images"`, i.e. ai2's inference config has keys the installed fme does not accept;
   - the `afterok` rollout 7704379 left no log. ✅
6. **ACE2 ports.** `DryAirFix` is a faithful port of fme's step 2 and does what it says: dry mass is
   flat over 5 yr. `SoilMoistureFix` is not a port: fme has no such corrector. fme's step 1 (water
   positivity, trained through) is not ported. Step 3 (moisture budget) is not closable on our channel
   set, and in fme it does not touch the rolled water state anyway. ✅ (§7)

Recommended next steps from the 2026-10-06 adversarial review are in §11. Queue choices are the
operator's; metric and loss definitions are jesswan's.

---

## 1. Checkpoint lineages

Recipes are read from each run's rendered yaml and its log (`multistep_count = N` ⇒ `n_future = N−1`).
A config-side `n_future` is ignored (§8 #5).

| name | run tag (`RUNS/`) | parent | depth | recipe | job | status | 5-yr protocol (8 members) | 1-yr screen |
|---|---|---|---|---|---|---|---|---|
| **A** | `prod1n_b32_sgdr` | scratch | 1 (`multistep_count = 1`) ✅ | LR 2e-3, SGDR T₀=20, 3-ep warmup, β₂ 0.95, clip 32, batch 32, 243 ep ✅ | 7585080 | done; best = epoch 243, val 0.01284 (single-step, §8 #11) | 8/8 non-finite, leads 317–849 `❓ secondary-only` (CHANGELOG 2026-09-24) | dies at 595 (f1092) and 490 (f1156) ✅ |
| **C1** | `c1_rollout_full_b16` | A e243 | 2 (`multistep_count = 2`) ✅ | batch 16, 24 ep | 7593272 | done | not run | 11 of 24 epochs survive f1092 ✅ (`docs/2026-09-24_climate_screen_7649647.csv`) |
| **B** | `nf4_prod_b16_r1` | A e243 | **5** (`multistep_count = 5`) ✅ | LR 4e-4, `CosineAnnealingLR` **`T_max 100`** ✅, 1-ep warmup, batch 16, 24 ep. LR at e24 ≈ 3.54e-4 (89 % of peak, `S1A_OK` 7649946) ✅ | 7630639 | done. `best_ckpt` = **epoch 1** (lowest single-step val) | **B_e01: 1/8** (survivor −269 hPa) `❓ secondary-only`. **B22: 8/8** ✅, **B24: 8/8** ✅ (critic) | B e01/e21–e24 all survive both starts ✅ (7649647, 7649792 CSVs) |
| **DRYAIR** | `fs_anneal_dryair_nf4_b16_rb01` | B_e01 | 5 ✅ | B's LR and batch, **`T_max 22`** (anneals to 1e-6), `conserve_dry_air True` ✅ | 7709268 | done | **DRYAIR24: 8/8** ✅. DRYAIR01: 0/8, truncates at 5662–6047 ✅ | all 6 survive; PS −0.04 to −0.49 hPa ✅ |
| **SOILFIX** | `fs_anneal_soilfix_nf4_b16_rb01` | B_e01 | 5 ✅ | DRYAIR's recipe with `conserve_soil_moisture True` in place of dry-air ✅ | 7715005 | done (`MAKANI_MN_SCALING_OK`) ✅ | not run | all 6 survive; PS −8.8 to −17.6 hPa (e01 +6.4) ✅ |
| **F** (warm) | `f_nosoil_2n_b32_e43_warm` | A e243 sliced to 99 channels (7646690) | **1** (`multistep_count = 1`) ✅ | **A's recipe** (provenance file: "base recipe = prod1n_b32_sgdr"), EMA 0.9995, batch 32, 43 ep ✅ | 7660250 | done | not run | **0/16** rollouts survive (281–1243 steps); EMA best (epoch 30) lasts longest ✅ |
| Fsurg | `surgical_nosoil_7646690` | A e243 sliced + 20 steps at LR 1e-5 | 1 | — | 7646690 | val 0.014266 (`SURGICAL_TRANSFER_PASS`) `❓ secondary-only` | not run | dies at 293 / 343 ✅ |
| **F-scratch** | `f_nosoil_2n_b32_e23_scratch` | random init | 1 ✅ | A's recipe, 23 ep ✅ | 7718436 | **running** (`capacity`; `qstat` 2026-10-06) | — | — |
| **G** | `g_ace2vars_*` | — | 1 (`n_future: 0` in its yaml) | ACE2-EAMv3-matched channels: 83 in / 77 out, no TMQ, soil or Z3 | — | built, smoke-tested, **not queued** (operator, 2026-10-03) `❓ secondary-only` | — | — |
| **H** | `h_ace2vars_train2020_*` | — | — | G's channels, train 2020–2044 | — | built, not queued `❓ secondary-only` | — | — |

Notes:
- **Stable copies:** `nf4_prod_b16_r1/training_checkpoints/ckpt_mp0_{e22,e24}_stable.tar`, epochs
  confirmed from inside the files (`ckpt_epoch=` in member logs). B's `ckpt_mp0_v{0..3}` are a
  rotating slot: never resume `nf4_prod_b16_r1`. `❓ secondary-only` (CHANGELOG 2026-10-02)
- **Epoch numbers count from the start of each run.** DRYAIR01 descends from B_e01 and is not B_e01.
- **Cancelled before running:** the A-based Stage-1 arms 7650020 (T-anneal), 7650769
  (`anneal_dryair` from A) and 7650770 (T-d8), on 2026-09-28, to be re-based on F. Source:
  `polaris_makani_f_finetune_handoff.md`, on branch `worktree-monitor-ace2` only. The F-based Stage-1
  arms (T-anneal-F, `anneal_dryair`-F) have **not** run either. `❓ secondary-only`
- **B differs from A by depth *and* epochs.** No single-step-continuation control of A exists, so
  "depth-4 is what stabilizes" is supported by C1-vs-B and the 1-epoch proxies (§3), not by a matched
  control.
- **`best_ckpt` (lowest single-step validation loss) picks badly for rollouts.** B's `best_ckpt` is
  B_e01: 1/8 at 5 yr, while B22/B24 go 8/8. Validation loss is never used to rank checkpoints.

## 2. Hyperparameters

| knob | value / finding | state | evidence |
|---|---|---|---|
| A's peak LR | **2.0e-3**, `CosineAnnealingWarmRestarts` T₀=20, T_mult 1, min 1e-6, 3-epoch warmup from `lr_start` 0.01 (= 2e-5) | ✅ | `RUNS/e3sm_mn_scaling.prod1n_b32_sgdr.yaml` |
| LR ceiling | **(2e-3, 3e-3]**, and it does not move with batch 32 → 48. Every arm at ≥ 3e-3 collapses into the same dead attractor (train ≈ 0.106) | ✅ 8 logged arms | `.o7587738`–`.o7587741` and `.o7587776`–`.o7587778` (3e-3), plus `.o7587742` (4.5e-3): per-epoch training loss ends at 0.105–0.107 after a spike. 7587779's log has no loss lines. CHANGELOG's "9 of 9 at 3e-3" was not re-counted |
| β₂ | **0.95. 0.999 is the fastest route to collapse**: 4 of 4 logged 0.999 arms blew up at epoch 2 (train 6.05e11 / 3.42e5 / 6.21e7 / 2.67e12) | ✅ (§9) | `.o7587739`, `.o7587741`, `.o7587742`, `.o7587777` |
| grad clip | 32 in production. At 3e-3, clip 1.0 delays collapse to epoch 6 (both batches) but prevents nothing. "32 never engages: grad-norm peak 0.2995" | ✅ delay / ❓ peak | `.o7587740`, `.o7587778` (epoch 6: 3.13e6 / 1.5). Peak: CHANGELOG 2026-09-03 `❓ secondary-only` |
| batch | 32 for A, 16 for every rollout fine-tune. Batch 48 was a dead end (no LR headroom, 87.6 % of card) | ✅ batches / ❓ 48 | rendered yamls and `.o` headers (`global_batch=16`). Batch 48: CHANGELOG 2026-09-03 |
| fine-tune schedule | LR 4e-4, `CosineAnnealingLR`, 1-epoch warmup. Warmup and cosine step **once per epoch**, so **epoch 1 runs at a constant 4e-6** and epoch e ≥ 2 at cosine t = e − 2. `SCHED_TMAX=22` puts epoch 24 at 1e-6 | ✅ | `S1A_OK` job 7649946: "saved_lr(=epoch 24)=3.542173919e-04 formula(t=e-2)=… ok" for B's checkpoint |
| B's own schedule | `T_max 100` over 24 epochs, so it was never annealed (e24 ≈ 3.54e-4). **Not matched to the DRYAIR/SOILFIX arms (`T_max 22`)** | ✅ | `RUNS/nf4_prod_b16_r1.log` `scheduler_T_max 100`; `fs_anneal_dryair_nf4_b16_rb01.log` `scheduler_T_max 22` |
| EMA | F uses 0.9995 (≈1.5 epochs at 1368 updates/epoch), deliberately not ACE2's 0.999. F3: EMA best lasted longest at both starts | ✅ value / ✅ F3 | `f_nosoil_2n_b32_e43_warm.warmstart_provenance.txt`; F3 CSVs |
| memory model | peak ≈ 2.31 + 2.12 × samples/GPU GiB + ~8 GiB fixed. Retrodicts the batch-64 OOM. Depth 8 fits (19.66 GB, 7650039); depth 16 OOMs (7650263) | ❓ secondary-only | CHANGELOG 2026-09-04, 2026-09-24 |

## 3. Depth / `n_future`

- **Depth 4 buys survival.** In the Stage-0 1-yr screen, B ranks above C1 (depth 1) at 4 of 5 matched
  epochs. All 5 B checkpoints survive; 2 of 5 matched C1 epochs do. The 1-epoch proxies agree: both
  nf1 die, nf3 and both nf4 survive. (`docs/2026-09-24_climate_screen_7649647.csv`; CHANGELOG
  2026-09-24) `❓ secondary-only` for the reading; the CSV is the job's output.
- **More depth-4 epochs buy 5-yr survival**: B_e01 1/8 → B22/B24 8/8, same run and schedule.
- **Skill per epoch:** `n_future=4` for **one** epoch gives −4.66 % / −4.63 % RMSE at 126 h (two
  seeds), against depth-1's −3.00 % over 24 epochs. Batch 8 costs 2.3 pp against batch 16.
  (`docs/2026-09-11_nfuture_ladder_result.md`, `docs/2026-09-15_nfuture_ladder_d1_and_replication.md`)
  `❓ secondary-only`
- **CRPS / a distributional loss is not indicated.** The K=56 (14-day) read-out has VR 0.99–1.02 at
  every lead, so no amplitude collapse; NRMSE336 is 0.970. Its `DRIFT_FIRST` verdict came from one
  channel, `Z3_l17`. (job 7639537; `docs/2026-09-20_k56_readout_prereg.md`) `❓ secondary-only`
- **The lagged ensemble does not improve the deterministic forecast**: its mean is 31.7 % worse than
  the freshest member, SSR 1.66, CRPS 28.7 % better. (job 7643271; `docs/2026-09-21_lagged_ensemble_result.md`)
  `❓ secondary-only`

## 4. Mass conservation / dry-air

**What the fix is** (✅ code, fork `sfno_training/models/mass_fix.py` `DryAirFix`): it shifts predicted
PS uniformly so that the area-weighted global mean of **PS − g·TMQ** equals the step input's.
- Computed in float64, with equiangular cos-lat cell weights (`equiangular_weights`) and
  `GRAVITY = 9.80665`, fme's value.
- The output dtype is promoted to at least fp32 (`torch.promote_types(pred_z.dtype, torch.float32)`).
  That is the bf16 fix `6c29bf18`. Before it, the shift rounded away in bf16 and job 7650512 became a
  spatially non-uniform PS distortion.
- Applied in `PlasimPreprocessor`'s `history_denormalize` hook, so it acts in training, validation and
  inference, and the corrected state is the one fed back. ✅ (critic: installed makani `models/stepper.py`)

**5-year budget** (8 members each; job 7719173 `DRYAIR_BUDGET_OK`,
`b-continuation-dryair/makani_sfno/makani_dryair_budget.o7719173`). ✅

| arm | dPS (hPa) | dDRY (hPa) | global-mean TMQ (kg/m²), start → end | monthly TMQ cell minimum |
|---|---|---|---|---|
| **DRYAIR24** | −0.14 to −0.24 | **0.00** | 26.4–27.1 → 24.3–25.7 (never negative) | −6.7 to −12.5 |
| **B22** | −5.97 to −8.49 | −5.86 to −8.37 | 26.4–27.1 → 25.4–26.2 | −7.9 to −10.5 |
| B22 + RELHUM clamp (inference) | −3.59 to −7.09 | −3.48 to −7.00 | — | — |
| B24 | +12.7 to +17.7 | +12.4 to +17.4 | gains about +3 | −21 to −38 |

B24's row is ✅ (critic) and CHANGELOG 2026-10-06.

- **Units trap:** `readout.log`'s `drift PS (phys, lead L − lead 1)` is in **Pa** (`lead1=98564.8`).
  ✅ `EVAL/climate_protocol_7711720/readout.log`
- **Plain-B drift is dry-mass leak** (dDRY ≈ dPS). DRYAIR24's residual drift is all water: global TMQ
  falls about 7 % in 5 yr, which is fme's missing step 1 (§7).
- **Survival is not attributable to dry-air:** B22 and B24 also go 8/8. ✅
- **Post-hoc dry-air hurts.** On the 16 B/C1 checkpoints that survived a year (7650652), survivors
  fell 16 → 14, which reads as "hurts" by prereg A2. 12 of 14 survivors ended colder at T_l17. On
  B_e01 over 5 yr it gave 0/8, collapsing at 2.28–2.48 yr (7707597). ✅ labels
  (`docs/2026-09-24_climate_screen_7650652_dryair_on.csv`: B_e01/21–24 and C1 epochs); 5-yr result
  `❓ secondary-only`. **Trained-with beats post-hoc on B_e01**, but post-hoc has never been tried on a
  mature checkpoint (B22/B24), see §11.
- **Temperature:** DRYAIR24 is colder than B22 at 1 yr. TREFHT vs truth is −1.44 K vs −0.33 K, T_l17
  −1.43 K vs −0.44 K (`climate_screen_7711659` vs `7650512`). ✅ Whether that comes from the constraint,
  the annealed schedule or the missing water guard is open.
- **Soil-fix** (`SoilMoistureFix`) leaves dry mass free: dry_drift ≈ ps_drift in every row of
  `climate_screen_7719183`, at −8.8 to −17.6 hPa (e01 +6.4). Its 1-yr TREFHT is the best measured
  (e21 −0.04 K). ✅
- **Negativity** (7650442, 1-yr, 4 checkpoints): truth is never negative in any moisture channel. In
  every rollout PRECT is negative over 10–17 % of the globe, RELHUM negatives sit at the model top
  (l00–l02), and SOILWATER_10CM's worst cells reach −89 to −289. `❓ secondary-only`
  (`EVAL/negativity_7650442/negativity.md`)

## 5. Multi-node scaling (brief)

Full record: `makani_bench_report.md`, `polaris_nccl_*.md`. Everything here is `❓ secondary-only`
unless marked.
- **Every TCP-era multi-node number is void for today's stack.** The 128-node production run
  (7566145) ran its collectives over TCP (2026-09-17, `nccl-tests` 7629065/7629082/7629096). Before the
  libfabric/CXI fix, "multi-node is not faster" was a fabric artifact.
- **On CXI, 2 nodes beat 1** at global batch 32: 225.2 ms/step vs 365.4 ms with the same knobs, 81.1 %
  per-GPU efficiency (7669001; 7580338). So F's and G's 2-node shape is justified.
- **Spatial parallelism is a memory tool, not a speed tool:** h4w1 +26.0 %, h2w2 +85.6 %, h2w4
  +173.7 % step time (7669001). **`w=4` no longer hangs, but h2w4's loss is ~26 % higher at epoch 2.**
  Do not use `w=4` in production before a same-checkpoint equivalence check. `DryAirFix` refuses
  h/w > 1 (its global mean would be per-tile).
- Synthetic data is not a neutral stand-in for scaling: it removes I/O–comms overlap, and at 4 nodes
  synthetic was 35 % slower than real (7633729). The 84 % and 92.9 % makani projections are retracted
  (§9).

## 6. Climate screening methodology

| stage | what | ranking rule | source |
|---|---|---|---|
| Stage-0 1-yr screen | one rollout per checkpoint from 2044 f1092 (Oct 1), 1460 leads; drift = model − truth global mean | survivors → fewest `n_past_3sigma` → smallest \|PS drift @1460\| | `docs/2026-09-24_climate_screen_prereg.md` §3 (✅ critic, L35–42). Addendum A1 (second start f1156, top-3 at both) |
| F3 / D7 | raw vs EMA for F | survival gate, then noise-scaled PS gap, then `n_past_3sigma`; default raw | same doc §A3 |
| 5-yr protocol (jesswan) | 8 members start at 2044 frame 1092 + 16i (Oct 1, 5 … 29) and all end 2049-12-31 18:00; scoring from 2045-01-01; deterministic; forcing read by valid time | **none pre-registered for 5 yr** | `polaris_climate_run.pbs`. Member lengths 7667 − 16i ✅ (7719173) |
| climate fidelity | time-mean bias and pattern RMSE per channel vs the true 2045–49 mean, area-weighted, physical and σ units | not part of any rule | `polaris/build_true_climatology.py`, `polaris/score_climate_fidelity.py`. Ran **once** (7709968) ✅ |

What the screens can and cannot see:
- **A 1-yr screen cannot see post-hoc dry-air collapse.** B/C1 + post-hoc dry-air looked clean at
  1 yr (7650652) and collapsed at 2.3–2.5 yr on B_e01 (7707597). With units corrected, DRYAIR's 1-yr
  (−0.145 hPa) and 5-yr (−0.21 hPa) PS agree, so that pair is **not** an example of the 1-yr trap
  (§9).
- **PS drift is a fixed property of each checkpoint, not IC noise:** it reproduces to about 1–2 hPa
  across the two starts. But **neighbouring epochs differ by up to 70 hPa/yr** (B e21–e24: −58, +0.7,
  −68, +4.6). A single "best" epoch samples a checkpoint lottery. (7649647 vs 7649792 CSVs in
  `docs/`) `❓ secondary-only` for the reproducibility reading
- **`survived` means finite, not sane:** C1 e23 stayed finite from f1156 with global means near 1e16.
  `❓ secondary-only`
- **Accuracy is time-mean only.** There is no seasonal-cycle, monthly-climatology or interannual-variance
  metric yet. The fidelity CSV's B22/B24 numbers are in §0. `B22fp32` is bit-identical to B22 (§7).
- Every blow-up so far starts at the model top (`V_l00`, `RELHUM_l00/l04`), not at near-surface Z3.
  (7649647 CSV `first_past_3sigma`; F3 CSVs)

## 7. ACE2 comparison and ports

The canonical code-level comparison is `docs/2026-10-01_makani_vs_ace2_code_audit.md` (job 7703518).
Its §9 corrects about 25 claims in older docs (`ace2_vs_makani_differences.md`,
`docs/2026-09-10_ace2_comparison_the_corrector.md`, `polaris_makani_ace2_ports_handoff.md`). Prefer it.
Its single biggest correction: ACE2 and makani do **not** share the same SFNO. They differ in skip
wiring, position embedding, internal resolution and SHT grid. `❓ secondary-only`

**fme's atmosphere corrector runs three steps, in order, every step**
(`ACE2_retrain/ace_exp/fme/core/corrector/atmosphere.py`, main checkout only) ✅:
1. **Force water positive first.** The code comment says to do this "before imposing other
   conservation correctors".
2. **Conserve dry air** (`_force_conserve_dry_air`).
3. **Moisture budget** (`_force_conserve_moisture`). It reads `total_water_path` but **only sets
   precipitation rate, evaporation rate and the advection tendency**. Those are diagnostics; it does
   not modify the rolled water state.

| ACE2 mechanism | in makani? | flag / where | state |
|---|---|---|---|
| dry-air conservation (step 2) | ✅ **ported, faithful** | `CONSERVE_DRY_AIR=1` (train) / `--dry-air-fix on` (inference); `mass_fix.py` | flag-off equivalence bitwise (`DRYAIR_OFF_EQUIV_OK` 7650461, 7669103; ✅ critic). Documented deviation: TMQ is held fixed where fme holds q, about 1e-3 of the correction (critic) |
| water positivity (step 1) | ⚠ **inference only** | `--force-positive {config,off,relhum}` in `climate_rollout.py` (reuses `rollout_driver`'s clamp); `NonNegativeConstraint` opt-in, default off | never trained through. The RELHUM 18-channel clamp on B22: dPS −3.6 to −7.1 vs −6.0 to −8.5 hPa, survival unchanged ✅ (7719173). Fidelity unchanged (0.250 vs 0.251 σ) ✅ |
| moisture budget (step 3) | ❌ not ported | — | **not closable on our contract**: no evaporation or advection channel (PRECT is the only flux in `e3sm_alldata_full.yaml`, ✅ critic). It would not pin TMQ even in fme (above) ✅ |
| "frozen soil moisture corrector" | ⚠ **not a port** | `CONSERVE_SOIL_MOISTURE=1`; `soil_moisture_fix.py` `SoilMoistureFix` | fme's `AtmosphereCorrectorConfig` has no such field. The module docstring calls it a fresh reimplementation. 18 unit tests (7713397); **no PBS-level flag-off equivalence gate yet** |
| tendency-unit loss normalization | ❌ probe only | `temp_diff_normalization` (makani `loss.py`, 1/r) | not a one-line flip: PRECT's weight would be 8.3e-4 and `Z3_l17`'s 9518 (7646192). Audit ranks it #1. No full-split `time_diff_stds.npy` exists `❓ secondary-only` |
| `channel_weights: auto` | ⚠ = `constant` here | installed makani keys `auto` on lowercase ERA5 names | every E3SM name falls through to the default `❓ secondary-only` |
| EMA | ✅ | launcher `EMA=1 EMA_DECAY=0.9995` | used in F ✅ |
| rollout-metric checkpoint selection | ❌ absent | — | ACE2 selects on a long-rollout metric. Ours picks by single-step val, which picked B_e01 (§1) |
| fp32 state between steps | ❌ **untested** | `--feedback-fp32` exists but is a no-op | it casts the bf16 prediction to fp32 *after* autocast. That is lossless, so the trajectory is bit-identical (7707597; budget rows identical in 7719173) ✅. True fp32 inference (autocast off) has never run |
| ACE2-EAMv3 channel set | built as port **G**, not queued | `e3sm_alldata_ace2vars.yaml` | no TMQ, so `DryAirFix` is refused by name on G `❓ secondary-only` |

- **The ACE2 5-yr reference never ran.** See §0 item 5. ✅
- **jesswan's sign-off** covers the ACE2-port mechanism list (operator, 2026-10-02). Don't raise it as
  a caveat for those mechanisms. A new channel list for trained-through positivity, or a TMQ-retaining
  G variant, may be a new ask (§11).

---

## 8. Known silent-failure traps

Each of these cost at least one job and none announces itself. Source list:
`polaris_makani_analysis_ensemble_handoff.md` §6 (10 numbered) plus four described separately in
its §1 and §3. That is **14, counted by opening the file**. The handoff's own §0a calls
`SKIP_TRAIN` "the tenth trap" while its §6 numbers it 9 (see §10). Each was re-checked
2026-10-06.

### 8a. The 2026-09-04 list

| # | trap | state | primary evidence (checked 2026-10-06) |
|---|---|---|---|
| 1 | **`ckpt_mp0_v0.tar` or nothing.** `resuming` is true only if that exact file exists. Seed a scratch dir under any other name and the job trains from scratch, silently. Assert `resuming True` in the log. | ❌ LIVE (by design) | installed makani `train.py`: `resuming` is computed from `checkpoint_path.format(..., checkpoint_version=0)` existing (L101–105) |
| 2 | **A seeded/forked expdir needs `WANDB=0`.** With wandb on *and* resuming, makani reads `<wandb_dir>/wandb/makani_restart.yaml`. Only a fresh run writes that file, so every rank dies at construction. Don't copy the file in: the job would write into the parent run's wandb history. | ❌ LIVE | installed makani `utils/driver.py` `Driver._init_wandb`: `if not params.resuming:` writes `makani_restart.yaml`; else reads it (L248/270/275; the handoff's L237–248 has drifted). The rank-death log was not re-opened: symptom `❓ secondary-only` |
| 3 | **`LOAD_COUNTERS=0` for any validation-only run.** A restored epoch counter (243) against `EPOCHS=1` makes the epoch loop empty. Validation lives inside it, so the job exits 0 with no loss. | ✅ CONFIRMED, LIVE | `makani_mn_scaling.o7592332`: `max_epochs 1`, `resuming True`, and **no** `validation loss` line anywhere in the log |
| 4 | **`LOAD_LOSS=0` when `n_future` changes.** `LossHandler`'s running stats are shaped by `n_future`. | ✅ CONFIRMED, LIVE | `makani_mn_scaling.o7590350`: `size mismatch for running_mean: copying a param with shape torch.Size([101]) from checkpoint, the shape in current model is torch.Size([202])` |
| 5 | **A config-side `n_future` does nothing.** It is overwritten from `--multistep_count`. `MULTISTEP` is the only handle. Rendered yamls all read `n_future: 0`, so read `multistep_count` from the log. | ✅ CONFIRMED, LIVE | both installed makani `train.py` (L119) and the fork's `train_plasim.py` (L343) set `params["n_future"] = args.multistep_count - 1` |
| 6 | **`pretrained` and `resuming` are mutually exclusive.** A fine-tune needs a **new** `RUN_NUM`, or resuming silently wins. | ✅ CONFIRMED, LIVE | installed makani `deterministic_trainer.py`: `if self.params.pretrained and not self.params.resuming:` (L237) |
| 7 | **`qalter` is refused on Polaris** for every attribute (rc=32). Walltime and dependencies are fixed at submit; stagger with `qhold`/`qrls`. | ❓ secondary-only | `polaris_pbs_notes.md` §1b; CHANGELOG 2026-09-03 records the error text. No job log captures it |
| 8 | ~~**`capacity` cannot hold a queued successor**~~ | 🔁 **RETIRED** (see §9) | `qstat -Qf capacity` on 2026-10-06: `max_queued = [p:PBS_GENERIC=2]`, `max_run = [p:PBS_GENERIC=1]`. 7718436 (capacity) was queued behind 7715005: the `.o7715005` header shows it started 2026-10-05T17:40; 7718436's header shows it started 2026-10-06T01:44, after 7715005 ended. The cap is per *project*, so the second slot is free only if no other member holds it |
| 9 | **`SKIP_TRAIN=1` validated nothing before 2026-09-04.** The fork's entrypoint short-circuited before `trainer.train()`. makani handles `skip_training` *inside* `train()`. | 🔧 FIXED | fork `train_plasim.py` now calls `trainer.train()` under `skip_training`, with a comment citing 7592332/3/6. `makani_mn_scaling.o7598662` (va=3): `validation loss: 0.012838906608521938`, reproducing production's 0.01284 |
| 10 | **Never re-run a stuck job before diagnosing.** Read the queued job's `comment`. | rule, not a code trap | CLAUDE.md #12; `polaris_pbs_notes.md` §1b |
| 11 | **`validation loss` is single-step at every `valid_autoreg_steps`.** No validation loss in this project is a multi-step score, production's 0.01284 included. | ✅ CONFIRMED, LIVE | `.o7598662`/`.o7598663`/`.o7598664` (va = 3 / 10 / 20) all print `validation loss: 0.012838906608521938` |
| 12 | **makani computed no per-lead metric on E3SM channels.** The handoff says they were "computed and discarded" because they went only to wandb. **That mechanism is retracted** (CHANGELOG 2026-09-10). `MetricsHandler` intersects its ERA5 default names (`u10m, t2m, sp, sst, u500, z500, q500, q50`) with ours. The intersection is empty, so no metric handle is ever built. | 🔧 FIXED (fork) | installed makani `utils/metric.py`: defaults (L239–245), intersection (L269–275), `if self.l1_var_names:` (L323). Fork `plasim_trainer.py` now builds per-lead metrics on E3SM names and prints `Per-lead validation metrics:`. First seen in `makani_mn_scaling.o7603089` |
| 13 | **`_extract_truth_sic` read forcing channel 5 by position.** E3SM has 7 forcings, so the length guard passed and returned a different variable labelled `truth_sic`. | 🔧 FIXED | fork `sfno_inference/rollout_driver.py`: sea ice is looked up by name (`_SIC_ALIASES = ("sic", "ice", ...)`). truth_sic is disabled with a warning when names or stats are missing |
| 14 | **`save_raw_forecasts: True` is a dead key**: nothing reads it. | ❌ still no reader | no `save_raw_forecasts` reader anywhere in installed makani 0.2.0 or the fork's `src/` (searched 2026-10-06) |

### 8b. Traps found after 2026-09-04

| trap | state | primary evidence |
|---|---|---|
| **A queued makani job freezes the worktree it was submitted from.** PBS copies the *script* at submit, but Python is imported from `$PBS_O_WORKDIR/src` when the job **starts**, and again on every preemption restart. Edit `src/` while a job is queued and the job runs the edit. | ✅ CONFIRMED, LIVE | `polaris_makani_multinode_scaling.pbs`: `MAKANI_ROOT` from `PBS_O_WORKDIR`, then `export PYTHONPATH="${MAKANI_ROOT}/src:..."`. Same in `polaris_climate_run.pbs` |
| **One `debug` job per user at a time, and a second submission can be refused at `qsub`.** Soil-fix screen 7719183 was first rejected this way. | ✅ `max_run` / ❓ `max_queued` | `qstat -Qf debug` 2026-10-06: `max_run = [u:PBS_GENERIC=1]`. It shows **no** queue-level `max_queued` (so the refusal comes from elsewhere, e.g. a server limit or hook). `polaris_pbs_notes.md` states `max_queued 1 per USER`; the rejection is CHANGELOG 2026-10-06 (`❓ secondary-only`) |
| **`readout.log` drifts are in physical units, Pa for PS.** Reading Pa as hPa reversed a conclusion for a day (§9). | ✅ CONFIRMED | `EVAL/climate_protocol_7711720/readout.log`: `drift PS … \| lead1=98564.8` |
| **The epoch counter steps warmup and cosine once per epoch**, so epoch 1 of every fine-tune runs at `lr_start × LR` (4e-6). A "1-epoch fine-tune" barely moves the weights. | ✅ CONFIRMED | `S1A_OK` 7649946 (§2) |
| **`--feedback-fp32` cannot test fp32 inference.** It casts a bf16 prediction after autocast, which is lossless, so it gives a bit-identical trajectory. | ✅ CONFIRMED | §7 |
| **makani's `_crps_skillspread_kernel` ignores `ensemble_weights`** (plain `torch.mean`). Weighted CRPS silently comes back unweighted. | ❓ secondary-only | CHANGELOG 2026-09-21 (pinned by `test_crps_matches_makani_unweighted`) |
| **Truth `.npz` files predate the summary's TMQ column.** Reusing `screen_truth_2044f{1092,1156}.npz` refuses with `TRUTH_MISSING_CHANNEL: TMQ`. Use the `_tmq` files. | ❓ secondary-only | CHANGELOG 2026-10-03/04 (it hit F3, 7711659 and 7671841). The `_tmq` files exist in `EVAL/` |
| **The makani launcher hardcodes `print_timings_frequency 10`**, so an arm shorter than 10 steps writes no timing row (`NO_STEP_TIMING`). | ❓ secondary-only | CHANGELOG 2026-09-19 |

---

## 9. Retired / contradicted claims

**Do not resurrect these.** Sources: analysis-handoff §5 (8 rows), the makani-matched Decisions-log
entries, and this session's checks. CHANGELOG's `## Known issues / failed approaches` has **no**
makani-specific entries beyond the 1.18 B-parameter note and the `torch_harmonics` version box (see
§10).

### 9a. Claims that steer current work (all re-checked 2026-10-06)

| retired claim | what is true | evidence | state |
|---|---|---|---|
| **"F and B share the same depth-4 recipe; dropping soil severely destabilizes it — F survives 2–10 months vs B's 2.5–5 years"** (CHANGELOG 2026-10-03 F3 result; repeated in the 2026-10-04 operator decision and in memory) | F was trained **single-step on A's recipe**: `multistep_count = 1`, LR 2e-3 SGDR, batch 32, warm from sliced A. B is `multistep_count = 5`, LR 4e-4, batch 16. The 2026-09-28 F handoff said "also single-step", and the 09-30 debate said "F = A + 43 single-step epochs". F's matched comparator is A: F3's 16 rollouts die at 281–1243 steps (medians ≈ 500 / 467), A e243 at 595 / 490. Indistinguishable at n = 2 starts. **Nothing measured isolates soil.** | `.o7660250` / `.o7718436`: `multistep_count = 1`; `RUNS/nf4_prod_b16_r1.log`: `= 5`; `f_nosoil_2n_b32_e43_warm.warmstart_provenance.txt`; `EVAL/climate_screen_770977{5,6}/screen_summary.csv`; `EVAL/surgical_soil_screen_7671841/f{1092,1156}` | 🔁 RETIRED ✅ |
| corollary: the reservoir-drift chain in `docs/2026-09-10_ace2_comparison_the_corrector.md` §4 | The 2026-09-23 entry pre-stated: "if divergence still occurs near step 500 with no soil reservoir in the state, the reservoir-drift chain was never the cause and … §4 needs retracting". Soil-free F dies near step 500. By that pre-stated test, the chain is not the cause. Science call: jesswan. | as above | 🔁 by its own pre-stated test ✅ |
| **"DRYAIR24's 5-yr PS drift is −21 hPa, larger than B22's; the 1-yr advantage does not hold at 5 yr"** (CHANGELOG 2026-10-04; findings handoff TL;DR 2) | **−0.14 to −0.24 hPa**, with dDRY 0.00. Pa was read as hPa. That is 30–75× smaller than B22/B24, and it agrees with the 1-yr screen (−0.145 hPa) | `.o7719173`; `readout.log` `lead1=98564.8` | 🔁 ✅ |
| "with the fix on, −21 hPa implies global TMQ fell ≈ 214 kg/m² (negative)" (findings handoff §2) | global TMQ 26.4–27.1 → 24.3–25.7 kg/m², never negative | `.o7719173` | 🔁 ✅ |
| "the fp32-feedback no-op is `Preprocessor2D.append_history` re-casting the history buffer" (CHANGELOG 2026-10-02; memory) | With `n_history: 0` (all our yamls), installed makani's `append_history` returns `x2` untouched. The flag up-casts an already-bf16 prediction, which is lossless | installed makani `models/preprocessor.py` `append_history` (`res = x2`); `RUNS/e3sm_mn_scaling.*.yaml` `n_history: 0`; `climate_driver.py` `fed = pred.to(torch.float32) if feedback_fp32 else pred` | 🔁 ✅ |
| **"the post-hoc dry-air 16→14 survivors result was on F, not B"** (CHANGELOG 2026-10-02, "the earlier post-hoc F result (16→14 survivors)"; memory) | 7650652's 16 checkpoints are **B e01/e21–e24 and 11 C1 epochs**. F had no checkpoints until 7660250 ran (queued 2026-09-25) | `docs/2026-09-24_climate_screen_7650652_dryair_on.csv` `label` column | 🔁 ✅ |
| "DRYAIR01 was trained at a high LR" (verbal, 2026-10-05) | epoch 1 runs at a constant 4e-6 | `S1A_OK` 7649946 | 🔁 ✅ |
| "B22/B24 are the approximate no-constraint control for DRYAIR24; only the restart differs" (findings handoff §1d, "not verified") | **Not matched:** B `T_max 100`, never annealed (e24 ≈ 3.54e-4); DRYAIR `T_max 22`, annealed to 1e-6. Flag, schedule and restart all differ | `RUNS/*.log` `scheduler_T_max` | 🔁 ✅ |
| "**`capacity` cannot hold a queued successor while a job runs**" (`polaris_pbs_notes.md` §1b #2; analysis handoff §6 #8; CHANGELOG 2026-09-03) | **2 queued+running per project**, `max_run 1`. A successor can be pre-staged if no other member holds the second slot | `qstat -Qf capacity` 2026-10-06; 7718436 behind 7715005 | 🔁 ✅ |
| **"the ACE2-EAMv3 5-yr verdict (7704379) exists and is unread"** (decision job 7707317 `decision.md`) | it **never ran**. Prep 7704377: `ERROR PROBE_INCOMPLETE` / `ERROR PROBE_FAILED` (`dacite` rejects `log_nino34_index`, `log_zonal_mean_images`); no `.o7704379` anywhere | `.claude/worktrees/makani-port/ace2_5yr_prep.o7704377`; `$MEMBER_ROOT/runs/ace2_5yr/7704377/probe.log` | 🔁 ✅ |
| "β₂ 0.95 → 0.999 will help" (handoff recommendation) | backwards: 4/4 logged 0.999 arms collapse at epoch 2. **Keep 0.95** | §2 | 🔁 ✅ |
| "per-lead metrics were computed every epoch and discarded" (analysis handoff §1; commit `ddd9bb5e`) | never computed: the ERA5-name intersection is empty (§8 #12) | installed makani `utils/metric.py` | 🔁 ✅ |
| "the pre-registered outcomes (a)/(b)/(c) are in `docs/2026-10-02_b_continuation_dryair_handoff.md`" (CHANGELOG 2026-10-02) | that file never existed: `git log --all -- '*b_continuation_dryair_handoff*'` is empty | git object DB | 🔁 ✅ |
| "the `ACE2 2 forward steps ≡ makani's 2`" (`2026-09-10_ace2_comparison_the_corrector.md` §1) | true of C1 only (`multistep_count = 2`). A, the shipped production checkpoint, is single-step | `RUNS/c1_rollout_full_b16.log`, `prod1n_b32_sgdr.log` | 🔁 ✅ |
| "the model predicts a tendency; add the output to the input" (`target: "tendency"`; 08-27 / 09-03 usage docs) | `target` is a dead key and the model predicts the **full next state**. Adding gives ≈ 2× state. The usage docs were corrected 2026-10-01 | CHANGELOG 2026-10-01; audit §9 | 🔁 `❓ secondary-only` |
| "ACE2 and makani are the same SFNO, one config knob apart" | different skip wiring, position embedding, internal resolution and SHT grid | audit §9 | 🔁 `❓ secondary-only` |
| "`w=4` fails outright on our grid" (analysis handoff §4; bench report §5b) | on CXI, h2w4 trains without hanging, but its loss is ~26 % higher. Not equivalent; no production use | 7669001 (CHANGELOG 2026-09-29) | 🔁 `❓ secondary-only` |
| "multi-node is not faster than 1 node" (TCP era) | on CXI, 2 nodes = 225.2 ms vs 365.4 ms, 81.1 % per-GPU efficiency | 7669001, 7580338 | 🔁 `❓ secondary-only` |

### 9b. Older retractions (scaling, memory, analysis), carried from their source docs

| retired claim | what is true | source | state |
|---|---|---|---|
| "sharding beats pure DDP 1.71×" | a confound, overturned by 7580297 (+43.6 % tax) | analysis handoff §5 | ❓ secondary-only |
| "memory ≈ 0.99 GB per sample-step, linear" / "≈ 2.5× per doubling" / "the memory cliff" | refuted by the batch-64 OOM, the batch-48 measurement, and a metric that was never a peak | analysis handoff §5 | ❓ secondary-only |
| "1-node is 29.8 % better than 128-node" | not converged-vs-converged. The defensible claim is cost: 11.3× more samples per node-hour | analysis handoff §5; CHANGELOG 2026-09-04 | ❓ secondary-only |
| "the late ensemble members are redundant" | unfounded: similar loss does not imply correlated errors | analysis handoff §5 | ❓ secondary-only |
| "makani has never been profiled" | true only of kernel level (first nsys 7591822: 34.9 % of compute is copy/layout) | analysis handoff §5; CHANGELOG 2026-09-04 | ❓ secondary-only |
| "production walls out ~3 epochs short" | measurement error: 685 s/epoch + 350 s startup ⇒ 46.33 h of 48 | CHANGELOG 2026-09-03 | ❓ secondary-only |
| makani 4-node scaling projections "84 %" and "92.9 %" | retracted; built on a mismeasured 1-node step and a falsified batch-invariant toll | CHANGELOG 2026-09-19 | ❓ secondary-only |
| "A takes 1.52× FCN3 stage-1 steps" | 1.60× (the paper's 208,320 steps govern) | CHANGELOG 2026-09-29 | ❓ secondary-only |
| "the 43× worst channel in the K=56 read-out is unnamed / a soil channel" | it is `Z3_l17` | CHANGELOG 2026-09-20 / 09-30 | ❓ secondary-only |
| "`Z3` leaves range first / near-surface Z3 drives the ~500-step blow-up" | `Z3_l17` is ~0.015 σ at day 36 (benign); blow-ups start at the model top | CHANGELOG 2026-09-24; 7649647 CSV | 🔁 ✅ CSV / ❓ reading |
| "the slow (loss-blind) channels are land reservoirs" (H3) | they are the Z3 levels and PS | CHANGELOG 2026-09-23 (7646192) | ❓ secondary-only |
| "removable systematic drift is possibly the largest single win" | only 2 of 101 channels carry material bias² share at 336 h (`Z3_l17`, `Z3_l16`) | CHANGELOG 2026-09-23 | ❓ secondary-only |
| "`Z3_l17`'s bias is steep-terrain spectral ringing" (A5) | the bias sits on the ocean (−5.41 m) not the land (+0.85 m) | CHANGELOG 2026-09-23 (7646252) | ❓ secondary-only |
| "`temp_diff_normalization: True` is a one-line flip" | PRECT's weight would be 8.3e-4 and `Z3_l17`'s 9518; it needs a cap and a PRECT decision | CHANGELOG 2026-09-23 | ❓ secondary-only |
| "`submit_nfuture_ladder.sh`: the entire first epoch was a ramp" | epoch 1 runs at a constant 4e-6 | `S1A_OK` 7649946 | 🔁 ✅ |
| "with `-v SEED=1..4`, `seed:` is the knob for independent replicas" | retracted twice on 2026-09-10 (setting `seed` changes the data order) | CHANGELOG 2026-09-10 | ❓ secondary-only |

---

## 10. Dangling citations and contradictions found

A required deliverable (handoff §3.2). Each item says what was checked. **Nothing listed here
has been fixed.** Fixing is out of scope for a documentation task (handoff §4). CHANGELOG is
append-only, so corrections go in a new dated entry.

1. **`polaris_makani_b_dryair_findings_handoff.md` §3** cites
   `ACE2_retrain/ace_exp/fme/fme/core/corrector/atmosphere.py`. That path does not exist. The file
   is at `ACE2_retrain/ace_exp/fme/core/corrector/atmosphere.py` (one `fme/` fewer), in the main
   checkout only. `ACE2_retrain/ace_exp` is not in the `b-continuation-dryair` worktree's tree.
2. **`polaris_makani_analysis_ensemble_handoff.md` numbers its own traps inconsistently:** §0a
   calls `SKIP_TRAIN` "the tenth silent-failure trap", while §6 lists it as #9 of 10.
3. **`polaris_pbs_notes.md` §1b #2 vs the live queue:** "capacity cannot hold a queued
   successor… the earlier 'max_queued 2 per PROJECT' reading was wrong". On 2026-10-06 PBS reports
   `max_queued = [p:PBS_GENERIC=2]` for `capacity`, and 7718436 was queued behind 7715005. Either
   the limit changed or the original refusal was another member's job holding the second slot.
   `❓` which one.
4. **Line drift in trap citations** (claims still hold): `driver.py:237-248` → the
   `makani_restart.yaml` read is at L275; `train_plasim.py:382` → the fixed call is at L433.
5. **CHANGELOG's `## Known issues / failed approaches` section has no makani entries.** The
   section the handoff calls "the most direct what-doesn't-work source" covers PanguWeather, data
   prep, S2S and SI. makani's failed approaches live only in Decisions-log entries and handoffs.
6. **CHANGELOG 2026-10-03 (F3 result)** says "F and B share the same depth-4 (`n_future=4`) fine-tune
   recipe". **False** (§9a). The 2026-10-04 operator-decision entry and the soil-fix arm's motivation
   build on it, and so do project memory notes (`makani-ace2-model-selection-open`,
   `makani-b-lineage-dryair-handoff`).
7. **CHANGELOG 2026-10-02** cites `docs/2026-10-02_b_continuation_dryair_handoff.md` (§2, §2.3, §3).
   That file never existed in any branch or in history (`git log --all` is empty). Two citations
   remain.
8. **CHANGELOG 2026-10-02** calls the 16→14 post-hoc dry-air result "the earlier post-hoc F result".
   It was B/C1 (§9a).
9. **CHANGELOG 2026-10-04 Step-2 table and TL;DR** carry DRYAIR's drift in Pa labelled hPa. They are
   corrected by the 2026-10-06 entry and the findings-handoff banner. `TODO.md` (B-continuation block)
   still says DRYAIR24 "converged back to B's own drift magnitude by year 5" (critic). That is stale.
10. **Findings handoff §1d** says "Not verified whether B's own schedule matches the fine-tune
    recipe". It does not match (§2).
11. **Findings handoff §3** says "steps 1 and 3 anchor water". Step 3 does not modify the rolled water
    state (§7).
12. **Findings handoff §4 item 4** says "nobody has checked that the correctors compose correctly on
    the same step". They were checked at unit level: 18 tests including composition, job 7713397. The
    PBS-level flag-off gate is still missing.
13. **Decision job 7707317 (`$MEMBER_ROOT/runs/decision/makani_vs_ace2/7707317/decision.md`)** treats
    7704379 as an unread verdict. It never ran (§9a).
14. **`polaris_makani_ace2_ports_handoff.md` §3 Port B rationale** says "our config already sets
    `target: 'tendency'` — the network emits a change". That key is dead (§9a). Per critic, the port-B
    argument itself survives.
15. **Audit §6 / §8 item 8 and Q12** ("fed-back dtype open"): now measured. Plain arms feed back bf16
    (`feedback_dtype=torch.bfloat16`), and DRYAIR arms feed back fp32 through `mass_fix`'s promotion.
    (critic, member logs of 7707597 / 7711720)
16. **`polaris_makani_f_finetune_handoff.md`** is the record of the 2026-09-28 re-base of Stage-1 onto
    F. It exists **only** on branch `worktree-monitor-ace2`. The inventory's first pass missed it
    because that branch name has no "makani".
17. **The memory note on B_e01** reads "only 7/8, one survivor at −269 hPa". CHANGELOG 2026-10-02 means
    7/8 *non-finite*, so 1/8 survive.

---

## 11. Adversarial review, 2026-10-06: open questions and next experiments

Two read-only critics ran on 2026-10-06 (`Workflow` run `wf_64bce84d-4fc`, Fable 5):
- **A, "5-year accuracy path"**: 10 findings.
- **B, "ACE2 ports: useful, needed, correct"**: 11 findings.

Their claims this session re-opened are folded into §0–§10 above. The ranked list below is
recommendations, not decisions. Queue and compute choices are the operator's; metric, loss and
variable decisions are jesswan's.

**Cheapest first. Inference-only, so no training is needed:**
1. **Accuracy-score what is already on disk** (debug, < 1 node-h). Run `score_climate_fidelity.py` on
   `EVAL/climate_protocol_7711720` DRYAIR24 members against the existing
   `climate_fidelity_7709968/true_climatology_b.npz`. Extend the scorer to monthly climatology and
   interannual std. That gives the first B22 / B24 / B22-clamp / DRYAIR24 accuracy table. Then
   **pre-register a 5-yr accuracy ranking** before the next `capacity` submission (metric
   definitions: jesswan). *(both critics)*
2. **Make the external bar exist.** Fix the ACE2-EAMv3 5-yr test's config mismatch (strip
   `log_nino34_index` / `log_zonal_mean_images`, or align the fme version) and re-run prep and rollout.
   The rollout was sized for `debug-scaling` at 4 nodes. *(this session)*
3. **Controls that need no retrain** (debug, ≈ 1–2 node-h total):
   - **B22/B24 with `--dry-air-fix on`** at inference. Post-hoc has only ever been tried on B_e01.
   - **True fp32 inference** (autocast off) on B22 and DRYAIR24. `--feedback-fp32` cannot test this.
   - **The 5-yr protocol on soil-fix e21 and e24**, plus a **checkpoint average** of B21–24 and of
     DRYAIR21–24, to address the measured checkpoint lottery. *(critics A, B)*

**Training arms, each ≈ 15 node-h at 2 nodes. Run after the free reads above:**
4. **A plain `anneal b01` control** (`CKPT=<B_e01> submit_finetune_stability_arm.sh anneal b01`). It
   separates annealing from dry-air for every existing and future arm. *(critic A)*
5. **dry-air + trained-through water positivity**, in fme's order (positivity on TMQ/RELHUM before
   `DryAirFix`, in the same preprocessor hook), on the matched anneal recipe. This should come before a
   soil-fix 5-yr promotion or a soil-fix + dry-air arm. It may need a channel-list confirmation from
   jesswan. *(critic B; critic A)*
6. **The tendency-unit loss normalizer**: build a full-split `time_diff_stds.npy` (debug), then run one
   capped-1/r fine-tune from B_e01. It is the audit's #1 mechanism and the only one aimed at the
   universal model-top 3σ crossing. It needs a PRECT decision. *(critic B)*
7. **The soil question, if it still matters:** fine-tune F (or F-scratch e23) at depth 4 on B's exact
   recipe, then screen. That is the matched test F3 was not. *(critic A)*

**Process:**
- Select by **rollout metric**, not single-step validation (ACE2 does). Re-screen saved epochs of each
  finished arm at ~2.5 yr with 2 members (debug).
- **Before G/H takes `capacity`:** pre-register its stability levers and a kill criterion. G drops TMQ,
  so it cannot use the only proven corrector, and it is single-step until a depth-4 stage is budgeted.
- `capacity` now holds 2 per project (§8 #8), so a successor can be chained there.

**Open questions:**
- Is the `V_l00` model-top 3σ crossing a real dynamics failure, or tiny anomaly-σ in E3SM's damped
  sponge layer? Science read needed before optimizing against it.
- Where does DRYAIR24's −1.4 K TREFHT bias come from: the constraint, the annealed schedule, or the
  missing water guard? Experiments 3–5 triangulate it.
- Do 8 October starts bias climatology estimation? All members share one spin-up season.
- Does the 2026-10-02 sign-off cover a trained-through positivity channel list and a TMQ-retaining G
  variant, or are those new asks?
