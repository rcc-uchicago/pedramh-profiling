# Inventory — makani documentation in scope for `KNOWLEDGE.md` (2026-10-06)

Generated mechanically by a script (path, add-commit, first heading, line count, branch) from
branch `docs/makani-knowledge` @ `44a3ea9c` (= `feat/makani-b-continuation-dryair` HEAD).
The **status** column is a *guess* from the filename and heading. **Nothing in this file has been
claim-verified.** Verified content lives in [`../KNOWLEDGE.md`](../KNOWLEDGE.md). Handoff:
`polaris_makani_docs_consolidation_handoff.md` §3.1.

Counts: 121 docs on this branch (56 ours, 65 upstream-subtree), 3 only on other branches, 5 subdirectories (one line each), 50 makani-matched CHANGELOG Decisions-log entries.

## 1. Our docs: E3SM / Polaris era, added on this repo's history

| path | added | first heading | lines | status (guess) |
|---|---|---|---|---|
| `makani_sfno/docs/2026-08-27_prod128_alldata_checkpoint_usage.md` | 2026-08-27 `5a8031ac` | Using the makani-E3SM production checkpoint (`prod128_alldata_v2`) | 137 | reference |
| `makani_sfno/docs/2026-09-03_prod1n_b32_sgdr_checkpoint_usage.md` | 2026-09-03 `9f591e52` | Using the makani-E3SM 1-node production checkpoint (`prod1n_b32_sgdr`) | 248 | reference |
| `makani_sfno/docs/2026-09-10_ace2_comparison_the_corrector.md` | 2026-09-10 `aefa746a` | ACE2 does 7300-step stable rollouts at the SAME 2-step training depth. The difference is a | 135 | result — partly superseded by 2026-10-01 audit §9 (memory; unverified here) |
| `makani_sfno/docs/2026-09-10_c1_verdict.md` | 2026-09-10 `e04490f6` | C1 verdict — the rollout fine-tune worked. The old number measured the one lead where it i | 168 | result |
| `makani_sfno/docs/2026-09-10_e3sm_inference_port_scope.md` | 2026-09-10 `37e723ca` | E3SM inference port — why it is needed, and exactly what it takes | 231 | plan |
| `makani_sfno/docs/2026-09-10_fcn3_recipe_vs_ours.md` | 2026-09-10 `56b2a962` | What FourCastNet 3's recipe actually is, and what we ran instead | 247 | unknown |
| `makani_sfno/docs/2026-09-10_lagged_ensemble_design.md` | 2026-09-10 `37e723ca` | Lagged ensemble vs. many-IC evaluation — which one we are building, and why | 210 | plan |
| `makani_sfno/docs/2026-09-10_lagged_ensemble_endtoend_plan.md` | 2026-09-10 `8c5fd643` | Lagged ensemble — end-to-end plan: data loading, training, inference, uncertainty | 323 | plan |
| `makani_sfno/docs/2026-09-10_longroll_blowup_analysis.md` | 2026-09-10 `9bdc1b99` | Why the 500-step rollout blows up — what the error curve says | 174 | result |
| `makani_sfno/docs/2026-09-10_nfuture_ladder_prereg.md` | 2026-09-10 `fd2fdcfe` | Pre-registration — does `n_future = 3/4` actually buy stability? An adversarial design | 252 | plan (prereg) |
| `makani_sfno/docs/2026-09-10_nfuture_plan_in_plain_english.md` | 2026-09-10 `6f3a1b36` | The `n_future` experiment, in plain English | 191 | plan |
| `makani_sfno/docs/2026-09-10_rollout_spectra_and_drift.md` | 2026-09-10 `4cafc9d9` | Measured: the rollout BLURS and DRIFTS — it does not accumulate small-scale energy | 140 | result |
| `makani_sfno/docs/2026-09-11_nfuture_ladder_result.md` | 2026-09-11 `78a17d80` | `n_future = 4` for ONE epoch matches `n_future = 1` for twenty-four | 115 | result |
| `makani_sfno/docs/2026-09-11_proxy_failure_diagnosis.md` | 2026-09-11 `10612b86` | Why the 1-epoch proxy failed to reproduce C1 — hypotheses and the discriminating runs | 128 | result |
| `makani_sfno/docs/2026-09-15_nfuture_ladder_d1_and_replication.md` | 2026-09-15 `fd24e92f` | The two open caveats are closed: `n_f=4` replicates, and batch 8 is not innocent | 139 | result |
| `makani_sfno/docs/2026-09-20_k56_readout_prereg.md` | 2026-09-21 `9ccddb9e` | PRE-REGISTRATION — reading the K=56 (14-day) curve | 222 | plan (prereg) |
| `makani_sfno/docs/2026-09-21_lagged_ensemble_result.md` | 2026-09-21 `a5b83ae2` | The lagged ensemble, finished — and it does not improve the deterministic forecast | 135 | result |
| `makani_sfno/docs/2026-09-23_architect_review_ace2_ports.md` | 2026-09-23 `11628839` | Architect review — the ACE2 ports plan (`polaris_makani_ace2_ports_handoff.md`) | 149 | plan |
| `makani_sfno/docs/2026-09-24_climate_driver_g2_prereg.md` | 2026-09-24 `fe89b902` | G2 pre-registration — streaming climate driver vs `rollout_one_ic` | 50 | plan (prereg) |
| `makani_sfno/docs/2026-09-24_climate_screen_7649647.csv` | 2026-09-24 `15cdf9af` | (csv) rank,label,ckpt_epoch,epoch_check,survived,truncated_at_step,truncated_channel,s | 39 | result (data) |
| `makani_sfno/docs/2026-09-24_climate_screen_7649792_f1156.csv` | 2026-09-24 `356dd2ea` | (csv) rank,label,ckpt_epoch,epoch_check,survived,truncated_at_step,truncated_channel,s | 17 | result (data) |
| `makani_sfno/docs/2026-09-24_climate_screen_7650652_dryair_on.csv` | 2026-09-24 `14043a95` | (csv) rank,label,ckpt_epoch,epoch_check,survived,truncated_at_step,truncated_channel,s | 17 | result (data) |
| `makani_sfno/docs/2026-09-24_climate_screen_prereg.md` | 2026-09-24 `7f6ecfb5` | Pre-registration — Stage-0 stability screen (2026-09-24) | 248 | plan (prereg) |
| `makani_sfno/docs/2026-09-29_f2_equiv_prereg.md` | 2026-09-29 `e036ede5` | F2 pre-registration — full-width paths unchanged on the merged tree | 52 | plan (prereg) |
| `makani_sfno/docs/2026-09-30_fsurg_nf4_proxy_prereg.md` | 2026-09-30 `04ff5b7f` | Pre-registration — does a 1-epoch depth-4 fine-tune rescue the soil-free model? | 46 | plan (prereg) |
| `makani_sfno/docs/2026-09-30_g_finetune_debate_brief.md` | 2026-09-30 `9adfc33a` | Debate brief — what the 2026-09-29 debug jobs mean for port G, for fine-tuning, and what t | 115 | plan |
| `makani_sfno/docs/2026-09-30_g_finetune_next_runs_analysis.md` | 2026-09-30 `e617895c` | Analysis — what the 2026-09-29 debug jobs mean for G and for fine-tuning, and what to run  | 185 | result |
| `makani_sfno/docs/2026-09-30_surgical_soil_screen_prereg.md` | 2026-09-30 `4855b999` | Pre-registration — does removing soil change A's ~500-lead blow-up? (surgical checkpoint s | 44 | plan (prereg) |
| `makani_sfno/docs/2026-10-01_makani_vs_ace2_code_audit.md` | 2026-10-01 `6d2357e0` | Audit: ACE2 (ai2 fme) vs makani SFNO climate emulators, code-level synthesis | 494 | result (canonical makani-vs-ACE2 audit) |
| `makani_sfno/docs/2026-10-03_jesswan_climate_rollout_instructions.md` | 2026-10-03 `c3133bc1` | Running the 5-year climate rollout | 19 | reference |
| `MONITOR_makani_streaming_driver.md` | 2026-09-24 `81758ce8` | Monitor prompt — makani streaming multi-year rollout driver | 451 | plan |
| `ace2_vs_makani_differences.md` | 2026-09-23 `50350a96` | ACE2 vs makani — what actually differs | 208 | result — partly superseded by 2026-10-01 audit §9 (memory; unverified here) |
| `makani_bench_report.md` | 2026-09-01 `9cb36d50` | makani-SFNO on Polaris — benchmark results | 1221 | result |
| `makani_port/HANDOFF_monitor.md` | 2026-10-01 `ed3d75c6` | HANDOFF (monitor) — audit the makani `main` port | 63 | result |
| `makani_port/HANDOFF_worker.md` | 2026-10-01 `ed3d75c6` | HANDOFF (worker) — port our makani fork to upstream makani `main`, keeping every checkpoin | 136 | plan |
| `makani_port/PROGRESS.md` | 2026-10-01 `ed3d75c6` | makani port — PROGRESS (newest first) | 347 | plan |
| `makani_port/_papercuts.md` | 2026-10-01 `ed3d75c6` | _papercuts — mistakes that cost time; read at session start and before every `qsub` | 50 | plan |
| `makani_port/api_delta.md` | 2026-10-01 `2ed54f1a` | makani API delta — pin `c9704308` → main `a0aa4c4f` | 152 | plan |
| `makani_port/equivalence_tolerance.md` | 2026-10-01 `072880dd` | Equivalence tolerance for the makani port: M4 (inference) and M5 (training) | 80 | plan |
| `makani_port/golden_checkpoints.json` | 2026-10-01 `347c6431` | (csv) { | 11 | unknown |
| `makani_port/golden_checkpoints_ext.json` | 2026-10-01 `2b1c9030` | (csv) { | 11 | unknown |
| `makani_port/grid_declaration.md` | 2026-10-01 `072880dd` | Grid declaration: what makani main requires, and why the port waives one check | 60 | plan |
| `makani_port/m0_golden_prereg.md` | 2026-10-01 `347c6431` | M0 pre-registration — golden baselines on the OLD venv | 84 | plan (prereg) |
| `makani_port/m5_train_prereg.md` | 2026-10-01 `215b4ec4` | M5 pre-registration — training equivalence (Part 0: slot-3 pre-tests) | 123 | plan (prereg) |
| `makani_port/makani_symbols.txt` | 2026-10-01 `6a30605a` | (csv) makani | 29 | unknown |
| `polaris_makani_128node_decision_prompt.md` | 2026-09-17 `5a7212c6` | DECISION PROMPT — is the 128-node makani production run worth re-running? | 509 | plan |
| `polaris_makani_1node_production_handoff.md` | 2026-09-01 `0fa21928` | makani-SFNO on Polaris — 1-node production handoff | 303 | plan |
| `polaris_makani_accuracy_handoff.md` | 2026-09-21 `9ccddb9e` | HANDOFF — the makani accuracy track, from 2026-09-18 | 199 | plan |
| `polaris_makani_ace2_ports_handoff.md` | 2026-09-23 `50350a96` | makani handoff — port the ACE2 mechanisms, cheapest and least-gated first | 534 | plan — partly superseded by 2026-10-01 audit §9 (memory; unverified here) |
| `polaris_makani_analysis_ensemble_handoff.md` | 2026-09-04 `3aaaf9e3` | makani handoff — finish the analysis, then build the ensemble | 326 | result |
| `polaris_makani_b_dryair_findings_handoff.md` | 2026-10-06 `44a3ea9c` | makani B: dry-air findings and the ACE2 corrector gap (2026-10-05) | 202 | result — TL;DR 2–3 retracted by its own 2026-10-06 banner (Pa/hPa) |
| `polaris_makani_climate_protocol_handoff.md` | 2026-09-24 `881af9aa` | HANDOFF — makani E3SM emulator: the ACE2-style multi-year climate evaluation | 329 | plan |
| `polaris_makani_docs_consolidation_handoff.md` | 2026-10-04 `1b3a5202` | HANDOFF — consolidate makani_sfno documentation (fixing doc drift) | 166 | plan — this task |
| `polaris_makani_finetune_stability_handoff.md` | 2026-09-24 `5f04e080` | HANDOFF — makani E3SM emulator: fine-tuning for long-rollout stability | 219 | plan |
| `polaris_makani_g_spatial_handoff.md` | 2026-09-29 `4c692d52` | HANDOFF — makani: port G, the 2020–2044 split, the F fine-tune base, and spatial paralleli | 151 | plan |
| `polaris_makani_streaming_driver_handoff.md` | 2026-09-24 `81758ce8` | HANDOFF — makani E3SM emulator: the streaming multi-year rollout driver | 226 | plan |

## 2. Docs that exist only on other branches

Found with `git ls-tree` over the 26 makani/sfno-named branches (local + origin); read with
`git show <branch>:<path>`. Nothing was checked out or merged.

| path | branch(es) | first heading | lines |
|---|---|---|---|
| `makani_multinode_ddp_plan.md` | origin/worktree-makani-multinode-ddp-profiling, worktree-makani-multinode-ddp-profiling | Makani multi-node DDP scaling — plan and prereg | 273 |
| `makani_sfno/docs/2026-09-29_spatial_cxi_prereg.md` | feat/makani-spatial-cxi, origin/feat/makani-spatial-cxi | Pre-registration — spatial parallelism on the fixed CXI stack, phase 1 (2026-09-29) | 74 |
| `makani_sfno/docs/2026-09-29_spatial_cxi_result.md` | feat/makani-spatial-cxi, origin/feat/makani-spatial-cxi | Result — spatial parallelism on the fixed CXI stack, phase 1 (job 7669001) | 52 |

| `polaris_makani_f_finetune_handoff.md` | worktree-monitor-ace2, origin/worktree-monitor-ace2 | HANDOFF — makani: re-base the stability fine-tune on Port F (the soil-free model) | 150 |

The first pass filtered branches by `makani|sfno` in the name and missed the last row. A second pass
over the other 39 branches found only that one file.

`makani_multinode_ddp_plan.md` has an add-commit in this branch's history but is not in its tree
(deleted or moved here); the copy above is from the profiling branch.

## 3. Subdirectories (one line each, not claim-audited in this pass)

- `makani_sfno/docs/audit_snapshots/`: 6 files
- `makani_sfno/docs/codex_reviews/`: 61 files
- `makani_sfno/docs/hpo_distill/`: 16 files
- `makani_sfno/docs/run_log/`: 3 files
- `makani_port/golden/`: 4 files

## 4. Upstream docs inherited through the subtree

These came in with the `makani_sfno/` subtree import `27980894` (2026-07-14) and predate our E3SM work.
They cover the upstream PlaSim / AI-RES / Stampede3 / Derecho / DSI project. They are out of scope for
E3SM claims, and are listed so nobody mistakes them for results on our runs.

| path | first heading | lines |
|---|---|---|
| `makani_sfno/README.md` | SFNO Climate Emulator | 30 |
| `makani_sfno/docs/2026-05-02_ema_implementation_plan.md` | EMA implementation plan for SFNO emulator training | 1003 |
| `makani_sfno/docs/2026-05-04_makani_local_patches.md` | Local patches against vendored Makani | 316 |
| `makani_sfno/docs/2026-05-04_makani_pipeline_efficiency_review.md` | PlaSim → SFNO training pipeline: efficiency review vs. NVIDIA Makani | 228 |
| `makani_sfno/docs/2026-05-04_phase1_efficiency_implementation_plan.md` | Phase-1 efficiency implementation plan (low/medium-risk items only) | 1088 |
| `makani_sfno/docs/2026-05-04_zg1000hpa_migration_plan.md` | zg pressure-level subset: drop 150 hPa, add 1000 hPa (v10.1) | 241 |
| `makani_sfno/docs/2026-05-05_ddp_throughput_fix_plan.md` | DDP throughput fix — implementation plan | 902 |
| `makani_sfno/docs/2026-05-06_group_sfno_5410_climatology_prompt_for_derecho.md` | Standalone prompt: build the SFNO-5410 climatology on Derecho/Glade | 201 |
| `makani_sfno/docs/2026-05-06_group_sfno_5410_eval_plan.md` | Group SFNO-5410 emulator evaluation plan **v6.2** — Stampede3 | 973 |
| `makani_sfno/docs/2026-05-08_ddp_throughput_fix_resolution.md` | DDP throughput fix — resolution | 120 |
| `makani_sfno/docs/2026-05-08_hpo_knob_inventory.md` | 2026-05-08 — HPO knob inventory (full reference) | 230 |
| `makani_sfno/docs/2026-05-08_hpo_skill_trim_plan.md` | 2026-05-08 — Trim plan for `skills/train-sfno-hpo/SKILL.md` | 76 |
| `makani_sfno/docs/2026-05-08_panguweather_local_patches.md` | PanguWeather/v2.0 local patches for SFNO-5410 user inference | 232 |
| `makani_sfno/docs/2026-05-08_sfno_5410_explicit_K_horizon_plan.md` | 5410 NWP eval — explicit K-step horizon (production-path fix) — v3 | 341 |
| `makani_sfno/docs/2026-05-08_sfno_5410_external_user_inference_plan.md` | Plan: SFNO-5410 external-user inference path on Stampede3 | 419 |
| `makani_sfno/docs/2026-05-08_sfno_5410_external_user_inference_plan_v2.md` | Plan v2: SFNO-5410 external-user inference path on Stampede3 | 555 |
| `makani_sfno/docs/2026-05-08_sfno_5410_external_user_inference_plan_v3.md` | Plan v3: SFNO-5410 external-user inference path on Stampede3 | 544 |
| `makani_sfno/docs/2026-05-08_sfno_5410_external_user_inference_plan_v4.md` | Plan v4: SFNO-5410 external-user inference path on Stampede3 | 460 |
| `makani_sfno/docs/2026-05-08_sfno_5410_external_user_inference_plan_v5.md` | Plan v5: SFNO-5410 external-user inference path on Stampede3 | 385 |
| `makani_sfno/docs/2026-05-08_sfno_5410_inproc_codex_round3_diff_summary.md` | 5410 NWP eval — in-process orchestrator — Codex round-3 diff summary | 339 |
| `makani_sfno/docs/2026-05-08_sfno_5410_inproc_orchestrator_plan.md` | 5410 NWP eval — in-process orchestrator (deep refactor) — v2.1 | 676 |
| `makani_sfno/docs/2026-05-08_sfno_5410_scoring_plan.md` | 5410 NWP scoring — minimal scorecard + full pipeline — v4.4 | 699 |
| `makani_sfno/docs/2026-05-09_group_code_training_track_plan.md` | Plan v5 — Group-code (PanguWeather v2.0 SFNO-v2) training track on Stampede3 | 533 |
| `makani_sfno/docs/2026-05-09_sfno_5410_byo_inference_user_guide.md` | SFNO-5410 inference from your own NetCDF — Stampede3 user guide | 496 |
| `makani_sfno/docs/2026-05-09_sfno_5410_investigation_history.md` | SFNO-5410 evaluation — investigation history (through 2026-05-09) | 280 |
| `makani_sfno/docs/2026-05-10_forcing_pipeline_numerical_diff.md` | Forcing-pipeline numerical comparison: upstream 5410 vs Makani own-track | 188 |
| `makani_sfno/docs/2026-05-10_phaseF_group_production_plan.md` | Phase F — Group-code SFNO sigma10 production training plan (v5) | 478 |
| `makani_sfno/docs/2026-05-10_sfno_param_count_forensic.md` | SFNO parameter-count forensic: GB4-prod vs GB8 group-clone vs SFNO-5410 | 315 |
| `makani_sfno/docs/2026-05-10_sst_sea_ice_handling_fix_plan.md` | Plan: SST sea-ice handling fix (PlaSim-faithful `surface` mode) — rev 5 | 352 |
| `makani_sfno/docs/2026-05-11_zgplev_gbhpo_40yr_lr_sweep_plan.md` | GB16/GB32 vs GB8: Minimal 40-yr LR Probe (No Anchor) | 432 |
| `makani_sfno/docs/2026-05-12_v11_clip_restore_plan.md` | v11 + grad-clip-restored training plan (v11_clip) | 151 |
| `makani_sfno/docs/2026-05-13_v11_clip_next_hpo_plan.md` | 2026-05-13 — Next-HPO plan after v11_clip lands | 278 |
| `makani_sfno/docs/2026-05-14_pr_6h_units_mismatch_ticket.md` | 2026-05-14 — Ticket: `pr_6h` units mismatch between our v11 stats and SFNO-5410's stats | 161 |
| `makani_sfno/docs/2026-05-14_tas_no_ice_metric_plan.md` | tas_no_ice: ice-free near-surface air-temperature metric | 425 |
| `makani_sfno/docs/2026-05-14_v10_warmstart_full_training_plan.md` | v10 group-clone warm-start full-training plan | 459 |
| `makani_sfno/docs/2026-05-14_v11_clip_warmstart_continuation_plan.md` | 2026-05-14 — v11_clip warm-start continuation diagnostic | 480 |
| `makani_sfno/docs/2026-05-14_v11_rollout2_gb16_y40_pilot_plan.md` | 2026-05-14 — v11 rollout2 GB16 40y Screening Pilot Plan | 296 |
| `makani_sfno/docs/2026-05-14_zg500_targeted_hpo_plan.md` | 2026-05-14 — zg500-Targeted HPO Plan (post Pre-Run 0) | 431 |
| `makani_sfno/docs/2026-05-16_work_quota_cleanup_plan.md` | /work Quota Cleanup Plan — 2026-05-16 | 95 |
| `makani_sfno/docs/2026-05-20_bundled_training_eval_plan.md` | Bundled training + eval in a single SLURM job | 557 |
| `makani_sfno/docs/2026-05-21_v11_noise_epochs_probes_plan.md` | v11_gb32_lr8e4_minlr1e5 — Two single-knob HPO probes (noise σ=0.07 + epochs=75) | 311 |
| `makani_sfno/docs/2026-05-23_hpo_prune_plan.md` | HPO prune plan — distill-then-delete (own-track only) | 330 |
| `makani_sfno/docs/2026-05-23_pr6h_unit_alignment_plan.md` | pr_6h Cross-Track Unit Alignment Plan | 618 |
| `makani_sfno/docs/2026-05-23_sfno_climate_emulator_migration_plan.md` | Migrate AI-RES → SFNO_Climate_Emulator | 790 |
| `makani_sfno/docs/2026-06-02_eval_inference_latitude_flip.md` | Latitude-flip in emulator inference output (`eval_inference.py`) | 207 |
| `makani_sfno/docs/2026-06-02_eval_inference_latitude_flip_fix_plan.md` | Implementation plan — fix latitude-coordinate flip in `eval_inference.py` | 234 |
| `makani_sfno/docs/INDEX.md` | AI-RES `docs/` index | 48 |
| `makani_sfno/docs/aires_rad_profile_plan.md` | Add `aires_rad` profile (radiation + heat fluxes) — Plan (v3) | 366 |
| `makani_sfno/docs/codex_review_plan_skill_for_derecho.md` | Install `codex-review-plan` skill on Derecho | 215 |
| `makani_sfno/docs/dsi_full_training_plan.md` | DSI full SFNO training plan (v4) — phase 2 | 1197 |
| `makani_sfno/docs/dsi_smoke_backup_plan.md` | DSI smoke-only backup plan (v3) — phase 1 prerequisite | 715 |
| `makani_sfno/docs/emulator_adaptor_audit.md` | Emulator Adaptor Audit (2026-04-21) | 190 |
| `makani_sfno/docs/plasim_expansion_and_adaptor_plan.md` | PlaSim Postprocessor Expansion + Emulator Adaptor — Implementation Plan | 471 |
| `makani_sfno/docs/plasim_makani_packager_plan.md` | PlaSim → Makani packager — implementation plan | 1208 |
| `makani_sfno/docs/plasim_postprocessor_audit.md` | PlaSim Post-processor Audit | 221 |
| `makani_sfno/docs/plasim_postprocessor_expand_plan.md` | PlaSim Post-processor — Variable Expansion + `--profile` Removal (Plan) | 418 |
| `makani_sfno/docs/plasim_postprocessor_refactor_plan.md` | PlaSim Post-processor Refactor — Plan (v4) | 374 |
| `makani_sfno/docs/plasim_zg_plev_migration_plan.md` | PlaSim → Makani packager — zg sigma → zg_plev migration plan (v7) | 789 |
| `makani_sfno/docs/sfno_data_preflight.md` | SFNO Data Preflight — Phase 0 Gate Artifact | 120 |
| `makani_sfno/docs/sfno_eval_plan.md` | SFNO PlaSim emulator accuracy evaluation plan **v2.8** — implementation-ready (locked) | 934 |
| `makani_sfno/docs/sfno_full_training_plan.md` | SFNO PLASIM full-emulator training plan **v1.1** | 455 |
| `makani_sfno/docs/sfno_tiny_short_training_plan.md` | SFNO tiny + short training plan | 494 |
| `makani_sfno/docs/sfno_training_extraction_plan.md` | SFNO Training Subproject — Extraction & Refactor Plan | 15 |
| `makani_sfno/docs/sfno_training_implementation_plan.md` | SFNO Training + Validation Subproject (Makani wrapper) — implementation plan | 464 |
| `makani_sfno/docs/weather_emulator_io_postprocessor_check.md` | Weather Emulator IO and PlaSim Postprocessor Coverage | 164 |

## 5. CHANGELOG.md makani entries

CHANGELOG.md is 8135 lines. The Decisions log is newest-first above `## Known issues / failed
approaches (do NOT re-attempt)` at **line 7977**. These are entries whose bold header matches
`makani|sfno|ace2` (case-insensitive), not just the `(makani)` tag. Line numbers are as of `44a3ea9c`.

| line | date | header |
|---|---|---|
| 145 | 2026-10-06 | (makani) — correction: DRYAIR24's 5-yr PS drift is −0.21 hPa, not −21 hPa (Pa read as hPa). Soil-fix 7715005 d |
| 185 | 2026-10-05 | (makani) — F from scratch queued: job 7718436, `capacity`, 2 nodes (+1 spare), 10 h, 23 epochs. |
| 197 | 2026-10-05 | (makani) — dry-air findings review: DRYAIR24's 5-yr survival is not attributable to dry-air, and our fix is on |
| 209 | 2026-10-05 | (makani) — session checkpoint: handed off to TODO.md P0. |
| 217 | 2026-10-05 | (makani) — `anneal_soilfix` moved to 24 epochs on `capacity`: job 7714685 (25 epochs, preemptable) cancelled b |
| 239 | 2026-10-04 | (makani) — `anneal_soilfix` launched: job 7714685, 25 epochs, `preemptable`, after a clean debug smoke. |
| 269 | 2026-10-04 | (makani) — operator decision: continue the B lineage as main; prototype an ACE2- inspired frozen-soil-moisture |
| 327 | 2026-10-04 | (makani) — docs consolidation handoff revised after a 2-seed Opus review; the review found the handoff itself  |
| 355 | 2026-10-04 | (makani) — docs consolidation handoff written: `polaris_makani_docs_consolidation_handoff.md`. |
| 366 | 2026-10-04 | (makani) — Step 2 climate comparison: `anneal_dryair` (B-continuation, trained-with dry-air conservation) surv |
| 426 | 2026-10-03 | (makani) — F3 result: D7 = "neither; F not shippable as screened" — none of F's 8 checkpoints survive 1 year,  |
| 459 | 2026-10-03 | (makani) — F3 launched: jobs 7709775 (2044 f1092) + 7709776 (f1156), both starts per §A3.2. |
| 471 | 2026-10-03 | (makani) — F3's missing prereg written: D7 (raw vs EMA) decision rule, drafted by a 3-process debate, committe |
| 496 | 2026-10-03 | (makani) — climate-fidelity scoring: 2-reviewer adversarial pass found real bugs, all fixed, 12/12 tests pass. |
| 522 | 2026-10-03 | (makani) — F4 (`lrcheck`, job 7709269) PASSED: `MAKANI_MN_SCALING_OK`, 3 epochs, `rc=0`, real per-channel vali |
| 527 | 2026-10-03 | (makani) — climate-fidelity scoring built for jesswan: true 5-year climatology vs rollout. |
| 546 | 2026-10-03 | (makani) — Step 2 launched: B-specific `anneal_dryair` (trained-with dry-air conservation) on B_e01, plus F4 ( |
| 582 | 2026-10-02 | (makani) — B-continuation handoff Step 1: B_e22 and B_e24 both run the full 5-year protocol clean (8/8 finite) |
| 634 | 2026-10-01 | (makani/ACE2) — code-level audit of makani vs ACE2 landed; ACE2 5-year rollout test submitted. |
| 653 | 2026-10-01 | (makani) — our E3SM SFNO predicts the FULL next state, not a tendency; the checkpoint-usage docs said otherwis |
| 664 | 2026-10-01 | (makani port) — operator rulings (relayed by the port monitor): |
| 671 | 2026-10-01 | (makani port) — M1 GREEN: `sfno-venv-main` built beside `sfno-venv`, old venv unchanged |
| 682 | 2026-10-01 | (makani port) — M0 GREEN: golden baselines on the old venv are bitwise reproducible |
| 691 | 2026-10-01 | (makani port) — operator ruling (relayed by the port monitor) on api_delta §3.2: |
| 697 | 2026-10-01 | (makani) — port to upstream makani `main` planned: worker + monitor handoffs in `makani_port/`. |
| 710 | 2026-09-30 | (makani) — surgical soil-free checkpoint screen: ✅ `SURGICAL_SOIL_SCREEN_OK`, outcome D (slicing damage) at bo |
| 726 | 2026-09-30 | (makani) — surgical soil-free checkpoint screen: pre-registered (submission note). |
| 737 | 2026-09-30 | (makani) — CORRECTION, operator: the 2020–2044 split is a SEPARATE experiment, not part of G. |
| 765 | 2026-09-30 | (makani, analysis only) — moderated three-agent review of G, fine-tuning and the next runs: run the multi-year |
| 788 | 2026-09-29 | (makani) — F fine-tune base built; PORT G (the ACE2-EAMv3 variable set) and the 2020–2044 train split implemen |
| 1045 | 2026-09-24 | (makani, cont.) — Stage-1 arms queued, one at a time on `preemptable` (operator-approved). |
| 1063 | 2026-09-24 | (makani, cont.) — dry-air mass fix, inference only: conserves mass exactly, and by the pre-registered rule it  |
| 1119 | 2026-09-24 | (makani, cont.) — negativity probe: E3SM truth is never negative in any moisture channel, but every rollout is |
| 1149 | 2026-09-24 | (makani, cont.) — Stage 1 started: S1a green (`SCHED_TMAX=22`), T-anneal queued, depth-8 fits, depth-16 OOMs. |
| 1189 | 2026-09-24 | (makani, cont.) — second-start re-screen: B epoch 22 is the Stage-0 winner, and PS drift is a fixed property o |
| 1230 | 2026-09-24 | (makani) — Stage-0 stability screen: depth 4 helps survival, more epochs help, and mass loss is the rule. Oper |
| 1315 | 2026-09-24 | (makani) — jesswan's multi-year protocol, inference only: neither A nor B yields an admissible 5-year climate. |
| 1338 | 2026-09-24 | (makani) — streaming multi-year climate driver BUILT; G1–G3 green; first stability read of checkpoints A and B |
| 1397 | 2026-09-24 | (makani, analysis only) — `Z3_l17`'s "NRMSE 152" is a denominator artefact over a real, linear drift; the chan |
| 1420 | 2026-09-24 | (makani, cont.) — jesswan's multi-year climate protocol, assessed; probe SUBMITTED. |
| 1476 | 2026-09-24 | (makani, docs only) |
| 1493 | 2026-09-23 | (makani/ACE2 ports — `polaris_makani_ace2_ports_handoff.md` implemented) |
| 1563 | 2026-09-23 | (makani/ACE2, analysis only — no jobs, no allocation) |
| 1780 | 2026-09-21 | (makani, cont.) |
| 1839 | 2026-09-21 | (makani) |
| 1890 | 2026-09-20 | (makani) |
| 1959 | 2026-09-18 | (ACE2) |
| 2031 | 2026-09-19 | (ACE2) |
| 2072 | 2026-09-19 | (makani) |
| 2124 | 2026-09-18 | (ACE2, cont.) |
