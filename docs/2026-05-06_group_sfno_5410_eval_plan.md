# Group SFNO-5410 emulator evaluation plan **v6.2** — Stampede3

> **Author:** Zhixing Liu (with Claude Code), drafted 2026-05-06, revised 2026-05-07 (×1), 2026-05-08 (×6 → v6.2 below).
> **v6.2 changes (2026-05-08, post-approval wording cleanup):**
> - **§K step 4 deliverable column** previously said "finite RMSE/ACC for `model="sfno_5410"`, `model="persistence"`, and `model="climatology"`". Persistence and climatology baselines are **RMSE-only** (the scorer at `score_nwp.py:155-190` does not emit ACC for these row classes — only the emulator-label rows get ACC), and persistence `pr_6h` is intentionally NaN per §C.2 / §D.7. Reworded to enumerate finiteness per row class and call out the NaN exception. Aligns with the §D.7 sanity-gate semantics introduced in v6.
>
> **v6.1 changes (2026-05-08, sixth reviewer audit — pre-approval cleanup):**
> - **§B.2 bash: `RUN_ROOT` is set + exported before the Python pre-resolution heredoc.** v6 read `os.environ["RUN_ROOT"]` in the heredoc but assigned `RUN_ROOT=...` only afterwards. Reordered, added explicit `export RUN_ROOT`. The post-cd assignment is removed.
> - **§H test row for `test_ic_nc_compatibility.py` updated** to match v6's §3 P-7: per-`(Y, s)` sanity-load over the full 96-tuple run plan; H5 cross-check coverage spelled out by source (3 spot-tuples for per-year sources `plev_data` / `sigma_data_transferred`; all 96 for per-IC source `ic_nc_built_from_h5`). v6 had updated §3 P-7 itself but left the §H row pinned to the v5 "Jan 1 Y 00:00 + (121, 0) only" wording.
> - **§F section 4 bias-map grid:** "paired `(our, 5410)`" → "side-by-side `(our, 5410)`" with an explicit note that the two panels are independent estimates over each track's own IC schedule, not paired per IC. Aligns with the §D.6/§F-section-2 unpaired-distributional framing introduced in v6.
>
> **v6 changes (2026-05-08, fifth reviewer audit):**
> - **Cross-report baseline assumption corrected (BLOCKER fix, §D.6 + §F).** v5 wrongly claimed persistence/climatology rows from either track could be reused because both tracks "should agree to floating-point tolerance." That is **not true**: the two tracks differ in IC anchor (Aug-1/Y+5 vs Jan-1/Y), IC stride (116 vs 122), `init_state` source (Makani `MOST.<year>.h5` vs group `<year>_<s>.h5`), truth-NetCDF source, and climatology source. v6 drops the cross-track persistence-equality assertion and the "use whichever track scored first" approach. §F section 2 now shows **per-track baseline columns** (`persistence_ours`, `persistence_5410`, optionally `climatology_ours`, `climatology_5410`) and labels the comparison as **unpaired distributional** ("difference of means over each track's own 96 ICs, not a paired per-IC delta").
> - **§3 P-7 IC-compatibility gate now per `(Y, s)` (§3 P-7).** v5's gate ran `get_data_given_path_nc` only at `Jan 1 Y 00:00` (8 calls) plus one h5 cross-check at `(121, 0)` — fine for sources `plev_data` / `sigma_data_transferred` (same per-year file 12 times) but **inadequate for contingency C-B** where each `(Y, s)` is a distinct file. v6 iterates the actual run plan: full per-`(Y, s)` sanity-load; h5 cross-check at 3 spot tuples for sources A/C-A or all 96 for C-B.
> - **PYTHONPATH explicit + IC paths pre-resolved before `cd` (§B.2).** v5's bash snippet did `cd /work2/.../v2.0/` then imported `sfno_inference_5410.ic_source` from CWD — which would fail. v6 sets `PYTHONPATH=$AI_RES/src:${PYTHONPATH}` explicitly and pre-resolves all 96 IC paths in a single Python invocation before `cd`, indexing into the precomputed list inside the shell loop.
> - **§B.2 "we pass the per-year truth NetCDF" cleanup.** That phrasing was specific to source `plev_data`; v6 enumerates all three resolved-source possibilities in the explanatory bullet.
> - **§D.7 sanity gate finiteness clause** now distinguishes by `model` row class: emulator + climatology rows finite for all 53 channels; persistence rows finite for the 52 state channels only, with `pr_6h` persistence intentionally NaN per §C.2 (and the gate explicitly does not flag that NaN).
>
> **v5 changes (2026-05-08, fourth reviewer audit):**
> - **IC source threaded through `resolve_ic_nc_path` (§B.2, §H).** No more hardcoded `plev_data/<Y>_gaussian.nc` in the command path or test asserts. New `src/sfno_inference_5410/ic_source.py` reads `<run_root>/inference/ic_source.json` (written once by §3 P-7 gate when it picks a contingency) and dispatches to one of `plev_data` / `sigma_data_transferred` / `ic_nc_built_from_h5`. `test_runtime_args_5410.py` validates the resolved path, not always the plev path.
> - **Compute/storage estimate in §B.0 reconciled with §G.1.** "~64 k forward steps" → **75,840** (matches §G.1 derivation). "~240 GB" → **~125 GB** (matches §G.1). Numbers now identical across the two sections.
> - **`pr_6h` allow-list assertion in `channel_map.py` removed (§B.6).** v4 dropped the strict `units in {"kg m-2", "mm"}` check from §C.1 but the inline code sketch in §B.6 still showed it. Now removed; comment instead documents that `units` is recorded verbatim and gating is on magnitude/§1a semantics.
> - **§4 source layout updated.** Per-Y configs (`SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y<Y>.yaml` × 8) replace the single config; per-Y single-file `ckpt_epoch_50.tar` symlink shims replace the v3 directory-symlink sketch; `ic_source.json`, optional `ic_nc/` (contingency C-B), `upstream_raw/`, plus the new `ic_source.py`, `build_ic_nc_from_h5.py`, `test_ic_nc_compatibility.py`, `test_lead_slice_offset.py`, `test_runtime_args_5410.py` are all listed.
> - **§D.6 clarified for multi-row CSV.** Each track's `nwp_scorecard.csv` now contains 3 distinct `model` labels (`{emulator|sfno_5410}`, `persistence`, `climatology`); `render_cross_emulator_report.py` filters by `model` (`EMULATOR_LABELS`/`BASELINE_LABELS` constants) rather than assuming one label per file. Adds a cross-track persistence-equality sanity check.
>
> **v4 changes (2026-05-08, third reviewer audit):**
> - **IC-NetCDF compatibility now a hard pre-flight gate (§3 P-7).** Reviewer surfaced that `--init_nc_filepaths` is consumed by `get_data_given_path_nc` (`utils/data_loader_multifiles.py:83-159`), which requires per-variable `levels_per_var` matching (sigma for `ta/ua/va/hus`, plev for `zg`). On Stampede3, only `plev_data/<Y>_gaussian.nc` is present (verified 2026-05-08); `sigma_data/<Y>_gaussian.nc` files are **not** transferred (only `climatology.nc` is in `sigma_data/`). The plan no longer assumes any path works — `test_ic_nc_compatibility.py` runs the actual upstream loader, and on failure forces one of two documented contingencies: **(C-A) transfer `sigma_data/<Y>_gaussian.nc` from Derecho**, or **(C-B) build per-IC NetCDFs from h5 via `scripts/build_ic_nc_from_h5.py`**. The §A.3 stale text claiming "consumes per-timestep h5s" is corrected.
> - **Off-by-one slicing now empirically resolved (§B.5).** `test_lead_slice_offset.py` runs a smoke rollout and compares `pred[time=0]` vs `pred[time=1]` against the de-z-scored IC; pins `LEAD_SLICE_OFFSET ∈ {0, 1}` as a constant in `output_adapter.py`. Adapter slices `pred[OFFSET : OFFSET + 60]`, not a hardcoded range. Adapter step 6 adds a lead-1-vs-truth magnitude cross-check (`0.1 K < tas_diff < 10 K`, `1 gpm < zg500_diff < 100 gpm`) to catch both off-by-one and silent unit-conversion regressions.
> - **Stale §G.1 / §H lines updated.** Forward-step total: 5760 → **75,840** (verified `6 × Σ(1460-122i) + 2 × Σ(1464-122i) = 6×9468 + 2×9516`). Score command and `test_score_nwp_cli.py` invocation now include `--clim-coord-name auto` and `--write-climatology-row` (added in v3 D.0 but missing from these consumer references).
> - **`pr_6h` units check relaxed (§C.1).** Strict `units in {"kg m-2", "mm"}` allow-list dropped; replaced with magnitude/range gate plus documented §1a semantics. Reviewer noted metadata spelling varies harmlessly across writer versions.
> - §K step 2 expanded to explicitly run §3 P-7 + the lead-slice resolution before SLURM submit; budget now ranges 0.5-1.0 d to cover the contingency-transfer or build path if `plev_data/` fails.
>
> **v3 changes (2026-05-08, second reviewer audit):**
> - **Rollout-window mechanics corrected (§B.0/§B.2):** `long_inference.py` always rolls full-remainder-of-year regardless of `--final_datetime` (verified `long_inference.py:823-836`). Plan no longer claims short K=60 windows; it runs full-remainder-of-year and slices the first 60 leads in the adapter (§B.5). `--final_datetime = Jan 1 (Y+1) 00:00` to drive `long_rollout_years = 1`. Wallclock estimate revised from "~96 × short windows" to "~64 k forward steps total ≈ 1-1.5 h".
> - **Required CLI flags added (§B.2):** `--init_nc_filepaths` (required by `long_inference.py:1227`; opens with `xr.open_dataset` and `get_loc(init_datetime)` — we pass `<Y>_gaussian.nc`), `--output_dir`, `--save_basename` (forecast saving uses these, NOT `exp_dir`; verified `long_inference.py:1189-1212`).
> - **Per-IC-year `val_year_start` (§B.0/§B.1):** boundary loader at `data_loader_multifiles.py:948-960` falls back to `leap_year=12 / no_leap_year=11` template years whenever `start_date.year != val_year_start`. Build 8 per-Y override yamls instead of one; new `test_runtime_args_5410.py` covers it.
> - **Climatology coord-name reconciliation (§C.1, §D.0):** Derecho prompt specifies `time_of_year`; existing `score_nwp.py` reads `doy`. New `--clim-coord-name auto` flag handles both; pre-flight blocks if neither is present.
> - **Single-file symlink shim (§3 P-2):** `ckpt_epoch_50.tar` symlink, not directory symlink. Pre-flight asserts `os.path.realpath` equals the absolute upstream path.
> - **Climatology baseline row (§D.0, §F):** new `--write-climatology-row` flag adds `model="climatology"` rows so the cross-report's climatology column is actually populated (was missing in `score_nwp.py:155-190`).
> - **Relaxed `pr_6h ≥ 0` for predictions (§C.3):** strict only for truth/clim; predictions get `[-2, 50]` gross bound (per reviewer: small negatives from numerics are not unit-bugs).
> - **Cross-report erratum re companion plan (§F):** explicit note that `render_cross_emulator_report.py` ignores any our-emulator climate outputs even if present, so the deferred-climate-for-both-tracks symmetry holds without touching `docs/sfno_eval_plan.md` v2.8.
> - Total Phase-1 effort estimate unchanged at ~3.5 d (the new requirements are inside the same step budget).
>
> **v2 changes (2026-05-07, first reviewer audit):** scope tightened to **NWP forecast skill only** (climate-mode + `score_climate.py` deferred to Phase 2 for both tracks); inference engine corrected to **`long_inference.py` only** (`inference.py` doesn't support `sfno_plasim`); calendar/anchor convention corrected to group's Jan-1 / 1460-or-1464 timesteps (was incorrectly inheriting our emulator's Aug-1 / Y+5 / 1455-or-1459); existing `scripts/score_nwp.py` **extended** with `--leads`, `--model-label`, `--ic-file-regex` so both tracks share one scorer (no fork); `pl in Pa` typo in §C.3 removed.
> **Companion plan:** `docs/sfno_eval_plan.md` v2.8 — defines the metric stack (`src/sfno_eval/`), Gauss–Legendre lat-weights, time-of-year-proleptic climatology schema `(366, 4, 53, 64, 128)`, NWP scorecard CSV layout, and sanity-gate thresholds. **This plan reuses that stack verbatim** — only the inference engine, the input data tree, and the output tree differ.
> **Companion prompt:** `docs/2026-05-06_group_sfno_5410_climatology_prompt_for_derecho.md` — produces the `climatology_proleptic_5410.nc` artifact this plan consumes.

This plan covers evaluation of the **group's SFNO-5410 emulator** (PanguWeather/v2.0, run 5410). It is a **separate evaluation track** from `docs/sfno_eval_plan.md`, which covers our own AI-RES emulator. The two tracks share a metric stack but have disjoint inputs, checkpoints, inference engines, and output trees, so the plans are kept side-by-side rather than merged.

---

## 0. Phasing & scope discipline

**Phase 1 (this plan):** **NWP forecast skill only** (RMSE / ACC / forecast-lead-time comparison) for SFNO-5410, scored under our `src/sfno_eval/` metric convention so the numbers are directly comparable to our emulator's. Climate-mode diagnostics are explicitly **out of Phase 1**.

**Out of scope (locked):**
- **Climate-mode scoring (`score_climate.py`, time-mean bias / variance / drift / zonal-mean) for *both* tracks.** Deferred to a separate **Phase 2** plan once the NWP report lands. Rationale (user, 2026-05-07): scientifically, Phase 1 only needs weather forecast skill; `score_climate.py` does not yet exist for either track, so deferring it for both keeps the cross-emulator comparison apples-to-apples and avoids a hidden prerequisite.
- Group's native scoring stack (`utils/metrics.py`, `plot_validation_error.py`, `compare_statistics.ipynb`, etc.). Not used as a primary or secondary scorer; final report uses our metrics only.
- Ensemble eval (deterministic-only).
- ERA5 / external truth.
- Reproducing any group-published numbers for 5410.
- SHT-based spectra (deferred to Phase 2 across both tracks).

**Cross-emulator report** (joining this track and our emulator's track) is in §F. Phase 1 cross-report is **NWP-scorecard-only**; no climate panels.

---

## 1. Context & current state

**Group emulator:** `/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0/results/SFNO/5410/`

```
checkpoints/ckpt_epoch_50.tar      # only checkpoint we will use; no best_ckpt or ckpt_latest exists
hyperparams.yaml                   # snapshot of training-time params
out.log                            # contains training git hash and run date 2026-01-13
plots/                             # group's training-time validation plots (informational only)
validation_data/                   # group's saved validation outputs (not consumed by this plan)
```

**Source config:** `/work2/.../v2.0/config/SFNO_PLASIM_H5_DERECHO_5410.yaml`. Still references `/glade/derecho/...` paths — needs a Stampede3 path override before any inference job.

**Channel slate (53 channels, identical *order* to our emulator's v10.1 contract; *unit conventions are group-pipeline-specific* — see §1a):**
```
0       pl                                     log surface pressure (dimensionless): pl = ln(p_s)
                                               NOT pressure in Pa
1       tas                                    near-surface air temperature (K)
2..11   ta1..ta10                              T (K) on sigma levels [0.0383, 0.1191, 0.2109, 0.3169, 0.4368,
                                                                       0.5668, 0.6994, 0.8234, 0.9241, 0.9833]
12..21  ua1..ua10                              u (m/s) on the same 10 sigma levels
22..31  va1..va10                              v (m/s) on the same 10 sigma levels
32..41  hus1..hus10                            specific humidity (kg/kg) on the same 10 sigma levels
42..51  zg200, zg250, zg300, zg400, zg500,     geopotential HEIGHT (gpm = geopotential metres),
        zg600, zg700, zg850, zg925, zg1000     NOT geopotential in m² s⁻². On pressure levels
                                                  [20000, 25000, 30000, 40000, 50000,
                                                   60000, 70000, 85000, 92500, 100000] Pa
52      pr_6h                                  snapshot precip rate × 6 h (mm or kg m⁻², group writer-dependent),
                                               NOT a true ∫₀⁶ rate dt accumulator. Diagnostic, output-only.
```

**Forcing slate (6, supplied by truth at every step):** `lsm, sg, z0, sst, rsdt, sic` — same channel set as our emulator, same order assumption (verify in §H).

**Test split:** years **121-128** (8 years, parity with our emulator's plan §A). Year 129 is available on disk but excluded for parity.

**Source data (transferred from Derecho/Glade, all on Stampede3 `$SCRATCH`):**

| Asset | Path | Notes |
|---|---|---|
| Per-timestep h5 inputs | `/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/h5/sigma_data/` | 13148 files; pattern `<year>_<timestep>.h5`, years 121-129. Verified per-year counts (2026-05-07): years **121,122,123,125,126,127,129 = 1460** (non-leap, Jan 1-Dec 31, 6-h cadence); years **124, 128 = 1464** (PlaSim leap years). Group convention is calendar-year-anchored (Jan 1 00:00 = sample idx 0), **not** the Aug-1 / Y+5 anchor used by our emulator. |
| Per-year aggregated truth | `/scratch/.../sim52/plev_data/<year>_gaussian.nc` | 9 files (121-129); 64×128 grid; 13 plev for `zg`. **Used as ground truth for §D scoring**, after channel-name normalization (§B.5). |
| Group bias files | `/scratch/.../sim52/bias/` | 635 `.npy` files (per-variable, 4 z-time strata + global). Consumed by upstream inference path; **not** used by our scorer. |
| Group stats | `/scratch/.../sim52/h5/sigma_data/data_12-132_{mean,std}_sigma.nc` | Used by upstream inference for z-score / de-z-score. |
| Group's existing climatology | `/scratch/.../sim52/sigma_data/climatology.nc` | Loaded by upstream inference internally. **Not** used by our scorer; we use the proleptic clim built per the Derecho prompt. |
| **Our scoring climatology** (per Derecho prompt) | `/scratch/.../sim52/baselines/climatology_proleptic_5410.nc` | Schema `(366, 4, 53, 64, 128)`, built from group post-processing of years 12-111. **This is what `src/sfno_eval/` reads for ACC.** Pre-flight checks in §H must verify presence + schema before scoring runs. |

**Upstream code:** `/work2/.../v2.0/` contains the full PanguWeather/v2.0 source tree, including `inference.py` (NWP-style, 616 lines), `long_inference.py` (climate-mode, 1447 lines), `train.py`, `networks/`, `utils/{metrics,weighted_acc_rmse,power_spectrum}.py`, and `enviornment.yml`. We run inference from this tree directly (no vendoring into the AI-RES repo) — see §B.

---

## 1a. Variable conventions — **preserve verbatim, do NOT convert** (Derecho update 2026-05-06)

These are group-post-processing conventions, not standard CF/ERA5 conventions. They are confirmed by the Derecho-side climatology agent and are common to **inference outputs, ground-truth NetCDFs (`<year>_gaussian.nc`), the group climatology, and the new `climatology_proleptic_5410.nc`**. Every component of this evaluation pipeline (output_adapter, scorer, sanity-gate thresholds, report text) must preserve them.

| Channel(s) | Convention | What NOT to do |
|---|---|---|
| `pl` | Dimensionless `ln(p_s)`. Typical climatological mean ≈ 11.3-11.6 (i.e., `ln(~10⁵ Pa)`). | Do **not** exponentiate to recover surface pressure in Pa. |
| `zg200..zg1000` | Geopotential **height** in `gpm` (geopotential metres). Typical climatological zg500 ≈ 5400-5900 gpm. | Do **not** multiply by g₀ ≈ 9.80665 to "recover" m² s⁻². |
| `pr_6h` | `instantaneous_pr_rate(t) × 6h`. A **6-hour proxy**, not a true ∫₀⁶ rate dt accumulator. | Do **not** re-derive from a higher-cadence rate field; do **not** describe in plots/report as "6-hour accumulated precipitation". Label as "6-hour precip proxy". |

**Implementation hooks:**

- `src/sfno_inference_5410/channel_map.py` carries an explicit `CHANNEL_UNITS` table alongside `CHANNEL_NAMES`. The output adapter (§B.5) sets per-variable `units` attributes on every NetCDF it writes from this table — no inference, no conversion.
- `tests/sfno_inference_5410/test_channel_map.py` asserts `CHANNEL_UNITS["pl"] in {"1", ""}`, `CHANNEL_UNITS["zg500"] == "gpm"`, `CHANNEL_UNITS["tas"] == "K"`, etc.
- The scorer (`src/sfno_eval/metrics.py`) is unit-agnostic — RMSE/ACC formulas only require `pred` and `truth` to share the same units, which they do because both come from the same group post-processing pipeline.
- §C.1 climatology pre-flight asserts climatological-mean magnitudes match the table above (catches an upstream g₀-multiplication regression).
- §F report header includes a `Variable conventions` block reproducing this table verbatim, so anyone reading the scorecard knows that `zg500 RMSE = 12 gpm` is geopotential height error, not m² s⁻² error.

If any inference, adapter, scorer, or report code path silently converts `pl`, `zg*`, or `pr_6h`, the cross-emulator comparison numbers will be wrong. Treat unit conversions in this track as a regression class on par with channel-order regressions.

---

## 2. Locked decisions

| Decision | Choice | Source |
|---|---|---|
| Doc layout | New dated plan file (this file) | Interview Q1, 2026-05-06 |
| Inference engine | Upstream PanguWeather/v2.0 **`long_inference.py`** (`inference.py` only handles `pangu_plasim` — see §B.0), with a Stampede3-path-overridden yaml | Interview Q2 + reviewer audit 2026-05-07 |
| Scoring stack | Our `src/sfno_eval/` only — Gauss–Legendre lat-weights, proleptic climatology binning. Existing `scripts/score_nwp.py` **extended** with `--leads`, `--model-label`, and a configurable `ic_file` regex so both tracks share one scorer (no `score_nwp_5410.py` fork) | User direction 2026-05-07 |
| Test years | 121-128 (parity with our emulator). **Per-IC-year yaml override**: each year `Y` gets its own yaml with `val_year_start=Y, val_year_end=Y+1` (forced by the boundary-loader contract at `data_loader_multifiles.py:948-960`, see §B.0). Year 129 is on disk but is not used as a val year (no Y=129 yaml is built); it can only be touched if a year-128 IC's full-remainder rollout crosses Jan 1 of year 129, which our offsets `s ∈ {0..1342}` avoid by construction. | Interview Q4 + reviewer audits 2026-05-07 + 2026-05-08 |
| Climatology source | Built fresh on Derecho/Glade from group post-processing of years 12-111, transferred to Stampede3 | Climatology prompt |
| Climatology destination | `/scratch/.../sim52/baselines/climatology_proleptic_5410.nc` | This conversation |
| NWP cadence | 12 ICs/year × 8 years = 96 ICs, **monthly stride 122 timesteps (≈30.5 days), K=60** (15 days, captures upstream's longest forecast lead). Offsets: `[0, 122, 244, …, 1342]` = `[122*i for i in range(12)]`. Same offsets used for leap (1464) and non-leap (1460) years; `1342 + 60 = 1402` fits inside both. | Reviewer audit 2026-05-07 (replaces interview-Q5 stride 116 derived from our-emulator 1455/1459 file counts, which do not apply here) |
| Climate-mode | **DEFERRED to Phase 2** (separate plan). Phase 1 = NWP forecast skill only. | User direction 2026-05-07 |
| Compute | Stampede3 `h100`, single GPU | Same as our emulator |
| Output tree | `results/sfno_eval_5410/<run_tag>/` (sibling to `results/sfno_eval/<run_tag>/`) | "Treat eval assets separately" |
| Checkpoint | `ckpt_epoch_50.tar`, discovered by `long_inference.py`'s `ckpt_epoch_*.tar` globstr via a **symlink shim** (§B.0) — no upstream patch | Reviewer audit + user direction 2026-05-07 |

---

## 3. Preconditions

**P-1 (✅ verified 2026-05-07).** Per-timestep h5 inputs and per-year truth NetCDFs are present on Stampede3 at the paths in §1. **Per-year h5 file counts (verified):** years `121,122,123,125,126,127,129 = 1460` (non-leap, Jan 1 → Dec 31 at 6-h cadence); years `124, 128 = 1464` (PlaSim Gregorian leap). Total `7×1460 + 2×1464 = 13148` h5s. Truth NetCDFs: 9 (`121_gaussian.nc`..`129_gaussian.nc`). The group's calendar-year (Jan 1) anchor differs from our emulator's Aug-1 / Y+5 anchor — IC-offset and file-count assumptions in `src/sfno_inference/` do **not** carry over and are re-derived in §A.

**P-2 (✅ verified 2026-05-06; checkpoint-discovery resolved 2026-05-07).** Checkpoint `ckpt_epoch_50.tar`, config `SFNO_PLASIM_H5_DERECHO_5410.yaml`, and supporting stats/bias files all present at the paths in §1. The 49 other `ckpt_epoch_<n>.tar` files are also present.

`long_inference.py:1393-1416` discovers checkpoints by globbing `<exp_dir>/<config>/<run_num>/checkpoints/ckpt_epoch_*.tar` and `natsorted`-picking the highest. We need that globstr to land on `ckpt_epoch_50.tar` *without* overloading `exp_dir` to point at the upstream `results/` tree (we want our own `exp_dir` so checkpoint discovery doesn't depend on the writer-target location). Resolution: **a single-file symlink shim** assembled by `scripts/build_5410_yaml_override.py` before any inference job runs (one shim per Y, since the override yaml filename — and thus `<config>` — is per-Y):

```
<our_exp_dir = $RESULTS/sfno_eval_5410/<run_tag>/inference>
└── SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y<Y>/      # one per Y ∈ 121..128
    └── 5410/
        └── checkpoints/
            └── ckpt_epoch_50.tar              # SINGLE-FILE symlink (not directory)
                -> /work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0/results/SFNO/5410/checkpoints/ckpt_epoch_50.tar
```

A single-file symlink (rather than a directory symlink to the whole upstream `checkpoints/`) means the globstr can match **only** `ckpt_epoch_50.tar`, eliminating any chance of `natsorted` picking some other epoch if the upstream tree later gains files. `test_yaml_override.py` asserts:

```python
target = "/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0/results/SFNO/5410/checkpoints/ckpt_epoch_50.tar"
shim   = f"<our_exp_dir>/SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y{Y}/5410/checkpoints/ckpt_epoch_50.tar"

assert os.path.islink(shim)
assert os.path.realpath(shim) == target          # absolute-path equality
assert natsorted(glob(os.path.join(os.path.dirname(shim), "ckpt_epoch_*.tar")))[-1] == shim
```

With this layout, `long_inference.py:1416` sets `params_i['best_checkpoint_path']` to the symlink (which `torch.load` follows transparently). Predictions written by `save_prediction` (long_inference.py:1189-1212) land under `<our_exp_dir>/inference/upstream_raw/` (set via `--output_dir`, see §B.2), **not** in the upstream tree.

**P-3 (✅ delivered 2026-05-06).** Derecho-side build complete per `docs/2026-05-06_group_sfno_5410_climatology_prompt_for_derecho.md`. Source: per-year `sigma_data/<year>_gaussian.nc` files (Derecho-confirmed bit-equivalent to per-timestep h5 for sampled channels). Variable conventions per §1a preserved (no unit conversions on `pl`/`zg*`/`pr_6h`). Build window years **12-111**, schema `(366, 4, 53, 64, 128)`. All 8 §7 validations PASS.

**Pinned manifest values:**

| Field | Value |
|---|---|
| Stampede3 path | `/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/baselines/climatology_proleptic_5410.nc` |
| Derecho source path | `/glade/derecho/scratch/zhil/AI-RES/climatology/sfno_5410_clim_yr12-111_proleptic_20260506.nc` |
| File size | `5,085,101,686 bytes` (5.09 GB) |
| `sha256` | `6b12a880637928eb537b6294399618df169426f794862b939ba62e41f2940876` |
| `years_used` | `12, 13, …, 111` (contiguous, 100 years) |
| `n_years_used` | `100` |
| `n_leap_years_used` | `24` (Feb-29 / day-59 populated 24/24) |
| `n_timesteps_aggregated` | `146,096` |
| `built_by_sha7` | `feec151b` (Derecho build script id) |
| Build job | PBS `3503674` (casper htc), exit 0, ~71.9 min wall, peak 24 GB |
| Transfer | Globus task `bede3215-49d0-11f1-b97b-0afffe4617ab` (NCAR GLADE → TACC Stampede3 Filesystems), SUCCEEDED, 2/2 subtasks, 0 faults, ~30 s elapsed @ 170 MB/s |
| Source variant | `yearly_netcdf_<year>_gaussian.nc` (sigma_data) |
| Manifests on Derecho | `*.manifest.md` (4.6 KB), `*.manifest.json` (15 KB), co-located with source `.nc` |

These values are baked into:
- `tests/sfno_eval/test_climatology_load_5410.py` — sha256 + `years_used` are hard-coded asserts (see §H).
- `scripts/submit_eval_score_5410.slurm` — pre-flight asserts the file exists at the Stampede3 path and matches sha256 before launching scoring.

Pre-flight in `test_climatology_load_5410.py` (§H) checks presence, sha256 against the pinned manifest, schema, channel names, and unit attributes (per §1a) before any scoring job runs. **The implementation steps in §K assume the climatology is in place; this is now satisfied.**

**P-4 (PENDING — Stampede3 env for upstream code).** PanguWeather/v2.0 lists deps in `enviornment.yml`. The Stampede3 evaluation needs a venv that satisfies those deps and is compatible with the trained `ckpt_epoch_50.tar` (built under torch 2.6.0+cu124 per `out.log:2`). We piggy-back on whatever venv the project's `requirements-stampede3.txt` already provides, plus a delta for any upstream-specific imports. The delta is identified at first smoke run (`Phase 0`, §K step 1).

**P-5 (residual sanity check, not blocking).** The 5410 yaml's `lat` array starts at `+87.864…` (north pole first). Our scorer's lat-weight build assumes the same north-first convention and Gaussian quadrature. Pre-flight in `test_eval_grid_5410.py` asserts the truth NetCDF's `lat` array matches the yaml's `lat` to 1e-6.

**P-6 (residual sanity check, not blocking).** All 8 truth NetCDFs (`121_gaussian.nc`..`128_gaussian.nc`) are expected to share an identical channel/coord schema. Pre-flight asserts the channel/level/lat/lon coords match `121_gaussian.nc` byte-for-byte across the 8 files.

**P-7 (BLOCKER — IC-NetCDF compatibility for `--init_nc_filepaths`).** Reviewer audit 2026-05-08 surfaced this: upstream's `get_data_given_path_nc` (`utils/data_loader_multifiles.py:83-159`) loads the IC by **looking up `init_datetime` in the file's `time` dim and selecting per-variable levels with a 1e-4 tolerance** against `levels_per_var` (sigma for `ta/ua/va/hus`, plev for `zg`). It is **not** automatically the case that any post-processed per-year NetCDF on Stampede3 satisfies this contract.

`tests/sfno_inference_5410/test_ic_nc_compatibility.py` (NEW, hard gate) — runs **before any inference SLURM job is submitted**. The gate iterates over **every `(Y, s)` tuple in the run plan** (8 years × 12 ICs = 96 tuples), not just one IC per year — required because under contingency C-B the IC files are **per-IC** (one NetCDF per `(Y, s)`), so a single-(Y,0) probe wouldn't exercise the other 88 files. For sources `plev_data` and `sigma_data_transferred` the same per-year file appears 12 times in the loop; that's fine — each call exercises a different `init_datetime` against `get_data_given_path_nc`'s `time`-lookup.

1. **Sanity-load (per `(Y, s)`).** For each `(Y, s)` in the run plan, call `get_data_given_path_nc(resolve_ic_nc_path(Y, s, run_root), ['ta','ua','va','hus','zg'], ['pl','tas'], init_datetime=cftime.DatetimeProlepticGregorian(Y, 1, 1, 0) + s × 6h, levels_per_var=...)` from upstream's actual code (import `utils.data_loader_multifiles`). Every call must return without raising; every level-match must succeed within tolerance; every `time`-lookup must find the requested `init_datetime` in the file's time index.
2. **Channel-by-channel agreement vs h5 (per `(Y, s)` for contingency C-B; one spot-check for sources A/C-A).** Load the IC tensor through the NetCDF path *and* through the h5 path (`get_data_given_path` on `<Y>_<s:04d>.h5`). Assert `max-abs-diff < 1e-4` across all 52 prognostic channels (pr_6h excluded; not in IC). Coverage:
   - **Source `plev_data` or `sigma_data_transferred`:** spot-check at `(121, 0)`, `(124, 0)` (leap year), and `(128, 1342)` (last IC of last test year). 3 calls. Same per-year NetCDF means inter-`s` variations are within-file timesteps, low novelty after the spot-checks; deferring full 96 saves test wall-time.
   - **Source `ic_nc_built_from_h5`:** all 96 (Y, s) tuples — each NetCDF is a distinct file produced by `build_ic_nc_from_h5.py`, so each must be exercised independently. Total ~96 × ~10 ms ≈ ~1 s test runtime; cheap.
3. **Tensor shape and ordering (per `(Y, s)`).** Asserts the NetCDF-loaded tensor has the same `(channel, lat, lon)` shape and ordering the SFNO model was trained on (`channel` order = `[pl, tas, ta1..10, ua1..10, va1..10, hus1..10, zg200..1000]` per §B.6 — 52 channels at the IC; pr_6h is not in the IC).
4. **Contingency cascade.** If step 1 fails on `plev_data/<Y>_gaussian.nc` (e.g., the file is plev-only and lacks sigma levels for ta/ua/va/hus), the gate trips and forces one of two documented contingencies — chosen by the user at smoke time, **not** silently:
   - **(C-A)** Transfer `sigma_data/<Y>_gaussian.nc` for `Y ∈ {121..128}` from Derecho/Glade to Stampede3 (mirror of the climatology transfer; ~4 GB × 8 ≈ 32 GB; one Globus task). Re-point `--init_nc_filepaths` at `sigma_data/`. Re-run the gate.
   - **(C-B)** Build per-IC single-timestep NetCDFs locally from the per-timestep h5s (`scripts/build_ic_nc_from_h5.py`, NEW). For each `(Y, s)`, read `<Y>_<s:04d>.h5`, write a 1-timestep NetCDF at `<run_root>/inference/ic_nc/<Y>_<s:04d>.nc` containing `ta(time=1, sigma_lev=10, lat, lon)`, `ua(...)`, ..., `zg(time=1, plev=10, lat, lon)`, `pl(time=1, lat, lon)`, `tas(time=1, lat, lon)`, with `time` coord = `cftime.DatetimeProlepticGregorian(Y, 1, 1, 0) + s × 6h` and a level coord matching the yaml's `sigma_levels`/`levels`. Total ~96 files × ~50 MB. Re-point `--init_nc_filepaths` at these. Re-run the gate.

The gate **must pass** on the chosen IC-NetCDF source before §K step 5 (full NWP run). §K step 2 explicitly includes resolving this; §G.1 SLURM pre-flight re-asserts the gate has been run.

---

## 4. Source layout (target after this plan ships)

```
src/sfno_inference_5410/                    # NEW — minimal Stampede3 adapter around upstream PanguWeather/v2.0
├── __init__.py
├── stampede3_yaml_override.py              # produces a Stampede3-pathed copy of SFNO_PLASIM_H5_DERECHO_5410.yaml
│                                              + assembles the checkpoint symlink shim (§3 P-2)
├── ic_offsets.py                           # 12 ICs/year × monthly stride 122, K=60 (§A.2)
├── ic_source.py                            # NEW — resolve_ic_nc_path(Y, s, run_root); reads <run_root>/inference/ic_source.json (§B.2, §3 P-7)
├── climatology_constants.py                # CLIMATOLOGY_PATH/EXPECTED_SHA256/YEARS_USED/... (§C.1)
├── output_adapter.py                       # reads upstream NetCDF outputs → scorer-format NetCDF
│                                              (writes to <run_root>/inference/nwp/ to match score_nwp.py:5;
│                                               carries pinned LEAD_SLICE_OFFSET constant per §B.5)
├── channel_map.py                          # canonical 53-channel name map + CHANNEL_UNITS table (§B.6)
└── README.md                               # explains long_inference.py-only path-overridden approach

scripts/
├── build_5410_yaml_override.py             # NEW — emit per-Y yaml (8 files) + per-Y single-file ckpt symlink shim
├── build_ic_nc_from_h5.py                  # NEW (contingency C-B, §3 P-7) — 1-timestep NetCDF per IC from <Y>_<s>.h5
├── eval_inference_5410.py                  # NEW — orchestrator: 96 single-IC long_inference.py invocations (uses resolve_ic_nc_path)
├── score_nwp.py                            # EXTENDED (existing file) — adds --leads, --model-label, --ic-file-regex,
│                                              #   --clim-coord-name, --write-climatology-row (§D.0)
├── render_eval_report_5410.py              # NEW — Phase-1 (NWP-only) 5410 report.md
├── render_cross_emulator_report.py         # NEW — combined NWP-only report
├── submit_eval_inference_5410.slurm        # NEW
├── submit_eval_score_5410.slurm            # NEW
├── submit_eval_report_5410.slurm           # NEW
└── submit_eval_5410.sh                     # NEW — chains the SLURM jobs with --dependency

tests/sfno_inference_5410/
├── test_yaml_override.py                   # incl. per-Y single-file symlink-shim resolution check
├── test_runtime_args_5410.py               # NEW — runtime CLI argv (incl. resolve_ic_nc_path)
├── test_ic_nc_compatibility.py             # NEW (§3 P-7 hard gate) — runs upstream get_data_given_path_nc + h5 cross-check
├── test_lead_slice_offset.py               # NEW (§B.5) — pins LEAD_SLICE_OFFSET ∈ {0, 1}
├── test_ic_offsets.py                      # asserts step=122 offsets [0, 122, ..., 1342]
├── test_output_adapter.py                  # incl. scorer-attr schema (ic_file, file_anchor, ...)
└── test_channel_map.py

tests/sfno_eval/
├── test_climatology_load_5410.py           # NEW — pre-flight on the transferred 5410 climatology (incl. doy/time_of_year resolution)
└── test_score_nwp_cli.py                   # NEW — defaults reproduce our-emulator fixture byte-for-byte;
                                              #       5410 flag set (incl. --clim-coord-name auto, --write-climatology-row) green

config/                                     # Stampede3-pathed yaml overrides live here (generated, gitignored)
├── SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y121.yaml   # one per test year Y ∈ {121..128}
├── SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y122.yaml
├── ...                                              # (val_year_start=Y, val_year_end=Y+1; rest identical)
└── SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y128.yaml

results/sfno_eval_5410/<run_tag>/           # → /work2/.../results/sfno_eval_5410/<run_tag>/ symlink
├── inference/
│   ├── ic_source.json                      # written by §3 P-7 gate; pins {"ic_source": "<plev_data | sigma_data_transferred | ic_nc_built_from_h5>", ...}
│   ├── ic_nc/                              # populated **only** under contingency C-B (build per-IC NCs from h5):
│   │   └── <Y>_<ssss>.nc                   #   96 single-timestep IC NetCDFs (~50 MB each, ~5 GB total)
│   ├── SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y<Y>/   # one tree per Y (matches per-Y --config)
│   │   └── 5410/
│   │       └── checkpoints/
│   │           └── ckpt_epoch_50.tar       # SINGLE-FILE symlink → /work2/.../v2.0/results/SFNO/5410/checkpoints/ckpt_epoch_50.tar
│   ├── upstream_raw/                       # 96 full-remainder-of-year upstream NetCDFs (~125 GB total)
│   │   └── Y<Y>_s<ssss>_member000_y<YYYY>.nc
│   └── nwp/<Y>_<ssss>.nc                   # 96 adapter-output NetCDFs (scorer-format, sliced to lead 1..60)
│                                              consumed directly by score_nwp.py:5 glob
├── scores/
│   ├── nwp_scorecard.csv                   # rows for model ∈ {sfno_5410, persistence, climatology}
│   ├── nwp_scorecard_summary.csv
│   └── bias_maps_<channel>_<lead>.npy
├── plots/
└── report.md
```

Climate-mode subtrees (`inference/climate/`, `scores/climate_stub/`) are **not** part of Phase 1 and will be added by the Phase 2 plan.

`results/sfno_eval_cross/<cross_run_tag>/report.md` lives in a sibling tree and consumes `results/sfno_eval/<our_run_tag>/` + `results/sfno_eval_5410/<5410_run_tag>/`.

---

## A. Test split & IC selection

### A.1 Test-year file inventory — **group convention (Jan-1 anchor)**.

For each year `Y ∈ {121..128}`:
- Per-timestep h5 list: `glob('/scratch/.../sim52/h5/sigma_data/{Y}_*.h5')`, sorted by timestep index. **Expected counts (verified 2026-05-07):** 1460 for `Y ∈ {121,122,123,125,126,127}` (non-leap) and 1464 for `Y ∈ {124, 128}` (PlaSim Gregorian leap, Feb 29 inserted). Year 129 = 1460 (non-leap), present on disk but not used as a val year — see §2.
- Per-year truth NetCDF: `/scratch/.../sim52/plev_data/{Y}_gaussian.nc`. Read once; provides ground-truth fields at every 6 h timestep on the same 64×128 Gaussian grid.
- **Anchor: Jan 1 of year `Y` 00:00:00**, `proleptic_gregorian` calendar (yaml: `calendar: 'proleptic_gregorian'`, `has_year_zero: True`). Sample idx `s` corresponds to `Jan 1 Y 00:00 + s × 6 h`. This **replaces** our emulator's Aug-1 / Y+5 anchor — that scheme was specific to the Makani-packaged data and does not apply here. Pre-flight in `test_eval_grid_5410.py` (§H) cross-checks: `time[0] == cftime.DatetimeProlepticGregorian(Y, 1, 1, 0, 0, 0, has_year_zero=True)` for all 8 truth files.

### A.2 NWP IC selection — **12 ICs/year, monthly stride 122, K=60**.

```python
def nwp_ic_offsets_5410(n_samples: int, K: int = 60, n_ic: int = 12,
                        step: int = 122) -> list[int]:
    """Return 12 IC sample indices on a monthly stride (Jan-1 anchor).

    With step=122 (~30.5 days × 4 timesteps/day) and n_ic=12, offsets are
    [0, 122, 244, ..., 1342]. Max IC + K = 1342 + 60 = 1402, which fits
    inside both 1460 (non-leap) and 1464 (leap) year files.

    The same offsets are used for every year so `nwp_scorecard.csv` rows
    are uniform across leap/non-leap. No floor() arithmetic per year —
    the stride is fixed by design, not derived from n_samples.
    """
    assert n_samples > 0 and step > 0 and n_ic > 0
    offsets = [step * i for i in range(n_ic)]
    for s in offsets:
        assert s + K < n_samples, (
            f"IC {s}+K={K} crosses file boundary at n_samples={n_samples}")
    return offsets
```

IC sample indices: `[0, 122, 244, 366, 488, 610, 732, 854, 976, 1098, 1220, 1342]`. Calendar-of-year semantics (proleptic_gregorian, Jan-1 anchor):

| IC idx | Sample | Approx date (non-leap) |
|---|---|---|
| 0 | 0 | Jan 1 00:00 |
| 1 | 122 | ~Feb 1 12:00 |
| 2 | 244 | ~Mar 4 |
| 3 | 366 | ~Apr 4 |
| 4 | 488 | ~May 5 |
| 5 | 610 | ~Jun 5 |
| 6 | 732 | ~Jul 6 |
| 7 | 854 | ~Aug 6 |
| 8 | 976 | ~Sep 6 |
| 9 | 1098 | ~Oct 7 |
| 10 | 1220 | ~Nov 7 |
| 11 | 1342 | ~Dec 8 |

Stride 122 is uniform on idx but **slightly drifts** in calendar-of-year between leap and non-leap years (Feb 29 in years 124/128 shifts post-Feb dates by 4 timesteps). For Phase-1 NWP scorecard this is irrelevant — we score IC-level deltas against truth at the same idx, not at the same calendar date — but the report header notes it for transparency.

### A.3 Per-IC source-file mapping.

For NWP mode at year `Y`, IC sample `s`, the upstream `long_inference.py` separates the **IC source** from the **forcing/boundary source**:

- **IC source — NetCDF**, passed via `--init_nc_filepaths` (required, `long_inference.py:1227`). The dataset class enters its `init_from_nc` branch, calling `_get_data_nc(index)` (`utils/data_loader_multifiles.py:905-924`) which invokes `get_data_given_path_nc` (line 83) on the NetCDF. The loader **looks up** `init_datetime` in the file's `time` dim (`get_data_given_path_nc:104-115`), then for each upper-air variable selects exactly the levels in `levels_per_var` (sigma for `ta/ua/va/hus` and plev for `zg`, with tolerance `1e-4`; line 144). **The IC NetCDF must therefore contain `ta/ua/va/hus` on the yaml's 10 sigma levels AND `zg` on the yaml's 10 plev values, both at `init_datetime` exactly.** This is **not** the per-timestep h5 path the our-emulator track uses.
- **Forcing/boundary source — per-timestep h5**, drawn via `_get_boundary_data` from `data_dir = /scratch/.../sim52/h5/sigma_data/`. The loader reads `<Y>_<idx>.h5` files (`get_out_path`, line 161-166) using leap_year/no_leap_year template-year mapping (line 932-934) when `start_date.year != val_year_start` — which is why the per-Y yaml override is required (§B.0).

**IC NetCDF availability on Stampede3 (verified 2026-05-08):**

| Path | What's there | Suitable for `--init_nc_filepaths`? |
|---|---|---|
| `/scratch/.../sim52/sigma_data/` | **only `climatology.nc`** — per-year `<Y>_gaussian.nc` files were used by Derecho during the climatology build but **were not transferred** to Stampede3 | No (files don't exist locally) |
| `/scratch/.../sim52/plev_data/<Y>_gaussian.nc` | 9 files for `Y ∈ {121..129}`, 64×128 grid, ~4 GB each | **TBD** — name suggests plev-only; SFNO loader needs sigma levels for `ta/ua/va/hus`, so this likely fails the level-match. Pre-flight (§3 P-7) decides. |
| `/scratch/.../sim52/h5/sigma_data/<Y>_<idx>.h5` | 13148 per-timestep h5s | Wrong format — h5 layout, not NetCDF. Could be **converted** to per-IC NetCDFs by us (contingency, §3 P-7). |

The chosen IC-NetCDF source is **decided by the pre-flight gate in §3 P-7**, not assumed.

### A.4 Climate-mode (deferred).

Climate-mode IC selection (1 IC/year × K=n_samples−1) is **deferred to Phase 2**. Phase 1 covers NWP only (§A.2).

---

## B. Inference — upstream PanguWeather/v2.0 `long_inference.py`, path-overridden

### B.0 Engine choice + rollout-window mechanics (Phase 1 = NWP only).

Verified against upstream code 2026-05-07 + 2026-05-08 reviewer audit:

- **`inference.py:106-132`** dispatches only on `params.nettype == 'pangu_plasim'` and raises `Exception("not implemented")` otherwise. SFNO is **not supported** by `inference.py`. It also hardcodes `'training_checkpoints/ckpt.tar'` (line 578) — wrong path for our `ckpt_epoch_50.tar` artifact. Async-save in `inference.py:251` uses `diagnostic_transform` (sync uses `diagnostic_inv_transform`, line 335) — would corrupt `pr_6h` units in async mode. **Not used by this plan.**
- **`long_inference.py:308`** has the `sfno_plasim` branch. Uses `ckpt_epoch_*.tar` globstr + `natsorted` (line 1396, 1406-1410) so it picks `ckpt_epoch_50.tar` automatically given the single-file symlink shim from §3 P-2. **Async-safe**: every diagnostic save in `long_inference.py` (lines 641, 668, 750, 915, 942, 1028) uses `diagnostic_inv_transform`. **This plan uses `long_inference.py` for the NWP rollouts.**

**Rollout-window mechanics — important upstream constraint (verified `long_inference.py:823-836`):**

`long_inference.py` rolls out from `init_datetime` to `next_year_offset_hours = init_datetime.hour % timedelta_hours; next_output_datetime = (current_year+1, 1, 1, hour=next_year_offset_hours)` — i.e. **always to Jan 1 of the next calendar year**, regardless of `final_datetime`. The output buffer is sized at `output_inference_steps = (next_output_datetime - init_datetime) // 6h`. `final_datetime` only governs how many year-blocks the outer loop runs (`long_rollout_years = final.year - init.year`, `long_inference.py:190`); it does **not** truncate within a year.

Implications for our K=60 (15-day) NWP windows:

- We **cannot** ask upstream for an exact 60-step partial save without patching it. We accept full-remainder-of-year rollouts and slice the first 60 leads in the output adapter (§B.5).
- `final_datetime.year` must be **strictly greater** than `init_datetime.year` for the outer loop to run at all (`long_rollout_years > 0`); we set `final_datetime = (init.year + 1, 1, 1, init.hour)` to make it run exactly one year-block.
- Compute estimate (verified): per-IC step count = `1460 - s` (non-leap) or `1464 - s` (leap), summed over `s ∈ {0, 122, 244, …, 1342}`. Per-year total = `Σᵢ₌₀..₁₁(1460 - 122i) = 9468` (non-leap) or `Σᵢ₌₀..₁₁(1464 - 122i) = 9516` (leap). Across all 8 test years: `6 × 9468 + 2 × 9516 = ` **75,840 forward steps**. At ~15 ms/step on H100 single-GPU → ~19 min compute; per-IC dataset+model init (~96 × 20 s = ~32 min) dominates. **Total wallclock ≈ 50-60 min**, fits 4-h SLURM with margin. (See §G.1 for the SLURM-row sizing — same number, expressed there as a job spec.)
- Many produced leads (>60) will be discarded by the adapter. Network egress and disk are the only cost; per-IC storage is one full-remainder-of-year NetCDF, ranging from ~2.5 GB (s=0, full year) to ~200 MB (s=1342, ~24 days remaining); average ~1.3 GB. Aggregate: **~125 GB across the 96 IC rollouts**, fits `$SCRATCH`. (Same number as §G.1.)

**Per-IC `val_year_start` requirement (verified `utils/data_loader_multifiles.py:948-960`):**

The boundary loader's single-IC branch maps `start_date.year` to `params.leap_year` (= 12) or `params.no_leap_year` (= 11) **whenever `start_date.year != val_year_start`**. With our test years 121-128, this means a single yaml with `val_year_start=121` would route year-122..128 ICs through nonexistent year-11/12 boundary files. **Resolution: per-IC `val_year_start = Y, val_year_end = Y+1`** — the orchestrator parametrizes the override yaml per IC year (cheap; same yaml template, only two int fields differ). Pre-flight in `test_yaml_override.py` asserts `start_date.year == val_year_start` for every (Y, s) tuple in the run plan.

### B.1 Yaml override (`scripts/build_5410_yaml_override.py`).

Reads the source yaml at `/work2/.../v2.0/config/SFNO_PLASIM_H5_DERECHO_5410.yaml` and writes **8 per-IC-year copies** to `config/SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y<Y>.yaml` (one per `Y ∈ {121..128}`). Also assembles the **single-file checkpoint symlink shim** described in §3 P-2 before any inference job submits. Per-Y is required by the boundary-loader constraint at `data_loader_multifiles.py:948-960` (see §B.0).

| Key | Source value | Stampede3 override |
|---|---|---|
| `data_dir` | `/glade/derecho/scratch/awikner/PLASIM/data/2100_year_sims_rerun/sim52/h5/sigma_data` | `/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/h5/sigma_data` |
| `bias_data_dir` | `/glade/derecho/.../sim52/bias` | `/scratch/.../sim52/bias` |
| `climatology_file` | `/glade/derecho/.../sim52/sigma_data/climatology.nc` | `/scratch/.../sim52/sigma_data/climatology.nc` (group's training-time clim — consumed by the upstream loader internally; **not** our scorer's clim, which lives at `/scratch/.../sim52/baselines/climatology_proleptic_5410.nc` per §C.1) |
| `load_exp_dir` | `/glade/work/marchakitus/PLASIM/PanguWeather/v2.0/results` | `/work2/.../PanguWeather/v2.0/results` (only used if upstream re-loads training state; with the symlink shim, redundant — kept for completeness) |
| `exp_dir` | `results` | `$RESULTS/sfno_eval_5410/<run_tag>/inference` — used by upstream **only** for the checkpoint-discovery globstr (via the §3 P-2 symlink shim). Forecast saving uses the explicit `--output_dir` / `--save_basename` CLI args (verified `long_inference.py:1189-1212`), **not** `exp_dir`. |
| `val_year_start` | `11` | **`<Y>` per IC year** (one of 121..128). **Critical** for boundary loader (§B.0): the single-IC branch at `data_loader_multifiles.py:948-960` falls back to the `leap_year=12 / no_leap_year=11` template years whenever `start_date.year != val_year_start`, so a one-size-fits-all `val_year_start=121` would route ICs in years 122-128 through nonexistent year-11/12 boundary files. |
| `val_year_end` | `12` | **`<Y> + 1`**. Exclusive in upstream loader (`long_inference.py:1320,1348`); covers exactly one year (`Y` itself) so `start_date.year == val_year_start` for every IC of that year. |
| `save_forecasts` | (absent → defaults False at `long_inference.py:170`) | `true` — **required** to actually write prediction NetCDFs to disk |
| `log_to_wandb` | `true` | `false` (also force-disabled at `long_inference.py:1427`, but documented here for clarity) |
| `wandb_*` (entity, project, group, name) | various | dropped from override yaml; with `log_to_wandb=false` they are unused, and removing them avoids accidental wandb init |
| `forecast_lead_times` | `[1, 12, 20, 40, 60]` | unchanged for upstream-internal logic; the rollout produces a full-remainder-of-year sequence regardless (§B.0), and the adapter slices the first 60 leads. |
| `num_inferences` | `128` | `1` per IC invocation (we drive 96 invocations from the orchestrator — see §B.2). |
| `batch_size` | `8` | `1` for single-IC inference |
| `max_epochs` | `50` | irrelevant; inference reads ckpt directly |

`test_yaml_override.py` asserts (per-Y): zero `/glade/...` substrings; loads via `YParams` cleanly; `val_year_start == Y == val_year_end - 1`; `save_forecasts == True`; `log_to_wandb == False`; all 8 generated yamls produce identical model architecture (only `val_year_start`/`val_year_end` differ). It also asserts the **single-file** symlink shim resolves: `os.path.realpath(<our_exp_dir>/.../checkpoints/ckpt_epoch_50.tar) == "/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0/results/SFNO/5410/checkpoints/ckpt_epoch_50.tar"` (absolute-path equality, not a directory listing).

### B.2 Run NWP via upstream `long_inference.py` — **96 single-IC invocations**.

**IC-source resolution.** The IC NetCDF path is **not hardcoded**; it is returned by `resolve_ic_nc_path(Y, s)`, defined in `src/sfno_inference_5410/ic_source.py` (NEW). The function reads a single source-of-truth file `<run_root>/inference/ic_source.json` (written by §3 P-7's gate when it picks a contingency) and dispatches accordingly:

```python
# src/sfno_inference_5410/ic_source.py  (sketch)
import json, os
from pathlib import Path

# IC_SOURCE ∈ {"plev_data", "sigma_data_transferred", "ic_nc_built_from_h5"}
# Determined by §3 P-7 gate; pinned in <run_root>/inference/ic_source.json.

def resolve_ic_nc_path(Y: int, s: int, run_root: Path) -> Path:
    cfg = json.loads((run_root / "inference" / "ic_source.json").read_text())
    src = cfg["ic_source"]
    if src == "plev_data":
        return Path("/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/plev_data") / f"{Y}_gaussian.nc"
    if src == "sigma_data_transferred":
        return Path("/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/sigma_data") / f"{Y}_gaussian.nc"
    if src == "ic_nc_built_from_h5":
        return run_root / "inference" / "ic_nc" / f"{Y}_{s:04d}.nc"
    raise ValueError(f"unknown ic_source: {src!r}")
```

`ic_source.json` is written **once** at the §3 P-7 gate-pass moment, with fields `{"ic_source": "<one-of-three>", "resolved_at": "<UTC-iso>", "gate_pass_sha256": "<sha256-of-the-passing-fixture>"}`. Every downstream consumer (orchestrator, scorer pre-flight, adapter) reads it via `resolve_ic_nc_path(...)` — never assumes `plev_data/`.

```bash
# Set + EXPORT RUN_ROOT before anything else reads it.
# (The Python pre-resolution heredoc below reads os.environ["RUN_ROOT"], so
#  RUN_ROOT must be exported into the environment of that subprocess. A bare
#  shell assignment without `export` is visible only inside the parent shell.)
RUN_ROOT=$RESULTS/sfno_eval_5410/<run_tag>
export RUN_ROOT

# Make AI-RES src importable BEFORE cd'ing into the upstream tree.
# (After `cd /work2/.../v2.0/`, Python's CWD-based path resolution would not
#  find sfno_inference_5410.* — explicit PYTHONPATH is required for the
#  resolve_ic_nc_path import in the heredoc below and for any AI-RES helpers
#  the orchestrator imports later. SLURM scripts MUST set this; the
#  orchestrator fails fast if AI-RES src is unimportable.)
export PYTHONPATH=$AI_RES/src:${PYTHONPATH:-}

# Pre-resolve all 96 IC paths up-front (single Python invocation, before any
# cd) so the loop body doesn't depend on CWD or repeated process startup.
# Writes one path per line to a tempfile, indexed by (Y, s) order.
IC_PATH_LIST=$(mktemp)
python - <<'PY' >"$IC_PATH_LIST"
from pathlib import Path
import os
from sfno_inference_5410.ic_source import resolve_ic_nc_path
run_root = Path(os.environ["RUN_ROOT"])
for Y in range(121, 129):
    for s in [0, 122, 244, 366, 488, 610, 732, 854, 976, 1098, 1220, 1342]:
        print(resolve_ic_nc_path(Y, s, run_root))
PY

cd /work2/.../PanguWeather/v2.0/
IC_IDX=0

for Y in 121 122 123 124 125 126 127 128; do
    YAML=$AI_RES/config/SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y${Y}.yaml
    CONFIG=SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y${Y}    # = yaml basename
    OUT_DIR=$RUN_ROOT/inference/upstream_raw

    for s in 0 122 244 366 488 610 732 854 976 1098 1220 1342; do
        # IC NetCDF path resolved through ic_source.json — NOT hardcoded.
        # Returns plev_data/<Y>_gaussian.nc OR sigma_data/<Y>_gaussian.nc
        # OR <run_root>/inference/ic_nc/<Y>_<s>.nc depending on which source
        # passed §3 P-7's compatibility gate. Pre-resolved above; here we
        # just index into the precomputed list.
        IC_NC=$(sed -n "$((IC_IDX + 1))p" "$IC_PATH_LIST")
        IC_IDX=$((IC_IDX + 1))

        # init datetime: Jan 1 Y 00:00 + s × 6h (proleptic_gregorian)
        INIT=$(python -c "import cftime, datetime as dt; \
            t = cftime.DatetimeProlepticGregorian($Y,1,1,0,has_year_zero=True) \
                + dt.timedelta(hours=$s*6); \
            print(t.strftime('%Y-%m-%d_%H:%M:%S'))")
        # final datetime: Jan 1 (Y+1) 00:00 — drives long_rollout_years = 1
        # (long_inference.py uses final.year-init.year as outer-loop count,
        #  and rolls to Jan 1 of the *next* year regardless; final_datetime
        #  must satisfy final.year > init.year — see §B.0.)
        FINAL=$(printf '%04d-01-01_00:00:00' $((Y+1)))

        python -u long_inference.py \
            --run_num=5410 \
            --yaml_config=$YAML \
            --config=$CONFIG \
            --init_datetime=$INIT \
            --final_datetime=$FINAL \
            --init_nc_filepaths=$IC_NC \
            --output_dir=$OUT_DIR \
            --save_basename=Y${Y}_s$(printf '%04d' $s) \
            --async_save
    done
done
```

Notes:
- **`--init_nc_filepaths`** is `required=True` at `long_inference.py:1227`. We pass **the IC NetCDF resolved by `resolve_ic_nc_path(Y, s, run_root)` (§3 P-7 gate result, §B.2 dispatcher)**, which is one of:
  - per-year `plev_data/<Y>_gaussian.nc` (source `plev_data`, if §3 P-7 step 1 passed there);
  - per-year `sigma_data/<Y>_gaussian.nc` after a contingency-C-A Globus transfer (source `sigma_data_transferred`); or
  - per-IC `<run_root>/inference/ic_nc/<Y>_<s:04d>.nc` built locally from h5 (source `ic_nc_built_from_h5`, contingency C-B).
  Whichever source is active, `long_inference.py:1333-1336` opens it with xarray and uses `ds.get_index("time").get_loc(init_datetime)` to find the IC time index. For per-year sources the `time` index is a CFTimeIndex on `proleptic_gregorian` at 6h cadence over the year (verified for `plev_data/` in `test_eval_grid_5410.py`); for per-IC NetCDFs the `time` index has length 1 with the matching `init_datetime`. Either way the lookup succeeds — but only because §3 P-7's per-`(Y, s)` gate exercised it before SLURM submit.
- **`--output_dir`** + **`--save_basename`** are how forecasts are saved (verified `long_inference.py:1189-1212`: `savedir = self.params.output_dir; save_basename = os.path.join(savedir, self.params.save_basename)`; filename = `<save_basename>_member<NNN>_y<YYYY>.nc`). `exp_dir` is **not** the output root — it is only used for the checkpoint-discovery globstr.
- **`--final_datetime = Jan 1 (Y+1) 00:00`** drives `long_rollout_years = (Y+1) - Y = 1`, so the outer loop runs exactly one year-block. The actual buffer size is `(Jan 1 (Y+1) - init_datetime)/6h` = `1460 - s` (non-leap) or `1464 - s` (leap), per `long_inference.py:823-836`. The adapter (§B.5) slices the first 60 leads from this remainder-of-year output.
- **`--config`** matches the per-Y yaml's filename (without `.yaml`), per `long_inference.py:1251` (`YParams(yaml_config, args.config)`).
- **`--run_num=5410`** matches the symlink-shim subdirectory name (§3 P-2).
- **`save_basename = "Y<Y>_s<ssss>"`** distinguishes the 96 IC invocations on disk. The upstream writer appends `_member<NNN>_y<YYYY>.nc` (line 1206); the adapter discovers files by globbing `Y${Y}_s${s}*.nc` and the adapter renames to `<Y>_<ssss>.nc` (the scorer-expected layout, §B.5).
- The orchestrator (`scripts/eval_inference_5410.py`) decides at smoke time whether 96 invocations can share one in-process `Stepper` (cache the model + ckpt across calls) or must each be a fresh subprocess. Subprocess fallback adds ~30 min total (per-call dataset+model init dominated by xarray/torch import overhead). Either way, the resulting NetCDF set must be 96 files in `inference/upstream_raw/`.
- **No** `inference.py` is used anywhere in this plan.

`long_inference.py:1320,1348` build `init_datetime` and `final_datetime` from `params.val_year_start` if `--init_datetime` / `--final_datetime` flags are not passed. We pass both explicitly to lock the per-IC window.

### B.3 Climate-mode (deferred to Phase 2).

Climate-mode rollouts (8 ICs × ~1-year, K=n_samples−1) and `score_climate.py` are **deferred to Phase 2** per the §0 scope discipline. No `inference/upstream_climate/` outputs in Phase 1; no climate panels in the cross-emulator report.

### B.4 Output schema produced by upstream — **TBD until first smoke**.

`long_inference.py:1116, 1189-1213` writes NetCDFs via xarray. The exact dim/variable layout is **not yet known**; the typical PanguWeather/SFNO convention has per-variable arrays:

```
dataset:
  dims:
    time, latitude, longitude, plev (or sigma), ensemble, ...
  variables:
    tas(time, latitude, longitude)
    pl(time, latitude, longitude)
    ta(time, plev_or_sigma, latitude, longitude)
    ua(time, plev_or_sigma, latitude, longitude)
    ...
    zg(time, plev, latitude, longitude)
    pr_6h(time, latitude, longitude)
```

The smoke run in §K step 2 inspects this with `ncdump -h` and pins the exact dim names, variable list, and level coordinate names for each variable. The output_adapter (§B.5) is finalized after that inspection.

### B.5 Output adapter (`src/sfno_inference_5410/output_adapter.py`).

Reads upstream NetCDFs **from `<run_root>/inference/upstream_raw/Y<Y>_s<ssss>_member000_y<YYYY>.nc`** (full-remainder-of-year rollouts, see §B.0/§B.2) and writes scorer-format NetCDFs into **`<run_root>/inference/nwp/<Y>_<ssss>.nc`** (path matches the existing `scripts/score_nwp.py:5` glob `out_root/inference/nwp/*.nc`, so the same scorer reads both tracks).

For each `(Y, s)` IC, the adapter:
1. Opens the upstream NetCDF (≤ ~1460 timesteps).
2. **Resolves the time-index convention** of the upstream NetCDF (one-time, persisted as a constant): does `pred[time=0]` represent the **IC** (de-z-scored input state, identical to `init_state`) or **lead-1** (model output after one forward step)? Answered by `tests/sfno_inference_5410/test_lead_slice_offset.py` (NEW): runs a 1-IC smoke with K=2 (or any short K), then for the resulting NetCDF asserts `max(abs(pred[time=0] - init_state)) < 1e-4` if `pred[0]==IC`, else `max(abs(pred[time=1] - init_state)) > 1e-2`. The result is recorded as `LEAD_SLICE_OFFSET ∈ {0, 1}` in `src/sfno_inference_5410/output_adapter.py`. Based on the `inference.py` pattern at line 241 (`val_output_surface[:,0] = surface_inv_transform(val_input_surface)`) the offset is **expected to be 1** — i.e. `pred[0]==IC`, slice `pred[1:61]` for the K=60 lead window. **The expectation is verified, not assumed.**
3. Reads truth from `<Y>_gaussian.nc` at sample indices `s+1 .. s+60` (matched timestep-by-timestep against the prediction's lead-1..60 outputs).
4. Reads `init_state` from per-timestep h5 `<Y>_<ssss>.h5` for the 52 state channels at sample `s`.
5. **Slices `pred[LEAD_SLICE_OFFSET : LEAD_SLICE_OFFSET + 60]`** from the prediction (discards beyond lead-60 — required because `long_inference.py` produces a full-remainder-of-year buffer regardless of K, see §B.0). Asserts `pred.shape[time_dim] >= LEAD_SLICE_OFFSET + 60`.
6. **Cross-checks the slice against truth at lead 1.** Runs `mean(abs(pred[LEAD_SLICE_OFFSET, c] - truth[s+1, c]))` for `c ∈ {tas, zg500}`. The 6-hour forecast error should be **comparable to** persistence at lead 6h (i.e. `max-abs-diff` of the right magnitude — order ~1 K for tas, ~10 gpm for zg500), **not** order-of-zero (would imply `pred[LEAD_SLICE_OFFSET]==truth[s+1]`, indicating a leakage bug) and **not** order-of-100× larger than persistence (would imply an off-by-one or unit drift). Numeric thresholds: `0.1 K < tas_diff < 10 K`, `1 gpm < zg500_diff < 100 gpm`. Hard fail if either bound is violated; warn if borderline. This catches both off-by-one and silent unit-conversion bugs at the slicing step.
7. Reshapes per-variable arrays into the flat 53-channel schema below.
8. Writes the result with the global attrs the scorer expects.

Target schema **identical to our emulator's** (`sfno_eval_plan.md` §B.4):

```
dims:
  init_time = 1
  lead_time = 60          # K=60 for Phase-1 NWP
  channel   = 53
  channel_ic = 52
  lat = 64
  lon = 128

vars:
  prediction(init_time, lead_time, channel, lat, lon)
  truth(init_time, lead_time, channel, lat, lon)
  init_state(init_time, channel_ic, lat, lon)

global_attrs (must match the existing scorer's expectations at score_nwp.py:130-133):
  ic_file             = "<Y>_<s:04d>.h5"           # group convention; e.g. "121_0000.h5"
                                                    # (NOT "MOST.<Y>.h5" — our-emulator naming)
  ic_sample_idx       = <s>                        # int, 0..n_samples-1
  file_anchor         = "<Y>-01-01 00:00:00"       # Jan-1 anchor of year Y, proleptic_gregorian
  time_plasim_at_ic   = <s × 0.25>                 # days since file_anchor (6h cadence → 0.25 d/step)
  ckpt_path           = /work2/.../checkpoints/ckpt_epoch_50.tar
  ckpt_basename       = ckpt_epoch_50
  model_label         = "sfno_5410"                # consumed by extended score_nwp.py --model-label
  emulator            = "group_sfno_5410"          # informational; report header reads this
  data_packager_sha   = (read from per-timestep h5 attrs; "unknown" if absent)
  upstream_code_sha   = group's PanguWeather/v2.0 git SHA at training (from out.log)
  eval_code_sha       = AI-RES git short SHA at adapter-run time
  rollout_K           = 60
  dhours              = 6
  plasim_calendar     = 'proleptic_gregorian'
```

`ic_file` and `file_anchor` use the group's Jan-1 calendar-year naming and **deliberately do not** match our emulator's `MOST.<year>.h5` / Aug-1-of-(Y+5) anchor. The scorer's `ic_file` parser at `score_nwp.py:139` is extended (§D) to handle both naming conventions via a `--ic-file-regex` flag; default falls back to our emulator's `MOST.<year>.h5` pattern so the our-emulator track is unaffected.

The adapter handles three reshape jobs:

1. **Per-variable → flat 53-channel.** Stack `pl, tas, ta[0..9], ua[0..9], va[0..9], hus[0..9], zg[0..9], pr_6h` along a new `channel` axis in the order from §1. Channel name strings are fixed by `src/sfno_inference_5410/channel_map.py:CHANNEL_NAMES` and asserted to match `len(CHANNEL_NAMES) == 53`.
2. **Level-coord normalization.** Map upstream's level coords (numerical sigma `[0.0383..0.9833]` and plev `[20000..100000]` Pa) to our integer indices `1..10` for sigma and to plev-named labels (`zg500`, `zg1000`, etc.) for `zg`. The `channel_map.py` table is the single source of truth.
3. **Truth alignment.** Read truth fields from `/scratch/.../sim52/plev_data/{Y}_gaussian.nc` at the K lead-time samples (samples `s+1..s+K`), de-z-score if needed (the truth NetCDF should already be in physical units — verify), and stack into the same channel order. The IC field at sample `s` becomes `init_state` (52 state channels only, no `pr_6h`).

`zg` is on plev in **both** upstream output and truth NetCDF, so no sigma↔plev resampling is needed. `ta/ua/va/hus` are on sigma in **both**, so likewise no resampling. This is convenient — alignment is variable-by-variable and channel-name renaming, no interpolation.

If at smoke time we find that the truth NetCDF carries 13 plev for `zg` (per metadata) instead of 10, the adapter selects the 10 levels in the channel slate and drops the other 3.

### B.6 Channel-map table (`src/sfno_inference_5410/channel_map.py`).

```python
CHANNEL_NAMES = [
    "pl", "tas",
    *[f"ta{i}" for i in range(1, 11)],
    *[f"ua{i}" for i in range(1, 11)],
    *[f"va{i}" for i in range(1, 11)],
    *[f"hus{i}" for i in range(1, 11)],
    "zg200", "zg250", "zg300", "zg400", "zg500",
    "zg600", "zg700", "zg850", "zg925", "zg1000",
    "pr_6h",
]
assert len(CHANNEL_NAMES) == 53

SIGMA_LEVELS = [0.0383, 0.1191, 0.2109, 0.3169, 0.4368,
                0.5668, 0.6994, 0.8234, 0.9241, 0.9833]
PLEV_PA      = [20000, 25000, 30000, 40000, 50000,
                60000, 70000, 85000, 92500, 100000]

# Per-channel units written to the adapted NetCDFs. These reflect the group
# post-processing conventions (see plan §1a) — they are NOT standard CF units
# for `pl` or `zg*`, and they MUST NOT be silently "fixed" to standard ones.
CHANNEL_UNITS = {
    "pl":    "1",          # dimensionless ln(p_s); not Pa
    "tas":   "K",
    **{f"ta{i}":  "K"      for i in range(1, 11)},
    **{f"ua{i}":  "m s-1"  for i in range(1, 11)},
    **{f"va{i}":  "m s-1"  for i in range(1, 11)},
    **{f"hus{i}": "kg kg-1" for i in range(1, 11)},
    "zg200":  "gpm", "zg250":  "gpm", "zg300":  "gpm", "zg400":  "gpm",
    "zg500":  "gpm", "zg600":  "gpm", "zg700":  "gpm", "zg850":  "gpm",
    "zg925":  "gpm", "zg1000": "gpm",
    # pr_6h units are recorded **verbatim** from the upstream variable's
    # `units` attr; metadata spelling varies harmlessly across writer versions
    # ("kg m-2", "kg m^-2", "mm", "mm/6h" …). The adapter copies the string
    # through without an allow-list check (§C.1, reviewer audit 2026-05-08);
    # gating happens on magnitude/range plus the documented §1a semantics,
    # not on the unit string.
    "pr_6h":  "<from_upstream>",
}
assert set(CHANNEL_UNITS.keys()) == set(CHANNEL_NAMES)
```

Tested in `test_channel_map.py`: 53-name uniqueness, ordering, `zg500` at index 46, sigma/plev list lengths, `CHANNEL_UNITS["pl"] in {"1", ""}`, `CHANNEL_UNITS["zg500"] == "gpm"`, `CHANNEL_UNITS["pr_6h"]` is the sentinel that triggers upstream-copy in the adapter, every `CHANNEL_NAMES` entry has a `CHANNEL_UNITS` entry. **No assertion** that `pr_6h` units fall in any fixed set.

### B.7 The 58→53 contract.

This contract is enforced inside the upstream model (which has the same 52 + 6 forcing → 53 architecture as our SFNO). We do **not** assert it from our adapter — that lives inside the upstream code. The output_adapter consumes the upstream's 53-channel result and writes it to disk; no in-loop hooks needed on our side.

If a future run of the upstream code produces a different channel count, `output_adapter.py` raises with a clear "expected 53 output channels, got <n>" message before writing anything.

---

## C. Climatology + persistence

### C.1 Climatology — **transferred, not built** (✅ delivered 2026-05-06).

Eval consumes `/scratch/.../sim52/baselines/climatology_proleptic_5410.nc` (built per `docs/2026-05-06_group_sfno_5410_climatology_prompt_for_derecho.md`). Schema `(366, 4, 53, 64, 128)` matches our scorer's expected layout exactly — no transformation needed.

**Pinned constants (single source of truth — copy into test + SLURM consumers, not re-derive):**

```python
# src/sfno_inference_5410/climatology_constants.py  (target file, created in §K step 4)
CLIMATOLOGY_PATH = "/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/baselines/climatology_proleptic_5410.nc"
EXPECTED_SHA256  = "6b12a880637928eb537b6294399618df169426f794862b939ba62e41f2940876"
EXPECTED_BYTES   = 5_085_101_686
YEARS_USED       = tuple(range(12, 112))   # 12..111 inclusive, 100 years
N_YEARS_USED     = 100
N_LEAP_YEARS_USED = 24
N_TIMESTEPS_AGGREGATED = 146_096
DERECHO_BUILD_SHA7 = "feec151b"
GLOBUS_TASK_ID   = "bede3215-49d0-11f1-b97b-0afffe4617ab"
```

Pre-flight (`tests/sfno_eval/test_climatology_load_5410.py`):
- `xarray.open_dataset(...)` succeeds, no warnings.
- File size on disk == `EXPECTED_BYTES`.
- `sha256sum(file) == EXPECTED_SHA256` (hard-coded assert, not "vs manifest").
- **Calendar-axis coord-name reconciliation (§D.0).** The Derecho prompt (§4) specifies dim/coord name `time_of_year` (`docs/2026-05-06_group_sfno_5410_climatology_prompt_for_derecho.md:74`); the existing `scripts/score_nwp.py:79-89` reader uses `ds["doy"]` (matches our-emulator builder `scripts/compute_climatology.py:126,131,145`). The pre-flight asserts the file has **at least one** of `{"doy", "time_of_year"}` as a dim/coord with length 366 and records which one was used so the scorer fallback (§D.0) routes correctly. If both are present (unlikely but harmless), `doy` wins for compatibility.
- `channel` coord matches `channel_map.CHANNEL_NAMES` byte-for-byte.
- `mean.shape == (366, 4, 53, 64, 128)`, `std.shape == (366, 4, 53, 64, 128)`, `n_contributors.shape == (366, 4)`.
- Spot-check: `n_contributors[59, :].max() == 24` (Feb-29 / day-59 must be populated by exactly the 24 leap years in 12-111).
- Spot-check: `n_contributors[0, :].max() == 100` (day-1 populated by all 100 years).
- **Convention checks (§1a regression guards):**
  - `mean["tas"]` averaged over `(time_of_year, hour_quarter, lat, lon)` lies in `[250, 310]` K.
  - `mean["pl"]` averaged similarly lies in `[11.0, 12.0]` (i.e., `ln(p_s)` ≈ 11.51 ± a small spatial spread). If it lands near `1e5`, the Derecho build silently exponentiated — fail loudly.
  - `mean["zg500"]` averaged similarly lies in `[5300, 6000]` gpm. If it lands near `5e4`, somebody multiplied by g₀ — fail loudly.
  - `mean["zg1000"]` lies in `[-200, 400]` gpm; `mean["zg200"]` lies in `[11000, 13000]` gpm.
  - `mean["pr_6h"] >= 0` everywhere (climatological mean of a non-negative quantity is non-negative; this is a strict check on truth/clim, not predictions); channel-mean magnitude is consistent with a 6-hour proxy (typically O(1) mm; not zero, not absurdly large).
  - Per-variable `units` attribute (if present) matches `channel_map.CHANNEL_UNITS` for the strict entries (`pl`, `tas`, `ta*`, `ua*`, `va*`, `hus*`, `zg*`). For `pr_6h` the `units` string is **recorded for the report header but not gated on a fixed allow-list** — metadata spelling varies across writer versions (`"kg m-2"`, `"kg m^-2"`, `"mm"`, `"mm/6h"`, etc. all describe the same physical quantity given §1a's snapshot-rate × 6h convention). The pre-flight relies on the **magnitude/range** check above plus the documented semantics in §1a, not on the unit string.

If any check fails, scoring jobs do **not** run — the SLURM dependency chain breaks at the score step.

### C.2 Persistence — same definition as our emulator.

`persistence(c, lead) = init_state[s, c, ...]` for `c ∈ {0..51}` (52 state channels). For `c = 52` (`pr_6h`), `persistence_rmse[c, lead] = NaN`. Same rationale as `sfno_eval_plan.md` §C.1 (pr_6h is diagnostic-only, no IC value).

The `init_state` field for the 52 state channels is read from the per-timestep h5 at sample `s` (the IC sample). `pr_6h` at sample `s` exists in `fields_diagnostic` but is deliberately not used for persistence.

### C.3 Normalization stats source.

Upstream `long_inference.py` de-z-scores predictions internally using `/scratch/.../sim52/h5/sigma_data/data_12-132_{mean,std}_sigma.nc` (`surface_inv_transform`, `upper_air_inv_transform`, `diagnostic_inv_transform`). The adapter writes **already-de-z-scored** physical-unit fields in the **group post-processing conventions of §1a**, which the scorer (also unit-agnostic) consumes directly. Sanity checks in `test_output_adapter.py` follow §1a, **not** standard CF/ERA5:

- `tas` ∈ [180, 340] K
- `pl` ≈ `ln(p_s)`, climatological mean ≈ 11.3-11.6 (dimensionless) — **NOT in Pa**. If predicted `pl` falls in `[5e4, 1.1e5]` (Pa range), the adapter has silently exponentiated → fail.
- `zg500` ∈ [4500, 6500] gpm — **NOT m² s⁻²**. If in `[4.5e4, 6.5e4]`, somebody multiplied by g₀ → fail.
- `pr_6h`:
  - **Truth and climatology**: strict `>= 0` everywhere (these come from the simulation post-processing and have no negative values by construction; any negative is a unit-convention regression).
  - **Predictions**: gross magnitude bound `pr_6h ∈ [-2, 50]` (mm or kg m⁻²) — predictions can produce small negative values from numerical artifacts of the network without indicating a unit-convention bug. A few-percent of grid cells with `pred_pr_6h ∈ [-2, 0]` is acceptable; whole-field magnitude away from this range is a fail.

---

## D. NWP scoring

**Both tracks call the same `scripts/score_nwp.py`** — extended once with three configurability hooks (decided 2026-05-07: extend, do not fork). The changes are backward-compatible for our emulator's existing invocation.

### D.0 `score_nwp.py` extensions (one-time edit, both tracks benefit).

| New flag | Default | Effect |
|---|---|---|
| `--leads <h_csv>` | `6,24,72,120,240,336` (= existing `_SCORED_LEADS_H`) | Comma-separated lead-time hours to score. SFNO-5410 invocation passes `--leads 6,24,72,120,240,336,360` (adds k=60). |
| `--model-label <str>` | `emulator` (= our emulator's existing label) | Value written into the `model` column of `nwp_scorecard.csv`. SFNO-5410 invocation passes `--model-label sfno_5410`. |
| `--ic-file-regex <re>` | `r"MOST\.(\d{4})\.h5"` (= our emulator's `MOST.<year>.h5` convention; capture group 1 = ic_year) | Regex applied to the `ic_file` global attr to extract `ic_year`. SFNO-5410 invocation passes `--ic-file-regex '(\d+)_\d+\.h5'` (group convention `<year>_<idx>.h5`). |
| `--clim-coord-name <doy|time_of_year|auto>` | `auto` (probes the climatology file: `doy` if present, else `time_of_year`) | Selects the calendar-axis coord/dim name. Both our-emulator's `compute_climatology.py` (uses `doy`) and the Derecho prompt (uses `time_of_year`) are now supported. The pre-flight test in §C.1 records which name the active climatology file uses. |
| `--write-climatology-row` (boolean flag, default off) | off (preserves existing CSV layout) | When set, also writes RMSE rows with `model="climatology"` for every (channel, lead) pair, computed as `rmse_lat_weighted(clim_mean[doy_idx, hq_idx, c], truth[k, c], lat_w)`. Used by the cross-emulator report (§F) so it can show a climatology baseline column. |

Code refactor needed in `scripts/score_nwp.py`:
- Replace module-level `_SCORED_LEADS_H` (line 54) with `parse_leads(args.leads) -> tuple[int, ...]` and thread it through `_compute_metrics_for_one_ic`.
- Replace hardcoded `ic_file.replace("MOST.", "").replace(".h5", "")` (line 139) with `re.match(args.ic_file_regex, ic_file).group(1)`.
- Replace hardcoded `model="emulator"` (lines 156, 168) with `args.model_label`.
- In `_load_clim_for_lookup` (line 79): replace `ds["doy"].values` with `ds[args.clim_coord_name].values` — and resolve `auto` once at startup by checking which name is present.
- In the inner loop after `# === PERSISTENCE RMSE === ` (line 173): when `args.write_climatology_row`, append a `model="climatology"` row using `clim_mean[doy_idx, hq_idx, c]` as the prediction surrogate (no trick: the RMSE is well-defined since `clim_mean[doy_idx, hq_idx, c]` and `truth[k, c]` have the same shape `(H, W)`).

`tests/sfno_eval/test_score_nwp_cli.py` (NEW) asserts:
- Defaults reproduce the our-emulator numbers byte-for-byte on a fixed fixture (regression guard).
- SFNO-5410 flags `--leads ... --model-label sfno_5410 --ic-file-regex '(\d+)_\d+\.h5' --clim-coord-name auto --write-climatology-row` resolve correctly on a synthetic 5410 fixture (`<year>_<idx>.h5` ic_file, climatology with either `doy` or `time_of_year`).
- The `--write-climatology-row` flag adds rows with `model="climatology"` and does **not** alter `model="emulator"` / `model="persistence"` row values.

### D.1 Lat-weights (Gauss–Legendre).

Identical to `sfno_eval_plan.md` §B.5 / §D.1. Cached at `stats/lat_weights_legendre_gauss.npy` — shared across both tracks.

### D.2 RMSE / D.3 ACC / D.4 Bias maps.

Identical formulas (`rmse_lat_weighted`, `acc`, `bias_map`) to `sfno_eval_plan.md` §D.1-D.3. Bias maps written for the same 5 key channels (`tas, pr_6h, zg500, ua5, ta5`).

### D.5 Lead times scored.

5410: `{6h (k=1), 24h (k=4), 72h (k=12), 120h (k=20), 240h (k=40), 336h (k=56), 360h (k=60)}` — adds **k=60** relative to our emulator's 6-lead set, passed via `--leads 6,24,72,120,240,336,360`. The cross-emulator report (§F) reports the **intersection** `{6, 24, 72, 120, 240, 336}`; `360h` appears only in the 5410-only Phase-1 report and in the appendix of the cross report.

### D.6 Output: `scores/nwp_scorecard.csv`.

Columns: `model, channel, lead_hours, ic_year, ic_sample_idx, metric, value`. Same tidy long format as our emulator's CSV; both tracks write into their own `<run_root>/scores/nwp_scorecard.csv` (sibling output trees, not mixed). With the v3 scorer extensions (§D.0), each track's CSV contains rows for **three** distinct `model` labels:

- `model = "<emulator_label>"` — `"emulator"` (our-emulator track, default `--model-label`) or `"sfno_5410"` (this track).
- `model = "persistence"` — RMSE of `init_state` against truth at each lead, written for all 53 channels (NaN for `pr_6h`, the diagnostic-only channel; §C.2).
- `model = "climatology"` — RMSE of `clim_mean[doy_idx, hq_idx, c]` against truth at each lead, written **only** when `--write-climatology-row` is passed.

Aggregation: `nwp_scorecard_summary.csv` averaged over IC dimension, grouped by `(model, channel, lead_hours, metric)`.

`scripts/render_cross_emulator_report.py` is responsible for **filtering by `model` and treating baselines as per-track artifacts** (not shared):

```python
# Row classification used by render_cross_emulator_report.py
EMULATOR_LABELS = {"emulator", "sfno_5410"}        # one per track
BASELINE_LABELS = {"persistence", "climatology"}   # PER-TRACK; not assumed shared
```

The joined table in §F section 2 takes the **emulator row from each track** and the **baseline rows from each track separately**. **Persistence and climatology RMSEs are NOT comparable across tracks**, because the two tracks have different IC schedules and different baseline-source files (verified 2026-05-08, reviewer audit):

- **IC anchor & stride differ.** Our-emulator uses Aug-1 / Y+5 anchor, stride 116 (`docs/sfno_eval_plan.md` §A.4); 5410 uses Jan-1 / Y anchor, stride 122 (this plan §A.2). The two tracks evaluate **different IC moments** — even though both cover test years 121-128, the actual `(year, calendar_date)` of each IC differs between tracks. Persistence RMSE depends on `init_state` value, which depends on the IC moment — so per-track persistence values cannot be assumed equal.
- **`init_state` source files differ.** Our-emulator reads from Makani-packaged `MOST.<year>.h5`; 5410 reads from group `<year>_<s>.h5` (or per-IC NetCDF under contingency C-B). Although Derecho confirmed bit-equivalence for sampled channels (climatology-prompt §2), this is not byte-equality on all 53 channels at every IC.
- **Truth-NetCDF source files differ.** Our-emulator scores against Makani-packaged truth; 5410 scores against `plev_data/<Y>_gaussian.nc`. Both originate in the same PlaSim simulation but go through different post-processing pipelines.
- **Climatology source files differ.** Our-emulator uses `<our-clim>.nc` built from `sim52_full/train` years 12-111; 5410 uses `climatology_proleptic_5410.nc` built from group-processed `sigma_data/<year>_gaussian.nc` years 12-111 (§3 P-3). Even with identical year sets, the post-processing pipelines diverge.

**Consequence — cross-report rules (replaces v5's "use whichever track scored first"):**

1. **No persistence-equality assertion.** The cross-report does **not** compare persistence values across tracks; the `|persistence_ours - persistence_5410| < 1e-6` check from v5 is removed. (It would routinely fail, and even when it passes that's coincidence, not a load-bearing invariant.)
2. **Per-track baseline columns.** §F section 2's table has 4 model columns: `our_emulator | persistence_ours | sfno_5410 | persistence_5410` (and `climatology_ours | climatology_5410` if both tracks ran with `--write-climatology-row`). Bold the better RMSE per `(channel, lead)` cell across the two emulators only; baselines are presented for context, not bolded against each other.
3. **Unpaired distributional comparison.** The cross-emulator delta is **unpaired**: it averages each track's RMSE/ACC over its own 96 ICs, then reports the difference of means. The report header explicitly states this and notes the per-track IC-schedule differences (anchor + stride). An exact paired comparison would require a unified IC schedule, which is out of scope for Phase 1.
4. **If a track lacks `--write-climatology-row`:** the report drops that track's `climatology_*` column with a one-line note rather than emitting blank cells.

### D.7 Sanity gate (same thresholds).

- 5410 RMSE on `tas` at 6 h **<** persistence RMSE on `tas` at 6 h.
- 5410 ACC on `zg500` at 24 h **>** 0.6.
- **Finiteness, by row class:**
  - `model="sfno_5410"` rows: RMSE finite (no NaN/Inf) for **all** 53 channels × 7 leads.
  - `model="climatology"` rows (when `--write-climatology-row` is passed): RMSE finite for **all** 53 channels × 7 leads.
  - `model="persistence"` rows: RMSE finite for the **52 state channels** only. Persistence RMSE for `pr_6h` is **intentionally NaN** by design (`§C.2`: pr_6h is diagnostic-only, no IC value), and that NaN must propagate into the CSV — **the gate does not flag it**. Any other persistence NaN (e.g. on a state channel) is a fail.

If any fail: STOP the chain, escalate, do not render the cross-emulator report.

The 0.6 ACC threshold is inherited from `sfno_eval_plan.md` §D.6; it was set conservatively for PlaSim's simpler atmosphere. The 5410 emulator was trained for 50 epochs on this same simulation, so it should clear this gate easily; failure indicates an evaluation-pipeline bug, not a model deficiency.

---

## E. Climate scoring — **DEFERRED to Phase 2** (both tracks).

Per the §0 scope discipline (decided 2026-05-07): `score_climate.py` does not exist for either track, and Phase 1 is scientifically complete with NWP forecast skill alone. Climate-mode rollouts (1 IC/year × ~1-year), time-mean bias maps, zonal-mean heatmaps, variance ratios, and drift windows are all moved to a separate **Phase 2 plan** to be drafted after the NWP report lands.

In Phase 1: no `inference/climate/` outputs, no `scores/climate_*.csv`, no `climate_stub/` directory, no climate panels in the cross-emulator report (§F). KE / temperature variance spectra also stay deferred (already locked in §0).

---

## F. Cross-emulator report

`scripts/render_cross_emulator_report.py` produces `results/sfno_eval_cross/<cross_run_tag>/report.md`, joining outputs from both tracks. **Phase 1 = NWP only; no climate panels.**

**Erratum re companion plan.** `docs/sfno_eval_plan.md` v2.8 (our-emulator plan) still describes a Phase-1 climate stub in its §E. Per the cross-emulator-symmetry decision in §0, the cross-report **explicitly ignores** any `results/sfno_eval/<our_run_tag>/scores/climate_stub/` outputs even if they exist on disk: `scripts/render_cross_emulator_report.py` reads only `scores/nwp_scorecard.csv` from each track and never globs for climate artifacts. Climate-mode for both tracks is reserved for the future Phase 2 plan (§0). The companion plan will get its own Phase-2 erratum when that plan is drafted; touching it now is out of scope here.

Sections:

1. **Header** — both run tags, climatology source identifiers (Derecho manifest SHA + Stampede3 path), eval-code SHA, sanity-gate results for each emulator, **explicit note that test years 121-128 are the only years scored on both sides; year 129 is excluded from every comparison row**. Includes a fixed **Variable conventions** block reproducing the §1a table verbatim so the reader knows that `zg500` numbers are gpm-error (not m² s⁻²-error), `pl` numbers are dimensionless `ln(p_s)`-error, and `pr_6h` is scored as a 6-hour proxy rather than a true accumulator.
2. **NWP scorecard table — unpaired distributional comparison, per-track baselines (§D.6).** Channels × lead times × **per-track columns**: `our_emulator | persistence_ours | climatology_ours | sfno_5410 | persistence_5410 | climatology_5410`. Each cell is that track's mean RMSE/ACC over its own 96-IC schedule (different anchors and strides per §D.6). Lead-time set is the **intersection** `{6, 24, 72, 120, 240, 336}` (k=60 = 360h dropped, since our emulator is K=56). Bold the better RMSE per `(channel, lead)` cell **across the two emulator columns only**; baselines are presented for context, not bolded against each other and not assumed equal across tracks. If a track was not run with `--write-climatology-row`, that track's climatology column is omitted (with a one-line note); persistence columns are always present. **The table caption explicitly states the comparison is unpaired: "each track averages its own 96 ICs (Aug-1/stride-116 for our_emulator, Jan-1/stride-122 for sfno_5410); the cross-track delta is a difference of means, not a paired per-IC delta."**
3. **ACC line plots** — per channel, both emulators on the same axes (PNG embedded).
4. **Bias-map grid** — `(our, 5410)` shown **side-by-side** for each `(channel, lead)` ∈ {5 × 3}. Each panel is rendered from its own track's IC schedule (Aug-1/stride-116 vs Jan-1/stride-122 per §D.6), so the two sides are not paired per IC; they are independent estimates of time-mean bias on overlapping test years 121-128.
5. **Sanity gates** — explicit PASS/FAIL per emulator with which check failed.
6. **5410-only appendix** — k=60 (360h) lead row for the 5410 track, presented with no our-emulator comparison column.
7. **Manual commentary slot** — placeholder for Zhixing's writeup; the script never auto-fills this.

`cross_run_tag = ${YYYYMMDD}_eval-${EVAL_SHA7}_ours-${OURS_RUN_TAG_SHORT}_5410-${5410_RUN_TAG_SHORT}` where `*_SHORT` are the date+ckpt fields of each track's run-tag.

---

## G. Compute & SLURM

### G.1 Three jobs (chained) — separate from our-emulator chain.

| Job | Wallclock | GPUs | What it does |
|---|---|---|---|
| `submit_eval_inference_5410.slurm` | 4 h | 1 × H100 | NWP only via `long_inference.py`, **full-remainder-of-year per IC** (§B.0). Total forward steps: `6 non-leap × Σᵢ₌₀..₁₁(1460-122i) + 2 leap × Σᵢ₌₀..₁₁(1464-122i) = 6×9468 + 2×9516 = 75 840 forward steps`. At ~15 ms/step on H100 single-GPU → ~19 min compute; per-invocation dataset+model init (~20 s × 96) ≈ 32 min. **Total wallclock ≈ 50-60 min** + I/O + ckpt-loads, comfortably inside 4 h. Per-IC NetCDFs land at `<run_root>/inference/upstream_raw/Y<Y>_s<ssss>_member000_y<YYYY>.nc`; storage budget per IC ≤ ~2.5 GB (s=0, full year), down to ~200 MB (s=1342); aggregate ~125 GB. SLURM script re-asserts §3 P-7 (IC-NetCDF compatibility) before launching. |
| `submit_eval_score_5410.slurm` | 1 h | 0 (CPU) | Adapter (upstream → scorer schema, slices first 60 leads per §B.5) + `score_nwp.py --model-label sfno_5410 --leads 6,24,72,120,240,336,360 --ic-file-regex '(\d+)_\d+\.h5' --clim-coord-name auto --write-climatology-row`. Pre-flight asserts climatology presence + sha256 (§3 P-3) and the §C.1 calendar-axis-name probe ran. No new climatology build. |
| `submit_eval_report_5410.slurm` | 30 min | 0 (CPU) | `render_eval_report_5410.py` for the 5410-only Phase-1 report (NWP only). |

### G.2 Cross-report job.

`submit_eval_report_cross.slurm` (30 min, CPU): depends on **both** our-emulator's report job AND `submit_eval_report_5410.slurm`. Renders the combined `results/sfno_eval_cross/<cross_run_tag>/report.md`.

### G.3 `submit_eval_5410.sh`.

```bash
JOB_INF=$(sbatch --parsable submit_eval_inference_5410.slurm)
JOB_SCO=$(sbatch --parsable --dependency=afterok:$JOB_INF submit_eval_score_5410.slurm)
JOB_REP=$(sbatch --parsable --dependency=afterok:$JOB_SCO submit_eval_report_5410.slurm)
echo "Inference: $JOB_INF, Score: $JOB_SCO, Report: $JOB_REP"
```

Cross-report submission is a separate manual step (`submit_eval_report_cross.slurm`), since it requires **both** tracks' report jobs to be done.

### G.4 Run tag.

```
run_tag = ${YYYYMMDD}_eval-${EVAL_SHA7}_5410_data-${DATA_SHA7}_train-${UPSTREAM_SHA7}_ckpt-ckpt_epoch_50
```

- `EVAL_SHA7` — AI-RES git short SHA at eval-submit time.
- `DATA_SHA7` — `packager_git_sha[:7]` from per-timestep h5 attrs (if present); else `unknown`. The group's per-timestep h5s may not carry this attribute — verify at preflight; the run-tag scheme tolerates `unknown`.
- `UPSTREAM_SHA7` — PanguWeather/v2.0 git short SHA at training time. Extract from `/work2/.../results/SFNO/5410/out.log` first line that mentions a git hash; fall back to `unknown` if absent.

### G.5 Stampede3 env.

The eval submit script sources a venv that satisfies upstream's `enviornment.yml`. Smoke run (§K step 1) identifies which deps are missing from the AI-RES baseline env and pins them. Document the resulting env spec at `external/PanguWeather_stampede3_env.txt`.

---

## H. Tests

| Test | Asserts |
|---|---|
| `test_yaml_override.py` | For each `Y ∈ {121..128}`: `build_5410_yaml_override.py --year Y` produces a yaml with **zero** `/glade/` substrings; loads via `YParams` without error; `val_year_start == Y`, `val_year_end == Y + 1`, `save_forecasts == True`, `log_to_wandb == False`. All 8 yamls have identical model architecture (only the two int fields differ). The **single-file symlink shim** for that Y resolves: `os.path.islink(...ckpt_epoch_50.tar)` AND `os.path.realpath(...) == "/work2/.../v2.0/results/SFNO/5410/checkpoints/ckpt_epoch_50.tar"` AND `natsorted(glob(...ckpt_epoch_*.tar))[-1]` is exactly that file. Sanity-loads `latitudes` and `longitudes` lists from the override unchanged. |
| `test_runtime_args_5410.py` (NEW) | The orchestrator's argv constructor produces, for every `(Y, s)` in the run plan: `--init_nc_filepaths == resolve_ic_nc_path(Y, s, run_root)` (path **as the IC-source dispatcher returns it** — `plev_data/<Y>_gaussian.nc` for source `plev_data`, `sigma_data/<Y>_gaussian.nc` for source `sigma_data_transferred`, or `<run_root>/inference/ic_nc/<Y>_<s>.nc` for source `ic_nc_built_from_h5`), and that file exists; `--init_datetime` parses to `cftime.DatetimeProlepticGregorian(Y, ...)` such that `start_date.year == Y == params.val_year_start` (boundary-loader contract, §B.0); `--final_datetime` parses to `cftime.DatetimeProlepticGregorian(Y+1, 1, 1, 0)` (drives `long_rollout_years = 1`); `--output_dir` exists and is writable; `--save_basename == f"Y{Y}_s{s:04d}"`; `--config` matches the per-Y yaml filename (without `.yaml`); `--run_num == "5410"` (matches symlink-shim subdir). Also asserts that `xr.open_dataset(resolve_ic_nc_path(Y, s, run_root)).get_index("time").get_loc(parse_init_datetime)` succeeds for every `(Y, s)`, and that `<run_root>/inference/ic_source.json` exists with a recognized `ic_source` value. |
| `test_ic_offsets.py` | `nwp_ic_offsets_5410(1460, K=60, n_ic=12, step=122) == [0, 122, 244, 366, 488, 610, 732, 854, 976, 1098, 1220, 1342]`; identical for `n_samples=1464`; `s + K < n_samples` for all returned `s` in both cases; the function raises if `step × (n_ic-1) + K >= n_samples`. |
| `test_channel_map.py` | `len(CHANNEL_NAMES) == 53`; uniqueness; `CHANNEL_NAMES[46] == "zg500"`; `len(SIGMA_LEVELS) == 10` and `len(PLEV_PA) == 10`; `CHANNEL_UNITS["pl"] in {"1", ""}`, `CHANNEL_UNITS["zg500"] == "gpm"`, `CHANNEL_UNITS.keys() == set(CHANNEL_NAMES)`. |
| `test_output_adapter.py` | Round-trip on a synthetic upstream NetCDF: adapter produces shape `(1, 60, 53, 64, 128)` with the right `channel` coord; `init_state.shape == (1, 52, 64, 128)`; **attrs `ic_file = "<Y>_<s:04d>.h5"`, `ic_sample_idx = <s>`, `file_anchor = "<Y>-01-01 00:00:00"`, `time_plasim_at_ic = <s × 0.25>`, `model_label = "sfno_5410"`**; physical-unit sanity per §C.3 (`tas` ∈ [180, 340] K; `pl` ≈ 11.3-11.6, **NOT** ~10⁵; `zg500` ≈ 5400-5900 gpm, **NOT** ~5e4). |
| `test_climatology_load_5410.py` | Pre-flight check on the transferred 5410 climatology: presence, sha256 == `6b12a880…0876`, schema `(366, 4, 53, 64, 128)`, `channel` coord, no NaN in populated bins, `n_contributors[59].max() == 24`, `n_contributors[0].max() == 100`. Also resolves which calendar-axis name (`doy` vs `time_of_year`) is present and records it for the score-job pre-flight (§C.1, §D.0); fails if neither is present. |
| `test_eval_grid_5410.py` | Truth NetCDF's `lat` matches yaml `lat` to 1e-6; `lon` likewise; channel/level coords are consistent across all 8 truth files (`121_gaussian.nc`..`128_gaussian.nc`); `time[0]` of each is `cftime.DatetimeProlepticGregorian(Y, 1, 1, 0, 0, 0, has_year_zero=True)`. |
| `test_score_nwp_cli.py` (NEW) | `score_nwp.py` defaults reproduce the our-emulator existing fixture byte-for-byte (regression guard); `--leads 6,24,72,120,240,336,360 --model-label sfno_5410 --ic-file-regex '(\d+)_\d+\.h5' --clim-coord-name auto --write-climatology-row` runs without crash on a synthetic fixture with `<Y>_<s>.h5` ic_file attrs and a climatology with either `doy` or `time_of_year` calendar coord; yields a 7-lead × 53-channel scorecard with rows for `model ∈ {"sfno_5410", "persistence", "climatology"}`. The climatology rows have finite RMSE; the `--write-climatology-row` flag does **not** alter `model="sfno_5410"` or `model="persistence"` row values (defaults-vs-flag regression). |
| `test_ic_nc_compatibility.py` (NEW, hard gate, §3 P-7) | **Per-`(Y, s)` sanity-load over the full 96-tuple run plan**: for every `(Y, s)`, invokes upstream's actual `utils.data_loader_multifiles.get_data_given_path_nc` against `resolve_ic_nc_path(Y, s, run_root)` with `(['ta','ua','va','hus','zg'], ['pl','tas'], init_datetime = cftime.DatetimeProlepticGregorian(Y, 1, 1, 0) + s × 6h, levels_per_var=...)`. Asserts (a) the call returns without raising for every (Y, s); (b) the resulting tensor shape matches `(<flat-channel-count>, 64, 128)`; (c) every level-match succeeds within tolerance and every `time`-lookup hits the requested `init_datetime`. **H5 cross-check coverage by source**: for `plev_data` / `sigma_data_transferred` (per-year files), spot-check `(121, 0)`, `(124, 0)` (leap), `(128, 1342)` — 3 calls; for `ic_nc_built_from_h5` (per-IC files), **all 96 (Y, s) tuples** since each NetCDF is distinct. Each cross-check asserts `max-abs-diff < 1e-4` against `get_data_given_path('<Y>_<s:04d>.h5', ...)` for all 52 prognostic channels. Failure trips the contingency cascade in §3 P-7. |
| `test_lead_slice_offset.py` (NEW, §B.5) | After a 1-IC smoke rollout: opens the upstream NetCDF, computes `LEAD_SLICE_OFFSET ∈ {0, 1}` by checking which time index matches `init_state` (read from h5 at sample s) within `1e-4`. Records the value as a constant in `output_adapter.py` and asserts it equals the expected upstream-pattern value (1 for `long_inference.py`, matching `inference.py:241`). Also asserts the lead-1 cross-check thresholds in §B.5 step 6 are satisfied on the smoke output. |
| `test_smoke_eval_5410_cpu.py` | CPU smoke: 1 NWP IC, K=2 (not 60), end-to-end through adapter + extended scorer with synthetic clim. Validates orchestration, not numerics. |

The 5410-specific tests live under `tests/sfno_inference_5410/` and `tests/sfno_eval/`. The `test_score_nwp_cli.py` test goes under `tests/sfno_eval/` since it exercises the shared scorer.

---

## I. Risks & open issues

| Risk | Likelihood | Mitigation |
|---|---|---|
| **IC-NetCDF source `plev_data/<Y>_gaussian.nc` doesn't satisfy SFNO loader's `levels_per_var` (sigma for ta/ua/va/hus, plev for zg) — `get_data_given_path_nc` raises `ValueError: Level ... not found in file`** | **High** | §3 P-7 hard gate runs the actual upstream loader before any SLURM submit. Failure forces a documented contingency: (C-A) transfer `sigma_data/<Y>_gaussian.nc` from Derecho, or (C-B) build per-IC NetCDFs from h5 via `scripts/build_ic_nc_from_h5.py`. No silent fallback. |
| Lead-slicing off-by-one in adapter (whether `pred[time=0]` is the IC or lead-1) silently shifts every reported RMSE/ACC by one 6-h step | High if regressed | `test_lead_slice_offset.py` empirically determines `LEAD_SLICE_OFFSET` from a smoke rollout (compares `pred[0]` and `pred[1]` against the de-z-scored IC at 1e-4 tol); cross-check at §B.5 step 6 catches both directions of off-by-one and unit drift before scoring. |
| `long_inference.py:Stepper` cannot be re-driven with different `init_datetime` per IC without rebuilding the model — orchestrator must use 96 subprocess invocations instead of 96 in-process calls | Medium | §B.2 fallback: 96 independent subprocess calls. Adds ~30 min to inference wallclock (per-call dataset+model init). Decided at smoke (§K step 2). |
| Full-remainder-of-year rollout (§B.0) accidentally truncated by a future upstream change, producing fewer than 60 lead steps | Low | Adapter (§B.5) asserts `pred.shape[lead_dim] >= 60` before slicing; failure → fail-fast at score-step pre-flight, not silent truncation. |
| 8 per-Y yaml + 8 per-Y symlink shims fall out of sync (e.g., one Y's shim missing) | Low-Medium | `test_yaml_override.py` iterates `Y ∈ {121..128}` and asserts the shim+yaml exist for each before any SLURM submission; `submit_eval_inference_5410.slurm` re-asserts at job start. |
| Climatology calendar-axis coord is named differently than expected (`time_of_year` per Derecho prompt vs `doy` in our existing `compute_climatology.py`) — would crash the scorer with `KeyError: 'doy'` mid-run | Medium | §C.1 pre-flight resolves the actual coord name; §D.0 `--clim-coord-name auto` falls back to whichever exists. `test_climatology_load_5410.py` blocks the SLURM chain if neither is present. |
| `--init_nc_filepaths`-targeted `<Y>_gaussian.nc` truth NetCDF doesn't have `init_datetime` in its time index (e.g., wrong calendar or 0-indexing offset) → `get_index("time").get_loc(...)` raises | Low-Medium | `test_runtime_args_5410.py` exercises `get_loc` for every `(Y, s)` tuple before SLURM submit. Smoke run (§K step 2) validates one IC end-to-end. |
| Upstream NetCDF schema differs per variable in unanticipated ways | Medium | First implementation step is a 1-IC smoke + `ncdump -h`; adapter pinned afterwards. |
| Stampede3 venv missing PanguWeather/v2.0 deps (e.g., specific `cftime`, `xarray`, `torch_harmonics` versions) | Medium | Phase-0 smoke identifies the delta; pinned at `external/PanguWeather_stampede3_env.txt`. |
| `ckpt_epoch_50.tar` saved under torch 2.6.0+cu124 but Stampede3 env is older | Low-Medium | Verify torch version in venv; upgrade if needed. CUDA 12.4 → 12.x compatibility usually fine on H100. |
| Channel-name mismatch between upstream NetCDF and our scorer (e.g., `zg_50000.0` vs `zg500`) | Medium | `channel_map.py` is the single source of truth; adapter normalizes on read. |
| **Silent unit conversion of `pl`, `zg*`, or `pr_6h`** (e.g., adapter exponentiates `pl`, multiplies `zg*` by g₀, or re-derives `pr_6h` from a rate field) — would corrupt RMSE/ACC across both emulators if the climatology and predictions diverge in convention | High if regressed | §1a documents conventions explicitly; `channel_map.CHANNEL_UNITS` is the single source of truth; adapter sets `units` from this table verbatim; §C.1 + §C.3 pre-flight magnitude checks catch convention drift; report header reproduces the convention table so reviewers see what was scored. |
| `score_nwp.py` extension introduces a regression in the our-emulator track (e.g., the new `--ic-file-regex` parser changes behavior for `MOST.<year>.h5`) | High if regressed | `test_score_nwp_cli.py` asserts default-flag invocation reproduces existing our-emulator fixture byte-for-byte. CI runs both default and 5410 invocations. |
| Symlink shim breaks (upstream `results/SFNO/5410/checkpoints/` moved or deleted) | Low | `test_yaml_override.py` resolves the symlink and asserts `ckpt_epoch_50.tar` is the natsort-last globstr match; failure aborts before SLURM submit. |
| Climatology Feb-29 bins underweighted in ACC (n_contributors ≈ 24 vs ~100) | Low | Report annotates per-bin n_contributors; bins with `n < 10` flagged "low-N", not used for cross-emulator deltas. |
| Per-Y rollout `Y=128` reaches Jan 1 (Y+1)=129 within its full-remainder buffer (§B.0), causing the loader to touch year-129 h5s | Low | The loader's per-Y val window is `[Y, Y+1)` (exclusive end). With `val_year_end=129` only when `Y=128`, full-remainder rollouts conceptually run to the end of year 128. `output_inference_steps` is sized as `(Jan 1 (Y+1) - init_datetime)/6h = 1464 - s`, all within year 128. Smoke run inspects which h5s the dataset opens to confirm. |
| Truth NetCDF `lat` orientation differs from yaml (south-first vs north-first) | Low | `test_eval_grid_5410.py` catches; adapter flips if mismatch. |
| Group climatology source years mismatch SFNO-5410's actual training years (e.g., 5410 used a subset) | Low | Derecho prompt's manifest lists exact years used; preflight checks the manifest's `years_used` covers ≥80 of years 12-111. |

---

## J. Open questions

1. **(B.2)** Can `long_inference.Stepper` be reused across 96 ICs in-process (rebuilding `init_datetime` / `final_datetime` between calls), or must each IC be a fresh subprocess? Resolved at smoke (§K step 2); fallback (subprocess) already specified.
2. **(B.4)** Exact dim/variable layout of upstream NetCDFs from `long_inference.py:save_prediction` (line 1189-1213). Resolved at smoke.
3. **(B.5)** Does the per-year truth NetCDF (`<year>_gaussian.nc`) carry 10 plev (config) or 13 plev (transfer description) for `zg`? Resolved by `ncdump -h` on `121_gaussian.nc` at first implementation step.
4. **(C.1)** Does our `src/sfno_eval/` climatology reader assume the climatology was built in-pipeline (with a sidecar manifest/JSON) or accept a stand-alone NetCDF? Resolved via a tiny `read_climatology(path, manifest_path=None)` patch in the scorer if needed.
5. **(D.0)** Does our-emulator's existing fixture for `score_nwp.py` regression-test cover all defaulted code paths (lead set, ic_file regex, model label)? Audit in §K step 1; if not, add the fixture before extending.
6. **(F)** Should the cross-emulator report also include a row showing the **group's published 5410 numbers** for triangulation, even though we are not using the group scoring stack as a primary metric? (Out of scope by current decision; flagging as a possible future addition.)
7. **(G.4)** Is the per-timestep h5 `packager_git_sha` attribute present on the group's processed h5s, or is `data_sha7` simply `unknown` for this track? Confirmed at preflight.

---

## K. Estimated implementation order (post-approval, post-climatology-transfer)

| Step | Days | Deliverable |
|---|---|---|
| 0. ✅ **DONE 2026-05-06** — climatology built on Derecho (PBS 3503674, sha7 `feec151b`), transferred via Globus (task `bede3215-49d0-11f1-b97b-0afffe4617ab`, 5.09 GB @ 170 MB/s), sha256 verified on Stampede3. See §3 P-3 + §C.1 pinned constants. | — | `/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/baselines/climatology_proleptic_5410.nc` present, hash-verified. |
| 1. Stampede3 venv delta for PanguWeather/v2.0 deps; smoke-import upstream `long_inference.py` (NOT `inference.py`); audit our-emulator fixture coverage for `score_nwp.py` regression test (§D.0, §J.5) | 0.5 | venv spec at `external/PanguWeather_stampede3_env.txt`; upstream code imports cleanly on h100 node; existing scorer fixture verified or extended. |
| 2. `build_5410_yaml_override.py` (per-Y, single-file symlink shim) + orchestrator argv constructor + `test_yaml_override.py` + `test_runtime_args_5410.py`. **Run §3 P-7 hard gate** (`test_ic_nc_compatibility.py`): probe `plev_data/<Y>_gaussian.nc` first; if it fails, choose contingency (C-A transfer `sigma_data/` from Derecho, or C-B build per-IC NCs from h5) and re-run gate until pass. Then 1-IC NWP smoke via `long_inference.py` (incl. required `--init_nc_filepaths`/`--output_dir`/`--save_basename`/`--final_datetime=Jan-1-(Y+1)`). | 0.5-1.0 (range covers contingency C-A or C-B if needed) | One full-remainder-of-year upstream NetCDF in `<run_root>/inference/upstream_raw/`; `ncdump -h` documented; `long_inference.Stepper` reuse-vs-subprocess decision recorded (§J.1); per-Y shim+yaml correct; **IC-NC source decided and gate green**. |
| 3. `output_adapter.py` + `channel_map.py` + `tests/sfno_inference_5410/`; **run `test_lead_slice_offset.py` to pin `LEAD_SLICE_OFFSET`** (§B.5 step 2); slice first 60 leads from the smoke rollout per the resolved offset; round-trip into scorer-format NetCDF at `<run_root>/inference/nwp/`; verify §B.5 step 6 cross-check (lead-1 vs truth thresholds). | 0.5 | One scorer-format NetCDF with attrs `ic_file/ic_sample_idx/file_anchor/time_plasim_at_ic/model_label`; lead dim == 60; **off-by-one resolved and asserted**; physical-unit sanity (§C.3) passes; `pr_6h` predictions checked with relaxed bounds, truth/clim with strict `≥0`. |
| 4. **Extend `scripts/score_nwp.py`** with `--leads`, `--model-label`, `--ic-file-regex`, `--clim-coord-name`, `--write-climatology-row` (§D.0); add `tests/sfno_eval/test_score_nwp_cli.py` (default = our-emulator regression; 5410-flag invocation = green path); `test_climatology_load_5410.py` (incl. coord-name resolution). Score the smoke output. | 0.5 | 5410 `nwp_scorecard.csv` with finite RMSE & ACC for `model="sfno_5410"` (53 channels), finite RMSE for `model="persistence"` (52 state channels; `pr_6h` row intentionally NaN per §C.2/§D.7), and finite RMSE for `model="climatology"` (53 channels) — RMSE-only for the two baseline classes (no ACC rows). Our-emulator fixture still byte-equal under default flags; pre-flight clim checks pass under both `doy` and `time_of_year` axis names. |
| 5. Full NWP run (96 ICs × K=60) | 1.0 | All 96 NetCDFs at `<run_root>/inference/nwp/`; full scorecard; sanity gate PASS. |
| 6. Phase-1 5410 report (NWP-only) | 0.5 | `results/sfno_eval_5410/<run_tag>/report.md`. |
| 7. Cross-emulator report (NWP-only) | 0.5 | `results/sfno_eval_cross/<cross_run_tag>/report.md`. |
| **Total** | **~3.5 days** | NWP-only Phase 1. Climate-mode and `score_climate.py` deferred to Phase 2 plan. |

Phase 2 (climate-mode rollouts + `score_climate.py` + spectra + full climate scorecard) deferred for both tracks; covered by a separate plan to be drafted after this plan's Phase-1 cross-report lands.

---
