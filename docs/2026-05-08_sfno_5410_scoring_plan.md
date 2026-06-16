# 5410 NWP scoring — minimal scorecard + full pipeline — v4.4

_Plan for review. Date: 2026-05-08. Branch: zgplev-migration-dsi-bootstrap._
_v1 → v2: Codex round-1 fixed 4 blockers (truth source, schema, clim coord, init_state channels)._
_v2 → v3: extended from score-only to full own-track-style pipeline (4-job afterok chain via `submit_eval_5410.sh`)._
_v3 → v3.1: Codex round-2 fixed `file_anchor` regex parseability + `ic_file` strip-then-extract compatibility + lead-1 magnitude bounds._
_v3.1 → v4: Codex round-3 fixed two-root design + conditional skip-mode deps + pr_6h cap + `render_eval_figures.py --track 5410` flag._
_v4 → v4.1: Codex round-4 fixed lead-1 test split + two-root layout consistency + figure smoke `--track 5410` + stale out-of-scope line._
_v4.1 → v4.2: Codex round-5 fixed forecast-error bounds + `init_time` calendar-equivalence + driver refuses non-skip on populated `upstream_raw` + smoke trimmed-plan preflight._
_v4.2 → v4.3: Codex round-6 added subset-mode for `assert_output_dir_complete`, promoted `OUT_ROOT/inference/nwp` rerun safety to driver preflight, fixed `init_time` test math for s ≥ 4, normalized `SCORE_ONLY` alias._
_v4.3 → v4.4: Codex round-7 fixed (a) shell bug — `OUT_ROOT` was used in the rerun-safety check before the block that defines it (would crash under `set -u`); reordered the driver so OUT_ROOT/RUN_TAG/SHAs are computed BEFORE any preflight that references them; (b) adapted-output rerun safety enforced inside `score_5410.py` itself (not just the driver) so direct `python scripts/score_5410.py …` smoke calls also refuse stale `inference/nwp/`; (c) `FORCE=1` semantics tightened — now actively deletes the prior adapted NCs before writing (was previously a soft "bypass"); (d) truth-h5 preflight refined from yearly-counts to per-IC `(Y, s+k)` existence checks for k=0..K._

## Context

The 5410 in-process production sweep (job 3098459) finished cleanly: 96 NetCDFs at `/work2/.../sfno_eval_5410/20260507_phase1_gate/inference/upstream_raw/`, each `time=61` (IC + 60 forecast leads at 6 h). The eval-sfno-5410 skill says scoring SLURMs are TODO.

The user wants to compare 5410 vs the GB=4 own-track emulator (`/scratch/.../sfno_zgplev_full/plasim_sim52_zgplev_full/0`) using the same scorecard schema as `/work2/.../sfno_eval/20260504_eval-8377f46.../report.md` (RMSE + ACC at leads 6h/24h/72h/120h/240h/336h on tas, pr_6h, zg500, ua5, ta5; vs persistence baseline; sanity gate; n=96 ICs).

Goal: minimal scoring SLURM for 5410 producing a `report.md` with comparable numbers, **without unit-converting on top of 5410 conventions** per the user's saved memory.

## What changed in v2 (Codex round-1 fixes)

1. **Wrong truth source.** v1 said use Makani `MOST.0{Y}.h5` (own-track truth, Aug-1/Y+5 anchored). Codex verified live: `Y121_s0000` raw time-0 ≈ Derecho `121_0000.h5` while `MOST.0121.h5` differs by ~72 K on `tas`. Different calendar entirely. **v2 truth source: `/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/h5/sigma_data/{Y}_{ssss:04d}.h5`** — 1460 (non-leap) / 1464 (leap) files per Y, one per 6-hour timestep. Confirmed shape `(64, 128)` per channel, all 8 years (121–128) populated.
2. **Adapter schema mismatch.** v1 wrote `prediction[lead_time, channel, ...]` and a 53-channel `init_state` with cftime `lead_time`. v2 mirrors `src/sfno_inference/nc_writer.py:113-…` exactly: `prediction[init_time=1, lead_time, channel=53, lat, lon]`, `truth[init_time=1, lead_time, channel=53, lat, lon]`, `init_state[init_time=1, channel_ic=52, lat, lon]`, integer-hour `lead_time = [6, 12, …, 360]`, datetime64 `init_time`, plus the full attr set (`ic_file`, `ic_sample_idx`, `ic_global_idx`, `file_anchor`, `time_plasim_at_ic`, `K`, `dt_hours`, `rollout_mode`, …).
3. **Climatology coord name.** 5410 climatology has `time_of_year` dim/coord (366 entries). `score_nwp.py:79` hardcodes `ds["doy"]`. v2 fix: the driver writes a one-line-renamed compatibility climatology to `{out_root}/baselines/climatology_proleptic.nc` (`ds.rename({"time_of_year": "doy"})`), so `score_nwp.py` runs unchanged.
4. **52-channel `init_state` (no `pr_6h`).** `score_nwp.py:173` uses `n_state = init_state.shape[0]` and intentionally NaNs persistence for diagnostics. v2 makes `init_state` exactly 52 channels (state-only, drops `pr_6h`); the 53-channel `channel` coord is kept for prediction/truth where pr_6h is meaningful.

## Inventory — what already exists

| Asset | Path | Schema |
|---|---|---|
| 5410 raw outputs | `/work2/.../upstream_raw/Y{Y}_s{s:04d}_member000_y{Y:04d}.nc` | per-variable: `pl`, `tas`, `pr_6h` (time, lat, lon); `ta`, `ua`, `va`, `hus` (time, lev=10, lat, lon); `zg` (time, plev=10, lat, lon); time=61 |
| 5410 climatology | `/scratch/.../baselines/climatology_proleptic_5410.nc` | dims: `time_of_year=366, hour_quarter=4, channel=53, lat=64, lon=128`. vars: mean, std, n_contributors, channel_units. Channels: pl, tas, ta1–ta10, ua1–ua10, va1–va10, hus1–hus10, zg200..zg1000, pr_6h |
| **Truth (Derecho per-timestep)** | `/scratch/.../sim52/h5/sigma_data/{Y}_{ssss:04d}.h5` | one file per (Y, sample). Top-level group `input` containing `pl(64,128)`, `tas(64,128)`, `pr_6h(64,128)`, `ta_<sigma>(64,128)` × 10, `ua_<sigma>` × 10, `va_<sigma>` × 10, `hus_<sigma>` × 10, `zg_<plev>(64,128)` × 13 (incl. 5000, 10000, 15000 Pa not in climatology), plus surface-condition + forcing channels we ignore. **Verified live: matches 5410 raw time-0 within rounding.** |
| Existing scoring | `scripts/score_nwp.py`, `scripts/_eval_utils.py`, `src/sfno_eval/metrics.py`, `src/sfno_eval/climatology.py` | Reads `inference/nwp/*.nc` w/ `prediction[init_time, lead_time, 53, H, W]`, `truth[…]`, `init_state[init_time, 52, H, W]`, integer-hour `lead_time`, climatology w/ `doy` coord. RMSE lat-weighted, ACC vs clim DOY/HQ bin, persistence (52 state ch only). |
| Reference (GB=4 own-track) | `/work2/.../20260504_eval-8377f46.../report.md` | RMSE/ACC tables for 5 channels × 6 leads. n=96 ICs. |

## Channel mapping — climatology canonical order ↔ 5410 raw ↔ Derecho h5 truth

**Canonical 53-channel order (from `climatology_proleptic_5410.nc::channel`):**

```
[ 0] pl
[ 1] tas
[ 2-11] ta1..ta10           # sigma levels, top→surface
[12-21] ua1..ua10
[22-31] va1..va10
[32-41] hus1..hus10
[42-51] zg200, zg250, zg300, zg400, zg500, zg600, zg700, zg850, zg925, zg1000
[ 52] pr_6h
```

**State channels (channel_ic, 52)** = indices 0..51 (drops index 52 `pr_6h`).

**Sigma index convention:** `ta1` = first sigma level in the climatology. Truth h5 keys are `ta_0.0383`, `ta_0.1191`, …, `ta_0.9833` (top-of-atmosphere first → surface last). 5410 raw NC has `lev = [0.0383, 0.1191, …, 0.9833]` in the same order. **So `ta_k` ↔ raw `ta[:, k-1, :, :]` ↔ truth `input/ta_<sigma_str(k-1)>`.**

**Pressure index convention:** climatology `zgN` ↔ pressure `N hPa = N*100 Pa`. 5410 raw NC has `plev = [20000, 25000, 30000, 40000, 50000, 60000, 70000, 85000, 92500, 100000]`. Truth h5 has 13 zg_<plev> keys; we slice the 10 we need by literal pressure value.

The mapping is computed once from the climatology's `channel` coord at adapter setup, then reused for all 96 ICs.

## Design

### Pipeline shape (mirrors own-track `submit_eval.sh`)

```
scripts/submit_eval_5410.sh   ← single entry point
  │
  ├── sbatch submit_eval_inference_5410.slurm        → JOB_INF (already shipped)
  ├── sbatch --dependency=afterok:JOB_INF submit_eval_score_5410.slurm   → JOB_SCO
  ├── sbatch --dependency=afterok:JOB_SCO submit_eval_report_5410.slurm  → JOB_REP
  └── sbatch --dependency=afterok:JOB_REP submit_eval_figures_5410.slurm → JOB_FIG

Final artifacts on success (v4: two-root design — see §"Top-level driver" below):

  RUN_ROOT  (prepared inference root — INPUT to the chain):
    $RUN_ROOT/inference/ic_source.json
    $RUN_ROOT/inference/ic_nc/{Y}_{ssss:04d}.nc                          (96 IC NCs)
    $RUN_ROOT/inference/SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y{Y}.yaml  (8 yamls)
    $RUN_ROOT/inference/SFNO/5410/checkpoints/ckpt_epoch_50.tar          (ckpt shim)
    $RUN_ROOT/inference/upstream_raw/Y{Y}_s{ssss}_member000_y{Y:04d}.nc  (96 raw — written by JOB_INF)

  OUT_ROOT  (per-RUN_TAG scoring root — created fresh by the driver):
    $OUT_ROOT/inference/nwp/Y{Y}_s{ssss}.nc                              (96 adapted, written by JOB_SCO)
    $OUT_ROOT/baselines/climatology_proleptic.nc                          (compat clim)
    $OUT_ROOT/scores/nwp_scorecard_summary.csv
    $OUT_ROOT/scores/bias_maps_<channel>_<lead>h.npy                      (5×6 = 30)
    $OUT_ROOT/report.md
    $OUT_ROOT/figures/                                                     (7 PNGs)
    $OUT_ROOT/provenance.txt
```

**Read-vs-write convention:** the score adapter reads `$RUN_ROOT/inference/upstream_raw/*.nc` (raw inference outputs) and writes adapted NCs to `$OUT_ROOT/inference/nwp/*.nc`. All scoring/report/figures stages read from and write to `$OUT_ROOT` only.

Stages 1 (inference) is already in production. Stages 2-4 are new — but `score_nwp.py`, `render_eval_report.py`, `render_eval_figures.py` are reusable as-is.

### Files to add / change

**New files (zero changes to existing code except the figures-script patch below):**

| File | Lines (est.) | Role |
|---|---:|---|
| `src/sfno_inference_5410/score_adapter.py` | ~280 | Adapter: 5410 raw NC + Derecho per-timestep truth + IC state → score_nwp inference NC (canonical schema) |
| `src/sfno_inference_5410/score_climatology_compat.py` | ~30 | `write_compat_clim(src, dst)` — rename `time_of_year → doy`. |
| `scripts/score_5410.py` | ~120 | Score driver: compat clim + adapt 96 ICs + invoke `score_nwp.main()`. |
| `scripts/submit_eval_score_5410.slurm` | ~50 | Score SLURM. |
| `scripts/submit_eval_report_5410.slurm` | ~40 | Report SLURM. Invokes existing `render_eval_report.py` with 5410-flavored provenance strings. |
| `scripts/submit_eval_figures_5410.slurm` | ~30 | Figures SLURM. Invokes patched `render_eval_figures.py --track 5410`. |
| `scripts/submit_eval_5410.sh` | ~180 | **Top-level driver.** Mirrors `submit_eval.sh`: separate `RUN_ROOT` (prepared inference root) vs `OUT_ROOT` (per-eval scoring root); precondition checks; conditional 4-job afterok chain. |
| `tests/sfno_inference_5410/test_score_adapter.py` | ~280 | Schema + truth-alignment + clim compat + magnitude-bound tests. |
| `tests/sfno_inference_5410/test_pipeline_chain.py` | ~120 | Mocks sbatch; verifies driver computes correct deps under all 16 (SKIP_INF, SKIP_SCO, SKIP_REP, SKIP_FIG) combos + RUN_ROOT preconditions + provenance.txt content. |

**Existing files modified (round-3 fix d):**

| File | Patch | Reason |
|---|---|---|
| `scripts/render_eval_figures.py` | + `--track {own,5410}` argparse flag (default `own`); when `track=='5410'`, set `CHANNEL_UNIT_SCALE = {}` (empty) and `CHANNEL_UNITS["pr_6h"] = "kg m$^{-2}$ (6h)"`. ~15 lines added, no removed lines. Backward compat: own-track callers don't pass `--track` and get the existing `m s^-1 → mm day^-1` scaling. | Codex round-3 major: 5410 `pr_6h` is "rate × 6h" in kg/m² (max ≈ 0.013 live), NOT m/s. The hard-coded `86400*1000` scaling at `render_eval_figures.py:56` would mislabel + mis-scale the bias map. The user's saved memory explicitly forbids unit-converting on top of 5410 outputs. |

### Adapter surface (`score_adapter.py`)

```python
def adapt_5410_ic_to_score_nwp(
    *,
    raw_nc_path: Path,            # one of the 96 raw 5410 outputs
    truth_h5_dir: Path,           # /scratch/.../sigma_data/
    Y: int, s: int,               # IC identifiers
    K: int,                       # forecast leads
    channel_names: list[str],     # 53-channel order from clim NC
    out_nc_path: Path,            # where to write the adapted NC
    ckpt_path: Path,              # for attrs
    eval_sha7: str, data_sha7: str, train_sha7: str, run_tag: str,
) -> None:
    """Convert one 5410 raw NetCDF + corresponding Derecho per-timestep
    truth + IC state into the inference-NetCDF schema score_nwp.py
    consumes (mirrors src/sfno_inference/nc_writer.py:113-...).

    Output schema (verified against nc_writer.py 2026-05-08):
      data_vars:
        prediction(init_time=1, lead_time=K, channel=53, lat, lon)
        truth     (init_time=1, lead_time=K, channel=53, lat, lon)
        init_state(init_time=1, channel_ic=52, lat, lon)
      coords:
        init_time = [datetime64(year=Y, month=1, day=1, hour=0)
                     + timedelta(hours=s*6)]
        lead_time = [6, 12, ..., K*6]   # int hours, NOT cftime
        channel = list of 53 names (incl pr_6h at idx 52)
        channel_ic = list of 52 names (channel[:52], drops pr_6h)
        lat, lon = same as raw NC
      attrs:
        ckpt_path, eval_sha7, data_sha7, train_sha7, run_tag,
        # ic_file must follow the XXXX.h5 pattern that
        # score_nwp.py:139 strips MOST. + .h5 from. We pass just the
        # year so ic_year extraction yields "0121" (matches own-track
        # behavior where ic_year='0121' from "MOST.0121.h5"). The
        # actual per-timestep truth file is stashed in truth_h5_file.
        ic_file = f"{Y:04d}.h5",
        truth_h5_file = f"{Y}_{s:04d}.h5",   # provenance only
        ic_sample_idx = s,
        ic_global_idx = s,
        # file_anchor MUST be a parseable YYYY-MM-DD HH:MM:SS string
        # (score_nwp.py:92 regex). Anchored at Y-Jan-1; the
        # per-IC offset rides on time_plasim_at_ic (in days).
        # _date_for_lead does base + timedelta(days=t_pic) +
        # timedelta(hours=lead). Math:
        #   For Y=121 s=122: base = cftime(121,1,1,0); t_pic=30.5d
        #   (122 * 6h / 24h); base + 30.5d = 121-01-31 12:00. ✓
        file_anchor = f"{Y:04d}-01-01 00:00:00",
        time_plasim_at_ic = s * 0.25,        # days; 6h-step → 0.25 d
        rollout_mode = "nwp",
        K = K, dt_hours = 6
    """
```

Implementation outline (~280 lines; helpers below):

1. `_canonical_channel_map(channel_names)` — once per driver run, builds `{name: ("raw"|"truth_h5", lookup_args)}` for each of the 53 names. Returns the map + the 52 state-channel subset.
2. `_read_raw_prediction(raw_nc_path, channel_map, K)` — opens 5410 raw NC, builds `prediction[K, 53, H, W]` by per-channel lookup. Slices `time[1:K+1]` (drops the IC step at index 0).
3. `_read_truth_for_leads(truth_h5_dir, Y, s, K, channel_map)` — for k=1..K, opens `{truth_h5_dir}/{Y}_{s+k:04d}.h5`, reads each of the 53 channels via `channel_map`, stacks into `truth[K, 53, H, W]`. Handles year rollovers correctly (s+k might exceed 1459/1463 → wraps to next Y, but with K=60 and last s=1342, s+K=1402 < 1459, so no rollover).
4. `_read_ic_state(truth_h5_dir, Y, s, channel_map_state)` — opens `{Y}_{s:04d}.h5`, reads 52 state channels, returns `init_state[52, H, W]`.
5. `_pack_dataset(...)` — writes the xarray Dataset with the canonical schema + attrs.

### Compat climatology helper (`score_climatology_compat.py`)

```python
def write_compat_clim(src: Path, dst: Path) -> None:
    """Rename time_of_year → doy in the 5410 climatology NetCDF and
    write to dst (so score_nwp.py:79 ds["doy"] works unchanged).
    Also passes through n_contributors and channel coords."""
    import xarray as xr
    ds = xr.open_dataset(src)
    if "time_of_year" not in ds.dims:
        # Already a doy-form clim; just symlink/copy.
        ds.close()
        if dst != src:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src.resolve())
        return
    out = ds.rename({"time_of_year": "doy"})
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(dst)
    ds.close()
```

### Driver (`score_5410.py`)

```python
def main():
    args = parse([
        "--run-root",   # 5410 production run-root (RUN_ROOT, raw inputs)
        "--K",          # 60
        "--years",      # default 121..128
        "--ic-subset",  # explicit Y:s,Y:s,... for cross-year smoke (mirrors orchestrator)
        "--limit-ics",  # smoke knob (1 IC for fast iteration)
        "--truth-h5-dir",  # default /scratch/.../sim52/h5/sigma_data
        "--clim-src",   # default /scratch/.../baselines/climatology_proleptic_5410.nc
        "--out-root",   # OUT_ROOT (scoring outputs)
        "--run-tag",    # required for provenance
        "--force",      # if --out-root/inference/nwp/ is non-empty, delete + rebuild
                        # (Codex round-7 fix #2/#3: enforced INSIDE score_5410.py,
                        # not just the driver, so direct invocations are safe.)
    ])
    # 1. Preflight (in order):
    #    1a. assert_clim_src_present(clim_src)
    #    1b. assert_clim_channels_canonical(clim_src)
    #    1c. assert_truth_h5_for_plan(truth_h5_dir, plan, K)   # per-IC, per-lead
    #    1d. assert_raw_outputs_complete(run_root/inference/upstream_raw, plan, K,
    #                                    mode='exact' if full else 'subset')
    #    1e. assert_adapted_dir_empty_or_force(out_root, force=args.force)
    # 2. Build compat climatology at out_root/baselines/climatology_proleptic.nc.
    # 3. Read 53 channel names from the clim.
    # 4. For each (Y, s) in plan:
    #      adapt_5410_ic_to_score_nwp(...) → out_root/inference/nwp/{stem}.nc
    # 5. Invoke score_nwp.main() with sys.argv set to:
    #      --out-root <out_root> --clim-nc <out_root>/baselines/climatology_proleptic.nc
    #    score_nwp produces scorecard CSV + bias maps + sanity gate.
    # 6. (Report rendering moved to submit_eval_report_5410.slurm, which calls
    #    render_eval_report.py. score_5410.py stops at the scorecard CSV.)
```

### Score SLURM (`submit_eval_score_5410.slurm`)

Patterns after `scripts/submit_eval_score.slurm` (own-track scoring SLURM): 1 h budget on h100 (purely for queue parity — scoring is CPU-bound, but we have the allocation), invokes `python scripts/score_5410.py …` with the right paths.

### Report SLURM (`submit_eval_report_5410.slurm`) — reuses own-track renderer

Mirrors `scripts/submit_eval_report.slurm` (~40 lines). 30 min budget on h100 (or skx-dev — we move it to skx-dev for queue parity since the report is pure CPU work):

```bash
python scripts/render_eval_report.py \
    --out-root  "$OUT_ROOT" \
    --run-tag   "$RUN_TAG" \
    --eval-sha7 "$EVAL_SHA7" \
    --data-sha7 "$GROUP_SHA7" \      # 5410 group source label, e.g. "5410-v2.0"
    --train-sha7 "$MODEL_SHA7" \     # 5410 ckpt label, e.g. "ckpt_epoch_50"
    --ckpt-path "$CKPT"              # /work2/.../v2.0/results/SFNO/5410/checkpoints/ckpt_epoch_50.tar
```

`render_eval_report.py` accepts these as plain strings — no model-specific knowledge. The output `report.md` header will say:

```
**Run tag:** `20260508_eval-<EVAL_SHA7>_5410-v2.0_ckpt-50`
| Eval code SHA | `<EVAL_SHA7>` |
| Data packager SHA | `5410-v2.0` |
| Training code SHA | `ckpt_epoch_50` |
| Checkpoint | `/work2/.../v2.0/results/SFNO/5410/checkpoints/ckpt_epoch_50.tar` |
```

The label semantics differ from the own-track ("Data packager SHA" actually points at the 5410 group source release tag), but the values are honest about what they reference and the report is otherwise structurally identical to the own-track report — which is what makes head-to-head comparison straightforward.

**Optional v3.x follow-up (not for now):** add a `--track {own,5410}` flag to `render_eval_report.py` that swaps the header field labels. Out of scope for this plan; the literal strings above are sufficient for the comparison the user asked for.

### Figures SLURM (`submit_eval_figures_5410.slurm`) — patched renderer + 5410 track flag

Mirrors `scripts/submit_eval_figures.slurm` exactly. 30 min on **skx-dev** (CPU-only — explicitly avoids tying up an h100 slot per the own-track precedent):

```bash
python scripts/render_eval_figures.py --out-root "$OUT_ROOT" --track 5410
```

**`render_eval_figures.py` patch (Codex round-3 major fix #4):**

`render_eval_figures.py:56` hard-codes `CHANNEL_UNIT_SCALE = {"pr_6h": 86400.0 * 1000.0}` to convert own-track's `pr_6h` (m/s rate) to mm/day for axis labels. **5410's `pr_6h` is "rate × 6h" in kg/m² (max ≈ 1.3e-2 live)**, NOT m/s. Applying the same scaling would give axis values ≈ 1e6 mm/day (six orders of magnitude wrong) and mislabel the bias map.

The user's saved memory explicitly forbids unit-converting on top of 5410 outputs. Minimum patch:

```diff
+ p.add_argument("--track", choices=("own", "5410"), default="own",
+                help="Emulator track. 5410 disables the m/s→mm/day "
+                     "scaling for pr_6h (5410 outputs are already in "
+                     "kg m^-2 per 6h accumulation; do not convert).")
  ...
  args = p.parse_args()
+ if args.track == "5410":
+     CHANNEL_UNIT_SCALE.clear()                            # disable all scaling
+     CHANNEL_UNITS["pr_6h"] = "kg m$^{-2}$ (6h accum.)"    # honest label
```

~15 lines added, no removed lines. Default `--track own` preserves existing behavior; own-track callers don't need to change anything.

**Tested by `test_pipeline_chain.py::test_render_figures_track_flag`** — invokes `render_eval_figures.py --track 5410 --help` and confirms the flag is exposed; runs against a stub OUT_ROOT with 5410 conventions and confirms the output bias_pr_6h.png label string is "kg m$^{-2}$ (6h accum.)" not "mm day$^{-1}$".

`render_eval_figures.py` reads `out_root/scores/nwp_scorecard_summary.csv` + `out_root/scores/bias_maps_*.npy` and writes 7 PNGs to `out_root/figures/` (rmse_vs_lead.png, acc_vs_lead.png, bias_<channel>.png × 5). It hard-codes `REPORT_CHANNELS = ["tas", "pr_6h", "zg500", "ua5", "ta5"]` and `REPORT_LEADS = [6, 24, 72, 120, 240, 336]` — both match what `score_nwp.py` writes for 5410.

### Top-level driver (`scripts/submit_eval_5410.sh`)

**Two roots, separate concerns** (Codex round-3 fix #1):

| Variable | Role | Created by |
|---|---|---|
| `RUN_ROOT` | The **prepared inference root**. Contains `inference/ic_source.json`, the 96 IC NCs at `inference/ic_nc/{Y}_{ssss}.nc`, the 8 per-Y yamls, the ckpt symlink shim, and (after inference runs) `inference/upstream_raw/`. Same convention as `submit_eval_inference_5410.slurm`'s preconditions. | Manual prep with `scripts/build_5410_yaml_override.py --all-years --K 60` + IC NC builder + `ic_source.json` setup. (A future `scripts/prepare_5410_run_root.py` would consolidate this — explicitly out of scope for v4.) |
| `OUT_ROOT` | The **per-eval scoring root**. Contains `inference/nwp/` (adapted NCs), `baselines/climatology_proleptic.nc` (compat clim), `scores/`, `report.md`, `figures/`, `provenance.txt`. Fresh per `RUN_TAG`. | Driver creates this. |

This diverges from the own-track convention (where `OUT_ROOT == inference root`) because the 5410 inference SLURM was built with explicit RUN_ROOT preconditions. Document this divergence in the driver header.

**Driver structure:**

```bash
set -euo pipefail
cd "$HOME/projects/SFNO_Climate_Emulator"

# === required env (defaults shown) ===
: "${UPSTREAM_REPO:=/work2/.../v2.0}"
: "${RUN_ROOT:=/work2/.../sfno_eval_5410/20260507_phase1_gate}"   # MUST be prepared
: "${CKPT:=$UPSTREAM_REPO/results/SFNO/5410/checkpoints/ckpt_epoch_50.tar}"
: "${TRUTH_H5_DIR:=/scratch/.../sim52/h5/sigma_data}"
: "${CLIM_SRC:=/scratch/.../sim52/baselines/climatology_proleptic_5410.nc}"
: "${K:=60}"

# === SCORE_ONLY alias normalization (Codex round-6 fix #4) ===
# SCORE_ONLY=1 is shorthand for "skip everything except score". Apply
# the alias HERE so all downstream skip-logic sees the canonical flags.
if [[ "${SCORE_ONLY:-0}" == "1" ]]; then
    SKIP_INF=1
    SKIP_REP=1
    SKIP_FIG=1
fi

# === Compute SHAs / RUN_TAG / OUT_ROOT FIRST (Codex round-7 fix #1 — ===
# === these must be defined before any preflight that references them) ===
EVAL_SHA7=$(git rev-parse --short=7 HEAD)
GROUP_SHA7="$(git -C "$UPSTREAM_REPO" rev-parse --short=7 HEAD 2>/dev/null || echo 5410-v2.0)"
MODEL_SHA7="$(basename "$CKPT" .tar)"
DATE_STR=$(date +%Y%m%d)
: "${RUN_TAG:=${DATE_STR}_eval-${EVAL_SHA7}_5410-${GROUP_SHA7}_${MODEL_SHA7}}"
: "${OUT_ROOT:=$WORK2/SFNO_Climate_Emulator/results/sfno_eval_5410/$RUN_TAG}"
mkdir -p "$OUT_ROOT" logs

# === RUN_ROOT precondition check (skip if SKIP_INF=1) ===
if [[ "${SKIP_INF:-0}" != "1" ]]; then
    # Mirror the inference SLURM's preconditions so we fail FAST at submit
    # time instead of waiting for SLURM to schedule the job.
    test -f "$RUN_ROOT/inference/ic_source.json" \
        || { echo "FATAL: $RUN_ROOT/inference/ic_source.json missing — RUN_ROOT not prepared" >&2; exit 2; }
    test -L "$RUN_ROOT/inference/SFNO/5410/checkpoints/ckpt_epoch_50.tar" \
        || { echo "FATAL: ckpt symlink shim missing under $RUN_ROOT" >&2; exit 2; }
    for Y in 121 122 123 124 125 126 127 128; do
        test -f "$RUN_ROOT/inference/SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y${Y}.yaml" \
            || { echo "FATAL: per-Y yaml missing for Y=$Y under $RUN_ROOT" >&2; exit 2; }
    done
    # IC NC presence check happens inside submit_eval_inference_5410.slurm.

    # Codex round-5 fix #3: refuse to run inference into a populated
    # upstream_raw. The orchestrator's assert_output_dir_empty would
    # eventually catch this, but failing in the driver pre-submit gives
    # the user an actionable error before any compute is consumed.
    if [[ -d "$RUN_ROOT/inference/upstream_raw" ]]; then
        n_nc=$(ls "$RUN_ROOT/inference/upstream_raw/"Y*_member*_y*.nc 2>/dev/null | wc -l)
        if [[ "$n_nc" -gt 0 ]]; then
            echo "FATAL: $RUN_ROOT/inference/upstream_raw is non-empty ($n_nc prior NetCDFs)." >&2
            echo "  Either:" >&2
            echo "    (a) Pass SKIP_INF=1 to score against the existing outputs." >&2
            echo "    (b) Backup/delete the existing upstream_raw before launch." >&2
            echo "    (c) Use a fresh RUN_ROOT." >&2
            exit 2
        fi
    fi
else
    # SKIP_INF: must already have 96 prior NetCDFs in upstream_raw.
    n_nc=$(ls "$RUN_ROOT/inference/upstream_raw/"Y*_member*_y*.nc 2>/dev/null | wc -l)
    [[ "$n_nc" -eq 96 ]] \
        || { echo "FATAL: SKIP_INF=1 but $RUN_ROOT/inference/upstream_raw has $n_nc files (expect 96)" >&2; exit 2; }
fi

# === OUT_ROOT/inference/nwp rerun safety (Codex round-6 fix #2 + ===
# === round-7 fix #3 — FORCE=1 now actively deletes prior adapted ===
# === NCs instead of soft-bypassing). Note OUT_ROOT is now defined ===
# === above this block. ===
# score_nwp.py reads ALL *.nc under OUT_ROOT/inference/nwp/ as ICs;
# stale adapted NCs from a prior run would silently contaminate the
# scorecard. Refuse non-empty unless FORCE=1; with FORCE=1 we DELETE
# the prior set, which guarantees no extras can survive into the
# new scorecard.
if [[ "${SKIP_SCO:-0}" != "1" ]]; then
    if [[ -d "$OUT_ROOT/inference/nwp" ]]; then
        n_adapted=$(ls "$OUT_ROOT/inference/nwp/"*.nc 2>/dev/null | wc -l)
        if [[ "$n_adapted" -gt 0 ]]; then
            if [[ "${FORCE:-0}" == "1" ]]; then
                echo "[driver] FORCE=1: deleting $n_adapted prior adapted NCs at $OUT_ROOT/inference/nwp/"
                rm -f "$OUT_ROOT/inference/nwp/"*.nc
            else
                echo "FATAL: $OUT_ROOT/inference/nwp is non-empty ($n_adapted prior adapted NCs)." >&2
                echo "  score_nwp.py would silently include these in the scorecard." >&2
                echo "  Either:" >&2
                echo "    (a) Pass FORCE=1 to delete the prior adapted set and rebuild from raw." >&2
                echo "    (b) Delete \$OUT_ROOT/inference/nwp/*.nc manually." >&2
                echo "    (c) Use a fresh OUT_ROOT (different RUN_TAG)." >&2
                exit 2
            fi
        fi
    fi
fi

cat > "$OUT_ROOT/provenance.txt" <<EOF
RUN_TAG=$RUN_TAG
RUN_ROOT=$RUN_ROOT          # prepared inference root (input)
OUT_ROOT=$OUT_ROOT          # scoring outputs (output)
EVAL_SHA7=$EVAL_SHA7
GROUP_SHA7=$GROUP_SHA7
MODEL_SHA7=$MODEL_SHA7
CKPT=$CKPT
UPSTREAM_REPO=$UPSTREAM_REPO
TRUTH_H5_DIR=$TRUTH_H5_DIR
CLIM_SRC=$CLIM_SRC
K=$K
SKIP_INF=${SKIP_INF:-0}
SKIP_SCO=${SKIP_SCO:-0}
SKIP_REP=${SKIP_REP:-0}
SKIP_FIG=${SKIP_FIG:-0}
DATE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

export RUN_ROOT OUT_ROOT EVAL_SHA7 GROUP_SHA7 MODEL_SHA7 RUN_TAG \
       CKPT TRUTH_H5_DIR CLIM_SRC K UPSTREAM_REPO

# === conditional afterok chain (Codex round-3 fix #2) ===
# Each subsequent stage's dep is the LAST job we actually submitted —
# regardless of which earlier stages were skipped. This works for any
# subset of {SKIP_INF, SKIP_SCO, SKIP_REP, SKIP_FIG}.
prev_job=""

submit_with_dep() {
    local slurm="$1"
    local args=()
    if [[ -n "$prev_job" ]]; then
        args+=("--dependency=afterok:$prev_job")
    fi
    args+=(--parsable "$slurm")
    submit_sbatch "${args[@]}"
}

if [[ "${SKIP_INF:-0}" != "1" ]]; then
    # BLOCKER_JOB_ID lets us chain after a non-pipeline job (e.g.
    # late-arriving training run that produces a new ckpt).
    if [[ -n "${BLOCKER_JOB_ID:-}" ]]; then
        prev_job="$BLOCKER_JOB_ID"
    fi
    JOB_INF=$(submit_with_dep scripts/submit_eval_inference_5410.slurm)
    echo "[submit_eval_5410] inference job: $JOB_INF (deps: ${prev_job:-none})"
    prev_job="$JOB_INF"
fi

if [[ "${SKIP_SCO:-0}" != "1" ]]; then
    JOB_SCO=$(submit_with_dep scripts/submit_eval_score_5410.slurm)
    echo "[submit_eval_5410] scoring   job: $JOB_SCO (deps: ${prev_job:-none})"
    prev_job="$JOB_SCO"
fi

if [[ "${SKIP_REP:-0}" != "1" ]]; then
    JOB_REP=$(submit_with_dep scripts/submit_eval_report_5410.slurm)
    echo "[submit_eval_5410] report    job: $JOB_REP (deps: ${prev_job:-none})"
    prev_job="$JOB_REP"
fi

if [[ "${SKIP_FIG:-0}" != "1" ]]; then
    JOB_FIG=$(submit_with_dep scripts/submit_eval_figures_5410.slurm)
    echo "[submit_eval_5410] figures   job: $JOB_FIG (deps: ${prev_job:-none})"
fi

echo "Final artifacts on success: $OUT_ROOT/report.md, $OUT_ROOT/figures/"
```

**Skip-mode behavior (verified by `test_pipeline_chain.py` for all 16 combos):**

| Mode | Deps actually emitted |
|---|---|
| (no skip) | INF; SCO afterok:INF; REP afterok:SCO; FIG afterok:REP |
| `SKIP_INF=1` | SCO (no dep); REP afterok:SCO; FIG afterok:REP |
| `SKIP_SCO=1` | INF; REP afterok:INF; FIG afterok:REP |
| `SKIP_INF=1 SKIP_SCO=1` | REP (no dep); FIG afterok:REP |
| `SKIP_INF=1 SKIP_REP=1 SKIP_FIG=1` (= `SCORE_ONLY`) | SCO (no dep); nothing else |
| ... | (16 total combinations, test enumerates each) |

`BLOCKER_JOB_ID=NNN` is honored only when inference is NOT skipped (chains the inference job after some external SLURM).

**For the user's immediate ask** (compare 5410 vs GB=4 emulator without re-running inference):

```bash
SKIP_INF=1 \
RUN_ROOT=/work2/.../sfno_eval_5410/20260507_phase1_gate \
RUN_TAG=20260508_5410-vs-gb4 \
./scripts/submit_eval_5410.sh
```

The 96 raw NetCDFs from job 3098459 already live at `$RUN_ROOT/inference/upstream_raw/`; SKIP_INF skips the inference SLURM and runs only score → report → figures. **End-to-end ~1 h.**

## Tests (Codex round-1 recommendation: prove alignment, not just shapes)

`tests/sfno_inference_5410/test_score_adapter.py`. Each test gated on the upstream + truth + raw NetCDF + climatology being present (skip cleanly otherwise).

**Schema correctness:**
- `test_canonical_channel_order`: `_canonical_channel_map(climatology_channels)` returns exactly 53 entries in the order pl, tas, ta1..ta10, ua1..ua10, va1..va10, hus1..hus10, zg200..zg1000, pr_6h. `channel_ic` is a 52-element prefix (drops pr_6h).
- `test_adapt_one_ic_shapes`: run `adapt_5410_ic_to_score_nwp` on `Y121_s0000_*.nc`. Open the adapted NC and assert `prediction.shape == (1, 60, 53, 64, 128)`, `truth.shape == (1, 60, 53, 64, 128)`, `init_state.shape == (1, 52, 64, 128)`, `lead_time.dtype == int64`, `lead_time.values.tolist() == [6, 12, ..., 360]`. **`init_time` calendar-equivalence check** (Codex round-5 fix #2 — xarray decodes pre-1582 dates as `cftime.DatetimeProlepticGregorian` object-dtype not `datetime64`, so kind `O` not `M`; round-6 fix #3 — full cftime arithmetic, not `(s*6)%24` which is wrong for s ≥ 4 because day rolls over):
  ```python
  import cftime
  from datetime import timedelta
  expected = cftime.DatetimeProlepticGregorian(Y, 1, 1, 0, has_year_zero=True) + timedelta(hours=6 * s)
  v = ds.init_time.values[0]
  if hasattr(v, 'year'):       # cftime path (object-dtype after decode)
      y, mo, d, h = v.year, v.month, v.day, v.hour
  else:                        # numpy datetime64 path (post-1582 dates)
      import pandas as pd
      ts = pd.Timestamp(v)
      y, mo, d, h = ts.year, ts.month, ts.day, ts.hour
  assert (y, mo, d, h) == (expected.year, expected.month, expected.day, expected.hour), (
      f"init_time round-trip mismatch: got ({y},{mo},{d},{h}), "
      f"expected ({expected.year},{expected.month},{expected.day},{expected.hour}) "
      f"for Y={Y} s={s}"
  )
  ```
  Run for several (Y, s) tuples that exercise day/month rollover: `(121, 0)`, `(121, 4)` (Jan 2 00:00), `(121, 124)` (Feb 1 00:00), `(125, 1342)` (~Dec 1 12:00 of year 125).
  Same defensive shape works for the `_date_for_lead` cross-check in `test_file_anchor_parses`.
- `test_attrs_match_nc_writer_contract`: every attr `score_nwp.py` reads (`ic_file`, `ic_sample_idx`, `file_anchor`, `time_plasim_at_ic`, `ckpt_path`, `eval_sha7`, `data_sha7`, `train_sha7`, `run_tag`, `K`, `dt_hours`, `rollout_mode`) is present and has the correct type.
- `test_file_anchor_parses`: for a sample of ICs (Y=121 s=0; Y=121 s=122; Y=125 s=1342; Y=128 s=976), the adapted NC's `file_anchor` matches `score_nwp.py:92`'s regex `r"(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})"` AND `_date_for_lead(file_anchor, time_plasim_at_ic, lead=6h)` returns `(month, day, hour)` matching `init_datetime + 6h` computed from raw cftime arithmetic. Codex round-2: this is the load-bearing test for the ACC climatology lookup — without it the verification calendar is silently wrong.
- `test_ic_file_extracts_clean_year`: `ic_file.replace("MOST.", "").replace(".h5", "")` (per `score_nwp.py:139`) yields a clean 4-digit year `"0121"` — not `"121_0000"`. The per-timestep truth filename is in a separate `truth_h5_file` provenance attr.

**Truth-alignment (Codex's specific tests):**
- `test_raw_time0_equals_ic_h5`: for IC Y=121 s=0, open the raw 5410 NC and read `tas` at `time=0` (the IC step). Open `121_0000.h5` and read `input/tas`. **Assert max-abs diff == 0** (or below fp32 eps). This is the smoking-gun test that the truth source matches the IC source.
- `test_truth_at_lead_6h_equals_h5_at_s_plus_1` (was misnamed v3.1 — Codex round-4 fix #1; raw `time=1` is the **forecast**, NOT truth): the **adapted `truth[lead_time=6h]`** equals the contents of `121_0001.h5/input/<channel>` bit-exact for all 52 state channels (pl, tas, ta1..ta10, ua1..ua10, va1..va10, hus1..hus10, zg200..zg1000) AND the diagnostic pr_6h. Each channel's max-abs-diff against the h5 read must be exactly 0.0.
- `test_truth_magnitude_bounds_at_lead_6h` (Codex round-2 strengthening — catches wrong-channel/wrong-unit/wrong-pressure-level mistakes that bit-exact alone wouldn't): on the same `truth.sel(lead_time=6h)`:
     * `tas` mean ∈ [240 K, 310 K], min ≥ 180 K, max ≤ 340 K.
     * `zg500` mean ∈ [5400 m, 5800 m], min ≥ 4800 m, max ≤ 6200 m.
     * `pl` mean ∈ [11.4, 11.6] (ln(p_s) ≈ ln(95000)–ln(102000)).
     * `pr_6h` ≥ 0 everywhere; **mean ∈ [1e-5, 5e-3] kg m⁻²**; **99th percentile ≤ 0.02 kg m⁻²**; **max ≤ 0.05 kg m⁻²**. (Live spot-check on `121_0001.h5/input/pr_6h`: min=0, mean=5.76e-4, p99=5.98e-3, max=1.27e-2 — well inside.)
     * `ua5` |mean| ≤ 50 m/s.
- `test_prediction_at_lead_6h_equals_raw_time1` (NEW — Codex round-4 fix #1 splits the misnamed v3.1 test into two halves): the **adapted `prediction[lead_time=6h]`** comes from the 5410 raw NetCDF's `time=1` slice (the first model forecast step), not the IC step (`time=0`). Test the channel-flatten remap is correct:
  1. **Per-channel match against raw**: for `tas` (a flat-2D channel), `prediction.sel(channel='tas', lead_time=6h) == raw_nc.tas.isel(time=1)` bit-exact. For `ta5` (a sigma-indexed channel), `prediction.sel(channel='ta5', lead_time=6h) == raw_nc.ta.isel(time=1, lev=4)` bit-exact (sigma index = climatology channel suffix - 1; ta5 → lev=4 = ~0.43σ). For `zg500` (a plev-keyed channel), `prediction.sel(channel='zg500', lead_time=6h) == raw_nc.zg.isel(time=1).sel(plev=50000)` bit-exact (plev selection by literal Pa value, NOT positional index).
  2. **Differs from truth (model is doing something)**: on `(prediction[lead_time=6h] - truth[lead_time=6h])`, the per-channel max-abs > 0 for `tas` (typical 6h forecast error ≥ 0.1 K), for `zg500` (≥ 1 m), for `ua5` (≥ 0.1 m/s). pr_6h prediction may differ in mean even if instantaneous max-abs is small — assert max-abs > 0 only.
  3. **Order-of-magnitude sanity bounds (catches accidental ×1000 scaling)** on `(prediction - truth)` per channel. Bounds calibrated against live data on Y121 s=0 (Codex round-5 fix #1; v4.1's `tas≤5 K`/`zg500≤50 m` caps were too tight — live values were 8.9 K and 87 m). Uses **three nested bounds** (mean / p99 / max) so a single hot grid point can't fail the test:
     | Channel | mean(\|err\|) | p99(\|err\|) | max(\|err\|) | Live (Y121 s=0) |
     |---|---|---|---|---|
     | `tas` | ≤ 3 K | ≤ 8 K | ≤ 20 K | mean=1.42, p99=6.49, max=8.90 |
     | `zg500` | ≤ 30 m | ≤ 80 m | ≤ 200 m | mean=19.5, p99=66.4, max=87.1 |
     | `ua5` | ≤ 3 m/s | ≤ 8 m/s | ≤ 20 m/s | mean=1.50, p99=5.29, max=6.78 |
     A ×1000 unit mistake would push max ≈ 8000+ for tas / 87000+ for zg500 — well above the max bounds. The mean+p99 bounds catch subtler systematic biases that a max-only bound can't (a single contaminated grid point inflates max but not mean).
- `test_init_state_matches_h5_sample_s_state_only`: for IC Y=121 s=0, the adapted NC's `init_state[0, c, :, :]` for each c=0..51 matches `121_0000.h5` reading. `init_state` has no pr_6h channel.
- `test_pr_6h_in_truth_but_not_init_state`: `truth[0, :, 52, :, :]` (the pr_6h slice) is finite (read from `input/pr_6h` in the truth h5). `init_state` has no channel-52 entry; `channel_ic` length is 52.
- `test_zg_plev_slicing_uses_sel_not_position`: open the adapted NC for Y=121 s=0; verify `truth[0, 0, channel_idx_zg500, :, :]` matches `121_0001.h5`'s `input/zg_50000.0`. Specifically test that we're not relying on the order of `plev` in the raw NC.

**Climatology compat:**
- `test_compat_clim_has_doy_dim`: run `write_compat_clim(...)`; open the result; assert `"doy" in ds.dims and ds.sizes["doy"] == 366`; `n_contributors` and `channel` coords still present.
- `test_compat_clim_idempotent_if_already_doy`: pass a clim with `doy` already as the dim; expect a symlink (or copy), no error.

**End-to-end smoke (slow, gated by RUN_SCORE_AB=1):**
- `test_score_nwp_consumes_adapted_nc`: `adapt_5410_ic_to_score_nwp(Y=121, s=0)` → adapted NC. Then call `score_nwp.main()` via subprocess with `--out-root` pointing at the test run-root. Confirm `scorecard_summary.csv` exists, contains rows for `tas` channel × leads 6h/24h/.../336h × {emulator, persistence}, and that `tas`@6h emulator RMSE is finite + within the same order of magnitude as the GB=4 number (~0.246 K). No exit-1 from the sanity gate.

## Preflights (in `score_5410.py main()` before any IC is adapted)

1. **`assert_clim_src_present(clim_src)`**: file exists, has `time_of_year` or `doy` dim of length 366, `channel` length 53.
2. **`assert_clim_channels_canonical(clim_src)`**: channel coord matches the canonical 53-name list (pl, tas, ta1..10, ua1..10, va1..10, hus1..10, zg200..zg1000, pr_6h). zg500 is at the expected index.
3. **`assert_truth_h5_for_plan(truth_dir, plan, K)`** (Codex round-7 fix #4 — was a yearly-count check): for **each** `(Y, s)` in the plan, assert each of `{Y}_{s:04d}.h5` (IC) and `{Y}_{s+k:04d}.h5` for k=1..K (forecast-target truths) exists. Catches missing/holey truth coverage at submit time. Sample-checks one file per Y for `input/tas` shape `(64, 128)`.
4. **`assert_raw_outputs_complete(upstream_raw, plan, K, mode)`** (re-uses existing `assert_output_dir_complete` from `preflight.py` with the v4.3 `mode` parameter): for the full plan, `mode='exact'` (96 raw NCs, no extras); for trimmed plans (`--limit-ics`/`--ic-subset`), `mode='subset'` (the trimmed-plan files present + well-formed; allows extras from prior production sweeps to coexist). Each expected file has `time=K+1` and the 8-variable schema.
5. **`assert_adapted_dir_empty_or_force(out_root, force)`** (Codex round-7 fix #2 — must live inside `score_5410.py` itself, not just the driver, so direct `python scripts/score_5410.py …` invocations are also safe): refuses if `out_root/inference/nwp/` is non-empty unless `force=True`. With `force=True`, deletes the prior `*.nc` files before adaptation begins. Mirrors the driver's behavior so smoke and direct-pytest paths get the same protection. Argparse-exposed as `--force` flag (default False).

These run before any compute. **Smoke runs (`--limit-ics 1` or `--ic-subset Y:s,…`) build the limited plan first and call preflight #4 with the trimmed plan + subset mode** (Codex round-6 fix #1):

`assert_output_dir_complete` already exists in `preflight.py` but rejects extras (designed for production where exactly the 96 expected files must be present, no more, no less). For smoke against the **existing** 96-IC `upstream_raw`, a 1-IC expected set would fail with 95 "extras". Solution: add a `mode` parameter:

```python
def assert_output_dir_complete(
    out_dir: Path, plan, K: int,
    *,
    mode: Literal["exact", "subset"] = "exact",
    expected_vars: frozenset = _EXPECTED_OUTPUT_VARS,
) -> None:
    """
    mode='exact'   : expected_filenames == actual_filenames (production).
    mode='subset'  : expected_filenames ⊆ actual_filenames (smoke);
                     extras are OK; the K+1 time-dim and variable-set
                     checks still run on every expected file.
    """
    expected = {...}
    actual = {f.name for f in p.glob("Y*_member*_y*.nc")}
    missing = expected - actual
    if missing:
        raise ValueError(...)
    if mode == "exact":
        extra = actual - expected
        if extra:
            raise ValueError(f"unexpected extra files in production output dir: ...")
    # mode == "subset": ignore extras
    for fname in sorted(expected):
        # time-dim + variable-set checks still run on each expected file.
        ...
```

Backward-compat: existing callers (production, smoke) need to pass `mode="exact"` or `mode="subset"` explicitly. Update the orchestrator's postflight call to pass `mode="exact"` (which is also the default).

For smoke: `assert_output_dir_complete(upstream_raw_dir, trimmed_plan, K, mode="subset")` — asserts the 1 expected raw NetCDF is present and well-formed; allows the other 95 from the production sweep to coexist. This catches partial inference for the targeted IC + ensures the score adapter reads a valid raw, without forcing the operator to delete 95 files just to run a smoke.

## Verification plan (sequenced gates)

1. **Implement** `score_adapter.py` + `score_climatology_compat.py` + `score_5410.py` + the 3 SLURMs + `submit_eval_5410.sh` + tests.
2. **Unit tests:** `pytest tests/sfno_inference_5410/test_score_adapter.py tests/sfno_inference_5410/test_pipeline_chain.py`. All schema + truth-alignment + driver-chain tests pass.
3. **One-IC score smoke:** `python scripts/score_5410.py --years 121 --limit-ics 1 --run-tag smoke_$(date +%Y%m%d) …`. Inspect the partial scorecard CSV; spot-check `tas`@6h is finite + within order of magnitude of GB=4.
4. **Render dry-run** on the same smoke output: `python scripts/render_eval_report.py --out-root <smoke_out> …` and `python scripts/render_eval_figures.py --out-root <smoke_out> --track 5410` (note `--track 5410` — Codex round-4 fix #3; without it the smoke wouldn't exercise the unit-scaling fix and the bias_pr_6h.png would be mislabeled). Confirm `report.md` + 7 PNGs land at expected paths AND `bias_pr_6h.png`'s colorbar label is "kg m^-2 (6h accum.)" not "mm day^-1".
5. **Codex round-2 review** of the full diff (adapter + helpers + 3 SLURMs + driver + tests).
6. **Score-only path on existing inference** (the immediate ask): the 96 raw NetCDFs from job 3098459 already live at `/work2/.../sfno_eval_5410/20260507_phase1_gate/inference/upstream_raw/`. Run:
   ```bash
   SKIP_INF=1 \
   RUN_ROOT=/work2/.../sfno_eval_5410/20260507_phase1_gate \
   ./scripts/submit_eval_5410.sh
   ```
   Driver creates a fresh `OUT_ROOT` from RUN_TAG; no symlinking required (Codex round-4 fix #2 — v4's two-root design means scoring reads raw from RUN_ROOT directly). Wallclock end-to-end: ~1 h.
7. **Read `$OUT_ROOT/report.md` side-by-side with GB=4 own-track report.** Compare RMSE/ACC at 5 channels × 6 leads.
8. (Future) **Full chain on a fresh inference run:** `./scripts/submit_eval_5410.sh` (no SKIP_*). End-to-end ~2 h (10 min inference + 1 h score + 30 min each report/figures, queue gaps aside).

## Out of scope

- ~~Score / report / figures SLURM chain for the full 5410 eval pipeline.~~ **Now in scope as of v3.**
- Cross-emulator side-by-side report (own + 5410 in one document). The eval plan §G.3 references `submit_eval_report_cross.slurm` — that's a separate piece of work and depends on having BOTH JOB_REP outputs in the same directory tree. Not for this plan; once both pipelines produce their own `report.md` you can read them side-by-side manually until the cross job is built.
- `--track {own,5410}` flag in `render_eval_report.py` to swap header field labels. Cosmetic; out of scope. v3 emits 5410-flavored values into the existing field labels (e.g., "Data packager SHA: 5410-v2.0").
- DDP fan-out for scoring/report/figures — all CPU-bound; ≤ 1 h on a single node.
- Modifying `score_nwp.py` or `render_eval_report.py` — adaptation happens via the score adapter, the compat-clim helper, and the provenance strings the driver passes. (As of v4 we DO patch `render_eval_figures.py` with a single `--track {own,5410}` flag; that's the only existing-code change. See §"Files to add / change" → "Existing files modified".)
- Unit conversion between 5410 and own-track — both pipelines train on Derecho-derived data with identical conventions; comparison is valid in absolute units.
- Climate-mode rollouts (`MODE=climate`). Own-track has it; 5410 doesn't. NWP-only is enough for the immediate cross-emulator comparison.

## What I'd like you to look at most carefully (round-7)

**Round-1 through round-6 blockers are all addressed:**
- ✅ Truth source (Derecho per-timestep h5).
- ✅ Schema (mirrors nc_writer.py: init_time=1, integer-hour leads, 52-channel init_state).
- ✅ Climatology coord (compat clim with `time_of_year → doy`).
- ✅ `file_anchor` parseable + `ic_file` clean-year extraction.
- ✅ Truth/prediction lead-1 tests correctly split.
- ✅ Magnitude bounds + correct pr_6h cap + correct forecast-error bounds (live-calibrated).
- ✅ Two-root design (RUN_ROOT prepared, OUT_ROOT fresh per run-tag).
- ✅ Conditional afterok chain (handles all 16 SKIP combos via `prev_job` accumulator).
- ✅ `render_eval_figures.py --track 5410` flag disables pr_6h scaling.
- ✅ `init_time` round-trip uses calendar-equivalence + full cftime arithmetic for `s ≥ 4`.
- ✅ Driver refuses non-skip inference launch on populated `upstream_raw`.
- ✅ Subset-mode `assert_output_dir_complete` for smoke against existing 96-IC raw dirs.
- ✅ `SCORE_ONLY=1` alias normalized at top of driver into canonical SKIP_* flags.
- ✅ Driver computes OUT_ROOT/RUN_TAG/SHAs **before** any preflight that references OUT_ROOT (no `set -u` unbound-variable crash).
- ✅ Adapted-output rerun safety enforced inside `score_5410.py` itself, not just the driver (direct `python scripts/score_5410.py …` invocations also refuse stale `inference/nwp/`).
- ✅ `FORCE=1` semantics tightened: actively deletes prior adapted NCs before writing (was a soft "bypass").
- ✅ Truth-h5 preflight checks per-IC `(Y, s+k)` files for k=0..K (not just yearly counts).

Open items I'd like you to spot-check:

1. **`_date_for_lead` arithmetic at year boundaries.** For Y=121 s=1342 (last IC of the year, K=60): anchor=`0121-01-01 00:00:00`, t_at_ic=335.5d, lead=336h=14d → base + 335.5d + 14d = Jan 1 + 349.5d = Dec 16 12:00. Still in year 121 (last day-of-year is 365). For leap year Y=124 s=1342, K=60: 1342 + 60 = 1402 < 1464 (still safe). For Y=128 s=1342, K=60: 1402 < 1464. No year overflow. Worth a confirming test that the `_date_for_lead` result's `month` is in [1, 12] for every (Y, s, lead) in the plan — if any returns an out-of-range month it means we wrapped past Dec 31.
2. **`channel_ic` 52-channel order.** Per `nc_writer.py:111`: `channel_ic = list(channel_names[:n_chan_ic])`. We set `n_chan_ic = 52` so channel_ic = pl..zg1000 (drops pr_6h at index 52). `score_nwp.py:173` iterates `range(n_state)` where `n_state = init_state.shape[0] = 52` — so persistence runs only for c=0..51 and pr_6h gets NaN as designed. Worth confirming by reading the score_nwp.py persistence block end-to-end.
3. **`zg` plev string formatting.** Adapter uses `h5['input'][f'zg_{plev:.1f}']` to match the `zg_50000.0` key style. Verify all 10 plev values (200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa = 20000, 25000, ..., 100000 Pa) format correctly. Note `zg_92500.0` not `zg_92500.5` — `:.1f` should be safe.
4. **No-op safe re-runs.** The driver writes adapted NCs into `{out_root}/inference/nwp/`. If a prior run partially populated this dir, do we clobber? Recommendation: refuse if non-empty (mirrors orchestrator's `assert_output_dir_empty`). Override knob: `--force` flag.
5. **Anchor-form math verification.** With `file_anchor = "0121-01-01 00:00:00"` + `time_plasim_at_ic = s * 0.25`, for any (Y, s) the verification time at lead h should equal `cftime(Y, 1, 1) + s*6h + h` (i.e. `init_datetime + lead`). Test `test_file_anchor_parses` (added per Codex round-2) cross-checks this against direct cftime arithmetic on the raw NC's `init_time`.

## Revision history

- **v1 (2026-05-08, evening):** initial draft. Used Makani `MOST.0{Y}.h5` truth (wrong source), 53-channel `init_state`, cftime `lead_time`, no climatology coord rename.
- **v2 (2026-05-08, evening):** Codex round-1 fixes — Derecho per-timestep truth, `nc_writer.py`-canonical schema, `time_of_year → doy` compat clim, 52-channel `init_state`. Plus Codex's recommended truth-alignment tests (`raw time=0 == IC h5`, `raw time=1 == truth s+1`).
- **v3 (2026-05-08, evening):** extends from "score-only" to the full own-track-style pipeline (inference → score → report → figures, single-entry-point `submit_eval_5410.sh` with afterok chain). Both `render_eval_report.py` and `render_eval_figures.py` are reusable as-is; only need new SLURM wrappers + the top-level driver. SKIP_INF / SKIP_SCO / SKIP_REP / SCORE_ONLY env-var knobs added so the immediate cross-emulator comparison can run without a fresh inference sweep (use the existing 96 NetCDFs from job 3098459).
- **v3.1 (2026-05-08, evening):** Codex round-2 fixes:
  1. **`file_anchor` made parseable** (`score_nwp.py:92` regex requires `YYYY-MM-DD HH:MM:SS`). Anchor-form chosen per Codex preference: `f"{Y:04d}-01-01 00:00:00"` + `time_plasim_at_ic = s * 0.25` (days). Matches the group's Jan-1/Y anchor convention.
  2. **`ic_file` made compatible with `score_nwp.py:139`'s strip-then-extract logic** (`f"{Y:04d}.h5"` → `ic_year = "0121"`, not `"121_0000"`). Per-timestep truth filename now lives in a separate `truth_h5_file` provenance attr.
  3. **Truth-alignment test strengthened** with magnitude bounds for tas / zg500 / pl / pr_6h / ua5. Catches off-by-one / wrong-channel / wrong-unit mistakes that bit-exact alone wouldn't.
  4. **New tests** `test_file_anchor_parses` and `test_ic_file_extracts_clean_year` directly enforce these contracts via the same regex / strip logic `score_nwp.py` uses.
- **v4 (2026-05-08, evening):** Codex round-3 fixes:
  1. **Two-root design** (`RUN_ROOT` ≠ `OUT_ROOT`) — `RUN_ROOT` is the prepared inference root (with ic_source.json, IC NCs, yamls, ckpt shim); `OUT_ROOT` is the per-eval scoring root (created fresh per RUN_TAG). Driver checks RUN_ROOT preconditions explicitly (mirrors what `submit_eval_inference_5410.slurm` itself checks) before submitting inference. Diverges from own-track's `OUT_ROOT == inference root` convention; documented in driver header.
  2. **Conditional afterok chain via `prev_job` accumulator.** Each stage's dep is built from the last *actually submitted* job, not from a hardcoded `JOB_INF`. All 16 (SKIP_INF, SKIP_SCO, SKIP_REP, SKIP_FIG) combinations work; `test_pipeline_chain.py` enumerates each.
  3. **pr_6h max cap raised** from 0.001 → 0.05 kg/m². Live `121_0001.h5` shows max=1.27e-2; v3.1's cap was wrong by 10×. Now uses mean (∈ [1e-5, 5e-3]) + 99th percentile (≤ 0.02) + max (≤ 0.05) — three nested bounds catch different failure modes.
  4. **`render_eval_figures.py` patched with `--track {own,5410}` flag.** Default `own` preserves existing m/s → mm/day scaling for own-track. `--track 5410` clears `CHANNEL_UNIT_SCALE` and re-labels `pr_6h` as "kg m^-2 (6h accum.)". Backward-compatible, ~15 added lines, zero removed lines. Honors the user's "no unit conversion on top of 5410 outputs" memory.
- **v4.1 (2026-05-08, evening):** Codex round-4 fixes:
  1. **Lead-1 test split into truth and prediction halves.** v3.1 had `test_raw_time1_equals_truth_at_s_plus_1_for_state_channels` saying "raw `time=1` vs `121_0001.h5`" should be bit-exact — wrong, raw `time=1` is the model forecast, not truth. Replaced with `test_truth_at_lead_6h_equals_h5_at_s_plus_1` (adapted truth side bit-equals h5) AND `test_prediction_at_lead_6h_equals_raw_time1` (adapted prediction comes from raw `time=1`, differs from truth within sane error magnitudes). Plus order-of-magnitude sanity bounds on the prediction-vs-truth diff (catches accidental ×1000 unit mistakes).
  2. **v3-vs-v4 layout inconsistencies removed.** Final-artifact list, score-only command example, and verification step 6 all corrected to v4's two-root semantics: `$RUN_ROOT/inference/upstream_raw/` holds raws (input), `$OUT_ROOT/...` holds adapted/scoring outputs. No symlinking needed for the score-only path; the score adapter reads raws directly from RUN_ROOT.
  3. **Verification step 4 figure dry-run** now passes `--track 5410` so the smoke actually exercises the unit-scaling fix and verifies the bias_pr_6h.png label.
  4. **"Out of scope" section** corrected: `render_eval_figures.py` IS modified (the `--track` flag patch); only `score_nwp.py` and `render_eval_report.py` remain unchanged.
- **v4.2 (2026-05-08, evening):** Codex round-5 fixes (calibrated against live Y121 s=0 data):
  1. **Forecast-error bounds relaxed and triple-bounded.** v4.1 caps `tas≤5 K` and `zg500≤50 m` failed against live values 8.9 K and 87 m. Replaced with mean+p99+max bounds.
  2. **`init_time` calendar-equivalence check** replaces the `dtype.kind == "M"` assertion (xarray uses cftime fallback for pre-1582 dates).
  3. **Driver refuses non-skip inference launch on populated `upstream_raw`.**
  4. **Smoke preflight runs `assert_output_dir_complete` on the trimmed plan.**
- **v4.3 (2026-05-08, evening):** Codex round-6 fixes:
  1. Subset mode for `assert_output_dir_complete`.
  2. `OUT_ROOT/inference/nwp` rerun safety promoted from recommendation to required driver preflight.
  3. `init_time` test math fixed for s ≥ 4 (full cftime arithmetic).
  4. `SCORE_ONLY=1` alias normalized at top of driver.
- **v4.4 (this doc):** Codex round-7 fixes:
  1. **Driver order-of-operations bug.** v4.3 referenced `$OUT_ROOT/inference/nwp` in the rerun-safety preflight (line 362) before `OUT_ROOT` was defined (line 387). Under `set -u` this crashes with "OUT_ROOT: unbound variable" on every default-RUN_ROOT run. Fix: hoisted `EVAL_SHA7`, `GROUP_SHA7`, `MODEL_SHA7`, `RUN_TAG`, `OUT_ROOT` computation to the top of the driver, before any preflight. Stale duplicate "provenance + OUT_ROOT setup" block deleted; the `provenance.txt` write now uses the already-defined `$OUT_ROOT`.
  2. **Adapted-output safety enforced inside `score_5410.py` itself**, not only the driver. Direct `python scripts/score_5410.py …` invocations (e.g., the 1-IC smoke at verification step 3) bypass the driver's preflight. Added `assert_adapted_dir_empty_or_force(out_root, force)` to the score_5410.py preflight chain (step 1e); argparse exposes `--force` (default False).
  3. **`FORCE=1` semantics tightened.** v4.3 was a soft bypass: if extras remained, score_nwp.py would still include them. v4.4 makes FORCE=1 actively `rm -f $OUT_ROOT/inference/nwp/*.nc` before adaptation. Same in driver and in score_5410.py (single source of truth). Guarantees no extras can survive into the new scorecard.
  4. **Truth-h5 preflight refined to per-IC.** v4.3's `assert_truth_h5_complete` only checked yearly file counts. v4.4 replaces it with `assert_truth_h5_for_plan(truth_dir, plan, K)` which iterates each `(Y, s)` in the plan and asserts `{Y}_{s+k:04d}.h5` exists for k=0..K. Catches holey truth coverage at submit time before any adaptation is attempted.
