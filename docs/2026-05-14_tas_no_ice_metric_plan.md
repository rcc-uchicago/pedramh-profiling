# tas_no_ice: ice-free near-surface air-temperature metric

**Date:** 2026-05-14
**Status:** draft for review
**Scope:** Add a sea-ice-masked variant of `tas` (`tas_no_ice`) to the SFNO eval pipeline, on both eval tracks (own-track v10 zgplev and group SFNO-5410). Existing `tas` rows remain untouched.

---

## 1. Motivation

PLASIM's `tas` is a **global** 2-m air-temperature field. Over sea-ice cells the value is the temperature of the air *above* ice, not over open ocean — this can dominate certain error patterns (and bias maps) without being directly tied to model skill over land + open ocean. We currently report a single `tas` RMSE/ACC/bias number that lumps all three surfaces. We want a second metric, `tas_no_ice`, that restricts the lat-weighted spatial integral to **land + open-ocean cells** (i.e. drops only the sea-ice cells).

`tas` itself stays in the scorecard for backward comparability and as the default published number.

---

## 2. Locked-in design (from interview)

**Channel name (FINAL):** `tas_no_ice`. User-confirmed in chat 2026-05-14 and re-confirmed under Codex review round 3 the same day. Do not re-open. Filename slug, dataclass field, table heading, figure filenames, and test names all use `tas_no_ice` verbatim.

| Knob | Decision |
| ---- | -------- |
| Cells kept | Land + open ocean (drop only sea-ice cells) |
| Mask source | Per-IC truth `sic` at lead h (time-varying with each forecast step) |
| Threshold | `sic >= 0.15` ⇒ "ice" (NSIDC ice-edge convention) |
| Metrics covered | Emulator RMSE, persistence RMSE, ACC. Bias maps stay unmasked. |
| Output | New channel name `tas_no_ice` alongside `tas`. `tas` rows unchanged. |
| Tracks | Both own-track (`scripts/score_nwp.py`) and group SFNO-5410 (`scripts/score_5410.py` → `score_nwp.py`) |
| Gate | Existing shipping gate (`tas` 6h RMSE < persistence) **unchanged**. `tas_no_ice` is reported but not gated, for v1. |

---

## 3. Where truth `sic` lives (verified)

### Own-track (eval-sfno-own, v10 zgplev)

- File: `$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full/test_holdout/MOST.{YEAR}.h5`
- Dataset: `forcing` of shape `(T=1455, 6, 64, 128)` float32
- Channel order (`channel_forcing`): `['lsm', 'sg', 'z0', 'sst', 'rsdt', 'sic']` → **sic at index 5**
- Per-year H5; for leads that spill past the year boundary the loader continues into `MOST.{YEAR+1}.h5`

Already loaded by the dataset as `tar_forcing[n_future+1, 6, H, W]` (z-scored) inside `rollout_one_ic` (`src/sfno_inference/rollout_driver.py:168`).

### 5410 (eval-sfno-5410)

- Truth source: per-timestep H5 at `$SCRATCH/SFNO_Climate_Emulator/data/derecho_glade/sim52/h5/sigma_data/{YEAR}_{step:04d}.h5`
- Dataset: `input/sic` of shape `(64, 128)` float32 (raw units, 0..1)
- One file per (year, 6h step); the adapter at `src/sfno_inference_5410/score_adapter.py:188-196` already opens these files lead-by-lead to read `truth[K, 53, H, W]`.

**Conclusion:** on both tracks, the per-(IC, lead) sic field is already being read upstream of the NetCDF write. The cheapest change is to **embed `truth_sic[K, H, W]` into the per-IC inference NetCDF** at write time, and have `score_nwp.py` read it back.

---

## 4. NetCDF schema change

Add **one new optional variable** to the per-IC inference NetCDF (own-track + 5410 adapter both emit it):

| Name | Shape | Dims | Units | Notes |
| ---- | ----- | ---- | ----- | ----- |
| `truth_sic` | `(1, K, H, W)` | `(init_time, lead_time, lat, lon)` | sea-ice fraction (0..1) | raw, NOT z-scored |

`truth_sic` mirrors the (init_time, lead_time, …, lat, lon) layout of `truth` minus the `channel` axis. `init_state_sic` (IC-time mask) is **not** added — we agreed mask is time-varying with each lead, not anchored at IC.

**Backward compatibility:** `score_nwp.py` checks for `truth_sic` presence. If absent (older NetCDFs from prior runs), masked rows are **skipped**, not errored — the scorecard simply omits `tas_no_ice` rows for those ICs. This means old NetCDFs continue to score under the existing `tas`-only flow.

---

## 5. File-level edit map

### 5.1 `src/sfno_eval/metrics.py`

Add two functions next to the existing `rmse_lat_weighted` (line 67) and `acc` (line 104):

```python
def rmse_lat_weighted_masked(pred, truth, lat_weights, mask):
    """Latitude-weighted RMSE over unmasked cells.

    mask : (..., lat, lon) bool — True = keep, False = drop.
    Weights renormalize over kept cells: w_ij = lat_weights[i] / W * mask[i,j],
    then divide by w.sum() to make it a proper weighted mean.
    Returns NaN if every cell is masked out.
    """

def acc_masked(pred, truth, clim_mean, lat_weights, mask, eps=1e-12):
    """ACC computed only over unmasked cells (same weighting rule)."""
```

Implementation sketch (RMSE):

```python
W = pred.shape[-1]
w = (lat_weights.view(*([1]*(pred.ndim - 2)), -1, 1) / W) * mask.float()
w_sum = w.sum(dim=(-2, -1))
num = ((pred - truth) ** 2 * w).sum(dim=(-2, -1))
return torch.where(w_sum > 0, (num / w_sum).sqrt(), torch.full_like(w_sum, float("nan")))
```

ACC follows the same masking rule applied symmetrically to `(pred - clim_mean)`, `(truth - clim_mean)`, and their normalising L2 norms.

### 5.2 `src/sfno_inference/rollout_driver.py` (own-track only)

Extend `RolloutResult` (line 38) with an **optional** field (default None).
The existing fields (`prediction`, `truth`, `init_state`, `K`,
`ic_global_idx`, `ic_sample_idx`, `ic_file`, `file_anchor`,
`time_plasim_at_ic`, `rollout_mode`) are all non-default. Python's
dataclass rule requires default-bearing fields to come **after** all
non-default fields, so `truth_sic` must be appended **at the end**,
after `rollout_mode`:

```python
@dataclass
class RolloutResult:
    prediction: torch.Tensor
    truth: torch.Tensor
    init_state: torch.Tensor
    K: int
    ic_global_idx: int
    ic_sample_idx: int
    ic_file: str
    file_anchor: str
    time_plasim_at_ic: float
    rollout_mode: str
    truth_sic: torch.Tensor | None = None   # (K, H, W) raw fraction in [0,1], NaN over land
```

Also extend `RolloutResult.to_dict` (line 62) to include
`"truth_sic": self.truth_sic` so any code path that round-trips through
the dict keeps the variable.

**Slice (corrected per code review):** `PlasimForcingDataset.get_sample_at_index`
(`src/sfno_training/data/plasim_forcing_dataset.py:337-340`) reads
`tar_forcing = self._read_forcing(global_idx, n_hist+1, n_hist+n_fut+2)`,
so the returned `tar_forcing` is **unbatched, shape `(K, 6, H, W)`**,
already aligned to leads `(+6h, +12h, …, +K*6h)` (no IC included).
The dataset z-scores it as `(tar_forcing - forcing_bias) / forcing_scale`
and stores those normalization arrays on the dataset instance:
`dataset.forcing_bias` / `dataset.forcing_scale`, both shaped
`(1, n_forcing_channels=6, 1, 1)` (`plasim_forcing_dataset.py:81-82`).

In `rollout_one_ic` (line 107), after the validate body finishes (line 248):

1. **Use the original CPU tensor.** The `dataset[ic_global_idx]` return
   tuple lives on CPU (line 169); only `gdata` is unsqueeze+`.to(device)`'d
   (line 188-191). To avoid a needless device round-trip, work off the
   CPU `tar_forcing`:
   ```python
   sic_z = tar_forcing[:, 5, :, :].float()   # CPU, fp32, shape (K, H, W)
   ```
   (`tar_forcing` here is the local var from the original tuple-unpack
   at line 176, not `gdata[3]`.)
2. Pull forcing stats **from the dataset instance** (not from a re-loaded
   `.npy` path — `checkpoint_loader.py:113` documents that forcing stats
   are config-inherited, and `PlasimForcingDataset` already holds the
   broadcast-shaped arrays at `plasim_forcing_dataset.py:81-82`):
   ```python
   fb5 = float(dataset.forcing_bias.reshape(-1)[5])
   fs5 = float(dataset.forcing_scale.reshape(-1)[5])
   sic_phys = sic_z * fs5 + fb5
   ```
   Stats are scalars; this is a CPU op. NaN cells from the packager
   (`packager.py:226` keeps NaN over land) round-trip as NaN (z-scoring
   NaN is NaN, inverse z-scoring NaN is NaN).
3. **Do not clamp to `[0,1]`.** Clamping would replace land-NaNs with 0.0
   and silently flip the mask semantics. Leave physical values as-is.
4. Attach as `result.truth_sic = sic_phys` (already CPU, fp32).

If for any reason `dataset.forcing_bias`/`forcing_scale` are missing or
have wrong shape (defensive — this would indicate a misconfigured run),
leave `result.truth_sic = None`. `score_nwp.py` then treats the NetCDF
as having no `truth_sic` (see §5.5 skip rule). No NaN-fallback variable
is written.

### 5.3 `src/sfno_inference/nc_writer.py` (own-track)

Write `truth_sic` **conditionally** — `RolloutResult.truth_sic` is
optional (§5.2), so the writer must not unconditionally call
`.numpy()` on it.

Approach:

1. Build `data_vars` as a `dict` (instead of inline kwargs) so a
   conditional insert is easy.
2. After the unconditional `prediction` / `truth` / `init_state` entries,
   add:
   ```python
   if result.truth_sic is not None:
       data_vars["truth_sic"] = (
           ("init_time", "lead_time", "lat", "lon"),
           result.truth_sic.numpy()[np.newaxis, ...].astype(np.float32),
       )
   ```
3. Build the encoding dict (line 173) the same way — only include
   `truth_sic` when the variable was inserted.
4. After construction, only when present, set
   `ds["truth_sic"].attrs["units"] = "fraction"` and
   `ds["truth_sic"].attrs["description"] = "Truth sea-ice fraction at each lead; NaN over land. Downstream tas_no_ice mask uses sic >= 0.15 to drop sea-ice cells."`.

This keeps the existing 3-variable schema as the backward-compat
baseline and makes the new variable's presence the **signal** that the
NetCDF is `tas_no_ice`-capable.

### 5.4 `src/sfno_inference_5410/score_adapter.py`

In `adapt_5410_ic_to_score_nwp` (line 138):

1. Allocate `truth_sic = np.empty((K, H, W), dtype=np.float32)` alongside `truth` (line 187).
2. In the `for k_lead in range(K)` loop (line 188), inside the same h5py open, read `f["input/sic"][...]` → `truth_sic[k_lead]`.
3. Add the variable to `out_ds` (line 215-228) with dims `(init_time, lead_time, lat, lon)` and `units="fraction"` attr.

### 5.4a `scripts/score_5410.py` preflight

The existing per-IC truth-h5 preflight (`_preflight_truth_h5_for_plan`, line 152-180) sample-checks `f["input/tas"].shape == (64, 128)` on the first plan entry. Extend it so the same sampled file also asserts:

- `"input/sic"` exists,
- `f["input/sic"].shape == (64, 128)`,
- dtype is float32 (or coerces cleanly),
- **value range** (catches sentinel values like `-9999` that pass `np.isfinite`):
  ```python
  arr = f["input/sic"][...].astype(np.float32, copy=False)
  finite = np.isfinite(arr)
  if not np.all(finite | np.isnan(arr)):
      raise ValueError("input/sic has +/-inf values")
  finite_vals = arr[finite]
  tol = 1e-4
  if finite_vals.size and (finite_vals.min() < -tol or finite_vals.max() > 1 + tol):
      raise ValueError(
          f"input/sic finite values out of [0,1] (with tol={tol}): "
          f"min={float(finite_vals.min()):.6g}, max={float(finite_vals.max()):.6g}"
      )
  ```
  The `tol=1e-4` slack mirrors the packager's clip behavior (`packager.py:213-258` clips `sic` to `[0,1]` and validates against MOST; tiny float noise is fine, sentinels like `-9999` are not). NaN cells (land) are tolerated separately.

This catches missing-sic / sentinel-encoded-sic upstream-data regressions at submit time rather than mid-scoring, mirroring the existing `input/tas` check.

### 5.5 `scripts/score_nwp.py`

In `_compute_metrics_for_one_ic` (line 106):

1. After `truth = torch.from_numpy(ds["truth"].values[0]).float()` (line 126), conditionally:
   ```python
   has_sic = "truth_sic" in ds.variables
   if has_sic:
       truth_sic = torch.from_numpy(ds["truth_sic"].values[0]).float()   # (K, H, W)
   ```
2. Inside the `for h in _SCORED_LEADS_H` loop, after the existing `tas` row is emitted (around line 154-159), if `has_sic`:
   - Build mask: `mask = ~(truth_sic[k] >= 0.15)`. This is the **NaN-safe** form. The packager preserves `sic == NaN` over land (`src/plasim_makani_packager/packager.py:226`); for a NaN cell `(NaN >= 0.15) → False`, so `~False → True` → **kept** (land stays in the metric, per the locked decision §2). Using the naive `(truth_sic[k] < 0.15)` would treat NaN as False → drop land. The expression still drops `sic == 0.15` (NSIDC ice-edge convention).
   - Do **not** add a `~torch.isnan(truth_sic)` term to the mask — that would re-drop the land cells we just took care to keep. If a future packager regression produces NaN tas (separate variable), the metric would surface as NaN through `(pred-truth)**2`; relying on that as a data-quality smoke is acceptable.
   - **If `mask.sum() == 0` (extreme — fully ice-covered slice), do not emit any `tas_no_ice` row for this (IC, lead).** This keeps the finite-row gate at lines 258-262 honest: missing rows are tolerated; NaN rows are not.
   - Otherwise find `c_tas = chan_names.index("tas")` and emit:
     - emulator RMSE row with `channel="tas_no_ice"` from `rmse_lat_weighted_masked(pred[k, c_tas], truth[k, c_tas], lat_w, mask)`
     - persistence RMSE row with `channel="tas_no_ice"` from `rmse_lat_weighted_masked(init_state[c_tas], truth[k, c_tas], lat_w, mask)`
     - ACC row (when climatology covers the bin) from `acc_masked(pred[k, c_tas], truth[k, c_tas], cm, lat_w, mask)`
   - Bias map accumulator is **not** updated for `tas_no_ice` (per decision).

The shipping gate at line 240-247 still keys on `("emulator", "tas", 6, "rmse")` — **unchanged**.

The finite-row gate at lines 258-262 (`if r["model"] == "emulator" and r["value"] != r["value"]`) is **not** softened: by skipping rather than emitting NaN we keep the invariant "every emitted emulator row is finite". The aggregator `_mean` already tolerates a missing key (returns NaN, which only affects the new tas_no_ice section render).

### 5.6 `scripts/render_eval_report.py`

Report structure (verified at line 290-316): `_render_header → benchmark_banner → _render_table(summary, key_channels=bias_channels(channel_names)) → _render_gate → _render_bias_maps → _render_climate → _render_provenance`. The scorecard table channels come from `bias_channels()` (`scripts/_eval_utils.py:86`), which intentionally excludes `tas_no_ice` because that list also drives the bias-map grid.

Add a **new dedicated section** `_render_masked_tas(summary, benchmark_summary)`, slotted between `_render_table` and `_render_gate` in the `parts = [...]` list (line 303-311). It produces:

- A short markdown subsection titled "Sea-ice-masked tas (`tas_no_ice`, sic < 0.15)".
- A small table with rows for leads `{6, 24, 120, 240}` h and columns `emulator RMSE | persistence RMSE | ACC | n_ICs`. Mirror the formatting of `_render_table`.
- The 5410 benchmark column rendered when `benchmark_summary` is non-None (same gate as the main table).
- Soft-skip the whole section if `("emulator", "tas_no_ice", 6, "rmse")` is absent (older runs / NetCDFs lacking `truth_sic`).

The gate function `_render_gate` is **not** modified — the shipping gate stays on `tas`.

`bias_channels()` is **not** modified — keeps the bias-map grid honest (no orphan `bias_maps_tas_no_ice_*.png` will ever be written, see §5.7).

### 5.7 `scripts/render_eval_figures.py`

The existing line-plot function hard-codes a 2x2 layout (`fig, axes = plt.subplots(2, 2, …)` at line 176, zipped with `LINE_PLOT_CHANNELS`). A 5th channel appended to `LINE_PLOT_CHANNELS` would be **silently dropped**. Note: the current outputs are split into **two** files — `figures/rmse_vs_lead.png` (line 439) and `figures/acc_vs_lead.png` (line 441), each calling the same `plot_lines(summary, metric, out_path, ...)` once per metric. Per-channel bias maps land at `figures/bias_{channel}.png` (line 444).

Instead, write **two new dedicated figures** for the masked tas, mirroring the existing split:

- New helper `plot_masked_tas_lines(summary, metric, out_path, *, include_persistence, benchmark_summary)` produces a single-panel figure of `tas_no_ice` only, vs lead — emulator (+ optional persistence, + optional 5410 benchmark) — reusing `SERIES_STYLE`, `REPORT_LEADS`, `_lead_days`.
- `main()` dispatcher (around line 439) gets two new calls:
  - `plot_masked_tas_lines(summary, "rmse", fig_dir / "rmse_vs_lead_tas_no_ice.png", …)`
  - `plot_masked_tas_lines(summary, "acc",  fig_dir / "acc_vs_lead_tas_no_ice.png",  …)`
  Both soft-skip if no `tas_no_ice` rows are present in the summary.
- Add `CHANNEL_LABELS["tas_no_ice"] = "tas (ice-free cells, sic<0.15)"` and `CHANNEL_UNITS["tas_no_ice"] = "K"`.
- Do **not** add `tas_no_ice` to `LINE_PLOT_CHANNELS` and do **not** add to `REPORT_CHANNELS` (bias-map grid stays 5-channel; no `scores/bias_maps_tas_no_ice_*.npy` accumulator written and no `figures/bias_tas_no_ice.png` emitted).
- `render_eval_report.py` (§5.6) references the two new PNGs in the masked-tas section.

### 5.8 `scripts/_eval_utils.py`

`STATE_BIAS_CHANNELS` at line 86 is unchanged (bias maps don't carry `tas_no_ice`). No edits.

---

## 6. Tests

Tests must cover **alignment + invariants**, not just shape.

### 6.1 Math helpers — `tests/sfno_eval/test_rmse_lat_weighted_masked.py`

- Identity mask (all-true) reproduces `rmse_lat_weighted` **to within float tolerance** — use `torch.allclose(..., atol=1e-6, rtol=1e-6)` (or `np.testing.assert_allclose` if computing in numpy). The masked implementation reduces in a different order from the original (`(err2 * w).sum(-2,-1) / w.sum(-2,-1)` vs the original's `err2.mean(-1)` then weighted-sum over lat), so floating-point reduction order may differ — bit-exact equality is too strict.
- Hand-checked 3×4 synthetic case with a known mask: verify against manual computation, again with `allclose` (atol matching the hand computation's precision).
- All-False mask returns NaN (not 0, not inf).
- Per-row mask (some lat bands fully masked): verify weights renormalize correctly.
- **Threshold-edge invariant** (tested at the scoring layer, not in the helper): given `truth_sic` with values `{0.0, 0.149999, 0.15, 0.15001, 0.5}` across cells, the score_nwp mask drops `>= 0.15` (third onward) and keeps the first two. Verified in §6.6.
- **NaN-keeps-cell invariant**: given `truth_sic` with NaN cells (simulating land per packager.py:226), the score_nwp mask via `~(>= 0.15)` **keeps** those cells. Verified in §6.6.

### 6.2 Math helpers — `tests/sfno_eval/test_acc_masked.py`

- Identity mask reproduces `acc` **to within float tolerance** (`torch.allclose(..., atol=1e-6, rtol=1e-6)`) — same reduction-order rationale as §6.1.
- **Masked-region invariance**: sign-flipping a masked region does not change ACC (proves it's actually masked, not silently included in either numerator or denominator). Use `allclose` here too — the sign-flipped path still goes through the masked-out branch, which has the same FP reduction order.
- All-False mask returns NaN.

### 6.3 Own-track write — `tests/sfno_inference/test_nc_writer_truth_sic.py`

- Writing a `RolloutResult` with synthetic `truth_sic` produces a NetCDF that has the variable with the expected dims `(init_time, lead_time, lat, lon)`, shape `(1, K, H, W)`, dtype float32, and `units="fraction"` attr.
- Old `RolloutResult`-shaped synthetic input without `truth_sic` writes a NetCDF that lacks the variable (backward-compat path holds).

### 6.4 Own-track alignment — `tests/sfno_inference/test_rollout_truth_sic_alignment.py` (new)

This is the load-bearing one for #1. Build a synthetic `PlasimForcingDataset` (or mock its return tuple) whose `tar_forcing[:, 5, :, :]` carries known per-lead patterns (e.g. lead h → all-`h/600.0` floats), z-scored against known forcing stats. Run `rollout_one_ic` (with a no-op wrapper that returns its input) and assert:

- `result.truth_sic[0]` equals the source sic at lead +6h (proves the dataset's slice starts at +6h, not at IC).
- `result.truth_sic[k]` equals the source sic at lead `(k+1)*6h` for all `k ∈ [0, K)`.
- `result.truth_sic` is in physical units (the z-scoring round-trip is correct).

### 6.5 5410 adapter — extend `tests/sfno_inference_5410/test_score_adapter.py`

- The adapter copies `input/sic` from per-timestep h5s into `truth_sic` aligned with `truth`'s lead-time axis (assertion: for each `k_lead`, `truth_sic[k_lead] == h5["input/sic"]` of file `{Y}_{s + k_lead + 1:04d}.h5`).
- Output NetCDF has `truth_sic` with the same `init_time, lead_time, lat, lon` coords as `truth` (no channel axis).

### 6.6 Scoring — `tests/score_nwp/test_tas_no_ice_rows.py` (new)

- Build a 1-IC NetCDF with `truth_sic` containing a hand-crafted mask pattern → run `_compute_metrics_for_one_ic` → confirm:
  - `tas_no_ice` rows for `(emulator, rmse), (persistence, rmse), (emulator, acc)` appear at each scored lead.
  - The masked numbers **differ** from the unmasked `tas` numbers (proves the mask is biting).
  - The unmasked `tas` rows are **bit-identical** to a baseline run on the same NetCDF without `truth_sic` (proves we didn't accidentally change the existing metric).
- **Threshold-edge case**: `truth_sic` set to `{0.0, 0.149999, 0.15, 0.15001, 0.5}` across 5 cells in a single lat band, with known errors. Verify the masked metric uses exactly the first two cells (strict-less-than convention via `~(>= 0.15)`).
- **NaN-over-land case**: a `truth_sic` field with NaN cells over (synthetic) "land" and 0.0 over ocean. Verify those land cells are **kept** in the masked metric (Codex round-2 finding #1). Note: we do **not** assert that NaN-over-ice would be detectable — with only `truth_sic` carrying provenance, a NaN is indistinguishable from land NaN. If you ever need to distinguish, you'd have to add a separate land-mask provenance channel; that's out of scope here.
- Fully-masked lead (all `sic >= 0.15`, no NaN): no `tas_no_ice` rows emitted for that (IC, lead); no NaN row written; finite-row gate passes.
- NetCDF without `truth_sic`: `_compute_metrics_for_one_ic` emits only the existing rows (no `tas_no_ice`, no crash).

### 6.7 Report + figures invariants

- `tests/render_eval_figures/test_no_tas_no_ice_bias_artifacts.py`: after a scoring smoke,
  - no file `scores/bias_maps_tas_no_ice_*.npy` is written (bias accumulator stays 5-channel);
  - no file `figures/bias_tas_no_ice.png` is written (bias-map renderer never sees the new channel);
  - the new `figures/rmse_vs_lead_tas_no_ice.png` and `figures/acc_vs_lead_tas_no_ice.png` **are** written when `tas_no_ice` rows are present in the scorecard, and **are not** written when they are absent.
- `tests/render_eval_report/test_masked_tas_section.py`: report contains the new section header when `tas_no_ice` rows are present, and **omits it** when absent.

### 6.8 Smoke runs

- One-IC own-track smoke: re-run a tiny inference (existing smoke harness) so the new NetCDF carries `truth_sic`, then `score_nwp.py`. Confirm `tas_no_ice` rows in `scorecard.csv` and the new report section.
- One-IC 5410 smoke through the existing smoke driver to confirm parity (5410 adapter writes `truth_sic`).

---

## 7. What this does NOT change

- `tas` RMSE/ACC/bias rows in the scorecard. Same numbers as today.
- The shipping gate (`tas` 6h RMSE < persistence). Unchanged.
- Climatology builder. Climatology has no `sic` and doesn't need it for this metric — the mask comes from truth, not climatology.
- The 4-job SLURM chain in `scripts/submit_eval.sh`. No new job; the masked metric drops into the existing score → report → figures flow.
- 5410 production driver `scripts/infer_sfno5410_blocking_h100_packed.py`. No change — the adapter is the one that writes `truth_sic`.
- Existing eval NetCDFs on disk. They simply won't have `tas_no_ice` rows in re-scored scorecards (soft-skip).

---

## 8. Rollout

1. Land §5.1 + §5.2 + §5.3 (own-track write path) + tests.
2. Land §5.4 (5410 adapter) + extended test.
3. Land §5.5 (scoring) + §5.6 + §5.7 (report/figures) + smoke.
4. Re-run own-track NWP eval on the v10 production run → verify `tas_no_ice` rows in scorecard and the new report section. (One full 96-IC run; same SLURM chain.)
5. Optional: re-run 5410 smoke + production scoring to surface the same rows.

---

## 9. Open questions

- **Climatology coverage of marginal-ice zones for ACC:** climatological tas values over (sometimes-icy, sometimes-open) cells are an average of both regimes. Masking by truth sic means a cell can be in-mask one lead and out-of-mask the next — the ACC anomaly there is computed against a "mixed" climatology. This is the intended behavior (we asked for a truth-sic-based mask), but worth flagging: if numbers look noisy at long leads, we may want to revisit whether the climatology mean is the right reference for an ice-edge-jittery sample.

- **Edge case — fully-masked lead:** resolved per Codex review #3. We **skip** emitting `tas_no_ice` rows for any (IC, lead) where `mask.sum() == 0`. This avoids tripping the finite-row gate at `score_nwp.py:258-262`. Downstream `_mean` over the rows simply averages over fewer ICs at that lead; the new report section prints "n/a" if no IC has a valid row at all.

---

## 10. Codex review log

### Round 1 (2026-05-14)

| # | Severity | Issue | Resolved in |
| - | -------- | ----- | ----------- |
| 1 | High | `tar_forcing` slice was `[0, 1:K+1, 5, :, :]` (wrong — implies batched/lead-shifted layout). Dataset returns unbatched `(K, 6, H, W)` already aligned to leads +6h..+K*6h. | §5.2 (slice now `tar_forcing[:, 5, :, :]` + explicit code-reference) |
| 2 | High | Adding `tas_no_ice` to `LINE_PLOT_CHANNELS` is silently dropped by the hard-coded `2×2` layout (`render_eval_figures.py:176`). | §5.7 (dedicated figures, `LINE_PLOT_CHANNELS` left at 4) |
| 3 | High | Emitting NaN rows for fully-masked leads trips the finite-row emulator gate (`score_nwp.py:258-262`). | §5.5 (skip emission entirely when `mask.sum()==0`; gate untouched) |
| 4 | Medium | Naming `tas_no_ice` vs `tas_ice_free`. | User chose `tas_no_ice` in chat; plan was already on it. No change. |
| 5 | Medium | Report integration referenced wrong lines (166-176 are the gate). Scorecard table channels come from `bias_channels()`. | §5.6 (new dedicated `_render_masked_tas` section between `_render_table` and `_render_gate`; `bias_channels()` untouched) |
| 6 | Medium | Tests should cover alignment + invariants, not just shape. | §6 split into 6.1–6.8 with alignment, threshold-edge, masked-region-invariance, bit-identical `tas` checks |

### Round 2 (2026-05-14)

| # | Severity | Issue | Resolved in |
| - | -------- | ----- | ----------- |
| 1 | High | `mask = (truth_sic < 0.15)` drops land where the packager preserves `sic == NaN` (`packager.py:226`). `(NaN < 0.15) → False` would incorrectly treat land as ice. | §5.5 (mask now `~(truth_sic >= 0.15)`; NaN cells kept); new NaN-over-land test in §6.6 |
| 2 | Medium | Naming flagged again — but user chose `tas_no_ice` in chat. | No change; Round-1 row 4 already records this. |
| 3 | Medium | `RolloutResult.truth_sic` was added as a required dataclass field while writer was supposed to soft-skip — contradictory. | §5.2 makes it `Optional` with default `None`; §5.3 writes the variable conditionally based on presence |
| 4 | Medium | Forcing-stats inverse transform should use the dataset's loaded arrays, not re-load run-dir `.npy` paths (`checkpoint_loader.py:113` notes forcing stats are config-inherited, not copied). | §5.2 now uses `dataset.forcing_bias.reshape(-1)[5]` / `dataset.forcing_scale.reshape(-1)[5]`; no NaN-fallback variable; if missing, `truth_sic` stays `None` |
| 5 | Low | Stale figure filenames in plan: actual outputs are `rmse_vs_lead.png` / `acc_vs_lead.png` and `bias_{ch}.png`, not `line_plots.png` / `bias_maps_*.png`. | §5.7 corrected; new files are `rmse_vs_lead_tas_no_ice.png` and `acc_vs_lead_tas_no_ice.png`; §6.7 invariant test targets `scores/bias_maps_tas_no_ice_*.npy` AND `figures/bias_tas_no_ice.png` (both must be absent) |

### Round 3 (2026-05-14)

| # | Severity | Issue | Resolved in |
| - | -------- | ----- | ----------- |
| 1 | Medium | Naming flagged for the third time. | §2 now has an explicit **FINAL** marker. User re-confirmed `tas_no_ice` 2026-05-14 under round-3 review. No further re-litigation. |
| 2 | Low/Medium | §6.6 had an impossible assertion: "flipping all sea-ice cells to NaN does NOT keep them as land". With only `truth_sic` as provenance, NaN is indistinguishable from land-NaN; the mask `~(>=0.15)` keeps it either way. | §6.6 reworded — explicitly notes that NaN provenance is not distinguishable and that adding a land-mask channel is out of scope. |
| 3 | Medium | `truth_sic: torch.Tensor \| None = None` is correct *in isolation* but illegal **placement** in a `@dataclass` where all existing fields are non-default — Python raises "non-default argument follows default argument". | §5.2 now shows the full dataclass with `truth_sic` appended **after** `rollout_mode`, and notes the dataclass-default-ordering rule. `to_dict` also extended. |
| 4 | Medium | 5410 preflight only sample-checks `input/tas`. The adapter will now require `input/sic` too — a missing-sic upstream regression would fail mid-scoring rather than at submit time. | §5.4a (new) extends `_preflight_truth_h5_for_plan` (`score_5410.py:152-180`) to also check `input/sic` presence, shape `(64,128)`, and finite-or-NaN values. |

### Round 4 (2026-05-14)

| # | Severity | Issue | Resolved in |
| - | -------- | ----- | ----------- |
| 1 | Medium | §5.4a "finite or NaN" check passes sentinel values like `-9999` (which *are* finite). Prose said reject `-9999`, code didn't. | §5.4a now adds an explicit `[0, 1]` range check on finite cells (with `tol=1e-4` slack to match the packager's clip behavior at `packager.py:213-258`). |
| 2 | Medium | §6.1/§6.2 tests demanded exact numerical equality for identity-mask. The masked impl reduces in a different order from `rmse_lat_weighted`/`acc`, so FP reduction order may diverge. | §6.1/§6.2 now specify `allclose` (`atol=1e-6, rtol=1e-6`) for identity-mask equivalence. |
| 3 | Low | §5.2 said `sic_z` is "on device, fp32". The original `tar_forcing` from `dataset[…]` lives on **CPU** (line 169); only `gdata` is moved to device (line 188-191). | §5.2 now explicitly works off the CPU `tar_forcing` from the original tuple-unpack at line 176, avoiding a needless device round-trip; all ops are CPU/scalar. |
