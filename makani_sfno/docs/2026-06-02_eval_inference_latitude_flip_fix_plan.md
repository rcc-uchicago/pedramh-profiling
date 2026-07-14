# Implementation plan — fix latitude-coordinate flip in `eval_inference.py`

**Date:** 2026-06-02
**Issue doc:** [`docs/2026-06-02_eval_inference_latitude_flip.md`](./2026-06-02_eval_inference_latitude_flip.md)
**Status:** Implemented 2026-06-02 (core: §3.1 reader fix + §3.2 writer guard + test migration;
issue-doc §3 corrected). Backfill (§3.3) skipped — no existing own-track `eval_inference.py` outputs
found to repair. Open §6 defaults taken: ascending torch_harmonics fallback **removed**; no shared
cross-writer assertion added (deferred).
**Scope:** `SFNO_Climate_Emulator` only. `AI-RES-clean` is **not affected** and needs no changes
(verified: it never consumes `eval_inference.py` NetCDFs, sources lat from a descending yaml, and
routes every field through `standardize_coordinates()` before any lat-coordinate join).

---

## 1. Problem statement (verified, with a correction to the issue doc)

`scripts/eval_inference.py` writes NWP- **and** climate-mode inference NetCDFs whose `lat`
**coordinate** is the reverse of the **data array** it labels. The data rows are **descending**
(row 0 = +87.86°N … row 63 = −87.86°S, the PlaSim convention), but the written `lat` coordinate is
**ascending** (−87.86 → +87.86). Result: every written field (`prediction`/`truth`/`init_state`/
`truth_sic`) is **hemisphere-flipped relative to its label**. Data values are intact — it is a
**label-only** flip.

**Corrected root cause** (the issue doc §3 attributes this to `metadata/data.json`'s `lat` being
ascending; that is *not* what happens):

- `_read_lat_lon_from_run` (`scripts/eval_inference.py:102`) looks for a **top-level** `lat`/`lon`
  key: `if "lat" in md and "lon" in md` (line ~125).
- **Every** `metadata/data.json` in the repo stores latitude under **`coords/lat`**, not top-level,
  and that `coords/lat` is **descending (correct)**. Verified across all 14 makani datasets:
  `top_lat=False`, `coords_lat=DESC` for every one.
- So the top-level check **always misses** and the function **always falls through to the
  torch_harmonics fallback** (lines ~128-138), which returns Gauss–Legendre nodes
  **ascending (−87.86 → +87.86)** — confirmed empirically with the `aires` env.

Therefore the true flip source is the **ascending torch_harmonics fallback**, while the metadata it
ignores actually holds the correct descending coordinate. The downstream symptom is identical to the
issue doc; only the mechanism differs — and it matters because the cleanest fix is to read a
coordinate the data/metadata already provide in the correct order.

### Affected vs. not affected (verified by reading the code)

| Component | Affected? | Evidence |
|---|---|---|
| `eval_inference.py` NWP output `lat` coord (`run_nwp`, line 237) | **Yes** | ascending label on descending data |
| `eval_inference.py` climate output `lat` coord (`run_climate`, line 317) | **Yes** | same call to `_read_lat_lon_from_run` |
| Training / in-training validation / forward pass | No | positional; quadrature weights from `grid_type`+`nlat`; never uses lat array |
| Own-track scorer `score_nwp.py` | No | loads `ds["prediction"].values[0]` **positionally**; weights via `legendre_gauss_lat_weights(H)` (symmetric); climatology indexed positionally; **never reads `ds["lat"]`** |
| Climatology builder `compute_climatology.py` | No | writes `lat=np.arange(H)` (integer indices, data order) |
| Figures `render_eval_figures.py` | No | never opens NC `lat`; `imshow(origin="upper", extent=[…,-90,90])` on data-order arrays |
| 5410 / group→AIRES convert `convert_group_inference_to_aires_nc.py` | No | reads lat from the **h5** (`f["lat"]`, descending) |
| `AI-RES-clean` | No | never consumes these NetCDFs; descending yaml lat; `standardize_coordinates()` guard |

**Consequence:** No *internal* published number is wrong (the whole own-track chain is positional and
self-consistent). The flip only bites **external, coordinate-joining consumers** (e.g. the downstream
`ensemble-perturbation-study`), which is how it was discovered.

**Zero-regression note:** because no internal consumer reads `ds["lat"]`, relabeling the coordinate
from ascending → descending changes **no** existing own-track metric, gate, or figure. The fix is
purely additive correctness for external consumers.

---

## 2. Goals (verifiable)

1. `_read_lat_lon_from_run` returns lat in **descending** (data) order for every run.
   → verify: re-run issue-doc Claim A repro; expect `lat[0] ≈ +87.9`, `lat[-1] ≈ −87.9`.
2. `write_rollout_nc` **refuses** to write a non-descending `lat` (fail-loud guard), and the
   existing `test_nc_writer.py` success fixtures are migrated to descending lat (they currently
   pass ascending `np.linspace(-90, 90, H)`).
   → verify: `pytest tests/sfno_inference/test_nc_writer.py` green, incl. a new ascending-rejection test.
3. A freshly produced NWP NetCDF reads correctly by coordinate.
   → verify: `OUTPUT zg500 .sel(lat=+70)` ≈ NH value (~5500), not ~4994.
4. (Optional) existing flipped NetCDFs are repaired in place, data untouched.
   → verify: post-relabel `.sel(lat=+70)` ≈ NH value; array bytes for data vars unchanged.

---

## 3. Changes

### 3.1 Core fix — `_read_lat_lon_from_run` (`scripts/eval_inference.py:102`)

Source lat/lon from the **authoritative descending data** and cross-check against metadata; remove
the orientation-flipping fallback. Both call sites already have the test h5 available, so pass it in.

**New behaviour (preference order):**

1. **Read lat/lon from the source h5** (`h5_path['lat']`, `h5_path['lon']`) — authoritative,
   descending, identical grid the model trained/forecasts on.
2. **Cross-check** against `metadata/data.json` `coords/lat` when present: assert the rounded
   value-*sets* match (`sorted(round(h5_lat)) == sorted(round(meta_lat))`). This catches a genuine
   grid mismatch (wrong run vs wrong data) while being orientation-agnostic. **Fail loud** on
   mismatch.
   - **Metadata-path order:** resolve `cfg["metadata_json_path"]` **first** (it is
     `{{OUTPUT_ROOT}}/metadata/data.json`, the reliable location — see
     `src/sfno_training/config/plasim_sim52_zgplev_full.yaml:26`). Do **not** rely on the existing
     `train_data_path`-derived fallback as-is: with `train_data_path = {{OUTPUT_ROOT}}/train`
     (yaml line 27), `Path(train_path).parent.parent / "metadata"` resolves **one level above**
     `$OUTPUT_ROOT` and is wrong. Either drop that fallback or correct it to
     `Path(train_path).parent / "metadata" / "data.json"`.
   - Read `coords/lat` / `coords/lon` (nested), **not** a top-level `lat` key (which never exists).
3. **Remove** the torch_harmonics ascending fallback. In the eval path the source h5 is always
   present (we enumerate `test_files` before calling this), so the fallback is dead weight and is the
   exact thing that introduced the flip. If a defensive fallback is still wanted, it MUST return
   **descending** lat and be asserted as such (see §6 open question).

**Signature change:** `_read_lat_lon_from_run(run_dir, h5_path)`.

**Callers to update (pass `test_files[0]`):**
- `run_nwp`: line 237 → `lat, lon = _read_lat_lon_from_run(args.run_dir, test_files[0])`
- `run_climate`: line 317 → same.

(`test_files` is in scope at both sites: `run_nwp` line 225, `run_climate` line 312.)

### 3.2 Defense-in-depth guard — `src/sfno_inference/nc_writer.py`

In `write_rollout_nc`, after the existing length check (line ~100-103) and before building the
dataset, assert `lat` is **strictly monotonically decreasing** (the descending-by-contract grid):

```python
lat_arr = np.asarray(lat, dtype=np.float64)
if not np.all(np.diff(lat_arr) < 0):
    raise ValueError(
        f"lat must be strictly descending (data is North-first); "
        f"got lat[0]={lat_arr[0]:.3f} lat[-1]={lat_arr[-1]:.3f}. "
        f"Refusing to write a mislabeled grid."
    )
```

Reuse `lat_arr` for the existing `lat=("lat", …)` coord assignment (line ~141).

**Test-fixture migration (required — the guard breaks current happy-path tests):**
`tests/sfno_inference/test_nc_writer.py` success cases pass **ascending** `np.linspace(-90, 90, H)`
lat (lines 68, 109, 140, 207, 227). Convert these fixtures to **descending**
(`np.linspace(90, -90, H)`) so they exercise the contract, then add one **ascending-lat rejection**
test asserting `write_rollout_nc` raises. (The existing wrong-length tests at lines 167+ still raise
on the length check first, so they are unaffected.)

**Recurrence scope (narrowed).** This guard covers **only** the own-track `eval_inference.py` path,
since that is the sole writer routed through `write_rollout_nc`. Other direct `to_netcdf` writers
exist and are **out of scope** for this fix; their lat sources differ, so no single blanket claim
applies (verified):
- `scripts/convert_group_inference_to_aires_nc.py` / `convert_group_long_inference_to_aires_nc.py`
  — lat from the **source h5** (descending, correct).
- `src/sfno_training_group/score_function/group_emulator.py:save_rollout_netcdf` — lat from
  `params.lat` (the yaml `PLASIM.lat`, descending).
- `src/sfno_inference_5410/score_adapter.py:171` — lat from the **raw 5410 NetCDF** (orientation
  follows that source file, not an h5).

None are *fixed* by this plan; whether to add a shared assertion across them is the deferred §6 Q4.

### 3.3 (Optional) Backfill existing flipped outputs — new `scripts/relabel_inference_lat.py`

A small, idempotent one-shot that repairs already-written NetCDFs **without re-forecasting**:

- Open each target `*.nc` with `netCDF4` in `r+`.
- Detect the flip: if `lat[0] < lat[-1]` (ascending) → **reverse only the `lat` variable** in place,
  leave all data variables untouched. (This matches the relabel — *not* a data re-sort — because the
  data rows are already correct; only the label is wrong.)
- Idempotent: if `lat` is already descending, skip and log.
- Dry-run by default (`--apply` to write); log every file's before/after `lat[0], lat[-1]`.
- Cross-check after write: `.sel(lat=+70)` Z500 ≈ NH value as a sanity gate (warn if not).
  Resolve the Z500 channel **adaptively** via `scripts/_eval_utils.py:detect_z500_channel(channel_names)`
  rather than hard-coding `zg500` (v9 vs v10 differ); or, if scoping the backfill to v10 only,
  reject non-v10 roots loudly. Do not assume a fixed channel name across "every run".

Run only against output roots that may feed a coordinate-aware scorer.

### 3.4 (Optional) Correct the issue doc

Add a one-paragraph correction to `docs/2026-06-02_eval_inference_latitude_flip.md` §3 noting the
flip originates in the torch_harmonics fallback (top-level `lat` lookup always misses;
`coords/lat` is descending/correct), so future readers aren't misdirected to the metadata.

---

## 4. Step-by-step execution

```
0. Preflight the smoke env: pick a FRESH empty OUT_ROOT/RUN_TAG (never a production root); pin CKPT
   explicitly (EMA `best_ckpt_ema_mp0.tar` vs raw `best_ckpt_mp0.tar`) and reuse the SAME CKPT for
   pre- and post-fix runs                                  → verify: out dir empty; CKPT path echoed
1. Edit _read_lat_lon_from_run + both callers (3.1)        → verify: Claim A repro returns descending
2. Add nc_writer.py guard + migrate test fixtures (3.2)    → verify: pytest test_nc_writer.py green; ascending raises
3. Re-run a 1-IC smoke (`--limit-files 1 --limit-ics 1`)   → verify: .sel(lat=+70) Z500 ≈ ~5500 (NH)
4. Score it with an EXISTING same-family --clim-nc          → verify: gate metrics match the SAME-CKPT, SAME-SUBSET pre-fix baseline (no regression)
5. (Optional) write + run relabel_inference_lat.py --apply → verify: patched files read correct by coord; data bytes unchanged
6. (Optional) correct issue doc §3                         → verify: prose matches the empirical mechanism
```

> **Smoke-env preflight (P1-b/P1-c).** `eval_inference.py` writes deterministic filenames into an
> existing `inference/nwp` dir (line 277) and `score_nwp.py` scores **every** `*.nc` in that dir
> (line 318) — so a 1-IC smoke in a production root silently overwrites one file and scores a mixed
> set. The `submit_eval_prelude.sh` rerun guard (line 153) protects only the wrapper path and is
> bypassed by direct script use or `ALLOW_RERUN=1`. Always use a fresh root for the smoke. Likewise,
> "metrics unchanged" is only meaningful if the pre- and post-fix runs use the **identical** `CKPT`
> — the prelude prefers EMA while the inline fallback defaults to raw, so pin it. Because the smoke
> uses `--limit-files 1 --limit-ics 1`, both baselines must score the **same single (file, IC)**
> subset.
>
> **Climatology reuse (avoid recompute).** Direct `score_nwp.py` requires `--clim-nc`
> (`score_nwp.py:63`); the score wrapper (`eval_run_score_inline.sh:29`) will **build a full
> climatology if one is missing** — costly and pointless for a smoke. Reuse an existing same-family
> `climatology_proleptic.nc` by passing it via `--clim-nc` (or symlinking it into the fresh root);
> only rebuild if that compute is intentional.

## 5. Risks & mitigations

- **Signature change ripples:** `_read_lat_lon_from_run` is internal to `eval_inference.py`; only two
  callers. Grep to confirm no other importers before editing.
- **A real grid mismatch surfacing as a hard failure:** intended — fail loud beats silent corruption.
  The cross-check compares value-*sets* (orientation-agnostic), so it only fires on a genuine
  wrong-data/wrong-run pairing, which should never pass eval anyway.
- **Backfill on the wrong files / double-apply:** mitigated by idempotent ascending-detection +
  dry-run default + explicit `--apply` + per-file logging.
- **Hidden coordinate-aware consumer we missed:** the guard (3.2) plus the descending-by-contract
  invariant make any future flip a hard error rather than silent.

## 6. Open questions for the reviewer

1. **Keep or drop the fallback?** Preference: drop it (h5 always present in eval). If kept for
   non-eval reuse, it must emit **descending** lat + assert. Which do you want?
2. **Backfill scope:** do any existing `eval_inference.py` NWP/climate NetCDFs feed a coordinate-aware
   scorer, i.e. is §3.3 needed now, or is the forward fix (3.1+3.2) sufficient?
3. **Cross-check strictness:** assert value-set equality at `round(…, 2)` tolerance — acceptable, or
   prefer `np.allclose` on sorted arrays with an explicit atol?
4. **Shared lat-orientation assertion?** The `write_rollout_nc` guard covers only the own-track
   eval writer; the other direct `to_netcdf` writers (see §3.2) are unguarded. Options: (a) leave
   them — they source from h5/yaml/raw-NC and are not part of this bug; (b) extract a shared
   `assert_descending_lat(lat)` helper and call it in the **degree-coordinate** schema writers for
   defense-in-depth. **Exclude `compute_climatology.py`** from any such helper: it intentionally
   writes `lat=np.arange(args.H)` (integer positional indices, `compute_climatology.py:130`), so a
   descending-degrees assertion would either fail or force an unrelated schema change. Which do you
   want?
