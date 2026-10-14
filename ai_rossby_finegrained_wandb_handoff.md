# HANDOFF — give ai-rossby the same per-variable/per-level wandb detail as Pangu

Read this whole file before touching code. Work in
`/eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling`, branch
`fix/tsoi-fill-270` (or a fresh branch off it).

**Do not implement this yet.** Both production runs are live (Pangu `7368237` on
`capacity`, ai-rossby `7368547` on `preemptable`, SFNO-E3SM parity comparison —
`polaris_sfno_comparison_handoff.md`). Editing the source files below is safe —
Python has already loaded the running processes' modules into memory, so editing
`train.py`/`train_loop.py` on disk does not touch either live job — but the new
code only takes effect on the **next** launch. **Land this when the queue
rotation happens** (Pangu's capacity link ends near epoch ~50 on its 168 h wall;
ai-rossby's capacity companion `7368536` is held behind it — see
`polaris_pbs_notes.md` §1b and the live CHANGELOG entries from 2026-08-08/09).
That restart is the natural point to also flip in this change, smoke-test it, and
launch the next link with it.

## 1. Why (the gap this closes)

Both harnesses run the same SFNO-E3SM parity architecture (bitwise param-count
match, `compare_sfno_parity.py`) and log to the same wandb project
(`pedramh-profiling`), but their metric schemas diverge in both **naming** and
**granularity**:

| | Pangu (`PanguWeather/v2.0/train.py`) | ai-rossby (`physicsnemo_ai_rossby/.../train.py`) |
|---|---|---|
| key scheme | flat (`wandb.log({...})` direct calls) | namespaced (`LaunchLogger` prefixes `f"{name_space}/{key}"`) |
| train loss | `train_loss` | `train/loss` |
| valid loss | `valid_loss` | `valid/val_loss` |
| granularity | **one metric per variable, per pressure level** — `train_{var}_lwrmse`, `train_{var}_level{level:.4f}_lwrmse` (~101 keys every iteration) | **four aggregate buckets** — `surface`, `upper_air`, `diagnostic`, `vae_kl` |

Even the aggregate quantities can't be overlaid in wandb today because the key
strings differ. And ai-rossby has no way to answer "how well does the model
predict `T` at 500 hPa" at all — that number is never separated out.

`vae_kl` reading `0.000e+00` every step is **not a bug** — confirmed at
`train_loop.py:346`: SfnoPlasim has no VAE head, so it's a structural placeholder
shared with a different model variant in this codebase. Leave it alone.

## 2. Ground truth, verified directly (don't re-derive from memory)

### 2a. The variable lists are already fully aligned — no remapping table needed

Pulled from the two **actually-running** rendered configs, not from any older
summary:

- Pangu production:
  `runs/pangu_sfno_alldata_full/E3SM_SFNO_H5_POLARIS_ALLDATA.full.rendered.yaml`
- ai-rossby production:
  `physicsnemo_ai_rossby/examples/weather/ai_rossby/conf/model/sfno_e3sm_parity.yaml`

Pangu's YAML splits `surface_variables` (6) from `land_variables` (2) for
readability, but **at runtime they're concatenated**:
`utils/data_loader_multifiles.py:457`:
```python
self.surface_variables = self.surface_variables + params.land_variables
```
So `current_dataset.surface_variables` used by the diagnostic block below is the
full 8-item list, **same order** as ai-rossby's folded 8-item list:

| group | names (identical order, both harnesses) |
|---|---|
| surface (8) | `TREFHT, U10, RHREFHT, PS, PSL, TMQ, SOILWATER_10CM, TSOI_10CM` |
| upper-air (5 × 18 levels) | `T, U, V, Z3, RELHUM` × the same 18 hybrid-pressure levels in both configs |
| diagnostic (3) | `FSNTOA, FSNT, PRECT` |

8 + 5×18 + 3 = **101** — matches the parity check's `out_chans`. Use these exact
strings and this exact order; no cross-harness name mapping is required.

### 2b. Pangu's exact formula (`PanguWeather/v2.0/train.py`)

Weight (`train.py:141-144`):
```python
def latitude_weighting_factor_torch(latitudes):
    lat_weights_unweighted = torch.cos(3.1416/180. * latitudes)
    return latitudes.size()[0] * lat_weights_unweighted / torch.sum(lat_weights_unweighted)
```
Per-channel(-level) RMSE (`train.py:147-157`), reduced over lat/lon only:
```python
def weighted_rmse_torch_channels(pred, target, latitudes):   # pred/target: (n, c, h, w)
    weight = torch.reshape(latitude_weighting_factor_torch(latitudes), (1, 1, -1, 1))
    return torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1, -2)))   # -> (n, c)

def weighted_rmse_torch_3D(pred, target, latitudes):         # pred/target: (n, c, l, h, w)
    weight = torch.reshape(latitude_weighting_factor_torch(latitudes), (1, 1, 1, -1, 1))
    return torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1, -2)))   # -> (n, c, l)
```
This runs on **normalized-space** `output`/`target` (the same tensors the loss is
computed on), inside a `torch.no_grad()` block placed **after the loss/optimizer
step and after `self._epoch_telemetry.step_end()`** (`train.py:1218-1226`,
comment: *"this block is per-iteration diagnostics (RMSE + wandb)... That block
has no counterpart inside ai-rossby's window, and including it here would make
the two harnesses' step times incomparable."*). It then denormalizes by
multiplying by the per-variable std (`train.py:1677,1680,1684`):
```python
diagnostic_logs[f'train_{var}_lwrmse'] = torch.mean(surface_lwrmse[:, j]) * current_dataset.surface_std[j]
diagnostic_logs[f'train_{var}_lwrmse'] = torch.mean(diagnostic_lwrmse[:, j]) * current_dataset.diagnostic_std[j]
diagnostic_logs[f'train_{var}_level{level:.4f}_lwrmse'] = torch.mean(upper_air_lwrmse[:, j, k]) * current_dataset.upper_air_std[j, k]
```
This is mathematically valid because for z-scored `pred_z = (pred_phys - mean) /
std`, the residual `pred_z - target_z = (pred_phys - target_phys) / std` — so
`std * RMSE(pred_z, target_z) = RMSE(pred_phys, target_phys)` exactly; no need to
denormalize the full tensors first, just scale the scalar result.

**Runs every iteration, unconditionally** (no `% freq` gate) whenever `not BENCH`.
This is a real, measured per-step cost on the production run (not reflected in
the bench numbers, which skip this block entirely) — expect ai-rossby's
equivalent to add similar overhead once wired in.

**Validation-side per-variable metrics are out of scope for this handoff.** Pangu
only emits `valid_{var}_bias_lwrmse` / `valid_{var}_level{level:.3f}_bias_lwrmse`
when `long_validation=True` (`train.py:2661-2670`), and the running production
config has it **off** — `long_validation: False  # smoke: bias .npy files not
staged on Polaris` (`E3SM_SFNO_H5_POLARIS_ALLDATA.full.rendered.yaml:118`).
There is currently nothing on Pangu's own dashboard to match on the validation
side. If that changes later (bias `.npy` staged, `long_validation` flipped on
Pangu), extend this work then — don't build it now against a target that isn't
live.

### 2c. ai-rossby's existing building blocks (`physicsnemo_ai_rossby/`)

- `examples/weather/ai_rossby/loss.py` already has `cos_lat_weights(num_lat,
  device, dtype)` and `lat_weighted_residual`/`per_var_lat_weighted_residual` —
  but both **reduce to a scalar** (`.mean()` over channels too). Neither returns
  a per-channel tensor. You need a new function, not a reuse of these, mirroring
  `weighted_rmse_torch_channels`/`_3D` above: same `sqrt(mean(weight *
  (pred-target)**2, dim=(-1,-2)))` shape, but keep the channel (and level) axis.
- `physicsnemo/experimental/datapipes/climate/transforms.py`'s
  `ClimateNormalizer` already exposes `.surface_std`, `.upper_air_std`,
  `.diagnostic_std` as buffers in the same channel order as the config's
  variable lists (`transforms.py:145,161,177`) — same shape and role as Pangu's
  `current_dataset.surface_std[j]` / `.upper_air_std[j,k]` /
  `.diagnostic_std[j]`. Reuse these directly; don't add a second stats source.
- `LaunchLogger` (`physicsnemo/utils/logging/launch.py`) prefixes every key with
  `f"{name_space}/{key}"`. If you route the new metrics through
  `log.log_minibatch(...)`, they'll come out as `train/{var}_lwrmse` —
  **not** `train_{var}_lwrmse`. Decide explicitly (§4) whether to accept the
  `train/` prefix mismatch or call `wandb.log(...)` directly for this block to
  get byte-identical keys with Pangu.

## 3. ⚠ Required verification before trusting any new number

**The latitude weighting is not yet confirmed equivalent.** Pangu's
`self.latitudes` is `torch.from_numpy(np.array(self.params.lat))` — a real grid
array (its ultimate source wasn't pinned down during this handoff; the obvious
in-code candidates are commented out at `data_loader_multifiles.py:526,1008`, so
trace `params.lat`'s actual assignment before relying on it). ai-rossby's
`cos_lat_weights` instead **synthesizes** an idealized equiangular grid:
```python
phi = torch.linspace(pi/2 - pi/(2N), -pi/2 + pi/(2N), N)   # cell-centered, pole-to-pole
```
If E3SM's real grid isn't exactly this formula (e.g. a different pole offset or
a non-uniform spacing), the two harnesses' "physical unit" RMSE numbers carry a
small systematic mismatch even when everything else is right — silently, since
both would still look like sane physical values. **Before merging:** dump
Pangu's actual `self.params.lat` array once (any smoke run, `log_to_screen`) and
diff it numerically against `cos_lat_weights`'s implied `phi` grid (converted
back to degrees). If they match to float precision, reuse the existing
`cos_lat_weights` as-is. If they don't, add a `lat_values` path to the new
function that takes real latitudes and reuses the *Pangu* weight formula (also
copy it verbatim — the `N * w / sum(w)` normalization is subtly different from
`cos_lat_weights`'s `w / w.mean()` only in this edge case: they're actually
equal, since `sum(w)/N == mean(w)`, but verify this once rather than assume).

## 4. What to build

1. New function in `ai_rossby`'s loss/metrics path (`loss.py` or a new
   `diagnostics.py` next to it — don't overload `lat_weighted_residual`, its
   contract is "returns a scalar" and callers rely on that):
   ```python
   def per_channel_lat_weighted_rmse(pred, target, lat_weights) -> torch.Tensor:
       # pred/target: (n, c, [l,] h, w) -> returns (n, c) or (n, c, l)
       # mirrors weighted_rmse_torch_channels / weighted_rmse_torch_3D exactly
   ```
2. In `train.py`, inside a `torch.no_grad()` block placed **after** the training
   step and **after** `telemetry.step_end()` (mirror Pangu's placement and its
   comment verbatim — this is the CLAUDE.md #10 contract: per-harness step-time
   comparisons must stay apples-to-apples), compute:
   - `per_channel_lat_weighted_rmse` on the normalized `output`/`target` for
     surface, upper-air, and diagnostic groups
   - multiply by `normalizer.surface_std[j]`, `.upper_air_std[j,k]`,
     `.diagnostic_std[j]` respectively
   - log with **Pangu's exact key strings**: `train_{var}_lwrmse` for surface
     and diagnostic vars, `train_{var}_level{level:.4f}_lwrmse` for upper-air
     (same `.4f` precision, same level values — the configs already share the
     literal level list, so the formatted strings will match character-for-
     character)
3. Decide the namespace question from §2c explicitly: call `wandb.log({...})`
   directly (bypassing `LaunchLogger`'s prefix) so keys land identically to
   Pangu's and the wandb UI can overlay both runs on one panel. Using
   `log.log_minibatch` instead is easier but produces `train/{var}_lwrmse`,
   which won't merge with Pangu's `train_{var}_lwrmse` panel — pick on purpose,
   don't default into it.
4. Gate nothing behind a frequency check unless you also add one to Pangu's
   side — right now both should run every iteration to stay comparable. If the
   added wandb-call volume (~101 keys × 10,950 steps/epoch) turns out to be
   expensive, that's a joint change to both harnesses, not a unilateral one
   here (rule #10: knobs are per-project, but the contract itself isn't).

## 5. Before it ships (CLAUDE.md #6, #10, #11)

- **Smoke test both harnesses** after the change — a short run, confirm the new
  wandb keys appear with sane physical-unit magnitudes (e.g. `TREFHT` in
  Kelvin-ish range, not O(1) normalized units — an easy way to catch a missed
  `* std` multiply).
- **Cross-check against the existing aggregate loss.** `train_loss` (Pangu) and
  the sum/mean of the new per-var panels should be in the same ballpark as the
  aggregate `surface`/`upper_air`/`diagnostic` buckets ai-rossby already logs —
  not bit-identical (different reduction: RMS-of-mean vs the training loss's own
  reduction), but same order of magnitude. A mismatch of 10x+ means a units bug
  (missed or double-applied `std`), not a modeling difference — trace it, don't
  wave it off (rule #11).
- **Confirm §3's latitude-weighting check** before treating the new panels as
  authoritative for any physical claim in a write-up.
- Update `epoch_telemetry_test.py`'s drift assertion if this touches the
  duplicated `epoch_telemetry.py` files — it shouldn't (this is a separate
  per-iteration wandb block, not the epoch telemetry CSV), but confirm before
  committing.
