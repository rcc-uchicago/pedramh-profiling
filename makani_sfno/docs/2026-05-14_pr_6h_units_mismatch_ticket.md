# 2026-05-14 — Ticket: `pr_6h` units mismatch between our v11 stats and SFNO-5410's stats

Author: Claude (with Zhixing)
Status: open — diagnostic ticket, NOT a training/HPO run.
Scope: investigate and document; **no behavior change required for the
v11_clip warmstart continuation diagnostic** (separate plan at
`docs/2026-05-14_v11_clip_warmstart_continuation_plan.md`).

---

## 1. Observation

The 2026-05-14 channel-by-channel normalization comparison showed every
state channel matches within ≤ 1 % on std and ≤ 0.25 σ on mean — **except
`pr_6h`**, which differs by roughly 4 400 ×:

| stat | ours (v11 packager, yr 12–111) | 5410 (`data_12-132_mean_sigma.nc`, yr 12–132) | ratio |
|---|---:|---:|---:|
| mean | 1.6995 e-07 | 6.1137 e-04 | 0.000278 (× 3 598) |
| std | 2.8087 e-07 | 1.2277 e-03 | 0.000229 (× 4 372) |

Source files:
- Ours: `/scratch/.../sim52_zgplev_full_v11/stats/global_means.npy`,
  `global_stds.npy` (index 52 = `pr_6h`).
- 5410: `/scratch/.../derecho_glade/sim52/h5/sigma_data/data_12-132_{mean,std}_sigma.nc[pr_6h]`.

## 2. Why this matters (and doesn't)

**Doesn't matter for the zg500 6 h RMSE gap.** Each pipeline trains and
evaluates with its own normalization stats; the model never sees absolute
units. The pr_6h discrepancy is irrelevant to zg500 / tas skill.

**Does matter for**:
1. **5410-benchmark overlay comparisons of pr_6h.** The 5410-benchmark
   overlay rendered by `scripts/render_eval_figures.py` and the
   `nwp_scorecard_summary.csv` rollup in `report.md` display
   pr_6h numbers side-by-side. If the unit convention diverges by 4 000 ×,
   any direct numerical comparison is meaningless.
2. **The note in `project_5410_eval_track`** which asserts that "group
   conventions `pl=ln(p_s)`, `zg=gpm`, `pr_6h=rate × 6h` must NOT be
   converted." If 5410's pr_6h is in different physical units than ours,
   the note is incomplete or our packager output is in a different
   convention than we believe.

## 3. Candidate explanations (untested)

1. **Our packager doesn't actually integrate × 6h.** Maybe the postprocessor
   emits `pr_6h` as a 6-hour-window-average rate in kg m⁻² s⁻¹ (≈ 1.7 e-07
   on average over a 100-yr PlaSim run is the right order of magnitude for
   global precip rate). 5410's std 1.23 e-03 in kg m⁻² per 6h
   = 1.23 mm / 6h is the right order of magnitude for precip accumulation.
   In that case `5410_std / our_std = 21600 s/6h * 1 kg/m²/(kg/m²/s)` → 21 600
   if it were a pure unit conversion. We see 4 372. So a pure
   "rate-in-s vs accumulation-in-6h" conversion doesn't explain the ratio
   exactly; there's an additional ~5× factor unaccounted for.
2. **5410 stores pr_6h with a density / mass-flux conversion** the source
   PanguPlasim pipeline applied (e.g., m water-equivalent vs kg/m²,
   1 mm = 1 kg/m² but PlaSim's native unit might be m/s or m/6h, and 5410
   may have multiplied by ρ_water = 1 000 kg/m³ at some stage). Our
   packager would then NOT be doing the same conversion.
3. **Stats were computed on different masked subsets.** PlaSim emits some
   pr_6h = 0 cells often; if 5410's stats are over a different
   cell-mask (e.g., land-only, or precipitation-event-only), the std
   would be much larger than ours computed over all-pixels-all-times.
4. **PlaSim version difference in `pr_6h` diagnostic definition.** The
   2100-year derecho rerun may use a slightly different precipitation
   accumulation convention than what our `plasim_postprocessor` extracts.
   Unlikely (both are sim52), but worth confirming.

## 4. Recommended investigation (cheap, non-training)

All steps are read-only and don't require any new compute.

### Step 1: Look at a single matched h5 raw sample

Read one raw h5 from each pipeline at a matched (year, day, hour) and
compare the `pr_6h` field byte-for-byte.

```bash
# Our v11 raw h5 sample
ls $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full_v11/train/ | head -3
# 5410 raw h5 sample
ls /scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/h5/sigma_data/ | grep -E '^[0-9]+_' | head -3
```

Then in python:
```python
import h5py
import numpy as np

with h5py.File('<our h5>', 'r') as f:
    our_pr = f['fields_diagnostic'][:, 0]  # pr_6h is index 0 in diagnostic
print('ours:', our_pr.shape, 'mean=', our_pr.mean(), 'std=', our_pr.std(), 'max=', our_pr.max())

with h5py.File('<5410 h5>', 'r') as f:
    # 5410's h5 layout: check the dataset names
    print(list(f.keys()))
    # Find pr_6h in the appropriate group
```

If the raw arrays differ by ~4 400 ×, the units gap is in the source data,
not in stats computation. If they're similar, the gap is in how
`data_12-132_mean_sigma.nc` was computed.

### Step 2: Check packager docstrings + postprocessor source

```bash
grep -rnE "(pr_6h|precip|kg.*m.*s|kg.*m.*6h)" \
    src/plasim_makani_packager/ \
    src/plasim_postprocessor/ 2>/dev/null | head
```

Confirm what our `pr_6h` channel is physically (rate × 6h, rate, depth,
mass, energy, ...).

### Step 3: Cross-check with the 5410 source tree

Inspect `data_loader_multifiles.py` in the group's training repo for how
`pr_6h` is consumed at training time. If they re-scale by std read from
`data_12-132_std_sigma.nc[pr_6h]` (= 1.23 e-03), then their model expects
the raw h5 values to be on the same scale as that std.

## 5. Action items (small, can do without a training run)

- [ ] Step 1 above: byte-compare raw `pr_6h` at one matched timestamp.
- [ ] Step 2: confirm what physical quantity our packager emits for
      `pr_6h` (rate × 6h? rate? accumulation?).
- [ ] Document the resolution in `project_5410_eval_track` memory note —
      either the convention note is correct and the stats discrepancy is
      explained by something else, or the note needs a unit qualifier
      added.
- [ ] If 5410's `pr_6h` is in different absolute units than ours, then the
      5410-benchmark overlay in `report.md` and `figures/` should:
      either convert at render-time, or flag the columns with a unit
      annotation, or drop pr_6h from the cross-track comparison entirely.
- [ ] Update `scripts/render_eval_report.py` and
      `scripts/render_eval_figures.py` accordingly. Likely a < 50-line
      change once the convention is confirmed.

## 6. Explicit non-scope

- This ticket does **not** propose retraining anything. The pr_6h units
  question is a reporting / interpretation issue, not a model-quality issue.
- This ticket does **not** affect the v11_clip warmstart continuation
  diagnostic; the warmstart experiment can proceed independently.
- Per `project_5410_eval_track`, do NOT silently re-scale our pr_6h to
  match the 5410 stats without auditing what each side's units actually
  represent — silent rescaling would hide the convention question rather
  than resolve it.

## 7. Cross-references

- `docs/2026-05-14_v11_clip_warmstart_continuation_plan.md` — the parent
  forensic which surfaced this discrepancy.
- `project_5410_eval_track.md` memory note — the convention claim that
  needs verification.
- `src/plasim_makani_packager/channels.py:60` — `DIAGNOSTIC_CHANNELS =
  ["pr_6h"]`.
- `src/plasim_postprocessor/` — our pr_6h source.
- `/scratch/.../derecho_glade/sim52/h5/sigma_data/data_12-132_{mean,std}_sigma.nc` —
  5410 stats source.
