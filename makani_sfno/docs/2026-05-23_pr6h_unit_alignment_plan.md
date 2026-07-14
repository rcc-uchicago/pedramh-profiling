# pr_6h Cross-Track Unit Alignment Plan

**Date:** 2026-05-23
**Author:** Zhixing Liu (with Claude)
**Status:** Draft — Codex rounds 1-6 applied + self-audit class-sweep
**Supersedes:** `docs/2026-05-14_pr_6h_units_mismatch_ticket.md` — that
ticket flagged the symptom (pr_6h unit mismatch between v11 stats and
5410's stats) and deferred resolution; this plan provides the resolution
(report-side suppression with banner) and the upstream-bug citation
that explains the root cause.

## §1. Background

The own-track NWP scorecard table now overlays a "5410 benchmark" row for
each channel × lead — added at `scripts/render_eval_report.py:174` inside
the shared RMSE/ACC loop (`_render_table`, lines 128-189). A second
benchmark-row append at `:220` is part of the `_render_masked_tas`
helper and is `tas_no_ice`-only; it is **not** in scope for this plan.
For tas, zg500, ua5, ta5 this lets us compare RMSE/ACC directly. For
`pr_6h` the report currently disclaims comparability:

> "5410 benchmark values are in the group's native units (no unit
> conversion); for `pr_6h` this is `kg m^-2` per 6h, so the 5410 row is
> not directly comparable to own-track `pr_6h` (m s^-1)."
> — `scripts/render_eval_report.py:158-160`

That disclaimer is half right (the unit *names* differ) and half wrong
(the disclaimer assumes the 5410 emits clean physical values; it does
not). Evidence from the 2026-05-22 own-track run + 2026-05-09 5410
benchmark `inference/nwp/Y121_s0000.nc`:

```
        own truth (m/s)   5410 truth (m/6h)   5410 PREDICTION (?)
median  7.89e-08          1.14e-04            -319
mean    1.74e-07          6.14e-04            -0.18
max     6.33e-06          2.72e-02            16,840
```

Two distinct issues:

1. **Truth-side unit gap.** 5410's truth is `pr_6h` in the group's
   "6-hour precip proxy" convention — `instantaneous_pr_rate(t) × 6h`
   per `docs/2026-05-06_group_sfno_5410_eval_plan.md:127` (which
   explicitly warns *not* to describe it as "6-hour accumulated
   precipitation"); own's truth is `pr_6h` as an instantaneous rate
   in m/s. The nominal convention factor is 21,600 s/6h, but the
   empirically observed truth-stats ratio is ~3,600-4,400× (mean
   3,598×, std 4,372× per
   `docs/2026-05-14_pr_6h_units_mismatch_ticket.md:11-25`) — there is
   a ~5× unexplained gap on top of the unit factor. The
   [[project-5410-eval-track]] memory note that "pr_6h = rate × 6h"
   captures the convention but is not sufficient to define a clean
   scalar conversion between own and 5410 truth. This is exactly why
   §2 chooses suppression rather than scalar conversion.

2. **Prediction-side scale chaos.** 5410's emitted prediction has range
   [-484, +16840] with median -319. Own's prediction range matches its
   truth tightly (median 7.94e-8 vs truth 7.89e-8). High pr_6h ACC at
   short lead on 5410 (0.82 @ 6h) confirms the spatial *pattern* is
   right; only the absolute magnitude is off by ~10^6.

The cause of (2) is now narrowed by code audit
(`scripts/infer_sfno5410_blocking_h100_packed.py:348-349`):

```python
diagnostic_prediction[:, step - 1] = (
    stepper.dataset.diagnostic_transform(out_diagnostic.detach().cpu()).numpy()
)
```

This calls the **forward** z-score transform on the diagnostic
prediction — asymmetric with surface/upper_air at lines 343/346 which
use `*_inv_transform`. The upstream loader defines both
`diagnostic_transform` (`utils/data_loader_multifiles.py:705`) and
`diagnostic_inv_transform` (`:722`); the inference script is using the
wrong one for the diagnostic channel. Net effect: pr_6h prediction is
written in **forward-z-scored space** while truth is in physical
m/6h — which exactly matches the empirically observed scale chaos
(prediction range ±10⁴, truth range 0–0.03).

Whether `out_diagnostic` coming out of the stepper is itself already
in z-space (and the forward call is a double-transform) or in physical
space (and the forward call is the only transform) needs §3 to decide
— but **both possibilities produce non-physical writes**.

## §2. Goal

Make the own-track report.md's `pr_6h` cross-track comparison
**scientifically honest** rather than misleadingly numeric.

After Codex round-1 review surfaced a P0 (Phase 2 below), the user
chose **suppression** as the resolution: the 5410 pr_6h row is dropped
from the cross-track RMSE *and* ACC tables, with a banner citing the
forward-z-score-transform anomaly at
`infer_sfno5410_blocking_h100_packed.py:348-349`. The own-track pr_6h
row stays in its native m/s — readers who want a human-readable
precip number can multiply by `21600 * 1000` for mm/6h (≡ kg/m²/6h) in
their head, but the table doesn't fake the comparison.

**Why suppression over scalar conversion:** RMSE and ACC are field
functionals (`sqrt(E[(pred - truth)²])` and anomaly correlation), not
scalar-recoverable from the already-IC-averaged CSV. Per-IC RMSE and
ACC are computed field-by-field at `score_nwp.py:160` (RMSE) and
`:172` (ACC), then IC-averaged and written to
`scores/nwp_scorecard_summary.csv` by `_summarize` at
`score_nwp.py:238-253`. No post-hoc scalar on those mean/std
values can produce the metric you'd get from `RMSE(f(pred), truth)`
when `f` is an inverse transform that mixes shift and scale (which
zscore inversion does). The renderer would either need to recompute
metrics from the NetCDF fields (heavier and outside the
report-only-WHERE constraint) or to suppress — user chose the latter.

The fix lives at **report rendering time only**
(`scripts/render_eval_report.py`). Score CSVs and figures remain in
native units. This bounds blast radius and preserves the
"5410-on-disk-is-sacred" invariant per [[project-5410-eval-track]].

Permanent integration — every future own-track NWP eval that overlays
5410 will have the suppressed pr_6h cross-track row, with a banner.
(Per user's 2026-05-23 interview: "Permanent (Recommended)".)

## §3. Phase 1 — banner-text audit (suppression path)

Suppression doesn't need a working inverse transform. It only needs
**enough evidence to justify the banner text** so a reader can trust
the suppression decision (and so a future maintainer who wonders
"can't we just multiply by something?" finds the answer in code).

### 3.1 Cite the forward-transform anomaly

Confirm and cite in a doc-comment in `render_eval_report.py`:
- `scripts/infer_sfno5410_blocking_h100_packed.py:343` —
  `surface_inv_transform(input_surface)` (correct direction).
- `:346` — `upper_air_inv_transform(input_upper_air)` (correct).
- `:348-349` — `diagnostic_transform(out_diagnostic)` (**forward**,
  asymmetric — root cause for the packed-benchmark path this report
  consumes).
- `scripts/infer_sfno5410_byo_ic.py:425-432` — same pattern (BYO
  path also calls `surface_inv_transform`, `upper_air_inv_transform`,
  then `diagnostic_transform` forward), so any BYO-generated 5410
  prediction NetCDF carries the same anomaly. Cited for maintainer
  awareness; the report's benchmark overlay path is the packed one
  above.

### 3.2 Cite the upstream loader symmetry

Confirm both transform directions exist in the upstream loader, so the
anomaly is "wrong direction chosen", not "no inverse available":
- `/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_blocking/source_trees/forecast_modules/PanguPlasim/utils/data_loader_multifiles.py:705`
  — `diagnostic_transform(data)`
- `:722` — `diagnostic_inv_transform(data)`

### 3.3 Empirical signature

Cite the truth-vs-prediction scale comparison from §1 (own truth and
5410 truth observably differ by ~3,600-4,400× per
`docs/2026-05-14_pr_6h_units_mismatch_ticket.md:11-25` — not cleanly
21,600, leaving a ~5× unexplained gap; 5410 prediction off by ~10⁶ on
top of that) as the empirical proof that the on-disk 5410 prediction
is in a transformed space.
This evidence is independent of the upstream code audit and survives
even if upstream details change.

**Belt-and-suspenders check (cheap):** confirm the empirical signature
on a second 5410 NetCDF (e.g. `Y121_s0122.nc`) so we aren't acting on
a `Y121_s0000.nc`-only artifact. **Scope is within-5410 only** — open
the second 5410 file and check that `truth pr_6h` has the same order
of magnitude as the first file (~1e-4 m/6h) and that `prediction
pr_6h` has the same wildly-out-of-physical-range signature
(median ≪ 0, max ≫ 10³). This avoids any need to match own-track
IC offsets, which are generated by `nwp_ic_offsets()` in
`src/sfno_inference/rollout_driver.py:309` using a `(n_samples - K) //
n_ic` cadence that need not align with 5410's `IC_OFFSETS` constant in
`scripts/infer_sfno5410_blocking_h100_packed.py:45`. Fail-loud if the
second-IC 5410 prediction is suddenly in physical range — that would
mean the forward-transform anomaly is per-file (not global) and the
suppression should be conditional, not default.

### 3.4 Deliverable from Phase 1

A `_PR6H_SUPPRESSION_RATIONALE` constant or doc-comment in
`render_eval_report.py` with the three file:line citations above and a
one-paragraph banner text consumed by §5.3. **No µ/σ chase, no inverse
transform implementation, no field-level recompute** — all of that is
out of scope under the suppression decision.

### 3.5 What this phase does NOT do

- Does not extract `mean_diag`, `std_diag` from training stats.
- Does not implement `diagnostic_inv_transform` in Python here.
- Does not patch the upstream inference script.
- (The within-5410 second-IC scale check IS done — see §3.3
  belt-and-suspenders. A multi-IC own-vs-5410 truth-ratio audit is
  NOT done, because the 2026-05-14 ticket already shows a ~5×
  unexplained gap on top of the 21,600 unit factor — any scalar
  conversion would be unaudited regardless. Suppression sidesteps
  the entire question.)

If a future plan wants to UNSUPPRESS pr_6h (by either fixing the
upstream `diagnostic_transform` call or by adding a field-level
recompute in the scorer), it should restore these audit steps.

## §4. Suppression design (Phase 2)

### 4.1 Suppress 5410 pr_6h from the RMSE cross-track row

When `--track own` and `--benchmark-5410-out-root <path>` are both
set, prevent the `("emulator", "5410 benchmark", benchmark_summary)`
entry from being appended to `row_specs` at
`render_eval_report.py:173-174` for `ch == "pr_6h"`. Because the
formatted-row loop iterates `row_specs` (`:175`), suppressing the
append cleanly omits the row from the rendered RMSE table. The
emulator and persistence rows for pr_6h are unaffected.

### 4.2 Suppress 5410 pr_6h from the ACC cross-track row

Symmetric to §4.1, applied to the ACC table. **Reason (per Codex P1):**
ACC is anomaly correlation against a *physical* climatology
(`src/sfno_eval/metrics.py:169`, climatology fed in by
`score_nwp.py:167`). If 5410's pr_6h prediction is in
forward-z-scored space, the anomaly subtraction is dimensionally
incoherent — the resulting ACC value is not interpretable as "spatial
pattern match in physical anomaly space" even though the formula
mechanically produces a number in [-1, 1].

### 4.3 Banner text

Above the RMSE/ACC tables (or in the captions), emit:

> **Note on `pr_6h` cross-track comparison.** The 5410 benchmark row
> is suppressed for `pr_6h` for two compounding reasons:
> (a) **Prediction-side anomaly.** The upstream 5410 inference scripts
> apply the *forward* z-score transform to the diagnostic channel — at
> `infer_sfno5410_blocking_h100_packed.py:348-349` (packed benchmark
> path, which is what this report's overlay consumes) and equivalently
> at `infer_sfno5410_byo_ic.py:431` (BYO path) — asymmetric with the
> surface (`:343`) and upper-air (`:346`) paths that use the inverse.
> As a result the on-disk 5410 `pr_6h` prediction is in a transformed
> space and RMSE/ACC against the physical-units truth and climatology
> is not scalar-recoverable. (b) **Truth-side unit convention.** 5410
> truth uses the group's "6-hour precip proxy" `instantaneous_rate × 6h`
> (per `docs/2026-05-06_group_sfno_5410_eval_plan.md:127`), while own
> keeps the instantaneous rate (m/s). The nominal factor would be
> 21,600 s/6h, but the empirically observed truth-stats ratio is
> ~3,600-4,400× (per `docs/2026-05-14_pr_6h_units_mismatch_ticket.md`)
> — a ~5× unexplained gap on top, so even fixing (a) would still leave
> an unaudited scalar between own and 5410 `pr_6h`. Own-track `pr_6h`
> rows remain in their native m/s. See
> `docs/2026-05-23_pr6h_unit_alignment_plan.md`.

### 4.4 Own pr_6h row stays in native m/s

No conversion of own's `pr_6h` values. The own row remains in m/s
(value at 6h ≈ 6.5e-8 m/s). Readers wanting mm/6h can apply
`× 21600 × 1000` mentally. This avoids introducing a hand-edited
unit string in the table that would conflict with the CSV and the
figures-side rendering, which uses `× 86400 × 1000` to get mm/day
**for the own-track pr_6h bias maps only** (see
`render_eval_figures.py:46, :58`; note that pr_6h is excluded from the
line plots — see §5.5). Display-unit harmonization across the table
and the bias-map captions is a deferred follow-up; see §10.

## §5. Implementation (Phase 3) — at `render_eval_report.py`

### 5.1 New CLI flag

```
--pr6h-unit-align {suppress, none}
    Default: suppress.
    `suppress` drops the 5410-benchmark pr_6h row from both the RMSE
        and ACC cross-track tables and emits the §4.3 banner.
    `none` preserves today's behavior (5410 pr_6h row present, with
        the existing partial disclaimer at lines 158-160).
```

### 5.2 Suppression site

The existing renderer has a **single shared loop** `for metric in
("rmse", "acc")` (`render_eval_report.py:163`) that, for each channel,
builds one `row_specs` list per (metric, channel) iteration
(`:170-174`). `row_specs` entries are `(model_key, model_label, summary)`
— NOT `(channel, ...)` — because channel is the outer iterator
(`ch` from `:169`'s per-channel loop).

The conditional benchmark append lives at `:174`:

```python
if benchmark_summary is not None:
    row_specs.append(("emulator", "5410 benchmark", benchmark_summary))
```

Suppression is a one-line gate at that append site, parameterized by
the new `pr6h_unit_align` parameter, the current channel `ch`, and the
existing `--track` value:

```python
if benchmark_summary is not None and not (
    pr6h_unit_align == "suppress" and track == "own" and ch == "pr_6h"
):
    row_specs.append(("emulator", "5410 benchmark", benchmark_summary))
```

**Signature extension.** `_render_table` currently has
`def _render_table(summary, key_channels, *, benchmark_summary=None,
track="own")` (`render_eval_report.py:128`). Extend it with:

```python
def _render_table(
    summary,
    key_channels,
    *,
    benchmark_summary=None,
    track="own",
    pr6h_unit_align="suppress",
):
```

**Call-site update.** The single call site at
`render_eval_report.py:425` currently passes
`track=args.track` only — add `pr6h_unit_align=args.pr6h_unit_align`.

The **`track == "own"` clause is load-bearing.** The renderer already
has track-aware caption logic at `render_eval_report.py:149-160` and a
`--track {own, 5410}` CLI flag at `:87`. When `--track 5410` (used for
group_clone runs and direct 5410 evals), both rows are in matching
group-native units and ARE directly comparable; suppressing in that
case would be wrong. The guard is therefore specifically:
"the OWN row would mislead the reader when shown next to the 5410 row
because their units differ." That condition is false in 5410-track
mode.

Because the same `row_specs` is consumed in both the RMSE and ACC
inner formatting passes (the outer `for metric` loop at `:163`
reconstructs row_specs each pass but applies the same logic), **one
guard handles both tables**. No separate RMSE/ACC patches.

The `tas_no_ice` table at `_render_masked_tas` (`:210+`) has its own
`row_specs.append(("emulator", "5410 benchmark", ...))` but that
section is `tas_no_ice`-only and never sees `pr_6h`, so the guard does
not need to be replicated there.

### 5.3 Caption / banner injection

When `pr6h_unit_align == "suppress" and track == "own" and
benchmark_summary is not None`, replace the partial disclaimer at
`render_eval_report.py:158-160` with the §4.3 banner. Keep the
existing disclaimer when `pr6h_unit_align == "none"` so legacy
invocations stay unchanged. In `--track 5410` mode the existing
caption at `:151-156` (which states rows are directly comparable) is
correct and stays as-is.

**Also update three stale callouts** that currently advertise the
unconditional benchmark overlay:

- CLI help for `--benchmark-5410-out-root` at
  `render_eval_report.py:65-70` — currently says "the scorecard table
  gains a '5410 benchmark' model row per channel". Add a parenthetical
  note that this is suppressed by default for own-track `pr_6h` (per
  `--pr6h-unit-align suppress`).
- `_load_benchmark` banner at `:391-396` — the function signature is
  currently `_load_benchmark(bench_root: Path | None)` with no track
  or mode awareness (`:364`). Extend it to
  `_load_benchmark(bench_root, *, track, pr6h_unit_align)` and gate
  the suppression-mention text on `track == "own" and
  pr6h_unit_align == "suppress"` only. Update the call site at `:410`
  to pass `track=args.track, pr6h_unit_align=args.pr6h_unit_align`.
  In the suppress+own case the banner reads: "Side-by-side rows
  appear in the scorecard table (**5410 benchmark `pr_6h` row
  suppressed by default** — see the note above the table; own-track
  `pr_6h` rows remain in native m/s); bias maps overlay the benchmark
  in the figures job." In the 5410-track or `none` case the
  banner is the existing wording, **unchanged** — preserving the
  byte-for-byte `none`-mode promise of §6. (The "line plots" mention
  in the existing wording is pedantically inaccurate, but fixing it
  here would conflict with the byte-for-byte test; defer to a
  follow-up.)
- The disclaimer at `:158-160` is replaced by the §4.3 banner under
  the suppress+own condition; no other branch needs to change.

### 5.4 Thread the flag through `eval_run_report_inline.sh`

`scripts/eval_run_report_inline.sh:28` builds a fixed `REPORT_ARGS`
array and invokes the renderer at `:45`. To preserve env-override
ergonomics (so future fixed-5410 builds can flip to `none` without
editing the script per Codex round-2 P2), use the same
`${VAR:-default}` pattern the rest of the script already uses for
`TRACK`, `OUT_ROOT`, `CKPT`:

```bash
# After the existing `: "${TRACK:=own}"` line (~:20):
: "${PR6H_UNIT_ALIGN:=suppress}"

# In the REPORT_ARGS array construction:
REPORT_ARGS+=( --pr6h-unit-align "$PR6H_UNIT_ALIGN" )
```

Callers override via `PR6H_UNIT_ALIGN=none scripts/submit_eval.sh` or
by setting the env var in `bundled_eval.sh`. The default `suppress`
applies to all current and chained submissions.

### 5.5 No-touch zones

- `scripts/score_nwp.py` — unchanged. CSVs stay native.
- `scripts/render_eval_figures.py` — UNCHANGED. Note that `pr_6h` is
  already excluded from the **line plots** (`rmse_vs_lead.png`,
  `acc_vs_lead.png`) — the active filter is `LINE_PLOT_CHANNELS` at
  `render_eval_figures.py:33` (= `["tas", "zg500", "ua5", "ta5"]`,
  notably NO pr_6h) used at `:180, :187`. `REPORT_CHANNELS` at `:28`
  is a *broader* list that DOES include pr_6h, but it drives bias-map
  iteration only (`:611`). The "pr_6h omitted" annotation at `:278`
  documents the line-plot exclusion. The bias-map overlay (own mm/day
  vs 5410 transformed) also lies in the mixed-units regime; we treat
  that as a deferred follow-up (§10), not a blocker for the
  report-side suppression.
- 5410-side eval (`eval-sfno-5410` skill / scripts) — unchanged.
- Upstream blocking source tree
  (`/work2/.../artifacts/derecho_blocking/source_trees/forecast_modules/PanguPlasim/`)
  — unchanged. The forward-transform bug at
  `infer_sfno5410_blocking_h100_packed.py:348-349` is *cited* by the
  banner but not fixed here; a separate plan would do that.

## §6. Tests + validation

- **Unit test** in `tests/scripts/test_render_eval_report_pr6h.py`:
  - Drive `_render_table` (the actual helper name at
    `render_eval_report.py:128`)
    with a synthetic `summary` + `benchmark_summary` covering both
    `pr_6h` and a control channel (e.g. `tas`) across the scored
    leads in `_SCORED_LEADS_H`.
  - Assert that when `pr6h_unit_align == "suppress"`:
    - The rendered RMSE table contains no row with
      `| pr_6h | 5410 benchmark | ...`.
    - The rendered ACC table also contains no such row (proves the
      single guard covers both metrics).
    - `pr_6h` `emulator` and `persistence` rows remain.
    - `tas` `5410 benchmark` row remains.
  - Assert `pr6h_unit_align == "none"` reproduces the pre-change
    output byte-for-byte.
  - **Assert `track == "5410" and pr6h_unit_align == "suppress"` does
    NOT suppress the `pr_6h | 5410 benchmark` row.** This is the guard
    that protects valid group-native cross-track comparison from being
    accidentally hidden when someone runs a 5410-track report with the
    default `suppress` mode. The guard's `track == "own"` clause is
    load-bearing and must be tested directly.
- **Regression test** following the existing pattern at
  `tests/scripts/test_render_eval_report_warmstart.py:53` (must
  supply `--metadata-json` because the renderer otherwise reads
  channel_names from inference NetCDFs — see
  `render_eval_report.py:415`):
  - Frozen tiny `nwp_scorecard_summary.csv` for own and for the
    benchmark, plus a stub `metadata.json` of the exact shape
    consumed by `_eval_utils.resolve_channel_names` (which reads
    `["coords"]["channel"]`):

    ```json
    {"coords": {"channel": ["pl", "tas", ..., "pr_6h"]}}
    ```

    The warmstart test's `_write_metadata_json` helper already uses
    this shape — mirror it.
  - Build a temporary OUT_ROOT layout matching the renderer's
    expectations (`scores/`, `provenance.txt`, etc.).
  - Run with all required renderer args (per
    `scripts/render_eval_report.py:45-51`, mirroring the warmstart
    test helper at
    `tests/scripts/test_render_eval_report_warmstart.py:86-97`):

    ```bash
    render_eval_report.py \
      --out-root <own_root_fixture> \
      --run-tag test_pr6h_suppress \
      --eval-sha7 abc1234 \
      --data-sha7 def5678 \
      --train-sha7 fed3210 \
      --ckpt-path /fake/ckpt.tar \
      --benchmark-5410-out-root <bench_root_fixture> \
      --pr6h-unit-align suppress \
      --metadata-json <stub> \
      --report-out /tmp/test_pr6h_suppress_report.md
    ```

    Diff the output against a golden `report.md` that has no
    `(pr_6h, "5410 benchmark")` row in either table and includes the
    §4.3 banner. Without the `--benchmark-5410-out-root` arg the
    benchmark code path doesn't activate and the test would be
    vacuous — the benchmark root is load-bearing for exercising
    suppression.
- **Smoke check** on the live 2026-05-22 report (per Codex P2). The
  required SHAs/run-tag/ckpt-path are read from the live OUT_ROOT's
  `provenance.txt` (which already records `EVAL_SHA7`, `DATA_SHA7`,
  `TRAIN_SHA7`, `RUN_TAG`, and the EMA ckpt path) — same pattern as
  `submit_eval_report.slurm` uses when chaining the renderer in
  production:

  ```bash
  # read provenance.txt into shell vars (illustrative)
  # NOTE: the key is CKPT=, not CKPT_PATH=, per
  #   scripts/submit_eval_prelude.sh:190 (own-track) and
  #   scripts/submit_eval_5410.sh:131,:138 (5410). We map CKPT→ckpt-path
  #   for the renderer's --ckpt-path arg below.
  source <(awk -F= '/^(EVAL_SHA7|DATA_SHA7|TRAIN_SHA7|RUN_TAG|CKPT)=/ {print}' \
           <noise0p020_epochs75_h100retry_root>/provenance.txt)

  render_eval_report.py \
    --out-root <noise0p020_epochs75_h100retry_root> \
    --run-tag "$RUN_TAG" \
    --eval-sha7 "$EVAL_SHA7" \
    --data-sha7 "$DATA_SHA7" \
    --train-sha7 "$TRAIN_SHA7" \
    --ckpt-path "$CKPT" \
    --benchmark-5410-out-root <pinned_5410_root> \
    --pr6h-unit-align suppress \
    --report-out /tmp/pr6h_suppress_smoke_$$.md
  ```

  Routed at `/tmp/...` so the production `report.md` is not
  overwritten. The benchmark root must be passed explicitly —
  without it the suppression code path doesn't activate.
  - Assert the smoke output contains the §4.3 banner.
  - Assert the smoke output contains no `| pr_6h | 5410 benchmark | ...`
    row.
  - Assert other channels (tas, zg500, ua5, ta5) still have their 5410
    rows present.
- **`none`-mode byte-for-byte scope.** The byte-for-byte equivalence
  is asserted on the rendered scorecard table only — the part of the
  output that suppression actually touches. The `_load_benchmark`
  banner in `none`/`5410` modes is left unchanged from current code so
  there's no test-vs-spec drift (see §5.3).
- The §3.3 belt-and-suspenders within-5410 scale check is Phase 1
  work, not Phase 3 test work — listed here only to flag that the
  "no Phase-1 sanity check" wording of earlier drafts was wrong.

## §7. Risks + open questions

| risk | mitigation |
|---|---|
| Suppression loses pr_6h cross-track signal. | Accepted — the user chose suppression after weighing this against a heavier renderer-side recompute (Codex P0 resolution). A reader who needs cross-track pr_6h must wait for either the upstream `diagnostic_transform` fix or a follow-up plan that adds a field-level recompute path. |
| Reader doesn't realize what got suppressed. | §4.3 banner makes the suppression explicit with file:line citations. The banner replaces, not augments, the existing partial disclaimer. |
| Future 5410 reruns under a different upstream commit may finally fix the diagnostic-transform asymmetry and emit physical pr_6h. The suppression would then be hiding a *valid* comparison. | The default `--pr6h-unit-align suppress` is overrideable. A future eval against a fixed 5410 build can pass `--pr6h-unit-align none` to restore the displayed row. **But note:** `none` only restores the row to the table — it does **not** make own-track RMSE directly numeric-comparable, because the truth-side own (m/s) vs 5410 (m/6h) unit gap remains (§1). A truly comparable future state requires either a scalar conversion on the displayed values (still requires field-level recompute for RMSE/ACC per §2) or the upstream `diagnostic_transform` fix combined with a downstream unit-conversion pass. Document this nuance in the SKILL.md (§8). |
| Pre-existing track-mismatch in 5410-track callers (`submit_eval_report_5410.slurm` and `submit_eval_group_prod.sh` don't pass `--track 5410` / export `TRACK=5410`). Codex r6 flagged this. | **Out of scope for this plan.** The suppression guard's four-clause AND (`benchmark_summary is not None AND pr6h_unit_align == "suppress" AND track == "own" AND ch == "pr_6h"`) means suppression only triggers when all four hold. Per the Class A caller-sweep audit: `submit_eval_report_5410.slurm` doesn't pass `--benchmark-5410-out-root`, and `submit_eval_group_prod.sh` defaults `TRACK=own` (which is actually correct for the own-clone "phaseF group" use case that script targets, per its header at `:1-2`). So pr_6h suppression behaves correctly for the existing callers despite the pre-existing track-mismatch. Filing the caption-correctness bug as a separate ticket. |
| User wants the suppression applied retroactively to past reports (2026-05-21 noise sweep, beta1 sweep, etc.). | Out of scope. Past reports stay as historical artifacts; if needed, they can be re-rendered with the new flag by re-running `render_eval_report.py` against the existing CSVs. |
| Touching `render_eval_report.py` while a different branch (`zgplev-migration-dsi-bootstrap` currently active) is mid-flight may cause merge friction. | The current branch *does* have prior renderer edits (see commits `99d0180` 5410 benchmark overlay, `e3beb57` v10 zgplev defaults, etc.) — the `--track` flag at `:87` and track-aware caption at `:149` are recent. Before patching, run `git diff main -- scripts/render_eval_report.py scripts/eval_run_report_inline.sh .claude/skills/eval-sfno-own/SKILL.md` to see the existing diff across all three files this plan touches, so the new edits compose cleanly on top. The new helper + flag are additive and should not collide. |
| The empirical signature in §3.3 (own-vs-5410 truth-ratio ~3,600-4,400×, not a clean 21,600) was checked on one IC pair only. | The §3.3 second-IC check verifies the **within-5410** transformed-prediction signature (truth physical, prediction wildly out-of-range) on a second 5410 NetCDF — not the cross-run own-vs-5410 truth-ratio, which remains single-IC evidence by design. The cross-run truth-ratio is sufficient single-IC evidence because both pipelines deterministically write physical-units truth; the ratio is independent of model behavior. The numeric value of the ratio is not load-bearing for suppression — the *existence* of any unaudited gap (21,600 nominal vs ~4,000 observed = ~5× unexplained) is sufficient justification. |

## §8. Rollout

1. **Phase 1 (today, 2026-05-23)**: banner-text audit per §3. ~20 min.
   Output: `_PR6H_SUPPRESSION_RATIONALE` doc-comment in
   `render_eval_report.py` with the three file:line citations.
2. **Phase 2 (today)**: code `_maybe_suppress_pr6h_benchmark` helper +
   CLI flag (`--pr6h-unit-align`). Land with default `suppress`. ~20
   min.
3. **Phase 3 (today)**: thread the flag through
   `eval_run_report_inline.sh:28`. ~5 min.
4. **Phase 4 (today)**: add unit tests + regression fixture per §6.
   ~30 min.
5. **Phase 5 (today)**: smoke re-render the 2026-05-22 report with
   `--report-out /tmp/...` per Codex P2. ~5 min. Confirm banner +
   suppression land as expected.
6. **Phase 6 (today)**: SKILL.md updates — two sites, not one.
   ~10 min.
   - **`.claude/skills/eval-sfno-own/SKILL.md:71`** — the
     `BENCHMARK_5410_OUT_ROOT` row in the env-var table currently
     reads "always shown next to the own-track result". Replace
     "always" with "side-by-side except for own-track `pr_6h`, which
     is suppressed by default; override via `PR6H_UNIT_ALIGN=none`".
   - **Add a new `PR6H_UNIT_ALIGN` row to the same env-var table**
     with default `suppress`, override `none`, pointing at this plan
     doc.
   - **`.claude/skills/eval-sfno-own/SKILL.md:224`** — rewrite the
     "Group SFNO-5410 benchmark overlay" paragraph. The current
     paragraph claims the scorecard table "always" includes 5410
     side-by-side including pr_6h — that claim must change.
     Replacement paragraph should:
     - State the suppression behavior (5410 pr_6h row dropped from
       both RMSE and ACC tables by default for own-track).
     - Cite the new env var `PR6H_UNIT_ALIGN` (default `suppress`,
       override with `none`; passes through to `--pr6h-unit-align`).
     - Point at this plan doc for the rationale.
     - Keep the existing bias-map-layout description (which is
       correct and not changed by this plan).
   - Include `.claude/skills/eval-sfno-own/SKILL.md` in the
     `git diff main -- ...` checklist from §7's risk row so any
     existing in-flight skill edits compose cleanly.
7. **No retro re-render** of older reports unless explicitly requested.

## §9. Memory updates

After this plan lands:
- New project memory `project_5410_pr6h_forward_transform_anomaly`:
  "5410 inference at `infer_sfno5410_blocking_h100_packed.py:348-349`
  applies forward (not inverse) z-score transform to the diagnostic
  pr_6h channel, asymmetric with surface/upper_air. On-disk 5410
  pr_6h prediction is therefore in transformed space and not
  comparable to physical truth without an upstream fix or a
  field-level recompute. Renderer suppresses the cross-track pr_6h
  row by default."
- Update [[project-5410-eval-track]] with a cross-reference to the
  new memory.

## §10. Out of scope (deferred follow-ups)

- **Patching the upstream forward-transform bug** at
  `infer_sfno5410_blocking_h100_packed.py:348-349`. Touching the
  upstream pipeline is its own decision (compute, group conventions,
  potential re-eval); deferred.
- **Field-level pr_6h recompute** in the renderer or in
  `score_nwp.py`. Would unsuppress the cross-track comparison but is
  heavier work; only worth it if the user later wants pr_6h
  cross-track back.
- **Display-unit harmonization** between report (m/s) and figures
  (mm/day) for own pr_6h. The two channels disagree on display unit
  but both are mathematically correct in their native CSV space —
  cleanup is cosmetic.
- **Modifying the own-track packager** to emit pr_6h in m/6h instead
  of m/s. Would break every past own-track checkpoint's calibration.
- **Retraining either model** to fix pr_6h.
- **Adding a second pr_6h channel** ("pr_6h_mm") to either NetCDF.
