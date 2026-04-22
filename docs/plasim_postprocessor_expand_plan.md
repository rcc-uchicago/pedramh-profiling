# PlaSim Post-processor — Variable Expansion + `--profile` Removal (Plan)

> **SUPERSEDED 2026-04-21** by `docs/plasim_expansion_and_adaptor_plan.md`,
> which is the plan that was actually executed (commits `7fc8a7c …
> 504079e` on main). Four of this v1 draft's decisions were reversed
> during the superseding plan after the 2026-04-21 interview:
>
>   - `pl` is kept in the postprocess output. v1 dropped it under the
>     misreading that `ps` (surface_air_pressure) could substitute for
>     `log_surface_pressure`; the SFNO emulator's `SINGLE_LEVEL_VARS`
>     entry named `pl` maps to burn7 code 152, not to `ps`.
>   - `zg` is dual-output: sigma (canonical, 10 midpoints, co-located with
>     ta/ua/va/hus, produced by a patched burn7) and `zg_plev` (13
>     pressure levels, the prior pressure-level zg renamed to avoid a
>     NetCDF name collision). v1 only described pressure-zg.
>   - The `sst` / `rsdt` adaptor for the emulator's varying-boundary tuple
>     now lives in a separate module at `src/emulator_adaptor/`, not in
>     the postprocess. v1 did not consider the adaptor layer at all.
>   - `td2m` is NOT in the final contract. sim30/MOST.0012 does not emit
>     code 168, and no SFNO emulator config references it. v1 had listed
>     it in the sigma namelist code set.
>
> v1 text preserved below for history. Do not act on it.
>
> Status (v1, as originally drafted): draft, awaiting user approval + external Codex review before implementation.

## Goal

Two coupled changes to `src/plasim_postprocessor/plasim_postprocessor.py`:

1. **Remove the `--profile` CLI flag.** Delete the legacy `exp15`, `exp25`, `exp26` profiles (not used by this project). With only one variable set remaining, there is nothing to choose between, so the flag itself goes away. The script has a single purpose: produce the audited variable set from raw PlaSim output.
2. **Expand the variable set** to the union of the current `aires_rad` contract + additions from the 2026-04-18 request (with q2m→td2m and pl dropped per interview).

Plus the paperwork: re-audit to lock the new contract, delete legacy snapshots/docs, update the SKILL, narrative audit doc, and refactor-plan record.

## Final variable set (the new single contract)

All variables are produced by **burn7** except `pr_6h`, which is `cdo runsum,6` applied to burn7's `pr`. No other CDO-derived variables (per user directive: physical derivations like q2m-from-td2m belong downstream, not in post-processing).

### Sigma-namelist codes (one burn7 call)

Atmosphere 3D (on 10 sigma levels):
`ta` (130), `ua` (131), `va` (132), `hus` (133)

Single-level surface / near-surface / flux / static-boundary fields (burn7 emits these as 2D despite the sigma namelist; it dispatches by internal code metadata):
`tas` (167), `td2m` (168), `ts` (139), `ps` (134), `psl` (151), `clt` (164),
`mrso` (140),
`rss` (176), `rls` (177), `rst` (178), `rlut` (179), `rsut` (203),
`hfss` (146), `hfls` (147),
`lsm` (172), `z0` (173), `sg` (129),
`pr` (260).

With `--with-sea-ice`: append `sic` (210).

### Pressure-namelist codes (separate burn7 call)

`zg` (156) interpolated to 13 pressure levels:
`50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000` hPa.

### CDO-derived

`pr_6h` from `pr` via `selname,pr → runsum,6 → chname,pr,pr_6h → merge`. `pr` is retained alongside `pr_6h` (matches existing contract; 5-step `runsum` warm-up still trims the time dimension by 5).

### Variables removed from the user's draft list

| Requested | Action | Reason (from interview) |
|---|---|---|
| `q2m` | replaced with `td2m` (code 168, 2-m dew-point) | burn7 has no 2-m specific humidity code. Derive q2m downstream from `td2m + ps` via Clausius-Clapeyron in training code. Keeps the "burn7 for variables, CDO only for pr_6h" contract intact. |
| `pl` | removed entirely | `pl` in burn7 is `log_surface_pressure` (PlaSim-internal, for dynamical-core numerical stability). `ps` covers surface pressure; `pl = ln(ps/p0)` is trivially recoverable downstream if ever needed. |
| `surface geopotential` as orography | kept as raw `sg` (m² s⁻²) | No derived variables in the post-processor. Downstream divides by `g = 9.80665` if orography in meters is wanted. |
| `sic` unconditional | gated by `--with-sea-ice` CLI flag (default off) | PlaSim run-time namelist (NSEAICE/NLSG) not in this repo; sims 30–36 belong to amarchakitus. A CLI flag lets the operator declare it per run without a namelist parser. Default-off so the script doesn't fail on sims without sea ice. |

### Expected output manifest (dimensions + variables)

```
=== dimensions ===
lat = 64
lev = 10
lev_2 = 13
lon = 128
time = UNLIMITED   // 1459 for MOST.0012 (1464 native − 5 runsum,6 warm-up)

=== variables ===
  double lat(lat)
  double lev_2(lev_2)
  double lev(lev)
  double lon(lon)
  double time(time)
  float clt(time, lat, lon)
  float hfls(time, lat, lon)
  float hfss(time, lat, lon)
  float hus(time, lev, lat, lon)
  float lsm(time, lat, lon)
  float mrso(time, lat, lon)
  float pr(time, lat, lon)
  float pr_6h(time, lat, lon)
  float ps(time, lat, lon)
  float psl(time, lat, lon)
  float rls(time, lat, lon)
  float rlut(time, lat, lon)
  float rss(time, lat, lon)
  float rst(time, lat, lon)
  float rsut(time, lat, lon)
  float sg(time, lat, lon)          // may be (lat, lon) if burn7 emits it as static — verify
  float ta(time, lev, lat, lon)
  float tas(time, lat, lon)
  float td2m(time, lat, lon)
  float ts(time, lat, lon)
  float ua(time, lev, lat, lon)
  float va(time, lev, lat, lon)
  float z0(time, lat, lon)          // may be time-varying via snow; verify at audit
  float zg(time, lev_2, lat, lon)
  // +float sic(time, lat, lon) iff --with-sea-ice

=== lev (sigma) coordinate values ===
lev=0.0383,0.1191,0.21085,0.31685,0.4368,0.5668,0.69935,0.82335,0.9241,0.9833   // exact values from burn7

=== lev_2 (pressure) coordinate values ===
lev_2=50,100,150,200,250,300,400,500,600,700,850,925,1000

=== time coordinate summary ===
time units = <captured from ncdump -v time>
time size  = 1459
first 4 values: <captured>
last  4 values: <captured>
spacing (diff[0..2]): <captured>   // expect 0.25 day or 6 h — confirms native 6-hourly cadence
```

The exact `sg` and `z0` time-dimension behaviour is an audit-time question — burn7 treats some "static" fields as once-per-file, others as time-replicated. The locked `manifest.txt` records whatever shape burn7 actually emits; it is not a correctness judgement, just the contract.

**Time-coordinate capture (change from prior audit convention).** The existing `manifest()` helper intentionally dropped time values because they're "calendar-dependent" (see `docs/plasim_postprocessor_audit.md` method section). This plan reverses that call per the user's explicit request: the new `manifest()` helper captures time units + size + first 4 / last 4 values + sample spacings. The prior blind spot (silent calendar shifts, wrong cadence, off-by-one trims from `runsum`) is closed by this. Full-time-array equality is *not* the contract — first/last/spacing is enough to catch meaningful drift without making the snapshot hostile to small calendar-attribute changes.

## Code changes (single Python file)

### `src/plasim_postprocessor/plasim_postprocessor.py`

Remove:
- `PROFILES` dict (4 entries)
- `PROVISIONAL_PROFILES` set
- `--profile` argparse flag
- `profile` argument from `process_one`, `_write_sigma_namelist`, `_write_zg_namelist`
- Runtime provisional-warning block in `main()`

Replace with module-level constants (single contract):

```python
SIGMA_CODES = [
    # 3D atmosphere
    "ta", "ua", "va", "hus",
    # near-surface + surface state
    "tas", "td2m", "ts", "ps", "psl", "clt",
    # land
    "mrso",
    # radiation + heat fluxes
    "rss", "rls", "rst", "rlut", "rsut", "hfss", "hfls",
    # static boundary
    "lsm", "z0", "sg",
    # precipitation (source for pr_6h)
    "pr",
]
PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
ACCUMULATE_PRECIP_HOURS = [6]
SEA_ICE_CODE = "sic"
```

`_write_sigma_namelist(path, with_sea_ice)` builds `code=`:
- base: `SIGMA_CODES`
- if `with_sea_ice`: `+ [SEA_ICE_CODE]`

`_write_zg_namelist(path)` no longer needs a profile argument — `PRESSURE_LEVELS` is a module constant.

`process_one(sim, year, opts)` drops `profile`, reads `opts.with_sea_ice`, passes it into `_write_sigma_namelist`. Precip-accumulation loop still iterates `ACCUMULATE_PRECIP_HOURS` (a one-element list; kept as a list for trivial extensibility, but no CLI knob exposes it).

Add argparse flags:
- `--with-sea-ice` / `--no-sea-ice`: `argparse.BooleanOptionalAction`, default `False`. Help text: "Include `sic` (code 210, sea_ice_cover). Default off — enabling it on a sim whose PlaSim run had sea ice disabled will cause burn7 to fail. The operator must know whether the source sims have sea ice; this script cannot infer it."

Delete from argparse:
- `--profile` line (and its sorted-choices reference)
- Any docstring/epilog mention of profiles

Update `_parse_args()` description: "Single-purpose burn7-based PlaSim post-processor for the SFNO emulator training variable set."

Update module docstring: delete the profile contract table, replace with the single variable-set summary above. Keep the toolchain, output-layout, and module-load sections.

### `src/plasim_postprocessor/submit.slurm`

Remove `PROFILE=""` placeholder and the `--profile "$PROFILE"` line. Add a commented `WITH_SEA_ICE=` toggle (default empty; if set to `1`, dispatch appends `--with-sea-ice`). Fail-fast guards otherwise unchanged.

### Nothing else in `src/plasim_postprocessor/` changes.

The `burn7/` directory, `burn7` binary, and module-load contract are untouched.

## Docs changes

### Delete outright (explicitly authorized by interview Q6 + Q5)

- `docs/audit_snapshots/exp15_manifest.txt`
- `docs/audit_snapshots/exp25_manifest.txt`
- `docs/audit_snapshots/exp26_manifest.txt`
- `docs/audit_snapshots/aires_rad_manifest.txt` (replaced by `manifest.txt` per Q5)
- `docs/plasim_postprocessor_refactor_plan.md` (v1–v4 history; git preserves it — per Q6)

### Extra deletions — NOT in original authorization; flagging for explicit sign-off

These were in the initial draft of this plan but are not covered by interview Q6 ("delete exp15/25/26 manifests, the --profile section of the audit doc, and refactor plan v1–v4"). Listed separately so you can decide each:

- `docs/aires_rad_profile_plan.md` — this is the provisional→audited workflow doc for the `aires_rad` profile. With the `--profile` flag gone and the provisional-flow section removed from SKILL.md, this doc has no referents left in the repo. **Default recommendation: delete** (git preserves it and the audit doc already carries forward the sign-convention findings). **Alternative: keep** as a historical record of how `aires_rad` was validated. Your call.
- `docs/plasim_postprocessor_expand_plan.md` — this file itself, after the three commits land. **Default recommendation: delete** (its job ends when the code + audit + SKILL rewrite ship; the audit doc becomes the durable record of the contract). **Alternative: keep** in case you want a rolling log of planning docs in `docs/`. Your call.

### Rewrite

**`docs/plasim_postprocessor_audit.md`** — delete the EXP15/25/26 reconciliation tables, delete the "Addendum: aires_rad profile" section's provisional framing. Replace with a single "Locked variable set (2026-04-18)" section:
- Method (sample input `/scratch/10000/amarchakitus/PLASIM/data/sim30/MOST.0012`, toolchain versions, module set — unchanged from current audit).
- Full expected manifest (copied from this plan once audit confirms it).
- Per-variable physical sanity table (see "Audit plan" below).
- `sic` probe result: whether code 210 is present in sim30/MOST.0012 (interview Q7 context). Recorded as an informational note, does *not* affect the default CLI flag state.
- Sign-convention block for the 7 radiation/flux variables (unchanged; carried forward from the current audit addendum).

**`skills/plasim-postprocess/SKILL.md`** —
- Drop the "Profiles (locked by audit snapshots)" table; replace with a single "Variable set (locked by `docs/audit_snapshots/manifest.txt`)" section.
- Drop `--profile` from the CLI reference.
- Add `--with-sea-ice` / `--no-sea-ice` to the CLI reference (with the default-off rationale).
- Drop the "Adding a new profile (provisional → audited workflow)" section outright — irrelevant with a single contract.
- Update the verification recipe. The new recipe is:
  - Always: run without `--with-sea-ice`, manifest, `diff` against `docs/audit_snapshots/manifest.txt` (must be empty).
  - If `docs/audit_snapshots/manifest_with_sea_ice.txt` exists in the repo: also run with `--with-sea-ice` on the same reference input, manifest, `diff` against that snapshot (must be empty).
  - If the sea-ice snapshot does *not* exist in the repo: the `--with-sea-ice` path has no in-repo verification target. SKILL.md documents this as "operators enabling `--with-sea-ice` on other sims take responsibility for their source data having code 210; if you want test coverage for that branch, run an audit on a sim you know has sea ice and lock a `manifest_with_sea_ice.txt` for that reference." This is the honest framing — not "one diff always."
- Delete the "provisional" row from the Common failure modes table.
- Update the description frontmatter: drop "four audited profiles (aires_rad, exp15, exp25, exp26)" → "the audited variable set".

### Leave as-is

- `src/plasim_postprocessor/burn7/**` — all source/binary files.
- `skills/plasim-postprocess/` — no other files.

## Audit plan (lock the new contract)

Reference input: `/scratch/10000/amarchakitus/PLASIM/data/sim30/MOST.0012` (same as previous audits, per interview Q7).

### Step 0 — probe sic presence in sim30/MOST.0012

Before anything else, run a standalone burn7 call with `code=sic` only:

```bash
cd $(mktemp -d)
cat > sic_probe.nl <<EOF
code=sic
MODLEV=10,9,8,7,6,5,4,3,2,1,0
vtype=sigma,htype=g,mean=0,netcdf=1
EOF
/path/to/burn7 /scratch/.../sim30/MOST.0012 sic_probe.nc < sic_probe.nl
ncdump -h sic_probe.nc | grep -E "^\s*float sic"
```

Record the outcome in `docs/plasim_postprocessor_audit.md` — either "code 210 present in sim30/MOST.0012" or "absent; burn7 returns error X". Informational; does not change the default-off flag.

### Step 1 — lock the default (no-sea-ice) manifest

Run the new script:

```bash
python3 src/plasim_postprocessor/plasim_postprocessor.py \
    --sims 30 --years 12 12 \
    --input-root /scratch/10000/amarchakitus/PLASIM/data \
    --output-root /tmp/audit_default
```

Manifest it with a **new** helper (the existing one in `docs/plasim_postprocessor_audit.md` / `SKILL.md` does not capture time — see Finding 1 of round 2 review). The new helper replaces the old one in both places:

```bash
manifest() {
    local nc="$1"
    echo "=== dimensions ==="
    ncdump -h "$nc" \
        | awk '/^dimensions:/{flag=1; next} /^variables:/{flag=0} flag' \
        | sed 's/^[[:space:]]*//' | sort
    echo "=== variables ==="
    ncdump -h "$nc" \
        | awk '/^variables:/{flag=1; next} /^\/\/ global attributes:|^data:/{flag=0} flag' \
        | grep -E "^[[:space:]]*(float|double|int|short|char) " | sort
    echo "=== lev (sigma) coordinate values ==="
    ncdump -v lev "$nc" 2>/dev/null | awk '/^ lev =/,/;/' | tr -d '\n ' | sed 's/;.*//'; echo
    echo "=== lev_2 (pressure) coordinate values ==="
    ncdump -v lev_2 "$nc" 2>/dev/null | awk '/^ lev_2 =/,/;/' | tr -d '\n ' | sed 's/;.*//'; echo
    echo "=== time coordinate summary ==="
    ncdump -h "$nc" | grep -E '^[[:space:]]*(double|float) time\(' | head -1
    ncdump -h "$nc" | grep -E 'time:units'          # units — frozen for contract
    ncdump -h "$nc" | grep -E 'time:calendar' || echo "(no calendar attribute)"
    ncdump -h "$nc" | grep -E '^[[:space:]]+time = '   # dim size from dimensions block
    # Extract time values once, reuse for first-4 / last-4 / cadence.
    local tvals
    tvals=$(ncdump -v time "$nc" | awk '/^ time =/,/;/' | tr -d '\n ' | sed 's/;.*//' | tr ',' '\n' | grep -v '^$')
    local n
    n=$(printf '%s\n' "$tvals" | wc -l)
    echo "first 4 values:"
    printf '%s\n' "$tvals" | awk -v n="$n" 'NR<=4'
    echo "last 4 values:"
    printf '%s\n' "$tvals" | awk -v n="$n" 'NR>=n-3'
    echo "cadence (first three diffs):"
    printf '%s\n' "$tvals" | awk 'NR<=4 {t[NR]=$1} END {for (i=2;i<=4;i++) printf "dt[%d]=%g\n", i-1, t[i]-t[i-1]}'
    echo "=== lat/lon shape (sanity) ==="
    ncdump -h "$nc" | grep -E "^[[:space:]]+(lat|lon) =" | sort
}
```

Save output to `docs/audit_snapshots/manifest.txt`. This is the new locked-contract format; it supersedes the old 5-section helper (the old lat/lon/lev/lev_2/variables contract still holds — we're adding time, not replacing anything).

### Step 2 — capture the (conditional) sea-ice manifest

Only if Step 0 shows sic is present:

```bash
python3 src/plasim_postprocessor/plasim_postprocessor.py \
    --sims 30 --years 12 12 \
    --input-root /scratch/10000/amarchakitus/PLASIM/data \
    --output-root /tmp/audit_sic \
    --with-sea-ice
```

Save to `docs/audit_snapshots/manifest_with_sea_ice.txt`. If Step 0 shows sic is absent, the audit doc records that `--with-sea-ice` has no test coverage in this repo; users enabling it take responsibility for their sims having sea ice data. No separate snapshot file.

### Step 3 — per-variable physical sanity

**Provenance rule.** All `Observed mean / min / max` cells in the table below are populated from the Step 1 run (`/tmp/audit_default/sim30/MOST.0012.nc`). The `aires_rad` §B.1 numbers in the existing audit doc are **baselines for expected-range sanity**, not values copied into the new table — the new contract is self-contained. If a new-run number drifts from its old counterpart by more than ±5 %, that is a finding logged in the audit-doc rewrite and investigated (likely a sampling artefact from the expanded sigma namelist, but it needs explicit sign-off).

**Statistics, per variable family.** `cdo -s output -fldmean -timmean` does not collapse vertical dimensions, so multi-level variables need an explicit vertical reduction. The new contract picks one canonical reduction per variable (not per level) so each row in the table is a single scalar:

| Variable family | Mean command | Min command | Max command |
|---|---|---|---|
| 2D (lat, lon, time) — every surface / TOA / static-boundary variable | `cdo -s output -fldmean -timmean -selname,<v> file.nc` | `cdo -s output -fldmin -timmin -selname,<v> file.nc` | `cdo -s output -fldmax -timmax -selname,<v> file.nc` |
| 3D on sigma (ta, ua, va, hus) | `cdo -s output -fldmean -vertmean -timmean -selname,<v> file.nc` | `cdo -s output -fldmin -vertmin -timmin -selname,<v> file.nc` | `cdo -s output -fldmax -vertmax -timmax -selname,<v> file.nc` |
| 3D on pressure (zg) — single canonical level | `cdo -s output -fldmean -timmean -sellevel,500 -selname,zg file.nc` | `cdo -s output -fldmin -timmin -sellevel,500 -selname,zg file.nc` | `cdo -s output -fldmax -timmax -sellevel,500 -selname,zg file.nc` |

`vertmean` is a simple level-average (not mass-weighted — burn7 outputs `lev` as sigma values but CDO has no pressure data to weight with at this point in the pipeline). Adequate for order-of-magnitude sanity; the per-level structure is already constrained by the manifest snapshot (dimension sizes + `lev` coordinate values). For `zg`, 500 hPa is the canonical aviation/synoptic level and the value that `aires_rad` implicitly targeted.

**Verdict rules (per row, not a blanket tolerance).** The "±30 % of expected magnitude" shorthand from `aires_rad` §B.1 does *not* generalise — it fails on near-zero variables (va) and on mask/fraction variables (lsm, sic). Each row declares its own pass criterion:

| Verdict kind | Applies to | Pass criterion |
|---|---|---|
| Range-on-mean | rss, rst (positive means); `|mean|` within range for rls, rlut, rsut, hfss, hfls; ta, ts, tas, td2m, ps, psl, z0, sg, pr, pr_6h, hus, ua, zg, mrso | Observed mean falls inside the stated "Expected mean range" column |
| Near-zero | va | `|mean| < 0.5 m s⁻¹` (global-temporal vertmean; continuity makes this much tighter than 30 %) |
| Binary / fraction | lsm | min = 0, max = 1 exactly (or ≤ 1 + 1e-6); mean ∈ [0.25, 0.33]; separately record whether the value set on (lat, lon) is strictly {0, 1} or has fractional coastal cells — if fractional, the row records "fractional" and flags for the audit doc |
| Fraction | clt, sic | min ≥ 0, max ≤ 1 + 1e-6; mean in stated range |

**Table.** Populate `Observed` columns from the Step 1 output (no carry-forward):

| Variable | Reduction | Units | Expected mean range | Observed mean | Observed min | Observed max | Verdict rule | Verdict |
|---|---|---|---|---|---|---|---|---|
| `ta` | `vertmean` over 10 sigma | K | 245–260 (simple level-average of tropospheric profile) | — | — | — | range-on-mean | — |
| `ua` | `vertmean` | m s⁻¹ | 0–10 (zonal + vertical average; dominated by mid-lat westerlies) | — | — | — | range-on-mean | — |
| `va` | `vertmean` | m s⁻¹ | ≈ 0 | — | — | — | near-zero (\|mean\| < 0.5) | — |
| `hus` | `vertmean` | 1 | 1e-3 – 5e-3 (simple level-average) | — | — | — | range-on-mean | — |
| `zg` | `sellevel,500` | m | 5400–5700 | — | — | — | range-on-mean | — |
| `tas` | — | K | 287–289 | — | — | — | range-on-mean | — |
| `ts` | — | K | 287–292 | — | — | — | range-on-mean | — |
| `td2m` | — | K | 275–285 | — | — | — | range-on-mean | — |
| `ps` | — | hPa | 970–990 | — | — | — | range-on-mean | — |
| `psl` | — | hPa | 1010–1015 | — | — | — | range-on-mean | — |
| `clt` | — | 1 | 0.55–0.70 | — | — | — | fraction | — |
| `mrso` | — | m | 0.01–0.08 (global mean *including* ocean cells, which are 0 in this field; land-only mean would be ~10× larger, but the table reports the whole-field stat the CDO command yields) | — | — | — | range-on-mean | — |
| `z0` | — | m | 0.05–0.3 (global; ocean cells near 1e-4, forest cells 1–3) | — | — | — | range-on-mean | — |
| `sg` | — | m² s⁻² | 2000–6000 (global; 0 ocean, up to ~60000 Himalayas) | — | — | — | range-on-mean | — |
| `lsm` | — | 1 | 0.27–0.31 | — | 0 | 1 | binary / fraction | — |
| `pr` | — | m s⁻¹ | 2e-8 – 5e-8 (≈ 1.7 – 4.3 mm day⁻¹) | — | — | — | range-on-mean | — |
| `pr_6h` | — | m | 4e-4 – 11e-4 (≈ pr × 21600 s) | — | — | — | range-on-mean | — |
| `rss` | — | W m⁻² | +150 – +180 | — | — | — | range-on-mean | — |
| `rls` | — | W m⁻² | −70 – −40 | — | — | — | range-on-mean | — |
| `rst` | — | W m⁻² | +230 – +260 | — | — | — | range-on-mean | — |
| `rlut` | — | W m⁻² | −260 – −220 | — | — | — | range-on-mean | — |
| `rsut` | — | W m⁻² | −120 – −80 | — | — | — | range-on-mean | — |
| `hfss` | — | W m⁻² | −35 – −15 | — | — | — | range-on-mean | — |
| `hfls` | — | W m⁻² | −100 – −70 | — | — | — | range-on-mean | — |
| `sic` (if Step 0 present) | — | 1 | 0.03–0.08 (global; non-zero only in polar) | — | ≥ 0 | ≤ 1 | fraction | — |

Also record (non-numeric, per variable):
- `sg`, `lsm`, `z0` — whether burn7 emits them as `(time, lat, lon)` or `(lat, lon)`. The manifest captures this in its variable block; also note in the audit narrative.
- `lsm` — binary vs fractional coastlines, per the binary-row rule above.
- `z0` — whether it is time-varying (snow-cover coupling in some configurations). Observable from the manifest by shape, from the values by running `cdo timvar` and checking non-zero temporal variance.

Non-numeric shape/coverage sanity (applies to every variable in the contract):
- `ncdump -h` confirms the expected `(time, ...)` shape on every data variable; the manifest diff already enforces this.
- `cdo info` confirms `Miss = 0` for every timestep on every variable (no unexpected missing-value cells introduced by the expanded namelist).

### Step 4 — round-trip verification

- Always: re-run Step 1, re-manifest, `diff` against `docs/audit_snapshots/manifest.txt` — must be empty.
- If Step 2 produced a sea-ice snapshot: re-run Step 2, re-manifest, `diff` against `docs/audit_snapshots/manifest_with_sea_ice.txt` — must be empty.
- If Step 2 did not run (sic absent from reference): the `--with-sea-ice` code path has no locked contract. This is documented in the audit doc and SKILL.md; it is *not* a failure of the audit, it is a true statement about what this reference input can test.

## Commit order (atomic, verifiable)

1. **Commit A: code + slurm.** Edit `plasim_postprocessor.py` and `submit.slurm` per the code-changes section. Do *not* touch docs or snapshots yet. This commit alone will make existing SKILL recipes invalid until the docs catch up, but code review can read this commit in isolation.
2. **Commit B: run audit, lock snapshot.** Run Steps 0–4. Commit the new `docs/audit_snapshots/manifest.txt` (and `manifest_with_sea_ice.txt` if Step 2 ran). Append the new audit section to `docs/plasim_postprocessor_audit.md`.
3. **Commit C: doc cleanup.** Delete the authorized items (legacy exp15/25/26 + aires_rad snapshots, `refactor_plan.md`), rewrite `docs/plasim_postprocessor_audit.md` for the single-contract narrative, rewrite `SKILL.md` per the docs-changes section. **Any items from the "Extra deletions" list are included in this commit only if you signed off on them at plan-approval time** — otherwise they stay in the tree.

Each commit stands alone on `git log`; C depends on B depends on A.

## Open questions / risks

None surfaced from the interview — all 8 clarifications resolved. The one live unknown is the audit-time behaviour of `sg`/`lsm`/`z0` time dimensionality, which is a "record-what-burn7-does" question, not a design question.

## Files touched (summary)

| File | Change |
|---|---|
| `src/plasim_postprocessor/plasim_postprocessor.py` | edit: remove PROFILES/--profile, add `--with-sea-ice`, flatten to module constants |
| `src/plasim_postprocessor/submit.slurm` | edit: drop `PROFILE`, add optional `WITH_SEA_ICE` toggle |
| `docs/audit_snapshots/manifest.txt` | **new** (locked contract) |
| `docs/audit_snapshots/manifest_with_sea_ice.txt` | new (conditional on sic probe) |
| `docs/audit_snapshots/exp15_manifest.txt` | **delete** |
| `docs/audit_snapshots/exp25_manifest.txt` | **delete** |
| `docs/audit_snapshots/exp26_manifest.txt` | **delete** |
| `docs/audit_snapshots/aires_rad_manifest.txt` | **delete** (replaced by manifest.txt) |
| `docs/plasim_postprocessor_audit.md` | rewrite (single-contract narrative + new sanity table) |
| `docs/plasim_postprocessor_refactor_plan.md` | **delete** (authorized; git preserves v1–v4) |
| `docs/aires_rad_profile_plan.md` | **delete pending approval** (extra; see "Extra deletions" section) |
| `docs/plasim_postprocessor_expand_plan.md` | **delete pending approval** after work lands (extra; see "Extra deletions" section) |
| `skills/plasim-postprocess/SKILL.md` | rewrite: drop profiles, add sea-ice flag, simplify verification |
