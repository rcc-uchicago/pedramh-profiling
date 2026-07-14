# Add `aires_rad` profile (radiation + heat fluxes) — Plan (v3)

> Revision history:
> - **v1**: hard-gated implementation on sim30 having radiation; "first run = truth" snapshot; weak grep probe; under-specified --help update.
> - **v2**: decoupled feature from snapshot (Stream A vs B), added semantic-validation step, replaced grep probe with `cdo infon`, listed argparse description in edits.
> - **v3**: resolved contract ambiguity (provisional ship); parameterized reference data via REF_SIM/REF_YEAR; added semantic-validation step.
> - **v4 (this doc)**: responds to Codex round 3 by enumerating *every* "three audited profiles / frozen-by-snapshot" string the v3 edit list referenced loosely. All such strings are now listed with file:line and explicit before/after text, both for Stream A (set provisional) and Stream B.4 (flip back).

## Context

The post-processor at `src/plasim_postprocessor/plasim_postprocessor.py` exposes three audited profiles (`exp15`, `exp25`, `exp26`); each has a locked manifest in `docs/audit_snapshots/` and is treated as a frozen contract by the script docstring (`:7`) and skill (`SKILL.md:20`).

You want a fourth profile, `aires_rad`, bundling exp26's vars with a 7-code radiation + heat-flux block (`rss, rls, rst, rlut, rsut, hfss, hfls`), suitable for energy-budget-aware AI emulator training.

Three earlier interview decisions stand:
- **Variables**: 7 codes — `rss, rls, rst, rlut, rsut, hfss, hfls`.
- **Profile shape**: new sibling profile (`aires_rad`) that's a strict superset of `exp26`. Existing audited profiles untouched.
- **Audit method**: snapshot directly, no diff against legacy pipeline.

Round-2 decisions added:
- **Ship gate**: provisional — Stream A merges without a snapshot; Stream B lands the snapshot later when a rad-enabled reference file is identified. Provisional status is visible in *every* user-facing surface.
- **Reference data**: parameterized via `REF_SIM`/`REF_YEAR` so audit isn't bound to sim30.

`/scratch/.../sim30-36` remains *reference sample data*, not the production target.

## Profile contract: `aires_rad` (provisional until snapshot lands)

| Group | Variables | Producing tool |
|---|---|---|
| Sigma 11-level | ta, ua, va, hus | burn7 sigma namelist |
| Surface (single-level via sigma namelist) | pl, tas | burn7 sigma namelist |
| Land (surface) | mrso | burn7 sigma namelist |
| Radiation (surface, all `(time, lat, lon)`) | rss (176), rls (177), rst (178), rlut (179), rsut (203) | burn7 sigma namelist |
| Heat fluxes (surface) | hfss (146), hfls (147) | burn7 sigma namelist |
| Pressure (zg only) | zg @ 13 levels: 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa | burn7 zg namelist |
| Precip accumulation | pr_6h (and pr retained alongside) | CDO `runsum,6` + `chname,pr,pr_6h` |

burn7 metadata (`burn7.cpp:5508-5547`) only stores label/units (CF-style names like `surface_net_shortwave_flux`, `toa_net_longwave_flux`, `toa_outgoing_shortwave_flux`); it does not encode sign conventions. PlaSim sets the sign — Stream B's semantic-validation step pins this down empirically before the snapshot is locked.

## Approach

### Stream A — Generic implementation (provisional ship)

Independent of reference-data availability. Lands the feature with explicit "provisional" markers throughout.

#### A.1 Edits to `src/plasim_postprocessor/plasim_postprocessor.py`

**Code additions:**

- Add `"aires_rad"` entry to `PROFILES` dict (after `exp26`):
  ```python
  "aires_rad": {
      "sigma_vars": ["ta", "ua", "va", "hus", "pl", "tas",
                     "rss", "rls", "rst", "rlut", "rsut",
                     "hfss", "hfls"],
      "land_vars": ["mrso"],
      "pressure_levels": [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
      "accumulate_precip_hours": [6],
  },
  ```
- Add a module-level constant immediately after `PROFILES`:
  ```python
  # Profiles whose output contract has not yet been locked by an audit snapshot
  # in docs/audit_snapshots/{profile}_manifest.txt. Removed when the snapshot lands.
  PROVISIONAL_PROFILES = {"aires_rad"}
  ```
- Add a **runtime warning** in `main()` immediately after argparse, before processing or `--count-tasks`:
  ```python
  if args.profile in PROVISIONAL_PROFILES:
      logger.warning(
          "Profile %r is PROVISIONAL — its output contract is not yet locked "
          "by an audit snapshot in docs/audit_snapshots/. Outputs may change "
          "in a future revision. Do not use for downstream work that requires "
          "a stable contract.",
          args.profile,
      )
  ```
  Warning fires once per invocation (not per task) and goes to stderr via the existing logger. Does not gate execution — the user has been told.

**Stale-text rewrites** (every "three audited profiles / frozen" reference in this file):

| Line(s) | Current | Replace with |
|---|---|---|
| `:5-7` (docstring opener) | `One of three audited profiles (exp15 / exp25 / exp26) selects the variable set; profile definitions are internal constants frozen by docs/audit_snapshots/{profile}_manifest.txt.` | `A profile (--profile) selects the variable set. Audited profiles (exp15, exp25, exp26) are frozen by docs/audit_snapshots/{profile}_manifest.txt; provisional profiles (aires_rad) are usable but their output contract is not yet locked.` |
| `:14` (docstring section header) | `Profile contracts (full output variable list locked by audit snapshots)` | `Profile contracts` |
| `:14-17` (docstring body block) | `exp15 …` / `exp25 …` / `exp26 …` (3 lines) | Two-section block: `Audited (locked by docs/audit_snapshots/{profile}_manifest.txt):` followed by exp15/25/26 lines, blank line, `Provisional (no locked snapshot — outputs may shift; do not pin downstream work):` followed by `aires_rad  exp26 vars + rss rls rst rlut rsut hfss hfls (radiation + surface heat fluxes; sign conventions not yet validated)` |
| `:174-175` (argparse description) | `"Single-purpose burn7-based PlaSim post-processor (profiles: exp15, exp25, exp26)."` | `"Single-purpose burn7-based PlaSim post-processor (audited profiles: exp15, exp25, exp26; provisional: aires_rad)."` |
| `:180` (`--profile` help text) | `help="Audited variable-set profile to produce."` | `help="Variable-set profile to produce. exp15/exp25/exp26 are audited (frozen by docs/audit_snapshots/); aires_rad is provisional (no locked snapshot)."` |

`argparse` choices come from `sorted(PROFILES.keys())`, so the new key is auto-picked-up. No other code changes needed.

Note on field placement: rad/heat-flux codes go in `sigma_vars` because `_write_sigma_namelist` concatenates `sigma_vars + land_vars + (["pr"] if accumulate_precip_hours)` into burn7's `code=` line. burn7 emits 2D vs 3D based on its own internal code metadata, not the `vtype` flag — surface fields like `tas`, `pl`, `rss`, `hfss` come out as `(time, lat, lon)` even when the namelist says `vtype=sigma`. Renaming `sigma_vars → burn7_vars` would be more honest but is **out of scope**.

#### A.2 Edits to `skills/plasim-postprocess/SKILL.md`

**Stale-text rewrites** (every "three / audited / frozen" reference):

| Line(s) | Current | Replace with |
|---|---|---|
| `:3` (frontmatter `description`) | `Use when the user wants to convert raw PlaSim binary output files (MOST.NNNN, one per simulated year) into per-sim-year NetCDF files using the audited burn7-based post-processor at src/plasim_postprocessor/. Covers three profiles (exp15, exp25, exp26), …` | `Use when the user wants to convert raw PlaSim binary output files (MOST.NNNN, one per simulated year) into per-sim-year NetCDF files using the burn7-based post-processor at src/plasim_postprocessor/. Covers four profiles — audited: exp15, exp25, exp26; provisional: aires_rad (radiation + heat fluxes, no locked snapshot yet) — …` |
| `:8` (intro paragraph) | `The variable set is selected by --profile and is **frozen** by audit snapshots in docs/audit_snapshots/.` | `The variable set is selected by --profile. Audited profiles (exp15/exp25/exp26) are frozen by audit snapshots in docs/audit_snapshots/. Provisional profiles (aires_rad) are usable but their contract is not yet locked; the CLI emits a stderr warning when a provisional profile is selected.` |
| `:14` (when-to-use bullet) | `Modifying the script (always re-verify against the locked audit snapshots after any change).` | `Modifying the script (always re-verify audited profiles against the locked snapshots after any change; provisional profiles have no snapshot to diff against).` |
| `:18` (section header) | `## Profiles (locked by audit snapshots)` | `## Profiles` |
| Profiles table (after `:18`) | Single 3-row table for exp15/25/26. | Split into two: "**Audited profiles** (locked by `docs/audit_snapshots/`)" with the existing 3 rows, then "**Provisional profiles** (no locked snapshot — outputs may shift)" with one row for aires_rad listing the full variable set. |
| `:55` (CLI block) | `--profile           {exp15,exp25,exp26}    selects audited variable set` | `--profile           {aires_rad,exp15,exp25,exp26}  selects variable set; aires_rad is provisional` |
| `:121` (verification intro) | `…re-verify all three profiles against the locked snapshots **before declaring the change done**.` | `…re-verify all audited profiles against the locked snapshots **before declaring the change done** (provisional profiles like aires_rad have no snapshot to verify against).` |
| `:142` (verification loop) | `for prof in exp15 exp25 exp26; do` | unchanged for Stream A — verification iterates only audited profiles (provisional has no diff target). Add a one-line comment above: `# Iterate only audited profiles; provisional profiles (e.g. aires_rad) have no snapshot to diff against.` |
| `:155` (post-loop assertion) | `All three diffs must be empty.` | `All audited diffs must be empty.` |
| `:179` (snapshots reference) | `Locked snapshots: docs/audit_snapshots/{exp15,exp25,exp26}_manifest.txt — the verification target.` | `Locked snapshots: docs/audit_snapshots/{exp15,exp25,exp26}_manifest.txt — the verification target. (aires_rad snapshot pending Stream B.)` |

**New content additions to SKILL.md:**

- Callout immediately after the new split-tables block:
  > **Provisional profiles** are usable but unaudited. `aires_rad` emits the documented variables today, but the field shapes, sign conventions, and exact variable list are not yet frozen by a snapshot in `docs/audit_snapshots/`. Do not pin downstream training data to a provisional profile until its snapshot lands. The CLI emits a stderr warning when a provisional profile is selected.
- New row in "Common failure modes" table: stderr `WARNING: Profile 'aires_rad' is PROVISIONAL …` → cause: profile has no locked snapshot yet (Stream B pending) → fix: safe to ignore for one-off runs; do not use for reproducible training data until snapshot lands.
- Update "Adding a new profile" section to document the provisional-vs-audited workflow: profile lands in both `PROFILES` and `PROVISIONAL_PROFILES`; running Stream B (probe → semantic validation → snapshot → round-trip) flips it (removes from `PROVISIONAL_PROFILES`, creates the manifest file, updates docstrings/SKILL/audit doc in one commit).

#### A.3 Update audit doc (provisional addendum)

`docs/plasim_postprocessor_audit.md` — append "Addendum: `aires_rad` profile (provisional)":
- Variable list (per the contract table above).
- Audit-method departure: snapshot-as-truth, no legacy diff (the legacy pipeline never emitted radiation).
- **Status: provisional**. Snapshot file `docs/audit_snapshots/aires_rad_manifest.txt` does not yet exist; will be created by Stream B.
- TODO list pointing at Stream B's required steps (probe, semantic validation, snapshot, round-trip, then provisional flip).

#### A.4 Regression test

Stream A must not break the existing 3 profiles or the CLI surface:

```bash
# 1. argparse description includes aires_rad and "provisional"
python3 src/plasim_postprocessor/plasim_postprocessor.py --help 2>&1 | head -3 \
    | grep -E "aires_rad.*provisional"

# 2. argparse choices include aires_rad
python3 src/plasim_postprocessor/plasim_postprocessor.py --help 2>&1 | grep -A1 "profile" \
    | grep -q aires_rad

# 3. Provisional warning fires when aires_rad selected
python3 src/plasim_postprocessor/plasim_postprocessor.py \
    --profile aires_rad --sims 30 --years 12 12 --count-tasks 2>&1 \
    | grep -q "PROVISIONAL"

# 4. Provisional warning does NOT fire for audited profiles
test -z "$(python3 src/plasim_postprocessor/plasim_postprocessor.py \
    --profile exp15 --sims 30 --years 12 12 --count-tasks 2>&1 | grep PROVISIONAL)"

# 5. Existing 3 profiles regress-test clean
for p in exp15 exp25 exp26; do
    rm -rf /tmp/regress_$p
    python3 src/plasim_postprocessor/plasim_postprocessor.py \
        --profile $p --sims 30 --years 12 12 \
        --input-root /scratch/10000/amarchakitus/PLASIM/data \
        --output-root /tmp/regress_$p
    manifest /tmp/regress_$p/sim30/MOST.0012.nc > /tmp/${p}_check.txt
    diff /tmp/${p}_check.txt docs/audit_snapshots/${p}_manifest.txt && echo "$p OK"
done
```

Stream A is shippable when all 5 pass. Feature exists, is honestly labeled provisional, doesn't break any audited profile.

### Stream B — Audit and snapshot lockdown (gated on rad-enabled reference data)

Locks the contract using whatever rad-enabled MOST file is available. Reference is parameterized.

#### B.0 Pick a reference + probe

Defaults assume sim30/MOST.0012 (the existing audit reference) but the entire workflow uses env vars so it can substitute any rad-enabled MOST.NNNN:

```bash
module purge && module load intel/24.0 cdo netcdf python/3.12.11
export LD_LIBRARY_PATH=/opt/apps/intel24/netcdf/4.9.2/x86_64/lib:/home1/09979/awikner/netcdf-4.2/lib:$LD_LIBRARY_PATH

# === EDIT THESE TO POINT AT YOUR REFERENCE ===
REF_INPUT_ROOT=${REF_INPUT_ROOT:-/scratch/10000/amarchakitus/PLASIM/data}
REF_SIM=${REF_SIM:-30}
REF_YEAR=${REF_YEAR:-12}

REF_FILE="$REF_INPUT_ROOT/sim$REF_SIM/MOST.$(printf '%04d' "$REF_YEAR")"
echo "Audit reference: $REF_FILE"
test -f "$REF_FILE" || { echo "Reference file not found"; exit 1; }

cat > /tmp/rad_probe.nl <<'EOF'
code=rss,rls,rst,rlut,rsut,hfss,hfls
vtype=sigma,htype=g,mean=0,netcdf=1
EOF

src/plasim_postprocessor/burn7/Stampede3/burn7 \
    < /tmp/rad_probe.nl "$REF_FILE" /tmp/rad_probe.nc

# Per-variable summary (handles negatives, sci notation, missing values)
cdo -s infon /tmp/rad_probe.nc | tail -n +2

# Variable presence: must be 7
ncdump -h /tmp/rad_probe.nc | grep -cE "^\s+float (rss|rls|rst|rlut|rsut|hfss|hfls)\("
```

**Pass criteria** (all three must hold for the chosen reference):
1. burn7 exits 0 (no "code not found").
2. All 7 vars in `ncdump -h` (count == 7).
3. For each var, `cdo infon` shows min ≠ max AND missing-fraction < 50%.

**Fail handling**: surface to user with the failing reference. Stream A is already merged so the feature is live; just ask for an alternative `REF_SIM`/`REF_YEAR` (or a different `REF_INPUT_ROOT`). Do not modify the script.

#### B.1 Semantic validation (pre-snapshot)

Repeatability ≠ correctness for radiation. Compute global temporal means and check against textbook ranges:

```bash
cdo -s fldmean -timmean -selname,rss,rls,rst,rlut,rsut,hfss,hfls /tmp/rad_probe.nc /tmp/rad_globalmean.nc
ncdump /tmp/rad_globalmean.nc
```

Expected ranges (PlaSim convention TBD; this is the validation):

| Var | Label (burn7) | Expected global mean | Sign convention check |
|-----|---------------|----------------------|-----------------------|
| rss | surface_net_shortwave_flux | ±150 to ±180 W/m² | sign ⇒ direction of net SW at surface. CF: positive = down-into-surface. |
| rls | surface_net_longwave_flux | ±40 to ±60 W/m² (net cooling) | CF: positive = down-into-surface (so net surface LW is *negative* — outgoing). |
| rst | toa_net_shortwave_flux | ±230 to ±250 W/m² | CF: positive = down-into-atmosphere. |
| rlut | toa_net_longwave_flux ("OLR" per common usage) | ±230 to ±250 W/m² | **Sign disambiguates**: positive = down (CF, so OLR is negative); negative = outgoing (PlaSim may use this). Document. |
| rsut | toa_outgoing_shortwave_flux | +90 to +110 W/m² | Should be positive (named "outgoing"). |
| hfss | surface_sensible_heat_flux | ±15 to ±25 W/m² | CF: positive = up-from-surface. |
| hfls | surface_latent_heat_flux | ±75 to ±90 W/m² | CF: positive = up-from-surface. |

**Outputs to write into the audit doc** before snapshotting:
1. Observed global means for all 7 variables.
2. Sign convention determined per variable (positive ⇒ which direction).
3. The OLR-vs-net-LW interpretation for `rlut` (positive or negative when outgoing?).
4. Plausibility verdict per variable.

If any variable falls outside the expected range by >30%, **stop and surface** before locking the snapshot.

#### B.2 Snapshot the contract

Once B.0 and B.1 pass:

```bash
mkdir -p /tmp/aires_rad_audit
python3 src/plasim_postprocessor/plasim_postprocessor.py \
    --profile aires_rad --sims "$REF_SIM" --years "$REF_YEAR" "$REF_YEAR" \
    --input-root "$REF_INPUT_ROOT" --output-root /tmp/aires_rad_audit
manifest "/tmp/aires_rad_audit/sim$REF_SIM/MOST.$(printf '%04d' "$REF_YEAR").nc" \
    > docs/audit_snapshots/aires_rad_manifest.txt
```

Sanity-check before committing:
- 22 vars alphabetical: `hfls, hfss, hus, lat, lev, lev_2, lon, mrso, pl, pr, pr_6h, rls, rlut, rss, rst, rsut, ta, tas, time, ua, va, zg`.
- `lev_2 = 13` pressure values.
- `lev = 10` sigma values.
- `time` count matches `(native cadence count) − 5` for the reference year.

#### B.3 Round-trip verification

Re-run, manifest, diff against snapshot — must be byte-identical:

```bash
rm -rf /tmp/aires_rad_verify
python3 src/plasim_postprocessor/plasim_postprocessor.py \
    --profile aires_rad --sims "$REF_SIM" --years "$REF_YEAR" "$REF_YEAR" \
    --input-root "$REF_INPUT_ROOT" --output-root /tmp/aires_rad_verify
manifest "/tmp/aires_rad_verify/sim$REF_SIM/MOST.$(printf '%04d' "$REF_YEAR").nc" \
    > /tmp/aires_rad_verify.txt
diff /tmp/aires_rad_verify.txt docs/audit_snapshots/aires_rad_manifest.txt
# must exit 0
```

#### B.4 Flip provisional → audited (one atomic commit)

When B.0–B.3 all pass, edit *every* string A.1/A.2 set to "provisional" — same line-by-line table, in reverse:

**`src/plasim_postprocessor/plasim_postprocessor.py`:**

| Line(s) | Provisional (Stream A) | Audited (Stream B.4) |
|---|---|---|
| `PROVISIONAL_PROFILES` constant | `{"aires_rad"}` | `set()` (keep constant + comment for future profiles) |
| `:5-7` docstring opener | "Audited profiles … provisional profiles (aires_rad) …" | "A profile (`--profile`) selects the variable set; all profile contracts are frozen by `docs/audit_snapshots/{profile}_manifest.txt`." |
| `:14-17` docstring body | Two-section Audited / Provisional block | Single block listing exp15/25/26/aires_rad as audited |
| `:174-175` argparse description | `"…(audited profiles: exp15, exp25, exp26; provisional: aires_rad)."` | `"…(audited profiles: aires_rad, exp15, exp25, exp26)."` |
| `:180` `--profile` help | `"…aires_rad is provisional (no locked snapshot)."` | `"Audited variable-set profile to produce."` |

**`skills/plasim-postprocess/SKILL.md`:**

| Line(s) | Provisional (Stream A) | Audited (Stream B.4) |
|---|---|---|
| `:3` frontmatter description | "audited: exp15, exp25, exp26; provisional: aires_rad …" | "Covers four audited profiles (aires_rad, exp15, exp25, exp26), …" |
| `:8` intro paragraph | Audited / Provisional split | "The variable set is selected by `--profile` and is **frozen** by audit snapshots in `docs/audit_snapshots/`." |
| `:14` when-to-use bullet | "audited profiles … provisional profiles have no snapshot …" | "(always re-verify against the locked audit snapshots after any change)." |
| `:18` section header | `## Profiles` | `## Profiles (locked by audit snapshots)` |
| Profiles tables | Two tables (Audited / Provisional) | Single 4-row table including aires_rad |
| `:55` CLI block | `{aires_rad,exp15,exp25,exp26}  selects variable set; aires_rad is provisional` | `{aires_rad,exp15,exp25,exp26}    selects audited variable set` |
| `:121` verification intro | "audited profiles … provisional have no snapshot …" | "all four profiles" |
| `:142` verification loop | `for prof in exp15 exp25 exp26` | `for prof in aires_rad exp15 exp25 exp26` |
| `:155` post-loop | "All audited diffs must be empty." | "All four diffs must be empty." |
| `:179` snapshots ref | "(aires_rad snapshot pending Stream B.)" | `Locked snapshots: docs/audit_snapshots/{aires_rad,exp15,exp25,exp26}_manifest.txt` |
| Provisional callout (added in A.2) | present | **delete** |
| Common-failure-modes row about `WARNING: PROVISIONAL` | present | **delete** |
| "Adding a new profile" workflow text | mentions provisional-vs-audited | retains the workflow description (still applies to future profiles) |

**`docs/plasim_postprocessor_audit.md`:**
- Replace "TODO list pointing at Stream B" in the addendum with "completed; snapshot at `docs/audit_snapshots/aires_rad_manifest.txt`".
- Insert B.1 findings: global-mean table for all 7 rad/flux variables, sign convention determined for each, OLR-vs-net-LW interpretation for `rlut`.

Commit message: `Lock aires_rad audit snapshot; flip provisional → audited`.

After this commit:
- `PROVISIONAL_PROFILES` is empty for aires_rad → the runtime warning no longer fires.
- The contract invariant "every `--profile` choice is audited and frozen by a snapshot" holds again.
- All "audited profiles" / "frozen by snapshots" wording across the codebase consistently includes aires_rad.

## Critical files

Stream A (provisional ship):
- **Edit:** `src/plasim_postprocessor/plasim_postprocessor.py` — add `PROFILES` entry; add `PROVISIONAL_PROFILES` constant; update module docstring (`:14-17`); update argparse description (`:174-175`) + `--profile` help (`:179-180`); add provisional warning in `main()`.
- **Edit:** `skills/plasim-postprocess/SKILL.md` — split into Audited / Provisional tables; provisional callout; failure-mode row; update "Adding a new profile" workflow.
- **Edit:** `docs/plasim_postprocessor_audit.md` — provisional addendum.
- **Edit:** `docs/plasim_postprocessor_refactor_plan.md` — v5 revision-history bullet pointing at this doc.

Stream B (snapshot lockdown):
- **Create:** `docs/audit_snapshots/aires_rad_manifest.txt` — locked contract.
- **Edit:** Same Stream A files, in reverse — flip provisional language to audited; remove from `PROVISIONAL_PROFILES`; add B.1 findings to audit doc.

No changes to argparse `choices`, SLURM template, or the burn7 binary in either stream.

## Verification

Stream A (always required, blocks merge of Stream A):
```bash
# 1. argparse description signals provisional
python3 src/plasim_postprocessor/plasim_postprocessor.py --help 2>&1 | head -3 \
    | grep -E "aires_rad.*provisional"

# 2. argparse choices include aires_rad
python3 src/plasim_postprocessor/plasim_postprocessor.py --help 2>&1 | grep -A1 "profile" \
    | grep -q aires_rad

# 3. Provisional warning fires for aires_rad
python3 src/plasim_postprocessor/plasim_postprocessor.py \
    --profile aires_rad --sims 30 --years 12 12 --count-tasks 2>&1 \
    | grep -q "PROVISIONAL"

# 4. No false-positive warning for audited profiles
test -z "$(python3 src/plasim_postprocessor/plasim_postprocessor.py \
    --profile exp15 --sims 30 --years 12 12 --count-tasks 2>&1 | grep PROVISIONAL)"

# 5. Existing audited profiles regress clean
for p in exp15 exp25 exp26; do
    diff /tmp/${p}_check.txt docs/audit_snapshots/${p}_manifest.txt
done   # all exit 0
```

Stream B (gated on B.0; if probe fails for the chosen reference, ask user for an alternative — Stream A stays shipped):
```bash
# B.0 probe passes for chosen REF_SIM / REF_YEAR
test "$(ncdump -h /tmp/rad_probe.nc | grep -cE '^\s+float (rss|rls|rst|rlut|rsut|hfss|hfls)\(')" = "7"

# B.1 semantic validation: documented global means + sign conventions in audit doc

# B.2/B.3 snapshot exists, round-trip clean
diff /tmp/aires_rad_verify.txt docs/audit_snapshots/aires_rad_manifest.txt   # exit 0

# B.4 flip: PROVISIONAL_PROFILES now empty for aires_rad; warning no longer fires
test -z "$(python3 src/plasim_postprocessor/plasim_postprocessor.py \
    --profile aires_rad --sims "$REF_SIM" --years "$REF_YEAR" "$REF_YEAR" --count-tasks 2>&1 \
    | grep PROVISIONAL)"
```
