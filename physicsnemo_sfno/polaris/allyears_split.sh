# shellcheck shell=bash
# ============================================================================
# allyears_split.sh — single source of truth for the ALL-DATA (2015-2049) year split.
#
# Sourced by:
#   polaris/polaris_zarr_e3sm_allyears.pbs      (builds the full stores from this split)
#   polaris/polaris_sfno_allyears_smoke.pbs     (certifies this split's shape cheaply)
# The trainer (polaris/polaris_sfno_allyears.pbs) does NOT read this file — it gates on
# the stores' own recorded attrs, so a legitimately different prepped split can't be
# failed against a stale copy of the defaults.
#
# The archive is ONE continuous SSP245 AMIP run: years 2015-2049, exactly 1460
# samples/year (noleap), 51,100 files total (measured 2026-07-15, adversarially
# re-verified 2026-07-16). This split uses EVERY year:
#
#   train 2015-2046   32 years  46,720 samples
#   val   2047-2049    3 years   4,380 samples
#
# vs polaris_zarr_e3sm_full.pbs's 2015-2044 / 2045-2047, which leaves 2048-2049 unused.
# Val stays at the END of the run: E3SM SSP245 carries a warming trend, so validating on
# years EARLIER than some training years would leak the future into training.
#
# Overridable (qsub -v TRAIN_YEARS="..." ) but GUARDED: allyears_split_check refuses
# overlap, out-of-range years, and — the trap that motivated this file — NON-CONTIGUOUS
# spans. A store built from "2015 2020 2030" records sampling_mode=contiguous, earns a
# valid conversion_complete sentinel, passes every size gate, and hands the trainer
# 8766-hour year seams it silently learns as t->t+6h.
# ============================================================================

TRAIN_YEARS="${TRAIN_YEARS:-$(seq 2015 2046 | tr '\n' ' ')}"
VAL_YEARS="${VAL_YEARS:-$(seq 2047 2049 | tr '\n' ' ')}"

# Gate the split itself. PASS token: "SPLIT_OK". Escape hatch for a deliberately partial
# span (that is polaris_zarr_e3sm_full.pbs's territory): -v ALLYEARS_ALLOW_PARTIAL=1.
allyears_split_check() {
    python - "${TRAIN_YEARS}" "${VAL_YEARS}" <<'PY'
import os, sys
tr = [int(x) for x in sys.argv[1].split()]
va = [int(x) for x in sys.argv[2].split()]
for ys, label in ((tr, "train"), (va, "val")):
    if not ys:
        print("ERROR SPLIT_EMPTY: no %s years" % label); sys.exit(2)
    if ys != sorted(set(ys)):
        print("ERROR SPLIT_UNSORTED_OR_DUP: %s years %s" % (label, ys)); sys.exit(2)
    if ys != list(range(ys[0], ys[-1] + 1)):
        print("ERROR SPLIT_NOT_CONTIGUOUS: %s years %s have gaps." % (label, ys))
        print("  A gapped span converts 'successfully' (sampling_mode=contiguous, valid")
        print("  conversion_complete sentinel) but carries multi-year seams the trainer")
        print("  learns as t->t+6h. Use one unbroken run of years.")
        sys.exit(2)
overlap = set(tr) & set(va)
if overlap:
    print("ERROR SPLIT_OVERLAP: train and val share %s — train/val leakage." % sorted(overlap))
    sys.exit(2)
bad = (set(tr) | set(va)) - set(range(2015, 2050))
if bad:
    print("ERROR YEAR_OUT_OF_RANGE: %s (the archive covers 2015-2049)" % sorted(bad))
    sys.exit(2)
missing = sorted(set(range(2015, 2050)) - set(tr) - set(va))
if missing and os.environ.get("ALLYEARS_ALLOW_PARTIAL") != "1":
    print("ERROR NOT_ALL_YEARS: %s unused. This is the ALL-data variant; a partial span" % missing)
    print("  belongs to polaris_zarr_e3sm_full.pbs. Override with -v ALLYEARS_ALLOW_PARTIAL=1")
    print("  only if you mean it (the run's results would be mislabeled otherwise).")
    sys.exit(2)
print("SPLIT_OK: train %d-%d (%dy, %d samples), val %d-%d (%dy, %d samples)%s"
      % (tr[0], tr[-1], len(tr), len(tr) * 1460,
         va[0], va[-1], len(va), len(va) * 1460,
         ", all 35 years covered" if not missing else " — PARTIAL (allowed by override)"))
PY
}

# Fail fast if the archive does not actually hold 1460 files for every split year —
# discovering that at hour 9 of a conversion wastes the allocation. One os.scandir of
# the (flat, 51,100-entry) plev_data dir; NOT a recursive find.
# PASS token: "CENSUS_OK". Requires $E3SM_ROOT (exported by polaris_env.sh).
allyears_archive_census() {
    python - "${TRAIN_YEARS}" "${VAL_YEARS}" <<'PY'
import collections, os, sys
src = os.path.join(os.environ["E3SM_ROOT"], "h5", "plev_data")
years = [int(x) for a in sys.argv[1:3] for x in a.split()]
counts = collections.Counter()
with os.scandir(src) as it:
    for e in it:
        if e.name.endswith(".h5") and "_" in e.name:
            counts[e.name.split("_")[0]] += 1
bad = {y: counts.get(str(y), 0) for y in years if counts.get(str(y), 0) != 1460}
if bad:
    print("ERROR ARCHIVE_CENSUS_FAILED: expected exactly 1460 files/year (noleap) under")
    print("  %s" % src)
    for y, n in sorted(bad.items()):
        print("    %d: %d files" % (y, n))
    print("  The 2026-07-15 measurement said 1460 for all of 2015-2049; the archive has")
    print("  CHANGED (or E3SM_ROOT points somewhere else). Re-verify before converting.")
    sys.exit(2)
print("CENSUS_OK: %d split years x 1460 files each under %s" % (len(years), src))
PY
}
