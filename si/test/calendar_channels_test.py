"""Tests for the two SI dataset-coupling bugs (see polaris_pbs_notes.md).

Both bugs share one root cause: **SI was silently coupled to the Midway AMIP dataset.**
Midway has 6 surface + 15 diagnostic variables and a `standard` calendar — which is exactly
what the code hardcoded, so nothing ever surfaced there. E3SM has **3** diagnostics and a
**noleap** calendar, and both assumptions break at once.

    python si/test/calendar_channels_test.py       # PASS = "SI_FIXES_OK"

Needs no GPU, no cluster and no data — deliberately, because the failures it pins are
data-shape assumptions, not compute.

  1. calendar : has_year_zero must come from the CALENDAR, not the config
  2. channels : disassemble_input must REQUIRE the channel counts
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cftime  # noqa: E402
import torch  # noqa: E402

from common.utils import disassemble_input  # noqa: E402

# E3SM ships exactly 1460 files per year (365 days x 4/day) for EVERY year, 2016 and 2020
# included — i.e. the archive is noleap. The loader derives the FILENAME from the date
# (amip_new.py: seconds_into_year // 3600 // 6), so the calendar decides which file is read.
SIX_HOURLY = 6


def _file_index(cls, y, m, d):
    """Reproduce amip_new.py's date -> file-index arithmetic."""
    return int((cls(y, m, d) - cls(y, 1, 1)).total_seconds()) // 3600 // SIX_HOURLY


def test_noleap_is_the_archives_calendar():
    """1460 files/yr means every year is 365 days. Pin that noleap agrees and Gregorian doesn't."""
    assert _file_index(cftime.DatetimeNoLeap, 2016, 12, 31) == 1456, "noleap year should end < 1460"
    # The last 6-hourly slot of a noleap year is index 1459 (0-based).
    last = int((cftime.DatetimeNoLeap(2016, 12, 31, 18) - cftime.DatetimeNoLeap(2016, 1, 1)).total_seconds()) // 3600 // SIX_HOURLY
    assert last == 1459, "noleap 2016 must end at file 1459, got %d" % last

    # Gregorian thinks 2016 has 366 days, so it runs off the end of the archive.
    greg_last = int((cftime.DatetimeGregorian(2016, 12, 31, 18) - cftime.DatetimeGregorian(2016, 1, 1)).total_seconds()) // 3600 // SIX_HOURLY
    assert greg_last == 1463, "sanity: Gregorian 2016 should reach 1463"
    assert greg_last > 1459, "Gregorian overruns the 1460-file year — that is the bug"


def test_leap_year_offset_is_exactly_one_day():
    """THE bug: after Feb in a leap year, Gregorian reads the wrong day's file — silently."""
    noleap = _file_index(cftime.DatetimeNoLeap, 2016, 3, 1)
    greg = _file_index(cftime.DatetimeGregorian, 2016, 3, 1)
    assert noleap == 236, "Mar 1 2016 must map to file 0236 under noleap, got %d" % noleap
    assert greg == 240, "sanity: Gregorian maps Mar 1 2016 to 0240, got %d" % greg
    assert greg - noleap == 4, "the drift is exactly one day (4 six-hourly slots)"

    # Before Feb 29 the two agree — which is why a smoke that never leaves early 2016 passes.
    assert _file_index(cftime.DatetimeNoLeap, 2016, 2, 1) == _file_index(cftime.DatetimeGregorian, 2016, 2, 1), \
        "Jan/Feb must agree — that is why the smoke never caught this"


def test_has_year_zero_defaults_are_a_property_of_the_calendar():
    """The root cause: the config pinned False, which only ever matched Gregorian."""
    assert cftime.DatetimeGregorian(2000, 1, 1).has_year_zero is False, \
        "Gregorian's default is False — which is why the Midway configs' False was invisible"
    for cls in (cftime.DatetimeNoLeap, cftime.DatetimeAllLeap, cftime.Datetime360Day):
        assert cls(2000, 1, 1).has_year_zero is True, \
            "%s is idealized: its default is True, so forcing False breaks it" % cls.__name__


def test_forcing_has_year_zero_false_on_noleap_raises():
    """Pin the exact failure the fix removes, so a regression is unmistakable."""
    a = cftime.DatetimeNoLeap(2016, 3, 1)                       # default has_year_zero=True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                          # cftime warns it "ignores" the kwarg
        b = cftime.DatetimeNoLeap(2016, 1, 1, has_year_zero=False)
    try:
        (a - b).total_seconds()
    except TypeError as e:
        assert "year zero" in str(e), "expected the year-zero TypeError, got: %s" % e
    else:
        raise AssertionError(
            "forcing has_year_zero=False on noleap no longer raises — cftime changed; "
            "re-check amip_new.py's assumption before trusting this fix")

    # Not forcing it is what makes the arithmetic work at all.
    assert int((a - cftime.DatetimeNoLeap(2016, 1, 1)).total_seconds()) // 3600 // SIX_HOURLY == 236


# ---------------------------------------------------------------------------
# 2. disassemble_input must require the channel counts
# ---------------------------------------------------------------------------

def _packed(nsurface, ndiagnostic, nlevels, nvars_ml=2, h=4, w=8):
    """A packed (b, c, h, w) tensor whose channels are labelled by value."""
    c = nsurface + ndiagnostic + nvars_ml * nlevels
    x = torch.zeros(1, c, h, w)
    for i in range(c):
        x[:, i] = i
    return x


def test_counts_are_required_not_defaulted():
    """The fix: a caller that forgets the counts must FAIL, not silently mis-split."""
    x = _packed(6, 3, 18)
    try:
        disassemble_input(x)
    except TypeError:
        pass
    else:
        raise AssertionError(
            "disassemble_input still has defaults — a caller that forgets the counts will "
            "silently slice the Midway shape (6/15) out of an E3SM tensor")


def test_e3sm_split_is_correct():
    """E3SM: 6 surface + 3 diagnostic. The old default of 15 diagnostics mis-slices this."""
    nsurface, ndiag, nlev, nml = 6, 3, 18, 2
    x = _packed(nsurface, ndiag, nlev, nvars_ml=nml)
    surface, multilevel, diagnostic = disassemble_input(
        x, nsurface=nsurface, ndiagnostic=ndiag, nlevels=nlev)

    assert surface.shape[1] == nsurface, surface.shape
    assert diagnostic.shape[1] == ndiag, diagnostic.shape
    assert multilevel.shape[1:3] == (nml, nlev), multilevel.shape

    # Channels are labelled by index, so we can assert the split landed in the right place.
    assert surface[0, 0, 0, 0].item() == 0
    assert surface[0, -1, 0, 0].item() == nsurface - 1
    assert diagnostic[0, 0, 0, 0].item() == nsurface, "diagnostics must start right after surface"
    assert diagnostic[0, -1, 0, 0].item() == nsurface + ndiag - 1
    assert multilevel[0, 0, 0, 0, 0].item() == nsurface + ndiag, "multilevel starts after diagnostics"


def test_midway_shape_still_works():
    """The fix must not change Midway: 6 surface + 15 diagnostic, explicitly passed."""
    nsurface, ndiag, nlev, nml = 6, 15, 26, 2
    x = _packed(nsurface, ndiag, nlev, nvars_ml=nml)
    surface, multilevel, diagnostic = disassemble_input(
        x, nsurface=nsurface, ndiagnostic=ndiag, nlevels=nlev)
    assert surface.shape[1] == nsurface
    assert diagnostic.shape[1] == ndiag
    assert multilevel.shape[1:3] == (nml, nlev)


def test_wrong_counts_produce_wrong_answers_not_errors():
    """WHY the counts had to become required, rather than merely documented.

    Using Midway's 15 diagnostics on an E3SM tensor does not raise here — it returns
    confidently wrong tensors. That is the whole hazard: a silent mis-split trains fine.
    """
    nsurface, nlev, nml = 6, 18, 2
    x = _packed(nsurface, 3, nlev, nvars_ml=nml)          # E3SM: 3 diagnostics
    right = disassemble_input(x, nsurface=nsurface, ndiagnostic=3, nlevels=nlev)
    try:
        wrong = disassemble_input(x, nsurface=nsurface, ndiagnostic=15, nlevels=nlev)
    except Exception:
        return                                             # raising is an acceptable outcome too
    assert wrong[2].shape[1] != right[2].shape[1], \
        "the wrong diagnostic count should change the split (this test is the canary)"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print("  ok    %s" % t.__name__)
        except AssertionError as e:
            print("  FAIL  %s: %s" % (t.__name__, e)); failed += 1
        except Exception as e:  # noqa: BLE001
            print("  ERROR %s: %s: %s" % (t.__name__, type(e).__name__, e)); failed += 1
    print()
    if failed:
        print("ERROR %d/%d SI fix tests failed" % (failed, len(tests)))
        sys.exit(1)
    print("SI_FIXES_OK (%d tests)" % len(tests))
