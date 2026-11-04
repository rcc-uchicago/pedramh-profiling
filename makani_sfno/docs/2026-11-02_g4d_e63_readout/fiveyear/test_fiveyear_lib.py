"""Tests for fiveyear_lib.py (numpy only, python3.6+). Prints TEST_FIVEYEAR_LIB_OK."""
import bisect
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fiveyear_lib as fl  # noqa: E402

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)


def test_calendar():
    e = fl.month_edges()
    check(e[0] == 0 and e[-1] == 1460 and len(e) == 13, "edges %r" % e)
    check(e[1] == 124 and e[2] == 236, "Jan/Feb edges %r" % e[:3])
    frames = np.arange(1460)
    brute = np.array([bisect.bisect_right(e, int(f)) for f in frames])      # climate_driver.month_of_frame
    check(np.array_equal(fl.month_of_frames(frames), brute), "month_of_frames != bisect_right")
    check(fl.month_of_frames([0, 123, 124, 1459]).tolist() == [1, 1, 2, 12], "month boundaries")
    check(fl.N_MONTHS == 63 and fl.MONTHS[0] == (2044, 10) and fl.MONTHS[-1] == (2049, 12), "MONTHS")
    check(fl.month_slot([2044, 2045, 2049, 2044, 2050], [10, 1, 12, 9, 1]).tolist() == [0, 3, 62, -1, -1],
          "month_slot")
    try:
        fl.month_of_frames([1460])
        check(False, "frame 1460 accepted")
    except ValueError:
        pass


def fake_member(start_frame=1092, n=7667):
    """(years, frames) of steps 1..n from (2044, start_frame), as climate_driver.valid_time."""
    k = start_frame + np.arange(1, n + 1)
    return 2044 + k // 1460, k % 1460


def test_member_alignment():
    y, f = fake_member()
    check((y[-1], f[-1]) == (2049, 1459), "member 0 should end on 2049 frame 1459, got %r" % ((y[-1], f[-1]),))
    y7, f7 = fake_member(1092 + 16 * 7, 7667 - 16 * 7)
    check((y7[-1], f7[-1]) == (2049, 1459), "member 7 end")
    keep, slot = fl.aligned(y, f)
    check(not keep[(f == 0)].any(), "frame 0 kept")
    check(keep.sum() == len(y) - 5, "exactly the five Jan-1 00:00 frames dropped (%d)" % (len(y) - keep.sum()))


def test_binning():
    y, f = fake_member()
    vals = np.stack([y * 100.0 + fl.month_of_frames(f), np.ones(len(y))], axis=1)      # (n, 2)
    means, counts = fl.bin_by_month(vals, y, f)
    check(means.shape == (63, 2) and counts.shape == (63,), "shapes")
    for i, (yy, mm) in enumerate(fl.MONTHS):
        check(abs(means[i, 0] - (yy * 100 + mm)) < 1e-9, "month %d/%d mean wrong" % (yy, mm))
    nov = fl.MONTHS.index((2044, 11))
    jan45, feb45 = fl.MONTHS.index((2045, 1)), fl.MONTHS.index((2045, 2))
    check(counts[nov] == 120 and counts[jan45] == 123 and counts[feb45] == 112, "counts %r" % counts[:6].tolist())
    check(counts[0] == 1216 - 1093, "Oct 2044 partial count %d" % counts[0])         # frames 1093..1215
    v2 = np.zeros((len(y), 3, 4))
    m2, c2 = fl.bin_by_month(v2, y, f)
    check(m2.shape == (63, 3, 4), "multi-dim binning shape")
    yshort, fshort = fake_member(1092, 400)
    m3, c3 = fl.bin_by_month(np.ones(len(yshort)), yshort, fshort)
    check(np.isnan(m3[-1]) and c3[-1] == 0, "empty month should be NaN")


def test_year_means():
    y, f = fake_member()
    vals = (y * 10.0)[:, None] + np.zeros((1, 2))
    vals[f == 0] = 1e9                                   # frame 0 must not leak in
    ym, n = fl.year_means(vals, y, f)
    check(n.tolist() == [1459] * 5, "year counts %r" % n.tolist())
    check(np.allclose(ym[:, 0], np.array(fl.YEARS) * 10.0), "year means %r" % ym[:, 0].tolist())


def test_bias_stats():
    rng = np.random.default_rng(0)
    m = rng.normal(2.0, 1.0, size=(8, 5))
    truth = np.zeros(5)
    bias, t, sd_scale, dof = fl.bias_stats(m, truth)
    check(dof == 7, "dof")
    check(np.allclose(bias, m.mean(axis=0)), "bias")
    check(np.allclose(sd_scale, m.std(axis=0, ddof=1) * np.sqrt(1 / 8 + 1)), "inflation")
    check(np.allclose(t, bias / sd_scale), "t")
    _, t0, _, _ = fl.bias_stats(np.ones((8, 3)), np.ones(3))
    check(np.isnan(t0).all(), "zero spread and zero bias should give NaN t, not an error")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    if FAILS:
        for x in FAILS:
            print("ERROR TEST_FIVEYEAR_LIB: %s" % x)
        sys.exit(1)
    print("TEST_FIVEYEAR_LIB_OK")
