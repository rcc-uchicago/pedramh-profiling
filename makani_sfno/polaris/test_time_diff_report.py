"""Tests for time_diff_report (port B step 1 table, handoff §3)."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from time_diff_report import CLAMP, build_table, spearman, summarize  # noqa: E402


def test_weights_match_makani_and_the_clamp_is_physical():
    names = ["Z3_l17", "T", "PRECT"]
    sigma = [0.271, 10.0, 8.3e-8 / 1e-3]      # PRECT: sigma ~8.3e-5 m/s
    delta = [2.8e-5, 5.0, 7.95e-8]            # PRECT's delta far below 1e-4
    rows = {x["channel"]: x for x in build_table(names, sigma, delta)}
    # makani: sigma / clamp(delta, 1e-4) -- Z3_l17 AND PRECT both hit the clamp
    assert rows["Z3_l17"]["clamped"] and rows["PRECT"]["clamped"] and not rows["T"]["clamped"]
    np.testing.assert_allclose(rows["PRECT"]["weight_realised"], sigma[2] / CLAMP)
    np.testing.assert_allclose(rows["T"]["weight_realised"], 2.0)
    # the clamp destroys PRECT's weight relative to its true 1/r
    assert rows["PRECT"]["weight_realised"] < 1e-2 * rows["PRECT"]["weight_1_over_r"]


def test_sorted_slowest_first_and_summary():
    rows = build_table(["a", "b", "c"], [1.0, 1.0, 1.0], [0.5, 0.01, 1.0])
    assert [x["channel"] for x in rows] == ["b", "a", "c"]
    s = summarize(rows, "t")
    assert s["weight_max_channel"] == "b" and s["r_span"] == 100.0


def test_spearman_is_rank_based():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 1000]) == 1.0
    assert spearman([1, 2, 3], [3, 2, 1]) == -1.0
