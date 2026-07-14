"""Tests for src/sfno_eval/climatology.py.

Coverage (per docs/sfno_eval_plan.md §C.2 and §H):

  - ``parse_anchor`` round-trips the packager's
    ``"days since 0YYY-MM-DD HH:MM:SS"`` format.
  - ``calendar_bin`` maps (month, day, hour) to (doy, hq) correctly,
    including Feb 29 (doy=59).
  - ``ClimatologyAccumulator``:
      * mean/std match a numpy reference on synthetic data;
      * empty bins return mean=0, std=0, n=0 (no NaN);
      * single-contributor bins return std=0 (Welford degenerate case).
  - ``ingest_file`` on synthetic h5 produces the right binning for
    a leap and a non-leap file.
  - ``build_climatology`` end-to-end on 3 fake h5 files.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest


from sfno_eval import climatology as cm  # noqa: E402


# ---------------------------------------------------------------------------
# parse_anchor + calendar_bin
# ---------------------------------------------------------------------------

class TestAnchor:
    def test_basic(self):
        assert cm.parse_anchor("days since 0126-08-01 00:00:00") == (126, 8, 1, 0, 0, 0)

    def test_bytes(self):
        assert cm.parse_anchor(b"days since 0019-08-01 06:00:00") == (19, 8, 1, 6, 0, 0)

    def test_rejects_garbage(self):
        with pytest.raises(ValueError, match="unparseable"):
            cm.parse_anchor("the dawn of time")


class TestCalendarBin:
    def test_jan_1_midnight(self):
        assert cm.calendar_bin(1, 1, 0) == (0, 0)

    def test_feb_29_noon(self):
        # Jan (31) + Feb 29 - 1 = 59
        assert cm.calendar_bin(2, 29, 12) == (59, 2)

    def test_dec_31_eighteen_z(self):
        # 31+29+31+30+31+30+31+31+30+31+30 = 335; +30 = 365
        assert cm.calendar_bin(12, 31, 18) == (365, 3)

    def test_hour_quarter_mapping(self):
        for h, want_hq in [(0, 0), (5, 0), (6, 1), (11, 1), (12, 2), (17, 2), (18, 3), (23, 3)]:
            _, hq = cm.calendar_bin(1, 1, h)
            assert hq == want_hq, f"hour {h} → hq {hq}, expected {want_hq}"


# ---------------------------------------------------------------------------
# ClimatologyAccumulator
# ---------------------------------------------------------------------------

class TestAccumulator:
    def test_mean_and_std_match_numpy(self):
        rng = np.random.default_rng(42)
        n_chan, H, W = 3, 4, 8
        acc = cm.ClimatologyAccumulator(n_chan=n_chan, H=H, W=W)
        # Drop 5 samples into the same bin; verify mean/std.
        samples = [
            rng.standard_normal((n_chan, H, W)).astype(np.float32) for _ in range(5)
        ]
        for s in samples:
            acc.update(0, 0, s)
        out = acc.finalize()
        ref_mean = np.mean(np.stack(samples), axis=0)
        ref_std = np.std(np.stack(samples), axis=0, ddof=1)
        np.testing.assert_allclose(out["mean"][0, 0], ref_mean, rtol=1e-5, atol=1e-5)
        # Welford variance is sample variance (ddof=1).
        np.testing.assert_allclose(out["std"][0, 0], ref_std, rtol=1e-4, atol=1e-4)
        assert out["n_contributors"][0, 0] == 5
        # Other bins remain empty.
        assert out["n_contributors"][0, 1] == 0
        np.testing.assert_array_equal(out["mean"][0, 1], np.zeros((n_chan, H, W)))

    def test_empty_bin_no_nan(self):
        acc = cm.ClimatologyAccumulator(n_chan=2, H=2, W=2)
        # No updates at all — finalize should still return finite arrays.
        out = acc.finalize()
        assert np.all(np.isfinite(out["mean"]))
        assert np.all(np.isfinite(out["std"]))
        assert (out["n_contributors"] == 0).all()

    def test_single_contributor_std_zero(self):
        acc = cm.ClimatologyAccumulator(n_chan=1, H=2, W=2)
        acc.update(10, 1, np.ones((1, 2, 2), dtype=np.float32) * 3.7)
        out = acc.finalize()
        # With ddof=1, n=1 has zero variance by convention.
        np.testing.assert_array_equal(out["std"][10, 1], np.zeros((1, 2, 2)))
        np.testing.assert_allclose(out["mean"][10, 1], np.full((1, 2, 2), 3.7), rtol=1e-6)

    def test_rejects_wrong_sample_shape(self):
        acc = cm.ClimatologyAccumulator(n_chan=3, H=4, W=8)
        with pytest.raises(ValueError, match="sample shape"):
            acc.update(0, 0, np.zeros((3, 4, 7), dtype=np.float32))


# ---------------------------------------------------------------------------
# ingest_file + build_climatology
# ---------------------------------------------------------------------------

def _make_synthetic_h5(
    path: Path, *, anchor_year: int, n_samples: int,
    n_state: int = 2, n_diag: int = 1, H: int = 4, W: int = 8,
) -> None:
    """Write a stand-in h5 with state ‖ diagnostic + 6h-spaced time_plasim."""
    rng = np.random.default_rng(anchor_year)
    with h5py.File(path, "w") as f:
        f.attrs["plasim_time_units"] = f"days since {anchor_year:04d}-08-01 00:00:00"
        f.attrs["split"] = "train"
        time = np.arange(n_samples, dtype=np.float64) * 0.25
        f.create_dataset("time_plasim", data=time)
        f.create_dataset(
            "fields_state",
            data=rng.standard_normal((n_samples, n_state, H, W)).astype(np.float32),
        )
        f.create_dataset(
            "fields_diagnostic",
            data=rng.standard_normal((n_samples, n_diag, H, W)).astype(np.float32),
        )


class TestIngestFile:
    def test_ingest_non_leap_file(self, tmp_path):
        """anchor 99 → candidate 100, NOT leap; n_samples=1455 (not 1459)."""
        path = tmp_path / "MOST.0094.h5"
        _make_synthetic_h5(path, anchor_year=99, n_samples=1455)
        acc = cm.ClimatologyAccumulator(n_chan=3, H=4, W=8)
        n = cm.ingest_file(acc, path)
        assert n == 1455
        # Total contributors across all bins = n_samples
        assert acc.n_contrib.sum() == 1455
        # No Feb-29 contributor (year 100 is not leap).
        assert acc.n_contrib[59, :].sum() == 0  # doy 59 = Feb 29

    def test_ingest_leap_file_populates_feb29(self, tmp_path):
        """anchor 19 → candidate 20, IS leap; should hit doy=59 four times."""
        path = tmp_path / "MOST.0014.h5"
        _make_synthetic_h5(path, anchor_year=19, n_samples=1459)
        acc = cm.ClimatologyAccumulator(n_chan=3, H=4, W=8)
        n = cm.ingest_file(acc, path)
        assert n == 1459
        assert acc.n_contrib[59, :].sum() == 4  # Feb 29 has 4 6h-slots

    def test_rejects_channel_mismatch(self, tmp_path):
        path = tmp_path / "MOST.0094.h5"
        _make_synthetic_h5(path, anchor_year=99, n_samples=8, n_state=2, n_diag=1)
        # Accumulator built for 5 channels but file has 3 (2+1).
        acc = cm.ClimatologyAccumulator(n_chan=5, H=4, W=8)
        with pytest.raises(RuntimeError, match="channels"):
            cm.ingest_file(acc, path)


class TestBuildClimatology:
    def test_three_files(self, tmp_path):
        # Use a small n_samples so the test runs in <1 s.
        files = []
        for y in (19, 20, 21):  # mix of leap (20→21 leap), non-leap
            p = tmp_path / f"MOST.{y:04d}.h5"
            _make_synthetic_h5(p, anchor_year=y, n_samples=64)
            files.append(p)
        out = cm.build_climatology(files, n_chan=3, H=4, W=8)
        # The total contributors across all bins should equal sum of n_samples.
        assert out["n_contributors"].sum() == 3 * 64
        # mean / std arrays have the right shape.
        assert out["mean"].shape == (366, 4, 3, 4, 8)
        assert out["std"].shape == (366, 4, 3, 4, 8)
        # std is finite everywhere (no NaN).
        assert np.all(np.isfinite(out["std"]))


# ---------------------------------------------------------------------------
# lookup_clim_at
# ---------------------------------------------------------------------------

class TestLookupClimAt:
    def test_returns_mean_when_populated(self):
        clim_n = np.zeros((366, 4), dtype=np.int64)
        clim_n[59, 2] = 24  # Feb 29 noon, leap-year contributors
        clim_mean = np.full((366, 4, 1, 2, 2), -1.0, dtype=np.float32)
        clim_mean[59, 2] = np.full((1, 2, 2), 7.7, dtype=np.float32)
        out = cm.lookup_clim_at(clim_mean, clim_n, month=2, day=29, hour=12)
        assert out is not None
        np.testing.assert_array_equal(out, np.full((1, 2, 2), 7.7, dtype=np.float32))

    def test_returns_none_when_empty(self):
        clim_n = np.zeros((366, 4), dtype=np.int64)
        clim_mean = np.zeros((366, 4, 1, 2, 2), dtype=np.float32)
        out = cm.lookup_clim_at(clim_mean, clim_n, month=2, day=29, hour=12)
        assert out is None
