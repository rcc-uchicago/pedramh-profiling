"""Tests for polaris/climate_screen_summary.py (fine-tune stability handoff §2, gate S0a).

Member NetCDFs are written by the **real** ``climate_driver.MemberWriter`` (so the
format under test is the one the screen job produces), and the truth series is
read from tiny real ``YYYY.h5`` files laid out like the pack. The handoff's seeded
faults are each shown red here:

  - a truncated member ranks last, even when it looks cleaner before it dies;
  - drift sign: model below truth is negative, model above truth positive;
  - truth alignment off by one lead is refused, both as a shifted start and as
    shifted per-lead valid times.

Needs torch (the writer's module imports it), h5py and netCDF4: run on a compute
node (``polaris_climate_screen_test.pbs``), never the login node.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")
pytest.importorskip("netCDF4")

_MAKANI = Path(__file__).resolve().parents[1]
for p in (_MAKANI / "polaris", _MAKANI / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import climate_screen_summary as S  # noqa: E402
from sfno_ensemble.scores import equiangular_weights  # noqa: E402
from sfno_inference import climate_driver as cd  # noqa: E402

H, W = 4, 8
FPY = S.FRAMES_PER_YEAR
START = (2044, 1092)
N = 1460
NAMES = ["PS", "T_l17", "Z3_l10", "TREFHT", "TMQ", "X"]
NT = len(S.TRUTH_CHANNELS)                 # PS, T_l17, Z3_l10, TREFHT, TMQ


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _truth_value(ch: int, year: int, frame: np.ndarray) -> np.ndarray:
    """A per-(channel, year, frame) level that makes misalignment visible."""
    return 1000.0 * (ch + 1) + 10.0 * (year - 2044) * FPY + frame


LAT_PATTERN = np.array([0.0, 0.0, 0.0, 8.0])     # asymmetric: cos weights != plain mean


def _write_year(path: Path, year: int) -> None:
    frames = np.arange(FPY, dtype=np.float64)
    # every truth channel is state except TREFHT, which sits in the diagnostic
    # dataset to exercise the by-name fallback; value index = TRUTH_CHANNELS position
    tc = list(S.TRUTH_CHANNELS)
    st_names = [n for n in tc if n != "TREFHT"]
    state = np.empty((FPY, len(st_names), H, W), np.float64)   # float64: exact asserts
    for j, n in enumerate(st_names):
        state[:, j] = (_truth_value(tc.index(n), year, frames)[:, None, None]
                       + LAT_PATTERN[None, :, None])
    diag = np.empty((FPY, 1, H, W), np.float64)
    diag[:, 0] = (_truth_value(tc.index("TREFHT"), year, frames)[:, None, None]
                  + LAT_PATTERN[None, :, None])
    with h5py.File(path, "w") as f:
        f["fields_state"] = state
        f["fields_diagnostic"] = diag
        f["channel_state"] = np.array(st_names, dtype="S")
        f["channel_diagnostic"] = np.array(["TREFHT"], dtype="S")


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    root = tmp_path_factory.mktemp("pack")
    for year, split in ((2044, "train"), (2045, "valid")):
        (root / split).mkdir()
        _write_year(root / split / f"{year}.h5", year)
    return root


@pytest.fixture(scope="module")
def truth(pack):
    return S.build_truth(pack, START, N)


def _expected_truth(k: int, ch: int) -> float:
    y, fr = S.valid_time(START, k)
    w = equiangular_weights(H)
    return float(_truth_value(ch, y, np.array(fr)) + LAT_PATTERN @ w)


def _write_member(path: Path, truth: dict, *, label: str, epoch: int, steps: int,
                  truncated_at: int = -1, offsets: dict | None = None,
                  past: dict | None = None, start=START, frame_shift: int = 0) -> Path:
    """Member whose global mean = truth + ``offsets[ch](lead)``; anomaly σ from ``past``.

    ``past = {channel: first lead > 3σ}`` sets ``anom_rms_sigma`` to 5 from that
    lead, 0.5 before it. ``frame_shift`` seeds per-lead valid-time misalignment.
    """
    C = len(NAMES)
    leads = np.arange(1, steps + 1)
    gm = np.zeros((steps, C))
    tch = truth["channels"]
    for c, name in enumerate(NAMES):
        base = truth["global_mean"][:steps, tch.index(name)] if name in tch else np.zeros(steps)
        off = (offsets or {}).get(name)
        gm[:, c] = base + (off(leads) if off else 0.0)
    anom = np.full((steps, C), 0.5)
    for name, lead in (past or {}).items():
        if lead <= steps:
            anom[lead - 1:, NAMES.index(name)] = 5.0
    series = {"global_mean_z": gm, "std_sigma": np.full((steps, C), 0.5), "anom_rms_sigma": anom}
    vt = np.array([cd.valid_time(start[0], start[1], k + frame_shift) for k in leads])
    w = cd.MemberWriter(path, channel_names=NAMES, lat=90.0 - (np.arange(H) + 0.5) * 180.0 / H,
                        lon=np.arange(W) * 360.0 / W, out_bias=np.zeros(C), out_scale=np.ones(C))
    w.finish(series=series, stability=cd.stability_summary(series), time_mean_z=None,
             valid_year=vt[:, 0].astype(np.int32), valid_frame=vt[:, 1].astype(np.int32),
             score_start_step=368,
             attrs=dict(member_id=label, ckpt_epoch=epoch, start_year=start[0],
                        start_frame=start[1], truncated_at_step=truncated_at,
                        truncated_channel="PS" if truncated_at > 0 else "",
                        run_dir="/runs/x", ckpt=f"/runs/x/{label}.tar"))
    return path


# ---------------------------------------------------------------------------
# truth series
# ---------------------------------------------------------------------------

def test_truth_matches_direct_read_across_year_boundary(truth):
    assert truth["global_mean"].shape == (N, NT)
    assert (int(truth["valid_year"][367]), int(truth["valid_frame"][367])) == (2045, 0)
    assert (int(truth["valid_year"][366]), int(truth["valid_frame"][366])) == (2044, 1459)
    assert (int(truth["valid_year"][-1]), int(truth["valid_frame"][-1])) == (2045, 1092)
    for k in (1, 367, 368, 369, 600, 1460):
        for c in range(NT):
            assert truth["global_mean"][k - 1, c] == pytest.approx(_expected_truth(k, c), abs=1e-6)


def test_truth_uses_equiangular_area_weights(truth):
    # the plain lat mean of LAT_PATTERN is 2.0; the cos-weighted one is ~1.17
    w = equiangular_weights(H)
    assert abs(LAT_PATTERN @ w - LAT_PATTERN.mean()) > 0.1
    y, fr = S.valid_time(START, 1)
    assert truth["global_mean"][0, 0] - _truth_value(0, y, np.array(fr)) == pytest.approx(
        LAT_PATTERN @ w, abs=1e-6)


def test_truth_reads_diagnostic_channel_by_name(truth):
    assert truth["channels"].tolist() == list(S.TRUTH_CHANNELS)
    assert truth["global_mean"][599, 3] == pytest.approx(_expected_truth(600, 3), abs=1e-6)


def test_truth_npz_roundtrip(truth, tmp_path):
    S.save_truth(truth, tmp_path / "t.npz")
    t = S.load_truth(tmp_path / "t.npz")
    assert t["channels"] == list(S.TRUTH_CHANNELS) and t["start"] == START
    np.testing.assert_array_equal(t["global_mean"], truth["global_mean"])


# ---------------------------------------------------------------------------
# per-member metrics
# ---------------------------------------------------------------------------

def _load_truth(truth):
    t = dict(truth)
    t["channels"] = [str(x) for x in truth["channels"]]
    t["start"] = tuple(int(x) for x in truth["start"])
    return t


def test_survivor_metrics_and_drift_sign(truth, tmp_path):
    t = _load_truth(truth)
    # PS loses 1350 Pa by lead 600 (linear), T_l17 warms +2 K by 1460, Z3_l10 -30 m
    p = _write_member(tmp_path / "a.nc", t, label="C1_e05", epoch=5, steps=N,
                      offsets={"PS": lambda L: -1350.0 * L / 600,
                               "T_l17": lambda L: 2.0 * L / N,
                               "Z3_l10": lambda L: -30.0 * L / N},
                      past={"X": 700, "T_l17": 900})
    r = S.summarize_member(S.load_member(p), t, N)
    assert r["survived"] == 1 and r["truncated_at_step"] == -1 and r["steps_run"] == N
    assert r["n_past_3sigma"] == 2
    assert r["first_past_3sigma"] == "X@700 T_l17@900"
    assert r["median_cross_3sigma"] == -1           # 2 of 5 past never move the median
    assert r["ps_drift_hpa@600"] == pytest.approx(-13.5, abs=1e-6)       # sign: loss < 0
    assert r["ps_drift_hpa@1460"] == pytest.approx(-13.5 * N / 600, abs=1e-6)
    assert r["t17_drift_k@1460"] == pytest.approx(+2.0, abs=1e-6)          # sign: warm > 0
    assert r["z10_drift_m@1460"] == pytest.approx(-30.0, abs=1e-6)
    assert r["trefht_drift_k@1460"] == pytest.approx(0.0, abs=1e-6)
    assert r["tmq_drift_kgm2@1460"] == pytest.approx(0.0, abs=1e-6)
    assert r["dry_drift_hpa@1460"] == pytest.approx(r["ps_drift_hpa@1460"], abs=1e-6)
    assert r["epoch_check"] == "ok"


def test_median_cross_needs_a_channel_majority(truth, tmp_path):
    t = _load_truth(truth)
    p = _write_member(tmp_path / "m.nc", t, label="m", epoch=1, steps=N,
                      past={"PS": 100, "T_l17": 200, "Z3_l10": 300, "TMQ": 400})
    r = S.summarize_member(S.load_member(p), t, N)
    # 6 channels: at 300 the median is (0.5+5)/2 = 2.75; at 400 four are past -> 5
    assert r["n_past_3sigma"] == 4 and r["median_cross_3sigma"] == 400


def test_truncated_member_is_not_a_survivor_and_has_no_1460_drift(truth, tmp_path):
    t = _load_truth(truth)
    p = _write_member(tmp_path / "tr.nc", t, label="A_e243", epoch=243, steps=594,
                      truncated_at=595, offsets={"PS": lambda L: -100.0 * np.ones_like(L)})
    r = S.summarize_member(S.load_member(p), t, N)
    assert r["survived"] == 0 and r["truncated_at_step"] == 595 and r["steps_run"] == 594
    assert math.isnan(r["ps_drift_hpa@1460"]) and math.isnan(r["ps_drift_hpa@600"])
    assert r["ps_drift_hpa@last"] == pytest.approx(-1.0, abs=1e-9)


def test_seeded_truncated_member_ranks_last(truth, tmp_path):
    """Fault: a member that dies clean (0 channels past 3σ) vs survivors that are ugly."""
    t = _load_truth(truth)
    dead = _write_member(tmp_path / "dead.nc", t, label="dead", epoch=1, steps=400,
                         truncated_at=401)
    ugly = _write_member(tmp_path / "ugly.nc", t, label="ugly", epoch=1, steps=N,
                         past={n: 50 for n in NAMES},
                         offsets={"PS": lambda L: -9000.0 * L / N})
    ok = _write_member(tmp_path / "ok.nc", t, label="ok", epoch=1, steps=N)
    rows, errors = S.summarize([dead, ugly, ok], t, N)
    assert errors == []
    assert [r["label"] for r in rows] == ["ok", "ugly", "dead"]
    assert [r["rank"] for r in rows] == [1, 2, 3]


def test_seeded_truth_start_off_by_one_is_refused(pack, tmp_path):
    t_shift = _load_truth(S.build_truth(pack, (2044, 1093), N))
    t = _load_truth(S.build_truth(pack, START, N))
    p = _write_member(tmp_path / "s.nc", t, label="s", epoch=1, steps=N)
    with pytest.raises(S.ScreenError, match="TRUTH_MISALIGNED"):
        S.summarize_member(S.load_member(p), t_shift, N)
    # and the aligned pair is clean
    assert S.summarize_member(S.load_member(p), t, N)["survived"] == 1


def test_seeded_per_lead_valid_time_off_by_one_is_refused(truth, tmp_path):
    t = _load_truth(truth)
    p = _write_member(tmp_path / "v.nc", t, label="v", epoch=1, steps=N, frame_shift=1)
    with pytest.raises(S.ScreenError, match="TRUTH_MISALIGNED: lead 1 "):
        S.summarize_member(S.load_member(p), t, N)


def test_incomplete_member_without_truncation_is_refused(truth, tmp_path):
    t = _load_truth(truth)
    p = _write_member(tmp_path / "i.nc", t, label="i", epoch=1, steps=1000)
    with pytest.raises(S.ScreenError, match="MEMBER_INCOMPLETE"):
        S.summarize_member(S.load_member(p), t, N)


def test_epoch_label_mismatch_is_flagged(truth, tmp_path):
    t = _load_truth(truth)
    p = _write_member(tmp_path / "e.nc", t, label="B_e21", epoch=22, steps=N)
    assert S.summarize_member(S.load_member(p), t, N)["epoch_check"] == "MISMATCH(label 21)"
    assert S.expected_epoch("nf4p_r1") is None


# ---------------------------------------------------------------------------
# ranking rule
# ---------------------------------------------------------------------------

def _row(label, survived, n_past, ps1460, trunc=-1):
    return {"label": label, "survived": survived, "n_past_3sigma": n_past,
            "ps_drift_hpa@1460": ps1460, "truncated_at_step": trunc}


def test_ranking_rule_order():
    rows = S.rank_rows([
        _row("dead_late", 0, 0, float("nan"), trunc=1400),
        _row("surv_3past_small_ps", 1, 3, -0.1),
        _row("surv_0past_big_ps", 1, 0, -40.0),
        _row("surv_0past_small_ps_pos", 1, 0, +5.0),
        _row("dead_early", 0, 0, float("nan"), trunc=300),
        _row("dead_earliest", 0, 0, float("nan"), trunc=200),
        _row("dead_late_many_past", 0, 50, float("nan"), trunc=1450),
    ])
    assert [r["label"] for r in rows] == [
        "surv_0past_small_ps_pos",   # |+5| < |-40|: magnitude, not signed value
        "surv_0past_big_ps",
        "surv_3past_small_ps",       # fewer past-3σ beats smaller drift
        "dead_late",                 # among non-survivors, later truncation first …
        "dead_early",
        "dead_earliest",
        "dead_late_many_past",       # … but only after n_past_3sigma
    ]


# ---------------------------------------------------------------------------
# CLI end to end
# ---------------------------------------------------------------------------

def test_cli_truth_then_summarize(pack, tmp_path, capsys):
    tnpz = tmp_path / "truth.npz"
    assert S.main(["truth", "--pack", str(pack), "--start", "2044", "1092",
                   "--n-leads", str(N), "--out", str(tnpz)]) == 0
    assert "CLIMATE_SCREEN_TRUTH_OK leads=1460" in capsys.readouterr().out
    t = S.load_truth(tnpz)
    ncs = [_write_member(tmp_path / f"{lab}.nc", t, label=lab, epoch=e, steps=N)
           for lab, e in (("C1_e01", 1), ("C1_e02", 2))]
    csv_p, md_p = tmp_path / "s.csv", tmp_path / "s.md"
    assert S.main(["summarize", "--truth", str(tnpz), "--csv", str(csv_p),
                   "--md", str(md_p), *map(str, ncs)]) == 0
    out = capsys.readouterr().out
    assert "CLIMATE_SCREEN_SUMMARY_OK n=2" in out
    rows = list(csv.DictReader(open(csv_p)))
    assert [r["label"] for r in rows] == ["C1_e01", "C1_e02"]
    assert list(rows[0].keys()) == list(S.CSV_COLUMNS)
    assert md_p.read_text().count("\n| ") == 3            # header + 2 rows


def test_cli_summarize_fails_loudly_on_a_bad_member(pack, tmp_path, capsys):
    tnpz = tmp_path / "truth.npz"
    S.save_truth(S.build_truth(pack, START, N), tnpz)
    t = S.load_truth(tnpz)
    good = _write_member(tmp_path / "g.nc", t, label="g", epoch=1, steps=N)
    bad = _write_member(tmp_path / "b.nc", t, label="b", epoch=1, steps=N, frame_shift=1)
    rc = S.main(["summarize", "--truth", str(tnpz), "--csv", str(tmp_path / "s.csv"),
                 "--md", str(tmp_path / "s.md"), str(good), str(bad)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "ERROR CLIMATE_SCREEN_MEMBER" in out and "TRUTH_MISALIGNED" in out
    assert "CLIMATE_SCREEN_SUMMARY_OK" not in out


def test_dry_air_drift_subtracts_the_water_column(truth, tmp_path):
    """PS low by 300 Pa and TMQ low by 10 kg/m2: dry-air drift = -300 + g*10 Pa."""
    t = _load_truth(truth)
    p = _write_member(tmp_path / "d.nc", t, label="d", epoch=1, steps=N,
                      offsets={"PS": lambda L: -300.0 + 0 * L, "TMQ": lambda L: -10.0 + 0 * L})
    r = S.summarize_member(S.load_member(p), t, N)
    assert r["ps_drift_hpa@1460"] == pytest.approx(-3.0, abs=1e-9)
    assert r["tmq_drift_kgm2@1460"] == pytest.approx(-10.0, abs=1e-9)
    assert r["dry_drift_hpa@1460"] == pytest.approx((-300.0 + S.GRAVITY * 10.0) / 100, abs=1e-9)
