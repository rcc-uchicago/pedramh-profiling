"""Tests for scripts/negativity_rollout.py (the per-lead statistic) and
polaris/negativity_summary.py (truth series + summary). Run on a compute node
(polaris_negativity_probe.pbs MODE=test): needs torch, h5py.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

_M = Path(__file__).resolve().parents[1]
for p in (_M / "scripts", _M / "polaris", _M / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import negativity_rollout as R  # noqa: E402
import negativity_summary as S  # noqa: E402
from sfno_ensemble.scores import equiangular_weights  # noqa: E402

H, W, FPY = 4, 8, 1460
START = (2044, 1092)
N = 1460


# ---------------------------------------------------------------------------
# per-lead statistic
# ---------------------------------------------------------------------------

def test_negativity_stats_physical_units_and_area_weights():
    names = ["PS", "RELHUM_l00", "PRECT"]
    mean = torch.tensor([1e5, 50.0, 1e-5], dtype=torch.float64)
    std = torch.tensor([1e3, 25.0, 1e-5], dtype=torch.float64)
    w = torch.as_tensor(equiangular_weights(H))
    phys = torch.zeros(3, H, W, dtype=torch.float64)
    phys[0] = 1e5
    phys[1] = 40.0
    phys[1, 0, :] = -5.0              # top row (polar, small weight) negative
    phys[2] = 2e-5
    phys[2, 1, :4] = -1e-6            # half of row 1 negative
    z = ((phys - mean[:, None, None]) / std[:, None, None])[None].to(torch.bfloat16).float()
    idx = R.select_channels(names)
    assert idx == [1, 2]
    frac, mn, negm = R.negativity_stats(z, idx, mean, std, w)
    wn = equiangular_weights(H)
    assert float(frac[0]) == pytest.approx(wn[0], rel=1e-6)                # whole row 0
    assert float(frac[1]) == pytest.approx(0.5 * wn[1], rel=1e-6)          # half of row 1
    assert float(mn[0]) == pytest.approx(-5.0, abs=0.3)                    # bf16 z rounding
    assert float(negm[0]) == pytest.approx(-5.0 * wn[0], rel=0.1)
    assert float(negm[0]) < 0 and float(negm[1]) < 0


def test_negativity_stats_zero_when_all_positive():
    mean = torch.zeros(2, dtype=torch.float64)
    std = torch.ones(2, dtype=torch.float64)
    z = torch.rand(1, 2, H, W) + 0.1
    frac, mn, negm = R.negativity_stats(z, [0, 1], mean, std,
                                        torch.as_tensor(equiangular_weights(H)))
    assert torch.all(frac == 0) and torch.all(negm == 0) and torch.all(mn > 0)


def test_select_channels_default_and_empty():
    names = ["PS", "TMQ", "RELHUM_l00", "RELHUM_l17", "RHREFHT", "SOILWATER_10CM", "PRECT", "T_l00"]
    assert [names[i] for i in R.select_channels(names)] == names[1:7]
    with pytest.raises(ValueError, match="NO_CHANNELS"):
        R.select_channels(["PS", "T_l00"])


# ---------------------------------------------------------------------------
# truth + summary
# ---------------------------------------------------------------------------

CH = ["RELHUM_l00", "RELHUM_l01", "RHREFHT", "PRECT"]


def _write_year(path, year):
    """RELHUM_l00 negative on one cell in frames divisible by 8 of 2045 only; others >= 0."""
    st = np.full((FPY, 3, H, W), 50.0)
    if year == 2045:
        st[::8, 0, 2, 3] = -1.0
    dg = np.full((FPY, 1, H, W), 1e-5)
    with h5py.File(path, "w") as f:
        f["fields_state"] = st
        f["fields_diagnostic"] = dg
        f["channel_state"] = np.array(CH[:3], dtype="S")
        f["channel_diagnostic"] = np.array(["PRECT"], dtype="S")


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    root = tmp_path_factory.mktemp("pack")
    for y, s in ((2044, "train"), (2045, "valid")):
        (root / s).mkdir()
        _write_year(root / s / f"{y}.h5", y)
    return root


def test_truth_stride_and_year_boundary(pack):
    t = S.build_truth(pack, START, N, CH, stride=4)
    assert t["leads"][0] == 4 and t["leads"][-1] == 1460 and t["leads"].size == 365
    # lead 368 = 2045 frame 0 (negative), lead 372 = frame 4 (not), lead 376 = frame 8 (negative)
    j = {int(k): i for i, k in enumerate(t["leads"])}
    assert (int(t["valid_year"][j[368]]), int(t["valid_frame"][j[368]])) == (2045, 0)
    w = equiangular_weights(H)
    assert t["neg_frac"][j[368], 0] == pytest.approx(w[2] / W)
    assert t["neg_frac"][j[372], 0] == 0 and t["neg_frac"][j[376], 0] > 0
    assert t["neg_frac"][j[364], 0] == 0                          # 2044 never negative
    assert np.all(t["neg_frac"][:, 1:] == 0) and t["min"][j[368], 0] == -1.0


def _member(tmp_path, truth, *, label, steps, frac_fn, trunc=-1, shift=0, chans=CH):
    vt = np.array([S.valid_time(START, k + shift) for k in range(1, steps + 1)])
    fr = np.array([[frac_fn(k, c) for c in range(len(chans))] for k in range(1, steps + 1)],
                  dtype=float)                                   # c = channel INDEX
    mn = np.where(fr > 0, -2.0, 1.0)
    p = tmp_path / f"neg_{label}.npz"
    np.savez(p, channels=np.array(chans), neg_frac=fr, min=mn, neg_mean=-fr,
             valid_year=vt[:, 0], valid_frame=vt[:, 1], start=np.array(START),
             truncated_at_step=trunc, member_id=label, ckpt="x", run_dir="y", dry_air_fix=0)
    return p


def test_summarize_member_metrics_and_truth_columns(pack, tmp_path):
    t = S.build_truth(pack, START, N, CH, stride=4)
    p = _member(tmp_path, t, label="m", steps=N,
                frac_fn=lambda k, c: 0.02 if (c == 3 and k >= 100) else 0.0)
    rows = {r["channel"]: r for r in S.summarize_member(S._load(p), t)}
    pr = rows["PRECT"]
    assert pr["ever_neg"] == 1 and pr["first_neg_lead"] == 100
    assert pr["max_frac_pct"] == pytest.approx(2.0)
    assert pr["mean_frac_pct"] == pytest.approx(2.0 * (N - 99) / N)
    assert pr["min_value"] == -2.0 and pr["truth_ever_neg"] == 0
    assert rows["RELHUM_l00"]["ever_neg"] == 0 and rows["RELHUM_l00"]["truth_ever_neg"] == 1


def test_truncated_member_uses_only_its_leads(pack, tmp_path):
    t = S.build_truth(pack, START, N, CH, stride=4)
    p = _member(tmp_path, t, label="tr", steps=300, trunc=301, frac_fn=lambda k, c: 0.0)
    rows = S.summarize_member(S._load(p), t)
    assert rows[0]["steps"] == 300 and rows[0]["truncated_at_step"] == 301
    assert rows[0]["truth_ever_neg"] == 0          # truth's 2045 negatives start at lead 368


def test_misaligned_member_is_refused(pack, tmp_path):
    t = S.build_truth(pack, START, N, CH, stride=4)
    p = _member(tmp_path, t, label="bad", steps=N, shift=1, frac_fn=lambda k, c: 0.0)
    with pytest.raises(S.NegativityError, match="MISALIGNED"):
        S.summarize_member(S._load(p), t)


def test_group_table_shows_worst_relhum_level(pack, tmp_path):
    t = S.build_truth(pack, START, N, CH, stride=4)
    p = _member(tmp_path, t, label="g", steps=N,
                frac_fn=lambda k, c: {0: 0.001, 1: 0.05}.get(c, 0.0))
    md = S.group_table(S.summarize_member(S._load(p), t))
    assert "RELHUM_l01: 5 /" in md


def test_cli_end_to_end(pack, tmp_path, capsys):
    tnpz = tmp_path / "t.npz"
    assert S.main(["truth", "--pack", str(pack), "--start", "2044", "1092", "--n-leads", str(N),
                   "--channels", *CH, "--out", str(tnpz)]) == 0
    assert "NEGATIVITY_TRUTH_OK leads=365" in capsys.readouterr().out
    t = S._load(tnpz)
    ps = [_member(tmp_path, t, label=f"m{i}", steps=N, frac_fn=lambda k, c: 0.0) for i in range(2)]
    assert S.main(["summarize", "--truth", str(tnpz), "--csv", str(tmp_path / "n.csv"),
                   "--md", str(tmp_path / "n.md"), *map(str, ps)]) == 0
    assert "NEGATIVITY_SUMMARY_OK n=2" in capsys.readouterr().out
    assert len(list(csv.DictReader(open(tmp_path / "n.csv")))) == 2 * len(CH)
