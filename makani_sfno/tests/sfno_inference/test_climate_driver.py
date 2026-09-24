"""Tests for src/sfno_inference/climate_driver.py (handoff §5, tests 1–8).

Everything here runs the **real** code paths the traps live in:

  - a real ``PlasimForcingDataset`` on tiny ``YYYY.h5`` files (1460 frames/year,
    5 state + 1 diagnostic + 3 forcing channels, 4×8 grid — deliberately not 101);
  - the real ``PlasimPreprocessor`` / stock ``Preprocessor2D`` forcing-cache
    methods (instance built without ``__init__``, which needs comm + an SHT grid);
  - the real ``rollout_one_ic`` as the block reference.

Each trap test seeds the fault and asserts it is caught (the "shown red" half),
next to a test showing the unseeded driver is clean. Needs torch + makani + h5py
+ netCDF4: run on a compute node (``polaris_climate_equiv.pbs``), never the
login node.
"""
from __future__ import annotations

import bisect
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("makani")
h5py = pytest.importorskip("h5py")
netCDF4 = pytest.importorskip("netCDF4")

from sfno_inference import climate_driver as cd  # noqa: E402
from sfno_inference import rollout_driver as rd  # noqa: E402

FPY = cd.FRAMES_PER_YEAR
CS, CD_, CF, H, W = 5, 1, 3, 4, 8          # state, diagnostic, forcing, lat, lon
C_OUT = CS + CD_
EXPECTED_MONTH_COUNTS = [124, 112, 124, 120, 124, 120, 124, 124, 120, 124, 120, 124]


# ---------------------------------------------------------------------------
# fixture: a tiny pack of year files
# ---------------------------------------------------------------------------

def _write_year(path, year, *, cs, cf, ts0, seed):
    rng = np.random.default_rng(seed)
    T = FPY
    with h5py.File(path, "w") as f:
        state = f.create_dataset("fields_state", data=rng.standard_normal(
            (T, cs, H, W)).astype(np.float32) * 3 + 10)
        diag = f.create_dataset("fields_diagnostic", data=rng.standard_normal(
            (T, 1, H, W)).astype(np.float32))
        forcing = rng.standard_normal((T, cf, H, W)).astype(np.float32)
        forcing[:, 0] = year                                   # encodes the year …
        forcing[:, 1] = np.arange(T, dtype=np.float32)[:, None, None]   # … and the frame
        forc = f.create_dataset("forcing", data=forcing)
        f["timestamp"] = ts0 + np.arange(T, dtype=np.int64) * 21600
        f["timestamp"].make_scale("timestamp")
        f["lat"] = 90.0 - (np.arange(H) + 0.5) * 180.0 / H
        f["lat"].make_scale("lat")
        f["lon"] = np.arange(W) * 360.0 / W
        f["lon"].make_scale("lon")
        for ds in (state, diag, forc):
            ds.dims[0].attach_scale(f["timestamp"])
        state.dims[2].attach_scale(f["lat"])
        state.dims[3].attach_scale(f["lon"])
        f["time_plasim"] = np.arange(T, dtype=np.float64) / 4.0
        f["channel_state"] = np.array([f"s{i}" for i in range(cs)], dtype="S")
        f["channel_diagnostic"] = np.array(["diag"], dtype="S")


def _make_pack(root, years, *, cs=CS, cf=CF, continuous=True):
    splits = {}
    for i, (y, split) in enumerate(years.items()):
        d = root / split
        d.mkdir(parents=True, exist_ok=True)
        ts0 = (i * FPY if continuous else 0) * 21600
        _write_year(d / f"{y}.h5", y, cs=cs, cf=cf, ts0=ts0, seed=y)
        splits[y] = d / f"{y}.h5"
    rng = np.random.default_rng(7)
    c_out = cs + 1
    stats = root / "stats"
    stats.mkdir()
    np.save(stats / "global_means.npy", (10 + rng.standard_normal((1, c_out, 1, 1))).astype(np.float32))
    np.save(stats / "global_stds.npy", (2 + rng.random((1, c_out, 1, 1))).astype(np.float32))
    np.save(stats / "forcing_means.npy", np.full((1, cf, 1, 1), 0.5, np.float32))
    np.save(stats / "forcing_stds.npy", np.full((1, cf, 1, 1), 2.0, np.float32))
    np.save(stats / "forcing_id_means.npy", np.zeros((1, cf, 1, 1), np.float32))
    np.save(stats / "forcing_id_stds.npy", np.ones((1, cf, 1, 1), np.float32))
    np.save(stats / "time_means.npy", (10 + rng.standard_normal((1, c_out, H, W))).astype(np.float32))
    return root


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    root = tmp_path_factory.mktemp("pack")
    return _make_pack(root, {2043: "train", 2044: "train", 2045: "valid"})


def _year_dir(pack, years, dest):
    return cd.build_year_dir(cd.locate_year_files(pack, years), dest)


def _dataset(location, pack, *, n_future=0, identity_forcing=False, cs=CS, cf=CF):
    from sfno_training.data import PlasimForcingDataset
    fm = "forcing_id_means" if identity_forcing else "forcing_means"
    fs = "forcing_id_stds" if identity_forcing else "forcing_stds"
    return PlasimForcingDataset(
        location=str(location), dt=1,
        in_channels=list(range(cs)), out_channels=list(range(cs + 1)),
        n_forcing_channels=cf, n_history=0, n_future=n_future,
        relative_timestamp=True, data_grid_type="equiangular", model_grid_type="equiangular",
        bias=np.load(pack / "stats/global_means.npy"), scale=np.load(pack / "stats/global_stds.npy"),
        forcing_bias=np.load(pack / f"stats/{fm}.npy"), forcing_scale=np.load(pack / f"stats/{fs}.npy"),
        enable_logging=False,
    )


def _eval_params(pack, K=1, cs=CS, cf=CF):
    return SimpleNamespace(
        valid_autoreg_steps=K - 1, n_future=K - 1, n_history=0,
        N_in_channels=cs + cf, N_out_channels=cs + 1, n_state_channels=cs,
        n_diagnostic_channels=1, n_forcing_channels=cf,
        amp_enabled=False, amp_dtype=torch.float32,
        global_means_path=str(pack / "stats/global_means.npy"),
        global_stds_path=str(pack / "stats/global_stds.npy"),
    )


# ---------------------------------------------------------------------------
# model stand-ins: real preprocessor, deterministic nonlinear model
# ---------------------------------------------------------------------------

def _real_preprocessor(n_state, n_diag):
    """A real PlasimPreprocessor with the eval forcing cache, built without
    ``__init__`` (which needs makani comm + an SHT grid). Every method the driver
    and rollout_one_ic call is the real one."""
    from sfno_training.models.preprocessor import PlasimPreprocessor
    pp = PlasimPreprocessor.__new__(PlasimPreprocessor)
    torch.nn.Module.__init__(pp)
    pp.n_history = 0
    pp.unpredicted_inp_train = pp.unpredicted_tar_train = None
    pp.unpredicted_inp_eval = pp.unpredicted_tar_eval = None
    pp.n_state_channels = n_state
    pp.n_full_out_channels = n_state + n_diag
    pp.do_add_static_features = False
    pp.history_normalization_mode = "none"
    return pp.eval()


class MixWrapper(torch.nn.Module):
    """Deterministic nonlinear map (state ‖ forcing) → (state ‖ diag).

    Forcing enters nonlinearly, so a wrong forcing frame changes every later
    prediction. Records the forcing it saw on each call.
    """

    def __init__(self, cs=CS, cf=CF, *, nan_at=None, seed=0):
        super().__init__()
        self.preprocessor = _real_preprocessor(cs, 1)
        g = torch.Generator().manual_seed(seed)
        self.cs, self.cf = cs, cf
        self.register_buffer("M", torch.randn(cs + 1, cs + cf, generator=g) / (cs + cf) ** 0.5)
        self.nan_at = nan_at
        self.calls = 0
        self.seen_forcing = []
        self.outputs = []

    def forward(self, inp):
        inpa = self.preprocessor.append_unpredicted_features(inp)
        assert inpa.shape[1] == self.cs + self.cf, f"model got {inpa.shape[1]} channels"
        self.calls += 1
        self.seen_forcing.append(inpa[0, self.cs:].clone())
        # sin() on the forcing: its channels encode year (~2044) and frame (~0..1459),
        # which saturate tanh directly -- then a one-frame forcing shift changes no
        # output bit and the §5.2/§5.3 seeded faults pass unseen (job 7649561).
        feat = torch.cat([inpa[:, : self.cs], torch.sin(inpa[:, self.cs:])], 1)
        mix = torch.tanh(torch.einsum("oc,bchw->bohw", self.M, feat))
        out = torch.cat([0.8 * inp[:, : self.cs] + 0.3 * mix[:, : self.cs], mix[:, self.cs:]], 1)
        if self.nan_at == self.calls:
            out = out.clone()
            out[0, 0, 0, 0] = float("nan")
        self.outputs.append(out.detach().clone())
        return out


class RampWrapper(MixWrapper):
    """Emits the training time-mean exactly, plus ``a·k`` on one channel."""

    def __init__(self, tm_z, channel, a):
        super().__init__()
        self.tm = torch.as_tensor(tm_z, dtype=torch.float32)[None]
        self.channel, self.a = channel, a

    def forward(self, inp):
        self.preprocessor.append_unpredicted_features(inp)
        self.calls += 1
        out = self.tm.clone()
        out[0, self.channel] += self.a * self.calls
        return out


def _stream(wrapper, dataset, ep, ic, n, chunk):
    preds = []
    out_bias, out_scale = rd._load_run_norm_stats(ep, "cpu")

    def on_step(k, pred):
        preds.append(pred * out_scale + out_bias)
        return True

    cd.stream_rollout(wrapper=wrapper, dataset=dataset, ic_global_idx=ic, n_steps=n,
                      chunk_len=chunk, eval_params=ep, device="cpu", on_step=on_step)
    return torch.cat(preds, 0)


def _block(pack, location, ic, K):
    ep = _eval_params(pack, K=K)
    res = rd.rollout_one_ic(wrapper=MixWrapper(), dataset=_dataset(location, pack, n_future=K - 1),
                            ic_global_idx=ic, eval_params=ep, device="cpu")
    return res.prediction


def _mismatch_steps(a, b):
    return [k + 1 for k in range(a.shape[0]) if not torch.equal(a[k], b[k])]


@pytest.fixture
def yd(pack, tmp_path):
    return _year_dir(pack, [2043, 2044, 2045], tmp_path / "years")


# ---------------------------------------------------------------------------
# §5.1  streaming == block
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ic_frame", [100, 1455])            # in-file, and across 2044→2045
@pytest.mark.parametrize("chunk", [1, 2, 4])
def test_stream_equals_block_bitwise(pack, yd, ic_frame, chunk):
    K = 10
    ic = FPY + ic_frame                                       # 2044 is file 1
    ref = _block(pack, yd, ic, K)
    got = _stream(MixWrapper(), _dataset(yd, pack), _eval_params(pack), ic, K, chunk)
    assert got.shape == ref.shape == (K, C_OUT, H, W)
    assert _mismatch_steps(got, ref) == []


def test_chunk_invariance_1_7_40(pack, yd):
    ic = FPY + 1440
    runs = [_stream(MixWrapper(), _dataset(yd, pack), _eval_params(pack), ic, 50, c)
            for c in (1, 7, 40)]
    assert _mismatch_steps(runs[0], runs[1]) == [] and _mismatch_steps(runs[0], runs[2]) == []


# ---------------------------------------------------------------------------
# §5.2  forcing index off-by-one (trap §2a.1)
# ---------------------------------------------------------------------------

def test_seeded_forcing_offbyone_trips_the_guard(pack, yd, monkeypatch):
    monkeypatch.setattr(cd, "_forcing_step_index", lambda i: i + 1)
    with pytest.raises(cd.ForcingIndexError, match="FORCING_INDEX_OUT_OF_RANGE"):
        _stream(MixWrapper(), _dataset(yd, pack), _eval_params(pack), FPY + 100, 8, 4)


def test_seeded_forcing_offbyone_breaks_equivalence_without_the_guard(pack, yd, monkeypatch):
    """With the guard also removed, stock append_history silently keeps stale
    forcing; the streaming-vs-block comparison must see it."""
    ic, K = FPY + 100, 8
    ref = _block(pack, yd, ic, K)
    monkeypatch.setattr(cd, "_forcing_step_index", lambda i: i + 1)
    monkeypatch.setattr(cd, "_check_local_step", lambda local, pp: None)
    got = _stream(MixWrapper(), _dataset(yd, pack), _eval_params(pack), ic, K, 4)
    assert _mismatch_steps(got, ref) != []


# ---------------------------------------------------------------------------
# §5.3  chunk-boundary re-cache (trap §2a.2)
# ---------------------------------------------------------------------------

def test_seeded_xz_none_at_boundary_is_refused(pack, yd, monkeypatch):
    monkeypatch.setattr(cd, "_boundary_input_forcing", lambda pp: None)
    with pytest.raises(cd.ForcingCacheError, match="FORCING_CACHE_WIPED"):
        _stream(MixWrapper(), _dataset(yd, pack), _eval_params(pack), FPY + 100, 6, 2)


def test_seeded_stale_xz_at_boundary_breaks_equivalence(pack, yd, monkeypatch):
    ic, K = FPY + 100, 6
    ref = _block(pack, yd, ic, K)
    ic_forcing = _dataset(yd, pack)[ic][2].unsqueeze(0)          # the IC's forcing, stale
    monkeypatch.setattr(cd, "_boundary_input_forcing", lambda pp: ic_forcing.clone())
    got = _stream(MixWrapper(), _dataset(yd, pack), _eval_params(pack), ic, K, 2)
    bad = _mismatch_steps(got, ref)
    assert bad and bad[0] == 3                                 # first step of chunk 2


# ---------------------------------------------------------------------------
# §5.4  noleap month binning
# ---------------------------------------------------------------------------

def test_calendar_arithmetic():
    assert cd.month_edges() == [0, 124, 236, 360, 480, 604, 724, 848, 972, 1092, 1216, 1336, 1460]
    assert cd.month_edges()[9] == 1092                          # Oct 1 00:00
    assert [cd.month_of_frame(f) for f in (0, 235, 236, 1091, 1092, 1459)] == [1, 2, 3, 9, 10, 12]
    assert cd.valid_time(2043, 1459, 1) == (2044, 0)
    assert cd.step_of((2044, 1092), (2045, 0)) == 368
    assert cd.valid_time(2044, 1092, 7667) == (2049, 1459)
    assert cd.years_needed(2044, 1092, 7667) == list(range(2044, 2050))
    counts = np.bincount([cd.month_of_frame(f) for f in range(FPY)], minlength=13)[1:]
    assert counts.tolist() == EXPECTED_MONTH_COUNTS


def _run(pack, yd, tmp_path, *, start, n, score_start, wrapper=None, chunk=7,
         identity_forcing=False, time_means=True, name="m.nc", **kw):
    wrapper = wrapper or MixWrapper()
    ds = _dataset(yd, pack, identity_forcing=identity_forcing)
    names = [f"s{i}" for i in range(CS)] + ["diag"]
    res = cd.run_member(
        wrapper=wrapper, dataset=ds, eval_params=_eval_params(pack), device="cpu",
        start=start, n_steps=n, score_start=score_start, chunk_len=chunk,
        out_path=tmp_path / name, channel_names=names, lat=ds.lat_lon[0], lon=ds.lat_lon[1],
        time_means_path=str(pack / "stats/time_means.npy") if time_means else None,
        provenance={"member_id": "test"}, **kw)
    return res, wrapper


def _read(path):
    with netCDF4.Dataset(path) as f:
        out = {k: np.array(f[k][:]) for k in f.variables if k not in ("channel", "metric", "snapshot_kind")}
        out["snapshot_kind"] = list(f["snapshot_kind"][:])
        out["attrs"] = {k: f.getncattr(k) for k in f.ncattrs()}
    return out


def test_full_year_month_counts_through_driver(pack, yd, tmp_path):
    res, _ = _run(pack, yd, tmp_path, start=(2043, 1459), n=FPY, score_start=(2044, 0), chunk=40)
    nc = _read(res.out_path)
    assert nc["month_count"].tolist() == EXPECTED_MONTH_COUNTS
    assert nc["month_year"].tolist() == [2044] * 12
    assert nc["month_index"].tolist() == list(range(1, 13))


def test_dec31_to_jan1_binning(pack, yd, tmp_path):
    res, _ = _run(pack, yd, tmp_path, start=(2043, 1450), n=20, score_start=(2043, 1451))
    nc = _read(res.out_path)
    got = list(zip(nc["month_year"].tolist(), nc["month_index"].tolist(), nc["month_count"].tolist()))
    assert got == [(2043, 12, 9), (2044, 1, 11)]


def test_seeded_leap_calendar_breaks_month_counts(pack, yd, tmp_path, monkeypatch):
    """A Feb-29 (Gregorian-leap) calendar must not reproduce the noleap bins."""
    leap = [0]
    for d in (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31):
        leap.append(leap[-1] + 4 * d)
    monkeypatch.setattr(cd, "month_of_frame",
                        lambda f, fpy=FPY: min(bisect.bisect_right(leap, f), 12))
    res, _ = _run(pack, yd, tmp_path, start=(2043, 1459), n=FPY, score_start=(2044, 0), chunk=40)
    assert _read(res.out_path)["month_count"].tolist() != EXPECTED_MONTH_COUNTS


# ---------------------------------------------------------------------------
# §5.5  accumulators vs a numpy float64 reference
# ---------------------------------------------------------------------------

def _numpy_reference(preds_z, mean, std, tm_z, w, score_step, valid_frames):
    P = preds_z.astype(np.float64)                             # (T, C, H, W)
    area = lambda x: np.einsum("h,...h->...", w, x.mean(axis=-1))  # noqa: E731
    gm = area(P)
    ref = {
        "global_mean": gm * std + mean,
        "std_sigma": np.sqrt(area((P - gm[..., None, None]) ** 2)),
        "anom_rms_sigma": np.sqrt(area((P - tm_z) ** 2)),
    }
    sc = P[score_step - 1:]
    ref["time_mean"] = sc.mean(axis=0) * std[:, None, None] + mean[:, None, None]
    months = np.array([cd.month_of_frame(f) for f in valid_frames[score_step - 1:]])
    ref["monthly_mean"] = np.stack([sc[months == m].mean(axis=0) * std[:, None, None]
                                    + mean[:, None, None] for m in dict.fromkeys(months)])
    return ref


def test_accumulators_match_numpy_float64(pack, yd, tmp_path):
    from sfno_ensemble.scores import equiangular_weights
    start, n, score = (2044, 100), 60, (2044, 110)
    res, wr = _run(pack, yd, tmp_path, start=start, n=n, score_start=score)
    nc = _read(res.out_path)
    mean = np.load(pack / "stats/global_means.npy").astype(np.float64).reshape(-1)
    std = np.load(pack / "stats/global_stds.npy").astype(np.float64).reshape(-1)
    tm_z = cd.load_time_means_z(pack / "stats/time_means.npy", mean, std, H, W)
    preds = torch.cat(wr.outputs).numpy()
    frames = [cd.valid_time(*start, k)[1] for k in range(1, n + 1)]
    score_step = cd.step_of(start, score)
    ref = _numpy_reference(preds, mean, std, tm_z, equiangular_weights(H), score_step, frames)

    assert nc["attrs"]["score_start_step"] == score_step == 10
    assert nc["month_count"].tolist() == [14, 37]              # Jan 110..123, Feb 124..160
    for key in ("global_mean", "std_sigma", "anom_rms_sigma", "time_mean"):
        np.testing.assert_allclose(nc[key], ref[key], rtol=1e-12, atol=1e-12, err_msg=key)
    np.testing.assert_allclose(nc["monthly_mean"], ref["monthly_mean"], rtol=1e-6, atol=1e-5)

    # Shown red: the same reference with the window one lead early disagrees.
    early = _numpy_reference(preds, mean, std, tm_z, equiangular_weights(H), score_step - 1, frames)
    assert not np.allclose(nc["time_mean"], early["time_mean"], rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# §5.6  stability monitor
# ---------------------------------------------------------------------------

def test_seeded_nan_truncates_and_writes(pack, yd, tmp_path):
    k = 7
    res, _ = _run(pack, yd, tmp_path, start=(2044, 100), n=30, score_start=(2044, 103),
                  wrapper=MixWrapper(nan_at=k))
    assert res.truncated_at_step == k and res.truncated_channel == "s0"
    assert res.out_path.is_file() and not res.out_path.with_name(res.out_path.name + ".partial").exists()
    nc = _read(res.out_path)
    assert nc["attrs"]["truncated_at_step"] == k and nc["attrs"]["status"] == "truncated"
    assert nc["global_mean"].shape[0] == k - 1 and np.isfinite(nc["global_mean"]).all()
    assert nc["snapshot_step"].tolist() == [k - 1, k]
    assert nc["snapshot_kind"] == ["last_finite", "first_nonfinite"]
    assert nc["month_count"].sum() == k - 1 - 3 + 1            # scored leads 3..6


def test_ramp_sets_first_bad_step(pack, yd, tmp_path):
    mean = np.load(pack / "stats/global_means.npy").astype(np.float64).reshape(-1)
    std = np.load(pack / "stats/global_stds.npy").astype(np.float64).reshape(-1)
    tm_z = cd.load_time_means_z(pack / "stats/time_means.npy", mean, std, H, W)
    c, a = 2, 0.28                                             # 3/0.28 = 10.7, 10/0.28 = 35.7
    res, _ = _run(pack, yd, tmp_path, start=(2044, 100), n=40, score_start=(2044, 101),
                  wrapper=RampWrapper(tm_z, c, a))
    fb = res.stability["first_bad_step"]
    anom = cd.STABILITY_METRICS.index("anom_rms_sigma")
    assert fb[anom, 0, c] == 11 and fb[anom, 1, c] == 36
    others = [i for i in range(C_OUT) if i != c]
    assert (fb[anom, :, others] == -1).all()
    assert (res.stability["median_cross_step"] == -1).all()


# ---------------------------------------------------------------------------
# §5.7  cross-file: every step's forcing is the right (year, frame)
# ---------------------------------------------------------------------------

def _forcing_seen_errors(wr, start, n):
    bad = []
    for k in range(1, n + 1):
        year, frame = cd.valid_time(*start, k - 1)            # input time of lead k
        f = wr.seen_forcing[k - 1]
        if not (torch.all(f[0] == year) and torch.all(f[1] == frame)):
            bad.append((k, float(f[0, 0, 0]), float(f[1, 0, 0]), year, frame))
    return bad


def test_cross_file_forcing_year_and_frame(pack, yd, tmp_path):
    start, n = (2044, 1450), 30
    res, wr = _run(pack, yd, tmp_path, start=start, n=n, score_start=(2045, 0),
                   identity_forcing=True, chunk=7)
    assert _forcing_seen_errors(wr, start, n) == []
    (h,) = res.handoffs
    assert (h["step"], h["from"], h["to"], h["to_local_idx"]) == (10, "2044.h5", "2045.h5", 0)
    assert h["forcing_direct_match"] is True and h["verified_at_step"] == 11
    nc = _read(res.out_path)
    assert json.loads(nc["attrs"]["handoffs"])[0]["step"] == 10
    assert nc["valid_year"][9] == 2045 and nc["valid_frame"][9] == 0


def test_seeded_offbyone_is_seen_by_the_cross_file_check(pack, yd, tmp_path, monkeypatch):
    monkeypatch.setattr(cd, "_forcing_step_index", lambda i: i + 1)
    monkeypatch.setattr(cd, "_check_local_step", lambda local, pp: None)
    start, n = (2044, 1450), 30
    _, wr = _run(pack, yd, tmp_path, start=start, n=n, score_start=(2045, 0),
                 identity_forcing=True, chunk=7)
    assert _forcing_seen_errors(wr, start, n) != []


def test_year_dir_refuses_gap_real_files_and_strays(pack, tmp_path):
    with pytest.raises(cd.ClimateDriverError, match="YEAR_GAP"):
        cd.locate_year_files(pack, [2044, 2045, 2046])
    d = _year_dir(pack, [2044, 2045], tmp_path / "y")
    assert sorted(p.name for p in d.iterdir()) == ["2044.h5", "2045.h5"]
    assert all(p.is_symlink() for p in d.iterdir())
    _year_dir(pack, [2044, 2045], tmp_path / "y")               # idempotent
    with pytest.raises(cd.ClimateDriverError, match="YEAR_DIR_STRAY"):
        _year_dir(pack, [2044], tmp_path / "y")
    real = tmp_path / "r"
    real.mkdir()
    (real / "2044.h5").write_bytes(b"")
    with pytest.raises(cd.ClimateDriverError, match="YEAR_DIR_NOT_SYMLINK"):
        _year_dir(pack, [2044], real)


def test_timestamp_axis_mode_is_reported(pack, yd, tmp_path):
    assert cd.timestamp_axis_mode(_dataset(yd, pack))["mode"] == "continuous"
    reset = _make_pack(tmp_path / "reset", {2044: "train", 2045: "valid"}, continuous=False)
    d = _year_dir(reset, [2044, 2045], tmp_path / "ry")
    m = cd.timestamp_axis_mode(_dataset(d, reset))
    assert m["mode"] == "synthesized" and m["first_bad_concat_index"] == FPY - 1


# ---------------------------------------------------------------------------
# §5.8  channel-agnostic
# ---------------------------------------------------------------------------

def test_other_channel_count_runs(tmp_path):
    cs, cf = 3, 2
    p = _make_pack(tmp_path / "p", {2044: "train", 2045: "valid"}, cs=cs, cf=cf)
    d = _year_dir(p, [2044, 2045], tmp_path / "y")
    ds = _dataset(d, p, cs=cs, cf=cf)
    res = cd.run_member(
        wrapper=MixWrapper(cs, cf), dataset=ds, eval_params=_eval_params(p, cs=cs, cf=cf),
        device="cpu", start=(2044, 1455), n_steps=10, score_start=(2044, 1456), chunk_len=3,
        out_path=tmp_path / "o.nc", channel_names=["a", "b", "c", "d"],
        lat=ds.lat_lon[0], lon=ds.lat_lon[1], time_means_path=str(p / "stats/time_means.npy"))
    assert res.n_steps_run == 10 and _read(res.out_path)["time_mean"].shape == (cs + 1, H, W)


def test_mismatched_stats_length_raises(pack, yd, tmp_path):
    bad = tmp_path / "bad_means.npy"
    np.save(bad, np.zeros((1, C_OUT + 1, 1, 1), np.float32))
    ep = _eval_params(pack)
    ep.global_means_path = str(bad)
    ds = _dataset(yd, pack)
    with pytest.raises(cd.ClimateDriverError, match="STATS_CHANNEL_MISMATCH"):
        cd.run_member(wrapper=MixWrapper(), dataset=ds, eval_params=ep, device="cpu",
                      start=(2044, 100), n_steps=4, score_start=(2044, 101), chunk_len=2,
                      out_path=tmp_path / "x.nc", channel_names=[str(i) for i in range(C_OUT)],
                      lat=ds.lat_lon[0], lon=ds.lat_lon[1])


def test_mismatched_time_means_raises(pack):
    bad_tm = pack / "stats" / "time_means_bad.npy"
    np.save(bad_tm, np.zeros((1, C_OUT - 1, H, W), np.float32))
    mean = np.zeros(C_OUT)
    with pytest.raises(cd.ClimateDriverError, match="STATS_CHANNEL_MISMATCH"):
        cd.load_time_means_z(bad_tm, mean, mean + 1, H, W)
    os.remove(bad_tm)
