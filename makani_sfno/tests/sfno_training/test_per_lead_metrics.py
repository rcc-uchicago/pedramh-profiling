"""Per-lead validation metrics must actually be computed on our channel names.

The bug this pins: makani builds ``MetricsHandler`` with ERA5 default variable
names (``u10m, t2m, sp, sst, u500, z500, q500, q50``) at
``deterministic_trainer.py:168-176``, intersects them with
``params.channel_names`` (``metric.py:305-311``), and constructs a metric handle
only if the survivors are non-empty (``if self.l1_var_names:`` at ``:361``).
Against the 101-channel E3SM ALLDATA contract (``PS, TREFHT, U10, RHREFHT,
PSL, TMQ, T_l00…``) the intersection is EMPTY, so zero handles are built,
``finalize()`` iterates nothing, and **no per-lead metric is ever computed** —
with no error, no warning and an exit code of 0.

Measured consequence (jobs 7598662/3/4): the same checkpoint scored at
``valid_autoreg_steps`` 3 / 10 / 20 returned byte-identical
``0.012838906608521938`` while validation time scaled 17.3 → 41.9 → 74.4 s.

Asserts:
  (a) ``PlasimTrainer`` builds a non-empty handle set covering EVERY channel
      the dataset emits (L1, RMSE, ACC).
  (b) The silent-failure semantics are real and unchanged upstream: names that
      are absent from ``channel_names`` yield zero handles and raise nothing.
      This is the regression guard — if it ever starts raising, the workaround
      in ``_rebuild_metrics_for_dataset_channels`` can be revisited.
  (c) ``validate_one_epoch`` writes the full (lead time × channel) curve to
      HDF5, correctly shaped and with its dimension scales attached. Only two
      slices of that curve reach ``valid_logs["metrics"]``
      (``metric.py:694-704``), so the file is the artifact that carries the
      lead-time ladder.
  (d) An unknown name in ``metric_var_names`` is rejected loudly rather than
      silently dropped — silent dropping is the whole defect.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
makani = pytest.importorskip("makani")
h5py = pytest.importorskip("h5py")

import numpy as np  # noqa: E402

from makani.utils.dataloaders.data_helpers import get_climatology  # noqa: E402
from makani.utils.metric import MetricsHandler  # noqa: E402

from sfno_training.trainer import PlasimTrainer  # noqa: E402

from test_trainer_ci import (  # noqa: E402  reuse helpers
    _load_yparams,
    _override_for_smoke,
    _populate_runtime_params,
)

# The exact defaults makani ships. Kept literal so an upstream change to them
# shows up here as a diff rather than as silently different coverage.
MAKANI_ERA5_DEFAULT_VAR_NAMES = ["u10m", "t2m", "sp", "sst", "u500", "z500", "q500", "q50"]


def _build_trainer(packaged_dataset: Path, tmp_path: Path, *, valid_autoreg_steps: int = 2):
    params = _load_yparams(packaged_dataset)
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir(exist_ok=True)
    (exp_dir / "training_checkpoints").mkdir(exist_ok=True)

    _populate_runtime_params(params, exp_dir)
    _override_for_smoke(params, n_future=0)
    params["valid_autoreg_steps"] = valid_autoreg_steps

    return PlasimTrainer(params, world_rank=0, device="cpu"), params, exp_dir


def test_handles_cover_every_dataset_channel(packaged_dataset: Path, tmp_path: Path):
    """(a) Non-empty handles, and every channel measured — not a subset."""
    pt, params, _ = _build_trainer(packaged_dataset, tmp_path)

    handles = pt.metrics.metric_handles
    assert handles, (
        "MetricsHandler built ZERO metric handles — this is the silent failure "
        "the rebuild exists to prevent (no per-lead metric would be computed, "
        "and nothing would say so)."
    )

    names = {h.metric_name for h in handles}
    assert {"L1", "RMSE", "ACC"} <= names, f"expected L1/RMSE/ACC handles, got {sorted(names)}"

    channel_names = list(params.channel_names)
    for h in handles:
        assert list(h.metric_channels) == channel_names, (
            f"{h.metric_name} measures {len(h.metric_channels)} channels, but the dataset "
            f"emits {len(channel_names)}. All channels is deliberate: choosing a headline "
            "subset is a science call, not an instrumentation one."
        )
        # channel_mask must stay in range of the normalization/climatology
        # arrays, which are sized to N_out_channels.
        assert max(h.channel_mask) < int(params.N_out_channels)

    assert pt.metrics.num_rollout_steps == params["valid_autoreg_steps"] + 1


def test_absent_names_build_zero_handles_and_raise_nothing(
    packaged_dataset: Path, tmp_path: Path
):
    """(b) Regression guard on the upstream silent-failure semantics."""
    pt, params, _ = _build_trainer(packaged_dataset, tmp_path)
    clim = torch.from_numpy(get_climatology(params)).to(torch.float32)

    absent = MetricsHandler(
        params=params,
        climatology=clim,
        num_rollout_steps=params["valid_autoreg_steps"] + 1,
        device=torch.device("cpu"),
        l1_var_names=["__absent_channel__"],
        rmse_var_names=["__absent_channel__"],
        acc_var_names=["__absent_channel__"],
        crps_var_names=[],
        spread_var_names=[],
        ssr_var_names=[],
    )
    assert absent.metric_handles == [], (
        "upstream no longer silently drops unknown variable names — re-read "
        "metric.py:305-311/:361 before keeping the rebuild workaround"
    )

    # And the same thing via makani's own shipped defaults, whenever those do
    # not intersect the dataset (which is the real-world E3SM case).
    survivors = [c for c in MAKANI_ERA5_DEFAULT_VAR_NAMES if c in list(params.channel_names)]
    if not survivors:
        stock = MetricsHandler(
            params=params,
            climatology=clim,
            num_rollout_steps=params["valid_autoreg_steps"] + 1,
            device=torch.device("cpu"),
            crps_var_names=[],
            spread_var_names=[],
            ssr_var_names=[],
        )
        assert stock.metric_handles == [], (
            "expected makani's ERA5 defaults to produce zero handles on this dataset"
        )


def test_validate_writes_full_lead_time_curve(packaged_dataset: Path, tmp_path: Path):
    """(c) The (lead × channel) curve reaches disk, shaped and labelled."""
    valid_autoreg_steps = 2
    pt, params, exp_dir = _build_trainer(
        packaged_dataset, tmp_path, valid_autoreg_steps=valid_autoreg_steps
    )
    n_leads = valid_autoreg_steps + 1
    n_channels = len(list(params.channel_names))

    pt.validate_one_epoch(epoch=0)

    path = Path(pt._per_lead_metrics_path)
    assert path.is_file(), "per-lead metric file was not written"
    assert path == exp_dir / "scores" / "metrics_ep0000.h5"

    with h5py.File(path, "r") as f:
        assert {"L1", "RMSE", "ACC"} <= set(f.keys()), f"groups: {sorted(f.keys())}"
        for name in ("L1", "RMSE", "ACC"):
            data = f[name]["metric_data"][:]
            assert data.shape == (n_leads, n_channels), (
                f"{name}: expected (leads={n_leads}, channels={n_channels}), got {data.shape}"
            )
            assert np.isfinite(data).all(), f"{name} curve holds non-finite values"

            # Lead axis is labelled in hours: dt * dhours * [1..n_leads].
            lead = f[name]["lead_time"][:]
            expected = pt.metrics.dtxdh * np.arange(1, n_leads + 1)
            assert np.array_equal(lead, expected), f"{name}: lead_time {lead} != {expected}"

            chan = [c.decode() if isinstance(c, bytes) else c for c in f[name]["channel"][:]]
            assert chan == list(params.channel_names)

    # The scalars promoted into the log are only two slices of that curve, so
    # the file is not redundant with them.
    assert n_leads > 2, "pick valid_autoreg_steps >= 2 or this assertion is vacuous"


def test_unknown_metric_var_name_is_rejected(packaged_dataset: Path, tmp_path: Path):
    """(d) Loud rejection, because silent dropping is the defect."""
    params = _load_yparams(packaged_dataset)
    exp_dir = tmp_path / "exp_reject"
    exp_dir.mkdir()
    (exp_dir / "training_checkpoints").mkdir()

    _populate_runtime_params(params, exp_dir)
    _override_for_smoke(params, n_future=0)
    params["metric_var_names"] = [list(params.channel_names)[0], "__not_a_channel__"]

    with pytest.raises(AssertionError, match="__not_a_channel__"):
        PlasimTrainer(params, world_rank=0, device="cpu")
