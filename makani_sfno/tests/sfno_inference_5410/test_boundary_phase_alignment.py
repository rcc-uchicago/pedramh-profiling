"""Regression test for 5410 single-IC boundary forcing phase.

The training/validation autoregression path feeds the model boundary
variables from the current step: the first forecast step consumes the
IC-time boundary field. The 5410 NWP inference path uses the upstream
single_ic BCS loader, so ``nc_bc_offset`` must be 0. An 18-hour offset
produces plausible-looking files but corrupts short-lead RMSE and
long-lead ACC.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
import sys

import cftime
import numpy as np
import pytest


_UPSTREAM_REPO = Path(
    "/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0"
)
_RUN_ROOT = Path(
    "/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results/sfno_eval_5410/20260507_phase1_gate"
)


def _need_live_5410_assets():
    yaml_path = _RUN_ROOT / "inference" / "SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y121.yaml"
    ic_nc = _RUN_ROOT / "inference" / "ic_nc" / "121_0000.nc"
    truth_h5 = Path(
        "/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/h5/sigma_data/121_0000.h5"
    )
    missing = [p for p in (_UPSTREAM_REPO, yaml_path, ic_nc, truth_h5) if not p.exists()]
    if missing:
        pytest.skip(f"live 5410 assets not present: {missing}")
    return yaml_path, ic_nc


def test_single_ic_bcs_first_step_matches_validation_boundary_phase():
    pytest.importorskip("torch")
    pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    yaml_path, ic_nc = _need_live_5410_assets()

    if str(_UPSTREAM_REPO) not in sys.path:
        sys.path.insert(0, str(_UPSTREAM_REPO))

    from sfno_inference_5410.upstream_hydration import (
        hydrate_static_params,
        set_per_ic_params,
        set_per_y_params,
    )
    from utils.data_loader_multifiles import GetDataset  # type: ignore

    Y = 121
    K = 60
    init_dt = cftime.DatetimeProlepticGregorian(Y, 1, 1, 0, has_year_zero=True)
    final_dt = init_dt + dt.timedelta(hours=(K + 1) * 6)

    params = hydrate_static_params(yaml_path, K=K, upstream_repo=_UPSTREAM_REPO)
    params["num_data_workers"] = 0
    set_per_y_params(params, Y=Y)
    set_per_ic_params(
        params,
        init_datetime=init_dt,
        final_datetime=final_dt,
        init_nc_filepaths=[ic_nc],
        save_basename="Y121_s0000",
        output_dir=_RUN_ROOT / "test_boundary_phase_alignment",
    )
    assert params.nc_bc_offset == 0

    # Reference: validation/autoregression path takes its step-0 boundary
    # variables from data_in at the IC time.
    ref_ds = GetDataset(
        params, params.data_dir, Y, Y + 1,
        train=False, num_inferences=1, validate=True,
        single_ic=False, ensemble=False, init_from_nc=False,
    )
    raw_ref = ref_ds._get_data(ref_ds.start_date, out=False)
    _, _, ref_boundary = ref_ds._reshape_and_mask_variables(raw_ref, out=False)
    ref_boundary = ref_ds.boundary_transform(ref_boundary).numpy()

    # Inference BCS loader: item 0 is the boundary forcing consumed by the
    # first model forward pass.
    params["single_ic_offset"] = 0
    bcs_ds = GetDataset(
        params, params.data_dir, Y, Y,
        train=False, single_ic=True,
    )
    _, _, bcs_boundary, _ = bcs_ds[0]

    np.testing.assert_array_equal(bcs_boundary.numpy(), ref_boundary)
