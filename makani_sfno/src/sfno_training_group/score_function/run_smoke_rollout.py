"""Phase E.1 inference smoke entry script.

Loads the GroupEmulator wrapper, reads an init NC + boundary trajectory from
the converted h5, runs `rollout(steps=K)`, and saves a NetCDF.

This sidesteps long_inference.py's three blockers for short-horizon smokes:
- WORLD_SIZE/--debug requirement (we don't use distributed init).
- --init_datetime default (we accept the IC datetime explicitly).
- year-boundary-only save semantics (we save whatever K we ran).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cftime
import numpy as np
import torch
import xarray as xr

from sfno_training_group.score_function.group_emulator import GroupEmulator
from sfno_training_group.score_function._boundary_loader import load_varying_boundary_trajectory

logger = logging.getLogger("run_smoke_rollout")


def _load_ic(init_nc: Path, init_dt: cftime.datetime) -> tuple[torch.Tensor, torch.Tensor]:
    ds = xr.open_dataset(init_nc, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
    time_idx = list(ds["time"].values).index(init_dt)
    # surface: (n_surf=2, H, W)
    pl = ds["pl"].values[time_idx]
    tas = ds["tas"].values[time_idx]
    surface = np.stack([pl, tas], axis=0).astype(np.float32)
    # upper-air: (5, 10, H, W)
    ua_blocks: list[np.ndarray] = []
    for var in ("ta", "ua", "va", "hus", "zg"):
        ua_blocks.append(ds[var].values[time_idx].astype(np.float32))
    upper_air = np.stack(ua_blocks, axis=0)
    ds.close()
    return torch.tensor(surface), torch.tensor(upper_air)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--yaml", required=True, type=Path)
    parser.add_argument("--config", default="SFNO", type=str)
    parser.add_argument("--init-nc", required=True, type=Path)
    parser.add_argument("--boundary-data-dir", required=True, type=Path)
    parser.add_argument("--init-dt", required=True, type=str,
                        help='IC datetime "YYYY-MM-DD HH:MM:SS".')
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--no-prefer-ema", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    init_dt = cftime.datetime.strptime(
        args.init_dt, "%Y-%m-%d %H:%M:%S",
        calendar="proleptic_gregorian", has_year_zero=True,
    )

    em = GroupEmulator(
        ckpt_path=args.ckpt, yaml_path=args.yaml, config_name=args.config,
        device=args.device, prefer_ema=not args.no_prefer_ema,
    )
    logger.info("Loaded SFNO_v2 (in_chans=%d, out_chans=%d) using state %s",
                em.shim.in_chans, em.shim.out_chans, em.loaded_state_kind)

    surface, upper_air = _load_ic(args.init_nc, init_dt)
    bdry = load_varying_boundary_trajectory(
        args.boundary_data_dir, init_dt, args.steps,
        varying_boundary_variables=list(em.params.varying_boundary_variables),
    )
    logger.info("IC shapes: surface=%s, upper_air=%s; bdry=%s",
                tuple(surface.shape), tuple(upper_air.shape), bdry.shape)

    s_traj, u_traj, d_traj = em.rollout(
        init_surface=surface, init_upper_air=upper_air,
        boundary_trajectory=torch.tensor(bdry, dtype=torch.float32),
        steps=args.steps,
    )
    logger.info("rollout done: surface=%s upper=%s diag=%s",
                tuple(s_traj.shape), tuple(u_traj.shape), tuple(d_traj.shape))

    em.save_rollout_netcdf(s_traj, u_traj, d_traj, init_dt, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
