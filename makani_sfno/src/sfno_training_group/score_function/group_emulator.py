"""Programmatic Python wrapper around a group-trained SFNO_v2 checkpoint.

Phase G of the plan v5. AI-RES uses this for in-process score-function calls;
the Phase E.1 inference smoke also drives this (sidesteps long_inference's
WORLD_SIZE / --init_datetime / year-boundary-only-save constraints).

Public API:
    GroupEmulator(ckpt_path, yaml_path, config_name="SFNO", device="cuda:0",
                  prefer_ema=True)
        .step(surface, upper_air, varying_boundary)
            -> (out_surface, out_upper_air, out_diagnostic)  physical-space
        .rollout(init_surface, init_upper_air, boundary_trajectory, steps)
            -> (surface_traj, upper_air_traj, diagnostic_traj)
        .save_rollout_netcdf(surface_traj, upper_air_traj, diagnostic_traj,
                             init_dt, out_path)
        .loaded_state_kind  ('ema_state' or 'model_state')

Design notes:
- We mirror group's normalization exactly:
  * Upper-air / surface / varying-boundary / diagnostic stats: from NetCDF with
    Z (plev) and Z_2 (sigma) coords as required by `load_mean_std`
    (data_loader_multifiles.py:751-780).
  * Constant boundary (lsm, sg): loaded from `{val_year_start}_0000.h5` and
    *spatially* z-scored per-variable, NOT divided by global stats — mirrors
    `_load_constant_boundary_data` (line 740-749).
- `params.has_diagnostic` is set before SFNO construction (mirrors train.py:3435).
- Checkpoint loader prefers `ema_state` over `model_state` if present, matching
  long_inference.py:370-380. `'module.'` DDP prefixes are stripped.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import cftime
import h5py
import numpy as np
import torch
import xarray as xr

from sfno_training_group.score_function._dataset_shim import DatasetShim

logger = logging.getLogger("group_emulator")

_DEFAULT_GROUP_ROOT = "/work2/09979/awikner/stampede3/PanguWeather/v2.0"


def _ensure_group_on_path() -> str:
    root = os.environ.get("GROUP_PANGU_ROOT", _DEFAULT_GROUP_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _load_yaml_params(yaml_path: str | Path, config_name: str) -> Any:
    _ensure_group_on_path()
    from utils.YParams import YParams
    return YParams(str(yaml_path), config_name)


def _select_levels_from_xr(
    da: xr.DataArray, coord_name: str, desired_levels: list[float],
    delta: float = 1e-4,
) -> np.ndarray:
    """Mirror `load_mean_std` level filtering (data_loader_multifiles.py:756-770)."""
    file_levels = np.asarray(da[coord_name].values, dtype=np.float64)
    out_idx: list[int] = []
    for lev in desired_levels:
        diffs = np.abs(file_levels - float(lev))
        best = int(np.argmin(diffs))
        if diffs[best] > delta:
            raise ValueError(
                f"level {lev} for {da.name!r}: closest file level {file_levels[best]} "
                f"(diff={diffs[best]:.2e}, delta={delta})"
            )
        out_idx.append(best)
    return da.values[np.array(out_idx)]


class GroupEmulator:
    def __init__(
        self,
        ckpt_path: str | Path,
        yaml_path: str | Path,
        config_name: str = "SFNO",
        device: str = "cuda:0",
        *,
        prefer_ema: bool = True,
    ) -> None:
        # 1. Parse YAML.
        self.params = _load_yaml_params(yaml_path, config_name)

        # 2. has_diagnostic per train.py:3435.
        if hasattr(self.params, "diagnostic_variables"):
            self.params["has_diagnostic"] = len(self.params.diagnostic_variables) > 0
        else:
            self.params["has_diagnostic"] = False

        # 3. Build dataset shim (sufficient for SFNO_v2 init).
        self.shim = DatasetShim(
            upper_air_variables=self.params.upper_air_variables,
            surface_variables=self.params.surface_variables,
            diagnostic_variables=self.params.diagnostic_variables,
            varying_boundary_variables=self.params.varying_boundary_variables,
            constant_boundary_variables=self.params.constant_boundary_variables,
            sigma_levels=self.params.sigma_levels,
            levels=self.params.levels,
            use_sigma_levels=bool(self.params.use_sigma_levels),
        )
        self.device = torch.device(device)

        # 4. Load NetCDF stats (mirrors load_mean_std contract).
        mean_path = Path(self.params.data_dir) / self.params.upper_air_mean
        std_path = Path(self.params.data_dir) / self.params.upper_air_std
        mean_ds = xr.open_dataset(mean_path)
        std_ds = xr.open_dataset(std_path)
        self._build_stats_tensors(mean_ds, std_ds)
        mean_ds.close(); std_ds.close()

        # 5. Constant boundary: load + spatial z-score. Mirrors
        #    data_loader_multifiles._load_constant_boundary_data (line 740-749).
        self.constant_boundary, self.land_mask = self._load_constant_boundary()

        # 6. Construct SFNO_v2.
        _ensure_group_on_path()
        from networks.modulus_sfno.sfnonet import SphericalFourierNeuralOperatorNet_v2
        self.model = SphericalFourierNeuralOperatorNet_v2(self.params, self.shim).to(self.device).eval()

        # 7. Load checkpoint with EMA preference.
        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        if prefer_ema and ckpt.get("ema_state") is not None:
            state_dict = ckpt["ema_state"]
            self._loaded_state = "ema_state"
        else:
            state_dict = ckpt["model_state"]
            self._loaded_state = "model_state"
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        # Allow strict=False so sgd_state buffers (etc.) don't block load.
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        logger.info("loaded %s; missing=%d, unexpected=%d", self._loaded_state,
                    len(missing), len(unexpected))
        if missing:
            logger.debug("    missing keys (sample): %s", missing[:5])
        if unexpected:
            logger.debug("    unexpected keys (sample): %s", unexpected[:5])

    # ----- stats -----

    def _build_stats_tensors(self, mean_ds: xr.Dataset, std_ds: xr.Dataset) -> None:
        # Upper-air: per-(var, level). use_sigma_levels=True -> ta/ua/va/hus on Z_2 (sigma);
        # zg on Z (plev).
        ua_means: list[np.ndarray] = []
        ua_stds: list[np.ndarray] = []
        for var in self.params.upper_air_variables:
            if self.params.use_sigma_levels and var not in ("zg", "geopotential_height"):
                coord, desired = "Z_2", list(self.params.sigma_levels)
            else:
                coord, desired = "Z", list(self.params.levels)
            ua_means.append(_select_levels_from_xr(mean_ds[var], coord, desired))
            ua_stds.append(_select_levels_from_xr(std_ds[var], coord, desired))
        # (n_vars, n_levels)
        self.upper_air_mean = torch.tensor(np.stack(ua_means, axis=0), dtype=torch.float32, device=self.device)
        self.upper_air_std = torch.tensor(np.stack(ua_stds, axis=0), dtype=torch.float32, device=self.device)

        # Scalars (0-D in NetCDF).
        def _scalar_stack(ds: xr.Dataset, varlist: list[str]) -> torch.Tensor:
            return torch.tensor(
                np.array([float(ds[v].values) for v in varlist], dtype=np.float32),
                device=self.device,
            )

        self.surface_mean = _scalar_stack(mean_ds, list(self.params.surface_variables))
        self.surface_std = _scalar_stack(std_ds, list(self.params.surface_variables))
        self.varying_boundary_mean = _scalar_stack(mean_ds, list(self.params.varying_boundary_variables))
        self.varying_boundary_std = _scalar_stack(std_ds, list(self.params.varying_boundary_variables))
        self.diagnostic_mean = _scalar_stack(mean_ds, list(self.params.diagnostic_variables))
        self.diagnostic_std = _scalar_stack(std_ds, list(self.params.diagnostic_variables))

    # ----- constant boundary -----

    def _load_constant_boundary(self) -> tuple[torch.Tensor, torch.Tensor]:
        path = Path(self.params.data_dir) / f"{int(self.params.val_year_start)}_0000.h5"
        with h5py.File(path, "r") as f:
            stack = np.stack(
                [f["input"][v][:] for v in self.params.constant_boundary_variables],
                axis=0,
            ).astype(np.float64)
        # Mask-fill (mirrors data_loader_multifiles._fill_mask).
        for i, var in enumerate(self.params.constant_boundary_variables):
            nans = np.isnan(stack[i])
            if nans.any():
                stack[i] = np.where(nans, float(self.params.mask_fill.get(var, 0.0)), stack[i])
        # Land mask = lsm before normalization.
        lsm_idx = list(self.params.constant_boundary_variables).index("lsm")
        land_mask_np = stack[lsm_idx].copy()
        # Spatial z-score per variable (NOT global stats — see _load_constant_boundary_data).
        spatial_mean = stack.mean(axis=(1, 2), keepdims=True)
        spatial_std = stack.std(axis=(1, 2), keepdims=True)
        if (spatial_std < 1e-12).any():
            raise RuntimeError("constant_boundary spatial std too small for some variable.")
        normalized = (stack - spatial_mean) / spatial_std
        return (
            torch.tensor(normalized, dtype=torch.float32, device=self.device),
            torch.tensor(land_mask_np, dtype=torch.float32, device=self.device),
        )

    # ----- forward / inverse normalization -----

    def _surface_fwd(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.surface_mean.view(-1, 1, 1)) / self.surface_std.view(-1, 1, 1)

    def _surface_inv(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (B, surface_vars, lat, lon)
        return x * self.surface_std.view(1, -1, 1, 1) + self.surface_mean.view(1, -1, 1, 1)

    def _upper_fwd(self, x: torch.Tensor) -> torch.Tensor:
        n = len(self.params.upper_air_variables)
        return (x - self.upper_air_mean.view(n, -1, 1, 1)) / self.upper_air_std.view(n, -1, 1, 1)

    def _upper_inv(self, x: torch.Tensor) -> torch.Tensor:
        n = len(self.params.upper_air_variables)
        # x shape (B, n_vars, n_levels, lat, lon)
        return x * self.upper_air_std.view(1, n, -1, 1, 1) + self.upper_air_mean.view(1, n, -1, 1, 1)

    def _vbdry_fwd(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.varying_boundary_mean.view(-1, 1, 1)) / self.varying_boundary_std.view(-1, 1, 1)

    def _diag_inv(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.diagnostic_std.view(1, -1, 1, 1) + self.diagnostic_mean.view(1, -1, 1, 1)

    # ----- public API -----

    @property
    def loaded_state_kind(self) -> str:
        return self._loaded_state

    @torch.inference_mode()
    def step(
        self,
        surface: torch.Tensor,
        upper_air: torch.Tensor,
        varying_boundary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One 6h forward step.

        Inputs (physical-space, batch-less):
            surface:          (n_surface=2, 64, 128)
            upper_air:        (n_upper_air=5, n_levels=10, 64, 128)
            varying_boundary: (n_vbdry=4, 64, 128)
        Returns physical-space, batch-less:
            (out_surface, out_upper_air, out_diagnostic) with shapes
            (2, 64, 128), (5, 10, 64, 128), (1, 64, 128)
        """
        s_n = self._surface_fwd(surface).unsqueeze(0).to(self.device)            # (1, 2, H, W)
        u_n = self._upper_fwd(upper_air).unsqueeze(0).to(self.device)            # (1, 5, 10, H, W)
        v_n = self._vbdry_fwd(varying_boundary).unsqueeze(0).to(self.device)     # (1, 4, H, W)
        c_n = self.constant_boundary.unsqueeze(0)                                # (1, 2, H, W)

        out = self.model(s_n, c_n, v_n, u_n)
        if self.params.has_diagnostic:
            out_s, out_u, out_d, *_ = out
            out_s_phys = self._surface_inv(out_s).squeeze(0)
            out_u_phys = self._upper_inv(out_u).squeeze(0)
            out_d_phys = self._diag_inv(out_d).squeeze(0)
            return out_s_phys, out_u_phys, out_d_phys
        else:
            out_s, out_u, *_ = out
            return self._surface_inv(out_s).squeeze(0), self._upper_inv(out_u).squeeze(0), torch.empty(0, device=self.device)

    @torch.inference_mode()
    def rollout(
        self,
        init_surface: torch.Tensor,
        init_upper_air: torch.Tensor,
        boundary_trajectory: torch.Tensor,  # (steps + 1, 4, H, W) — IC + steps boundary samples
        steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Auto-regressive rollout.

        Returns (surface_traj, upper_air_traj, diagnostic_traj):
            surface_traj:    (steps + 1, 2, H, W)         (IC included at t=0)
            upper_air_traj:  (steps + 1, 5, 10, H, W)     (IC included at t=0)
            diagnostic_traj: (steps,     1, H, W)         (no IC; pr_6h is between steps)
        """
        if boundary_trajectory.shape[0] < steps + 1:
            raise ValueError(
                f"boundary_trajectory has {boundary_trajectory.shape[0]} timesteps; "
                f"need at least steps + 1 = {steps + 1}"
            )
        H, W = init_surface.shape[-2:]
        surface_traj = torch.empty(steps + 1, init_surface.shape[0], H, W,
                                    dtype=torch.float32, device="cpu")
        upper_traj = torch.empty(steps + 1, init_upper_air.shape[0], init_upper_air.shape[1], H, W,
                                  dtype=torch.float32, device="cpu")
        diag_traj = torch.empty(steps, len(self.params.diagnostic_variables), H, W,
                                 dtype=torch.float32, device="cpu")
        surface_traj[0] = init_surface.detach().to("cpu")
        upper_traj[0] = init_upper_air.detach().to("cpu")

        s = init_surface.to(self.device)
        u = init_upper_air.to(self.device)
        for k in range(steps):
            v = boundary_trajectory[k].to(self.device)
            out_s, out_u, out_d = self.step(s, u, v)
            surface_traj[k + 1] = out_s.detach().to("cpu")
            upper_traj[k + 1] = out_u.detach().to("cpu")
            if self.params.has_diagnostic:
                diag_traj[k] = out_d.detach().to("cpu")
            s, u = out_s, out_u
        return surface_traj, upper_traj, diag_traj

    def save_rollout_netcdf(
        self,
        surface_traj: torch.Tensor,        # (T+1, n_surf, H, W)
        upper_air_traj: torch.Tensor,      # (T+1, n_upper, n_levels, H, W)
        diagnostic_traj: torch.Tensor,     # (T, n_diag, H, W)
        init_dt: cftime.datetime,
        out_path: str | Path,
        *,
        extra_attrs: dict[str, Any] | None = None,
    ) -> None:
        """Save rollout as NetCDF compatible with the eval converter."""
        T = surface_traj.shape[0]
        from datetime import timedelta
        times = np.array(
            [init_dt + timedelta(hours=6 * k) for k in range(T)],
            dtype=object,
        )
        diag_times = times[1:]  # diagnostic is between-steps (no IC entry)
        coords = {
            "time": (("time",), times),
            "diag_time": (("diag_time",), diag_times),
            "sigma": (("sigma",), np.array(self.params.sigma_levels, dtype=np.float64)),
            "lev": (("lev",), np.array(self.params.levels, dtype=np.float64)),
            "lat": (("lat",), np.array(self.params.lat, dtype=np.float64)),
            "lon": (("lon",), np.array(self.params.lon, dtype=np.float64)),
        }
        data_vars: dict[str, xr.DataArray] = {}
        # Surface
        for i, v in enumerate(self.params.surface_variables):
            data_vars[v] = xr.DataArray(
                surface_traj[:, i].numpy().astype(np.float32),
                dims=("time", "lat", "lon"),
            )
        # Upper-air per-var on the appropriate level coord.
        for vi, var in enumerate(self.params.upper_air_variables):
            if self.params.use_sigma_levels and var not in ("zg", "geopotential_height"):
                lev_dim = "sigma"
            else:
                lev_dim = "lev"
            data_vars[var] = xr.DataArray(
                upper_air_traj[:, vi].numpy().astype(np.float32),
                dims=("time", lev_dim, "lat", "lon"),
            )
        # Diagnostic (between-steps).
        for di, var in enumerate(self.params.diagnostic_variables):
            data_vars[var] = xr.DataArray(
                diagnostic_traj[:, di].numpy().astype(np.float32),
                dims=("diag_time", "lat", "lon"),
            )
        ds = xr.Dataset(data_vars, coords=coords)
        ds.attrs["loaded_state_kind"] = self._loaded_state
        ds.attrs["init_dt"] = init_dt.strftime("%Y-%m-%d %H:%M:%S")
        ds.attrs["steps"] = int(T - 1)
        ds.attrs["source"] = "sfno_training_group.score_function.GroupEmulator"
        if extra_attrs:
            for k, v in extra_attrs.items():
                ds.attrs[k] = v
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        encoding = {v: {"dtype": "float32", "_FillValue": None} for v in ds.data_vars}
        encoding["time"] = {
            "units": f"hours since {init_dt.strftime('%Y-%m-%d %H:%M:%S')}",
            "calendar": "proleptic_gregorian",
        }
        encoding["diag_time"] = {
            "units": f"hours since {init_dt.strftime('%Y-%m-%d %H:%M:%S')}",
            "calendar": "proleptic_gregorian",
        }
        ds.to_netcdf(out_path, encoding=encoding)
        logger.info("Wrote rollout %s (%d KB)", out_path, out_path.stat().st_size // 1024)
