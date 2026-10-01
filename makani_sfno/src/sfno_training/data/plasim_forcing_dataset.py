"""PlasimForcingDataset — MultifilesDataset subclass for the PlaSim → Makani
asymmetric three-dataset HDF5 contract.

    /fields_state      (T, 52, H, W) float32   state, fed back at rollout
    /fields_diagnostic (T, 1,  H, W) float32   pr_6h, loss-only
    /forcing           (T, 6,  H, W) float32   prescribed forcing, never predicted

Returns the 4-tuple ``(inp_state, tar, inp_forcing, tar_forcing)``. The
trainer caches forcing into the preprocessor's ``unpredicted_*`` slots so
that stock ``Preprocessor2D.append_unpredicted_features`` concatenates
forcing back at the model boundary (52 + 6 = 58 input channels).

See docs/plasim_makani_packager_plan.md §"Custom dataloader
PlasimForcingDataset" and docs/sfno_training_implementation_plan.md §1.
"""

from __future__ import annotations

import glob
import logging
import math
import operator
import os
from itertools import accumulate
from typing import Tuple

import h5py
import numpy as np
import torch

# Side-effect import: installs the Python 3.12 timedelta shim before any
# MultifilesDataset reads the int64 /timestamp. Required on Stampede3
# (Python 3.12). Removable once upstream Makani fixes the cast.
from sfno_training import compat  # noqa: F401
from sfno_training.compat import MAKANI_HAS_BACKENDS

from makani.utils.dataloaders.data_loader_multifiles import MultifilesDataset


class _ChunkOnlyBackend:
    """What makani main's ``MultifilesDataset.__init__`` and ``DataShapes.from_loader`` read off
    ``self.backend`` -- the rank's grid chunk -- and nothing else. Reads never go through it."""

    def __init__(self, chunk):
        self.chunk = chunk


class PlasimForcingDataset(MultifilesDataset):
    """Dataset for the asymmetric PlaSim → Makani SFNO contract.

    Built on stock :class:`MultifilesDataset` but overrides
    :meth:`get_sample_at_index` because the stock implementation assumes
    ``in_channels == out_channels`` and a single ``/fields`` HDF5 key.

    Parameters
    ----------
    diagnostic_dataset_path : str, default ``"fields_diagnostic"``
        HDF5 path of the diagnostic dataset.
    forcing_dataset_path : str, default ``"forcing"``
        HDF5 path of the forcing dataset.
    n_forcing_channels : int, default 6
        Number of forcing channels (must equal ``forcing.shape[1]``).
    forcing_bias, forcing_scale : np.ndarray, optional
        Per-channel mean / std with shape ``(1, n_forcing_channels, 1, 1)``.
        Identity normalization (mean 0, std 1) when omitted.
    **kwargs : dict
        Forwarded to :class:`MultifilesDataset`. ``dataset_path`` defaults
        to ``"fields_state"`` (stock default is ``"fields"``).
    """

    def __init__(
        self,
        *,
        diagnostic_dataset_path: str = "fields_diagnostic",
        forcing_dataset_path: str = "forcing",
        n_forcing_channels: int = 6,
        forcing_bias: np.ndarray | None = None,
        forcing_scale: np.ndarray | None = None,
        **kwargs,
    ):
        kwargs.setdefault("dataset_path", "fields_state")
        if MAKANI_HAS_BACKENDS:
            # makani main dropped these from MultifilesDataset (its kwargs swallow them);
            # our own HDF5 reads below still need them.
            if kwargs.get("enable_s3", False):
                raise NotImplementedError("PlasimForcingDataset on makani main: enable_s3 is not supported")
            self.dataset_path = kwargs["dataset_path"]
            self.file_suffix = kwargs.get("file_suffix", "h5")
            self.file_driver = None
            self.file_driver_kwargs = {}
        super().__init__(**kwargs)

        self.diagnostic_dataset_path = diagnostic_dataset_path
        self.forcing_dataset_path = forcing_dataset_path
        self.n_forcing_channels = n_forcing_channels

        self._diag_files = [None] * len(self.files_paths)
        self._forcing_files = [None] * len(self.files_paths)

        if forcing_bias is None:
            forcing_bias = np.zeros((1, n_forcing_channels, 1, 1), dtype=np.float32)
        if forcing_scale is None:
            forcing_scale = np.ones((1, n_forcing_channels, 1, 1), dtype=np.float32)
        self.forcing_bias = forcing_bias.astype(np.float32, copy=False)
        self.forcing_scale = forcing_scale.astype(np.float32, copy=False)

    def _get_stats_h5(self, enable_logging):
        """Read file metadata, tolerating timestamp resets across source splits.

        The packaged data has uniform 6-hour timestamps inside each original
        split. A training subset can intentionally span multiple source splits
        via symlinks, e.g. train years 0012-0100 plus valid years 0101-0111.
        In that case the raw /timestamp values reset at the split boundary,
        while the MOST.xxxx file order is still the intended physical order.
        Stock Makani sorts/checks by raw timestamps and fails there, so this
        PlaSim dataset keeps filename order and synthesizes one continuous
        timestamp axis when it detects a boundary reset.
        """

        self.n_samples_file = []
        raw_timestamps = []

        for file_idx, filename in enumerate(self.files_paths):
            with h5py.File(
                filename,
                "r",
                driver=self.file_driver,
                **self.file_driver_kwargs,
            ) as _f:
                if file_idx == 0:
                    if enable_logging:
                        logging.info("Getting file stats from {}".format(filename))
                    self.img_shape = _f[self.dataset_path].shape[2:4]
                    self.total_channels = _f[self.dataset_path].shape[1]
                    lat = _f[self.dataset_path].dims[2]["lat"][...]
                    lon = _f[self.dataset_path].dims[3]["lon"][...]
                    self.lat_lon = (lat.tolist(), lon.tolist())

                self.n_samples_file.append(_f[self.dataset_path].shape[0])
                raw_timestamps.append(
                    np.asarray(
                        _f[self.dataset_path].dims[0]["timestamp"][...],
                        dtype=np.int64,
                    )
                )

        step_seconds = self._infer_step_seconds(raw_timestamps)
        raw_concat = np.concatenate(raw_timestamps, axis=0)
        raw_diffs = np.diff(raw_concat)
        if raw_diffs.size and np.all(raw_diffs == step_seconds):
            self.timestamps = raw_concat
        else:
            bad = np.flatnonzero(raw_diffs != step_seconds)
            if enable_logging:
                first_bad = int(bad[0]) if bad.size else -1
                logging.warning(
                    "PlaSim timestamps reset or jump across file boundaries "
                    "(first bad concat index %d). Keeping MOST.xxxx file order "
                    "and synthesizing continuous %d-second timestamps.",
                    first_bad,
                    step_seconds,
                )
            start = int(raw_timestamps[0][0])
            self.timestamps = start + (
                np.arange(raw_concat.size, dtype=np.int64) * np.int64(step_seconds)
            )

        self.datestamps = self.date_fn(self.timestamps)
        self.date_ranges = []
        offset = 0
        for count in self.n_samples_file:
            self.date_ranges.append(
                (self.datestamps[offset], self.datestamps[offset + count - 1])
            )
            offset += count

        if step_seconds % 3600 != 0:
            raise RuntimeError(
                f"PlaSim timestamp step {step_seconds} seconds is not an "
                "integer number of hours."
            )
        self.dhours = step_seconds // 3600

        return

    if MAKANI_HAS_BACKENDS:
        # makani main builds a storage backend here, which would (a) re-order files by time and
        # refuse PlaSim's split-spanning timestamp resets, and (b) verify the grid declaration.
        # Keep the pin's file-stats logic instead (c9704308 data_loader_multifiles.py), so file
        # order, timestamps, crop/read geometry and every read stay what they were.
        def _get_files_stats(self, enable_logging):
            from torch_harmonics.distributed import compute_split_shapes
            from makani.utils.dataloaders.backends.base import GridSpec

            if isinstance(self.location, str):
                self.location = [self.location]
            if os.path.isfile(self.location[0]):
                self.files_paths = list(self.location)
            else:
                self.files_paths = []
                for location in self.location:
                    if not os.path.isdir(location):
                        raise IOError(f"Location {location} is neither a path nor a directory.")
                    self.files_paths += glob.glob(os.path.join(location, f"*.{self.file_suffix}"))
            if not self.files_paths:
                raise IOError(f"Error, the specified file path {self.location} does not contain hdf5 files.")
            self.file_format = "h5"
            self.files_paths.sort()
            self._get_stats_h5(enable_logging)

            if not self.relative_timestamp:
                self.years = sorted({date.year for date in self.datestamps.tolist()})
                self.n_years = len(self.years)
            self.files = [None for _ in self.files_paths]
            self.start_date = self.datestamps[0]
            self.end_date = self.datestamps[-1]

            crop_x, crop_y = self.crop_size
            crop_x = self.img_shape[0] if crop_x is None else crop_x
            crop_y = self.img_shape[1] if crop_y is None else crop_y
            self.crop_size = (crop_x, crop_y)
            assert self.crop_anchor[0] + self.crop_size[0] <= self.img_shape[0]
            assert self.crop_anchor[1] + self.crop_size[1] <= self.img_shape[1]
            split_x = compute_split_shapes(self.crop_size[0], self.io_grid[0])
            split_y = compute_split_shapes(self.crop_size[1], self.io_grid[1])
            self.read_anchor = (self.crop_anchor[0] + sum(split_x[: self.io_rank[0]]),
                                self.crop_anchor[1] + sum(split_y[: self.io_rank[1]]))
            self.read_shape = (split_x[self.io_rank[0]], split_y[self.io_rank[1]])
            self.return_shape = (math.ceil(self.read_shape[0] / self.subsampling_factor),
                                 math.ceil(self.read_shape[1] / self.subsampling_factor))

            self.file_offsets = list(accumulate(self.n_samples_file, operator.add))[:-1]
            self.file_offsets.insert(0, 0)
            self.n_samples_available = sum(self.n_samples_file)
            self.n_samples_total = self.n_samples_available
            if enable_logging:
                logging.info("Found data at path %s. Number of examples: %d. Full image shape: %d x %d x %d. "
                             "Read shape: %d x %d x %d", self.location, self.n_samples_available,
                             self.img_shape[0], self.img_shape[1], self.total_channels,
                             self.read_shape[0], self.read_shape[1], self.n_in_channels)

            self.img_shape_x, self.img_shape_y = self.img_shape[0], self.img_shape[1]
            self.img_crop_shape_x, self.img_crop_shape_y = self.crop_size
            self.img_crop_offset_x, self.img_crop_offset_y = self.crop_anchor[0], self.crop_anchor[1]
            self.img_local_shape_x, self.img_local_shape_y = self.read_shape
            self.img_local_offset_x, self.img_local_offset_y = self.read_anchor
            self.img_shape_resampled = (math.ceil(self.img_shape[0] / self.subsampling_factor),
                                        math.ceil(self.img_shape[1] / self.subsampling_factor))
            self.img_local_shape_x_resampled, self.img_local_shape_y_resampled = self.return_shape
            self.img_shape_x_resampled, self.img_shape_y_resampled = self.img_shape_resampled

            # local coordinates exactly as the pin's __init__ derives them
            latitude, longitude = np.array(self.lat_lon[0]), np.array(self.lat_lon[1])
            lat_loc = latitude[self.read_anchor[0]: self.read_anchor[0] + self.read_shape[0]][:: self.subsampling_factor]
            lon_loc = longitude[self.read_anchor[1]: self.read_anchor[1] + self.read_shape[1]][:: self.subsampling_factor]
            self.lat_lon_local = (lat_loc.tolist(), lon_loc.tolist())
            self.backend = _ChunkOnlyBackend(
                GridSpec(getattr(self, "grid_type", "equiangular"), tuple(self.return_shape), lat_loc, lon_loc))

        def _open_file(self, file_idx):
            _file = h5py.File(self.files_paths[file_idx], "r", driver=self.file_driver, **self.file_driver_kwargs)
            self.files[file_idx] = _file[self.dataset_path]

    @staticmethod
    def _infer_step_seconds(timestamps: list[np.ndarray]) -> int:
        steps: list[int] = []
        for ts in timestamps:
            diffs = np.diff(ts)
            if diffs.size == 0:
                continue
            unique = np.unique(diffs)
            if unique.size != 1:
                raise RuntimeError(
                    "PlaSim file has non-uniform within-file timestamps: "
                    f"{unique[:10].tolist()}"
                )
            steps.append(int(unique[0]))

        if not steps:
            raise RuntimeError("PlaSim dataset needs at least one timestamp step.")
        if len(set(steps)) != 1:
            raise RuntimeError(
                "PlaSim files disagree on within-file timestamp step: "
                f"{sorted(set(steps))}"
            )

        step_seconds = steps[0]
        if step_seconds <= 0:
            raise RuntimeError(f"PlaSim timestamp step must be positive: {step_seconds}")
        return step_seconds

    # ---- file handles --------------------------------------------------
    def _open_diag(self, file_idx: int) -> None:
        if self._diag_files[file_idx] is None:
            f = h5py.File(
                self.files_paths[file_idx],
                "r",
                driver=self.file_driver,
                **self.file_driver_kwargs,
            )
            self._diag_files[file_idx] = f[self.diagnostic_dataset_path]

    def _open_forcing(self, file_idx: int) -> None:
        if self._forcing_files[file_idx] is None:
            f = h5py.File(
                self.files_paths[file_idx],
                "r",
                driver=self.file_driver,
                **self.file_driver_kwargs,
            )
            self._forcing_files[file_idx] = f[self.forcing_dataset_path]

    # ---- slicing helpers ----------------------------------------------
    def _crop_bounds(self) -> Tuple[int, int, int, int]:
        sx = self.read_anchor[0]
        ex = sx + self.read_shape[0]
        sy = self.read_anchor[1]
        ey = sy + self.read_shape[1]
        return sx, ex, sy, ey

    # ---- reads --------------------------------------------------------
    def _read_state(
        self,
        global_idx: int,
        offset_start: int,
        offset_end: int,
        *,
        channels: np.ndarray,
    ) -> np.ndarray:
        """Read ``/fields_state`` for a range of offsets, restricted to
        ``channels``.

        Caller passes the explicit channel list — no mutation of
        ``self.in_channels`` / ``self.out_channels``. Codex round 1 fix #8
        of docs/sfno_training_implementation_plan.md.
        """
        sx, ex, sy, ey = self._crop_bounds()
        ch = np.asarray(channels)
        ch_sorted = np.sort(ch)
        ch_unsort = np.argsort(np.argsort(ch))
        ch_is_sorted = bool(np.all(ch_sorted == ch))

        items: list[np.ndarray] = []
        for off in range(offset_start, offset_end):
            file_idx, local_idx = self._get_indices(global_idx + self.dt * off)
            if self.files[file_idx] is None:
                self._open_file(file_idx)
            arr = self.files[file_idx][
                local_idx : local_idx + 1,
                ch_sorted,
                sx:ex,
                sy:ey,
            ]
            if not ch_is_sorted:
                arr = arr[:, ch_unsort, :, :]
            items.append(arr)
        return np.concatenate(items, axis=0)

    def _read_diagnostic(
        self, global_idx: int, offset_start: int, offset_end: int
    ) -> np.ndarray:
        sx, ex, sy, ey = self._crop_bounds()
        items: list[np.ndarray] = []
        for off in range(offset_start, offset_end):
            file_idx, local_idx = self._get_indices(global_idx + self.dt * off)
            self._open_diag(file_idx)
            arr = self._diag_files[file_idx][
                local_idx : local_idx + 1,
                :,
                sx:ex,
                sy:ey,
            ]
            items.append(arr)
        return np.concatenate(items, axis=0)

    def _read_forcing(
        self, global_idx: int, offset_start: int, offset_end: int
    ) -> np.ndarray:
        sx, ex, sy, ey = self._crop_bounds()
        items: list[np.ndarray] = []
        for off in range(offset_start, offset_end):
            file_idx, local_idx = self._get_indices(global_idx + self.dt * off)
            self._open_forcing(file_idx)
            arr = self._forcing_files[file_idx][
                local_idx : local_idx + 1,
                :,
                sx:ex,
                sy:ey,
            ]
            items.append(arr)
        return np.concatenate(items, axis=0)

    # ---- sample assembly ---------------------------------------------
    def get_sample_at_index(self, global_idx: int, return_target: bool = True):
        """Return ``(inp_state, tar, inp_forcing, tar_forcing)``.

        Shapes (4D ``(T, C, H, W)`` per stock convention; ``T = n_history+1``
        on input, ``T = n_future+1`` on target):

          - ``inp_state``   : ``(n_history+1, 52, H, W)``  -- normalized by ``in_bias/in_scale``
          - ``tar``         : ``(n_future+1, 53, H, W)``  -- state ‖ diagnostic, by ``out_bias/out_scale``
          - ``inp_forcing`` : ``(n_history+1, 6, H, W)``   -- by ``forcing_bias/forcing_scale``
          - ``tar_forcing`` : ``(n_future+1, 6, H, W)``    -- same
        """
        n_hist = self.n_history
        n_fut = self.n_future

        inp_state = self._read_state(
            global_idx, 0, n_hist + 1, channels=self.in_channels
        )
        inp_state = (inp_state - self.in_bias) / self.in_scale

        inp_forcing = self._read_forcing(global_idx, 0, n_hist + 1)
        inp_forcing = (inp_forcing - self.forcing_bias) / self.forcing_scale

        result: list[np.ndarray] = [inp_state]

        if return_target:
            # Target state: out_channels[:52] (the 53rd index is the
            # diagnostic, read from /fields_diagnostic separately).
            state_channels = np.asarray(self.out_channels)[: self.n_out_channels - 1]
            tar_state = self._read_state(
                global_idx,
                n_hist + 1,
                n_hist + n_fut + 2,
                channels=state_channels,
            )
            tar_diag = self._read_diagnostic(
                global_idx, n_hist + 1, n_hist + n_fut + 2
            )
            tar = np.concatenate([tar_state, tar_diag], axis=1)
            tar = (tar - self.out_bias) / self.out_scale
            result.append(tar)

        result.append(inp_forcing)

        if return_target:
            tar_forcing = self._read_forcing(
                global_idx, n_hist + 1, n_hist + n_fut + 2
            )
            tar_forcing = (tar_forcing - self.forcing_bias) / self.forcing_scale
            result.append(tar_forcing)

        tensors = tuple(
            torch.as_tensor(arr, dtype=torch.float32) for arr in result
        )
        tensors = tuple(self.grid_converter(t) for t in tensors)
        return tensors

    def __getitem__(self, global_idx: int):
        return self.get_sample_at_index(global_idx, return_target=self.return_target)
