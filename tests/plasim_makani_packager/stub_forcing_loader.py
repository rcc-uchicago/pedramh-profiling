"""Stub PlaSim-specific Makani components used by Phase 4b smoke.

Minimal stand-ins for the production implementations that will ship in
src/sfno_training/ (follow-up PR per docs/plasim_makani_packager_plan.md
§7). These stubs are not production code and are deliberately kept
small; their job is to exercise the data contract / training-loop wiring
end-to-end on real HDF5 files, so that the packager PR cannot silently
regress the interface.

Requires torch + makani to be importable.
"""

from __future__ import annotations

from typing import Tuple

import datetime as _dt

import h5py
import numpy as np
import torch

from makani.models.preprocessor import Preprocessor2D
from makani.models.stepper import MultiStepWrapper, SingleStepWrapper
from makani.utils.dataloaders.data_loader_multifiles import MultifilesDataset

# --- Python 3.12 compat shim ------------------------------------------------
# `dt.timedelta(seconds=x)` on Python 3.12 rejects numpy integers (pre-3.12
# auto-converted). Makani reads our int64 /timestamp via h5py and passes the
# raw np.int64 into `get_timedelta_from_timestamp` → TypeError. Stock code
# does `from makani.utils.dataloaders.data_helpers import get_timedelta_from_timestamp`
# at module scope (data_loader_multifiles.py:31), which binds the name into
# that module — so both the source and the importer's binding need to be
# patched. Same fix will need to land in the production trainer path too
# (tracked in the src/sfno_training/ follow-up).
from makani.utils.dataloaders import data_helpers as _dh  # noqa: E402
from makani.utils.dataloaders import data_loader_multifiles as _dlm  # noqa: E402


def _timedelta_cast(t):
    return _dt.timedelta(seconds=int(t))


_dh.get_timedelta_from_timestamp = _timedelta_cast  # type: ignore[assignment]
_dlm.get_timedelta_from_timestamp = _timedelta_cast  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PlasimForcingDataset(MultifilesDataset):
    """Subclass returning (inp_state, tar, inp_forcing, tar_forcing).

    - Input state: reads ``/fields_state`` columns ``in_channels`` (52).
    - Target: reads ``/fields_state`` columns ``out_channels[:52]`` +
      ``/fields_diagnostic`` column 0 → 53 channels along axis 1.
    - Forcing (input and target): reads ``/forcing`` full (6 channels).

    All tensors are 4D ``(n_history+1, C, H, W)`` / ``(n_future+1, C, H, W)``
    to match the stock MultifilesDataset convention.
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
        self, global_idx: int, offset_start: int, offset_end: int, *, is_target: bool
    ) -> np.ndarray:
        """Read ``/fields_state`` for a range of offsets."""
        sx, ex, sy, ey = self._crop_bounds()
        items: list[np.ndarray] = []
        ch = (
            self.out_channels if is_target else self.in_channels
        )  # out_channels[:52] on target path — handled by caller
        ch_sorted = np.sort(ch)
        ch_unsort = np.argsort(np.argsort(ch))
        ch_is_sorted = bool(np.all(ch_sorted == ch))
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
        n_hist = self.n_history
        n_fut = self.n_future

        # Input state (52 channels via self.in_channels)
        inp_state = self._read_state(global_idx, 0, n_hist + 1, is_target=False)
        inp_state = (inp_state - self.in_bias) / self.in_scale

        # Input forcing (6 channels)
        inp_forcing = self._read_forcing(global_idx, 0, n_hist + 1)
        inp_forcing = (inp_forcing - self.forcing_bias) / self.forcing_scale

        result: list[np.ndarray] = [inp_state]

        if return_target:
            # Target state: out_channels[:52] (state portion)
            state_out = list(self.out_channels[: self.out_channels.size - 1])
            # Re-read state for target using state-only slice, then concat diagnostic.
            # The _read_state helper uses self.out_channels directly when is_target=True,
            # but we only want the state portion; temporarily trim.
            orig_out = self.out_channels
            self.out_channels = np.array(state_out)
            try:
                tar_state = self._read_state(
                    global_idx, n_hist + 1, n_hist + n_fut + 2, is_target=True
                )
            finally:
                self.out_channels = orig_out
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


# ---------------------------------------------------------------------------
# Preprocessor — strips diagnostic at append_history time (plan v9)
# ---------------------------------------------------------------------------
class PlasimPreprocessor(Preprocessor2D):
    """Auto-strip diagnostic channels from ``pred`` before the feedback copy.

    Covers two in-scope rollout call sites (inference explicitly out of scope):
      - makani/makani/models/stepper.py:112 (MultiStepWrapper training rollout)
      - makani/makani/utils/training/deterministic_trainer.py:661 (validation)
    """

    def __init__(self, params):
        super().__init__(params)
        self.n_state_channels = params.n_state_channels
        self.n_full_out_channels = (
            params.n_state_channels + params.n_diagnostic_channels
        )

    def append_history(self, x1, x2, step, update_state=True):
        assert x2.dim() == 4, (
            f"expected x2 4D (B, C, H, W), got {x2.dim()}D shape {tuple(x2.shape)}"
        )
        assert x2.shape[1] in (self.n_state_channels, self.n_full_out_channels), (
            f"PlasimPreprocessor.append_history: x2 channels must be "
            f"{self.n_state_channels} or {self.n_full_out_channels}, "
            f"got {x2.shape[1]}"
        )
        if x2.shape[1] == self.n_full_out_channels:
            x2 = x2[:, : self.n_state_channels, ...]
        return super().append_history(x1, x2, step, update_state=update_state)


class PlasimSingleStepWrapper(SingleStepWrapper):
    def __init__(self, params, model_handle):
        super().__init__(params, model_handle)
        self.preprocessor = PlasimPreprocessor(params)


class PlasimMultiStepWrapper(MultiStepWrapper):
    def __init__(self, params, model_handle):
        super().__init__(params, model_handle)
        self.preprocessor = PlasimPreprocessor(params)
