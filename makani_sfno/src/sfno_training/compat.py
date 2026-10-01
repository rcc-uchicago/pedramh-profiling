"""Python 3.12 shim for Makani's get_timedelta_from_timestamp.

Background
----------
On Python 3.12 ``datetime.timedelta(seconds=x)`` rejects numpy integer
inputs (pre-3.12 they were auto-converted via ``__int__``). Makani reads
the int64 ``/timestamp`` via h5py and passes the raw ``np.int64`` into
``get_timedelta_from_timestamp`` -> TypeError.

The stock import form at
``makani/makani/utils/dataloaders/data_loader_multifiles.py:32`` is::

    from makani.utils.dataloaders.data_helpers import (
        get_date_from_timestamp,
        get_timedelta_from_timestamp,
        ...,
    )

That binds the name into ``data_loader_multifiles``'s namespace at import
time, so we have to patch BOTH the source and the importer's local
binding for the override to take effect on subsequent reads.

makani main (the port target, ``makani_port/``) adds a third binding in
``dataloaders/backends/base.py:74``, used by the backends' timestamp
converter. That package does not exist at the pin, so patching it is
feature-detected by import.

The module also carries the small helpers that let our code run on both the
pin and makani main (checkpoint safe globals, model-parallel layout, loss
compile, grid declaration); see the sections below.

Idempotent: importing this module twice is a no-op.
"""

from __future__ import annotations

import datetime as _dt
import inspect

from makani.utils.dataloaders import data_helpers as _dh
from makani.utils.dataloaders import data_loader_multifiles as _dlm

#: makani main reworked MultifilesDataset onto storage backends and dropped the
#: per-file HDF5 internals PlasimForcingDataset builds on (api_delta §2).
MAKANI_HAS_BACKENDS = not hasattr(_dlm.MultifilesDataset, "_get_stats_h5")


def _timedelta_cast(t) -> _dt.timedelta:
    return _dt.timedelta(seconds=int(t))


_dh.get_timedelta_from_timestamp = _timedelta_cast  # type: ignore[assignment]
_dlm.get_timedelta_from_timestamp = _timedelta_cast  # type: ignore[assignment]
try:
    from makani.utils.dataloaders.backends import base as _backends_base
except ImportError:  # the pin has no backends package
    _backends_base = None
else:
    _backends_base.get_timedelta_from_timestamp = _timedelta_cast  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Legacy checkpoints on makani main (api_delta §3.1, operator ruling 2026-10-01)
#
# main loads checkpoints with torch.load(weights_only=True)
# (checkpoint_helpers.load_checkpoint). Ours pickle two ruamel types, the YAML-parsed
# lr/eps inside optimizer and scheduler state, so even a model-only load is refused.
# ---------------------------------------------------------------------------
def _register_checkpoint_safe_globals() -> bool:
    """Allow exactly ``ScalarFloat`` and ``Anchor`` (measured on all ten golden checkpoints).

    Never the global ``MAKANI_ALLOW_UNSAFE_CHECKPOINT_LOAD`` escape.
    """
    try:
        from makani.utils import checkpoint_helpers as _ch
    except ImportError:
        return False
    if not hasattr(_ch, "load_checkpoint"):
        return False  # the pin loads with weights_only=False; nothing to allow
    import torch
    from ruamel.yaml.anchor import Anchor
    from ruamel.yaml.scalarfloat import ScalarFloat

    torch.serialization.add_safe_globals([ScalarFloat, Anchor])
    return True


CHECKPOINT_SAFE_GLOBALS_REGISTERED = _register_checkpoint_safe_globals()


# ---------------------------------------------------------------------------
# Model-parallel layout: fin x fout (pin) vs one matmul group (main, f9b6e787)
#
# Everything below feature-detects the installed makani and, on the pin,
# reproduces the call it replaces exactly.
# ---------------------------------------------------------------------------
def _comm_uses_fin_fout() -> bool:
    from makani.utils import comm

    names = inspect.signature(comm.init).parameters["model_parallel_names"].default
    return "fin" in list(names)


def model_parallel_layout(h: int, w: int, fin: int = 1, fout: int = 1, matmul: int | None = None):
    """``(sizes, names)`` for ``comm.init`` on the installed makani.

    The pin wants ``[h, w, fin, fout]``; main has one ``matmul`` group, sized
    ``matmul`` if given, else ``fin * fout``.
    """
    if _comm_uses_fin_fout():
        if matmul not in (None, 1):
            raise ValueError(f"matmul_parallel_size={matmul} needs makani main; the pin only has fin/fout")
        return [h, w, fin, fout], ["h", "w", "fin", "fout"]
    m = int(matmul) if matmul is not None else int(fin) * int(fout)
    return [h, w, m], ["h", "w", "matmul"]


def normalize_model_parallel(sizes, names):
    """Re-express a stored ``(model_parallel_sizes, model_parallel_names)`` pair, e.g. a pin
    checkpoint's ``config.json``, for the installed makani."""
    by_name = dict(zip(list(names), [int(s) for s in sizes]))
    return model_parallel_layout(
        by_name.get("h", 1), by_name.get("w", 1), by_name.get("fin", 1), by_name.get("fout", 1),
        by_name.get("matmul"),
    )


def set_model_parallel_params(params, sizes, names) -> None:
    """Write ``<name>_parallel_size`` per group plus the two lists, as ``train.py`` does."""
    for name, size in zip(names, sizes):
        params[f"{name}_parallel_size"] = size
    params["model_parallel_sizes"] = list(sizes)
    params["model_parallel_names"] = list(names)


# ---------------------------------------------------------------------------
# Loss handler: main compiles every non-SHT loss term by default (b4e9c6a, 74ba136)
# ---------------------------------------------------------------------------
def loss_handler_compile_off(loss_handler_cls):
    """``loss_handler_cls`` defaulting to ``compile=False`` where it has that kwarg (main);
    the class itself on the pin. Compiling the loss changes its values (fused reductions),
    so it is an optimisation behind the DESIGN §4 gate -- not part of the port."""
    if "compile" not in inspect.signature(loss_handler_cls.__init__).parameters:
        return loss_handler_cls

    class EagerLossHandler(loss_handler_cls):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("compile", False)
            super().__init__(*args, **kwargs)

    return EagerLossHandler


# ---------------------------------------------------------------------------
# Grid declaration: main verifies coords.lat against coords.grid_type (798245b)
#
# Our packs declare ``equiangular`` but store 180 cell-centred rows, 89.5 ... -89.5,
# which match no grid type makani knows, so main raises at config load
# (``parse_dataset_metada.py:54``). Operator ruling on api_delta §3.2 (2026-10-01):
# waive the check for exactly this declaration, around our own call only, outputs
# bitwise; band-area weights are applied when scoring (makani_port/grid_declaration.md).
# ---------------------------------------------------------------------------
GRID_VERIFY_OPTOUT = "GRID_VERIFY_OPTOUT equiangular_cellcentred"
_CELLCENTRED_LAT = [89.5 - i for i in range(180)]  # convert_e3sm_to_makani.py: arange(89.5, -90, -1)
_CELLCENTRED_ATOL = 1e-3  # makani main's GRID_TYPE_TOLERANCE_DEGREES
_optout_logged = False


def is_our_cellcentred_grid(grid_type, latitudes) -> bool:
    """True only for ``equiangular`` with our 180 cell-centred rows, stored north to south."""
    lat = [float(x) for x in latitudes]
    return (
        grid_type == "equiangular"
        and len(lat) == len(_CELLCENTRED_LAT)
        and all(abs(a - b) <= _CELLCENTRED_ATOL for a, b in zip(lat, _CELLCENTRED_LAT))
    )


def parse_dataset_metadata_scoped(metadata_json_path, params):
    """makani's ``parse_dataset_metadata`` with the grid check waived for our declaration only.

    ``verify_grid_type`` is rebound for the duration of this call and restored in
    ``finally``; any other grid is still checked by the stock function. On the pin,
    which has no check, this is the stock call.
    """
    global _optout_logged
    from makani.utils import parse_dataset_metada as pdm

    stock = getattr(pdm, "verify_grid_type", None)
    if stock is None:
        return pdm.parse_dataset_metadata(metadata_json_path, params=params)

    waived = []

    def verify_unless_ours(grid_type, latitudes, *args, **kwargs):
        if is_our_cellcentred_grid(grid_type, latitudes):
            waived.append(len(latitudes))
            return None
        return stock(grid_type, latitudes, *args, **kwargs)

    pdm.verify_grid_type = verify_unless_ours
    try:
        result = pdm.parse_dataset_metadata(metadata_json_path, params=params)
    finally:
        pdm.verify_grid_type = stock
    if waived and not _optout_logged:
        print(f"{GRID_VERIFY_OPTOUT} n_lat={waived[0]} source={metadata_json_path}", flush=True)
        _optout_logged = True
    return result
