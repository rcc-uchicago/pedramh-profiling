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
pin and makani main (model-parallel layout, ...); see the sections below.

Idempotent: importing this module twice is a no-op.
"""

from __future__ import annotations

import datetime as _dt
import inspect

from makani.utils.dataloaders import data_helpers as _dh
from makani.utils.dataloaders import data_loader_multifiles as _dlm


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
