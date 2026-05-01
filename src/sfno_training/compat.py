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

Idempotent: importing this module twice is a no-op.
"""

from __future__ import annotations

import datetime as _dt

from makani.utils.dataloaders import data_helpers as _dh
from makani.utils.dataloaders import data_loader_multifiles as _dlm


def _timedelta_cast(t) -> _dt.timedelta:
    return _dt.timedelta(seconds=int(t))


_dh.get_timedelta_from_timestamp = _timedelta_cast  # type: ignore[assignment]
_dlm.get_timedelta_from_timestamp = _timedelta_cast  # type: ignore[assignment]
