# Grid declaration: what makani main requires, and why the port waives one check

**Status (2026-10-01):** describes the operator's ruling on `api_delta.md` §3.2 (dataset-scoped opt-out, outputs
bitwise) and commit c8 (scoped `verify_grid_type` rebind, awaiting operator approval in the worker session).

## What makani main requires

Each dataset's `data.json` declares `coords.grid_type` and lists `coords.lat`. makani main checks that the latitudes
are the nodes of the declared grid type, to within 1e-3° (`makani/utils/grid_types.py:37`, check at `:88`). If they
are not, `parse_dataset_metadata` raises at configuration load (`makani/utils/parse_dataset_metada.py:54`). The
makani version we have used so far (`c9704308`) has no such check.

| declared grid type | latitudes makani expects | our latitudes, 180 rows, 89.5° … −89.5° |
|---|---|---|
| `equiangular`, `clenshaw-curtiss`, `weatherbench2` | equally spaced from −90° to 90°, poles included: `linspace(-90, 90, n)` (`grid_types.py:54`) | do not match: offset of up to 0.5° |
| `legendre-gauss` | Gauss–Legendre nodes | do not match |
| `euclidean` | not checked; declares the data not to lie on a sphere | not applicable to global fields |

Our packs place values at the centres of 1° latitude bands, without pole rows. None of the grid types known to makani
has these nodes, so no correct declaration passes the check. The packs declare `equiangular`.

## What the declaration controls

The declared type (`data_grid_type`) is separate from the grid the network computes on (`model_grid_type`, which stays
`equiangular`). The spherical harmonic transforms of the network, the loss and the metrics use `model_grid_type`
(`sfnonet.py:802-803`, `loss.py:142`, `metric.py:343`). A checkpoint's inference output therefore does not depend on
the quadrature weights. The weights enter the area-weighted loss and metrics.

Our cell-centred rows are weighted as if they were equiangular nodes. Measured on checkpoint A e243, one step ahead,
64 samples (M1 job 7680816): the channel-mean loss changes by −0.074 % under cell-centred weights. The largest changes
are in near-surface relative humidity: RELHUM level 0 +2.45 %, level 1 +1.79 %, level 2 +1.67 %.

## Options

| option | effect on existing checkpoints | effect on outputs |
|---|---|---|
| 1. waive the check for this grid only (c8) | unchanged | unchanged; each job logs `GRID_VERIFY_OPTOUT equiangular_cellcentred`; any other grid is still checked |
| 2. regrid the data to a supported grid (181 rows with poles, or 180 Gauss rows) | inputs move by up to 0.5° in latitude; 181 rows also changes the input shape; amounts to retraining | changed; a science decision |
| 3. declare a different `data_grid_type` | the dataloader receives both grid types; converting between them when they differ is unconfirmed | changed if it resamples; `euclidean` misdescribes the data |
| 4. write `linspace(-90, 90, 180)` into `coords.lat` | unchanged | unchanged, but the metadata no longer records where the values are; not recommended |

The port takes option 1. The weighting error stays as it is in every existing run. Correct cell-centred weights for
new training runs are a separate opt-in item (S-GW), which needs jesswan's written approval.
