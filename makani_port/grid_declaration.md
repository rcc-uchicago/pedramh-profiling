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
(`sfnonet.py:802-803`, `loss.py:142`, `metric.py:343`). A checkpoint's inference output does not depend on the
declared data grid type or on the loss and metric weights. It does depend on the equiangular nodes assumed by the
spherical harmonic transforms, which are the same in both makani versions and under every option that keeps the data.

The model has always computed on this assumption. makani's equiangular grid of 180 rows has nodes at
`linspace(-90, 90, 180)`, 180/179 = 1.006° apart with both poles included; our rows are 1° apart and half a degree in
from each pole. Each row is placed up to 0.5° from its true latitude, with the largest offset at the poles and almost
none at the equator. The old makani (`c9704308`) made the same assumption without checking it.

## Existing weighting error (present with or without the waiver)

The loss and metrics use makani's `naive` rule, `sin(linspace(0, π, 180))` (`grids.py:113`). It gives the first and
last rows, which are our 89–90° latitude bands, exactly zero weight, and gives rows within about 5° of each pole 67 to
92 % of their true area. Measured on checkpoint A e243, one step ahead, 64 samples (M1 job 7680816): under the true
band areas the channel-mean loss changes by −0.074 %, and near-surface relative humidity by +2.45 % (level 0), +1.79 %
(level 1) and +1.67 % (level 2). These numbers describe what the loss would be under band-area weights; they are not
an effect of the waiver. Our offline scorer (`sfno_eval/metrics.py:51-63`, used by `polaris/score_rollout_nc.py`)
already uses band-area weights, so reported scores are not affected.

## Options

| option | existing checkpoints | model outputs | loss and metric values | notes |
|---|---|---|---|---|
| 1. waive the check for this grid only (c8) | unchanged | unchanged | unchanged | each job logs `GRID_VERIFY_OPTOUT equiangular_cellcentred`; any other grid is still checked |
| 2. regrid to a supported grid (181 rows with poles, or 180 Gauss rows) | inputs move by up to 0.5° (181 rows) or 0.26° (Gauss); amounts to retraining | changed | changed | a science decision |
| 3. declare a different `data_grid_type` | depends on whether makani resamples (unconfirmed) | changed if it resamples | changed if it resamples | `euclidean` misdescribes the data |
| 4. write `linspace(-90, 90, 180)` into `coords.lat` | unchanged | unchanged | unchanged | the zero-weight rows remain; the metadata no longer records where the values are; not recommended |
| 5. add two pole rows (182) | input shape and spectral band limit change; amounts to retraining | changed | polar rows over-weighted up to 2× | still not evenly spaced, so the check still fails |

The port takes option 1. The training loss is not changed: the weighting error stays as it is in every existing
run, and the validation loss that selects `best_ckpt` and drives `ReduceLROnPlateau` keeps makani's weights. Band-area
weights are used for scoring. A different grid or different training weights would change what the model learns and
is a decision for jesswan before a new from-scratch training campaign.
