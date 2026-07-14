# `common/` — shared Lightning-port utilities

> Status: port complete through Phase 6. The "Contents" list below is the
> original Phase-0 plan, retained as build history; landed entries are
> present-tense and entries never created (no S2S analogue) are marked.

This directory mirrors `$SI_DIR/common/` (`/project/pedramh/shared/anthonyz/common/`)
as part of the S2S -> PyTorch Lightning port (restructure-in-place; the `v2.0/`
originals are left untouched).

## Contents

- `bench_callback.py` — **landed.** `BenchCallback(L.Callback)`, mirrors
  `$SI_DIR/common/bench_callback.py`; this is where S2S's in-loop `S2S_BENCH` /
  NVTX instrumentation moved onto Lightning hooks (Phase 3).
- `set_epoch_callback.py` — **landed.** `SetEpochCallback(L.Callback)` (no SI
  analogue); calls `set_epoch` on S2S's own `DistributedSampler` each epoch,
  needed because the entry points run `use_distributed_sampler=False`.
- `utils.py` — **not ported.** SI's `common/utils.py` (`get_yaml`,
  `dict2namespace`, ...) is for its flat YAML config; S2S keeps its own
  `v2.0/utils/YParams.py` + sectioned YAML (a necessary-S2S divergence settled
  at Phase 5), so no `common/utils.py` was created.
- `loss.py` — **not ported.** S2S's training losses are **reused** from
  `v2.0/utils/losses.py`, not reimplemented; SI's `common/loss.py`
  (`latitude_weighted_rmse`) has no S2S consumer.
- `plotting.py` — **not ported** (no S2S analogue needed).

## Packaging note

SI's source packages contain **no `__init__.py`** — they are PEP 420 implicit
namespace packages imported absolutely (e.g.
`from common.bench_callback import BenchCallback`) with the repo root on the
path. This scaffold mirrors that exactly: **do not add `__init__.py` here.** Run
Lightning entry points from the repo root.

## Environment (LPORT_ENV — recorded for later phases)

The known-good interpreter for this port is:

    LPORT_ENV=/project/pedramh/shared/S2S/v2.0/venv

It is S2S's own venv (where `v2.0/networks/pangu.py`, `v2.0/utils/losses.py`,
and the HDF5 loaders run) and already provides `lightning`:
`torch 2.6.0+cu124`, `lightning 2.5.0.post0`, Python 3.11.11. Do **not** clone it
or run a conda/mamba solve. Unifying with SI's torch-2.11 / py-3.13 env is a
Phase-5 concern.

Note: `PYTHONPATH=v2.0/` is still required so `from utils...` / `from networks...`
resolve against S2S's existing code (see the S2S hard constraints).

## `data/` coexistence (Phase 0 collision, resolved without clobbering)

The scaffold mirrors SI's `modules/`, `data/`, `configs/`, `common/`. Of these,
**`data/` already existed in S2S** as a data-asset folder holding
`data/constant_mask/` (`land_mask.npy`, `soil_type.npy`, `topography.npy`). It is
**left exactly as-is** — not clobbered. SI's `data/` is a PEP 420 namespace
package with no `__init__.py`, so in Phase 1 the ported `data/datamodule.py`
(wrapping S2S's existing `get_data_loader` / `get_infer_data` / `GetDataset` from
`v2.0/utils/data_loader_multifiles.py`) can be added **alongside**
`data/constant_mask/` without conflict.

Caveat for later phases: S2S's `.gitignore` has a blanket `data/` rule (the
`constant_mask/*.npy` assets are tracked only because they predate that rule).
New tracked Python sources placed under `data/` in Phase 1 will need an explicit
`git add -f` (or a `.gitignore` carve-out) — a deliberate decision to make then,
not silently in Phase 0. That is why this scaffold ships **no** tracked
`data/README.md`.
