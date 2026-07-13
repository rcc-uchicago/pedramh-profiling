# `modules/` — Lightning modules + ported model

> Status: port complete through Phase 6. The "Contents" list below is the
> original Phase-0 plan, retained as build history; landed entries are
> present-tense and entries never created (no S2S analogue) are marked.

Mirrors `$SNFO_DIR/modules/` (`/project/pedramh/shared/anthonyz/modules/`) for the
S2S -> PyTorch Lightning port (restructure-in-place; `v2.0/` is left untouched).

## Contents

- `train_module.py` — **landed.** `TrainModule(L.LightningModule)`, mirrors
  `$SNFO_DIR/modules/train_module.py`. Holds S2S's `PanguModel_Plasim`
  (imported from `v2.0/networks/pangu.py`) as `self.model`; instantiates the
  losses from `v2.0/utils/losses.py`; ports `v2.0/train.py::Trainer.cal_loss`'s
  7-tuple / CRPS+KL logic into `training_step` (AMP/scheduler delegated to the
  Trainer).
- `ae_module.py` / `combined_module.py` — **not ported** (SNFO-specific
  autoencoder / evaluation modules; S2S has no analogue).
- `models/` — intentional **empty placeholder** (this README + `models/README.md`
  only). SNFO keeps its model definitions under `modules/models/` (`DiT.py`,
  `Unet.py`, ...), but S2S's `PanguModel_Plasim` (and its blocks
  `EarthSpecificLayer` / `EarthAttention3D` / patch embed/recover) is **reused in
  place from `v2.0/networks/pangu.py`** — held as `self.model` in
  `train_module.py`, *not* copied here. **This model is the one intended
  S2S<->SNFO difference.**
- `layers/` — intentional **empty placeholder** (`layers/README.md` only). The
  model is kept whole in `v2.0/networks/pangu.py` rather than split into
  `modules/layers/` blocks, so nothing landed here.

## Packaging note

SNFO source packages have **no `__init__.py`** (PEP 420 namespace packages,
absolute imports, run from repo root). Mirror that: **do not add `__init__.py`.**

See `common/README.md` for the LPORT_ENV designation.
