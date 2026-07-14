# `modules/models/` — ported model definitions

> Status: port complete through Phase 6. This directory mirrors SI's
> `modules/models/` slot but is an intentional **empty placeholder** — see below.

Mirrors `$SI_DIR/modules/models/` (`/project/pedramh/shared/anthonyz/modules/models/`,
which holds `AE.py`, `Decoder.py`, `DiT.py`, `Unet.py`).

This is SI's slot for model definitions. In S2S, `PanguModel_Plasim` (and its
blocks `EarthSpecificLayer` / `EarthAttention3D` / patch embed/recover) is
**reused in place from `v2.0/networks/pangu.py`** — it is *not* copied here.
`modules/train_module.py::TrainModule` imports it directly
(`from networks.pangu import PanguModel_Plasim`, `modules/train_module.py:59`)
and holds it as `self.model`. So this directory stays an empty placeholder
(this README only). The model is the **only** intended material difference
between the S2S and SI trees — everything else (entry points, DataModule,
LightningModule scaffolding, bench harness, common utils) converges on SI's
shape.

Reuse, do not rewrite: the architecture in `v2.0/networks/pangu.py` is held as
`self.model` inside `modules/train_module.py::TrainModule`, not reimplemented.

No `__init__.py` (mirrors SI's namespace-package style).
