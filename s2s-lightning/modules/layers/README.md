# `modules/layers/` — supporting model building blocks

> Status: port complete through Phase 6. This directory is an intentional empty
> placeholder (this README only), retained as build history.

Mirrors `$SI_DIR/modules/layers/` (`/project/pedramh/shared/anthonyz/modules/layers/`,
which holds `basics.py`, `conv.py`, `embedding.py`, `patchify.py`, ...).

Intentional **empty placeholder** (this README only). S2S's `PanguModel_Plasim`
is **kept whole** in `v2.0/networks/pangu.py` (held as `self.model` in
`modules/train_module.py`) rather than split into reusable `modules/layers/`
blocks (`EarthSpecificLayer`, `EarthAttention3D`, patch embed/recover,
up/down-sample blocks) — so **nothing landed here**. Reuse, do not rewrite.
See `modules/README.md`.

No `__init__.py` (mirrors SI's namespace-package style).
