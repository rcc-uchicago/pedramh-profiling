"""The ensemble/CRPS path must be wired correctly, and every failure here is silent.

``PlasimEnsembleTrainer`` has an empty body and relies entirely on cooperative
multiple inheritance. That is elegant but fragile against upstream change, and
each way it can break produces WRONG TRAINING rather than an exception:

  * if ``EnsembleTrainer`` stops subclassing ``Trainer``, the MRO collapses and
    the PlaSim overrides silently stop applying;
  * if ``_install_plasim_patches`` does not rebind ``ensemble_trainer``'s own
    module-scope ``get_dataloader``, an ensemble run gets STOCK makani's
    dataloader, which has no slot for our 7 forcing channels -- it would train
    happily on wrong inputs;
  * if ``local_ensemble_size`` is unset, ``EnsembleTrainer`` reads it off
    ``params`` and dies -- makani sets it in ``train_stochastic.py`` but NOT in
    ``train.py``, which is the entrypoint our fork is modelled on;
  * if the CRPS loss is paired with ``ensemble_size: 1`` it degenerates
    algebraically to MAE and trains something that is not CRPS at all.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
makani = pytest.importorskip("makani")
yaml = pytest.importorskip("yaml")

from makani.utils.training.deterministic_trainer import Trainer  # noqa: E402
from makani.utils.training.ensemble_trainer import EnsembleTrainer  # noqa: E402
from makani.utils.training import deterministic_trainer, ensemble_trainer  # noqa: E402

from sfno_training.trainer import (  # noqa: E402
    PlasimEnsembleTrainer,
    PlasimTrainer,
    _install_plasim_patches,
    _plasim_get_dataloader,
)

_POLARIS = Path(__file__).resolve().parents[2] / "polaris"


def test_upstream_ensemble_trainer_still_subclasses_trainer():
    """The whole design rests on this. If upstream changes it, fail loudly."""
    assert issubclass(EnsembleTrainer, Trainer), (
        "EnsembleTrainer no longer subclasses Trainer -- PlasimEnsembleTrainer's "
        "cooperative-inheritance design is invalid and its overrides would "
        "silently stop applying."
    )


def test_mro_puts_ensemble_trainer_between_plasim_and_trainer():
    """super() inside PlasimTrainer must resolve to EnsembleTrainer, not Trainer."""
    mro = PlasimEnsembleTrainer.__mro__
    names = [c.__name__ for c in mro]
    assert names[:4] == [
        "PlasimEnsembleTrainer",
        "PlasimTrainer",
        "EnsembleTrainer",
        "Trainer",
    ], f"unexpected MRO: {names[:6]}"

    # The load-bearing consequence: the methods PlasimTrainer delegates through
    # must land on EnsembleTrainer's implementations.
    i_plasim = mro.index(PlasimTrainer)
    i_ens = mro.index(EnsembleTrainer)
    i_det = mro.index(Trainer)
    assert i_plasim < i_ens < i_det


def test_patches_rebind_ensemble_trainer_too():
    """Rebinding deterministic_trainer alone leaves the ensemble path on stock."""
    _install_plasim_patches()
    assert deterministic_trainer.get_dataloader is _plasim_get_dataloader
    assert ensemble_trainer.get_dataloader is _plasim_get_dataloader, (
        "ensemble_trainer.get_dataloader was NOT patched -- an ensemble run "
        "would use stock makani's dataloader, which drops our 7 forcing "
        "channels and would train on silently wrong inputs."
    )
    # sync_params matters for the multi-node model-parallel deadlock fix.
    assert ensemble_trainer.sync_params is deterministic_trainer.sync_params


def test_entrypoint_sets_local_ensemble_size():
    """makani sets this in train_stochastic.py, NOT train.py. We must."""
    src = (Path(__file__).resolve().parents[2]
           / "src" / "sfno_training" / "train_plasim.py").read_text()
    assert 'params["local_ensemble_size"]' in src, (
        "train_plasim.py does not set local_ensemble_size; EnsembleTrainer reads "
        "it off params (ensemble_trainer.py:497) and would raise AttributeError."
    )
    assert "PlasimEnsembleTrainer(params, world_rank)" in src
    assert "PlasimTrainer(params, world_rank)" in src, "deterministic path lost"


def test_crps_config_is_internally_consistent():
    """A CRPS loss with ensemble_size 1 degenerates to MAE -- catch it here."""
    cfg_path = _POLARIS / "e3sm_alldata_crps.yaml"
    assert cfg_path.is_file(), cfg_path
    cfg = yaml.safe_load(cfg_path.read_text())
    # the file holds one or more named configs; find the one carrying losses
    sections = [v for v in cfg.values() if isinstance(v, dict) and "losses" in v]
    assert sections, "no config section with a `losses` key"
    for sec in sections:
        losses = sec["losses"]
        types = [l.get("type") for l in losses]
        assert any(str(t).startswith("ensemble_") for t in types), (
            f"CRPS config carries no ensemble_* loss: {types}"
        )
        ens = int(sec.get("ensemble_size", 1))
        assert ens > 1, (
            f"ensemble_size={ens} with an ensemble loss {types}: CRPS over a "
            "single member is algebraically MAE, so this would train something "
            "that is not CRPS while looking like it is."
        )


def test_crps_loss_type_exists_in_makani_registry():
    """Guard the loss name against upstream renames."""
    from makani.utils.loss import _LOSS_REGISTRY

    cfg = yaml.safe_load((_POLARIS / "e3sm_alldata_crps.yaml").read_text())
    sections = [v for v in cfg.values() if isinstance(v, dict) and "losses" in v]
    for sec in sections:
        for l in sec["losses"]:
            t = l.get("type")
            assert t in _LOSS_REGISTRY, (
                f"loss type {t!r} is not in makani's registry; available: "
                f"{sorted(_LOSS_REGISTRY)}"
            )


def test_deterministic_config_unchanged_by_the_crps_variant():
    """The CRPS file must differ from the baseline in loss/ensemble ONLY."""
    base = yaml.safe_load((_POLARIS / "e3sm_alldata_full.yaml").read_text())
    crps = yaml.safe_load((_POLARIS / "e3sm_alldata_crps.yaml").read_text())
    common = set(base) & set(crps)
    assert common, "no shared config section between the two files"
    for name in common:
        b, c = base[name], crps[name]
        if not (isinstance(b, dict) and isinstance(c, dict)):
            continue
        allowed = {"losses", "ensemble_size"}
        diffs = {
            k for k in set(b) | set(c)
            if b.get(k) != c.get(k)
        } - allowed
        assert not diffs, (
            f"section {name!r} differs outside the loss/ensemble axis: {sorted(diffs)}. "
            "The CRPS arm must be a single-variable change against the baseline."
        )
