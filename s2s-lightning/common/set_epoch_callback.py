"""Lightning callback that calls ``set_epoch`` on S2S's own DistributedSampler.

Because the ported entry points build the ``Trainer`` with
``use_distributed_sampler=False`` (so Lightning does not inject a second
sampler on top of the one S2S's
:func:`utils.data_loader_multifiles.get_data_loader` already builds), Lightning
also stops calling ``set_epoch`` on the dataloader's sampler each epoch --
Lightning only does that for samplers it injects itself. Under DDP that sampler
is a :class:`torch.utils.data.distributed.DistributedSampler`, whose per-epoch
shuffle is only re-seeded when ``set_epoch`` is called; without this callback
every epoch would replay the same shuffle.

This callback restores that behaviour: :meth:`on_train_epoch_start` calls
``set_epoch(current_epoch)`` on the DataModule's retained sampler
(:attr:`data.datamodule.ClimateDataModule._train_sampler`), guarded behind an
active process group, exactly as ``v2.0/train.py::Trainer.train`` did. In a
single-process run the sampler is a :class:`torch.utils.data.RandomSampler`
(which has no ``set_epoch`` and re-seeds itself), so the guard makes this a
no-op there.

See Also:
    data.datamodule.ClimateDataModule: Exposes the retained training sampler as
        ``_train_sampler`` (watch-point (a) in its class docstring).
"""

import lightning as L
import torch


class SetEpochCallback(L.Callback):
    """Forward the epoch index to S2S's DistributedSampler each epoch.

    Restores the per-epoch ``set_epoch`` call that Lightning omits when
    ``use_distributed_sampler=False`` (see the module docstring). The call is
    guarded behind ``torch.distributed.is_available() and
    torch.distributed.is_initialized()``, so it is a no-op for single-process
    runs whose sampler is a :class:`torch.utils.data.RandomSampler`.

    Wire this into the ``Trainer``'s callback list in the entry point (it is
    added by ``train.py`` alongside the checkpoints / LR monitor).
    """

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        """Call ``set_epoch(current_epoch)`` on the DataModule's train sampler.

        No-op unless a process group is initialized and the DataModule exposes a
        ``_train_sampler`` with a ``set_epoch`` method (the
        :class:`~torch.utils.data.distributed.DistributedSampler` case under
        DDP).

        Args:
            trainer: The Lightning ``Trainer``; its ``datamodule`` is expected to
                be a :class:`data.datamodule.ClimateDataModule` (it carries the
                ``_train_sampler``) and ``current_epoch`` supplies the seed.
            pl_module: The Lightning module being trained (unused).
        """
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return
        datamodule = getattr(trainer, "datamodule", None)
        sampler = getattr(datamodule, "_train_sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(trainer.current_epoch)
