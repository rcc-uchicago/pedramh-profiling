"""Lightning ``DataModule`` wrapping the S2S HDF5 data loaders.

This module ports the data-loading half of ``v2.0/train.py::Trainer.get_dataset``
onto a :class:`lightning.LightningDataModule`, mirroring the SNFO template at
``$SNFO_DIR/data/datamodule.py``. It **reuses** the existing S2S loaders rather
than reimplementing them: the HDF5-backed
:class:`utils.data_loader_multifiles.GetDataset` is constructed once in
:meth:`ClimateDataModule.__init__` (so it is available as the normalizer before
``trainer.fit`` runs), and the train/validation ``DataLoader`` + sampler are
built per-rank in :meth:`ClimateDataModule.setup` -- *inside* ``fit()``, after
Lightning's DDP strategy has initialized the process group -- by reproducing
:func:`utils.data_loader_multifiles.get_data_loader`'s sampler choice and
``DataLoader`` settings against the prebuilt dataset (``setup`` does not call
``get_data_loader`` itself, which would re-read the HDF5 tree).
The inference path (Phase 4) routes through
:func:`utils.data_loader_multifiles.get_infer_data` in
:meth:`ClimateDataModule.predict_dataloader` (used by ``trainer.predict``); the
``trainer.validate`` path used by :mod:`val.py` reuses :meth:`val_dataloader`,
which wraps the same validate loader the canonical ``v2.0/inference.py`` reads.

The ``utils.*`` imports resolve only when ``v2.0/`` is on ``PYTHONPATH``
(``PYTHONPATH=v2.0/``), matching the rest of the ported tree.
"""

import lightning as L
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from utils.data_loader_multifiles import (
    get_data_loader,  # noqa: F401  (canonical loader; referenced in docstrings; setup reproduces it without calling it)
    get_infer_data,  # Phase 4 inference loader (called from predict_dataloader)
    GetDataset,
)


class ClimateDataModule(L.LightningDataModule):
    """Wrap S2S's train/val HDF5 loaders as a Lightning ``DataModule``.

    The heavy :class:`~utils.data_loader_multifiles.GetDataset` objects (reading
    dates, constant-boundary data, the land mask, and mean/std statistics off the
    HDF5 tree at ``params.data_dir``) are constructed once in :meth:`__init__`,
    because the training dataset doubles as the normalizer and the entry point
    needs it *before* ``trainer.fit`` (watch-point (b)). The ``DataLoader`` + its
    sampler, however, are deferred to :meth:`setup` so they are built per-rank
    **inside** ``fit()`` -- after Lightning's DDP strategy has initialized the
    process group -- which is what makes watch-point (a) correct under DDP.

    Three integration watch-points carried over from the manual
    ``Trainer.get_dataset`` loop are resolved here:

    (a) **Distributed sampler (built in** :meth:`setup` **, not** :meth:`__init__`
        **).** S2S's :func:`~utils.data_loader_multifiles.get_data_loader` chooses
        the sampler from the process-group state at call time: a
        :class:`torch.utils.data.distributed.DistributedSampler` when a process
        group is initialized, otherwise a :class:`torch.utils.data.RandomSampler`
        (single-process). Under Lightning DDP the process group is brought up
        *inside* ``trainer.fit`` (``DDPStrategy.setup_environment`` ->
        ``init_process_group``), so the loader/sampler must be created from within
        :meth:`setup` (which Lightning calls per-rank after ``fit`` starts) rather
        than from :meth:`__init__` (which the entry point runs *before* ``fit``,
        when no process group exists and every rank would otherwise get an
        unsharded ``RandomSampler``). Because the sampler is built per-rank with
        the group up, it is a real ``DistributedSampler`` under DDP, faithful to
        the canonical ``v2.0/train.py`` (which calls ``init_process_group`` at
        import and therefore always gets ``DistributedSampler``s for both train
        and val). The prebuilt loader is already correctly sampled, so Lightning
        must not inject a *second* distributed sampler: the entry point builds the
        ``Trainer`` with ``use_distributed_sampler=False`` (wired in Phase 3) and
        this module retains the train sampler as :attr:`_train_sampler`.
        :class:`common.set_epoch_callback.SetEpochCallback` then calls
        ``set_epoch`` on it each epoch for correct cross-epoch shuffling -- a
        no-op for the single-process ``RandomSampler`` (no ``set_epoch``), which
        re-seeds itself. This lifecycle governs **both** the train and validation
        loaders, which share the same ``distributed`` flag resolved in
        :meth:`setup`.

    (b) **Normalizer.** The training dataset doubles as the normalization /
        statistics source — it holds ``constant_boundary_data``, ``land_mask``,
        and the surface/upper-air/diagnostic/boundary means and stds. It is built
        in :meth:`__init__` and exposed as :attr:`train_dataset` exactly as the
        SNFO template does, so the entry point can pass
        ``normalizer=datamodule.train_dataset`` into the ``LightningModule``
        *before* ``fit`` (Phase 2/3).

    (c) **Batch size.** The per-GPU batch size is read directly from
        ``params.batch_size`` by the loader. Unlike the old ``__main__`` in
        ``v2.0/train.py`` (which divided ``batch_size`` by world size before
        constructing loaders), this module does **not** divide: under Lightning
        the config carries the per-GPU batch size and the DDP strategy handles
        scaling across ranks. ``params`` is passed through untouched.

    Attributes:
        params: The S2S parameter object (e.g. a
            :class:`utils.YParams.YParams` instance, or any mapping exposing the
            same attribute/item access). Passed through to the loaders.
        train_dataset (GetDataset): The training dataset; also the normalizer /
            statistics source (see watch-point (b)). Built in :meth:`__init__`.
        val_dataset (GetDataset): The validation dataset. Built in :meth:`__init__`.

    See Also:
        utils.data_loader_multifiles.get_data_loader: The canonical S2S loader
            whose sampler/DataLoader construction :meth:`setup` reproduces against
            the prebuilt datasets.
        utils.data_loader_multifiles.get_infer_data: The inference loader used by
            the Phase 4 inference path.
        utils.data_loader_multifiles.GetDataset: The HDF5-backed dataset built in
            :meth:`__init__` and wrapped by both loaders.
        common.set_epoch_callback.SetEpochCallback: Drives ``set_epoch`` on
            :attr:`_train_sampler` each epoch (watch-point (a)).
    """

    def __init__(self, params) -> None:
        """Build the train and validation datasets (the normalizer source).

        Only the HDF5-backed datasets are constructed here -- the ``DataLoader``
        and its sampler are deferred to :meth:`setup` so they are built per-rank
        after Lightning brings up the DDP process group (watch-point (a)). The
        training dataset is needed eagerly because it doubles as the normalizer
        passed into the ``LightningModule`` before ``trainer.fit`` (watch-point
        (b)).

        Args:
            params: S2S parameter object (attribute- and item-accessible, e.g.
                :class:`utils.YParams.YParams`) carrying ``data_dir``,
                ``batch_size``, ``num_data_workers``, the year ranges
                (``train_year_start`` / ``train_year_end`` / ``val_year_start`` /
                ``val_year_end``), ``num_inferences``, and the variable / stats
                configuration consumed by
                :class:`~utils.data_loader_multifiles.GetDataset`.
        """
        super().__init__()
        self.params = params

        # Build the datasets only (the heavy HDF5 / .nc reads) -- NOT the
        # DataLoader/sampler, which setup() builds per-rank once the process group
        # is up. The constructor args mirror get_data_loader's internal call:
        #   GetDataset(params, files_pattern, year_start, year_end, train,
        #              num_inferences, validate)
        # with files_pattern == params.data_dir, matching
        # v2.0/train.py::Trainer.get_dataset.
        self.train_dataset = GetDataset(
            params,
            params.data_dir,
            params.train_year_start,
            params.train_year_end,
            True,        # train
            0,           # num_inferences
            False,       # validate
        )
        self.val_dataset = GetDataset(
            params,
            params.data_dir,
            params.val_year_start,
            params.val_year_end,
            False,                   # train
            params.num_inferences,   # num_inferences
            True,                    # validate
        )

        # DataLoaders + sampler are created in setup() (watch-point (a)).
        self._train_loader = None
        self._train_sampler = None
        self._val_loader = None

    def prepare_data(self) -> None:
        """Lightning hook for one-time, single-process data preparation.

        No-op: the S2S HDF5 dataset at ``params.data_dir`` is read in place on a
        shared filesystem (no download or staging step), so there is nothing to
        do here. Mirrors the SNFO template.
        """
        pass

    def setup(self, stage: str) -> None:
        """Build the train/val ``DataLoader`` + sampler per-rank, inside ``fit``.

        This is where watch-point (a) is enforced: Lightning calls :meth:`setup`
        per-rank *after* the DDP strategy has initialized the process group, so
        the ``distributed`` flag below is ``True`` under DDP and the sampler is a
        real :class:`torch.utils.data.distributed.DistributedSampler` (sharded
        across ranks). In a single-process run (no process group) it is a
        :class:`torch.utils.data.RandomSampler`, leaving the existing 1-GPU path
        unchanged. The loader/sampler construction reproduces
        :func:`utils.data_loader_multifiles.get_data_loader`'s sampler-selection
        logic (and, via :meth:`_make_loader`, its ``DataLoader`` settings) against
        the datasets already built in :meth:`__init__`, so the heavy HDF5 reads are
        not repeated; ``setup`` does not call ``get_data_loader`` itself.

        Idempotent: Lightning may call :meth:`setup` more than once (e.g. ``fit``
        then ``validate``); the train loader is built only when it does not
        already exist, and the val loader only when missing.

        Args:
            stage: The Lightning stage (``"fit"``, ``"validate"``, ``"test"``, or
                ``"predict"``).
        """
        distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )

        # Train loader: built for the "fit" stage (or when called with no stage).
        # The sampler choice mirrors the sampler block in get_data_loader:
        # DistributedSampler under a live process group, else RandomSampler.
        if stage in (None, "fit") and self._train_loader is None:
            self._train_sampler = (
                DistributedSampler(self.train_dataset, shuffle=True)
                if distributed
                else RandomSampler(self.train_dataset)
            )
            self._train_loader = self._make_loader(
                self.train_dataset, self._train_sampler
            )

        # Val loader: built for "fit" (validation during training) and "validate".
        # Validation uses no shuffle; under DDP that is a non-shuffling
        # DistributedSampler (sharded), faithful to canonical v2.0/train.py.
        if stage in (None, "fit", "validate") and self._val_loader is None:
            val_sampler = (
                DistributedSampler(self.val_dataset, shuffle=False)
                if distributed
                else None
            )
            self._val_loader = self._make_loader(self.val_dataset, val_sampler)

    def _make_loader(self, dataset, sampler) -> DataLoader:
        """Construct a ``DataLoader`` matching ``get_data_loader``'s settings.

        Reproduces the ``DataLoader(...)`` call in
        :func:`utils.data_loader_multifiles.get_data_loader`
        verbatim -- same ``batch_size`` / ``num_workers`` / ``shuffle=False``
        (the sampler owns ordering) / ``drop_last`` / ``pin_memory`` -- so the
        deferred-construction path stays byte-for-byte faithful to the canonical
        loader, differing only in *when* it runs (watch-point (a)).

        Args:
            dataset: The HDF5-backed :class:`~utils.data_loader_multifiles.GetDataset`.
            sampler: The sampler to drive ordering (a
                :class:`~torch.utils.data.distributed.DistributedSampler`,
                :class:`~torch.utils.data.RandomSampler`, or ``None`` for the
                sequential validation case).

        Returns:
            torch.utils.data.DataLoader: The configured loader.
        """
        return DataLoader(
            dataset,
            batch_size=int(self.params.batch_size),
            num_workers=self.params.num_data_workers,
            shuffle=False,  # sampler (or sequential default) owns ordering
            sampler=sampler,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
        )

    def train_dataloader(self):
        """Return the training dataloader built in :meth:`setup`.

        Lightning calls this after :meth:`setup` (inside ``fit``), so the loader
        and its sampler exist by now.

        Returns:
            torch.utils.data.DataLoader: The training loader built in
            :meth:`setup`, including its own sampler (a
            :class:`~torch.utils.data.distributed.DistributedSampler` under DDP,
            else a :class:`~torch.utils.data.RandomSampler`; see watch-point (a)
            in the class docstring).
        """
        return self._train_loader

    def val_dataloader(self):
        """Return the validation dataloader built in :meth:`setup`.

        Returns:
            torch.utils.data.DataLoader: The validation loader built in
            :meth:`setup`.
        """
        return self._val_loader

    def test_dataloader(self):
        """Return the test dataloader.

        Returns:
            None: S2S has no separate test split; mirrors the SNFO template.
        """
        return None

    def predict_dataloader(self):
        """Return the prediction/inference dataloader (Phase 4).

        Built lazily here via S2S's dedicated inference loader
        :func:`utils.data_loader_multifiles.get_infer_data` -- which constructs
        its own ``GetDataset(train=False, validate=...)`` plus a
        ``DataLoader(sampler=None, persistent_workers=True, prefetch_factor=8)``
        -- so the ``trainer.predict`` path uses the same dedicated loader the
        canonical ``v2.0/inference.py`` would have used.

        Note:
            The canonical inference path in
            :class:`v2.0/inference.py::Stepper` (read via ``get_data_loader(...,
            train=False, validate=True)``) is the **same validate loader** this
            module already exposes as :meth:`val_dataloader`; ``get_infer_data``
            differs only by forcing ``sampler=None`` and adding
            ``persistent_workers``/``prefetch_factor``. The entry point
            :mod:`val.py` therefore drives inference through ``trainer.validate``
            (reusing the already-built, already-smoked validation loader and the
            rank0/batch0 save hook in
            :meth:`modules.train_module.TrainModule.validation_step`); this
            ``predict_dataloader`` exists so a ``trainer.predict`` driver can use
            S2S's dedicated inference loader without losing it.

        Returns:
            torch.utils.data.DataLoader: The inference loader from
            :func:`utils.data_loader_multifiles.get_infer_data` (single-process;
            no distributed sampler).
        """
        distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        loader, _dataset = get_infer_data(
            self.params,
            self.params.data_dir,
            distributed,
            self.params.val_year_start,
            self.params.val_year_end,
            num_inferences=self.params.num_inferences,
            validate=True,
        )
        return loader
