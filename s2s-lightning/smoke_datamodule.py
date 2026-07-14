"""Phase 1 smoke test for :class:`data.datamodule.ClimateDataModule`.

Loads the S2S ``test.yaml`` config via :class:`utils.YParams.YParams`,
instantiates the ported :class:`~data.datamodule.ClimateDataModule` (which builds
the train/val datasets in its ``__init__``), runs ``setup("fit")`` to build the
``DataLoader`` + sampler, and pulls a single training batch to prove the
DataModule wiring is correct end-to-end against the real HDF5 dataset.

Run from the repo root with ``v2.0/`` on ``PYTHONPATH`` so that ``data.*``
resolves (repo root / script dir) and ``utils.*`` resolves (``PYTHONPATH``)::

    PYTHONPATH=/project/pedramh/shared/S2S/v2.0 python smoke_datamodule.py

Prints ``SMOKE_OK`` on success (the commit gate keys on that token).
"""

import os
import sys

import torch

from utils.YParams import YParams

from data.datamodule import ClimateDataModule


def _describe(name, obj):
    """Print a one-line shape/length summary for a batch element.

    Args:
        name: Label for the element.
        obj: A tensor (printed as its shape) or any other object (printed as its
            type, and length when available).
    """
    if isinstance(obj, torch.Tensor):
        print(f"  {name}: Tensor shape={tuple(obj.shape)} dtype={obj.dtype}")
    else:
        extra = f" len={len(obj)}" if hasattr(obj, "__len__") else ""
        print(f"  {name}: {type(obj).__name__}{extra}")


def main():
    """Run the Phase-1 DataModule smoke and print ``SMOKE_OK`` on success.

    Loads ``v2.0/config/test.yaml`` via :class:`utils.YParams.YParams`, forces a
    single-process loader (``num_data_workers=0``), and instantiates
    :class:`data.datamodule.ClimateDataModule` -- whose ``__init__`` builds the
    train/val :class:`~utils.data_loader_multifiles.GetDataset` objects
    (``data/datamodule.py:142``/``151``). It then runs ``setup("fit")`` to build
    the ``DataLoader`` + sampler, asserts the (already-built) datasets and their
    normalizer buffers (``constant_boundary_data`` / ``land_mask``) are present,
    and pulls one training batch -- proving the DataModule wiring works
    end-to-end against the real HDF5 dataset at ``/project/pedramh/h5data/h5data``.

    Returns:
        None: The commit gate keys on the printed ``SMOKE_OK`` token, not a
        return value (``sys.exit(main())`` therefore exits 0).
    """
    # Resolve the config relative to THIS file, not cwd, so the smoke works from
    # any directory in the monorepo (s2s/v2.0/ is a sibling of s2s-lightning/).
    config_path = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "s2s", "v2.0", "config", "test.yaml")
    )
    print(f"Loading config: {config_path} (section S2S)")
    params = YParams(config_path, "S2S")

    # Smoke-only override: load batches single-process so the one-batch pull is
    # fast and deterministic on a 4-CPU smoke node. This does NOT alter the
    # config file; production runs keep params.num_data_workers from the YAML.
    params["num_data_workers"] = 0

    print("Instantiating ClimateDataModule...")
    dm = ClimateDataModule(params)
    dm.setup("fit")

    assert dm.train_dataset is not None, "train_dataset (normalizer source) is None"
    assert dm.val_dataset is not None, "val_dataset is None"
    print(
        f"train_dataset len={len(dm.train_dataset)} "
        f"val_dataset len={len(dm.val_dataset)}"
    )

    # Normalizer-source sanity: the training dataset must expose the stats
    # buffers the LightningModule will consume in Phase 2/3.
    assert hasattr(dm.train_dataset, "constant_boundary_data")
    assert hasattr(dm.train_dataset, "land_mask")
    print(
        "normalizer buffers present: "
        f"constant_boundary_data shape={tuple(dm.train_dataset.constant_boundary_data.shape)} "
        f"land_mask shape={tuple(dm.train_dataset.land_mask.shape)}"
    )

    print("Pulling one training batch...")
    batch = next(iter(dm.train_dataloader()))
    print(f"train batch is a {type(batch).__name__} of {len(batch)} elements:")
    for i, elem in enumerate(batch):
        _describe(f"[{i}]", elem)

    print("SMOKE_OK")


if __name__ == "__main__":
    sys.exit(main())
