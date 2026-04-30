"""checkpoint_loader — load eval params + build a PlasimMultiStepWrapper.

Implements docs/sfno_eval_plan.md §B.0. The two public helpers are:

  - :func:`load_eval_params` — read ``runs/.../config.json`` (the only
    authoritative copy of the model-shape parameters), assert the
    58→53 channel contract, override the eval-only fields (horizon,
    paths, sharding, batch size), and derive ``amp_enabled`` /
    ``amp_dtype`` from the serialised ``amp_mode`` (matching the
    trainer's runtime derivation).
  - :func:`build_wrapper_from_checkpoint` — install the PlaSim patches,
    build a ``PlasimMultiStepWrapper`` via ``model_registry.get_model``,
    move it to the requested device, restore weights with
    ``Driver.restore_from_checkpoint``, and assert the wrapper landed
    on the right device with a safe type+index comparison.

The corresponding eval-time *normalization* stats (``global_means.npy`` /
``global_stds.npy``) are loaded by :mod:`sfno_inference.rollout_driver`,
not here — see §B.2.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# load_eval_params
# ---------------------------------------------------------------------------

# The trainer pickles a ParamsBase via ``params.to_dict()`` so any string
# stored as 'None' should round-trip back to a Python None — matching
# ParamsBase.update_params' behavior.
_NONE_STR = "None"


def load_eval_params(run_dir, *, K: int):
    """Load the run-dir ``config.json`` and apply eval-only overrides.

    Parameters
    ----------
    run_dir : str | Path
        Path to a Makani run directory (must contain ``config.json``,
        ``global_means.npy``, ``global_stds.npy``).
    K : int
        Rollout horizon in 6 h steps (number of predictions to produce
        per IC). For NWP scoring use ``K=56`` (= 14 days). For climate
        rollout use ``K = n_samples_in_file - 1`` per file.

    Returns
    -------
    eval_params : ParamsBase
        Populated with channel-count assertions enforced and AMP fields
        derived from ``amp_mode``.
    """
    from makani.utils.YParams import ParamsBase

    run_dir = Path(run_dir)
    cfg_path = run_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing config.json in run dir: {cfg_path}")

    eval_params = ParamsBase.from_json(str(cfg_path))

    # Hard contract — these are the values the metadata-time patch
    # installed at training; if they ever drift, abort before model build.
    assert eval_params.N_in_channels == 58, (
        f"expected 58 (52 state + 6 forcing); got {eval_params.N_in_channels}"
    )
    assert eval_params.N_out_channels == 53, (
        f"expected 53 (52 state + 1 diagnostic); got {eval_params.N_out_channels}"
    )
    assert eval_params.n_state_channels == 52
    assert eval_params.n_diagnostic_channels == 1
    assert eval_params.n_forcing_channels == 6

    # Eval-only overrides.
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    eval_params.valid_autoreg_steps = K - 1   # ACTIVE handle in eval mode (plasim_trainer.py:103-105)
    eval_params.n_future = K - 1              # set redundantly for any code that reads it directly
    eval_params.n_history = 0
    eval_params.data_num_shards = 1
    eval_params.data_shard_id = 0
    eval_params.batch_size = 1

    # Pin normalization to the run dir, NOT the dataset-stats dir.
    # Q5 (Codex round 6) verified the two are byte-identical for this run,
    # but run-dir is the canonical source so future per-checkpoint overrides
    # remain self-consistent. We compare-and-warn below.
    run_means = run_dir / "global_means.npy"
    run_stds = run_dir / "global_stds.npy"
    if not run_means.is_file() or not run_stds.is_file():
        raise FileNotFoundError(
            f"run dir is missing normalization stats: {run_means}, {run_stds}"
        )

    dataset_means = getattr(eval_params, "global_means_path", None)
    dataset_stds = getattr(eval_params, "global_stds_path", None)
    if dataset_means and dataset_stds:
        _warn_if_stats_diverge(run_means, Path(dataset_means))
        _warn_if_stats_diverge(run_stds, Path(dataset_stds))

    eval_params.global_means_path = str(run_means)
    eval_params.global_stds_path = str(run_stds)
    # forcing_global_means_path / forcing_global_stds_path are inherited
    # from cfg (the dataset-stats dir; the training run did not produce a
    # run-dir copy).

    # AMP — config.json stores only `amp_mode`. `amp_enabled` and `amp_dtype`
    # are derived in the trainer at runtime (deterministic_trainer.py:84-97).
    amp_mode = getattr(eval_params, "amp_mode", "none")
    if amp_mode == "none":
        eval_params.amp_enabled = False
        eval_params.amp_dtype = torch.float32
    elif amp_mode == "fp16":
        eval_params.amp_enabled = True
        eval_params.amp_dtype = torch.float16
    elif amp_mode == "bf16":
        eval_params.amp_enabled = True
        eval_params.amp_dtype = torch.bfloat16
    else:
        raise ValueError(f"unknown amp_mode: {amp_mode!r}")

    return eval_params


def _warn_if_stats_diverge(run_path: Path, dataset_path: Path) -> None:
    """SHA256-compare two npy files; log a warning if they diverge."""
    if not dataset_path.is_file():
        # Dataset-stats path may be unreadable from this node; not fatal.
        return
    try:
        run_sha = _sha256(run_path)
        ds_sha = _sha256(dataset_path)
    except OSError as exc:
        logger.warning("could not compare stats files: %s", exc)
        return
    if run_sha != ds_sha:
        logger.warning(
            "Normalization stats diverge between run-dir and dataset-stats:\n"
            "  run:     %s  sha256=%s\n"
            "  dataset: %s  sha256=%s\n"
            "Eval will use the run-dir copy (the model was trained against it).",
            run_path, run_sha, dataset_path, ds_sha,
        )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# build_wrapper_from_checkpoint
# ---------------------------------------------------------------------------

def build_wrapper_from_checkpoint(eval_params, ckpt_path, device):
    """Build PlasimMultiStepWrapper, restore weights, place on device.

    Mirrors the trainer's model-build path (deterministic_trainer.py:132)::

        self.model = model_registry.get_model(self.params, multistep=self.multistep).to(self.device)

    then calls ``Driver.restore_from_checkpoint`` to load weights.

    Parameters
    ----------
    eval_params : ParamsBase
        From :func:`load_eval_params`.
    ckpt_path : str | Path
        Path to ``best_ckpt_mp0.tar`` (or any legacy-format checkpoint).
        ``Driver._restore_checkpoint_legacy`` formats this with
        ``mp_rank=comm.get_rank("model")`` so for our 1-rank setup it is
        used as-is.
    device : torch.device | str
        Where to place the model. For GPU runs prefer an explicit index
        (``f"cuda:{torch.cuda.current_device()}"``) so the post-build
        device assertion is unambiguous.

    Returns
    -------
    wrapper : nn.Module
        ``PlasimMultiStepWrapper`` in ``eval()`` mode with weights
        restored and parameters on ``device``.
    """
    from makani.models import model_registry
    from makani.utils.driver import Driver
    from sfno_training.trainer.plasim_trainer import _install_plasim_patches

    _ensure_comm_initialized(eval_params)
    _install_plasim_patches()

    wrapper = model_registry.get_model(eval_params, multistep=True).to(device)

    # Driver.restore_from_checkpoint signature
    # (makani/utils/driver.py:347-356):
    #   restore_from_checkpoint(checkpoint_path, model, loss=None,
    #       optimizer=None, scheduler=None, counters=None,
    #       checkpoint_mode='legacy', strict=True)
    # @staticmethod, model passed second.
    Driver.restore_from_checkpoint(str(ckpt_path), wrapper, checkpoint_mode="legacy")
    wrapper.eval()

    # Post-build assertions on the actual SFNO module. SFNO uses
    # `inp_chans` / `out_chans` (sfnonet.py:298-299), NOT `in_chans`.
    assert wrapper.model.inp_chans == 58, (
        f"wrapper.model.inp_chans={wrapper.model.inp_chans}, expected 58"
    )
    assert wrapper.model.out_chans == 53, (
        f"wrapper.model.out_chans={wrapper.model.out_chans}, expected 53"
    )

    _assert_on_device(wrapper, device)
    return wrapper


def _assert_on_device(wrapper: torch.nn.Module, device) -> None:
    """Compare wrapper's parameter device to ``device`` safely.

    ``torch.device("cuda")`` (no index) does NOT compare equal to
    ``torch.device("cuda:0")`` under ``==`` even though ``.to("cuda")``
    resolves to ``cuda:<current_device>``. The comparison here:
      - matches ``type`` exactly (cpu vs cuda);
      - accepts a missing index on the *expected* side (treats it as a
        wildcard against any index of the same type).
    """
    actual = next(wrapper.parameters()).device
    expected = torch.device(device) if not isinstance(device, torch.device) else device
    assert actual.type == expected.type and (
        expected.index is None or actual.index == expected.index
    ), f"wrapper on {actual}, expected {expected}"


def _ensure_comm_initialized(eval_params) -> None:
    """Initialise Makani's comm subsystem for a 1-rank eval if needed.

    Makani's ``Driver._restore_checkpoint_legacy`` calls
    ``comm.get_rank("model")``; for our 1-rank setup the resulting
    ``mp_rank=0`` matches ``best_ckpt_mp0.tar``. This helper makes the
    eval driver runnable both inside SLURM (where the launcher already
    set MASTER_ADDR/PORT/WORLD_SIZE/RANK and called ``comm.init``) and
    interactively (where it has not).
    """
    try:
        from makani.utils import comm
    except ImportError as exc:  # pragma: no cover — Makani is a hard dep
        raise RuntimeError("makani is not importable") from exc

    # If already initialised (e.g., SLURM job already called comm.init),
    # do nothing.
    if getattr(comm, "_DM", None) is not None:
        return

    sizes = list(getattr(eval_params, "model_parallel_sizes", [1, 1, 1, 1]))
    names = list(getattr(eval_params, "model_parallel_names", ["h", "w", "fin", "fout"]))
    comm.init(model_parallel_sizes=sizes, model_parallel_names=names, verbose=False)
