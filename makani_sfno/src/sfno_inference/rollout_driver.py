"""rollout_driver — arbitrary-K SFNO rollout that mirrors validate_one_epoch.

Implements docs/sfno_eval_plan.md §B.1-§B.3. Two public entry points:

  - :func:`rollout_one_ic` — run a single rollout from one IC. Returns
    a :class:`RolloutResult` with predictions and truth in **physical
    units** plus the IC tensor.
  - :func:`run_one_file` — orchestrate rollouts over all configured ICs
    in one h5 file (NWP mode: 12 ICs at monthly stride; climate mode:
    1 IC at sample 0 with K = n_samples - 1) and write a NetCDF per IC.

The rollout body is a near-line-for-line copy of
``validate_one_epoch`` (deterministic_trainer.py:617-661) so any
behavioural drift between training validation and eval rollout shows up
as a code-diff against that reference.

The 58→53 channel contract is enforced inside the loop in **debug
mode** (``params.assert_contract: bool = True``). Production runs can
flip to False for speed; the smoke test always runs with assertions on.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# data classes
# ---------------------------------------------------------------------------

@dataclass
class RolloutResult:
    """One IC's worth of predictions, truth, and provenance.

    All tensors are in **physical units** (de-z-scored using run-dir
    stats) and on CPU.

    Shapes:
      - ``prediction`` : ``(K, n_out, H, W)``     where ``n_out = 53``
      - ``truth``      : ``(K, n_out, H, W)``
      - ``init_state`` : ``(n_state, H, W)``      where ``n_state = 52``
    """

    prediction: torch.Tensor
    truth: torch.Tensor
    init_state: torch.Tensor
    K: int
    ic_global_idx: int
    ic_sample_idx: int
    ic_file: str
    file_anchor: str  # e.g. '0126-08-01 00:00:00'
    time_plasim_at_ic: float
    rollout_mode: str  # 'nwp' or 'climate'
    truth_sic: torch.Tensor | None = None  # (K, H, W) raw fraction; NaN over land

    def to_dict(self) -> dict:
        return {
            "prediction": self.prediction,
            "truth": self.truth,
            "init_state": self.init_state,
            "truth_sic": self.truth_sic,
            "K": self.K,
            "ic_global_idx": self.ic_global_idx,
            "ic_sample_idx": self.ic_sample_idx,
            "ic_file": self.ic_file,
            "file_anchor": self.file_anchor,
            "time_plasim_at_ic": self.time_plasim_at_ic,
            "rollout_mode": self.rollout_mode,
        }


# ---------------------------------------------------------------------------
# core single-IC rollout
# ---------------------------------------------------------------------------

def _load_run_norm_stats(eval_params, device, *, n_out: int = 53):
    """Load run-dir global_means / global_stds and shape for broadcast.

    Returns ``(out_bias, out_scale)`` each shaped ``(1, n_out, 1, 1)``
    and on ``device`` ready to broadcast against ``(K, 53, H, W)``.
    """
    out_bias_np = np.load(eval_params.global_means_path).astype(np.float32)
    out_scale_np = np.load(eval_params.global_stds_path).astype(np.float32)
    # The stats files on disk are saved as (1, n_out, 1, 1) by the data
    # packager (verified for plasim_sim52_full: both run-dir and dataset-stats
    # copies are (1, 53, 1, 1)). The plan §B.2 documented (53,) but reality
    # is broadcast-shaped — accept either and reshape canonically.
    accepted = ((n_out,), (1, n_out, 1, 1))
    if out_bias_np.shape not in accepted:
        raise RuntimeError(
            f"unexpected global_means shape {out_bias_np.shape}; expected one of {accepted}"
        )
    if out_scale_np.shape not in accepted:
        raise RuntimeError(
            f"unexpected global_stds shape {out_scale_np.shape}; expected one of {accepted}"
        )
    out_bias = torch.from_numpy(out_bias_np).to(device).reshape(1, n_out, 1, 1)
    out_scale = torch.from_numpy(out_scale_np).to(device).reshape(1, n_out, 1, 1)
    return out_bias, out_scale


def rollout_one_ic(
    *,
    wrapper,
    dataset,
    ic_global_idx: int,
    eval_params,
    device,
    out_bias: torch.Tensor | None = None,
    out_scale: torch.Tensor | None = None,
    assert_contract: bool = True,
) -> RolloutResult:
    """Run one K-step rollout from ``ic_global_idx`` and return physical-unit tensors.

    The autoregressive body mirrors ``validate_one_epoch`` exactly.

    Parameters
    ----------
    wrapper : nn.Module
        ``PlasimMultiStepWrapper`` from
        :func:`sfno_inference.checkpoint_loader.build_wrapper_from_checkpoint`.
        Must have ``.preprocessor`` (the canonical instance owned by the
        wrapper — do NOT pass a separate preprocessor).
    dataset : PlasimForcingDataset
        From ``_plasim_get_dataloader(eval_params, ..., mode='eval')``.
    ic_global_idx : int
        Sample index into the dataset (NOT the file-local index — for
        the canonical 1-file dataset they coincide).
    eval_params : ParamsBase
        From :func:`sfno_inference.checkpoint_loader.load_eval_params`.
    device : torch.device | str
        Where to run.
    out_bias, out_scale : torch.Tensor, optional
        Pre-loaded normalization tensors, each ``(1, 53, 1, 1)`` on
        ``device``. If ``None``, loaded from ``eval_params``. Pass
        explicitly when running many ICs back-to-back.
    assert_contract : bool, default True
        Toggle the 58→53 channel-contract guards in the loop. Off for
        production wallclock-sensitive runs; on for tests and the smoke
        run.
    """
    K = int(eval_params.valid_autoreg_steps) + 1
    if K < 1:
        raise ValueError(f"derived K={K} from valid_autoreg_steps; must be >= 1")

    if dataset.n_future != K - 1:
        raise RuntimeError(
            f"dataset.n_future={dataset.n_future} but expected {K - 1}; "
            "did valid_autoreg_steps fail to propagate via _plasim_get_dataloader?"
        )

    if out_bias is None or out_scale is None:
        out_bias, out_scale = _load_run_norm_stats(eval_params, device)

    preprocessor = wrapper.preprocessor  # share the wrapper's instance
    n_state = int(getattr(eval_params, "n_state_channels", 52))
    n_out = int(getattr(eval_params, "N_out_channels", 53))

    # Pull one sample. PlasimForcingDataset returns 4 tensors:
    #   inp_state    (n_history+1, 52, H, W)  z-scored
    #   tar          (n_future+1, 53, H, W)  z-scored
    #   inp_forcing  (n_history+1, 6,  H, W)  z-scored
    #   tar_forcing  (n_future+1, 6,  H, W)  z-scored
    sample = dataset[ic_global_idx]
    if not (isinstance(sample, tuple) and len(sample) == 4):
        raise RuntimeError(
            f"PlasimForcingDataset.get_sample_at_index returned {type(sample).__name__} "
            f"of length {len(sample) if hasattr(sample, '__len__') else '?'}; "
            "expected a 4-tuple (inp_state, tar, inp_forcing, tar_forcing)"
        )
    inp_state, tar, inp_forcing, tar_forcing = sample

    # Sanity-check shapes BEFORE moving to device — cheaper to fail here.
    if tar.shape[0] != K:
        raise RuntimeError(
            f"tar T-axis = {tar.shape[0]}, expected K={K}; "
            f"valid_autoreg_steps={eval_params.valid_autoreg_steps}"
        )

    H, W = inp_state.shape[-2], inp_state.shape[-1]

    # Add batch dim and move to device.
    gdata = tuple(
        t.unsqueeze(0).to(device)
        for t in (inp_state, tar, inp_forcing, tar_forcing)
    )

    # === BEGIN: copy of validate_one_epoch body (deterministic_trainer.py:617-661) ===

    # cache_unpredicted_features stores forcing in the preprocessor and returns
    # only the state-pair (inp_state_z, tar_z).
    inp, tar_z = preprocessor.cache_unpredicted_features(*gdata)
    inp = preprocessor.flatten_history(inp)  # (1, 52, H, W) at n_history=0

    if assert_contract:
        assert inp.shape == (1, n_state, H, W), (
            f"flattened input shape {tuple(inp.shape)} != (1, {n_state}, {H}, {W})"
        )

    tarlist = torch.split(tar_z, 1, dim=1)  # K-tuple of (1, 1, 53, H, W)

    autocast_enabled = bool(eval_params.amp_enabled) and (torch.device(device).type == "cuda")
    autocast_dtype = eval_params.amp_dtype if autocast_enabled else torch.float32

    predictions: list[torch.Tensor] = []
    inpt = inp
    for idt, targ in enumerate(tarlist):
        targ = preprocessor.flatten_history(targ)  # (1, 53, H, W)

        with torch.inference_mode(), torch.amp.autocast(
            device_type=torch.device(device).type,
            enabled=autocast_enabled,
            dtype=autocast_dtype,
        ):
            pred = wrapper(inpt)  # _forward_eval → single-step forward → (1, 53, H, W)

        if assert_contract:
            assert pred.shape == (1, n_out, H, W), (
                f"step {idt}: pred shape {tuple(pred.shape)} != (1, {n_out}, {H}, {W})"
            )

        # Cast to fp32 before stashing so downstream math (RMSE/ACC) is
        # bit-stable across runs with bf16 autocast.
        predictions.append(pred.detach().to(torch.float32).clone())

        # PlasimPreprocessor.append_history (preprocessor.py:39) asserts
        # x2 has 53 channels and slices to first 52 internally. Driver
        # does NOT do the slice itself.
        inpt = preprocessor.append_history(inpt, pred, idt)

        if assert_contract:
            assert inpt.shape == (1, n_state, H, W), (
                f"step {idt}: post-append_history inpt shape "
                f"{tuple(inpt.shape)} != (1, {n_state}, {H}, {W})"
            )
    # === END: validate_one_epoch body ===

    predictions_z = torch.cat(predictions, dim=0)        # (K, 53, H, W) z-scored, fp32
    predictions_phys = predictions_z * out_scale + out_bias

    # Truth: also de-z-score. tar_z is (1, K, 53, H, W); squeeze batch.
    truth_z = tar_z.squeeze(0).to(torch.float32)         # (K, 53, H, W)
    truth_phys = truth_z * out_scale + out_bias

    # IC: de-z-score the (1, 1, 52, H, W) inp_state. The preprocessor
    # stored the original (z-scored) inp_state in `inp`; we use that
    # post-flatten value here — same float content, just batched.
    in_bias = torch.from_numpy(np.asarray(dataset.in_bias).astype(np.float32)).to(device)
    in_scale = torch.from_numpy(np.asarray(dataset.in_scale).astype(np.float32)).to(device)
    in_bias = in_bias.reshape(1, n_state, 1, 1)
    in_scale = in_scale.reshape(1, n_state, 1, 1)
    init_state_phys = (inp.to(torch.float32) * in_scale + in_bias).squeeze(0)  # (52, H, W)

    # Provenance: locate the file and sample-within-file for this global idx.
    ic_file, ic_sample_idx, file_anchor, t_plasim = _resolve_ic_provenance(dataset, ic_global_idx)

    truth_sic = _extract_truth_sic(tar_forcing, dataset)

    return RolloutResult(
        prediction=predictions_phys.cpu(),
        truth=truth_phys.cpu(),
        init_state=init_state_phys.cpu(),
        K=K,
        ic_global_idx=ic_global_idx,
        ic_sample_idx=ic_sample_idx,
        ic_file=ic_file,
        file_anchor=file_anchor,
        time_plasim_at_ic=float(t_plasim),
        rollout_mode="",  # caller sets this
        truth_sic=truth_sic,
    )


def _extract_truth_sic(tar_forcing: torch.Tensor, dataset) -> torch.Tensor | None:
    """Recover physical sic at each lead from the CPU `tar_forcing` tensor.

    `tar_forcing` is `(K, 6, H, W)` z-scored on CPU, with channel order
    `['lsm','sg','z0','sst','rsdt','sic']`. Inverse-transform channel 5
    using the dataset's loaded forcing stats. NaN over land (per
    packager.py:226) round-trips as NaN. Returns None if stats are
    missing or mis-shaped; the writer then skips truth_sic.
    """
    try:
        fb = np.asarray(dataset.forcing_bias, dtype=np.float32).reshape(-1)
        fs = np.asarray(dataset.forcing_scale, dtype=np.float32).reshape(-1)
    except (AttributeError, TypeError, ValueError):
        logger.warning("dataset has no forcing_bias/forcing_scale; truth_sic disabled")
        return None
    if fb.shape[0] < 6 or fs.shape[0] < 6:
        logger.warning("forcing stats too short (%d, %d); truth_sic disabled",
                       fb.shape[0], fs.shape[0])
        return None
    sic_z = tar_forcing[:, 5, :, :].to(torch.float32)        # (K, H, W) CPU
    sic_phys = sic_z * float(fs[5]) + float(fb[5])
    return sic_phys


# ---------------------------------------------------------------------------
# IC selection (§A.4)
# ---------------------------------------------------------------------------

def nwp_ic_offsets(n_samples: int, K: int = 56, n_ic: int = 12) -> list[int]:
    """Return ``n_ic`` IC sample indices that fit within ``[0, n_samples - K - 1]``.

    Spacing is ``(n_samples - K) // n_ic`` ≈ monthly cadence (see
    docs/sfno_eval_plan.md §A.4). Cross-file rollout is **explicitly out
    of scope** because the dataset's ``_get_indices`` would silently
    return samples from the next file, breaking ``time_plasim``-based
    provenance.
    """
    if n_samples <= K + n_ic:
        raise ValueError(
            f"n_samples={n_samples} too small for K={K}, n_ic={n_ic}"
        )
    step = (n_samples - K) // n_ic
    offsets = [i * step for i in range(n_ic)]
    for s in offsets:
        if s + K >= n_samples:
            raise RuntimeError(
                f"IC offset {s} + K={K} = {s + K} would cross file boundary at "
                f"n_samples={n_samples}; cross-file rollout is not supported"
            )
    return offsets


# ---------------------------------------------------------------------------
# IC provenance resolution
# ---------------------------------------------------------------------------

def _resolve_ic_provenance(dataset, ic_global_idx: int):
    """Map a global dataset idx to (file_name, sample_idx, anchor_str, time_plasim)."""
    file_idx, local_idx = dataset._get_indices(ic_global_idx)
    file_path = Path(dataset.files_paths[file_idx])
    # Re-open just the metadata; the dataset already has h5 handles open
    # but exposing them through the _state_files private attr is fragile.
    # The h5 attrs and time_plasim dataset are tiny so this re-read is
    # cheap (and it keeps the dataset class as a black box).
    import h5py
    with h5py.File(file_path, "r") as f:
        anchor = f.attrs.get("plasim_time_units", "")
        if isinstance(anchor, bytes):
            anchor = anchor.decode("utf-8")
        # Strip leading "days since "
        if anchor.startswith("days since "):
            anchor = anchor[len("days since "):]
        t_plasim = float(f["time_plasim"][local_idx])
    return file_path.name, int(local_idx), anchor, t_plasim
