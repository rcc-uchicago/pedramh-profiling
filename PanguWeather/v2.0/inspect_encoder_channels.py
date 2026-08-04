#!/usr/bin/env python
"""Per-input-channel diagnostics for a trained PanguWeather SFNO checkpoint.

Answers "did the model actually use input channel X?" without a GPU or a forward
pass, by reading the encoder's 1x1 convolution:

* **weight norm** ``||W[:, c]||`` — the channel's learned gain. Near zero means
  the channel was effectively pruned; much larger than its peers means the model
  amplified it.
* **cosine similarity** between channels' 512-dim weight vectors — whether the
  model reads two channels along the same direction (redundant) or different
  ones. Compared against the ``1/sqrt(d_model)`` noise floor for random vectors.

Motivated by the SST-fill investigation: SST is degC but was NaN-filled with 270,
compressing the ocean's variation to 0.093 sigma of the channel. See the
2026-08-04 CHANGELOG entry.

⚠ **What this CANNOT tell you.** Normalization makes the whole channel unit
variance, so an input like SST arrives with std 1.0 regardless — the 0.093 sigma
is only its *ocean* part, the rest being the static land/ocean step. One weight
vector serves both, so a normal-looking norm does NOT prove the ocean signal was
learned. Settling that needs a forward-pass perturbation probe on a GPU.

Usage::

    python inspect_encoder_channels.py --checkpoint <run>/checkpoints/best_ckpt.tar
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch

# Channel order into the encoder, from
# networks/modulus_sfno/sfnonet.py::SphericalFourierNeuralOperatorNet_v2.forward:
#   x = cat((surface, c_boundary, v_boundary, upper_air), dim=1)
# `surface` is surface_variables + land_variables + ocean_variables.
E3SM_NAMED_CHANNELS = (
    ["TREFHT", "U10", "RHREFHT", "PS", "PSL", "TMQ"]      # surface (6)
    + ["SOILWATER_10CM", "TSOI_10CM"]                      # land (2)
    + ["PCT_GLACIER", "PFTDATA_MASK", "PCT_NATVEG", "TOPO"]  # constant boundary (4)
    + ["SST", "ICE", "sol_in"]                             # varying boundary (3)
)                                                          # + 90 upper-air = 105


def _stub_ruamel() -> None:
    """Let the unpickler resolve the ruamel config object the checkpoint carries.

    The trainer pickles its YParams alongside the tensors. We only want tensors,
    and installing ruamel just to read weights is not worth it.
    """
    class _Any:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, s):
            self.__dict__.update(s if isinstance(s, dict) else {})

    for name in (
        "ruamel", "ruamel.yaml", "ruamel.yaml.comments", "ruamel.yaml.scalarstring",
        "ruamel.yaml.compat", "ruamel.yaml.nodes", "ruamel.yaml.scalarfloat",
        "ruamel.yaml.scalarint",
    ):
        mod = types.ModuleType(name)
        mod.__getattr__ = lambda _n: _Any        # noqa: B023 — intentional catch-all
        mod.__path__ = []
        sys.modules[name] = mod


def encoder_weight(state: dict) -> torch.Tensor:
    key = next(k for k in state if k.endswith("encoder.0.weight"))
    return state[key].float()[:, :, 0, 0]        # (d_model, n_channels)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--state", default="model_state", choices=["model_state", "ema_state"])
    p.add_argument("--focus", default="SST", help="Channel to report cosine similarities for.")
    a = p.parse_args(argv)

    _stub_ruamel()
    # mmap: these checkpoints are ~19 GB (weights + optimizer + EMA).
    ck = torch.load(a.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    W = encoder_weight(ck[a.state])
    d_model, n_ch = W.shape
    names = list(E3SM_NAMED_CHANNELS)
    n_named = len(names)
    if n_ch < n_named:
        print(f"ERROR CHANNEL_COUNT {n_ch} < {n_named} named — wrong variable set?")
        return 1

    norms = W.norm(dim=0).numpy()
    median = float(np.median(norms))
    upper = norms[n_named:]

    print(f"=== {a.checkpoint.name} [{a.state}] epoch {ck.get('epoch')} ===")
    print(f"encoder.0.weight: d_model={d_model}, channels={n_ch} "
          f"({n_named} named + {n_ch - n_named} upper-air)\n")
    print(f"  {'ch':>3} {'name':16s} {'||W||':>8} {'vs median':>10}")
    for i, nm in enumerate(names):
        print(f"  {i:3d} {nm:16s} {norms[i]:8.4f} {norms[i] / median:9.2f}x")
    print(f"  {'':3} {'upper-air (median)':16s} {np.median(upper):8.4f} "
          f"{np.median(upper) / median:9.2f}x")
    print(f"\n  all-channel median {median:.4f}")

    if a.focus in names:
        f = names.index(a.focus)
        Wn = (W / W.norm(dim=0, keepdim=True)).numpy()
        cos = Wn.T @ Wn
        noise = 1.0 / np.sqrt(d_model)
        rank = int(np.argsort(-norms).tolist().index(f)) + 1
        print(f"\n  {a.focus}: rank {rank} of {n_ch} by weight norm")
        print(f"\n  cosine similarity vs the other named channels "
              f"(noise floor +/-{noise:.3f}; |cos| > {3 * noise:.3f} is 3 sigma):")
        for j in np.argsort(-np.abs(cos[f][:n_named])):
            if j == f:
                continue
            s = cos[f, j]
            print(f"    {names[j]:16s} {s:+.3f}  ({abs(s) / noise:4.1f} sigma)"
                  f"{' **' if abs(s) > 3 * noise else ''}")
        ua = cos[f, n_named:]
        print(f"    {'upper-air (90)':16s} mean |cos| {np.abs(ua).mean():.3f}, "
              f"max {np.abs(ua).max():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
