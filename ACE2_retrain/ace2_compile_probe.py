"""Apply torch.compile to ONE region of fme, then run fme.ace.train unchanged.

Why monkeypatch instead of editing ACE2_retrain/ace_exp: that tree is a git
subtree, so edits can conflict on a future subtree pull, and the committed
baseline must stay untouched while we measure. DESIGN 5 allows measuring a rung
freely; adopting one is what the DESIGN 4 equivalence gate governs. Nothing here
is intended to be adopted as-is.

Targets, chosen from the 8x H200 profile (elementwise/copy = 47.9% of GPU time,
of which 58% is copies that fusion CANNOT remove -- so the reachable prize is
the ~20% that is add/unary/fill):

  none        control
  normalizer  fme.core.normalizer._normalize/_denormalize -- a dict
              comprehension over ~50 named variables, ~2 tiny kernels each
  network     the SFNO module itself (MLP / activation / instance_norm)
  corrector   AtmosphereCorrector.__call__ -- conserve_dry_air, moisture budget,
              force_positive over 16 variables, all pointwise
  all         all three

Env: ACE2_COMPILE_TARGET (above), ACE2_COMPILE_MODE (torch.compile mode).
"""

import logging
import os
import runpy
import sys

import torch

TARGET = os.environ.get("ACE2_COMPILE_TARGET", "none")
MODE = os.environ.get("ACE2_COMPILE_MODE", "default")
_applied = []


def _compile(fn, label):
    _applied.append(label)
    return torch.compile(fn, mode=MODE, dynamic=False)


if TARGET in ("normalizer", "all"):
    import fme.core.normalizer as _norm

    # Module-attribute patch, not a rebind of the callers: normalizer.normalize()
    # looks _normalize up on the module at call time, so this takes effect.
    _norm._normalize = _compile(_norm._normalize, "normalizer._normalize")
    _norm._denormalize = _compile(_norm._denormalize, "normalizer._denormalize")

if TARGET in ("network", "all"):
    from fme.core.registry.module import ModuleSelector

    _orig_build = ModuleSelector.build

    def _build_compiled(self, *args, **kwargs):
        module = _orig_build(self, *args, **kwargs)
        _applied.append("network(SFNO module)")
        return torch.compile(module, mode=MODE)

    ModuleSelector.build = _build_compiled

if TARGET in ("corrector", "all"):
    from fme.core.corrector.atmosphere import AtmosphereCorrector

    # __call__ is looked up on the type, so patching the class attribute works.
    AtmosphereCorrector.__call__ = _compile(
        AtmosphereCorrector.__call__, "AtmosphereCorrector.__call__"
    )

logging.basicConfig(level=logging.INFO)
print(f"ACE2_COMPILE_TARGET={TARGET} mode={MODE} patched={_applied or ['<none>']}", flush=True)

# argv is already (config, --override ...) as fme.ace.train expects.
runpy.run_module("fme.ace.train", run_name="__main__")
