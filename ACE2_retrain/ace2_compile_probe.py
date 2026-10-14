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
  mlp         the SFNO's real-valued MLP blocks (the spectral path CANNOT be
              compiled -- Inductor has no complex64 lowering)
  safe        corrector + mlp: everything that compiled without breaking or
              moving numerics more than run-to-run noise
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
    # Patch AFTER SingleModuleStep has built and .to(device)'d the module, not
    # at ModuleSelector.build. Wrapping the builder's return value made the
    # subsequent `module.to(get_device())` fail with
    #   AttributeError: 'function' object has no attribute 'to'
    # (job 53525035) -- the builder's return does not survive being handed to
    # torch.compile at that point. Wrapping the finished attribute avoids the
    # question entirely and is also closer to what a real adoption would do.
    import fme.core.step.single_module as _sm

    _orig_init = _sm.SingleModuleStep.__init__

    def _init_compiled(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        # self.module is fme's OWN wrapper (fme.core.registry.module.Module),
        # which is NOT an nn.Module -- it holds the real network in `_module`
        # alongside a label encoding. Compiling the wrapper is what broke job
        # 53525035: torch.compile() on a non-nn.Module callable returns a plain
        # function, so the caller's `module.to(get_device())` then failed with
        # AttributeError: 'function' object has no attribute 'to'.
        inner = getattr(self.module, "_module", None)
        if isinstance(inner, torch.nn.Module):
            self.module._module = torch.compile(inner, mode=MODE)
            _applied.append(f"network({type(inner).__name__})")
            print(f"ACE2_COMPILE compiled inner network {type(inner).__name__}", flush=True)
        else:
            print(
                f"ACE2_COMPILE WARNING: no inner nn.Module on {type(self.module)} "
                f"(got {type(inner)}) -- left uncompiled",
                flush=True,
            )

    _sm.SingleModuleStep.__init__ = _init_compiled

if TARGET in ("mlp", "safe"):
    # The ONLY part of the SFNO that Inductor can legally take. Compiling the
    # whole network fails with InductorError: KeyError: 'complex64' (jobs
    # 53525182/183) -- the spherical harmonic transform path is complex-valued
    # and Inductor has no lowering for complex64. The MLP blocks are real-valued
    # and, with use_mlp: true in this config, are live in every block.
    import fme.ace.models.modulus.layers as _layers

    _layers.MLP.forward = _compile(_layers.MLP.forward, "SFNO MLP.forward")

if TARGET in ("corrector", "all", "safe"):
    from fme.core.corrector.atmosphere import AtmosphereCorrector

    # __call__ is looked up on the type, so patching the class attribute works.
    AtmosphereCorrector.__call__ = _compile(
        AtmosphereCorrector.__call__, "AtmosphereCorrector.__call__"
    )

logging.basicConfig(level=logging.INFO)
print(f"ACE2_COMPILE_TARGET={TARGET} mode={MODE} patched={_applied or ['<none>']}", flush=True)

# argv is already (config, --override ...) as fme.ace.train expects.
runpy.run_module("fme.ace.train", run_name="__main__")
