"""Settle the §4.5c OPEN question: is the SFNO spectral weight an nn.Parameter
or a FactorizedTensor — and why doesn't the assert fire?

Source reading says it MUST fire: `--config=SFNO` -> YParams converts the YAML
string 'None' to Python None (utils/YParams.py:20) -> sfnonet.py:442 sets
self.factorization = None -> the block passes it through (sfnonet.py:175) ->
SpectralFilterLayer sets `use_tensorly = False if factorization is None else True`
(sfnonet.py:115) -> SpectralConvS2 takes the `else` branch at
s2convolutions.py:150 -> `assert factorization == "ComplexDense"` -> raises.

But jobs 7255503/7255557/7366939/7366940 all ran 40+ steps. So a premise in that
chain is wrong. This prints each link so we can see WHICH one, instead of
guessing. CPU only, no training, no GPU needed.
"""
import os
import sys

print(f"python: {sys.executable}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '<unset>')}")

from utils.YParams import YParams                                  # noqa: E402

yaml_path = sys.argv[1]
params = YParams(os.path.abspath(yaml_path), "SFNO")
f = getattr(params, "factorization", "<ATTR MISSING>")
print(f"\n1. params.factorization = {f!r}   type={type(f).__name__}")
print(f"   is None? {f is None}")
print(f"   => use_tensorly would be: {False if f is None else True}")
for k in ("filter_type", "operator_type", "separable", "rank", "embed_dim",
          "num_layers", "hard_thresholding_fraction", "num_blocks"):
    print(f"   {k} = {getattr(params, k, '<missing>')!r}")

print("\n2. are asserts active in this interpreter?")
try:
    assert False
    print("   NO — asserts are DISABLED (-O / PYTHONOPTIMIZE). That alone would "
          "explain it.")
except AssertionError:
    print("   YES — asserts are active, so the assert would really raise.")

print("\n3. build the real net and inspect the actual weight object:")
import torch                                                        # noqa: E402
from networks.modulus_sfno.sfnonet import (                         # noqa: E402
    SphericalFourierNeuralOperatorNet_v2 as SFNO,
)


class _DS:
    """Minimal stand-in for the dataset arg the net's __init__ takes."""
    def __init__(self, p):
        self.img_shape = tuple(p.horizontal_resolution)


try:
    with torch.device("meta"):          # no memory, no GPU
        net = SFNO(params, _DS(params))
    print("   net constructed on the meta device — the assert did NOT fire.")
except AssertionError as exc:
    print(f"   AssertionError: {exc!r}")
    print("   ⇒ the source reading is right and the running jobs must differ "
          "some other way.")
    raise SystemExit(0)
except Exception as exc:                                            # noqa: BLE001
    print(f"   {type(exc).__name__}: {exc}")
    print("   (construction failed for an unrelated reason — see the traceback "
          "below; the factorization question is still answered by 1 and 2.)")
    import traceback
    traceback.print_exc()
    raise SystemExit(0)

blk = net.blocks[0]
w = blk.filter.filter.weight
print(f"\n4. weight type   : {type(w).__module__}.{type(w).__name__}")
print(f"   is nn.Parameter: {isinstance(w, torch.nn.Parameter)}")
print(f"   shape          : {tuple(w.shape)}")
n = 1
for d in w.shape:
    n *= d
print(f"   elements       : {n:,}   (complex elements: {n // 2:,})")
print(f"   contiguous     : {w.is_contiguous()}   stride={w.stride()}")
print(f"\n   §4.5a predicted 512x512x180 complex = 47,185,920 complex elements.")
print(f"   MATCH: {n // 2 == 512 * 512 * 180}")
print("\nFACTORIZATION_PROBE_OK")
