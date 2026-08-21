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

# YParams is REPLICATED, not imported: it needs `ruamel.yaml`, which the only
# working torch env (the ai-rossby venv) does not have — that cost attempt 4.
# The three lines that matter are YParams.py:18-23, read directly: load the named
# section, convert the string 'None' to Python None, expose as attributes. The
# conversion is what makes `factorization` None, and it is verified below.
import yaml                                                        # noqa: E402


class _P(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


yaml_path = sys.argv[1]
with open(os.path.abspath(yaml_path)) as fh:
    _sec = yaml.safe_load(fh)["SFNO"]
params = _P({k: (None if v == 'None' else v) for k, v in _sec.items()})
print("  (YParams replicated via PyYAML — ruamel is absent from this env)")
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
    """Stand-in for the dataset arg. The net reads exactly three attributes off
    it (sfnonet.py:762-766): variable_list_in, constant_boundary_variables and
    variable_list_out — used only to size in_chans/out_chans. §4.5a established
    those as 105 and 101, pinned by the logged 1,182,108,160 parameter total."""

    def __init__(self, p):
        n_upper = len(p['upper_air_variables']) * p['num_levels']
        self.variable_list_in = ['x'] * (n_upper + len(p['surface_variables'])
                                         + len(p['land_variables'])
                                         + len(p['varying_boundary_variables']))
        self.constant_boundary_variables = list(p['constant_boundary_variables'])
        self.variable_list_out = ['y'] * (n_upper + len(p['surface_variables'])
                                          + len(p['land_variables'])
                                          + len(p['diagnostic_variables']))
        print(f"  in_chans={len(self.variable_list_in) + len(self.constant_boundary_variables)}"
              f"  out_chans={len(self.variable_list_out)}  (§4.5a: 105 / 101)")


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
