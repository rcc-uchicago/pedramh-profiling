"""test_mixprec.py -- mixprec.py's rounding and adaptive rule. numpy part on any python >= 3.6 (login node); the torch
part runs when CUDA is available (the job's stage 0). Prints MIXPREC_TEST_OK [torch=...] last.

  python3 test_mixprec.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mixprec as MP                                                 # noqa: E402


def ref_round(x, bits):
    """Independent reference: round-half-even to `bits` significant bits by frexp / rint (normal float32 values)."""
    m, e = np.frexp(x.astype(np.float64))
    return (np.rint(m * 2.0 ** bits) / 2.0 ** bits * 2.0 ** e.astype(np.float64)).astype(np.float32)


def test_numpy():
    rs = np.random.RandomState(0)
    # bf16: 8 significant bits, float32's range; ties to even
    x = (rs.standard_normal(200000) * np.exp(rs.uniform(-60, 60, 200000))).astype(np.float32)
    assert np.array_equal(MP.np_round(x[:, None], "bf16")[:, 0], ref_round(x, 8))
    ties = np.array([1 + 2 ** -8, 1 + 3 * 2 ** -8, -(1 + 2 ** -8), 3.0 + 2 ** -7], dtype=np.float32)
    want = np.array([1.0, 1 + 2 ** -6, -1.0, 3.0], dtype=np.float32)
    assert np.array_equal(MP.np_round(ties[:, None], "bf16")[:, 0], want), MP.np_round(ties[:, None], "bf16")
    # fp16 with the per-column scale: 11 significant bits for values within 2^14 of the column max
    cols = 64
    y = (rs.standard_normal((512, cols)) * np.exp(rs.uniform(-40, 10, cols))[None, :]).astype(np.float32)
    q = MP.np_round(y, "fp16")
    amax = np.max(np.abs(y), axis=0)
    big = np.abs(y) >= amax[None, :] * 2.0 ** -13
    assert np.array_equal(q[big], ref_round(y, 11)[big])
    err = np.abs(q.astype(np.float64) - y)
    assert np.all(err <= amax[None, :] * 2.0 ** -24 * 1.0001 + np.abs(y) * 2.0 ** -11 * 1.0001)
    assert np.array_equal(MP.np_round(np.zeros((4, 3), np.float32), "fp16"), np.zeros((4, 3), np.float32))
    # policies
    assert MP.parse_policy("none") == {"kind": "none"}
    assert MP.parse_policy("fp16_kx128") == {"kind": "kx", "fmt": "fp16", "K": 128}
    assert MP.parse_policy("ad32") == {"kind": "ad", "tau": 32.0}
    assert MP.parse_policy("bf16_all") == {"kind": "all", "fmt": "bf16"}
    for bad in ("fp8_all", "ad0", "adx", ""):
        try:
            MP.parse_policy(bad)
            raise AssertionError(f"accepted {bad!r}")
        except ValueError:
            pass
    # adaptive buckets: the error budget holds, lower energy never gets more precision, the extremes
    k = np.arange(1, 1367, dtype=np.float64)
    e = k ** (-8 / 3) * np.exp(-((k / 900.0) ** 8)) * rs.uniform(0.5, 2.0, k.size)
    for tau in (1.0, 32.0, 1024.0):
        b = MP.buckets(e, tau)
        added = np.sum(e[b == 2]) * MP.U["bf16"] ** 2 + np.sum(e[b == 1]) * MP.U["fp16"] ** 2
        assert added <= (tau * MP.U32) ** 2 * e.sum() * (1 + 1e-12), (tau, added)
        o = np.argsort(e, kind="stable")
        assert np.all(np.diff(b[o]) <= 0), tau
    assert np.all(MP.buckets(e, 1e9) == 2) and np.all(MP.buckets(e, 1e-9) == 0)
    fr = {a: {"f16": 0.5} for a in MP.POINTS}
    assert abs(MP.projected_ms(fr, 1.47) - 0.3675) < 1e-12


def test_no_shadowing():
    """InRowQ's methods and attributes must not reuse a name InRow assigns on self, or the instance attribute hides
    the method: job 7859359 died on InRow's `self.q = None` (theta_phys's buffer) hiding InRowQ.q, the rounding
    called after every op. Static (ast), so it runs on the login node; `ops` is the one intended override."""
    import ast

    def self_attrs(cls):
        return {t.attr for n in ast.walk(cls) if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign))
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self"}

    def find_class(path, name):
        tree = ast.parse(open(path).read())
        return next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name)

    here = os.path.dirname(os.path.abspath(__file__))
    base = find_class(os.path.join(os.path.dirname(here), "inrow", "inrow_step.py"), "InRow")
    sub = find_class(os.path.join(here, "mixprec.py"), "InRowQ")
    base_attrs = self_attrs(base)
    base_methods = {n.name for n in base.body if isinstance(n, ast.FunctionDef)}
    sub_methods = {n.name for n in sub.body if isinstance(n, ast.FunctionDef)} - {"__init__"}
    hidden = sub_methods & base_attrs
    assert not hidden, f"InRowQ methods hidden by InRow instance attributes: {sorted(hidden)}"
    clobbered = self_attrs(sub) & (base_attrs | base_methods)
    assert not clobbered, f"InRowQ overwrites InRow names: {sorted(clobbered)}"
    assert sub_methods & base_methods == {"ops"}, sorted(sub_methods & base_methods)


def bits(a):
    """The bit pattern of a complex64 array as uint32 (copied contiguous first: numpy cannot re-view a strided slice
    as a smaller dtype -- job 7854381 died on exactly that, in this GPU-only branch)."""
    return np.ascontiguousarray(a, dtype=np.complex64).view(np.uint32)


def test_torch():
    import torch
    if not torch.cuda.is_available():
        return "skipped (no CUDA)"
    rs = np.random.RandomState(1)
    R, C = 256, 48
    mag = np.exp(rs.uniform(-40, 10, C))[None, :]
    z = ((rs.standard_normal((R, C)) + 1j * rs.standard_normal((R, C))) * mag).astype(np.complex64)
    for fmt in ("bf16", "fp16"):
        X = torch.as_tensor(z, device="cuda")
        idx = torch.as_tensor(np.arange(0, C, 2), device="cuda")
        MP.torch_round_cols_(X, idx, fmt)
        got = X.cpu().numpy()
        assert np.array_equal(bits(got[:, 1::2]), bits(z[:, 1::2])), f"{fmt}: untouched columns"
        sel = z[:, 0::2]
        r = np.stack([sel.real, sel.imag], 1)                          # (R, 2, m): np_round's column = last axis
        want = MP.np_round(r, fmt)
        want = (want[:, 0] + 1j * want[:, 1]).astype(np.complex64)
        assert np.array_equal(bits(got[:, 0::2]), bits(want)), f"{fmt}: torch != numpy"
    b = MP.buckets(torch.as_tensor(np.linspace(1e-30, 1.0, 100), device="cuda").double(), 1.0)
    assert b.dtype == torch.int64 and int(b.max()) == 2 and int(b[-1]) == 0
    return f"ok ({torch.__version__})"


def main():
    test_numpy()
    test_no_shadowing()
    print(f"MIXPREC_TEST_OK torch={test_torch() if 'torch' in sys.argv[1:] or os.environ.get('MIXPREC_TORCH') else 'not run'}")


if __name__ == "__main__":
    main()
