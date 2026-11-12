"""test_tilerows_advance.py -- the pure-Python parts of tilerows_advance.py and its wiring into o12_polaris.py (no GPU;
python >= 3.6). Prints O12_TR_TEST_OK.

  * parse: if32_25 / tilerows40 / tilerows_if25 -> (kind, if_mode, steps per frame); bad names raise.
  * frame_steps: a segment's first IF frame is 2 switching steps + (n - 2); every frame spans stride DT exactly
    (2e-4 = 25 x 8e-6 in float64 to 1 ulp).
  * o12_polaris: every tile-row solver is in SOLVERS, none is BITWISE, each parses at o1.2's STRIDE 40.
  * full_from_compact (the scatter-free expand) == tilerows_step.from_compact exactly, in numpy.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tilerows_advance as TA                                        # noqa: E402
import o12_polaris as OP                                             # noqa: E402

STRIDE, DT = 40, 5e-6


def main():
    assert TA.parse("if32_25", STRIDE) == ("if32", True, 25)
    assert TA.parse("tilerows_if25", STRIDE) == ("tr", True, 25)
    assert TA.parse("tilerows40", STRIDE) == ("tr", False, 40)
    for bad in ("tilerows32", "if32_2", "if32_41", "lean32ek_fast", "tilerows_if"):
        try:
            TA.parse(bad, STRIDE)
        except ValueError:
            continue
        raise AssertionError(f"parse accepted {bad!r}")
    for scheme in ("if32_25", "tilerows_if25", "tilerows40"):
        kind, if_mode, n = TA.parse(scheme, STRIDE)
        h = STRIDE * DT / n
        first = TA.frame_steps(if_mode, n, True) + (2 if if_mode else 0)
        assert first == n and TA.frame_steps(if_mode, n, False) == n, scheme
        assert abs(first * h - STRIDE * DT) <= 2 * 2.0 ** -52 * STRIDE * DT, (scheme, first * h)   # <= 2 ulp
    import numpy as np
    import tilerows_step as TS                                       # numpy-only at import
    rng = np.random.default_rng(0)
    c = rng.standard_normal((TS.CR, TS.CC)) + 1j * rng.standard_normal((TS.CR, TS.CC))
    full = TA.full_from_compact(np, c, TS.N, TS.NK, TS.KM)
    assert full.shape == (TS.N, TS.NK) and np.array_equal(full, TS.from_compact(c, np.complex128)), "expand order"
    assert set(OP.TILEROWS_SOLVERS) <= set(OP.SOLVERS)
    assert not set(OP.TILEROWS_SOLVERS) & set(OP.BITWISE_SOLVERS)
    for s in OP.TILEROWS_SOLVERS:
        TA.parse(s, STRIDE)
    print(f"O12_TR_TEST_OK schemes={len(OP.TILEROWS_SOLVERS)} frame grid exact; wiring consistent "
          f"(python {sys.version.split()[0]})")


if __name__ == "__main__":
    main()
