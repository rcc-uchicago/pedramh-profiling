"""Tests for slice_checkpoint.slice_state_dict (port F surgical transfer)."""

from __future__ import annotations

import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from slice_checkpoint import slice_state_dict  # noqa: E402

N_IN, N_OUT, E = 9, 7, 16      # 7 state+diag-ish outputs, 9 inputs, embed 16
DROP = [2, 3]


class Toy(torch.nn.Module):
    """encoder 1x1 (N_IN->E), trunk (E->E), decoder 1x1 (E->N_OUT), big_skip."""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.enc = torch.nn.Conv2d(n_in, E, 1)
        self.trunk = torch.nn.Conv2d(E, E, 1)
        self.dec = torch.nn.Conv2d(E, n_out, 1)
        self.residual_transform = torch.nn.Conv2d(n_in, n_out, 1, bias=False)

    def forward(self, x):
        return self.dec(torch.relu(self.trunk(torch.relu(self.enc(x))))) + self.residual_transform(x)


def test_sliced_model_equals_original_with_dropped_inputs_zeroed():
    torch.manual_seed(0)
    big = Toy(N_IN, N_OUT)
    sd, touched = slice_state_dict(big.state_dict(), N_IN, N_OUT, DROP, DROP)
    small = Toy(N_IN - 2, N_OUT - 2)
    small.load_state_dict(sd, strict=True)  # the fork's warm-start is strict
    keep_in = [i for i in range(N_IN) if i not in DROP]
    keep_out = [i for i in range(N_OUT) if i not in DROP]
    x = torch.randn(2, N_IN, 3, 4)
    x0 = x.clone()
    x0[:, DROP] = 0.0
    with torch.no_grad():
        torch.testing.assert_close(small(x[:, keep_in]), big(x0)[:, keep_out])
    names = {k for k, *_ in touched}
    assert names == {"enc.weight", "dec.weight", "dec.bias", "residual_transform.weight"}
    assert "trunk.weight" not in names


def test_equal_widths_are_refused():
    with pytest.raises(ValueError, match="SLICE_AMBIGUOUS"):
        slice_state_dict({}, 5, 5, [1], [1])
