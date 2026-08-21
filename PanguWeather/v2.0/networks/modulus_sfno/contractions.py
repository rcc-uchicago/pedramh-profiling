
# flake8: noqa
# Copied from https://github.com/ai2cm/modulus/commit/22df4a9427f5f12ff6ac891083220e7f2f54d229
# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch


@torch.jit.script
def compl_mul1d_fwd(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a complex-valued multiplication operation between two 1-dimensional
    tensors.
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bix,io->box", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def compl_muladd1d_fwd(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs complex multiplication of two 1-dimensional tensors 'a' and 'b', and then
    adds a third tensor 'c'.
    """
    tmpcc = torch.view_as_complex(compl_mul1d_fwd(a, b))
    cc = torch.view_as_complex(c)
    return torch.view_as_real(tmpcc + cc)


@torch.jit.script
def compl_mul2d_fwd(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a complex-valued multiplication operation between two 2-dimensional
    tensors.
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,io->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def compl_muladd2d_fwd(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs complex multiplication of two 2-dimensional tensors 'a' and 'b', and then
    adds a third tensor 'c'.
    """
    tmpcc = torch.view_as_complex(compl_mul2d_fwd(a, b))
    cc = torch.view_as_complex(c)
    return torch.view_as_real(tmpcc + cc)


@torch.jit.script  # TODO remove
def _contract_localconv_fwd(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a complex local convolution operation between two tensors 'a' and 'b'.
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,iox->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script  # TODO remove
def _contract_blockconv_fwd(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a complex block convolution operation between two tensors 'a' and 'b'.
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bim,imn->bin", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script  # TODO remove
def _contractadd_blockconv_fwd(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a complex block convolution operation between two tensors 'a' and 'b', and
    then adds a third tensor 'c'.
    """
    tmpcc = torch.view_as_complex(_contract_blockconv_fwd(a, b))
    cc = torch.view_as_complex(c)
    return torch.view_as_real(tmpcc + cc)


# for the experimental layer
@torch.jit.script  # TODO remove
def compl_exp_mul2d_fwd(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a 2D complex multiplication operation between two tensors 'a' and 'b'.
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,xio->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def compl_exp_muladd2d_fwd(  # TODO remove
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a 2D complex multiplication operation between two tensors 'a' and 'b',
    and then adds a third tensor 'c'.
    """
    tmpcc = torch.view_as_complex(compl_exp_mul2d_fwd(a, b))
    cc = torch.view_as_complex(c)
    return torch.view_as_real(tmpcc + cc)


@torch.jit.script
def real_mul2d_fwd(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a 2D real multiplication operation between two tensors 'a' and 'b'.
    """
    res = torch.einsum("bixy,io->boxy", a, b)
    return res


@torch.jit.script
def real_muladd2d_fwd(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a 2D real multiplication operation between two tensors 'a' and 'b', and
    then adds a third tensor 'c'.
    """
    res = real_mul2d_fwd(a, b) + c
    return res


# new contractions set to replace older ones. We use complex
@torch.jit.script
def _contract_diagonal(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a complex diagonal operation between two tensors 'a' and 'b'.
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,ioxy->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


def dhconv_weight_is_xio():
    """True when the dhconv weight is stored `[modes_lat, in, out]` (§4.9's layout fix).

    Off by default and deliberately so: flipping it changes the **shape** of 95.8% of the
    model's parameters, so every existing checkpoint would fail `load_state_dict`.
    Adoption needs a conversion pass; this knob makes the change *testable* against the
    §4.12 gate first.

    Read in plain Python, never inside the scripted contraction: `_contract_dhconv` is
    `@torch.jit.script`, and TorchScript cannot compile `os.environ`. Hence two scripted
    variants selected by the (unscripted) caller rather than one branching function.
    """
    return os.environ.get("PANGU_DHCONV_XIO", "0") == "1"


@torch.jit.script
def _contract_dhconv(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a complex Driscoll-Healy style convolution operation between two tensors
    'a' and 'b'.
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,iox->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def _contract_dhconv_xio(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """`_contract_dhconv` with the weight stored x-major — §4.9's layout fix.

    Identical arithmetic to `_contract_dhconv`; the ONLY difference is that `b` arrives as
    `[modes_lat, in, out]` instead of `[in, out, modes_lat]`, so `einsum`'s lowering to a
    batched matmul (`x` is a batch index, `i` is contracted) needs no permute. Stored
    `[i, o, x]` that permute moves 377 MB per call at **32.00 sectors/request** — the
    counter's ceiling — reading 2043 MB to move 377 MB (§4.8).
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,xio->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def _contract_sep_diagonal(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a complex convolution operation between two tensors 'a' and 'b'
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,ixy->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def _contract_sep_dhconv(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a complex convolution operation between two tensors 'a' and 'b'
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,ix->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res

