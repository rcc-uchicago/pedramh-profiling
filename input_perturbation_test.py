#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the PanguWeather-parity input perturbation (`epsilon_factor`).

The science owner set the noise value to **0.01 for all models** (2026-08-06).
That decision touched two different defects, and this file pins both:

1. **The two PanguWeather references disagreed by 10x** — the Derecho config
   carried 0.1, the Stampede config 0.01, and our Polaris configs inherited the
   Derecho value. Both Polaris configs must now read 0.01.
2. **ai-rossby had no perturbation at all**, so identical bytes on disk reached
   the two models as different inputs. It now applies the same noise.

What makes the perturbation correct is narrow and easy to get wrong, so each
property is checked rather than argued:

* it hits the INPUT state only — never targets, boundaries, or diagnostics;
* sigma is exactly `epsilon_factor` (PanguWeather's `ff_std/std` scaling is 1.0
  because both resolve to the same file);
* it defaults to OFF, so every config that does not set it is unchanged;
* it is drawn from the per-rank-seeded global RNG rather than inside DataLoader
  workers — which is what keeps `num_workers` output-neutral here, unlike
  PanguWeather where it is not (CHANGELOG next-action #4).

**Standard library + torch-optional.** The tensor properties run only if torch
imports; the config assertions are text-only so they always run, including on a
login node (CLAUDE.md #3).

Run::

    python3.12 input_perturbation_test.py

PASS = ``INPUT_PERTURBATION_TEST_OK (<n> tests)``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
RECIPE = REPO / "physicsnemo_ai_rossby/examples/weather/ai_rossby"
TRAIN_LOOP = RECIPE / "train_loop.py"
TRAIN = RECIPE / "train.py"
PROFILE = RECIPE / "profile_train.py"
PARITY_TRAINING = RECIPE / "conf/training/sfno_e3sm_parity.yaml"
PANGU_CFGS = [
    REPO / "PanguWeather/v2.0/config/E3SM_SFNO_H5_POLARIS_ALLDATA.yaml",
    REPO / "PanguWeather/v2.0/config/E3SM_SFNO_H5_POLARIS.yaml",
]

EPS = 0.01


def _eps_of(path: Path) -> float:
    m = re.search(r"(?m)^\s*epsilon_factor:\s*([0-9.eE+-]+)", path.read_text())
    assert m, f"epsilon_factor not found in {path.name}"
    return float(m.group(1))


# --- the decision, as recorded in the configs ------------------------------


def test_pangu_polaris_configs_are_0p01():
    for p in PANGU_CFGS:
        v = _eps_of(p)
        assert v == EPS, f"{p.name}: epsilon_factor={v}, expected {EPS}"


def test_ai_rossby_parity_config_is_0p01():
    assert _eps_of(PARITY_TRAINING) == EPS


def test_jesswan_reference_configs_untouched():
    """We must not edit the science owner's own reference files.

    Her two configs disagree (Derecho 0.1, Stampede 0.01) and that discrepancy
    is hers to resolve — we changed only OUR Polaris configs. If a future edit
    "tidies" hers, the provenance of the 10x finding disappears.
    """
    d = _eps_of(REPO / "PanguWeather/v2.0/config/E3SM_SFNO_H5_DERECHO_jsw.yaml")
    s = _eps_of(REPO / "PanguWeather/v2.0/config/E3SM_SFNO_H5_STAMPEDE_jsw.yaml")
    assert (d, s) == (0.1, 0.01), f"reference configs changed: derecho={d} stampede={s}"


# --- wiring ----------------------------------------------------------------


def test_default_is_off_everywhere():
    """Unset => 0.0 => byte-identical behaviour for every other config."""
    src = TRAIN_LOOP.read_text()
    assert src.count("epsilon_factor: float = 0.0") == 2, (
        "both train_step and multistep_train_step must default epsilon_factor to 0.0"
    )
    for p in (TRAIN, PROFILE):
        assert 'get("epsilon_factor", 0.0)' in p.read_text(), f"{p.name} must default to 0.0"


def test_threaded_into_both_step_functions():
    src = TRAIN_LOOP.read_text()
    for fn in ("def train_step(", "def multistep_train_step("):
        body = src.split(fn)[1].split("\n) ->")[0]
        assert "epsilon_factor" in body, f"{fn} does not accept epsilon_factor"
    # train.py has TWO call sites (single-step and multistep); both must pass it.
    assert TRAIN.read_text().count("epsilon_factor=epsilon_factor,") == 2
    assert PROFILE.read_text().count("epsilon_factor=epsilon_factor,") == 1


def test_perturbs_inputs_only_not_targets():
    """The single most consequential property: targets must stay clean.

    Perturbing the target would change the loss into something else entirely,
    and it is a one-word mistake away.
    """
    src = TRAIN_LOOP.read_text()
    block = src.split("if epsilon_factor and epsilon_factor > 0.0:")[1].split("\n\n")[0]
    assert "surface_in" in block and "upper_air_in" in block
    for forbidden in ("target_surface", "target_upper_air",
                      "constant_boundary", "varying_boundary"):
        assert forbidden not in block, f"perturbation must not touch {forbidden}"


def test_not_drawn_inside_dataloader_workers():
    """Keeps num_workers output-neutral on this side.

    PanguWeather draws its noise in DataLoader workers from the global RNG with
    no worker_init_fn, which is precisely why its num_data_workers changes
    results. The ai-rossby dataset must stay RNG-free.
    """
    ds = (REPO / "physicsnemo_ai_rossby/physicsnemo/experimental/datapipes/"
          "climate/dataset.py").read_text()
    for tok in ("torch.randn", "np.random", "epsilon_factor"):
        assert tok not in ds, f"{tok} leaked into the dataset — worker count would matter"


# --- tensor properties (torch only) ----------------------------------------


def _torch():
    try:
        import torch  # noqa: PLC0415
        return torch
    except Exception:
        return None


def test_sigma_equals_epsilon_and_targets_untouched():
    torch = _torch()
    if torch is None:
        return  # login node without torch — config tests above still ran
    sys.path.insert(0, str(RECIPE))
    from train_loop import perturb_inputs  # noqa: PLC0415

    torch.manual_seed(0)
    s = torch.zeros(4, 8, 32, 64)
    u = torch.zeros(4, 5, 18, 32, 64)
    s2, u2 = perturb_inputs(s, u, EPS)
    # Zero-mean, sigma == epsilon_factor. Loose bounds: this is a sanity check
    # on the SCALE, not a distribution test.
    assert abs(s2.std().item() - EPS) < 0.1 * EPS, s2.std().item()
    assert abs(u2.std().item() - EPS) < 0.1 * EPS, u2.std().item()
    assert abs(s2.mean().item()) < 0.05 * EPS
    # Inputs are not mutated in place — the caller's tensors must survive.
    assert s.abs().max().item() == 0.0
    assert u.abs().max().item() == 0.0


def test_zero_epsilon_is_exactly_identity():
    torch = _torch()
    if torch is None:
        return
    sys.path.insert(0, str(RECIPE))
    from train_loop import perturb_inputs  # noqa: PLC0415

    s = torch.randn(2, 8, 16, 32)
    u = torch.randn(2, 5, 18, 16, 32)
    s2, u2 = perturb_inputs(s, u, 0.0)
    assert s2 is s and u2 is u, "epsilon 0 must return the SAME objects, not copies"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            failed += 1
            print("ERROR %s: %s" % (t.__name__, exc))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("ERROR %s raised %s: %s" % (t.__name__, type(exc).__name__, exc))
    if failed:
        print("ERROR INPUT_PERTURBATION_TEST_FAILED (%d/%d)" % (failed, len(tests)))
        sys.exit(1)
    print("INPUT_PERTURBATION_TEST_OK (%d tests)" % len(tests))
