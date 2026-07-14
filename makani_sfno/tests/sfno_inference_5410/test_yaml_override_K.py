"""K-threading regression test for stampede3_yaml_override.

Asserts the new K-aware contract (per docs/2026-05-08_sfno_5410_explicit_K_horizon_plan.md):
  * ``build_per_y_yaml(..., K=K)`` writes ``ensemble_inference_hours = (K+1)*6``
    AND ``prediction_duration_days = (K+1)*6/24`` on the SFNO section.
  * K is required (call without K raises TypeError).
  * Bool / non-int / non-positive K are rejected with ValueError
    (bool rejection is critical because ``isinstance(True, int)`` is True).
  * K > 1463 (multi-year) is rejected with ValueError.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ruamel.yaml")

from sfno_inference_5410.stampede3_yaml_override import (  # noqa: E402
    UPSTREAM_CKPT_PATH,
    UPSTREAM_YAML_PATH,
    _horizon_hours_for_K,
    _raw_steps_for_K,
    build_per_y_yaml,
)


def _need_upstream_artifacts():
    if not UPSTREAM_YAML_PATH.is_file():
        pytest.skip(f"upstream yaml not present: {UPSTREAM_YAML_PATH}")
    if not UPSTREAM_CKPT_PATH.is_file():
        pytest.skip(f"upstream ckpt not present: {UPSTREAM_CKPT_PATH}")


@pytest.mark.parametrize("K,expected_hours,expected_days", [
    (60, 366, 15.25),
    (56, 342, 14.25),
    (1, 12, 0.5),       # smallest K — sanity bound
    (1463, 8784, 366.0),  # largest allowed K (one leap year)
])
def test_horizon_helpers(K, expected_hours, expected_days):
    assert _raw_steps_for_K(K) == K + 1
    assert _horizon_hours_for_K(K) == expected_hours
    # Cross-check the day form used by prediction_duration_days.
    assert (K + 1) * 6 / 24.0 == expected_days


def test_K60_yaml_horizon_keys(tmp_path):
    """Y=121, K=60 → ensemble_inference_hours=366, prediction_duration_days=15.25."""
    _need_upstream_artifacts()
    from ruamel.yaml import YAML

    yaml_path = build_per_y_yaml(
        121, tmp_path / "config", tmp_path / "exp", K=60,
    )
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(yaml_path) as f:
        doc = yaml.load(f)
    sfno = doc["SFNO"]
    assert int(sfno["ensemble_inference_hours"]) == 366, (
        f"K=60 yaml ensemble_inference_hours expected 366; got {sfno['ensemble_inference_hours']}"
    )
    assert abs(float(sfno["prediction_duration_days"]) - 15.25) < 1e-9, (
        f"K=60 yaml prediction_duration_days expected 15.25; got {sfno['prediction_duration_days']}"
    )
    # Year-long sentinels must NOT leak through.
    assert int(sfno["ensemble_inference_hours"]) not in (8760, 8784)


def test_K56_yaml_horizon_keys(tmp_path):
    """K=56 → 342 / 14.25 (alternative scoreboard window)."""
    _need_upstream_artifacts()
    from ruamel.yaml import YAML

    yaml_path = build_per_y_yaml(
        121, tmp_path / "config", tmp_path / "exp", K=56,
    )
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(yaml_path) as f:
        doc = yaml.load(f)
    sfno = doc["SFNO"]
    assert int(sfno["ensemble_inference_hours"]) == 342
    assert abs(float(sfno["prediction_duration_days"]) - 14.25) < 1e-9


def test_K_required_keyword_only(tmp_path):
    """Calling build_per_y_yaml without K must raise TypeError."""
    _need_upstream_artifacts()
    with pytest.raises(TypeError):
        # Intentional: K omitted → TypeError from the required kw-only arg.
        build_per_y_yaml(121, tmp_path / "config", tmp_path / "exp")  # type: ignore[call-arg]


@pytest.mark.parametrize("bad", [True, False, 0, -1, 60.0, "60", None, 1.5])
def test_K_validation_rejects_bad_inputs(tmp_path, bad):
    """ValueError (not assert) for bool / non-int / <1 / float."""
    _need_upstream_artifacts()
    with pytest.raises(ValueError):
        build_per_y_yaml(121, tmp_path / "config", tmp_path / "exp", K=bad)


def test_K_too_large_rejected(tmp_path):
    """K such that (K+1)*6 > 8784 must raise ValueError."""
    _need_upstream_artifacts()
    # K=1464 → (1464+1)*6 = 8790 > 8784.
    with pytest.raises(ValueError):
        build_per_y_yaml(121, tmp_path / "config", tmp_path / "exp", K=1464)


def test_helpers_reject_bool_directly():
    """The validation helper must reject bool even though isinstance(True, int) is True."""
    for bad in (True, False):
        with pytest.raises(ValueError):
            _raw_steps_for_K(bad)
        with pytest.raises(ValueError):
            _horizon_hours_for_K(bad)
