"""Cross-yaml static-architecture invariant.

All 8 per-Y yamls produced by ``build_all(K=60)`` must agree on every
architecture / checkpoint / normalization / precision field. Only the
four per-Y fields (val_year_start, val_year_end, leap_year, no_leap_year)
may differ.

This invariant is what justifies the in-process orchestrator's design:
  * Build the Stepper ONCE from yaml_paths[0]'s params.
  * Reuse self.model + ckpt across all 96 ICs.
  * Mutate only the per-Y + per-IC fields when crossing IC boundaries.

If any non-per-Y field diverges between yamls, swapping the per-Y yaml
mid-run would silently change the model architecture or normalization,
which would invalidate the reuse of self.model.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_UPSTREAM_REPO = Path(
    "/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0"
)


@pytest.fixture(scope="module")
def all_yamls(tmp_path_factory):
    pytest.importorskip("ruamel.yaml")
    if not _UPSTREAM_REPO.is_dir():
        pytest.skip(f"upstream repo not present: {_UPSTREAM_REPO}")
    from sfno_inference_5410.stampede3_yaml_override import (
        UPSTREAM_CKPT_PATH,
        UPSTREAM_YAML_PATH,
        build_all,
    )
    if not UPSTREAM_YAML_PATH.is_file():
        pytest.skip(f"upstream yaml not present: {UPSTREAM_YAML_PATH}")
    if not UPSTREAM_CKPT_PATH.is_file():
        pytest.skip(f"upstream ckpt not present: {UPSTREAM_CKPT_PATH}")
    root = tmp_path_factory.mktemp("static_arch")
    out = build_all(root / "config", root / "exp", K=60)
    return [out[Y]["yaml"] for Y in sorted(out)]


def test_assert_yamls_share_static_arch_passes_on_all_8(all_yamls):
    """Live integration: the 8 yamls produced by build_all(K=60) pass
    the static-arch invariant."""
    from sfno_inference_5410.upstream_hydration import (
        assert_yamls_share_static_arch,
    )
    assert len(all_yamls) == 8
    assert_yamls_share_static_arch(all_yamls)


def test_assert_yamls_share_static_arch_rejects_arch_drift(tmp_path):
    """Sanity: deliberately divergent yaml triggers ValueError."""
    pytest.importorskip("ruamel.yaml")
    from ruamel.yaml import YAML
    from sfno_inference_5410.upstream_hydration import (
        assert_yamls_share_static_arch,
    )
    yaml = YAML()
    yaml.preserve_quotes = True

    base_doc = {"SFNO": {
        "embed_dim": 256, "num_layers": 8,
        "val_year_start": 121, "val_year_end": 122,
        "leap_year": 121, "no_leap_year": 121,
    }}
    p1 = tmp_path / "y121.yaml"
    with open(p1, "w") as f:
        yaml.dump(base_doc, f)

    drift_doc = {"SFNO": {
        "embed_dim": 384,  # CHANGED — architecture drift!
        "num_layers": 8,
        "val_year_start": 122, "val_year_end": 123,
        "leap_year": 122, "no_leap_year": 122,
    }}
    p2 = tmp_path / "y122.yaml"
    with open(p2, "w") as f:
        yaml.dump(drift_doc, f)

    with pytest.raises(ValueError, match="embed_dim"):
        assert_yamls_share_static_arch([p1, p2])


def test_assert_yamls_share_static_arch_allows_per_y_diff(tmp_path):
    """Sanity: differing only in per-Y fields is allowed."""
    pytest.importorskip("ruamel.yaml")
    from ruamel.yaml import YAML
    from sfno_inference_5410.upstream_hydration import (
        assert_yamls_share_static_arch,
    )
    yaml = YAML()
    yaml.preserve_quotes = True

    common = {"embed_dim": 256, "num_layers": 8}
    p1 = tmp_path / "y121.yaml"
    with open(p1, "w") as f:
        yaml.dump({"SFNO": dict(common, val_year_start=121, val_year_end=122,
                                  leap_year=121, no_leap_year=121)}, f)
    p2 = tmp_path / "y122.yaml"
    with open(p2, "w") as f:
        yaml.dump({"SFNO": dict(common, val_year_start=122, val_year_end=123,
                                  leap_year=122, no_leap_year=122)}, f)
    # No raise.
    assert_yamls_share_static_arch([p1, p2])


def test_assert_yamls_share_static_arch_singleton_is_noop(tmp_path):
    """One yaml is trivially self-consistent."""
    pytest.importorskip("ruamel.yaml")
    from ruamel.yaml import YAML
    from sfno_inference_5410.upstream_hydration import (
        assert_yamls_share_static_arch,
    )
    yaml = YAML()
    p = tmp_path / "single.yaml"
    with open(p, "w") as f:
        yaml.dump({"SFNO": {"embed_dim": 256}}, f)
    # No raise.
    assert_yamls_share_static_arch([p])
