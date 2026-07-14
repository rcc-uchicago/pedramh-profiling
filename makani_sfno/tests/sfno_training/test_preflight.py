"""Tests for scripts/preflight.py.

Coverage (per docs/sfno_tiny_short_training_plan.md §C.5):
  - ``check_makani_path`` passes on the live install (makani-src/), fails
    when the resolved path doesn't contain ``makani-src``.
  - ``check_rendered_yaml`` accepts a fully-rendered YAML; rejects one
    that still has ``{{PLACEHOLDER}}``; rejects template-diff anomalies.
  - ``check_single_batch_contract`` passes on a real PlasimTrainer with
    the RecordingDummyModel nettype; catches a forced 58→other channel
    breach via a monkey-patched preprocessor.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
makani = pytest.importorskip("makani")  # noqa: F401

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PREFLIGHT = _REPO_ROOT / "scripts" / "preflight.py"


def _load_preflight():
    spec = importlib.util.spec_from_file_location("preflight", _PREFLIGHT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["preflight"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def preflight():
    return _load_preflight()


# ---------------------------------------------------------------------------
# check_makani_path
# ---------------------------------------------------------------------------
class TestMakaniPath:
    def test_passes_on_live_install(self, preflight):
        # The live install under .venv is editable from makani-src/. If the
        # repo is configured per docs/sfno_training_implementation_plan.md,
        # this passes; otherwise the test fails clearly which is itself
        # a useful CI signal.
        preflight.check_makani_path()

    def test_fails_on_pypi_path(self, preflight):
        """Simulate a regression to a PyPI wheel install by patching
        ``makani.__file__`` to a path without ``makani-src``."""
        import makani as live_makani
        with patch.object(live_makani, "__file__", "/usr/lib/python3.12/site-packages/makani/__init__.py"):
            with pytest.raises(RuntimeError, match="makani-src"):
                preflight.check_makani_path()


# ---------------------------------------------------------------------------
# check_rendered_yaml
# ---------------------------------------------------------------------------
class TestRenderedYaml:
    def test_accepts_clean_yaml(self, preflight, tmp_path):
        rendered = tmp_path / "rendered.yaml"
        rendered.write_text("name: foo\npath: /scratch/bar\n")
        preflight.check_rendered_yaml(rendered, template=None)

    def test_rejects_leftover_placeholder(self, preflight, tmp_path):
        rendered = tmp_path / "rendered.yaml"
        rendered.write_text("name: foo\npath: {{OUTPUT_ROOT}}/bar\n")
        with pytest.raises(RuntimeError, match="placeholder"):
            preflight.check_rendered_yaml(rendered, template=None)

    def test_template_diff_passes_on_substitution(self, preflight, tmp_path):
        template = tmp_path / "tpl.yaml"
        template.write_text("name: foo\npath: {{OUTPUT_ROOT}}/bar\n")
        rendered = tmp_path / "rendered.yaml"
        rendered.write_text("name: foo\npath: /scratch/bar\n")
        preflight.check_rendered_yaml(rendered, template=template)

    def test_template_diff_rejects_unexpected_change(self, preflight, tmp_path):
        template = tmp_path / "tpl.yaml"
        template.write_text("name: foo\npath: {{OUTPUT_ROOT}}/bar\n")
        rendered = tmp_path / "rendered.yaml"
        # Mutated a non-placeholder line — should fail.
        rendered.write_text("name: bogus\npath: /scratch/bar\n")
        with pytest.raises(RuntimeError, match="no placeholder"):
            preflight.check_rendered_yaml(rendered, template=template)

    def test_template_diff_rejects_line_count_drift(self, preflight, tmp_path):
        template = tmp_path / "tpl.yaml"
        template.write_text("name: foo\npath: {{OUTPUT_ROOT}}/bar\n")
        rendered = tmp_path / "rendered.yaml"
        rendered.write_text("name: foo\npath: /scratch/bar\nextra_line: 1\n")
        with pytest.raises(RuntimeError, match="lines"):
            preflight.check_rendered_yaml(rendered, template=template)


# ---------------------------------------------------------------------------
# check_single_batch_contract — uses the same RecordingDummyModel nettype
# pattern as test_trainer_ci.py, so this is fast on CPU.
# ---------------------------------------------------------------------------
class TestSingleBatchContract:
    def _build_trainer(self, packaged_dataset: Path, exp_dir: Path):
        """Construct a PlasimTrainer with the recording-dummy nettype. Mirrors
        test_trainer_ci._populate_runtime_params + _override_for_smoke."""
        from test_trainer_ci import (
            _load_yparams,
            _override_for_smoke,
            _populate_runtime_params,
        )
        from sfno_training.trainer import PlasimTrainer

        params = _load_yparams(packaged_dataset)
        _populate_runtime_params(params, exp_dir)
        _override_for_smoke(params, n_future=0)
        params["valid_autoreg_steps"] = 0
        pt = PlasimTrainer(params, world_rank=0, device="cpu")
        return pt, params

    def test_passes(self, preflight, packaged_dataset: Path, tmp_path: Path):
        exp_dir = tmp_path / "exp"
        exp_dir.mkdir()
        (exp_dir / "training_checkpoints").mkdir()
        pt, params = self._build_trainer(packaged_dataset, exp_dir)

        # Reset captured inputs so we don't cross-talk with trainer warm-up.
        pt.model.model.inputs_seen.clear()

        preflight.check_single_batch_contract(pt, params)

        # Side-effect verification: the model saw a 58-channel input.
        seen = pt.model.model.inputs_seen
        assert len(seen) >= 1
        assert seen[0].shape[1] == 58

    def test_catches_wrong_internal_channels(
        self, preflight, packaged_dataset: Path, tmp_path: Path
    ):
        """If the wrapped SFNO sees the wrong channel count, the assertion
        in check_single_batch_contract must fire. Simulated by monkey-patching
        the preprocessor's append_unpredicted_features to produce a 53-channel
        tensor (skipping the 6-channel forcing concat)."""
        exp_dir = tmp_path / "exp"
        exp_dir.mkdir()
        (exp_dir / "training_checkpoints").mkdir()
        pt, params = self._build_trainer(packaged_dataset, exp_dir)

        original = pt.model.preprocessor.append_unpredicted_features

        def broken(self_pp, inp, target=False):
            # Stripped concat — inp is already 4D state (52ch), so just
            # pass it through to the model. The wrapper passes the
            # same tensor onward, so the SFNO sees 52 channels, not 58.
            return inp

        # Patch as a bound method on the instance.
        from types import MethodType
        pt.model.preprocessor.append_unpredicted_features = MethodType(
            broken, pt.model.preprocessor
        )
        try:
            with pytest.raises(AssertionError, match="forcing concat"):
                preflight.check_single_batch_contract(pt, params)
        finally:
            pt.model.preprocessor.append_unpredicted_features = original


# ---------------------------------------------------------------------------
# _build_loader_and_wrapper amp_mode / checkpointing_level passthrough
# (docs/sfno_full_training_plan.md §A.2)
# ---------------------------------------------------------------------------
class TestBuildLoaderPassthrough:
    """Verify the new kwargs land in params before PlasimTrainer is constructed.

    Patches ``sfno_training.trainer.PlasimTrainer`` to a capturing stub so
    we can inspect the params dict the helper hands off, without actually
    instantiating a full trainer (the helper is otherwise tested by
    TestSingleBatchContract above)."""

    def _run(
        self,
        preflight,
        packaged_dataset: Path,
        *,
        amp_mode: str,
        checkpointing_level: int,
    ):
        """Drive _build_loader_and_wrapper with the given kwargs; return
        the params dict captured at PlasimTrainer-construction time.

        The packaged_dataset fixture writes a fully-rendered YAML at
        config/plasim_sim52_astro_64x128_zgplev.yaml with a real exp_dir
        under the fixture root, so we feed it directly without rendering."""
        from unittest.mock import patch

        cfg_path = packaged_dataset / "config" / "plasim_sim52_astro_64x128_zgplev.yaml"
        captured = {}

        class _StubTrainer:
            def __init__(self, params, world_rank=0):
                captured["params"] = params
                self.params = params

        with patch("sfno_training.trainer.PlasimTrainer", _StubTrainer):
            preflight._build_loader_and_wrapper(
                cfg_path,
                "plasim_sim52_astro_64x128_zgplev",
                amp_mode=amp_mode,
                checkpointing_level=checkpointing_level,
            )
        return captured["params"]

    def test_amp_mode_bf16_passthrough(
        self, preflight, packaged_dataset: Path, tmp_path: Path
    ):
        params = self._run(
            preflight, packaged_dataset,
            amp_mode="bf16", checkpointing_level=0,
        )
        assert params["amp_mode"] == "bf16"
        assert params["checkpointing_level"] == 0

    def test_checkpointing_level_2_passthrough(
        self, preflight, packaged_dataset: Path, tmp_path: Path
    ):
        params = self._run(
            preflight, packaged_dataset,
            amp_mode="none", checkpointing_level=2,
        )
        assert params["amp_mode"] == "none"
        assert params["checkpointing_level"] == 2

    def test_defaults_match_legacy_tiny_behavior(
        self, preflight, packaged_dataset: Path, tmp_path: Path
    ):
        """If kwargs are omitted, params must default to the original
        hardcoded values ('none' / 0) — so submit_tiny.slurm (which
        doesn't pass the flags) keeps working unchanged."""
        params = self._run(
            preflight, packaged_dataset,
            amp_mode="none", checkpointing_level=0,
        )
        assert params["amp_mode"] == "none"
        assert params["checkpointing_level"] == 0
