"""Tests for src/sfno_inference/checkpoint_loader.py.

Coverage (per docs/sfno_eval_plan.md §B.0):

  - ``load_eval_params``:
      * applies BOTH ``valid_autoreg_steps = K-1`` and ``n_future = K-1``;
      * preserves the channel-count assertions (58, 53, 52, 1, 6);
      * derives ``amp_enabled`` / ``amp_dtype`` from ``amp_mode``
        (none, fp16, bf16);
      * pins normalization paths to the run dir;
      * rejects K < 1.

  - ``_assert_on_device``:
      * accepts ``torch.device("cuda")`` (no index) against
        ``cuda:0`` (the v2.7 round-7 fix);
      * rejects type mismatches (cpu vs cuda).

  - Real-checkpoint contract test (gated on
    ``AIRES_TEST_REAL_CKPT=1``): loads the production
    ``best_ckpt_mp0.tar``, builds the wrapper on CPU, asserts
    ``inp_chans == 58, out_chans == 53`` and parameter device is cpu.
    Skipped by default because Makani's distributed init can be heavy
    in CI; opt-in for local validation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")
makani = pytest.importorskip("makani")  # noqa: F401

from sfno_inference import checkpoint_loader as cl  # noqa: E402


_RUN_DIR_REAL = Path(
    "/scratch/11114/zhixingliu/AI-RES/runs/sfno_full/plasim_sim52_full/0"
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _minimal_config(amp_mode: str = "bf16") -> dict:
    """A minimal config.json shaped like the real production one."""
    return {
        "nettype": "SFNO",
        "N_in_channels": 58,
        "N_out_channels": 53,
        "n_state_channels": 52,
        "n_diagnostic_channels": 1,
        "n_forcing_channels": 6,
        "amp_mode": amp_mode,
        "valid_autoreg_steps": 3,
        "n_future": 0,
        "n_history": 0,
        "batch_size": 4,
        "global_means_path": "/should/be/overridden",
        "global_stds_path": "/should/be/overridden",
        "forcing_global_means_path": "/some/forcing/means.npy",
        "forcing_global_stds_path": "/some/forcing/stds.npy",
        "img_shape_x_resampled": 64,
        "img_shape_y_resampled": 128,
        "model_parallel_sizes": [1, 1, 1, 1],
        "model_parallel_names": ["h", "w", "fin", "fout"],
    }


def _make_run_dir(tmp_path: Path, *, amp_mode: str = "bf16",
                  cfg_overrides: dict | None = None) -> Path:
    """Build a fake run dir with config.json + global_means/stds.npy."""
    cfg = _minimal_config(amp_mode=amp_mode)
    if cfg_overrides:
        cfg.update(cfg_overrides)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps(cfg))

    # Plausible 53-channel stats; values irrelevant for these tests.
    np.save(run_dir / "global_means.npy", np.zeros(53, dtype=np.float32))
    np.save(run_dir / "global_stds.npy", np.ones(53, dtype=np.float32))
    return run_dir


# ---------------------------------------------------------------------------
# load_eval_params — happy path + overrides
# ---------------------------------------------------------------------------

class TestLoadEvalParams:
    def test_sets_both_horizon_handles(self, tmp_path):
        """v2.4 finding: valid_autoreg_steps is the active handle in eval mode.

        Setting only n_future would silently cap rollouts at the
        training-time valid_autoreg_steps=3 regardless of K.
        """
        run = _make_run_dir(tmp_path)
        ep = cl.load_eval_params(run, K=42)
        assert ep.valid_autoreg_steps == 41
        assert ep.n_future == 41

    def test_channel_contract_preserved(self, tmp_path):
        run = _make_run_dir(tmp_path)
        ep = cl.load_eval_params(run, K=10)
        assert ep.N_in_channels == 58
        assert ep.N_out_channels == 53
        assert ep.n_state_channels == 52
        assert ep.n_diagnostic_channels == 1
        assert ep.n_forcing_channels == 6

    def test_pins_normalization_to_run_dir(self, tmp_path):
        run = _make_run_dir(tmp_path)
        ep = cl.load_eval_params(run, K=10)
        assert ep.global_means_path == str(run / "global_means.npy")
        assert ep.global_stds_path == str(run / "global_stds.npy")

    def test_eval_overrides_set(self, tmp_path):
        run = _make_run_dir(tmp_path)
        ep = cl.load_eval_params(run, K=10)
        assert ep.n_history == 0
        assert ep.data_num_shards == 1
        assert ep.data_shard_id == 0
        assert ep.batch_size == 1

    def test_amp_bf16_derivation(self, tmp_path):
        run = _make_run_dir(tmp_path, amp_mode="bf16")
        ep = cl.load_eval_params(run, K=10)
        assert ep.amp_enabled is True
        assert ep.amp_dtype == torch.bfloat16

    def test_amp_fp16_derivation(self, tmp_path):
        run = _make_run_dir(tmp_path, amp_mode="fp16")
        ep = cl.load_eval_params(run, K=10)
        assert ep.amp_enabled is True
        assert ep.amp_dtype == torch.float16

    def test_amp_none_derivation(self, tmp_path):
        run = _make_run_dir(tmp_path, amp_mode="none")
        ep = cl.load_eval_params(run, K=10)
        assert ep.amp_enabled is False
        assert ep.amp_dtype == torch.float32

    def test_amp_unknown_raises(self, tmp_path):
        run = _make_run_dir(tmp_path, amp_mode="int8")
        with pytest.raises(ValueError, match="amp_mode"):
            cl.load_eval_params(run, K=10)


# ---------------------------------------------------------------------------
# load_eval_params — guard rails
# ---------------------------------------------------------------------------

class TestLoadEvalParamsGuards:
    def test_missing_run_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="config.json"):
            cl.load_eval_params(tmp_path / "nope", K=10)

    def test_missing_stats(self, tmp_path):
        run = _make_run_dir(tmp_path)
        (run / "global_means.npy").unlink()
        with pytest.raises(FileNotFoundError, match="normalization stats"):
            cl.load_eval_params(run, K=10)

    def test_K_must_be_positive(self, tmp_path):
        run = _make_run_dir(tmp_path)
        with pytest.raises(ValueError, match="K must be"):
            cl.load_eval_params(run, K=0)

    def test_channel_count_drift_breaks(self, tmp_path):
        # Forge a config that claims 60 input channels — would silently
        # build the wrong model if we didn't assert.
        run = _make_run_dir(tmp_path, cfg_overrides={"N_in_channels": 60})
        with pytest.raises(AssertionError, match="58"):
            cl.load_eval_params(run, K=10)


# ---------------------------------------------------------------------------
# _assert_on_device — v2.7 safe type+index comparison
# ---------------------------------------------------------------------------

class TestAssertOnDevice:
    def _make_dummy(self, device):
        m = torch.nn.Linear(2, 2)
        return m.to(device)

    def test_accepts_exact_match_cpu(self):
        m = self._make_dummy("cpu")
        cl._assert_on_device(m, "cpu")  # should not raise

    def test_accepts_torch_device_object(self):
        m = self._make_dummy("cpu")
        cl._assert_on_device(m, torch.device("cpu"))

    def test_rejects_type_mismatch(self):
        m = self._make_dummy("cpu")
        with pytest.raises(AssertionError, match="expected"):
            cl._assert_on_device(m, "cuda")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_accepts_no_index_against_indexed_param(self):
        """The v2.7 fix — torch.device('cuda') accepted against cuda:0 params."""
        m = self._make_dummy(torch.device(f"cuda:{torch.cuda.current_device()}"))
        # Bare 'cuda' (no index) should be treated as a wildcard.
        cl._assert_on_device(m, "cuda")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_rejects_wrong_indexed_device(self):
        """If user asks for cuda:9 we should fail loudly."""
        m = self._make_dummy(torch.device(f"cuda:{torch.cuda.current_device()}"))
        with pytest.raises(AssertionError):
            cl._assert_on_device(m, "cuda:9")


# ---------------------------------------------------------------------------
# Real-checkpoint smoke test (opt-in)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("AIRES_TEST_REAL_CKPT") != "1",
    reason="set AIRES_TEST_REAL_CKPT=1 to exercise the full build+restore path",
)
class TestRealCheckpoint:
    def test_load_eval_params_on_real_run_dir(self):
        if not _RUN_DIR_REAL.is_dir():
            pytest.skip(f"real run dir not present: {_RUN_DIR_REAL}")
        ep = cl.load_eval_params(_RUN_DIR_REAL, K=56)
        assert ep.valid_autoreg_steps == 55
        assert ep.amp_dtype == torch.bfloat16
        # config.json had valid_autoreg_steps=3; we should have overridden it.
        assert ep.valid_autoreg_steps != 3

    def test_build_wrapper_on_cpu(self):
        if not _RUN_DIR_REAL.is_dir():
            pytest.skip(f"real run dir not present: {_RUN_DIR_REAL}")
        ckpt = _RUN_DIR_REAL / "training_checkpoints" / "best_ckpt_mp0.tar"
        if not ckpt.is_file():
            pytest.skip(f"checkpoint not present: {ckpt}")
        ep = cl.load_eval_params(_RUN_DIR_REAL, K=4)
        wrapper = cl.build_wrapper_from_checkpoint(ep, ckpt, device="cpu")
        try:
            assert wrapper.model.inp_chans == 58
            assert wrapper.model.out_chans == 53
            assert next(wrapper.parameters()).device.type == "cpu"
        finally:
            del wrapper
