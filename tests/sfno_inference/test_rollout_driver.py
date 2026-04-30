"""Tests for src/sfno_inference/rollout_driver.py.

Coverage (per docs/sfno_eval_plan.md §B.1-§B.3 and §H):

  - ``nwp_ic_offsets``:
      * monthly stride, all ICs satisfy s + K < n_samples;
      * rejects too-small files;
      * step is identical for n=1455 and n=1459 at K=56, n_ic=12.

  - ``rollout_one_ic`` end-to-end on a synthetic 6-step rollout with a
    stub wrapper + stub PlasimForcingDataset:
      * predictions shape (K, 53, H, W) and physical units;
      * truth shape (K, 53, H, W);
      * init_state shape (52, H, W) — no diagnostic;
      * AssertionError if the wrapper returns a wrong-channel-count tensor;
      * AssertionError if dataset.n_future does not match
        valid_autoreg_steps (regression guard for the v2.4 finding).

  - The 58→53 contract assertions inside the loop:
      * pred shape (1, 53, H, W) is enforced when ``assert_contract=True``;
      * inpt shape (1, 52, H, W) after append_history is enforced.

The tests stub PlasimPreprocessor with a behavioral lookalike so the
driver's logic is exercised without pulling in Makani's full Preprocessor2D
(which needs ``comm.init`` and a SHT grid setup).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from sfno_inference import rollout_driver as rd  # noqa: E402


# ---------------------------------------------------------------------------
# stub preprocessor + wrapper + dataset
# ---------------------------------------------------------------------------

class StubPreprocessor:
    """Minimal stand-in for PlasimPreprocessor.

    Reproduces only the methods + attrs the rollout driver touches:
      - cache_unpredicted_features(x, y, xz, yz) -> (x, y), caches xz/yz
      - flatten_history(x) — collapses dim=1 if size==1
      - append_history(inpt, pred, idt) — asserts pred has 53 channels
        and slices to first 52, returns the result reshaped like inpt.
    """

    def __init__(self, n_state: int = 52, n_full: int = 53):
        self.n_state = n_state
        self.n_full = n_full
        self.unpredicted_inp_eval = None
        self.unpredicted_tar_eval = None

    def cache_unpredicted_features(self, x, y, xz=None, yz=None):
        self.unpredicted_inp_eval = xz
        self.unpredicted_tar_eval = yz
        return x, y

    def flatten_history(self, x):
        # x: (B, T, C, H, W) → (B, T*C, H, W), but in our tests T is 1.
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            return x.reshape(B, T * C, H, W)
        return x

    def append_history(self, inpt, pred, idt):
        assert pred.dim() == 4
        assert pred.shape[1] in (self.n_state, self.n_full), (
            f"append_history: expected {self.n_state} or {self.n_full} channels, "
            f"got {pred.shape[1]}"
        )
        if pred.shape[1] == self.n_full:
            pred = pred[:, : self.n_state, ...]
        return pred  # next-step input is the new state


class StubWrapper(torch.nn.Module):
    """SFNO surrogate that returns ``(B, 53, H, W)`` filled with a constant.

    The constant lets the test verify that forward was called K times
    (constant per-step indexed by ``self.step_counter``).
    """

    def __init__(self, n_in: int = 58, n_out: int = 53,
                 H: int = 8, W: int = 16):
        super().__init__()
        self.preprocessor = StubPreprocessor()
        self.n_in = n_in
        self.n_out = n_out
        self.H = H
        self.W = W
        # A trainable parameter so .parameters() and .to(device) work.
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.step_counter = 0

    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        # Mimic: input is (B, 52, H, W); output is (B, 53, H, W).
        # For a real wrapper, the (B, 58, H, W) → (B, 53, H, W) mapping
        # happens *inside* via append_unpredicted_features. Our stub
        # accepts (B, 52, H, W) directly.
        B, _, H, W = inpt.shape
        out = torch.full(
            (B, self.n_out, H, W),
            float(self.step_counter),
            dtype=torch.float32,
            device=inpt.device,
        )
        self.step_counter += 1
        return out


class StubDataset:
    """Behavioral stand-in for PlasimForcingDataset.

    Returns a fixed 4-tuple at any global_idx; n_future is configurable.
    """

    def __init__(self, *, K: int, H: int = 8, W: int = 16,
                 n_state: int = 52, n_full: int = 53, n_forcing: int = 6):
        self.n_future = K - 1
        self.n_history = 0
        self.dt = 1
        self.H = H
        self.W = W
        self.n_state = n_state
        self.n_full = n_full
        self.n_forcing = n_forcing

        # Deterministic z-scored values for reproducibility.
        rng = np.random.default_rng(42)
        self._inp_state = torch.from_numpy(
            rng.standard_normal((1, n_state, H, W), dtype=np.float32)
        )
        self._tar = torch.from_numpy(
            rng.standard_normal((K, n_full, H, W), dtype=np.float32)
        )
        self._inp_forcing = torch.from_numpy(
            rng.standard_normal((1, n_forcing, H, W), dtype=np.float32)
        )
        self._tar_forcing = torch.from_numpy(
            rng.standard_normal((K, n_forcing, H, W), dtype=np.float32)
        )

        # Bias/scale for in_state — used by the driver to de-z-score.
        self.in_bias = np.zeros(n_state, dtype=np.float32)
        self.in_scale = np.ones(n_state, dtype=np.float32)

        # Provenance attrs needed by _resolve_ic_provenance.
        self.files_paths = ["/dev/null/MOST.0121.h5"]
        self.file_offsets = [0]

    def __getitem__(self, idx):
        return (
            self._inp_state.clone(),
            self._tar.clone(),
            self._inp_forcing.clone(),
            self._tar_forcing.clone(),
        )

    def _get_indices(self, global_idx: int):
        return 0, global_idx


def _make_eval_params(K: int, *, amp_enabled: bool = False, run_dir=None):
    """Return a SimpleNamespace standing in for ParamsBase."""
    p = SimpleNamespace(
        valid_autoreg_steps=K - 1,
        n_future=K - 1,
        n_history=0,
        N_in_channels=58,
        N_out_channels=53,
        n_state_channels=52,
        n_diagnostic_channels=1,
        n_forcing_channels=6,
        amp_enabled=amp_enabled,
        amp_dtype=torch.float32,
        global_means_path=None,
        global_stds_path=None,
    )
    if run_dir is not None:
        p.global_means_path = str(run_dir / "global_means.npy")
        p.global_stds_path = str(run_dir / "global_stds.npy")
    return p


# ---------------------------------------------------------------------------
# nwp_ic_offsets
# ---------------------------------------------------------------------------

class TestNwpIcOffsets:
    def test_non_leap_file(self):
        offsets = rd.nwp_ic_offsets(1455, K=56, n_ic=12)
        assert len(offsets) == 12
        assert offsets[0] == 0
        for s in offsets:
            assert s + 56 < 1455

    def test_leap_file(self):
        offsets = rd.nwp_ic_offsets(1459, K=56, n_ic=12)
        assert len(offsets) == 12
        assert offsets[0] == 0
        for s in offsets:
            assert s + 56 < 1459

    def test_step_identical_for_leap_and_non_leap(self):
        """At K=56, n_ic=12, step should be 116 for both file lengths."""
        non_leap = rd.nwp_ic_offsets(1455, K=56, n_ic=12)
        leap = rd.nwp_ic_offsets(1459, K=56, n_ic=12)
        # First and last IC should match because step=116 in both cases.
        assert non_leap == leap
        # And step is 116.
        if len(non_leap) > 1:
            assert non_leap[1] - non_leap[0] == 116

    def test_too_small_file_rejected(self):
        with pytest.raises(ValueError, match="too small"):
            rd.nwp_ic_offsets(60, K=56, n_ic=12)


# ---------------------------------------------------------------------------
# rollout_one_ic — end-to-end
# ---------------------------------------------------------------------------

class TestRolloutOneIc:
    def _setup(self, K=6, tmp_path=None):
        H, W = 8, 16
        wrapper = StubWrapper(H=H, W=W)
        dataset = StubDataset(K=K, H=H, W=W)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        np.save(run_dir / "global_means.npy", np.zeros(53, dtype=np.float32))
        np.save(run_dir / "global_stds.npy", np.ones(53, dtype=np.float32))
        eval_params = _make_eval_params(K, run_dir=run_dir)
        return wrapper, dataset, eval_params, H, W

    def test_basic_rollout_shapes(self, tmp_path, monkeypatch):
        K = 6
        wrapper, dataset, eval_params, H, W = self._setup(K, tmp_path)

        # Stub h5py.File for _resolve_ic_provenance.
        _patch_h5_resolution(monkeypatch)

        result = rd.rollout_one_ic(
            wrapper=wrapper,
            dataset=dataset,
            ic_global_idx=0,
            eval_params=eval_params,
            device="cpu",
        )

        assert result.K == K
        assert result.prediction.shape == (K, 53, H, W)
        assert result.truth.shape == (K, 53, H, W)
        assert result.init_state.shape == (52, H, W)
        # Wrapper was called K times (verified via step_counter).
        assert wrapper.step_counter == K

    def test_predictions_carry_step_indexed_constant(self, tmp_path, monkeypatch):
        """The stub wrapper returns a constant equal to its step_counter.

        After de-z-scoring (bias=0, scale=1), each lead-time slice should
        contain a single constant matching its index.
        """
        K = 6
        wrapper, dataset, eval_params, H, W = self._setup(K, tmp_path)
        _patch_h5_resolution(monkeypatch)

        result = rd.rollout_one_ic(
            wrapper=wrapper, dataset=dataset, ic_global_idx=0,
            eval_params=eval_params, device="cpu",
        )
        for k in range(K):
            assert torch.allclose(
                result.prediction[k], torch.full((53, H, W), float(k))
            ), f"lead {k} has unexpected values"

    def test_assertion_blocks_n_future_drift(self, tmp_path, monkeypatch):
        """v2.4 regression guard: dataset.n_future must equal K-1."""
        K = 6
        wrapper, dataset, eval_params, H, W = self._setup(K, tmp_path)
        _patch_h5_resolution(monkeypatch)
        # Sabotage: leave eval_params at K-1 but force the dataset to n_future=2.
        dataset.n_future = 2
        with pytest.raises(RuntimeError, match="n_future"):
            rd.rollout_one_ic(
                wrapper=wrapper, dataset=dataset, ic_global_idx=0,
                eval_params=eval_params, device="cpu",
            )

    def test_assertion_blocks_wrong_channel_count(self, tmp_path, monkeypatch):
        """assert_contract=True catches a model that returns the wrong shape."""
        K = 4
        wrapper, dataset, eval_params, H, W = self._setup(K, tmp_path)
        _patch_h5_resolution(monkeypatch)

        # Sabotage: wrapper returns 50 channels instead of 53.
        original_forward = wrapper.forward

        def bad_forward(inpt):
            out = original_forward(inpt)
            return out[:, :50, ...]

        wrapper.forward = bad_forward  # type: ignore[method-assign]

        with pytest.raises(AssertionError, match="pred shape"):
            rd.rollout_one_ic(
                wrapper=wrapper, dataset=dataset, ic_global_idx=0,
                eval_params=eval_params, device="cpu",
            )

    def test_pr6h_never_appears_in_inpt(self, tmp_path, monkeypatch):
        """53→52 slice happens inside append_history; channel 52 must never feed back.

        We hook the stub wrapper to record the inpt on every call, then
        check that no inpt has more than 52 channels.
        """
        K = 5
        wrapper, dataset, eval_params, H, W = self._setup(K, tmp_path)
        _patch_h5_resolution(monkeypatch)

        seen_shapes: list[tuple[int, ...]] = []
        original_forward = wrapper.forward

        def recording_forward(inpt):
            seen_shapes.append(tuple(inpt.shape))
            return original_forward(inpt)

        wrapper.forward = recording_forward  # type: ignore[method-assign]

        rd.rollout_one_ic(
            wrapper=wrapper, dataset=dataset, ic_global_idx=0,
            eval_params=eval_params, device="cpu",
        )
        for shape in seen_shapes:
            assert shape[1] == 52, (
                f"inpt had {shape[1]} channels — pr_6h leaked into feedback!"
            )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _patch_h5_resolution(monkeypatch):
    """Stub h5py.File so _resolve_ic_provenance does not need a real file."""

    class _StubH5:
        def __init__(self, *a, **k):
            self.attrs = {"plasim_time_units": "days since 0126-08-01 00:00:00"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __getitem__(self, key):
            if key == "time_plasim":
                return np.zeros(2000, dtype=np.float64)
            raise KeyError(key)

    import h5py
    monkeypatch.setattr(h5py, "File", _StubH5)
