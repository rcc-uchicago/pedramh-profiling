"""DDP throughput-fix I0 unit tests.

Covers (docs/2026-05-05_ddp_throughput_fix_plan.md):
  * T1 — ``_resolve_batch_sizes`` divisibility contract + happy path +
    single-rank pass-through.
  * T2 — ``_log_ddp_launch_summary`` block content (key presence; not
    parsed positionally).
  * T3b — ``plasim_sim52_zgplev_full.yaml`` loads with the post-I3
    ``batch_size`` (global=32) and ``lr`` (sqrt-scaled = 2.83e-4)
    chosen by the I2 microbench.
  * T6 — Per-rank step count under sampler+loader ``drop_last=True``
    matches ``floor(len(train) / global_batch_size)`` — the formula
    surfaced by the I0 launch summary.
  * T7 — ``PlasimTrainer.log_epoch`` backfills ``valid_logs`` keys
    consumed by the upstream rank-0 screen path so a ``--skip_validation``
    run does not raise ``KeyError`` (which previously left ranks 1-N
    blocked on the next AllReduce until NCCL timed out).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
makani = pytest.importorskip("makani")

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from makani.utils.YParams import YParams

from sfno_training.train_plasim import (
    _log_ddp_launch_summary,
    _resolve_batch_sizes,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FULL_YAML = _REPO_ROOT / "src/sfno_training/config/plasim_sim52_zgplev_full.yaml"


class _ParamsLike(dict):
    """Mimic YParams' attribute+item access for unit tests.

    YParams is a thin dict subclass (``YParams.YParams``) that supports
    both ``params.foo`` and ``params['foo']``. The two helpers under test
    use both forms, so the stub does too.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


# ---------------------------------------------------------------------------
# T1 — _resolve_batch_sizes
# ---------------------------------------------------------------------------
def test_resolve_batch_sizes_raises_on_indivisible():
    """T1a: global=15, dp=4 must fail the divisibility contract."""
    params = _ParamsLike(batch_size=15)
    with pytest.raises(AssertionError):
        _resolve_batch_sizes(params, data_parallel_size=4)


def test_resolve_batch_sizes_happy_path():
    """T1b: global=16 / dp=4 → per-rank=4 with both keys populated."""
    params = _ParamsLike(batch_size=16)
    per_rank = _resolve_batch_sizes(params, data_parallel_size=4)
    assert per_rank == 4
    assert params["global_batch_size"] == 16
    assert params["batch_size"] == 4


def test_resolve_batch_sizes_single_rank_passthrough():
    """T1c: single-rank disable_ddp / GH200 path keeps both keys at 4."""
    params = _ParamsLike(batch_size=4)
    per_rank = _resolve_batch_sizes(params, data_parallel_size=1)
    assert per_rank == 4
    assert params["global_batch_size"] == 4
    assert params["batch_size"] == 4


# ---------------------------------------------------------------------------
# T2 — launch summary content
# ---------------------------------------------------------------------------
def test_log_ddp_launch_summary_emits_all_labelled_keys(caplog):
    params = _ParamsLike(
        batch_size=4,
        global_batch_size=16,
        num_data_workers=4,
        prefetch_factor=4,
        persistent_workers=True,
        multistep_count=1,
        valid_autoreg_steps=3,
        ema={"enabled": True},
        ema_validation_period=1,
        amp_mode="bf16",
        checkpointing_level=2,
    )
    with caplog.at_level(logging.INFO, logger="sfno_training.train_plasim"):
        _log_ddp_launch_summary(params, world_size=4, data_parallel_size=4)

    text = "\n".join(rec.getMessage() for rec in caplog.records)
    expected_lines = [
        "===== DDP launch summary =====",
        "world_size                = 4",
        "data_parallel_size        = 4",
        "global_batch_size         = 16",
        "per_rank_batch_size       = 4",
        "expected_train_steps_per_epoch  = floor(len(train) / 16)",
        "num_data_workers          = 4",
        "prefetch_factor           = 4",
        "persistent_workers        = True",
        "multistep_count           = 1",
        "valid_autoreg_steps       = 3",
        "ema.enabled               = True",
        "ema_validation_period     = 1",
        "amp_mode                  = bf16",
        "checkpointing_level       = 2",
        "==============================",
    ]
    for needle in expected_lines:
        assert needle in text, f"missing summary line: {needle!r}"


def test_log_ddp_launch_summary_handles_missing_optional_keys(caplog):
    """Bare params (no EMA block, no phase-1 knobs) still emits a block.

    Defaults degrade gracefully so single-GPU / smoke runs do not crash
    inside the helper.
    """
    params = _ParamsLike(batch_size=4)
    with caplog.at_level(logging.INFO, logger="sfno_training.train_plasim"):
        _log_ddp_launch_summary(params, world_size=1, data_parallel_size=1)
    text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "===== DDP launch summary =====" in text
    assert "world_size                = 1" in text
    assert "global_batch_size         = 4" in text
    assert "ema.enabled               = False" in text


# ---------------------------------------------------------------------------
# T3b — full YAML loads with post-I3 batch_size + lr
# ---------------------------------------------------------------------------
def test_full_yaml_loads_with_post_i3_values():
    """T3b: pin the I3 production YAML edit (plan §I3).

    ``batch_size`` is the GLOBAL batch consumed by
    ``_resolve_batch_sizes`` (per-rank=8 at 4-GPU DDP). ``lr`` follows
    sqrt scaling from the prior global-4 baseline:
    ``2.83e-4 ≈ 1.0e-4 * sqrt(32/4)``.
    """
    params = YParams(str(_FULL_YAML), "plasim_sim52_zgplev_full")
    assert params.batch_size == 32
    assert params.lr == 2.83e-4


# ---------------------------------------------------------------------------
# T6 — expected-train-steps formula matches the actual sampler+loader
# ---------------------------------------------------------------------------
class _FixedLenDataset(Dataset):
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return torch.zeros(1)


@pytest.mark.parametrize(
    "length, world_size, global_batch",
    [
        (33, 4, 16),
        (145_592, 4, 16),
        (145_592, 4, 32),
        (145_592, 1, 4),  # single-GPU baseline shape
        (4_096, 4, 8),
    ],
)
def test_expected_train_steps_matches_distributedsampler(length, world_size, global_batch):
    """Per-rank step count under sampler+loader drop_last=True equals
    ``floor(length / global_batch)`` — the formula printed in the I0
    launch block."""
    per_rank_bs = global_batch // world_size
    ds = _FixedLenDataset(length)
    if world_size > 1:
        sampler = DistributedSampler(
            ds, shuffle=False, num_replicas=world_size, rank=0, drop_last=True
        )
        loader = DataLoader(ds, batch_size=per_rank_bs, sampler=sampler, drop_last=True)
    else:
        loader = DataLoader(ds, batch_size=per_rank_bs, drop_last=True)
    assert len(loader) == length // global_batch


# ---------------------------------------------------------------------------
# T7 — log_epoch backfills valid_logs under --skip_validation
# ---------------------------------------------------------------------------
def _make_bare_plasim_trainer(params):
    """Build a PlasimTrainer without running __init__ (which needs Makani
    distributed init, datasets, etc.). Only ``params`` is required by
    ``log_epoch``."""
    from sfno_training.trainer.plasim_trainer import PlasimTrainer

    inst = PlasimTrainer.__new__(PlasimTrainer)
    inst.params = params
    return inst


def test_log_epoch_backfills_valid_logs_when_skip_validation():
    """Reproduce the upstream skip_validation init and confirm the override
    inserts the keys the rank-0 screen path requires before delegating.

    Mirrors deterministic_trainer.py:374-376 (skip_validation init) and
    :709,724 (rank-0 screen reads). The parent ``log_epoch`` is patched to
    a no-op so the test does not require Makani's wandb/screen wiring.
    """
    from sfno_training.trainer.plasim_trainer import PlasimTrainer

    valid_logs = {"base": {}, "metrics": {}}
    train_logs = {"train_steps": 100, "loss": 0.5}
    timing_logs = {"training step time [ms]": 20.0}

    inst = _make_bare_plasim_trainer(_ParamsLike(batch_size=4, world_size=4))
    parent_cls = PlasimTrainer.__mro__[1]
    with patch.object(parent_cls, "log_epoch", lambda self, t, v, ti: None):
        inst.log_epoch(train_logs, valid_logs, timing_logs)

    assert valid_logs["base"]["validation steps"] == 0
    assert math.isnan(valid_logs["base"]["validation loss"])
    assert valid_logs["metrics"] == {}
    # Pre-existing samples/sec injection still fires.
    assert timing_logs["samples/sec"] == pytest.approx(4 * 4 / (20.0 / 1000.0))
    assert train_logs["samples/sec"] == timing_logs["samples/sec"]


def test_log_epoch_does_not_overwrite_existing_valid_logs():
    """When validation DID run, the override must not stomp real values."""
    from sfno_training.trainer.plasim_trainer import PlasimTrainer

    valid_logs = {
        "base": {"validation steps": 250, "validation loss": 0.123},
        "metrics": {"rmse": 0.5},
    }
    train_logs = {"train_steps": 100, "loss": 0.5}
    timing_logs = {"training step time [ms]": 20.0}

    inst = _make_bare_plasim_trainer(_ParamsLike(batch_size=4, world_size=4))
    parent_cls = PlasimTrainer.__mro__[1]
    with patch.object(parent_cls, "log_epoch", lambda self, t, v, ti: None):
        inst.log_epoch(train_logs, valid_logs, timing_logs)

    assert valid_logs["base"]["validation steps"] == 250
    assert valid_logs["base"]["validation loss"] == 0.123
    assert valid_logs["metrics"] == {"rmse": 0.5}
