from __future__ import annotations

from argparse import Namespace

from sfno_training.train_plasim import _should_skip_distributed_init


def _args(**overrides):
    values = {
        "disable_ddp": True,
        "h_parallel_size": 1,
        "w_parallel_size": 1,
        "fin_parallel_size": 1,
        "fout_parallel_size": 1,
    }
    values.update(overrides)
    return Namespace(**values)


def test_single_rank_disable_ddp_skips_distributed_init(monkeypatch):
    for name in ("WORLD_SIZE", "SLURM_NTASKS", "SLURM_NPROCS", "OMPI_COMM_WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)

    assert _should_skip_distributed_init(_args()) is True

    monkeypatch.setenv("SLURM_NTASKS", "1")
    assert _should_skip_distributed_init(_args()) is True


def test_multi_rank_or_model_parallel_keeps_distributed_init(monkeypatch):
    monkeypatch.setenv("SLURM_NTASKS", "2")
    assert _should_skip_distributed_init(_args()) is False

    monkeypatch.setenv("WORLD_SIZE", "1")
    assert _should_skip_distributed_init(_args()) is False

    monkeypatch.setenv("SLURM_NTASKS", "1")
    monkeypatch.delenv("WORLD_SIZE")
    assert _should_skip_distributed_init(_args(h_parallel_size=2)) is False
    assert _should_skip_distributed_init(_args(disable_ddp=False)) is False
