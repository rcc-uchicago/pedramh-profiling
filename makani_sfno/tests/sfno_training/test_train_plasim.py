from __future__ import annotations

from argparse import Namespace

from makani.utils import argument_parser

from sfno_training.train_plasim import _odirect_params, _should_skip_distributed_init


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


def test_model_parallel_size_reads_either_parser(monkeypatch):
    # makani main's parser has --matmul_parallel_size and no fin/fout flags
    for name in ("WORLD_SIZE", "SLURM_NTASKS", "SLURM_NPROCS", "OMPI_COMM_WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    main_args = Namespace(disable_ddp=True, h_parallel_size=1, w_parallel_size=1, matmul_parallel_size=1)
    assert _should_skip_distributed_init(main_args) is True
    main_args.matmul_parallel_size = 2
    assert _should_skip_distributed_init(main_args) is False
    assert _should_skip_distributed_init(_args(fin_parallel_size=2)) is False


def test_odirect_params_from_the_installed_parser():
    defaults = argument_parser.get_default_argument_parser().parse_args([])
    if hasattr(defaults, "odirect_config"):  # makani main
        assert _odirect_params(defaults) == {"enable_odirect": False, "odirect_alignment": 0}
        assert _odirect_params(Namespace(odirect_config="4K")) == {"enable_odirect": True, "odirect_alignment": 4096}
    else:  # the pin: the flag passes straight through, as before
        assert _odirect_params(defaults) == {"enable_odirect": defaults.enable_odirect}
        assert _odirect_params(Namespace(enable_odirect=True)) == {"enable_odirect": True}
