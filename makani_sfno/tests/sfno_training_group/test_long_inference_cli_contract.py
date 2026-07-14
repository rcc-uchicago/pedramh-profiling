"""T.1 — long_inference slurm CLI contract (Phase F.G).

Asserts the rendered torchrun command for submit_long_inference_full.slurm:
  - has NO --nc_bc_offset (hardcoded internally at long_inference.py:1267)
  - has --init_datetime "%Y-%m-%d_%H:%M:%S" with UNDERSCORE
  - has --final_datetime explicit (line 1347 fallback uses val_year_end)
  - launches via torchrun --standalone --nnodes=1 --nproc_per_node=1
  - tests both YEAR=121 (non-leap) AND YEAR=124 (leap)
"""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SLURM_PATH = REPO_ROOT / "src" / "sfno_training_group" / "slurm" / "submit_long_inference_full.slurm"


@pytest.fixture(scope="module")
def slurm_text() -> str:
    return SLURM_PATH.read_text()


@pytest.fixture(scope="module")
def slurm_code(slurm_text: str) -> str:
    """slurm_text minus #-comments (we want to check the executed commands, not the docs)."""
    out_lines = []
    for line in slurm_text.splitlines():
        stripped = line.lstrip()
        # Drop pure-comment lines (but keep #SBATCH directives).
        if stripped.startswith("#") and not stripped.startswith("#SBATCH"):
            continue
        # Strip trailing comments (best-effort: split on " #" not preceded by quote/var).
        # For our slurm we don't have inline comments after commands, so just keep line.
        out_lines.append(line)
    return "\n".join(out_lines)


def _shellcheck(slurm_text: str, year: int) -> str:
    """Render the slurm text into a shell command via env substitution.

    We can't actually run sbatch here; instead we extract the torchrun command
    region and substitute YEAR-related env vars to produce the literal command
    string a real run would invoke.
    """
    next_year = year + 1
    init_dt = f"{year:04d}-01-01_00:00:00"
    final_dt = f"{next_year:04d}-01-01_00:00:00"
    # Mock RUN_DIR / RUN_NUM / DATA_DIR / INIT_NC for substitution.
    mock = {
        "YEAR": str(year),
        "INIT_DT": init_dt,
        "FINAL_DT": final_dt,
        "RUN_DIR": "/mock/run_dir",
        "RUN_NUM": "prod_test",
        "DATA_DIR": "/mock/data_dir",
        "INIT_NC": f"/mock/init_year{year}.nc",
        "OUT_DIR": f"/mock/run_dir/inference_full_year{year}",
        "GROUP_PANGU_ROOT": "/mock/pangu",
    }
    text = slurm_text
    for k, v in mock.items():
        text = text.replace(f"${k}", v).replace(f"${{{k}}}", v)
    return text


def test_no_nc_bc_offset_flag(slurm_code: str) -> None:
    """`--nc_bc_offset` must NOT appear in executed code (it's hardcoded in
    long_inference.py:1267)."""
    assert "--nc_bc_offset" not in slurm_code, (
        "--nc_bc_offset is hardcoded at long_inference.py:1267 and is NOT a CLI flag. "
        "Passing it would cause argparse to fail."
    )


@pytest.mark.parametrize("year", [121, 124])
def test_init_datetime_underscore_format(slurm_code: str, year: int) -> None:
    """`--init_datetime` must use UNDERSCORE between date and time."""
    rendered = _shellcheck(slurm_code, year)
    expected = f'--init_datetime "{year:04d}-01-01_00:00:00"'
    assert expected in rendered, (
        f"YEAR={year}: expected substring not found in rendered slurm:\n"
        f"  expected: {expected!r}\n"
        f"  Got nearby:\n{_extract_torchrun_block(rendered)}"
    )


@pytest.mark.parametrize("year", [121, 124])
def test_final_datetime_explicit_and_year_plus_1(slurm_code: str, year: int) -> None:
    """`--final_datetime` must be `(year+1)-01-01_00:00:00` (NOT relying on fallback)."""
    rendered = _shellcheck(slurm_code, year)
    expected = f'--final_datetime "{year + 1:04d}-01-01_00:00:00"'
    assert expected in rendered, (
        f"YEAR={year}: --final_datetime missing or wrong:\n"
        f"  expected: {expected!r}\n"
        f"  Got: \n{_extract_torchrun_block(rendered)}"
    )


def test_torchrun_standalone(slurm_code: str) -> None:
    assert "torchrun --standalone --nnodes=1 --nproc_per_node=1" in slurm_code


def test_long_inference_invocation_present(slurm_code: str) -> None:
    assert 'long_inference.py' in slurm_code
    assert '--config SFNO' in slurm_code
    assert '--init_nc_filepaths' in slurm_code
    assert '--save_basename' in slurm_code


def test_no_prediction_duration_days(slurm_code: str) -> None:
    """The slurm should NOT pass --prediction_duration_days; YAML omits it."""
    assert "--prediction_duration_days" not in slurm_code
    assert "prediction_duration_days" not in slurm_code


def _extract_torchrun_block(text: str) -> str:
    """Best-effort: grab the block from 'torchrun' through the next blank line."""
    m = re.search(r"^torchrun\b.*?(?=\n\n|\Z)", text, re.S | re.M)
    return m.group(0) if m else "(no torchrun block found)"
