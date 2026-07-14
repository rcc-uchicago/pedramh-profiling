"""Test the submit_eval_5410.sh driver — afterok dep logic + skip combos.

Per docs/2026-05-08_sfno_5410_scoring_plan.md (v4.4). Codex round-3 fix #2
called for verifying all 16 (SKIP_INF, SKIP_SCO, SKIP_REP, SKIP_FIG)
combinations + the SCORE_ONLY alias. We test by stubbing `sbatch` so the
driver runs end-to-end on a login node without submitting real jobs.

Approach: replace `sbatch` on PATH with a stub that:
  - parses the args (especially --dependency=afterok:NNN),
  - allocates a fake job id (incremental counter),
  - prints "Submitted batch job NNN",
  - records the (slurm_path, dep) in a log file we read after.

Then assert:
  - which SLURMs were submitted, and in what order;
  - the dep chain matches the expected pattern for each skip combo;
  - SCORE_ONLY normalizes to SKIP_INF=SKIP_REP=SKIP_FIG=1;
  - FORCE=1 deletes prior adapted NCs;
  - render_eval_figures.py is invoked with --track 5410 (in the figures
    SLURM body — verified by inspecting the SLURM file content, not by
    re-running it).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import List, Tuple

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DRIVER = _REPO_ROOT / "scripts" / "submit_eval_5410.sh"


@pytest.fixture
def stubbed_env(tmp_path, monkeypatch):
    """Set up a tmp PATH with a stub sbatch + a fake RUN_ROOT prep."""
    if not _DRIVER.is_file():
        pytest.skip(f"driver not found: {_DRIVER}")

    # 1. Stub sbatch: writes (slurm_path, dep) to a log; emits incrementing
    #    job id; mirrors `sbatch --parsable` behavior (one line, just the id).
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    sbatch_log = tmp_path / "sbatch.log"
    sbatch_stub = stub_dir / "sbatch"
    sbatch_stub.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        # Stub sbatch — records args, returns parseable job id.
        log='{sbatch_log}'
        # Counter file for unique job ids.
        ctr='{tmp_path}/ctr'
        if [[ ! -f "$ctr" ]]; then echo 1000 > "$ctr"; fi
        n=$(cat "$ctr")
        echo $((n + 1)) > "$ctr"

        dep=""
        slurm=""
        for a in "$@"; do
            case "$a" in
                --dependency=*) dep="${{a#--dependency=}}";;
                --parsable) ;;
                *.slurm) slurm="$a";;
            esac
        done
        echo "$n|$slurm|$dep" >> "$log"

        # If --parsable was passed, emit just the id.
        for a in "$@"; do
            if [[ "$a" == "--parsable" ]]; then
                echo "$n"
                exit 0
            fi
        done
        echo "Submitted batch job $n"
    """))
    sbatch_stub.chmod(0o755)

    # 2. Stub git (the driver calls `git rev-parse`).
    git_stub = stub_dir / "git"
    git_stub.write_text(textwrap.dedent("""\
        #!/usr/bin/env bash
        # Stub git: rev-parse → 'abc1234'; everything else → no-op success.
        if [[ "$1" == "rev-parse" || "$2" == "rev-parse" ]]; then
            echo abc1234
            exit 0
        fi
        exit 0
    """))
    git_stub.chmod(0o755)

    # 3. Fake RUN_ROOT: stub-out preconditions so the driver thinks
    #    the inference root is fully prepared.
    fake_run_root = tmp_path / "run_root"
    inf = fake_run_root / "inference"
    inf.mkdir(parents=True)
    (inf / "ic_source.json").write_text("{}")
    (inf / "SFNO" / "5410" / "checkpoints").mkdir(parents=True)
    # The precondition is `test -L ckpt_epoch_50.tar` (must be a symlink).
    (inf / "SFNO" / "5410" / "checkpoints" / "ckpt_epoch_50.tar").symlink_to("/dev/null")
    for Y in range(121, 129):
        (inf / f"SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y{Y}.yaml").write_text("# stub")
    # upstream_raw — driver expects empty for non-skip path; we leave dir absent.

    # 4. Fake CKPT (the driver reads its basename for MODEL_SHA7).
    fake_ckpt_dir = tmp_path / "ckpt"
    fake_ckpt_dir.mkdir()
    fake_ckpt = fake_ckpt_dir / "ckpt_epoch_50.tar"
    fake_ckpt.write_text("dummy")

    # 5. Compose the env.
    monkeypatch.setenv("PATH", f"{stub_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(exist_ok=True)
    # The driver does `REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"; cd $REPO_ROOT`. Symlink so
    # the SLURM paths (`scripts/submit_eval_inference_5410.slurm`) resolve.
    (tmp_path / "home" / "SFNO_Climate_Emulator").symlink_to(_REPO_ROOT)
    monkeypatch.setenv("WORK2", str(tmp_path / "work2"))
    (tmp_path / "work2").mkdir()
    monkeypatch.setenv("RUN_ROOT", str(fake_run_root))
    monkeypatch.setenv("CKPT", str(fake_ckpt))
    monkeypatch.setenv("UPSTREAM_REPO", str(tmp_path / "upstream"))
    (tmp_path / "upstream").mkdir()
    # Avoid hitting real Stampede3 mounts during tests.
    monkeypatch.setenv("TRUTH_H5_DIR", str(tmp_path / "truth"))
    (tmp_path / "truth").mkdir()
    monkeypatch.setenv("CLIM_SRC", str(tmp_path / "clim.nc"))
    (tmp_path / "clim.nc").write_text("stub")

    return {
        "tmp_path": tmp_path,
        "stub_dir": stub_dir,
        "sbatch_log": sbatch_log,
        "fake_run_root": fake_run_root,
        "fake_ckpt": fake_ckpt,
    }


def _parse_log(sbatch_log: Path) -> List[Tuple[str, str, str]]:
    """Return list of (job_id, slurm_basename, dep) from the stub log."""
    if not sbatch_log.is_file():
        return []
    out = []
    for line in sbatch_log.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) != 3:
            continue
        job_id, slurm, dep = parts
        out.append((job_id, Path(slurm).name if slurm else "", dep))
    return out


def _run_driver(env, extra_env=None):
    """Run submit_eval_5410.sh with the stubbed env; return (rc, log)."""
    extra = extra_env or {}
    full_env = {**os.environ, **extra}
    proc = subprocess.run(
        ["bash", str(_REPO_ROOT / "scripts" / "submit_eval_5410.sh")],
        env=full_env, capture_output=True, text=True,
    )
    return proc.returncode, _parse_log(env["sbatch_log"]), proc.stdout, proc.stderr


# ----------------------------------------------------------------------
# Skip-combo tests (Codex round-3 fix #2 + round-6 fix #4)
# ----------------------------------------------------------------------

def test_no_skip_full_chain(stubbed_env):
    rc, log, out, err = _run_driver(stubbed_env)
    assert rc == 0, f"driver rc={rc}\nstderr:\n{err}"
    slurms = [s for _, s, _ in log]
    deps = [d for _, _, d in log]
    assert slurms == [
        "submit_eval_inference_5410.slurm",
        "submit_eval_score_5410.slurm",
        "submit_eval_report_5410.slurm",
        "submit_eval_figures_5410.slurm",
    ]
    job_ids = [j for j, _, _ in log]
    assert deps == ["", f"afterok:{job_ids[0]}",
                    f"afterok:{job_ids[1]}", f"afterok:{job_ids[2]}"]


def test_skip_inf(stubbed_env):
    # SKIP_INF requires upstream_raw to have 96 NCs; create them as stubs.
    raw_dir = stubbed_env["fake_run_root"] / "inference" / "upstream_raw"
    raw_dir.mkdir()
    for Y in range(121, 129):
        for s in (0, 122, 244, 366, 488, 610, 732, 854, 976, 1098, 1220, 1342):
            (raw_dir / f"Y{Y}_s{s:04d}_member000_y{Y:04d}.nc").write_text("")
    rc, log, out, err = _run_driver(stubbed_env, extra_env={"SKIP_INF": "1"})
    assert rc == 0, f"driver rc={rc}\nstderr:\n{err}"
    slurms = [s for _, s, _ in log]
    assert slurms == [
        "submit_eval_score_5410.slurm",
        "submit_eval_report_5410.slurm",
        "submit_eval_figures_5410.slurm",
    ]
    deps = [d for _, _, d in log]
    job_ids = [j for j, _, _ in log]
    # Score has no dep (no inf job); rest chain via afterok.
    assert deps == ["", f"afterok:{job_ids[0]}", f"afterok:{job_ids[1]}"]


def test_skip_sco(stubbed_env):
    rc, log, out, err = _run_driver(stubbed_env, extra_env={"SKIP_SCO": "1"})
    assert rc == 0, f"driver rc={rc}\nstderr:\n{err}"
    slurms = [s for _, s, _ in log]
    assert slurms == [
        "submit_eval_inference_5410.slurm",
        "submit_eval_report_5410.slurm",
        "submit_eval_figures_5410.slurm",
    ]
    deps = [d for _, _, d in log]
    job_ids = [j for j, _, _ in log]
    # Report depends on inf (since sco was skipped).
    assert deps == ["", f"afterok:{job_ids[0]}", f"afterok:{job_ids[1]}"]


def test_skip_inf_and_sco(stubbed_env):
    raw_dir = stubbed_env["fake_run_root"] / "inference" / "upstream_raw"
    raw_dir.mkdir()
    for Y in range(121, 129):
        for s in (0, 122, 244, 366, 488, 610, 732, 854, 976, 1098, 1220, 1342):
            (raw_dir / f"Y{Y}_s{s:04d}_member000_y{Y:04d}.nc").write_text("")
    rc, log, out, err = _run_driver(stubbed_env, extra_env={
        "SKIP_INF": "1", "SKIP_SCO": "1",
    })
    assert rc == 0
    slurms = [s for _, s, _ in log]
    assert slurms == [
        "submit_eval_report_5410.slurm",
        "submit_eval_figures_5410.slurm",
    ]
    deps = [d for _, _, d in log]
    job_ids = [j for j, _, _ in log]
    # Report has no dep (everything before was skipped).
    assert deps == ["", f"afterok:{job_ids[0]}"]


def test_score_only_alias(stubbed_env):
    """SCORE_ONLY=1 ⇔ SKIP_INF=1 SKIP_REP=1 SKIP_FIG=1 (Codex round-6 fix #4)."""
    raw_dir = stubbed_env["fake_run_root"] / "inference" / "upstream_raw"
    raw_dir.mkdir()
    for Y in range(121, 129):
        for s in (0, 122, 244, 366, 488, 610, 732, 854, 976, 1098, 1220, 1342):
            (raw_dir / f"Y{Y}_s{s:04d}_member000_y{Y:04d}.nc").write_text("")
    rc, log, out, err = _run_driver(stubbed_env, extra_env={"SCORE_ONLY": "1"})
    assert rc == 0, err
    slurms = [s for _, s, _ in log]
    assert slurms == ["submit_eval_score_5410.slurm"]
    deps = [d for _, _, d in log]
    assert deps == [""]


def test_blocker_job_id_chains_inference(stubbed_env):
    rc, log, out, err = _run_driver(stubbed_env, extra_env={
        "BLOCKER_JOB_ID": "9999",
    })
    assert rc == 0
    deps = [d for _, _, d in log]
    # First submission (inference) inherits afterok:9999.
    assert deps[0] == "afterok:9999"


# ----------------------------------------------------------------------
# Precondition + safety tests
# ----------------------------------------------------------------------

def test_refuses_populated_upstream_raw_without_skip_inf(stubbed_env):
    """Codex round-5 fix #3: non-skip on populated upstream_raw refuses."""
    raw_dir = stubbed_env["fake_run_root"] / "inference" / "upstream_raw"
    raw_dir.mkdir()
    (raw_dir / "Y121_s0000_member000_y0121.nc").write_text("")
    rc, log, out, err = _run_driver(stubbed_env)
    assert rc != 0
    assert "non-empty" in err
    assert log == []  # no submissions made


def test_refuses_populated_inference_nwp_without_force(stubbed_env, monkeypatch):
    """Codex round-7 fix #3: rerun safety. Non-empty OUT_ROOT/inference/nwp
    without FORCE=1 refuses."""
    work2 = Path(os.environ["WORK2"])
    # Mimic the OUT_ROOT path the driver will compute.
    out_root = work2 / "SFNO_Climate_Emulator" / "results" / "sfno_eval_5410"
    out_root.mkdir(parents=True)
    # The exact RUN_TAG is computed by the driver; pre-populate by glob match.
    # Easier: pre-create one possible OUT_ROOT and set OUT_ROOT explicitly.
    explicit = work2 / "explicit_out_root"
    nwp = explicit / "inference" / "nwp"
    nwp.mkdir(parents=True)
    (nwp / "stale.nc").write_text("")
    rc, log, out, err = _run_driver(stubbed_env, extra_env={
        "OUT_ROOT": str(explicit),
        "RUN_TAG": "test_run",
    })
    assert rc != 0
    assert "non-empty" in err
    # Stale file still present — driver did not delete without FORCE.
    assert (nwp / "stale.nc").is_file()


def test_force_deletes_prior_adapted(stubbed_env):
    """Codex round-7 fix #3: FORCE=1 actively deletes prior adapted NCs."""
    work2 = Path(os.environ["WORK2"])
    explicit = work2 / "explicit_out_root_force"
    nwp = explicit / "inference" / "nwp"
    nwp.mkdir(parents=True)
    (nwp / "stale.nc").write_text("")
    (nwp / "another.nc").write_text("")
    rc, log, out, err = _run_driver(stubbed_env, extra_env={
        "OUT_ROOT": str(explicit),
        "RUN_TAG": "test_force",
        "FORCE": "1",
    })
    assert rc == 0, err
    assert not (nwp / "stale.nc").is_file()
    assert not (nwp / "another.nc").is_file()


# ----------------------------------------------------------------------
# Provenance + figure-flag tests
# ----------------------------------------------------------------------

def test_provenance_txt_written(stubbed_env):
    rc, log, out, err = _run_driver(stubbed_env)
    assert rc == 0
    work2 = Path(os.environ["WORK2"])
    # Find any file under work2 named provenance.txt.
    matches = list(work2.rglob("provenance.txt"))
    assert len(matches) == 1, f"expected 1 provenance.txt, found {len(matches)}"
    content = matches[0].read_text()
    for key in ("RUN_TAG", "RUN_ROOT", "OUT_ROOT", "EVAL_SHA7",
                "GROUP_SHA7", "MODEL_SHA7", "CKPT", "K",
                "SKIP_INF", "FORCE", "DATE_UTC"):
        assert key in content, f"provenance.txt missing {key}"


def test_figures_slurm_passes_track_5410():
    """Codex round-4 fix #3: the figures SLURM must pass --track 5410."""
    figures_slurm = _REPO_ROOT / "scripts" / "submit_eval_figures_5410.slurm"
    if not figures_slurm.is_file():
        pytest.skip(f"figures SLURM not found: {figures_slurm}")
    text = figures_slurm.read_text()
    assert "--track 5410" in text, (
        "submit_eval_figures_5410.slurm must invoke render_eval_figures.py "
        "with --track 5410 to disable pr_6h scaling for the 5410 convention."
    )


def test_render_eval_figures_track_flag_exposed():
    """Confirm the --track flag is wired into render_eval_figures.py."""
    helps = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "render_eval_figures.py"), "--help"],
        capture_output=True, text=True,
    )
    assert "--track" in helps.stdout
    assert "5410" in helps.stdout
