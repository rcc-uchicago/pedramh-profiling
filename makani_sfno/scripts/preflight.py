#!/usr/bin/env python3
"""preflight.py — Runtime guardrail before sfno_training tiny / short launches.

Per docs/sfno_tiny_short_training_plan.md §C.5, runs five checks; exits
non-zero on any failure. Output captured to ``EXP_DIR/preflight_log.txt``.

  1. **Makani import path**: ``makani.__file__`` must resolve under
     ``makani-src/`` (catches PyPI wheel drift back to the unfixed
     ``cache_unpredicted_features`` clone).
  2. **Re-run rollout sentinel tests**: ``test_validation_rollout.py``
     and ``test_wrappers.py`` in the launch venv.
  3. **Single-batch contract dry-run**: build dataloader on the actual
     subset, build a fresh wrapper, eval-mode forward + append_history,
     assert 52→58→53 channel contract via a forward pre-hook, assert
     forcing buffer advanced to truth.
  4. **Print resolved sizes**: ``len(train_dataset)``, ``len(valid_dataset)``,
     batch counts, and resolved YAML keys.
  5. **YAML diff vs template**: assert no leftover ``{{...}}`` placeholders
     in the rendered config.

Usage::

    scripts/preflight.py \\
        --yaml_config "$EXP_DIR/plasim_sim52_tiny.rendered.yaml" \\
        --config plasim_sim52_tiny \\
        --template "$HOME/projects/SFNO_Climate_Emulator/src/sfno_training/config/plasim_sim52_tiny.yaml" \\
        --log "$EXP_DIR/preflight_log.txt"
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

logger = logging.getLogger("preflight")


# ---------------------------------------------------------------------------
# Step 1: makani import path
# ---------------------------------------------------------------------------
def check_makani_path() -> None:
    """Assert ``makani.__file__`` resolves under ``makani-src/``."""
    import makani

    makani_path = Path(makani.__file__).resolve()
    if "makani-src" not in makani_path.parts:
        raise RuntimeError(
            f"makani.__file__={makani_path} is not under a makani-src/ checkout. "
            f"PyPI wheel 0.2.0 is missing the cache_unpredicted_features clone fix. "
            f"Reinstall via: pip install --no-deps -e $HOME/projects/SFNO_Climate_Emulator/makani-src"
        )
    logger.info("[1/5] makani import path OK: %s", makani_path)


# ---------------------------------------------------------------------------
# Step 2: rollout sentinel tests
# ---------------------------------------------------------------------------
def run_sentinel_tests(repo_root: Path) -> None:
    """Re-run the 58-channel + rollout content tests in the launch venv."""
    targets = [
        "tests/sfno_training/test_validation_rollout.py",
        "tests/sfno_training/test_wrappers.py",
    ]
    cmd = [sys.executable, "-m", "pytest", *targets, "-v", "-x", "--no-header"]
    logger.info("[2/5] running sentinel tests: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"sentinel tests failed (returncode={proc.returncode}); see stdout/stderr above."
        )
    logger.info("[2/5] sentinel tests OK")


# ---------------------------------------------------------------------------
# Step 3: single-batch contract dry-run
# ---------------------------------------------------------------------------
def _build_loader_and_wrapper(
    yaml_config: Path,
    config_name: str,
    *,
    amp_mode: str = "none",
    checkpointing_level: int = 0,
    multistep_count: int = 1,
):
    """Build dataloader + freshly-instantiated wrapper from a rendered YAML.

    Mirrors what ``train_plasim.main`` does up to (and including) trainer
    construction, but stops short of starting training. Uses the
    ``PlasimTrainer`` so the patches install + hard asserts run; we then
    pull ``trainer.valid_dataloader`` and ``trainer.model_eval`` for the
    contract check.

    ``amp_mode`` and ``checkpointing_level`` mirror the values used by
    the launching submit script. Defaults preserve the original
    tiny-gate behavior (fp32, no activation checkpointing). The full
    run passes ``bf16`` + ``2`` so the memory probe represents the real
    training path (docs/sfno_full_training_plan.md §A.2).
    """
    import torch  # noqa: F401  -- ensure torch present
    from makani.utils import comm
    from makani.utils.parse_dataset_metada import parse_dataset_metadata
    from makani.utils.YParams import YParams

    from sfno_training.trainer import PlasimTrainer

    params = YParams(str(yaml_config), config_name, print_params=False)

    # Mirror train_plasim.main argument injection (single-process defaults).
    params["fin_parallel_size"] = 1
    params["fout_parallel_size"] = 1
    params["h_parallel_size"] = 1
    params["w_parallel_size"] = 1
    params["model_parallel_sizes"] = [1, 1, 1, 1]
    params["model_parallel_names"] = ["h", "w", "fin", "fout"]
    params["parameters_reduction_buffer_count"] = 1
    params["load_checkpoint"] = "legacy"
    params["save_checkpoint"] = "legacy"

    try:
        comm.init(
            model_parallel_sizes=params["model_parallel_sizes"],
            model_parallel_names=params["model_parallel_names"],
            verbose=False,
        )
    except Exception:
        # comm may already be initialized in this process; ignore.
        pass

    params["world_size"] = comm.get_world_size()
    params["global_batch_size"] = params.batch_size
    params["batch_size"] = int(params["global_batch_size"] // max(comm.get_size("data"), 1))
    params["amp_mode"] = amp_mode
    params["jit_mode"] = "none"
    params["skip_validation"] = False
    # We don't want preflight to fully start training — but PlasimTrainer
    # also doesn't start training in __init__; .train() is the kicker.
    params["skip_training"] = True
    params["enable_odirect"] = False
    params["enable_s3"] = False
    params["checkpointing_level"] = checkpointing_level
    params["enable_synthetic_data"] = False
    params["split_data_channels"] = False
    params["print_timings_frequency"] = -1
    params["multistep_count"] = multistep_count
    params["disable_ddp"] = True
    params["enable_grad_anomaly_detection"] = False

    exp_dir = Path(params.exp_dir).resolve() / config_name / "0"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "training_checkpoints").mkdir(parents=True, exist_ok=True)
    params["experiment_dir"] = str(exp_dir)
    params["checkpoint_path"] = str(
        exp_dir / "training_checkpoints" / "ckpt_mp{mp_rank}_v{checkpoint_version}.tar"
    )
    params["best_checkpoint_path"] = str(
        exp_dir / "training_checkpoints" / "best_ckpt_mp{mp_rank}.tar"
    )
    params["resuming"] = False
    if not hasattr(params, "wandb_dir") or params["wandb_dir"] is None:
        params["wandb_dir"] = str(exp_dir)

    parse_dataset_metadata(params["metadata_json_path"], params=params)

    trainer = PlasimTrainer(params, world_rank=0)
    return trainer, params


def check_single_batch_contract(trainer, params) -> None:
    """58-input / 53-output / 52-feedback / forcing-from-truth assertions."""
    import torch

    from sfno_training.models import PlasimPreprocessor

    wrapper = trainer.model_eval
    assert isinstance(wrapper.preprocessor, PlasimPreprocessor), (
        f"wrapper.preprocessor is {type(wrapper.preprocessor).__name__}, "
        f"expected PlasimPreprocessor — patch surface broken."
    )

    n_state = int(params.n_state_channels)
    n_diag = int(params.n_diagnostic_channels)
    n_forc = int(params.n_forcing_channels)
    n_in = n_state + n_forc
    n_out = n_state + n_diag

    # Pull one batch from the valid loader (avoids touching train, which
    # may be tiny but is still the live training source). Must mirror
    # what validate_one_epoch does: cache → flatten → forward.
    batch = next(iter(trainer.valid_dataloader))
    device = trainer.device
    gdata = tuple(t.to(device) for t in batch)

    wrapper.eval()
    preprocessor = wrapper.preprocessor

    internal_inputs: list[torch.Tensor] = []

    def _capture_pre_hook(_module, args):
        # Forward pre-hook on the wrapped SFNO; captures the post-concat
        # 58-channel tensor that append_unpredicted_features built.
        internal_inputs.append(args[0].detach().clone())

    handle = wrapper.model.register_forward_pre_hook(_capture_pre_hook)
    try:
        with torch.no_grad():
            inp5d, tar5d = preprocessor.cache_unpredicted_features(*gdata)
            inp_state = preprocessor.flatten_history(inp5d)

            # Sanity: pre-flatten 5D shapes from the dataloader.
            B = inp5d.shape[0]
            H = int(params.img_shape_x)
            W = int(params.img_shape_y)
            assert inp5d.shape == (B, 1, n_state, H, W), (
                f"inp5d.shape={tuple(inp5d.shape)}, expected (B, 1, {n_state}, {H}, {W})"
            )
            n_lead = tar5d.shape[1]
            assert tar5d.shape == (B, n_lead, n_out, H, W), (
                f"tar5d.shape={tuple(tar5d.shape)}, expected (B, n_lead, {n_out}, {H}, {W})"
            )
            assert inp_state.shape == (B, n_state, H, W), (
                f"inp_state.shape={tuple(inp_state.shape)} after flatten_history; "
                f"expected (B, {n_state}, {H}, {W})"
            )

            pred = wrapper(inp_state)
    finally:
        handle.remove()

    # Wrapper-internal forward must have seen the 58-channel concat.
    assert len(internal_inputs) >= 1, (
        "forward pre-hook never fired; the wrapped SFNO model wasn't called. "
        "Patch surface broken or wrapper bypassed."
    )
    internal = internal_inputs[0]
    assert internal.shape == (B, n_in, H, W), (
        f"wrapped model received internal.shape={tuple(internal.shape)}; "
        f"expected (B, {n_in}, {H}, {W}). The 52→58 forcing concat "
        f"(append_unpredicted_features) is broken."
    )

    # Outer wrapper interface is still 53-channel.
    assert pred.shape == (B, n_out, H, W), (
        f"pred.shape={tuple(pred.shape)}, expected (B, {n_out}, {H}, {W})"
    )

    # next_state feedback: 52 channels, no pr_6h leak.
    next_state = preprocessor.append_history(inp_state, pred, step=0)
    assert next_state.shape == (B, n_state, H, W), (
        f"append_history returned shape {tuple(next_state.shape)}; "
        f"expected (B, {n_state}, {H}, {W}) at n_history=0."
    )
    assert torch.equal(next_state, pred[:, :n_state]), (
        "append_history return must equal pred[:, :n_state] at n_history=0; "
        "PlasimPreprocessor strip is broken."
    )

    # Forcing buffer was advanced to the +6h target's forcing (from truth).
    forcing_after = preprocessor.unpredicted_inp_eval.detach().clone()
    expected = preprocessor.unpredicted_tar_eval[:, 0:1].detach().clone()
    assert torch.equal(forcing_after, expected), (
        "preprocessor.unpredicted_inp_eval was not advanced to "
        "unpredicted_tar_eval[:, 0:1] after append_history(step=0); "
        "rollout would feed the wrong forcing."
    )

    logger.info(
        "[3/5] contract dry-run OK (B=%d, in=%d, model_in=%d, out=%d, leads=%d)",
        B, n_state, n_in, n_out, n_lead,
    )


# ---------------------------------------------------------------------------
# Step 4: print resolved sizes
# ---------------------------------------------------------------------------
def print_resolved_sizes(trainer, params) -> None:
    """Log measured ``len(dataset)`` and ``len(dataloader)`` plus key params."""
    train_ds = trainer.train_dataset
    valid_ds = trainer.valid_dataset
    train_dl = trainer.train_dataloader
    valid_dl = trainer.valid_dataloader

    logger.info("[4/5] resolved sizes:")
    logger.info("  len(train_dataset)    = %d", len(train_ds))
    logger.info("  len(valid_dataset)    = %d", len(valid_ds))
    logger.info("  len(train_dataloader) = %d", len(train_dl))
    logger.info("  len(valid_dataloader) = %d", len(valid_dl))
    logger.info("  batch_size            = %d", int(params.batch_size))
    logger.info("  max_epochs            = %d", int(params.max_epochs))
    logger.info("  n_history / n_future  = %d / %d", int(params.n_history), int(params.n_future))
    logger.info("  valid_autoreg_steps   = %d", int(params.valid_autoreg_steps))
    logger.info("  N_in_channels         = %d", int(params.N_in_channels))
    logger.info("  N_out_channels        = %d", int(params.N_out_channels))
    logger.info("  lr                    = %g", float(params.lr))
    logger.info("  lr_warmup_steps       = %d", int(params.lr_warmup_steps))


# ---------------------------------------------------------------------------
# Step 5: YAML diff vs template
# ---------------------------------------------------------------------------
_PLACEHOLDER_RE = re.compile(r"\{\{[^}]+\}\}")


def check_rendered_yaml(rendered: Path, template: Path | None) -> None:
    """Assert the rendered YAML has no leftover ``{{PLACEHOLDER}}`` markers.

    If ``template`` is provided, also asserts that every line in the
    rendered file either matches the template or differs only on a line
    that contained a placeholder in the template (catches accidental
    sed-substitution on unintended lines).
    """
    rendered_text = rendered.read_text()
    leftover = _PLACEHOLDER_RE.findall(rendered_text)
    if leftover:
        raise RuntimeError(
            f"rendered YAML {rendered} still contains placeholder(s): {leftover[:5]}"
        )

    if template is None:
        logger.info("[5/5] rendered YAML has no leftover placeholders (template-diff skipped)")
        return

    template_text = template.read_text()
    rendered_lines = rendered_text.splitlines()
    template_lines = template_text.splitlines()
    if len(rendered_lines) != len(template_lines):
        raise RuntimeError(
            f"rendered YAML has {len(rendered_lines)} lines, template has "
            f"{len(template_lines)} — a sed substitution likely added/removed lines."
        )
    for i, (rl, tl) in enumerate(zip(rendered_lines, template_lines), start=1):
        if rl == tl:
            continue
        if not _PLACEHOLDER_RE.search(tl):
            raise RuntimeError(
                f"rendered YAML line {i} differs from template, but template "
                f"line had no placeholder:\n  template: {tl!r}\n  rendered: {rl!r}"
            )
    logger.info("[5/5] rendered YAML diff vs template OK")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="sfno_training tiny / short preflight")
    p.add_argument("--yaml_config", required=True, type=Path,
                   help="Rendered YAML config (placeholders substituted).")
    p.add_argument("--config", required=True, type=str,
                   help="YAML top-level config name (e.g. plasim_sim52_tiny).")
    p.add_argument("--template", type=Path, default=None,
                   help="Path to the unrendered template YAML for diff (optional).")
    p.add_argument("--log", type=Path, default=None,
                   help="Path to a preflight log file (optional, in addition to stdout).")
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent,
                   help="Repo root for sentinel-test cwd.")
    p.add_argument("--skip-tests", action="store_true",
                   help="Skip step 2 (sentinel pytest re-run). Use only for local debugging.")
    p.add_argument("--amp-mode", default="none", choices=["none", "fp16", "bf16"],
                   help="AMP mode for the memory-probe forward pass. Match the value the "
                        "launching submit script will pass to train_plasim.")
    p.add_argument("--checkpointing-level", default=0, type=int,
                   help="Activation-checkpointing level for the memory-probe forward pass. "
                        "Pass-through (no enum); match the value the launching submit "
                        "script will pass to train_plasim.")
    p.add_argument("--multistep-count", default=1, type=int,
                   help="Multi-step rollout depth for the memory probe (1 = single-step "
                        "default; 2 = rollout-2). Must match --multistep_count that the "
                        "launching submit script passes to train_plasim, otherwise the "
                        "memory probe exercises a different forward graph than training.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if args.log is not None:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log, mode="w"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )

    try:
        check_makani_path()

        if args.skip_tests:
            logger.warning("[2/5] sentinel tests SKIPPED (--skip-tests)")
        else:
            run_sentinel_tests(args.repo_root)

        # Step 5 (YAML check) before steps 3/4: cheap, and catches a bad
        # sed-render before we spin up the trainer.
        check_rendered_yaml(args.yaml_config, args.template)

        trainer, params = _build_loader_and_wrapper(
            args.yaml_config,
            args.config,
            amp_mode=args.amp_mode,
            checkpointing_level=args.checkpointing_level,
            multistep_count=args.multistep_count,
        )
        check_single_batch_contract(trainer, params)
        print_resolved_sizes(trainer, params)

    except Exception as e:  # noqa: BLE001  -- preflight should report any failure cleanly
        logger.error("PREFLIGHT FAILED: %s", e)
        traceback.print_exc()
        return 1

    logger.info("PREFLIGHT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
