"""Smoke test for the warm-start (pretrained_checkpoint_path) trainer path.

Implements the 7-assert contract from
docs/2026-05-14_v11_clip_warmstart_continuation_plan.md §4.1.2 plus the
resume-precedence variant. The asserts catch accidental optimizer /
scheduler / counter restore — which would silently turn a fresh 50-epoch
continuation into a one-tail-epoch run.

Single-rank, dummy nettype (``plasim_test_recording_dummy``) on the synthetic
packaged dataset. CPU only. No DDP, no AMP.

Asserts (clean warm-start construction):
    1. ``trainer.start_epoch == 0``
    2. ``trainer.iters == 0``
    3. ``optimizer.state`` is empty BEFORE the first step
    4. After one synthetic loss.backward() + optimizer.step(), the optimizer's
       per-param ``step`` counter is 1 (not ~891 752 — the v11_clip raw-best
       ckpt's iters)
    5. ``scheduler.last_epoch == 0``
    6. ``model.state_dict()`` is byte-equal to the loaded pretrained ckpt's
       ``model_state``
    7. ``ema._shadow`` is byte-equal to ``model.state_dict()`` (i.e. EMA seeded
       from the loaded weights, not from random init — the load-order check)

Resume-precedence variant: when ``params.resuming=True`` and an EXP_DIR
ckpt exists, the resume path wins (model_state from EXP_DIR ckpt; counters
match the EXP_DIR ckpt's iters/epoch; ``pretrained_checkpoint_path`` is
ignored).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
makani = pytest.importorskip("makani")

from makani.utils.driver import Driver  # noqa: E402

from helpers import RecordingDummyModel  # noqa: E402
from test_trainer_ci import (  # noqa: E402
    _load_yparams,
    _override_for_smoke,
    _populate_runtime_params,
)

from sfno_training.trainer import PlasimTrainer  # noqa: E402


def _build_pretrained_ckpt(
    packaged_dataset: Path,
    tmp_path: Path,
    *,
    dummy_param_value: float,
) -> tuple[Path, OrderedDict]:
    """Build a trainer once, force dummy_param to a sentinel value, and save
    a full legacy checkpoint (model+optimizer+scheduler+counters) so the
    smoke test can verify that the warm-start path bypasses optimizer/
    scheduler/counter restore even when the source ckpt contains them.

    Returns (checkpoint_path, model_state_dict_copy).
    """
    params = _load_yparams(packaged_dataset)
    src_exp_dir = tmp_path / "exp_src"
    src_exp_dir.mkdir()
    (src_exp_dir / "training_checkpoints").mkdir()
    _populate_runtime_params(params, src_exp_dir)
    _override_for_smoke(params, n_future=0)
    pt_src = PlasimTrainer(params, world_rank=0, device="cpu")

    # Drive one synthetic optimizer step so optimizer + scheduler have
    # non-empty state. If the warm-start path accidentally restored them,
    # the smoke test would observe step counter > 0 and last_epoch > 0
    # right at construction.
    pt_src.optimizer.zero_grad()
    loss = pt_src.model.model.dummy_param.sum()
    loss.backward()
    pt_src.optimizer.step()
    pt_src.scheduler.step()
    pt_src.iters = 891_751
    pt_src.epoch = 49

    # Write the sentinel AFTER the optimizer step so the saved model_state
    # carries the literal value (the step would otherwise nudge it off).
    with torch.no_grad():
        pt_src.model.model.dummy_param.fill_(dummy_param_value)

    ckpt_path = src_exp_dir / "training_checkpoints" / "pretrained_mp{mp_rank}.tar"
    Driver.save_checkpoint(
        str(ckpt_path),
        pt_src.model,
        loss=None,
        optimizer=pt_src.optimizer,
        scheduler=pt_src.scheduler,
        counters={"iters": pt_src.iters, "epoch": pt_src.epoch},
        checkpoint_mode="legacy",
    )
    ckpt_fname = Path(str(ckpt_path).format(mp_rank=0))
    assert ckpt_fname.is_file(), f"pretrained ckpt not written: {ckpt_fname}"

    saved_model_state = {
        k: v.detach().clone() for k, v in pt_src.model.state_dict().items()
    }
    return ckpt_fname, saved_model_state


def test_warmstart_clean_construction(packaged_dataset: Path, tmp_path: Path):
    """Asserts 1–7: counters fresh, optimizer/scheduler fresh, model + EMA
    seeded from the pretrained ckpt."""
    sentinel = 42.0
    ckpt_fname, saved_model_state = _build_pretrained_ckpt(
        packaged_dataset, tmp_path, dummy_param_value=sentinel
    )

    params = _load_yparams(packaged_dataset)
    warm_exp_dir = tmp_path / "exp_warm"
    warm_exp_dir.mkdir()
    (warm_exp_dir / "training_checkpoints").mkdir()
    _populate_runtime_params(params, warm_exp_dir)
    _override_for_smoke(params, n_future=0)

    # The knob under test.
    params["pretrained_checkpoint_path"] = str(ckpt_fname)
    params["resuming"] = False
    # Enable EMA so we can assert the shadow seeded from loaded weights.
    params["ema"] = {"enabled": True, "decay": 0.999, "warmup": True}

    pt = PlasimTrainer(params, world_rank=0, device="cpu")

    # (1) start_epoch fresh
    assert pt.start_epoch == 0, (
        f"start_epoch should be 0 after warm-start (got {pt.start_epoch}) — "
        "optimizer/scheduler/counters may have been restored from the "
        "pretrained ckpt (epoch=49)"
    )
    # (2) iters fresh
    assert pt.iters == 0, (
        f"iters should be 0 after warm-start (got {pt.iters}) — counter "
        "restore from the pretrained ckpt (iters=891,751) leaked through"
    )
    # (3) optimizer state empty pre-step. AdamW only populates state[*]
    # after the first step; a non-empty state here is a direct sign that
    # `optimizer.load_state_dict` ran against the pretrained ckpt's
    # optimizer_state_dict.
    assert len(pt.optimizer.state_dict()["state"]) == 0, (
        f"optimizer.state should be empty pre-step (got "
        f"{len(pt.optimizer.state_dict()['state'])} entries) — pretrained "
        "ckpt optimizer state was restored when it should not have been"
    )
    # (5) scheduler fresh (assert before stepping)
    assert pt.scheduler.last_epoch == 0, (
        f"scheduler.last_epoch should be 0 after warm-start (got "
        f"{pt.scheduler.last_epoch}) — scheduler state was restored from "
        "the pretrained ckpt"
    )
    # (6) model weights byte-equal to pretrained
    live_state = pt.model.state_dict()
    for k, v_saved in saved_model_state.items():
        v_live = live_state[k]
        assert torch.equal(v_saved, v_live), (
            f"model state[{k!r}] differs from pretrained ckpt — "
            f"warm-start did not load model_state"
        )
    # Sentinel was 42.0 — make sure the loaded weights reflect that, not the
    # fresh-random init the from-scratch path would produce.
    loaded_dummy = pt.model.model.dummy_param.detach()
    assert torch.allclose(loaded_dummy, torch.full_like(loaded_dummy, sentinel)), (
        f"dummy_param should be {sentinel} after warm-start (got "
        f"{loaded_dummy.tolist()})"
    )
    # (7) EMA shadow seeded from loaded weights (NOT from random init). If
    # EMAModel(self.model, ...) had run BEFORE the warm-start load, the
    # shadow would carry the random-init values; the loaded-vs-shadow
    # delta would be non-zero.
    assert pt.ema is not None, "EMA must be enabled for this assertion"
    # _shadow keys use the unwrapped model's prefix (see EMAModel /
    # _get_model_state_dict_prefix); compare against the same keys.
    for k, shadow_val in pt.ema._shadow.items():
        live_val = live_state[k]
        # Shadow may be cast to fp32 for fp16/bf16 lives; for our dummy
        # nettype everything is fp32 so the cast is a no-op.
        assert torch.allclose(shadow_val, live_val.to(shadow_val.dtype)), (
            f"EMA shadow[{k!r}] differs from model state — EMA was "
            "constructed BEFORE the warm-start load (wrong order)"
        )

    # (4) After one synthetic optimizer step, step==1 (not ~891,752). The
    # earlier optimizer.state empty check rules out direct state restore;
    # this check rules out a more subtle path where the step counter is
    # restored but per-param state is reinitialized lazily.
    pt.optimizer.zero_grad()
    pt.model.model.dummy_param.requires_grad_(True)
    loss = pt.model.model.dummy_param.sum()
    loss.backward()
    pt.optimizer.step()
    step_values = [s.get("step", 0) for s in pt.optimizer.state.values()]
    assert step_values, "optimizer.state should be populated post-step"
    max_step = max(int(getattr(s, "item", lambda: s)()) for s in step_values)
    assert max_step == 1, (
        f"optimizer step counter should be 1 after one update (got "
        f"{max_step}) — pretrained optimizer step counter (~891,751) "
        "leaked through"
    )


def test_resume_takes_precedence_over_warmstart(
    packaged_dataset: Path, tmp_path: Path
):
    """When params.resuming=True (set by train_plasim.py because EXP_DIR
    already has a ckpt), the resume path runs and the warm-start path is
    bypassed. Model weights come from the EXP_DIR ckpt, not from
    pretrained_checkpoint_path; counters match the EXP_DIR ckpt's
    iters/epoch."""
    sentinel_pretrained = 42.0
    sentinel_resume = 99.0

    ckpt_pretrained, _ = _build_pretrained_ckpt(
        packaged_dataset, tmp_path, dummy_param_value=sentinel_pretrained
    )

    # Build a second source trainer for the resume ckpt, with different
    # sentinel weights AND non-zero counters.
    params_src = _load_yparams(packaged_dataset)
    src_exp_dir = tmp_path / "exp_resume_src"
    src_exp_dir.mkdir()
    (src_exp_dir / "training_checkpoints").mkdir()
    _populate_runtime_params(params_src, src_exp_dir)
    _override_for_smoke(params_src, n_future=0)
    pt_resume_src = PlasimTrainer(params_src, world_rank=0, device="cpu")
    pt_resume_src.optimizer.zero_grad()
    loss = pt_resume_src.model.model.dummy_param.sum()
    loss.backward()
    pt_resume_src.optimizer.step()
    pt_resume_src.scheduler.step()
    # Sentinel must be written AFTER the optimizer step (the step would
    # otherwise nudge dummy_param off the literal).
    with torch.no_grad():
        pt_resume_src.model.model.dummy_param.fill_(sentinel_resume)

    # Stage the resume ckpt at EXP_DIR/training_checkpoints/ckpt_mp0_v0.tar
    # (the path Trainer.__init__ reads when params.resuming=True).
    resume_exp_dir = tmp_path / "exp_resume"
    resume_exp_dir.mkdir()
    (resume_exp_dir / "training_checkpoints").mkdir()
    resume_ckpt_template = (
        resume_exp_dir
        / "training_checkpoints"
        / "ckpt_mp{mp_rank}_v{checkpoint_version}.tar"
    )
    Driver.save_checkpoint(
        str(resume_ckpt_template).replace("{checkpoint_version}", "0"),
        pt_resume_src.model,
        loss=pt_resume_src.loss_obj,
        optimizer=pt_resume_src.optimizer,
        scheduler=pt_resume_src.scheduler,
        counters={"iters": 999, "epoch": 42},
        checkpoint_mode="legacy",
    )
    resume_ckpt_path = Path(
        str(resume_ckpt_template).replace("{checkpoint_version}", "0").format(mp_rank=0)
    )
    assert resume_ckpt_path.is_file()

    # Build the warm-start trainer with params.resuming=True. Both
    # pretrained_checkpoint_path AND a populated EXP_DIR are set; the
    # resume path must take precedence.
    params = _load_yparams(packaged_dataset)
    _populate_runtime_params(params, resume_exp_dir)
    _override_for_smoke(params, n_future=0)
    params["pretrained_checkpoint_path"] = str(ckpt_pretrained)
    params["resuming"] = True
    params["ema"] = {"enabled": True, "decay": 0.999, "warmup": True}

    pt = PlasimTrainer(params, world_rank=0, device="cpu")

    # Model weights come from the resume ckpt (99.0), not pretrained (42.0).
    loaded_dummy = pt.model.model.dummy_param.detach()
    assert torch.allclose(
        loaded_dummy, torch.full_like(loaded_dummy, sentinel_resume)
    ), (
        f"resume should have won: expected dummy_param={sentinel_resume}, "
        f"got {loaded_dummy.tolist()} — warm-start path fired despite "
        "resuming=True"
    )
    # Counters match the resume ckpt.
    assert pt.iters == 999, f"resume counters not restored: iters={pt.iters}"
    assert pt.start_epoch == 42, (
        f"resume counters not restored: start_epoch={pt.start_epoch}"
    )
