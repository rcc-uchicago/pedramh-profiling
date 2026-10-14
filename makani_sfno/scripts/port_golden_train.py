#!/usr/bin/env python3
"""port_golden_train.py — makani-port M0/M5: ``train_plasim.main()`` with a recorded per-step trace.

Runs the production entrypoint unchanged, with two OBSERVATION-only hooks on the stock training
step (neither alters a value):

* a forward hook on ``trainer.loss_obj`` records every training-mode loss (``float.hex``);
* ``deterministic_trainer.clip_grads`` is wrapped to record the grad norm it returns and whether
  the norm exceeded ``optimizer_max_grad_norm`` (i.e. whether clipping changed the update).

``log_epoch`` records the epoch's train and validation loss. After construction (model built,
warm start applied) the trainer reseeds python/numpy/torch with ``GOLDEN_SEED`` and selects
deterministic cuDNN, so the shuffle order depends on the seed alone and not on how many random
draws model initialisation consumed (api_delta N4).

Environment: ``GOLDEN_TRACE_OUT`` (JSON path, required), ``GOLDEN_SEED`` (default 1234).
All CLI arguments are passed through to ``train_plasim``. Prints ``PORT_GOLDEN_TRACE_WRITTEN``.

Optional knobs for the m5 prereg Part 0 pre-tests (``makani_port/m5_train_prereg.md``). With none
set, the run and the trace JSON are exactly the M0 golden ones:

* ``GOLDEN_MAX_GRAD_NORM`` -- override ``optimizer_max_grad_norm`` (P0.1 N2-PRE: 1e-3);
* ``GOLDEN_CLIP_ARITH=main`` -- swap the pin's ``clip_grads`` for a verbatim copy of makani
  main's (``a0aa4c4f`` ``training_helpers.py:123-180``; P0.1 arm b). This one changes the update
  whenever clipping fires: that is the measurement;
* ``GOLDEN_PERTENSOR_OUT`` -- write the step-1 per-tensor fp64 grad L2 norms, by name, before
  clipping (P0.6, ``trace_pertensor_ref.json``);
* ``GOLDEN_FOREACH_CHECK=1`` -- at step 1, before clipping, compare ``torch._foreach_norm`` with
  the fp64 reference per tensor and in total (P0.2), printing ``FOREACH_NORM_COMPLEX_OK``;
* ``GOLDEN_PARAMS_SHA=1`` -- record the sha256 of the parameters after training (P0.3 CRPS).

The ensemble trainer (``ensemble_size > 1``) gets the same hooks.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

OUT = Path(os.environ["GOLDEN_TRACE_OUT"])
SEED = int(os.environ.get("GOLDEN_SEED", "1234"))
TRACE = {"seed": SEED, "steps": [], "train_loss_epoch": None, "valid_loss": None, "valid_steps": None}


def _hex(x) -> str:
    return float(x).hex()


def _named_grads(model):
    out = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            out.append((name[len("module."):] if name.startswith("module.") else name, p.grad))
    return out


def _fp64_norm(g):
    """fp64 L2 norm. A complex grad goes to complex128: ``.double()`` would drop the imaginary
    part and compute exactly the real-part-only norm the P0.2 check exists to catch."""
    import torch

    return float(torch.linalg.vector_norm(g.to(torch.complex128) if g.is_complex() else g.double()))


def _write_pertensor(model, path):
    norms = {name: _hex(_fp64_norm(g)) for name, g in _named_grads(model)}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"step": 1, "norm": "fp64 L2", "n": len(norms), "grads": norms},
                                     indent=1, sort_keys=True))
    print(f"PORT_PERTENSOR_WRITTEN n={len(norms)} out={path}")


def _foreach_check(model):
    """P0.2 / row G4: torch._foreach_norm per tensor and in total vs the fp64 reference."""
    import math

    import torch

    named = _named_grads(model)
    grads = [g for _, g in named]
    n_complex = sum(1 for g in grads if g.is_complex())
    try:
        got = torch._foreach_norm(grads, 2)
    except Exception as exc:  # the check is whether it works at all
        print(f"ERROR FOREACH_NORM_COMPLEX - raised {type(exc).__name__}: {exc}")
        return
    max_rel, sq_got, sq_ref = 0.0, 0.0, 0.0
    for (name, g), f in zip(named, got):
        if f.is_complex() or not bool(torch.isfinite(f)):
            print(f"ERROR FOREACH_NORM_COMPLEX {name} dtype={f.dtype} value={f.item()}")
            return
        fv, rv = float(f), _fp64_norm(g)
        sq_got, sq_ref = sq_got + fv * fv, sq_ref + rv * rv
        if rv == 0.0:
            if abs(fv) > 1e-12:
                print(f"ERROR FOREACH_NORM_COMPLEX {name} zero_grad abs={abs(fv):.3e}")
                return
            continue
        rel = abs(fv - rv) / rv
        if rel > 1e-5:
            print(f"ERROR FOREACH_NORM_COMPLEX {name} rel={rel:.3e} complex={g.is_complex()}")
            return
        max_rel = max(max_rel, rel)
    total_rel = abs(math.sqrt(sq_got) - math.sqrt(sq_ref)) / math.sqrt(sq_ref) if sq_ref else 0.0
    if n_complex < 1:
        print(f"ERROR FOREACH_NORM_COMPLEX - n_complex=0 (the check proves nothing)")
    elif total_rel > 1e-5:
        print(f"ERROR FOREACH_NORM_COMPLEX total rel={total_rel:.3e}")
    else:
        print(f"FOREACH_NORM_COMPLEX_OK n={len(grads)} n_complex={n_complex} max_rel={max_rel:.3e} "
              f"total_rel={total_rel:.3e}")


def _main_clip_grads(model, max_grad_norm, norm_type=2.0, verbose=False):
    """Verbatim arithmetic of makani main a0aa4c4f training_helpers.py:123-180 (P0.1 arm b)."""
    import torch
    import torch.distributed as dist
    from makani.utils import comm

    with torch.no_grad():
        groups, param_map = {}, {}
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            key = tuple(g for g in getattr(param, "sharded_dims_mp", []) if g is not None and comm.get_size(g) > 1)
            groups.setdefault(key, []).append(param.grad)
            param_map.setdefault(key, []).append(name)
        ord_ = 2 if norm_type == 2.0 else 1
        partials = []
        for mp_groups, grads in groups.items():
            per_tensor_norms = torch._foreach_norm(grads, ord=ord_)
            if norm_type == 2.0:
                partial = torch.stack(per_tensor_norms).pow(2).sum()
            else:
                partial = torch.stack(per_tensor_norms).sum()
            for g in mp_groups:
                dist.all_reduce(partial, group=comm.get_group(g))
            partials.append(partial)
        total_gnorm = torch.stack(partials).sum() if partials else torch.tensor(0.0)
        if norm_type == 2.0:
            total_gnorm = total_gnorm.sqrt()
        clip_factor = torch.clamp(max_grad_norm / (total_gnorm + 1e-6), max=1.0)
        if clip_factor < 1.0:
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            torch._foreach_mul_(grads, clip_factor)
    return total_gnorm


def main() -> int:
    import hashlib

    import numpy as np
    import torch
    import sfno_training.train_plasim as tp
    from makani.utils.training import deterministic_trainer as dtm
    from makani.utils.training import ensemble_trainer as etm

    max_norm_override = os.environ.get("GOLDEN_MAX_GRAD_NORM")
    clip_arith = os.environ.get("GOLDEN_CLIP_ARITH", "stock")
    if clip_arith not in ("stock", "main"):
        raise SystemExit(f"ERROR GOLDEN_CLIP_ARITH={clip_arith} (stock|main)")
    pertensor_out = os.environ.get("GOLDEN_PERTENSOR_OUT")
    foreach_check = os.environ.get("GOLDEN_FOREACH_CHECK") == "1"
    params_sha = os.environ.get("GOLDEN_PARAMS_SHA") == "1"
    stock_clip = _main_clip_grads if clip_arith == "main" else dtm.clip_grads
    first_step_done = []

    def clip_spy(model, max_grad_norm, *a, **k):
        if not first_step_done:  # observation, before clipping touches the grads
            first_step_done.append(True)
            if pertensor_out:
                _write_pertensor(model, pertensor_out)
            if foreach_check:
                _foreach_check(model)
        gnorm = stock_clip(model, max_grad_norm, *a, **k)
        g = float(gnorm)
        if TRACE["steps"]:
            TRACE["steps"][-1].update(grad_norm=_hex(g), clipped=bool(g > float(max_grad_norm)))
        return gnorm

    dtm.clip_grads = clip_spy
    etm.clip_grads = clip_spy

    class _Golden:
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            if max_norm_override is not None:
                self.max_grad_norm = float(max_norm_override)
                self.params["optimizer_max_grad_norm"] = float(max_norm_override)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            self.loss_obj.register_forward_hook(self._record_loss)
            TRACE["flags"] = {
                "cudnn.benchmark": torch.backends.cudnn.benchmark,
                "cudnn.deterministic": torch.backends.cudnn.deterministic,
                "matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
                "amp_mode": str(self.params.get("amp_mode", "none")),
                "optimizer_max_grad_norm": float(self.params.get("optimizer_max_grad_norm", -1.0)),
                "batch_size": int(self.params.batch_size),
            }

        def _record_loss(self, module, inputs, output):
            if self.model_train.training:
                TRACE["steps"].append({"step": len(TRACE["steps"]) + 1, "loss": _hex(output.detach())})

        def log_epoch(self, train_logs, valid_logs, timing_logs):
            TRACE["train_loss_epoch"] = _hex(train_logs.get("loss", float("nan")))
            base = valid_logs.get("base", {})
            TRACE["valid_loss"] = _hex(base.get("validation loss", float("nan")))
            TRACE["valid_steps"] = int(base.get("validation steps", 0))
            if params_sha:
                h = hashlib.sha256()
                for name, p in self.model_train.named_parameters():
                    h.update(name.encode())
                    h.update(p.detach().cpu().contiguous().numpy().tobytes())
                TRACE["params_sha256_after"] = h.hexdigest()
            if max_norm_override is not None or clip_arith != "stock":
                TRACE["clip"] = {"arith": clip_arith, "max_grad_norm": _hex(self.max_grad_norm)}
            return super().log_epoch(train_logs, valid_logs, timing_logs)

    class GoldenTrainer(_Golden, tp.PlasimTrainer):
        pass

    class GoldenEnsembleTrainer(_Golden, tp.PlasimEnsembleTrainer):
        pass

    tp.PlasimTrainer = GoldenTrainer
    tp.PlasimEnsembleTrainer = GoldenEnsembleTrainer
    tp.main()

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(TRACE, indent=1, sort_keys=True))
        clipped = sum(1 for s in TRACE["steps"] if s.get("clipped"))
        print(f"PORT_GOLDEN_TRACE_WRITTEN steps={len(TRACE['steps'])} clipped={clipped} "
              f"valid_loss={TRACE['valid_loss']} valid_steps={TRACE['valid_steps']} out={OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
