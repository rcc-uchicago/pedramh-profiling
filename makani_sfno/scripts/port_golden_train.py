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


def main() -> int:
    import numpy as np
    import torch
    import sfno_training.train_plasim as tp
    from makani.utils.training import deterministic_trainer as dtm

    stock_clip = dtm.clip_grads

    def clip_spy(model, max_grad_norm, *a, **k):
        gnorm = stock_clip(model, max_grad_norm, *a, **k)
        g = float(gnorm)
        if TRACE["steps"]:
            TRACE["steps"][-1].update(grad_norm=_hex(g), clipped=bool(g > float(max_grad_norm)))
        return gnorm

    dtm.clip_grads = clip_spy

    class GoldenTrainer(tp.PlasimTrainer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
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
            return super().log_epoch(train_logs, valid_logs, timing_logs)

    tp.PlasimTrainer = GoldenTrainer
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
