"""F.F helper: read ckpt_latest.tar epoch attr; touch RUN_DIR/.done if >= max_epochs.

Used by submit_train_full.slurm post-step to short-circuit chain over-submission.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

logger = logging.getLogger("check_train_complete")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--max-epochs", type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(name)s %(levelname)s %(message)s")

    ckpt = args.run_dir / "checkpoints" / "ckpt_latest.tar"
    if not ckpt.is_file():
        logger.info("No ckpt_latest.tar at %s; train hasn't progressed.", ckpt)
        return 0

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    epoch = int(state.get("iters", state.get("epoch", -1)))
    logger.info("ckpt_latest epoch=%d, max_epochs=%d", epoch, args.max_epochs)
    if epoch >= args.max_epochs:
        done = args.run_dir / ".done"
        done.touch()
        logger.info("Wrote %s — train complete; subsequent chain segments will exit early.", done)
    return 0


if __name__ == "__main__":
    sys.exit(main())
