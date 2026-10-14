#!/usr/bin/env python
"""Read the training config out of ai2's published ACE2-EAMv3 checkpoint.

WHY: no public config lists ACE2-EAMv3's prognostic / diagnostic / forcing
split (model card, arXiv 2505.08742 and the repo's E3SMv3 data configs were all
checked on 2026-09-24). fme checkpoints carry the stepper config, so the
checkpoint itself is the authoritative source.

    python ace2_eamv3_read_config.py <ckpt> <out.json>

Writes the config (tensors replaced by their shape) to <out.json> and prints the
channel split. PASS = ACE2_EAMV3_CONFIG_OK.  CPU only.
"""

import json
import sys

import torch


def _jsonable(x):
    if isinstance(x, torch.Tensor):
        return f"<tensor {tuple(x.shape)} {x.dtype}>"
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return f"<{type(x).__name__}>"


def _find(obj, key, path="", hits=None):
    """Every (path, value) where a dict holds `key`, anywhere in the tree."""
    hits = [] if hits is None else hits
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            if k == key:
                hits.append((p, v))
            _find(v, key, p, hits)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _find(v, key, f"{path}[{i}]", hits)
    return hits


def main():
    ckpt_path, out_path = sys.argv[1], sys.argv[2]
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as exc:  # fme stores non-tensor objects; allenai-published file
        print(f"weights_only load refused ({type(exc).__name__}); retrying full load")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    print("top-level keys:", list(ckpt) if isinstance(ckpt, dict) else type(ckpt))
    # Drop the weights before serializing: the config is what we want.
    slim = {k: v for k, v in ckpt.items() if k not in ("module", "optimization", "ema")}
    stepper = slim.get("stepper")
    if isinstance(stepper, dict):
        slim["stepper"] = {k: v for k, v in stepper.items() if k != "module"}
    with open(out_path, "w") as f:
        json.dump(_jsonable(slim), f, indent=1, sort_keys=True)
    print("wrote", out_path)

    ins, outs = _find(ckpt, "in_names"), _find(ckpt, "out_names")
    if not ins or not outs:
        print("ERROR NO_IN_OUT_NAMES: config written, but no in_names/out_names key found")
        sys.exit(1)
    print("in_names at:", [p for p, _ in ins], " out_names at:", [p for p, _ in outs])
    i, o = list(ins[0][1]), list(outs[0][1])
    prog = [n for n in i if n in o]
    forc = [n for n in i if n not in o]
    diag = [n for n in o if n not in i]
    print(f"in={len(i)} out={len(o)} prognostic={len(prog)} forcing={len(forc)} diagnostic={len(diag)}")
    print("PROGNOSTIC:", prog)
    print("FORCING:   ", forc)
    print("DIAGNOSTIC:", diag)
    for key in ("corrector", "ocean", "loss", "n_forward_steps"):
        for p, v in _find(ckpt, key)[:2]:
            print(f"{key} @ {p}: {json.dumps(_jsonable(v))[:600]}")
    print("ACE2_EAMV3_CONFIG_OK")


if __name__ == "__main__":
    main()
