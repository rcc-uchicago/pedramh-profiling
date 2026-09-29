"""Port F surgical transfer: drop channels from a trained checkpoint's per-channel rows.

Architect review §4 / handoff §6a route 2. Only three kinds of SFNO tensor carry
per-channel rows -- the encoder's first layer (N_in), the decoder's last layer
(N_out) and big_skip's residual_transform (N_out x N_in); the trunk is
embed_dim-sized. Input order is [state, forcing] (makani preprocessor
`torch.cat([x, xc])`), output order is channel_names, so for the E3SM ALLDATA
pack SOILWATER_10CM / TSOI_10CM sit at position 8 / 9 on BOTH sides.

Every dim whose size equals the old N_in (or N_out) is sliced; the sizes 107 /
101 collide with nothing else in the net (embed_dim 384). Refuses if N_in ==
N_out (positions would be ambiguous) and lists every tensor it touched.

Writes {model_state, comm_grid, sliced_from, dropped} -- only what the fork's
warm-start (plasim_trainer: restore_from_checkpoint, model only, strict) reads.

    python slice_checkpoint.py --src best_ckpt_mp0.tar --dst sliced.tar \
        --n-in 107 --n-out 101 --drop 8 9
      -> SLICE_CKPT_OK ...
"""

from __future__ import annotations

import argparse
import sys


def slice_state_dict(sd: dict, n_in: int, n_out: int, drop_in, drop_out):
    """Return (new_sd, touched) with `drop_in` removed from every n_in-sized dim
    and `drop_out` from every n_out-sized dim. Tensors are torch or numpy-like."""
    if n_in == n_out:
        raise ValueError("SLICE_AMBIGUOUS: n_in == n_out, cannot tell the dims apart")
    keep_in = [i for i in range(n_in) if i not in set(drop_in)]
    keep_out = [i for i in range(n_out) if i not in set(drop_out)]
    new, touched = {}, []
    for k, v in sd.items():
        shape = tuple(getattr(v, "shape", ()))
        out = v
        for dim, size in enumerate(shape):
            if size == n_in:
                out = out.index_select(dim, _idx(out, keep_in))
            elif size == n_out:
                out = out.index_select(dim, _idx(out, keep_out))
        if tuple(getattr(out, "shape", ())) != shape:
            touched.append((k, shape, tuple(out.shape)))
        new[k] = out
    return new, touched


def _idx(t, keep):
    import torch
    return torch.as_tensor(keep, dtype=torch.long, device=t.device)


def main() -> int:
    import torch

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--n-in", type=int, required=True)
    p.add_argument("--n-out", type=int, required=True)
    p.add_argument("--drop", type=int, nargs="+", required=True,
                   help="positions to drop; the same on the input and output side")
    a = p.parse_args()

    ck = torch.load(a.src, map_location="cpu", weights_only=False)
    sd, touched = slice_state_dict(ck["model_state"], a.n_in, a.n_out, a.drop, a.drop)
    for k, s0, s1 in touched:
        print(f"  sliced {k}: {s0} -> {s1}")
    kinds = {"in": any(a.n_in in s0 and a.n_out not in s0 for _, s0, _ in touched),
             "out": any(a.n_out in s0 and a.n_in not in s0 for _, s0, _ in touched)}
    if not (kinds["in"] and kinds["out"]):
        print(f"ERROR SLICE_INCOMPLETE: found input-side={kinds['in']} output-side="
              f"{kinds['out']} tensors; expected both (is --n-in/--n-out right?)")
        return 2
    out = {"model_state": sd, "sliced_from": a.src, "dropped": a.drop,
           "epoch": ck.get("epoch"), "iters": ck.get("iters")}
    if "comm_grid" in ck:
        out["comm_grid"] = ck["comm_grid"]
    torch.save(out, a.dst)
    print(f"SLICE_CKPT_OK tensors={len(touched)} src_epoch={ck.get('epoch')} dst={a.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
