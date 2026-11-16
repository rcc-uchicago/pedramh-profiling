"""rollout_dose.py -- autoregressive rollouts of one gate-D arm (item 4), and its one-step predictions.

  python rollout_dose.py --frames <dir> --arm <arm dir> --which best|last [--val-members 4] [--steps 2500]
                         [--n-ic 16] [--n-pred 256]

The model is the arm's best.pt or last.pt. Each 0.02 step is x <- model(x), where model(x) = x + net(x).

**Initial conditions.** --n-ic held-out frames spread evenly over the val members' stored segments
(dose_common.rollout_ics). They are the same for every arm.

**Per step and initial condition, recorded:**
- the 8 band energies (per-pixel mean square, o1.2 units);
- the field's max |x|.

**Blow-ups.** A rollout counts as blown up at the first step where it is non-finite, or where max|x| exceeds 50× the
initial frames' max|x|. From then on it is NaN.

**One-step predictions** model(x) on the first --n-pred fixed held-out val pairs (dose_common.val_pairs). They are
used for the absorption slope.

Writes <arm>/rollout_<which>.npz. Prints ROLLOUT_DONE.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SQG = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
sys.path.insert(0, SQG)
import dose_common as DC                                             # noqa: E402
import train_dose as TD                                              # noqa: E402  (architecture constants, fp32)

BLOW_FACTOR = 50.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--which", choices=("best", "last"), required=True)
    ap.add_argument("--val-members", default="4")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--n-ic", type=int, default=16)
    ap.add_argument("--n-pred", type=int, default=256)
    ap.add_argument("--n-val", type=int, default=512)
    a = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F
    from model_UViT import ConvNextUViT
    device = torch.device("cuda:0")
    TD.strict_fp32(torch, F, device)
    G = DC.geometry()
    _, band_energy = DC.torch_tools(torch, G, device)
    model = ConvNextUViT(in_ch=1, out_ch=1, n_feat=TD.N_FEAT, depths=TD.DEPTHS, kernel_size=TD.KERNEL,
                         drop_path=TD.DROP_PATH, n_attn=TD.N_ATTN, n_heads=TD.N_HEADS, residual=True).to(device)
    ck = torch.load(os.path.join(a.arm, f"{a.which}.pt"), map_location="cpu")
    model.load_state_dict(ck["model"])
    model.eval()
    va_members = [int(m) for m in a.val_members.split(",")]
    _, va_arr = DC.open_segments(a.frames, va_members)
    x0, tags = DC.rollout_ics(va_arr, a.n_ic)
    t0 = time.time()
    with torch.no_grad():
        x = torch.from_numpy(x0)[:, None].to(device)
        n = x.shape[0]
        cap = BLOW_FACTOR * float(x.abs().amax())
        E = np.full((a.steps + 1, n, len(DC.BANDS)), np.nan)
        xmax = np.full((a.steps + 1, n), np.nan)
        blow = np.full(n, -1, dtype=np.int64)
        alive = torch.ones(n, dtype=torch.bool, device=device)
        E[0] = band_energy(x).cpu().numpy()
        xmax[0] = x.abs().amax(dim=(1, 2, 3)).cpu().numpy()
        for s in range(1, a.steps + 1):
            x = model(x)
            mx = x.abs().amax(dim=(1, 2, 3))
            bad = alive & (~torch.isfinite(mx) | (mx > cap))
            if bool(bad.any()):
                for i in torch.nonzero(bad).flatten().tolist():
                    blow[i] = s
                alive &= ~bad
                x = torch.where(alive[:, None, None, None], x, torch.zeros_like(x))
            if not bool(alive.any()):
                break
            e = band_energy(x).cpu().numpy()
            m = mx.cpu().numpy()
            al = alive.cpu().numpy()
            E[s, al] = e[al]
            xmax[s, al] = m[al]
        xv, yv, pick = DC.val_pairs(va_arr, a.n_val)
        xv, yv = xv[:a.n_pred], yv[:a.n_pred]
        preds = []
        for i in range(0, len(xv), 32):
            preds.append(model(torch.from_numpy(xv[i:i + 32])[:, None].to(device)).cpu().numpy()[:, 0])
        preds = np.concatenate(preds).astype(np.float32)
    out = os.path.join(a.arm, f"rollout_{a.which}.npz")
    np.savez(out, E=E, xmax=xmax, blow=blow, ic_tags=np.asarray(tags), preds=preds, val_pick=pick[:a.n_pred],
             step_of_ckpt=int(ck["step"]), alpha=float(ck["config"]["alpha"]), seed=int(ck["config"]["seed"]))
    n_blow = int((blow >= 0).sum())
    print(f"ROLLOUT_DONE arm={os.path.basename(a.arm.rstrip('/'))} which={a.which} ckpt_step={ck['step']} "
          f"steps={a.steps} ics={n} blown={n_blow} first_blow={int(blow[blow >= 0].min()) if n_blow else -1} "
          f"seconds={time.time() - t0:.0f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
