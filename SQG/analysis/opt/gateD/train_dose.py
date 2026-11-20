"""train_dose.py -- one gate-D training arm (item 4).

o1.2's network, loss and optimiser, trained OFFLINE on stored production frames, with the target shifted by a dose of
lean32ek's measured bias pattern.

  python train_dose.py --frames <dir> --out <arm dir> --alpha <dose> --seed <s>
                       [--train-members 1,2,3] [--val-members 4] [--steps 195250] [--batch 16]
                       [--val-every 2500] [--ckpt-every 2500] [--n-val 512] [--prec ieee|tf32|bf16]

As o1.2 (L90-102, L401-411, L476-492):
- ConvNextUViT(n_feat 32, depths 2/2/4/4/8, kernel 7, drop_path 0.1, 4 attention blocks, 8 heads, residual);
- MSE loss; AdamW(lr 5e-4, weight decay 0.05, betas 0.9/0.95); OneCycle(pct 0.1, div 25, final div 1e4) over --steps;
- gradient clip 1.0; strict IEEE fp32 (o1.2's check, in torch 2.3.1's spelling).

Different from o1.2:
- **Data.** Pairs are drawn uniformly, with replacement, from the stored 0.02 pairs of the training members, not from
  a solver FIFO.
- **Batch.** 16 on one A100.
- **Target.** y + alpha * B_hat(x, y) (dose_common.py).

Arms with the same --seed get the same initial weights and the same batch sequence. They differ only in alpha, so
their difference isolates the dose.

**Validation:** every --val-every steps, on the fixed held-out pairs (the same for every arm):
- val_clean against the true target y;
- val_own against the arm's own target y + alpha * B_hat;
- best.pt is the lowest val_own.

**Resume:** last.pt every --ckpt-every steps holds the weights, optimiser, scheduler, step, the numpy and torch RNG
states and the histories. A rerun continues from it, with the same batch sequence as an uninterrupted run.

**Precision (--prec; mixprec/PREDICTIONS_prec_outcome.md, 2026-11-20).** The network's precision, as o12_polaris.py's
O12_PREC (mixprec probe 2):
- ieee (default): o1.2's strict fp32. Gate D's arms all ran this path, and it is unchanged.
- tf32: TF32 matmul + cuDNN conv for the whole run, validation included; o1.2's check must then fail (the proof).
- bf16: strict fp32 flags, with the training forward and loss under bf16 autocast. Weights, optimiser and validation
  stay float32.
The first training step counts the Conv / Linear outputs that are bfloat16 (PREC_FORWARD): there must be some under
bf16 and none otherwise. Probe 2's M2 check could not see bf16 through x + net(x). Each val row also logs the peak GPU
memory.

Prints DOSE_VAL lines and DOSE_ARM_DONE last.
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SQG = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
sys.path.insert(0, SQG)
import dose_common as DC                                             # noqa: E402

N_FEAT, DEPTHS, KERNEL, DROP_PATH, N_ATTN, N_HEADS = 32, [2, 2, 4, 4, 8], 7, 0.1, 4, 8      # o1.2 L91-92
LR, PCT_START, DIV, FINAL_DIV, WD, BETAS, CLIP = 5e-4, 0.1, 25, 1e4, 0.05, (0.9, 0.95), 1.0   # o1.2 L100-102
PRECS = ("ieee", "tf32", "bf16")                                     # o12_polaris.PRECS


def fp32_errors(torch, F, device):
    """o1.2's check: the max relative error of a conv and a matmul against float64."""
    g = torch.Generator(device=device).manual_seed(0)
    r = lambda *s: torch.randn(*s, device=device, generator=g)                              # noqa: E731
    rel = lambda a, b: ((a.double() - b).abs().max() / b.abs().max()).item()               # noqa: E731
    x, w, a, b = r(1, 64, 128, 128), r(64, 64, 7, 7) / 50, r(1024, 1024), r(1024, 1024)
    return {"conv": rel(F.conv2d(x, w, padding=3), F.conv2d(x.double(), w.double(), padding=3)),
            "matmul": rel(a @ b, a.double() @ b.double())}


def strict_fp32(torch, F, device):
    """o1.2's precision block (L130-150) in torch 2.3.1's spelling, and its check."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    errs = fp32_errors(torch, F, device)
    assert all(e < 2e-5 for e in errs.values()), f"not IEEE fp32: {errs}"
    return errs


def set_precision(torch, F, device, prec):
    """After strict_fp32: switch the network to `prec` (as O12_PREC). Returns (a context factory for the training
    forward + loss, a record for config.json). tf32 must make o1.2's check fail, or TF32 is not on."""
    import contextlib
    if prec not in PRECS:
        raise SystemExit(f"ERROR DOSE_PREC={prec!r} not in {PRECS}")
    rec = {"prec": prec}
    if prec == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        rec["fp32_errors_after"] = errs = fp32_errors(torch, F, device)
        if all(e < 2e-5 for e in errs.values()):
            raise SystemExit(f"ERROR DOSE_TF32_NOT_ACTIVE o1.2's fp32 check still passes: {errs}")
    if prec == "bf16":
        return (lambda: torch.autocast("cuda", dtype=torch.bfloat16)), rec
    return contextlib.nullcontext, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train-members", default="1,2,3")
    ap.add_argument("--val-members", default="4")
    ap.add_argument("--steps", type=int, default=195250)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--val-every", type=int, default=2500)
    ap.add_argument("--ckpt-every", type=int, default=2500)
    ap.add_argument("--n-val", type=int, default=512)
    ap.add_argument("--prec", default="ieee", choices=PRECS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import OneCycleLR
    from model_UViT import ConvNextUViT
    device = torch.device("cuda:0")
    fp32 = strict_fp32(torch, F, device)
    train_ctx, prec_rec = set_precision(torch, F, device, a.prec)
    G = DC.geometry()
    bias_hat, _ = DC.torch_tools(torch, G, device)
    tr_members = [int(m) for m in a.train_members.split(",")]
    va_members = [int(m) for m in a.val_members.split(",")]
    tr_segs, tr_arr = DC.open_segments(a.frames, tr_members)
    va_segs, va_arr = DC.open_segments(a.frames, va_members)
    if not tr_arr or not va_arr:
        raise SystemExit(f"ERROR DOSE_NO_FRAMES train={len(tr_arr)} val={len(va_arr)} in {a.frames}")
    seg, kk = DC.pair_index(tr_arr)
    xv_np, yv_np, _ = DC.val_pairs(va_arr, a.n_val)
    xv = torch.from_numpy(xv_np)[:, None].to(device)
    yv = torch.from_numpy(yv_np)[:, None].to(device)
    with torch.no_grad():
        yv_own = yv + a.alpha * bias_hat(xv, yv) if a.alpha else yv
        persistence = float(F.mse_loss(xv, yv))

    torch.manual_seed(a.seed)
    model = ConvNextUViT(in_ch=1, out_ch=1, n_feat=N_FEAT, depths=DEPTHS, kernel_size=KERNEL, drop_path=DROP_PATH,
                         n_attn=N_ATTN, n_heads=N_HEADS, residual=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=BETAS)
    sched = OneCycleLR(opt, max_lr=LR, total_steps=a.steps, pct_start=PCT_START, div_factor=DIV,
                       final_div_factor=FINAL_DIV)
    rng = np.random.default_rng(a.seed)
    step, hist, best, census_before = 0, [], float("inf"), None
    last_p, best_p = os.path.join(a.out, "last.pt"), os.path.join(a.out, "best.pt")
    if os.path.exists(last_p):
        ck = torch.load(last_p, map_location="cpu")
        if ck["config"].get("prec", "ieee") != a.prec:                   # gate D's arms predate --prec: ieee
            raise SystemExit(f"ERROR DOSE_PREC_MISMATCH {last_p} was trained as {ck['config'].get('prec', 'ieee')}, "
                             f"not {a.prec}")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        rng.bit_generator.state = ck["np_rng"]
        torch.set_rng_state(ck["torch_rng"])
        torch.cuda.set_rng_state(ck["cuda_rng"])
        step, hist, best = ck["step"], ck["hist"], ck["best"]
        census_before = ck["config"].get("forward_census")      # kept if no step is left (a finished arm's rerun)
        print(f"DOSE_RESUME step={step} best_val_own={best:.6g}", flush=True)
    cfg = {"alpha": a.alpha, "seed": a.seed, "steps": a.steps, "batch": a.batch, "train_segments": [s[0] for s in
           tr_segs], "val_segments": [s[0] for s in va_segs], "n_train_pairs": int(len(seg)), "n_val": int(len(xv)),
           "persistence_val": persistence, "fp32_check": fp32,
           "params_M": sum(p.numel() for p in model.parameters()) / 1e6, "prec": a.prec, "prec_record": prec_rec}
    if census_before:
        cfg["forward_census"] = census_before
    with open(os.path.join(a.out, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=1)
    print(f"DOSE_ARM alpha={a.alpha:g} seed={a.seed} steps={a.steps} batch={a.batch} train_pairs={len(seg)} "
          f"val={len(xv)} persistence={persistence:.5f} params={cfg['params_M']:.2f}M fp32={fp32} prec={a.prec}",
          flush=True)

    def save(path):
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
              "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
              "cuda_rng": torch.cuda.get_rng_state(), "hist": hist, "best": best, "config": cfg}
        torch.save(ck, path + ".tmp")
        os.replace(path + ".tmp", path)

    def validate():
        model.eval()
        se_c = se_o = 0.0
        with torch.no_grad():
            for i in range(0, len(xv), 32):
                p = model(xv[i:i + 32])
                se_c += float(((p - yv[i:i + 32]) ** 2).sum())
                se_o += float(((p - yv_own[i:i + 32]) ** 2).sum())
        model.train()
        n = xv.numel()
        return se_c / n, se_o / n

    # the first step's Conv / Linear output dtypes (hooks draw no random numbers, so the batch sequence is unchanged)
    census = {}

    def census_hook(name):
        def hook(mod, inp, out):                                     # returns None: a returned value would replace out
            census.setdefault(name, str(out.dtype))
        return hook

    hooks = [m.register_forward_hook(census_hook(n)) for n, m in model.named_modules()
             if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear))]
    model.train()
    loss_sum, n_loss, t_last = 0.0, 0, time.time()
    while step < a.steps:
        idx = rng.integers(0, len(seg), size=a.batch)
        xb = np.stack([tr_arr[seg[i]][kk[i]] for i in idx])
        yb = np.stack([tr_arr[seg[i]][kk[i] + DC.LAG] for i in idx])
        x = torch.from_numpy(xb)[:, None].to(device, non_blocking=True)
        y = torch.from_numpy(yb)[:, None].to(device, non_blocking=True)
        if a.alpha:
            with torch.no_grad():
                y = y + a.alpha * bias_hat(x, y)
        with train_ctx():
            loss = F.mse_loss(model(x), y)
        if hooks:
            for h in hooks:
                h.remove()
            hooks = []
            n16 = sum(v == "torch.bfloat16" for v in census.values())
            print(f"PREC_FORWARD prec={a.prec} conv_linear_modules={len(census)} bf16={n16} "
                  f"loss_dtype={loss.dtype}", flush=True)
            if (n16 > 0) != (a.prec == "bf16"):
                raise SystemExit(f"ERROR PREC_FORWARD_DTYPE prec={a.prec} bf16 outputs {n16}/{len(census)}")
            cfg["forward_census"] = {"conv_linear_modules": len(census), "bf16": n16, "loss_dtype": str(loss.dtype)}
            with open(os.path.join(a.out, "config.json"), "w") as fh:
                json.dump(cfg, fh, indent=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        if step < a.steps - 1:
            sched.step()
        step += 1
        loss_sum, n_loss = loss_sum + float(loss), n_loss + 1
        if not math.isfinite(float(loss)):
            raise SystemExit(f"ERROR DOSE_NONFINITE_LOSS step={step}")
        if step % a.val_every == 0 or step == a.steps:
            vc, vo = validate()
            dt = (time.time() - t_last) / max(n_loss, 1)
            row = {"step": step, "train": loss_sum / max(n_loss, 1), "val_clean": vc, "val_own": vo,
                   "lr": opt.param_groups[0]["lr"], "s_step": dt,
                   "peak_GiB": torch.cuda.max_memory_allocated(device) / 2 ** 30}
            hist.append(row)
            print(f"DOSE_VAL alpha={a.alpha:g} seed={a.seed} step={step} train={row['train']:.6f} val_clean={vc:.6f} "
                  f"val_own={vo:.6f} lr={row['lr']:.2e} s_step={dt:.4f} peak={row['peak_GiB']:.2f}GiB", flush=True)
            if vo < best:
                best = vo
                save(best_p)
            loss_sum, n_loss, t_last = 0.0, 0, time.time()
        if step % a.ckpt_every == 0 or step == a.steps:
            save(last_p)
    with open(os.path.join(a.out, "history.json"), "w") as fh:
        json.dump({"config": cfg, "hist": hist, "best_val_own": best}, fh, indent=1)
    print(f"DOSE_ARM_DONE alpha={a.alpha:g} seed={a.seed} step={step} best_val_own={best:.6g} prec={a.prec}",
          flush=True)


if __name__ == "__main__":
    main()
