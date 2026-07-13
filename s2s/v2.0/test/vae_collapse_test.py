"""
Test for VAE posterior collapse in PanguModel_Plasim.

The model generates ensemble diversity entirely through the VAE's reparameterisation:
the same input is repeated num_ensemble_members times and each copy receives a
different random noise draw in reparameterize(). If sigma has collapsed to a very
negative value, std = exp(0.5 * sigma) ≈ 0, all four members become identical,
and the model is effectively deterministic.

Run from v2.0/ directory:
    python test/vae_collapse_test.py \
        --yaml_config config/exp2.yaml \
        --checkpoint results/S2S/.../ckpt.tar \
        --n_samples 8

The checkpoint path must exist. If omitted, the model runs with random weights
(useful only to verify the spread mechanism works at all before training).
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.distributed as dist

# Allow imports from v2.0/ without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.pangu import PanguModel_Plasim
from utils.YParams import YParams
from utils.data_loader_multifiles import get_data_loader

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_ensemble_batch(data, ens_members):
    """Repeat each sample ens_members times — mirrors train.py:185."""
    return data.unsqueeze(1).expand(-1, ens_members, *data.shape[1:]).reshape(-1, *data.shape[1:])


def relative_spread(outputs):
    """
    Returns ensemble spread relative to signal magnitude.
    outputs: list of tensors, each shape (batch, channels, ...) — one per ensemble member.
    Returns a scalar: mean(std across members) / mean(|mean across members|).
    Near zero means collapse; a few percent or more suggests real diversity.
    """
    stacked = torch.stack(outputs, dim=0)          # (M, batch, channels, ...)
    member_std  = stacked.std(dim=0)               # (batch, channels, ...)
    member_mean = stacked.mean(dim=0).abs()
    # Avoid division by zero on near-zero mean pixels.
    mask = member_mean > 1e-6
    rel = (member_std[mask] / member_mean[mask]).mean()
    return rel.item(), member_std.mean().item()


def sigma_stats(sigma_tensor):
    """Summary stats for the log-variance tensor from the VAE encoder."""
    s = sigma_tensor.detach().float()
    std_implied = torch.exp(0.5 * s)
    return {
        "logvar_mean":  s.mean().item(),
        "logvar_min":   s.min().item(),
        "logvar_max":   s.max().item(),
        "implied_std_mean": std_implied.mean().item(),
        "implied_std_min":  std_implied.min().item(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config",  required=True)
    parser.add_argument("--config",       default="S2S")
    parser.add_argument("--checkpoint",   default=None,
                        help="Path to ckpt.tar. Omit to test with random weights.")
    parser.add_argument("--n_samples",    type=int, default=8,
                        help="Number of unique input samples to test.")
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    params = YParams(args.yaml_config, args.config, print_params=False)

    # Single-process: no DDP needed.
    params["local_rank"] = 0
    params["global_batch_size"] = params.batch_size
    params["world_size"] = 1
    if not hasattr(params, "num_ensemble_members"):
        params["num_ensemble_members"] = 1

    ens = params.num_ensemble_members
    logging.info("num_ensemble_members = %d", ens)

    # -----------------------------------------------------------------------
    # Build model
    # -----------------------------------------------------------------------
    logging.info("Building model…")

    # Land mask placeholder — load if available, else zeros.
    land_mask = torch.zeros(1, 1, *params.horizontal_resolution, device=device)
    mask_fill = params.mask_fill if hasattr(params, "mask_fill") else {}
    model = PanguModel_Plasim(params, land_mask=land_mask,
                              mask_fill=mask_fill).to(device)
    model.eval()

    if args.checkpoint:
        logging.info("Loading checkpoint: %s", args.checkpoint)
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
        # Strip DDP 'module.' prefix if present.
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logging.warning("Missing keys (%d): %s …", len(missing), missing[:3])
        if unexpected:
            logging.warning("Unexpected keys (%d): %s …", len(unexpected), unexpected[:3])
    else:
        logging.warning("No checkpoint provided — running with random weights. "
                        "Spread will reflect architecture only, not trained behaviour.")

    # -----------------------------------------------------------------------
    # Data loader — one batch is enough
    # -----------------------------------------------------------------------
    logging.info("Loading data…")
    # Use params.batch_size = 1 unique sample; expand to ens copies below.
    orig_bs = params.batch_size
    params["batch_size"] = 1

    loader, dataset, _ = get_data_loader(
        params,
        files_pattern=params.data_dir,
        distributed=False,
        year_start=params.train_year_start,
        year_end=params.train_year_end,
        train=True,
    )
    params["batch_size"] = orig_bs

    # Load constant boundary data the same way as Trainer does.
    constant_boundary = dataset.constant_boundary_data.unsqueeze(0).to(device)
    if ens > 1:
        constant_boundary = to_ensemble_batch(constant_boundary, ens)

    # -----------------------------------------------------------------------
    # Run collapse test
    # -----------------------------------------------------------------------
    all_surface_spread, all_upper_spread = [], []
    all_sigma_stats = []
    samples_done = 0

    logging.info("Running %d samples through the model (ens=%d each)…", args.n_samples, ens)

    with torch.no_grad():
        for batch in loader:
            if samples_done >= args.n_samples:
                break

            inp_sfc, inp_upper, tgt_sfc, tgt_upper, tgt_diag, vb = (
                t.to(device) for t in batch[:6]
            )

            # Expand each unique sample to ens copies.
            if ens > 1:
                inp_sfc   = to_ensemble_batch(inp_sfc,   ens)
                inp_upper = to_ensemble_batch(inp_upper, ens)
                vb        = to_ensemble_batch(vb,        ens)

            # Forward — train=False so Encoder 2 doesn't run.
            out = model(inp_sfc, constant_boundary, vb, inp_upper, train=False)

            # out = (output_surface, output_upper_air, [output_diagnostic,] mu, sigma)
            if len(out) == 5:
                out_sfc, out_upper, _, mu, sigma = out
            elif len(out) == 4:
                out_sfc, out_upper, mu, sigma = out
            else:
                logging.error("Unexpected output length %d", len(out))
                break

            # Sigma stats (collapse indicator).
            stats = sigma_stats(sigma)
            all_sigma_stats.append(stats)

            # Split into individual ensemble members.
            # to_ensemble_batch lays out as [s0,s0,s0,s0, s1,s1,s1,s1, ...]
            # so member slicing is out_sfc[sample*ens : (sample+1)*ens]
            B_ens = out_sfc.shape[0]
            unique = B_ens // ens
            sfc_members   = [out_sfc  [j*ens:(j+1)*ens] for j in range(unique)]
            upper_members = [out_upper[j*ens:(j+1)*ens] for j in range(unique)]

            for j in range(unique):
                members_sfc   = [sfc_members[j][m:m+1]   for m in range(ens)]
                members_upper = [upper_members[j][m:m+1]  for m in range(ens)]
                rel_sfc,  abs_sfc   = relative_spread(members_sfc)
                rel_upper, abs_upper = relative_spread(members_upper)
                all_surface_spread.append((rel_sfc,  abs_sfc))
                all_upper_spread.append((rel_upper, abs_upper))
                samples_done += 1
                if samples_done >= args.n_samples:
                    break

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("  VAE Posterior Collapse Test")
    print("="*60)
    print(f"  Samples tested : {samples_done}")
    print(f"  Ensemble size  : {ens}")
    print(f"  Checkpoint     : {args.checkpoint or '(none — random weights)'}")

    print("\n--- VAE sigma (log-variance) statistics ---")
    print("  These come from Encoder 1 (the forecast encoder).")
    print("  If logvar_mean << 0 (e.g. -10 or lower), the encoder has")
    print("  collapsed: implied_std ≈ 0 and all ensemble members are identical.\n")
    lv_means = [s["logvar_mean"]       for s in all_sigma_stats]
    lv_mins  = [s["logvar_min"]        for s in all_sigma_stats]
    ist_means= [s["implied_std_mean"]  for s in all_sigma_stats]
    ist_mins = [s["implied_std_min"]   for s in all_sigma_stats]
    print(f"  logvar  mean across samples : {np.mean(lv_means):.3f}  "
          f"(min seen: {np.min(lv_mins):.3f})")
    print(f"  implied std  mean           : {np.mean(ist_means):.4f}  "
          f"(min seen: {np.min(ist_mins):.4f})")

    print("\n--- Ensemble spread across output fields ---")
    print("  relative_spread = mean(std across members) / mean(|mean across members|)")
    print("  A well-functioning ensemble should show a few percent relative spread.")
    print("  Near zero means collapsed (all members identical).\n")
    rel_sfcs   = [x[0] for x in all_surface_spread]
    abs_sfcs   = [x[1] for x in all_surface_spread]
    rel_uppers = [x[0] for x in all_upper_spread]
    abs_uppers = [x[1] for x in all_upper_spread]
    print(f"  Surface fields   — relative spread : {np.mean(rel_sfcs)*100:.3f}%  "
          f"  absolute std : {np.mean(abs_sfcs):.5f}")
    print(f"  Upper-air fields — relative spread : {np.mean(rel_uppers)*100:.3f}%  "
          f"  absolute std : {np.mean(abs_uppers):.5f}")

    print("\n--- Diagnosis ---")
    mean_rel = np.mean(rel_sfcs + rel_uppers)
    mean_logvar = np.mean(lv_means)
    if mean_logvar < -5 or mean_rel < 1e-4:
        print("  COLLAPSE DETECTED.")
        print(f"  logvar mean = {mean_logvar:.2f}  (implied std ≈ {np.mean(ist_means):.4f})")
        print(f"  Relative spread = {mean_rel*100:.4f}% — ensemble members are nearly identical.")
        print("  The VAE is not contributing meaningful ensemble diversity.")
        print("  Consider: increase vae_loss_weight, use noise injection instead,")
        print("  or evaluate whether ensemble forecasting is needed at this stage.")
    elif mean_rel < 0.01:
        print(f"  WEAK SPREAD: {mean_rel*100:.2f}% relative spread.")
        print("  The VAE is producing some diversity but it is likely too small")
        print("  to represent physically meaningful forecast uncertainty.")
    else:
        print(f"  HEALTHY SPREAD: {mean_rel*100:.2f}% relative spread.")
        print("  The VAE is generating meaningful ensemble diversity.")

    print("="*60 + "\n")


if __name__ == "__main__":
    main()
