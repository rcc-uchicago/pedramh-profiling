"""Pangu-parity per-variable wandb diagnostics for the PlaSim/E3SM trainer.

Purpose: make makani's production runs answerable in the SAME wandb panels as
PanguWeather and physicsnemo-ai-rossby.  Those two harnesses share a metrics
contract (`wandb_metrics_report.md`, `ai_rossby_finegrained_wandb_handoff.md`):
per-variable(-level) latitude-weighted RMSE in PHYSICAL units, computed on the
normalized train tensors every iteration and denormalized by the per-channel
std, logged flat (no namespace prefix, no explicit ``step=``) under

    train_{var}_lwrmse                       (surface / diagnostic vars)
    train_{var}_level{level:.4f}_lwrmse      (upper-air vars)

plus flat aggregate keys ``train_loss`` / ``valid_loss``.  Stock makani logs
none of this: epoch-level only, and under its own names ("loss",
"validation loss"), so nothing overlays.

What "same metrics" can and cannot mean here, stated so nobody over-reads a
panel: makani-E3SM trains a DIFFERENT channel contract (53 channels: PS,
TREFHT, T/U/V/RH/Z at the 10 plev levels 200..1000 hPa, PRECT) than the
Pangu/ai-rossby pair (101 channels at 18 hybrid levels).  Identical *keys* are
therefore impossible where the variables differ; what this module matches is
the SCHEME — formula, units, cadence, key format — and the base-variable names
where the physics is shared: channels ``RH{p}`` log as ``RELHUM`` and ``Z{p}``
as ``Z3`` (their E3SM source fields, and Pangu's base names), so e.g. a
``train_RELHUM_level500.0000_lwrmse`` panel is the same physical quantity in
both projects.  Levels are plain hPa; Pangu's hybrid-level key values differ
numerically, so most panels sit side-by-side rather than auto-overlaying.

Formula — copied from ``PanguWeather/v2.0/train.py:141-157`` per the handoff
(the ``N * w / sum(w)`` normalization and the lat/lon-only reduction are the
contract, not a style choice):

    w    = cos(pi/180 * lat);  w = N * w / sum(w)          # (H,)
    rmse = sqrt(mean(w * (pred - tar)^2, dim=(-1, -2)))    # -> (B, C)
    log( mean_B(rmse[:, c]) * std[c] )                     # physical units

The std multiply is exact denormalization for z-scored tensors:
``std * RMSE(pred_z, tar_z) == RMSE(pred_phys, tar_phys)``.

Mechanics: a **forward hook on ``trainer.loss_obj``** — the one per-step call
site where the normalized ``(pred, tar)`` pair exists
(``deterministic_trainer.py:493``).  A hook rather than a wrapper because
``loss_obj`` is passed to checkpoint save/restore (lines 252/281/399) and a
wrapper would silently change the checkpoint's state-dict keys.  Gated on
``module.training and torch.is_grad_enabled()`` so validation (eval + no_grad,
line 628) never fires it, and installed at all only when ``log_to_wandb`` is
on — which every bench/scaling render forces False, so the scaling CSV path
runs zero new instructions (CLAUDE.md #10: bench instrumentation must not
drift).  On the production path the cost is real and unhidden: ~55 scalar
reductions + one wandb.log per step, same trade Pangu made ("runs every
iteration, unconditionally").
"""

from __future__ import annotations

import logging
import re

import torch

logger = logging.getLogger(__name__)

# channel-name -> (pangu base var, level) parser for the upper-air channels.
# Two naming schemes exist across the makani-E3SM packs:
#   * locked 53-ch pack: T200/U850/RH500/Z1000 -- nominal-hPa names; RH->RELHUM
#     and Z->Z3 are deliberate renames to Pangu's base names for the same E3SM
#     source fields (RELHUM, Z3). NOTE these nominal levels do not numerically
#     equal Pangu's hybrid values, so those panels sit side-by-side, not merged.
#   * ALLDATA 101-ch pack: T_l00..T_l17 etc., BY LEVEL INDEX, bases already
#     Pangu-named. These map through the archive's actual 18 hybrid levels
#     below, so the formatted keys match Pangu/ai-rossby's CHARACTER FOR
#     CHARACTER (same literal level list, same %.4f) and the wandb panels merge.
_UPPER_AIR = re.compile(r"^(T|U|V|RH|Z)(\d+)$")
_BASE_NAME = {"T": "T", "U": "U", "V": "V", "RH": "RELHUM", "Z": "Z3"}
_UPPER_AIR_LIDX = re.compile(r"^(T|U|V|Z3|RELHUM)_l(\d\d)$")
# Pangu's NOMINAL level labels per index, TOA->surface. ⚠ NOT the archive's
# actual hybrid mid-level values (4.71..998.50): Pangu's wandb keys were read
# back from its production datastore (run j796bp1k, 2026-08-27) and use these
# round labels -- `train_T_level1000.0000_lwrmse`, 18 distinct values below.
# An earlier revision formatted the true hybrid floats here; every upper-air
# panel silently failed to overlay (caught by the operator on the live run).
# The index<->label correspondence is the archive's own nominal table
# (metadata attrs.level_table: hybrid 200.9989 -> label 200, 849.6612 -> 850).
_NOMINAL_LEVELS = [
    5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700,
    850, 925, 1000,
]


# Surface/diagnostic names that must NEVER hit the nominal-hPa regex: U10 ends
# in digits and would be misparsed as upper-air U at a fake 10 hPa level (it
# was, until the 7565972 smoke's datastore check caught the missing
# train_U10_lwrmse key). Same bug class, same fix as the pack-audit script:
# exact-name dispatch FIRST.
_SURFACE_DIAG = {
    "PS", "TREFHT", "U10", "RHREFHT", "PSL", "TMQ", "FSNT", "FSNTOA",
    "SOILWATER_10CM", "TSOI_10CM", "PRECT",
}


def build_keys(channel_names) -> list[str]:
    """One wandb key per channel, in channel order, Pangu key scheme."""
    keys = []
    for name in channel_names:
        if name in _SURFACE_DIAG:
            keys.append(f"train_{name}_lwrmse")
            continue
        m = _UPPER_AIR_LIDX.match(name)
        if m:
            level = _NOMINAL_LEVELS[int(m.group(2))]
            keys.append(f"train_{m.group(1)}_level{level:.4f}_lwrmse")
            continue
        m = _UPPER_AIR.match(name)
        if m:
            base, level = _BASE_NAME[m.group(1)], float(m.group(2))
            keys.append(f"train_{base}_level{level:.4f}_lwrmse")
        else:
            # PS, TREFHT, PRECT, U10, RHREFHT, PSL, TMQ, SOILWATER_10CM,
            # TSOI_10CM, FSNT, FSNTOA -- all already Pangu's own key names.
            keys.append(f"train_{name}_lwrmse")
    return keys


class PanguParityDiagnostics:
    """Holds the precomputed weights/stds/keys and logs one step's metrics."""

    def __init__(self, channel_names, lat_deg: torch.Tensor, out_scale, device):
        self.keys = build_keys(channel_names)

        # Pangu's latitude_weighting_factor_torch, verbatim semantics.
        lat = torch.as_tensor(lat_deg, dtype=torch.float32, device=device)
        w = torch.cos(3.1416 / 180.0 * lat)
        w = lat.shape[0] * w / torch.sum(w)
        self.weight = w.reshape(1, 1, -1, 1)  # (1, 1, H, 1)

        scale = torch.as_tensor(out_scale, dtype=torch.float32, device=device)
        self.std = scale.reshape(-1)  # (C,)

        if self.std.shape[0] != len(self.keys):
            # Louder than a comment, quieter than killing a production run at
            # step 1: refuse to install (caller checks .ok) with the mismatch
            # in the log. A wrong-length std would multiply the wrong channels.
            logger.error(
                "wandb diagnostics DISABLED: %d channel names vs %d stds",
                len(self.keys),
                self.std.shape[0],
            )
            self.ok = False
        else:
            self.ok = True
        self._shape_checked = False

    @torch.no_grad()
    def log_step(self, pred: torch.Tensor, tar: torch.Tensor, loss) -> None:
        import wandb  # deferred: only reachable when log_to_wandb is on

        # fp32 regardless of amp: bf16 squared-residual means lose real digits.
        with torch.autocast(device_type="cuda", enabled=False):
            p = pred.detach().float()
            t = tar.detach().float()
            if not self._shape_checked:
                # (B, C, H, W) with C == len(keys); anything else means the
                # contract drifted -- disable loudly rather than mislabel.
                if p.dim() != 4 or p.shape[1] != len(self.keys):
                    logger.error(
                        "wandb diagnostics DISABLED: pred shape %s vs %d keys",
                        tuple(p.shape),
                        len(self.keys),
                    )
                    self.ok = False
                    return
                self._shape_checked = True
            rmse = torch.sqrt(
                torch.mean(self.weight * (p - t) ** 2, dim=(-1, -2))
            )  # (B, C), normalized units
            vals = rmse.mean(dim=0) * self.std  # (C,), physical units

        logs = dict(zip(self.keys, vals.tolist()))
        # Pangu's 102nd key: channel-mean of the NORMALIZED (pre-std) rmse.
        logs["train_mean_norm_lwrmse"] = float(rmse.mean())
        # Flat aggregate for overlay with Pangu's per-iteration `train_loss`.
        logs["train_loss"] = float(loss)
        # No `step=`: the contract logs on wandb's auto-incrementing step, same
        # as Pangu. (Makani's native epoch logs use step=epoch and will be
        # dropped as out-of-order once this runs -- PlasimTrainer.log_epoch
        # re-logs their content flat; see that override.)
        wandb.log(logs)


def install(trainer) -> bool:
    """Register the per-step hook on ``trainer.loss_obj``. Returns success."""
    diag = PanguParityDiagnostics(
        channel_names=trainer.params.channel_names,
        lat_deg=trainer.lat_global,
        out_scale=trainer.train_dataloader.get_output_normalization()[1],
        device=trainer.device,
    )
    if not diag.ok:
        return False

    def _hook(module, args, output):
        # args is the positional (pred, tar) of loss_obj(pred, tar, inp=...).
        # Training-only: validation runs under eval() + no_grad (both gates).
        if module.training and torch.is_grad_enabled() and diag.ok:
            diag.log_step(args[0], args[1], output.detach())

    trainer.loss_obj.register_forward_hook(_hook)
    logger.info(
        "installed Pangu-parity wandb diagnostics: %d per-channel keys",
        len(diag.keys),
    )
    return True
