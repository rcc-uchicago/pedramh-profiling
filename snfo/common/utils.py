import yaml
import argparse
from einops import rearrange
import torch 

def get_yaml(path):
    with open(path) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def save_yaml(config, path):
    with open(path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def assemble_input(surface, multilevel, diagnostic=None):
    multilevel = rearrange(
        multilevel, "b c l h w -> b (c l) h w"
    )
    if diagnostic is None:
        out = torch.cat((surface, multilevel), dim=1) # b c h w
    else:
        out = torch.cat((surface, diagnostic, multilevel), dim=1) # b c h w

    return out

def assemble_forcing(forcing, invariant):
    out = torch.cat((forcing, invariant), dim=1) # b c h w

    return out

def disassemble_input(x, nsurface=6, ndiagnostic=15, nlevels=13, use_diagnostic=True):
    # x in b c h w
    if use_diagnostic:
        surface = x[:, : nsurface]
        diagnostic = x[:, nsurface : nsurface + ndiagnostic]
        multilevel = x[:, nsurface + ndiagnostic :]
    else:
        surface = x[:, : nsurface]
        multilevel = x[:, nsurface :]

    multilevel = rearrange(
        multilevel,
        "b (c l) h w -> b c l h w",
        l=nlevels,
    )

    if use_diagnostic:
        return surface, multilevel, diagnostic
    else:
        return surface, multilevel

def disassemble_forcing(x, nforcing=3, ninvariant=2):
    # x in b c h w

    forcing = x[:, : nforcing]
    invariant = x[:, nforcing : nforcing + ninvariant]

    return forcing, invariant

def fix_state_dict(state_dict, prefix="decoder."):
    return {
        k[len(prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(prefix)
    }

def load_vanilla_weights_for_subpixel(model, checkpoint_path, prefix="model."):
    """Load weights from a vanilla-unpatch DiT checkpoint into a subpixel-unpatch DiT.

    Discards `unpatchify_layer.*` and `out_proj.*` keys from the checkpoint
    so the subpixel model's own unpatchify and output layers keep their
    freshly-initialized weights.

    Args:
        model: DiT model instance with subpixel unpatching.
        checkpoint_path: Path to a PyTorch Lightning checkpoint (.ckpt).
        prefix: Key prefix added by Lightning (e.g. "model.").

    Returns:
        List of checkpoint keys that were skipped.
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]

    # Strip Lightning prefix
    state_dict = {
        k[len(prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(prefix)
    }

    # Filter out vanilla unpatch / output projection weights
    skip_prefixes = ("unpatchify_layer.", "out_proj.")
    filtered = {k: v for k, v in state_dict.items() if not k.startswith(skip_prefixes)}

    model.load_state_dict(filtered, strict=False)
    return model

