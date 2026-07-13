from functools import  partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
import math 
from natten import NeighborhoodAttention2D
from timm.models.vision_transformer import Mlp, PatchEmbed

def _ensure_int_tuple(tup, n_elements=2):
    if isinstance(tup, int):
        tup = (tup,) * n_elements
    return tup


def validate_patch_size(input_size, patch_size=None, n_patch=None):
    """Returns valid sizes, raises error if parameters incompatible"""
    if patch_size is None and n_patch is None:
        raise ValueError(
            "You passed both patch_size=None and n_patch=None. One should be defined as a tuple of ints."
        )

    if patch_size is not None and n_patch is not None:
        raise ValueError(
            f"You passed both {patch_size=} and {n_patch=}. Only one be defined, as a tuple of ints."
        )

    if patch_size is None:
        patch_size = _ensure_int_tuple(patch_size, 2)

        patch_size = (
            input_size[0] // n_patch[0] + bool(input_size[0] % n_patch[0]),
            input_size[1] // n_patch[1] + bool(input_size[1] % n_patch[1]),
        )
    else:
        n_patch = _ensure_int_tuple(n_patch, 2)

        n_patch = (
            input_size[0] // patch_size[0] + bool(input_size[0] % patch_size[0]),
            input_size[1] // patch_size[1] + bool(input_size[1] % patch_size[1]),
        )

    target_size = (patch_size[0] * n_patch[0], patch_size[1] * n_patch[1])

    return patch_size, n_patch, target_size


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class NattenAttention(nn.Module):
    """A neighborhood attention module with circular padding and natten."""

    def __init__(
        self,
        kernel_size: tuple[int, int],
        dim: int,
        num_heads: int,
        # The following are all standard
        qkv_bias=True,
        # if 'True', circularly pad the input
        # gets sufficient context in the East/West
        # direction.
        # this is probably useful if you are performing
        # prediction with this block, but likely less
        # useful for autoencoding
        circular_pad_width: bool = False,
    ):
        """Neighborhood attention."""
        super().__init__()
        self.kernel_size = kernel_size

        kernel_size_h, kernel_size_w = kernel_size

        self.circular_pad_width = circular_pad_width

        if circular_pad_width:
            pad_size = (kernel_size_w - 1) // 2
        else:
            pad_size = 0
        self.attn = NeighborhoodAttention2D(
            dim,
            num_heads=num_heads,
            kernel_size=[kernel_size_h, kernel_size_w],
            qkv_bias=qkv_bias,
        )

        self.circ_pad_reverse_dim_order = (
            0,
            0,  # channel dim
            pad_size,
            pad_size,  # width dim
            0,
            0,  # height dim
        )

        self.crop_pad = (
            0,
            0,  # channel dim
            -pad_size,
            -pad_size,  # width dim
            0,
            0,  # height dim
        )

    def forward(self, x: torch.Tensor):
        """x is of shape b h w c, from tokenizer"""

        b, h, w, c = x.shape
        x = F.pad(x, self.circ_pad_reverse_dim_order, mode="circular")

        x = self.attn(x)
        # center_crop
        x = F.pad(x, self.crop_pad)

        return x


class NattenDiTBlock(nn.Module):
    """A DiT Block with Natten Attention."""

    def __init__(
        self,
        grid_size: tuple[int, int],
        hidden_size: int,
        num_heads: int,
        kernel_size: tuple[int, int] = (7, 7),
        # This following are all standard
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        # for traditional DiT Block drop params are 0.0
        act_layer=partial(nn.GELU, approximate="tanh"),
        norm_layer=partial(nn.LayerNorm, elementwise_affine=False, eps=1e-6),
        attn_drop_rate=0.0,
        mlp_drop_rate=0.0,
        path_drop_rate=0.0,
        attn_mask=None,
        use_swiglu=False,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.attn = NattenAttention(
            kernel_size,
            hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.norm1 = norm_layer(hidden_size)
        self.norm2 = norm_layer(hidden_size)
        self.drop_path = nn.Identity()

        mlp_hidden_size = int(hidden_size * mlp_ratio)

        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_size,
            act_layer=act_layer,
            drop=0,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        b, t, c = x.shape
        h, w = self.grid_size

        res = self.adaLN_modulation(cond).chunk(6, dim=1)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = res

        y = modulate(self.norm1(x), shift_attn, scale_attn)
        z = gate_attn.unsqueeze(1) * self.attn(y.view(b, h, w, c)).view(b, h * w, c)
        x = x.view(b, h * w, c) + self.drop_path(z)

        y = modulate(self.norm2(x), shift_mlp, scale_mlp)
        z = gate_mlp.unsqueeze(1) * self.mlp(y)
        x = x + self.drop_path(z)

        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.norm_final = torch.nn.LayerNorm(
            hidden_dim, elementwise_affine=False, eps=1e-6
        )
        self.linear = torch.nn.Linear(hidden_dim, out_dim, bias=True)
        self.adaLN_modulation = torch.nn.Sequential(
            torch.nn.SiLU(), torch.nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    
class PatchPad(nn.Module):
    def __init__(self, input_size, target_size):
        super().__init__()

        self.input_size = input_size
        self.target_size = target_size
        self.lat_pad, self.lon_pad = abs(target_size[0] - input_size[0]), abs(
            target_size[1] - input_size[1]
        )

    def forward(self, x):
        if self.lat_pad:
            x = F.pad(x, pad=(0, 0, 0, self.lat_pad), mode="reflect")
        if self.lon_pad:
            x = F.pad(x, pad=(0, self.lon_pad, 0, 0), mode="circular")
        return x


class PatchUnpad(nn.Module):
    def __init__(self, input_size, target_size):
        super().__init__()
        self.input_size = input_size
        self.target_size = target_size
        if input_size[0] < target_size[0]:
            raise ValueError(
                f"Input_size[0] {input_size} is smaller than target_size[0] {target_size}, cannot unpad."
            )
        if input_size[1] < target_size[1]:
            raise ValueError(
                f"Input_size[1] {input_size} is smaller than target_size[1] {target_size}, cannot unpad."
            )

    def forward(self, x):
        return x[..., : self.target_size[0], : self.target_size[1]]
    
class FourierEmbedder(nn.Module):
    """
    Embeds scalar or vector into a cos-sin representation.
    """

    def __init__(
        self,
        out_dim,
        frequency_embedding_dim=256,
        input_multiplier=1000,
        max_period=10000,
    ):
        super().__init__()

        self.frequency_embedding_dim = frequency_embedding_dim
        self.input_multiplier = input_multiplier
        self.max_period = max_period

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(frequency_embedding_dim, out_dim, bias=True),
            torch.nn.SiLU(),
            torch.nn.Linear(out_dim, out_dim, bias=True),
        )

    def cos_sin_embedding(self, x):
        """
        Create cos-sin embeddings of a vector.
        Conventional time-step embedding on each
        dimension of the vector. Adapted from:
        https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(1)

        half = (self.frequency_embedding_dim // x.shape[-1]) // 2
        freqs = (
            torch.exp(
                -math.log(self.max_period)
                * torch.arange(start=0, end=half, dtype=torch.float32)
                / half
            )
            .to(device=x.device)
            .unsqueeze(1)
        )

        args = x[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-2)
        embedding = einops.rearrange(embedding, "... a b -> ... (a b)")

        emb_diff = self.frequency_embedding_dim - embedding.shape[-1]
        if emb_diff != 0:
            embedding = torch.cat(
                (embedding, torch.zeros(embedding.shape[0], emb_diff, device=x.device)),
                dim=-1,
            )

        return embedding

    def forward(self, x):
        x = self.input_multiplier * x
        x = self.cos_sin_embedding(x)
        x = self.mlp(x)

        return x


class NattenCombineDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size1=(90, 181),
        input_size2=(90, 181),
        input_channels1=3,
        input_channels2=3,
        hidden_channels1=1152,
        hidden_channels2=1152,
        patch_size=None,  # Either specify patch_size or n_patch, int or (int, int)
        n_patch=None,
        output_channels=None,  # If None, = input_channels
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        patch_processing="pad",  # resample
        patch_processing_add_conv=True,  # If resample only
        date_condition=False,
        combination_mode="channel_concatenation",  # 'token_adition', 'token_multiplication'
        use_natten=True,
        # natten_kernel_size=(3, 3),
        kernel_size=3,
        checkpoint=None,  # it int > 0, checkpoint every n blocks
        nsurface=6,
        ndiagnostic=9,
        nmultilevel=9,
        nlevels=26,
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads

        output_channels = (
            output_channels if output_channels is not None else input_channels1
        )
        self.output_channels = output_channels

        self.use_padding = False
        self.use_resampling = False

        natten_kernel_size = (kernel_size, kernel_size)

        # Embedding for first input
        input_size1 = _ensure_int_tuple(input_size1, 2)
        self.input_size1 = input_size1

        patch_size, n_patch, latent_size = validate_patch_size(
            input_size=input_size1, patch_size=patch_size, n_patch=n_patch
        )
        self.patch_size = patch_size
        self.n_patch = n_patch
        self.latent_size = latent_size

        self.combination_mode = combination_mode
        self.hidden_channels1 = hidden_channels1
        self.hidden_channels2 = hidden_channels2

        if patch_processing == "pad":
            self.preprocess1 = PatchPad(input_size1, target_size=latent_size)
            self.postprocess = PatchUnpad(latent_size, target_size=input_size1)
        elif patch_processing == "resample":
            self.preprocess1 = nn.Identity()
            self.postprocess = nn.Identity()
        self.x_embedder1 = PatchEmbed(
            latent_size, patch_size, input_channels1, hidden_channels1, bias=True
        )
        self.pos_embed1 = nn.Parameter(
            torch.zeros(1, self.x_embedder1.num_patches, hidden_channels1),
            requires_grad=False,
        )

        # Embedding for second input: NOTE that we use the same postprocessing
        input_size2 = _ensure_int_tuple(input_size2, 2)
        # Should have the same number of patches as the first one
        patch_size2, _, latent_size = validate_patch_size(
            input_size=input_size2, patch_size=None, n_patch=n_patch
        )
        self.input_size2 = input_size2
        if patch_processing == "pad":
            self.preprocess2 = PatchPad(input_size2, target_size=latent_size)
        elif patch_processing == "resample":
            self.preprocess2 = nn.Identity()  # removed as unused
        self.x_embedder2 = PatchEmbed(
            latent_size, patch_size2, input_channels2, hidden_channels2, bias=True
        )
        self.pos_embed2 = nn.Parameter(
            torch.zeros(1, self.x_embedder2.num_patches, hidden_channels2),
            requires_grad=False,
        )

        # Combine the two embeddings: results in hidden-channels sum of the two
        if (
            self.combination_mode == "token_concatenation"
            or self.combination_mode == "token_addition"
            or self.combination_mode == "token_multiplication"
        ):
            hidden_channels = hidden_channels1
        elif self.combination_mode == "channel_concatenation":
            hidden_channels = hidden_channels1 + hidden_channels2
        self.hidden_channels = hidden_channels

        # We always have a time embedding
        self.t_embedder = FourierEmbedder(hidden_channels)
        if date_condition:
            self.date_embedder = nn.Identity()  # removed as unused
        else:
            self.date_embedder = None

        block = self._get_dit_block(use_natten, natten_kernel_size)
        self.blocks = nn.ModuleList(
            [
                block(
                    hidden_size=hidden_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(
            hidden_channels, patch_size[0] * patch_size[1] * output_channels
        )
        self.checkpoint = checkpoint

        self.nsurface = nsurface
        self.ndiagnostic = ndiagnostic
        self.nmultilevel = nmultilevel
        self.nlevels = nlevels

    def _get_dit_block(
        self, use_natten: bool, natten_kernel_size: tuple[int, int] | None = None
    ):
        self.use_natten = use_natten
        if use_natten:
            self.natten_kernel_size = natten_kernel_size
            block = partial(
                NattenDiTBlock,
                grid_size=self.x_embedder1.grid_size,
                kernel_size=natten_kernel_size,
            )
        else:
            block = None

        return block

    def unpatchify(self, x):
        c = self.output_channels
        p1 = self.patch_size[0]
        p2 = self.patch_size[1]
        n1 = self.n_patch[0]
        n2 = self.n_patch[1]

        x = x.reshape(shape=(x.shape[0], n1, n2, p1, p2, c))
        return einops.rearrange(x, "a b c d e f -> a f (b d) (c e)")
    
    def assemble_input(self, surface, multilevel, diagnostic):
        multilevel = einops.rearrange(
            multilevel, "b l h w c -> b h w (l c)"
        )
        out = torch.cat((surface, diagnostic, multilevel), dim=-1) # b h w c
        out = einops.rearrange(
            out, "b h w c -> b c h w"
        )

        return out
    
    def disassemble_input(self, x):
        x = einops.rearrange(
            x, "b c h w -> b h w c"
        )

        surface = x[..., : self.nsurface]
        diagnostic = x[..., self.nsurface : self.nsurface + self.ndiagnostic]
        multilevel = x[..., self.nsurface + self.ndiagnostic :]

        multilevel = einops.rearrange(
            multilevel,
            "b h w (l c) -> b l h w c",
            l=self.nlevels,
        )

        return surface, multilevel, diagnostic

    def forward(self, surface_history, multilevel_history, diagnostic_history,
                z_surface, z_history, z_diagnostic, t=None, date=None, **kwargs):
        """
        Forward pass of DiT.
        x: (N, seq_length, seq_dim) tensor input
        t: (N,) tensor of diffusion timesteps
        """

        x_1 = self.assemble_input(
            surface_history, multilevel_history, diagnostic_history
        )
        x_2 = self.assemble_input(
            z_surface, z_history, z_diagnostic
        )

        # Pad or resample if needed
        x_1 = self.preprocess1(x_1)
        x_2 = self.preprocess2(x_2)

        # First, embed the patches + add fixed positional embedding:
        x_1 = (
            self.x_embedder1(x_1) + self.pos_embed1
        )  # (N, seq_length, seq_dim) -> (N, seq_length, hidden_dim)
        x_2 = (
            self.x_embedder2(x_2) + self.pos_embed2
        )  # (N, seq_length, seq_dim) -> (N, seq_length, hidden_dim)

        # x = torch.cat((x_1, x_2), dim=-1)

        #################################################
        # Recombination with x_t
        if (
            self.combination_mode == "token_addition"
            and self.hidden_channels1 == self.hidden_channels2
        ):
            x = x_1 + x_2
        elif (
            self.combination_mode == "token_multiplication"
            and self.hidden_channels1 == self.hidden_channels2
        ):
            x = x_1 * x_2
        elif self.combination_mode == "token_concatenation":
            x = torch.cat((x_1, x_2), dim=1)
        elif (
            self.combination_mode == "channel_concatenation"
            and x_1.shape[1] == x_2.shape[1]
        ):
            x = torch.cat((x_1, x_2), dim=-1)

        # Add optional time and date embeddings
        if t is None:
            t = torch.ones(x.shape[0], 1, device=x.device).view(-1)
        # time conditioning
        conditioning = self.t_embedder(t)  # (N, hidden_dim)
        # date conditioning
        if self.date_embedder is not None and date is not None:
            conditioning = conditioning + self.date_embedder(date)

        for j, block in enumerate(self.blocks):
            if self.checkpoint is not None and (j + 1) % self.checkpoint == 0:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, conditioning, use_reentrant=False
                )
            else:
                x = block(x, conditioning)

        x = self.final_layer(x, conditioning)  # (N, seq_length, out_dim)

        x = self.unpatchify(x)

        # Unpad or resample
        x = self.postprocess(x)

        return self.disassemble_input(x)