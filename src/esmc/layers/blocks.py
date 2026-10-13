import torch
import torch.nn as nn
import torch.nn.functional as F

from esmc.layers.attention import MultiHeadAttention


def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    # set hidden dimension to nearest multiple of 256 after expansion ratio
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class SwiGLU(nn.Module):
    """
    SwiGLU activation function as an nn.Module, allowing it to be used within nn.Sequential.
    Splits along the last dim, applies SiLU to the first half, multiplies by the second.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


def swiglu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool):
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(
            d_model, swiglu_correction_fn(expansion_ratio, d_model) * 2, bias=bias
        ),
        SwiGLU(),
        nn.Linear(swiglu_correction_fn(expansion_ratio, d_model), d_model, bias=bias),
    )


def gelu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool):
    hidden_dim = int(expansion_ratio * d_model)
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, hidden_dim, bias=bias),
        nn.GELU(),
        nn.Linear(hidden_dim, d_model, bias=bias),
    )


class UnifiedTransformerBlock(nn.Module):
    """Plain (non-geometric) transformer block used by ESM-C."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = False,
        expansion_ratio: float = 4.0,
        residue_scaling_factor: float = 1,
        qk_layernorm: bool = True,
        ffn_type: str = "swiglu",  # swiglu | gelu
    ):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, bias, qk_layernorm=qk_layernorm)
        if ffn_type == "swiglu":
            self.ffn = swiglu_ln_ffn(d_model, expansion_ratio, bias)
        elif ffn_type == "gelu":
            self.ffn = gelu_ln_ffn(d_model, expansion_ratio, bias)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")
        self.scaling_factor = residue_scaling_factor

    def forward(
        self,
        x: torch.Tensor,
        sequence_id: torch.Tensor | None,
    ) -> torch.Tensor:
        r1 = self.attn(x, sequence_id)
        x = x + r1 / self.scaling_factor
        r3 = self.ffn(x) / self.scaling_factor
        x = x + r3
        return x
