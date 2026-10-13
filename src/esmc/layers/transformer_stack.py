import math

import torch
import torch.nn as nn

from esmc.layers.blocks import UnifiedTransformerBlock


class TransformerStack(nn.Module):
    """Stack of plain transformer blocks (v_heads=None, no geometric layers)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        scale_residue: bool = True,
        bias: bool = False,
        qk_layernorm: bool = True,
        ffn_type: str = "swiglu",  # swiglu | gelu
        expansion_ratio: float = 8 / 3,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                UnifiedTransformerBlock(
                    d_model,
                    n_heads,
                    residue_scaling_factor=(
                        math.sqrt(n_layers / 36) if scale_residue else 1.0
                    ),
                    expansion_ratio=expansion_ratio,
                    bias=bias,
                    qk_layernorm=qk_layernorm,
                    ffn_type=ffn_type,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        sequence_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Forward pass.

        Returns:
            post_norm: (B, L, d_model) after final LayerNorm.
            pre_norm:  (B, L, d_model) before final LayerNorm, used as the model's
                       "embeddings" output.
            hiddens:   list of per-block (B, L, d_model) tensors.
        """
        hiddens = []
        for block in self.blocks:
            x = block(x, sequence_id)
            hiddens.append(x)
        return self.norm(x), x, hiddens
