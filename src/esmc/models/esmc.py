from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Protocol

from esmc.layers.regression_head import RegressionHead
from esmc.layers.transformer_stack import TransformerStack


class EsmSequenceTokenizer(Protocol):
    pad_token_id: int


@dataclass
class ESMCOutput:
    sequence_logits: torch.Tensor
    embeddings: torch.Tensor | None
    hidden_states: torch.Tensor | None


class ESMC(nn.Module):
    """ESM-C: sequence-only transformer encoder with a regression-head MLM output.

    Args:
        d_model: hidden width.
        n_heads: attention heads.
        n_layers: number of transformer blocks.
        tokenizer: kept on the module so callers can read pad/cls/eos ids, not
            used by `forward` itself.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        tokenizer: EsmSequenceTokenizer,
    ):
        super().__init__()
        self.embed = nn.Embedding(64, d_model)
        self.transformer = TransformerStack(d_model, n_heads, n_layers)
        self.sequence_head = RegressionHead(d_model, 64)
        self.tokenizer = tokenizer

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def raw_model(self):
        return self

    def forward(
        self,
        sequence_tokens: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
    ) -> ESMCOutput:
        """Args:
            sequence_tokens: int64 tensor (B, L) with CLS/EOS already in place.
            sequence_id: optional bool padding mask (B, L). If None, derived from
                `sequence_tokens != tokenizer.pad_token_id`.
        """
        if sequence_id is None:
            sequence_id = sequence_tokens != self.tokenizer.pad_token_id

        x = self.embed(sequence_tokens)
        x, _, hiddens = self.transformer(x, sequence_id=sequence_id)

        # Stack hidden states into a [n_layers, B, L, D] matrix.
        hiddens = torch.stack(hiddens, dim=0)

        sequence_logits = self.sequence_head(x)
        return ESMCOutput(
            sequence_logits=sequence_logits, embeddings=x, hidden_states=hiddens
        )
