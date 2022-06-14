"""
Created on 2022/06/13
@author Sangwoo Han
"""
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from ..modules import MLAttention, MLPLayer
from ..utils import filter_arguments


class ColBERT(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str = "monologg/koelectra-base-v3-discriminator",
        n_feature_layers: int = 1,
        proj_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        model_config = AutoConfig.from_pretrained(
            pretrained_model_name, output_hidden_states=True
        )
        self.bert: nn.Module = AutoModel.from_pretrained(
            pretrained_model_name, config=model_config
        )

        self.n_feature_layers = n_feature_layers
        if n_feature_layers > 1:
            self.proj = nn.Linear(
                self.n_feature_layers * model_config.hidden_size,
                model_config.hidden_size,
            )
            self.proj_dropout = nn.Dropout(proj_dropout)
        else:
            self.register_parameter("proj", None)
            self.register_parameter("proj_dropout", None)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        enc_outputs = self.bert(**filter_arguments(inputs, self.bert.forward))
        if self.proj is not None:
            outputs = torch.cat(
                [enc_outputs[-1][-i] for i in range(1, self.n_feature_layers + 1)],
                dim=-1,
            )
            outputs = self.proj_dropout(outputs)
            outputs = self.proj(outputs)
        else:
            outputs = enc_outputs[0]
        return outputs


class LateInteraction(nn.Module):
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x1_mask: Optional[torch.Tensor] = None,
        x2_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x1_mask is None:
            x1_mask = torch.ones_like(x1)
        if x2_mask is None:
            x2_mask = torch.ones_like(x2)

        if len(x1_mask.shape) == 2:
            x1_mask = x1_mask.unsqueeze(-1)

        if len(x2_mask.shape) == 2 or (len(x2.shape) == 4 and len(x2_mask.shape) == 3):
            x2_mask = x2_mask.unsqueeze(-1)

        x1 = F.normalize(x1, dim=-1) * x1_mask
        x2 = F.normalize(x2, dim=-1) * x2_mask
        if len(x2.shape) == 4:
            x1 = x1.unsqueeze(1)
        sim = x1 @ x2.transpose(-1, -2)
        return sim.max(dim=-1)[0].sum(dim=-1)


class AttentionLateInteraction(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        query_max_length: int,
        linear_size: List[int],
        dropout: float = 0.2,
        use_layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.attention1 = MLAttention(hidden_size, query_max_length)
        self.attention2 = MLAttention(hidden_size * 2, 1)
        self.linear = MLPLayer(hidden_size * 2, 1, dropout, linear_size, use_layernorm)

    def forward(
        self,
        query: torch.Tensor,
        passage: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        passage_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if query_mask is None:
            query_mask = torch.ones_like(query)
        if passage_mask is None:
            passage_mask = torch.ones_like(passage)

        outputs = self.attention1(passage, passage_mask)
        if len(outputs.shape) == 4:
            query = query.expand(query.shape[0], outputs.shape[1], *query.shape[1:])
        outputs = self.attention2(torch.cat([query, outputs], dim=-1), query_mask)
        outputs: torch.Tensor = self.linear(outputs)
        return outputs.squeeze()
