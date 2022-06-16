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
            x1_mask = torch.ones(x1.shape[:-1]).to(x1.device)
        if x2_mask is None:
            x2_mask = torch.ones(x2.shape[:-1]).to(x2.device)

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
            query_mask = torch.ones(query.shape[:-1]).to(query.device)
        if passage_mask is None:
            passage_mask = torch.ones(passage.shape[:-1]).to(passage.device)

        outputs = self.attention1(passage, passage_mask)
        if len(outputs.shape) == 4:
            query = query.expand(query.shape[0], outputs.shape[1], *query.shape[1:])
        outputs = self.attention2(torch.cat([query, outputs], dim=-1), query_mask)
        outputs: torch.Tensor = self.linear(outputs)
        return outputs.squeeze().sigmoid()


class TransformerLateInteraction(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_layernorm: bool = False,
        layernorm_eps: float = 1e-12,
        linear_size: List[int] = [128],
        linear_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layernorm_eps,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.type_embeddings = nn.Embedding(2, hidden_size)
        self.position_embeddings = nn.Embedding(512, hidden_size)
        self.linear = MLPLayer(
            input_size=hidden_size,
            output_size=1,
            linear_size=linear_size,
            dropout=linear_dropout,
            use_layernorm=use_layernorm,
            layernorm_eps=layernorm_eps,
        )
        self.register_buffer("position_ids", torch.arange(512).unsqueeze(0))

    def forward(
        self,
        query: torch.Tensor,
        passage: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        passage_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if query_mask is None:
            query_mask = torch.ones(query.shape[:-1]).to(query.device)
        if passage_mask is None:
            passage_mask = torch.ones(passage.shape[:-1]).to(passage.device)

        query_length = query.shape[-2]
        passage_length = passage.shape[-2]

        query_position_ids = self.position_ids[:, :query_length]
        passage_position_ids = self.position_ids[:, :passage_length]
        type_ids = (
            torch.LongTensor([0] * query_length + [1] * passage_length)
            .unsqueeze(0)
            .to(query.device)
        )

        query_position_embeds = self.position_embeddings(query_position_ids)
        passage_position_embeds = self.position_embeddings(passage_position_ids)
        position_embeds = torch.cat(
            [query_position_embeds, passage_position_embeds], dim=-2
        )
        type_embeds = self.type_embeddings(type_ids)

        if len(passage.shape) == 4:
            assert query.shape[0] == passage.shape[0] == 1
            query_mask = query_mask.expand(passage.shape[1], *query_mask.shape[1:])
            passage_mask = passage_mask.squeeze()
            query = query.expand(passage.shape[1], *query.shape[1:])
            passage = passage.squeeze()

        input_embeds = torch.cat([query, passage], dim=-2)
        input_embeds = input_embeds + position_embeds + type_embeds
        mask = torch.cat([query_mask, passage_mask], dim=-1)
        outputs = self.encoder(input_embeds, src_key_padding_mask=~mask.bool())
        outputs: torch.Tensor = self.linear(outputs[:, 0])
        return outputs.squeeze()
