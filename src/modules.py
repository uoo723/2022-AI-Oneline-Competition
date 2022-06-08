"""
Created on 2022/06/08
@author Sangwoo Han
"""
from typing import List, Optional

import torch
import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        dropout: float,
        use_layernorm: bool = False,
        layernorm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = (
            nn.LayerNorm(output_size, layernorm_eps) if use_layernorm else nn.Identity()
        )
        self.act = nn.GELU()

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        outputs = self.linear(inputs)
        outputs = self.dropout(outputs)
        outputs = self.layernorm(outputs)
        outputs = self.act(outputs)
        return outputs


class MLPLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        dropout: float,
        linear_size: Optional[List[int]] = None,
        use_layernorm: bool = False,
        layernorm_eps: float = 1e-12,
    ) -> None:
        super().__init__()

        linear_size = linear_size or []
        linear_size = [input_size] + linear_size

        if len(linear_size) == 1:
            self.input_layers = nn.Identity()
        else:
            self.input_layers = nn.Sequential(
                *[
                    LinearLayer(in_s, out_s, dropout, use_layernorm, layernorm_eps)
                    for in_s, out_s in zip(linear_size[:-1], linear_size[1:])
                ]
            )

        self.output_layer = nn.Linear(linear_size[-1], output_size)

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        outputs = self.input_layers(inputs)
        outputs = self.output_layer(outputs)
        return outputs
