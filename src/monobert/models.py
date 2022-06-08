"""
Created on 2022/06/08
@author Sangwoo Han
"""
from typing import Dict, List

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from ..modules import MLPLayer


class MonoBERT(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str = "monologg/koelectra-base-v3-discriminator",
        linear_size: List[int] = [256],
        dropout: float = 0.2,
        use_layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.bert: nn.Module = AutoModel.from_pretrained(pretrained_model_name)
        model_config = AutoConfig.from_pretrained(pretrained_model_name)
        self.mlp = MLPLayer(
            model_config.hidden_size,
            output_size=1,
            dropout=dropout,
            linear_size=linear_size,
            use_layernorm=use_layernorm,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        enc_outputs = self.bert(**inputs)[0][:, 0]
        outputs: torch.Tensor = self.mlp(enc_outputs)
        return outputs.squeeze()
