"""
Created on 2022/06/08
@author Sangwoo Han
"""
from typing import Dict, List

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from ..modules import MLPLayer
from ..utils import filter_arguments


class MonoBERT(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str = "monologg/koelectra-base-v3-discriminator",
        linear_size: List[int] = [256],
        dropout: float = 0.2,
        use_layernorm: bool = False,
        use_conv: bool = False,
        kernel_size: int = 2,
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
        self.mlp = MLPLayer(
            model_config.hidden_size,
            output_size=1,
            dropout=dropout,
            linear_size=linear_size,
            use_layernorm=use_layernorm,
        )

        self.n_feature_layers = n_feature_layers
        if n_feature_layers > 1:
            self.proj_dropout = nn.Dropout(proj_dropout)
            if use_conv:
                self.conv1d = nn.Conv1d(
                    n_feature_layers, 1, kernel_size, padding="same"
                )
                self.register_parameter("proj", None)
            else:
                self.proj = nn.Linear(
                    self.n_feature_layers * model_config.hidden_size,
                    model_config.hidden_size,
                )
                self.register_parameter("conv1d", None)
        else:
            self.register_parameter("proj", None)
            self.register_parameter("proj_dropout", None)
            self.register_parameter("conv1d", None)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        enc_outputs = self.bert(**filter_arguments(inputs, self.bert.forward))
        batch_size = enc_outputs[0].shape[0]

        if self.proj is not None:
            outputs = torch.cat(
                [
                    enc_outputs[-1][-i][:, 0]
                    for i in range(1, self.n_feature_layers + 1)
                ],
                dim=1,
            )
            outputs = self.proj_dropout(outputs)
            outputs = self.proj(outputs)
        elif self.conv1d is not None:
            outputs = torch.cat(
                [
                    enc_outputs[-1][-i][:, 0].unsqueeze(1)
                    for i in range(1, self.n_feature_layers + 1)
                ],
                dim=1,
            )
            outputs = self.proj_dropout(outputs)
            outputs = self.conv1d(outputs)
            outputs = outputs.reshape(batch_size, -1)
        else:
            outputs = enc_outputs[0][:, 0]

        outputs: torch.Tensor = self.mlp(outputs)
        return outputs.squeeze()
