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
            self.proj = nn.Linear(
                self.n_feature_layers * model_config.hidden_size,
                model_config.hidden_size,
            )
            self.proj_dropout = nn.Dropout(proj_dropout)
            self.conv1d = nn.Conv1d(5, 1, 2, padding='same')
        else:
            self.register_parameter("proj", None)
            self.register_parameter("proj_dropout", None)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        enc_outputs = self.bert(**filter_arguments(inputs, self.bert.forward))

        enc_out_shape = enc_outputs[-1][-1].shape
        if self.proj is not None:
            outputs = torch.cat(
                [
                    enc_outputs[-1][-i][:, 0].view([-1, 1, enc_out_shape[-1]])
                    for i in range(1, self.n_feature_layers + 1)
                ],
                dim=1,
            )
            outputs = self.proj_dropout(outputs)
            outputs = self.conv1d(outputs)
            outputs = outputs.view([-1, 768])
        else:
            outputs = enc_outputs[0][:, 0]

        outputs: torch.Tensor = self.mlp(outputs)
        return outputs.squeeze()