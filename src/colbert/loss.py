"""
Created on 2022/06/13
@author Sangwoo Han
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CircleLoss(nn.Module):
    """Implementaion of Circle loss"""

    def __init__(self, m: float = 0.15, gamma: float = 1.0) -> None:
        super().__init__()
        self.m = m
        self.gamma = gamma

    def forward(
        self,
        sp: Optional[torch.Tensor] = None,
        sn: Optional[torch.Tensor] = None,
        pos_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if sp is not None:
            if len(sp.shape) == 1:
                sp = sp.unsqueeze(-1)
            ap = torch.clamp_min(-sp.detach() + 1 + self.m, min=0.0)
            delta_p = 1 - self.m
            weights = 1.0 if pos_weights is None else pos_weights
            logit_p = -ap * (sp - delta_p) * self.gamma * weights
            logit_p_logsumexp = torch.logsumexp(logit_p, dim=-1)
        else:
            logit_p_logsumexp = torch.tensor(0.0)

        if sn is not None:
            if len(sn.shape) == 1:
                sn.unsqueeze(-1)
            an = torch.clamp_min(sn.detach() + self.m, min=0.0)
            delta_n = self.m
            neg_weights = 1.0 if pos_weights is None else pos_weights.mean()
            logit_n = an * (sn - delta_n) * self.gamma * neg_weights
            logit_n_logsumexp = torch.logsumexp(logit_n, dim=-1)
        else:
            logit_n_logsumexp = torch.tensor(0.0)

        return F.softplus(logit_p_logsumexp + logit_n_logsumexp).mean()
