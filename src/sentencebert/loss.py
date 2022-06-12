"""
Created on 2022/06/10
@author Sangwoo Han
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_similarity(
    x1: torch.Tensor, x2: torch.Tensor, metric: str = "cosine"
) -> torch.Tensor:
    assert metric in ["cosine", "euclidean"]
    if metric == "cosine":
        x1 = F.normalize(x1, dim=-1)
        x2 = F.normalize(x2, dim=-1)
        return (x1.unsqueeze(1) @ x2.transpose(2, 1)).squeeze()
    return 1 / (1 + torch.cdist(x1.unsqueeze(1), x2).squeeze())


# https://openaccess.thecvf.com/content_CVPR_2020/papers/Sun_Circle_Loss_A_Unified_Perspective_of_Pair_Similarity_Optimization_CVPR_2020_paper.pdf
class CircleLoss(nn.Module):
    """Implementaion of Circle loss"""

    def __init__(
        self,
        m: float = 0.15,
        gamma: float = 1.0,
        metric: str = "cosine",
    ) -> None:
        super().__init__()
        self.m = m
        self.gamma = gamma
        self.metric = metric

        assert self.metric in ["cosine", "euclidean"]

    def forward(
        self,
        anchor: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        neg: Optional[torch.Tensor] = None,
        pos_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pos is not None:
            sp = get_similarity(anchor, pos, self.metric)
            if len(sp.shape) == 1:
                sp = sp.unsqueeze(-1)
            ap = torch.clamp_min(-sp.detach() + 1 + self.m, min=0.0)
            delta_p = 1 - self.m
            weights = 1.0 if pos_weights is None else pos_weights
            logit_p = -ap * (sp - delta_p) * self.gamma * weights
            logit_p_logsumexp = torch.logsumexp(logit_p, dim=-1)
        else:
            logit_p_logsumexp = torch.tensor(0.0)

        if neg is not None:
            sn = get_similarity(anchor, neg, self.metric)
            if len(sn.shape) == 1:
                sn = sn.unsqueeze(-1)
            an = torch.clamp_min(sn.detach() + self.m, min=0.0)
            delta_n = self.m
            neg_weights = 1.0 if pos_weights is None else pos_weights.mean()
            logit_n = an * (sn - delta_n) * self.gamma * neg_weights
            logit_n_logsumexp = torch.logsumexp(logit_n, dim=-1)
        else:
            logit_n_logsumexp = torch.tensor(0.0)

        return F.softplus(logit_p_logsumexp + logit_n_logsumexp).mean()
