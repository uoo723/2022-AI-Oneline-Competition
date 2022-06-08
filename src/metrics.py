"""
Created on 2022/06/08
@author Sangwoo Han
"""
from typing import Dict, List


def get_mrr(y_true: Dict[str, List[str]], predicted: Dict[str, List[str]]) -> float:
    assert len(y_true) == len(predicted)
    acc_rr = 0
    for q_id, d_ids in y_true.items():
        for rank, d_id in enumerate(predicted[q_id], start=1):
            if d_id in d_ids:
                acc_rr += 1 / rank
    return acc_rr / len(predicted)
