"""
Created on 2022/06/08
@author Sangwoo Han
"""
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        queries: Dict[str, str],
        docs: Dict[str, str],
        query_to_docs: Dict[str, str],
        candidates: Dict[str, List[str]],
        shard_idx: int,
        shard_size: int = 10000,
        num_neg: int = 1,
    ) -> None:
        super().__init__()
        self.queries = list(queries.items())[
            shard_idx * shard_size : (shard_idx + 1) * shard_size
        ]
        self.docs = docs
        self.query_to_docs = query_to_docs
        self.candidates = candidates
        self.shard_idx = shard_idx
        self.num_neg = num_neg

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, idx: int) -> Tuple[List[str], List[str], List[int]]:
        q_id, query = self.queries[idx]
        pos_doc_ids = [d_id for d_id in self.query_to_docs[q_id]]
        neg_doc_ids = set()
        while len(neg_doc_ids) < self.num_neg:
            ridx = np.random.randint(len(self.candidates[q_id]))
            if self.candidates[q_id][ridx] not in pos_doc_ids:
                neg_doc_ids.add(self.candidates[q_id][ridx])

        neg_doc_ids = list(neg_doc_ids)
        pos_doc_str = [self.docs[doc_id] for doc_id in pos_doc_ids]
        neg_doc_str = [self.docs[doc_id] for doc_id in neg_doc_ids]

        queries = [query] * (len(pos_doc_str) + len(neg_doc_str))
        labels = [1] * len(pos_doc_str) + [0] * len(neg_doc_str)

        return queries, pos_doc_str + neg_doc_str, labels


def bert_collate_fn(
    batch: Iterable[Tuple[List[str], List[str], List[int]]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    queries = [q for b in batch for q in b[0]]
    docs = [d for b in batch for d in b[1]]
    labels = [l for b in batch for l in b[2]]

    inputs = tokenizer(
        queries,
        docs,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation="longest_first",
    )
    labels = torch.FloatTensor(labels)

    return inputs, labels
