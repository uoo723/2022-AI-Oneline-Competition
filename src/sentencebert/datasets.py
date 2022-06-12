"""
Created on 2022/06/10
@author Sangwoo Han
"""
import os
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        query_ids: np.ndarray,
        queries: np.ndarray,
        docs: Dict[str, str],
        query_to_docs: Dict[str, str],
        candidates: Dict[str, List[str]],
        topk: int = 1000,
        num_pos: int = 1,
        num_neg: int = 5,
        is_training: bool = False,
    ) -> None:
        super().__init__()
        assert len(queries) == len(query_ids)

        self.query_ids = query_ids
        self.queries = queries
        self.docs = docs
        self.query_to_docs = query_to_docs
        self.candidates = candidates
        self.topk = topk
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, idx: int) -> Tuple[str, List[str], List[str]]:
        q_id, query = self.query_ids[idx], self.queries[idx]
        pos_doc_ids = []
        neg_doc_ids = []

        if self.is_training:
            while len(pos_doc_ids) < self.num_pos:
                ridx = np.random.randint(len(self.query_to_docs[q_id]))
                pos_doc_ids.append(self.query_to_docs[q_id][ridx])

            while len(neg_doc_ids) < self.num_neg:
                ridx = np.random.randint(len(self.candidates[q_id]))
                if self.candidates[q_id][ridx] not in self.query_to_docs[q_id]:
                    neg_doc_ids.append(self.candidates[q_id][ridx])
        else:
            # candidates in test time
            pos_doc_ids = self.candidates[q_id][: self.topk]

        pos_doc_str = [self.docs[doc_id] for doc_id in pos_doc_ids]
        neg_doc_str = [self.docs[doc_id] for doc_id in neg_doc_ids]

        return query, pos_doc_str, neg_doc_str


def bert_collate_fn(
    batch: Iterable[Tuple[str, List[str], List[str]]],
    tokenizer: PreTrainedTokenizerBase,
    query_max_length: int,
    passage_max_length: int,
) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    pos_doc_len = torch.LongTensor([len(b[1]) for b in batch])
    neg_doc_len = torch.LongTensor([len(b[2]) for b in batch])

    queries = [b[0] for b in batch]
    pos_docs = [d for b in batch for d in b[1]]
    neg_docs = [d for b in batch for d in b[2]]

    query_inputs = tokenizer(
        queries,
        return_tensors="pt",
        max_length=query_max_length,
        padding="max_length",
        truncation="longest_first",
    )

    pos_doc_inputs = tokenizer(
        pos_docs,
        return_tensors="pt",
        max_length=passage_max_length,
        padding="max_length",
        truncation="longest_first",
    )

    if neg_docs:
        neg_doc_inputs = tokenizer(
            neg_docs,
            return_tensors="pt",
            max_length=passage_max_length,
            padding="max_length",
            truncation="longest_first",
        )
    else:
        neg_doc_inputs = None

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return {
        "query_inputs": query_inputs,
        "pos_doc_inputs": pos_doc_inputs,
        "pos_doc_len": pos_doc_len,
        "neg_doc_inputs": neg_doc_inputs,
        "neg_doc_len": neg_doc_len,
    }
