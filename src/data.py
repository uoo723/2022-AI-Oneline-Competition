"""
Created on 2022/06/09
@author Sangwoo Han
"""
import json
import os
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Tuple

import pandas as pd
from joblib import Parallel, delayed
from logzero import logger


def load_data(
    data_dir: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    train_data_path = os.path.join(data_dir, "train.json")
    test_data_path = os.path.join(data_dir, "test_data.json")
    with open(train_data_path, "r") as f1, open(test_data_path, "r") as f2:
        train_data = json.load(f1)
        test_data = json.load(f2)
    test_question = pd.read_csv("./data/test_questions.csv", encoding="utf8")
    return train_data, test_data, test_question


def preprocess_data(
    data: Dict[str, Any],
    return_query_to_docs: bool = True,
) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, str]]:
    queries = OrderedDict()
    docs = OrderedDict()
    query_to_docs = defaultdict(list)

    for d in data["data"]:
        for paragraph in d["paragraphs"]:
            if return_query_to_docs:
                for q in paragraph["qas"]:
                    queries[q["question_id"]] = q["question"]
                    query_to_docs[q["question_id"]].append(paragraph["paragraph_id"])
            docs[paragraph["paragraph_id"]] = paragraph["context"]

    ret = (queries, docs)
    if return_query_to_docs:
        ret += (query_to_docs,)
    return ret


def _load_single_shard(
    data_dir: str,
    shard_idx: int,
    shard_indices: List[int],
    query_id_i2s: Dict[int, str],
) -> Dict[str, List[str]]:
    filename = f"train_top1000_{shard_idx:02d}.txt"
    tsv_path = os.path.join(data_dir, "top1000", filename)
    df = pd.read_csv(tsv_path, sep=" ", header=None)
    df[0] = df[0].map(lambda x: query_id_i2s[x])
    candidates = df.groupby(0)[2].apply(list).to_dict()
    logger.info(f"Completed loading shard {shard_idx} among {shard_indices}")
    return candidates


def load_shard(
    data_dir: str,
    shard_indices: List[int],
    train_queries: Dict[str, str],
    n_jobs: int = 1,
) -> Dict[str, List[str]]:
    n_jobs = min(n_jobs, len(shard_indices))
    query_id_i2s = dict(zip(range(len(train_queries)), train_queries.keys()))
    candidates_list = Parallel(n_jobs=n_jobs)(
        delayed(_load_single_shard)(data_dir, shard_idx, shard_indices, query_id_i2s)
        for shard_idx in shard_indices
    )

    candidates = {}

    for c in candidates_list:
        candidates.update(c)

    return candidates
