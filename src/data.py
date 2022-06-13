"""
Created on 2022/06/09
@author Sangwoo Han
"""
import json
import os
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Tuple

import pandas as pd


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
