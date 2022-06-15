import json
import os
import subprocess
from collections import OrderedDict, defaultdict
from functools import partial

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from ast import literal_eval
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from joblib import Parallel, delayed
from pyserini.search.lucene import LuceneSearcher
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.base_trainer import get_ckpt_path, get_run
from src.metrics import get_mrr
from src.monobert.datasets import Dataset, bert_collate_fn
from src.monobert.models import MonoBERT
from src.sentencebert.datasets import Dataset as SenDataset
from src.sentencebert.datasets import bert_collate_fn as sen_bert_collate_fn
from src.sentencebert.loss import CircleLoss, get_similarity
from src.sentencebert.models import SentenceBERT

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==========================================================================================
# Load Data
train_data_path = "./data/train.json"
test_data_path = "./data/test_data.json"
with open(train_data_path, "r") as f1, open(test_data_path, "r") as f2:
    train_data = json.load(f1)
    test_data = json.load(f2)
test_question = pd.read_csv("./data/test_questions.csv", encoding="utf8")
sample = pd.read_csv("./data/sample_submission.csv", encoding="utf8")
print(f"# of train data: {len(train_data['data']):,}")
print(f"# of test data: {len(test_data['data']):,}")

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


train_queries, train_docs, train_query_to_docs = preprocess_data(train_data)
for t in train_queries:
    print(t, train_queries[t])
    print(type(t))
    break
for t in train_docs:
    print(t, train_docs[t])
    print(type(t))
    break
for t in train_query_to_docs:
    print(t, train_query_to_docs[t], type(train_query_to_docs[t]))
    print(type(t))
    break
print(type(train_docs))
_, test_docs = preprocess_data(test_data, return_query_to_docs=False)
test_queries = dict(zip(test_question["question_id"], test_question["question_text"]))
print(test_queries)
print(f"# of queries: {len(train_queries):,}")
print(f"# of documents: {len(train_docs):,}")

# ==========================================================================================
# Stats

query_lengths = np.array([len(q) for q in train_queries.values()])
print(f"Max length of query: {np.max(query_lengths):,}")
print(f"Min length of query: {np.min(query_lengths):,}")
print(f"Avg. length of query: {np.mean(query_lengths):.2f}")
print(f"Std. length of query: {np.std(query_lengths):.2f}")
print("-" * 40)
doc_lengths = np.array([len(d) for d in train_docs.values()])
print(f"Max length of document: {np.max(doc_lengths):,}")
print(f"Min length of document: {np.min(doc_lengths):,}")
print(f"Avg. length of document: {np.mean(doc_lengths):.2f}")
print(f"Std. length of document: {np.std(doc_lengths):.2f}")

# ==========================================================================================
# BM25 Baseline

def get_contents(data):
    contents = []
    for d in data["data"]:
        for paragraph in d["paragraphs"]:
            contents.append(
                {"id": paragraph["paragraph_id"], "contents": paragraph["context"]}
            )
    return contents


data_list = [
    (train_data, "./data/index/train/doc.json"),
    (test_data, "./data/index/test/doc.json"),
]

for data, index_path in data_list:
    contents = get_contents(data)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "w", encoding="utf8") as f:
        json.dump(contents, f)

command = 'python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input data/index/train/ \
  --language ko \
  --index data/train.index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 16 \
  --storePositions --storeDocvectors --storeRaw'

command2 = 'python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input data/index/test/ \
  --language ko \
  --index data/test.index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 16 \
  --storePositions --storeDocvectors --storeRaw'
  
subprocess.call(command, shell=True)
subprocess.call(command2, shell=True)

searcher = LuceneSearcher("./data/test.index")
searcher.set_language("ko")

answer = []
for q in tqdm(test_question["question_text"]):
    answer.append(",".join([r.docid for r in searcher.search(q, k=10)]))
sample["paragraph_id"] = answer
sample.to_csv("submission.csv", index=False)


# ==========================================================================================
# First Stage Retrieval
def make_shard(
    queries: Dict[str, str],
    tsv_root_path: str,
    tsv_filename: str,
    shard_size: int = 10000,
) -> None:
    query_ids = list(queries.keys())
    query_str = list(queries.values())
    query_df = pd.DataFrame(
        data={"q_id": np.arange(len(query_str)), "query": query_str}
    )
    num_shards = (len(query_df) + shard_size - 1) // shard_size

    for n in range(num_shards):
        tsv_path = os.path.join(tsv_root_path, f"{tsv_filename}_{n:02d}.tsv")
        os.makedirs(os.path.dirname(tsv_path), exist_ok=True)
        query_df[n * shard_size : (n + 1) * shard_size].to_csv(
            tsv_path, index=False, header=None, sep="\t"
        )


train_query_id_i2s = dict(zip(range(len(train_queries)), train_queries.keys()))
test_query_id_i2s = dict(zip(range(len(test_queries)), test_queries.keys()))


make_shard(train_queries, "./data/top1000", "train_queries")
make_shard(test_queries, "./data/top1000", "test_queries", len(test_queries))

command = 'scripts/gen_top1000.sh'
subprocess.call(command, shell=True)