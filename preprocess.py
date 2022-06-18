"""
Created on 2022/06/13
@author Sangwoo Han
"""
import json
import os
from typing import Any, Dict

import click
import pandas as pd

from main import cli
from src.data import load_data, preprocess_data
from src.utils import AttrDict, get_num_batches, log_elapsed_time


def _get_contents(data):
    contents = []
    for d in data["data"]:
        for paragraph in d["paragraphs"]:
            contents.append(
                {"id": paragraph["paragraph_id"], "contents": paragraph["context"]}
            )
    return contents


def _make_shard(
    queries: Dict[str, str],
    tsv_root_path: str,
    tsv_filename: str,
    shard_size: int = 10000,
) -> None:
    query_df = pd.DataFrame(
        data={"q_id": range(len(queries)), "query": queries.values()}
    )
    num_shards = get_num_batches(shard_size, len(query_df))

    for n in range(num_shards):
        tsv_path = os.path.join(tsv_root_path, f"{tsv_filename}_{n:02d}.tsv")
        os.makedirs(os.path.dirname(tsv_path), exist_ok=True)
        query_df[n * shard_size : (n + 1) * shard_size].to_csv(
            tsv_path, index=False, header=None, sep="\t"
        )


@cli.command(context_settings={"show_default": True})
@click.option(
    "--data-dir",
    type=click.Path(exists=True),
    default="./data",
    help="Data root directory",
)
@log_elapsed_time
def make_index_contents(**args: Any):
    """Make index contents"""
    args = AttrDict(args)
    train_data, test_data, _ = load_data(args.data_dir)
    data_list = [
        (train_data, os.path.join(args.data_dir, "index", "train", "doc.json")),
        (test_data, os.path.join(args.data_dir, "index", "test", "doc.json")),
    ]

    for data, index_path in data_list:
        contents = _get_contents(data)
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        with open(index_path, "w", encoding="utf8") as f:
            json.dump(contents, f)


@cli.command(context_settings={"show_default": True})
@click.option(
    "--data-dir",
    type=click.Path(exists=True),
    default="./data",
    help="Data root directory",
)
@click.option("--top-k", type=click.INT, default=1000, help="Top k candidates")
@click.option("--shard-size", type=click.INT, default=10000, help="Shard size")
@log_elapsed_time
def make_shard(**args: Any):
    """Make topk candidates shard"""
    args = AttrDict(args)
    train_data, _, _ = load_data(args.data_dir)
    train_queries, _, _ = preprocess_data(train_data)

    test_question = pd.read_csv(
        os.path.join(args.data_dir, "test_questions.csv"), encoding="utf8"
    )
    test_queries = dict(
        zip(test_question["question_id"], test_question["question_text"])
    )
    _make_shard(
        train_queries, os.path.join(args.data_dir, f"top{args.top_k}"), "train_queries"
    )
    _make_shard(
        test_queries,
        os.path.join(args.data_dir, f"top{args.top_k}"),
        "test_queries",
        len(test_queries),
    )
