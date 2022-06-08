"""
Created on 2022/06/07
@author Sangwoo Han
"""
import json
import os
from collections import OrderedDict, defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from logzero import logger
from optuna import Trial
from pytorch_lightning.utilities.types import (
    EPOCH_OUTPUT,
    EVAL_DATALOADERS,
    STEP_OUTPUT,
    TRAIN_DATALOADERS,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from .. import base_trainer
from ..base_trainer import BaseTrainerModel
from ..datasets import Dataset, bert_collate_fn
from ..metrics import get_mrr
from ..utils import AttrDict, filter_arguments
from .models import MonoBERT

BATCH = Tuple[Dict[str, torch.Tensor], torch.Tensor]


class MonoBERTTrainerModel(BaseTrainerModel):
    def __init__(
        self,
        pretrained_model_name: str = "monologg/koelectra-base-v3-discriminator",
        linear_size: List[int] = [256],
        dropout: int = 0.2,
        use_layernorm: bool = False,
        data_dir_path: str = "./data",
        max_length: int = 512,
        shard_idx: List[str] = [0],
        shard_size: int = 10000,
        topk_candidates: int = 50,
        final_topk: int = 10,
        num_neg: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.pretrained_model_name = pretrained_model_name
        self.linear_size = linear_size
        self.dropout = dropout
        self.use_layernorm = use_layernorm
        self.data_dir_path = data_dir_path
        self.max_length = max_length
        self.shard_idx = shard_idx
        self.shard_size = shard_size
        self.topk_candidates = topk_candidates
        self.final_topk = final_topk
        self.num_neg = num_neg
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.save_hyperparameters(ignore=self.IGNORE_HPARAMS)

    @property
    def model_hparams(self) -> List[str]:
        return {
            "pretrained_model_name",
            "dropout",
            "linear_size",
            "use_layernorm",
        }

    def prepare_data(self) -> None:
        train_data_path = os.path.join(self.data_dir_path, "train.json")
        test_data_path = os.path.join(self.data_dir_path, "test_data.json")
        with open(train_data_path, "r") as f1, open(test_data_path, "r") as f2:
            self.train_data = json.load(f1)
            self.test_data = json.load(f2)

        # test_question = pd.read_csv("./data/test_questions.csv", encoding="utf8")
        # sample = pd.read_csv("./data/sample_submission.csv", encoding="utf8")

        self.train_queries = OrderedDict()
        self.train_query_to_docs = defaultdict(list)
        self.train_docs = OrderedDict()

        for data in self.train_data["data"]:
            for paragraph in data["paragraphs"]:
                for q in paragraph["qas"]:
                    self.train_queries[q["question_id"]] = q["question"]
                    self.train_query_to_docs[q["question_id"]].append(
                        paragraph["paragraph_id"]
                    )
                self.train_docs[paragraph["paragraph_id"]] = paragraph["context"]

        self.train_query_ids = list(self.train_queries.keys())
        self.train_query_str = list(self.train_queries.values())
        query_id_i2s = dict(
            zip(np.arange(len(self.train_query_str)).tolist(), self.train_query_ids)
        )

        self.sub_train_query_ids = []
        self.sub_train_query_str = []
        self.candidates = {}

        logger.info("Load sharding...")
        for shard_idx in self.shard_idx:
            filename = f"train_top1000_{shard_idx:02d}.txt"
            tsv_path = os.path.join(self.data_dir_path, "top1000", filename)
            df = pd.read_csv(tsv_path, sep=" ", header=None)
            df[0] = df[0].map(lambda x: query_id_i2s[x])
            self.candidates.update(df.groupby(0)[2].apply(list).to_dict())
            self.sub_train_query_ids.extend(
                self.train_query_ids[
                    shard_idx * self.shard_size : (shard_idx + 1) * self.shard_size
                ]
            )
            self.sub_train_query_str.extend(
                self.train_query_str[
                    shard_idx * self.shard_size : (shard_idx + 1) * self.shard_size
                ]
            )
            logger.info(f"Completed loading shard {shard_idx} among {self.shard_idx}")

        self.sub_train_query_ids = np.array(self.sub_train_query_ids)
        self.sub_train_query_str = np.array(self.sub_train_query_str)

    def setup_dataset(self, stage: Optional[str] = None) -> None:
        if stage == "predict":
            raise ValueError(f"{stage} stage is not supported")

        if self.train_dataset is None:
            dataset = Dataset(
                self.sub_train_query_ids,
                self.sub_train_query_str,
                self.train_docs,
                self.train_query_to_docs,
                self.candidates,
                topk=self.topk_candidates,
                num_neg=self.num_neg,
                is_training=True,
            )

            self.train_ids, self.valid_ids = train_test_split(
                np.arange(len(dataset)),
                test_size=self.valid_size,
                random_state=self.seed,
            )

            self.train_dataset = Subset(dataset, self.train_ids)
            self.val_dataset = Subset(dataset, self.valid_ids)

        if self.test_dataset is None:
            self.test_dataset = self.val_dataset

    def _set_dataset_mode(
        self, dataset: Subset[Dataset], is_training: bool = True
    ) -> None:
        dataset.dataset.is_training = is_training

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=partial(
                bert_collate_fn, tokenizer=self.tokenizer, max_length=self.max_length
            ),
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            collate_fn=partial(
                bert_collate_fn, tokenizer=self.tokenizer, max_length=self.max_length
            ),
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            collate_fn=partial(
                bert_collate_fn, tokenizer=self.tokenizer, max_length=self.max_length
            ),
        )

    def setup_model(self, stage: Optional[str] = None) -> None:
        if self.model is not None:
            return

        if self.run_id is not None:
            hparams = self.load_model_hparams()
        else:
            hparams = {param: getattr(self, param) for param in self.model_hparams}

        self.model = MonoBERT(
            **filter_arguments(hparams, MonoBERT),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)

    def training_step(self, batch: BATCH, _) -> STEP_OUTPUT:
        batch_x, batch_y = batch
        outputs = self.model(batch_x)
        loss = self.loss_fn(outputs, batch_y)
        self.log("loss/train", loss)
        return loss

    def _validation_and_test_step(
        self, batch: BATCH, is_val: bool = True
    ) -> Optional[STEP_OUTPUT]:
        batch_x, batch_y = batch
        outputs = self.model(batch_x)
        scores, indices = torch.topk(outputs, k=self.final_topk)
        scores = scores.sigmoid().float().cpu()
        indices = indices.cpu()

        if is_val:
            loss = self.loss_fn(outputs, batch_y)
            self.log("loss/val", loss)

        return scores, indices

    def _validation_and_test_epoch_end(
        self, outputs: EPOCH_OUTPUT, is_val: bool = True
    ) -> None:
        _, indices = zip(*outputs)
        indices = np.stack(indices)

        prediction = {}
        for query_id, rank_indices in zip(
            self.sub_train_query_ids[self.valid_ids][:len(indices)], indices
        ):
            prediction[query_id] = [
                self.candidates[query_id][: self.topk_candidates][idx]
                for idx in rank_indices
            ]

        val_y_true = {
                k: self.train_query_to_docs[k]
                for k in self.sub_train_query_ids[self.valid_ids][:len(indices)]
            }

        mrr = get_mrr(val_y_true, prediction)

        if is_val:
            self.log_dict({"val/mrr": mrr}, prog_bar=True)
        else:
            self.log_dict({"test/mrr": mrr}, prog_bar=True)

    def validation_step(self, batch: BATCH, _) -> Optional[STEP_OUTPUT]:
        return self._validation_and_test_step(batch, is_val=True)

    def test_step(self, batch: BATCH, _) -> Optional[STEP_OUTPUT]:
        return self._validation_and_test_step(batch, is_val=False)

    def on_train_epoch_start(self) -> None:
        self._set_dataset_mode(self.train_dataset, is_training=True)

    def on_validation_epoch_start(self) -> None:
        self._set_dataset_mode(self.val_dataset, is_training=False)

    def on_test_epoch_start(self) -> None:
        self._set_dataset_mode(self.test_dataset, is_training=False)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self._validation_and_test_epoch_end(outputs, is_val=True)

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self._validation_and_test_epoch_end(outputs, is_val=False)


def check_args(args: AttrDict) -> None:
    valid_early_criterion = ["mrr"]
    valid_model_name = ["monoBERT"]
    valid_dataset_name = ["dataset"]
    base_trainer.check_args(
        args, valid_early_criterion, valid_model_name, valid_dataset_name
    )


def init_run(args: AttrDict) -> None:
    base_trainer.init_run(args)


def train(
    args: AttrDict,
    is_hptuning: bool = False,
    trial: Optional[Trial] = None,
    enable_trial_pruning: bool = False,
) -> Tuple[float, pl.Trainer]:
    return base_trainer.train(
        args,
        MonoBERTTrainerModel,
        is_hptuning=is_hptuning,
        trial=trial,
        enable_trial_pruning=enable_trial_pruning,
    )


def test(
    args: AttrDict, trainer: Optional[pl.Trainer] = None, is_hptuning: bool = False
) -> Dict[str, float]:
    return base_trainer.test(
        args,
        MonoBERTTrainerModel,
        metrics=["mrr"],
        trainer=trainer,
        is_hptuning=is_hptuning,
    )


def predict(args: AttrDict) -> Any:
    raise NotImplemented
