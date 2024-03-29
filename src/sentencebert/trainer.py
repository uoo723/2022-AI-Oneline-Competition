"""
Created on 2022/06/10
@author Sangwoo Han
"""
import os
from collections import OrderedDict
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple

import click
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
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
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .. import base_trainer
from ..base_trainer import BaseTrainerModel, get_ckpt_path, load_model_hparams
from ..data import load_data, preprocess_data
from ..metrics import get_mrr
from ..utils import AttrDict, copy_file, filter_arguments, get_num_batches
from .datasets import Dataset, bert_collate_fn
from .loss import CircleLoss, get_similarity
from .models import SentenceBERT

BATCH = Tuple[Dict[str, torch.Tensor], torch.Tensor]


class SentenceBERTTrainerModel(BaseTrainerModel):
    MODEL_HPARAMS: Iterable[str] = {
        "pretrained_model_name",
        "n_feature_layers",
        "proj_dropout",
    }

    def __init__(
        self,
        pretrained_model_name: str = "monologg/koelectra-base-v3-discriminator",
        n_feature_layers: int = 1,
        proj_dropout: float = 0.5,
        query_max_length: int = 60,
        passage_max_length: int = 512,
        shard_idx: List[str] = [0],
        shard_size: int = 10000,
        topk_candidates: int = 50,
        final_topk: int = 10,
        num_neg: int = 1,
        num_pos: int = 1,
        margin: float = 0.15,
        gamma: float = 1.0,
        metric: str = "cosine",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.pretrained_model_name = pretrained_model_name
        self.n_feature_layers = n_feature_layers
        self.proj_dropout = proj_dropout
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length
        self.shard_idx = shard_idx
        self.shard_size = shard_size
        self.topk_candidates = topk_candidates
        self.final_topk = final_topk
        self.num_neg = num_neg
        self.num_pos = num_pos
        self.metric = metric
        self.loss_fn = CircleLoss(margin, gamma, metric)
        self.save_hyperparameters(ignore=self.IGNORE_HPARAMS)

    @property
    def model_hparams(self) -> Iterable[str]:
        return SentenceBERTTrainerModel.MODEL_HPARAMS

    def prepare_data(self) -> None:
        self.train_data, _, _ = load_data(self.data_dir)

        self.train_queries, self.train_docs, self.train_query_to_docs = preprocess_data(
            self.train_data
        )

        self.train_query_ids = list(self.train_queries.keys())
        self.train_query_str = list(self.train_queries.values())
        query_id_i2s = dict(
            zip(np.arange(len(self.train_query_str)).tolist(), self.train_query_ids)
        )

        self.sub_train_query_ids = []
        self.sub_train_query_str = []
        self.candidates = {}

        logger.info("Loading shard...")
        for shard_idx in self.shard_idx:
            filename = f"train_top1000_{shard_idx:02d}.txt"
            tsv_path = os.path.join(self.data_dir, "top1000", filename)
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
                num_pos=self.num_pos,
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
                bert_collate_fn,
                tokenizer=self.tokenizer,
                query_max_length=self.query_max_length,
                passage_max_length=self.passage_max_length,
            ),
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            collate_fn=partial(
                bert_collate_fn,
                tokenizer=self.tokenizer,
                query_max_length=self.query_max_length,
                passage_max_length=self.passage_max_length,
            ),
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            collate_fn=partial(
                bert_collate_fn,
                tokenizer=self.tokenizer,
                query_max_length=self.query_max_length,
                passage_max_length=self.passage_max_length,
            ),
        )

    def setup_model(self, stage: Optional[str] = None) -> None:
        if self.model is not None:
            return

        if self.run_id is not None:
            hparams = load_model_hparams(self.log_dir, self.run_id, self.model_hparams)
        else:
            hparams = {param: getattr(self, param) for param in self.model_hparams}

        self.model = SentenceBERT(
            **filter_arguments(hparams, SentenceBERT),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)

    def training_step(self, batch: BATCH, _) -> STEP_OUTPUT:
        query_inputs: Dict[str, torch.Tensor] = batch["query_inputs"]
        pos_doc_inputs: Dict[str, torch.Tensor] = batch["pos_doc_inputs"]
        pos_doc_len: torch.Tensor = batch["pos_doc_len"]
        neg_doc_inputs: Dict[str, torch.Tensor] = batch["neg_doc_inputs"]
        neg_doc_len: torch.Tensor = batch["neg_doc_len"]

        batch_size = len(pos_doc_len)
        anchor: torch.Tensor = self.model(query_inputs)
        pos: torch.Tensor = self.model(pos_doc_inputs)
        neg: torch.Tensor = self.model(neg_doc_inputs)

        pos = pos.reshape(batch_size, pos_doc_len[0], -1)
        neg = neg.reshape(batch_size, neg_doc_len[0], -1)

        loss = self.loss_fn(anchor, pos, neg)
        self.log("loss/train", loss, batch_size=batch_size)
        return loss

    def _validation_and_test_step(
        self, batch: BATCH, is_val: bool = True
    ) -> Optional[STEP_OUTPUT]:
        query_inputs: Dict[str, torch.Tensor] = batch["query_inputs"]
        candidate_doc_inputs: Dict[str, torch.Tensor] = batch["pos_doc_inputs"]
        candidate_doc_len: torch.Tensor = batch["pos_doc_len"]

        batch_size = len(candidate_doc_len)

        anchor: torch.Tensor = self.model(query_inputs)
        scores = []
        indices = []
        losses = []
        bs = 0
        for i, doc_len in enumerate(candidate_doc_len):
            be = bs + doc_len.item()
            num_batches = get_num_batches(batch_size, doc_len.item())
            candidate = []
            for b in range(num_batches):
                candidate.append(
                    self.model(
                        {
                            k: v[bs:be][b * batch_size : (b + 1) * batch_size]
                            for k, v in candidate_doc_inputs.items()
                        }
                    )
                )
            bs = be
            candidate = torch.cat(candidate).unsqueeze(0)
            sim = get_similarity(anchor[i : i + 1], candidate)
            s, idx = torch.topk(sim, k=self.final_topk)
            scores.append(s.sigmoid().float().cpu())
            indices.append(idx.cpu())
            if is_val:
                losses.append(self.loss_fn(anchor[i : i + 1], candidate[:, :1]))

        scores = torch.stack(scores)
        indices = torch.stack(indices)

        if is_val:
            losses = torch.stack(losses)
            self.log("loss/val", losses.mean(), batch_size=batch_size)

        return scores, indices

    def _validation_and_test_epoch_end(
        self, outputs: EPOCH_OUTPUT, is_val: bool = True
    ) -> None:
        _, indices = zip(*outputs)
        indices = np.concatenate(indices)
        prediction = {}
        for query_id, rank_indices in zip(
            self.sub_train_query_ids[self.valid_ids][: len(indices)], indices
        ):
            prediction[query_id] = [
                self.candidates[query_id][: self.topk_candidates][idx]
                for idx in rank_indices
            ]

        val_y_true = {
            k: self.train_query_to_docs[k]
            for k in self.sub_train_query_ids[self.valid_ids][: len(indices)]
        }

        mrr = get_mrr(val_y_true, prediction)

        if is_val:
            self.log_dict({"val/mrr": mrr}, prog_bar=True)
        else:
            self.log_dict({"test/mrr": mrr})

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
    valid_model_name = ["sentenceBERT"]
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
        SentenceBERTTrainerModel,
        is_hptuning=is_hptuning,
        trial=trial,
        enable_trial_pruning=enable_trial_pruning,
    )


def test(
    args: AttrDict, trainer: Optional[pl.Trainer] = None, is_hptuning: bool = False
) -> Dict[str, float]:
    return base_trainer.test(
        args,
        SentenceBERTTrainerModel,
        metrics=["mrr"],
        trainer=trainer,
        is_hptuning=is_hptuning,
    )


def predict(args: AttrDict) -> Any:
    assert args.mode == "predict", "mode must be predict"
    assert args.run_id is not None, "run_id must be specified"
    assert args.submission_output is not None, "submission output must be specified"

    logger.info(f"run_id: {args.run_id}")
    logger.info(f"submission_output: {args.submission_output}")

    if os.path.exists(args.submission_output):
        click.confirm(
            f"{os.path.basename(args.submission_output)} is already existed."
            " Overwrite it?",
            abort=True,
        )

    ############################# Save runscript #######################################
    os.makedirs(os.path.dirname(args.submission_output), exist_ok=True)
    if args.run_script:
        dirname = os.path.dirname(args.submission_output)
        basename, ext = os.path.splitext(os.path.basename(args.run_script))
        basename += "_" + os.path.splitext(os.path.basename(args.submission_output))[0]
        copy_file(args.run_script, os.path.join(dirname, basename + ext))
    ####################################################################################

    ################################# Load Data ########################################
    logger.info("Load Data...")
    _, test_data, test_question = load_data(args.data_dir)
    _, test_docs = preprocess_data(test_data, return_query_to_docs=False)
    test_queries = dict(
        zip(test_question["question_id"], test_question["question_text"])
    )
    test_query_id_i2s = dict(zip(range(len(test_queries)), test_queries.keys()))

    tsv_path = os.path.join(args.data_dir, "top1000", "test_top1000_00.txt")
    df = pd.read_csv(tsv_path, sep=" ", header=None)
    df[0] = df[0].map(lambda x: test_query_id_i2s[x])
    test_candidates: Dict[str, List[str]] = df.groupby(0)[2].apply(list).to_dict()
    ####################################################################################

    ################################## Load Model ######################################
    logger.info("Load Model...")
    hparams = load_model_hparams(
        args.log_dir, args.run_id, SentenceBERTTrainerModel.MODEL_HPARAMS
    )

    model = SentenceBERT(**filter_arguments(hparams, SentenceBERT))

    ckpt_path = get_ckpt_path(args.log_dir, args.run_id, load_best=True)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    swa_callback_key = None
    callbacks: Dict[str, Any] = ckpt["callbacks"]
    for key in callbacks.keys():
        if "StochasticWeightAveraging" in key:
            swa_callback_key = key
            break

    state_dict: Dict[str, torch.Tensor] = ckpt["state_dict"]

    if swa_callback_key is not None and "average_model" in callbacks[swa_callback_key]:
        logger.info("Use averaged weights")
        avg_state_dict: Dict[str, torch.Tensor] = callbacks[swa_callback_key][
            "average_model"
        ]
        avg_state_dict.pop("models_num")
        state_dict.update(avg_state_dict)

    state_dict = OrderedDict(
        zip(
            [key.replace("model.", "") for key in state_dict.keys()],
            state_dict.values(),
        )
    )

    model.load_state_dict(state_dict)
    model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(hparams["pretrained_model_name"])
    ####################################################################################

    ################################## Inference #######################################
    logger.info("Start Inference")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    batch_size = args.test_batch_size
    max_length = args.max_length
    topk = args.topk_candidates
    answers = []

    model.eval()
    for q_id, doc_ids in tqdm(test_candidates.items(), desc="inference..."):
        query_str = test_queries[q_id]
        doc_ids = np.array(doc_ids)
        num_batches = get_num_batches(batch_size, topk)
        predictions = []
        for b in range(num_batches):
            doc_str = [
                test_docs[d_id]
                for d_id in doc_ids[:topk][b * batch_size : (b + 1) * batch_size]
            ]
            inputs: Dict[str, torch.Tensor] = tokenizer(
                [query_str] * len(doc_str),
                doc_str,
                return_tensors="pt",
                max_length=max_length,
                padding="max_length",
                truncation="longest_first",
            )
            with torch.no_grad():
                outputs: torch.Tensor = model(
                    {k: v.to(args.device) for k, v in inputs.items()}
                )
            predictions.append(outputs.cpu())
        rank = np.concatenate(predictions).argsort()[::-1]
        answers.append(",".join(doc_ids[:topk][rank][: args.final_topk]))

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    ####################################################################################

    ############################### Make Submission ####################################
    logger.info("Make submission")
    submission = pd.DataFrame(
        data={"question_id": test_question["question_id"], "paragraph_id": answers}
    )
    submission.to_csv(args.submission_output, index=False)
    logger.info(f"Saved into {args.submission_output}")
    ####################################################################################

    return submission
