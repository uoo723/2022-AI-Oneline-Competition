"""
Created on 2022/06/07
@author Sangwoo Han
"""
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from optuna import Trial
from pytorch_lightning.utilities.types import (
    EPOCH_OUTPUT,
    EVAL_DATALOADERS,
    STEP_OUTPUT,
    TRAIN_DATALOADERS,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Subset

from .. import base_trainer
from ..base_trainer import BaseTrainerModel
from ..utils import filter_arguments, AttrDict

BATCH = Tuple[Dict[str, torch.Tensor], torch.Tensor]


class MonoBERTTrainerModel(BaseTrainerModel):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=self.IGNORE_HPARAMS)

    @property
    def model_hparams(self) -> List[str]:
        return {}

    def prepare_data(self) -> None:
        pass

    def setup_dataset(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def setup_model(self, stage: Optional[str] = None) -> None:
        pass


def check_args(args: AttrDict) -> None:
    valid_early_criterion = []
    valid_model_name = ["monoBERT"]
    valid_dataset_name = []
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
        metrics=[],
        trainer=trainer,
        is_hptuning=is_hptuning,
    )


def predict(args: AttrDict) -> Any:
    raise NotImplemented
