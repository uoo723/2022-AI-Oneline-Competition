"""
Created on 2022/06/07
@author Sangwoo Han
"""
from typing import Any, Callable, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from logzero import logger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException

MODEL = Union[nn.DataParallel, nn.Module]
_AVG_FN = Callable[[torch.Tensor, torch.Tensor, torch.LongTensor], torch.FloatTensor]


class MLFlowExceptionCallback(Callback):
    def on_exception(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        exception: BaseException,
    ) -> None:
        logger = pl_module.logger
        if logger.experiment.get_run(logger.run_id):
            logger.experiment.set_terminated(logger.run_id, status="FAILED")

    def on_test_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        logger = pl_module.logger
        if logger.experiment.get_run(logger.run_id):
            logger.experiment.set_terminated(logger.run_id, status="RUNNING")

    def on_test_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        logger = pl_module.logger
        if logger.experiment.get_run(logger.run_id):
            logger.experiment.set_terminated(logger.run_id, status="FINISHED")


def swa_init(
    model: MODEL, swa_state: Dict[str, torch.Tensor], verbose: bool = True
) -> None:
    if verbose:
        logger.info("SWA Initializing")

    if isinstance(model, nn.DataParallel):
        model = model.module

    swa_state["models_num"] = 1
    for n, p in model.named_parameters():
        swa_state[n] = p.data.clone().detach()


def swa_step(
    model: MODEL, swa_state: Dict[str, torch.Tensor], verbose: bool = True
) -> None:
    if verbose:
        logger.info("SWA Step")

    if not swa_state:
        return

    if isinstance(model, nn.DataParallel):
        model = model.module

    swa_state["models_num"] += 1
    beta = 1.0 / swa_state["models_num"]
    with torch.no_grad():
        for n, p in model.named_parameters():
            swa_state[n].mul_(1.0 - beta).add_(p.data, alpha=beta)


def swap_swa_params(
    model: MODEL, swa_state: Dict[str, torch.Tensor], verbose: bool = True
) -> None:
    if verbose:
        logger.info("SWA Swap Params")

    if not swa_state:
        return

    if isinstance(model, nn.DataParallel):
        model = model.module

    for n, p in model.named_parameters():
        p.data, swa_state[n] = swa_state[n], p.data


class StochasticWeightAveraging(Callback):
    def __init__(
        self,
        swa_epoch_start: Union[int, float] = 0.8,
        swa_lrs: Optional[Union[float, List[float]]] = None,
        annealing_epochs: int = 10,
        annealing_strategy: str = "cos",
        avg_fn: Optional[_AVG_FN] = None,
        device: Optional[torch.device] = torch.device("cpu"),
    ):
        err_msg = "swa_epoch_start should be a >0 integer or a float between 0 and 1."
        if isinstance(swa_epoch_start, int) and swa_epoch_start < 1:
            raise MisconfigurationException(err_msg)
        if isinstance(swa_epoch_start, float) and not (0 <= swa_epoch_start <= 1):
            raise MisconfigurationException(err_msg)

        wrong_type = not isinstance(swa_lrs, (float, list))
        wrong_float = isinstance(swa_lrs, float) and swa_lrs <= 0
        wrong_list = isinstance(swa_lrs, list) and not all(
            lr > 0 and isinstance(lr, float) for lr in swa_lrs
        )
        if swa_lrs is not None and (wrong_type or wrong_float or wrong_list):
            raise MisconfigurationException(
                "The `swa_lrs` should be `None`, a positive float, or a list of positive floats"
            )

        if avg_fn is not None and not isinstance(avg_fn, Callable):
            raise MisconfigurationException("The `avg_fn` should be callable.")

        self._swa_epoch_start = swa_epoch_start
        self._swa_lrs = swa_lrs
        self._annealing_epochs = annealing_epochs
        self._annealing_strategy = annealing_strategy
        self._avg_fn = avg_fn or self.avg_fn
        self._device = device
        self._model_contains_batch_norm = None
        self._average_model = None
        self._should_stop = False
        self._is_transfer_device = True

    @property
    def state_key(self) -> str:
        return self._generate_state_key(
            swa_epoch_start=self._swa_epoch_start,
            annealing_strategy=self._annealing_strategy,
        )

    @property
    def swa_start(self) -> int:
        return max(self._swa_epoch_start - 1, 0)  # 0-based

    @property
    def swa_end(self) -> int:
        return self._max_epochs - 1  # 0-based

    @staticmethod
    def pl_module_contains_batch_norm(pl_module: "pl.LightningModule"):
        return any(
            isinstance(module, nn.modules.batchnorm._BatchNorm)
            for module in pl_module.modules()
        )

    def transfer_device(self, pl_module: "pl.LightningModule") -> None:
        if not self._average_model or self._is_transfer_device:
            return

        for n, p in pl_module.named_parameters():
            self._average_model[n] = self._average_model[n].to(p.device)

        self._is_transfer_device = True

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # optimizers = trainer.optimizers
        # lr_schedulers = trainer.lr_schedulers

        # if len(optimizers) != 1:
        #     raise MisconfigurationException("SWA currently works with 1 `optimizer`.")

        # if len(lr_schedulers) > 1:
        #     raise MisconfigurationException(
        #         "SWA currently not supported for more than 1 `lr_scheduler`."
        #     )

        if isinstance(self._swa_epoch_start, float):
            self._swa_epoch_start = int(trainer.max_epochs * self._swa_epoch_start)

        self._max_epochs = trainer.max_epochs

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        if trainer.current_epoch == self.swa_start:
            self._average_model = {}
            swa_init(pl_module, self._average_model, verbose=False)

    def on_validation_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if self._average_model:
            self.transfer_device(pl_module)
            swa_step(pl_module, self._average_model, verbose=False)
            swap_swa_params(pl_module, self._average_model, verbose=False)

    def on_validation_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if self._average_model:
            swap_swa_params(pl_module, self._average_model, verbose=False)

    def on_test_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if self._average_model:
            self.transfer_device(pl_module)
            swap_swa_params(pl_module, self._average_model, verbose=False)

    def on_test_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if self._average_model:
            swap_swa_params(pl_module, self._average_model, verbose=False)

    def on_predict_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if self._average_model:
            self.transfer_device(pl_module)
            swap_swa_params(pl_module, self._average_model, verbose=False)

    def on_predict_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if self._average_model:
            swap_swa_params(pl_module, self._average_model, verbose=False)

    def on_save_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        checkpoint: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self._average_model is not None:
            return {
                "average_model": self._average_model,
            }
        return {}

    def on_load_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        callback_state: Dict[str, Any],
    ) -> None:
        if "average_model" in callback_state:
            self._is_transfer_device = False
            self._average_model = callback_state["average_model"]

    @staticmethod
    def transfer_weights(
        src_pl_module: "pl.LightningModule", dst_pl_module: "pl.LightningModule"
    ):
        for src_param, dst_param in zip(
            src_pl_module.parameters(), dst_pl_module.parameters()
        ):
            dst_param.detach().copy_(src_param.to(dst_param.device))

    @staticmethod
    def swap_weights(
        src_pl_module: "pl.LightningModule", dst_pl_module: "pl.LightningModule"
    ):
        for src_param, dst_param in zip(
            src_pl_module.parameters(), dst_pl_module.parameters()
        ):
            src_param.data, dst_param.data = dst_param.data, src_param.data

    def reset_batch_norm_and_save_state(self, pl_module: "pl.LightningModule"):
        """Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L140-L154."""
        self.momenta = {}
        for module in pl_module.modules():
            if not isinstance(module, nn.modules.batchnorm._BatchNorm):
                continue
            module.running_mean = torch.zeros_like(
                module.running_mean,
                device=pl_module.device,
                dtype=module.running_mean.dtype,
            )
            module.running_var = torch.ones_like(
                module.running_var,
                device=pl_module.device,
                dtype=module.running_var.dtype,
            )
            self.momenta[module] = module.momentum
            module.momentum = None
            module.num_batches_tracked *= 0

    def reset_momenta(self):
        """Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L164-L165."""
        for bn_module in self.momenta:
            bn_module.momentum = self.momenta[bn_module]

    @staticmethod
    def update_parameters(
        average_model: "pl.LightningModule",
        model: "pl.LightningModule",
        n_averaged: torch.LongTensor,
        avg_fn: _AVG_FN,
    ):
        """Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L104-L112."""
        for p_swa, p_model in zip(average_model.parameters(), model.parameters()):
            device = p_swa.device
            p_swa_ = p_swa.detach()
            p_model_ = p_model.detach().to(device)
            src = (
                p_model_
                if n_averaged == 0
                else avg_fn(p_swa_, p_model_, n_averaged.to(device))
            )
            p_swa_.copy_(src)

        n_averaged += 1

    @staticmethod
    def avg_fn(
        averaged_model_parameter: torch.Tensor,
        model_parameter: torch.Tensor,
        num_averaged: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L95-L97."""
        return averaged_model_parameter + (
            model_parameter - averaged_model_parameter
        ) / (num_averaged + 1)
