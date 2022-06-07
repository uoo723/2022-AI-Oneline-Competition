"""
Created on 2022/06/07
@author Sangwoo Han
"""
from typing import Any, Optional

import click
from click_option_group import optgroup
from logzero import logger
from optuna import Trial

from main import cli
from src.utils import AttrDict, log_elapsed_time, save_args


class FloatIntParamType(click.ParamType):
    name = "float|int"

    def convert(self, value, param, ctx):
        try:
            value = float(value)
            new_value = int(value)
            if new_value == value:
                return new_value
            return value
        except ValueError:
            self.fail(f"{value!r} is not a valid float|int", param, ctx)


FLOAT_INT = FloatIntParamType()

# fmt: off

_train_options = [
    optgroup.group("Train Options"),
    optgroup.option(
        "--mode",
        type=click.Choice(["train", "test", "predict"]),
        default="train",
        help="train: train and test are executed. test: test only, predict: make prediction",
    ),
    optgroup.option("--run-id", type=click.STRING, help="MLFlow Run ID for resume training"),
    optgroup.option("--model-name", type=click.STRING, required=True, help="Model name"),
    optgroup.option("--dataset-name", type=click.STRING, required=True, help="Dataset name"),
    optgroup.option("--valid-size", type=FLOAT_INT, default=0.2, help="Validation dataset size"),
    optgroup.option("--seed", type=click.INT, default=0, help="Seed for reproducibility"),
    optgroup.option("--swa-warmup", type=click.INT, default=10, help="Warmup for SWA. Disable: 0"),
    optgroup.option("--mp-enabled", is_flag=True, default=False, help="Enable Mixed Precision"),
    optgroup.option("--early", type=click.INT, default=50, help="Early stopping step"),
    optgroup.option("--early-criterion", type=click.Choice(["p5", "n5", "psp5", "loss"]), default="n5", help="Early stopping criterion"),
    optgroup.option("--eval-step", type=click.INT, default=100, help="Evaluation step during training"),
    optgroup.option("--num-epochs", type=click.INT, default=40, help="Total number of epochs"),
    optgroup.option("--train-batch-size", type=click.INT, default=128, help="Batch size for training"),
    optgroup.option("--test-batch-size", type=click.INT, default=256, help="Batch size for test"),
    optgroup.option("--no-cuda", is_flag=True, default=False, help="Disable cuda"),
    optgroup.option("--num-workers", type=click.INT, default=4, help="Number of workers for data loader"),
    optgroup.option("--lr", type=click.FLOAT, default=1e-3, help="learning rate"),
    optgroup.option("--decay", type=click.FLOAT, default=1e-2, help="Weight decay"),
    optgroup.option("--accumulation-step", type=click.INT, default=1, help="accumlation step for small batch size"),
    optgroup.option("--gradient-max-norm", type=click.FLOAT, help="max norm for gradient clipping"),
    optgroup.option("--model-cnf", type=click.Path(exists=True), help="Model config file path"),
    optgroup.option("--data-cnf", type=click.Path(exists=True), help="Data config file path"),
    optgroup.option("--optim-name", type=click.Choice(["adamw", "sgd"]), default="adamw", help="Choose optimizer"),
    optgroup.option("--scheduler-warmup", type=click.FloatRange(0, 1), help="Ratio of warmup among total training steps"),
    optgroup.option(
        "--scheduler-type",
        type=click.Choice(
            [
                "linear",
                "cosine",
                "cosine_with_restarts",
                "polynomial",
                "constant",
                "constant_with_warmup",
            ]
        ),
        help="Set type of scheduler",
    ),
]

_log_options = [
    optgroup.group("Log Options"),
    optgroup.option("--log-dir", type=click.Path(), default="./logs", help="log directory"),
    optgroup.option("--experiment-name", type=click.STRING, default="baseline", help="experiment name"),
    optgroup.option("--run-name", type=click.STRING, help="Set Run Name for MLFLow"),
    optgroup.option("--tags", type=(str, str), multiple=True, help="Set mlflow run tags"),
    optgroup.option("--run-script", type=click.Path(exists=True), help="Run script file path to log"),
]

_dataset_options = [
    optgroup.group("Dataset Options"),
    optgroup.option("--maxlen", type=click.INT, default=500, help="Max length of tokens"),
    optgroup.option("--pad", type=click.STRING, default="<PAD>", help="Pad token"),
    optgroup.option("--unknown", type=click.STRING, default="<UNK>", help="Unknown token"),
    optgroup.option("--num-cores", type=click.INT, default=-1, help="# cores for preprocessing"),
]

_monobert_options = [
    optgroup.group("monoBERT Options"),
    optgroup.option("--test-opt", type=click.INT, default=256, help="test opt"),
]

_duobert_options = [
    optgroup.group("duoBERT Options"),
    optgroup.option("--test-opt", type=click.INT, default=256, help="test opt"),
]

# fmt: on


def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


@cli.command(context_settings={"show_default": True})
@add_options(_train_options)
@add_options(_log_options)
@add_options(_dataset_options)
@add_options(_monobert_options)
@click.pass_context
def train_monobert(ctx: click.core.Context, **args: Any) -> None:
    """Train monoBERT"""
    if ctx.obj["save_args"] is not None:
        save_args(args, ctx.obj["save_args"])
        return
    train_model("monoBERT", **args)


@cli.command(context_settings={"show_default": True})
@add_options(_train_options)
@add_options(_log_options)
@add_options(_dataset_options)
@add_options(_duobert_options)
@click.pass_context
def train_laroberta(ctx: click.core.Context, **args: Any) -> None:
    """Train duoBERT"""
    if ctx.obj["save_args"] is not None:
        save_args(args, ctx.obj["save_args"])
        return
    train_model("duoBERT", **args)


@log_elapsed_time
def train_model(
    train_name: str,
    is_hptuning: bool = False,
    trial: Optional[Trial] = None,
    enable_trial_pruning: bool = False,
    **args: Any,
) -> None:
    assert train_name in ["monoBERT", "duoBERT"]
    if train_name == "monoBERT":
        import src.monobert.trainer as trainer
    elif train_name == "duoBERT":
        import src.duobert.trainer as trainer

    args = AttrDict(args)

    trainer.check_args(args)
    trainer.init_run(args)

    if args.mode == "predict":
        logger.info("Predict mode")
        return trainer.predict(args)

    pl_trainer = None
    if args.mode == "train":
        _, pl_trainer = trainer.train(
            args,
            is_hptuning=is_hptuning,
            trial=trial,
            enable_trial_pruning=enable_trial_pruning,
        )

    if args.mode == "test":
        logger.info("Test mode")

    try:
        return trainer.test(args, pl_trainer, is_hptuning=is_hptuning)
    except Exception as e:
        if pl_trainer:
            pl_logger = pl_trainer.logger
            pl_logger.experiment.set_terminated(pl_logger.run_id, status="FAILED")
        raise e
