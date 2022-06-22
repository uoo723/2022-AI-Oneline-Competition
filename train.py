"""
Created on 2022/06/07
@author Sangwoo Han
"""
import os
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
    optgroup.option("--skip-test", is_flag=True, default=False, help="If set to true, skip test after training"),
    optgroup.option("--run-id", type=click.STRING, help="MLFlow Run ID for resume training"),
    optgroup.option("--model-name", type=click.STRING, required=True, help="Model name"),
    optgroup.option("--dataset-name", type=click.STRING, required=True, default="dataset", help="Dataset name"),
    optgroup.option("--valid-size", type=FLOAT_INT, default=0.2, help="Validation dataset size"),
    optgroup.option("--seed", type=click.INT, default=0, help="Seed for reproducibility"),
    optgroup.option("--swa-warmup", type=click.INT, default=10, help="Warmup for SWA. Disable: 0"),
    optgroup.option("--mp-enabled", is_flag=True, default=False, help="Enable Mixed Precision"),
    optgroup.option("--early", type=click.INT, default=10, help="Early stopping step"),
    optgroup.option("--reset-early", is_flag=True, default=False, help="Reset early"),
    optgroup.option("--early-criterion", type=click.Choice(["mrr", "loss"]), default="mrr", help="Early stopping criterion"),
    optgroup.option("--eval-step", type=click.INT, default=100, help="Evaluation step during training"),
    optgroup.option("--num-epochs", type=click.INT, default=40, help="Total number of epochs"),
    optgroup.option("--train-batch-size", type=click.INT, default=8, help="Batch size for training"),
    optgroup.option("--test-batch-size", type=click.INT, default=1, help="Batch size for test"),
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
    optgroup.option("--tmp-dir", type=click.Path(), default="./tmp", help="Temp file directory"),
    optgroup.option("--log-run-id", is_flag=True, default=False, help="Log run id to tmp dir"),
    optgroup.option("--experiment-name", type=click.STRING, default="baseline", help="experiment name"),
    optgroup.option("--run-name", type=click.STRING, help="Set Run Name for MLFLow"),
    optgroup.option("--tags", type=(str, str), multiple=True, help="Set mlflow run tags"),
    optgroup.option("--run-script", type=click.Path(exists=True), help="Run script file path to log"),
]

_dataset_options = [
    optgroup.group("Dataset Options"),
    optgroup.option("--data-dir", type=click.Path(exists=True), default="./data", help="Data root directory"),
    optgroup.option("--max-length", type=click.INT, default=512, help="Maximum length of tokens"),
    optgroup.option("--query-max-length", type=click.INT, default=60, help="Maximum length of query tokens"),
    optgroup.option("--passage-max-length", type=click.INT, default=512, help="Maximum length of passage tokens"),
    optgroup.option("--shard-idx", type=click.INT, multiple=True, help="Select shard idx"),
    optgroup.option("--shard-size", type=click.INT, default=10000, help="Size of shard"),
    optgroup.option("--topk-candidates", type=click.INT, default=50, help="Topk candidates"),
    optgroup.option("--final-topk", type=click.INT, default=10, help="Final topk predction"),
    optgroup.option("--num-neg", type=click.INT, default=1, help="# of negative samples"),
    optgroup.option("--num-pos", type=click.INT, default=1, help="# of positive samples"),
]

_submission_options = [
    optgroup.group("Submission Options"),
    optgroup.option("--submission-output", type=click.Path(), default="./submissions/submission.csv", help="Output path for submssion"),
    optgroup.option("--topk-filepath", type=click.Path(exists=True), help="Topk candidate list files"),
    optgroup.option("--silent", is_flag=True, default=False, help="If submission file exists, exit without prompt"),
    optgroup.option("--overwrite", is_flag=True, default=False, help="Submssion file will be overwritten without prompt"),
    optgroup.option("--load-last", is_flag=True, default=False, help="Load last model instead of best model"),
]

_monobert_options = [
    optgroup.group("monoBERT Options"),
    optgroup.option("--pretrained-model-name", type=click.STRING, default="monologg/koelectra-base-v3-discriminator", help="Bert pretrained model name"),
    optgroup.option('--linear-size', type=click.INT, multiple=True, default=[256], help="Linear size"),
    optgroup.option('--dropout', type=click.FloatRange(0, 1), default=0.2, help="Dropout for MLP"),
    optgroup.option("--use-layernorm", is_flag=True, default=False, help="Use layernorm in MLP"),
    optgroup.option("--n-feature-layers", type=click.INT, default=1, help="# of layers to be concatenated for outputs"),
    optgroup.option("--proj-dropout", type=click.FloatRange(0, 1), default=0.5, help="Dropout for projection layer"),
    optgroup.option("--use-conv", is_flag=True, default=False, help="Use conv layer to aggregate encoder outputs"),
    optgroup.option("--kernel-size", type=click.INT, default=2, help="Kernel size of Conv1d"),
]

_duobert_options = [
    optgroup.group("duoBERT Options"),
    optgroup.option("--test-opt", type=click.INT, default=256, help="test opt"),
]


_sentencebert_options = [
    optgroup.group("sentenceBERT Options"),
    optgroup.option("--pretrained-model-name", type=click.STRING, default="monologg/koelectra-base-v3-discriminator", help="Bert pretrained model name"),
    optgroup.option("--n-feature-layers", type=click.INT, default=1, help="# of layers to be concatenated for outputs"),
    optgroup.option("--proj-dropout", type=click.FloatRange(0, 1), default=0.5, help="Dropout for projection layer"),
    optgroup.option("--margin", type=click.FLOAT, default=0.15, help="Margin for circle loss"),
    optgroup.option("--gamma", type=click.FloatRange(0, 1, min_open=True), default=1.0, help="Scale factor for circle loss"),
    optgroup.option("--metric", type=click.Choice(["cosine", "euclidean"]), default="cosine", help="Metric for circle loss"),
]

_colbert_options = [
    optgroup.group("colBERT Options"),
    optgroup.option("--pretrained-model-name", type=click.STRING, default="monologg/koelectra-base-v3-discriminator", help="Bert pretrained model name"),
    optgroup.option("--n-feature-layers", type=click.INT, default=1, help="# of layers to be concatenated for outputs"),
    optgroup.option("--proj-dropout", type=click.FloatRange(0, 1), default=0.5, help="Dropout for projection layer"),
    optgroup.option("--margin", type=click.FLOAT, default=0.15, help="Margin for circle loss"),
    optgroup.option("--gamma", type=click.FloatRange(0, 1, min_open=True), default=1.0, help="Scale factor for circle loss"),
    optgroup.option("--metric", type=click.Choice(["cosine", "euclidean"]), default="cosine", help="Metric for circle loss"),
    optgroup.option("--use-transformer-late-interaction", is_flag=True, default=False, help="Use transformer late interaction"),
    optgroup.option("--linear-size", type=click.INT, multiple=True, default=[128], help="linear size for transformer late interaction"),
    optgroup.option('--linear-dropout', type=click.FloatRange(0, 1), default=0.2, help="Dropout for MLP of transformer late interaction"),
    optgroup.option('--dropout', type=click.FloatRange(0, 1), default=0.1, help="Dropout for transformer late interaction"),
    optgroup.option("--use-layernorm", is_flag=True, default=False, help="Use layernorm in MLP of transformer late interaction"),
    optgroup.option("--dim-feedforward", type=click.INT, default=1024, help="Feedforward dimension of transformer late interaction"),
    optgroup.option("--n-layers", type=click.INT, default=4, help="# of layers of transformer late interaction"),
    optgroup.option("--n-heads", type=click.INT, default=4, help="# of heads of transformer late interaction"),
    optgroup.option("--loss-type", type=click.Choice(['circle', 'bce']), default='circle', help="Type of loss function"),
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
@add_options(_submission_options)
@add_options(_monobert_options)
@click.pass_context
def train_monobert(ctx: click.core.Context, **args: Any) -> None:
    """Train monoBERT"""
    if ctx.obj["save_args"] is not None:
        save_args(args, ctx.obj["save_args"])
        return
    args["linear_size"] = list(args["linear_size"])
    args["shard_idx"] = list(args["shard_idx"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train_model("monoBERT", **args)


@cli.command(context_settings={"show_default": True})
@add_options(_train_options)
@add_options(_log_options)
@add_options(_dataset_options)
@add_options(_duobert_options)
@click.pass_context
def train_duobert(ctx: click.core.Context, **args: Any) -> None:
    """Train duoBERT"""
    if ctx.obj["save_args"] is not None:
        save_args(args, ctx.obj["save_args"])
        return
    args["shard_idx"] = list(args["shard_idx"])
    train_model("duoBERT", **args)


@cli.command(context_settings={"show_default": True})
@add_options(_train_options)
@add_options(_log_options)
@add_options(_dataset_options)
@add_options(_submission_options)
@add_options(_sentencebert_options)
@click.pass_context
def train_sentencebert(ctx: click.core.Context, **args: Any) -> None:
    """Train sentenceBERT"""
    if ctx.obj["save_args"] is not None:
        save_args(args, ctx.obj["save_args"])
        return
    args["shard_idx"] = list(args["shard_idx"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train_model("sentenceBERT", **args)


@cli.command(context_settings={"show_default": True})
@add_options(_train_options)
@add_options(_log_options)
@add_options(_dataset_options)
@add_options(_submission_options)
@add_options(_colbert_options)
@click.pass_context
def train_colbert(ctx: click.core.Context, **args: Any) -> None:
    """Train ColBERT"""
    if ctx.obj["save_args"] is not None:
        save_args(args, ctx.obj["save_args"])
        return
    args["linear_size"] = list(args["linear_size"])
    args["shard_idx"] = list(args["shard_idx"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train_model("colBERT", **args)


@log_elapsed_time
def train_model(
    train_name: str,
    is_hptuning: bool = False,
    trial: Optional[Trial] = None,
    enable_trial_pruning: bool = False,
    **args: Any,
) -> None:
    assert train_name in ["monoBERT", "duoBERT", "sentenceBERT", "colBERT"]
    if train_name == "monoBERT":
        import src.monobert.trainer as trainer
    elif train_name == "duoBERT":
        import src.duobert.trainer as trainer
    elif train_name == "sentenceBERT":
        import src.sentencebert.trainer as trainer
    elif train_name == "colBERT":
        import src.colbert.trainer as trainer

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

        if args.log_run_id:
            run_id_path = os.path.join(
                args.tmp_dir, f"{args.run_name or train_name}_run_id"
            )
            os.makedirs(os.path.dirname(run_id_path), exist_ok=True)
            with open(run_id_path, "w", encoding="utf8") as f:
                f.write(pl_trainer.logger.run_id)

    if args.mode == "test":
        logger.info("Test mode")

    if not args.skip_test:
        try:
            return trainer.test(args, pl_trainer, is_hptuning=is_hptuning)
        except Exception as e:
            if pl_trainer:
                pl_logger = pl_trainer.logger
                pl_logger.experiment.set_terminated(pl_logger.run_id, status="FAILED")
            raise e
