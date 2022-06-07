"""
Created on 2022/06/07
@author Sangwoo Han
"""
import copy
import inspect
import json
import random
import time
from datetime import timedelta
from functools import wraps
from typing import Any, Dict

import numpy as np
import torch
from attrdict import AttrDict as _AttrDict
from logzero import logger


class AttrDict(_AttrDict):
    def _build(self, obj: Any) -> Any:
        return obj


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def filter_arguments(args: Dict[str, Any], obj: Any) -> Dict[str, Any]:
    if isinstance(obj, type):
        param_names = []
        for cls in inspect.getmro(obj):
            param_names.extend(list(inspect.signature(cls).parameters.keys()))
    else:
        param_names = list(inspect.signature(obj).parameters.keys())
    return {k: v for k, v in args.items() if k in param_names}


def log_elapsed_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()

        elapsed = end - start
        logger.info(f"elapsed time: {end - start:.2f}s, {timedelta(seconds=elapsed)}")

        return ret

    return wrapper


def save_args(args: AttrDict, path: str) -> None:
    args = copy.deepcopy(args)
    with open(path, "w", encoding="utf8") as f:
        json.dump(args, f, indent=4, ensure_ascii=False)


def load_args(path: str) -> AttrDict:
    with open(path, "r", encoding="utf8") as f:
        return AttrDict(json.load(f))
