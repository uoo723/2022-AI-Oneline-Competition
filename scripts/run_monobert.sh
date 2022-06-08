#!/usr/bin/env bash

# export MLFLOW_TRACKING_URI=http://localhost:5000
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export MLFLOW_EXPERIMENT_NAME=monoBERT

DATASET=dataset
MODEL=monoBERT

args=(
    --model-name $MODEL
    --dataset-name $DATASET
    --run-script $0
    --pretrained-model-name "monologg/koelectra-small-v3-discriminator"
    --optim-name "adamw"
    --lr 1e-5
    --num-epochs 5
    --train-batch-size 16
    --test-batch-size 1
    --accumulation-step 2
    --early-criterion 'mrr'
    --seed $1
    --swa-warmup 1
    --eval-step 300
    --early 5
    --mp-enabled
    --gradient-max-norm 5.0
    --num-workers 8
    --experiment-name "monoBERT"
    --max-length 512
    --valid-size 200
    --use-layernorm
    --shard-idx 0
    --shard-idx 1
)

python main.py train-monobert "${args[@]}"
