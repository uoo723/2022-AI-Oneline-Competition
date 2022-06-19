#!/usr/bin/env bash

# export MLFLOW_TRACKING_URI=http://localhost:5000
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export MLFLOW_EXPERIMENT_NAME=ColBERT

DATASET=dataset
MODEL=colBERT
RUN_ID=${RUN_ID:-b6ec5451b76743229b9a40a41f53230a}
TOPK=${TOPK:-500}

args=(
    --model-name $MODEL
    --dataset-name $DATASET
    --run-script $0
    --mode "predict"
    --submission-output "./submissions/top${TOPK}_colbert_$RUN_ID.csv"
    --silent
    --run-id "$RUN_ID"
    --topk-candidates $TOPK
    --final-topk $TOPK
    --test-batch-size 64
    --query-max-length 80
    --passage-max-length 512
)

python main.py train-colbert "${args[@]}"
