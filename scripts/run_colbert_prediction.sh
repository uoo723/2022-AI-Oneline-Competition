#!/usr/bin/env bash

# export MLFLOW_TRACKING_URI=http://localhost:5000
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export MLFLOW_EXPERIMENT_NAME=ColBERT

DATASET=dataset
MODEL=colBERT

args=(
    --model-name $MODEL
    --dataset-name $DATASET
    --run-script $0
    --mode "predict"
    --submission-output "./submissions/submission8.csv"
    --run-id "7f26132ba3914fc1b85d656b358684ce"
    --topk-candidates 200
    --test-batch-size 50
    --query-max-length 60
    --passage-max-length 512
)

python main.py train-colbert "${args[@]}"
