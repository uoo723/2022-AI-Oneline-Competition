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
    --mode "predict"
    --submission-output "./submissions/submission3.csv"
    --run-id "bd45f678b6a041cb875ed1bb84b6bee4"
    --topk-candidates 100
    --test-batch-size 100
)

python main.py train-monobert "${args[@]}"
