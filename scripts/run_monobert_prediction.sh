#!/usr/bin/env bash

# export MLFLOW_TRACKING_URI=http://localhost:5000
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export MLFLOW_EXPERIMENT_NAME=monoBERT

DATASET=dataset
MODEL=monoBERT

RUN_ID=${RUN_ID:-e5f00870e79d42b392778ba40a171c01}
TOPK_FILE=${TOPK_FILE:-top500_colbert_b6ec5451b76743229b9a40a41f53230a.csv}
TOPK=${TOPK:-20}
SUBMISSION_FILE=${SUBMISSION_FILE:-submission24.csv}

args=(
    --model-name $MODEL
    --dataset-name $DATASET
    --run-script $0
    --mode "predict"
    --submission-output "./submissions/$SUBMISSION_FILE"
    --run-id "$RUN_ID"
    --topk-candidates $TOPK
    --test-batch-size $TOPK
    --topk-filepath "./submissions/$TOPK_FILE"
)

python main.py train-monobert "${args[@]}"
