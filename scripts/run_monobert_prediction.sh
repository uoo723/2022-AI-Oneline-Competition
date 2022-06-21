#!/usr/bin/env bash

# export MLFLOW_TRACKING_URI=http://localhost:5000
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export MLFLOW_EXPERIMENT_NAME=monoBERT

DATASET=dataset
MODEL=monoBERT

RUN_ID=${RUN_ID:-4a3cbf97ae1d4f14b0c7e4099a179c76}
TOPK_FILE=${TOPK_FILE:-top500_colbert_9d84388f242e44c289c7f459aa95bdca.csv}

args=(
    --model-name $MODEL
    --dataset-name $DATASET
    --run-script $0
    --mode "predict"
    --submission-output "./submissions/submission23.csv"
    --run-id "$RUN_ID"
    --topk-candidates 50
    --test-batch-size 50
    --topk-filepath "./submissions/$TOPK_FILE"
)

python main.py train-monobert "${args[@]}"