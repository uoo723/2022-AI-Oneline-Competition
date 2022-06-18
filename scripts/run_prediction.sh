#!/usr/bin/env bash
set -e

# export MLFLOW_TRACKING_URI=http://localhost:5000
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

TMP_DIR=./tmp
COLBERT_RUN_ID=$(cat $TMP_DIR/colBERT_run_id)
MONOBERT_RUN_ID=$(cat $TMP_DIR/monoBERT_run_id)
TOPK=500
TOPK_FILE=top${TOPK}_colbert_${COLBERT_RUN_ID}.csv

RUN_ID=$COLBERT_RUN_ID TOPK=$TOPK ./scripts/run_colbert_prediction.sh
RUN_ID=$MONOBERT_RUN_ID TOPK_FILE=$TOPK_FILE ./scripts/run_monobert_prediction.sh