#!/usr/bin/env bash
set -e

# export MLFLOW_TRACKING_URI=http://localhost:5000
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

TMP_DIR=./tmp
COLBERT_RUN_ID=${COLBERT_RUN_ID:-$(cat $TMP_DIR/colBERT_run_id)}
MONOBERT_RUN_ID=${MONOBERT_RUN_ID:-$(cat $TMP_DIR/monoBERT_run_id)}
TOPK=${TOPK:-500}
TOPK2=${TOPK2:-20}

if [[ -z "${SOURCE_DATA_DIR}" ]]; then
  mkdir -p ./data
  rsync -ahP ${SOURCE_DATA_DIR} ./data
fi

export SUBMISSION_FILE=${SUBMISSION_FILE:-submission24.csv}
export TOPK_FILE=top${TOPK}_colbert_${COLBERT_RUN_ID}.csv

RUN_ID=$COLBERT_RUN_ID TOPK=$TOPK ./scripts/run_colbert_prediction.sh
RUN_ID=$MONOBERT_RUN_ID TOPK=$TOPK2 ./scripts/run_monobert_prediction.sh
