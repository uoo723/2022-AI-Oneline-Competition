#!/usr/bin/env bash
set -e

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

DATA_DIR=./data
TOPK=1000

if [[ -z "${SOURCE_DATA_DIR}" ]]; then
  mkdir -p $DATA_DIR
  rsync -ahP ${SOURCE_DATA_DIR} $DATA_DIR
fi

python main.py make-index-contents --data-dir $DATA_DIR

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input $DATA_DIR/index/train/ \
  --language ko \
  --index $DATA_DIR/train.index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 16 \
  --storePositions --storeDocvectors --storeRaw

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input $DATA_DIR/index/test/ \
  --language ko \
  --index $DATA_DIR/test.index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 16 \
  --storePositions --storeDocvectors --storeRaw

python main.py make-shard --data-dir $DATA_DIR --top-k $TOPK

for i in $(seq 0 23); do
    python -m pyserini.search.lucene \
    --index $DATA_DIR/train.index \
    --topics $DATA_DIR/top$TOPK/train_queries_$(printf %02d $i).tsv \
    --output $DATA_DIR/top$TOPK/train_top${TOPK}_$(printf %02d $i).txt \
    --bm25 --language ko --hits $TOPK --batch-size 256 --threads 32
done

python -m pyserini.search.lucene \
    --index $DATA_DIR/test.index \
    --topics $DATA_DIR/top$TOPK/test_queries_00.tsv \
    --output $DATA_DIR/top$TOPK/test_top${TOPK}_00.txt \
    --bm25 --language ko --hits $TOPK --batch-size 256 --threads 32
