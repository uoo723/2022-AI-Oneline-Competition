#!/usr/bin/env bash
set -e

for i in $(seq 0 23); do
    python -m pyserini.search.lucene \
    --index data/train.index \
    --topics data/top1000/train_queries_$(printf %02d $i).tsv \
    --output data/top1000/train_top1000_$(printf %02d $i).txt \
    --bm25 --language ko --hits 1000 --batch-size 256 --threads 32
done
