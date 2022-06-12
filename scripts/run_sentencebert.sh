#!/usr/bin/env bash

# export MLFLOW_TRACKING_URI=http://localhost:5000
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export MLFLOW_EXPERIMENT_NAME=SentenceBERT

DATASET=dataset
MODEL=sentenceBERT

args=(
    --model-name $MODEL
    --dataset-name $DATASET
    --run-script $0
    # --pretrained-model-name "monologg/koelectra-small-v3-discriminator"
    --pretrained-model-name "monologg/koelectra-base-v3-discriminator"
    # --pretrained-model-name "gogamza/kobart-base-v2"
    --n-feature-layers 5
    --optim-name "adamw"
    --lr 2.5e-5
    --num-epochs 10
    --train-batch-size 4
    --test-batch-size 32
    --accumulation-step 8
    --early-criterion 'mrr'
    --seed $1
    --swa-warmup 1
    --eval-step 1000
    --early 5
    --mp-enabled
    --gradient-max-norm 5.0
    --num-workers 8
    --experiment-name "SentenceBERT"
    # --run-id "efe8a17d663645389ff4b92f9f82da44"
    # --reset-early
    --query-max-length 60
    --passage-max-length 512
    --valid-size 500
    --num-pos 1
    --num-neg 5
    --topk-candidates 100
    --shard-idx 0
    --shard-idx 1
    --shard-idx 2
    --shard-idx 3
    --shard-idx 4
    --shard-idx 5
)

python main.py train-sentencebert "${args[@]}"
