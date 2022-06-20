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
    --pretrained-model-name "monologg/koelectra-small-v3-discriminator"
    # --pretrained-model-name "monologg/koelectra-base-v3-discriminator"
    # --pretrained-model-name "hyunwoongko/kobart"
    # --pretrained-model-name "gogamza/kobart-base-v2"
    --skip-test
    --log-run-id
    --n-feature-layers 5
    --topk-candidates 100
    # --use-transformer-late-interaction
    # --use-layernorm
    # --n-heads 1
    # --dim-feedforward 512
    # --loss-type "bce"
    --optim-name "adamw"
    --lr 2.5e-5
    # --scheduler-type "linear"
    # --scheduler-warmup 0.05
    --num-epochs 5
    --train-batch-size 16
    --test-batch-size 32
    --accumulation-step 2
    --early-criterion 'mrr'
    --seed $1
    --swa-warmup 1
    --eval-step 2000
    --early 5
    --mp-enabled
    --gradient-max-norm 5.0
    --num-workers 8
    --experiment-name "ColBERT"
    # --run-id "efe8a17d663645389ff4b92f9f82da44"
    # --reset-early
    --query-max-length 80
    --passage-max-length 512
    --valid-size 500
    --num-pos 1
    --num-neg 1
    --shard-idx 0
    --shard-idx 1
    --shard-idx 2
    --shard-idx 3
    --shard-idx 4
    --shard-idx 5
    --shard-idx 6
    --shard-idx 7
    --shard-idx 8
    --shard-idx 9
    --shard-idx 10
    --shard-idx 11
    --shard-idx 12
    --shard-idx 13
    --shard-idx 14
    --shard-idx 15
    --shard-idx 16
    --shard-idx 17
    --shard-idx 18
    --shard-idx 19
    --shard-idx 20
    --shard-idx 21
    --shard-idx 22
    --shard-idx 23
)

python main.py train-colbert "${args[@]}"
