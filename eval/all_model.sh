#!/bin/bash

# Define the models and datasets
models=(
    "/data/DFO/saves/test_qwen_model_tang"
)
datasets=( "nqswap" )
# "nqopen" "nqswap" "xsum"

# Loop through each model and dataset combination
for model in "${models[@]}"; do
    if [[ "$model" == *"qwen"* || "$model" == *"Qwen"* ]]; then
        export CUDA_VISIBLE_DEVICES=4,5,6,7
        gpu_num=4
    else
        export CUDA_VISIBLE_DEVICES=4,5,6,7
        gpu_num=4
    fi
    for dataset in "${datasets[@]}"; do
        echo "Running model $model on dataset $dataset"
        python main.py --model-id "$model" --eval-dataset "$dataset" --gpu_num $gpu_num
    done
done