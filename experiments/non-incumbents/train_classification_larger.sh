#!/usr/bin/env bash

python run.py \
    --gpus=1 \
    --learning_rate=0.001 \
    --epochs=200 \
    --batch_size=32 \
    --dataset_path=./datasets/worksites.csv \
    --classification_task=true \
    --input_size=24 \
    --hidden_size=100 \
    --deterministic=true \
    --wandb_logging=true \
    --wandb_name=classification-task-larger
