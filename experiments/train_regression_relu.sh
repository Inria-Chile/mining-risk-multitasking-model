#!/usr/bin/env bash

python run.py \
    --gpus=1 \
    --learning_rate=0.001 \
    --epochs=200 \
    --batch_size=32 \
    --dataset_path=./datasets/worksites.csv \
    --regression_task=true \
    --fill_missing_regression=1460 \
    --input_size=24 \
    --hidden_size=50 \
    --deterministic=true \
    --wandb_logging=true \
    --wandb_name=regression-relu \
    --regressor_activation=relu
