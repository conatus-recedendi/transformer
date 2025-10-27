#!/bin/bash

# Base configuration training script
echo "Starting Transformer training with base configuration..."

python train_with_config.py \
    --config experiments/base_config.json \
    --use_dummy_data \
    --seed 42

echo "Training completed!"
