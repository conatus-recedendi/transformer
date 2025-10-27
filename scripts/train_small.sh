#!/bin/bash

# Small configuration training script for quick testing
echo "Starting Transformer training with small configuration..."

python train_with_config.py \
    --config experiments/small_config.json \
    --use_dummy_data \
    --seed 42

echo "Training completed!"
