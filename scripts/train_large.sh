#!/bin/bash

# Large configuration training script
echo "Starting Transformer training with large configuration..."

python train_with_config.py \
    --config experiments/large_config.json \
    --use_dummy_data \
    --seed 42

echo "Training completed!"
