#!/bin/bash

# Test script for evaluating trained models
echo "Testing trained Transformer model..."

if [ -z "$1" ]; then
    echo "Usage: $0 <config_file> [checkpoint_path]"
    echo "Example: $0 experiments/small_config.json checkpoints/small/best_model.pt"
    exit 1
fi

CONFIG_FILE=$1
CHECKPOINT_PATH=${2:-""}

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Running test without loading checkpoint..."
    python train_with_config.py \
        --config $CONFIG_FILE \
        --test_only \
        --use_dummy_data \
        --seed 42
else
    echo "Running test with checkpoint: $CHECKPOINT_PATH"
    python train_with_config.py \
        --config $CONFIG_FILE \
        --resume $CHECKPOINT_PATH \
        --test_only \
        --use_dummy_data \
        --seed 42
fi

echo "Testing completed!"
