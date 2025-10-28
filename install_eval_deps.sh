#!/bin/bash

# Install required packages for evaluation
echo "Installing evaluation dependencies..."

# Install SacreBLEU for BLEU score calculation
pip install sacrebleu

# Install SentencePiece for BPE tokenization (if not already installed)
pip install sentencepiece

echo "Dependencies installed successfully!"
echo ""
echo "Usage examples:"
echo ""
echo "1. Evaluate with base model (average last 5 checkpoints):"
echo "   python eval.py --config experiments/base_config.json --checkpoint-dir checkpoints/base/ --n-checkpoints 5"
echo ""
echo "2. Evaluate with big model (average last 20 checkpoints):"
echo "   python eval.py --config experiments/large_config.json --checkpoint-dir checkpoints/large/ --n-checkpoints 20"
echo ""
echo "3. Evaluate with custom beam search parameters:"
echo "   python eval.py --config experiments/base_config.json --checkpoint-dir checkpoints/base/ --beam-size 8 --length-penalty 0.8"
echo ""
echo "4. Evaluate with dummy data for testing:"
echo "   python eval.py --config experiments/base_config.json --checkpoint-dir checkpoints/base/ --use-dummy-data"
