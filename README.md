# Transformer Implementation

This is a complete implementation of the Transformer model as described in the paper "Attention is All You Need" by Vaswani et al.

## 🏗️ Architecture

### Core Components

- **MultiheadAttention**: Scaled dot-product attention with multiple heads
- **Encoder**: Stack of encoder layers with self-attention and feed-forward networks
- **Decoder**: Stack of decoder layers with masked self-attention and cross-attention
- **Transformer**: Complete encoder-decoder architecture

### Key Features

- ✅ Positional encoding
- ✅ Masked self-attention (causal masking)
- ✅ Cross-attention between encoder and decoder
- ✅ Layer normalization and residual connections
- ✅ Configurable model sizes
- ✅ Attention weight visualization
- ✅ Text generation capabilities

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd Transformer

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Training with Config Files

The easiest way to train a model is using configuration files:

```bash
# Train small model (fast, for testing)
python train_with_config.py --config experiments/small_config.json --use_dummy_data

# Train base model (original paper size)
python train_with_config.py --config experiments/base_config.json --use_dummy_data

# Train large model (high performance)
python train_with_config.py --config experiments/large_config.json --use_dummy_data
```

### Using Shell Scripts

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Train different model sizes
./scripts/train_small.sh
./scripts/train_base.sh
./scripts/train_large.sh

# Test a trained model
./scripts/test_model.sh experiments/small_config.json checkpoints/small/best_model.pt
```

## ⚙️ Configuration

### Model Configurations

#### Small Model (for testing)

```json
{
  "model": {
    "model_dim": 256,
    "num_heads": 4,
    "num_encoder_layers": 3,
    "num_decoder_layers": 3,
    "ffn_dim": 1024
  }
}
```

#### Base Model (original paper)

```json
{
  "model": {
    "model_dim": 512,
    "num_heads": 8,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "ffn_dim": 2048
  }
}
```

#### Large Model (high performance)

```json
{
  "model": {
    "model_dim": 1024,
    "num_heads": 16,
    "num_encoder_layers": 12,
    "num_decoder_layers": 12,
    "ffn_dim": 4096
  }
}
```

### Training Parameters

```json
{
  "training": {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "warmup_steps": 4000,
    "max_epochs": 100,
    "gradient_clip": 1.0
  }
}
```

## 🔧 Advanced Usage

### Resume Training

```bash
python train_with_config.py \
    --config experiments/base_config.json \
    --resume checkpoints/base/checkpoint_step_5000.pt \
    --use_dummy_data
```

### Test Only Mode

```bash
python train_with_config.py \
    --config experiments/base_config.json \
    --test_only \
    --use_dummy_data
```

### Custom Device

```bash
python train_with_config.py \
    --config experiments/base_config.json \
    --device cuda \
    --use_dummy_data
```

## 🧪 Testing

### Run All Tests

```bash
# Test individual components
python test_attention.py      # Test multi-head attention
python test_encoder.py        # Test encoder
python test_decoder.py        # Test decoder
python test_transformer.py    # Test complete model

# Test config-based training
python test_config_training.py
```

## 📊 Model Sizes

| Configuration | Parameters | Memory | Training Time |
| ------------- | ---------- | ------ | ------------- |
| Small         | ~1.8M      | ~2GB   | Fast          |
| Base          | ~65M       | ~8GB   | Medium        |
| Large         | ~213M      | ~16GB  | Slow          |

## 🎯 Features

### Training Features

- ✅ Warmup learning rate scheduling
- ✅ Gradient clipping
- ✅ Automatic checkpointing
- ✅ Validation monitoring
- ✅ Progress tracking with tqdm
- ✅ Comprehensive logging

### Model Features

- ✅ Attention weight extraction
- ✅ Multiple generation strategies (greedy, sampling, top-k, top-p)
- ✅ Configurable model architecture
- ✅ Weight tying option
- ✅ Proper masking (padding + causal)

## 📁 Project Structure

```
Transformer/
├── src/
│   ├── multi_head_attention.py    # Multi-head attention implementation
│   ├── encoder.py                 # Encoder with positional encoding
│   ├── decoder.py                 # Decoder with masking
│   ├── transformer.py             # Complete model
│   ├── trainer.py                 # Training utilities
│   ├── data_loader.py             # Data loading utilities
│   ├── config.py                  # Configuration class
│   └── utils.py                   # Utility functions
├── experiments/
│   ├── small_config.json          # Small model config
│   ├── base_config.json           # Base model config
│   └── large_config.json          # Large model config
├── scripts/
│   ├── train_small.sh             # Train small model
│   ├── train_base.sh              # Train base model
│   ├── train_large.sh             # Train large model
│   └── test_model.sh              # Test trained model
├── train_with_config.py           # Config-based training script
└── test_*.py                      # Test scripts
```

## 🎓 Paper Reference

This implementation follows the original Transformer paper:

**"Attention is All You Need"**  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin  
_Neural Information Processing Systems (NIPS) 2017_
