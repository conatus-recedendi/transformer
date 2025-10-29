"""
Utility functions for Transformer model
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
from typing import Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def create_padding_mask(seq: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    Create padding mask for attention mechanism

    Args:
        seq: Input sequence [batch_size, seq_len]
        pad_token_id: Padding token ID

    Returns:
        Padding mask [batch_size, 1, 1, seq_len]
    """
    mask = (
        (seq != pad_token_id).unsqueeze(1).unsqueeze(1)
    )  # [batch_size, 1, 1, seq_len]
    return mask


def create_look_ahead_mask(size: int) -> torch.Tensor:
    """
    Create look-ahead mask for decoder self-attention

    Args:
        size: Sequence length

    Returns:
        Look-ahead mask [size, size]
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return ~mask  # Invert mask (True for allowed positions)


def create_masks(
    src: torch.Tensor, tgt: torch.Tensor, pad_token_id: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create all masks for Transformer

    Args:
        src: Source sequence [batch_size, src_len]
        tgt: Target sequence [batch_size, tgt_len]
        pad_token_id: Padding token ID

    Returns:
        Tuple of (src_mask, tgt_mask, src_tgt_mask)
    """
    # Source padding mask
    src_mask = create_padding_mask(src, pad_token_id)

    # Target padding mask
    tgt_padding_mask = create_padding_mask(tgt, pad_token_id)

    # Target look-ahead mask
    tgt_len = tgt.size(1)
    tgt_look_ahead_mask = create_look_ahead_mask(tgt_len).to(tgt.device)

    # Combine target masks
    tgt_mask = tgt_padding_mask & tgt_look_ahead_mask.unsqueeze(0).unsqueeze(0)

    # Source-target mask (only padding mask needed)
    src_tgt_mask = src_mask

    return src_mask, tgt_mask, src_tgt_mask


def save_checkpoint(
    model: nn.Module, optimizer, epoch: int, loss: float, filepath: str, **kwargs
):
    """
    Save model checkpoint

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
        **kwargs: Additional items to save
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        **kwargs,
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(
    filepath: str, model: nn.Module, optimizer=None, device: str = "cpu"
) -> Dict[str, Any]:
    """
    Load model checkpoint

    Args:
        filepath: Path to checkpoint file
        model: PyTorch model
        optimizer: Optimizer (optional)
        device: Device to load model on

    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Checkpoint loaded: {filepath}")
    return checkpoint


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module):
    """
    Print detailed model summary with parameter breakdown

    Args:
        model: PyTorch model
    """
    print("\n" + "=" * 80)
    print("📊 DETAILED MODEL SUMMARY")
    print("=" * 80)

    # Count total parameters
    total_params = count_parameters(model)
    total_params_all = sum(p.numel() for p in model.parameters())

    print(f"🔢 Total trainable parameters: {total_params:,}")
    print(f"🔢 Total parameters (including non-trainable): {total_params_all:,}")

    # Calculate model size in MB
    param_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    print(f"💾 Model size (float32): {param_size_mb:.2f} MB")

    # Parameter breakdown by component
    print(f"\n📋 Parameter Breakdown by Component:")
    print("-" * 50)

    component_params = {}

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                # Categorize parameters
                if "encoder" in name:
                    category = "Encoder"
                elif "decoder" in name:
                    category = "Decoder"
                elif "embedding" in name or "embed" in name:
                    category = "Embeddings"
                elif "output_projection" in name or "projection" in name:
                    category = "Output Projection"
                elif "attention" in name or "attn" in name:
                    category = "Attention"
                elif "feed_forward" in name or "ffn" in name or "mlp" in name:
                    category = "Feed Forward"
                elif "norm" in name or "layer_norm" in name:
                    category = "Layer Norm"
                else:
                    category = "Other"

                if category not in component_params:
                    component_params[category] = 0
                component_params[category] += params

    # Sort by parameter count
    sorted_components = sorted(
        component_params.items(), key=lambda x: x[1], reverse=True
    )

    for component, params in sorted_components:
        percentage = (params / total_params) * 100
        print(f"  {component:.<25} {params:>12,} ({percentage:5.1f}%)")

    # Memory usage estimation
    print(f"\n💾 Memory Usage Estimation (Training):")
    print("-" * 50)

    # Forward pass memory (activations)
    # Rough estimation based on model size and sequence length
    seq_len = getattr(model, "d_model", 512)  # Use d_model as rough seq_len estimate
    d_model = getattr(model, "d_model", 512)
    batch_size = 32  # Assumption

    activation_memory_mb = (batch_size * seq_len * d_model * 4) / (1024 * 1024)
    gradient_memory_mb = param_size_mb  # Gradients same size as parameters
    optimizer_memory_mb = param_size_mb * 2  # Adam uses 2x parameter memory

    total_memory_mb = (
        param_size_mb + activation_memory_mb + gradient_memory_mb + optimizer_memory_mb
    )

    print(f"  Parameters:............ {param_size_mb:8.2f} MB")
    print(f"  Activations (est):...... {activation_memory_mb:8.2f} MB")
    print(f"  Gradients:.............. {gradient_memory_mb:8.2f} MB")
    print(f"  Optimizer state:........ {optimizer_memory_mb:8.2f} MB")
    print(f"  {'Total (estimated):':.<24} {total_memory_mb:8.2f} MB")

    # Model architecture comparison
    print(f"\n🏗️  Architecture Comparison:")
    print("-" * 50)

    # Try to extract key architectural info
    if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        num_encoder_layers = (
            len(model.encoder.layers)
            if hasattr(model.encoder.layers, "__len__")
            else "Unknown"
        )
        print(f"  Encoder layers:......... {num_encoder_layers}")

    if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
        num_decoder_layers = (
            len(model.decoder.layers)
            if hasattr(model.decoder.layers, "__len__")
            else "Unknown"
        )
        print(f"  Decoder layers:......... {num_decoder_layers}")

    if hasattr(model, "d_model"):
        print(f"  Model dimension:........ {model.d_model}")

    # Compare with known models
    print(f"\n📈 Model Scale Comparison:")
    print("-" * 50)

    if total_params < 10_000_000:
        scale = "Small (< 10M)"
    elif total_params < 100_000_000:
        scale = "Medium (10M - 100M)"
    elif total_params < 1_000_000_000:
        scale = "Large (100M - 1B)"
    else:
        scale = "Very Large (> 1B)"

    print(f"  Model scale:............ {scale}")

    # Reference models for comparison
    reference_models = {
        "Transformer Base": "65M parameters",
        "Transformer Big": "213M parameters",
        "GPT-2 Small": "117M parameters",
        "BERT Base": "110M parameters",
        "T5 Small": "60M parameters",
    }

    print(f"  Reference models:")
    for model_name, params in reference_models.items():
        print(f"    {model_name}:...... {params}")

    print("=" * 80)


def save_config(config: Dict[str, Any], filepath: str):
    """
    Save configuration to JSON file

    Args:
        config: Configuration dictionary
        filepath: Path to save configuration
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved: {filepath}")


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file

    Args:
        filepath: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(filepath, "r") as f:
        config = json.load(f)

    print(f"Configuration loaded: {filepath}")
    return config


def plot_training_curves(
    train_losses: list, val_losses: list = None, save_path: str = None
):
    """
    Plot training curves

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses (optional)
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, "b-", label="Training Loss")

    if val_losses:
        plt.plot(epochs, val_losses, "r-", label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curves")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved: {save_path}")

    plt.show()


def plot_attention_weights(
    attention_weights: torch.Tensor,
    src_tokens: list,
    tgt_tokens: list,
    save_path: str = None,
):
    """
    Plot attention weights heatmap

    Args:
        attention_weights: Attention weights [tgt_len, src_len]
        src_tokens: Source tokens
        tgt_tokens: Target tokens
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(12, 8))

    sns.heatmap(
        attention_weights.cpu().numpy(),
        xticklabels=src_tokens,
        yticklabels=tgt_tokens,
        cmap="Blues",
        annot=True,
        fmt=".2f",
    )

    plt.xlabel("Source Tokens")
    plt.ylabel("Target Tokens")
    plt.title("Attention Weights")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Attention plot saved: {save_path}")

    plt.show()


def compute_bleu_score(predictions: list, references: list) -> float:
    """
    Compute BLEU score (placeholder implementation)

    Args:
        predictions: List of predicted sequences
        references: List of reference sequences

    Returns:
        BLEU score
    """
    # TODO: Implement actual BLEU score calculation
    # This is a placeholder - you should use a proper BLEU implementation
    # such as from nltk or sacrebleu

    print("Warning: Using placeholder BLEU score implementation")
    return 0.0


def set_seed(seed: int):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed}")


def get_device() -> torch.device:
    """
    Get available device (CUDA/MPS/CPU)

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    return device


class EarlyStopping:
    """Early stopping utility"""

    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        """
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def warmup_lr_schedule(step: int, warmup_steps: int, d_model: int) -> float:
    """
    Warmup learning rate schedule as described in "Attention is All You Need"

    Args:
        step: Current training step
        warmup_steps: Number of warmup steps
        d_model: Model dimension

    Returns:
        Learning rate multiplier
    """
    step = max(step, 1)  # Avoid division by zero
    return (d_model**-0.5) * min(step**-0.5, step * warmup_steps**-1.5)
