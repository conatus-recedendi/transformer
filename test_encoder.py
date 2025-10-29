"""
Test script for Encoder module
"""

import torch
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from encoder import Encoder, PositionalEncoding, EncoderLayer


def test_positional_encoding():
    """Test PositionalEncoding module"""
    print("Testing PositionalEncoding...")

    d_model = 512
    seq_len = 100
    batch_size = 2

    pos_encoding = PositionalEncoding(d_model, max_seq_length=200, dropout=0.1)

    # Create dummy input
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"Input shape: {x.shape}")

    # Apply positional encoding
    output = pos_encoding(x)

    print(f"Output shape: {output.shape}")
    print(f"Positional encoding shape: {pos_encoding.pe.shape}")

    # Check that positional encoding is deterministic
    output2 = pos_encoding(x)
    print(
        f"Positional encoding is deterministic: {torch.allclose(output, output2, atol=1e-6)}"
    )

    return pos_encoding


def test_encoder_layer():
    """Test single EncoderLayer"""
    print("\n" + "=" * 50)
    print("Testing EncoderLayer...")

    d_model = 512
    num_heads = 8
    d_ff = 2048
    seq_len = 20
    batch_size = 2

    encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout=0.1)

    print(f"EncoderLayer parameters:")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_ff: {d_ff}")

    # Create dummy input
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\nInput shape: {x.shape}")

    # Test forward pass
    with torch.no_grad():
        output = encoder_layer(x)

    print(f"Output shape: {output.shape}")

    # Test with padding mask
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    padding_mask[0, -3:] = True  # Mask last 3 positions for first batch
    padding_mask[1, -5:] = True  # Mask last 5 positions for second batch

    print(f"\nTesting with padding mask: {padding_mask.shape}")

    with torch.no_grad():
        output_masked = encoder_layer(x, src_key_padding_mask=padding_mask)

    print(f"Masked output shape: {output_masked.shape}")

    return encoder_layer


def test_full_encoder():
    """Test complete Encoder with embedding and positional encoding"""
    print("\n" + "=" * 50)
    print("Testing Full Encoder...")

    # Parameters
    vocab_size = 10000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 5000
    dropout = 0.1
    kdim = 64
    vdim = 64

    # Create encoder
    encoder = Encoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
        dropout=dropout,
        kdim=kdim,
        vdim=vdim,
    )

    print(f"Encoder parameters:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  num_layers: {num_layers}")
    print(f"  d_ff: {d_ff}")

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"  total_params: {total_params:,}")

    # Create dummy input (token indices)
    batch_size = 2
    seq_len = 50
    src = torch.randint(1, vocab_size, (batch_size, seq_len))  # Avoid padding token (0)

    print(f"\nInput shape: {src.shape}")
    print(f"Input range: [{src.min().item()}, {src.max().item()}]")

    # Test forward pass
    with torch.no_grad():
        output = encoder(src)

    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.6f}")
    print(f"Output std: {output.std().item():.6f}")

    # Test with padding mask
    src_with_padding = src.clone()
    src_with_padding[0, -5:] = 0  # Add padding to first sequence
    src_with_padding[1, -8:] = 0  # Add padding to second sequence

    # Create padding mask
    padding_mask = src_with_padding == 0

    print(f"\nTesting with padding:")
    print(f"  src_with_padding shape: {src_with_padding.shape}")
    print(f"  padding_mask shape: {padding_mask.shape}")
    print(f"  padding positions: {padding_mask.sum().item()}")

    with torch.no_grad():
        output_padded = encoder(src_with_padding, src_key_padding_mask=padding_mask)

    print(f"  padded output shape: {output_padded.shape}")

    # Test attention weights extraction
    print(f"\nTesting attention weights extraction...")
    with torch.no_grad():
        attention_weights = encoder.get_attention_weights(src)

    print(f"  Number of layers: {len(attention_weights)}")
    print(f"  Attention weights shape per layer: {attention_weights[0].shape}")

    return encoder


def test_different_configurations():
    """Test encoder with different configurations"""
    print("\n" + "=" * 50)
    print("Testing Different Configurations...")

    configurations = [
        {"d_model": 256, "num_heads": 4, "num_layers": 3, "d_ff": 1024},
        {"d_model": 512, "num_heads": 8, "num_layers": 6, "d_ff": 2048},
        {"d_model": 768, "num_heads": 12, "num_layers": 12, "d_ff": 3072},
    ]

    for i, config in enumerate(configurations):
        print(f"\nConfiguration {i+1}: {config}")

        encoder = Encoder(
            vocab_size=5000, **config, max_seq_length=512, dropout=0.1, kdim=64, vdim=64
        )

        # Test with small input
        batch_size = 1
        seq_len = 10
        src = torch.randint(1, 5000, (batch_size, seq_len))

        with torch.no_grad():
            output = encoder(src)

        total_params = sum(p.numel() for p in encoder.parameters())
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {total_params:,}")


if __name__ == "__main__":
    print("Running Encoder tests...\n")

    # Test 1: Positional Encoding
    pos_enc = test_positional_encoding()

    # Test 2: Single Encoder Layer
    enc_layer = test_encoder_layer()

    # Test 3: Full Encoder
    full_encoder = test_full_encoder()

    # Test 4: Different Configurations
    test_different_configurations()

    print("\n" + "=" * 50)
    print("All encoder tests completed successfully! 🎉")
