"""
Test script for Decoder module
"""

import torch
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from decoder import Decoder, DecoderLayer, create_causal_mask, create_padding_mask
from encoder import Encoder


def test_causal_mask():
    """Test causal mask creation"""
    print("Testing Causal Mask...")

    seq_len = 5
    mask = create_causal_mask(seq_len)

    print(f"Causal mask for seq_len={seq_len}:")
    print(mask.int())
    print(f"Shape: {mask.shape}")
    print(f"Type: {mask.dtype}")

    # Test that future positions are masked (False)
    assert not mask[0, 1].item(), "Future positions should be masked"
    assert mask[1, 0].item(), "Past positions should be visible"
    assert mask[2, 2].item(), "Current position should be visible"

    print("✅ Causal mask test passed!")
    return mask


def test_decoder_layer():
    """Test single DecoderLayer"""
    print("\n" + "=" * 50)
    print("Testing DecoderLayer...")

    d_model = 512
    num_heads = 8
    d_ff = 2048
    seq_len = 20
    src_len = 25
    batch_size = 2

    decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout=0.1)

    print(f"DecoderLayer parameters:")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_ff: {d_ff}")

    # Create dummy inputs
    tgt = torch.randn(batch_size, seq_len, d_model)  # Target sequence
    encoder_output = torch.randn(batch_size, src_len, d_model)  # Encoder output

    print(f"\nInput shapes:")
    print(f"  tgt: {tgt.shape}")
    print(f"  encoder_output: {encoder_output.shape}")

    # Create causal mask for target
    tgt_mask = create_causal_mask(seq_len)

    print(f"  tgt_mask: {tgt_mask.shape}")

    # Test forward pass
    with torch.no_grad():
        output = decoder_layer(tgt, encoder_output, tgt_mask=tgt_mask)

    print(f"Output shape: {output.shape}")

    # Test with padding masks
    tgt_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    tgt_padding_mask[0, -3:] = True  # Mask last 3 positions for first batch

    memory_padding_mask = torch.zeros(batch_size, src_len, dtype=torch.bool)
    memory_padding_mask[1, -5:] = True  # Mask last 5 positions for second batch

    print(f"\nTesting with padding masks:")
    print(f"  tgt_padding_mask: {tgt_padding_mask.shape}")
    print(f"  memory_padding_mask: {memory_padding_mask.shape}")

    with torch.no_grad():
        output_masked = decoder_layer(
            tgt,
            encoder_output,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )

    print(f"Masked output shape: {output_masked.shape}")

    return decoder_layer


def test_full_decoder():
    """Test complete Decoder"""
    print("\n" + "=" * 50)
    print("Testing Full Decoder...")

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

    # Create decoder
    decoder = Decoder(
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

    print(f"Decoder parameters:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  num_layers: {num_layers}")
    print(f"  d_ff: {d_ff}")

    # Count parameters
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"  total_params: {total_params:,}")

    # Create dummy inputs
    batch_size = 2
    tgt_len = 30
    src_len = 40

    tgt = torch.randint(1, vocab_size, (batch_size, tgt_len))  # Target tokens
    encoder_output = torch.randn(batch_size, src_len, d_model)  # Encoder output

    print(f"\nInput shapes:")
    print(f"  tgt: {tgt.shape}")
    print(f"  encoder_output: {encoder_output.shape}")

    # Create causal mask
    tgt_mask = create_causal_mask(tgt_len)

    # Test forward pass
    with torch.no_grad():
        output = decoder(tgt, encoder_output, tgt_mask=tgt_mask)

    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.6f}")
    print(f"Output std: {output.std().item():.6f}")

    # Test with padding
    tgt_with_padding = tgt.clone()
    tgt_with_padding[0, -5:] = 0  # Add padding to first sequence
    tgt_with_padding[1, -8:] = 0  # Add padding to second sequence

    tgt_padding_mask = create_padding_mask(tgt_with_padding, pad_token_id=0)

    print(f"\nTesting with padding:")
    print(f"  tgt_with_padding shape: {tgt_with_padding.shape}")
    print(f"  tgt_padding_mask shape: {tgt_padding_mask.shape}")
    print(f"  padding positions: {tgt_padding_mask.sum().item()}")

    with torch.no_grad():
        output_padded = decoder(
            tgt_with_padding,
            encoder_output,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )

    print(f"  padded output shape: {output_padded.shape}")

    # Test attention weights extraction
    print(f"\nTesting attention weights extraction...")
    with torch.no_grad():
        self_attn_weights, cross_attn_weights = decoder.get_attention_weights(
            tgt, encoder_output, tgt_mask=tgt_mask
        )

    print(f"  Number of layers: {len(self_attn_weights)}")
    print(f"  Self-attention weights shape per layer: {self_attn_weights[0].shape}")
    print(f"  Cross-attention weights shape per layer: {cross_attn_weights[0].shape}")

    return decoder


def test_encoder_decoder_integration():
    """Test Encoder-Decoder integration"""
    print("\n" + "=" * 50)
    print("Testing Encoder-Decoder Integration...")

    # Shared parameters
    d_model = 256
    num_heads = 4
    num_layers = 3
    d_ff = 1024
    dropout = 0.1
    kdim = 64
    vdim = 64

    src_vocab_size = 5000
    tgt_vocab_size = 6000

    # Create encoder and decoder
    encoder = Encoder(
        vocab_size=src_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        kdim=kdim,
        vdim=vdim,
    )

    decoder = Decoder(
        vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        kdim=kdim,
        vdim=vdim,
    )

    print(f"Encoder-Decoder configuration:")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  num_layers: {num_layers}")
    print(f"  src_vocab_size: {src_vocab_size}")
    print(f"  tgt_vocab_size: {tgt_vocab_size}")

    # Create sample data
    batch_size = 2
    src_len = 20
    tgt_len = 15

    src = torch.randint(1, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))

    print(f"\nInput shapes:")
    print(f"  src: {src.shape}")
    print(f"  tgt: {tgt.shape}")

    # Create masks
    tgt_mask = create_causal_mask(tgt_len)

    # Forward pass
    with torch.no_grad():
        # Encoder
        encoder_output = encoder(src)
        print(f"  encoder_output: {encoder_output.shape}")

        # Decoder
        decoder_output = decoder(tgt, encoder_output, tgt_mask=tgt_mask)
        print(f"  decoder_output: {decoder_output.shape}")

    # Test with padding
    src_with_padding = src.clone()
    src_with_padding[0, -3:] = 0
    src_with_padding[1, -5:] = 0

    tgt_with_padding = tgt.clone()
    tgt_with_padding[0, -2:] = 0
    tgt_with_padding[1, -4:] = 0

    src_padding_mask = create_padding_mask(src_with_padding)
    tgt_padding_mask = create_padding_mask(tgt_with_padding)

    print(f"\nTesting with padding masks:")
    print(f"  src_padding_mask: {src_padding_mask.sum().item()} positions")
    print(f"  tgt_padding_mask: {tgt_padding_mask.sum().item()} positions")

    with torch.no_grad():
        encoder_output_padded = encoder(
            src_with_padding, src_key_padding_mask=src_padding_mask
        )

        decoder_output_padded = decoder(
            tgt_with_padding,
            encoder_output_padded,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )

    print(f"  encoder_output_padded: {encoder_output_padded.shape}")
    print(f"  decoder_output_padded: {decoder_output_padded.shape}")

    # Calculate total parameters
    total_encoder_params = sum(p.numel() for p in encoder.parameters())
    total_decoder_params = sum(p.numel() for p in decoder.parameters())
    total_params = total_encoder_params + total_decoder_params

    print(f"\nParameter counts:")
    print(f"  Encoder: {total_encoder_params:,}")
    print(f"  Decoder: {total_decoder_params:,}")
    print(f"  Total: {total_params:,}")

    return encoder, decoder


if __name__ == "__main__":
    print("Running Decoder tests...\n")

    # Test 1: Causal Mask
    causal_mask = test_causal_mask()

    # Test 2: Single Decoder Layer
    dec_layer = test_decoder_layer()

    # Test 3: Full Decoder
    full_decoder = test_full_decoder()

    # Test 4: Encoder-Decoder Integration
    encoder, decoder = test_encoder_decoder_integration()

    print("\n" + "=" * 50)
    print("All decoder tests completed successfully! 🎉")
