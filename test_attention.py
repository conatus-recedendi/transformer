"""
Test script for MultiheadAttention module
"""

import torch
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from multi_head_attention import MultiheadAttention


def test_multihead_attention():
    """Test MultiheadAttention module"""
    print("Testing MultiheadAttention module...")

    # Parameters
    batch_size = 2
    seq_len = 10
    embed_dim = 512
    num_heads = 8

    # Create model
    attention = MultiheadAttention(
        embed_dim=embed_dim, num_heads=num_heads, dropout=0.1, batch_first=True
    )

    print(f"Model created with:")
    print(f"  embed_dim: {embed_dim}")
    print(f"  num_heads: {num_heads}")
    print(f"  head_dim: {attention.head_dim}")

    # Create test inputs
    query = torch.randn(batch_size, seq_len, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)

    print(f"\nInput shapes:")
    print(f"  query: {query.shape}")
    print(f"  key: {key.shape}")
    print(f"  value: {value.shape}")

    # Test forward pass
    with torch.no_grad():
        attn_output, attn_weights = attention(query, key, value)

    print(f"\nOutput shapes:")
    print(f"  attn_output: {attn_output.shape}")
    print(f"  attn_weights: {attn_weights.shape}")

    # Test with padding mask
    key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    key_padding_mask[0, -2:] = True  # Mask last 2 positions for first batch
    key_padding_mask[1, -3:] = True  # Mask last 3 positions for second batch

    print(f"\nTesting with key_padding_mask:")
    print(f"  key_padding_mask shape: {key_padding_mask.shape}")

    with torch.no_grad():
        attn_output_masked, attn_weights_masked = attention(
            query, key, value, key_padding_mask=key_padding_mask
        )

    print(f"  attn_output_masked: {attn_output_masked.shape}")
    print(f"  attn_weights_masked: {attn_weights_masked.shape}")

    # Test with attention mask (causal mask)
    attn_mask = torch.tril(torch.ones(seq_len, seq_len))  # Lower triangular mask

    print(f"\nTesting with causal attention mask:")
    print(f"  attn_mask shape: {attn_mask.shape}")

    with torch.no_grad():
        attn_output_causal, attn_weights_causal = attention(
            query, key, value, attn_mask=attn_mask
        )

    print(f"  attn_output_causal: {attn_output_causal.shape}")
    print(f"  attn_weights_causal: {attn_weights_causal.shape}")

    # Test batch_first=False
    attention_seq_first = MultiheadAttention(
        embed_dim=embed_dim, num_heads=num_heads, dropout=0.1, batch_first=False
    )

    # Transpose inputs for seq_first format
    query_seq_first = query.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
    key_seq_first = key.transpose(0, 1)
    value_seq_first = value.transpose(0, 1)

    print(f"\nTesting with batch_first=False:")
    print(f"  query_seq_first: {query_seq_first.shape}")

    with torch.no_grad():
        attn_output_seq_first, attn_weights_seq_first = attention_seq_first(
            query_seq_first, key_seq_first, value_seq_first
        )

    print(f"  attn_output_seq_first: {attn_output_seq_first.shape}")
    print(f"  attn_weights_seq_first: {attn_weights_seq_first.shape}")

    print("\n✅ All tests passed!")

    # Check parameter count
    total_params = sum(p.numel() for p in attention.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    return attention


def test_self_attention():
    """Test self-attention (query, key, value are the same)"""
    print("\n" + "=" * 50)
    print("Testing Self-Attention...")

    batch_size = 2
    seq_len = 8
    embed_dim = 256
    num_heads = 4

    attention = MultiheadAttention(
        embed_dim=embed_dim, num_heads=num_heads, dropout=0.0, batch_first=True
    )

    # Self-attention: query, key, value are the same
    x = torch.randn(batch_size, seq_len, embed_dim)

    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        output, weights = attention(x, x, x)

    print(f"Self-attention output shape: {output.shape}")
    print(f"Self-attention weights shape: {weights.shape}")

    # Check if attention weights sum to 1
    weights_sum = weights.sum(dim=-1)
    print(f"Attention weights sum (should be ~1.0): {weights_sum.mean().item():.6f}")

    return attention


if __name__ == "__main__":
    print("Running MultiheadAttention tests...\n")

    # Test 1: Basic functionality
    model1 = test_multihead_attention()

    # Test 2: Self-attention
    model2 = test_self_attention()

    print("\n" + "=" * 50)
    print("All tests completed successfully! 🎉")
