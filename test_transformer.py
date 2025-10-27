"""
Test script for complete Transformer model
"""

import torch
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from transformer import Transformer
from decoder import create_causal_mask, create_padding_mask


def test_transformer_basic():
    """Test basic Transformer functionality"""
    print("Testing Basic Transformer Functionality...")

    # Model parameters
    src_vocab_size = 5000
    tgt_vocab_size = 6000
    d_model = 256
    num_heads = 4
    num_encoder_layers = 3
    num_decoder_layers = 3
    d_ff = 1024
    dropout = 0.1

    # Create model
    transformer = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        dropout=dropout,
    )

    print(f"Transformer model created:")
    print(f"  src_vocab_size: {src_vocab_size}")
    print(f"  tgt_vocab_size: {tgt_vocab_size}")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  encoder_layers: {num_encoder_layers}")
    print(f"  decoder_layers: {num_decoder_layers}")

    # Count parameters
    total_params = transformer.count_parameters()
    print(f"  total_parameters: {total_params:,}")

    # Create sample data
    batch_size = 2
    src_len = 20
    tgt_len = 15

    src = torch.randint(1, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))

    print(f"\nInput shapes:")
    print(f"  src: {src.shape}")
    print(f"  tgt: {tgt.shape}")

    # Test forward pass
    with torch.no_grad():
        output = transformer(src, tgt)

    print(f"Output shape: {output.shape}")
    print(f"Expected shape: [batch_size, tgt_len, tgt_vocab_size]")
    print(f"Output mean: {output.mean().item():.6f}")
    print(f"Output std: {output.std().item():.6f}")

    # Test with padding
    src_with_padding = src.clone()
    src_with_padding[0, -3:] = 0  # Add padding
    src_with_padding[1, -5:] = 0

    tgt_with_padding = tgt.clone()
    tgt_with_padding[0, -2:] = 0
    tgt_with_padding[1, -4:] = 0

    print(f"\nTesting with padding:")
    print(f"  src padding positions: {(src_with_padding == 0).sum().item()}")
    print(f"  tgt padding positions: {(tgt_with_padding == 0).sum().item()}")

    with torch.no_grad():
        output_padded = transformer(src_with_padding, tgt_with_padding)

    print(f"  padded output shape: {output_padded.shape}")

    return transformer


def test_encoder_decoder_separate():
    """Test encode and decode methods separately"""
    print("\n" + "=" * 50)
    print("Testing Encode/Decode Methods Separately...")

    transformer = Transformer(
        src_vocab_size=3000,
        tgt_vocab_size=4000,
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=512,
    )

    batch_size = 3
    src_len = 15
    tgt_len = 12

    src = torch.randint(1, 3000, (batch_size, src_len))
    tgt = torch.randint(1, 4000, (batch_size, tgt_len))

    print(f"Input shapes: src={src.shape}, tgt={tgt.shape}")

    # Test encoding
    with torch.no_grad():
        encoder_output = transformer.encode(src)

    print(f"Encoder output shape: {encoder_output.shape}")

    # Test decoding
    with torch.no_grad():
        decoder_output = transformer.decode(tgt, encoder_output)

    print(f"Decoder output shape: {decoder_output.shape}")

    # Compare with full forward pass
    with torch.no_grad():
        full_output = transformer(src, tgt)

    print(f"Full forward output shape: {full_output.shape}")
    print(f"Outputs match: {torch.allclose(decoder_output, full_output, atol=1e-6)}")

    return transformer


def test_generation():
    """Test generation functionality"""
    print("\n" + "=" * 50)
    print("Testing Generation...")

    transformer = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=256,
        dropout=0.0,  # No dropout for deterministic testing
    )

    transformer.eval()

    batch_size = 2
    src_len = 10
    max_gen_length = 15

    src = torch.randint(1, 1000, (batch_size, src_len))

    print(f"Source shape: {src.shape}")
    print(f"Max generation length: {max_gen_length}")

    # Test greedy generation
    with torch.no_grad():
        generated_greedy = transformer.generate(
            src,
            max_length=max_gen_length,
            start_token=1,
            end_token=2,
            do_sample=False,  # Greedy
        )

    print(f"Greedy generated shape: {generated_greedy.shape}")
    print(f"Generated sequences (greedy):")
    for i, seq in enumerate(generated_greedy):
        print(f"  Batch {i}: {seq.tolist()}")

    # Test sampling generation
    with torch.no_grad():
        generated_sample = transformer.generate(
            src,
            max_length=max_gen_length,
            start_token=1,
            end_token=2,
            temperature=0.8,
            do_sample=True,
        )

    print(f"Sampling generated shape: {generated_sample.shape}")
    print(f"Generated sequences (sampling):")
    for i, seq in enumerate(generated_sample):
        print(f"  Batch {i}: {seq.tolist()}")

    # Test top-k sampling
    with torch.no_grad():
        generated_topk = transformer.generate(
            src,
            max_length=max_gen_length,
            start_token=1,
            end_token=2,
            temperature=1.0,
            top_k=50,
            do_sample=True,
        )

    print(f"Top-k generated shape: {generated_topk.shape}")

    return transformer


def test_attention_weights():
    """Test attention weights extraction"""
    print("\n" + "=" * 50)
    print("Testing Attention Weights Extraction...")

    transformer = Transformer(
        src_vocab_size=500,
        tgt_vocab_size=600,
        d_model=64,
        num_heads=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
    )

    batch_size = 1
    src_len = 8
    tgt_len = 6

    src = torch.randint(1, 500, (batch_size, src_len))
    tgt = torch.randint(1, 600, (batch_size, tgt_len))

    print(f"Input shapes: src={src.shape}, tgt={tgt.shape}")

    with torch.no_grad():
        encoder_attn, decoder_self_attn, decoder_cross_attn = (
            transformer.get_attention_weights(src, tgt)
        )

    print(f"Attention weights shapes:")
    print(f"  Encoder layers: {len(encoder_attn)}")
    print(f"  Encoder attention shape per layer: {encoder_attn[0].shape}")
    print(f"  Decoder self-attention layers: {len(decoder_self_attn)}")
    print(f"  Decoder self-attention shape per layer: {decoder_self_attn[0].shape}")
    print(f"  Decoder cross-attention layers: {len(decoder_cross_attn)}")
    print(f"  Decoder cross-attention shape per layer: {decoder_cross_attn[0].shape}")

    # Check attention weights sum to 1
    enc_weights_sum = encoder_attn[0].sum(dim=-1).mean()
    dec_self_weights_sum = decoder_self_attn[0].sum(dim=-1).mean()
    dec_cross_weights_sum = decoder_cross_attn[0].sum(dim=-1).mean()

    print(f"Attention weights sums (should be ~1.0):")
    print(f"  Encoder: {enc_weights_sum.item():.6f}")
    print(f"  Decoder self: {dec_self_weights_sum.item():.6f}")
    print(f"  Decoder cross: {dec_cross_weights_sum.item():.6f}")

    return transformer


def test_different_configurations():
    """Test different model configurations"""
    print("\n" + "=" * 50)
    print("Testing Different Configurations...")

    configurations = [
        # Small model
        {
            "name": "Small",
            "src_vocab_size": 1000,
            "tgt_vocab_size": 1000,
            "d_model": 128,
            "num_heads": 4,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "d_ff": 256,
        },
        # Base model (similar to original Transformer)
        {
            "name": "Base",
            "src_vocab_size": 30000,
            "tgt_vocab_size": 30000,
            "d_model": 512,
            "num_heads": 8,
            "num_encoder_layers": 6,
            "num_decoder_layers": 6,
            "d_ff": 2048,
        },
        # Large model
        {
            "name": "Large",
            "src_vocab_size": 50000,
            "tgt_vocab_size": 50000,
            "d_model": 1024,
            "num_heads": 16,
            "num_encoder_layers": 12,
            "num_decoder_layers": 12,
            "d_ff": 4096,
        },
    ]

    for config in configurations:
        print(f"\n{config['name']} Configuration:")

        transformer = Transformer(**{k: v for k, v in config.items() if k != "name"})

        total_params = transformer.count_parameters()
        print(f"  Parameters: {total_params:,}")
        print(f"  d_model: {config['d_model']}")
        print(f"  num_heads: {config['num_heads']}")
        print(
            f"  layers: {config['num_encoder_layers']}/{config['num_decoder_layers']}"
        )

        # Test with small input
        batch_size = 1
        src_len = 5
        tgt_len = 4

        src = torch.randint(
            1, min(100, config["src_vocab_size"]), (batch_size, src_len)
        )
        tgt = torch.randint(
            1, min(100, config["tgt_vocab_size"]), (batch_size, tgt_len)
        )

        with torch.no_grad():
            output = transformer(src, tgt)

        print(f"  Test output shape: {output.shape}")


def test_loss_computation():
    """Test loss computation for training"""
    print("\n" + "=" * 50)
    print("Testing Loss Computation...")

    transformer = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=256,
    )

    batch_size = 4
    src_len = 12
    tgt_len = 10

    src = torch.randint(1, 1000, (batch_size, src_len))
    tgt = torch.randint(1, 1000, (batch_size, tgt_len))

    print(f"Input shapes: src={src.shape}, tgt={tgt.shape}")

    # Forward pass
    logits = transformer(src, tgt)
    print(f"Logits shape: {logits.shape}")

    # Prepare targets for loss (shift by 1 position)
    tgt_input = tgt[:, :-1]  # Remove last token
    tgt_output = tgt[:, 1:]  # Remove first token

    # Get predictions for the input portion
    pred_logits = transformer(src, tgt_input)

    print(f"Prediction logits shape: {pred_logits.shape}")
    print(f"Target output shape: {tgt_output.shape}")

    # Compute cross-entropy loss
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding

    loss = loss_fn(
        pred_logits.reshape(-1, pred_logits.size(-1)),  # [batch*seq, vocab]
        tgt_output.reshape(-1),  # [batch*seq]
    )

    print(f"Loss: {loss.item():.6f}")

    # Test backward pass
    loss.backward()
    print(f"Gradients computed successfully!")

    # Check gradient norms
    total_norm = 0
    for p in transformer.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)

    print(f"Total gradient norm: {total_norm:.6f}")


if __name__ == "__main__":
    print("Running Complete Transformer tests...\n")

    # Test 1: Basic functionality
    transformer1 = test_transformer_basic()

    # Test 2: Separate encode/decode
    transformer2 = test_encoder_decoder_separate()

    # Test 3: Generation
    transformer3 = test_generation()

    # Test 4: Attention weights
    transformer4 = test_attention_weights()

    # Test 5: Different configurations
    test_different_configurations()

    # Test 6: Loss computation
    test_loss_computation()

    print("\n" + "=" * 50)
    print("All Transformer tests completed successfully! 🎉")
    print("The complete Transformer model is ready for training!")
