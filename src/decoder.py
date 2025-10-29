import torch
import torch.nn as nn
import math
from .multi_head_attention import MultiheadAttention
from .encoder import PositionalEncoding


class DecoderLayer(nn.Module):
    """Single Transformer Decoder Layer"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, kdim=64, vdim=64):
        super(DecoderLayer, self).__init__()

        # Masked multi-head self-attention
        self.masked_self_attn = MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=kdim,
            vdim=vdim,
        )

        # Multi-head cross-attention (decoder-encoder attention)
        self.cross_attn = MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=kdim,
            vdim=vdim,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize all parameters with uniform distribution U[-0.1, 0.1]"""
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, -0.1, 0.1)
                if module.bias is not None:
                    nn.init.uniform_(module.bias, -0.1, 0.1)

        # LayerNorm parameters
        for module in [self.norm1, self.norm2, self.norm3]:
            nn.init.uniform_(module.weight, -0.1, 0.1)
            nn.init.uniform_(module.bias, -0.1, 0.1)

    def forward(
        self,
        x,
        encoder_output,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        Args:
            x: [batch_size, tgt_len, d_model] - target sequence
            encoder_output: [batch_size, src_len, d_model] - encoder output
            tgt_mask: [tgt_len, tgt_len] - causal mask for target
            memory_mask: [tgt_len, src_len] - mask for cross attention
            tgt_key_padding_mask: [batch_size, tgt_len] - padding mask for target
            memory_key_padding_mask: [batch_size, src_len] - padding mask for encoder output
        Returns:
            output: [batch_size, tgt_len, d_model]
        """
        # 1. Masked multi-head self-attention with residual connection and layer norm
        self_attn_output, _ = self.masked_self_attn(
            x, x, x, key_padding_mask=tgt_key_padding_mask, attn_mask=tgt_mask
        )
        x = self.norm1(x + self.dropout(self_attn_output))

        # 2. Multi-head cross-attention with residual connection and layer norm
        cross_attn_output, _ = self.cross_attn(
            x,
            encoder_output,
            encoder_output,
            key_padding_mask=memory_key_padding_mask,
            attn_mask=memory_mask,
        )
        x = self.norm2(x + self.dropout(cross_attn_output))

        # 3. Feed-forward with residual connection and layer norm
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))

        return x


class Decoder(nn.Module):
    """Transformer Decoder with Input Embedding and Positional Encoding"""

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length=5000,
        dropout=0.1,
        padding_idx=0,
        kdim=64,
        vdim=64,
    ):
        """
        Args:
            vocab_size: Size of the target vocabulary
            d_model: Model dimension (embedding dimension)
            num_heads: Number of attention heads
            num_layers: Number of decoder layers
            d_ff: Feed-forward dimension
            max_seq_length: Maximum sequence length for positional encoding
            dropout: Dropout probability
            padding_idx: Index of padding token
        """
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Target embedding
        # self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

        # Positional encoding
        # self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

        # Decoder layers
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, num_heads, d_ff, dropout, kdim, vdim)
                for _ in range(num_layers)
            ]
        )

        # Final layer normalization
        # self.norm = nn.LayerNorm(d_model)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize all parameters with uniform distribution U[-0.1, 0.1]"""
        # Initialize embedding
        # nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        # Initialize final layer norm
        # nn.init.uniform_(self.norm.weight, -0.1, 0.1)
        # nn.init.uniform_(self.norm.bias, -0.1, 0.1)

        print(f"Decoder: Initialized all parameters with U[-0.1, 0.1]")

    def forward(
        self,
        x,
        encoder_output,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        Args:
            tgt: [batch_size, tgt_len, d_model] target token indices
            encoder_output: [batch_size, src_len, d_model] encoder output
            tgt_mask: [tgt_len, tgt_len] causal mask for target
            memory_mask: [tgt_len, src_len] mask for cross attention
            tgt_key_padding_mask: [batch_size, tgt_len] padding mask for target
            memory_key_padding_mask: [batch_size, src_len] padding mask for encoder output
        Returns:
            output: [batch_size, tgt_len, d_model] decoded representations
        """
        # Target embedding with scaling
        # x = self.embedding(tgt) * math.sqrt(self.d_model)

        # Add positional encoding
        # x = self.pos_encoding(x)

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(
                x,
                encoder_output,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
            )

        # Final layer normalization
        # x = self.norm(x)

        return x

    def get_attention_weights(
        self,
        x,
        encoder_output,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        Get attention weights from all layers (for visualization)

        Returns:
            self_attention_weights: List of self-attention weights from each layer
            cross_attention_weights: List of cross-attention weights from each layer
        """
        # x = self.embedding(tgt) * math.sqrt(self.d_model)
        # x = self.pos_encoding(x)

        self_attention_weights = []
        cross_attention_weights = []

        for layer in self.layers:
            # Get self-attention weights
            self_attn_output, self_attn_weights = layer.masked_self_attn(
                x, x, x, key_padding_mask=tgt_key_padding_mask, attn_mask=tgt_mask
            )
            self_attention_weights.append(self_attn_weights)

            # Continue forward pass
            x = layer.norm1(x + layer.dropout(self_attn_output))

            # Get cross-attention weights
            cross_attn_output, cross_attn_weights = layer.cross_attn(
                x,
                encoder_output,
                encoder_output,
                key_padding_mask=memory_key_padding_mask,
                attn_mask=memory_mask,
            )
            cross_attention_weights.append(cross_attn_weights)

            # Continue forward pass
            x = layer.norm2(x + layer.dropout(cross_attn_output))
            ffn_output = layer.ffn(x)
            x = layer.norm3(x + layer.dropout(ffn_output))

        return self_attention_weights, cross_attention_weights


def create_causal_mask(seq_len, device=None):
    """
    Create causal (look-ahead) mask for decoder self-attention

    Args:
        seq_len: Sequence length
        device: Device to create tensor on

    Returns:
        mask: [seq_len, seq_len] - True for allowed positions, False for masked
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.bool()


def create_padding_mask(seq, pad_token_id=0):
    """
    Create padding mask

    Args:
        seq: [batch_size, seq_len] token indices
        pad_token_id: Padding token ID

    Returns:
        mask: [batch_size, seq_len] - True for padding positions
    """
    return seq == pad_token_id
