import torch
import torch.nn as nn
import math
from .multi_head_attention import MultiheadAttention
from .positional_encoding import PositionalEncoding


class EncoderLayer(nn.Module):
    """Single Transformer Encoder Layer"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, kdim=64, vdim=64):
        super(EncoderLayer, self).__init__()

        # Multi-head self-attention
        self.self_attn = MultiheadAttention(
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
        for module in [self.norm1, self.norm2]:
            nn.init.uniform_(module.weight, -0.1, 0.1)
            nn.init.uniform_(module.bias, -0.1, 0.1)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            src_mask: [seq_len, seq_len] attention mask
            src_key_padding_mask: [batch_size, seq_len] padding mask
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Multi-head self-attention with residual connection and layer norm
        attn_output, _ = self.self_attn(
            x, x, x, key_padding_mask=src_key_padding_mask, attn_mask=src_mask
        )
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection and layer norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x


class Encoder(nn.Module):
    """Transformer Encoder with Input Embedding and Positional Encoding"""

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
            vocab_size: Size of the vocabulary
            d_model: Model dimension (embedding dimension)
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
            d_ff: Feed-forward dimension
            max_seq_length: Maximum sequence length for positional encoding
            dropout: Dropout probability
            padding_idx: Index of padding token
            kdim: Dimension of key vectors
            vdim: Dimension of value vectors
        """
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Input embedding
        # self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

        # Positional encoding
        # self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

        # Encoder layers
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, d_ff, dropout, kdim=kdim, vdim=vdim)
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

        print(f"Encoder: Initialized all parameters with U[-0.1, 0.1]")

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: [batch_size, seq_len, d_model] input token indices
            src_mask: [seq_len, seq_len] attention mask (optional)
            src_key_padding_mask: [batch_size, seq_len] padding mask (optional)
        Returns:
            output: [batch_size, seq_len, d_model] encoded representations
        """
        # Input embedding with scaling
        # x = self.embedding(src) * math.sqrt(self.d_model)

        # Add positional encoding
        # x = self.pos_encoding(x)

        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, src_mask, src_key_padding_mask)

        # Final layer normalization
        # x = self.norm(x)  # TODO: is it necessary?

        return x

    def get_attention_weights(self, x, src_mask=None, src_key_padding_mask=None):
        """
        Get attention weights from all layers (for visualization)

        Args:
            src: [batch_size, seq_len, embed_dim] input token indices
            src_mask: [seq_len, seq_len] attention mask (optional)
            src_key_padding_mask: [batch_size, seq_len] padding mask (optional)
        Returns:
            attention_weights: List of attention weights from each layer
        """
        # x = self.embedding(src) * math.sqrt(self.d_model)
        # x = self.pos_encoding(x)

        attention_weights = []

        for layer in self.layers:
            # Get attention weights
            attn_output, attn_weights = layer.self_attn(
                x, x, x, key_padding_mask=src_key_padding_mask, attn_mask=src_mask
            )
            attention_weights.append(attn_weights)

            # Continue forward pass
            x = layer.norm1(x + layer.dropout(attn_output))
            ffn_output = layer.ffn(x)
            x = layer.norm2(x + layer.dropout(ffn_output))

        return attention_weights
