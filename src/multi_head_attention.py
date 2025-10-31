import torch
import torch.nn as nn

import math


class SingleHeadAttention(nn.Module):
    """Single head attention module"""

    def __init__(self, embed_dim, kdim, vdim, dropout=0.0, bias=True):
        super(SingleHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim
        self.vdim = vdim
        self.dropout = dropout

        # Linear projections for single head
        self.q_linear = nn.Linear(embed_dim, kdim, bias=bias)
        self.k_linear = nn.Linear(embed_dim, kdim, bias=bias)
        self.v_linear = nn.Linear(embed_dim, vdim, bias=bias)

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize all parameters with uniform distribution U[-0.1, 0.1]"""
        for module in [self.q_linear, self.k_linear, self.v_linear]:
            nn.init.uniform_(module.weight, -0.1, 0.1)
            if module.bias is not None:
                nn.init.uniform_(module.bias, -0.1, 0.1)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        Single head attention forward pass

        Args:
            query: [batch_size, tgt_len, embed_dim]
            key: [batch_size, src_len, embed_dim]
            value: [batch_size, src_len, embed_dim]
            key_padding_mask: [batch_size, src_len]
            attn_mask: [tgt_len, src_len]

        Returns:
            attn_output: [batch_size, tgt_len, vdim]
            attn_weights: [batch_size, tgt_len, src_len]
        """
        batch_size, tgt_len, _ = query.size()
        src_len = key.size(1)

        # Linear projections
        Q = self.q_linear(query)  # [batch_size, tgt_len, kdim]
        K = self.k_linear(key)  # [batch_size, src_len, kdim]
        V = self.v_linear(value)  # [batch_size, src_len, vdim]

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.kdim)
        # [batch_size, tgt_len, src_len]

        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # [tgt_len, src_len] -> [1, tgt_len, src_len]
                attn_mask = attn_mask.unsqueeze(0)
            scores = scores.masked_fill(attn_mask, float("-inf"))

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # [batch_size, src_len] -> [batch_size, 1, src_len]
            key_padding_mask = key_padding_mask.unsqueeze(1)
            scores = scores.masked_fill(key_padding_mask, float("-inf"))

        # Compute attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        # [batch_size, tgt_len, vdim]

        return attn_output, attn_weights


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
    ):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.bias = bias
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.batch_first = batch_first
        self.embed_dim = embed_dim

        # Head dimension (각 head가 사용할 실제 차원)
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads

        # Create multiple single head attention modules
        self.heads = nn.ModuleList(
            [
                SingleHeadAttention(
                    embed_dim=embed_dim,
                    kdim=self.kdim,
                    vdim=self.vdim,
                    dropout=dropout,
                    bias=bias,
                )
                for _ in range(num_heads)
            ]
        )

        # Output projection
        self.out_linear = nn.Linear(self.vdim * self.num_heads, embed_dim, bias=bias)

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize all parameters with uniform distribution U[-0.1, 0.1]"""
        # SingleHeadAttention modules initialize themselves
        nn.init.uniform_(self.out_linear.weight, -0.1, 0.1)
        if self.out_linear.bias is not None:
            nn.init.uniform_(self.out_linear.bias, -0.1, 0.1)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        Multi-head attention forward pass

        Args:
            query: [batch_size, tgt_len, embed_dim] or [tgt_len, batch_size, embed_dim]
            key: [batch_size, src_len, embed_dim] or [src_len, batch_size, embed_dim]
            value: [batch_size, src_len, embed_dim] or [src_len, batch_size, embed_dim]
            key_padding_mask: [batch_size, src_len] - True for padding positions
            attn_mask: [tgt_len, src_len] or [num_heads * batch_size, tgt_len, src_len]

        Returns:
            attn_output: [batch_size, tgt_len, embed_dim] or [tgt_len, batch_size, embed_dim]
            attn_weights: [batch_size, num_heads, tgt_len, src_len]
        """
        # Handle batch_first
        if not self.batch_first:
            # Convert from [seq_len, batch, embed_dim] to [batch, seq_len, embed_dim]
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        batch_size, tgt_len, d_embed = query.size()
        src_len = key.size(1)

        # Apply each head attention
        head_outputs = []
        head_weights = []

        for head in self.heads:
            head_output, head_weight = head(
                query, key, value, key_padding_mask, attn_mask
            )
            head_outputs.append(head_output)  # [batch_size, tgt_len, head_vdim]
            head_weights.append(
                head_weight.unsqueeze(1)
            )  # [batch_size, 1, tgt_len, src_len]

        # Concatenate all head outputs
        # [batch_size, tgt_len, head_vdim * num_heads] = [batch_size, tgt_len, vdim]
        attn_output = torch.cat(head_outputs, dim=-1)

        # Stack all head weights
        # [batch_size, num_heads, tgt_len, src_len]
        attn_weights = torch.cat(head_weights, dim=1)

        # Output projection
        attn_output = self.out_linear(attn_output)
        attn_output = self.dropout_layer(attn_output)

        # Handle batch_first
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, attn_weights
