import torch
import torch.nn as nn

import math


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
        # self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.bias = bias
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.batch_first = batch_first
        self.embed_dim = embed_dim

        # Head dimension
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.q_k_embed_dim = self.kdim * num_heads

        # Linear projections for Q, K, V (all heads combined)
        self.q_linear = nn.Linear(embed_dim, self.kdim * num_heads, bias=bias)
        self.k_linear = nn.Linear(embed_dim, self.kdim * num_heads, bias=bias)
        self.v_linear = nn.Linear(embed_dim, self.vdim * num_heads, bias=bias)

        # Output projection
        self.out_linear = nn.Linear(self.vdim * num_heads, embed_dim, bias=bias)

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize all parameters with uniform distribution U[-0.1, 0.1]"""
        for module in [self.q_linear, self.k_linear, self.v_linear, self.out_linear]:
            nn.init.uniform_(module.weight, -0.1, 0.1)
            if module.bias is not None:
                nn.init.uniform_(module.bias, -0.1, 0.1)

        # print(f"MultiheadAttention: Initialized all parameters with U[-0.1, 0.1]")

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

        # Linear projections for Q, K, V
        Q = self.q_linear(query)  # [batch_size, tgt_len, kdim * num_heads]
        K = self.k_linear(key)  # [batch_size, src_len, kdim * num_heads]
        V = self.v_linear(value)  # [batch_size, src_len, vdim * num_heads]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, tgt_len, self.num_heads, self.kdim)
        K = K.view(batch_size, src_len, self.num_heads, self.kdim)
        V = V.view(batch_size, src_len, self.num_heads, self.vdim)

        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention
        attn_output, attn_weights = self._scaled_dot_product_attention(
            Q, K, V, key_padding_mask, attn_mask
        )

        # Concatenate heads: [batch_size, num_heads, tgt_len, vdim] -> [batch_size, tgt_len, vdim * num_heads]
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, tgt_len, self.vdim * self.num_heads)
        )

        # Output projection
        attn_output = self.out_linear(attn_output)
        attn_output = self.dropout_layer(attn_output)

        # Handle batch_first
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, attn_weights

    def _scaled_dot_product_attention(
        self, Q, K, V, key_padding_mask=None, attn_mask=None
    ):
        """
        Scaled dot-product attention

        Args:
            Q: [batch_size, num_heads, tgt_len, kdim]
            K: [batch_size, num_heads, src_len, kdim]
            V: [batch_size, num_heads, src_len, vdim]
            key_padding_mask: [batch_size, src_len]
            attn_mask: [tgt_len, src_len]

        Returns:
            attn_output: [batch_size, num_heads, tgt_len, head_dim]
            attn_weights: [batch_size, num_heads, tgt_len, src_len]
        """
        batch_size, num_heads, tgt_len, kdim = Q.size()
        _, _, _, vdim = V.size()
        src_len = K.size(2)

        # Compute attention scores
        # [batch_size, num_heads, tgt_len, kdim] x [batch_size, num_heads, kdim, src_len]
        # -> [batch_size, num_heads, tgt_len, src_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(kdim)

        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # [tgt_len, src_len] -> [1, 1, tgt_len, src_len]
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).transpose(2, 3)
            # attn_mask: True=masked positions, False=allowed positions
            scores = scores.masked_fill(attn_mask, float("-inf"))

            # attn_mask:
            # 1 1 1
            # 0 1 1
            # 0 0 1

            # scores
            # -inf -inf -inf
            #  x    -inf -inf
            #  x     x   -inf

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # [batch_size, src_len] -> [batch_size, 1, 1, src_len]
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(key_padding_mask, float("-inf"))

        # Compute attention weights
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        # [batch_size, num_heads, tgt_len, src_len] x [batch_size, num_heads, src_len, vdim]
        # -> [batch_size, num_heads, tgt_len, vdim]

        return attn_output, attn_weights
