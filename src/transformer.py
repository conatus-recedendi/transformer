import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder
from .decoder import Decoder, create_causal_mask, create_padding_mask
from .positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    """Complete Transformer model for sequence-to-sequence tasks"""

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        max_seq_length=5000,
        dropout=0.1,
        pad_token_id=0,
        tie_weights=True,
        kdim=64,
        vdim=64,
    ):
        """
        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            d_model: Model dimension
            num_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            d_ff: Feed-forward dimension
            max_seq_length: Maximum sequence length
            dropout: Dropout probability
            pad_token_id: Padding token ID
            tie_weights: Whether to tie decoder embedding and output projection weights
            kdim: Dimension of key vectors
            vdim: Dimension of value vectors
        """
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.tie_weights = tie_weights

        # Input embedding
        # src_vocab_size alwasy equals tgt_vocab_size. bcz of shared BPE
        self.embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_token_id)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

        # Encoder
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            dropout=dropout,
            padding_idx=pad_token_id,
            kdim=kdim,
            vdim=vdim,
        )

        # Decoder
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            dropout=dropout,
            padding_idx=pad_token_id,
            kdim=kdim,
            vdim=vdim,
        )

        # Output projection layer
        self.output_projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

        # Optionally tie decoder embedding weights with output projection
        if tie_weights:
            self.output_projection.weight = self.embedding.weight

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize all parameters with uniform distribution U[-0.1, 0.1]"""
        # Initialize embedding layer
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        # Initialize output projection if not tied
        if not self.tie_weights:
            nn.init.uniform_(self.output_projection.weight, -0.1, 0.1)

        print(
            f"Transformer: Initialized embedding and output projection with U[-0.1, 0.1]"
        )

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        Forward pass of the transformer

        Args:
            src: [batch_size, src_len] source token indices
            tgt: [batch_size, tgt_len] target token indices
            src_mask: [src_len, src_len] source attention mask
            tgt_mask: [tgt_len, tgt_len] target attention mask (causal)
            src_key_padding_mask: [batch_size, src_len] source padding mask
            tgt_key_padding_mask: [batch_size, tgt_len] target padding mask
            memory_key_padding_mask: [batch_size, src_len] encoder output padding mask

        Returns:
            output: [batch_size, tgt_len, tgt_vocab_size] logits
        """
        # Create padding masks if not provided
        if src_key_padding_mask is None:
            src_key_padding_mask = create_padding_mask(src, self.pad_token_id)

        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = create_padding_mask(tgt, self.pad_token_id)

        if memory_key_padding_mask is None:
            memory_key_padding_mask = src_key_padding_mask

        # Create causal mask for target if not provided
        if tgt_mask is None:
            tgt_len = tgt.size(1)
            tgt_mask = create_causal_mask(tgt_len, device=tgt.device)

        x = self.embedding(src) * math.sqrt(
            self.d_model
        )  # [batch_size, src_len, d_model] Input embedding with scaling
        x = self.pos_encoding(x)
        # Encoder forward pass
        encoder_output = self.encoder(
            x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )
        y = self.embedding(tgt) * math.sqrt(
            self.d_model
        )  # [batch_size, tgt_len, d_model] Input embedding with scaling
        y = self.pos_encoding(y)
        # Decoder forward pass
        decoder_output = self.decoder(
            y,
            encoder_output,
            tgt_mask=tgt_mask,
            memory_mask=None,  # Usually not used
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        # Output projection to vocabulary
        output = self.output_projection(decoder_output)

        return output

    def encode(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Encode source sequence

        Args:
            src: [batch_size, src_len] source token indices
            src_mask: [src_len, src_len] source attention mask
            src_key_padding_mask: [batch_size, src_len] source padding mask

        Returns:
            encoder_output: [batch_size, src_len, d_model]
        """
        if src_key_padding_mask is None:
            src_key_padding_mask = create_padding_mask(src, self.pad_token_id)

        return self.encoder(
            src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )

    def decode(
        self,
        tgt,
        encoder_output,
        tgt_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        Decode target sequence given encoder output

        Args:
            tgt: [batch_size, tgt_len] target token indices
            encoder_output: [batch_size, src_len, d_model] encoder output
            tgt_mask: [tgt_len, tgt_len] target attention mask (causal)
            tgt_key_padding_mask: [batch_size, tgt_len] target padding mask
            memory_key_padding_mask: [batch_size, src_len] encoder output padding mask

        Returns:
            output: [batch_size, tgt_len, tgt_vocab_size] logits
        """
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = create_padding_mask(tgt, self.pad_token_id)

        if tgt_mask is None:
            tgt_len = tgt.size(1)
            tgt_mask = create_causal_mask(tgt_len, device=tgt.device)

        decoder_output = self.decoder(
            tgt,
            encoder_output,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        return self.output_projection(decoder_output)

    def generate(
        self,
        src,
        max_length=100,
        start_token=1,
        end_token=2,
        temperature=1.0,
        top_k=None,
        top_p=None,
        do_sample=True,
    ):
        """
        Generate target sequence given source sequence (inference)

        Args:
            src: [batch_size, src_len] source token indices
            max_length: Maximum generation length
            start_token: Start token ID
            end_token: End token ID
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            do_sample: Whether to sample or use greedy decoding

        Returns:
            generated: [batch_size, generated_len] generated token indices
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device

        # Encode source
        encoder_output = self.encode(src)

        # Initialize target with start token
        tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)

        # Generate tokens one by one
        for _ in range(max_length - 1):
            # Decode current target sequence
            logits = self.decode(
                tgt, encoder_output
            )  # [batch_size, tgt_len, vocab_size]

            # Get logits for the last position
            next_token_logits = (
                logits[:, -1, :] / temperature
            )  # [batch_size, vocab_size]

            # Apply top-k sampling
            if top_k is not None:
                indices_to_remove = (
                    next_token_logits
                    < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                )
                next_token_logits[indices_to_remove] = float("-inf")

            # Apply top-p sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float("-inf")

            # Sample next token
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to target sequence
            tgt = torch.cat([tgt, next_token], dim=1)

            # Check if all sequences have generated end token
            if (next_token == end_token).all():
                break

        return tgt

    def get_attention_weights(
        self,
        src,
        tgt,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        Get attention weights for visualization

        Returns:
            encoder_self_attn: List of encoder self-attention weights
            decoder_self_attn: List of decoder self-attention weights
            decoder_cross_attn: List of decoder cross-attention weights
        """
        if src_key_padding_mask is None:
            src_key_padding_mask = create_padding_mask(src, self.pad_token_id)

        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = create_padding_mask(tgt, self.pad_token_id)

        if memory_key_padding_mask is None:
            memory_key_padding_mask = src_key_padding_mask

        tgt_len = tgt.size(1)
        tgt_mask = create_causal_mask(tgt_len, device=tgt.device)

        src = self.embedding(src) * math.sqrt(
            self.d_model
        )  # [batch_size, src_len, d_model] Input embedding with scaling
        # Add positional encoding
        src = self.pos_encoding(src)

        # Get encoder attention weights
        encoder_self_attn = self.encoder.get_attention_weights(
            src, src_key_padding_mask=src_key_padding_mask
        )

        # Get encoder output
        encoder_output = self.encoder(src, src_key_padding_mask=src_key_padding_mask)

        tgt = self.embedding(tgt) * math.sqrt(
            self.d_model
        )  # [batch_size, tgt_len, d_model] Input embedding with scaling
        # Add positional encoding
        tgt = self.pos_encoding(tgt)

        # Get decoder attention weights
        decoder_self_attn, decoder_cross_attn = self.decoder.get_attention_weights(
            tgt,
            encoder_output,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        return encoder_self_attn, decoder_self_attn, decoder_cross_attn

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
