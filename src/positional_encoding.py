import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len=5000, device=torch.device("cpu")):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(
            1
        )
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model] input embeddings
        Returns:
            [batch_size, seq_len, d_model] embeddings with positional encoding added
        """
        batch_size, seq_len, d_model = x.size()
        return x + self.pe[:seq_len, :].squeeze(1)  # [seq_len, d_model]
