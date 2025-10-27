"""
Data loading utilities for Transformer model
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class TransformerDataset(Dataset):
    """Dataset class for sequence-to-sequence tasks"""

    def __init__(
        self,
        src_sequences: List[List[int]],
        tgt_sequences: List[List[int]],
        src_vocab_size: int,
        tgt_vocab_size: int,
        max_length: int = 512,
    ):
        """
        Args:
            src_sequences: List of source sequences (tokenized)
            tgt_sequences: List of target sequences (tokenized)
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            max_length: Maximum sequence length
        """
        assert len(src_sequences) == len(
            tgt_sequences
        ), "Source and target sequences must have same length"

        self.src_sequences = src_sequences
        self.tgt_sequences = tgt_sequences
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_length = max_length

    def __len__(self):
        return len(self.src_sequences)

    def __getitem__(self, idx):
        src = self.src_sequences[idx][: self.max_length]
        tgt = self.tgt_sequences[idx][: self.max_length]

        return {
            "src": torch.tensor(src, dtype=torch.long),
            "tgt": torch.tensor(tgt, dtype=torch.long),
            "src_length": len(src),
            "tgt_length": len(tgt),
        }


class TextDataset(Dataset):
    """Dataset for text data with tokenization"""

    def __init__(
        self,
        file_path: str,
        tokenizer,
        max_length: int = 512,
        task_type: str = "translation",
    ):
        """
        Args:
            file_path: Path to data file
            tokenizer: Tokenizer object
            max_length: Maximum sequence length
            task_type: Type of task ('translation', 'language_modeling', etc.)
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        self.data = self._load_data()

    def _load_data(self):
        """Load and preprocess data"""
        # TODO: Implement data loading logic based on task type
        # This is a placeholder - you should implement according to your data format
        data = []

        if self.task_type == "translation":
            # Expected format: source_text \t target_text
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "\t" in line:
                        src, tgt = line.strip().split("\t", 1)
                        data.append(
                            {
                                "src": self.tokenizer.encode(src),
                                "tgt": self.tokenizer.encode(tgt),
                            }
                        )

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "src": torch.tensor(item["src"][: self.max_length], dtype=torch.long),
            "tgt": torch.tensor(item["tgt"][: self.max_length], dtype=torch.long),
        }


def collate_fn(
    batch: List[Dict[str, torch.Tensor]], pad_token_id: int = 0
) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader

    Args:
        batch: List of batch items
        pad_token_id: Padding token ID

    Returns:
        Batched and padded tensors
    """
    src_sequences = [item["src"] for item in batch]
    tgt_sequences = [item["tgt"] for item in batch]

    # Pad sequences
    src_padded = pad_sequence(
        src_sequences, batch_first=True, padding_value=pad_token_id
    )
    tgt_padded = pad_sequence(
        tgt_sequences, batch_first=True, padding_value=pad_token_id
    )

    # Create attention masks
    src_mask = src_padded != pad_token_id
    tgt_mask = tgt_padded != pad_token_id

    return {
        "src": src_padded,
        "tgt": tgt_padded,
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
        "src_length": torch.tensor([len(seq) for seq in src_sequences]),
        "tgt_length": torch.tensor([len(seq) for seq in tgt_sequences]),
    }


def create_data_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pad_token_id: int = 0,
) -> DataLoader:
    """
    Create DataLoader with custom collate function

    Args:
        dataset: Dataset object
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pad_token_id: Padding token ID

    Returns:
        DataLoader object
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id),
        pin_memory=True,
    )


class DummyTokenizer:
    """Dummy tokenizer for testing purposes"""

    def __init__(self, vocab_size: int = 30000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

    def encode(self, text: str) -> List[int]:
        """Dummy encoding - replace with actual tokenizer"""
        # This is a placeholder - implement with actual tokenization logic
        words = text.split()
        return [hash(word) % (self.vocab_size - 4) + 4 for word in words]

    def decode(self, token_ids: List[int]) -> str:
        """Dummy decoding - replace with actual tokenizer"""
        # This is a placeholder - implement with actual detokenization logic
        return " ".join([f"token_{tid}" for tid in token_ids])


def load_dummy_data(num_samples: int = 1000) -> Tuple[List[List[int]], List[List[int]]]:
    """Generate dummy data for testing"""
    np.random.seed(42)

    src_sequences = []
    tgt_sequences = []

    for _ in range(num_samples):
        src_len = np.random.randint(5, 50)
        tgt_len = np.random.randint(5, 50)

        src = np.random.randint(4, 1000, src_len).tolist()
        tgt = np.random.randint(4, 1000, tgt_len).tolist()

        # Add BOS and EOS tokens
        src = [1] + src + [2]  # BOS + sequence + EOS
        tgt = [1] + tgt + [2]  # BOS + sequence + EOS

        src_sequences.append(src)
        tgt_sequences.append(tgt)

    return src_sequences, tgt_sequences
