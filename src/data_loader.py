"""
Data loading utilities for Transformer model
Includes data cleaning functions from Tensor2Tensor
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import re
import itertools
from typing import List, Tuple, Optional, Dict, Any


# Data cleaning patterns from Tensor2Tensor
_RE_GOOD_S_START = re.compile(r'^["""]?[A-Z]')
_RE_GOOD_S_END = re.compile(r'\w[.?!][""]?$', re.UNICODE)

_RE_LABEL_COLON = re.compile(r"^\w+\.?( \w+)?: ", re.UNICODE)
_RE_DIGIT_SPACE_DIGIT = re.compile(r"\d +\d", re.UNICODE)
_RE_ALL_CAP_WORDS = re.compile(r"^[A-Z]\S*(\s+[A-Z]\S+)+\s*$")

_RE_DQ_ONE = re.compile(r'^[^"""]*["""][^"""]*$')
_RE_DQ_INITIAL = re.compile(r'^["""]([^"""]+)$')
_RE_DQ_FINAL = re.compile(r'^[^"""]+["""]$')
_RE_DQ_LINE = re.compile(r'^["""].*["""]$')

_RE_DQ_MANY = re.compile(r'(["""].*){3,}')
_RE_SQ_MANY = re.compile(r"(['''][^st].*){3,}")
_RE_CHARS_QQ = re.compile(r"[\"\"\"''']\\s*[\"\"\"''']")
_RE_SPACE_PUNCT_SPACE = re.compile(r"\\s[\"\"\"''',:;]\\s")

_RE_COPYRIGHT = re.compile(r"©|^Copyright|^\(C\)")
_RE_UNMATCHED_PAREN_LEFT = re.compile(r"[(][^)]*$")
_RE_UNMATCHED_PAREN_RIGHT = re.compile(r"^[^(]*[)]")
_RE_TAGLINE_CITY = re.compile(r"^[A-Z]{2,}(\s+[A-Z]+)*\s+-")
_RE_CHARS_UPPER_UNDERSCORE = re.compile(r"^[A-Z]+[a-z]*_")


def clean_sentence_pairs(
    src_sentences: List[str], tgt_sentences: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Clean and filter sentence pairs using Tensor2Tensor cleaning rules

    Args:
        src_sentences: List of source sentences
        tgt_sentences: List of target sentences

    Returns:
        Tuple of (cleaned_src, cleaned_tgt) lists
    """
    cleaned_src = []
    cleaned_tgt = []

    total_pairs = len(src_sentences)
    filtered_count = 0
    split_count = 0

    for src, tgt in zip(src_sentences, tgt_sentences):
        if _regex_filter(src):
            filtered_count += 1
            continue

        src_list, tgt_list = _split_sentences(src, tgt)

        if len(src_list) != len(tgt_list):
            filtered_count += 1
            continue  # discard this pair
        elif len(src_list) == 1:
            cleaned_src.append(src)
            cleaned_tgt.append(tgt)
        else:
            split_count += len(src_list)
            for src_sub, tgt_sub in zip(src_list, tgt_list):
                if _regex_filter(src_sub):
                    continue
                cleaned_src.append(src_sub)
                cleaned_tgt.append(tgt_sub)

    print(f"Data cleaning results:")
    print(f"  Original pairs: {total_pairs:,}")
    print(f"  Filtered out: {filtered_count:,}")
    print(f"  Split sentences: {split_count:,}")
    print(f"  Final pairs: {len(cleaned_src):,}")
    print(f"  Retention rate: {len(cleaned_src)/total_pairs*100:.1f}%")

    return cleaned_src, cleaned_tgt


def _regex_filter(sentence: str) -> bool:
    """Apply regex filters to determine if sentence should be discarded"""
    return (
        not _is_match(sentence, _RE_GOOD_S_START)
        or not _is_match(sentence, _RE_GOOD_S_END)
        or _is_match(sentence, _RE_LABEL_COLON)
        or _is_match(sentence, _RE_DIGIT_SPACE_DIGIT)
        or _is_match(sentence, _RE_DQ_ONE)
        or _is_match(sentence, _RE_DQ_INITIAL)
        or _is_match(sentence, _RE_DQ_FINAL)
        or _is_match(sentence, _RE_DQ_LINE)
        or _is_match(sentence, _RE_DQ_MANY)
        or _is_match(sentence, _RE_SQ_MANY)
        or _is_match(sentence, _RE_CHARS_QQ)
        or _is_match(sentence, _RE_SPACE_PUNCT_SPACE)
        or _is_match(sentence, _RE_COPYRIGHT)
        or _is_match(sentence, _RE_UNMATCHED_PAREN_LEFT)
        or _is_match(sentence, _RE_UNMATCHED_PAREN_RIGHT)
        or _is_match(sentence, _RE_TAGLINE_CITY)
        or _is_match(sentence, _RE_CHARS_UPPER_UNDERSCORE)
    )


def _is_match(sentence: str, regex) -> bool:
    """Check if regex matches the sentence"""
    return regex.search(sentence) is not None


def _split_sentences(s1: str, s2: str) -> Tuple[List[str], List[str]]:
    """Split sentences at sentence boundaries"""
    # Convert to unicode if needed
    if isinstance(s1, bytes):
        s1 = s1.decode("utf-8", errors="ignore")
    if isinstance(s2, bytes):
        s2 = s2.decode("utf-8", errors="ignore")

    # Split sentences using regex patterns
    s1 = re.sub(r"(\w[A-Z]|[0-9a-z])([.!?]) ([A-Z])", r"\1\2__|__\3", s1)
    s2 = re.sub(r"([^0-9][.!?]) ([A-Z])", r"\1__|__\2", s2)

    s1_subsentences = s1.split("__|__")
    s2_subsentences = s2.split("__|__")

    return s1_subsentences, s2_subsentences


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
