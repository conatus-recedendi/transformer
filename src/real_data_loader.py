"""
실제 WMT 데이터 로딩 모듈
data/wmt14_en_de/train.txt, valid.txt, test.txt 형식으로 저장된 데이터 로드
"""

import os
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class SimpleVocabulary:
    """간단한 어휘 사전"""

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.special_tokens = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}

        # 특수 토큰 추가
        for token, idx in self.special_tokens.items():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def build_from_files(self, file_paths: List[str], vocab_size: int = 30000):
        """파일들로부터 어휘 사전 구축"""
        logger.info(f"Building vocabulary from {len(file_paths)} files...")

        token_counter = Counter()
        total_lines = 0

        for file_path in file_paths:
            if os.path.exists(file_path):
                logger.info(f"Processing {file_path}...")
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        # 탭으로 분리된 소스와 타겟 문장
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            src_tokens = parts[0].split()
                            tgt_tokens = parts[1].split()
                            token_counter.update(src_tokens)
                            token_counter.update(tgt_tokens)
                            total_lines += 1

                        if total_lines % 10000 == 0:
                            logger.info(f"  Processed {total_lines} lines...")

        logger.info(f"Total tokens: {sum(token_counter.values())}")
        logger.info(f"Unique tokens: {len(token_counter)}")

        # 가장 빈번한 토큰들 선택
        vocab_size_without_special = vocab_size - len(self.special_tokens)
        most_common = token_counter.most_common(vocab_size_without_special)

        # 어휘 사전 구축
        next_id = len(self.special_tokens)
        for token, freq in most_common:
            if token not in self.token_to_id:
                self.token_to_id[token] = next_id
                self.id_to_token[next_id] = token
                next_id += 1

        logger.info(f"Vocabulary built: {len(self.token_to_id)} tokens")

    def encode(self, tokens: List[str]) -> List[int]:
        """토큰들을 ID로 변환"""
        return [
            self.token_to_id.get(token, self.special_tokens["<unk>"])
            for token in tokens
        ]

    def decode(self, ids: List[int]) -> List[str]:
        """ID들을 토큰으로 변환"""
        return [self.id_to_token.get(id, "<unk>") for id in ids]

    def __len__(self):
        return len(self.token_to_id)


class RealWMTDataset(Dataset):
    """실제 WMT 데이터셋 클래스 (탭 분리 형식)"""

    def __init__(self, file_path: str, vocab: SimpleVocabulary, max_length: int = 512):
        self.file_path = file_path
        self.vocab = vocab
        self.max_length = max_length

        # 데이터 로드
        self.data_pairs = self._load_data()

        logger.info(f"Loaded {len(self.data_pairs)} sentence pairs from {file_path}")

    def _load_data(self) -> List[Tuple[List[str], List[str]]]:
        """데이터 파일 로드"""
        data_pairs = []

        if not os.path.exists(self.file_path):
            logger.warning(f"Data file not found: {self.file_path}")
            return data_pairs

        with open(self.file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                # 탭으로 분리된 소스와 타겟 문장
                parts = line.split("\t")
                if len(parts) >= 2:
                    src_tokens = parts[0].split()
                    tgt_tokens = parts[1].split()

                    # 길이 제한 및 빈 라인 필터링
                    if (
                        len(src_tokens) > 0
                        and len(tgt_tokens) > 0
                        and len(src_tokens) <= self.max_length
                        and len(tgt_tokens) <= self.max_length
                    ):
                        data_pairs.append((src_tokens, tgt_tokens))
                else:
                    logger.warning(
                        f"Invalid line format at line {line_num + 1}: {line}"
                    )

        return data_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        src_tokens, tgt_tokens = self.data_pairs[idx]

        # 토큰을 ID로 변환
        src_ids = self.vocab.encode(src_tokens)
        tgt_ids = self.vocab.encode(tgt_tokens)

        # BOS/EOS 토큰 추가
        tgt_input = [self.vocab.special_tokens["<bos>"]] + tgt_ids
        tgt_output = tgt_ids + [self.vocab.special_tokens["<eos>"]]

        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_input, dtype=torch.long),
            "tgt_y": torch.tensor(tgt_output, dtype=torch.long),
            "src_len": len(src_ids),
            "tgt_len": len(tgt_input),
        }


def collate_fn_real(batch, pad_token_id: int = 0):
    """배치 데이터 패딩"""
    src_seqs = [item["src"] for item in batch]
    tgt_seqs = [item["tgt"] for item in batch]
    tgt_y_seqs = [item["tgt_y"] for item in batch]

    # 패딩
    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_seqs, batch_first=True, padding_value=pad_token_id
    )
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_seqs, batch_first=True, padding_value=pad_token_id
    )
    tgt_y_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_y_seqs, batch_first=True, padding_value=pad_token_id
    )

    return {
        "src": src_padded,
        "tgt": tgt_padded,
        "tgt_y": tgt_y_padded,
    }


def load_real_wmt_data(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """실제 WMT 데이터 로드"""
    logger.info("Loading real WMT data...")

    # 데이터 경로 설정
    data_path = Path(config.DATA_PATH)
    dataset_path = data_path / config.DATASET

    train_file = dataset_path / "train.txt"
    valid_file = dataset_path / "valid.txt"
    test_file = dataset_path / "test.txt"

    # 파일 존재 확인
    files_exist = all(f.exists() for f in [train_file, valid_file, test_file])
    if not files_exist:
        logger.error(f"Data files not found in {dataset_path}")
        logger.error(f"Expected files: train.txt, valid.txt, test.txt")
        return None, None, None

    logger.info(f"Found data files in {dataset_path}")
    logger.info(f"  Train: {train_file}")
    logger.info(f"  Valid: {valid_file}")
    logger.info(f"  Test: {test_file}")

    # 어휘 사전 구축
    vocab = SimpleVocabulary()
    vocab.build_from_files([str(train_file)], config.VOCAB_SIZE)

    # 데이터셋 생성
    train_dataset = RealWMTDataset(str(train_file), vocab, config.MAX_SEQ_LENGTH)
    val_dataset = RealWMTDataset(str(valid_file), vocab, config.MAX_SEQ_LENGTH)
    test_dataset = RealWMTDataset(str(test_file), vocab, config.MAX_SEQ_LENGTH)

    logger.info(f"Dataset sizes:")
    logger.info(f"  Train: {len(train_dataset):,} samples")
    logger.info(f"  Valid: {len(val_dataset):,} samples")
    logger.info(f"  Test: {len(test_dataset):,} samples")

    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_real(batch, config.PAD_TOKEN),
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_real(batch, config.PAD_TOKEN),
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_real(batch, config.PAD_TOKEN),
        num_workers=0,
        pin_memory=True,
    )

    logger.info(f"Data loaders created:")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Valid batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader
