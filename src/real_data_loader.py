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
        """파일들로부터 어휘 사전 구축 (분리된 언어 파일들 처리)"""
        logger.info(f"Building vocabulary from {len(file_paths)} files...")

        token_counter = Counter()
        total_lines = 0

        for file_path in file_paths:
            if os.path.exists(file_path):
                logger.info(f"Processing {file_path}...")
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            tokens = line.split()
                            token_counter.update(tokens)
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
    """실제 WMT 데이터셋 클래스 (분리된 언어 파일 형식: train.en, train.de)"""

    def __init__(
        self,
        src_file: str,
        tgt_file: str,
        vocab: SimpleVocabulary,
        max_length: int = 512,
    ):
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.vocab = vocab
        self.max_length = max_length

        # 데이터 로드
        self.data_pairs = self._load_data()

        logger.info(f"Loaded {len(self.data_pairs)} sentence pairs")
        logger.info(f"  Source file: {src_file}")
        logger.info(f"  Target file: {tgt_file}")

    def _load_data(self) -> List[Tuple[List[str], List[str]]]:
        """분리된 언어 파일들 로드"""
        data_pairs = []

        if not os.path.exists(self.src_file) or not os.path.exists(self.tgt_file):
            logger.warning(f"Data files not found: {self.src_file} or {self.tgt_file}")
            return data_pairs

        with open(self.src_file, "r", encoding="utf-8") as f_src, open(
            self.tgt_file, "r", encoding="utf-8"
        ) as f_tgt:

            for line_num, (src_line, tgt_line) in enumerate(zip(f_src, f_tgt)):
                src_line = src_line.strip()
                tgt_line = tgt_line.strip()

                if not src_line or not tgt_line:
                    continue

                src_tokens = src_line.split()
                tgt_tokens = tgt_line.split()

                # 길이 제한 및 빈 라인 필터링
                if (
                    len(src_tokens) > 0
                    and len(tgt_tokens) > 0
                    and len(src_tokens) <= self.max_length
                    and len(tgt_tokens) <= self.max_length
                ):
                    data_pairs.append((src_tokens, tgt_tokens))

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
    """실제 WMT 데이터 로드 (분리된 언어 파일 형식: train.en, train.de 등)"""
    logger.info("Loading real WMT data...")

    # 데이터 경로 설정
    data_path = Path(config.DATA_PATH)
    dataset_path = data_path / config.DATASET

    # config에서 언어 정보 가져오기
    src_lang = getattr(config, "SRC_LANG", "en")
    tgt_lang = getattr(config, "TGT_LANG", "de")

    # 분리된 언어 파일 경로
    train_src_file = dataset_path / f"train.{src_lang}"
    train_tgt_file = dataset_path / f"train.{tgt_lang}"
    valid_src_file = dataset_path / f"valid.{src_lang}"
    valid_tgt_file = dataset_path / f"valid.{tgt_lang}"
    test_src_file = dataset_path / f"test.{src_lang}"
    test_tgt_file = dataset_path / f"test.{tgt_lang}"

    # 파일 존재 확인
    required_files = [
        train_src_file,
        train_tgt_file,
        valid_src_file,
        valid_tgt_file,
        test_src_file,
        test_tgt_file,
    ]
    files_exist = all(f.exists() for f in required_files)

    if not files_exist:
        logger.error(f"Data files not found in {dataset_path}")
        logger.error(f"Expected files:")
        for f in required_files:
            status = "✓" if f.exists() else "✗"
            logger.error(f"  {status} {f.name}")
        return None, None, None

    logger.info(f"Found data files in {dataset_path}")
    logger.info(f"  Language pair: {src_lang} → {tgt_lang}")
    logger.info(f"  Train: {train_src_file.name}, {train_tgt_file.name}")
    logger.info(f"  Valid: {valid_src_file.name}, {valid_tgt_file.name}")
    logger.info(f"  Test: {test_src_file.name}, {test_tgt_file.name}")

    # 어휘 사전 구축 (훈련 데이터에서)
    vocab = SimpleVocabulary()
    vocab_files = [str(train_src_file), str(train_tgt_file)]
    vocab.build_from_files(vocab_files, config.VOCAB_SIZE)

    # 데이터셋 생성
    train_dataset = RealWMTDataset(
        str(train_src_file), str(train_tgt_file), vocab, config.MAX_SEQ_LENGTH
    )
    val_dataset = RealWMTDataset(
        str(valid_src_file), str(valid_tgt_file), vocab, config.MAX_SEQ_LENGTH
    )
    test_dataset = RealWMTDataset(
        str(test_src_file), str(test_tgt_file), vocab, config.MAX_SEQ_LENGTH
    )

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
