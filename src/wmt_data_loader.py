"""
WMT 데이터셋 로딩 및 전처리 모듈
"""

import os
import json
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import logging
import spacy

logger = logging.getLogger(__name__)


class WMTVocabulary:
    """WMT 데이터셋용 어휘 사전"""

    def __init__(self, vocab_file: str = None):
        self.token_to_id = {}
        self.id_to_token = {}
        self.special_tokens = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}

        # 특수 토큰 추가
        for token, idx in self.special_tokens.items():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)

    def build_vocab_from_files(self, file_paths: List[str], vocab_size: int = 37000):
        """파일들로부터 어휘 사전 구축"""
        logger.info(f"Building vocabulary from {len(file_paths)} files...")

        # 토큰 빈도수 계산
        token_counter = Counter()
        total_lines = 0

        for file_path in file_paths:
            if os.path.exists(file_path):
                logger.info(f"Processing {file_path}...")
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        tokens = line.strip().split()
                        token_counter.update(tokens)
                        total_lines += 1

                        if total_lines % 100000 == 0:
                            logger.info(f"  Processed {total_lines} lines...")

        logger.info(f"Total tokens: {sum(token_counter.values())}")
        logger.info(f"Unique tokens: {len(token_counter)}")

        # 가장 빈번한 토큰들 선택 (특수 토큰 제외)
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

    def save_vocab(self, vocab_file: str):
        """어휘 사전 저장"""
        vocab_data = {
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "special_tokens": self.special_tokens,
        }

        os.makedirs(os.path.dirname(vocab_file), exist_ok=True)
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Vocabulary saved to {vocab_file}")

    def load_vocab(self, vocab_file: str):
        """어휘 사전 로드"""
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        self.token_to_id = {k: int(v) for k, v in vocab_data["token_to_id"].items()}
        self.id_to_token = {int(k): v for k, v in vocab_data["id_to_token"].items()}
        self.special_tokens = vocab_data["special_tokens"]

        logger.info(
            f"Vocabulary loaded from {vocab_file}: {len(self.token_to_id)} tokens"
        )

    def encode(self, tokens: List[str]) -> List[int]:
        """토큰들을 ID로 변환"""
        return [
            self.token_to_id.get(token, self.special_tokens["<UNK>"])
            for token in tokens
        ]

    def decode(self, ids: List[int]) -> List[str]:
        """ID들을 토큰으로 변환"""
        return [self.id_to_token.get(id, "<UNK>") for id in ids]

    def __len__(self):
        return len(self.token_to_id)


class WMTDataset(Dataset):
    """WMT 데이터셋 클래스"""

    def __init__(
        self,
        src_file: str,
        tgt_file: str,
        src_vocab: WMTVocabulary,
        tgt_vocab: WMTVocabulary,
        max_length: int = 512,
    ):
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length

        # 데이터 로드

        # self.spacy_src = spacy.load("en_core_web_sm")  # de
        # self.spacy_tgt = spacy.load("de_core_news_sm")  # en

        self.spacy_src = spacy.load("de_core_news_sm")
        self.spacy_tgt = spacy.load("en_core_web_sm")

        self.src_sentences, self.tgt_sentences = self._load_data()

        logger.info(f"Loaded {len(self.src_sentences)} sentence pairs")
        logger.info(f"  Source file: {src_file}")
        logger.info(f"  Target file: {tgt_file}")

    def _load_data(self) -> Tuple[List[List[str]], List[List[str]]]:
        """데이터 파일들 로드"""
        src_sentences = []
        tgt_sentences = []

        if not os.path.exists(self.src_file) or not os.path.exists(self.tgt_file):
            logger.warning(f"Data files not found: {self.src_file} or {self.tgt_file}")
            return src_sentences, tgt_sentences

        with open(self.src_file, "r", encoding="utf-8") as f_src, open(
            self.tgt_file, "r", encoding="utf-8"
        ) as f_tgt:

            for line_num, (src_line, tgt_line) in enumerate(zip(f_src, f_tgt)):
                # src_tokens = src_line.strip().split()
                # tgt_tokens = tgt_line.strip().split()
                src_tokens = [token.text for token in self.spacy_src(src_line.strip())]
                tgt_tokens = [token.text for token in self.spacy_tgt(tgt_line.strip())]

                # 길이 제한 및 빈 라인 필터링
                if (
                    len(src_tokens) > 0
                    and len(tgt_tokens) > 0
                    and len(src_tokens) <= self.max_length
                    and len(tgt_tokens) <= self.max_length
                ):
                    src_sentences.append(src_tokens)
                    tgt_sentences.append(tgt_tokens)

        return src_sentences, tgt_sentences

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_tokens = self.src_sentences[idx]
        tgt_tokens = self.tgt_sentences[idx]

        # 토큰을 ID로 변환
        src_ids = self.src_vocab.encode(src_tokens)
        tgt_ids = self.tgt_vocab.encode(tgt_tokens)

        # BOS/EOS 토큰 추가
        src_ids = src_ids + [self.src_vocab.special_tokens["<EOS>"]]
        tgt_input = [self.tgt_vocab.special_tokens["<BOS>"]] + tgt_ids
        tgt_output = tgt_ids + [self.tgt_vocab.special_tokens["<EOS>"]]

        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_input, dtype=torch.long),
            "tgt_y": torch.tensor(tgt_output, dtype=torch.long),
            "src_len": len(src_ids),
            "tgt_len": len(tgt_input),
        }


def collate_fn(batch, pad_token_id: int = 0):
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

    # 마스크 생성
    src_mask = src_padded != pad_token_id
    tgt_mask = tgt_padded != pad_token_id

    return {
        "src": src_padded,
        "tgt": tgt_padded,
        "tgt_y": tgt_y_padded,
        "src_key_padding_mask": ~src_mask,  # True for padding positions
        "tgt_key_padding_mask": ~tgt_mask,
    }


def setup_wmt_vocabulary(config) -> Tuple[WMTVocabulary, WMTVocabulary]:
    """WMT 어휘 사전 설정"""
    data_path = Path(config.DATA_PATH)
    vocab_path = data_path / "vocab"
    vocab_path.mkdir(parents=True, exist_ok=True)

    if config.SHARED_VOCAB:
        # 공유 어휘 사전
        vocab_file = vocab_path / "vocab_shared.json"

        if vocab_file.exists():
            logger.info("Loading existing shared vocabulary...")
            shared_vocab = WMTVocabulary(str(vocab_file))
            return shared_vocab, shared_vocab
        else:
            logger.info("Building shared vocabulary...")
            shared_vocab = WMTVocabulary()

            # 모든 훈련 파일 수집
            train_files = []
            lang_dir = data_path / f"{config.SRC_LANG}-{config.TGT_LANG}"

            for lang in [config.SRC_LANG, config.TGT_LANG]:
                train_file = lang_dir / "train" / f"train.{lang}.tok"
                if train_file.exists():
                    train_files.append(str(train_file))
                else:
                    # .tok 파일이 없으면 원본 파일 시도
                    train_file_raw = lang_dir / "train" / f"train.{lang}"
                    if train_file_raw.exists():
                        train_files.append(str(train_file_raw))

            if train_files:
                shared_vocab.build_vocab_from_files(train_files, config.VOCAB_SIZE)
                shared_vocab.save_vocab(str(vocab_file))
                return shared_vocab, shared_vocab
            else:
                logger.warning("No training files found for vocabulary building")
                return None, None

    else:
        # 분리된 어휘 사전
        src_vocab_file = vocab_path / f"vocab_{config.SRC_LANG}.json"
        tgt_vocab_file = vocab_path / f"vocab_{config.TGT_LANG}.json"

        src_vocab = WMTVocabulary(
            str(src_vocab_file) if src_vocab_file.exists() else None
        )
        tgt_vocab = WMTVocabulary(
            str(tgt_vocab_file) if tgt_vocab_file.exists() else None
        )

        # 어휘 사전이 없으면 구축
        if not src_vocab_file.exists() or not tgt_vocab_file.exists():
            lang_dir = data_path / f"{config.SRC_LANG}-{config.TGT_LANG}"

            # 소스 어휘 사전
            src_train_file = lang_dir / "train" / f"train.{config.SRC_LANG}"
            if src_train_file.exists():
                src_vocab.build_vocab_from_files(
                    [str(src_train_file)], config.SRC_VOCAB_SIZE
                )
                src_vocab.save_vocab(str(src_vocab_file))

            # 타겟 어휘 사전
            tgt_train_file = lang_dir / "train" / f"train.{config.TGT_LANG}"
            if tgt_train_file.exists():
                tgt_vocab.build_vocab_from_files(
                    [str(tgt_train_file)], config.TGT_VOCAB_SIZE
                )
                tgt_vocab.save_vocab(str(tgt_vocab_file))

        return src_vocab, tgt_vocab


def create_wmt_datasets(config) -> Tuple[WMTDataset, WMTDataset, WMTDataset]:
    """WMT 데이터셋 생성"""
    logger.info("Creating WMT datasets...")

    # 어휘 사전 설정
    src_vocab, tgt_vocab = setup_wmt_vocabulary(config)

    if src_vocab is None or tgt_vocab is None:
        logger.error("Failed to setup vocabularies")
        return None, None, None

    # 데이터 경로 설정
    data_path = Path(config.DATA_PATH)
    lang_dir = data_path / f"{config.SRC_LANG}-{config.TGT_LANG}"

    datasets = {}

    # 각 분할에 대해 데이터셋 생성
    for split in ["train", "val", "test"]:
        src_file = lang_dir / split / f"{split}.{config.SRC_LANG}"
        tgt_file = lang_dir / split / f"{split}.{config.TGT_LANG}"

        # .tok 파일이 있으면 우선 사용
        src_file_tok = lang_dir / split / f"{split}.{config.SRC_LANG}.tok"
        tgt_file_tok = lang_dir / split / f"{split}.{config.TGT_LANG}.tok"

        if src_file_tok.exists() and tgt_file_tok.exists():
            src_file = src_file_tok
            tgt_file = tgt_file_tok

        if src_file.exists() and tgt_file.exists():
            dataset = WMTDataset(
                str(src_file),
                str(tgt_file),
                src_vocab,
                tgt_vocab,
                config.MAX_SEQ_LENGTH,
            )
            datasets[split] = dataset
        else:
            logger.warning(f"Data files not found for {split}: {src_file}, {tgt_file}")
            datasets[split] = None

    return datasets.get("train"), datasets.get("val"), datasets.get("test")


def create_wmt_data_loaders(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """WMT 데이터 로더 생성"""
    # 데이터셋 생성
    train_dataset, val_dataset, test_dataset = create_wmt_datasets(config)

    loaders = []

    # 훈련 데이터 로더
    if train_dataset:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, config.PAD_TOKEN),
            num_workers=0,  # multiprocessing 이슈 방지
            pin_memory=True,
        )
        logger.info(
            f"Train loader: {len(train_dataset)} samples, {len(train_loader)} batches"
        )
    else:
        train_loader = None
        logger.warning("No training data loaded")

    # 검증 데이터 로더
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, config.PAD_TOKEN),
            num_workers=0,
            pin_memory=True,
        )
        logger.info(
            f"Validation loader: {len(val_dataset)} samples, {len(val_loader)} batches"
        )
    else:
        val_loader = None
        logger.warning("No validation data loaded")

    # 테스트 데이터 로더
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, config.PAD_TOKEN),
            num_workers=0,
            pin_memory=True,
        )
        logger.info(
            f"Test loader: {len(test_dataset)} samples, {len(test_loader)} batches"
        )
    else:
        test_loader = None
        logger.warning("No test data loaded")

    return train_loader, val_loader, test_loader


def prepare_sample_data(config):
    """샘플 데이터 준비 (테스트용)"""
    logger.info("Preparing sample WMT data for testing...")

    data_path = Path(config.DATA_PATH)
    lang_dir = data_path / f"{config.SRC_LANG}-{config.TGT_LANG}"

    # 디렉토리 생성
    for split in ["train", "val", "test"]:
        split_dir = lang_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

    # 샘플 데이터 생성
    sample_en = [
        "Hello world .",
        "This is a test sentence .",
        "Machine translation is fascinating .",
        "The cat sits on the mat .",
        "I enjoy learning new languages .",
        "Natural language processing is important .",
        "Deep learning models are powerful .",
        "Transformer models have revolutionized NLP .",
        "Attention mechanisms are key to understanding .",
        "Translation quality has improved significantly .",
    ]

    sample_de = [
        "Hallo Welt .",
        "Das ist ein Testsatz .",
        "Maschinelle Übersetzung ist faszinierend .",
        "Die Katze sitzt auf der Matte .",
        "Ich lerne gerne neue Sprachen .",
        "Natürliche Sprachverarbeitung ist wichtig .",
        "Deep-Learning-Modelle sind mächtig .",
        "Transformer-Modelle haben die NLP revolutioniert .",
        "Aufmerksamkeitsmechanismen sind der Schlüssel zum Verständnis .",
        "Die Übersetzungsqualität hat sich erheblich verbessert .",
    ]

    # 데이터 크기 설정
    sizes = {"train": 5000, "val": 500, "test": 100}

    for split, size in sizes.items():
        src_file = lang_dir / split / f"{split}.{config.SRC_LANG}"
        tgt_file = lang_dir / split / f"{split}.{config.TGT_LANG}"

        with open(src_file, "w", encoding="utf-8") as f_src, open(
            tgt_file, "w", encoding="utf-8"
        ) as f_tgt:

            for i in range(size):
                idx = i % len(sample_en)
                f_src.write(sample_en[idx] + "\n")
                f_tgt.write(sample_de[idx] + "\n")

        logger.info(f"Created {split} data: {size} sentence pairs")

    logger.info("Sample data preparation complete!")
