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
import sentencepiece as spm

logger = logging.getLogger(__name__)


class BPEVocabulary:
    """BPE(Byte Pair Encoding) 기반 어휘 사전"""

    def __init__(self):
        self.sp_model = None
        self.vocab_size = 30000
        self.special_tokens = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.model_path = None

    def train_bpe_model(
        self,
        file_paths: List[str],
        vocab_size: int = 30000,
        model_prefix: str = "bpe_model",
    ):
        """BPE 모델 훈련"""
        logger.info(f"Training BPE model from {len(file_paths)} files...")

        self.vocab_size = vocab_size
        self.model_path = f"{model_prefix}.model"

        # 모든 파일을 하나로 합치기
        combined_file = f"{model_prefix}_combined.txt"
        total_lines = 0

        with open(combined_file, "w", encoding="utf-8") as outf:
            for file_path in file_paths:
                if os.path.exists(file_path):
                    logger.info(f"Processing {file_path}...")
                    with open(file_path, "r", encoding="utf-8") as inf:
                        for line in inf:
                            line = line.strip()
                            if line:
                                outf.write(line + "\n")
                                total_lines += 1

                            if total_lines % 1_000_000 == 0:
                                logger.info(f"  Processed {total_lines} lines...")

        logger.info(f"Total lines for BPE training: {total_lines}")

        # BPE 모델 훈련
        spm.SentencePieceTrainer.train(
            input=combined_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=0.995,
            model_type="bpe",
            pad_id=self.special_tokens["<PAD>"],
            bos_id=self.special_tokens["<BOS>"],
            eos_id=self.special_tokens["<EOS>"],
            unk_id=self.special_tokens["<UNK>"],
            pad_piece="<PAD>",
            bos_piece="<BOS>",
            eos_piece="<EOS>",
            unk_piece="<UNK>",
            # <UNK>는 기본 제어 심볼이므로 user_defined_symbols에서 제외
            user_defined_symbols=["<PAD>", "<BOS>", "<EOS>"],
        )

        # 임시 파일 삭제
        os.remove(combined_file)

        # 모델 로드
        self.load_model(self.model_path)

        logger.info(f"BPE model trained and saved: {self.model_path}")
        logger.info(f"Vocabulary size: {len(self)}")

    def load_model(self, model_path: str):
        """훈련된 BPE 모델 로드"""
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_path)
        self.model_path = model_path
        logger.info(f"BPE model loaded from {model_path}")

    def encode(self, text: str) -> List[int]:
        """텍스트를 BPE 토큰 ID로 변환"""
        if self.sp_model is None:
            raise ValueError(
                "BPE model not loaded. Call train_bpe_model() or load_model() first."
            )

        if isinstance(text, list):
            # 토큰 리스트가 입력된 경우 공백으로 결합
            text = " ".join(text)

        return self.sp_model.encode_as_ids(text)

    def decode(self, ids: List[int]) -> str:
        """BPE 토큰 ID를 텍스트로 변환"""
        if self.sp_model is None:
            raise ValueError("BPE model not loaded.")

        return self.sp_model.decode_ids(ids)

    def encode_as_pieces(self, text: str) -> List[str]:
        """텍스트를 BPE 토큰 조각으로 변환"""
        if self.sp_model is None:
            raise ValueError("BPE model not loaded.")

        if isinstance(text, list):
            text = " ".join(text)

        return self.sp_model.encode_as_pieces(text)

    def __len__(self):
        if self.sp_model is None:
            return self.vocab_size
        return self.sp_model.get_piece_size()


class RealWMTDataset(Dataset):
    """실제 WMT 데이터셋 클래스 (분리된 언어 파일 형식: train.en, train.de)"""

    def __init__(
        self,
        src_file: str,
        tgt_file: str,
        vocab: BPEVocabulary,
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

    def _load_data(self) -> List[Tuple[str, str]]:
        """분리된 언어 파일들 로드 (BPE용으로 원문 텍스트 반환)"""
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

                src_tokens = src_line.strip()
                tgt_tokens = tgt_line.strip()

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
        src_text, tgt_text = self.data_pairs[idx]

        # BPE로 토큰을 ID로 변환
        src_ids = self.vocab.encode(src_text)
        tgt_ids = self.vocab.encode(tgt_text)

        # BOS/EOS 토큰 추가
        tgt_input = [self.vocab.special_tokens["<BOS>"]] + tgt_ids
        tgt_output = tgt_ids + [self.vocab.special_tokens["<EOS>"]]

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
    # train_src_file = dataset_path / f"train.{src_lang}"
    # train_tgt_file = dataset_path / f"train.{tgt_lang}"
    # valid_src_file = dataset_path / f"valid.{src_lang}"
    # valid_tgt_file = dataset_path / f"valid.{tgt_lang}"
    # test_src_file = dataset_path / f"test.{src_lang}"
    # test_tgt_file = dataset_path / f"test.{tgt_lang}"
    train_src_file = dataset_path / f"train.14.{src_lang}"
    train_tgt_file = dataset_path / f"train.14.{tgt_lang}"
    valid_src_file = dataset_path / f"test.14.{src_lang}"
    valid_tgt_file = dataset_path / f"test.14.{tgt_lang}"
    test_src_file = dataset_path / f"test.14.{src_lang}"
    test_tgt_file = dataset_path / f"test.14.{tgt_lang}"

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
    vocab = BPEVocabulary()
    vocab_files = [str(train_src_file), str(train_tgt_file)]

    # BPE 모델 경로 설정
    bpe_model_prefix = dataset_path / "bpe_model"
    bpe_model_path = f"{bpe_model_prefix}.model"

    # BPE 모델이 이미 존재하는지 확인
    if os.path.exists(bpe_model_path):
        logger.info(f"Loading existing BPE model: {bpe_model_path}")
        vocab.load_model(bpe_model_path)
    else:
        logger.info("Training new BPE model...")
        vocab.train_bpe_model(vocab_files, config.VOCAB_SIZE, str(bpe_model_prefix))

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
