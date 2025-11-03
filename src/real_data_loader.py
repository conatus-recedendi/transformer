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
            # 특수 토큰 ID를 명확히 분리하여 설정
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
            # 특수 토큰 문자열 설정
            pad_piece="<PAD>",
            bos_piece="<BOS>",
            eos_piece="<EOS>",
            unk_piece="<UNK>",
            # user_defined_symbols에서 UNK 제외 (자동으로 정의됨)
            user_defined_symbols=[],
            # 추가 설정으로 정확한 ID 매핑 보장
            control_symbols=["<PAD>", "<BOS>", "<EOS>"],
        )

        # 임시 파일 삭제
        os.remove(combined_file)

        # 모델 로드
        self.load_model(self.model_path)

        logger.info(f"BPE model trained and saved: {self.model_path}")
        logger.info(f"Vocabulary size: {len(self)}")

        # 훈련 직후 특수 토큰 검증
        logger.info("Verifying special tokens after training:")
        for token, expected_id in self.special_tokens.items():
            actual_id = self.sp_model.piece_to_id(token)
            if actual_id == expected_id:
                logger.info(f"  ✓ {token}: {actual_id}")
            else:
                logger.error(f"  ✗ {token}: expected {expected_id}, got {actual_id}")
                raise ValueError(
                    f"BPE training failed: {token} has wrong ID {actual_id}, expected {expected_id}"
                )

    def load_model(self, model_path: str):
        """훈련된 BPE 모델 로드"""
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_path)
        self.model_path = model_path

        # 특수 토큰 ID 검증 및 수정
        self._verify_special_tokens()

        logger.info(f"BPE model loaded from {model_path}")
        logger.info(f"Vocabulary size: {len(self)}")
        logger.info(f"Special token mapping:")
        for token, expected_id in self.special_tokens.items():
            actual_id = self.sp_model.piece_to_id(token)
            logger.info(f"  {token}: expected={expected_id}, actual={actual_id}")

    def _verify_special_tokens(self):
        """특수 토큰 ID가 올바르게 설정되었는지 검증"""
        for token, expected_id in self.special_tokens.items():
            actual_id = self.sp_model.piece_to_id(token)
            if actual_id != expected_id:
                logger.warning(
                    f"Special token ID mismatch: {token} expected={expected_id}, actual={actual_id}"
                )

        # 어휘 크기 확인
        vocab_size = self.sp_model.get_piece_size()
        logger.info(f"Loaded vocabulary size: {vocab_size}")

        # 처음 몇 개 토큰 확인
        logger.info("First 10 tokens:")
        for i in range(min(10, vocab_size)):
            piece = self.sp_model.id_to_piece(i)
            logger.info(f"  ID {i}: '{piece}'")

        # 특수 토큰이 제대로 설정되었는지 다시 확인
        for token in ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]:
            token_id = self.sp_model.piece_to_id(token)
            if token_id < 0:
                logger.error(f"Invalid token ID for {token}: {token_id}")
            else:
                logger.info(f"Valid token: {token} -> ID {token_id}")

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
        apply_cleaning: bool = True,
    ):
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.vocab = vocab
        self.max_length = max_length
        self.apply_cleaning = apply_cleaning

        # 데이터 로드
        self.data_pairs = self._load_data()

        logger.info(f"Loaded {len(self.data_pairs)} sentence pairs")
        logger.info(f"  Source file: {src_file}")
        logger.info(f"  Target file: {tgt_file}")
        logger.info(f"  Data cleaning: {'Enabled' if apply_cleaning else 'Disabled'}")

    def _load_data(self) -> List[Tuple[str, str]]:
        """분리된 언어 파일들 로드 (BPE용으로 원문 텍스트 반환)"""
        data_pairs = []

        if not os.path.exists(self.src_file) or not os.path.exists(self.tgt_file):
            logger.warning(f"Data files not found: {self.src_file} or {self.tgt_file}")
            return data_pairs

        # 🔍 바이너리 모드로 정확한 라인 수 확인
        def count_binary_lines(file_path):
            with open(file_path, "rb") as f:
                return f.read().count(b"\n")

        src_line_count = count_binary_lines(self.src_file)
        tgt_line_count = count_binary_lines(self.tgt_file)

        if src_line_count != tgt_line_count:
            logger.error(f"❌ Binary line count mismatch:")
            logger.error(f"  {self.src_file}: {src_line_count:,} lines")
            logger.error(f"  {self.tgt_file}: {tgt_line_count:,} lines")
            raise ValueError(
                "Source and target files must have the same number of lines"
            )

        logger.info(f"📊 Binary line counts match: {src_line_count:,} lines each")

        # 🚨 단독 \r 문자 문제 감지 및 경고
        def check_standalone_cr(file_path):
            with open(file_path, "rb") as f:
                data = f.read()
                cr_count = data.count(b"\r")
                crlf_count = data.count(b"\r\n")
                standalone_cr = cr_count - crlf_count
                return standalone_cr

        src_standalone_cr = check_standalone_cr(self.src_file)
        tgt_standalone_cr = check_standalone_cr(self.tgt_file)

        if src_standalone_cr > 0 or tgt_standalone_cr > 0:
            logger.warning(f"⚠️  Standalone \\r characters detected:")
            logger.warning(f"  {self.src_file}: {src_standalone_cr} standalone \\r")
            logger.warning(f"  {self.tgt_file}: {tgt_standalone_cr} standalone \\r")
            logger.warning(f"  Using newlines='\\n' mode to prevent misalignment")

        # 🚨 강력한 파일 읽기 - newlines='\n'으로 단독 \r 문제 해결
        processed_pairs = 0
        skipped_pairs = 0
        raw_src_sentences = []
        raw_tgt_sentences = []

        with open(
            self.src_file, "r", encoding="utf-8", errors="replace", newline="\n"
        ) as f_src, open(
            self.tgt_file, "r", encoding="utf-8", errors="replace", newline="\n"
        ) as f_tgt:

            for line_num, (src_line, tgt_line) in enumerate(zip(f_src, f_tgt), 1):
                src_line = src_line.strip()
                tgt_line = tgt_line.strip()

                # 🚨 빈 라인 쌍은 모두 건너뛰기
                if not src_line and not tgt_line:
                    skipped_pairs += 1
                    continue
                elif not src_line or not tgt_line:
                    # 한쪽만 비어있으면 경고하고 건너뛰기
                    if skipped_pairs < 10:  # 처음 10개만 로깅
                        logger.warning(f"Line {line_num}: One-sided empty - skipping")
                    skipped_pairs += 1
                    continue

                # 길이 제한 및 빈 라인 필터링
                if (
                    len(src_line) > 0
                    and len(tgt_line) > 0
                    and len(src_line) <= self.max_length
                    and len(tgt_line) <= self.max_length
                ):
                    raw_src_sentences.append(src_line)
                    raw_tgt_sentences.append(tgt_line)
                    processed_pairs += 1
                else:
                    skipped_pairs += 1

        logger.info(f"📊 Raw data loading completed:")
        logger.info(f"  Binary line count: {src_line_count:,}")
        logger.info(f"  Raw processed pairs: {processed_pairs:,}")
        logger.info(f"  Raw skipped pairs: {skipped_pairs:,}")

        # 🧹 데이터 클리닝 적용 (설정에 따라)
        apply_cleaning = getattr(self, "apply_cleaning", True)  # 기본값: True

        if apply_cleaning:
            from src.data_loader import clean_sentence_pairs

            logger.info(f"🧹 Applying Tensor2Tensor data cleaning rules...")
            cleaned_src, cleaned_tgt = clean_sentence_pairs(
                raw_src_sentences, raw_tgt_sentences
            )
        else:
            logger.info(f"⏭️ Skipping data cleaning (disabled in config)")
            cleaned_src, cleaned_tgt = raw_src_sentences, raw_tgt_sentences

        # 최종 데이터 쌍 생성
        for src_text, tgt_text in zip(cleaned_src, cleaned_tgt):
            data_pairs.append((src_text, tgt_text))

        logger.info(f"✅ Final data loading completed:")
        logger.info(f"  Final pairs: {len(data_pairs):,}")
        logger.info(
            f"  Overall success rate: {len(data_pairs)/src_line_count*100:.1f}%"
        )

        return data_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.data_pairs[idx]

        # BPE로 토큰을 ID로 변환
        src_ids = self.vocab.encode(src_text)
        tgt_ids = self.vocab.encode(tgt_text)

        # BOS/EOS 토큰 추가
        tgt_input = (
            [self.vocab.special_tokens["<BOS>"]]
            + tgt_ids
            # + [self.vocab.special_tokens["<EOS>"]]
        )
        tgt_output = (
            # [self.vocab.special_tokens["<BOS>"]]
            tgt_ids
            + [self.vocab.special_tokens["<EOS>"]]
        )

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
        logger.info(f"Found existing BPE model: {bpe_model_path}")
        vocab.load_model(bpe_model_path)

        # 특수 토큰이 올바르게 설정되었는지 확인
        pad_id = vocab.sp_model.piece_to_id("<PAD>")
        bos_id = vocab.sp_model.piece_to_id("<BOS>")
        eos_id = vocab.sp_model.piece_to_id("<EOS>")
        unk_id = vocab.sp_model.piece_to_id("<UNK>")

        # 특수 토큰 ID가 잘못되었으면 재훈련
        if (
            pad_id != 0
            or bos_id != 1
            or eos_id != 2
            or unk_id != 3
            or any(id < 0 for id in [pad_id, bos_id, eos_id, unk_id])
        ):
            logger.warning(
                "Special tokens are incorrectly configured. Retraining BPE model..."
            )
            # 기존 모델 파일들 삭제
            for ext in [".model", ".vocab"]:
                old_file = f"{bpe_model_prefix}{ext}"
                if os.path.exists(old_file):
                    os.remove(old_file)
                    logger.info(f"Removed corrupted BPE file: {old_file}")

            # 새로 훈련
            vocab.train_bpe_model(vocab_files, config.VOCAB_SIZE, str(bpe_model_prefix))
        else:
            logger.info("BPE model special tokens are correctly configured.")
    else:
        logger.info("Training new BPE model...")
        vocab.train_bpe_model(vocab_files, config.VOCAB_SIZE, str(bpe_model_prefix))

    # 데이터 클리닝 설정 가져오기
    apply_cleaning = getattr(config, "APPLY_DATA_CLEANING", True)
    logger.info(f"Data cleaning: {'Enabled' if apply_cleaning else 'Disabled'}")

    # 데이터셋 생성
    train_dataset = RealWMTDataset(
        str(train_src_file),
        str(train_tgt_file),
        vocab,
        config.MAX_SEQ_LENGTH,
        apply_cleaning,
    )
    val_dataset = RealWMTDataset(
        str(valid_src_file),
        str(valid_tgt_file),
        vocab,
        config.MAX_SEQ_LENGTH,
        apply_cleaning,
    )
    test_dataset = RealWMTDataset(
        str(test_src_file),
        str(test_tgt_file),
        vocab,
        config.MAX_SEQ_LENGTH,
        apply_cleaning,
    )

    logger.info(f"Dataset sizes:")
    logger.info(f"  Train: {len(train_dataset):,} samples")
    logger.info(f"  Valid: {len(val_dataset):,} samples")
    logger.info(f"  Test: {len(test_dataset):,} samples")

    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
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

    return train_loader, val_loader, test_loader, vocab
