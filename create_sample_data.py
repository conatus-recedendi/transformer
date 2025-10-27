#!/usr/bin/env python3
"""
WMT 샘플 데이터 생성 스크립트
"""

import sys
import os


from src.config import Config
from src.wmt_data_loader import prepare_sample_data
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """샘플 WMT 데이터 생성"""
    print("🔧 Creating sample WMT data...")

    # Config 로드
    config = Config()
    config.DATA_PATH = "data/"
    config.SRC_LANG = "en"
    config.TGT_LANG = "de"
    config.VOCAB_SIZE = 37000
    config.SHARED_VOCAB = True

    print(f"📊 Configuration:")
    print(f"  Data path: {config.DATA_PATH}")
    print(f"  Language pair: {config.SRC_LANG} → {config.TGT_LANG}")
    print(f"  Vocabulary size: {config.VOCAB_SIZE}")
    print(f"  Shared vocabulary: {config.SHARED_VOCAB}")

    # 샘플 데이터 생성
    prepare_sample_data(config)

    print("✅ Sample data creation complete!")
    print(f"📁 Data created in: {config.DATA_PATH}{config.SRC_LANG}-{config.TGT_LANG}/")

    # 생성된 파일 목록 출력
    from pathlib import Path

    data_path = Path(config.DATA_PATH)
    lang_dir = data_path / f"{config.SRC_LANG}-{config.TGT_LANG}"

    print("\n📋 Created files:")
    for split in ["train", "val", "test"]:
        split_dir = lang_dir / split
        if split_dir.exists():
            for file in split_dir.iterdir():
                if file.is_file():
                    size = file.stat().st_size
                    print(f"  {file}: {size:,} bytes")


if __name__ == "__main__":
    main()
