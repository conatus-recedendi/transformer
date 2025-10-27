#!/usr/bin/env python3
"""
분리된 언어 파일 형식으로 샘플 WMT 데이터 생성
data/wmt14_en_de/train.en, train.de, valid.en, valid.de, test.en, test.de
"""

import os
from pathlib import Path


def create_separated_language_data():
    """분리된 언어 파일 형식으로 샘플 데이터 생성"""

    # 데이터 디렉토리 생성
    data_dir = Path("data/wmt14_en_de")
    data_dir.mkdir(parents=True, exist_ok=True)

    # 샘플 문장 쌍 (영어 → 독일어)
    sample_pairs = [
        ("Hello world .", "Hallo Welt ."),
        ("This is a test sentence .", "Das ist ein Testsatz ."),
        (
            "Machine translation is fascinating .",
            "Maschinelle Übersetzung ist faszinierend .",
        ),
        ("The cat sits on the mat .", "Die Katze sitzt auf der Matte ."),
        ("I enjoy learning new languages .", "Ich lerne gerne neue Sprachen ."),
        (
            "Natural language processing is important .",
            "Natürliche Sprachverarbeitung ist wichtig .",
        ),
        ("Deep learning models are powerful .", "Deep-Learning-Modelle sind mächtig ."),
        (
            "Transformer models have revolutionized NLP .",
            "Transformer-Modelle haben die NLP revolutioniert .",
        ),
        (
            "Attention mechanisms are key to understanding .",
            "Aufmerksamkeitsmechanismen sind der Schlüssel zum Verständnis .",
        ),
        (
            "Translation quality has improved significantly .",
            "Die Übersetzungsqualität hat sich erheblich verbessert .",
        ),
        (
            "Artificial intelligence is advancing rapidly .",
            "Künstliche Intelligenz entwickelt sich schnell .",
        ),
        (
            "We are building a neural machine translation system .",
            "Wir bauen ein neuronales maschinelles Übersetzungssystem .",
        ),
        ("The weather is beautiful today .", "Das Wetter ist heute schön ."),
        (
            "I like to drink coffee in the morning .",
            "Ich trinke gerne morgens Kaffee .",
        ),
        (
            "Programming is both challenging and rewarding .",
            "Programmieren ist sowohl herausfordernd als auch lohnend .",
        ),
        (
            "Data science combines statistics and computer science .",
            "Data Science kombiniert Statistik und Informatik .",
        ),
        (
            "Machine learning requires large amounts of data .",
            "Maschinelles Lernen erfordert große Datenmengen .",
        ),
        ("The book is on the table .", "Das Buch liegt auf dem Tisch ."),
        (
            "Students are studying in the library .",
            "Studenten lernen in der Bibliothek .",
        ),
        (
            "Technology is changing our daily lives .",
            "Technologie verändert unser tägliches Leben .",
        ),
        (
            "Scientists are working on important research .",
            "Wissenschaftler arbeiten an wichtiger Forschung .",
        ),
        ("The sun is shining brightly today .", "Die Sonne scheint heute hell ."),
        ("Children are playing in the park .", "Kinder spielen im Park ."),
        ("Music brings people together .", "Musik bringt Menschen zusammen ."),
        ("Education is the key to success .", "Bildung ist der Schlüssel zum Erfolg ."),
    ]

    # 파일별 크기 설정
    file_sizes = {"train": 5000, "valid": 500, "test": 200}

    print("Creating separated language files...")

    for split, size in file_sizes.items():
        # 영어 파일과 독일어 파일 경로
        en_file = data_dir / f"{split}.en"
        de_file = data_dir / f"{split}.de"

        print(f"Creating {split} files with {size} sentence pairs...")

        with open(en_file, "w", encoding="utf-8") as f_en, open(
            de_file, "w", encoding="utf-8"
        ) as f_de:

            for i in range(size):
                # 샘플 쌍을 순환하면서 사용
                pair_idx = i % len(sample_pairs)
                en_sentence, de_sentence = sample_pairs[pair_idx]

                # 약간의 변형을 위해 문장 번호 추가 (훈련 데이터에만)
                if split == "train" and i % 100 == 0:
                    en_sentence = f"Example {i//100 + 1} : {en_sentence}"
                    de_sentence = f"Beispiel {i//100 + 1} : {de_sentence}"

                # 각 언어별로 분리된 파일에 저장
                f_en.write(f"{en_sentence}\n")
                f_de.write(f"{de_sentence}\n")

        print(f"✅ Created {en_file.name} ({en_file.stat().st_size} bytes)")
        print(f"✅ Created {de_file.name} ({de_file.stat().st_size} bytes)")

    print(f"\n🎉 Separated language files created successfully!")
    print(f"📁 Location: {data_dir.absolute()}")
    print(f"📋 Files created:")

    for split in file_sizes.keys():
        en_file = data_dir / f"{split}.en"
        de_file = data_dir / f"{split}.de"
        print(f"   {en_file.name}: {en_file.stat().st_size:,} bytes")
        print(f"   {de_file.name}: {de_file.stat().st_size:,} bytes")

    # 첫 번째 파일의 처음 5줄 표시
    print(f"\n📖 Sample from train files:")
    en_file = data_dir / "train.en"
    de_file = data_dir / "train.de"

    with open(en_file, "r", encoding="utf-8") as f_en, open(
        de_file, "r", encoding="utf-8"
    ) as f_de:

        for i, (en_line, de_line) in enumerate(zip(f_en, f_de)):
            if i >= 5:
                break
            print(f"   {i+1}: EN: {en_line.strip()}")
            print(f"      DE: {de_line.strip()}")


if __name__ == "__main__":
    create_separated_language_data()
