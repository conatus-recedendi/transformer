#!/usr/bin/env python3
"""
실제 형식에 맞는 샘플 WMT 데이터 생성 스크립트
data/wmt14_en_de/train.txt, valid.txt, test.txt 형식으로 생성
"""

import os
from pathlib import Path


def create_sample_wmt_data():
    """샘플 WMT 데이터 생성 (탭 분리 형식)"""

    # 데이터 디렉토리 생성
    data_dir = Path("data/wmt14_en_de")
    data_dir.mkdir(parents=True, exist_ok=True)

    # 샘플 문장 쌍 (영어 -> 독일어)
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
    ]

    # 파일별 크기 설정
    file_sizes = {"train.txt": 5000, "valid.txt": 500, "test.txt": 200}

    for filename, size in file_sizes.items():
        file_path = data_dir / filename

        print(f"Creating {filename} with {size} sentence pairs...")

        with open(file_path, "w", encoding="utf-8") as f:
            for i in range(size):
                # 샘플 쌍을 순환하면서 사용
                pair_idx = i % len(sample_pairs)
                en_sentence, de_sentence = sample_pairs[pair_idx]

                # 약간의 변형을 위해 문장 번호 추가 (훈련 데이터에만)
                if filename == "train.txt" and i % 100 == 0:
                    en_sentence = f"Example {i//100 + 1} : {en_sentence}"
                    de_sentence = f"Beispiel {i//100 + 1} : {de_sentence}"

                # 탭으로 분리하여 저장
                f.write(f"{en_sentence}\t{de_sentence}\n")

        print(f"✅ Created {file_path} ({file_path.stat().st_size} bytes)")

    print(f"\n🎉 Sample WMT data created successfully!")
    print(f"📁 Location: {data_dir.absolute()}")
    print(f"📋 Files created:")
    for filename in file_sizes.keys():
        file_path = data_dir / filename
        print(f"   {filename}: {file_path.stat().st_size:,} bytes")

    # 첫 번째 파일의 처음 5줄 표시
    print(f"\n📖 Sample from train.txt:")
    with open(data_dir / "train.txt", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                print(f"   {i+1}: EN: {parts[0]}")
                print(f"      DE: {parts[1]}")


if __name__ == "__main__":
    create_sample_wmt_data()
