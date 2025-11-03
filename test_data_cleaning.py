#!/usr/bin/env python3
"""
데이터 클리닝 기능 테스트 스크립트
"""

from src.data_loader import clean_sentence_pairs


def test_data_cleaning():
    """데이터 클리닝 함수 테스트"""
    print("🧹 Testing data cleaning functions")
    print("=" * 50)

    # 테스트 데이터 - 일부는 필터링되고 일부는 통과해야 함
    test_src = [
        "This is a good sentence.",  # 통과해야 함
        "BREAKING NEWS UPDATE",  # 필터링 (대문자만)
        "Copyright © 2023 Test",  # 필터링 (저작권)
        "Hello world. How are you?",  # 분할되어야 함
        "test: this should be filtered",  # 필터링 (라벨 콜론)
        "Another valid sentence!",  # 통과해야 함
        "www.example.com",  # 필터링 (좋지 않은 시작)
        "Good morning. Nice to meet you.",  # 분할되어야 함
    ]

    test_tgt = [
        "Dies ist ein guter Satz.",
        "EILMELDUNG UPDATE",
        "Urheberrecht © 2023 Test",
        "Hallo Welt. Wie geht es dir?",
        "test: das sollte gefiltert werden",
        "Ein weiterer gültiger Satz!",
        "www.beispiel.de",
        "Guten Morgen. Freut mich, Sie kennenzulernen.",
    ]

    print(f"Original pairs: {len(test_src)}")
    print("\nOriginal sentences:")
    for i, (src, tgt) in enumerate(zip(test_src, test_tgt)):
        print(f"  {i+1}. EN: {src}")
        print(f"     DE: {tgt}")

    # 클리닝 적용
    cleaned_src, cleaned_tgt = clean_sentence_pairs(test_src, test_tgt)

    print(f"\nCleaned sentences:")
    for i, (src, tgt) in enumerate(zip(cleaned_src, cleaned_tgt)):
        print(f"  {i+1}. EN: {src}")
        print(f"     DE: {tgt}")

    print(f"\n✅ Test completed!")
    print(f"Retention rate: {len(cleaned_src)/len(test_src)*100:.1f}%")


if __name__ == "__main__":
    test_data_cleaning()
