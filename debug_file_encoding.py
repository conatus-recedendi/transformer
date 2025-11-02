#!/usr/bin/env python3
"""
파일 인코딩 및 특수 문자 디버깅 스크립트
"""

import os
import subprocess

def main():
    en_file = "data/wmt14_en_de/train.14.en"
    de_file = "data/wmt14_en_de/train.14.de"
    
    print("🔍 File Encoding and Special Character Analysis")
    print("=" * 70)
    
    def analyze_file(file_path, name):
        print(f"\n📁 {name} File: {file_path}")
        
        # 1. 파일 크기
        size = os.path.getsize(file_path)
        print(f"  File size: {size:,} bytes")
        
        # 2. 파일 처음 몇 바이트로 인코딩 추정
        with open(file_path, "rb") as f:
            first_bytes = f.read(1000)
            print(f"  First 50 bytes: {first_bytes[:50]}")
            
            # BOM 확인
            if first_bytes.startswith(b'\xef\xbb\xbf'):
                print(f"  ⚠️  UTF-8 BOM detected!")
            elif first_bytes.startswith(b'\xff\xfe'):
                print(f"  ⚠️  UTF-16 LE BOM detected!")
            elif first_bytes.startswith(b'\xfe\xff'):
                print(f"  ⚠️  UTF-16 BE BOM detected!")
            else:
                print(f"  ✅ No BOM detected")
        
        # 3. 다양한 방법으로 라인 카운팅
        print(f"  Line counting comparison:")
        
        # Python 기본 방법
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                python_count = sum(1 for _ in f)
            print(f"    Python (UTF-8): {python_count:,}")
        except Exception as e:
            print(f"    Python (UTF-8): ERROR - {e}")
        
        # Python 다른 인코딩들
        for enc in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(file_path, "r", encoding=enc, errors='ignore') as f:
                    count = sum(1 for _ in f)
                print(f"    Python ({enc}): {count:,}")
            except Exception as e:
                print(f"    Python ({enc}): ERROR - {e}")
        
        # wc 명령어
        try:
            result = subprocess.run(['wc', '-l', file_path], 
                                  capture_output=True, text=True)
            wc_count = int(result.stdout.split()[0])
            print(f"    Shell wc -l: {wc_count:,}")
        except Exception as e:
            print(f"    Shell wc -l: ERROR - {e}")
        
        # 5. 마지막 몇 바이트 확인 (개행 문자)
        with open(file_path, "rb") as f:
            f.seek(-100, os.SEEK_END)
            tail_bytes = f.read()
            print(f"  Last 20 bytes: {tail_bytes[-20:]}")
            print(f"  Ends with \\n: {tail_bytes.endswith(b'\\n')}")
            print(f"  Ends with \\r\\n: {tail_bytes.endswith(b'\\r\\n')}")
        
        # 6. 특수 문자 통계
        null_count = 0
        cr_count = 0
        lf_count = 0
        crlf_count = 0
        
        with open(file_path, "rb") as f:
            data = f.read()
            null_count = data.count(b'\x00')
            cr_count = data.count(b'\r')
            lf_count = data.count(b'\n')
            crlf_count = data.count(b'\r\n')
        
        print(f"  Special character counts:")
        print(f"    NULL bytes (\\x00): {null_count}")
        print(f"    Carriage returns (\\r): {cr_count}")
        print(f"    Line feeds (\\n): {lf_count}")
        print(f"    CRLF sequences (\\r\\n): {crlf_count}")
        
        return {
            'size': size,
            'python_utf8': python_count if 'python_count' in locals() else 0,
            'wc_count': wc_count if 'wc_count' in locals() else 0,
            'null_count': null_count,
            'cr_count': cr_count,
            'lf_count': lf_count,
            'crlf_count': crlf_count
        }
    
    # 두 파일 분석
    en_stats = analyze_file(en_file, "ENGLISH")
    de_stats = analyze_file(de_file, "GERMAN")
    
    # 비교 결과
    print(f"\n🔄 Comparison Summary:")
    print(f"=" * 40)
    print(f"File size difference: {abs(en_stats['size'] - de_stats['size']):,} bytes")
    print(f"Python line count diff: {abs(en_stats['python_utf8'] - de_stats['python_utf8']):,}")
    print(f"Shell wc count diff: {abs(en_stats['wc_count'] - de_stats['wc_count']):,}")
    
    print(f"\nSpecial character differences:")
    print(f"  NULL bytes: EN={en_stats['null_count']}, DE={de_stats['null_count']}")
    print(f"  CR chars: EN={en_stats['cr_count']}, DE={de_stats['cr_count']}")
    print(f"  LF chars: EN={en_stats['lf_count']}, DE={de_stats['lf_count']}")
    print(f"  CRLF seqs: EN={en_stats['crlf_count']}, DE={de_stats['crlf_count']}")
    
    # 라인 수 차이 원인 추정
    print(f"\n🔍 Possible causes:")
    if abs(en_stats['python_utf8'] - de_stats['python_utf8']) > 0:
        print(f"  ❌ Python reads different line counts")
        if en_stats['cr_count'] != de_stats['cr_count']:
            print(f"    → Different carriage return counts detected")
        if en_stats['null_count'] != de_stats['null_count']:
            print(f"    → Different NULL byte counts detected")
    
    if abs(en_stats['wc_count'] - de_stats['wc_count']) == 0:
        print(f"  ✅ Shell wc counts match - encoding issue likely")
    else:
        print(f"  ❌ Even shell wc counts differ - file corruption possible")

if __name__ == "__main__":
    main()
