#!/usr/bin/env python3
"""
PyTorch Transformer 구현 테스트 스크립트
"""

import torch
import torch.nn as nn
from src.pytorch_transformer import PyTorchTransformerWrapper


def test_pytorch_transformer():
    """PyTorch Transformer 래퍼 테스트"""
    print("🔬 Testing PyTorch Transformer Wrapper")
    print("=" * 50)

    # 모델 설정
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 256
    num_heads = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    d_ff = 1024
    max_seq_length = 100
    dropout = 0.1
    pad_token_id = 0

    # 모델 생성
    model = PyTorchTransformerWrapper(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
        dropout=dropout,
        pad_token_id=pad_token_id,
        device="cpu",
    )

    print(f"✅ Model created successfully")

    # 파라미터 수 계산
    from src.pytorch_transformer import get_parameter_count

    total_params, trainable_params = get_parameter_count(model)
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,}")

    # 테스트 데이터 생성
    batch_size = 4
    src_seq_len = 20
    tgt_seq_len = 15

    # 랜덤 토큰 ID 생성 (패딩 제외)
    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))

    # 일부 토큰을 패딩으로 설정
    src[:, -5:] = pad_token_id  # 마지막 5개 토큰을 패딩으로
    tgt[:, -3:] = pad_token_id  # 마지막 3개 토큰을 패딩으로

    print(f"\n🧪 Test input shapes:")
    print(f"  Source: {src.shape}")
    print(f"  Target: {tgt.shape}")

    # Forward pass 테스트
    model.eval()
    with torch.no_grad():
        try:
            logits = model(src, tgt)
            print(f"✅ Forward pass successful")
            print(f"  Output shape: {logits.shape}")
            print(f"  Expected shape: ({batch_size}, {tgt_seq_len}, {tgt_vocab_size})")

            # 출력 검증
            assert logits.shape == (
                batch_size,
                tgt_seq_len,
                tgt_vocab_size,
            ), f"Wrong output shape: {logits.shape}"

            # 손실 계산 테스트
            criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

            # 타겟에서 다음 토큰 예측을 위한 shift
            tgt_input = tgt[:, :-1]  # BOS ~ (n-1)
            tgt_output = tgt[:, 1:]  # 1 ~ EOS

            # 다시 forward pass
            logits = model(src, tgt_input)

            # 손실 계산
            loss = criterion(logits.reshape(-1, tgt_vocab_size), tgt_output.reshape(-1))
            print(f"✅ Loss calculation successful: {loss.item():.4f}")

            # 인코더만 테스트
            memory = model.encode(src)
            print(f"✅ Encoder test successful")
            print(f"  Memory shape: {memory.shape}")

            # 디코더만 테스트
            decoder_logits = model.decode(tgt_input, memory)
            print(f"✅ Decoder test successful")
            print(f"  Decoder output shape: {decoder_logits.shape}")

            # 출력이 동일한지 확인
            torch.testing.assert_close(logits, decoder_logits, rtol=1e-5, atol=1e-5)
            print(f"✅ Encoder-Decoder consistency verified")

        except Exception as e:
            print(f"❌ Error during forward pass: {e}")
            import traceback

            traceback.print_exc()
            return False

    print(f"\n🎉 All tests passed!")
    return True


if __name__ == "__main__":
    test_pytorch_transformer()
