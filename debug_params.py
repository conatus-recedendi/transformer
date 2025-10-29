#!/usr/bin/env python3
"""
실제 모델 파라미터 확인을 위한 디버깅 스크립트
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.transformer import Transformer
from src.config import Config
from src.encoder import EncoderLayer
from src.multi_head_attention import MultiheadAttention


def count_parameters(model):
    """모델의 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyze_encoder_layer():
    """단일 Encoder 레이어 파라미터 분석"""
    print("=== Single Encoder Layer Analysis ===")

    # 설정값
    d_model = 512
    num_heads = 8
    d_ff = 2048
    kdim = 64
    vdim = 64

    # EncoderLayer 생성
    encoder_layer = EncoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.1,
        kdim=kdim,
        vdim=vdim,
    )

    print(f"총 파라미터: {count_parameters(encoder_layer):,}")

    # 컴포넌트별 분석
    print(f"Self-Attention 파라미터: {count_parameters(encoder_layer.self_attn):,}")
    print(f"FFN 파라미터: {count_parameters(encoder_layer.ffn):,}")
    print(f"LayerNorm1 파라미터: {count_parameters(encoder_layer.norm1):,}")
    print(f"LayerNorm2 파라미터: {count_parameters(encoder_layer.norm2):,}")

    # MultiHeadAttention 내부 분석
    print("\n--- MultiHeadAttention 내부 ---")
    attn = encoder_layer.self_attn
    print(f"Q linear: {count_parameters(attn.q_linear):,}")
    print(f"K linear: {count_parameters(attn.k_linear):,}")
    print(f"V linear: {count_parameters(attn.v_linear):,}")
    print(f"Out linear: {count_parameters(attn.out_linear):,}")

    # FFN 내부 분석
    print("\n--- FFN 내부 ---")
    print(f"Linear1 (512->2048): {count_parameters(encoder_layer.ffn[0]):,}")
    print(f"Linear2 (2048->512): {count_parameters(encoder_layer.ffn[3]):,}")


def analyze_full_encoder():
    """전체 Encoder 파라미터 분석"""
    print("\n=== Full Encoder Analysis ===")

    config = Config()

    # Transformer 모델 생성
    model = Transformer(
        src_vocab_size=config.VOCAB_SIZE,
        tgt_vocab_size=config.VOCAB_SIZE,
        d_model=config.MODEL_DIM,
        num_heads=config.NUM_HEADS,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        d_ff=config.FFN_DIM,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dropout=config.DROPOUT,
        kdim=config.KDIM,
        vdim=config.VDIM,
    )

    print(f"전체 모델 파라미터: {count_parameters(model):,}")
    print(f"Encoder 파라미터: {count_parameters(model.encoder):,}")
    print(f"Decoder 파라미터: {count_parameters(model.decoder):,}")
    print(f"공유 Embedding 파라미터: {count_parameters(model.embedding):,}")
    print(f"Output Projection 파라미터: {count_parameters(model.output_projection):,}")

    # 6레이어니까 레이어당 평균
    encoder_per_layer = count_parameters(model.encoder) // config.NUM_ENCODER_LAYERS
    print(f"Encoder 레이어당 평균: {encoder_per_layer:,}")


if __name__ == "__main__":
    analyze_encoder_layer()
    analyze_full_encoder()
