"""
PyTorch의 내장 Transformer를 래핑한 seq2seq 모델
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """위치 인코딩 (PyTorch Transformer용)"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [seq_len, batch_size, d_model] or [batch_size, seq_len, d_model]
        """
        if x.dim() == 3 and x.size(1) == self.pe.size(1):  # [seq_len, batch_size, d_model]
            return x + self.pe[:x.size(0), :]
        elif x.dim() == 3:  # [batch_size, seq_len, d_model]
            return x + self.pe[:x.size(1), :].transpose(0, 1)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")


class PyTorchTransformerWrapper(nn.Module):
    """PyTorch의 nn.Transformer를 seq2seq 태스크에 맞게 래핑"""
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        device: str = "cpu"
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.device_name = device
        
        # 임베딩 레이어
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_token_id)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_token_id)
        
        # 위치 인코딩
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # PyTorch Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=False  # [seq_len, batch_size, d_model] 형식 사용
        )
        
        # 출력 프로젝션
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # 임베딩 스케일링
        self.embedding_scale = math.sqrt(d_model)
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """디코더용 마스크 생성 (causal mask)"""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
    
    def _create_padding_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """패딩 마스크 생성"""
        # tokens: [batch_size, seq_len]
        # 반환: [batch_size, seq_len] (True = 패딩)
        return tokens == self.pad_token_id
    
    def forward(
        self, 
        src: torch.Tensor, 
        tgt: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            src: [batch_size, src_seq_len] - 소스 토큰 ID
            tgt: [batch_size, tgt_seq_len] - 타겟 토큰 ID (BOS 포함)
            
        Returns:
            logits: [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        batch_size, src_seq_len = src.shape
        _, tgt_seq_len = tgt.shape
        
        # 패딩 마스크 생성
        if src_key_padding_mask is None:
            src_key_padding_mask = self._create_padding_mask(src)  # [batch_size, src_seq_len]
        
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = self._create_padding_mask(tgt)  # [batch_size, tgt_seq_len]
        
        if memory_key_padding_mask is None:
            memory_key_padding_mask = src_key_padding_mask
        
        # 타겟 마스크 (causal mask)
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len).to(src.device)
        
        # 임베딩 + 위치 인코딩
        src_emb = self.src_embedding(src) * self.embedding_scale  # [batch_size, src_seq_len, d_model]
        tgt_emb = self.tgt_embedding(tgt) * self.embedding_scale  # [batch_size, tgt_seq_len, d_model]
        
        # PyTorch Transformer는 [seq_len, batch_size, d_model] 형식 요구
        src_emb = src_emb.transpose(0, 1)  # [src_seq_len, batch_size, d_model]
        tgt_emb = tgt_emb.transpose(0, 1)  # [tgt_seq_len, batch_size, d_model]
        
        # 위치 인코딩 적용
        src_emb = self.pos_encoding(src_emb)
        tgt_emb = self.pos_encoding(tgt_emb)
        
        # Transformer forward
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )  # [tgt_seq_len, batch_size, d_model]
        
        # [batch_size, tgt_seq_len, d_model]로 변환
        output = output.transpose(0, 1)
        
        # 출력 프로젝션
        logits = self.output_projection(output)  # [batch_size, tgt_seq_len, tgt_vocab_size]
        
        return logits
    
    def encode(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """인코더만 실행"""
        if src_key_padding_mask is None:
            src_key_padding_mask = self._create_padding_mask(src)
        
        src_emb = self.src_embedding(src) * self.embedding_scale
        src_emb = src_emb.transpose(0, 1)  # [src_seq_len, batch_size, d_model]
        src_emb = self.pos_encoding(src_emb)
        
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        return memory  # [src_seq_len, batch_size, d_model]
    
    def decode(
        self, 
        tgt: torch.Tensor, 
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """디코더만 실행"""
        _, tgt_seq_len = tgt.shape
        
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = self._create_padding_mask(tgt)
        
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)
        
        tgt_emb = self.tgt_embedding(tgt) * self.embedding_scale
        tgt_emb = tgt_emb.transpose(0, 1)  # [tgt_seq_len, batch_size, d_model]
        tgt_emb = self.pos_encoding(tgt_emb)
        
        output = self.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )  # [tgt_seq_len, batch_size, d_model]
        
        output = output.transpose(0, 1)  # [batch_size, tgt_seq_len, d_model]
        logits = self.output_projection(output)
        
        return logits


def get_parameter_count(model: nn.Module) -> Tuple[int, int]:
    """모델의 파라미터 수 계산"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
