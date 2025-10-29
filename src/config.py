"""
Configuration file for Transformer model training
Matches the JSON configuration file structure
"""


class Config:
    """Base configuration class that matches JSON config structure"""

    def __init__(self, **kwargs):
        """Initialize with default values and update with custom parameters"""
        # Experiment metadata
        self.EXPERIMENT_NAME = "transformer_base"
        self.DESCRIPTION = "Base Transformer model configuration"

        # Model parameters (matching model section in JSON)
        self.MODEL_DIM = 512
        self.NUM_HEADS = 8
        self.NUM_ENCODER_LAYERS = 6
        self.NUM_DECODER_LAYERS = 6
        self.FFN_DIM = 2048
        self.DROPOUT = 0.1
        self.MAX_SEQ_LENGTH = 512
        self.KDIM = 64  # Key dimension for attention
        self.VDIM = 64  # Value dimension for attention

        # Training parameters (matching training section in JSON)
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 1e-4
        self.WARMUP_STEPS = 4000
        self.MAX_EPOCHS = 100
        self.GRADIENT_CLIP = 1.0
        self.TOKENS_PER_BATCH = 25000
        self.MAX_TOKENS_PER_BATCH = 25000
        self.ACCUMULATE_GRAD_BATCHES = 1

        # Data parameters (matching data section in JSON)
        self.DATASET = "wmt14_en_de"
        self.VOCAB_SIZE = 37000
        self.SRC_VOCAB_SIZE = 37000  # For separate source/target vocabularies
        self.TGT_VOCAB_SIZE = 37000
        self.SHARED_VOCAB = True
        self.ENCODING = "bpe"
        self.APPROXIMATE_SENTENCE_PAIRS = 4500000
        self.TRAIN_PAIRS = 4500000
        self.VAL_PAIRS = 500000
        self.PAD_TOKEN = 0
        self.BOS_TOKEN = 1
        self.EOS_TOKEN = 2
        self.UNK_TOKEN = 3
        self.SRC_LANG = "en"
        self.TGT_LANG = "de"

        # Evaluation parameters (matching evaluation section in JSON)
        self.EVAL_EVERY = 500
        self.SAVE_EVERY = 1000
        self.N_CHECKPOINTS = (
            5  # Number of checkpoints to average (5 for base, 20 for big model)
        )
        self.BEAM_SIZE = 4  # Beam size for beam search
        self.LENGTH_PENALTY = 0.6  # Length penalty for beam search

        # File paths (matching paths section in JSON)
        self.DATA_PATH = "data/"
        self.MODEL_SAVE_PATH = "checkpoints/base/"
        self.LOG_PATH = "logs/base/"

        # Device (not in JSON but needed for runtime)
        self.DEVICE = "cuda"  # or "cpu", "mps", "auto"

        # Update with any provided kwargs
        self.update(**kwargs)

        # 초기화 후 적절한 배치 크기 계산 및 표시
        self._show_optimal_batch_size_recommendation()

    def update(self, **kwargs):
        """Update config with custom parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Allow new attributes for flexibility
                setattr(self, key, value)

    def to_dict(self):
        """Convert config to dictionary (flat structure for backward compatibility)"""
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }

    def to_json_structure(self):
        """Convert config to hierarchical JSON-like structure"""
        return {
            "experiment_name": self.EXPERIMENT_NAME,
            "description": self.DESCRIPTION,
            "model": {
                "model_dim": self.MODEL_DIM,
                "num_heads": self.NUM_HEADS,
                "num_encoder_layers": self.NUM_ENCODER_LAYERS,
                "num_decoder_layers": self.NUM_DECODER_LAYERS,
                "ffn_dim": self.FFN_DIM,
                "dropout": self.DROPOUT,
                "max_seq_length": self.MAX_SEQ_LENGTH,
                "kdim": self.KDIM,
                "vdim": self.VDIM,
            },
            "training": {
                "batch_size": self.BATCH_SIZE,
                "learning_rate": self.LEARNING_RATE,
                "warmup_steps": self.WARMUP_STEPS,
                "max_epochs": self.MAX_EPOCHS,
                "gradient_clip": self.GRADIENT_CLIP,
                "tokens_per_batch": self.TOKENS_PER_BATCH,
                "max_tokens_per_batch": self.MAX_TOKENS_PER_BATCH,
                "accumulate_grad_batches": self.ACCUMULATE_GRAD_BATCHES,
            },
            "data": {
                "dataset": self.DATASET,
                "vocab_size": self.VOCAB_SIZE,
                "src_vocab_size": self.SRC_VOCAB_SIZE,
                "tgt_vocab_size": self.TGT_VOCAB_SIZE,
                "shared_vocab": self.SHARED_VOCAB,
                "encoding": self.ENCODING,
                "approximate_sentence_pairs": self.APPROXIMATE_SENTENCE_PAIRS,
                "train_pairs": self.TRAIN_PAIRS,
                "val_pairs": self.VAL_PAIRS,
                "pad_token": self.PAD_TOKEN,
                "bos_token": self.BOS_TOKEN,
                "eos_token": self.EOS_TOKEN,
                "unk_token": self.UNK_TOKEN,
                "src_lang": self.SRC_LANG,
                "tgt_lang": self.TGT_LANG,
            },
            "evaluation": {
                "eval_every": self.EVAL_EVERY,
                "save_every": self.SAVE_EVERY,
                "n_checkpoints": self.N_CHECKPOINTS,
                "beam_size": self.BEAM_SIZE,
                "length_penalty": self.LENGTH_PENALTY,
            },
            "paths": {
                "data_path": self.DATA_PATH,
                "model_save_path": self.MODEL_SAVE_PATH,
                "log_path": self.LOG_PATH,
            },
        }

    @classmethod
    def from_json_structure(cls, json_data):
        """Create Config instance from hierarchical JSON structure"""
        config = cls()

        # Update from hierarchical JSON structure
        if "experiment_name" in json_data:
            config.EXPERIMENT_NAME = json_data["experiment_name"]
        if "description" in json_data:
            config.DESCRIPTION = json_data["description"]

        # Model parameters
        if "model" in json_data:
            model = json_data["model"]
            config.MODEL_DIM = model.get("model_dim", config.MODEL_DIM)
            config.NUM_HEADS = model.get("num_heads", config.NUM_HEADS)
            config.NUM_ENCODER_LAYERS = model.get(
                "num_encoder_layers", config.NUM_ENCODER_LAYERS
            )
            config.NUM_DECODER_LAYERS = model.get(
                "num_decoder_layers", config.NUM_DECODER_LAYERS
            )
            config.FFN_DIM = model.get("ffn_dim", config.FFN_DIM)
            config.DROPOUT = model.get("dropout", config.DROPOUT)
            config.MAX_SEQ_LENGTH = model.get("max_seq_length", config.MAX_SEQ_LENGTH)
            config.KDIM = model.get("kdim", config.KDIM)
            config.VDIM = model.get("vdim", config.VDIM)

        # Training parameters
        if "training" in json_data:
            training = json_data["training"]
            config.BATCH_SIZE = training.get("batch_size", config.BATCH_SIZE)
            config.LEARNING_RATE = training.get("learning_rate", config.LEARNING_RATE)
            config.WARMUP_STEPS = training.get("warmup_steps", config.WARMUP_STEPS)
            config.MAX_EPOCHS = training.get("max_epochs", config.MAX_EPOCHS)
            config.GRADIENT_CLIP = training.get("gradient_clip", config.GRADIENT_CLIP)
            config.TOKENS_PER_BATCH = training.get(
                "tokens_per_batch", config.TOKENS_PER_BATCH
            )
            config.MAX_TOKENS_PER_BATCH = training.get(
                "max_tokens_per_batch", config.MAX_TOKENS_PER_BATCH
            )
            config.ACCUMULATE_GRAD_BATCHES = training.get(
                "accumulate_grad_batches", config.ACCUMULATE_GRAD_BATCHES
            )

        # Data parameters
        if "data" in json_data:
            data = json_data["data"]
            config.DATASET = data.get("dataset", config.DATASET)
            config.VOCAB_SIZE = data.get("vocab_size", config.VOCAB_SIZE)
            config.SRC_VOCAB_SIZE = data.get(
                "src_vocab_size", data.get("vocab_size", config.SRC_VOCAB_SIZE)
            )
            config.TGT_VOCAB_SIZE = data.get(
                "tgt_vocab_size", data.get("vocab_size", config.TGT_VOCAB_SIZE)
            )
            config.SHARED_VOCAB = data.get("shared_vocab", config.SHARED_VOCAB)
            config.ENCODING = data.get("encoding", config.ENCODING)
            config.APPROXIMATE_SENTENCE_PAIRS = data.get(
                "approximate_sentence_pairs", config.APPROXIMATE_SENTENCE_PAIRS
            )
            config.TRAIN_PAIRS = data.get(
                "train_pairs", config.APPROXIMATE_SENTENCE_PAIRS
            )
            config.VAL_PAIRS = data.get("val_pairs", config.VAL_PAIRS)
            config.PAD_TOKEN = data.get("pad_token", config.PAD_TOKEN)
            config.BOS_TOKEN = data.get("bos_token", config.BOS_TOKEN)
            config.EOS_TOKEN = data.get("eos_token", config.EOS_TOKEN)
            config.UNK_TOKEN = data.get("unk_token", config.UNK_TOKEN)
            config.SRC_LANG = data.get("src_lang", config.SRC_LANG)
            config.TGT_LANG = data.get("tgt_lang", config.TGT_LANG)

        # Evaluation parameters
        if "evaluation" in json_data:
            evaluation = json_data["evaluation"]
            config.EVAL_EVERY = evaluation.get("eval_every", config.EVAL_EVERY)
            config.SAVE_EVERY = evaluation.get("save_every", config.SAVE_EVERY)
            config.N_CHECKPOINTS = evaluation.get("n_checkpoints", config.N_CHECKPOINTS)
            config.BEAM_SIZE = evaluation.get("beam_size", config.BEAM_SIZE)
            config.LENGTH_PENALTY = evaluation.get(
                "length_penalty", config.LENGTH_PENALTY
            )

        # Paths
        if "paths" in json_data:
            paths = json_data["paths"]
            config.DATA_PATH = paths.get("data_path", config.DATA_PATH)
            config.MODEL_SAVE_PATH = paths.get(
                "model_save_path", config.MODEL_SAVE_PATH
            )
            config.LOG_PATH = paths.get("log_path", config.LOG_PATH)

        return config

    def _show_optimal_batch_size_recommendation(self):
        """적절한 배치 크기 추천 계산 및 표시"""

        # 데이터셋별 평균 시퀀스 길이 (실제 WMT 데이터 기준)
        dataset_avg_lengths = {
            "wmt14_en_de": 35,
            "wmt14_en_fr": 40,
            "wmt14_cs_en": 32,
            "wmt16_en_de": 36,
            "opus": 30,
            "dummy": 20,
        }

        # 현재 데이터셋의 평균 길이 추정
        avg_seq_length = dataset_avg_lengths.get(self.DATASET, 35)

        # 소스 + 타겟 시퀀스 길이 (대략 비슷함)
        total_avg_length = avg_seq_length * 2

        # 최적 배치 크기 계산
        optimal_batch_size = self.TOKENS_PER_BATCH // total_avg_length
        optimal_batch_size = max(1, min(512, optimal_batch_size))

        # 현재 배치 크기로 실제 토큰 수 계산
        current_tokens = self.BATCH_SIZE * total_avg_length

        # 효율성 계산
        efficiency = (
            min(
                current_tokens / self.TOKENS_PER_BATCH,
                self.TOKENS_PER_BATCH / current_tokens,
            )
            * 100
        )

        print("\n" + "=" * 60)
        print("📊 BATCH SIZE OPTIMIZATION ANALYSIS")
        print("=" * 60)
        print(f"📋 Dataset: {self.DATASET}")
        print(f"📏 Estimated avg sequence length: {avg_seq_length} tokens")
        print(f"🎯 Target tokens per batch: {self.TOKENS_PER_BATCH:,}")
        print("\n📈 Current Configuration:")
        print(f"  - Current batch size: {self.BATCH_SIZE}")
        print(f"  - Estimated tokens per batch: {current_tokens:,}")
        print(f"  - Token efficiency: {efficiency:.1f}%")

        print("\n✨ Recommended Configuration:")
        print(f"  - Optimal batch size: {optimal_batch_size}")
        print(
            f"  - Expected tokens per batch: {optimal_batch_size * total_avg_length:,}"
        )
        print(f"  - Memory efficiency: ~100%")

        # 메모리 및 성능 팁
        if current_tokens < self.TOKENS_PER_BATCH * 0.7:
            print("\n⚠️  Current batch size is TOO SMALL:")
            print(f"    - You're using only {efficiency:.1f}% of target capacity")
            print(f"    - Consider increasing batch_size to ~{optimal_batch_size}")
            print(f"    - This will improve GPU utilization and training speed")

        elif current_tokens > self.TOKENS_PER_BATCH * 1.3:
            print("\n⚠️  Current batch size is TOO LARGE:")
            print(
                f"    - You're using {current_tokens/self.TOKENS_PER_BATCH:.1f}x target capacity"
            )
            print(f"    - Consider decreasing batch_size to ~{optimal_batch_size}")
            print(f"    - This will reduce memory usage and prevent OOM errors")

        else:
            print("\n✅ Current batch size is reasonably optimal!")
            print(f"    - Token efficiency: {efficiency:.1f}%")
            print(f"    - Memory usage is well balanced")

        # 다양한 시나리오 제시
        print(f"\n📋 Alternative Batch Sizes (for different sequence lengths):")
        scenarios = [20, 30, 40, 50, 60, 80]
        for seq_len in scenarios:
            total_len = seq_len * 2
            batch_size = self.TOKENS_PER_BATCH // total_len
            batch_size = max(1, min(512, batch_size))
            tokens = batch_size * total_len
            print(
                f"  - Avg seq {seq_len:2d} tokens → batch_size: {batch_size:3d} → {tokens:,} tokens"
            )

        print("\n💡 Tips:")
        print("  - Longer sequences = smaller batch sizes")
        print("  - Shorter sequences = larger batch sizes")
        print("  - Adjust batch_size based on your actual data statistics")
        print("  - Monitor GPU memory usage during training")
        print("=" * 60)

    def calculate_optimal_batch_size(self, avg_seq_length: int = None) -> int:
        """평균 시퀀스 길이 기준으로 최적 배치 크기 계산"""
        if avg_seq_length is None:
            # 데이터셋별 기본값
            dataset_avg_lengths = {"wmt14_en_de": 35, "wmt14_en_fr": 40, "dummy": 20}
            avg_seq_length = dataset_avg_lengths.get(self.DATASET, 35)

        # 소스 + 타겟 고려
        total_avg_length = avg_seq_length * 2

        # 배치 크기 계산
        optimal_batch_size = self.TOKENS_PER_BATCH // total_avg_length
        optimal_batch_size = max(1, min(512, optimal_batch_size))

        return optimal_batch_size

    def auto_adjust_batch_size(self, avg_seq_length: int = None) -> int:
        """배치 크기 자동 조정 (실제로 변경함)"""
        optimal_batch_size = self.calculate_optimal_batch_size(avg_seq_length)

        print(f"🔧 Auto-adjusting batch size:")
        print(f"  - Old batch size: {self.BATCH_SIZE}")
        print(f"  - New batch size: {optimal_batch_size}")

        self.BATCH_SIZE = optimal_batch_size
        return optimal_batch_size

    def estimate_model_parameters(self) -> int:
        """모델 파라미터 수 추정"""

        # 어휘 크기 결정
        if hasattr(self, "SRC_VOCAB_SIZE") and hasattr(self, "TGT_VOCAB_SIZE"):
            src_vocab_size = self.SRC_VOCAB_SIZE
            tgt_vocab_size = self.TGT_VOCAB_SIZE
        else:
            src_vocab_size = self.VOCAB_SIZE
            tgt_vocab_size = self.VOCAB_SIZE

        d_model = self.MODEL_DIM
        num_heads = self.NUM_HEADS
        num_encoder_layers = self.NUM_ENCODER_LAYERS
        num_decoder_layers = self.NUM_DECODER_LAYERS
        d_ff = self.FFN_DIM
        kdim = self.KDIM
        vdim = self.VDIM

        # 임베딩 레이어 파라미터
        src_embedding_params = src_vocab_size * d_model
        tgt_embedding_params = tgt_vocab_size * d_model
        positional_embedding_params = self.MAX_SEQ_LENGTH * d_model

        # 어텐션 레이어 파라미터 (per layer) - 실제 구현에 맞게 수정
        # 실제로는 Q, K: d_model -> d_model, V: d_model -> d_model, Output: d_model -> d_model
        # MultiHeadAttention에서 실제로는 모든 projection이 d_model -> d_model 차원임
        attention_params_per_layer = (
            (d_model * d_model)
            + (d_model * d_model)
            + (d_model * d_model)
            + (d_model * d_model)
        )

        # Feed-forward 레이어 파라미터 (per layer)
        # Two linear layers
        ff_params_per_layer = d_model * d_ff + d_ff * d_model

        # Layer norm 파라미터 (per layer)
        # 2 layer norms per encoder layer, 3 per decoder layer
        encoder_ln_params_per_layer = 2 * d_model * 2  # weight + bias
        decoder_ln_params_per_layer = 3 * d_model * 2  # weight + bias

        # 총 파라미터 계산
        embedding_params = (
            src_embedding_params + tgt_embedding_params + positional_embedding_params
        )

        encoder_params = num_encoder_layers * (
            attention_params_per_layer
            + ff_params_per_layer
            + encoder_ln_params_per_layer
        )

        decoder_params = num_decoder_layers * (
            2 * attention_params_per_layer
            + ff_params_per_layer
            + decoder_ln_params_per_layer  # 2x attention (self + cross)
        )

        # Output projection (종종 tgt embedding과 weight sharing)
        if getattr(self, "tie_weights", True):
            output_projection_params = 0  # Shared with tgt embedding
        else:
            output_projection_params = d_model * tgt_vocab_size

        total_params = (
            embedding_params
            + encoder_params
            + decoder_params
            + output_projection_params
        )

        return total_params

    def print_estimated_model_info(self):
        """추정된 모델 정보를 컴포넌트별로 상세하게 출력"""

        # 어휘 크기 결정
        if hasattr(self, "SRC_VOCAB_SIZE") and hasattr(self, "TGT_VOCAB_SIZE"):
            src_vocab_size = self.SRC_VOCAB_SIZE
            tgt_vocab_size = self.TGT_VOCAB_SIZE
        else:
            src_vocab_size = self.VOCAB_SIZE
            tgt_vocab_size = self.VOCAB_SIZE

        d_model = self.MODEL_DIM
        num_heads = self.NUM_HEADS
        num_encoder_layers = self.NUM_ENCODER_LAYERS
        num_decoder_layers = self.NUM_DECODER_LAYERS
        d_ff = self.FFN_DIM
        kdim = self.KDIM
        vdim = self.VDIM

        # 컴포넌트별 파라미터 계산

        # 1. 임베딩 레이어
        src_embedding_params = src_vocab_size * d_model
        tgt_embedding_params = tgt_vocab_size * d_model
        positional_embedding_params = self.MAX_SEQ_LENGTH * d_model
        embedding_params = (
            src_embedding_params + tgt_embedding_params + positional_embedding_params
        )

        # 2. 인코더
        attention_params_per_layer = (
            (d_model * kdim) + (d_model * kdim) + (d_model * vdim) + (vdim * d_model)
        )
        ff_params_per_layer = d_model * d_ff + d_ff * d_model
        encoder_ln_params_per_layer = 2 * d_model * 2  # 2 layer norms, weight + bias
        encoder_params_per_layer = (
            attention_params_per_layer
            + ff_params_per_layer
            + encoder_ln_params_per_layer
        )
        encoder_params = num_encoder_layers * encoder_params_per_layer

        # 3. 디코더
        decoder_attention_params_per_layer = (
            2 * attention_params_per_layer
        )  # self + cross attention
        decoder_ln_params_per_layer = 3 * d_model * 2  # 3 layer norms, weight + bias
        decoder_params_per_layer = (
            decoder_attention_params_per_layer
            + ff_params_per_layer
            + decoder_ln_params_per_layer
        )
        decoder_params = num_decoder_layers * decoder_params_per_layer

        # 4. 출력 프로젝션
        if getattr(self, "tie_weights", True):
            output_projection_params = 0  # Shared with tgt embedding
        else:
            output_projection_params = d_model * tgt_vocab_size

        # 총 파라미터
        total_params = (
            embedding_params
            + encoder_params
            + decoder_params
            + output_projection_params
        )

        print(f"\n🔮 Estimated Model Information by Config:")
        print("=" * 70)

        # 전체 요약
        print(f"📊 Total Parameters: {total_params:,}")
        print(f"💾 Model Size: {total_params * 4 / (1024**2):.1f} MB (float32)")

        # 스케일 분류
        if total_params < 10_000_000:
            scale = "Small (< 10M)"
        elif total_params < 100_000_000:
            scale = "Medium (10M - 100M)"
        elif total_params < 1_000_000_000:
            scale = "Large (100M - 1B)"
        else:
            scale = "Very Large (> 1B)"
        print(f"🏷️  Model Scale: {scale}")

        print("\n📋 Component Breakdown:")
        print("-" * 70)

        # 1. 임베딩
        print(
            f"🔤 Embeddings: {embedding_params:,} parameters ({embedding_params/total_params*100:.1f}%)"
        )
        print(
            f"  ├─ Source Embedding: {src_embedding_params:,} ({src_vocab_size:,} × {d_model})"
        )
        print(
            f"  ├─ Target Embedding: {tgt_embedding_params:,} ({tgt_vocab_size:,} × {d_model})"
        )
        print(
            f"  └─ Positional Encoding: {positional_embedding_params:,} ({self.MAX_SEQ_LENGTH} × {d_model})"
        )

        # 2. 인코더
        print(
            f"\n🔄 Encoder ({num_encoder_layers} layers): {encoder_params:,} parameters ({encoder_params/total_params*100:.1f}%)"
        )
        print(f"  ├─ Per Layer: {encoder_params_per_layer:,} parameters")
        print(f"  │  ├─ Self-Attention: {attention_params_per_layer:,}")
        print(f"  │  │  ├─ Q projection: {d_model * kdim:,} ({d_model} → {kdim})")
        print(f"  │  │  ├─ K projection: {d_model * kdim:,} ({d_model} → {kdim})")
        print(f"  │  │  ├─ V projection: {d_model * vdim:,} ({d_model} → {vdim})")
        print(f"  │  │  └─ Output projection: {vdim * d_model:,} ({vdim} → {d_model})")
        print(f"  │  ├─ Feed-Forward: {ff_params_per_layer:,}")
        print(f"  │  │  ├─ Linear 1: {d_model * d_ff:,} ({d_model} → {d_ff})")
        print(f"  │  │  └─ Linear 2: {d_ff * d_model:,} ({d_ff} → {d_model})")
        print(
            f"  │  └─ Layer Norms: {encoder_ln_params_per_layer:,} (2 layers × {d_model} × 2)"
        )

        # 3. 디코더
        print(
            f"\n🎯 Decoder ({num_decoder_layers} layers): {decoder_params:,} parameters ({decoder_params/total_params*100:.1f}%)"
        )
        print(f"  ├─ Per Layer: {decoder_params_per_layer:,} parameters")
        print(f"  │  ├─ Self-Attention: {attention_params_per_layer:,}")
        print(f"  │  ├─ Cross-Attention: {attention_params_per_layer:,}")
        print(f"  │  ├─ Feed-Forward: {ff_params_per_layer:,}")
        print(
            f"  │  └─ Layer Norms: {decoder_ln_params_per_layer:,} (3 layers × {d_model} × 2)"
        )

        # 4. 출력 프로젝션
        if output_projection_params > 0:
            print(
                f"\n📤 Output Projection: {output_projection_params:,} parameters ({output_projection_params/total_params*100:.1f}%)"
            )
            print(
                f"  └─ Linear: {output_projection_params:,} ({d_model} → {tgt_vocab_size:,})"
            )
        else:
            print(
                f"\n📤 Output Projection: Shared with target embedding (tie_weights=True)"
            )

        print("\n📈 Architecture Summary:")
        print(f"  ├─ Model Dimension (d_model): {d_model}")
        print(f"  ├─ Attention Heads: {num_heads}")
        print(f"  ├─ Key/Value Dimensions: {kdim}/{vdim}")
        print(f"  ├─ Feed-Forward Dimension: {d_ff}")
        print(f"  ├─ Encoder Layers: {num_encoder_layers}")
        print(f"  ├─ Decoder Layers: {num_decoder_layers}")
        print(f"  ├─ Source Vocabulary: {src_vocab_size:,}")
        print(f"  ├─ Target Vocabulary: {tgt_vocab_size:,}")
        print(f"  └─ Max Sequence Length: {self.MAX_SEQ_LENGTH}")

        # 메모리 추정
        print(f"\n💾 Memory Estimation (Training):")
        params_memory = total_params * 4 / (1024**2)  # float32
        gradients_memory = total_params * 4 / (1024**2)  # gradients
        optimizer_memory = total_params * 8 / (1024**2)  # Adam (2 states)
        total_memory = params_memory + gradients_memory + optimizer_memory

        print(f"  ├─ Parameters: {params_memory:.1f} MB")
        print(f"  ├─ Gradients: {gradients_memory:.1f} MB")
        print(f"  ├─ Optimizer States: {optimizer_memory:.1f} MB")
        print(f"  └─ Total Model Memory: {total_memory:.1f} MB")

        # 유명한 모델들과 비교
        print(f"\n🏆 Model Comparison:")
        comparisons = [
            ("Transformer Base (Paper)", 65_000_000),
            ("Transformer Big (Paper)", 213_000_000),
            ("BERT Base", 110_000_000),
            ("BERT Large", 340_000_000),
            ("GPT-2 Small", 117_000_000),
            ("GPT-2 Medium", 345_000_000),
            ("T5 Base", 220_000_000),
        ]

        closest_model = None
        min_diff = float("inf")

        for name, params in comparisons:
            diff = abs(total_params - params) / params
            if diff < min_diff:
                min_diff = diff
                closest_model = (name, params, diff)

            if diff < 0.1:  # Within 10%
                print(
                    f"  🎯 Similar to {name}: {params:,} parameters ({diff*100:.1f}% difference)"
                )
                break

        if closest_model and min_diff >= 0.1:
            name, params, diff = closest_model
            if total_params < params:
                print(
                    f"  📉 Closest to {name}: {params:,} parameters ({diff*100:.1f}% smaller)"
                )
            else:
                print(
                    f"  📈 Closest to {name}: {params:,} parameters ({diff*100:.1f}% larger)"
                )

        # 상세 파라미터 분포
        print(f"\n📊 Detailed Parameter Distribution:")
        components = [
            ("Embeddings", embedding_params),
            ("Encoder", encoder_params),
            ("Decoder", decoder_params),
            (
                "Output Projection",
                output_projection_params if output_projection_params > 0 else 0,
            ),
        ]

        print("  Component           Parameters      Percentage")
        print("  " + "-" * 50)
        for name, params in components:
            if params > 0:
                percentage = params / total_params * 100
                print(f"  {name:<18} {params:>12,} {percentage:>9.1f}%")

        print("=" * 70)
