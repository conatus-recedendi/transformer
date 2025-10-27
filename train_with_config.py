"""
Config-based training script for Transformer model
"""

import os
import json
import argparse
import torch
from typing import Dict, Any
from pathlib import Path

from src.config import Config
from src.data_loader import (
    TransformerDataset,
    create_data_loader,
    load_dummy_data,
    DummyTokenizer,
)
from src.trainer import create_trainer
from src.transformer import Transformer
from src.utils import set_seed, get_device, print_model_summary


def load_config_from_json(config_path: str) -> Config:
    """Load configuration from JSON file"""
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    # Flatten nested dictionaries
    flattened_config = {}

    # Model parameters
    if "model" in config_dict:
        model_config = config_dict["model"]
        flattened_config.update(
            {
                "MODEL_DIM": model_config.get("model_dim", 512),
                "NUM_HEADS": model_config.get("num_heads", 8),
                "NUM_ENCODER_LAYERS": model_config.get("num_encoder_layers", 6),
                "NUM_DECODER_LAYERS": model_config.get("num_decoder_layers", 6),
                "FFN_DIM": model_config.get("ffn_dim", 2048),
                "DROPOUT": model_config.get("dropout", 0.1),
                "MAX_SEQ_LENGTH": model_config.get("max_seq_length", 512),
            }
        )

    # Training parameters
    if "training" in config_dict:
        training_config = config_dict["training"]
        flattened_config.update(
            {
                "BATCH_SIZE": training_config.get("batch_size", 32),
                "LEARNING_RATE": training_config.get("learning_rate", 1e-4),
                "WARMUP_STEPS": training_config.get("warmup_steps", 4000),
                "MAX_EPOCHS": training_config.get("max_epochs", 100),
                "GRADIENT_CLIP": training_config.get("gradient_clip", 1.0),
                "TOKENS_PER_BATCH": training_config.get("tokens_per_batch", 25000),
                "MAX_TOKENS_PER_BATCH": training_config.get(
                    "max_tokens_per_batch", 25000
                ),
                "ACCUMULATE_GRAD_BATCHES": training_config.get(
                    "accumulate_grad_batches", 1
                ),
            }
        )

    # Data parameters
    if "data" in config_dict:
        data_config = config_dict["data"]
        flattened_config.update(
            {
                "VOCAB_SIZE": data_config.get("vocab_size", 30000),
                "PAD_TOKEN": data_config.get("pad_token", 0),
                "BOS_TOKEN": data_config.get("bos_token", 1),
                "EOS_TOKEN": data_config.get("eos_token", 2),
                "UNK_TOKEN": data_config.get("unk_token", 3),
                "DATASET": data_config.get("dataset", "dummy"),
                "SHARED_VOCAB": data_config.get("shared_vocab", True),
                "ENCODING": data_config.get("encoding", "bpe"),
                "SRC_LANG": data_config.get("src_lang", "en"),
                "TGT_LANG": data_config.get("tgt_lang", "de"),
                "APPROXIMATE_SENTENCE_PAIRS": data_config.get(
                    "approximate_sentence_pairs", 1000000
                ),
            }
        )

    # Evaluation parameters
    if "evaluation" in config_dict:
        eval_config = config_dict["evaluation"]
        flattened_config.update(
            {
                "EVAL_EVERY": eval_config.get("eval_every", 500),
                "SAVE_EVERY": eval_config.get("save_every", 1000),
            }
        )

    # Path parameters
    if "paths" in config_dict:
        paths_config = config_dict["paths"]
        flattened_config.update(
            {
                "DATA_PATH": paths_config.get("data_path", "data/"),
                "MODEL_SAVE_PATH": paths_config.get("model_save_path", "checkpoints/"),
                "LOG_PATH": paths_config.get("log_path", "logs/"),
            }
        )
    else:
        # Set default paths if not specified
        flattened_config.update(
            {
                "DATA_PATH": "data/",
                "MODEL_SAVE_PATH": "checkpoints/",
                "LOG_PATH": "logs/",
            }
        )

    # Set device
    flattened_config["DEVICE"] = "auto"

    return Config(**flattened_config)


def create_model_from_config(config: Config) -> Transformer:
    """Create Transformer model from config"""
    print("Creating Transformer model from config...")
    print(f"Model configuration:")
    print(f"  - Model dimension: {config.MODEL_DIM}")
    print(f"  - Number of heads: {config.NUM_HEADS}")
    print(f"  - Encoder layers: {config.NUM_ENCODER_LAYERS}")
    print(f"  - Decoder layers: {config.NUM_DECODER_LAYERS}")
    print(f"  - FFN dimension: {config.FFN_DIM}")
    print(f"  - Vocabulary size: {config.VOCAB_SIZE}")
    print(f"  - Max sequence length: {config.MAX_SEQ_LENGTH}")
    print(f"  - Dropout: {config.DROPOUT}")

    model = Transformer(
        src_vocab_size=config.VOCAB_SIZE,
        tgt_vocab_size=config.VOCAB_SIZE,  # Assuming same vocab for src and tgt
        d_model=config.MODEL_DIM,
        num_heads=config.NUM_HEADS,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        d_ff=config.FFN_DIM,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dropout=config.DROPOUT,
        pad_token_id=config.PAD_TOKEN,
        tie_weights=True,
    )

    return model


def create_data_loaders_from_config(config, use_dummy_data=False):
    """Create data loaders based on configuration."""
    print("📊 Initializing data loaders...")

    # Display dataset information
    if hasattr(config, "DATASET") and not use_dummy_data:
        print(f"\n📂 Dataset: {config.DATASET}")

        if config.DATASET == "WMT14_EN_DE":
            print("   - Language pair: English → German")
            print("   - Training pairs: ~4.5M sentence pairs")
            print("   - Source vocabulary: ~37,000 subword units")
            print("   - Target vocabulary: ~37,000 subword units")
            print("   - Shared vocabulary: Yes")
            print("   - Encoding: Byte Pair Encoding (BPE)")

        elif config.DATASET == "WMT14_EN_FR":
            print("   - Language pair: English → French")
            print("   - Training pairs: ~36M sentence pairs")
            print("   - Source vocabulary: ~32,000 subword units")
            print("   - Target vocabulary: ~32,000 subword units")
            print("   - Shared vocabulary: Yes")
            print("   - Encoding: Byte Pair Encoding (BPE)")

        if hasattr(config, "TOKENS_PER_BATCH"):
            print(f"   - Target tokens per batch: {config.TOKENS_PER_BATCH:,}")

        print("\n⚠️  Note: Real WMT dataset loading not yet implemented.")
        print("   Currently using dummy data. To implement real WMT loading:")
        print("   1. Download WMT 2014 dataset")
        print("   2. Apply BPE tokenization")
        print("   3. Implement dynamic batching for token-based batches")
        print("   4. Add proper data preprocessing pipeline")

    else:
        print("📊 Using dummy dataset for testing")

    if use_dummy_data or not hasattr(config, "DATASET"):
        print("\n🔄 Loading dummy data...")

        # Determine vocabulary sizes for dummy data
        if hasattr(config, "SRC_VOCAB_SIZE") and hasattr(config, "TGT_VOCAB_SIZE"):
            src_vocab_size = config.SRC_VOCAB_SIZE
            tgt_vocab_size = config.TGT_VOCAB_SIZE
        else:
            src_vocab_size = config.VOCAB_SIZE
            tgt_vocab_size = config.VOCAB_SIZE

        # Scale dummy data based on dataset size
        if hasattr(config, "TRAIN_PAIRS"):
            # Use a fraction of the real dataset size for dummy data
            total_samples = min(100000, config.TRAIN_PAIRS // 100)
        else:
            total_samples = 50000

        src_sequences, tgt_sequences = load_dummy_data(
            num_samples=total_samples
        )  # Split data
        train_size = int(0.8 * len(src_sequences))
        val_size = int(0.1 * len(src_sequences))

        train_src = src_sequences[:train_size]
        train_tgt = tgt_sequences[:train_size]

        val_src = src_sequences[train_size : train_size + val_size]
        val_tgt = tgt_sequences[train_size : train_size + val_size]

        test_src = src_sequences[train_size + val_size :]
        test_tgt = tgt_sequences[train_size + val_size :]

        # Create datasets
        train_dataset = TransformerDataset(
            train_src,
            train_tgt,
            src_vocab_size,
            tgt_vocab_size,
            config.MAX_SEQ_LENGTH,
        )

        val_dataset = TransformerDataset(
            val_src,
            val_tgt,
            src_vocab_size,
            tgt_vocab_size,
            config.MAX_SEQ_LENGTH,
        )

        test_dataset = TransformerDataset(
            test_src,
            test_tgt,
            src_vocab_size,
            tgt_vocab_size,
            config.MAX_SEQ_LENGTH,
        )

        print(f"Dataset sizes:")
        print(f"  - Train: {len(train_dataset):,} samples")
        print(f"  - Validation: {len(val_dataset):,} samples")
        print(f"  - Test: {len(test_dataset):,} samples")

    else:
        # 실제 WMT 데이터 로딩
        print("🔄 Loading real WMT dataset...")
        
        # WMT 데이터 로더 임포트
        try:
            from src.wmt_data_loader import create_wmt_data_loaders, prepare_sample_data
            
            # 데이터 파일이 없으면 샘플 데이터 생성
            data_path = Path(config.DATA_PATH)
            lang_dir = data_path / f"{config.SRC_LANG}-{config.TGT_LANG}"
            train_dir = lang_dir / "train"
            
            if not train_dir.exists() or not any(train_dir.iterdir()):
                print("� No existing data found. Creating sample data...")
                prepare_sample_data(config)
            
            # WMT 데이터 로더 생성
            train_loader, val_loader, test_loader = create_wmt_data_loaders(config)
            
            if train_loader is None:
                print("⚠️  Failed to load WMT data. Falling back to dummy data...")
                total_samples = min(100000, config.TRAIN_PAIRS // 100)
                src_sequences, tgt_sequences = load_dummy_data(num_samples=total_samples)
                
                # 더미 데이터로 데이터셋 생성
                train_size = int(0.8 * len(src_sequences))
                val_size = int(0.1 * len(src_sequences))

                train_src = src_sequences[:train_size]
                train_tgt = tgt_sequences[:train_size]
                val_src = src_sequences[train_size : train_size + val_size]
                val_tgt = tgt_sequences[train_size : train_size + val_size]
                test_src = src_sequences[train_size + val_size :]
                test_tgt = tgt_sequences[train_size + val_size :]

                # 더미 데이터셋 생성
                train_dataset = TransformerDataset(
                    train_src, train_tgt, src_vocab_size, tgt_vocab_size, config.MAX_SEQ_LENGTH,
                )
                val_dataset = TransformerDataset(
                    val_src, val_tgt, src_vocab_size, tgt_vocab_size, config.MAX_SEQ_LENGTH,
                )
                test_dataset = TransformerDataset(
                    test_src, test_tgt, src_vocab_size, tgt_vocab_size, config.MAX_SEQ_LENGTH,
                )
                
                # 더미 데이터 로더 생성
                train_loader = create_data_loader(train_dataset, config.BATCH_SIZE, shuffle=True)
                val_loader = create_data_loader(val_dataset, config.BATCH_SIZE, shuffle=False)
                test_loader = create_data_loader(test_dataset, config.BATCH_SIZE, shuffle=False)
                
                print(f"Dataset sizes (dummy data):")
                print(f"  - Train: {len(train_dataset):,} samples")
                print(f"  - Validation: {len(val_dataset):,} samples")
                print(f"  - Test: {len(test_dataset):,} samples")
            else:
                print("✅ Successfully loaded WMT data!")
                
            return train_loader, val_loader, test_loader
            
        except ImportError as e:
            print(f"⚠️  Failed to import WMT data loader: {e}")
            print("⏳ Falling back to dummy data...")
            total_samples = min(100000, config.TRAIN_PAIRS // 100)
            src_sequences, tgt_sequences = load_dummy_data(num_samples=total_samples)
            
            # 더미 데이터로 데이터셋 생성 (else 블록과 동일)
            train_size = int(0.8 * len(src_sequences))
            val_size = int(0.1 * len(src_sequences))

            train_src = src_sequences[:train_size]
            train_tgt = tgt_sequences[:train_size]
            val_src = src_sequences[train_size : train_size + val_size]
            val_tgt = tgt_sequences[train_size : train_size + val_size]
            test_src = src_sequences[train_size + val_size :]
            test_tgt = tgt_sequences[train_size + val_size :]

            # 더미 데이터셋 생성
            train_dataset = TransformerDataset(
                train_src, train_tgt, src_vocab_size, tgt_vocab_size, config.MAX_SEQ_LENGTH,
            )
            val_dataset = TransformerDataset(
                val_src, val_tgt, src_vocab_size, tgt_vocab_size, config.MAX_SEQ_LENGTH,
            )
            test_dataset = TransformerDataset(
                test_src, test_tgt, src_vocab_size, tgt_vocab_size, config.MAX_SEQ_LENGTH,
            )
            
            # 더미 데이터 로더 생성
            train_loader = create_data_loader(train_dataset, config.BATCH_SIZE, shuffle=True)
            val_loader = create_data_loader(val_dataset, config.BATCH_SIZE, shuffle=False)
            test_loader = create_data_loader(test_dataset, config.BATCH_SIZE, shuffle=False)
            
            print(f"Dataset sizes (dummy fallback):")
            print(f"  - Train: {len(train_dataset):,} samples")
            print(f"  - Validation: {len(val_dataset):,} samples")
            print(f"  - Test: {len(test_dataset):,} samples")
            
            return train_loader, val_loader, test_loader

    # 더미 데이터용 (use_dummy_data == True인 경우)
    # Split data
    train_size = int(0.8 * len(src_sequences))
    val_size = int(0.1 * len(src_sequences))

    train_src = src_sequences[:train_size]
    train_tgt = tgt_sequences[:train_size]

    val_src = src_sequences[train_size : train_size + val_size]
    val_tgt = tgt_sequences[train_size : train_size + val_size]

    test_src = src_sequences[train_size + val_size :]
    test_tgt = tgt_sequences[train_size + val_size :]

    # Create datasets
    train_dataset = TransformerDataset(
        train_src,
        train_tgt,
        src_vocab_size,
        tgt_vocab_size,
        config.MAX_SEQ_LENGTH,
    )

    val_dataset = TransformerDataset(
        val_src,
        val_tgt,
        src_vocab_size,
        tgt_vocab_size,
        config.MAX_SEQ_LENGTH,
    )

    test_dataset = TransformerDataset(
        test_src,
        test_tgt,
        src_vocab_size,
        tgt_vocab_size,
        config.MAX_SEQ_LENGTH,
    )

    print(f"Dataset sizes:")
    print(f"  - Train: {len(train_dataset):,} samples")
    print(f"  - Validation: {len(val_dataset):,} samples")
    print(f"  - Test: {len(test_dataset):,} samples")

    # Create data loaders
    train_loader = create_data_loader(
        train_dataset, config.BATCH_SIZE, shuffle=True, pad_token_id=config.PAD_TOKEN
    )

    val_loader = create_data_loader(
        val_dataset, config.BATCH_SIZE, shuffle=False, pad_token_id=config.PAD_TOKEN
    )

    test_loader = create_data_loader(
        test_dataset, config.BATCH_SIZE, shuffle=False, pad_token_id=config.PAD_TOKEN
    )

    print(f"Data loaders created:")
    print(f"  - Train batches: {len(train_loader):,}")
    print(f"  - Val batches: {len(val_loader):,}")
    print(f"  - Test batches: {len(test_loader):,}")

    return train_loader, val_loader, test_loader


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Transformer with config file")

    parser.add_argument(
        "--config", type=str, required=True, help="Path to config JSON file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument("--test_only", action="store_true", help="Only run testing")
    parser.add_argument(
        "--use_dummy_data",
        action="store_true",
        default=True,
        help="Use dummy data for testing",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training",
    )

    args = parser.parse_args()

    # Check if config file exists
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    print(f"Loading configuration from: {args.config}")

    # Load configuration
    config = load_config_from_json(args.config)

    # Override device if specified
    if args.device != "auto":
        config.DEVICE = args.device
    else:
        device = get_device()
        config.DEVICE = str(device)

    # Set random seed
    set_seed(args.seed)

    print("\n" + "=" * 60)
    print("TRANSFORMER TRAINING CONFIGURATION")
    print("=" * 60)

    print("\nModel Configuration:")
    print(f"  Model Dimension: {config.MODEL_DIM}")
    print(f"  Number of Heads: {config.NUM_HEADS}")
    print(f"  Encoder Layers: {config.NUM_ENCODER_LAYERS}")
    print(f"  Decoder Layers: {config.NUM_DECODER_LAYERS}")
    print(f"  FFN Dimension: {config.FFN_DIM}")
    print(f"  Vocabulary Size: {config.VOCAB_SIZE}")
    print(f"  Max Sequence Length: {config.MAX_SEQ_LENGTH}")
    print(f"  Dropout: {config.DROPOUT}")

    print("\nTraining Configuration:")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Learning Rate: {config.LEARNING_RATE}")
    print(f"  Warmup Steps: {config.WARMUP_STEPS}")
    print(f"  Max Epochs: {config.MAX_EPOCHS}")
    print(f"  Gradient Clip: {config.GRADIENT_CLIP}")

    print("\nData Configuration:")
    print(f"  PAD Token: {config.PAD_TOKEN}")
    print(f"  BOS Token: {config.BOS_TOKEN}")
    print(f"  EOS Token: {config.EOS_TOKEN}")
    print(f"  UNK Token: {config.UNK_TOKEN}")

    # Display dataset-specific configuration if available
    if hasattr(config, "DATASET"):
        print(f"\nDataset Configuration:")
        print(f"  Dataset: {config.DATASET}")

        if hasattr(config, "SRC_VOCAB_SIZE") and hasattr(config, "TGT_VOCAB_SIZE"):
            print(f"  Source Vocabulary Size: {config.SRC_VOCAB_SIZE:,}")
            print(f"  Target Vocabulary Size: {config.TGT_VOCAB_SIZE:,}")

        if hasattr(config, "SHARED_VOCAB"):
            print(f"  Shared Vocabulary: {config.SHARED_VOCAB}")

        if hasattr(config, "ENCODING"):
            print(f"  Encoding: {config.ENCODING}")

        if hasattr(config, "TOKENS_PER_BATCH"):
            print(f"  Tokens per Batch: {config.TOKENS_PER_BATCH:,}")

        if hasattr(config, "TRAIN_PAIRS") and hasattr(config, "VAL_PAIRS"):
            print(f"  Training Pairs: {config.TRAIN_PAIRS:,}")
            print(f"  Validation Pairs: {config.VAL_PAIRS:,}")

    print("\nEvaluation Configuration:")
    print(f"  Eval Every: {config.EVAL_EVERY} steps")
    print(f"  Save Every: {config.SAVE_EVERY} steps")

    print(f"\nDevice: {config.DEVICE}")
    print(f"Random Seed: {args.seed}")

    print("=" * 60)

    # Create data loaders
    print("\n📊 Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders_from_config(
        config, use_dummy_data=args.use_dummy_data
    )

    # Create model
    print("\n🤖 Creating model...")
    model = create_model_from_config(config)
    print_model_summary(model)

    # Create trainer
    print("\n🏋️ Creating trainer...")
    trainer = create_trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    print(f"Trainer created successfully!")
    print(f"  - Optimizer: Adam (betas=(0.9, 0.98), eps=1e-9)")
    print(f"  - Scheduler: Warmup + Decay")
    print(f"  - Loss Function: CrossEntropyLoss (ignore_index={config.PAD_TOKEN})")

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\n📂 Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Create directories
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.LOG_PATH, exist_ok=True)
    print(f"\n📁 Directories created:")
    print(f"  - Checkpoints: {config.MODEL_SAVE_PATH}")
    print(f"  - Logs: {config.LOG_PATH}")

    # Run training or testing
    if args.test_only:
        print("\n🧪 Running test only...")
        test_results = trainer.test()
        print(f"\n📈 Test Results:")
        print(f"  Test Loss: {test_results['loss']:.6f}")
    else:
        print("\n🚀 Starting training...")
        print(f"Training will run for {config.MAX_EPOCHS} epochs")
        print(f"Validation every {config.EVAL_EVERY} steps")
        print(f"Checkpoints saved every {config.SAVE_EVERY} steps")
        print("\nPress Ctrl+C to stop training gracefully...\n")

        try:
            trainer.train()

            # Run final test
            print("\n🧪 Running final test...")
            test_results = trainer.test()
            print(f"\n📈 Final Test Results:")
            print(f"  Test Loss: {test_results['loss']:.6f}")

        except KeyboardInterrupt:
            print("\n⏹️ Training interrupted by user")
            print("Saving final checkpoint...")
            trainer._save_checkpoint()

        except Exception as e:
            print(f"\n❌ Training failed with error: {e}")
            print("Saving emergency checkpoint...")
            trainer._save_checkpoint()
            raise

    print("\n✅ Training completed successfully!")


if __name__ == "__main__":
    main()
