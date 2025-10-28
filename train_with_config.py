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
        model = config_dict["model"]
        flattened_config.update(
            {
                "MODEL_DIM": model.get("model_dim", 512),
                "NUM_HEADS": model.get("num_heads", 8),
                "NUM_ENCODER_LAYERS": model.get("num_encoder_layers", 6),
                "NUM_DECODER_LAYERS": model.get("num_decoder_layers", 6),
                "FFN_DIM": model.get("ffn_dim", 2048),
                "DROPOUT": model.get("dropout", 0.1),
                "MAX_SEQ_LENGTH": model.get("max_seq_length", 512),
            }
        )

    # Training parameters
    if "training" in config_dict:
        training = config_dict["training"]
        flattened_config.update(
            {
                "BATCH_SIZE": training.get("batch_size", 32),
                "LEARNING_RATE": training.get("learning_rate", 1e-4),
                "WARMUP_STEPS": training.get("warmup_steps", 4000),
                "MAX_EPOCHS": training.get("max_epochs", 100),
                "GRADIENT_CLIP": training.get("gradient_clip", 1.0),
                "TOKENS_PER_BATCH": training.get("tokens_per_batch", 25000),
                "MAX_TOKENS_PER_BATCH": training.get("max_tokens_per_batch", 25000),
                "ACCUMULATE_GRAD_BATCHES": training.get("accumulate_grad_batches", 1),
            }
        )

    # Data parameters
    if "data" in config_dict:
        data = config_dict["data"]
        flattened_config.update(
            {
                "DATASET": data.get("dataset", "wmt14_en_de"),
                "VOCAB_SIZE": data.get("vocab_size", 30000),
                "SRC_VOCAB_SIZE": data.get(
                    "src_vocab_size", data.get("vocab_size", 30000)
                ),
                "TGT_VOCAB_SIZE": data.get(
                    "tgt_vocab_size", data.get("vocab_size", 30000)
                ),
                "SHARED_VOCAB": data.get("shared_vocab", True),
                "ENCODING": data.get("encoding", "bpe"),
                "APPROXIMATE_SENTENCE_PAIRS": data.get(
                    "approximate_sentence_pairs", 4500000
                ),
                "TRAIN_PAIRS": data.get(
                    "train_pairs", data.get("approximate_sentence_pairs", 4500000)
                ),
                "VAL_PAIRS": data.get("val_pairs", 500000),
                "PAD_TOKEN": data.get("pad_token", 0),
                "BOS_TOKEN": data.get("bos_token", 1),
                "EOS_TOKEN": data.get("eos_token", 2),
                "UNK_TOKEN": data.get("unk_token", 3),
                "SRC_LANG": data.get("src_lang", "en"),
                "TGT_LANG": data.get("tgt_lang", "de"),
            }
        )

    # Evaluation parameters
    if "evaluation" in config_dict:
        evaluation = config_dict["evaluation"]
        flattened_config.update(
            {
                "EVAL_EVERY": evaluation.get("eval_every", 500),
                "SAVE_EVERY": evaluation.get("save_every", 1000),
                "N_CHECKPOINTS": evaluation.get("n_checkpoints", 5),
                "BEAM_SIZE": evaluation.get("beam_size", 4),
                "LENGTH_PENALTY": evaluation.get("length_penalty", 0.6),
            }
        )

    # Paths
    if "paths" in config_dict:
        paths = config_dict["paths"]
        flattened_config.update(
            {
                "DATA_PATH": paths.get("data_path", "data/"),
                "MODEL_SAVE_PATH": paths.get("model_save_path", "checkpoints/"),
                "LOG_PATH": paths.get("log_path", "logs/"),
            }
        )

    # Additional metadata
    if "experiment_name" in config_dict:
        flattened_config["EXPERIMENT_NAME"] = config_dict["experiment_name"]
    if "description" in config_dict:
        flattened_config["DESCRIPTION"] = config_dict["description"]

    return Config(**flattened_config)


def load_data(config: Config, use_dummy: bool = True):
    """Load training, validation, and test data"""

    if use_dummy:
        # Use dummy data for testing
        print("🔄 Loading dummy data...")
        src_sequences, tgt_sequences = load_dummy_data(num_samples=10000)

        # Split data
        train_size = int(0.8 * len(src_sequences))
        val_size = int(0.1 * len(src_sequences))

        train_src = src_sequences[:train_size]
        train_tgt = tgt_sequences[:train_size]

        val_src = src_sequences[train_size : train_size + val_size]
        val_tgt = tgt_sequences[train_size : train_size + val_size]

        test_src = src_sequences[train_size + val_size :]
        test_tgt = tgt_sequences[train_size + val_size :]

        # Determine vocabulary sizes for dummy data
        if hasattr(config, "SRC_VOCAB_SIZE") and hasattr(config, "TGT_VOCAB_SIZE"):
            src_vocab_size = config.SRC_VOCAB_SIZE
            tgt_vocab_size = config.TGT_VOCAB_SIZE
        else:
            src_vocab_size = config.VOCAB_SIZE
            tgt_vocab_size = config.VOCAB_SIZE

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

        # Create data loaders
        train_loader = create_data_loader(
            train_dataset,
            config.BATCH_SIZE,
            shuffle=True,
            pad_token_id=config.PAD_TOKEN,
        )

        val_loader = create_data_loader(
            val_dataset, config.BATCH_SIZE, shuffle=False, pad_token_id=config.PAD_TOKEN
        )

        test_loader = create_data_loader(
            test_dataset,
            config.BATCH_SIZE,
            shuffle=False,
            pad_token_id=config.PAD_TOKEN,
        )

        print(f"Dataset sizes:")
        print(f"  - Train: {len(train_dataset):,} samples")
        print(f"  - Validation: {len(val_dataset):,} samples")
        print(f"  - Test: {len(test_dataset):,} samples")

    else:
        # 실제 WMT 데이터 로딩 (분리된 언어 파일 형식)
        print("🔄 Loading real WMT dataset...")

        try:
            from src.real_data_loader import load_real_wmt_data

            # 언어 설정이 없으면 기본값 설정
            if not hasattr(config, "SRC_LANG"):
                config.SRC_LANG = "en"
            if not hasattr(config, "TGT_LANG"):
                config.TGT_LANG = "de"
            if not hasattr(config, "DATASET"):
                config.DATASET = "wmt14_en_de"

            train_loader, val_loader, test_loader = load_real_wmt_data(config)

            if train_loader is None:
                print("❌ Failed to load real data. Check if data files exist:")
                print(f"   Expected location: {config.DATA_PATH}{config.DATASET}/")
                print(
                    f"   Expected files: train.{config.SRC_LANG}, train.{config.TGT_LANG}, valid.{config.SRC_LANG}, valid.{config.TGT_LANG}, test.{config.SRC_LANG}, test.{config.TGT_LANG}"
                )
                print("   File format: each line should contain one sentence")
                raise FileNotFoundError("Real data files not found")

        except (ImportError, FileNotFoundError) as e:
            print(f"⚠️  Error loading real data: {e}")
            print("   Falling back to dummy data...")
            return load_data(config, use_dummy=True)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


def create_model_from_config(config: Config) -> torch.nn.Module:
    """Create Transformer model from configuration"""
    print("🤖 Creating Transformer model...")

    # 어휘 크기 결정
    if hasattr(config, "SRC_VOCAB_SIZE") and hasattr(config, "TGT_VOCAB_SIZE"):
        src_vocab_size = config.SRC_VOCAB_SIZE
        tgt_vocab_size = config.TGT_VOCAB_SIZE
    else:
        src_vocab_size = config.VOCAB_SIZE
        tgt_vocab_size = config.VOCAB_SIZE

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config.MODEL_DIM,
        num_heads=config.NUM_HEADS,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        d_ff=config.FFN_DIM,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dropout=config.DROPOUT,
        pad_token_id=config.PAD_TOKEN,
        tie_weights=True,  # Tie embedding and output projection weights
    )

    return model


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train Transformer model with JSON config"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--use-dummy-data",
        action="store_true",
        help="Use dummy data instead of real data",
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

        if hasattr(config, "SRC_LANG") and hasattr(config, "TGT_LANG"):
            print(f"  Language Pair: {config.SRC_LANG} → {config.TGT_LANG}")

    print("\nEvaluation Configuration:")
    print(f"  Eval Every: {config.EVAL_EVERY} steps")
    print(f"  Save Every: {config.SAVE_EVERY} steps")

    print(f"\nDevice: {config.DEVICE}")
    print(f"Random Seed: {args.seed}")

    print("=" * 60)

    # Load data
    train_loader, val_loader, test_loader = load_data(
        config, use_dummy=args.use_dummy_data
    )

    # Create model
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

    # Start training
    print("\n🚀 Starting training...")
    trainer.train()

    # Run final test
    print("\n🧪 Running final test...")
    test_results = trainer.test()
    print(f"Final test results: {test_results}")

    print("\n✅ Training completed!")


if __name__ == "__main__":
    main()
