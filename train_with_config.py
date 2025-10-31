"""
Config-based training script for Transformer model
"""

import os
import json
import argparse
import logging
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
                "KDIM": model.get("kdim", 64),
                "VDIM": model.get("vdim", 64),
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
                "DEBUG_OUTPUT_EVERY": training.get("debug_output_every", 100),
                "ENABLE_OUTPUT_DEBUG": training.get("enable_output_debug", False),
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
    logger = logging.getLogger(__name__)
    vocab = None
    if use_dummy:
        # Use dummy data for testing
        logger.info("🔄 Loading dummy data...")
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

        logger.info(f"Dataset sizes:")
        logger.info(f"  - Train: {len(train_dataset):,} samples")
        logger.info(f"  - Validation: {len(val_dataset):,} samples")
        logger.info(f"  - Test: {len(test_dataset):,} samples")

    else:
        # 실제 WMT 데이터 로딩 (분리된 언어 파일 형식)
        logger.info("🔄 Loading real WMT dataset...")

        try:
            from src.real_data_loader import load_real_wmt_data

            # 언어 설정이 없으면 기본값 설정
            if not hasattr(config, "SRC_LANG"):
                config.SRC_LANG = "en"
            if not hasattr(config, "TGT_LANG"):
                config.TGT_LANG = "de"
            if not hasattr(config, "DATASET"):
                config.DATASET = "wmt14_en_de"

            train_loader, val_loader, test_loader, vocab = load_real_wmt_data(config)

            if train_loader is None:
                logger.error("❌ Failed to load real data. Check if data files exist:")
                logger.error(
                    f"   Expected location: {config.DATA_PATH}{config.DATASET}/"
                )
                logger.error(
                    f"   Expected files: train.{config.SRC_LANG}, train.{config.TGT_LANG}, valid.{config.SRC_LANG}, valid.{config.TGT_LANG}, test.{config.SRC_LANG}, test.{config.TGT_LANG}"
                )
                logger.error("   File format: each line should contain one sentence")
                raise FileNotFoundError("Real data files not found")

        except (ImportError, FileNotFoundError) as e:
            logger.warning(f"⚠️  Error loading real data: {e}")
            logger.warning("   Falling back to dummy data...")
            return load_data(config, use_dummy=True)

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, vocab


def create_model_from_config(config: Config) -> torch.nn.Module:
    """Create Transformer model from configuration"""
    logger = logging.getLogger(__name__)
    logger.info("🤖 Creating Transformer model...")

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
        tie_weights=False,  # Tie embedding and output projection weights
        kdim=config.KDIM,
        vdim=config.VDIM,
        device=config.DEVICE,
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
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging BEFORE importing modules that use logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Console output
        ],
    )

    # Create logger for this module
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Transformer training script...")

    # Check if config file exists
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    logger.info(f"Loading configuration from: {args.config}")

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

    logger.info("\n" + "=" * 60)
    logger.info("TRANSFORMER TRAINING CONFIGURATION")
    logger.info("=" * 60)

    logger.info("\nModel Configuration:")
    logger.info(f"  Model Dimension: {config.MODEL_DIM}")
    logger.info(f"  Number of Heads: {config.NUM_HEADS}")
    logger.info(f"  Encoder Layers: {config.NUM_ENCODER_LAYERS}")
    logger.info(f"  Decoder Layers: {config.NUM_DECODER_LAYERS}")
    logger.info(f"  FFN Dimension: {config.FFN_DIM}")
    logger.info(f"  Vocabulary Size: {config.VOCAB_SIZE}")
    logger.info(f"  Max Sequence Length: {config.MAX_SEQ_LENGTH}")
    logger.info(f"  Dropout: {config.DROPOUT}")
    logger.info(f"  Key Dimension (kdim): {config.KDIM}")
    logger.info(f"  Value Dimension (vdim): {config.VDIM}")

    logger.info("\nTraining Configuration:")
    logger.info(f"  Batch Size: {config.BATCH_SIZE}")
    logger.info(f"  Learning Rate: {config.LEARNING_RATE}")
    logger.info(f"  Warmup Steps: {config.WARMUP_STEPS}")
    logger.info(f"  Max Epochs: {config.MAX_EPOCHS}")
    logger.info(f"  Gradient Clip: {config.GRADIENT_CLIP}")

    logger.info("\nData Configuration:")
    logger.info(f"  PAD Token: {config.PAD_TOKEN}")
    logger.info(f"  BOS Token: {config.BOS_TOKEN}")
    logger.info(f"  EOS Token: {config.EOS_TOKEN}")
    logger.info(f"  UNK Token: {config.UNK_TOKEN}")

    # Display dataset-specific configuration if available
    if hasattr(config, "DATASET"):
        logger.info(f"\nDataset Configuration:")
        logger.info(f"  Dataset: {config.DATASET}")

        if hasattr(config, "SRC_VOCAB_SIZE") and hasattr(config, "TGT_VOCAB_SIZE"):
            logger.info(f"  Source Vocabulary Size: {config.SRC_VOCAB_SIZE:,}")
            logger.info(f"  Target Vocabulary Size: {config.TGT_VOCAB_SIZE:,}")

        if hasattr(config, "SHARED_VOCAB"):
            logger.info(f"  Shared Vocabulary: {config.SHARED_VOCAB}")

        if hasattr(config, "ENCODING"):
            logger.info(f"  Encoding: {config.ENCODING}")

        if hasattr(config, "TOKENS_PER_BATCH"):
            logger.info(f"  Tokens per Batch: {config.TOKENS_PER_BATCH:,}")

        if hasattr(config, "TRAIN_PAIRS") and hasattr(config, "VAL_PAIRS"):
            logger.info(f"  Training Pairs: {config.TRAIN_PAIRS:,}")
            logger.info(f"  Validation Pairs: {config.VAL_PAIRS:,}")

        if hasattr(config, "SRC_LANG") and hasattr(config, "TGT_LANG"):
            logger.info(f"  Language Pair: {config.SRC_LANG} → {config.TGT_LANG}")

    logger.info("\nEvaluation Configuration:")
    logger.info(f"  Eval Every: {config.EVAL_EVERY} steps")
    logger.info(f"  Save Every: {config.SAVE_EVERY} steps")

    logger.info(f"\nDevice: {config.DEVICE}")
    logger.info(f"Random Seed: {args.seed}")

    # Print estimated model information
    config.print_estimated_model_info()

    logger.info("=" * 60)

    # Load data
    train_loader, val_loader, test_loader, vocab = load_data(
        config, use_dummy=args.use_dummy_data
    )

    # Create model
    logger.info("\n🤖 Creating Transformer model...")
    model = create_model_from_config(config)

    # Print detailed model summary with parameter information
    print_model_summary(model)

    # Create trainer
    logger.info("\n🏋️ Creating trainer...")
    trainer = create_trainer(
        model=model,
        config=config,
        vocab=vocab,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    logger.info(f"Trainer created successfully!")
    logger.info(f"  - Optimizer: Adam (betas=(0.9, 0.98), eps=1e-9)")
    logger.info(f"  - Scheduler: Warmup + Decay")
    logger.info(
        f"  - Loss Function: CrossEntropyLoss (ignore_index={config.PAD_TOKEN})"
    )

    # Start training
    logger.info("\n🚀 Starting training...")
    trainer.train()

    # Run final test
    logger.info("\n🧪 Running final test...")
    test_results = trainer.test()
    logger.info(f"Final test results: {test_results}")

    logger.info("\n✅ Training completed!")


if __name__ == "__main__":
    main()
    logging.basicConfig(filename="run.log")
