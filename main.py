"""
Main training script for Transformer model
"""

import os
import torch
import argparse
from typing import Optional

from src.config import Config
from src.data_loader import (
    TransformerDataset,
    create_data_loader,
    load_dummy_data,
    DummyTokenizer,
)
from src.trainer import create_trainer
from src.utils import set_seed, get_device, print_model_summary
from src.transformer import Transformer  # You'll need to implement this


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Transformer model")

    # Model parameters
    parser.add_argument("--model_dim", type=int, default=512, help="Model dimension")
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_encoder_layers", type=int, default=6, help="Number of encoder layers"
    )
    parser.add_argument(
        "--num_decoder_layers", type=int, default=6, help="Number of decoder layers"
    )
    parser.add_argument(
        "--ffn_dim", type=int, default=2048, help="Feed-forward network dimension"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--kdim", type=int, default=64, help="Key dimension for attention"
    )
    parser.add_argument(
        "--vdim", type=int, default=64, help="Value dimension for attention"
    )

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=4000, help="Number of warmup steps"
    )

    # Data parameters
    parser.add_argument("--vocab_size", type=int, default=30000, help="Vocabulary size")
    parser.add_argument(
        "--max_seq_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--dataset", type=str, default="wmt14_en_de", help="Dataset name"
    )
    parser.add_argument("--src_lang", type=str, default="en", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="de", help="Target language")

    # File paths
    parser.add_argument(
        "--data_path", type=str, default="data/", help="Path to data directory"
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="checkpoints/",
        help="Path to save model checkpoints",
    )
    parser.add_argument(
        "--log_path", type=str, default="logs/", help="Path to save logs"
    )

    # Training options
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument("--test_only", action="store_true", help="Only run testing")
    parser.add_argument(
        "--use_dummy_data", action="store_true", help="Use dummy data for testing"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training",
    )

    return parser.parse_args()


def setup_config(args) -> Config:
    """Setup configuration from arguments"""
    config_dict = {
        "MODEL_DIM": args.model_dim,
        "NUM_HEADS": args.num_heads,
        "NUM_ENCODER_LAYERS": args.num_encoder_layers,
        "NUM_DECODER_LAYERS": args.num_decoder_layers,
        "FFN_DIM": args.ffn_dim,
        "DROPOUT": args.dropout,
        "KDIM": args.kdim,
        "VDIM": args.vdim,
        "BATCH_SIZE": args.batch_size,
        "LEARNING_RATE": args.learning_rate,
        "MAX_EPOCHS": args.max_epochs,
        "WARMUP_STEPS": args.warmup_steps,
        "VOCAB_SIZE": args.vocab_size,
        "MAX_SEQ_LENGTH": args.max_seq_length,
        "DATA_PATH": args.data_path,
        "MODEL_SAVE_PATH": args.model_save_path,
        "LOG_PATH": args.log_path,
        "DATASET": args.dataset,
        "SRC_LANG": args.src_lang,
        "TGT_LANG": args.tgt_lang,
    }

    # Setup device
    if args.device == "auto":
        device = get_device()
        config_dict["DEVICE"] = str(device)
    else:
        config_dict["DEVICE"] = args.device

    return Config(**config_dict)


def load_data(config: Config, use_dummy: bool = True):
    """Load training, validation, and test data"""

    if use_dummy:
        # Use dummy data for testing
        print("Loading dummy data...")
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

        # Create datasets
        train_dataset = TransformerDataset(
            train_src,
            train_tgt,
            config.VOCAB_SIZE,
            config.VOCAB_SIZE,
            config.MAX_SEQ_LENGTH,
        )

        val_dataset = TransformerDataset(
            val_src,
            val_tgt,
            config.VOCAB_SIZE,
            config.VOCAB_SIZE,
            config.MAX_SEQ_LENGTH,
        )

        test_dataset = TransformerDataset(
            test_src,
            test_tgt,
            config.VOCAB_SIZE,
            config.VOCAB_SIZE,
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

    else:
        # 실제 WMT 데이터 로딩
        print("Loading real WMT data...")

        try:
            from src.real_data_loader import load_real_wmt_data

            # config에 DATASET 속성이 없으면 기본값 설정
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


def create_model(config: Config) -> torch.nn.Module:
    """Create Transformer model"""
    from src.transformer import Transformer

    print("Creating Transformer model...")
    print(f"Model configuration:")
    print(f"  - Model dimension: {config.MODEL_DIM}")
    print(f"  - Number of heads: {config.NUM_HEADS}")
    print(f"  - Encoder layers: {config.NUM_ENCODER_LAYERS}")
    print(f"  - Decoder layers: {config.NUM_DECODER_LAYERS}")
    print(f"  - FFN dimension: {config.FFN_DIM}")
    print(f"  - Source vocabulary size: {config.VOCAB_SIZE}")
    print(f"  - Target vocabulary size: {config.VOCAB_SIZE}")
    print(f"  - Max sequence length: {config.MAX_SEQ_LENGTH}")
    print(f"  - Dropout: {config.DROPOUT}")
    print(f"  - Key dimension (kdim): {config.KDIM}")
    print(f"  - Value dimension (vdim): {config.VDIM}")

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
        tie_weights=True,  # Tie embedding and output projection weights
        kdim=config.KDIM,
        vdim=config.VDIM,
        device=config.DEVICE,
    )

    return model


def main():
    """Main function"""
    args = parse_arguments()

    # Set random seed
    set_seed(args.seed)

    # Setup configuration
    config = setup_config(args)
    print("Configuration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")

    # Load data
    train_loader, val_loader, test_loader, vocab = load_data(
        config, use_dummy=args.use_dummy_data
    )

    # Create model
    model = create_model(config)
    print_model_summary(model)

    # Create trainer
    trainer = create_trainer(
        model=model,
        config=config,
        vocab=vocab,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Run training or testing
    if args.test_only:
        print("Running test...")
        test_results = trainer.test()
        print(f"Test results: {test_results}")
    else:
        print("Starting training...")
        trainer.train()

        # Run final test
        print("Running final test...")
        test_results = trainer.test()
        print(f"Final test results: {test_results}")


if __name__ == "__main__":
    main()
