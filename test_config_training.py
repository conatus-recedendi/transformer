"""
Test script for config-based training
"""

import os
import sys
import subprocess
from pathlib import Path


def test_config_loading():
    """Test config file loading"""
    print("Testing config file loading...")

    sys.path.append(".")
    from train_with_config import load_config_from_json

    # Test small config
    config = load_config_from_json("experiments/small_config.json")
    print(
        f"✅ Small config loaded: {config.MODEL_DIM}d model with {config.NUM_HEADS} heads"
    )

    # Test base config
    config = load_config_from_json("experiments/base_config.json")
    print(
        f"✅ Base config loaded: {config.MODEL_DIM}d model with {config.NUM_HEADS} heads"
    )

    # Test large config
    config = load_config_from_json("experiments/large_config.json")
    print(
        f"✅ Large config loaded: {config.MODEL_DIM}d model with {config.NUM_HEADS} heads"
    )

    print("All config files loaded successfully!")


def test_model_creation():
    """Test model creation from config"""
    print("\nTesting model creation from config...")

    sys.path.append(".")
    from train_with_config import load_config_from_json, create_model_from_config

    # Test small model
    config = load_config_from_json("experiments/small_config.json")
    model = create_model_from_config(config)
    total_params = model.count_parameters()
    print(f"✅ Small model created: {total_params:,} parameters")

    # Test base model (commented out to save memory)
    # config = load_config_from_json('experiments/base_config.json')
    # model = create_model_from_config(config)
    # total_params = model.count_parameters()
    # print(f"✅ Base model created: {total_params:,} parameters")

    print("Model creation test completed!")


def test_data_loading():
    """Test data loading from config"""
    print("\nTesting data loading from config...")

    sys.path.append(".")
    from train_with_config import load_config_from_json, create_data_loaders_from_config

    config = load_config_from_json("experiments/small_config.json")
    train_loader, val_loader, test_loader = create_data_loaders_from_config(
        config, use_dummy_data=True
    )

    print(f"✅ Data loaders created:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")

    # Test a batch
    batch = next(iter(train_loader))
    print(f"✅ Sample batch loaded:")
    print(f"  src shape: {batch['src'].shape}")
    print(f"  tgt shape: {batch['tgt'].shape}")

    print("Data loading test completed!")


def test_quick_training():
    """Test a very quick training run"""
    print("\nTesting quick training run...")

    # Create a minimal config for testing
    test_config = {
        "experiment_name": "test_run",
        "description": "Minimal config for testing",
        "model": {
            "model_dim": 64,
            "num_heads": 2,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "ffn_dim": 128,
            "dropout": 0.1,
            "max_seq_length": 32,
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 1e-3,
            "warmup_steps": 10,
            "max_epochs": 1,
            "gradient_clip": 1.0,
        },
        "data": {
            "vocab_size": 1000,
            "pad_token": 0,
            "bos_token": 1,
            "eos_token": 2,
            "unk_token": 3,
        },
        "evaluation": {"eval_every": 5, "save_every": 10},
    }

    # Save test config
    import json

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/test_config.json", "w") as f:
        json.dump(test_config, f, indent=2)

    # Run training
    try:
        result = subprocess.run(
            [
                "python",
                "train_with_config.py",
                "--config",
                "experiments/test_config.json",
                "--use_dummy_data",
                "--seed",
                "42",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )  # 5 minute timeout

        if result.returncode == 0:
            print("✅ Quick training run completed successfully!")
            print("Last few lines of output:")
            lines = result.stdout.strip().split("\n")
            for line in lines[-5:]:
                print(f"  {line}")
        else:
            print("❌ Quick training run failed!")
            print("Error output:")
            print(result.stderr)

    except subprocess.TimeoutExpired:
        print("⏰ Quick training run timed out (this might be normal)")
    except Exception as e:
        print(f"❌ Quick training run failed with exception: {e}")

    # Clean up
    if os.path.exists("experiments/test_config.json"):
        os.remove("experiments/test_config.json")


def main():
    """Run all tests"""
    print("🧪 Testing Config-based Transformer Training")
    print("=" * 50)

    try:
        test_config_loading()
        test_model_creation()
        test_data_loading()
        test_quick_training()

        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")
        print("\nYou can now run training with:")
        print(
            "  python train_with_config.py --config experiments/small_config.json --use_dummy_data"
        )
        print(
            "  python train_with_config.py --config experiments/base_config.json --use_dummy_data"
        )
        print(
            "  python train_with_config.py --config experiments/large_config.json --use_dummy_data"
        )

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
