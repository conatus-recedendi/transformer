"""
Configuration file for Transformer model training
"""


class Config:
    """Base configuration class"""

    # Model parameters
    MODEL_DIM = 512
    NUM_HEADS = 8
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    FFN_DIM = 2048
    DROPOUT = 0.1
    MAX_SEQ_LENGTH = 512

    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WARMUP_STEPS = 4000
    MAX_EPOCHS = 100
    GRADIENT_CLIP = 1.0

    # Data parameters
    VOCAB_SIZE = 30000
    PAD_TOKEN = 0
    BOS_TOKEN = 1
    EOS_TOKEN = 2
    UNK_TOKEN = 3

    # File paths
    DATA_PATH = "data/"
    MODEL_SAVE_PATH = "checkpoints/"
    LOG_PATH = "logs/"

    # Device
    DEVICE = "cuda"  # or "cpu"

    # Checkpoint
    SAVE_EVERY = 1000
    EVAL_EVERY = 500

    def __init__(self, **kwargs):
        """Update config with custom parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")

    def to_dict(self):
        """Convert config to dictionary"""
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }
