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

        # File paths (matching paths section in JSON)
        self.DATA_PATH = "data/"
        self.MODEL_SAVE_PATH = "checkpoints/base/"
        self.LOG_PATH = "logs/base/"

        # Device (not in JSON but needed for runtime)
        self.DEVICE = "cuda"  # or "cpu", "mps", "auto"

        # Update with any provided kwargs
        self.update(**kwargs)

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

        # Paths
        if "paths" in json_data:
            paths = json_data["paths"]
            config.DATA_PATH = paths.get("data_path", config.DATA_PATH)
            config.MODEL_SAVE_PATH = paths.get(
                "model_save_path", config.MODEL_SAVE_PATH
            )
            config.LOG_PATH = paths.get("log_path", config.LOG_PATH)

        return config
