"""
Evaluation script for Transformer model
- Load configuration from JSON
- Average last N checkpoints (paper-compliant)
- Use beam search for inference
- Calculate BLEU scores
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np

from src.config import Config
from src.transformer import Transformer
from src.utils import set_seed, get_device
from train_with_config import load_config_from_json, load_data, create_model_from_config

# BLEU score calculation
try:
    from sacrebleu import BLEU

    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    print("Warning: sacrebleu not available. Install with: pip install sacrebleu")

# Alternative BLEU implementation
from collections import Counter
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleBleuCalculator:
    """Simple BLEU calculator when sacrebleu is not available"""

    @staticmethod
    def calculate_bleu(
        references: List[List[str]], hypotheses: List[List[str]], max_n: int = 4
    ) -> float:
        """Calculate BLEU score"""
        if len(references) != len(hypotheses):
            raise ValueError("Number of references and hypotheses must match")

        # Calculate n-gram precisions
        precisions = []
        for n in range(1, max_n + 1):
            precision = SimpleBleuCalculator._calculate_ngram_precision(
                references, hypotheses, n
            )
            precisions.append(precision)

        # Calculate brevity penalty
        ref_len = sum(len(ref) for ref in references)
        hyp_len = sum(len(hyp) for hyp in hypotheses)

        if hyp_len == 0:
            return 0.0

        if hyp_len > ref_len:
            brevity_penalty = 1.0
        else:
            brevity_penalty = math.exp(1 - ref_len / hyp_len)

        # Calculate geometric mean of precisions
        if any(p == 0 for p in precisions):
            return 0.0

        log_precision_sum = sum(math.log(p) for p in precisions)
        geometric_mean = math.exp(log_precision_sum / len(precisions))

        bleu = brevity_penalty * geometric_mean
        return bleu * 100  # Convert to percentage

    @staticmethod
    def _calculate_ngram_precision(
        references: List[List[str]], hypotheses: List[List[str]], n: int
    ) -> float:
        """Calculate n-gram precision"""
        ref_ngrams = Counter()
        hyp_ngrams = Counter()

        for ref, hyp in zip(references, hypotheses):
            ref_ngrams.update(SimpleBleuCalculator._get_ngrams(ref, n))
            hyp_ngrams.update(SimpleBleuCalculator._get_ngrams(hyp, n))

        if sum(hyp_ngrams.values()) == 0:
            return 0.0

        # Count matches (clipped)
        matches = 0
        for ngram in hyp_ngrams:
            matches += min(hyp_ngrams[ngram], ref_ngrams[ngram])

        return matches / sum(hyp_ngrams.values())

    @staticmethod
    def _get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Extract n-grams from tokens"""
        if len(tokens) < n:
            return []
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class BeamSearchDecoder:
    """Beam search decoder for Transformer"""

    def __init__(
        self,
        model: torch.nn.Module,
        beam_size: int = 4,
        max_length: int = 100,
        length_penalty: float = 0.6,
        early_stopping: bool = True,
    ):
        self.model = model
        self.beam_size = beam_size
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.device = next(model.parameters()).device

    def decode(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        bos_token: int,
        eos_token: int,
        pad_token: int,
    ) -> List[int]:
        """
        Beam search decoding

        Args:
            src: Source sequence [1, src_len]
            src_mask: Source mask [1, src_len]
            bos_token: Beginning of sequence token ID
            eos_token: End of sequence token ID
            pad_token: Padding token ID

        Returns:
            List of token IDs representing the best decoded sequence
        """
        batch_size = src.size(0)
        assert batch_size == 1, "Beam search currently supports batch_size=1 only"

        # Store source for beam search
        self.src = src
        self.pad_token = pad_token

        # Initialize beam
        # Each beam item: (sequence, score, finished)
        beams = [(torch.tensor([bos_token], device=self.device), 0.0, False)]
        finished_beams = []

        for step in range(self.max_length):
            new_beams = []

            for seq, score, finished in beams:
                logger.info("test0")
                if finished:
                    finished_beams.append((seq, score, finished))
                    continue

                # Prepare decoder input
                tgt = tgt.unsqueeze(0)  # [1, seq_len]
                tgt_len = tgt.size(1)

                # Create target mask (causal mask)
                tgt_mask = torch.tril(
                    torch.ones(tgt_len, tgt_len, device=self.device)
                ).bool()
                # Convert to attention mask format (False for allowed, True for masked)
                # tgt_mask = ~tgt_mask  # Invert for attention mask

                # logger.info(self.src.shape)
                # Forward pass using the complete model
                logger.info("test1")
                with torch.no_grad():
                    # logger.debug(self.src.shape)

                    # Use full model forward pass
                    model_output = self.model(self.src, tgt)  # [1, seq_len, vocab_size]
                    logits = model_output.reshape(
                        -1, model_output.size(-1)
                    )  # [1, vocab_size] - last position

                    log_probs = F.log_softmax(logits, dim=-1)  # [1, vocab_size]

                # Get top-k candidates
                top_log_probs, top_indices = log_probs.topk(self.beam_size, dim=-1)

                # Debug: log predictions for first step
                # if step == 0 and len(beams) == 1:
                # print(f"Step {step}: Top predictions: {top_indices.cpu().numpy()}")
                # print(f"Step {step}: Top log_probs: {top_log_probs.cpu().numpy()}")
                logger.info("test2")
                for i in range(self.beam_size):
                    token_id = top_indices[0, i].item()
                    token_log_prob = top_log_probs[0, i].item()

                    new_seq = torch.cat(
                        [seq, torch.tensor([token_id], device=self.device)]
                    )
                    new_score = score + token_log_prob

                    # Check if finished
                    is_finished = token_id == eos_token

                    # Apply length penalty if finished
                    if is_finished and self.length_penalty > 0:
                        length_penalty = ((5 + len(new_seq)) / 6) ** self.length_penalty
                        new_score = new_score / length_penalty

                    new_beams.append((new_seq, new_score, is_finished))

            # Select top beams
            if new_beams:
                # Sort by score (higher is better)
                new_beams.sort(key=lambda x: x[1], reverse=True)
                beams = new_beams[: self.beam_size]

            # Early stopping check
            if self.early_stopping and all(finished for _, _, finished in beams):
                print(f"Early stopping at step {step}")
                break

        # Combine beams and finished beams
        all_beams = beams + finished_beams
        if not all_beams:
            return [eos_token]

        # Select best beam
        best_beam = max(all_beams, key=lambda x: x[1])
        return best_beam[0].tolist()


class ModelEvaluator:
    """Evaluator for Transformer model"""

    def __init__(self, config: Config, device: str = "auto"):
        self.config = config
        self.device = torch.device(device if device != "auto" else get_device())

        # Setup logging
        self._setup_logging()

        # Initialize model
        self.model = None
        self.vocab = None  # Will be set when loading data

        # Beam search parameters (from config or paper defaults)
        self.beam_size = getattr(config, "BEAM_SIZE", 4)
        self.length_penalty = getattr(config, "LENGTH_PENALTY", 0.6)
        self.max_output_length_offset = 50  # input_length + 50

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def load_and_average_checkpoints(
        self, checkpoint_paths: List[str]
    ) -> torch.nn.Module:
        """
        Load and average multiple checkpoints (as mentioned in the paper)

        Args:
            checkpoint_paths: List of checkpoint file paths

        Returns:
            Model with averaged weights
        """
        self.logger.info(
            f"Loading and averaging {len(checkpoint_paths)} checkpoints..."
        )

        # Load first checkpoint to get model structure
        first_checkpoint = torch.load(checkpoint_paths[0], map_location=self.device)

        # Create model
        self.model = create_model_from_config(self.config)
        self.model.to(self.device)

        # Initialize averaged state dict
        averaged_state_dict = {}

        # Load and average all checkpoints
        for i, checkpoint_path in enumerate(checkpoint_paths):
            self.logger.info(
                f"Loading checkpoint {i+1}/{len(checkpoint_paths)}: {checkpoint_path}"
            )

            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint["model_state_dict"]

            if i == 0:
                # Initialize with first checkpoint
                for key, value in state_dict.items():
                    averaged_state_dict[key] = value.clone()
            else:
                # Add to average
                for key, value in state_dict.items():
                    if key in averaged_state_dict:
                        averaged_state_dict[key] += value
                    else:
                        averaged_state_dict[key] = value.clone()

        # Divide by number of checkpoints to get average
        for key in averaged_state_dict:
            averaged_state_dict[key] /= len(checkpoint_paths)

        # Load averaged weights
        self.model.load_state_dict(averaged_state_dict)
        self.model.eval()

        self.logger.info("Model averaging completed successfully!")
        return self.model

    def find_last_n_checkpoints(
        self, checkpoint_dir: str, n_checkpoints: int
    ) -> List[str]:
        """
        Find the last N checkpoints in a directory

        Args:
            checkpoint_dir: Directory containing checkpoints
            n_checkpoints: Number of checkpoints to find

        Returns:
            List of checkpoint paths sorted by step number
        """
        checkpoint_dir = Path(checkpoint_dir)

        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

        # Find all checkpoint files
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_step_*.pt"))

        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

        # Extract step numbers and sort
        def extract_step(path):
            filename = path.name
            # checkpoint_step_1000.pt -> 1000
            try:
                step = int(filename.split("_")[-1].split(".")[0])
                return step
            except (ValueError, IndexError):
                return 0

        checkpoint_files.sort(key=extract_step)

        # Get last N checkpoints
        last_n = (
            checkpoint_files[-n_checkpoints:]
            if len(checkpoint_files) >= n_checkpoints
            else checkpoint_files
        )

        self.logger.info(f"Found {len(checkpoint_files)} total checkpoints")
        self.logger.info(f"Using last {len(last_n)} checkpoints for averaging:")
        for cp in last_n:
            step = extract_step(cp)
            self.logger.info(f"  - {cp.name} (step {step})")

        return [str(cp) for cp in last_n]

    def evaluate_on_test_set(self, test_loader, vocab) -> Dict[str, float]:
        """
        Evaluate model on test set using beam search and calculate BLEU

        Args:
            test_loader: Test data loader
            vocab: Vocabulary object

        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError(
                "Model not loaded. Call load_and_average_checkpoints first."
            )

        self.model.eval()
        self.vocab = vocab

        # Initialize beam search decoder
        beam_decoder = BeamSearchDecoder(
            model=self.model,
            beam_size=self.beam_size,
            max_length=512,  # Will be adjusted per input
            length_penalty=self.length_penalty,
            early_stopping=True,
        )

        # Collect predictions and references
        all_predictions = []
        all_references = []

        total_batches = len(test_loader)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
                # Move batch to device
                src = batch["src"].to(self.device)  # [batch_size, src_len]
                tgt_y = batch["tgt_y"].to(self.device)  # [batch_size, tgt_len]

                batch_size = src.size(0)

                for i in range(batch_size):
                    # Get single example
                    src_seq = src[i : i + 1]  # [src_len]
                    tgt_seq = tgt_y[i]  # [tgt_len]

                    # 🔍 디버깅: 각 배치의 첫 번째 문장 상세 정보 출력
                    if i == 0:  # 각 배치의 첫 번째 문장만
                        self._debug_batch_sample(batch_idx, src_seq, tgt_seq, vocab)

                    # Create source mask (ignore padding)
                    src_mask = (src_seq != self.config.PAD_TOKEN).unsqueeze(
                        1
                    )  # [1, 1, src_len]

                    # Set max output length: input_length + 50
                    max_length = src_seq.size(1) + self.max_output_length_offset
                    beam_decoder.max_length = min(max_length, 512)

                    # Generate prediction using beam search
                    try:
                        pred_tokens = beam_decoder.decode(
                            src=src_seq,
                            tgt=tgt_seq,
                            src_mask=src_mask,
                            tgt_mask=None,
                            bos_token=self.config.BOS_TOKEN,
                            eos_token=self.config.EOS_TOKEN,
                            pad_token=self.config.PAD_TOKEN,
                        )

                        # 🔍 디버깅: 첫 번째 문장의 예측 결과 상세 정보
                        if i == 0:
                            self._debug_prediction_result(
                                batch_idx, pred_tokens, tgt_seq, vocab
                            )

                    except Exception as e:
                        self.logger.warning(
                            f"Error in beam search for batch {batch_idx}, example {i}: {e}"
                        )
                        pred_tokens = [self.config.EOS_TOKEN]

                    # Remove BOS/EOS tokens and convert to text
                    pred_tokens_clean = [
                        t
                        for t in pred_tokens
                        if t
                        not in [
                            self.config.BOS_TOKEN,
                            self.config.EOS_TOKEN,
                            self.config.PAD_TOKEN,
                        ]
                    ]
                    ref_tokens = [
                        t.item()
                        for t in tgt_seq
                        if t.item()
                        not in [
                            self.config.BOS_TOKEN,
                            self.config.EOS_TOKEN,
                            self.config.PAD_TOKEN,
                        ]
                    ]

                    # Convert to text using vocabulary
                    try:
                        if hasattr(vocab, "decode"):
                            # BPE vocabulary
                            if pred_tokens_clean:  # Check if not empty
                                pred_text = vocab.decode(pred_tokens_clean)
                            else:
                                pred_text = ""

                            if ref_tokens:  # Check if not empty
                                ref_text = vocab.decode(ref_tokens)
                            else:
                                ref_text = ""

                            # Tokenize for BLEU calculation
                            pred_words = (
                                pred_text.split() if pred_text.strip() else ["<EMPTY>"]
                            )
                            ref_words = (
                                ref_text.split() if ref_text.strip() else ["<EMPTY>"]
                            )
                        else:
                            # Simple vocabulary
                            if pred_tokens_clean:
                                pred_words = vocab.decode(pred_tokens_clean)
                            else:
                                pred_words = ["<EMPTY>"]

                            if ref_tokens:
                                ref_words = vocab.decode(ref_tokens)
                            else:
                                ref_words = ["<EMPTY>"]

                    except Exception as e:
                        self.logger.warning(f"Error in vocabulary decoding: {e}")
                        pred_words = ["<UNK>"]
                        ref_words = ["<UNK>"]

                    all_predictions.append(pred_words)
                    all_references.append(ref_words)

                # Log progress
                if (batch_idx + 1) % 100 == 0:
                    self.logger.info(
                        f"Processed {batch_idx + 1}/{total_batches} batches"
                    )

        # Calculate BLEU score
        self.logger.info("Calculating BLEU score...")

        if SACREBLEU_AVAILABLE:
            # Use sacrebleu
            bleu = BLEU()

            # Convert to strings for sacrebleu
            pred_strings = [" ".join(pred) for pred in all_predictions]
            ref_strings = [
                [" ".join(ref)] for ref in all_references
            ]  # sacrebleu expects list of lists

            bleu_score = bleu.corpus_score(pred_strings, ref_strings).score
        else:
            # Use simple BLEU calculator
            bleu_score = SimpleBleuCalculator.calculate_bleu(
                all_references, all_predictions
            )

        # Log some examples
        self.logger.info("\nSample translations:")
        for i in range(min(5, len(all_predictions))):
            self.logger.info(f"Reference: {' '.join(all_references[i])}")
            self.logger.info(f"Prediction: {' '.join(all_predictions[i])}")
            self.logger.info("-" * 50)

        results = {"bleu_score": bleu_score, "num_examples": len(all_predictions)}

        return results

    def _debug_batch_sample(self, batch_idx, src_seq, tgt_seq, vocab):
        """각 배치의 첫 번째 문장에 대한 상세 디버깅 정보 출력"""
        if batch_idx >= 10:  # 처음 10배치만
            return

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🔍 BATCH {batch_idx} - EVALUATION DEBUG")
        self.logger.info(f"{'='*80}")

        # 원본 토큰 정보
        src_tokens = src_seq[0].cpu().tolist()  # [src_len]
        tgt_tokens = tgt_seq.cpu().tolist()  # [tgt_len]

        # PAD 토큰 제거
        src_tokens_clean = [t for t in src_tokens if t != self.config.PAD_TOKEN]
        tgt_tokens_clean = [
            t
            for t in tgt_tokens
            if t
            not in [self.config.PAD_TOKEN, self.config.BOS_TOKEN, self.config.EOS_TOKEN]
        ]

        self.logger.info(f"📤 Source (원본 입력):")
        self.logger.info(
            f"   토큰 ID: {src_tokens_clean[:20]}{'...' if len(src_tokens_clean) > 20 else ''}"
        )

        # 텍스트 변환
        try:
            if hasattr(vocab, "decode"):
                src_text = vocab.decode(src_tokens_clean)
                tgt_text = vocab.decode(tgt_tokens_clean)
            else:
                src_text = " ".join([str(t) for t in src_tokens_clean])
                tgt_text = " ".join([str(t) for t in tgt_tokens_clean])

            self.logger.info(f"   텍스트: '{src_text}'")

        except Exception as e:
            self.logger.warning(f"   텍스트 변환 실패: {e}")
            src_text = "<변환실패>"
            tgt_text = "<변환실패>"

        self.logger.info(f"📥 Target (예상 출력):")
        self.logger.info(
            f"   토큰 ID: {tgt_tokens_clean[:20]}{'...' if len(tgt_tokens_clean) > 20 else ''}"
        )
        self.logger.info(f"   텍스트: '{tgt_text}'")

        self.logger.info(
            f"🏷️  특수 토큰: PAD={self.config.PAD_TOKEN}, BOS={self.config.BOS_TOKEN}, EOS={self.config.EOS_TOKEN}"
        )
        self.logger.info(
            f"📊 길이 정보: Source={len(src_tokens_clean)}, Target={len(tgt_tokens_clean)}"
        )

    def _debug_prediction_result(self, batch_idx, pred_tokens, tgt_seq, vocab):
        """예측 결과에 대한 상세 디버깅 정보 출력"""
        if batch_idx >= 10:  # 처음 10배치만
            return

        # 예측 토큰 정리
        pred_tokens_clean = [
            t
            for t in pred_tokens
            if t
            not in [self.config.BOS_TOKEN, self.config.EOS_TOKEN, self.config.PAD_TOKEN]
        ]

        # 타겟 토큰 정리
        tgt_tokens = tgt_seq.cpu().tolist()
        tgt_tokens_clean = [
            t
            for t in tgt_tokens
            if t
            not in [self.config.PAD_TOKEN, self.config.BOS_TOKEN, self.config.EOS_TOKEN]
        ]

        self.logger.info(f"🤖 Prediction (실제 예측):")
        self.logger.info(f"   원본 예측: {pred_tokens}")
        self.logger.info(
            f"   정제된 토큰: {pred_tokens_clean[:20]}{'...' if len(pred_tokens_clean) > 20 else ''}"
        )

        # 텍스트 변환
        try:
            if hasattr(vocab, "decode"):
                pred_text = (
                    vocab.decode(pred_tokens_clean) if pred_tokens_clean else "<EMPTY>"
                )
                tgt_text = (
                    vocab.decode(tgt_tokens_clean) if tgt_tokens_clean else "<EMPTY>"
                )
            else:
                pred_text = (
                    " ".join([str(t) for t in pred_tokens_clean])
                    if pred_tokens_clean
                    else "<EMPTY>"
                )
                tgt_text = (
                    " ".join([str(t) for t in tgt_tokens_clean])
                    if tgt_tokens_clean
                    else "<EMPTY>"
                )

            self.logger.info(f"   예측 텍스트: '{pred_text}'")

        except Exception as e:
            self.logger.warning(f"   예측 텍스트 변환 실패: {e}")
            pred_text = "<변환실패>"
            tgt_text = "<변환실패>"

        # 토큰 레벨 비교 (처음 10개)
        self.logger.info(f"🔄 토큰 비교 (처음 10개):")
        max_compare = min(10, max(len(pred_tokens_clean), len(tgt_tokens_clean)))

        for j in range(max_compare):
            pred_token = pred_tokens_clean[j] if j < len(pred_tokens_clean) else "<PAD>"
            tgt_token = tgt_tokens_clean[j] if j < len(tgt_tokens_clean) else "<PAD>"

            match_symbol = "✅" if pred_token == tgt_token else "❌"

            # 개별 토큰 텍스트 변환
            try:
                if (
                    hasattr(vocab, "decode")
                    and isinstance(pred_token, int)
                    and isinstance(tgt_token, int)
                ):
                    pred_word = (
                        vocab.decode([pred_token]) if pred_token != "<PAD>" else "<PAD>"
                    )
                    tgt_word = (
                        vocab.decode([tgt_token]) if tgt_token != "<PAD>" else "<PAD>"
                    )
                else:
                    pred_word = str(pred_token)
                    tgt_word = str(tgt_token)
            except:
                pred_word = str(pred_token)
                tgt_word = str(tgt_token)

            self.logger.info(
                f"   [{j:2d}] {match_symbol} 예측:{pred_token:>6}('{pred_word}') vs 정답:{tgt_token:>6}('{tgt_word}')"
            )

        # 길이 비교
        self.logger.info(
            f"📏 길이 비교: 예측={len(pred_tokens_clean)}, 정답={len(tgt_tokens_clean)}"
        )

        # 전체 텍스트 비교
        self.logger.info(f"📝 전체 비교:")
        self.logger.info(f"   정답: '{tgt_text}'")
        self.logger.info(f"   예측: '{pred_text}'")
        self.logger.info(f"{'='*80}\n")


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate Transformer model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory containing checkpoints",
    )
    parser.add_argument(
        "--n-checkpoints",
        type=int,
        default=5,
        help="Number of last checkpoints to average (5 for base, 20 for big model)",
    )
    parser.add_argument(
        "--use-dummy-data",
        action="store_true",
        help="Use dummy data instead of real data",
    )
    parser.add_argument(
        "--beam-size", type=int, default=4, help="Beam size for beam search"
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=0.6,
        help="Length penalty for beam search",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Load configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    print(f"Loading configuration from: {args.config}")
    config = load_config_from_json(args.config)

    # Override device if specified
    if args.device != "auto":
        config.DEVICE = args.device
    else:
        device = get_device()
        config.DEVICE = str(device)

    # Override command line arguments with config values if not explicitly set
    if args.n_checkpoints == 5:  # Default value
        args.n_checkpoints = getattr(config, "N_CHECKPOINTS", 5)
    if args.beam_size == 4:  # Default value
        args.beam_size = getattr(config, "BEAM_SIZE", 4)
    if args.length_penalty == 0.6:  # Default value
        args.length_penalty = getattr(config, "LENGTH_PENALTY", 0.6)

    print("\n" + "=" * 60)
    print("TRANSFORMER MODEL EVALUATION")
    print("=" * 60)
    print(f"Model: {config.EXPERIMENT_NAME}")
    print(f"Dataset: {config.DATASET}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Number of checkpoints to average: {args.n_checkpoints}")
    print(f"Beam size: {args.beam_size}")
    print(f"Length penalty: {args.length_penalty}")
    print(f"Device: {config.DEVICE}")
    print("=" * 60)

    # Create evaluator
    evaluator = ModelEvaluator(config, device=config.DEVICE)
    evaluator.beam_size = args.beam_size
    evaluator.length_penalty = args.length_penalty

    # Find and load checkpoints
    try:
        checkpoint_paths = evaluator.find_last_n_checkpoints(
            args.checkpoint_dir, args.n_checkpoints
        )
        model = evaluator.load_and_average_checkpoints(checkpoint_paths)
    except FileNotFoundError as e:
        print(f"Error loading checkpoints: {e}")
        return

    # Load test data
    print("\n🔄 Loading test data...")
    try:
        train_loader, val_loader, test_loader, vocab = load_data(
            config, use_dummy=args.use_dummy_data
        )

        if test_loader is None:
            print("❌ Failed to load test data")
            return

        # Vocab is now directly returned from load_data function
        if vocab is None:
            # For dummy data, create a simple vocab as fallback
            from src.data_loader import DummyTokenizer

            vocab = DummyTokenizer(config.VOCAB_SIZE)

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Run evaluation
    print(f"\n🧪 Starting evaluation on {len(test_loader)} test batches...")

    try:
        results = evaluator.evaluate_on_test_set(test_loader, vocab)

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"BLEU Score: {results['bleu_score']:.2f}")
        print(f"Number of examples: {results['num_examples']:,}")
        print("=" * 60)

        # Save results
        results_file = Path(args.checkpoint_dir) / "evaluation_results.json"
        results_data = {
            "bleu_score": results["bleu_score"],
            "num_examples": results["num_examples"],
            "config": config.to_dict(),
            "beam_size": args.beam_size,
            "length_penalty": args.length_penalty,
            "n_checkpoints_averaged": args.n_checkpoints,
            "checkpoint_paths": checkpoint_paths,
        }

        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"\n✅ Results saved to: {results_file}")

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
