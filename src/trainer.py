"""
Training utilities and trainer class for Transformer model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
from typing import Dict, Any, Optional, Tuple
import logging
from tqdm import tqdm

from .config import Config
from .utils import save_checkpoint, load_checkpoint, warmup_lr_schedule


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Loss (논문에서 사용)"""

    def __init__(
        self, num_classes: int, smoothing: float = 0.1, ignore_index: int = -100
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        """
        Args:
            pred: [N, C] where C = number of classes
            target: [N] where each value is 0 <= target[i] <= C-1
        """
        pred = pred.log_softmax(dim=-1)

        # Create one-hot encoding
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))

        # Mask for ignore_index
        mask = target != self.ignore_index
        target_masked = target.masked_fill(~mask, 0)

        true_dist.scatter_(1, target_masked.unsqueeze(1), self.confidence)

        # Apply mask to both prediction and target
        pred_masked = pred * mask.unsqueeze(1).float()
        true_dist_masked = true_dist * mask.unsqueeze(1).float()

        return torch.sum(-true_dist_masked * pred_masked) / mask.sum().float()


class LRScheduler:
    """Learning rate scheduler for Transformer (Warmup + Decay)"""

    def __init__(
        self, optimizer, d_model: int, warmup_steps: int = 4000, batch_size=128
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
        self.batch_size = batch_size

    def step(self):
        """Update learning rate"""
        self.step_num += 1 / (
            self.batch_size / 128
        )  # step num added by batch_size /128
        lr = self._calculate_lr()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _calculate_lr(self):
        """Calculate learning rate based on step number"""
        step_num = self.step_num
        warmup_steps = self.warmup_steps
        d_model = self.d_model

        lr = (d_model**-0.5) * min(step_num**-0.5, step_num * warmup_steps**-1.5)
        return lr


class Trainer:
    """Trainer class for Transformer model"""

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        vocab: Any,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
    ):
        """
        Args:
            model: Transformer model
            config: Configuration object
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
        """
        self.model = model
        self.config = config
        self.vocab = vocab
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Setup device
        self.device = torch.device(
            config.DEVICE if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Setup optimizer and scheduler (논문의 정확한 설정)
        # 초기 learning rate는 0으로 설정하고 scheduler가 관리
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=1e-8,  # 매우 작은 초기값, scheduler가 실제 lr 관리
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        self.scheduler = LRScheduler(
            self.optimizer, config.MODEL_DIM, config.WARMUP_STEPS, config.BATCH_SIZE
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_TOKEN)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Setup logging
        self._setup_logging()

        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.LOG_PATH, exist_ok=True)

    def _setup_logging(self):
        """Setup logging configuration"""
        # Ensure log directory exists
        os.makedirs(self.config.LOG_PATH, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(self.config.LOG_PATH, "training.log")),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward pass
            self.scheduler.step()
            loss = self._forward_step(batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.GRADIENT_CLIP
            )

            # Update parameters
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{total_loss / (batch_idx + 1):.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                }
            )

            # Validation and saving
            if self.global_step % self.config.EVAL_EVERY == 0 and self.val_loader:
                val_metrics = self.validate()
                self.logger.info(
                    f"Step {self.global_step} - Val Loss: {val_metrics['loss']:.4f}"
                )

            if self.global_step % self.config.SAVE_EVERY == 0:
                self._save_checkpoint()

        return {"loss": total_loss / num_batches}

    def _forward_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward step for training"""
        src = batch["src"]  # [batch_size, src_len]
        tgt = batch["tgt"]  # [batch_size, tgt_len] - decoder input (with BOS)

        # 데이터 로더에서 제공하는 tgt_y 사용 (target output with EOS)
        if "tgt_y" in batch:
            tgt_output = batch[
                "tgt_y"
            ]  # [batch_size, tgt_len] - target output (with EOS)
            tgt_input = tgt  # decoder input (with BOS, without EOS)
        else:
            # 백워드 호환성: tgt_y가 없으면 기존 방식 사용
            tgt_input = tgt[:, :-1]  # [batch_size, tgt_len-1] - remove last token
            tgt_output = tgt[:, 1:]  # [batch_size, tgt_len-1] - remove first token

        # Forward pass through model
        logits = self.model(src, tgt_input)  # [batch_size, tgt_len-1, vocab_size]

        # 🔍 모델 출력 토큰 디버깅 (설정 가능)
        if (
            getattr(self.config, "ENABLE_OUTPUT_DEBUG", False)
            and self.global_step % getattr(self.config, "DEBUG_OUTPUT_EVERY", 100) == 0
        ):
            self._debug_model_output(src, tgt_input, tgt_output, logits)

        # Calculate loss
        logits_flat = logits.reshape(-1, logits.size(-1))  # [batch*seq, vocab_size]
        # First logits
        self.logger.info(f"Logits sample: {logits_flat[0][:10].detach().cpu().numpy()}")

        tgt_flat = tgt_output.reshape(-1)  # [batch*seq]
        self.logger.info(f"Target sample: {tgt_flat[0:10].detach().cpu().numpy()}")

        loss = self.criterion(logits_flat, tgt_flat)
        return loss

    def _debug_batch_sample(self, src, tgt, tgt_input, tgt_output, batch_idx):
        """매 배치마다 첫 번째 샘플의 상세한 데이터 분석"""
        if batch_idx > 10:  # 처음 10배치만
            return

        # 첫 번째 샘플 가져오기
        src_sample = src[0]  # [src_len]
        tgt_sample = tgt[0]  # [tgt_len]
        tgt_input_sample = tgt_input[0]  # [tgt_len-1 or tgt_len]
        tgt_output_sample = tgt_output[0]  # [tgt_len-1 or tgt_len]

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🔍 BATCH {batch_idx} - SAMPLE DEBUG")
        self.logger.info(f"{'='*60}")

        # 기본 정보
        self.logger.info(f"📊 Batch shapes:")
        self.logger.info(f"   src: {src.shape}")
        self.logger.info(f"   tgt: {tgt.shape}")
        self.logger.info(f"   tgt_input: {tgt_input.shape}")
        self.logger.info(f"   tgt_output: {tgt_output.shape}")

        # 특수 토큰 정보
        self.logger.info(
            f"🏷️  Special tokens: PAD={self.config.PAD_TOKEN}, BOS={self.config.BOS_TOKEN}, EOS={self.config.EOS_TOKEN}, UNK={self.config.UNK_TOKEN}"
        )

        # Source 분석
        src_list = src_sample.tolist()
        src_nonpad = [x for x in src_list if x != self.config.PAD_TOKEN]
        self.logger.info(f"🔤 Source (길이 {len(src_list)}):")
        self.logger.info(
            f"   전체: {src_list[:20]}{'...' if len(src_list) > 20 else ''}"
        )
        self.logger.info(
            f"   PAD 제외: {src_nonpad[:20]}{'...' if len(src_nonpad) > 15 else ''}"
        )

        # Target 원본 분석
        tgt_list = tgt_sample.tolist()
        tgt_nonpad = [x for x in tgt_list if x != self.config.PAD_TOKEN]
        self.logger.info(f"🎯 Target 원본 (길이 {len(tgt_list)}):")
        self.logger.info(
            f"   전체: {tgt_list[:20]}{'...' if len(tgt_list) > 20 else ''}"
        )
        self.logger.info(
            f"   PAD 제외: {tgt_nonpad[:20]}{'...' if len(tgt_nonpad) > 15 else ''}"
        )

        # Target Input 분석 (디코더 입력)
        tgt_input_list = tgt_input_sample.tolist()
        tgt_input_nonpad = [x for x in tgt_input_list if x != self.config.PAD_TOKEN]
        self.logger.info(f"� Target Input - 디코더 입력 (길이 {len(tgt_input_list)}):")
        self.logger.info(
            f"   전체: {tgt_input_list[:20]}{'...' if len(tgt_input_list) > 20 else ''}"
        )
        self.logger.info(
            f"   PAD 제외: {tgt_input_nonpad[:20]}{'...' if len(tgt_input_nonpad) > 15 else ''}"
        )

        # Target Output 분석 (예측 대상)
        tgt_output_list = tgt_output_sample.tolist()
        tgt_output_nonpad = [x for x in tgt_output_list if x != self.config.PAD_TOKEN]
        self.logger.info(f"� Target Output - 예측 대상 (길이 {len(tgt_output_list)}):")
        self.logger.info(
            f"   전체: {tgt_output_list[:20]}{'...' if len(tgt_output_list) > 20 else ''}"
        )
        self.logger.info(
            f"   PAD 제외: {tgt_output_nonpad[:20]}{'...' if len(tgt_output_nonpad) > 15 else ''}"
        )

        self.logger.info(f"🔍 원본 src 데이터(text):")
        src_list = src_sample.tolist()
        src_nonpad = [x for x in src_list if x != self.config.PAD_TOKEN]
        self.logger.info(
            f"   전체: {self.vocab.decode(src_list[:20])}{'...' if len(src_list) > 20 else ''}"
        )

        self.logger.info(f"🔍 target 데이터 (text) :")
        tgt_list = tgt_sample.tolist()
        tgt_nonpad = [x for x in tgt_list if x != self.config.PAD_TOKEN]
        self.logger.info(
            f"   전체: {self.vocab.decode(tgt_list[:20])}{'...' if len(tgt_list) > 20 else ''}"
        )

        # 특수 토큰 존재 확인
        has_bos_in_input = self.config.BOS_TOKEN in tgt_input_list
        has_eos_in_input = self.config.EOS_TOKEN in tgt_input_list
        has_bos_in_output = self.config.BOS_TOKEN in tgt_output_list
        has_eos_in_output = self.config.EOS_TOKEN in tgt_output_list

        self.logger.info(f"🔍 특수 토큰 분포:")
        self.logger.info(f"   Input에 BOS: {has_bos_in_input}, EOS: {has_eos_in_input}")
        self.logger.info(
            f"   Output에 BOS: {has_bos_in_output}, EOS: {has_eos_in_output}"
        )

        if len(tgt_input_list) > 0 and len(tgt_output_list) > 0:
            self.logger.info(
                f"   Input 첫 토큰: {tgt_input_list[0]}, 마지막 토큰: {tgt_input_list[-1]}"
            )
            self.logger.info(
                f"   Output 첫 토큰: {tgt_output_list[0]}, 마지막 토큰: {tgt_output_list[-1]}"
            )

        # Teacher Forcing 매핑 예시 (처음 5개)
        if len(tgt_input_list) > 0 and len(tgt_output_list) > 0:
            self.logger.info(f"📝 Teacher Forcing 매핑 예시:")
            max_examples = min(5, len(tgt_input_list), len(tgt_output_list))
            for i in range(max_examples):
                input_token = tgt_input_list[i] if i < len(tgt_input_list) else "N/A"
                output_token = tgt_output_list[i] if i < len(tgt_output_list) else "N/A"
                self.logger.info(
                    f"   위치[{i}]: 입력={input_token} → 예측해야할값={output_token}"
                )

        # PAD 토큰 통계
        src_pad_count = (src_sample == self.config.PAD_TOKEN).sum().item()
        tgt_pad_count = (tgt_sample == self.config.PAD_TOKEN).sum().item()
        tgt_input_pad_count = (tgt_input_sample == self.config.PAD_TOKEN).sum().item()
        tgt_output_pad_count = (tgt_output_sample == self.config.PAD_TOKEN).sum().item()

        self.logger.info(f"📊 PAD 토큰 통계:")
        self.logger.info(
            f"   Source PAD: {src_pad_count}/{src_sample.size(0)} ({src_pad_count/src_sample.size(0)*100:.1f}%)"
        )
        self.logger.info(
            f"   Target 원본 PAD: {tgt_pad_count}/{tgt_sample.size(0)} ({tgt_pad_count/tgt_sample.size(0)*100:.1f}%)"
        )
        self.logger.info(
            f"   Target Input PAD: {tgt_input_pad_count}/{tgt_input_sample.size(0)} ({tgt_input_pad_count/tgt_input_sample.size(0)*100:.1f}%)"
        )
        self.logger.info(
            f"   Target Output PAD: {tgt_output_pad_count}/{tgt_output_sample.size(0)} ({tgt_output_pad_count/tgt_output_sample.size(0)*100:.1f}%)"
        )

        self.logger.info(f"{'='*60}\n")

    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        if not self.val_loader:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Forward pass
                loss = self._forward_step(batch)
                total_loss += loss.item()

        avg_loss = total_loss / num_batches

        # Update best validation loss
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self._save_checkpoint(is_best=True)

        self.model.train()
        return {"loss": avg_loss}

    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        self.logger.info(f"Configuration: {self.config.to_dict()}")

        for epoch in range(self.config.MAX_EPOCHS):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = {}
            if self.val_loader:
                val_metrics = self.validate()

            # Log epoch results
            log_msg = f"Epoch {epoch + 1}/{self.config.MAX_EPOCHS} - "
            log_msg += f"Train Loss: {train_metrics['loss']:.4f}"

            if val_metrics:
                log_msg += f" - Val Loss: {val_metrics['loss']:.4f}"

            self.logger.info(log_msg)

        self.logger.info("Training completed!")

    def test(self) -> Dict[str, float]:
        """Test the model"""
        if not self.test_loader:
            self.logger.warning("No test loader provided")
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.test_loader)

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                # Move batch to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Forward pass
                loss = self._forward_step(batch)
                total_loss += loss.item()

        avg_loss = total_loss / num_batches
        self.logger.info(f"Test Loss: {avg_loss:.4f}")

        return {"loss": avg_loss}

    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint and manage checkpoint history"""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "best_val_loss": self.best_val_loss,
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config.MODEL_SAVE_PATH, f"checkpoint_step_{self.global_step}.pt"
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config.MODEL_SAVE_PATH, "best_model.pt")
            torch.save(checkpoint, best_path)
            self.logger.info(
                f"New best model saved with validation loss: {self.best_val_loss:.4f}"
            )

        # Manage checkpoint history - keep only last N checkpoints
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """Keep only the last N checkpoints for model averaging"""
        from pathlib import Path
        import glob

        checkpoint_dir = Path(self.config.MODEL_SAVE_PATH)
        checkpoint_pattern = str(checkpoint_dir / "checkpoint_step_*.pt")
        checkpoint_files = glob.glob(checkpoint_pattern)

        if len(checkpoint_files) <= getattr(self.config, "N_CHECKPOINTS", 5):
            return

        # Extract step numbers and sort
        def extract_step(path):
            filename = os.path.basename(path)
            try:
                step = int(filename.split("_")[-1].split(".")[0])
                return step
            except (ValueError, IndexError):
                return 0

        checkpoint_files.sort(key=extract_step)

        # Keep only last N checkpoints
        n_checkpoints = getattr(self.config, "N_CHECKPOINTS", 5)
        files_to_delete = checkpoint_files[:-n_checkpoints]

        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                step = extract_step(file_path)
                self.logger.debug(f"Removed old checkpoint: step_{step}.pt")
            except OSError as e:
                self.logger.warning(f"Failed to remove checkpoint {file_path}: {e}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        self.logger.info(
            f"Resumed from epoch {self.current_epoch}, step {self.global_step}"
        )

    def _debug_model_output(self, src, tgt_input, tgt_output, logits):
        """모델 출력 토큰 디버깅"""
        self.logger.info(f"\n{'🤖'*20}")
        self.logger.info(f"🔍 MODEL OUTPUT DEBUG - Step {self.global_step}")
        self.logger.info(
            f"📈 Current Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}"
        )
        self.logger.info(f"{'🤖'*20}")

        with torch.no_grad():
            # 첫 번째 배치 샘플만 분석
            src_sample = src[0]  # [src_len]
            tgt_input_sample = tgt_input[0]  # [tgt_len]
            tgt_output_sample = tgt_output[0]  # [tgt_len]
            logits_sample = logits[0]  # [tgt_len, vocab_size]

            # 예측된 토큰들 (argmax)
            predicted_tokens = torch.argmax(logits_sample, dim=-1)  # [tgt_len]

            # 토큰을 텍스트로 변환하는 헬퍼 함수
            def tokens_to_text(tokens, name):
                tokens_list = tokens.cpu().tolist()
                tokens_clean = [
                    t
                    for t in tokens_list
                    if t
                    not in [
                        self.config.PAD_TOKEN,
                        self.config.BOS_TOKEN,
                        self.config.EOS_TOKEN,
                    ]
                ]

                try:
                    if hasattr(self.vocab, "decode"):
                        text = (
                            self.vocab.decode(tokens_clean)
                            if tokens_clean
                            else "<EMPTY>"
                        )
                    else:
                        text = (
                            " ".join([str(t) for t in tokens_clean])
                            if tokens_clean
                            else "<EMPTY>"
                        )
                    return tokens_clean[:10], text
                except Exception as e:
                    return tokens_clean[:10], f"<DECODE_ERROR: {e}>"

            # Source 정보
            src_tokens, src_text = tokens_to_text(src_sample, "Source")
            self.logger.info(f"📤 Source: {src_tokens} → '{src_text}'")

            # Target input 정보
            tgt_input_tokens, tgt_input_text = tokens_to_text(
                tgt_input_sample, "Target Input"
            )
            tgt_input_raw = tgt_input_sample.cpu().tolist()[:15]  # 처음 15개 토큰
            self.logger.info(
                f"⬇️  Target Input: {tgt_input_tokens} → '{tgt_input_text}'"
            )
            self.logger.info(f"   Raw tokens: {tgt_input_raw}")

            # Target output (정답) 정보
            tgt_output_tokens, tgt_output_text = tokens_to_text(
                tgt_output_sample, "Target Output"
            )
            tgt_output_raw = tgt_output_sample.cpu().tolist()[:15]  # 처음 15개 토큰
            self.logger.info(
                f"🎯 Target Output (정답): {tgt_output_tokens} → '{tgt_output_text}'"
            )
            self.logger.info(f"   Raw tokens: {tgt_output_raw}")

            # 모델 예측 정보
            pred_tokens, pred_text = tokens_to_text(
                predicted_tokens, "Model Prediction"
            )
            self.logger.info(f"🤖 Model Prediction: {pred_tokens} → '{pred_text}'")

            # 토큰별 정확도 계산
            correct = (predicted_tokens == tgt_output_sample).float()
            # PAD 토큰 제외한 정확도
            non_pad_mask = tgt_output_sample != self.config.PAD_TOKEN
            if non_pad_mask.sum() > 0:
                accuracy = correct[non_pad_mask].mean().item()
                self.logger.info(f"📊 Token Accuracy (PAD 제외): {accuracy:.2%}")

            # 상위 5개 예측 토큰 확률 분포 (첫 번째 위치)
            if logits_sample.size(0) > 0:
                first_pos_probs = torch.softmax(logits_sample[0], dim=-1)
                top5_probs, top5_tokens = torch.topk(first_pos_probs, 5)

                self.logger.info(f"🏆 첫 번째 위치 Top-5 예측:")
                for i, (token, prob) in enumerate(
                    zip(top5_tokens.cpu().tolist(), top5_probs.cpu().tolist())
                ):
                    try:
                        if hasattr(self.vocab, "decode"):
                            token_text = self.vocab.decode([token])
                        else:
                            token_text = str(token)
                    except:
                        token_text = f"<ERR:{token}>"

                    self.logger.info(
                        f"   {i+1}. Token {token:>6} ({prob:.2%}) → '{token_text}'"
                    )

            # 토큰별 비교 (처음 5개)
            self.logger.info(f"🔄 위치별 예측 vs 정답 (처음 5개):")
            compare_len = min(5, len(predicted_tokens), len(tgt_output_sample))
            for i in range(compare_len):
                pred_token = predicted_tokens[i].item()
                true_token = tgt_output_sample[i].item()

                if true_token == self.config.PAD_TOKEN:
                    continue

                match_symbol = "✅" if pred_token == true_token else "❌"

                try:
                    if hasattr(self.vocab, "decode"):
                        pred_word = self.vocab.decode([pred_token])
                        true_word = self.vocab.decode([true_token])
                    else:
                        pred_word = str(pred_token)
                        true_word = str(true_token)
                except:
                    pred_word = f"<ERR:{pred_token}>"
                    true_word = f"<ERR:{true_token}>"

                self.logger.info(
                    f"   [{i}] {match_symbol} 예측:{pred_token:>6}('{pred_word}') vs 정답:{true_token:>6}('{true_word}')"
                )

        self.logger.info(f"{'🤖'*20}\n")


def create_trainer(
    model: nn.Module,
    config: Config,
    vocab: Any,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
) -> Trainer:
    """Factory function to create trainer"""
    return Trainer(model, config, vocab, train_loader, val_loader, test_loader)
