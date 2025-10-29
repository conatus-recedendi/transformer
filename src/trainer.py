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


class LRScheduler:
    """Learning rate scheduler for Transformer (Warmup + Decay)"""

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        """Update learning rate"""
        self.step_num += 1
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
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Setup device
        self.device = torch.device(
            config.DEVICE if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Setup optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        self.scheduler = LRScheduler(
            self.optimizer, config.MODEL_DIM, config.WARMUP_STEPS
        )

        # Setup loss function
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
            self.scheduler.step()

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
        tgt = batch["tgt"]  # [batch_size, tgt_len]

        # 🔍 매 배치마다 첫 번째 샘플 데이터 보여주기
        if self.global_step % 1 == 0:  # 매 배치마다
            self._debug_batch_sample(src, tgt, batch_idx=self.global_step)

        # Create input and target for decoder
        tgt_input = tgt[:, :-1]  # [batch_size, tgt_len-1] - remove last token
        tgt_output = tgt[
            :, 1:
        ]  # [batch_size, tgt_len-1] - remove first token (usually BOS)

        # Forward pass through model
        logits = self.model(src, tgt_input)  # [batch_size, tgt_len-1, vocab_size]

        # Calculate loss
        logits_flat = logits.reshape(-1, logits.size(-1))  # [batch*seq, vocab_size]
        tgt_flat = tgt_output.reshape(-1)  # [batch*seq]

        loss = self.criterion(logits_flat, tgt_flat)
        return loss

    def _debug_batch_sample(self, src, tgt, batch_idx):
        """매 배치마다 첫 번째 샘플의 상세한 데이터 분석"""
        if batch_idx > 10:  # 처음 10배치만
            return

        # 첫 번째 샘플 가져오기
        src_sample = src[0]  # [src_len]
        tgt_sample = tgt[0]  # [tgt_len]

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🔍 BATCH {batch_idx} - SAMPLE DEBUG")
        self.logger.info(f"{'='*60}")

        # 기본 정보
        self.logger.info(f"📊 Batch shape - src: {src.shape}, tgt: {tgt.shape}")
        self.logger.info(
            f"📊 Sample shape - src: {src_sample.shape}, tgt: {tgt_sample.shape}"
        )

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
            f"   PAD 제외: {src_nonpad[:15]}{'...' if len(src_nonpad) > 15 else ''}"
        )

        # Target 분석
        tgt_list = tgt_sample.tolist()
        tgt_nonpad = [x for x in tgt_list if x != self.config.PAD_TOKEN]
        self.logger.info(f"🎯 Target (길이 {len(tgt_list)}):")
        self.logger.info(
            f"   전체: {tgt_list[:20]}{'...' if len(tgt_list) > 20 else ''}"
        )
        self.logger.info(
            f"   PAD 제외: {tgt_nonpad[:15]}{'...' if len(tgt_nonpad) > 15 else ''}"
        )

        # 특수 토큰 존재 확인
        has_bos = self.config.BOS_TOKEN in tgt_list
        has_eos = self.config.EOS_TOKEN in tgt_list
        bos_pos = tgt_list.index(self.config.BOS_TOKEN) if has_bos else -1
        eos_pos = tgt_list.index(self.config.EOS_TOKEN) if has_eos else -1

        self.logger.info(f"🔍 Target 특수 토큰:")
        self.logger.info(f"   BOS 존재: {has_bos} (위치: {bos_pos})")
        self.logger.info(f"   EOS 존재: {has_eos} (위치: {eos_pos})")
        self.logger.info(f"   첫 토큰: {tgt_list[0] if len(tgt_list) > 0 else 'N/A'}")
        self.logger.info(
            f"   마지막 토큰: {tgt_list[-1] if len(tgt_list) > 0 else 'N/A'}"
        )

        # Teacher Forcing 처리 결과
        if len(tgt_list) > 1:
            tgt_input = tgt_sample[:-1].tolist()  # BOS 포함, EOS 제외
            tgt_output = tgt_sample[1:].tolist()  # BOS 제외, EOS 포함

            self.logger.info(f"📚 Teacher Forcing 처리:")
            self.logger.info(
                f"   입력 (decoder input): {tgt_input[:15]}{'...' if len(tgt_input) > 15 else ''}"
            )
            self.logger.info(
                f"   출력 (target output): {tgt_output[:15]}{'...' if len(tgt_output) > 15 else ''}"
            )
            self.logger.info(
                f"   입력 길이: {len(tgt_input)}, 출력 길이: {len(tgt_output)}"
            )

            # 토큰 매핑 예시 (처음 5개)
            self.logger.info(f"📝 토큰 매핑 예시:")
            for i in range(min(5, len(tgt_input))):
                self.logger.info(
                    f"   입력[{i}]={tgt_input[i]} → 예측해야할값={tgt_output[i]}"
                )
        else:
            self.logger.info(f"⚠️  Target 시퀀스가 너무 짧음 (길이: {len(tgt_list)})")

        # PAD 토큰 통계
        src_pad_count = (src_sample == self.config.PAD_TOKEN).sum().item()
        tgt_pad_count = (tgt_sample == self.config.PAD_TOKEN).sum().item()
        self.logger.info(f"📊 PAD 토큰 통계:")
        self.logger.info(
            f"   Source PAD: {src_pad_count}/{src_sample.size(0)} ({src_pad_count/src_sample.size(0)*100:.1f}%)"
        )
        self.logger.info(
            f"   Target PAD: {tgt_pad_count}/{tgt_sample.size(0)} ({tgt_pad_count/tgt_sample.size(0)*100:.1f}%)"
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


def create_trainer(
    model: nn.Module,
    config: Config,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
) -> Trainer:
    """Factory function to create trainer"""
    return Trainer(model, config, train_loader, val_loader, test_loader)
