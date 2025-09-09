"""
Training utilities and trainer class
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import os
import time
import logging
from tqdm import tqdm
import wandb

from ..model.gpt import SmallGPT
from ..utils.config import TrainingConfig
from .dataset import TextDataset


class GPTTrainer:
    """
    Trainer class for SmallGPT model.
    
    Args:
        model: SmallGPT model instance
        config: Training configuration
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        device: Device to train on
    """
    
    def __init__(
        self,
        model: SmallGPT,
        config: TrainingConfig,
        train_dataset: TextDataset,
        val_dataset: Optional[TextDataset] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Setup wandb if enabled
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=config.experiment_name,
                config=config.__dict__
            )
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer with weight decay."""
        
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Don't apply weight decay to biases and layer norms
                if 'bias' in name or 'ln' in name or 'norm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        if self.config.optimizer == 'adam':
            return optim.Adam(optim_groups, lr=self.config.learning_rate, betas=self.config.betas)
        elif self.config.optimizer == 'adamw':
            return optim.AdamW(optim_groups, lr=self.config.learning_rate, betas=self.config.betas)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        
        if self.config.lr_scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.max_steps,
                eta_min=self.config.min_lr
            )
        elif self.config.lr_scheduler == 'linear_warmup':
            return LinearWarmupScheduler(
                self.optimizer,
                warmup_steps=self.config.warmup_steps,
                total_steps=self.config.max_steps,
                min_lr=self.config.min_lr
            )
        elif self.config.lr_scheduler is None:
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.lr_scheduler}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger('GPTTrainer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def train(self) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Returns:
            Dictionary containing training history
        """
        self.logger.info(f"Starting training on {self.device}")
        self.logger.info(f"Model parameters: {self.model.get_num_params():,}")
        
        # Create data loader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Training loop
        self.model.train()
        start_time = time.time()
        
        while self.global_step < self.config.max_steps:
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {self.epoch}"):
                # Forward pass
                loss = self._training_step(batch)
                epoch_loss += loss
                num_batches += 1
                
                # Log metrics
                if self.global_step % self.config.log_interval == 0:
                    self._log_metrics(loss, time.time() - start_time)
                
                # Validation
                if self.global_step % self.config.eval_interval == 0 and self.val_dataset is not None:
                    val_loss = self._validate()
                    history['val_loss'].append(val_loss)
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint('best_model.pt')
                
                # Save checkpoint
                if self.global_step % self.config.save_interval == 0:
                    self._save_checkpoint(f'checkpoint_{self.global_step}.pt')
                
                # Check if we've reached max steps
                if self.global_step >= self.config.max_steps:
                    break
            
            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            history['train_loss'].append(avg_epoch_loss)
            history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            self.logger.info(f"Epoch {self.epoch} completed. Average loss: {avg_epoch_loss:.4f}")
            self.epoch += 1
        
        # Final checkpoint
        self._save_checkpoint('final_model.pt')
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        
        return history
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids, attention_mask)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        
        # Compute loss (next token prediction)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.model.pad_token_id
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
        
        # Optimizer step
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
        
        self.global_step += 1
        return loss.item()
    
    def _validate(self) -> float:
        """Validation step."""
        if self.val_dataset is None:
            return float('inf')
        
        self.model.eval()
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                # Compute loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=self.model.pad_token_id
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        avg_val_loss = total_loss / num_batches
        self.logger.info(f"Validation loss: {avg_val_loss:.4f}")
        
        return avg_val_loss
    
    def _log_metrics(self, loss: float, elapsed_time: float):
        """Log training metrics."""
        lr = self.optimizer.param_groups[0]['lr']
        
        self.logger.info(
            f"Step {self.global_step} | Loss: {loss:.4f} | LR: {lr:.2e} | "
            f"Time: {elapsed_time:.1f}s"
        )
        
        if self.config.use_wandb:
            wandb.log({
                'train/loss': loss,
                'train/learning_rate': lr,
                'train/step': self.global_step,
                'train/elapsed_time': elapsed_time
            })
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': self.config.__dict__,
            'best_val_loss': self.best_val_loss
        }
        
        filepath = os.path.join(self.config.output_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        self.logger.info(f"Checkpoint loaded: {filepath}")


class LinearWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """
    Linear warmup scheduler followed by linear decay.
    """
    
    def __init__(
        self, 
        optimizer: optim.Optimizer, 
        warmup_steps: int, 
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # Linear decay
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            return [
                base_lr * (1.0 - progress) + self.min_lr * progress
                for base_lr in self.base_lrs
            ]
