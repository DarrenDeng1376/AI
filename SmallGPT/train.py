#!/usr/bin/env python3
"""
Training script for SmallGPT
"""
import argparse
import torch
import torch.distributed as dist
from pathlib import Path
import logging
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model.gpt import SmallGPT
from src.training.trainer import GPTTrainer
from src.training.dataset import TextDataset, LanguageModelingDataset
from src.utils.tokenizer import SimpleTokenizer
from src.utils.config import load_config, create_default_config


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )


def prepare_data(config, tokenizer):
    """Prepare training and validation datasets."""
    data_path = Path(config.training.data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    # Find text files
    text_files = list(data_path.glob("*.txt"))
    if not text_files:
        raise FileNotFoundError(f"No .txt files found in {data_path}")
    
    print(f"Found {len(text_files)} text files")
    
    # Load all texts
    all_texts = []
    for file_path in text_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                all_texts.append(content)
    
    print(f"Loaded {len(all_texts)} text documents")
    
    # Split into train/validation
    split_idx = int(0.9 * len(all_texts))
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:] if split_idx < len(all_texts) else None
    
    # Create datasets
    train_dataset = LanguageModelingDataset(
        train_texts, 
        tokenizer, 
        block_size=config.training.max_seq_len
    )
    
    val_dataset = None
    if val_texts:
        val_dataset = LanguageModelingDataset(
            val_texts,
            tokenizer,
            block_size=config.training.max_seq_len
        )
    
    print(f"Training dataset size: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation dataset size: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def main():
    parser = argparse.ArgumentParser(description='Train SmallGPT model')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data', type=str, help='Path to training data')
    parser.add_argument('--output-dir', type=str, default='./models', help='Output directory')
    parser.add_argument('--tokenizer', type=str, help='Path to pretrained tokenizer')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    # Model configuration
    parser.add_argument('--vocab-size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--embed-dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--max-seq-len', type=int, default=1024, help='Maximum sequence length')
    
    # Training configuration
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--max-steps', type=int, default=100000, help='Maximum training steps')
    parser.add_argument('--warmup-steps', type=int, default=2000, help='Warmup steps')
    parser.add_argument('--save-interval', type=int, default=5000, help='Save interval')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = create_default_config()
        logger.info("Using default configuration")
    
    # Override config with command line arguments
    if args.data:
        config.training.data_dir = args.data
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.vocab_size:
        config.model.vocab_size = args.vocab_size
        config.tokenizer.vocab_size = args.vocab_size
    if args.embed_dim:
        config.model.embed_dim = args.embed_dim
    if args.num_heads:
        config.model.num_heads = args.num_heads
    if args.num_layers:
        config.model.num_layers = args.num_layers
    if args.max_seq_len:
        config.model.max_seq_len = args.max_seq_len
        config.training.max_seq_len = args.max_seq_len
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.max_steps:
        config.training.max_steps = args.max_steps
    if args.warmup_steps:
        config.training.warmup_steps = args.warmup_steps
    if args.save_interval:
        config.training.save_interval = args.save_interval
    
    # Validate configuration
    config.validate()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load or create tokenizer
    if args.tokenizer and Path(args.tokenizer).exists():
        tokenizer = SimpleTokenizer.load(args.tokenizer)
        logger.info(f"Loaded tokenizer from {args.tokenizer}")
    else:
        logger.info("Training new tokenizer...")
        tokenizer = SimpleTokenizer(
            vocab_size=config.tokenizer.vocab_size,
            special_tokens=config.tokenizer.special_tokens
        )
        
        # Load training texts for tokenizer training
        data_path = Path(config.training.data_dir)
        text_files = list(data_path.glob("*.txt"))
        
        all_texts = []
        for file_path in text_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                all_texts.append(f.read())
        
        tokenizer.train(all_texts)
        
        # Save tokenizer
        tokenizer_path = Path(config.training.output_dir) / 'tokenizer.pkl'
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(tokenizer_path)
        logger.info(f"Saved tokenizer to {tokenizer_path}")
    
    # Update config with actual vocab size
    config.model.vocab_size = tokenizer.get_vocab_size()
    config.tokenizer.vocab_size = tokenizer.get_vocab_size()
    
    # Prepare data
    train_dataset, val_dataset = prepare_data(config, tokenizer)
    
    # Create model
    model = SmallGPT(
        vocab_size=config.model.vocab_size,
        embed_dim=config.model.embed_dim,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout,
        pad_token_id=tokenizer.pad_token_id
    )
    
    logger.info(f"Created model with {model.get_num_params():,} parameters")
    
    # Create trainer
    trainer = GPTTrainer(
        model=model,
        config=config.training,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed training from {args.resume}")
    
    # Start training
    logger.info("Starting training...")
    history = trainer.train()
    
    logger.info("Training completed!")
    
    # Save final model
    final_model_path = Path(config.training.output_dir) / 'final_model.pt'
    model.save_pretrained(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")


if __name__ == '__main__':
    main()
