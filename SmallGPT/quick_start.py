#!/usr/bin/env python3
"""
Quick start example for SmallGPT
"""
import torch
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model.gpt import SmallGPT
from src.utils.tokenizer import SimpleTokenizer
from src.utils.config import create_small_config
from src.training.trainer import GPTTrainer
from src.training.dataset import TextDataset
from src.utils.sampling import generate_text


def create_sample_data():
    """Create sample training data."""
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    # Sample texts from different domains
    sample_texts = [
        # Simple stories
        """Once upon a time, in a small village, there lived a curious young girl named Alice. 
        She loved to explore the nearby forest and discover new things. One day, while walking 
        through the woods, she found a mysterious door hidden behind some bushes. The door was 
        old and wooden, with strange symbols carved into its surface. Alice wondered what 
        secrets lay beyond this magical door.""",
        
        # Science content
        """The human brain is one of the most complex organs in the body. It contains billions 
        of neurons that communicate through electrical and chemical signals. These neurons form 
        intricate networks that allow us to think, learn, and remember. Scientists are still 
        discovering new things about how the brain works and how different regions contribute 
        to our consciousness and behavior.""",
        
        # Historical content
        """The Renaissance was a period of great cultural and artistic achievement in Europe. 
        It began in Italy during the 14th century and spread throughout Europe over the next 
        few centuries. This era saw remarkable developments in art, science, literature, and 
        philosophy. Famous figures like Leonardo da Vinci, Michelangelo, and Shakespeare made 
        contributions that still influence our world today.""",
        
        # Technology content
        """Artificial intelligence is transforming the way we live and work. Machine learning 
        algorithms can now recognize patterns in data, understand natural language, and even 
        create art and music. As AI technology continues to advance, it promises to solve many 
        of humanity's greatest challenges while also raising important questions about the 
        future of work and society."""
    ]
    
    # Write sample data
    with open(data_dir / "sample_texts.txt", "w", encoding="utf-8") as f:
        f.write("\\n\\n".join(sample_texts))
    
    print(f"Sample data created in {data_dir}")
    return data_dir


def quick_train_example():
    """Quick training example with small model."""
    print("SmallGPT Quick Start Example")
    print("=" * 40)
    
    # Create sample data
    data_dir = create_sample_data()
    
    # Load configuration
    config = create_small_config()
    config.training.data_dir = str(data_dir)
    config.training.max_steps = 1000  # Very short training for demo
    config.training.log_interval = 50
    config.training.eval_interval = 200
    config.training.save_interval = 500
    
    print(f"\\nConfiguration:")
    print(f"  Model size: {config.model.embed_dim}d, {config.model.num_layers} layers")
    print(f"  Vocab size: {config.model.vocab_size}")
    print(f"  Max sequence length: {config.model.max_seq_len}")
    print(f"  Training steps: {config.training.max_steps}")
    
    # Create and train tokenizer
    print("\\nTraining tokenizer...")
    with open(data_dir / "sample_texts.txt", "r", encoding="utf-8") as f:
        texts = f.read().split("\\n\\n")
    
    tokenizer = SimpleTokenizer(
        vocab_size=config.tokenizer.vocab_size,
        special_tokens=config.tokenizer.special_tokens
    )
    tokenizer.train(texts)
    
    # Save tokenizer
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    tokenizer.save(models_dir / "quick_tokenizer.pkl")
    print(f"Tokenizer saved with {tokenizer.get_vocab_size()} tokens")
    
    # Create model
    print("\\nCreating model...")
    model = SmallGPT(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=config.model.embed_dim,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout,
        pad_token_id=tokenizer.pad_token_id
    )
    
    print(f"Model created with {model.get_num_params():,} parameters")
    
    # Create dataset
    print("\\nPreparing dataset...")
    dataset = TextDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_seq_len=config.model.max_seq_len
    )
    print(f"Dataset created with {len(dataset)} sequences")
    
    # Train model
    print("\\nStarting training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    trainer = GPTTrainer(
        model=model,
        config=config.training,
        train_dataset=dataset,
        device=device
    )
    
    # Quick training
    history = trainer.train()
    
    # Save model
    model_path = models_dir / "quick_model.pt"
    model.save_pretrained(model_path)
    print(f"\\nModel saved to {model_path}")
    
    return model, tokenizer, device


def generation_example(model, tokenizer, device):
    """Example of text generation."""
    print("\\n" + "=" * 40)
    print("TEXT GENERATION EXAMPLES")
    print("=" * 40)
    
    model.eval()
    
    # Example prompts
    prompts = [
        "Once upon a time",
        "The future of artificial intelligence",
        "In the depths of the ocean",
        "Scientists have discovered"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\\nExample {i+1}:")
        print(f"Prompt: '{prompt}'")
        print("-" * 30)
        
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=50,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            device=device
        )
        
        print(generated_text)


def main():
    """Main function for quick start example."""
    try:
        # Train a small model
        model, tokenizer, device = quick_train_example()
        
        # Generate some text
        generation_example(model, tokenizer, device)
        
        print("\\n" + "=" * 40)
        print("QUICK START COMPLETE!")
        print("=" * 40)
        print("\\nNext steps:")
        print("1. Try training on your own data")
        print("2. Experiment with different model sizes")
        print("3. Adjust generation parameters")
        print("4. Use the interactive generation mode:")
        print("   python generate.py interactive --model models/quick_model.pt --tokenizer models/quick_tokenizer.pkl")
        
    except Exception as e:
        print(f"\\nError: {e}")
        print("Make sure you have all dependencies installed:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    main()
