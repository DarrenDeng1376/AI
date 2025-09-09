#!/usr/bin/env python3
"""
Comprehensive example demonstrating SmallGPT capabilities
"""
import torch
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model.gpt import SmallGPT
from src.utils.tokenizer import SimpleTokenizer
from src.utils.config import create_small_config, create_medium_config
from src.training.trainer import GPTTrainer
from src.training.dataset import TextDataset, LanguageModelingDataset
from src.utils.sampling import generate_text
from src.utils.metrics import calculate_perplexity, evaluate_generation_quality


def create_shakespeare_data():
    """Create sample Shakespeare-like data for training."""
    shakespeare_texts = [
        """To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer 
        The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles 
        And by opposing end them. To die—to sleep, No more; and by a sleep to say we end 
        The heart-ache and the thousand natural shocks That flesh is heir to: 'tis a consummation 
        Devoutly to be wished.""",
        
        """All the world's a stage, And all the men and women merely players; They have their exits 
        and their entrances, And one man in his time plays many parts, His acts being seven ages. 
        At first the infant, Mewling and puking in the nurse's arms. Then the whining schoolboy, 
        with his satchel And shining morning face, creeping like snail Unwillingly to school.""",
        
        """Shall I compare thee to a summer's day? Thou art more lovely and more temperate: 
        Rough winds do shake the darling buds of May, And summer's lease hath all too short a date: 
        Sometime too hot the eye of heaven shines, And often is his gold complexion dimmed; 
        And every fair from fair sometime declines, By chance or nature's changing course, untrimmed.""",
        
        """Romeo, Romeo, wherefore art thou Romeo? Deny thy father and refuse thy name; 
        Or, if thou wilt not, be but sworn my love, And I'll no longer be a Capulet. 
        'Tis but thy name that is my enemy: Thou art thyself, though not a Montague. 
        What's Montague? It is nor hand, nor foot, Nor arm, nor face, nor any other part 
        Belonging to a man. O, be some other name!""",
        
        """Tomorrow, and tomorrow, and tomorrow, Creeps in this petty pace from day to day 
        To the last syllable of recorded time, And all our yesterdays have lighted fools 
        The way to dusty death. Out, out, brief candle! Life's but a walking shadow, a poor player 
        That struts and frets his hour upon the stage And then is heard no more."""
    ]
    
    # Create data directory
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    # Write Shakespeare data
    with open(data_dir / "shakespeare.txt", "w", encoding="utf-8") as f:
        f.write("\\n\\n".join(shakespeare_texts))
    
    print(f"Shakespeare training data created in {data_dir}")
    return data_dir


def train_shakespeare_model():
    """Train a model on Shakespeare-like text."""
    print("\\n" + "="*50)
    print("TRAINING SHAKESPEARE MODEL")
    print("="*50)
    
    # Create training data
    data_dir = create_shakespeare_data()
    
    # Load configuration
    config = create_small_config()
    config.training.data_dir = str(data_dir)
    config.training.max_steps = 2000
    config.training.log_interval = 100
    config.training.eval_interval = 500
    config.training.save_interval = 1000
    config.training.output_dir = "./models/shakespeare"
    
    # Create tokenizer
    print("\\nTraining tokenizer...")
    with open(data_dir / "shakespeare.txt", "r", encoding="utf-8") as f:
        texts = f.read().split("\\n\\n")
    
    tokenizer = SimpleTokenizer(
        vocab_size=config.tokenizer.vocab_size,
        special_tokens=config.tokenizer.special_tokens
    )
    tokenizer.train(texts)
    
    # Save tokenizer
    models_dir = Path(config.training.output_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(models_dir / "tokenizer.pkl")
    
    print(f"Tokenizer trained with {tokenizer.get_vocab_size()} tokens")
    
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
    dataset = LanguageModelingDataset(
        texts=texts,
        tokenizer=tokenizer,
        block_size=config.model.max_seq_len
    )
    
    print(f"Dataset created with {len(dataset)} sequences")
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\\nTraining on device: {device}")
    
    trainer = GPTTrainer(
        model=model,
        config=config.training,
        train_dataset=dataset,
        device=device
    )
    
    # Train
    history = trainer.train()
    
    # Save final model
    model.save_pretrained(models_dir / "final_model.pt")
    
    print(f"\\nTraining completed!")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    
    return model, tokenizer, device, history


def demonstrate_generation(model, tokenizer, device):
    """Demonstrate text generation capabilities."""
    print("\\n" + "="*50)
    print("TEXT GENERATION DEMONSTRATION")
    print("="*50)
    
    model.eval()
    
    # Shakespeare-style prompts
    prompts = [
        "To be or not to be",
        "All the world's a stage",
        "Shall I compare thee",
        "Romeo, Romeo",
        "Tomorrow, and tomorrow"
    ]
    
    print("\\nGenerating Shakespeare-style text...")
    
    for i, prompt in enumerate(prompts):
        print(f"\\n--- Example {i+1} ---")
        print(f"Prompt: '{prompt}'")
        print("Generated text:")
        
        # Generate with different sampling strategies
        strategies = [
            {"name": "Greedy", "params": {"do_sample": False}},
            {"name": "Temperature=0.7", "params": {"temperature": 0.7, "do_sample": True}},
            {"name": "Top-k=20", "params": {"top_k": 20, "do_sample": True}},
            {"name": "Top-p=0.8", "params": {"top_p": 0.8, "do_sample": True}}
        ]
        
        for strategy in strategies:
            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_length=30,
                device=device,
                **strategy["params"]
            )
            print(f"  {strategy['name']}: {generated_text}")


def analyze_model_performance(model, tokenizer, device):
    """Analyze model performance and metrics."""
    print("\\n" + "="*50)
    print("MODEL PERFORMANCE ANALYSIS")
    print("="*50)
    
    # Create test prompts
    test_prompts = [
        "To be or not to be",
        "All the world's a stage",
        "Shall I compare thee to a summer's day",
        "Romeo, Romeo, wherefore art thou Romeo"
    ]
    
    print("\\nEvaluating generation quality...")
    
    # Evaluate generation quality
    metrics = evaluate_generation_quality(
        model=model,
        tokenizer=tokenizer,
        prompts=test_prompts,
        num_samples=3,
        max_length=50,
        device=device
    )
    
    print("\\nGeneration Quality Metrics:")
    print("-" * 30)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Analyze vocabulary usage
    print("\\n\\nVocabulary Analysis:")
    print("-" * 30)
    print(f"Total vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Sample some tokens
    sample_tokens = []
    for i in range(min(20, tokenizer.get_vocab_size())):
        if i in tokenizer.id_to_token:
            sample_tokens.append(tokenizer.id_to_token[i])
    
    print(f"Sample tokens: {sample_tokens}")
    
    return metrics


def create_training_visualization(history):
    """Create visualizations of training progress."""
    print("\\n" + "="*50)
    print("TRAINING VISUALIZATION")
    print("="*50)
    
    try:
        import matplotlib.pyplot as plt
        
        # Plot training loss
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['learning_rate'])
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch') 
        plt.ylabel('Learning Rate')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Training visualization saved as 'training_progress.png'")
        
    except ImportError:
        print("Matplotlib not available for visualization")
    except Exception as e:
        print(f"Error creating visualization: {e}")


def run_comprehensive_example():
    """Run the comprehensive example demonstrating all features."""
    print("SmallGPT Comprehensive Example")
    print("="*50)
    print("This example demonstrates:")
    print("- Training a custom tokenizer")
    print("- Creating and training a small GPT model")
    print("- Text generation with different sampling strategies")
    print("- Model performance analysis")
    print("- Training progress visualization")
    
    try:
        # Train model
        model, tokenizer, device, history = train_shakespeare_model()
        
        # Demonstrate generation
        demonstrate_generation(model, tokenizer, device)
        
        # Analyze performance
        metrics = analyze_model_performance(model, tokenizer, device)
        
        # Create visualizations
        create_training_visualization(history)
        
        # Save results
        results = {
            "model_params": model.get_num_params(),
            "vocab_size": tokenizer.get_vocab_size(),
            "final_loss": history['train_loss'][-1],
            "metrics": metrics
        }
        
        with open("example_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("\\n" + "="*50)
        print("EXAMPLE COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("\\nResults saved to:")
        print("- Model: ./models/shakespeare/")
        print("- Tokenizer: ./models/shakespeare/tokenizer.pkl")
        print("- Training plot: training_progress.png")
        print("- Metrics: example_results.json")
        
        print("\\nNext steps:")
        print("1. Try the interactive generation mode:")
        print("   python generate.py interactive --model models/shakespeare/final_model.pt --tokenizer models/shakespeare/tokenizer.pkl")
        print("2. Train on your own data")
        print("3. Experiment with different model architectures")
        
    except Exception as e:
        print(f"\\nError running example: {e}")
        print("\\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check that you have sufficient disk space")
        print("3. For GPU training, ensure CUDA is properly installed")


if __name__ == "__main__":
    run_comprehensive_example()
