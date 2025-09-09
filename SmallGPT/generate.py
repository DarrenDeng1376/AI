#!/usr/bin/env python3
"""
Text generation script for SmallGPT
"""
import argparse
import torch
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model.gpt import SmallGPT
from src.utils.tokenizer import SimpleTokenizer
from src.utils.sampling import generate_text
from src.utils.config import GenerationConfig


def main():
    parser = argparse.ArgumentParser(description='Generate text with SmallGPT')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to tokenizer')
    parser.add_argument('--prompt', type=str, default='', help='Input prompt')
    parser.add_argument('--max-length', type=int, default=100, help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=0.9, help='Top-p (nucleus) sampling')
    parser.add_argument('--num-samples', type=int, default=1, help='Number of samples to generate')
    parser.add_argument('--no-sample', action='store_true', help='Use greedy decoding instead of sampling')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = SimpleTokenizer.load(args.tokenizer)
    print(f"Tokenizer loaded. Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = SmallGPT.from_pretrained(args.model)
    model.to(device)
    model.eval()
    print(f"Model loaded. Parameters: {model.get_num_params():,}")
    
    # Generate text
    print("\\n" + "="*50)
    print("GENERATING TEXT")
    print("="*50)
    
    if args.prompt:
        print(f"Prompt: {args.prompt}")
        print("-" * 50)
    
    generated_texts = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k if not args.no_sample else None,
        top_p=args.top_p if not args.no_sample else None,
        do_sample=not args.no_sample,
        num_return_sequences=args.num_samples,
        device=device
    )
    
    # Display results
    if isinstance(generated_texts, str):
        generated_texts = [generated_texts]
    
    for i, text in enumerate(generated_texts):
        if args.num_samples > 1:
            print(f"\\nSample {i+1}:")
            print("-" * 20)
        print(text)
        if i < len(generated_texts) - 1:
            print()
    
    print("\\n" + "="*50)


def interactive_mode():
    """Interactive text generation mode."""
    parser = argparse.ArgumentParser(description='Interactive text generation with SmallGPT')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to tokenizer')
    parser.add_argument('--temperature', type=float, default=0.8, help='Default sampling temperature')
    parser.add_argument('--top-k', type=int, default=50, help='Default top-k sampling')
    parser.add_argument('--top-p', type=float, default=0.9, help='Default top-p sampling')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = SimpleTokenizer.load(args.tokenizer)
    
    print(f"Loading model from {args.model}...")
    model = SmallGPT.from_pretrained(args.model)
    model.to(device)
    model.eval()
    
    print(f"\\nModel loaded successfully!")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Model parameters: {model.get_num_params():,}")
    print("\\nInteractive mode started. Type 'quit' to exit.")
    print("Commands:")
    print("  /temp <value>  - Set temperature")
    print("  /topk <value>  - Set top-k")
    print("  /topp <value>  - Set top-p")
    print("  /len <value>   - Set max length")
    print("  /help          - Show this help")
    print("\\n" + "="*50)
    
    # Default generation parameters
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    max_length = 100
    
    while True:
        try:
            # Get user input
            prompt = input("\\n> ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            # Handle commands
            if prompt.startswith('/'):
                parts = prompt.split()
                command = parts[0].lower()
                
                if command == '/help':
                    print("Commands:")
                    print("  /temp <value>  - Set temperature")
                    print("  /topk <value>  - Set top-k")
                    print("  /topp <value>  - Set top-p")
                    print("  /len <value>   - Set max length")
                    print("  /help          - Show this help")
                    continue
                elif command == '/temp' and len(parts) == 2:
                    try:
                        temperature = float(parts[1])
                        print(f"Temperature set to {temperature}")
                    except ValueError:
                        print("Invalid temperature value")
                    continue
                elif command == '/topk' and len(parts) == 2:
                    try:
                        top_k = int(parts[1])
                        print(f"Top-k set to {top_k}")
                    except ValueError:
                        print("Invalid top-k value")
                    continue
                elif command == '/topp' and len(parts) == 2:
                    try:
                        top_p = float(parts[1])
                        print(f"Top-p set to {top_p}")
                    except ValueError:
                        print("Invalid top-p value")
                    continue
                elif command == '/len' and len(parts) == 2:
                    try:
                        max_length = int(parts[1])
                        print(f"Max length set to {max_length}")
                    except ValueError:
                        print("Invalid max length value")
                    continue
                else:
                    print("Unknown command. Type /help for available commands.")
                    continue
            
            if not prompt:
                continue
            
            # Generate text
            print("\\nGenerating...")
            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                device=device
            )
            
            print(f"\\n{generated_text}")
            
        except KeyboardInterrupt:
            print("\\n\\nExiting...")
            break
        except Exception as e:
            print(f"\\nError: {e}")
    
    print("Goodbye!")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'interactive':
        # Remove 'interactive' from args
        sys.argv.pop(1)
        interactive_mode()
    else:
        main()
