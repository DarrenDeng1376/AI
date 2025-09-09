"""
Simple tests for SmallGPT components
"""
import sys
import os
import tempfile
import torch
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_tokenizer():
    """Test the SimpleTokenizer."""
    print("Testing SimpleTokenizer...")
    
    from src.utils.tokenizer import SimpleTokenizer
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=1000)
    
    # Sample texts
    texts = [
        "Hello world, this is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world."
    ]
    
    # Train tokenizer
    tokenizer.train(texts)
    
    # Test encoding/decoding
    text = "Hello world"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    assert isinstance(encoded, list)
    assert len(encoded) > 0
    assert isinstance(decoded, str)
    
    print("✅ Tokenizer test passed!")
    return tokenizer


def test_model():
    """Test the SmallGPT model."""
    print("\\nTesting SmallGPT model...")
    
    from src.model.gpt import SmallGPT
    
    # Create small model for testing
    model = SmallGPT(
        vocab_size=1000,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        max_seq_len=64,
        dropout=0.1
    )
    
    # Test forward pass
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    outputs = model(input_ids)
    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
    
    expected_shape = (batch_size, seq_len, 1000)
    assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
    
    # Test generation
    prompt = torch.randint(0, 1000, (1, 5))
    generated = model.generate(prompt, max_length=10, do_sample=False)
    
    assert generated.shape[0] == 1
    assert generated.shape[1] >= prompt.shape[1]
    
    print(f"Model parameters: {model.get_num_params():,}")
    print("✅ Model test passed!")
    return model


def test_config():
    """Test configuration management."""
    print("\\nTesting configuration...")
    
    from src.utils.config import create_small_config, Config
    
    # Test creating config
    config = create_small_config()
    config.validate()
    
    # Test serialization
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        config_dict = {
            'model': config.model.__dict__,
            'training': config.training.__dict__,
            'tokenizer': config.tokenizer.__dict__,
            'generation': config.generation.__dict__
        }
        yaml.dump(config_dict, f)
        temp_path = f.name
    
    # Test loading config
    from src.utils.config import load_config
    loaded_config = load_config(temp_path)
    loaded_config.validate()
    
    # Cleanup
    os.unlink(temp_path)
    
    print("✅ Configuration test passed!")
    return config


def test_dataset():
    """Test dataset functionality."""
    print("\\nTesting dataset...")
    
    from src.training.dataset import TextDataset
    from src.utils.tokenizer import SimpleTokenizer
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=500)
    texts = ["Hello world", "This is a test", "AI is amazing"]
    tokenizer.train(texts)
    
    # Create dataset
    dataset = TextDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_seq_len=32
    )
    
    assert len(dataset) > 0
    
    # Test getting item
    item = dataset[0]
    assert 'input_ids' in item
    assert isinstance(item['input_ids'], torch.Tensor)
    
    print(f"Dataset size: {len(dataset)}")
    print("✅ Dataset test passed!")
    return dataset


def test_generation():
    """Test text generation utilities."""
    print("\\nTesting text generation...")
    
    from src.utils.sampling import generate_text
    
    # Use previously created model and tokenizer
    tokenizer = test_tokenizer()
    model = test_model()
    
    # Test generation
    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt="Hello",
        max_length=20,
        temperature=0.8,
        do_sample=True
    )
    
    assert isinstance(generated, str)
    assert len(generated) > 0
    
    print(f"Generated text: {generated}")
    print("✅ Generation test passed!")


def run_all_tests():
    """Run all tests."""
    print("Running SmallGPT Tests")
    print("=" * 40)
    
    try:
        # Run tests
        test_tokenizer()
        test_model()
        test_config()
        test_dataset()
        test_generation()
        
        print("\\n" + "=" * 40)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 40)
        
    except Exception as e:
        print(f"\\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
