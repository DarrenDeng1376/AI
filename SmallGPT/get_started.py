#!/usr/bin/env python3
"""
Getting started script for SmallGPT
"""
import subprocess
import sys
import os
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("\\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def run_tests():
    """Run basic tests to verify installation."""
    print("\\n🧪 Running tests...")
    try:
        subprocess.check_call([sys.executable, "tests/test_basic.py"])
        print("✅ All tests passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed: {e}")
        return False


def run_quick_example():
    """Run the quick start example."""
    print("\\n🚀 Running quick example...")
    try:
        subprocess.check_call([sys.executable, "quick_start.py"])
        print("✅ Quick example completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Quick example failed: {e}")
        return False


def show_usage_examples():
    """Show usage examples."""
    print("\\n" + "="*50)
    print("USAGE EXAMPLES")
    print("="*50)
    
    print("""
1. Quick Start (minimal example):
   python quick_start.py

2. Comprehensive Example:
   python example.py

3. Train a custom model:
   python train.py --config configs/small_model.yaml --data ./data

4. Generate text:
   python generate.py --model models/final_model.pt --tokenizer models/tokenizer.pkl --prompt "Hello world"

5. Interactive generation:
   python generate.py interactive --model models/final_model.pt --tokenizer models/tokenizer.pkl

6. Train with custom parameters:
   python train.py --vocab-size 5000 --embed-dim 256 --num-layers 4 --max-steps 5000

7. Using as a Python library:
   ```python
   from src import SmallGPT, SimpleTokenizer, generate_text
   
   # Load model
   model = SmallGPT.from_pretrained('models/model.pt')
   tokenizer = SimpleTokenizer.load('models/tokenizer.pkl')
   
   # Generate text
   text = generate_text(model, tokenizer, "Hello", max_length=50)
   print(text)
   ```

8. Configuration files:
   - configs/small_model.yaml  (quick training)
   - configs/medium_model.yaml (better quality)
""")


def create_sample_data():
    """Create sample training data."""
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    if not (data_dir / "sample.txt").exists():
        sample_text = """The sun was setting over the quiet village as Maria walked down the cobblestone path. 
She had lived here all her life, but tonight everything felt different. The familiar houses seemed to glow 
with an ethereal light, and the evening breeze carried whispers of ancient secrets.

In the distance, she could see the old lighthouse standing tall against the darkening sky. Her grandmother 
had told her stories about that lighthouse, tales of ships guided safely to shore and of mysterious lights 
that appeared on foggy nights. Maria had always thought they were just stories, but now she wondered if 
there might be some truth to them.

As she approached the lighthouse, she noticed something unusual. A warm, golden light was emanating from 
the windows, even though the lighthouse had been abandoned for decades. Her heart raced with curiosity 
and a touch of fear. Should she investigate, or should she turn back and pretend she had never seen anything?

The decision was made for her when she heard a gentle voice calling her name from within the lighthouse. 
It was a voice she had never heard before, yet somehow it felt familiar, like a half-remembered dream. 
With trembling hands, she reached for the old wooden door and slowly pushed it open.

Inside, the lighthouse was not as she had expected. Instead of dust and decay, she found a spiral staircase 
that seemed to shimmer with starlight. Each step she took upward felt like stepping into another world, 
a place where magic was not just possible but inevitable."""
        
        with open(data_dir / "sample.txt", "w", encoding="utf-8") as f:
            f.write(sample_text)
        
        print(f"✅ Sample training data created in {data_dir}")


def main():
    """Main getting started function."""
    print("🤖 SmallGPT - Getting Started")
    print("=" * 50)
    print("This script will help you set up and run SmallGPT")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Create sample data
    create_sample_data()
    
    # Run tests
    if not run_tests():
        print("⚠️  Tests failed, but you can still try the examples")
    
    # Ask user what they want to do
    print("\\n" + "="*50)
    print("WHAT WOULD YOU LIKE TO DO?")
    print("="*50)
    print("1. Run quick example (5-10 minutes)")
    print("2. Run comprehensive example (15-30 minutes)")
    print("3. Show usage examples")
    print("4. Exit")
    
    while True:
        choice = input("\\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            if run_quick_example():
                print("\\n🎉 Quick example completed successfully!")
            break
        elif choice == "2":
            print("\\n🔄 Running comprehensive example...")
            try:
                subprocess.check_call([sys.executable, "example.py"])
                print("\\n🎉 Comprehensive example completed!")
            except subprocess.CalledProcessError as e:
                print(f"❌ Comprehensive example failed: {e}")
            break
        elif choice == "3":
            show_usage_examples()
            break
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-4.")
    
    # Final instructions
    print("\\n" + "="*50)
    print("NEXT STEPS")
    print("="*50)
    print("""
📚 Documentation: Check README.md for detailed documentation
🔧 Configuration: Edit configs/*.yaml files for custom training
📁 Data: Place your training data in the ./data directory
🎮 Interactive: Try the interactive generation mode
🧪 Experiments: Create your own model configurations

Happy experimenting with SmallGPT! 🚀
""")
    
    return True


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\n👋 Setup interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("Please check the error message and try again")
