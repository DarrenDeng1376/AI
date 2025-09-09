#!/usr/bin/env python3
"""
SmallGPT - Get Started Script (Minimal Version)
==================================================
This script helps you get started with SmallGPT using only essential dependencies.
"""

import sys
import subprocess
import os

def print_colored(text, color='green'):
    """Print colored text (simplified for Windows compatibility)"""
    colors = {
        'green': '✓',
        'yellow': '⚠',
        'red': '✗',
        'blue': 'ℹ'
    }
    prefix = colors.get(color, '•')
    print(f"{prefix} {text}")

def check_python_version():
    """Check if Python version is 3.9+"""
    print_colored("Checking Python version...", 'blue')
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print_colored(f"Error: Python 3.9+ required. You have {version.major}.{version.minor}", 'red')
        return False
    
    print_colored(f"Python {version.major}.{version.minor}.{version.micro} ✓", 'green')
    return True

def install_minimal_requirements():
    """Install minimal requirements one by one"""
    print_colored("Installing minimal requirements...", 'blue')
    
    # Essential packages only
    packages = [
        "torch>=2.0.0",
        "numpy>=1.24.0", 
        "pyyaml>=6.0",
        "tqdm>=4.65.0"
    ]
    
    failed_packages = []
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print_colored(f"Successfully installed {package}", 'green')
            else:
                print_colored(f"Failed to install {package}: {result.stderr[:100]}", 'yellow')
                failed_packages.append(package)
                
        except subprocess.TimeoutExpired:
            print_colored(f"Timeout installing {package}", 'yellow')
            failed_packages.append(package)
        except Exception as e:
            print_colored(f"Error installing {package}: {str(e)[:100]}", 'yellow')
            failed_packages.append(package)
    
    if failed_packages:
        print_colored(f"Some packages failed to install: {failed_packages}", 'yellow')
        print_colored("You can try installing them manually later.", 'blue')
    else:
        print_colored("All minimal requirements installed successfully!", 'green')
    
    return len(failed_packages) == 0

def check_imports():
    """Check if essential imports work"""
    print_colored("Testing essential imports...", 'blue')
    
    imports_to_test = [
        ("numpy", "np"),
        ("yaml", None),
        ("tqdm", None)
    ]
    
    failed_imports = []
    
    for module, alias in imports_to_test:
        try:
            if alias:
                exec(f"import {module} as {alias}")
            else:
                exec(f"import {module}")
            print_colored(f"✓ {module}", 'green')
        except ImportError:
            print_colored(f"✗ {module} (not available)", 'yellow')
            failed_imports.append(module)
    
    # Special check for torch (might take longer)
    try:
        import torch
        print_colored(f"✓ torch {torch.__version__}", 'green')
    except ImportError:
        print_colored("✗ torch (not available)", 'yellow')
        failed_imports.append("torch")
    
    if failed_imports:
        print_colored(f"Missing imports: {failed_imports}", 'yellow')
        print_colored("SmallGPT will run with reduced functionality.", 'blue')
    else:
        print_colored("All essential imports working!", 'green')
    
    return len(failed_imports) == 0

def run_minimal_demo():
    """Run a minimal demo that works without all dependencies"""
    print_colored("Running minimal demo...", 'blue')
    
    try:
        # Simple tokenizer test
        print("Testing tokenizer...")
        from src.utils.tokenizer import BPETokenizer
        
        tokenizer = BPETokenizer()
        test_text = "Hello, world! This is a test."
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        
        print(f"Original: {test_text}")
        print(f"Tokens: {tokens[:10]}...")  # Show first 10 tokens
        print(f"Decoded: {decoded}")
        
        print_colored("Tokenizer test passed!", 'green')
        
        # Try to load model config
        print("Testing config loading...")
        from src.utils.config import load_config
        
        config = load_config('configs/small_model.yaml')
        print(f"Model config: {config['model']['name']} with {config['model']['n_layers']} layers")
        
        print_colored("Config test passed!", 'green')
        print_colored("Minimal demo completed successfully!", 'green')
        
        return True
        
    except Exception as e:
        print_colored(f"Demo failed: {str(e)}", 'red')
        print_colored("This is normal if dependencies are missing.", 'blue')
        return False

def main():
    """Main setup function"""
    print("=" * 60)
    print("🚀 SmallGPT - Minimal Setup")
    print("=" * 60)
    
    # Step 1: Check Python version
    if not check_python_version():
        return
    
    # Step 2: Install minimal requirements
    print("\n" + "=" * 40)
    install_success = install_minimal_requirements()
    
    # Step 3: Test imports
    print("\n" + "=" * 40)
    import_success = check_imports()
    
    # Step 4: Run minimal demo
    print("\n" + "=" * 40)
    demo_success = run_minimal_demo()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Setup Summary")
    print("=" * 60)
    
    print_colored(f"Installation: {'Success' if install_success else 'Partial'}", 'green' if install_success else 'yellow')
    print_colored(f"Imports: {'Success' if import_success else 'Partial'}", 'green' if import_success else 'yellow')
    print_colored(f"Demo: {'Success' if demo_success else 'Failed'}", 'green' if demo_success else 'yellow')
    
    if install_success and import_success and demo_success:
        print_colored("🎉 SmallGPT is ready to use!", 'green')
        print_colored("Next steps:", 'blue')
        print("  - Run: python quick_start.py")
        print("  - Train a model: python scripts/train.py")
        print("  - Generate text: python scripts/generate.py")
    else:
        print_colored("⚠ SmallGPT partially set up", 'yellow')
        print_colored("You can still:", 'blue')
        print("  - View the code structure")
        print("  - Install missing packages manually")
        print("  - Run components that don't need missing dependencies")

if __name__ == "__main__":
    main()
