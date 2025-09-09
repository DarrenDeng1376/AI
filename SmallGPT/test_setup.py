#!/usr/bin/env python3
"""
SmallGPT - Test Existing Setup
===============================
This script tests what's already available without installing anything new.
"""

import sys
import os

def print_status(text, status='info'):
    """Print status with simple indicators"""
    indicators = {'ok': '✓', 'warn': '⚠', 'error': '✗', 'info': 'ℹ'}
    print(f"{indicators.get(status, '•')} {text}")

def test_python_version():
    """Test Python version"""
    version = sys.version_info
    print_status(f"Python {version.major}.{version.minor}.{version.micro}", 'ok')
    return version.major >= 3 and version.minor >= 9

def test_existing_imports():
    """Test what packages are already available"""
    print_status("Testing existing packages...", 'info')
    
    packages = {
        'numpy': 'np',
        'torch': None,
        'yaml': None,
        'tqdm': None,
        'matplotlib': 'plt',
        'pandas': 'pd'
    }
    
    available = []
    missing = []
    
    for pkg, alias in packages.items():
        try:
            if alias:
                exec(f"import {pkg} as {alias}")
            else:
                exec(f"import {pkg}")
            
            # Get version if available
            try:
                version = eval(f"{pkg}.__version__")
                print_status(f"{pkg} {version}", 'ok')
            except:
                print_status(f"{pkg} (available)", 'ok')
            available.append(pkg)
        except ImportError:
            print_status(f"{pkg} (not installed)", 'warn')
            missing.append(pkg)
    
    return available, missing

def test_project_structure():
    """Test if SmallGPT project files exist"""
    print_status("Checking project structure...", 'info')
    
    required_files = [
        'src/model/transformer.py',
        'src/utils/tokenizer.py',
        'src/training/train.py',
        'configs/small_model.yaml',
        'quick_start.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print_status(f"{file_path}", 'ok')
        else:
            print_status(f"{file_path} (missing)", 'error')
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def test_basic_functionality():
    """Test basic functionality with available packages"""
    print_status("Testing basic functionality...", 'info')
    
    try:
        # Test config loading
        if os.path.exists('src/utils/config.py'):
            sys.path.insert(0, '.')
            from src.utils.config import load_config
            config = load_config('configs/small_model.yaml')
            print_status(f"Config loaded: {config['model']['name']}", 'ok')
        
        # Test tokenizer (basic functionality)
        if os.path.exists('src/utils/tokenizer.py'):
            from src.utils.tokenizer import BPETokenizer
            tokenizer = BPETokenizer()
            # Just test basic methods exist
            print_status("Tokenizer class loaded", 'ok')
        
        return True
        
    except Exception as e:
        print_status(f"Functionality test failed: {str(e)[:50]}...", 'error')
        return False

def show_next_steps(available_packages, structure_ok):
    """Show what the user can do next"""
    print_status("What you can do now:", 'info')
    
    if structure_ok:
        print("📁 Project structure is complete")
        
        if 'torch' in available_packages:
            print("🚀 Full functionality available:")
            print("   python quick_start.py")
            print("   python scripts/train.py")
            print("   python scripts/generate.py")
        elif 'numpy' in available_packages:
            print("⚡ Basic functionality available:")
            print("   View and modify code")
            print("   Test tokenizer and config")
            print("   Install torch: pip install torch")
        else:
            print("📚 Learning mode:")
            print("   Read the code structure")
            print("   Install packages: pip install -r requirements_minimal.txt")
    
    print("\n📖 Documentation available:")
    print("   README.md - Overview")
    print("   src/ - Source code with comments")
    print("   configs/ - Model configurations")

def main():
    """Main test function"""
    print("=" * 50)
    print("🔍 SmallGPT - Test Existing Setup")
    print("=" * 50)
    
    # Test Python
    python_ok = test_python_version()
    
    print("\n" + "─" * 30)
    
    # Test packages
    available, missing = test_existing_imports()
    
    print("\n" + "─" * 30)
    
    # Test structure
    structure_ok = test_project_structure()
    
    print("\n" + "─" * 30)
    
    # Test functionality
    functionality_ok = test_basic_functionality()
    
    print("\n" + "=" * 50)
    print("📋 Summary")
    print("=" * 50)
    
    print_status(f"Python 3.9+: {'Yes' if python_ok else 'No'}", 'ok' if python_ok else 'error')
    print_status(f"Available packages: {len(available)}/{len(available) + len(missing)}", 'ok' if available else 'warn')
    print_status(f"Project structure: {'Complete' if structure_ok else 'Incomplete'}", 'ok' if structure_ok else 'error')
    print_status(f"Basic functionality: {'Working' if functionality_ok else 'Issues'}", 'ok' if functionality_ok else 'warn')
    
    if missing:
        print_status(f"Missing packages: {', '.join(missing)}", 'warn')
    
    print("\n" + "=" * 50)
    show_next_steps(available, structure_ok)

if __name__ == "__main__":
    main()
