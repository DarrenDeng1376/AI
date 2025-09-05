#!/usr/bin/env python3
"""
Intelligent Document Processor - Setup and Launch Script
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        "streamlit",
        "openai",
        "chromadb",
        "azure-ai-formrecognizer",
        "python-dotenv",
        "tiktoken",
        "pandas",
        "numpy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install dependencies from requirements.txt"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_env_file():
    """Check if .env file exists and has required variables"""
    env_file = Path(".env")
    
    if not env_file.exists():
        print("❌ .env file not found")
        print("📝 Creating .env file from template...")
        
        # Copy template
        template_file = Path("env_template.txt")
        if template_file.exists():
            shutil.copy(template_file, ".env")
            print("✅ .env file created from template")
            print("⚠️  Please edit .env file with your Azure credentials")
            return False
        else:
            print("❌ Template file not found")
            return False
    
    # Check if env file has required variables
    required_vars = [
        "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
        "AZURE_DOCUMENT_INTELLIGENCE_KEY", 
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT_NAME",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
    ]
    
    env_content = env_file.read_text()
    missing_vars = []
    
    for var in required_vars:
        if f"{var}=" not in env_content or f"{var}=your_" in env_content:
            missing_vars.append(var)
    
    if missing_vars:
        print("⚠️  Missing or incomplete environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        return False
    
    print("✅ Environment file configured")
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "data/vector_store",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Directory: {directory}")

def run_health_check():
    """Run application health check"""
    print("\n🔧 Running health check...")
    try:
        from example import run_health_check
        return run_health_check()
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def launch_streamlit():
    """Launch the Streamlit application"""
    print("\n🚀 Launching Streamlit application...")
    print("📝 Note: Press Ctrl+C to stop the application")
    print("🌐 The application will open in your default web browser")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped")
    except Exception as e:
        print(f"❌ Failed to launch application: {e}")

def main():
    """Main setup and launch function"""
    print("🤖 Intelligent Document Processor - Setup & Launch")
    print("=" * 60)
    
    # Step 1: Check Python version
    print("\n1. Checking Python version...")
    if not check_python_version():
        return
    
    # Step 2: Check dependencies
    print("\n2. Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"\n📦 Missing packages: {', '.join(missing)}")
        install = input("Install missing packages? (y/n): ").lower().strip()
        
        if install == 'y':
            if not install_dependencies():
                return
        else:
            print("❌ Cannot proceed without required packages")
            return
    
    # Step 3: Check environment configuration
    print("\n3. Checking environment configuration...")
    if not check_env_file():
        print("⚠️  Please configure your .env file and run this script again")
        return
    
    # Step 4: Create directories
    print("\n4. Creating directories...")
    create_directories()
    
    # Step 5: Health check
    if not run_health_check():
        print("❌ Health check failed. Please fix configuration issues.")
        return
    
    print("\n✅ Setup complete!")
    
    # Step 6: Launch options
    print("\nChoose launch option:")
    print("1. Launch Streamlit web application")
    print("2. Run example script")
    print("3. Exit")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        launch_streamlit()
    elif choice == "2":
        print("\n🔍 Running example script...")
        try:
            subprocess.run([sys.executable, "example.py"])
        except Exception as e:
            print(f"❌ Failed to run example: {e}")
    elif choice == "3":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Setup interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("Please check your setup and try again")
