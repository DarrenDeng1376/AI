"""
LangChain Learning Setup Script

This script helps you set up your LangChain learning environment.
It will:
1. Check Python version
2. Install required packages
3. Verify installations
4. Set up environment variables
5. Run basic tests
"""

import sys
import subprocess
import os
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible!")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.8+")
        return False

def install_packages():
    """Install required packages"""
    print("\n📦 Installing LangChain packages...")
    
    packages = [
        "langchain>=0.1.0",
        "langchain-openai>=0.0.5",
        "langchain-community>=0.0.10",
        "python-dotenv>=1.0.0",
        "chromadb>=0.4.0",
        "tiktoken>=0.5.0",
        "faiss-cpu>=1.7.4",
        "pydantic>=2.0.0"
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def check_installations():
    """Verify that packages are installed correctly"""
    print("\n🔍 Verifying installations...")
    
    packages_to_check = [
        ("langchain", "LangChain core"),
        ("langchain_openai", "LangChain OpenAI"),
        ("dotenv", "Python Dotenv"),
        ("chromadb", "ChromaDB"),
        ("tiktoken", "TikToken"),
        ("faiss", "FAISS"),
        ("pydantic", "Pydantic")
    ]
    
    all_good = True
    for package, name in packages_to_check:
        try:
            spec = importlib.util.find_spec(package)
            if spec is not None:
                print(f"✅ {name} - OK")
            else:
                print(f"❌ {name} - Not found")
                all_good = False
        except ImportError:
            print(f"❌ {name} - Import error")
            all_good = False
    
    return all_good

def setup_environment():
    """Help user set up environment variables"""
    print("\n🔧 Setting up environment variables...")
    
    env_file = ".env"
    env_example = ".env.example"
    
    if not os.path.exists(env_file):
        if os.path.exists(env_example):
            # Copy example file
            with open(env_example, 'r') as src, open(env_file, 'w') as dst:
                dst.write(src.read())
            print(f"✅ Created {env_file} from {env_example}")
        else:
            # Create basic .env file for Azure OpenAI
            env_content = """# LangChain Environment Variables - Azure OpenAI Configuration
# Add your Azure OpenAI details below

# Azure OpenAI (Primary)
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Azure OpenAI Deployment Names
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-35-turbo
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-ada-002

# Optional: Other providers
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
"""
            with open(env_file, 'w') as f:
                f.write(env_content)
            print(f"✅ Created basic {env_file}")
    else:
        print(f"✅ {env_file} already exists")
    
    print(f"\n⚠️  IMPORTANT: Edit {env_file} and add your Azure OpenAI details!")
    print("   - Get Azure OpenAI credentials from Azure Portal")
    print("   - You need: API Key, Endpoint, and Deployment Names")
    print("   - Most examples require at least Azure OpenAI configuration")

def run_basic_test():
    """Run a basic test to ensure everything works"""
    print("\n🧪 Running basic test...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check for Azure OpenAI configuration
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        if not azure_api_key or azure_api_key == "your_azure_openai_api_key_here":
            print("⚠️  Azure OpenAI API key not set. Please update your .env file to run examples.")
            return False
        
        if not azure_endpoint or azure_endpoint == "https://your-resource-name.openai.azure.com/":
            print("⚠️  Azure OpenAI endpoint not set. Please update your .env file to run examples.")
            return False
        
        # Test basic imports
        from langchain.schema import Document
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Test basic functionality
        doc = Document(page_content="This is a test document.")
        splitter = RecursiveCharacterTextSplitter(chunk_size=100)
        chunks = splitter.split_documents([doc])
        
        if len(chunks) > 0:
            print("✅ Basic functionality test passed!")
            print("✅ Azure OpenAI configuration looks good!")
            return True
        else:
            print("❌ Basic functionality test failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def show_next_steps():
    """Show user what to do next"""
    print("\n🚀 Next Steps:")
    print("1. Make sure you've added your OpenAI API key to .env")
    print("2. Start with: cd 01_Basics && python examples.py")
    print("3. Follow the learning path in README.md")
    print("4. Complete exercises in each module")
    print("5. Build your own projects!")
    print("\n📚 Learning Path:")
    print("   01_Basics → 02_Prompts → 03_LLMs → ... → 09_RAG → Production")
    print("\n💡 Tips:")
    print("   - Read the README.md in each folder first")
    print("   - Try exercises before looking at solutions")
    print("   - Experiment with different parameters")
    print("   - Join LangChain community for help")

def main():
    """Main setup function"""
    print("🎉 Welcome to LangChain Learning Setup!")
    print("=" * 50)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Install packages
    if success and not install_packages():
        success = False
    
    # Verify installations
    if success and not check_installations():
        success = False
    
    # Setup environment
    if success:
        setup_environment()
    
    # Run basic test
    if success:
        run_basic_test()
    
    # Show next steps
    show_next_steps()
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print("You're ready to start learning LangChain!")
    else:
        print("\n❌ Setup encountered some issues.")
        print("Please resolve the errors above and run the script again.")

if __name__ == "__main__":
    main()
