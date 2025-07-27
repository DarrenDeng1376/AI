"""
Setup script for Azure AI Search example
This script helps you set up the environment and test the connection
"""

import os
import sys
from dotenv import load_dotenv

def check_environment():
    """Check if all required environment variables are set"""
    load_dotenv()
    
    required_vars = [
        "AZURE_SEARCH_SERVICE_ENDPOINT",
        "AZURE_SEARCH_ADMIN_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease update your .env file with the correct values.")
        return False
    
    print("✅ All required environment variables are set!")
    return True

def test_connection():
    """Test connection to Azure AI Search service"""
    try:
        from azure.search.documents.indexes import SearchIndexClient
        from azure.core.credentials import AzureKeyCredential
        
        load_dotenv()
        endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
        key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
        
        credential = AzureKeyCredential(key)
        client = SearchIndexClient(endpoint=endpoint, credential=credential)
        
        # Try to list indexes to test connection
        indexes = list(client.list_indexes())
        print(f"✅ Successfully connected to Azure AI Search!")
        print(f"   Found {len(indexes)} existing indexes")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to Azure AI Search: {e}")
        return False

def install_requirements():
    """Install required packages"""
    try:
        import subprocess
        
        print("Installing required packages...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Successfully installed all requirements!")
            return True
        else:
            print(f"❌ Failed to install requirements: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def main():
    """Main setup function"""
    print("Azure AI Search Setup")
    print("=" * 30)
    
    # Step 1: Install requirements
    print("\n1. Installing requirements...")
    if not install_requirements():
        return
    
    # Step 2: Check environment variables
    print("\n2. Checking environment variables...")
    if not check_environment():
        print("\nSetup incomplete. Please configure your .env file and run again.")
        return
    
    # Step 3: Test connection
    print("\n3. Testing connection to Azure AI Search...")
    if not test_connection():
        print("\nSetup incomplete. Please check your credentials and try again.")
        return
    
    print("\n✅ Setup complete! You can now run the examples:")
    print("   python azure_search_example.py")
    print("   python advanced_search_examples.py")

if __name__ == "__main__":
    main()
