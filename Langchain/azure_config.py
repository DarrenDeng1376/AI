"""
Azure OpenAI Configuration Helper

This module provides utilities for configuring Azure OpenAI
across all LangChain examples using Azure Default Credentials.
"""

import os
from typing import Optional
from azure.identity import DefaultAzureCredential

def get_azure_openai_config():
    """
    Get Azure OpenAI configuration from environment variables
    Uses Azure Default Credentials for authentication
    """
    config = {
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        "chat_deployment": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-35-turbo"),
        "embeddings_deployment": os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "text-embedding-ada-002"),
        "credential": DefaultAzureCredential()
    }
    
    # Validate required fields (API key is no longer required)
    required_fields = ["azure_endpoint"]
    missing_fields = [field for field in required_fields if not config[field]]
    
    if missing_fields:
        raise ValueError(f"Missing required Azure OpenAI environment variables: {missing_fields}")
    
    return config

def create_azure_chat_openai(temperature: float = 0.7, **kwargs):
    """
    Create an Azure OpenAI chat instance with Default Credentials
    """
    try:
        from langchain_openai import AzureChatOpenAI
        
        config = get_azure_openai_config()
        
        return AzureChatOpenAI(
            azure_endpoint=config["azure_endpoint"],
            azure_ad_token_provider=lambda: config["credential"].get_token("https://cognitiveservices.azure.com/.default").token,
            api_version=config["api_version"],
            azure_deployment=config["chat_deployment"],
            temperature=temperature,
            **kwargs
        )
    except ImportError:
        raise ImportError("langchain-openai package is required. Install with: pip install langchain-openai")
    except Exception as e:
        raise Exception(f"Failed to create Azure Chat OpenAI: {e}. Make sure you're authenticated with Azure CLI or have appropriate managed identity permissions.")

def create_azure_openai_llm(temperature: float = 0.7, **kwargs):
    """
    Create an Azure OpenAI LLM instance (for completion models) with Default Credentials
    """
    try:
        from langchain_openai import AzureOpenAI
        
        config = get_azure_openai_config()
        
        return AzureOpenAI(
            azure_endpoint=config["azure_endpoint"],
            azure_ad_token_provider=lambda: config["credential"].get_token("https://cognitiveservices.azure.com/.default").token,
            api_version=config["api_version"],
            azure_deployment=config["chat_deployment"],
            temperature=temperature,
            **kwargs
        )
    except ImportError:
        raise ImportError("langchain-openai package is required. Install with: pip install langchain-openai")
    except Exception as e:
        raise Exception(f"Failed to create Azure OpenAI LLM: {e}. Make sure you're authenticated with Azure CLI or have appropriate managed identity permissions.")

def create_azure_openai_embeddings(**kwargs):
    """
    Create an Azure OpenAI embeddings instance with Default Credentials
    """
    try:
        from langchain_openai import AzureOpenAIEmbeddings
        
        config = get_azure_openai_config()
        
        return AzureOpenAIEmbeddings(
            azure_endpoint=config["azure_endpoint"],
            azure_ad_token_provider=lambda: config["credential"].get_token("https://cognitiveservices.azure.com/.default").token,
            api_version=config["api_version"],
            azure_deployment=config["embeddings_deployment"],
            **kwargs
        )
    except ImportError:
        raise ImportError("langchain-openai package is required. Install with: pip install langchain-openai")
    except Exception as e:
        raise Exception(f"Failed to create Azure OpenAI Embeddings: {e}. Make sure you're authenticated with Azure CLI or have appropriate managed identity permissions.")

def test_azure_openai_connection():
    """
    Test the Azure OpenAI connection
    """
    try:
        print("🔍 Testing Azure OpenAI connection...")
        
        # Test chat model
        llm = create_azure_chat_openai(temperature=0)
        response = llm.invoke("Say 'Hello from Azure OpenAI!'")
        print(f"✅ Chat model working: {response.content}")
        
        # Test embeddings
        embeddings = create_azure_openai_embeddings()
        embed_result = embeddings.embed_query("Test embedding")
        print(f"✅ Embeddings working: Generated {len(embed_result)} dimensional vector")
        
        print("🎉 Azure OpenAI connection successful!")
        return True
        
    except Exception as e:
        print(f"❌ Azure OpenAI connection failed: {e}")
        return False

if __name__ == "__main__":
    # Test configuration when run directly
    from dotenv import load_dotenv
    load_dotenv()
    
    try:
        config = get_azure_openai_config()
        print("📋 Azure OpenAI Configuration (using Default Credentials):")
        print(f"Endpoint: {config['azure_endpoint']}")
        print(f"API Version: {config['api_version']}")
        print(f"Chat Deployment: {config['chat_deployment']}")
        print(f"Embeddings Deployment: {config['embeddings_deployment']}")
        print(f"Authentication: Azure Default Credentials")
        print()
        
        test_azure_openai_connection()
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        print("\n💡 Make sure to:")
        print("1. Set these environment variables in your .env file:")
        print("   - AZURE_OPENAI_ENDPOINT")
        print("   - AZURE_OPENAI_CHAT_DEPLOYMENT") 
        print("   - AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
        print("\n2. Authenticate with Azure using one of these methods:")
        print("   - Azure CLI: az login")
        print("   - Service Principal with environment variables")
        print("   - Managed Identity (if running on Azure)")
        print("   - Visual Studio Code Azure Account extension")
        print("\n3. Ensure your account has 'Cognitive Services OpenAI User' role on the Azure OpenAI resource")
