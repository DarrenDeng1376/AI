"""
Configuration management for the Intelligent Document Processor
"""
import os
from typing import List, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class AzureConfig:
    """Azure service configuration"""
    
    # Document Intelligence
    document_intelligence_endpoint: str = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "")
    document_intelligence_key: str = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "")
    
    # OpenAI - Using default Azure credentials
    openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")  # Optional - will use default credentials if not provided
    openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    openai_deployment_name: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
    openai_embedding_deployment: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "")
    use_default_credentials: bool = os.getenv("AZURE_USE_DEFAULT_CREDENTIALS", "true").lower() == "true"
    
    def validate(self) -> List[str]:
        """Validate configuration and return any missing required fields"""
        missing = []
        
        if not self.document_intelligence_endpoint:
            missing.append("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        if not self.document_intelligence_key:
            missing.append("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        if not self.openai_endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT")
        
        # Only require API key if not using default credentials
        if not self.use_default_credentials and not self.openai_api_key:
            missing.append("AZURE_OPENAI_API_KEY (or set AZURE_USE_DEFAULT_CREDENTIALS=true)")
            
        if not self.openai_deployment_name:
            missing.append("AZURE_OPENAI_DEPLOYMENT_NAME")
        if not self.openai_embedding_deployment:
            missing.append("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
            
        return missing

@dataclass
class AppConfig:
    """Application configuration"""
    
    # File processing
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    max_chunks_per_document: int = int(os.getenv("MAX_CHUNKS_PER_DOCUMENT", "500"))
    
    # Security
    allowed_file_types: List[str] = field(default_factory=lambda: os.getenv(
        "ALLOWED_FILE_TYPES", 
        "pdf,png,jpg,jpeg,tiff,bmp,docx,xlsx,pptx,txt"
    ).split(","))
    max_documents_per_session: int = int(os.getenv("MAX_DOCUMENTS_PER_SESSION", "20"))
    
    # Performance
    max_concurrent_requests: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    
    # Streamlit
    streamlit_port: int = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
    streamlit_address: str = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", "./logs/app.log")

@dataclass
class EmbeddingConfig:
    """Embedding and vector search configuration"""
    
    embedding_model: str = "text-embedding-ada-002"
    embedding_dimension: int = 1536
    similarity_threshold: float = 0.7
    max_search_results: int = 10
    
    # Search settings
    enable_hybrid_search: bool = True
    keyword_weight: float = 0.3
    semantic_weight: float = 0.7

@dataclass
class QAConfig:
    """Question-answering configuration"""
    
    # Model settings
    temperature: float = 0.1
    max_tokens: int = 1000
    top_p: float = 0.9
    
    # Context settings
    max_context_chunks: int = 5
    context_relevance_threshold: float = 0.6
    
    # Response settings
    include_sources: bool = True
    confidence_threshold: float = 0.5
    max_follow_up_questions: int = 3

# Global configuration instances
azure_config = AzureConfig()
app_config = AppConfig()
embedding_config = EmbeddingConfig()
qa_config = QAConfig()

def get_config_summary() -> dict:
    """Get a summary of current configuration"""
    return {
        "azure_services_configured": len(azure_config.validate()) == 0,
        "vector_store_path": app_config.vector_store_path,
        "max_file_size_mb": app_config.max_file_size_mb,
        "chunk_size": app_config.chunk_size,
        "allowed_file_types": app_config.allowed_file_types,
        "embedding_model": embedding_config.embedding_model,
        "qa_temperature": qa_config.temperature,
    }

def validate_configuration() -> tuple[bool, List[str]]:
    """Validate all configuration and return status and any issues"""
    issues = []
    
    # Validate Azure configuration
    azure_missing = azure_config.validate()
    if azure_missing:
        issues.extend([f"Missing Azure config: {field}" for field in azure_missing])
    
    # Validate paths
    if not os.path.exists(os.path.dirname(app_config.vector_store_path)):
        try:
            os.makedirs(os.path.dirname(app_config.vector_store_path), exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create vector store directory: {e}")
    
    if not os.path.exists(os.path.dirname(app_config.log_file)):
        try:
            os.makedirs(os.path.dirname(app_config.log_file), exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create log directory: {e}")
    
    # Validate ranges
    if app_config.chunk_size <= 0:
        issues.append("Chunk size must be positive")
    
    if app_config.chunk_overlap >= app_config.chunk_size:
        issues.append("Chunk overlap must be less than chunk size")
    
    if embedding_config.similarity_threshold < 0 or embedding_config.similarity_threshold > 1:
        issues.append("Similarity threshold must be between 0 and 1")
    
    return len(issues) == 0, issues

if __name__ == "__main__":
    """Test configuration validation"""
    is_valid, issues = validate_configuration()
    print(f"Configuration valid: {is_valid}")
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    
    print("\nConfiguration summary:")
    summary = get_config_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
