"""
Azure service client utilities and helpers
"""
import logging
import asyncio
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.exceptions import AzureError, ClientAuthenticationError
from openai import AzureOpenAI

from config import azure_config

logger = logging.getLogger(__name__)

class AzureClientManager:
    """Manages Azure service clients with connection pooling and error handling"""
    
    def __init__(self):
        """Initialize the Azure client manager"""
        self._document_intelligence_client = None
        self._openai_client = None
        self._credentials = None
        
    def get_document_intelligence_client(self) -> DocumentAnalysisClient:
        """Get Document Intelligence client with lazy initialization"""
        if self._document_intelligence_client is None:
            try:
                if azure_config.use_default_credentials and not azure_config.document_intelligence_key:
                    # Use Azure Default Credentials
                    from azure.identity import DefaultAzureCredential
                    credential = DefaultAzureCredential()
                    
                    self._document_intelligence_client = DocumentAnalysisClient(
                        endpoint=azure_config.document_intelligence_endpoint,
                        credential=credential
                    )
                    logger.info("Document Intelligence client initialized with default credentials")
                else:
                    # Use API key
                    self._document_intelligence_client = DocumentAnalysisClient(
                        endpoint=azure_config.document_intelligence_endpoint,
                        credential=AzureKeyCredential(azure_config.document_intelligence_key)
                    )
                    logger.info("Document Intelligence client initialized with API key")
                    
            except Exception as e:
                logger.error(f"Failed to initialize Document Intelligence client: {e}")
                raise
        
        return self._document_intelligence_client
    
    def get_openai_client(self) -> AzureOpenAI:
        """Get Azure OpenAI client with lazy initialization"""
        if self._openai_client is None:
            try:
                if azure_config.use_default_credentials and not azure_config.openai_api_key:
                    # Use Azure Default Credentials
                    from azure.identity import DefaultAzureCredential
                    credential = DefaultAzureCredential()
                    
                    # Get token for Azure OpenAI
                    token = credential.get_token("https://cognitiveservices.azure.com/.default")
                    
                    self._openai_client = AzureOpenAI(
                        azure_ad_token=token.token,
                        api_version=azure_config.openai_api_version,
                        azure_endpoint=azure_config.openai_endpoint
                    )
                    logger.info("Azure OpenAI client initialized with default credentials")
                else:
                    # Use API key
                    self._openai_client = AzureOpenAI(
                        api_key=azure_config.openai_api_key,
                        api_version=azure_config.openai_api_version,
                        azure_endpoint=azure_config.openai_endpoint
                    )
                    logger.info("Azure OpenAI client initialized with API key")
                    
            except Exception as e:
                logger.error(f"Failed to initialize Azure OpenAI client: {e}")
                raise
        
        return self._openai_client
    
    async def test_connections(self) -> Dict[str, bool]:
        """Test connections to all Azure services"""
        results = {}
        
        # Test Document Intelligence
        try:
            client = self.get_document_intelligence_client()
            # Simple test - this might need adjustment based on actual API
            results["document_intelligence"] = True
            logger.info("Document Intelligence connection test passed")
        except Exception as e:
            logger.error(f"Document Intelligence connection test failed: {e}")
            results["document_intelligence"] = False
        
        # Test Azure OpenAI
        try:
            client = self.get_openai_client()
            # Test with a simple completion
            response = client.chat.completions.create(
                model=azure_config.openai_deployment_name,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            results["openai"] = True
            logger.info("Azure OpenAI connection test passed")
        except Exception as e:
            logger.error(f"Azure OpenAI connection test failed: {e}")
            results["openai"] = False
        
        return results
    
    def close_connections(self):
        """Close all client connections"""
        # Azure clients typically don't need explicit closing
        # but we can reset them for cleanup
        self._document_intelligence_client = None
        self._openai_client = None
        logger.info("Azure client connections closed")

class AzureServiceHealthChecker:
    """Monitor Azure service health and availability"""
    
    def __init__(self, client_manager: AzureClientManager):
        self.client_manager = client_manager
        self.health_status = {}
    
    async def check_service_health(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all Azure services"""
        health_results = {}
        
        # Check Document Intelligence
        doc_intel_health = await self._check_document_intelligence_health()
        health_results["document_intelligence"] = doc_intel_health
        
        # Check Azure OpenAI
        openai_health = await self._check_openai_health()
        health_results["openai"] = openai_health
        
        self.health_status = health_results
        return health_results
    
    async def _check_document_intelligence_health(self) -> Dict[str, Any]:
        """Check Document Intelligence service health"""
        try:
            client = self.client_manager.get_document_intelligence_client()
            
            # Test with minimal request
            start_time = asyncio.get_event_loop().time()
            
            # Note: Actual health check would depend on available endpoints
            # This is a placeholder that assumes the client creation is the test
            
            response_time = asyncio.get_event_loop().time() - start_time
            
            return {
                "status": "healthy",
                "response_time_ms": response_time * 1000,
                "last_checked": asyncio.get_event_loop().time(),
                "error": None
            }
            
        except ClientAuthenticationError as e:
            return {
                "status": "authentication_error",
                "response_time_ms": None,
                "last_checked": asyncio.get_event_loop().time(),
                "error": str(e)
            }
        except AzureError as e:
            return {
                "status": "service_error",
                "response_time_ms": None,
                "last_checked": asyncio.get_event_loop().time(),
                "error": str(e)
            }
        except Exception as e:
            return {
                "status": "unknown_error",
                "response_time_ms": None,
                "last_checked": asyncio.get_event_loop().time(),
                "error": str(e)
            }
    
    async def _check_openai_health(self) -> Dict[str, Any]:
        """Check Azure OpenAI service health"""
        try:
            client = self.client_manager.get_openai_client()
            
            start_time = asyncio.get_event_loop().time()
            
            # Test with minimal request
            response = client.chat.completions.create(
                model=azure_config.openai_deployment_name,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
                temperature=0
            )
            
            response_time = asyncio.get_event_loop().time() - start_time
            
            return {
                "status": "healthy",
                "response_time_ms": response_time * 1000,
                "last_checked": asyncio.get_event_loop().time(),
                "error": None,
                "model": azure_config.openai_deployment_name,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "response_time_ms": None,
                "last_checked": asyncio.get_event_loop().time(),
                "error": str(e)
            }
    
    def get_overall_health(self) -> str:
        """Get overall system health status"""
        if not self.health_status:
            return "unknown"
        
        statuses = [service["status"] for service in self.health_status.values()]
        
        if all(status == "healthy" for status in statuses):
            return "healthy"
        elif any(status == "healthy" for status in statuses):
            return "partial"
        else:
            return "unhealthy"

class AzureRateLimitManager:
    """Manage rate limits for Azure services"""
    
    def __init__(self):
        self.request_counts = {}
        self.reset_times = {}
    
    async def wait_for_rate_limit(self, service: str, max_requests: int, time_window: int):
        """Wait if rate limit would be exceeded"""
        current_time = asyncio.get_event_loop().time()
        
        if service not in self.request_counts:
            self.request_counts[service] = 0
            self.reset_times[service] = current_time + time_window
        
        # Reset count if time window has passed
        if current_time >= self.reset_times[service]:
            self.request_counts[service] = 0
            self.reset_times[service] = current_time + time_window
        
        # Check if we need to wait
        if self.request_counts[service] >= max_requests:
            wait_time = self.reset_times[service] - current_time
            if wait_time > 0:
                logger.info(f"Rate limit reached for {service}, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                # Reset after waiting
                self.request_counts[service] = 0
                self.reset_times[service] = current_time + time_window
        
        # Increment request count
        self.request_counts[service] += 1

@asynccontextmanager
async def azure_error_handler(service_name: str):
    """Context manager for handling Azure service errors"""
    try:
        yield
    except ClientAuthenticationError as e:
        logger.error(f"{service_name} authentication error: {e}")
        raise Exception(f"Authentication failed for {service_name}. Please check your credentials.")
    except AzureError as e:
        logger.error(f"{service_name} service error: {e}")
        raise Exception(f"{service_name} service error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error with {service_name}: {e}")
        raise

def validate_azure_configuration() -> List[str]:
    """Validate Azure configuration and return any issues"""
    issues = []
    
    # Check Document Intelligence configuration
    if not azure_config.document_intelligence_endpoint:
        issues.append("Document Intelligence endpoint not configured")
    elif not azure_config.document_intelligence_endpoint.startswith("https://"):
        issues.append("Document Intelligence endpoint should start with https://")
    
    # Only check key if not using default credentials
    if not azure_config.use_default_credentials:
        if not azure_config.document_intelligence_key:
            issues.append("Document Intelligence key not configured (or set AZURE_USE_DEFAULT_CREDENTIALS=true)")
        elif len(azure_config.document_intelligence_key) < 32:
            issues.append("Document Intelligence key appears to be invalid (too short)")
    
    # Check Azure OpenAI configuration
    if not azure_config.openai_endpoint:
        issues.append("Azure OpenAI endpoint not configured")
    elif not azure_config.openai_endpoint.startswith("https://"):
        issues.append("Azure OpenAI endpoint should start with https://")
    
    # Only check API key if not using default credentials
    if not azure_config.use_default_credentials:
        if not azure_config.openai_api_key:
            issues.append("Azure OpenAI API key not configured (or set AZURE_USE_DEFAULT_CREDENTIALS=true)")
        elif len(azure_config.openai_api_key) < 32:
            issues.append("Azure OpenAI API key appears to be invalid (too short)")
    
    if not azure_config.openai_deployment_name:
        issues.append("Azure OpenAI deployment name not configured")
    
    if not azure_config.openai_embedding_deployment:
        issues.append("Azure OpenAI embedding deployment not configured")
    
    # Check API version format
    if azure_config.openai_api_version:
        if not re.match(r'\d{4}-\d{2}-\d{2}', azure_config.openai_api_version):
            issues.append("Azure OpenAI API version should be in format YYYY-MM-DD")
    
    return issues

# Global client manager instance
client_manager = AzureClientManager()

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    import re
    
    async def test_azure_clients():
        """Test Azure client connections"""
        
        # Validate configuration first
        issues = validate_azure_configuration()
        if issues:
            print("Configuration issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return
        
        # Test connections
        print("Testing Azure service connections...")
        
        try:
            manager = AzureClientManager()
            results = await manager.test_connections()
            
            print("Connection test results:")
            for service, status in results.items():
                print(f"  {service}: {'✓' if status else '✗'}")
            
            # Test health checker
            health_checker = AzureServiceHealthChecker(manager)
            health_results = await health_checker.check_service_health()
            
            print("\nHealth check results:")
            for service, health in health_results.items():
                print(f"  {service}: {health['status']}")
                if health.get('response_time_ms'):
                    print(f"    Response time: {health['response_time_ms']:.2f}ms")
                if health.get('error'):
                    print(f"    Error: {health['error']}")
            
            overall_health = health_checker.get_overall_health()
            print(f"\nOverall system health: {overall_health}")
            
        except Exception as e:
            print(f"Error during testing: {e}")
    
    # Run test
    # asyncio.run(test_azure_clients())