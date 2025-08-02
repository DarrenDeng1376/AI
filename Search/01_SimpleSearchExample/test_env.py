import os
from dotenv import load_dotenv

load_dotenv()

print("Testing environment variables...")
print(f"Endpoint: {os.getenv('AZURE_SEARCH_SERVICE_ENDPOINT')}")
print(f"Key: {os.getenv('AZURE_SEARCH_ADMIN_KEY')[:10]}...")

print("Testing Azure Search import...")
try:
    from azure.search.documents import SearchClient
    print("✅ Azure Search package imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")

print("Environment test completed!")
