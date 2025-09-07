# 🤖 Intelligent Document Processor - Complete Setup Guide

This guide will help you set up and run the Intelligent Document Processor, a comprehensive document processing and Q&A system using Azure OpenAI and embeddings.

## 📋 Prerequisites

### 1. Python Requirements
- Python 3.9 or higher
- pip package manager

### 2. Azure Services
You'll need access to the following Azure services:

#### Azure Document Intelligence (Form Recognizer)
**What is Azure Document Intelligence?**
Azure Document Intelligence is Microsoft's AI-powered document analysis service that extracts:
- **Text from any document** (PDFs, images, scanned documents)
- **Tables and structured data** 
- **Key-value pairs** (form fields)
- **Document layout and structure**

**Supported Formats:**
- Documents: PDF, DOCX, TXT
- Images: PNG, JPG, JPEG, TIFF, BMP (with OCR)
- Spreadsheets: XLSX
- Presentations: PPTX

**Setup:**
- Create an Azure Document Intelligence resource in the Azure portal
- Note down the endpoint and key
- Choose the appropriate pricing tier for your usage

#### Azure OpenAI Service
**What you'll need:**
- Create an Azure OpenAI resource
- Deploy the following models:
  - **GPT-4** (or GPT-3.5-turbo) for question answering
  - **text-embedding-ada-002** for document embeddings
- Note down the endpoint and deployment names

**Authentication Options:**
- **Azure Default Credentials** (recommended): Uses your Azure login automatically
- **API Key**: Manual key-based authentication

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
1. Clone or download this project
2. Navigate to the project directory
3. Run the setup script:
   ```bash
   python run.py
   ```
4. Follow the prompts to install dependencies and configure the application

### Option 2: Manual Setup

#### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 2: Configure Environment
1. Copy the environment template:
   ```bash
   copy env_template.txt .env
   ```
   
2. Edit `.env` file with your Azure credentials:
   ```bash
   # Azure Document Intelligence (extracts text, tables, structure from documents)
   AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
   
   # Azure OpenAI
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_DEPLOYMENT_NAME=your_gpt4_deployment
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your_embedding_deployment
   
   # Authentication (choose one):
   # Option 1: Use Azure Default Credentials (recommended)
   AZURE_USE_DEFAULT_CREDENTIALS=true
   
   # Option 2: Use API Keys (for local development)
   # AZURE_USE_DEFAULT_CREDENTIALS=false
   # AZURE_DOCUMENT_INTELLIGENCE_KEY=your_doc_intel_key_here
   # AZURE_OPENAI_API_KEY=your_openai_key_here
   ```

#### Step 3: Create Required Directories
```bash
mkdir data
mkdir data/vector_store
mkdir logs
```

#### Step 4: Test Configuration
```bash
python example.py
```
Choose option 2 to run a health check.

#### Step 5: Launch Application
```bash
streamlit run app.py
```

## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` | Document Intelligence service endpoint | `https://your-resource.cognitiveservices.azure.com/` |
| `AZURE_DOCUMENT_INTELLIGENCE_KEY` | Document Intelligence API key | `your_api_key` |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI service endpoint | `https://your-resource.openai.azure.com/` |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | `your_api_key` |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | GPT model deployment name | `gpt-4` |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding model deployment name | `text-embedding-ada-002` |
| `MAX_FILE_SIZE_MB` | Maximum file size in MB | `50` |
| `CHUNK_SIZE` | Text chunk size for embeddings | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `100` |

### Supported File Types
- **Documents**: PDF, DOCX, TXT
- **Spreadsheets**: XLSX
- **Presentations**: PPTX
- **Images**: PNG, JPG, JPEG, TIFF, BMP

## 📱 Using the Application

### 1. Document Upload & Processing
1. Navigate to the "Document Upload & Processing" tab
2. Upload your documents (drag & drop or click to browse)
3. Review file information and click "Process Documents"
4. Wait for processing to complete

### 2. Question & Answer
1. Go to the "Question & Answer" tab
2. Select documents to search (or "All Documents")
3. Enter your question in the text area
4. Click "Get Answer" to receive an AI-generated response
5. Review sources and follow-up questions

### 3. Document Analytics
1. Visit the "Document Analytics" tab
2. View document overview and statistics
3. Select specific documents for detailed analysis
4. Explore extracted tables and key-value pairs

### 4. System Status
1. Check the "System Status" tab for:
   - Azure service health
   - Application statistics
   - Configuration details
   - Session management

## 🎯 Example Use Cases

### Legal Documents
- Contract analysis and comparison
- Legal document search and review
- Compliance checking
- Due diligence processing

### Healthcare
- Medical records analysis
- Research paper insights
- Clinical trial documentation
- Patient information extraction

### Finance
- Financial report analysis
- Invoice processing
- Audit document review
- Risk assessment reports

### HR & Recruitment
- Resume screening
- Policy document Q&A
- Training material insights
- Employee handbook search

## 🛠️ Troubleshooting

### Common Issues

#### Configuration Errors
- **Issue**: "Configuration issues detected"
- **Solution**: Check your `.env` file and ensure all Azure credentials are correct

#### Import Errors
- **Issue**: Module import errors
- **Solution**: Reinstall dependencies with `pip install -r requirements.txt`

#### Azure Service Errors
- **Issue**: Authentication or service errors
- **Solution**: 
  - Verify your Azure credentials
  - Check service quotas and limits
  - Ensure proper permissions

#### Memory Issues
- **Issue**: Out of memory errors
- **Solution**: 
  - Reduce `CHUNK_SIZE` in configuration
  - Process smaller documents
  - Restart the application

#### Slow Processing
- **Issue**: Documents process slowly
- **Solution**:
  - Check your internet connection
  - Verify Azure service region
  - Reduce concurrent requests

### Getting Help

1. **Check the logs**: Look in the `logs/` directory for error details
2. **Run health check**: Use `python example.py` and select option 2
3. **Review configuration**: Ensure all environment variables are set correctly
4. **Test Azure services**: Use the "System Status" tab to check service health

## 🚀 Advanced Usage

### Programmatic Usage
You can use the components programmatically:

```python
import asyncio
from src.document_processor import DocumentProcessor
from src.embedding_manager import EmbeddingManager
from src.qa_engine import QAEngine

async def process_document():
    # Initialize components
    processor = DocumentProcessor()
    embedding_manager = EmbeddingManager()
    qa_engine = QAEngine(embedding_manager)
    
    # Process document
    with open("document.pdf", "rb") as f:
        content = f.read()
    
    result = await processor.process_document(content, "document.pdf")
    
    if result.success:
        # Create embeddings
        embedding_result = await embedding_manager.create_embeddings(
            result.content, "doc_1", "document.pdf"
        )
        
        # Ask questions
        answer = await qa_engine.answer_question("What is this document about?")
        print(answer.answer)

# Run
asyncio.run(process_document())
```

### Custom Configuration
You can customize the behavior by modifying `config.py`:

```python
# Adjust chunk size for different document types
app_config.chunk_size = 1500  # Larger chunks for technical documents

# Modify search parameters
embedding_config.similarity_threshold = 0.8  # Higher threshold for precision
embedding_config.max_search_results = 15  # More search results

# Adjust Q&A settings
qa_config.temperature = 0.2  # More focused answers
qa_config.max_tokens = 1500  # Longer responses
```

## 📊 Performance Optimization

### For Large Documents
- Increase `MAX_CHUNKS_PER_DOCUMENT` if needed
- Use appropriate `CHUNK_SIZE` for your content type
- Consider processing documents in batches

### For High Volume
- Monitor Azure service quotas
- Implement caching for frequently accessed documents
- Consider using Azure Cognitive Search for large-scale deployments

### Cost Optimization
- Monitor token usage in the application
- Adjust chunk sizes to optimize embedding costs
- Use appropriate confidence thresholds to reduce API calls

## 🔒 Security Considerations

### Environment Variables
- Never commit `.env` files to version control
- Use Azure Key Vault for production deployments
- Rotate API keys regularly

### Data Handling
- Documents are processed in memory and cleaned up automatically
- Vector embeddings are stored locally in ChromaDB
- No document content is stored permanently unless explicitly configured

### Network Security
- Ensure secure connections to Azure services
- Consider VPN or private endpoints for sensitive data
- Implement rate limiting and monitoring

## 📈 Monitoring & Analytics

### Application Metrics
- Processing times and success rates
- Token usage and costs
- User interaction patterns
- Error rates and types

### Azure Service Monitoring
- API call volumes and latency
- Service availability and health
- Quota usage and limits

## 🎛️ Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Docker Deployment
```bash
# Build image
docker build -t intelligent-doc-processor .

# Run container
docker run -p 8501:8501 intelligent-doc-processor
```

### Azure Container Instances
Deploy using the provided Azure deployment templates in the `deploy/` directory.

---

## 📚 Additional Resources

- [Azure Document Intelligence Documentation](https://docs.microsoft.com/azure/applied-ai-services/form-recognizer/)
- [Azure OpenAI Service Documentation](https://docs.microsoft.com/azure/cognitive-services/openai/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

## 🤝 Support

For issues and questions:
1. Check this documentation
2. Review the example scripts
3. Test with the health check functionality
4. Check Azure service status and quotas

---

**Built with ❤️ for intelligent document processing**
