# Intelligent Document Processor

A production-ready application that combines Azure Document Intelligence, Azure OpenAI, and vector embeddings to create an intelligent document processing and Q&A system.

## 🎯 Project Overview

This system can:
- Extract text, tables, and structure from various document formats (PDF, images, Office docs)
- Generate embeddings for semantic search and retrieval
- Answer questions about document content using RAG (Retrieval-Augmented Generation)
- Provide structured analysis of documents with confidence scores
- Handle multiple document types with different processing strategies

## 🏗️ Architecture

```
Document Upload → Azure Document Intelligence → Text Extraction & Structure
                                               ↓
Text Chunking → Azure OpenAI Embeddings → Vector Store (ChromaDB)
                                               ↓
User Question → Vector Search → Context Retrieval → Azure OpenAI → Answer
```

## 🚀 Features

### Document Processing
- **Multi-format Support**: PDF, PNG, JPG, TIFF, BMP, DOCX, XLSX, PPTX
- **Intelligent OCR**: Extract text from scanned documents
- **Structure Recognition**: Tables, forms, headers, paragraphs
- **Layout Analysis**: Preserve document structure and formatting
- **Batch Processing**: Handle multiple documents simultaneously

### Advanced Search & QA
- **Semantic Search**: Find relevant content using vector similarity
- **Hybrid Search**: Combine keyword and semantic search
- **Contextual Answers**: Generate answers with source citations
- **Confidence Scoring**: Provide reliability metrics for answers
- **Follow-up Questions**: Handle conversational context

### Analytics & Insights
- **Document Insights**: Extract key information, entities, and themes
- **Comparison Analysis**: Compare multiple documents
- **Summary Generation**: Create executive summaries
- **Trend Analysis**: Identify patterns across document collections

## 🛠️ Tech Stack

- **Azure Document Intelligence**: Document structure and text extraction
- **Azure OpenAI**: Text embeddings and GPT-4 for Q&A
- **LangChain**: Framework for LLM applications
- **ChromaDB**: Vector database for embeddings
- **Streamlit**: Web interface
- **Python 3.9+**: Core development

## 📋 Prerequisites

- Azure subscription with access to:
  - Azure Document Intelligence (Form Recognizer)
  - Azure OpenAI Service
- Python 3.9 or higher
- Git

## 🚀 Quick Start

### 1. Clone and Setup
```bash
cd intelligent_document_processor
pip install -r requirements.txt
```

### 2. Configure Azure Services
```bash
cp .env.example .env
# Edit .env with your Azure credentials
```

### 3. Run the Application
```bash
streamlit run app.py
```

## 📁 Project Structure

```
intelligent_document_processor/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── app.py                    # Streamlit web application
├── config.py                 # Configuration management
├── src/                      # Source code
│   ├── __init__.py
│   ├── document_processor.py  # Azure Document Intelligence integration
│   ├── embedding_manager.py   # Vector embeddings and search
│   ├── qa_engine.py          # Question-answering system
│   ├── document_analyzer.py   # Document insights and analysis
│   └── utils/
│       ├── __init__.py
│       ├── text_processing.py # Text chunking and preprocessing
│       ├── file_handlers.py   # File upload and management
│       └── azure_clients.py   # Azure service clients
├── prompts/                  # Prompt templates
│   ├── qa_prompts.py
│   ├── analysis_prompts.py
│   └── summary_prompts.py
├── data/                     # Sample documents and test data
│   ├── sample_documents/
│   └── vector_store/
├── tests/                    # Unit tests
│   ├── test_document_processor.py
│   ├── test_embedding_manager.py
│   └── test_qa_engine.py
└── deploy/                   # Deployment configurations
    ├── Dockerfile
    ├── docker-compose.yml
    └── azure_deployment.yml
```

## 🔧 Configuration

### Environment Variables
```bash
# Azure Document Intelligence
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=your_endpoint

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT_NAME=your_gpt4_deployment
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your_embedding_deployment

# Authentication (choose one):
# Option 1: Use Azure Default Credentials (recommended)
AZURE_USE_DEFAULT_CREDENTIALS=true

# Option 2: Use API Keys (for local development)
# AZURE_USE_DEFAULT_CREDENTIALS=false
# AZURE_DOCUMENT_INTELLIGENCE_KEY=your_doc_intel_key
# AZURE_OPENAI_API_KEY=your_openai_key

# Application Settings
VECTOR_STORE_PATH=./data/vector_store
MAX_FILE_SIZE_MB=50
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
```

## 💻 Usage Examples

### Basic Document Q&A
1. Upload a document (PDF, image, or Office file)
2. Wait for processing and indexing
3. Ask questions about the document content
4. Get answers with source citations and confidence scores

### Advanced Analytics
1. Upload multiple related documents
2. Use the "Document Analysis" tab
3. Generate insights, summaries, and comparisons
4. Export results in various formats

### Batch Processing
1. Upload a folder of documents
2. Configure processing options
3. Monitor progress in real-time
4. Access processed results via the search interface

## 🎯 Use Cases

### Legal & Compliance
- Contract analysis and comparison
- Legal document search and review
- Compliance checking against regulations
- Due diligence document processing

### Healthcare
- Medical records analysis
- Research paper insights
- Clinical trial document processing
- Patient information extraction

### Finance
- Financial report analysis
- Invoice and receipt processing
- Audit document review
- Risk assessment reports

### HR & Recruitment
- Resume screening and analysis
- Policy document Q&A
- Training material insights
- Employee handbook search

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

Run integration tests:
```bash
pytest tests/ -m integration
```

Load testing:
```bash
python tests/load_test.py
```

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Docker
```bash
docker build -t intelligent-doc-processor .
docker run -p 8501:8501 intelligent-doc-processor
```

### Azure Container Instances
```bash
az deployment group create --resource-group myResourceGroup --template-file deploy/azure_deployment.yml
```

## 📊 Performance & Scaling

### Optimization Tips
- Use appropriate chunk sizes for your document types
- Implement caching for frequently accessed documents
- Consider using Azure Cognitive Search for large-scale deployments
- Monitor token usage and costs

### Scaling Considerations
- Horizontal scaling with multiple app instances
- Database optimization for large vector stores
- CDN for static assets
- Load balancing for high availability

## 🔒 Security

- API key management with Azure Key Vault
- Input validation and sanitization
- Rate limiting and throttling
- Audit logging for document access
- Data encryption at rest and in transit

## 📈 Monitoring & Analytics

### Application Metrics
- Document processing times
- Query response times
- User satisfaction scores
- Error rates and types

### Business Metrics
- Document volume processed
- User engagement patterns
- Feature usage statistics
- Cost optimization insights

## 🎛️ Advanced Features

### Custom Document Types
- Extend the processor for industry-specific documents
- Create custom extraction rules
- Implement domain-specific preprocessing

### Multi-language Support
- Language detection and processing
- Cross-language search capabilities
- Localized user interfaces

### API Integration
- RESTful API for programmatic access
- Webhook notifications for processing completion
- Integration with existing business systems

## 🛠️ Troubleshooting

### Common Issues
- **Azure Service Quotas**: Monitor and request increases as needed
- **Large Document Processing**: Implement chunking strategies
- **Memory Usage**: Optimize vector store operations
- **API Rate Limits**: Implement exponential backoff

### Performance Tuning
- Adjust chunk sizes based on document types
- Optimize embedding dimensions
- Fine-tune search parameters
- Implement result caching

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📚 Additional Resources

- [Azure Document Intelligence Documentation](https://docs.microsoft.com/en-us/azure/applied-ai-services/form-recognizer/)
- [Azure OpenAI Service Documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/openai/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction.html)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*Built with ❤️ for intelligent document processing*
