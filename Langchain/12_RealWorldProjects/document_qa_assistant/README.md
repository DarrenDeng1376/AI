# Document Q&A Assistant

A production-ready document Q&A system built with LangChain, Streamlit, and OpenAI.

## Features

- 📄 **Multi-format Support**: PDF, DOCX, TXT files
- 🧠 **Intelligent Answers**: RAG-powered responses with source citations
- 💬 **Conversation Memory**: Maintains context across questions
- 🎯 **Source Citations**: Shows which documents were used for answers
- 🚀 **Clean Interface**: Professional Streamlit web app

## Quick Start

### 1. Install Dependencies
```bash
pip install streamlit langchain langchain-openai chromadb python-dotenv
```

### 2. Set Environment Variables
Create a `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run the Application
```bash
streamlit run app.py
```

### 4. Use the App
1. Click "Load Sample Documents" in the sidebar
2. Wait for processing to complete
3. Ask questions about the documents
4. View answers with source citations

## Sample Questions

Try asking these questions about the company handbook:

- "What is the remote work policy?"
- "What benefits does the company offer?"
- "When do performance reviews happen?"
- "How much is the learning budget?"

## Architecture

```
User Question → Vector Search → Relevant Documents → LLM + Context → Answer + Sources
```

### Components:
- **Document Loader**: Processes multiple file formats
- **Text Splitter**: Breaks documents into searchable chunks
- **Vector Store**: ChromaDB for similarity search
- **LLM Chain**: OpenAI for answer generation
- **Web Interface**: Streamlit for user interaction

## Production Considerations

This is a simplified demo. For production use, consider:

### Scalability
- Use a persistent vector database (Pinecone, Weaviate)
- Implement caching for expensive operations
- Add horizontal scaling with load balancers

### Security
- Implement user authentication
- Add input validation and sanitization
- Secure API key management
- Rate limiting for API calls

### Monitoring
- Track user interactions and feedback
- Monitor API costs and usage
- Log errors and performance metrics
- A/B testing for different prompt strategies

### User Experience
- File upload functionality
- Better error handling
- Progress indicators for long operations
- Mobile-responsive design

## Customization

### Adding New Document Types
```python
# In load_documents() function
from langchain.document_loaders import UnstructuredWordDocumentLoader

if file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
    loader = UnstructuredWordDocumentLoader(file_path)
```

### Custom Prompt Templates
```python
from langchain.prompts import PromptTemplate

custom_prompt = PromptTemplate(
    template="""Use the following pieces of context to answer the question. 
    If you don't know the answer, just say that you don't know.

    Context: {context}
    Question: {question}
    
    Answer in a professional, helpful tone:""",
    input_variables=["context", "question"]
)
```

### Different LLM Models
```python
# Use different OpenAI models
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.1
)

# Or use other providers
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=0
)
```

## Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Docker Container
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Cloud Deployment
- **Streamlit Cloud**: Connect GitHub repo for automatic deployment
- **Heroku**: Use Procfile with `web: streamlit run app.py --server.port=$PORT`
- **AWS/GCP/Azure**: Deploy using container services

## Learning Outcomes

Building this project teaches:

- **RAG Implementation**: Document processing and retrieval
- **Vector Databases**: Similarity search and embeddings
- **Web Interfaces**: Streamlit for AI applications
- **Production Patterns**: Error handling, logging, monitoring
- **LangChain Advanced**: Custom chains and prompt engineering

## Next Steps

1. **Add File Upload**: Let users upload their own documents
2. **Implement Caching**: Speed up repeated queries
3. **Add User Feedback**: Thumbs up/down for answer quality
4. **Multi-language Support**: Handle documents in different languages
5. **Advanced RAG**: Implement query expansion and re-ranking

## Troubleshooting

### Common Issues

**Import Errors**: Install missing packages
```bash
pip install streamlit langchain langchain-openai chromadb
```

**API Key Errors**: Check your `.env` file
```
OPENAI_API_KEY=sk-your-key-here
```

**ChromaDB Issues**: Clear the database directory
```bash
rm -rf ./chroma_db
```

**Memory Issues**: Reduce chunk size or document count

## Contributing

This is a learning project. Feel free to:
- Add new features
- Improve the UI
- Fix bugs
- Add tests
- Improve documentation

---

*This project demonstrates production-ready patterns for LangChain applications. Use it as a foundation for your own document Q&A systems!*
