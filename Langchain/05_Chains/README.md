# Modern Chains - Building Complex Workflows

## Overview

This module demonstrates modern LangChain patterns for building complex workflows using the latest LCEL (LangChain Expression Language) and runnable chains. We focus on composable, type-safe, and production-ready chain patterns.

## Modern Chain Patterns

### 1. **LCEL Chains (Recommended)**
- **Pipe Operator**: `prompt | llm | output_parser`
- **RunnableSequence**: Explicit sequential composition
- **RunnableParallel**: Concurrent execution
- **RunnableBranch**: Conditional routing

### 2. **Streaming and Async**
- **Streaming responses**: Real-time output generation
- **Async execution**: Non-blocking chain operations
- **Batch processing**: Efficient bulk operations
- **Parallel branches**: Concurrent chain execution

### 3. **Advanced Composition**
- **Runnable binding**: Parameter binding and configuration
- **Chain nesting**: Chains within chains
- **Dynamic routing**: Context-aware chain selection
- **Error handling**: Robust failure management

### 4. **Production Patterns**
- **Input validation**: Type-safe chain inputs
- **Output parsing**: Structured response handling
- **Monitoring**: Comprehensive chain observability
- **Caching**: Performance optimization

## Key Modern Concepts

### LCEL (LangChain Expression Language)
```python
# Modern approach
chain = prompt | llm | output_parser

# vs Legacy approach (deprecated)
chain = LLMChain(llm=llm, prompt=prompt)
```

### Runnable Interface
- **invoke()**: Single synchronous execution
- **stream()**: Streaming responses
- **batch()**: Bulk processing
- **ainvoke()**: Async single execution

### Type Safety
- **Input schemas**: Pydantic models for validation
- **Output schemas**: Structured response parsing
- **Runtime validation**: Automatic input/output checking

## Modern Chain Architecture

### Composable Design
```
Input → Transform → Process → Parse → Output
  ↓        ↓         ↓        ↓       ↓
Validate → Map →   LLM →   Parse → Format
```

### Parallel Processing
```
Input → Branch → [Chain1, Chain2, Chain3] → Merge → Output
```

### Conditional Flow
```
Input → Condition → Chain A (if true) or Chain B (if false) → Output
```

## Examples in This Module

1. **lcel_basics.py** - LCEL fundamentals and pipe operations
2. **parallel_chains.py** - Concurrent chain execution
3. **conditional_chains.py** - Dynamic routing and branching
4. **streaming_chains.py** - Real-time response generation
5. **production_chains.py** - Enterprise-ready patterns

## Modern Best Practices

### Performance
- **Use streaming** for long-running operations
- **Implement caching** for repeated computations
- **Batch operations** when possible
- **Monitor execution** times and costs

### Reliability
- **Input validation** with Pydantic schemas
- **Error boundaries** with graceful degradation
- **Retry mechanisms** for transient failures
- **Comprehensive logging** for debugging

### Maintainability
- **Modular design** with small, focused chains
- **Type annotations** for better IDE support
- **Clear naming** for chain components
- **Documentation** of chain behavior

## Migration from Legacy Chains

### Before (Legacy)
```python
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(input="text")
```

### After (Modern)
```python
chain = prompt | llm
result = chain.invoke({"input": "text"})
```

## Production Deployment

### Monitoring
- **Trace chain execution** with callbacks
- **Track performance metrics** (latency, cost)
- **Log errors and exceptions** for debugging
- **Monitor token usage** and rate limits

### Scaling
- **Async/await patterns** for I/O operations
- **Connection pooling** for external services
- **Load balancing** across multiple instances
- **Horizontal scaling** with stateless chains

### Security
- **Input sanitization** to prevent injection
- **Output filtering** for sensitive content
- **Rate limiting** to prevent abuse
- **Access control** for chain endpoints

## Integration Patterns

### Web APIs
- **FastAPI integration** with async chains
- **Streaming endpoints** for real-time responses
- **WebSocket support** for interactive chains
- **Request validation** with Pydantic

### Message Queues
- **Celery integration** for background processing
- **Redis streams** for chain queuing
- **Event-driven chains** with pub/sub patterns
- **Dead letter queues** for failure handling

## Next Steps

After mastering modern chains:
1. Explore `06_Agents/` for autonomous decision-making
2. Learn about tool integration and function calling
3. Build production-ready chain applications
4. Implement advanced monitoring and observability
