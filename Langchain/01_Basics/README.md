# LangChain Basics - Getting Started

## What is LangChain?

LangChain is a framework for developing applications powered by language models. It enables you to:
- **Connect LLMs** to other sources of data
- **Chain together** multiple components to create sophisticated applications
- **Add memory** to conversations
- **Create agents** that can use tools and make decisions

## Core Concepts

### 1. Components
- **LLMs**: Large Language Models (OpenAI, Anthropic, etc.)
- **Prompts**: Templates for structuring inputs
- **Chains**: Sequences of operations
- **Memory**: Storing conversation history
- **Agents**: AI that can use tools and make decisions

### 2. Architecture
```
Input → Prompt Template → LLM → Output Parser → Final Output
```

### 3. Key Benefits
- **Modularity**: Reusable components
- **Flexibility**: Mix and match different LLMs
- **Extensibility**: Easy to add custom tools
- **Community**: Large ecosystem of integrations

## Your First LangChain Application

Let's start with a simple example that demonstrates the core concepts.

## Installation

```bash
pip install langchain langchain-openai python-dotenv
```

## Environment Setup

1. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Examples Walkthrough

Run the examples in order:
1. `basic_llm.py` - Simple LLM call
2. `prompt_template.py` - Using templates
3. `simple_chain.py` - Chaining components
4. `conversation_memory.py` - Adding memory

## Exercise

After running the examples, try to:
1. Create a chain that translates text to different languages
2. Add a custom prompt that includes context about the user
3. Experiment with different temperature settings

## Next Steps

Once comfortable with basics, move to `02_Prompts/` to learn advanced prompt engineering techniques.
