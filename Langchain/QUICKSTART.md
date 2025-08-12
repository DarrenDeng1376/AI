# LangChain Quick Start Guide

Welcome to your LangChain learning journey! This guide will get you up and running in minutes.

## 🚀 Quick Setup (5 minutes)

### Step 1: Run Setup Script
```powershell
cd Langchain
python setup.py
```

This will:
- Check your Python version (need 3.8+)
- Install all required packages
- Create environment file
- Run basic tests

### Step 2: Add API Key
1. Get an OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Edit `.env` file and replace `your_openai_api_key_here` with your actual key:
```
OPENAI_API_KEY=sk-your-actual-key-here
```

### Step 3: Test Your Setup
```powershell
cd 01_Basics
python examples.py
```

If you see outputs from the examples, you're ready to go! 🎉

## 📚 Learning Path

### Week 1-2: Foundations
```
01_Basics/          ← Start here!
├── README.md       ← Read this first
├── examples.py     ← Run these examples
├── exercises.py    ← Practice problems
└── solutions.py    ← Check your work

02_Prompts/         ← Advanced prompting
├── README.md
├── examples.py
└── exercises.py
```

### Week 3-4: Building Applications
```
05_Chains/          ← Complex workflows
06_Agents/          ← Autonomous AI
07_Tools/           ← External integrations
08_VectorStores/    ← Document search
```

### Week 5-6: Production Ready
```
09_RAG/             ← Document Q&A systems
10_CustomComponents/← Reusable components
11_Production/      ← Deployment
12_RealWorldProjects/← Complete apps
```

## 💡 How to Learn Effectively

### 1. Follow the Pattern
Each module follows the same structure:
1. **Read** `README.md` for concepts
2. **Run** `examples.py` to see it working
3. **Practice** with `exercises.py`
4. **Check** `solutions.py` if needed

### 2. Experiment
Don't just run the code - modify it:
- Change temperature values
- Try different prompts
- Use different models
- Add your own examples

### 3. Build Projects
After each module, try to build something:
- **01_Basics**: Simple chatbot
- **02_Prompts**: Content generator
- **09_RAG**: Personal document assistant

## 🛠️ Common Issues & Solutions

### Import Errors
```python
# If you see: ImportError: No module named 'langchain_openai'
pip install langchain-openai
```

### API Key Issues
```python
# If you see: openai.AuthenticationError
# Check your .env file has the correct API key format:
OPENAI_API_KEY=sk-your-key-starts-with-sk
```

### Rate Limits
```python
# If you hit OpenAI rate limits, add delays:
import time
time.sleep(1)  # Wait 1 second between calls
```

## 📖 Essential Concepts

### 1. The LangChain Flow
```
Input → Prompt Template → LLM → Output Parser → Result
```

### 2. Core Components
- **LLM**: The language model (OpenAI, Anthropic, etc.)
- **Prompts**: Templates for structuring input
- **Chains**: Sequences of operations
- **Memory**: Conversation history
- **Agents**: Decision-making AI

### 3. Key Patterns
- **Simple Chain**: Input → LLM → Output
- **Sequential Chain**: Chain1 → Chain2 → Chain3
- **RAG**: Query → Retrieve → Generate
- **Agent**: Input → Reason → Act → Observe → Repeat

## 🎯 Learning Goals

### Beginner (Week 1-2)
- [ ] Understand LangChain basics
- [ ] Create simple chains
- [ ] Use prompt templates
- [ ] Add conversation memory

### Intermediate (Week 3-4)  
- [ ] Build complex chains
- [ ] Create AI agents
- [ ] Integrate external tools
- [ ] Work with documents

### Advanced (Week 5-6)
- [ ] Build RAG systems
- [ ] Create custom components
- [ ] Deploy to production
- [ ] Build complete applications

## 🤝 Getting Help

### Resources
- [LangChain Documentation](https://python.langchain.com/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [LangChain Discord](https://discord.gg/langchain)

### In This Repository
- Each `README.md` has detailed explanations
- `examples.py` files are heavily commented
- `solutions.py` files show best practices

### Community
- Ask questions in issues
- Share your projects
- Contribute improvements

## 🚀 Ready to Start?

1. **Complete setup** (if you haven't): `python setup.py`
2. **Start learning**: `cd 01_Basics`
3. **Read the README**: Open `01_Basics/README.md`
4. **Run examples**: `python examples.py`
5. **Practice**: Work through exercises

**Happy Learning! 🎉**

---

*Remember: The best way to learn is by doing. Don't just read - run the code, modify it, break it, fix it, and build something new!*
