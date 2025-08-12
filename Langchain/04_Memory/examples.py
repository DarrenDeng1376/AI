"""
Memory and Conversation Management - Using Azure OpenAI

This module demonstrates different types of memory in LangChain:
1. Buffer memory types and when to use them
2. Conversation chains with memory
3. Custom memory implementations
4. Persistent memory storage
5. Memory optimization for long conversations
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from azure_config import create_azure_chat_openai, create_azure_openai
from dotenv import load_dotenv

load_dotenv()

# Example 1: Different Memory Types
def memory_types_example():
    """
    Compare different memory types and their characteristics
    """
    print("=== Example 1: Memory Types Comparison ===")
    
    from langchain.chains import ConversationChain
    from langchain.memory import (
        ConversationBufferMemory,
        ConversationBufferWindowMemory,
        ConversationSummaryMemory,
        ConversationSummaryBufferMemory
    )
    
    llm = create_azure_chat_openai(temperature=0.7)
    
    # 1. Buffer Memory - stores everything
    print("🧠 1. ConversationBufferMemory:")
    buffer_memory = ConversationBufferMemory()
    buffer_chain = ConversationChain(llm=llm, memory=buffer_memory, verbose=False)
    
    # Test conversation
    buffer_chain.predict(input="Hi, I'm Alice")
    buffer_chain.predict(input="What's my name?")
    
    print(f"Buffer Memory: {buffer_memory.buffer}")
    print()
    
    # 2. Window Memory - keeps last N exchanges  
    print("🪟 2. ConversationBufferWindowMemory:")
    window_memory = ConversationBufferWindowMemory(k=2)  # Keep last 2 exchanges
    window_chain = ConversationChain(llm=llm, memory=window_memory, verbose=False)
    
    window_chain.predict(input="Hi, I'm Bob")
    window_chain.predict(input="I like pizza")
    window_chain.predict(input="What's my name?")  # Should remember
    window_chain.predict(input="What do I like?")  # May not remember pizza
    
    print(f"Window Memory: {window_memory.buffer}")
    print()
    
    # 3. Summary Memory - summarizes old conversations
    print("📄 3. ConversationSummaryMemory:")
    summary_memory = ConversationSummaryMemory(llm=llm)
    summary_chain = ConversationChain(llm=llm, memory=summary_memory, verbose=False)
    
    summary_chain.predict(input="Hi, I'm Charlie and I work as a software engineer")
    summary_chain.predict(input="I'm working on a machine learning project")
    
    print(f"Summary: {summary_memory.buffer}")
    print()
    
    print("="*50 + "\n")


# Example 2: Conversation Chain with Memory
def conversation_chain_example():
    """
    Build a sophisticated conversation system with memory
    """
    print("=== Example 2: Advanced Conversation Chain ===")
    
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationSummaryBufferMemory
    from langchain.prompts import PromptTemplate
    
    llm = create_azure_chat_openai(temperature=0.7)
    
    # Custom conversation prompt
    template = """You are a helpful AI assistant with a good memory. 
    Use the conversation history to provide personalized responses.

    Current conversation:
    {history}
    Human: {input}
    AI Assistant:"""
    
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )
    
    # Use summary buffer memory for efficiency
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=150,
        return_messages=True
    )
    
    conversation = ConversationChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=False
    )
    
    # Simulate a conversation
    conversation_flow = [
        "Hi! I'm Sarah, a data scientist at TechCorp",
        "I'm working on a customer segmentation project using Python",
        "What programming languages are good for machine learning?",
        "Can you remind me what I told you about my job?",
        "What project am I working on?"
    ]
    
    print("💬 Conversation with Advanced Memory:")
    for user_input in conversation_flow:
        response = conversation.predict(input=user_input)
        print(f"Human: {user_input}")
        print(f"AI: {response}")
        print("-" * 40)
    
    # Show memory state
    print(f"📋 Final Memory State: {memory.buffer}")
    print("="*50 + "\n")


# Example 3: Custom Memory Implementation
def custom_memory_example():
    """
    Create a custom memory that stores structured information
    """
    print("=== Example 3: Custom Memory Implementation ===")
    
    from langchain.memory import BaseChatMemory
    from langchain.schema import BaseMessage, HumanMessage, AIMessage
    from typing import List, Dict, Any
    
    class PersonalizedMemory(BaseChatMemory):
        """Custom memory that extracts and stores personal information"""
        
        def __init__(self):
            super().__init__()
            self.personal_info = {}
            self.conversation_history = []
        
        def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
            """Save conversation and extract personal info"""
            human_message = inputs["input"]
            ai_message = outputs["response"]
            
            # Store conversation
            self.conversation_history.append(f"Human: {human_message}")
            self.conversation_history.append(f"AI: {ai_message}")
            
            # Extract personal info (simplified)
            if "my name is" in human_message.lower():
                name = human_message.lower().split("my name is")[1].strip().split()[0]
                self.personal_info["name"] = name
            
            if "i work as" in human_message.lower():
                job = human_message.lower().split("i work as")[1].strip()
                self.personal_info["job"] = job
            
            if "i like" in human_message.lower():
                interest = human_message.lower().split("i like")[1].strip()
                self.personal_info.setdefault("interests", []).append(interest)
        
        def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Load memory variables for the prompt"""
            recent_history = self.conversation_history[-6:] if len(self.conversation_history) > 6 else self.conversation_history
            
            return {
                "history": "\n".join(recent_history),
                "personal_info": str(self.personal_info) if self.personal_info else "No personal info yet"
            }
        
        def clear(self) -> None:
            """Clear memory"""
            self.personal_info = {}
            self.conversation_history = []
    
    # Test custom memory
    custom_memory = PersonalizedMemory()
    
    # Simulate saving context
    test_inputs = [
        {"input": "Hi, my name is David"},
        {"input": "I work as a marketing manager"},
        {"input": "I like hiking and photography"}
    ]
    
    test_outputs = [
        {"response": "Nice to meet you, David!"},
        {"response": "Marketing is an interesting field!"},
        {"response": "Hiking and photography are great hobbies!"}
    ]
    
    for inp, out in zip(test_inputs, test_outputs):
        custom_memory.save_context(inp, out)
    
    # Load memory
    memory_vars = custom_memory.load_memory_variables({})
    
    print("🧠 Custom Memory Results:")
    print(f"Personal Info: {custom_memory.personal_info}")
    print(f"History: {memory_vars['history']}")
    print("="*50 + "\n")


# Example 4: Persistent Memory
def persistent_memory_example():
    """
    Save and load memory between sessions
    """
    print("=== Example 4: Persistent Memory ===")
    
    import json
    import os
    from langchain.memory import ConversationBufferMemory
    
    class PersistentMemory:
        """Memory that can be saved to and loaded from disk"""
        
        def __init__(self, file_path="memory.json"):
            self.file_path = file_path
            self.memory = ConversationBufferMemory()
            self.load_memory()
        
        def save_memory(self):
            """Save memory to disk"""
            memory_data = {
                "buffer": self.memory.buffer,
                "chat_memory": [
                    {"type": type(msg).__name__, "content": msg.content}
                    for msg in self.memory.chat_memory.messages
                ]
            }
            
            with open(self.file_path, 'w') as f:
                json.dump(memory_data, f, indent=2)
            
            print(f"💾 Memory saved to {self.file_path}")
        
        def load_memory(self):
            """Load memory from disk"""
            if os.path.exists(self.file_path):
                try:
                    with open(self.file_path, 'r') as f:
                        memory_data = json.load(f)
                    
                    self.memory.buffer = memory_data.get("buffer", "")
                    print(f"📂 Memory loaded from {self.file_path}")
                    
                except Exception as e:
                    print(f"Error loading memory: {e}")
            else:
                print("📝 Starting with fresh memory")
        
        def add_message(self, human_message, ai_message):
            """Add messages to memory"""
            self.memory.save_context(
                {"input": human_message},
                {"output": ai_message}
            )
            self.save_memory()  # Auto-save
    
    # Test persistent memory
    persistent = PersistentMemory("test_memory.json")
    
    # Add some conversation
    persistent.add_message("Hello, I'm testing persistent memory", "Great! I'll remember this conversation")
    persistent.add_message("What did I just say?", "You said you're testing persistent memory")
    
    print(f"📋 Current memory buffer: {persistent.memory.buffer}")
    
    # Clean up test file
    if os.path.exists("test_memory.json"):
        os.remove("test_memory.json")
    
    print("="*50 + "\n")


# Example 5: Memory Optimization
def memory_optimization_example():
    """
    Strategies for managing memory in long conversations
    """
    print("=== Example 5: Memory Optimization ===")
    
    from langchain_openai import OpenAI
    from langchain.memory import ConversationSummaryBufferMemory
    import tiktoken
    
    llm = OpenAI(temperature=0.7)
    
    # Token counting function
    def count_tokens(text):
        """Count tokens in text"""
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))
    
    # Create memory with token limit
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=200,  # Small limit for demo
        return_messages=True
    )
    
    # Simulate a long conversation
    long_conversation = [
        ("Hello, I'm interested in learning about machine learning", "Great! Machine learning is fascinating"),
        ("What are the main types of ML algorithms?", "There are supervised, unsupervised, and reinforcement learning"),
        ("Can you explain supervised learning in detail?", "Supervised learning uses labeled data to train models"),
        ("What about unsupervised learning?", "Unsupervised learning finds patterns in unlabeled data"),
        ("How does reinforcement learning work?", "It learns through trial and error with rewards"),
        ("What programming languages are best for ML?", "Python and R are the most popular choices"),
        ("Can you recommend some ML libraries?", "Scikit-learn, TensorFlow, and PyTorch are excellent"),
        ("What's the difference between AI and ML?", "AI is broader, ML is a subset focused on learning from data")
    ]
    
    print("💬 Long Conversation Simulation:")
    
    for i, (human_msg, ai_msg) in enumerate(long_conversation):
        memory.save_context({"input": human_msg}, {"output": ai_msg})
        
        # Check memory state
        buffer = memory.buffer
        token_count = count_tokens(buffer)
        
        print(f"Turn {i+1}: {token_count} tokens")
        
        if i % 3 == 2:  # Every 3 turns, show memory state
            print(f"Memory state: {buffer[:100]}...")
            print("-" * 40)
    
    print(f"📊 Final memory state:")
    print(f"Buffer: {memory.buffer}")
    print(f"Total tokens: {count_tokens(memory.buffer)}")
    
    print("\n💡 Optimization Tips:")
    print("1. Use ConversationSummaryBufferMemory for long conversations")
    print("2. Set appropriate token limits based on your model")
    print("3. Monitor token usage to avoid exceeding limits")
    print("4. Consider pruning old conversations periodically")
    print("5. Use vector store memory for very long histories")
    
    print("="*50 + "\n")


if __name__ == "__main__":
    print("Memory and Conversation Management Examples")
    print("=" * 60)
    
    try:
        memory_types_example()
        conversation_chain_example()
        custom_memory_example()
        persistent_memory_example()
        memory_optimization_example()
        
        print("🎉 All memory examples completed!")
        print("Next: Check out 05_Chains/ for complex workflows")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have your OPENAI_API_KEY set in the .env file")
