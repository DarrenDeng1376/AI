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

from azure_config import create_azure_chat_openai
from dotenv import load_dotenv

load_dotenv()

# Example 1: Different Memory Types
def memory_types_example():
    """
    Compare different memory approaches using modern LangChain patterns
    """
    print("=== Example 1: Modern Memory Management ===")
    
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    
    llm = create_azure_chat_openai(temperature=0.7)
    
    # 1. Simple Message History - stores everything
    print("🧠 1. InMemoryChatMessageHistory:")
    
    # Create prompt template with message history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the conversation history to provide personalized responses."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # Create chain with message history
    chain = prompt | llm
    
    # Simple message history storage
    message_history = InMemoryChatMessageHistory()
    
    # Function to get session history
    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        return message_history
    
    # Create runnable with message history
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )
    
    # Test conversation
    response1 = chain_with_history.invoke(
        {"input": "Hi, I'm Alice"},
        config={"configurable": {"session_id": "test1"}}
    )
    print(f"Response 1: {response1.content}")
    
    response2 = chain_with_history.invoke(
        {"input": "What's my name?"},
        config={"configurable": {"session_id": "test1"}}
    )
    print(f"Response 2: {response2.content}")
    
    print(f"Message History: {len(message_history.messages)} messages stored")
    print()
    
    # 2. Windowed Memory - manually manage last N messages
    print("🪟 2. Windowed Message History:")
    
    class WindowedMessageHistory:
        def __init__(self, window_size=4):  # Keep last 4 messages (2 exchanges)
            self.messages = []
            self.window_size = window_size
        
        def add_user_message(self, message: str):
            self.messages.append(HumanMessage(content=message))
            self._trim_messages()
        
        def add_ai_message(self, message: str):
            self.messages.append(AIMessage(content=message))
            self._trim_messages()
        
        def _trim_messages(self):
            if len(self.messages) > self.window_size:
                self.messages = self.messages[-self.window_size:]
        
        def get_messages(self):
            return self.messages
    
    windowed_history = WindowedMessageHistory(window_size=4)
    
    # Simulate conversation
    conversations = [
        ("Hi, I'm Bob", "Nice to meet you, Bob!"),
        ("I like pizza", "Pizza is delicious!"),
        ("I also enjoy hiking", "Hiking is great exercise!"),
        ("What's my name?", "Your name is Bob."),
        ("What do I like?", "You mentioned liking pizza and hiking.")
    ]
    
    for human_msg, ai_response in conversations:
        windowed_history.add_user_message(human_msg)
        windowed_history.add_ai_message(ai_response)
        
        # Show current window
        recent_messages = windowed_history.get_messages()
        print(f"Window size: {len(recent_messages)} messages")
        if len(recent_messages) >= 2:
            last_exchange = recent_messages[-2:]
            for msg in last_exchange:
                msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
                print(f"  {msg_type}: {msg.content}")
        print("-" * 30)
    
    print()
    
    # 3. Summary-based approach (manual implementation)
    print("📄 3. Summary-based Memory:")
    
    class SummaryMemory:
        def __init__(self, llm, max_exchanges=3):
            self.llm = llm
            self.max_exchanges = max_exchanges
            self.summary = ""
            self.recent_messages = []
        
        def add_exchange(self, human_msg: str, ai_msg: str):
            self.recent_messages.extend([
                HumanMessage(content=human_msg),
                AIMessage(content=ai_msg)
            ])
            
            # If we have too many messages, summarize older ones
            if len(self.recent_messages) > self.max_exchanges * 2:
                self._create_summary()
        
        def _create_summary(self):
            # Take older messages for summary
            messages_to_summarize = self.recent_messages[:-4]  # Keep last 2 exchanges
            
            if messages_to_summarize:
                conversation_text = "\n".join([
                    f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
                    for msg in messages_to_summarize
                ])
                
                summary_prompt = f"""
                Summarize the following conversation in a concise way:
                
                {conversation_text}
                
                Summary:"""
                
                summary_response = self.llm.invoke(summary_prompt)
                self.summary = summary_response.content
                
                # Keep only recent messages
                self.recent_messages = self.recent_messages[-4:]
        
        def get_context(self):
            context_parts = []
            if self.summary:
                context_parts.append(f"Previous conversation summary: {self.summary}")
            
            if self.recent_messages:
                recent_text = "\n".join([
                    f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
                    for msg in self.recent_messages
                ])
                context_parts.append(f"Recent conversation:\n{recent_text}")
            
            return "\n\n".join(context_parts)
    
    summary_memory = SummaryMemory(llm, max_exchanges=2)
    
    # Add several exchanges
    exchanges = [
        ("Hi, I'm Charlie and I work as a software engineer", "Nice to meet you, Charlie!"),
        ("I'm working on a machine learning project", "That sounds exciting!"),
        ("It's about customer segmentation", "ML for customer segmentation is very valuable."),
        ("What did I tell you about my job?", "You work as a software engineer."),
        ("What project am I working on?", "You're working on a machine learning project for customer segmentation.")
    ]
    
    for human_msg, ai_msg in exchanges:
        summary_memory.add_exchange(human_msg, ai_msg)
    
    final_context = summary_memory.get_context()
    print(f"Summary Memory Context:\n{final_context}")
    
    print("="*50 + "\n")


# Example 2: Conversation Chain with Memory
def conversation_chain_example():
    """
    Build a sophisticated conversation system with modern memory patterns
    """
    print("=== Example 2: Modern Conversation System ===")
    
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    
    llm = create_azure_chat_openai(temperature=0.7)
    
    # Create advanced prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant with excellent memory. 
        Use the conversation history to provide personalized, context-aware responses.
        Remember details about the user and reference them naturally in conversation."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # Create chain
    chain = prompt | llm
    
    # Session-based memory storage
    memory_store = {}
    
    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in memory_store:
            memory_store[session_id] = InMemoryChatMessageHistory()
        return memory_store[session_id]
    
    # Create runnable with message history
    conversation_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )
    
    # Simulate a conversation
    session_id = "user_sarah"
    conversation_flow = [
        "Hi! I'm Sarah, a data scientist at TechCorp",
        "I'm working on a customer segmentation project using Python",
        "What programming languages are good for machine learning?",
        "Can you remind me what I told you about my job?",
        "What project am I working on?"
    ]
    
    print("💬 Advanced Conversation with Memory:")
    for user_input in conversation_flow:
        response = conversation_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        
        print(f"Human: {user_input}")
        print(f"AI: {response.content[:200]}...")
        print("-" * 50)
    
    # Show memory state
    session_memory = memory_store[session_id]
    print(f"📋 Final Memory State: {len(session_memory.messages)} messages stored")
    
    # Show last few messages
    print("Last few messages:")
    for msg in session_memory.messages[-4:]:
        msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {msg_type}: {msg.content[:100]}...")
    
    print("="*50 + "\n")


# Example 3: Custom Memory Implementation
def custom_memory_example():
    """
    Create a custom memory implementation using modern LangChain patterns
    """
    print("=== Example 3: Custom Memory Implementation ===")
    
    from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
    from langchain_core.chat_history import BaseChatMessageHistory
    from typing import List, Dict, Any
    import json
    
    class PersonalizedChatHistory(BaseChatMessageHistory):
        """Custom chat history that extracts and stores personal information"""
        
        def __init__(self):
            self._messages: List[BaseMessage] = []
            self.personal_info = {}
        
        @property
        def messages(self) -> List[BaseMessage]:
            return self._messages
        
        def add_message(self, message: BaseMessage) -> None:
            """Add a message and extract personal info if it's from human"""
            self._messages.append(message)
            
            if isinstance(message, HumanMessage):
                self._extract_personal_info(message.content)
        
        def _extract_personal_info(self, message: str) -> None:
            """Extract personal information from human messages"""
            message_lower = message.lower()
            
            # Extract name
            if "my name is" in message_lower:
                name_part = message_lower.split("my name is")[1].strip()
                name = name_part.split()[0] if name_part.split() else ""
                if name:
                    self.personal_info["name"] = name.title()
            
            # Extract job/profession
            if "i work as" in message_lower or "i am a" in message_lower:
                if "i work as" in message_lower:
                    job = message_lower.split("i work as")[1].strip()
                else:
                    job = message_lower.split("i am a")[1].strip()
                
                # Clean up the job description (take first sentence)
                job = job.split('.')[0].split(',')[0]
                self.personal_info["job"] = job
            
            # Extract interests/hobbies
            if "i like" in message_lower or "i enjoy" in message_lower:
                if "i like" in message_lower:
                    interest = message_lower.split("i like")[1].strip()
                else:
                    interest = message_lower.split("i enjoy")[1].strip()
                
                interest = interest.split('.')[0].split(',')[0]
                if "interests" not in self.personal_info:
                    self.personal_info["interests"] = []
                self.personal_info["interests"].append(interest)
        
        def clear(self) -> None:
            """Clear all messages and personal info"""
            self._messages = []
            self.personal_info = {}
        
        def get_context_summary(self) -> str:
            """Get a summary of the conversation with personal info"""
            summary_parts = []
            
            if self.personal_info:
                summary_parts.append("Personal Information:")
                for key, value in self.personal_info.items():
                    if isinstance(value, list):
                        summary_parts.append(f"- {key.title()}: {', '.join(value)}")
                    else:
                        summary_parts.append(f"- {key.title()}: {value}")
            
            if len(self._messages) > 4:  # Show recent conversation
                summary_parts.append("\nRecent conversation:")
                recent_messages = self._messages[-4:]
                for msg in recent_messages:
                    msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
                    summary_parts.append(f"{msg_type}: {msg.content}")
            
            return "\n".join(summary_parts)
    
    # Test the custom memory
    custom_history = PersonalizedChatHistory()
    
    # Simulate a conversation
    test_conversations = [
        ("Hi, my name is David", "Nice to meet you, David!"),
        ("I work as a marketing manager at a tech company", "Marketing is an interesting field!"),
        ("I like hiking and photography in my free time", "Hiking and photography are great hobbies!"),
        ("I also enjoy cooking Italian food", "Italian cuisine is delicious!"),
        ("What do you remember about me?", "Let me recall what you've shared...")
    ]
    
    print("🧠 Custom Memory with Personal Info Extraction:")
    
    for human_msg, ai_response in test_conversations:
        # Add messages to history
        custom_history.add_message(HumanMessage(content=human_msg))
        custom_history.add_message(AIMessage(content=ai_response))
        
        print(f"Human: {human_msg}")
        print(f"AI: {ai_response}")
        print()
    
    # Show extracted information
    print("📋 Extracted Personal Information:")
    print(json.dumps(custom_history.personal_info, indent=2))
    
    print("\n📝 Context Summary:")
    print(custom_history.get_context_summary())
    
    print("\n💡 Benefits of Custom Memory:")
    print("- Automatically extracts structured data from conversations")
    print("- Maintains both raw conversation and processed information") 
    print("- Can be easily extended for domain-specific information")
    print("- Enables personalized responses based on user profile")
    
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
    from langchain.memory.chat_message_histories import ChatMessageHistory
    
    class PersistentMemory:
        """Memory that can be saved to and loaded from disk"""
        
        def __init__(self, file_path="memory.json"):
            self.file_path = file_path
            self.memory = ConversationBufferMemory()
            self.load_memory()
        
        def save_memory(self):
            """Save memory to disk"""
            messages = self.memory.chat_memory.messages
            memory_data = {
                "buffer": self.memory.buffer,
                "messages": [
                    {
                        "type": type(msg).__name__,
                        "content": msg.content
                    }
                    for msg in messages
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
                    
                    # Recreate chat history
                    chat_history = ChatMessageHistory()
                    for msg_data in memory_data.get("messages", []):
                        if msg_data["type"] == "HumanMessage":
                            chat_history.add_user_message(msg_data["content"])
                        elif msg_data["type"] == "AIMessage":
                            chat_history.add_ai_message(msg_data["content"])
                    
                    self.memory.chat_memory = chat_history
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
    
    print(f"📋 Current memory buffer: {persistent.memory.buffer[:200]}...")
    
    # Clean up test file
    if os.path.exists("test_memory.json"):
        os.remove("test_memory.json")
        print("🧹 Cleaned up test file")
    
    print("="*50 + "\n")


# Example 5: Memory Optimization
def memory_optimization_example():
    """
    Strategies for managing memory in long conversations using modern patterns
    """
    print("=== Example 5: Modern Memory Optimization ===")
    
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.chat_history import InMemoryChatMessageHistory
    import tiktoken
    
    llm = create_azure_chat_openai(temperature=0.7)
    
    # Token counting function
    def count_tokens(text):
        """Count tokens in text"""
        try:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            return len(encoding.encode(text))
        except Exception:
            # Fallback: rough estimate (words * 1.3)
            return int(len(text.split()) * 1.3)
    
    # Smart Memory Manager
    class SmartMemoryManager:
        def __init__(self, llm, max_tokens=500):
            self.llm = llm
            self.max_tokens = max_tokens
            self.messages = []
            self.summary = ""
        
        def add_exchange(self, human_msg: str, ai_msg: str):
            """Add a conversation exchange"""
            self.messages.extend([
                HumanMessage(content=human_msg),
                AIMessage(content=ai_msg)
            ])
            
            # Check if we need to optimize memory
            self._optimize_memory()
        
        def _optimize_memory(self):
            """Optimize memory usage when it gets too large"""
            # Convert messages to text for token counting
            full_conversation = self._messages_to_text(self.messages)
            current_tokens = count_tokens(full_conversation)
            
            if current_tokens > self.max_tokens:
                # Keep the most recent exchanges and summarize the rest
                messages_to_keep = 6  # Last 3 exchanges
                messages_to_summarize = self.messages[:-messages_to_keep]
                
                if messages_to_summarize:
                    # Create summary of older messages
                    old_conversation = self._messages_to_text(messages_to_summarize)
                    
                    summary_prompt = f"""
                    Please provide a concise summary of this conversation, focusing on:
                    - Key personal information about the user
                    - Important topics discussed
                    - Any decisions or conclusions reached
                    
                    Conversation:
                    {old_conversation}
                    
                    Summary:"""
                    
                    try:
                        summary_response = self.llm.invoke(summary_prompt)
                        self.summary = summary_response.content
                    except Exception as e:
                        print(f"Error creating summary: {e}")
                        # Fallback: simple truncation
                        self.summary = f"Previous conversation covered: {old_conversation[:200]}..."
                    
                    # Keep only recent messages
                    self.messages = self.messages[-messages_to_keep:]
        
        def _messages_to_text(self, messages):
            """Convert messages to text format"""
            return "\n".join([
                f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
                for msg in messages
            ])
        
        def get_full_context(self):
            """Get the complete context (summary + recent messages)"""
            context_parts = []
            
            if self.summary:
                context_parts.append(f"Previous conversation summary:\n{self.summary}")
            
            if self.messages:
                recent_text = self._messages_to_text(self.messages)
                context_parts.append(f"Recent conversation:\n{recent_text}")
            
            return "\n\n".join(context_parts)
        
        def get_stats(self):
            """Get memory usage statistics"""
            full_context = self.get_full_context()
            return {
                "total_messages": len(self.messages),
                "has_summary": bool(self.summary),
                "total_tokens": count_tokens(full_context),
                "summary_tokens": count_tokens(self.summary) if self.summary else 0,
                "recent_tokens": count_tokens(self._messages_to_text(self.messages))
            }
    
    # Test the smart memory manager
    memory_manager = SmartMemoryManager(llm, max_tokens=300)  # Small limit for demo
    
    # Simulate a long conversation
    long_conversation = [
        ("Hello, I'm interested in learning about machine learning", "Great! Machine learning is fascinating"),
        ("What are the main types of ML algorithms?", "There are supervised, unsupervised, and reinforcement learning"),
        ("Can you explain supervised learning in detail?", "Supervised learning uses labeled data to train models for prediction"),
        ("What about unsupervised learning?", "Unsupervised learning finds patterns in unlabeled data without target variables"),
        ("How does reinforcement learning work?", "It learns through trial and error using rewards and punishments"),
        ("What programming languages are best for ML?", "Python and R are the most popular, with Python leading due to its libraries"),
        ("Can you recommend some ML libraries?", "Scikit-learn for beginners, TensorFlow and PyTorch for deep learning"),
        ("What's the difference between AI and ML?", "AI is broader - ML is a subset of AI focused on learning from data"),
        ("How do I start learning ML practically?", "Start with Python basics, then move to pandas, numpy, and scikit-learn"),
        ("What kind of projects should I work on?", "Begin with classification problems like iris dataset or titanic survival")
    ]
    
    print("💬 Long Conversation with Smart Memory Management:")
    
    for i, (human_msg, ai_msg) in enumerate(long_conversation):
        memory_manager.add_exchange(human_msg, ai_msg)
        stats = memory_manager.get_stats()
        
        print(f"Turn {i+1}: {stats['total_tokens']} tokens, {stats['total_messages']} recent messages")
        
        if stats['has_summary']:
            print(f"  📝 Summary created ({stats['summary_tokens']} tokens)")
        
        if i % 3 == 2:  # Every 3 turns, show memory state
            print(f"  Current context preview: {memory_manager.get_full_context()[:150]}...")
            print("-" * 50)
    
    print(f"\n📊 Final Memory Statistics:")
    final_stats = memory_manager.get_stats()
    for key, value in final_stats.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n📝 Final Context:")
    print(memory_manager.get_full_context())
    
    print(f"\n💡 Modern Memory Optimization Tips:")
    print("1. Use message-based history instead of deprecated memory classes")
    print("2. Implement smart summarization when context gets too long")
    print("3. Monitor token usage with proper counting tools")
    print("4. Keep recent messages and summarize older ones")
    print("5. Use structured data extraction for important information")
    print("6. Consider vector-based storage for very long conversations")
    
    print("="*50 + "\n")


if __name__ == "__main__":
    print("Memory and Conversation Management Examples with Azure OpenAI")
    print("=" * 70)
    
    try:
        # Test Azure OpenAI connection first
        print("🔍 Testing Azure OpenAI connection...")
        test_llm = create_azure_chat_openai()
        test_response = test_llm.invoke("Hello!")
        print(f"✅ Connection successful: {test_response.content}")
        print()
        
        # Run examples
        memory_types_example()
        conversation_chain_example()
        custom_memory_example()
        persistent_memory_example()
        memory_optimization_example()
        
        print("🎉 All memory examples completed!")
        print("Next: Check out 05_Chains/ for complex workflows")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Configured Azure OpenAI settings in .env file")
        print("2. Installed required packages: pip install tiktoken")
        print("3. Valid Azure OpenAI deployments")
