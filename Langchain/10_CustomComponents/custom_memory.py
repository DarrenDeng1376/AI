"""
Custom Memory Implementations with Azure Integration

This module demonstrates how to build custom memory systems for LangChain
applications, including Azure-backed persistence and advanced memory patterns.

Key concepts covered:
1. Custom memory base classes
2. Azure storage integration
3. Conversation management
4. Memory compression and summarization
5. Multi-user memory isolation
6. Context-aware memory retrieval
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from azure_config import create_azure_chat_openai, create_azure_openai_embeddings
from dotenv import load_dotenv

load_dotenv()

from typing import Dict, List, Any, Optional, Union
from langchain.memory.chat_memory import BaseChatMemory
from langchain_core.memory import BaseMemory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json
import sqlite3
import asyncio
from datetime import datetime, timedelta
import hashlib
from collections import defaultdict


# ============================================================================
# 1. AZURE-BACKED PERSISTENT MEMORY
# ============================================================================

class AzurePersistentMemory(BaseChatMemory):
    """
    Custom memory that persists conversations to Azure storage
    """
    
    session_id: str = Field(description="Session identifier for the conversation")
    user_id: str = Field(default="default", description="User identifier")
    max_token_limit: int = Field(default=2000, description="Maximum tokens before compression")
    storage_path: str = Field(default="./memory_storage.db", description="SQLite database path")
    
    class Config:
        arbitrary_types_allowed = True
        # Allow extra attributes for database connection
        extra = "allow"
    
    def __init__(self, **kwargs):
        # Set default input and output keys if not provided
        if 'input_key' not in kwargs:
            kwargs['input_key'] = 'input'
        if 'output_key' not in kwargs:
            kwargs['output_key'] = 'output'
        
        super().__init__(**kwargs)
        # Initialize connection as a private attribute
        object.__setattr__(self, '_conn', None)
        self._init_storage()
        self._load_messages()
    
    @property
    def conn(self):
        """Get database connection"""
        return self._conn
    
    @conn.setter
    def conn(self, value):
        """Set database connection"""
        object.__setattr__(self, '_conn', value)
    
    def _init_storage(self):
        """Initialize SQLite storage for conversations"""
        self.conn = sqlite3.connect(self.storage_path)
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                message_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                timestamp TEXT NOT NULL,
                token_count INTEGER DEFAULT 0
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_session 
            ON conversations(user_id, session_id)
        """)
        
        self.conn.commit()
    
    def _load_messages(self):
        """Load existing messages from storage"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT message_type, content, metadata, timestamp 
            FROM conversations 
            WHERE user_id = ? AND session_id = ? 
            ORDER BY timestamp ASC
        """, (self.user_id, self.session_id))
        
        messages = []
        for row in cursor.fetchall():
            message_type, content, metadata_str, timestamp = row
            metadata = json.loads(metadata_str) if metadata_str else {}
            
            if message_type == "human":
                messages.append(HumanMessage(content=content, additional_kwargs=metadata))
            elif message_type == "ai":
                messages.append(AIMessage(content=content, additional_kwargs=metadata))
            elif message_type == "system":
                messages.append(SystemMessage(content=content, additional_kwargs=metadata))
        
        self.chat_memory.messages = messages
    
    def _save_message(self, message: BaseMessage):
        """Save a message to storage"""
        cursor = self.conn.cursor()
        
        message_type = "human" if isinstance(message, HumanMessage) else \
                      "ai" if isinstance(message, AIMessage) else "system"
        
        metadata_str = json.dumps(message.additional_kwargs) if message.additional_kwargs else None
        token_count = len(message.content.split())  # Simple token estimation
        
        cursor.execute("""
            INSERT INTO conversations 
            (user_id, session_id, message_type, content, metadata, timestamp, token_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            self.user_id, 
            self.session_id, 
            message_type, 
            message.content,
            metadata_str,
            datetime.now().isoformat(),
            token_count
        ))
        
        self.conn.commit()
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save the context from this conversation to buffer."""
        # Save human message
        if self.input_key in inputs:
            human_message = HumanMessage(content=inputs[self.input_key])
            self.chat_memory.add_message(human_message)
            self._save_message(human_message)
        
        # Save AI message
        if self.output_key in outputs:
            ai_message = AIMessage(content=outputs[self.output_key])
            self.chat_memory.add_message(ai_message)
            self._save_message(ai_message)
        
        # Check token limit and compress if needed
        self._manage_memory_size()
    
    def _manage_memory_size(self):
        """Manage memory size by compressing old conversations"""
        total_tokens = sum(len(msg.content.split()) for msg in self.chat_memory.messages)
        
        if total_tokens > self.max_token_limit:
            print(f"🗜️ Memory size ({total_tokens} tokens) exceeds limit, compressing...")
            self._compress_old_messages()
    
    def _compress_old_messages(self):
        """Compress old messages using summarization"""
        if len(self.chat_memory.messages) <= 4:  # Keep at least 4 messages
            return
        
        # Get messages to compress (all but last 4)
        messages_to_compress = self.chat_memory.messages[:-4]
        recent_messages = self.chat_memory.messages[-4:]
        
        # Create summary of old messages
        try:
            llm = create_azure_chat_openai(temperature=0.1)
            
            # Format messages for summarization
            conversation_text = "\n".join([
                f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
                for msg in messages_to_compress
            ])
            
            summary_prompt = f"""
            Summarize the following conversation, preserving key context and important details:
            
            {conversation_text}
            
            Summary:
            """
            
            summary = llm.invoke(summary_prompt).content
            
            # Replace old messages with summary
            summary_message = SystemMessage(
                content=f"[Previous conversation summary: {summary}]",
                additional_kwargs={"type": "summary", "original_count": len(messages_to_compress)}
            )
            
            self.chat_memory.messages = [summary_message] + recent_messages
            
            # Update storage
            self._clear_old_messages()
            self._save_message(summary_message)
            
            print(f"✅ Compressed {len(messages_to_compress)} messages into summary")
            
        except Exception as e:
            print(f"❌ Compression failed: {e}")
    
    def _clear_old_messages(self):
        """Clear old messages from storage"""
        cursor = self.conn.cursor()
        cursor.execute("""
            DELETE FROM conversations 
            WHERE user_id = ? AND session_id = ?
        """, (self.user_id, self.session_id))
        self.conn.commit()
    
    def clear(self) -> None:
        """Clear all messages from memory and storage"""
        super().clear()
        self._clear_old_messages()
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables for the chain"""
        # Return conversation history formatted for prompts
        if self.return_messages:
            return {"history": self.chat_memory.messages}
        else:
            # Return as string format
            history_str = ""
            for message in self.chat_memory.messages:
                if isinstance(message, HumanMessage):
                    history_str += f"Human: {message.content}\n"
                elif isinstance(message, AIMessage):
                    history_str += f"AI: {message.content}\n"
                elif isinstance(message, SystemMessage):
                    history_str += f"System: {message.content}\n"
            return {"history": history_str}
    
    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables"""
        return ["history"]


# ============================================================================
# 2. CONTEXT-AWARE SEMANTIC MEMORY
# ============================================================================

class SemanticMemory(BaseMemory):
    """
    Memory that uses embeddings to retrieve contextually relevant past conversations
    """
    
    user_id: str = Field(default="default", description="User identifier")
    max_memories: int = Field(default=10, description="Maximum number of memories to store")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity threshold for retrieval")
    
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize embeddings and memory store as private attributes
        object.__setattr__(self, '_embeddings', create_azure_openai_embeddings())
        object.__setattr__(self, '_memories', [])
        self._init_memory_store()
    
    @property
    def embeddings(self):
        """Get embeddings client"""
        return self._embeddings
    
    @property
    def memories(self):
        """Get memories list"""
        return self._memories
    
    @memories.setter
    def memories(self, value):
        """Set memories list"""
        object.__setattr__(self, '_memories', value)
    
    def _init_memory_store(self):
        """Initialize in-memory store for semantic memories"""
        # memories property is already initialized in __init__
        pass
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save conversation context with semantic embedding"""
        try:
            # Combine input and output for context
            context = f"Human: {inputs.get('input', '')}\nAI: {outputs.get('output', '')}"
            
            # Generate embedding
            embedding = self.embeddings.embed_query(context)
            
            # Calculate importance score (simple heuristic)
            importance = self._calculate_importance(context)
            
            # Store memory
            memory = {
                "content": context,
                "embedding": embedding,
                "timestamp": datetime.now().isoformat(),
                "importance": importance
            }
            
            self.memories.append(memory)
            
            # Maintain memory limit
            if len(self.memories) > self.max_memories:
                # Remove least important memory
                self.memories.sort(key=lambda x: x["importance"])
                self.memories = self.memories[1:]
            
        except Exception as e:
            print(f"❌ Error saving semantic memory: {e}")
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load relevant memories based on current input"""
        try:
            query = inputs.get("input", "")
            if not query or not self.memories:
                return {"relevant_memories": ""}
            
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Calculate similarities
            similarities = []
            for memory in self.memories:
                similarity = self._cosine_similarity(query_embedding, memory["embedding"])
                if similarity >= self.similarity_threshold:
                    similarities.append((similarity, memory))
            
            # Sort by similarity and get top memories
            similarities.sort(key=lambda x: x[0], reverse=True)
            relevant_memories = [mem for _, mem in similarities[:3]]
            
            # Format memories for prompt
            if relevant_memories:
                memory_text = "\n".join([
                    f"[Previous context ({mem['timestamp'][:10]}): {mem['content'][:200]}...]"
                    for mem in relevant_memories
                ])
                return {"relevant_memories": f"\nRelevant past conversations:\n{memory_text}\n"}
            else:
                return {"relevant_memories": ""}
                
        except Exception as e:
            print(f"❌ Error loading semantic memory: {e}")
            return {"relevant_memories": ""}
    
    def _calculate_importance(self, content: str) -> float:
        """Calculate importance score for memory"""
        # Simple heuristics for importance
        importance = 0.5  # Base importance
        
        # Longer conversations might be more important
        importance += min(len(content) / 1000, 0.3)
        
        # Certain keywords indicate importance
        important_keywords = ["problem", "error", "urgent", "help", "issue", "bug", "critical"]
        keyword_count = sum(1 for keyword in important_keywords if keyword in content.lower())
        importance += keyword_count * 0.1
        
        # Questions might be more important to remember
        if "?" in content:
            importance += 0.1
        
        return min(importance, 1.0)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            import numpy as np
            vec1, vec2 = np.array(vec1), np.array(vec2)
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        except ImportError:
            # Fallback without numpy
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            return dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0
    
    def clear(self) -> None:
        """Clear all memories"""
        self.memories = []
    
    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables"""
        return ["relevant_memories"]


# ============================================================================
# 3. MULTI-USER CONVERSATION MANAGER
# ============================================================================

class MultiUserConversationManager:
    """
    Manages conversations for multiple users with isolation and sharing capabilities
    """
    
    def __init__(self):
        self.user_memories: Dict[str, AzurePersistentMemory] = {}
        self.shared_context: Dict[str, Any] = {}
        self.llm = create_azure_chat_openai(temperature=0.3)
    
    def get_or_create_memory(self, user_id: str, session_id: str) -> AzurePersistentMemory:
        """Get or create memory for a specific user and session"""
        memory_key = f"{user_id}:{session_id}"
        
        if memory_key not in self.user_memories:
            self.user_memories[memory_key] = AzurePersistentMemory(
                session_id=session_id,
                user_id=user_id,
                max_token_limit=1500
            )
        
        return self.user_memories[memory_key]
    
    async def chat_with_user(
        self, 
        user_id: str, 
        session_id: str, 
        message: str,
        include_shared_context: bool = True
    ) -> str:
        """Handle chat for a specific user"""
        try:
            # Get user's memory
            memory = self.get_or_create_memory(user_id, session_id)
            
            # Build prompt with memory
            # Format shared context properly
            shared_context_str = ""
            if include_shared_context and self.shared_context:
                context_items = []
                for key, value in self.shared_context.items():
                    if isinstance(value, dict):
                        context_items.append(f"- {key}: {value.get('value', value)}")
                    else:
                        context_items.append(f"- {key}: {value}")
                shared_context_str = "Shared team context:\n" + "\n".join(context_items)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a helpful assistant. Remember the conversation context for user {user_id}.
                
                {shared_context_str}
                """),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])
            
            # Create chain with memory
            chain = prompt | self.llm | StrOutputParser()
            
            # Get conversation history
            history = memory.chat_memory.messages
            
            # Generate response
            response = await chain.ainvoke({
                "input": message,
                "history": history
            })
            
            # Save to memory
            memory.save_context({"input": message}, {"output": response})
            
            return response
            
        except Exception as e:
            return f"Error processing message: {str(e)}"
    
    def share_context_with_team(self, context_key: str, context_value: Any):
        """Share context information across all users"""
        self.shared_context[context_key] = {
            "value": context_value,
            "timestamp": datetime.now().isoformat(),
            "type": type(context_value).__name__
        }
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get conversation statistics for a user"""
        user_memories = {k: v for k, v in self.user_memories.items() if k.startswith(f"{user_id}:")}
        
        total_messages = sum(len(memory.chat_memory.messages) for memory in user_memories.values())
        active_sessions = len(user_memories)
        
        return {
            "user_id": user_id,
            "active_sessions": active_sessions,
            "total_messages": total_messages,
            "session_ids": [k.split(":")[1] for k in user_memories.keys()]
        }
    
    def export_conversation(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Export conversation history for a user"""
        memory_key = f"{user_id}:{session_id}"
        
        if memory_key not in self.user_memories:
            return {"error": "Conversation not found"}
        
        memory = self.user_memories[memory_key]
        messages = []
        
        for msg in memory.chat_memory.messages:
            messages.append({
                "type": type(msg).__name__,
                "content": msg.content,
                "metadata": msg.additional_kwargs
            })
        
        return {
            "user_id": user_id,
            "session_id": session_id,
            "message_count": len(messages),
            "messages": messages,
            "exported_at": datetime.now().isoformat()
        }


# ============================================================================
# 4. DEMONSTRATION FUNCTIONS
# ============================================================================

async def demonstrate_persistent_memory():
    """Demonstrate Azure-backed persistent memory"""
    print("💾 Persistent Memory Demo")
    print("=" * 40)
    
    memory = AzurePersistentMemory(
        session_id="demo_session_1",
        user_id="test_user",
        max_token_limit=500  # Low limit to trigger compression
    )
    
    # Simulate conversation
    conversations = [
        ("Hello, I'm working on integrating Azure OpenAI", "Hello! I'd be happy to help you with Azure OpenAI integration. What specific aspect are you working on?"),
        ("I need help with authentication", "For authentication with Azure OpenAI, you can use Azure Active Directory, API keys, or managed identity. Which method would you prefer?"),
        ("Tell me about managed identity", "Managed identity is a great choice for Azure services. It provides automatic credential management without storing secrets in your code."),
        ("How do I implement it in Python?", "You can use the Azure Identity library with DefaultAzureCredential. Here's a basic example: from azure.identity import DefaultAzureCredential"),
        ("Can you show me a complete example?", "Certainly! Here's a complete example of using managed identity with Azure OpenAI in Python..."),
        ("This is very helpful, thank you!", "You're welcome! I'm glad I could help you with the Azure OpenAI integration. Let me know if you have any other questions!")
    ]
    
    for human_msg, ai_msg in conversations:
        memory.save_context({"input": human_msg}, {"output": ai_msg})
        print(f"💬 Human: {human_msg}")
        print(f"🤖 AI: {ai_msg}")
        print(f"📊 Memory size: {len(memory.chat_memory.messages)} messages")
        print()
    
    print("Final memory contents:")
    for i, msg in enumerate(memory.chat_memory.messages):
        msg_type = type(msg).__name__
        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        print(f"  {i+1}. [{msg_type}]: {content}")


async def demonstrate_semantic_memory():
    """Demonstrate semantic memory with contextual retrieval"""
    print("\n🧠 Semantic Memory Demo")
    print("=" * 40)
    
    semantic_memory = SemanticMemory(user_id="semantic_test", max_memories=5)
    
    # Save some contexts
    contexts = [
        ({"input": "How do I deploy Azure OpenAI?"}, {"output": "You can deploy Azure OpenAI through the Azure portal, ARM templates, or Azure CLI. The portal is the easiest for beginners."}),
        ({"input": "What's the pricing for GPT-4?"}, {"output": "GPT-4 pricing in Azure OpenAI is based on tokens: $0.03 per 1K prompt tokens and $0.06 per 1K completion tokens."}),
        ({"input": "How to handle rate limits?"}, {"output": "Azure OpenAI has rate limits per minute and per day. Implement exponential backoff and retry logic to handle 429 errors gracefully."}),
        ({"input": "Can I fine-tune models?"}, {"output": "Yes, Azure OpenAI supports fine-tuning for certain models like GPT-3.5-turbo. You'll need training data in JSONL format."}),
        ({"input": "What about data privacy?"}, {"output": "Azure OpenAI provides enterprise-grade security. Your data is not used to train models and is processed within your Azure region."})
    ]
    
    print("Saving conversation contexts...")
    for inp, out in contexts:
        semantic_memory.save_context(inp, out)
        print(f"💾 Saved: {inp['input'][:50]}...")
    
    # Test retrieval
    test_queries = [
        "Tell me about Azure OpenAI costs",
        "I'm getting 429 errors, what should I do?",
        "How secure is my data?",
        "Completely unrelated question about weather"
    ]
    
    print("\nTesting semantic retrieval:")
    for query in test_queries:
        relevant = semantic_memory.load_memory_variables({"input": query})
        print(f"\n🔍 Query: {query}")
        memories = relevant.get("relevant_memories", "No relevant memories")
        if memories.strip():
            print(f"📚 Relevant memories found: {memories[:200]}...")
        else:
            print("📚 No relevant memories found")


async def demonstrate_multi_user_manager():
    """Demonstrate multi-user conversation management"""
    print("\n👥 Multi-User Conversation Manager Demo")
    print("=" * 50)
    
    manager = MultiUserConversationManager()
    
    # Share some team context
    manager.share_context_with_team("current_project", "Azure OpenAI Integration")
    manager.share_context_with_team("deployment_status", "In Progress")
    
    # Simulate conversations for different users
    users = ["alice", "bob", "charlie"]
    
    for user in users:
        print(f"\n💬 Conversation with {user}:")
        
        # Different session for each user
        session = f"session_{user}_1"
        
        # User-specific messages
        user_messages = {
            "alice": [
                "Hi, I'm working on the frontend integration",
                "I need help with API error handling"
            ],
            "bob": [
                "Hello, I'm setting up the backend services",
                "What's the best way to manage API keys?"
            ],
            "charlie": [
                "Hey, I'm working on deployment automation",
                "Can you help with Azure ARM templates?"
            ]
        }
        
        for message in user_messages[user]:
            response = await manager.chat_with_user(user, session, message)
            print(f"  {user}: {message}")
            print(f"  🤖: {response[:100]}...")
        
        # Get user stats
        stats = manager.get_user_stats(user)
        print(f"  📊 Stats: {stats['total_messages']} messages, {stats['active_sessions']} sessions")
    
    # Export a conversation
    print(f"\n📤 Exporting alice's conversation:")
    export = manager.export_conversation("alice", "session_alice_1")
    print(f"Exported {export.get('message_count', 0)} messages")


# ============================================================================
# 5. MAIN DEMONSTRATION
# ============================================================================

async def main():
    """Main demonstration of custom memory implementations"""
    print("🚀 Custom Memory Implementations with Azure Integration")
    print("=" * 70)
    
    try:
        # Test persistent memory
        await demonstrate_persistent_memory()
        
        # Test semantic memory
        await demonstrate_semantic_memory()
        
        # Test multi-user manager
        await demonstrate_multi_user_manager()
        
        print("\n🎉 All memory demos completed!")
        print("\n💡 Key Features Demonstrated:")
        print("✅ Persistent conversation storage with compression")
        print("✅ Semantic memory retrieval using embeddings")
        print("✅ Multi-user conversation isolation")
        print("✅ Shared context across team members")
        print("✅ Memory size management and optimization")
        print("✅ Conversation export and analytics")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("\n🔧 Make sure:")
        print("1. Azure OpenAI is configured properly")
        print("2. Required packages are installed")
        print("3. Storage directory is writable")

if __name__ == "__main__":
    asyncio.run(main())
