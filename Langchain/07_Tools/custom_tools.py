"""
Custom Tools - Production-Ready Tool Development

This module demonstrates creating robust custom tools with proper schemas,
error handling, authentication, and production patterns using modern LangChain.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from azure_config import create_azure_chat_openai
from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, validator
from typing import Type, Optional, Dict, Any, List
import json
import time
import hashlib
import re
from datetime import datetime, timedelta
import sqlite3
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example 1: Data Processing Tools
class TextAnalysisInput(BaseModel):
    """Input schema for text analysis tool"""
    text: str = Field(description="The text to analyze")
    analysis_type: str = Field(
        description="Type of analysis: sentiment, keywords, readability, or summary",
        default="summary"
    )
    options: Optional[Dict[str, Any]] = Field(
        default={},
        description="Additional options for analysis"
    )
    
    @validator('analysis_type')
    def validate_analysis_type(cls, v):
        valid_types = ['sentiment', 'keywords', 'readability', 'summary']
        if v not in valid_types:
            raise ValueError(f"analysis_type must be one of {valid_types}")
        return v

class TextAnalysisTool(BaseTool):
    """Advanced text analysis tool with multiple analysis types"""
    name: str = "text_analyzer"
    description: str = "Analyze text for sentiment, keywords, readability, or generate summaries"
    args_schema: Type[BaseModel] = TextAnalysisInput
    
    def _run(self, text: str, analysis_type: str = "summary", options: Dict[str, Any] = {}) -> str:
        """Execute text analysis"""
        try:
            logger.info(f"Analyzing text with type: {analysis_type}")
            
            if analysis_type == "sentiment":
                return self._analyze_sentiment(text, options)
            elif analysis_type == "keywords":
                return self._extract_keywords(text, options)
            elif analysis_type == "readability":
                return self._calculate_readability(text, options)
            elif analysis_type == "summary":
                return self._generate_summary(text, options)
            else:
                return f"Error: Unknown analysis type '{analysis_type}'"
                
        except Exception as e:
            logger.error(f"Text analysis error: {e}")
            return f"Error analyzing text: {e}"
    
    def _analyze_sentiment(self, text: str, options: Dict[str, Any]) -> str:
        """Simple sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'happy', 'joy']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'frustrated', 'disappointed', 'poor']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        positive_ratio = positive_count / total_words if total_words > 0 else 0
        negative_ratio = negative_count / total_words if total_words > 0 else 0
        
        if positive_count > negative_count:
            sentiment = "Positive"
        elif negative_count > positive_count:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        return f"""Sentiment Analysis:
        - Overall Sentiment: {sentiment}
        - Positive words found: {positive_count}
        - Negative words found: {negative_count}
        - Positive ratio: {positive_ratio:.3f}
        - Negative ratio: {negative_ratio:.3f}"""
    
    def _extract_keywords(self, text: str, options: Dict[str, Any]) -> str:
        """Extract keywords from text"""
        # Simple keyword extraction using word frequency
        import re
        from collections import Counter
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        # Extract words (remove punctuation and convert to lowercase)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out stop words
        keywords = [word for word in words if word not in stop_words]
        
        # Count frequency
        word_freq = Counter(keywords)
        
        # Get top keywords
        top_count = options.get('top_k', 10)
        top_keywords = word_freq.most_common(top_count)
        
        result = "Top Keywords:\n"
        for word, count in top_keywords:
            result += f"- {word}: {count} occurrences\n"
        
        return result
    
    def _calculate_readability(self, text: str, options: Dict[str, Any]) -> str:
        """Calculate readability statistics"""
        sentences = len([s for s in text.split('.') if s.strip()])
        words = len(text.split())
        
        # Count syllables (simple approximation)
        syllable_count = 0
        for word in text.split():
            word_clean = re.sub(r'[^a-zA-Z]', '', word.lower())
            syllables = max(1, len(re.findall(r'[aeiouy]+', word_clean)))
            syllable_count += syllables
        
        # Calculate metrics
        avg_sentence_length = words / sentences if sentences > 0 else 0
        avg_syllables_per_word = syllable_count / words if words > 0 else 0
        
        # Simple readability score (Flesch approximation)
        readability_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        if readability_score >= 90:
            level = "Very Easy"
        elif readability_score >= 80:
            level = "Easy"
        elif readability_score >= 70:
            level = "Fairly Easy"
        elif readability_score >= 60:
            level = "Standard"
        elif readability_score >= 50:
            level = "Fairly Difficult"
        elif readability_score >= 30:
            level = "Difficult"
        else:
            level = "Very Difficult"
        
        return f"""Readability Analysis:
        - Total words: {words}
        - Total sentences: {sentences}
        - Total syllables: {syllable_count}
        - Average sentence length: {avg_sentence_length:.1f} words
        - Average syllables per word: {avg_syllables_per_word:.1f}
        - Readability score: {readability_score:.1f}
        - Reading level: {level}"""
    
    def _generate_summary(self, text: str, options: Dict[str, Any]) -> str:
        """Generate a simple extractive summary"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) <= 3:
            return f"Summary: {text}"
        
        # Simple scoring based on sentence length and position
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            words = len(sentence.split())
            position_score = 1.0 if i < 2 or i >= len(sentences) - 2 else 0.5
            length_score = min(words / 20, 1.0)  # Prefer moderate length sentences
            total_score = position_score + length_score
            scored_sentences.append((total_score, sentence))
        
        # Sort by score and take top sentences
        scored_sentences.sort(reverse=True)
        summary_count = min(3, len(sentences) // 3)
        summary_sentences = [sent for _, sent in scored_sentences[:summary_count]]
        
        return f"Summary: {'. '.join(summary_sentences)}."

# Example 2: Cache Tool
class CacheInput(BaseModel):
    """Input schema for cache operations"""
    operation: str = Field(description="Cache operation: get, set, delete, or clear")
    key: str = Field(description="Cache key")
    value: Optional[str] = Field(default=None, description="Value to cache (for set operation)")
    ttl: Optional[int] = Field(default=3600, description="Time to live in seconds")

class CacheTool(BaseTool):
    """Simple in-memory cache tool with TTL support"""
    name: str = "cache_manager"
    description: str = "Manage cached data with get, set, delete, and clear operations"
    args_schema: Type[BaseModel] = CacheInput
    
    def __init__(self):
        super().__init__()
        self._cache = {}
        self._expiry = {}
    
    def _run(self, operation: str, key: str, value: Optional[str] = None, ttl: int = 3600) -> str:
        """Execute cache operation"""
        try:
            current_time = time.time()
            
            # Clean expired entries
            self._cleanup_expired(current_time)
            
            if operation == "get":
                return self._get(key, current_time)
            elif operation == "set":
                return self._set(key, value, ttl, current_time)
            elif operation == "delete":
                return self._delete(key)
            elif operation == "clear":
                return self._clear()
            else:
                return f"Error: Unknown operation '{operation}'. Use: get, set, delete, clear"
                
        except Exception as e:
            logger.error(f"Cache operation error: {e}")
            return f"Error in cache operation: {e}"
    
    def _cleanup_expired(self, current_time: float):
        """Remove expired cache entries"""
        expired_keys = [key for key, expiry in self._expiry.items() if current_time > expiry]
        for key in expired_keys:
            del self._cache[key]
            del self._expiry[key]
    
    def _get(self, key: str, current_time: float) -> str:
        """Get value from cache"""
        if key in self._cache and current_time <= self._expiry[key]:
            return f"Cache hit: {self._cache[key]}"
        else:
            return f"Cache miss: Key '{key}' not found or expired"
    
    def _set(self, key: str, value: Optional[str], ttl: int, current_time: float) -> str:
        """Set value in cache"""
        if value is None:
            return "Error: Value is required for set operation"
        
        self._cache[key] = value
        self._expiry[key] = current_time + ttl
        return f"Cached '{key}' with TTL {ttl} seconds"
    
    def _delete(self, key: str) -> str:
        """Delete key from cache"""
        if key in self._cache:
            del self._cache[key]
            del self._expiry[key]
            return f"Deleted key '{key}' from cache"
        else:
            return f"Key '{key}' not found in cache"
    
    def _clear(self) -> str:
        """Clear all cache entries"""
        count = len(self._cache)
        self._cache.clear()
        self._expiry.clear()
        return f"Cleared {count} entries from cache"

# Example 3: Database Tool
class DatabaseInput(BaseModel):
    """Input schema for database operations"""
    operation: str = Field(description="Database operation: query, insert, update, delete, or schema")
    table: Optional[str] = Field(default=None, description="Table name")
    query: Optional[str] = Field(default=None, description="SQL query (for query operation)")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Data for insert/update operations")
    where_clause: Optional[str] = Field(default=None, description="WHERE clause for update/delete")

class SimpleDBTool(BaseTool):
    """Simple SQLite database tool for demonstration"""
    name: str = "database_manager"
    description: str = "Execute database operations: query, insert, update, delete, schema"
    args_schema: Type[BaseModel] = DatabaseInput
    
    def __init__(self, db_path: str = ":memory:"):
        super().__init__()
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database with sample tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create sample tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    price REAL,
                    category TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert sample data if tables are empty
            cursor.execute("SELECT COUNT(*) FROM users")
            if cursor.fetchone()[0] == 0:
                cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)", ("John Doe", "john@example.com"))
                cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)", ("Jane Smith", "jane@example.com"))
            
            cursor.execute("SELECT COUNT(*) FROM products")
            if cursor.fetchone()[0] == 0:
                cursor.execute("INSERT INTO products (name, price, category) VALUES (?, ?, ?)", ("Laptop", 999.99, "Electronics"))
                cursor.execute("INSERT INTO products (name, price, category) VALUES (?, ?, ?)", ("Book", 19.99, "Education"))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def _run(self, operation: str, table: Optional[str] = None, query: Optional[str] = None, 
             data: Optional[Dict[str, Any]] = None, where_clause: Optional[str] = None) -> str:
        """Execute database operation"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if operation == "query":
                return self._execute_query(cursor, query, table)
            elif operation == "insert":
                return self._insert_data(cursor, conn, table, data)
            elif operation == "update":
                return self._update_data(cursor, conn, table, data, where_clause)
            elif operation == "delete":
                return self._delete_data(cursor, conn, table, where_clause)
            elif operation == "schema":
                return self._get_schema(cursor, table)
            else:
                return f"Error: Unknown operation '{operation}'. Use: query, insert, update, delete, schema"
                
        except Exception as e:
            logger.error(f"Database operation error: {e}")
            return f"Error in database operation: {e}"
        finally:
            if 'conn' in locals():
                conn.close()
    
    def _execute_query(self, cursor, query: Optional[str], table: Optional[str]) -> str:
        """Execute SELECT query"""
        if query:
            cursor.execute(query)
        elif table:
            cursor.execute(f"SELECT * FROM {table} LIMIT 10")
        else:
            return "Error: Either query or table must be specified for query operation"
        
        results = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        
        if not results:
            return "No results found"
        
        # Format results as table
        result_str = "Query Results:\n"
        result_str += " | ".join(columns) + "\n"
        result_str += "-" * (len(columns) * 10) + "\n"
        
        for row in results:
            result_str += " | ".join(str(cell) for cell in row) + "\n"
        
        return result_str
    
    def _insert_data(self, cursor, conn, table: Optional[str], data: Optional[Dict[str, Any]]) -> str:
        """Insert data into table"""
        if not table or not data:
            return "Error: Table and data are required for insert operation"
        
        columns = list(data.keys())
        values = list(data.values())
        placeholders = ", ".join(["?" for _ in values])
        
        query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
        cursor.execute(query, values)
        conn.commit()
        
        return f"Inserted 1 row into {table} (ID: {cursor.lastrowid})"
    
    def _update_data(self, cursor, conn, table: Optional[str], data: Optional[Dict[str, Any]], where_clause: Optional[str]) -> str:
        """Update data in table"""
        if not table or not data:
            return "Error: Table and data are required for update operation"
        
        set_clause = ", ".join([f"{key} = ?" for key in data.keys()])
        query = f"UPDATE {table} SET {set_clause}"
        values = list(data.values())
        
        if where_clause:
            query += f" WHERE {where_clause}"
        
        cursor.execute(query, values)
        conn.commit()
        
        return f"Updated {cursor.rowcount} rows in {table}"
    
    def _delete_data(self, cursor, conn, table: Optional[str], where_clause: Optional[str]) -> str:
        """Delete data from table"""
        if not table:
            return "Error: Table is required for delete operation"
        
        query = f"DELETE FROM {table}"
        if where_clause:
            query += f" WHERE {where_clause}"
        else:
            return "Error: WHERE clause is required for delete operation (safety measure)"
        
        cursor.execute(query)
        conn.commit()
        
        return f"Deleted {cursor.rowcount} rows from {table}"
    
    def _get_schema(self, cursor, table: Optional[str]) -> str:
        """Get database schema information"""
        if table:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            
            if not columns:
                return f"Table '{table}' not found"
            
            result = f"Schema for table '{table}':\n"
            for col in columns:
                result += f"- {col[1]} ({col[2]}) {'PRIMARY KEY' if col[5] else ''} {'NOT NULL' if col[3] else ''}\n"
            
            return result
        else:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            return "Available tables:\n" + "\n".join([f"- {table[0]}" for table in tables])

# Demo function
def custom_tools_demo():
    """Demonstrate custom tools in action"""
    print("=== Custom Tools Demo ===")
    
    # Create tools
    text_analyzer = TextAnalysisTool()
    cache_manager = CacheTool()
    db_manager = SimpleDBTool()
    
    # Create agent
    llm = create_azure_chat_openai(temperature=0.1)
    
    custom_tools = [text_analyzer, cache_manager, db_manager]
    
    custom_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant with access to powerful custom tools for data processing.
        
        Available tools:
        - text_analyzer: Analyze text for sentiment, keywords, readability, or summaries
        - cache_manager: Manage cached data with get/set/delete/clear operations
        - database_manager: Execute database operations (query, insert, update, delete, schema)
        
        Use these tools to help users with data analysis, text processing, and information management.
        Always explain what you're doing and provide clear results."""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    custom_agent = create_tool_calling_agent(llm, custom_tools, custom_prompt)
    custom_executor = AgentExecutor(agent=custom_agent, tools=custom_tools, verbose=True)
    
    # Test queries
    test_queries = [
        "Analyze this text for sentiment: 'I love this new product! It's amazing and works perfectly. Highly recommended!'",
        "Cache the value 'user_session_123' with key 'current_user' for 1 hour",
        "Show me the database schema and then query all users",
        "Extract keywords from this text: 'Machine learning and artificial intelligence are transforming the technology industry with innovative solutions'"
    ]
    
    print("🛠️ Custom Tools Results:")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📋 Query {i}: {query}")
        print("-" * 60)
        
        try:
            result = custom_executor.invoke({"input": query})
            print(f"🎯 Result: {result['output']}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    print("Custom Tools - Production-Ready Tool Development")
    print("=" * 70)
    
    try:
        # Test Azure OpenAI connection
        print("🔍 Testing Azure OpenAI connection...")
        test_llm = create_azure_chat_openai()
        test_response = test_llm.invoke("Hello!")
        print(f"✅ Connection successful: {test_response.content[:50]}...")
        print()
        
        # Run custom tools demo
        custom_tools_demo()
        
        print("🎉 Custom tools examples completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Configured Azure OpenAI settings in .env file")
        print("2. Installed required dependencies")
        print("3. Proper file permissions for database operations")
