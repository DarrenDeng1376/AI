"""
Modern Agents - Autonomous Decision Making with LangChain

This module demonstrates modern LangChain agent patterns:
1. Tool-calling agents with function definitions
2. ReAct (Reasoning + Acting) agents
3. OpenAI Functions agents
4. Custom tool creation and integration
5. Multi-agent coordination
6. Agent memory and state management
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from azure_config import create_azure_chat_openai
from dotenv import load_dotenv
import asyncio
from typing import Dict, Any, List, Optional, Type
import json
import time
from datetime import datetime
import random

load_dotenv()

# Example 1: Basic Tool-Calling Agent
def basic_tool_agent_example():
    """
    Demonstrate a basic agent with tool-calling capabilities
    """
    print("=== Example 1: Basic Tool-Calling Agent ===")
    
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage
    
    # Create Azure OpenAI LLM
    llm = create_azure_chat_openai(temperature=0.1)
    
    # Define tools
    @tool
    def calculator(expression: str) -> str:
        """Evaluate mathematical expressions safely"""
        try:
            # Basic safety check
            allowed_chars = set('0123456789+-*/().e ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            
            # Evaluate safely
            result = eval(expression)
            return f"The result of {expression} is {result}"
        except Exception as e:
            return f"Error calculating {expression}: {str(e)}"
    
    @tool
    def word_counter(text: str) -> str:
        """Count words, characters, and sentences in text"""
        words = len(text.split())
        chars = len(text)
        sentences = text.count('.') + text.count('!') + text.count('?')
        
        return f"Text analysis:\n- Words: {words}\n- Characters: {chars}\n- Sentences: {sentences}"
    
    @tool
    def current_time() -> str:
        """Get the current date and time"""
        now = datetime.now()
        return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
    
    # Create tools list
    tools = [calculator, word_counter, current_time]
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant with access to tools.
        Use the tools when needed to answer questions accurately.
        Think step by step and explain your reasoning."""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    # Create agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Test queries
    test_queries = [
        "What is 15 * 23 + 7?",
        "Count the words in this sentence: 'LangChain agents are powerful tools for automation.'",
        "What time is it right now?",
        "Calculate 2^10 and tell me what time it is"
    ]
    
    print("🤖 Tool-Calling Agent Results:")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("-" * 50)
        
        try:
            result = agent_executor.invoke({"input": query})
            print(f"✅ Answer: {result['output']}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "="*60 + "\n")

# Example 2: ReAct Agent with Web Search
def react_agent_example():
    """
    Demonstrate ReAct (Reasoning + Acting) agent pattern
    """
    print("=== Example 2: ReAct Agent with Research Tools ===")
    
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain_core.prompts import PromptTemplate
    from langchain_core.tools import tool
    
    llm = create_azure_chat_openai(temperature=0.3)
    
    # Mock web search tool (replace with real search API)
    @tool
    def web_search(query: str) -> str:
        """Search the web for information"""
        # Mock search results for demonstration
        mock_results = {
            "langchain": "LangChain is a framework for developing applications powered by language models. It enables building context-aware and reasoning applications.",
            "azure openai": "Azure OpenAI Service provides REST API access to OpenAI's powerful language models including GPT-4, GPT-3.5-turbo, and Embeddings models.",
            "python": "Python is a high-level, interpreted programming language known for its readability and versatility.",
            "ai": "Artificial Intelligence (AI) is the simulation of human intelligence in machines programmed to think and learn."
        }
        
        # Simple keyword matching for demo
        query_lower = query.lower()
        for key, value in mock_results.items():
            if key in query_lower:
                return f"Search results for '{query}': {value}"
        
        return f"Search results for '{query}': No specific information found in mock database."
    
    @tool
    def document_analyzer(text: str) -> str:
        """Analyze document content for key information"""
        lines = text.split('\n')
        word_count = len(text.split())
        
        # Extract potential key points (simple heuristic)
        key_points = []
        for line in lines:
            if line.strip() and (len(line.split()) > 5):
                key_points.append(line.strip())
        
        analysis = f"Document Analysis:\n"
        analysis += f"- Total words: {word_count}\n"
        analysis += f"- Key points found: {len(key_points)}\n"
        if key_points:
            analysis += f"- First key point: {key_points[0][:100]}..."
        
        return analysis
    
    @tool
    def knowledge_synthesizer(topic: str, sources: str) -> str:
        """Synthesize information from multiple sources about a topic"""
        return f"Knowledge Synthesis for '{topic}':\n" \
               f"Based on available sources, here's a comprehensive overview:\n" \
               f"Sources analyzed: {sources[:200]}...\n" \
               f"Synthesis: This topic involves multiple interconnected concepts that require careful analysis."
    
    tools = [web_search, document_analyzer, knowledge_synthesizer]
    
    # ReAct prompt template
    react_prompt = PromptTemplate.from_template("""
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """)
    
    # Create ReAct agent
    agent = create_react_agent(llm, tools, react_prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True
    )
    
    # Test research queries
    research_queries = [
        "What is LangChain and how does it work?",
        "Research Azure OpenAI and analyze its key features",
        "Find information about Python and create a summary"
    ]
    
    print("🔬 ReAct Agent Research Results:")
    
    for i, query in enumerate(research_queries, 1):
        print(f"\n📚 Research Query {i}: {query}")
        print("-" * 60)
        
        try:
            result = agent_executor.invoke({"input": query})
            print(f"✅ Research Complete:")
            print(f"Final Answer: {result['output']}")
        except Exception as e:
            print(f"❌ Research Failed: {e}")
    
    print("\n" + "="*60 + "\n")

# Example 3: Custom Tools and Structured Data
def custom_tools_example():
    """
    Demonstrate creating custom tools with structured inputs/outputs
    """
    print("=== Example 3: Custom Tools with Structured Data ===")
    
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.tools import BaseTool
    from langchain_core.pydantic_v1 import BaseModel, Field
    from typing import Optional, Type
    
    llm = create_azure_chat_openai(temperature=0.2)
    
    # Define input schemas
    class DataAnalysisInput(BaseModel):
        data: List[float] = Field(description="List of numerical data points")
        analysis_type: str = Field(description="Type of analysis: mean, median, std, or summary")
    
    class TextProcessingInput(BaseModel):
        text: str = Field(description="Text to process")
        operation: str = Field(description="Operation: clean, extract_emails, or format")
        options: Optional[Dict[str, Any]] = Field(default={}, description="Additional options")
    
    class TaskPlannerInput(BaseModel):
        goal: str = Field(description="The main goal or objective")
        constraints: Optional[List[str]] = Field(default=[], description="Any constraints or limitations")
        deadline: Optional[str] = Field(default=None, description="Deadline if any")
    
    # Custom tool implementations
    class DataAnalyzer(BaseTool):
        name = "data_analyzer"
        description = "Analyze numerical data with statistical operations"
        args_schema: Type[BaseModel] = DataAnalysisInput
        
        def _run(self, data: List[float], analysis_type: str) -> str:
            try:
                if not data:
                    return "Error: No data provided"
                
                if analysis_type == "mean":
                    result = sum(data) / len(data)
                    return f"Mean of data: {result:.2f}"
                elif analysis_type == "median":
                    sorted_data = sorted(data)
                    n = len(sorted_data)
                    median = sorted_data[n//2] if n % 2 == 1 else (sorted_data[n//2-1] + sorted_data[n//2]) / 2
                    return f"Median of data: {median:.2f}"
                elif analysis_type == "std":
                    mean = sum(data) / len(data)
                    variance = sum((x - mean) ** 2 for x in data) / len(data)
                    std = variance ** 0.5
                    return f"Standard deviation: {std:.2f}"
                elif analysis_type == "summary":
                    mean = sum(data) / len(data)
                    minimum = min(data)
                    maximum = max(data)
                    return f"Data Summary:\n- Count: {len(data)}\n- Mean: {mean:.2f}\n- Min: {minimum}\n- Max: {maximum}"
                else:
                    return f"Error: Unknown analysis type '{analysis_type}'"
            except Exception as e:
                return f"Error in data analysis: {str(e)}"
    
    class TextProcessor(BaseTool):
        name = "text_processor"
        description = "Process and manipulate text in various ways"
        args_schema: Type[BaseModel] = TextProcessingInput
        
        def _run(self, text: str, operation: str, options: Dict[str, Any] = {}) -> str:
            try:
                if operation == "clean":
                    import re
                    # Remove extra whitespace and special characters
                    cleaned = re.sub(r'\s+', ' ', text.strip())
                    cleaned = re.sub(r'[^\w\s.,!?-]', '', cleaned)
                    return f"Cleaned text: {cleaned}"
                
                elif operation == "extract_emails":
                    import re
                    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                    emails = re.findall(email_pattern, text)
                    return f"Extracted emails: {', '.join(emails) if emails else 'No emails found'}"
                
                elif operation == "format":
                    format_type = options.get("format", "uppercase")
                    if format_type == "uppercase":
                        return f"Formatted text: {text.upper()}"
                    elif format_type == "lowercase":
                        return f"Formatted text: {text.lower()}"
                    elif format_type == "title":
                        return f"Formatted text: {text.title()}"
                    else:
                        return f"Unknown format type: {format_type}"
                
                else:
                    return f"Error: Unknown operation '{operation}'"
            except Exception as e:
                return f"Error in text processing: {str(e)}"
    
    class TaskPlanner(BaseTool):
        name = "task_planner"
        description = "Create structured plans for achieving goals"
        args_schema: Type[BaseModel] = TaskPlannerInput
        
        def _run(self, goal: str, constraints: List[str] = [], deadline: Optional[str] = None) -> str:
            try:
                # Simple task breakdown logic
                steps = []
                
                # Basic step generation based on goal keywords
                if "research" in goal.lower():
                    steps.extend(["Define research scope", "Gather sources", "Analyze information", "Synthesize findings"])
                elif "project" in goal.lower():
                    steps.extend(["Plan project scope", "Allocate resources", "Execute phases", "Review and deliver"])
                elif "learn" in goal.lower():
                    steps.extend(["Identify learning objectives", "Find resources", "Study materials", "Practice and apply"])
                else:
                    steps.extend(["Analyze requirements", "Plan approach", "Execute tasks", "Review results"])
                
                plan = f"Task Plan for: {goal}\n\n"
                plan += "Steps:\n"
                for i, step in enumerate(steps, 1):
                    plan += f"{i}. {step}\n"
                
                if constraints:
                    plan += f"\nConstraints to consider:\n"
                    for constraint in constraints:
                        plan += f"- {constraint}\n"
                
                if deadline:
                    plan += f"\nDeadline: {deadline}\n"
                
                plan += f"\nEstimated effort: {len(steps)} main phases"
                
                return plan
            except Exception as e:
                return f"Error in task planning: {str(e)}"
    
    # Create tools
    tools = [DataAnalyzer(), TextProcessor(), TaskPlanner()]
    
    # Create agent with custom tools
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant with access to specialized tools.
        Use the appropriate tools to help users with data analysis, text processing, and task planning.
        Always explain what you're doing and why."""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Test custom tools
    test_queries = [
        "Analyze this data for me: [10, 15, 20, 25, 30, 12, 18] - give me a summary",
        "Clean this text: 'Hello!!!   World@@@   How   are   you???'",
        "Create a plan to learn Python programming with a 3-month deadline and constraint that I can only study 2 hours per day"
    ]
    
    print("🛠️ Custom Tools Agent Results:")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🎯 Query {i}: {query}")
        print("-" * 60)
        
        try:
            result = agent_executor.invoke({"input": query})
            print(f"✅ Result: {result['output']}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "="*60 + "\n")

# Example 4: Multi-Agent Coordination
def multi_agent_example():
    """
    Demonstrate coordination between multiple specialized agents
    """
    print("=== Example 4: Multi-Agent Coordination ===")
    
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.tools import tool
    
    # Create specialized LLMs for different agents
    researcher_llm = create_azure_chat_openai(temperature=0.1)  # Conservative for research
    writer_llm = create_azure_chat_openai(temperature=0.7)     # Creative for writing
    analyst_llm = create_azure_chat_openai(temperature=0.3)    # Balanced for analysis
    
    # Researcher Agent Tools
    @tool
    def research_topic(topic: str) -> str:
        """Research a topic and gather key information"""
        # Mock research data
        research_db = {
            "artificial intelligence": {
                "definition": "AI is the simulation of human intelligence in machines",
                "applications": ["machine learning", "natural language processing", "computer vision"],
                "trends": ["generative AI", "large language models", "autonomous systems"]
            },
            "climate change": {
                "definition": "Long-term shifts in global temperatures and weather patterns",
                "causes": ["greenhouse gas emissions", "deforestation", "industrial processes"],
                "solutions": ["renewable energy", "carbon capture", "sustainable practices"]
            }
        }
        
        topic_lower = topic.lower()
        for key, data in research_db.items():
            if key in topic_lower:
                return f"Research on {topic}:\n" + \
                       f"Definition: {data['definition']}\n" + \
                       f"Key aspects: {', '.join(data.get('applications', data.get('causes', [])))}\n" + \
                       f"Current trends/solutions: {', '.join(data.get('trends', data.get('solutions', [])))}"
        
        return f"Research on {topic}: General information available. Topic requires further investigation."
    
    # Writer Agent Tools  
    @tool
    def create_content(content_type: str, topic: str, research_data: str) -> str:
        """Create content based on research data"""
        if content_type.lower() == "summary":
            return f"Summary of {topic}:\n\nBased on research, {topic} is a significant area with multiple facets. " + \
                   f"The research shows: {research_data[:200]}... This summary provides a concise overview of the key points."
        elif content_type.lower() == "blog":
            return f"Blog Post: Understanding {topic}\n\n" + \
                   f"In today's world, {topic} plays a crucial role. " + \
                   f"Research indicates: {research_data[:150]}... " + \
                   f"This blog explores the implications and future prospects."
        else:
            return f"Content created for {topic} in {content_type} format based on provided research."
    
    # Analyst Agent Tools
    @tool
    def analyze_content(content: str, analysis_type: str) -> str:
        """Analyze content for various metrics"""
        word_count = len(content.split())
        char_count = len(content)
        
        if analysis_type.lower() == "readability":
            avg_word_length = sum(len(word) for word in content.split()) / word_count if word_count > 0 else 0
            return f"Readability Analysis:\n- Word count: {word_count}\n- Average word length: {avg_word_length:.1f} characters\n- Estimated reading level: {'Easy' if avg_word_length < 5 else 'Moderate' if avg_word_length < 7 else 'Advanced'}"
        elif analysis_type.lower() == "structure":
            sentences = content.count('.') + content.count('!') + content.count('?')
            paragraphs = content.count('\n\n') + 1
            return f"Structure Analysis:\n- Sentences: {sentences}\n- Paragraphs: {paragraphs}\n- Words per sentence: {word_count/sentences:.1f if sentences > 0 else 0}"
        else:
            return f"General analysis of content: {word_count} words, {char_count} characters."
    
    # Create specialized agents
    research_tools = [research_topic]
    writer_tools = [create_content]
    analyst_tools = [analyze_content]
    
    # Researcher Agent
    research_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research specialist. Your job is to gather comprehensive information on topics."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    research_agent = create_tool_calling_agent(researcher_llm, research_tools, research_prompt)
    researcher = AgentExecutor(agent=research_agent, tools=research_tools, verbose=False)
    
    # Writer Agent
    writer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a content writer. Create engaging content based on research data provided."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    writer_agent = create_tool_calling_agent(writer_llm, writer_tools, writer_prompt)
    writer = AgentExecutor(agent=writer_agent, tools=writer_tools, verbose=False)
    
    # Analyst Agent
    analyst_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a content analyst. Analyze content for quality, structure, and readability."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    analyst_agent = create_tool_calling_agent(analyst_llm, analyst_tools, analyst_prompt)
    analyst = AgentExecutor(agent=analyst_agent, tools=analyst_tools, verbose=False)
    
    # Coordination workflow
    def multi_agent_workflow(topic: str, content_type: str = "summary"):
        """Coordinate multiple agents to complete a complex task"""
        print(f"🤝 Multi-Agent Workflow: Creating {content_type} on {topic}")
        
        # Step 1: Research phase
        print("\n🔍 Phase 1: Research")
        research_result = researcher.invoke({"input": f"Research the topic: {topic}"})
        research_data = research_result["output"]
        print(f"Research completed: {research_data[:100]}...")
        
        # Step 2: Content creation phase
        print("\n✍️ Phase 2: Content Creation")
        writer_input = f"Create a {content_type} about {topic} using this research: {research_data}"
        content_result = writer.invoke({"input": writer_input})
        content = content_result["output"]
        print(f"Content created: {content[:100]}...")
        
        # Step 3: Analysis phase
        print("\n📊 Phase 3: Content Analysis")
        analysis_result = analyst.invoke({"input": f"Analyze this content for readability: {content}"})
        analysis = analysis_result["output"]
        print(f"Analysis completed: {analysis}")
        
        return {
            "research": research_data,
            "content": content,
            "analysis": analysis
        }
    
    # Test multi-agent workflows
    test_topics = [
        ("artificial intelligence", "summary"),
        ("climate change", "blog")
    ]
    
    print("🚀 Multi-Agent Coordination Results:")
    
    for topic, content_type in test_topics:
        print(f"\n" + "="*60)
        print(f"📋 Task: Create {content_type} about {topic}")
        print("="*60)
        
        try:
            result = multi_agent_workflow(topic, content_type)
            print(f"\n✅ Workflow completed successfully!")
            print(f"\nFinal deliverable preview:")
            print(f"Content: {result['content'][:200]}...")
            print(f"Quality check: {result['analysis']}")
        except Exception as e:
            print(f"❌ Workflow failed: {e}")
    
    print("\n" + "="*60 + "\n")

# Example 5: Agent with Memory and State Management
def agent_memory_example():
    """
    Demonstrate agents with persistent memory and state management
    """
    print("=== Example 5: Agent with Memory and State Management ===")
    
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.tools import tool
    from langchain.memory import ConversationBufferWindowMemory
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_core.chat_history import InMemoryChatMessageHistory
    
    llm = create_azure_chat_openai(temperature=0.5)
    
    # Simulated user database
    user_profiles = {}
    user_preferences = {}
    conversation_history = {}
    
    @tool
    def save_user_preference(user_id: str, preference_type: str, value: str) -> str:
        """Save a user preference to memory"""
        if user_id not in user_preferences:
            user_preferences[user_id] = {}
        user_preferences[user_id][preference_type] = value
        return f"Saved {preference_type} preference '{value}' for user {user_id}"
    
    @tool
    def get_user_preference(user_id: str, preference_type: str) -> str:
        """Retrieve a user preference from memory"""
        if user_id in user_preferences and preference_type in user_preferences[user_id]:
            value = user_preferences[user_id][preference_type]
            return f"User {user_id}'s {preference_type} preference: {value}"
        return f"No {preference_type} preference found for user {user_id}"
    
    @tool
    def update_user_profile(user_id: str, field: str, value: str) -> str:
        """Update user profile information"""
        if user_id not in user_profiles:
            user_profiles[user_id] = {}
        user_profiles[user_id][field] = value
        return f"Updated {field} to '{value}' for user {user_id}"
    
    @tool
    def get_user_profile(user_id: str) -> str:
        """Get user profile information"""
        if user_id in user_profiles:
            profile = user_profiles[user_id]
            return f"User {user_id} profile: {json.dumps(profile, indent=2)}"
        return f"No profile found for user {user_id}"
    
    @tool
    def personalized_recommendation(user_id: str, category: str) -> str:
        """Generate personalized recommendations based on user profile and preferences"""
        # Get user data
        profile = user_profiles.get(user_id, {})
        preferences = user_preferences.get(user_id, {})
        
        # Simple recommendation logic
        recommendations = []
        
        if category.lower() == "books":
            if preferences.get("genre") == "sci-fi":
                recommendations = ["Dune by Frank Herbert", "Foundation by Isaac Asimov", "Neuromancer by William Gibson"]
            elif preferences.get("genre") == "mystery":
                recommendations = ["The Girl with the Dragon Tattoo", "Gone Girl", "The Big Sleep"]
            else:
                recommendations = ["To Kill a Mockingbird", "1984", "Pride and Prejudice"]
        
        elif category.lower() == "movies":
            if preferences.get("genre") == "action":
                recommendations = ["Mad Max: Fury Road", "John Wick", "Mission Impossible"]
            else:
                recommendations = ["The Shawshank Redemption", "Forrest Gump", "The Godfather"]
        
        else:
            recommendations = [f"General {category} recommendation 1", f"General {category} recommendation 2"]
        
        rec_text = f"Personalized {category} recommendations for {user_id}:\n"
        for i, rec in enumerate(recommendations, 1):
            rec_text += f"{i}. {rec}\n"
        
        if preferences:
            rec_text += f"\nBased on your preferences: {json.dumps(preferences)}"
        
        return rec_text
    
    # Create tools
    tools = [save_user_preference, get_user_preference, update_user_profile, get_user_profile, personalized_recommendation]
    
    # Create memory-aware agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a personalized assistant with memory capabilities.
        Remember user preferences and profile information across conversations.
        Use the available tools to save and retrieve user data.
        Be helpful and personalize your responses based on what you know about the user."""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create chat message history store
    store = {}
    
    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]
    
    # Create agent with memory
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    agent_with_memory = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )
    
    # Simulate conversation with memory
    def simulate_conversation(user_id: str):
        """Simulate a multi-turn conversation with memory"""
        print(f"\n👤 Starting conversation with User: {user_id}")
        print("-" * 50)
        
        conversations = [
            f"Hi, I'm {user_id}. I love science fiction books and action movies.",
            "Can you save that I prefer sci-fi books?",
            "What's my favorite book genre?",
            "Can you recommend some books for me?",
            "Also update my profile to show I'm a software engineer.",
            "What's in my profile now?",
            "Recommend some movies too!"
        ]
        
        session_config = {"configurable": {"session_id": user_id}}
        
        for i, message in enumerate(conversations, 1):
            print(f"\n💬 Turn {i}: {message}")
            try:
                result = agent_with_memory.invoke({"input": message}, config=session_config)
                print(f"🤖 Assistant: {result['output']}")
            except Exception as e:
                print(f"❌ Error: {e}")
            
            print("-" * 30)
    
    # Test memory with different users
    test_users = ["alice", "bob"]
    
    print("🧠 Memory-Enabled Agent Results:")
    
    for user in test_users:
        simulate_conversation(user)
    
    # Show memory persistence
    print(f"\n📚 Memory State Summary:")
    print(f"User Profiles: {json.dumps(user_profiles, indent=2)}")
    print(f"User Preferences: {json.dumps(user_preferences, indent=2)}")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    print("Modern Agents - Autonomous Decision Making with LangChain")
    print("=" * 70)
    
    try:
        # Test Azure OpenAI connection first
        print("🔍 Testing Azure OpenAI connection...")
        test_llm = create_azure_chat_openai()
        test_response = test_llm.invoke("Hello!")
        print(f"✅ Connection successful: {test_response.content[:50]}...")
        print()
        
        # Run all agent examples
        basic_tool_agent_example()
        react_agent_example()
        custom_tools_example()
        multi_agent_example()
        agent_memory_example()
        
        print("🎉 All modern agent examples completed!")
        print("Next: Check out 07_Tools/ for advanced tool integration")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Configured Azure OpenAI settings in .env file")
        print("2. Valid Azure OpenAI deployments")
        print("3. Sufficient quota in your Azure OpenAI resource")
