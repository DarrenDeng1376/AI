"""
Modern Custom Chains with Azure OpenAI and LCEL

This module demonstrates how to build custom LangChain chains using modern 
LangChain Expression Language (LCEL) patterns with Azure OpenAI integration.

Key concepts covered:
1. LCEL composition patterns
2. Custom business logic chains
3. Error handling and retry mechanisms
4. Streaming support
5. Azure OpenAI integration
6. Multi-step workflows
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from azure_config import create_azure_chat_openai, create_azure_openai_embeddings
from dotenv import load_dotenv

load_dotenv()

from typing import Dict, List, Any, Optional, AsyncGenerator
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableBranch
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
import asyncio
import json
import time
from datetime import datetime


# ============================================================================
# 1. CUSTOM BUSINESS WORKFLOW CHAINS
# ============================================================================

class CustomerQuery(BaseModel):
    """Structured customer query for business processing"""
    query: str = Field(description="The customer's question or request")
    priority: str = Field(description="Priority level: high, medium, low")
    category: str = Field(description="Query category: technical, billing, sales, support")
    customer_tier: str = Field(description="Customer tier: enterprise, pro, standard")

class CustomerResponse(BaseModel):
    """Structured response for customer queries"""
    response: str = Field(description="The main response to the customer")
    follow_up_actions: List[str] = Field(description="Recommended follow-up actions")
    escalation_needed: bool = Field(description="Whether escalation is required")
    estimated_resolution_time: str = Field(description="Estimated time to resolve")

def create_customer_service_chain():
    """
    Creates a modern customer service chain using LCEL patterns
    """
    print("🔧 Creating Customer Service Chain with Azure OpenAI...")
    
    # Initialize Azure OpenAI components
    llm = create_azure_chat_openai(temperature=0.3)
    
    # Define the analysis prompt
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert customer service analyzer. Analyze the customer query and extract:
        1. Priority level (high, medium, low)
        2. Category (technical, billing, sales, support)  
        3. Customer tier inference from language/complexity (enterprise, pro, standard)
        
        Customer Query: {query}
        
        Respond in JSON format with the extracted information."""),
        ("human", "Analyze this customer query: {query}")
    ])
    
    # Define the response generation prompt
    response_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional customer service representative for AzureTech Solutions.
        
        Customer Analysis:
        - Priority: {priority}
        - Category: {category}
        - Tier: {customer_tier}
        
        Generate a helpful, professional response that:
        1. Addresses their specific concern
        2. Provides actionable solutions
        3. Suggests appropriate follow-up actions
        4. Indicates if escalation is needed
        5. Estimates resolution time
        
        Be empathetic, clear, and solution-focused."""),
        ("human", "Customer Query: {query}")
    ])
    
    # Parser for structured output
    response_parser = PydanticOutputParser(pydantic_object=CustomerResponse)
    
    # Helper function to parse analysis
    def parse_analysis(analysis_result: str) -> Dict[str, str]:
        """Parse the LLM analysis result"""
        try:
            parsed = json.loads(analysis_result)
            return {
                "priority": parsed.get("priority", "medium"),
                "category": parsed.get("category", "support"),
                "customer_tier": parsed.get("customer_tier", "standard")
            }
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "priority": "medium",
                "category": "support", 
                "customer_tier": "standard"
            }
    
    # Build the LCEL chain
    chain = (
        # Step 1: Analyze the query
        {"query": RunnablePassthrough()}
        | RunnableLambda(lambda x: {"query": x["query"]})
        | analysis_prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(parse_analysis)
        # Step 2: Combine analysis with original query
        | RunnableLambda(lambda analysis: {
            "query": analysis.get("original_query", ""),
            **analysis
        })
        # Step 3: Generate response based on analysis
        | response_prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

async def test_customer_service_chain():
    """Test the customer service chain with various scenarios"""
    print("\n=== Testing Customer Service Chain ===")
    
    chain = create_customer_service_chain()
    
    test_queries = [
        "Hi, I'm having trouble connecting to the Azure OpenAI API from my enterprise application. This is blocking our production deployment.",
        "Can you tell me about your pricing plans? I'm interested in the professional tier.",
        "My bill seems wrong this month. I was charged twice for the same service.",
        "How do I integrate your ChatBot Pro with Microsoft Teams?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📋 Test Query {i}:")
        print(f"Customer: {query}")
        print("-" * 50)
        
        try:
            # Process with timing
            start_time = time.time()
            response = await chain.ainvoke({"query": query})
            processing_time = time.time() - start_time
            
            print(f"🤖 Response: {response}")
            print(f"⏱️ Processing time: {processing_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Error: {e}")


# ============================================================================
# 2. DOCUMENT ANALYSIS WORKFLOW CHAIN
# ============================================================================

class DocumentAnalysis(BaseModel):
    """Structured document analysis result"""
    summary: str = Field(description="Brief summary of the document")
    key_points: List[str] = Field(description="Main points or findings")
    sentiment: str = Field(description="Overall sentiment: positive, neutral, negative")
    action_items: List[str] = Field(description="Recommended actions")
    confidence_score: float = Field(description="Confidence in analysis (0-1)")

def create_document_analysis_chain():
    """
    Creates a document analysis chain with multiple processing steps
    """
    print("🔧 Creating Document Analysis Chain...")
    
    llm = create_azure_chat_openai(temperature=0.1)
    
    # Step 1: Document preprocessing
    def preprocess_document(doc: str) -> Dict[str, Any]:
        """Preprocess document for analysis"""
        word_count = len(doc.split())
        char_count = len(doc)
        
        return {
            "content": doc,
            "word_count": word_count,
            "char_count": char_count,
            "complexity": "high" if word_count > 1000 else "medium" if word_count > 300 else "low"
        }
    
    # Step 2: Analysis prompt
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert document analyzer. Analyze the provided document and extract:

        Document Metadata:
        - Word count: {word_count}
        - Complexity: {complexity}

        Provide a comprehensive analysis including:
        1. A clear summary (2-3 sentences)
        2. Key points (3-5 main findings)
        3. Overall sentiment
        4. Recommended action items
        5. Your confidence score (0-1)

        Be thorough but concise."""),
        ("human", "Document to analyze:\n\n{content}")
    ])
    
    # Step 3: Output parser
    output_parser = PydanticOutputParser(pydantic_object=DocumentAnalysis)
    
    # Build the chain
    chain = (
        RunnableLambda(preprocess_document)
        | analysis_prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(lambda x: parse_analysis_output(x))
    )
    
    return chain

def parse_analysis_output(output: str) -> Dict[str, Any]:
    """Parse the analysis output into structured format"""
    try:
        # Try to extract structured information from the response
        lines = output.strip().split('\n')
        
        result = {
            "summary": "",
            "key_points": [],
            "sentiment": "neutral",
            "action_items": [],
            "confidence_score": 0.8
        }
        
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if "summary" in line.lower():
                current_section = "summary"
            elif "key points" in line.lower() or "main points" in line.lower():
                current_section = "key_points"
            elif "sentiment" in line.lower():
                current_section = "sentiment"
            elif "action" in line.lower():
                current_section = "action_items"
            elif line.startswith("- ") or line.startswith("• "):
                if current_section == "key_points":
                    result["key_points"].append(line[2:])
                elif current_section == "action_items":
                    result["action_items"].append(line[2:])
            else:
                if current_section == "summary" and not result["summary"]:
                    result["summary"] = line
                elif current_section == "sentiment" and not result["sentiment"]:
                    result["sentiment"] = line.lower()
        
        return result
        
    except Exception:
        # Fallback to simple parsing
        return {
            "summary": output[:200] + "..." if len(output) > 200 else output,
            "key_points": ["Analysis completed"],
            "sentiment": "neutral",
            "action_items": ["Review analysis"],
            "confidence_score": 0.7
        }


# ============================================================================
# 3. CONDITIONAL ROUTING CHAIN
# ============================================================================

def create_smart_routing_chain():
    """
    Creates a chain that routes queries to different specialized handlers
    """
    print("🔧 Creating Smart Routing Chain...")
    
    llm = create_azure_chat_openai(temperature=0.1)
    
    # Classification prompt
    classifier_prompt = ChatPromptTemplate.from_messages([
        ("system", """Classify the user query into one of these categories:
        - technical: Technical questions, troubleshooting, API issues
        - sales: Pricing, features, comparisons, purchasing
        - support: Account issues, billing, general help
        - creative: Content generation, writing assistance
        
        Respond with just the category name."""),
        ("human", "{query}")
    ])
    
    # Specialized prompts for each category
    technical_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior technical expert. Provide detailed, accurate technical guidance.
        Include code examples, best practices, and troubleshooting steps where relevant."""),
        ("human", "{query}")
    ])
    
    sales_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a knowledgeable sales consultant. Focus on business value, 
        ROI, and how our solutions solve customer problems. Be persuasive but honest."""),
        ("human", "{query}")
    ])
    
    support_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful customer support agent. Be empathetic, clear,
        and solution-focused. Provide step-by-step guidance when needed."""),
        ("human", "{query}")
    ])
    
    creative_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a creative writing assistant. Be imaginative, engaging,
        and help with content creation, writing, and creative tasks."""),
        ("human", "{query}")
    ])
    
    # Classification chain
    classifier = classifier_prompt | llm | StrOutputParser()
    
    # Route based on classification
    def route_query(inputs: Dict[str, Any]) -> Any:
        """Route query to appropriate handler"""
        query = inputs["query"]
        category = inputs["category"].strip().lower()
        
        if category == "technical":
            return technical_prompt | llm | StrOutputParser()
        elif category == "sales":
            return sales_prompt | llm | StrOutputParser()
        elif category == "support":
            return support_prompt | llm | StrOutputParser()
        elif category == "creative":
            return creative_prompt | llm | StrOutputParser()
        else:
            # Default to support
            return support_prompt | llm | StrOutputParser()
    
    # Complete routing chain
    chain = (
        {"query": RunnablePassthrough()}
        | RunnableLambda(lambda x: {
            "query": x["query"],
            "category": classifier.invoke({"query": x["query"]})
        })
        | RunnableLambda(lambda x: {
            **x,
            "handler": route_query(x)
        })
        | RunnableLambda(lambda x: x["handler"].invoke({"query": x["query"]}))
    )
    
    return chain, classifier


# ============================================================================
# 4. STREAMING CONVERSATION CHAIN
# ============================================================================

async def create_streaming_conversation_chain():
    """
    Creates a conversation chain with streaming support
    """
    print("🔧 Creating Streaming Conversation Chain...")
    
    llm = create_azure_chat_openai(temperature=0.7, streaming=True)
    
    conversation_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant. Maintain context across the conversation
        and provide thoughtful, engaging responses. Be conversational and personable."""),
        ("human", "{query}")
    ])
    
    # Streaming chain
    chain = conversation_prompt | llm | StrOutputParser()
    
    return chain

async def demonstrate_streaming():
    """Demonstrate streaming conversation"""
    print("\n=== Streaming Conversation Demo ===")
    
    chain = await create_streaming_conversation_chain()
    
    query = "Tell me about the benefits of using Azure OpenAI for enterprise applications"
    
    print(f"🗣️ Query: {query}")
    print("🤖 Streaming Response:")
    print("-" * 40)
    
    try:
        async for chunk in chain.astream({"query": query}):
            print(chunk, end="", flush=True)
        print("\n" + "-" * 40)
        
    except Exception as e:
        print(f"❌ Streaming error: {e}")


# ============================================================================
# 5. ERROR HANDLING AND RETRY PATTERNS
# ============================================================================

class ResilientChain:
    """A chain wrapper that implements retry logic and error handling"""
    
    def __init__(self, chain, max_retries: int = 3, fallback_response: str = None):
        self.chain = chain
        self.max_retries = max_retries
        self.fallback_response = fallback_response or "I apologize, but I'm having trouble processing your request right now. Please try again later."
    
    async def invoke_with_retry(self, inputs: Dict[str, Any]) -> str:
        """Invoke chain with retry logic"""
        
        for attempt in range(self.max_retries):
            try:
                print(f"🔄 Attempt {attempt + 1}/{self.max_retries}")
                result = await self.chain.ainvoke(inputs)
                print(f"✅ Success on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {e}")
                
                if attempt == self.max_retries - 1:
                    print(f"🚨 All attempts failed, using fallback response")
                    return self.fallback_response
                
                # Wait before retrying (exponential backoff)
                wait_time = 2 ** attempt
                print(f"⏳ Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
        
        return self.fallback_response


# ============================================================================
# 6. MAIN DEMONSTRATION
# ============================================================================

async def main():
    """Main demonstration of custom chains"""
    print("🚀 Custom Chains with Azure OpenAI and Modern LCEL")
    print("=" * 60)
    
    try:
        # Test customer service chain
        await test_customer_service_chain()
        
        # Test document analysis
        print("\n" + "=" * 60)
        print("📄 Document Analysis Chain Demo")
        
        sample_doc = """
        Azure OpenAI Service provides enterprise-grade AI capabilities with the security and compliance features that enterprises need. The service offers powerful language models including GPT-4, GPT-3.5, and embeddings models through a simple REST API.

        Key benefits include:
        - Enterprise security with private endpoints and managed identity
        - Global availability across multiple Azure regions
        - Built-in responsible AI features and content filtering
        - Seamless integration with other Azure services
        - Pay-as-you-go pricing with predictable costs

        Organizations are using Azure OpenAI for customer service automation, content generation, code assistance, and data analysis. The service has shown significant ROI improvements in productivity and customer satisfaction.
        """
        
        doc_chain = create_document_analysis_chain()
        doc_result = await doc_chain.ainvoke(sample_doc)
        print(f"📊 Analysis Result: {doc_result}")
        
        # Test routing chain
        print("\n" + "=" * 60)
        print("🔀 Smart Routing Chain Demo")
        
        routing_chain, classifier = create_smart_routing_chain()
        
        test_queries = [
            "How do I authenticate with Azure OpenAI API using managed identity?",
            "What's the pricing difference between GPT-4 and GPT-3.5?",
            "I can't access my account dashboard",
            "Write a creative story about AI assistants"
        ]
        
        for query in test_queries:
            category = await classifier.ainvoke({"query": query})
            response = await routing_chain.ainvoke({"query": query})
            print(f"\n📝 Query: {query}")
            print(f"🏷️ Category: {category}")
            print(f"🎯 Response: {response[:100]}...")
        
        # Test streaming
        print("\n" + "=" * 60)
        await demonstrate_streaming()
        
        # Test resilient chain
        print("\n" + "=" * 60)
        print("🛡️ Resilient Chain Demo")
        
        basic_chain = ChatPromptTemplate.from_messages([
            ("human", "{query}")
        ]) | create_azure_chat_openai() | StrOutputParser()
        
        resilient = ResilientChain(basic_chain, max_retries=2)
        result = await resilient.invoke_with_retry({
            "query": "Explain the benefits of custom LangChain components"
        })
        print(f"🎯 Resilient Result: {result[:100]}...")
        
        print("\n🎉 All custom chain demos completed!")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("\n🔧 Make sure:")
        print("1. Azure OpenAI is configured in .env")
        print("2. You're authenticated with Azure CLI")
        print("3. Required packages are installed")

if __name__ == "__main__":
    asyncio.run(main())
