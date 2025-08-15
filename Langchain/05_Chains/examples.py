"""
Modern Chains - Building Complex Workflows with LCEL

This module demonstrates modern LangChain patterns using LCEL (LangChain Expression Language):
1. LCEL basics and pipe operations
2. Parallel chain execution with RunnableParallel
3. Conditional chains with RunnableBranch
4. Streaming chains for real-time responses
5. Production-ready patterns with validation and error handling
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from azure_config import create_azure_chat_openai
from dotenv import load_dotenv
import asyncio
from typing import Dict, Any, List
import json
import time

load_dotenv()

# Example 1: LCEL Fundamentals
def lcel_basics_example():
    """
    Demonstrate LCEL (LangChain Expression Language) fundamentals
    """
    print("=== Example 1: LCEL Basics ===")
    
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from pydantic import BaseModel, Field
    
    llm = create_azure_chat_openai(temperature=0.7)
    
    # 1. Basic Chain: Prompt | LLM | Output Parser
    print("🔗 1. Basic LCEL Chain:")
    
    prompt = ChatPromptTemplate.from_template(
        "Write a {length} {content_type} about {topic}."
    )
    
    # Simple string output chain
    basic_chain = prompt | llm | StrOutputParser()
    
    result = basic_chain.invoke({
        "length": "short",
        "content_type": "poem",
        "topic": "artificial intelligence"
    })
    
    print(f"Basic chain result: {result[:150]}...")
    print()
    
    # 2. Structured Output with Pydantic
    print("📋 2. Structured Output:")
    
    class StoryStructure(BaseModel):
        title: str = Field(description="The title of the story")
        characters: List[str] = Field(description="Main characters in the story")
        setting: str = Field(description="Where the story takes place")
        plot_summary: str = Field(description="Brief summary of the plot")
    
    structured_prompt = ChatPromptTemplate.from_template("""
    Create a story structure for a {genre} story about {theme}.
    
    Return the response in the following JSON format:
    {{
        "title": "Story title",
        "characters": ["character1", "character2"],
        "setting": "Story setting",
        "plot_summary": "Brief plot summary"
    }}
    """)
    
    json_parser = JsonOutputParser(pydantic_object=StoryStructure)
    structured_chain = structured_prompt | llm | json_parser
    
    try:
        structured_result = structured_chain.invoke({
            "genre": "science fiction",
            "theme": "time travel"
        })
        
        print("Structured output:")
        print(f"Title: {structured_result['title']}")
        print(f"Characters: {', '.join(structured_result['characters'])}")
        print(f"Setting: {structured_result['setting']}")
        print(f"Plot: {structured_result['plot_summary']}")
    except Exception as e:
        print(f"Structured parsing failed (trying fallback): {e}")
        # Fallback to string parsing
        fallback_chain = structured_prompt | llm | StrOutputParser()
        fallback_result = fallback_chain.invoke({
            "genre": "science fiction",
            "theme": "time travel"
        })
        print(f"Fallback result: {fallback_result[:200]}...")
    
    print()
    
    # 3. Chain with Preprocessing
    print("🔄 3. Chain with Preprocessing:")
    
    def preprocess_input(inputs):
        """Preprocess and validate inputs"""
        topic = inputs.get("topic", "").strip().lower()
        if not topic:
            return {"topic": "general knowledge", "processed": True}
        
        # Add context based on topic
        context_map = {
            "ai": "artificial intelligence and machine learning",
            "tech": "technology and innovation", 
            "science": "scientific discoveries and research"
        }
        
        enhanced_topic = context_map.get(topic, topic)
        return {"topic": enhanced_topic, "processed": True}
    
    preprocessing_chain = (
        RunnablePassthrough.assign(**preprocess_input) | 
        ChatPromptTemplate.from_template("Explain {topic} in simple terms.") |
        llm | 
        StrOutputParser()
    )
    
    preprocessed_result = preprocessing_chain.invoke({"topic": "ai"})
    print(f"Preprocessed result: {preprocessed_result[:200]}...")
    print()
    
    print("="*50 + "\n")

# Example 2: Parallel Chain Execution
def parallel_chains_example():
    """
    Demonstrate parallel chain execution with RunnableParallel
    """
    print("=== Example 2: Parallel Chain Execution ===")
    
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough
    
    llm = create_azure_chat_openai(temperature=0.7)
    
    # Define individual analysis chains
    sentiment_prompt = ChatPromptTemplate.from_template(
        "Analyze the sentiment of this text: '{text}'. Respond with: Positive, Negative, or Neutral with a brief explanation."
    )
    
    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize this text in one sentence: '{text}'"
    )
    
    keywords_prompt = ChatPromptTemplate.from_template(
        "Extract 3-5 key topics from this text: '{text}'. Return as comma-separated list."
    )
    
    # Create individual chains
    sentiment_chain = sentiment_prompt | llm | StrOutputParser()
    summary_chain = summary_prompt | llm | StrOutputParser()
    keywords_chain = keywords_prompt | llm | StrOutputParser()
    
    # Combine into parallel execution
    analysis_chain = RunnableParallel(
        sentiment=sentiment_chain,
        summary=summary_chain, 
        keywords=keywords_chain,
        original_text=RunnablePassthrough()
    )
    
    # Test with sample text
    sample_text = """
    The new AI assistant is incredibly helpful and has transformed how our team works. 
    It can analyze documents quickly, generate creative content, and provide intelligent 
    suggestions. However, we need to be careful about data privacy and ensure we're 
    using it responsibly. Overall, it's been a game-changer for productivity.
    """
    
    print("🔄 Parallel Analysis Results:")
    result = analysis_chain.invoke({"text": sample_text.strip()})
    
    print(f"Original Text: {result['original_text']['text'][:100]}...")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Summary: {result['summary']}")
    print(f"Keywords: {result['keywords']}")
    print()
    
    # Advanced parallel processing with different LLM configurations
    print("⚡ Advanced Parallel Processing:")
    
    # Different temperature settings for different tasks
    creative_llm = create_azure_chat_openai(temperature=0.9)
    factual_llm = create_azure_chat_openai(temperature=0.1)
    
    creative_chain = (
        ChatPromptTemplate.from_template("Write a creative tagline for: {product}") |
        creative_llm | 
        StrOutputParser()
    )
    
    factual_chain = (
        ChatPromptTemplate.from_template("List 3 practical features of: {product}") |
        factual_llm | 
        StrOutputParser()
    )
    
    marketing_analysis = RunnableParallel(
        creative_tagline=creative_chain,
        practical_features=factual_chain
    )
    
    marketing_result = marketing_analysis.invoke({"product": "smart home security system"})
    
    print("Marketing Analysis:")
    print(f"Creative Tagline: {marketing_result['creative_tagline']}")
    print(f"Practical Features: {marketing_result['practical_features']}")
    
    print("="*50 + "\n")

# Example 3: Conditional Chains
def conditional_chains_example():
    """
    Demonstrate conditional chain execution with RunnableBranch
    """
    print("=== Example 3: Conditional Chain Execution ===")
    
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableBranch
    
    llm = create_azure_chat_openai(temperature=0.7)
    
    # Define specialized chains for different content types
    technical_prompt = ChatPromptTemplate.from_template("""
    You are a technical expert. Provide a detailed technical explanation of: {query}
    Include technical terms and implementation details.
    """)
    
    beginner_prompt = ChatPromptTemplate.from_template("""
    You are explaining to a complete beginner. Explain {query} in very simple terms
    with analogies and examples anyone can understand.
    """)
    
    business_prompt = ChatPromptTemplate.from_template("""
    You are a business consultant. Explain {query} from a business perspective,
    focusing on ROI, implementation costs, and business impact.
    """)
    
    # Create conditional routing function
    def route_query(inputs):
        """Route based on query content and user type"""
        query = inputs.get("query", "").lower()
        user_type = inputs.get("user_type", "general").lower()
        
        # Route based on user type
        if user_type in ["developer", "engineer", "technical"]:
            return "technical"
        elif user_type in ["beginner", "student", "newcomer"]:
            return "beginner"
        elif user_type in ["business", "manager", "executive"]:
            return "business"
        
        # Route based on query content
        technical_keywords = ["api", "algorithm", "code", "programming", "implementation"]
        business_keywords = ["roi", "cost", "revenue", "market", "strategy"]
        
        if any(keyword in query for keyword in technical_keywords):
            return "technical"
        elif any(keyword in query for keyword in business_keywords):
            return "business"
        else:
            return "beginner"
    
    # Create the conditional chain
    routing_chain = RunnableBranch(
        (lambda x: route_query(x) == "technical", technical_prompt | llm | StrOutputParser()),
        (lambda x: route_query(x) == "business", business_prompt | llm | StrOutputParser()),
        beginner_prompt | llm | StrOutputParser()  # Default case
    )
    
    # Test with different scenarios
    test_cases = [
        {"query": "machine learning", "user_type": "developer"},
        {"query": "artificial intelligence", "user_type": "beginner"},
        {"query": "AI implementation costs", "user_type": "business"},
        {"query": "neural networks", "user_type": "general"}  # Will route based on content
    ]
    
    print("🔀 Conditional Routing Results:")
    
    for i, test_case in enumerate(test_cases, 1):
        route = route_query(test_case)
        result = routing_chain.invoke(test_case)
        
        print(f"Test {i}:")
        print(f"Query: {test_case['query']}")
        print(f"User Type: {test_case['user_type']}")
        print(f"Route: {route}")
        print(f"Response: {result[:150]}...")
        print("-" * 40)
    
    print("="*50 + "\n")

# Example 4: Streaming Chains
def streaming_chains_example():
    """
    Demonstrate streaming chain execution for real-time responses
    """
    print("=== Example 4: Streaming Chains ===")
    
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.callbacks import StreamingStdOutCallbackHandler
    
    llm = create_azure_chat_openai(
        temperature=0.7,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    
    # Create streaming chain
    streaming_prompt = ChatPromptTemplate.from_template(
        "Write a detailed explanation of {topic}. Include examples and practical applications."
    )
    
    streaming_chain = streaming_prompt | llm | StrOutputParser()
    
    print("📡 Streaming Response (real-time):")
    print("Topic: Quantum Computing")
    print("-" * 40)
    
    # Stream the response
    response_chunks = []
    try:
        for chunk in streaming_chain.stream({"topic": "quantum computing"}):
            print(chunk, end="", flush=True)
            response_chunks.append(chunk)
        
        full_response = "".join(response_chunks)
        print(f"\n\n✅ Streaming completed. Total length: {len(full_response)} characters")
        
    except Exception as e:
        print(f"\n❌ Streaming failed: {e}")
        # Fallback to non-streaming
        print("Using fallback non-streaming approach:")
        non_streaming_llm = create_azure_chat_openai(temperature=0.7)
        fallback_chain = streaming_prompt | non_streaming_llm | StrOutputParser()
        result = fallback_chain.invoke({"topic": "quantum computing"})
        print(result[:300] + "...")
    
    print("\n" + "="*50 + "\n")

# Example 5: Production-Ready Patterns
def production_chains_example():
    """
    Demonstrate production-ready chain patterns with validation and error handling
    """
    print("=== Example 5: Production-Ready Patterns ===")
    
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    from pydantic import BaseModel, Field, ValidationError
    from typing import Optional
    import time
    
    class ProductAnalysis(BaseModel):
        sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")
        confidence: float = Field(description="Confidence score between 0 and 1")
        key_points: List[str] = Field(description="Main points mentioned in the review")
        recommendation: str = Field(description="Recommendation based on the analysis")
    
    def validate_input(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize inputs"""
        text = inputs.get("text", "").strip()
        
        if not text:
            raise ValueError("Input text cannot be empty")
        
        if len(text) > 5000:
            text = text[:5000] + "..."
            inputs["text"] = text
            inputs["truncated"] = True
        
        # Add metadata
        inputs["processed_at"] = time.time()
        inputs["word_count"] = len(text.split())
        
        return inputs
    
    def handle_llm_error(error: Exception) -> Dict[str, Any]:
        """Handle LLM errors gracefully"""
        return {
            "error": True,
            "message": f"Analysis failed: {str(error)}",
            "sentiment": "neutral",
            "confidence": 0.0,
            "key_points": ["Error occurred during analysis"],
            "recommendation": "Please try again with different input"
        }
    
    def post_process_output(result: Any) -> Dict[str, Any]:
        """Post-process and validate output"""
        if isinstance(result, dict) and "error" in result:
            return result
        
        try:
            # Ensure confidence is within valid range
            if isinstance(result, dict) and "confidence" in result:
                confidence = float(result["confidence"])
                result["confidence"] = max(0.0, min(1.0, confidence))
            
            return result
        except Exception as e:
            return handle_llm_error(e)
    
    # Create production chain with error handling
    llm = create_azure_chat_openai(temperature=0.3)
    
    analysis_prompt = ChatPromptTemplate.from_template("""
    Analyze the following product review and provide a JSON response:
    
    Review: {text}
    
    Provide analysis in this exact JSON format:
    {{
        "sentiment": "positive/negative/neutral",
        "confidence": 0.85,
        "key_points": ["point1", "point2", "point3"],
        "recommendation": "your recommendation"
    }}
    """)
    
    # Build the production chain
    production_chain = (
        RunnableLambda(validate_input) |
        analysis_prompt |
        llm |
        JsonOutputParser(pydantic_object=ProductAnalysis) |
        RunnableLambda(post_process_output)
    )
    
    # Test with various inputs including edge cases
    test_reviews = [
        "This product is amazing! Great quality and fast shipping.",
        "Terrible product. Broke after one day. Waste of money.",
        "",  # Empty input (should fail gracefully)
        "A" * 6000,  # Very long input (should be truncated)
        "The product is okay. Nothing special but does the job."
    ]
    
    print("🏭 Production Chain Results:")
    
    for i, review in enumerate(test_reviews, 1):
        try:
            print(f"\nTest {i}:")
            print(f"Input: {review[:100]}{'...' if len(review) > 100 else ''}")
            
            result = production_chain.invoke({"text": review})
            
            if result.get("error"):
                print(f"❌ Error: {result['message']}")
            else:
                print(f"✅ Analysis completed:")
                print(f"   Sentiment: {result.get('sentiment', 'N/A')}")
                print(f"   Confidence: {result.get('confidence', 0):.2f}")
                print(f"   Key Points: {result.get('key_points', [])}")
                print(f"   Recommendation: {result.get('recommendation', 'N/A')}")
                
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            error_result = handle_llm_error(e)
            print(f"   Fallback: {error_result['message']}")
    
    print(f"\n💡 Production Best Practices Demonstrated:")
    print("1. ✅ Input validation and sanitization")
    print("2. ✅ Error handling with graceful degradation")
    print("3. ✅ Output validation and post-processing")
    print("4. ✅ Structured output with Pydantic models")
    print("5. ✅ Metadata tracking and logging")
    print("6. ✅ Edge case handling (empty input, long text)")
    
    print("="*50 + "\n")

# Example 6: Async Chain Execution
async def async_chains_example():
    """
    Demonstrate asynchronous chain execution for better performance
    """
    print("=== Example 6: Async Chain Execution ===")
    
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableParallel
    
    llm = create_azure_chat_openai(temperature=0.7)
    
    # Create multiple chains for async processing
    chains = {
        "summary": ChatPromptTemplate.from_template("Summarize: {text}") | llm | StrOutputParser(),
        "sentiment": ChatPromptTemplate.from_template("Analyze sentiment of: {text}") | llm | StrOutputParser(),
        "keywords": ChatPromptTemplate.from_template("Extract keywords from: {text}") | llm | StrOutputParser()
    }
    
    # Test data
    documents = [
        "Artificial intelligence is transforming industries worldwide.",
        "Climate change requires immediate global action and cooperation.",
        "The future of work will be shaped by automation and AI."
    ]
    
    print("⚡ Processing multiple documents asynchronously:")
    
    start_time = time.time()
    
    # Process all documents in parallel
    parallel_chain = RunnableParallel(**chains)
    
    tasks = []
    for i, doc in enumerate(documents):
        task = parallel_chain.ainvoke({"text": doc})
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    
    for i, (doc, result) in enumerate(zip(documents, results)):
        print(f"\nDocument {i+1}: {doc}")
        print(f"Summary: {result['summary'][:100]}...")
        print(f"Sentiment: {result['sentiment'][:50]}...")
        print(f"Keywords: {result['keywords'][:50]}...")
    
    print(f"\n⏱️ Total processing time: {end_time - start_time:.2f} seconds")
    print("✅ Async processing completed")

if __name__ == "__main__":
    print("Modern Chains - Building Complex Workflows with LCEL")
    print("=" * 70)
    
    try:
        # Test Azure OpenAI connection first
        print("🔍 Testing Azure OpenAI connection...")
        test_llm = create_azure_chat_openai()
        test_response = test_llm.invoke("Hello!")
        print(f"✅ Connection successful: {test_response.content[:50]}...")
        print()
        
        # Run synchronous examples
        # lcel_basics_example()
        parallel_chains_example() 
        conditional_chains_example()
        streaming_chains_example()
        production_chains_example()
        
        # Run async example
        print("🔄 Running async example...")
        asyncio.run(async_chains_example())
        
        print("🎉 All modern chain examples completed!")
        print("Next: Check out 06_Agents/ for autonomous AI systems")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Configured Azure OpenAI settings in .env file")
        print("2. Valid Azure OpenAI deployments")
        print("3. Sufficient quota in your Azure OpenAI resource")
