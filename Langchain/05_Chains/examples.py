"""
Advanced Chains - Building Complex Workflows

This module demonstrates different types of chains in LangChain:
1. Basic chain types and their uses
2. Sequential chains for multi-step workflows
3. Parallel chain execution
4. Custom chain implementations
5. Chain debugging and optimization
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Example 1: Basic Chain Types
def basic_chains_example():
    """
    Demonstrates fundamental chain types
    """
    print("=== Example 1: Basic Chain Types ===")
    
    from langchain_openai import OpenAI
    from langchain.chains import LLMChain, ConversationChain
    from langchain.prompts import PromptTemplate
    from langchain.memory import ConversationBufferMemory
    from langchain.chains.transform import TransformChain
    
    llm = OpenAI(temperature=0.7)
    
    # 1. Basic LLMChain
    print("🔗 1. LLMChain:")
    template = "Write a {length} summary of {topic}:"
    prompt = PromptTemplate(
        input_variables=["length", "topic"],
        template=template
    )
    
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    result = llm_chain.run(length="brief", topic="artificial intelligence")
    print(f"Summary: {result.strip()}")
    print()
    
    # 2. ConversationChain with Memory
    print("💬 2. ConversationChain:")
    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )
    
    response1 = conversation.predict(input="My favorite color is blue")
    print(f"Human: My favorite color is blue")
    print(f"AI: {response1}")
    
    response2 = conversation.predict(input="What's my favorite color?")
    print(f"Human: What's my favorite color?")
    print(f"AI: {response2}")
    print()
    
    # 3. TransformChain (no LLM)
    print("🔄 3. TransformChain:")
    def transform_text(inputs):
        text = inputs["text"]
        return {"transformed_text": text.upper().replace(" ", "_")}
    
    transform_chain = TransformChain(
        input_variables=["text"],
        output_variables=["transformed_text"],
        transform=transform_text
    )
    
    result = transform_chain.run(text="hello world from langchain")
    print(f"Original: hello world from langchain")
    print(f"Transformed: {result}")
    print()
    
    print("="*50 + "\n")


# Example 2: Sequential Chains
def sequential_chains_example():
    """
    Build multi-step workflows with sequential chains
    """
    print("=== Example 2: Sequential Chains ===")
    
    from langchain_openai import OpenAI
    from langchain.chains import LLMChain, SequentialChain
    from langchain.prompts import PromptTemplate
    
    llm = OpenAI(temperature=0.7)
    
    # Step 1: Generate a product name
    name_template = """
    Create a creative name for a {product_type} that targets {target_audience}.
    The name should be memorable and modern.
    
    Product name:"""
    
    name_prompt = PromptTemplate(
        input_variables=["product_type", "target_audience"],
        template=name_template
    )
    
    name_chain = LLMChain(
        llm=llm,
        prompt=name_prompt,
        output_key="product_name"
    )
    
    # Step 2: Create a tagline
    tagline_template = """
    Create a catchy tagline for a product called "{product_name}".
    The product is a {product_type} for {target_audience}.
    
    Tagline:"""
    
    tagline_prompt = PromptTemplate(
        input_variables=["product_name", "product_type", "target_audience"],
        template=tagline_template
    )
    
    tagline_chain = LLMChain(
        llm=llm,
        prompt=tagline_prompt,
        output_key="tagline"
    )
    
    # Step 3: Write a product description
    description_template = """
    Write a compelling product description for "{product_name}".
    Tagline: "{tagline}"
    Product type: {product_type}
    Target audience: {target_audience}
    
    Product description:"""
    
    description_prompt = PromptTemplate(
        input_variables=["product_name", "tagline", "product_type", "target_audience"],
        template=description_template
    )
    
    description_chain = LLMChain(
        llm=llm,
        prompt=description_prompt,
        output_key="description"
    )
    
    # Combine into sequential chain
    overall_chain = SequentialChain(
        chains=[name_chain, tagline_chain, description_chain],
        input_variables=["product_type", "target_audience"],
        output_variables=["product_name", "tagline", "description"]
    )
    
    # Test the chain
    result = overall_chain({
        "product_type": "smartphone app",
        "target_audience": "busy professionals"
    })
    
    print("📱 Product Marketing Chain Results:")
    print(f"Product Name: {result['product_name']}")
    print(f"Tagline: {result['tagline']}")
    print(f"Description: {result['description']}")
    print()
    print("="*50 + "\n")


# Example 3: Router Chain
def router_chain_example():
    """
    Route inputs to different chains based on content
    """
    print("=== Example 3: Router Chain ===")
    
    from langchain_openai import OpenAI
    from langchain.chains import LLMChain
    from langchain.chains.router import MultiPromptChain
    from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
    from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
    from langchain.prompts import PromptTemplate
    
    llm = OpenAI(temperature=0)
    
    # Define specialized chains
    math_template = """
    You are a mathematics expert. Solve this problem step by step:
    
    Problem: {input}
    
    Solution:"""
    
    science_template = """
    You are a science expert. Explain this scientific concept clearly:
    
    Topic: {input}
    
    Explanation:"""
    
    history_template = """
    You are a history expert. Provide historical context and facts about:
    
    Topic: {input}
    
    Historical information:"""
    
    # Create prompt infos for router
    prompt_infos = [
        {
            "name": "math",
            "description": "Good for solving mathematical problems and calculations",
            "prompt_template": math_template
        },
        {
            "name": "science", 
            "description": "Good for explaining scientific concepts and theories",
            "prompt_template": science_template
        },
        {
            "name": "history",
            "description": "Good for providing historical information and context",
            "prompt_template": history_template
        }
    ]
    
    # Create destination chains
    destination_chains = {}
    for p_info in prompt_infos:
        name = p_info["name"]
        prompt_template = p_info["prompt_template"]
        prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
        chain = LLMChain(llm=llm, prompt=prompt)
        destination_chains[name] = chain
    
    # Create router chain
    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)
    
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser()
    )
    
    router_chain = LLMRouterChain.from_llm(llm, router_prompt)
    
    # Create multi-prompt chain
    chain = MultiPromptChain(
        router_chain=router_chain,
        destination_chains=destination_chains,
        default_chain=destination_chains["science"],
        verbose=True
    )
    
    # Test different types of queries
    test_queries = [
        "What is the square root of 144?",
        "Explain photosynthesis",
        "When did World War II end?"
    ]
    
    print("🔀 Router Chain Results:")
    for query in test_queries:
        result = chain.run(query)
        print(f"Query: {query}")
        print(f"Response: {result.strip()[:100]}...")
        print("-" * 40)
    
    print("="*50 + "\n")


# Example 4: Custom Chain
def custom_chain_example():
    """
    Build a custom chain for specific business logic
    """
    print("=== Example 4: Custom Chain ===")
    
    from langchain.chains.base import Chain
    from langchain_openai import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from typing import Dict, List
    
    class ContentModerationChain(Chain):
        """Custom chain for content moderation"""
        
        def __init__(self, llm):
            super().__init__()
            self.llm = llm
            
            # Toxicity detection prompt
            toxicity_template = """
            Analyze the following text for toxicity, hate speech, or inappropriate content.
            Respond with SAFE or UNSAFE followed by a brief explanation.
            
            Text: {text}
            
            Analysis:"""
            
            self.toxicity_chain = LLMChain(
                llm=llm,
                prompt=PromptTemplate(
                    input_variables=["text"],
                    template=toxicity_template
                )
            )
            
            # Content improvement prompt  
            improvement_template = """
            Rewrite the following text to be more professional and appropriate:
            
            Original: {text}
            
            Improved version:"""
            
            self.improvement_chain = LLMChain(
                llm=llm,
                prompt=PromptTemplate(
                    input_variables=["text"],
                    template=improvement_template
                )
            )
        
        @property
        def input_keys(self) -> List[str]:
            return ["text"]
        
        @property  
        def output_keys(self) -> List[str]:
            return ["is_safe", "analysis", "improved_text"]
        
        def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
            text = inputs["text"]
            
            # Step 1: Check for toxicity
            toxicity_result = self.toxicity_chain.run(text=text)
            is_safe = "SAFE" in toxicity_result.upper()
            
            # Step 2: Generate improved version if needed
            improved_text = text
            if not is_safe:
                improved_text = self.improvement_chain.run(text=text)
            
            return {
                "is_safe": is_safe,
                "analysis": toxicity_result,
                "improved_text": improved_text
            }
    
    # Test the custom chain
    llm = OpenAI(temperature=0.3)
    moderation_chain = ContentModerationChain(llm)
    
    test_texts = [
        "This is a great product, I love it!",
        "This product is terrible and the company is awful!",
        "Hello, how are you doing today?"
    ]
    
    print("🛡️ Content Moderation Chain Results:")
    for text in test_texts:
        result = moderation_chain.run(text=text)
        print(f"Original: {text}")
        print(f"Safe: {result['is_safe']}")
        print(f"Analysis: {result['analysis']}")
        if not result['is_safe']:
            print(f"Improved: {result['improved_text']}")
        print("-" * 40)
    
    print("="*50 + "\n")


# Example 5: Chain Debugging and Optimization
def chain_debugging_example():
    """
    Techniques for debugging and optimizing chains
    """
    print("=== Example 5: Chain Debugging ===")
    
    from langchain_openai import OpenAI
    from langchain.chains import LLMChain, SequentialChain
    from langchain.prompts import PromptTemplate
    from langchain.callbacks import StdOutCallbackHandler
    import time
    
    llm = OpenAI(temperature=0.7)
    
    # Create a chain with timing
    class TimingChain(LLMChain):
        def _call(self, inputs, run_manager=None):
            start_time = time.time()
            result = super()._call(inputs, run_manager)
            end_time = time.time()
            print(f"⏱️ Chain execution time: {end_time - start_time:.2f} seconds")
            return result
    
    # Build a debugging chain
    template = """
    Generate a {content_type} about {topic}.
    Make it {tone} and {length}.
    
    Content:"""
    
    prompt = PromptTemplate(
        input_variables=["content_type", "topic", "tone", "length"],
        template=template
    )
    
    timing_chain = TimingChain(
        llm=llm,
        prompt=prompt,
        verbose=True  # Enable verbose logging
    )
    
    print("🔍 Chain Debugging Features:")
    print("1. Verbose logging enabled")
    print("2. Timing measurements")
    print("3. Input/output validation")
    print()
    
    # Test with different inputs
    test_cases = [
        {
            "content_type": "blog post",
            "topic": "artificial intelligence",
            "tone": "professional",
            "length": "brief"
        },
        {
            "content_type": "poem",
            "topic": "nature",
            "tone": "whimsical", 
            "length": "short"
        }
    ]
    
    for i, inputs in enumerate(test_cases):
        print(f"🧪 Test Case {i+1}:")
        print(f"Inputs: {inputs}")
        
        try:
            result = timing_chain.run(**inputs)
            print(f"✅ Success! Output length: {len(result)} characters")
            print(f"Sample: {result[:100]}...")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 40)
    
    print("\n💡 Debugging Tips:")
    print("- Use verbose=True to see chain internals")
    print("- Add timing measurements for performance")
    print("- Validate inputs and outputs")
    print("- Test with edge cases and invalid inputs")
    print("- Use callbacks for custom monitoring")
    print("- Log intermediate results in complex chains")
    
    print("="*50 + "\n")


if __name__ == "__main__":
    print("Advanced Chains - Building Complex Workflows")
    print("=" * 60)
    
    try:
        basic_chains_example()
        sequential_chains_example()
        # router_chain_example()  # Uncomment if you want to test router
        custom_chain_example()
        chain_debugging_example()
        
        print("🎉 All chain examples completed!")
        print("Next: Check out 06_Agents/ for autonomous AI systems")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have your OPENAI_API_KEY set in the .env file")
