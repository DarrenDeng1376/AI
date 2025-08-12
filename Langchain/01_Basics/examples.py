"""
LangChain Basics - Core Examples

This file demonstrates the fundamental concepts of LangChain:
1. Setting up Azure OpenAI LLM
2. Using prompt templates
3. Creating simple chains
4. Basic conversation memory

Run each example separately to understand the concepts.
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path for azure_config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from azure_config import create_azure_chat_openai, create_azure_openai_llm

# Load environment variables
load_dotenv()

# Example 1: Basic LLM Usage
def basic_llm_example():
    """
    Demonstrates how to use Azure OpenAI with LangChain
    """
    print("=== Example 1: Basic Azure OpenAI LLM Usage ===")
    
    try:
        # Initialize Azure OpenAI LLM
        llm = create_azure_chat_openai(
            temperature=0.7,  # Controls randomness (0-1)
        )
        
        # Simple prompt
        prompt = "What are the benefits of using LangChain?"
        
        # Get response
        response = llm.invoke(prompt)
        print(f"Prompt: {prompt}")
        print(f"Response: {response.content}")
        print("\n" + "="*50 + "\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your Azure OpenAI configuration is correct in .env file")
        print("\n" + "="*50 + "\n")


# Example 2: Prompt Templates
def prompt_template_example():
    """
    Shows how to use prompt templates with Azure OpenAI
    """
    print("=== Example 2: Prompt Templates ===")
    
    try:
        from langchain.prompts import PromptTemplate
        
        llm = create_azure_chat_openai(temperature=0.7)
        
        # Create a prompt template
        template = """
        You are a helpful AI assistant specialized in {topic}.
        
        Question: {question}
        
        Please provide a detailed and helpful answer.
        """
        
        prompt = PromptTemplate(
            input_variables=["topic", "question"],
            template=template
        )
        
        # Format the prompt
        formatted_prompt = prompt.format(
            topic="Python programming",
            question="How do I handle exceptions in Python?"
        )
        
        print(f"Formatted Prompt:\n{formatted_prompt}")
        
        # Get response
        response = llm.invoke(formatted_prompt)
        print(f"Response: {response.content}")
        print("\n" + "="*50 + "\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n" + "="*50 + "\n")


# Example 3: Simple Chain
def simple_chain_example():
    """
    Demonstrates how to create a simple LangChain chain with Azure OpenAI
    """
    print("=== Example 3: Simple Chain ===")
    
    try:
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
        
        llm = create_azure_chat_openai(temperature=0.7)
        
        # Create prompt template
        prompt = PromptTemplate(
            input_variables=["product"],
            template="Write a creative marketing slogan for {product}:"
        )
        
        # Create chain
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Run chain
        result = chain.run(product="eco-friendly water bottles")
        
        print(f"Marketing slogan: {result}")
        print("\n" + "="*50 + "\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n" + "="*50 + "\n")


# Example 4: Conversation Memory
def conversation_memory_example():
    """
    Shows how to add memory to maintain conversation context with Azure OpenAI
    """
    print("=== Example 4: Conversation Memory ===")
    
    try:
        from langchain.chains import ConversationChain
        from langchain.memory import ConversationBufferMemory
        
        llm = create_azure_chat_openai(temperature=0.7)
        
        # Create conversation chain with memory
        conversation = ConversationChain(
            llm=llm,
            memory=ConversationBufferMemory(),
            verbose=True  # Shows the conversation history
        )
        
        # Have a conversation
        print("Starting conversation...")
        
        response1 = conversation.predict(input="Hi, my name is Alice")
        print(f"Response 1: {response1}\n")
        
        response2 = conversation.predict(input="What's my name?")
        print(f"Response 2: {response2}\n")
        
        response3 = conversation.predict(input="What did we just talk about?")
        print(f"Response 3: {response3}\n")
        
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("="*50 + "\n")


# Example 5: Custom Chain with Multiple Steps
def custom_chain_example():
    """
    Creates a custom chain that performs multiple steps with Azure OpenAI
    """
    print("=== Example 5: Custom Chain ===")
    
    try:
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain, SequentialChain
        
        llm = create_azure_chat_openai(temperature=0.7)
        
        # Step 1: Generate a story idea
        idea_template = """
        Generate a creative story idea about {theme}.
        Just provide the main concept in one sentence.
        """
        idea_prompt = PromptTemplate(
            input_variables=["theme"],
            template=idea_template
        )
        idea_chain = LLMChain(
            llm=llm,
            prompt=idea_prompt,
            output_key="story_idea"
        )
        
        # Step 2: Create characters for the story
        character_template = """
        Based on this story idea: {story_idea}
        
        Create 2-3 main characters for this story.
        """
        character_prompt = PromptTemplate(
            input_variables=["story_idea"],
            template=character_template
        )
        character_chain = LLMChain(
            llm=llm,
            prompt=character_prompt,
            output_key="characters"
        )
        
        # Combine chains
        overall_chain = SequentialChain(
            chains=[idea_chain, character_chain],
            input_variables=["theme"],
            output_variables=["story_idea", "characters"]
        )
        
        # Run the chain
        result = overall_chain({"theme": "time travel"})
        
        print(f"Story Idea: {result['story_idea']}")
        print(f"Characters: {result['characters']}")
        print("\n" + "="*50 + "\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    print("Welcome to LangChain Basics Examples!")
    print("Using Azure OpenAI - Make sure you have configured your Azure OpenAI settings in .env")
    print("\n")
    
    try:
        # Test Azure OpenAI connection first
        print("🔍 Testing Azure OpenAI connection...")
        test_llm = create_azure_chat_openai(temperature=0)
        test_response = test_llm.invoke("Say hello!")
        print(f"✅ Connection successful: {test_response.content}")
        print()
        
        # Run examples
        basic_llm_example()
        prompt_template_example()
        simple_chain_example()
        conversation_memory_example()
        custom_chain_example()
        
        print("🎉 All examples completed successfully!")
        print("Next: Check out 02_Prompts/ for advanced prompt engineering")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Installed required packages: pip install langchain langchain-openai python-dotenv")
        print("2. Set your Azure OpenAI configuration in the .env file:")
        print("   - AZURE_OPENAI_API_KEY")
        print("   - AZURE_OPENAI_ENDPOINT") 
        print("   - AZURE_OPENAI_CHAT_DEPLOYMENT")
        print("3. Your Azure OpenAI resource has the required deployments")
