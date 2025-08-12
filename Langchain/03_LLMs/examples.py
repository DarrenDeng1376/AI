"""
Working with Azure OpenAI Language Models

This module demonstrates how to work with Azure OpenAI models in LangChain:
1. Different Azure OpenAI model configurations
2. Model comparison and selection
3. Cost optimization strategies
4. Deployment management
5. Fallback mechanisms
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from azure_config import create_azure_chat_openai, create_azure_openai
from dotenv import load_dotenv

load_dotenv()

# Example 1: Azure OpenAI Model Configurations
def azure_model_configurations_example():
    """
    Demonstrates different Azure OpenAI model configurations
    """
    print("=== Example 1: Azure OpenAI Model Configurations ===")
    
    try:
        print("🤖 Azure OpenAI Model Configurations:")
        
        # Chat model with different temperatures
        print("\n📊 Temperature Comparison:")
        temperatures = [0.1, 0.5, 0.9]
        prompt = "Write a creative opening line for a story."
        
        for temp in temperatures:
            llm = create_azure_chat_openai(temperature=temp)
            response = llm.invoke(prompt)
            print(f"Temperature {temp}: {response.content}")
        
        print("\n💬 Chat vs Completion Models:")
        
        # Chat model (recommended for most use cases)
        chat_model = create_azure_chat_openai(temperature=0.7)
        chat_response = chat_model.invoke("Explain machine learning in simple terms.")
        print(f"Chat Model: {chat_response.content[:150]}...")
        
        # Completion model (legacy, but useful for specific tasks)
        completion_model = create_azure_openai(temperature=0.7)
        completion_response = completion_model.invoke("Machine learning is")
        print(f"Completion Model: {completion_response[:150]}...")
        
    except Exception as e:
        print(f"Configuration Error: {e}")
        print("Make sure your Azure OpenAI deployment names are set correctly")
    
    print("\n" + "="*50 + "\n")

def model_parameter_tuning_example():
    """
    Demonstrates various model parameters and their effects
    """
    print("=== Example 2: Model Parameter Tuning ===")
    
    print("⚙️ Key Parameters and Their Effects:")
    
    parameters = {
        "temperature": {
            "0.0-0.3": "Factual, consistent responses",
            "0.4-0.7": "Balanced creativity and accuracy", 
            "0.8-1.0": "Highly creative, more random"
        },
        "max_tokens": {
            "50-100": "Short, concise responses",
            "200-500": "Medium-length responses",
            "1000+": "Detailed, comprehensive responses"
        },
        "top_p": {
            "0.1": "Very focused responses",
            "0.5": "Balanced selection",
            "0.9": "More diverse vocabulary"
        }
    }
    
    for param, values in parameters.items():
        print(f"\n📈 {param.upper()}:")
        for setting, description in values.items():
            print(f"  {setting}: {description}")
    
    # Practical demonstration
    print("\n🧪 Practical Demonstration:")
    prompt = "Describe the color blue."
    
    configs = [
        ("Conservative", {"temperature": 0.1, "max_tokens": 50}),
        ("Balanced", {"temperature": 0.5, "max_tokens": 100}),
        ("Creative", {"temperature": 0.9, "max_tokens": 150})
    ]
    
    for config_name, config in configs:
        try:
            llm = create_azure_chat_openai(**config)
            response = llm.invoke(prompt)
            print(f"{config_name}: {response.content}")
        except Exception as e:
            print(f"Error with {config_name}: {e}")
    
    print("\n" + "="*50 + "\n")

def cost_optimization_example():
    """
    Demonstrates strategies for optimizing Azure OpenAI costs
    """
    print("=== Example 3: Azure OpenAI Cost Optimization ===")
    
    print("💰 Cost Optimization Strategies:")
    
    print("\n1. Model Selection:")
    print("   - GPT-3.5 Turbo: Fast, cost-effective for most tasks")
    print("   - GPT-4: Higher quality but more expensive")
    print("   - Use GPT-3.5 for simple tasks, GPT-4 for complex reasoning")
    
    print("\n2. Token Management:")
    
    # Demonstrate token limiting
    prompt = "Explain quantum computing in detail with examples and applications."
    
    # Without limits
    print("📏 Without token limits:")
    try:
        unlimited = create_azure_chat_openai(temperature=0.3)
        full_response = unlimited.invoke(prompt)
        print(f"Full response: {len(full_response.content)} characters")
        print(f"Preview: {full_response.content[:100]}...")
    except Exception as e:
        print(f"Error: {e}")
    
    # With token limits  
    print("\n📏 With token limits (max_tokens=100):")
    try:
        limited = create_azure_chat_openai(temperature=0.3, max_tokens=100)
        limited_response = limited.invoke(prompt)
        print(f"Limited response: {len(limited_response.content)} characters")
        print(f"Response: {limited_response.content}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n3. Prompt Optimization:")
    print("   - Be specific and concise")
    print("   - Use system messages to set context once")
    print("   - Avoid redundant information")
    
    print("\n4. Batch Processing:")
    print("   - Process multiple queries together when possible")
    print("   - Use caching for repeated queries")
    print("   - Implement request queuing for high-volume apps")
    
    print("\n" + "="*50 + "\n")

def deployment_management_example():
    """
    Demonstrates Azure OpenAI deployment management concepts
    """
    print("=== Example 4: Deployment Management ===")
    
    print("🚀 Azure OpenAI Deployment Best Practices:")
    
    print("\n📋 Deployment Planning:")
    print("1. Choose appropriate regions for latency")
    print("2. Set up multiple deployments for different models")
    print("3. Configure appropriate scaling settings")
    print("4. Monitor usage and performance")
    
    print("\n⚙️ Configuration Example:")
    print("""
    Environment Variables Setup:
    - AZURE_OPENAI_ENDPOINT: Your Azure endpoint
    - AZURE_OPENAI_API_KEY: Your API key
    - AZURE_OPENAI_CHAT_DEPLOYMENT: Your chat model deployment
    - AZURE_OPENAI_COMPLETION_DEPLOYMENT: Your completion model deployment
    """)
    
    print("\n🔍 Monitoring and Alerts:")
    print("- Set up Azure Monitor for API usage")
    print("- Configure alerts for quota limits") 
    print("- Monitor response times and error rates")
    print("- Track costs and usage patterns")
    
    print("\n🔄 Load Balancing:")
    print("- Use multiple deployments across regions")
    print("- Implement client-side load balancing")
    print("- Set up failover mechanisms")
    print("- Monitor deployment health")
    
    # Demonstrate current deployment info
    try:
        print("\n� Current Deployment Test:")
        test_model = create_azure_chat_openai(temperature=0)
        test_response = test_model.invoke("Hello, Azure OpenAI!")
        print(f"✅ Deployment working: {test_response.content}")
    except Exception as e:
        print(f"❌ Deployment issue: {e}")
    
    print("\n" + "="*50 + "\n")

def azure_fallback_strategy_example():
    """
    Implement fallback mechanisms for Azure OpenAI
    """
    print("=== Example 5: Azure Fallback Strategy ===")
    
    class AzureFallbackManager:
        def __init__(self):
            # Different deployment configurations
            self.strategies = [
                ("Primary Chat", lambda: create_azure_chat_openai(temperature=0.7)),
                ("Conservative Chat", lambda: create_azure_chat_openai(temperature=0.1)),
                ("Completion Fallback", lambda: create_azure_openai(temperature=0.7))
            ]
    
        def invoke_with_fallback(self, prompt, max_retries=2):
            for strategy_name, model_creator in self.strategies:
                for attempt in range(max_retries):
                    try:
                        print(f"🔄 Trying {strategy_name} (attempt {attempt + 1})...")
                        model = model_creator()
                        
                        if "Chat" in strategy_name:
                            result = model.invoke(prompt)
                            content = result.content
                        else:
                            result = model.invoke(prompt)
                            content = result
                        
                        print(f"✅ Success with {strategy_name}")
                        return content
                        
                    except Exception as e:
                        print(f"❌ {strategy_name} failed: {str(e)[:100]}...")
                        if attempt == max_retries - 1:
                            print(f"Max retries reached for {strategy_name}")
                            break
            
            raise Exception("All fallback strategies failed")
    
    # Test fallback system
    print("🛡️ Testing Fallback System:")
    fallback = AzureFallbackManager()
    
    try:
        result = fallback.invoke_with_fallback("What is machine learning?")
        print(f"📝 Final result: {result[:150]}...")
    except Exception as e:
        print(f"💥 All strategies failed: {e}")
    
    print("\n💡 Fallback Best Practices:")
    print("- Have multiple deployment regions")
    print("- Use different model types (chat vs completion)")
    print("- Implement exponential backoff")
    print("- Log failures for monitoring")
    print("- Consider local model fallbacks for critical apps")
    
    print("\n" + "="*50 + "\n")

def model_selection_guide_example():
    """
    Guide for selecting the right Azure OpenAI model
    """
    print("=== Example 6: Azure OpenAI Model Selection Guide ===")
    
    print("🎯 Azure OpenAI Model Selection Framework:")
    
    selection_guide = {
        "Task Type": {
            "Simple Q&A": "GPT-3.5 Turbo",
            "Complex reasoning": "GPT-4",
            "Creative writing": "GPT-4 or GPT-3.5 with high temperature",
            "Code generation": "GPT-4 for complex, GPT-3.5 for simple",
            "Text completion": "GPT-3.5 Completion model"
        },
        "Response Length": {
            "Short (< 200 tokens)": "GPT-3.5 Turbo",
            "Medium (200-1000 tokens)": "GPT-3.5 Turbo or GPT-4",
            "Long (> 1000 tokens)": "GPT-4 with higher max_tokens"
        },
        "Accuracy Requirements": {
            "High precision": "GPT-4 with low temperature (0.1-0.3)",
            "Balanced": "GPT-3.5 with medium temperature (0.5-0.7)",
            "Creative": "Any model with high temperature (0.8-1.0)"
        },
        "Budget Considerations": {
            "Cost-sensitive": "GPT-3.5 Turbo",
            "Quality-first": "GPT-4",
            "Balanced": "GPT-3.5 for most tasks, GPT-4 for critical ones"
        }
    }
    
    for category, options in selection_guide.items():
        print(f"\n📊 {category}:")
        for use_case, recommendation in options.items():
            print(f"  • {use_case}: {recommendation}")
    
    print("\n🔍 Decision Tree:")
    print("1. Is high accuracy critical? → GPT-4")
    print("2. Is cost a major factor? → GPT-3.5 Turbo")
    print("3. Need creative output? → High temperature settings")
    print("4. Simple factual queries? → GPT-3.5 with low temperature")
    print("5. Complex reasoning needed? → GPT-4")
    
    print("\n⚡ Performance Tips:")
    print("- Use system messages to set context efficiently")
    print("- Adjust temperature based on task requirements")
    print("- Set appropriate max_tokens limits")
    print("- Use streaming for real-time applications")
    print("- Implement caching for repeated queries")
    
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    print("Working with Azure OpenAI Language Models")
    print("=" * 60)
    
    try:
        # Test Azure OpenAI connection first
        print("🔍 Testing Azure OpenAI connection...")
        test_llm = create_azure_chat_openai()
        test_response = test_llm.invoke("Hello!")
        print(f"✅ Connection successful: {test_response.content}")
        print()
        
        # Run Azure OpenAI focused examples
        azure_model_configurations_example()
        model_parameter_tuning_example()
        cost_optimization_example()
        deployment_management_example()
        azure_fallback_strategy_example()
        model_selection_guide_example()
        
        print("🎉 All Azure OpenAI LLM examples completed!")
        print("Next: Check out 04_Memory/ for conversation handling and memory management")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Configured Azure OpenAI settings in .env file")
        print("2. Valid Azure OpenAI deployments")
        print("3. Sufficient quota in your Azure OpenAI resource")
