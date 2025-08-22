"""
Built-in Tools - Modern LangChain Tool Usage

This module demonstrates using modern LangChain built-in tools with Azure OpenAI,
showcasing search tools, web APIs, and computational tools with proper error handling.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from azure_config import create_azure_chat_openai
from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL
import requests
import json
from typing import Dict, Any, Optional
import time

load_dotenv()

# Example 1: Search Tools
def search_tools_demo():
    """Demonstrate modern search tools"""
    print("=== Example 1: Modern Search Tools ===")
    
    # Wikipedia Tool
    wikipedia = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(
            top_k_results=2,
            doc_content_chars_max=500
        )
    )
    
    # DuckDuckGo Search Tool
    duckduckgo_search = DuckDuckGoSearchRun(
        api_wrapper=DuckDuckGoSearchAPIWrapper(
            region="us-en",
            time="y",  # Past year
            max_results=3
        )
    )
    
    # Tavily Search (if API key available)
    tavily_search = None
    if os.getenv("TAVILY_API_KEY"):
        tavily_search = TavilySearchResults(
            max_results=3,
            search_depth="advanced",
            include_images=False,
            include_answer=True
        )
    
    # Create search agent
    llm = create_azure_chat_openai(temperature=0.1)
    
    tools = [wikipedia, duckduckgo_search]
    if tavily_search:
        tools.append(tavily_search)
    
    search_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research assistant with access to multiple search tools.
        
        Available tools:
        - Wikipedia: For encyclopedic information and background knowledge
        - DuckDuckGo: For general web search and current information
        - Tavily: For advanced web search with AI-powered filtering (if available)
        
        When searching:
        1. Start with Wikipedia for background information
        2. Use DuckDuckGo for current events and specific queries
        3. Use Tavily for complex research that needs AI filtering
        4. Always cite your sources and provide accurate information"""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    search_agent = create_tool_calling_agent(llm, tools, search_prompt)
    search_executor = AgentExecutor(agent=search_agent, tools=tools, verbose=True)
    
    # Test search queries
    search_queries = [
        "What is artificial intelligence and what are the latest developments in 2024?",
        "Find information about climate change policies implemented in 2024",
        "Research the current state of quantum computing technology"
    ]
    
    print("🔍 Search Tools Results:")
    
    for i, query in enumerate(search_queries, 1):
        print(f"\n📋 Query {i}: {query}")
        print("-" * 60)
        
        try:
            result = search_executor.invoke({"input": query})
            print(f"🎯 Search Result: {result['output']}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "="*70 + "\n")

# Example 2: Web API Tools
def web_api_tools_demo():
    """Demonstrate web API integration tools"""
    print("=== Example 2: Web API Integration Tools ===")
    
    @tool
    def get_weather_info(city: str) -> str:
        """Get current weather information for a city using OpenWeatherMap API"""
        api_key = os.getenv("OPENWEATHER_API_KEY")
        if not api_key:
            return "Weather API key not configured. Please set OPENWEATHER_API_KEY environment variable."
        
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            weather = data["weather"][0]
            main = data["main"]
            
            return f"""Weather in {city}:
            - Condition: {weather['description'].title()}
            - Temperature: {main['temp']}°C (feels like {main['feels_like']}°C)
            - Humidity: {main['humidity']}%
            - Pressure: {main['pressure']} hPa"""
            
        except requests.RequestException as e:
            return f"Error fetching weather data: {e}"
        except KeyError as e:
            return f"Error parsing weather data: {e}"
    
    @tool
    def get_news_headlines(category: str = "general", country: str = "us") -> str:
        """Get latest news headlines by category using NewsAPI"""
        api_key = os.getenv("NEWS_API_KEY")
        if not api_key:
            return "News API key not configured. Please set NEWS_API_KEY environment variable."
        
        try:
            url = f"https://newsapi.org/v2/top-headlines?country={country}&category={category}&apiKey={api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            articles = data["articles"][:5]  # Get top 5 articles
            
            headlines = []
            for article in articles:
                headlines.append(f"• {article['title']} - {article['source']['name']}")
            
            return f"Latest {category} news headlines:\n" + "\n".join(headlines)
            
        except requests.RequestException as e:
            return f"Error fetching news data: {e}"
        except KeyError as e:
            return f"Error parsing news data: {e}"
    
    @tool
    def get_crypto_price(symbol: str) -> str:
        """Get current cryptocurrency price using CoinGecko API"""
        try:
            # Convert symbol to CoinGecko ID (simplified mapping)
            symbol_map = {
                "btc": "bitcoin",
                "eth": "ethereum", 
                "ada": "cardano",
                "sol": "solana",
                "dot": "polkadot"
            }
            
            coin_id = symbol_map.get(symbol.lower(), symbol.lower())
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_change=true"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if coin_id in data:
                price_data = data[coin_id]
                price = price_data["usd"]
                change_24h = price_data.get("usd_24h_change", 0)
                
                return f"{symbol.upper()} Price: ${price:,.2f} USD (24h change: {change_24h:+.2f}%)"
            else:
                return f"Cryptocurrency '{symbol}' not found"
                
        except requests.RequestException as e:
            return f"Error fetching crypto data: {e}"
        except Exception as e:
            return f"Error processing crypto data: {e}"
    
    # Create API integration agent
    llm = create_azure_chat_openai(temperature=0.1)
    
    api_tools = [get_weather_info, get_news_headlines, get_crypto_price]
    
    api_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an information assistant with access to various web APIs.
        
        Available tools:
        - get_weather_info: Current weather for any city
        - get_news_headlines: Latest news by category (general, business, technology, sports, etc.)
        - get_crypto_price: Cryptocurrency prices (btc, eth, ada, sol, dot, etc.)
        
        When using APIs:
        1. Always validate inputs before making calls
        2. Handle errors gracefully and inform users
        3. Provide clear, formatted responses
        4. Respect rate limits and be efficient"""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    api_agent = create_tool_calling_agent(llm, api_tools, api_prompt)
    api_executor = AgentExecutor(agent=api_agent, tools=api_tools, verbose=True)
    
    # Test API queries
    api_queries = [
        "What's the weather like in Tokyo right now?",
        "Get me the latest technology news headlines",
        "What's the current price of Bitcoin and Ethereum?"
    ]
    
    print("🌐 Web API Tools Results:")
    
    for i, query in enumerate(api_queries, 1):
        print(f"\n📋 Query {i}: {query}")
        print("-" * 60)
        
        try:
            result = api_executor.invoke({"input": query})
            print(f"🎯 API Result: {result['output']}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Add delay to respect rate limits
        time.sleep(1)
    
    print("\n" + "="*70 + "\n")

# Example 3: Computational Tools
def computational_tools_demo():
    """Demonstrate computational and analysis tools"""
    print("=== Example 3: Computational Tools ===")
    
    # Python REPL Tool (with safety restrictions)
    python_repl = PythonREPL()
    
    @tool
    def safe_python_execution(code: str) -> str:
        """Execute Python code safely with restrictions"""
        # Basic safety checks
        dangerous_imports = ['os', 'sys', 'subprocess', 'shutil', 'glob']
        dangerous_functions = ['exec', 'eval', 'open', '__import__']
        
        for dangerous in dangerous_imports + dangerous_functions:
            if dangerous in code:
                return f"Error: Use of '{dangerous}' is not allowed for security reasons"
        
        try:
            result = python_repl.run(code)
            return f"Execution successful:\n{result}"
        except Exception as e:
            return f"Execution error: {e}"
    
    @tool
    def calculate_statistics(numbers: str) -> str:
        """Calculate basic statistics for a list of numbers"""
        try:
            # Parse numbers from string
            nums = [float(x.strip()) for x in numbers.split(',')]
            
            if not nums:
                return "Error: No valid numbers provided"
            
            import statistics
            
            stats = {
                "count": len(nums),
                "sum": sum(nums),
                "mean": statistics.mean(nums),
                "median": statistics.median(nums),
                "min": min(nums),
                "max": max(nums)
            }
            
            if len(nums) > 1:
                stats["std_dev"] = statistics.stdev(nums)
                stats["variance"] = statistics.variance(nums)
            
            result = "Statistical Analysis:\n"
            for key, value in stats.items():
                result += f"- {key.replace('_', ' ').title()}: {value:.4f}\n"
            
            return result
            
        except ValueError:
            return "Error: Please provide comma-separated numbers"
        except Exception as e:
            return f"Error calculating statistics: {e}"
    
    @tool
    def unit_converter(value: float, from_unit: str, to_unit: str, unit_type: str) -> str:
        """Convert between different units"""
        conversions = {
            "length": {
                "meter": 1.0,
                "kilometer": 1000.0,
                "centimeter": 0.01,
                "millimeter": 0.001,
                "inch": 0.0254,
                "foot": 0.3048,
                "yard": 0.9144,
                "mile": 1609.34
            },
            "weight": {
                "kilogram": 1.0,
                "gram": 0.001,
                "pound": 0.453592,
                "ounce": 0.0283495,
                "ton": 1000.0
            },
            "temperature": {
                # Special handling for temperature
            }
        }
        
        try:
            if unit_type == "temperature":
                # Temperature conversion logic
                if from_unit == "celsius" and to_unit == "fahrenheit":
                    result = (value * 9/5) + 32
                elif from_unit == "fahrenheit" and to_unit == "celsius":
                    result = (value - 32) * 5/9
                elif from_unit == "celsius" and to_unit == "kelvin":
                    result = value + 273.15
                elif from_unit == "kelvin" and to_unit == "celsius":
                    result = value - 273.15
                else:
                    return f"Unsupported temperature conversion: {from_unit} to {to_unit}"
            else:
                if unit_type not in conversions:
                    return f"Unsupported unit type: {unit_type}"
                
                units = conversions[unit_type]
                
                if from_unit not in units or to_unit not in units:
                    return f"Unsupported units for {unit_type}: {from_unit}, {to_unit}"
                
                # Convert to base unit, then to target unit
                base_value = value * units[from_unit]
                result = base_value / units[to_unit]
            
            return f"{value} {from_unit} = {result:.6f} {to_unit}"
            
        except Exception as e:
            return f"Error in unit conversion: {e}"
    
    # Create computational agent
    llm = create_azure_chat_openai(temperature=0.1)
    
    comp_tools = [safe_python_execution, calculate_statistics, unit_converter]
    
    comp_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a computational assistant with tools for calculations and analysis.
        
        Available tools:
        - safe_python_execution: Execute Python code safely (no file operations or dangerous imports)
        - calculate_statistics: Basic statistical analysis of number lists
        - unit_converter: Convert between different units (length, weight, temperature)
        
        When performing computations:
        1. Use appropriate tools for different types of calculations
        2. Validate inputs and handle errors gracefully
        3. Provide clear explanations of results
        4. Show step-by-step calculations when helpful"""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    comp_agent = create_tool_calling_agent(llm, comp_tools, comp_prompt)
    comp_executor = AgentExecutor(agent=comp_agent, tools=comp_tools, verbose=True)
    
    # Test computational queries
    comp_queries = [
        "Calculate the statistics for these numbers: 10, 15, 20, 25, 30, 35, 40",
        "Convert 25 degrees Celsius to Fahrenheit",
        "Calculate the area of a circle with radius 5 using Python"
    ]
    
    print("🧮 Computational Tools Results:")
    
    for i, query in enumerate(comp_queries, 1):
        print(f"\n📋 Query {i}: {query}")
        print("-" * 60)
        
        try:
            result = comp_executor.invoke({"input": query})
            print(f"🎯 Computation Result: {result['output']}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    print("Built-in Tools - Modern LangChain Tool Usage")
    print("=" * 70)
    
    try:
        # Test Azure OpenAI connection
        print("🔍 Testing Azure OpenAI connection...")
        test_llm = create_azure_chat_openai()
        test_response = test_llm.invoke("Hello!")
        print(f"✅ Connection successful: {test_response.content[:50]}...")
        print()
        
        # Run tool examples
        search_tools_demo()
        web_api_tools_demo()
        computational_tools_demo()
        
        print("🎉 Built-in tools examples completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Configured Azure OpenAI settings in .env file")
        print("2. Installed required dependencies: pip install langchain-community")
        print("3. Optional API keys for external services (OpenWeather, NewsAPI, etc.)")
