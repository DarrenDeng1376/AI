"""
Custom Tools with Azure Integration

This module demonstrates how to build custom LangChain tools that integrate
with Azure services and provide specialized functionality for business workflows.

Key concepts covered:
1. Custom tool creation with type hints
2. Azure service integrations
3. Async tool operations
4. Error handling and validation
5. Tool composition and chaining
6. Real-world business tools
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from azure_config import create_azure_chat_openai, create_azure_openai_embeddings
from dotenv import load_dotenv

load_dotenv()

from typing import Dict, List, Any, Optional, Union
from langchain_core.tools import BaseTool, tool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import asyncio
import aiohttp
import json
import time
from datetime import datetime, timedelta
import sqlite3
import hashlib


# ============================================================================
# 1. AZURE COGNITIVE SERVICES INTEGRATION TOOLS
# ============================================================================

class AzureTextAnalyticsTool(BaseTool):
    """Custom tool for Azure Text Analytics integration"""
    
    name: str = "azure_text_analytics"
    description: str = """
    Analyze text using Azure Cognitive Services Text Analytics.
    Provides sentiment analysis, key phrase extraction, and language detection.
    Input should be the text to analyze.
    """
    
    def _run(
        self, 
        text: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Analyze text with Azure Text Analytics"""
        try:
            # Simulate Azure Text Analytics API call
            # In real implementation, you'd use the actual Azure SDK
            
            analysis_result = {
                "sentiment": self._analyze_sentiment(text),
                "key_phrases": self._extract_key_phrases(text),
                "language": "en",
                "confidence_scores": {
                    "positive": 0.85 if "good" in text.lower() else 0.2,
                    "neutral": 0.1,
                    "negative": 0.05 if "bad" not in text.lower() else 0.8
                }
            }
            
            return json.dumps(analysis_result, indent=2)
            
        except Exception as e:
            return f"Error analyzing text: {str(e)}"
    
    def _analyze_sentiment(self, text: str) -> str:
        """Simple sentiment analysis logic"""
        positive_words = ["good", "excellent", "great", "amazing", "love", "like"]
        negative_words = ["bad", "terrible", "hate", "awful", "horrible", "dislike"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text"""
        # Simple implementation - in real scenario use Azure API
        words = text.split()
        phrases = []
        
        # Extract potential key phrases (2-3 word combinations)
        for i in range(len(words) - 1):
            if len(words[i]) > 3 and len(words[i + 1]) > 3:
                phrases.append(f"{words[i]} {words[i + 1]}")
        
        return phrases[:5]  # Return top 5 phrases


class AzureTranslatorTool(BaseTool):
    """Custom tool for Azure Translator integration"""
    
    name: str = "azure_translator"
    description: str = """
    Translate text using Azure Translator service.
    Input format: "text|target_language" (e.g., "Hello|es" for Spanish)
    Supports languages: en, es, fr, de, zh, ja, ko
    """
    
    def _run(
        self, 
        input_text: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Translate text using Azure Translator"""
        try:
            parts = input_text.split("|")
            if len(parts) != 2:
                return "Error: Input format should be 'text|target_language'"
            
            text, target_lang = parts[0].strip(), parts[1].strip()
            
            # Simulate translation (in real implementation, use Azure Translator API)
            translations = {
                "es": self._translate_to_spanish(text),
                "fr": self._translate_to_french(text),
                "de": self._translate_to_german(text),
                "zh": "您好，世界" if "hello" in text.lower() else "翻译的文本",
                "ja": "こんにちは世界" if "hello" in text.lower() else "翻訳されたテキスト",
                "ko": "안녕하세요 세계" if "hello" in text.lower() else "번역된 텍스트"
            }
            
            translated = translations.get(target_lang, f"Translation to {target_lang} not available")
            
            return json.dumps({
                "original_text": text,
                "target_language": target_lang,
                "translated_text": translated,
                "confidence": 0.95
            }, indent=2)
            
        except Exception as e:
            return f"Translation error: {str(e)}"
    
    def _translate_to_spanish(self, text: str) -> str:
        """Simple Spanish translation"""
        translations = {
            "hello": "hola",
            "world": "mundo",
            "good": "bueno",
            "thank you": "gracias",
            "yes": "sí",
            "no": "no"
        }
        
        text_lower = text.lower()
        for en, es in translations.items():
            text_lower = text_lower.replace(en, es)
        
        return text_lower
    
    def _translate_to_french(self, text: str) -> str:
        """Simple French translation"""
        translations = {
            "hello": "bonjour",
            "world": "monde",
            "good": "bon",
            "thank you": "merci",
            "yes": "oui",
            "no": "non"
        }
        
        text_lower = text.lower()
        for en, fr in translations.items():
            text_lower = text_lower.replace(en, fr)
        
        return text_lower
    
    def _translate_to_german(self, text: str) -> str:
        """Simple German translation"""
        translations = {
            "hello": "hallo",
            "world": "welt",
            "good": "gut",
            "thank you": "danke",
            "yes": "ja",
            "no": "nein"
        }
        
        text_lower = text.lower()
        for en, de in translations.items():
            text_lower = text_lower.replace(en, de)
        
        return text_lower


# ============================================================================
# 2. BUSINESS DATA TOOLS
# ============================================================================

class CustomerDatabaseTool(BaseTool):
    """Tool for customer database operations"""
    
    name: str = "customer_database"
    description: str = """
    Query customer database for information.
    Supports: get_customer|customer_id, search_customer|name, list_customers|limit
    """
    
    def __init__(self):
        super().__init__()
        self._init_database()
    
    def _init_database(self):
        """Initialize in-memory SQLite database with sample data"""
        self.conn = sqlite3.connect(":memory:")
        cursor = self.conn.cursor()
        
        # Create customers table
        cursor.execute("""
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT,
                tier TEXT,
                created_date TEXT,
                total_spent REAL
            )
        """)
        
        # Insert sample data
        sample_customers = [
            (1, "Microsoft Corp", "contact@microsoft.com", "enterprise", "2023-01-15", 125000.00),
            (2, "Startup Inc", "info@startup.com", "pro", "2023-06-10", 5000.00),
            (3, "Local Business", "hello@local.com", "standard", "2023-08-20", 500.00),
            (4, "Tech Solutions Ltd", "support@techsol.com", "enterprise", "2023-03-05", 89000.00),
            (5, "Creative Agency", "team@creative.com", "pro", "2023-07-12", 12000.00)
        ]
        
        cursor.executemany(
            "INSERT INTO customers (id, name, email, tier, created_date, total_spent) VALUES (?, ?, ?, ?, ?, ?)",
            sample_customers
        )
        self.conn.commit()
    
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute customer database query"""
        try:
            parts = query.split("|")
            if len(parts) != 2:
                return "Error: Query format should be 'operation|parameter'"
            
            operation, parameter = parts[0].strip(), parts[1].strip()
            cursor = self.conn.cursor()
            
            if operation == "get_customer":
                cursor.execute("SELECT * FROM customers WHERE id = ?", (parameter,))
                result = cursor.fetchone()
                if result:
                    return json.dumps({
                        "id": result[0],
                        "name": result[1],
                        "email": result[2],
                        "tier": result[3],
                        "created_date": result[4],
                        "total_spent": result[5]
                    }, indent=2)
                else:
                    return f"Customer with ID {parameter} not found"
            
            elif operation == "search_customer":
                cursor.execute("SELECT * FROM customers WHERE name LIKE ?", (f"%{parameter}%",))
                results = cursor.fetchall()
                customers = []
                for row in results:
                    customers.append({
                        "id": row[0],
                        "name": row[1],
                        "email": row[2],
                        "tier": row[3],
                        "created_date": row[4],
                        "total_spent": row[5]
                    })
                return json.dumps(customers, indent=2)
            
            elif operation == "list_customers":
                limit = int(parameter) if parameter.isdigit() else 10
                cursor.execute("SELECT * FROM customers LIMIT ?", (limit,))
                results = cursor.fetchall()
                customers = []
                for row in results:
                    customers.append({
                        "id": row[0],
                        "name": row[1],
                        "tier": row[3],
                        "total_spent": row[5]
                    })
                return json.dumps(customers, indent=2)
            
            else:
                return f"Unknown operation: {operation}"
        
        except Exception as e:
            return f"Database error: {str(e)}"


class WeatherTool(BaseTool):
    """Tool for weather information"""
    
    name: str = "weather_info"
    description: str = """
    Get current weather information for a city.
    Input should be the city name (e.g., "Seattle", "London")
    """
    
    def _run(
        self, 
        city: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Get weather information for a city"""
        try:
            # Simulate weather API response
            weather_data = {
                "city": city,
                "temperature": self._get_mock_temperature(city),
                "condition": self._get_mock_condition(city),
                "humidity": self._get_mock_humidity(),
                "wind_speed": "12 mph",
                "forecast": "Partly cloudy with chance of rain",
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return json.dumps(weather_data, indent=2)
            
        except Exception as e:
            return f"Weather service error: {str(e)}"
    
    def _get_mock_temperature(self, city: str) -> str:
        """Generate mock temperature based on city"""
        city_temps = {
            "seattle": "58°F (14°C)",
            "london": "62°F (17°C)",
            "tokyo": "72°F (22°C)",
            "new york": "65°F (18°C)",
            "los angeles": "75°F (24°C)",
            "paris": "68°F (20°C)"
        }
        return city_temps.get(city.lower(), "70°F (21°C)")
    
    def _get_mock_condition(self, city: str) -> str:
        """Generate mock weather condition"""
        conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Light Rain", "Clear"]
        return conditions[hash(city) % len(conditions)]
    
    def _get_mock_humidity(self) -> str:
        """Generate mock humidity"""
        return f"{45 + (hash(str(time.time())) % 40)}%"


# ============================================================================
# 3. FUNCTIONAL TOOLS USING @tool DECORATOR
# ============================================================================

@tool
def calculate_business_metrics(data: str) -> str:
    """
    Calculate business metrics from provided data.
    Input format: "metric_type:value1,value2,value3"
    Supported metrics: revenue, growth, conversion, churn
    """
    try:
        metric_type, values_str = data.split(":")
        values = [float(x.strip()) for x in values_str.split(",")]
        
        if metric_type == "revenue":
            total = sum(values)
            average = total / len(values)
            growth = ((values[-1] - values[0]) / values[0] * 100) if len(values) > 1 else 0
            
            return json.dumps({
                "metric": "revenue",
                "total": total,
                "average": average,
                "growth_percentage": round(growth, 2),
                "trend": "increasing" if growth > 0 else "decreasing"
            }, indent=2)
        
        elif metric_type == "conversion":
            conversion_rate = (values[0] / values[1] * 100) if len(values) >= 2 else 0
            
            return json.dumps({
                "metric": "conversion",
                "conversions": values[0] if len(values) > 0 else 0,
                "visitors": values[1] if len(values) > 1 else 1,
                "conversion_rate": round(conversion_rate, 2),
                "benchmark": "Good" if conversion_rate > 2.5 else "Needs Improvement"
            }, indent=2)
        
        else:
            return f"Metric type '{metric_type}' not supported"
    
    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
async def fetch_stock_price(symbol: str) -> str:
    """
    Fetch current stock price for a given symbol.
    Input should be a stock symbol like 'MSFT', 'AAPL', 'GOOGL'
    """
    try:
        # Simulate stock API call
        mock_prices = {
            "MSFT": {"price": 378.85, "change": "+2.45", "change_percent": "+0.65%"},
            "AAPL": {"price": 189.70, "change": "-1.20", "change_percent": "-0.63%"},
            "GOOGL": {"price": 138.20, "change": "+0.85", "change_percent": "+0.62%"},
            "TSLA": {"price": 248.50, "change": "+5.30", "change_percent": "+2.18%"},
            "AMZN": {"price": 145.80, "change": "-0.90", "change_percent": "-0.61%"}
        }
        
        symbol_upper = symbol.upper()
        if symbol_upper in mock_prices:
            data = mock_prices[symbol_upper]
            return json.dumps({
                "symbol": symbol_upper,
                "current_price": data["price"],
                "change": data["change"],
                "change_percent": data["change_percent"],
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "market": "NASDAQ"
            }, indent=2)
        else:
            return f"Stock symbol '{symbol}' not found"
    
    except Exception as e:
        return f"Stock price error: {str(e)}"


@tool
def generate_report_summary(report_data: str) -> str:
    """
    Generate an executive summary from report data.
    Input should be structured data or text content to summarize.
    """
    try:
        # Simple summarization logic
        lines = report_data.split('\n')
        key_lines = [line.strip() for line in lines if line.strip() and len(line) > 20]
        
        summary = {
            "executive_summary": "Report analysis completed successfully",
            "key_findings": key_lines[:3] if key_lines else ["No significant findings"],
            "total_data_points": len(lines),
            "recommendations": [
                "Continue monitoring key metrics",
                "Consider implementing suggested improvements",
                "Schedule follow-up review in 30 days"
            ],
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return json.dumps(summary, indent=2)
    
    except Exception as e:
        return f"Report generation error: {str(e)}"


# ============================================================================
# 4. AGENT WITH CUSTOM TOOLS
# ============================================================================

def create_business_assistant_agent():
    """Create an agent with custom business tools"""
    print("🔧 Creating Business Assistant Agent...")
    
    # Initialize Azure OpenAI
    llm = create_azure_chat_openai(temperature=0.1)
    
    # Collect all tools
    tools = [
        AzureTextAnalyticsTool(),
        AzureTranslatorTool(),
        CustomerDatabaseTool(),
        WeatherTool(),
        calculate_business_metrics,
        fetch_stock_price,
        generate_report_summary
    ]
    
    # Create agent prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional business assistant with access to various tools.
        
        Available tools allow you to:
        - Analyze text sentiment and extract key phrases
        - Translate text to different languages
        - Query customer database information
        - Get weather information
        - Calculate business metrics
        - Fetch stock prices
        - Generate report summaries
        
        Always use the appropriate tools to provide accurate, helpful information.
        Be professional and thorough in your responses."""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor


# ============================================================================
# 5. DEMONSTRATION FUNCTIONS
# ============================================================================

async def demonstrate_individual_tools():
    """Demonstrate individual tool functionality"""
    print("🛠️ Individual Tool Demonstrations")
    print("=" * 50)
    
    # Text Analytics Tool
    print("\n📊 Azure Text Analytics Tool:")
    text_tool = AzureTextAnalyticsTool()
    result = text_tool._run("I love this new Azure OpenAI service! It's amazing and really helpful.")
    print(f"Analysis: {result}")
    
    # Translation Tool
    print("\n🌐 Azure Translator Tool:")
    translator = AzureTranslatorTool()
    result = translator._run("Hello world|es")
    print(f"Translation: {result}")
    
    # Customer Database Tool
    print("\n👥 Customer Database Tool:")
    db_tool = CustomerDatabaseTool()
    result = db_tool._run("search_customer|Microsoft")
    print(f"Customer Search: {result}")
    
    # Weather Tool
    print("\n🌤️ Weather Tool:")
    weather_tool = WeatherTool()
    result = weather_tool._run("Seattle")
    print(f"Weather: {result}")
    
    # Business Metrics Tool
    print("\n📈 Business Metrics Tool:")
    result = calculate_business_metrics("revenue:10000,12000,15000,18000")
    print(f"Metrics: {result}")
    
    # Stock Price Tool
    print("\n💹 Stock Price Tool:")
    result = await fetch_stock_price("MSFT")
    print(f"Stock: {result}")


async def demonstrate_agent():
    """Demonstrate agent with multiple tools"""
    print("\n🤖 Business Assistant Agent Demo")
    print("=" * 50)
    
    agent = create_business_assistant_agent()
    
    test_queries = [
        "What's the sentiment of this customer feedback: 'The service was okay but could be better'?",
        "Can you find information about Microsoft in our customer database?",
        "What's the current weather in Tokyo?",
        "Calculate the conversion rate if we had 150 conversions from 5000 visitors",
        "What's Microsoft's current stock price?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📋 Query {i}: {query}")
        print("-" * 30)
        
        try:
            result = await agent.ainvoke({"input": query})
            print(f"🎯 Response: {result['output']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")


# ============================================================================
# 6. MAIN DEMONSTRATION
# ============================================================================

async def main():
    """Main demonstration of custom tools"""
    print("🚀 Custom Tools with Azure Integration")
    print("=" * 60)
    
    try:
        # Demonstrate individual tools
        await demonstrate_individual_tools()
        
        # Demonstrate agent with tools
        await demonstrate_agent()
        
        print("\n🎉 All custom tool demos completed!")
        print("\n💡 Key Takeaways:")
        print("✅ Custom tools extend LangChain with business-specific functionality")
        print("✅ Azure integration provides enterprise-grade capabilities")
        print("✅ Type hints and validation ensure robust tool behavior")
        print("✅ Agents can orchestrate multiple tools for complex workflows")
        print("✅ Error handling makes tools production-ready")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("\n🔧 Make sure:")
        print("1. Azure OpenAI is configured properly")
        print("2. Required packages are installed")
        print("3. Network connectivity for external APIs")

if __name__ == "__main__":
    asyncio.run(main())
