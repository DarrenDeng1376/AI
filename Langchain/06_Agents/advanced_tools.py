"""
Advanced Agent Tools - Specialized Tools for Complex Tasks

This module demonstrates advanced tool creation patterns:
1. Database query tools with safety validation
2. File processing tools with format detection
3. API integration tools with retry logic
4. Data visualization tools with multiple formats
5. Workflow automation tools
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from azure_config import create_azure_chat_openai
from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Dict, Any, List, Optional, Type, Union
import json
import time
import re
from datetime import datetime, timedelta

load_dotenv()

# Example 1: Database Query Tool (Mock Implementation)
class DatabaseQueryInput(BaseModel):
    query_type: str = Field(description="Type of query: select, count, search")
    table: str = Field(description="Table name to query")
    conditions: Optional[Dict[str, Any]] = Field(default={}, description="Query conditions")
    limit: Optional[int] = Field(default=10, description="Maximum number of results")

class DatabaseQueryTool(BaseTool):
    name: str = "database_query"
    description: str = "Query database tables safely with validation"
    args_schema: Type[BaseModel] = DatabaseQueryInput
    
    def get_mock_db(self):
        """Get mock database data"""
        return {
            "users": [
                {"id": 1, "name": "Alice", "email": "alice@example.com", "role": "admin"},
                {"id": 2, "name": "Bob", "email": "bob@example.com", "role": "user"},
                {"id": 3, "name": "Carol", "email": "carol@example.com", "role": "user"}
            ],
            "products": [
                {"id": 1, "name": "Laptop", "price": 1200, "category": "electronics"},
                {"id": 2, "name": "Book", "price": 25, "category": "books"},
                {"id": 3, "name": "Desk", "price": 300, "category": "furniture"}
            ],
            "orders": [
                {"id": 1, "user_id": 1, "product_id": 1, "quantity": 1, "date": "2024-01-15"},
                {"id": 2, "user_id": 2, "product_id": 2, "quantity": 2, "date": "2024-01-16"}
            ]
        }
    
    def _run(self, query_type: str, table: str, conditions: Dict[str, Any] = {}, limit: int = 10) -> str:
        try:
            mock_db = self.get_mock_db()
            # Validate table exists
            if table not in mock_db:
                return f"Error: Table '{table}' not found. Available tables: {list(mock_db.keys())}"
            
            data = mock_db[table]
            
            if query_type == "select":
                # Apply conditions
                filtered_data = data
                for key, value in conditions.items():
                    filtered_data = [row for row in filtered_data if row.get(key) == value]
                
                # Apply limit
                result_data = filtered_data[:limit]
                
                return f"Query Results ({len(result_data)} rows):\n" + \
                       json.dumps(result_data, indent=2)
            
            elif query_type == "count":
                # Apply conditions
                filtered_data = data
                for key, value in conditions.items():
                    filtered_data = [row for row in filtered_data if row.get(key) == value]
                
                return f"Count: {len(filtered_data)} rows match the conditions"
            
            elif query_type == "search":
                # Simple text search across all fields
                search_term = conditions.get("search_term", "")
                if not search_term:
                    return "Error: search_term required for search queries"
                
                search_results = []
                for row in data:
                    for value in row.values():
                        if search_term.lower() in str(value).lower():
                            search_results.append(row)
                            break
                
                result_data = search_results[:limit]
                return f"Search Results for '{search_term}' ({len(result_data)} rows):\n" + \
                       json.dumps(result_data, indent=2)
            
            else:
                return f"Error: Unknown query type '{query_type}'. Supported: select, count, search"
        
        except Exception as e:
            return f"Database query error: {str(e)}"

# Example 2: File Processing Tool
class FileProcessingInput(BaseModel):
    operation: str = Field(description="Operation: read, write, analyze, convert")
    file_path: str = Field(description="Path to the file (for demo purposes)")
    content: Optional[str] = Field(default="", description="Content to write (for write operations)")
    options: Optional[Dict[str, Any]] = Field(default={}, description="Additional options")

class FileProcessingTool(BaseTool):
    name: str = "file_processor"
    description: str = "Process files with format detection and validation"
    args_schema: Type[BaseModel] = FileProcessingInput
    
    def get_mock_files(self):
        """Get mock file system data"""
        return {
            "data.csv": "name,age,city\nAlice,30,New York\nBob,25,San Francisco",
            "report.txt": "This is a sample report with important information about the project.",
            "config.json": '{"database": {"host": "localhost", "port": 5432}, "debug": true}',
            "readme.md": "# Project Title\n\nThis is a sample project with documentation."
        }
    
    def _run(self, operation: str, file_path: str, content: str = "", options: Dict[str, Any] = {}) -> str:
        try:
            mock_files = self.get_mock_files()
            file_name = file_path.split("/")[-1] if "/" in file_path else file_path
            
            if operation == "read":
                if file_name in mock_files:
                    file_content = mock_files[file_name]
                    
                    # Detect file type and provide appropriate analysis
                    if file_name.endswith('.csv'):
                        lines = file_content.split('\n')
                        headers = lines[0].split(',') if lines else []
                        data_rows = len(lines) - 1 if len(lines) > 1 else 0
                        return f"CSV File: {file_name}\nHeaders: {headers}\nData rows: {data_rows}\n\nContent:\n{file_content}"
                    
                    elif file_name.endswith('.json'):
                        try:
                            parsed_json = json.loads(file_content)
                            return f"JSON File: {file_name}\nStructure: {list(parsed_json.keys()) if isinstance(parsed_json, dict) else 'Array'}\n\nContent:\n{json.dumps(parsed_json, indent=2)}"
                        except:
                            return f"JSON File: {file_name}\n\nContent:\n{file_content}"
                    
                    else:
                        word_count = len(file_content.split())
                        line_count = len(file_content.split('\n'))
                        return f"Text File: {file_name}\nWords: {word_count}, Lines: {line_count}\n\nContent:\n{file_content}"
                else:
                    return f"Error: File '{file_name}' not found. Available files: {list(self.mock_files.keys())}"
            
            elif operation == "write":
                if not content:
                    return "Error: Content required for write operation"
                
                self.mock_files[file_name] = content
                return f"Successfully wrote {len(content)} characters to {file_name}"
            
            elif operation == "analyze":
                if file_name in self.mock_files:
                    file_content = self.mock_files[file_name]
                    
                    analysis = {
                        "file_name": file_name,
                        "size_chars": len(file_content),
                        "size_words": len(file_content.split()),
                        "lines": len(file_content.split('\n')),
                        "file_type": file_name.split('.')[-1] if '.' in file_name else "unknown"
                    }
                    
                    # Type-specific analysis
                    if file_name.endswith('.csv'):
                        lines = file_content.split('\n')
                        analysis["csv_columns"] = len(lines[0].split(',')) if lines else 0
                        analysis["csv_rows"] = len(lines) - 1 if len(lines) > 1 else 0
                    
                    elif file_name.endswith('.json'):
                        try:
                            parsed = json.loads(file_content)
                            analysis["json_keys"] = list(parsed.keys()) if isinstance(parsed, dict) else []
                            analysis["json_type"] = "object" if isinstance(parsed, dict) else "array"
                        except:
                            analysis["json_valid"] = False
                    
                    return f"File Analysis:\n{json.dumps(analysis, indent=2)}"
                else:
                    return f"Error: File '{file_name}' not found"
            
            elif operation == "convert":
                if file_name not in self.mock_files:
                    return f"Error: File '{file_name}' not found"
                
                target_format = options.get("target_format", "txt")
                file_content = self.mock_files[file_name]
                
                if target_format == "json" and file_name.endswith('.csv'):
                    # Convert CSV to JSON
                    lines = file_content.split('\n')
                    if len(lines) < 2:
                        return "Error: Invalid CSV format"
                    
                    headers = lines[0].split(',')
                    json_data = []
                    
                    for line in lines[1:]:
                        if line.strip():
                            values = line.split(',')
                            row_dict = {headers[i]: values[i] if i < len(values) else "" for i in range(len(headers))}
                            json_data.append(row_dict)
                    
                    converted_content = json.dumps(json_data, indent=2)
                    new_file_name = file_name.replace('.csv', '.json')
                    self.mock_files[new_file_name] = converted_content
                    
                    return f"Converted {file_name} to {new_file_name}\nResult:\n{converted_content}"
                
                else:
                    return f"Conversion from {file_name.split('.')[-1]} to {target_format} not supported"
            
            else:
                return f"Error: Unknown operation '{operation}'. Supported: read, write, analyze, convert"
        
        except Exception as e:
            return f"File processing error: {str(e)}"

# Example 3: API Integration Tool with Retry Logic
class APICallInput(BaseModel):
    endpoint: str = Field(description="API endpoint name")
    method: str = Field(description="HTTP method: GET, POST, PUT, DELETE")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="API parameters")
    data: Optional[Dict[str, Any]] = Field(default={}, description="Request body data")

class APIIntegrationTool(BaseTool):
    name: str = "api_client"
    description: str = "Make API calls with retry logic and error handling"
    args_schema: Type[BaseModel] = APICallInput
    
    def get_mock_apis(self):
        """Get mock API responses"""
        return {
            "weather": {
                "GET": {"temperature": 22, "condition": "sunny", "humidity": 65}
            },
            "news": {
                "GET": {
                    "articles": [
                        {"title": "AI Breakthrough", "summary": "New developments in AI technology"},
                        {"title": "Tech Update", "summary": "Latest technology trends"}
                    ]
                }
            },
            "users": {
                "GET": {"users": [{"id": 1, "name": "John"}, {"id": 2, "name": "Jane"}]},
                "POST": {"status": "created", "id": 3}
            }
        }
    
    def _run(self, endpoint: str, method: str, parameters: Dict[str, Any] = {}, data: Dict[str, Any] = {}) -> str:
        try:
            mock_apis = self.get_mock_apis()
            # Simulate API call with retry logic
            max_retries = 3
            delay = 1
            
            for attempt in range(max_retries):
                try:
                    # Simulate potential network issues
                    if attempt < 2 and endpoint == "unreliable":
                        raise Exception("Network timeout")
                    
                    # Check if endpoint exists
                    if endpoint not in mock_apis:
                        return f"Error: API endpoint '{endpoint}' not found. Available: {list(mock_apis.keys())}"
                    
                    # Check if method is supported
                    if method not in mock_apis[endpoint]:
                        return f"Error: Method '{method}' not supported for '{endpoint}'. Available: {list(mock_apis[endpoint].keys())}"
                    
                    # Get mock response
                    response_data = mock_apis[endpoint][method].copy()
                    
                    # Apply parameters (mock filtering)
                    if parameters and endpoint == "users" and method == "GET":
                        user_id = parameters.get("id")
                        if user_id:
                            users = response_data["users"]
                            filtered_users = [u for u in users if u["id"] == int(user_id)]
                            response_data = {"users": filtered_users}
                    
                    # Add request metadata
                    result = {
                        "status": "success",
                        "endpoint": endpoint,
                        "method": method,
                        "attempt": attempt + 1,
                        "data": response_data
                    }
                    
                    return f"API Call Successful:\n{json.dumps(result, indent=2)}"
                
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                        continue
                    else:
                        return f"API call failed after {max_retries} attempts: {str(e)}"
        
        except Exception as e:
            return f"API integration error: {str(e)}"

# Example 4: Data Visualization Tool
class DataVisualizationInput(BaseModel):
    data_source: str = Field(description="Source of data: manual, file, or database")
    chart_type: str = Field(description="Type of chart: bar, line, pie, scatter")
    data: Optional[List[Dict[str, Any]]] = Field(default=[], description="Manual data input")
    options: Optional[Dict[str, Any]] = Field(default={}, description="Chart options")

class DataVisualizationTool(BaseTool):
    name: str = "data_visualizer"
    description: str = "Create data visualizations with multiple chart types"
    args_schema: Type[BaseModel] = DataVisualizationInput
    
    def _run(self, data_source: str, chart_type: str, data: List[Dict[str, Any]] = [], options: Dict[str, Any] = {}) -> str:
        try:
            # Get data based on source
            if data_source == "manual":
                chart_data = data
            elif data_source == "file":
                # Mock file data
                chart_data = [
                    {"category": "A", "value": 10},
                    {"category": "B", "value": 15},
                    {"category": "C", "value": 8},
                    {"category": "D", "value": 12}
                ]
            elif data_source == "database":
                # Mock database data
                chart_data = [
                    {"month": "Jan", "sales": 1200},
                    {"month": "Feb", "sales": 1350},
                    {"month": "Mar", "sales": 1100},
                    {"month": "Apr", "sales": 1450}
                ]
            else:
                return f"Error: Unknown data source '{data_source}'. Supported: manual, file, database"
            
            if not chart_data:
                return "Error: No data available for visualization"
            
            # Generate chart description based on type
            title = options.get("title", f"{chart_type.title()} Chart")
            
            if chart_type == "bar":
                chart_desc = f"Bar Chart: {title}\n"
                chart_desc += "Data points:\n"
                for item in chart_data:
                    x_val = item.get("category", item.get("month", "Unknown"))
                    y_val = item.get("value", item.get("sales", 0))
                    chart_desc += f"  {x_val}: {y_val}\n"
            
            elif chart_type == "line":
                chart_desc = f"Line Chart: {title}\n"
                chart_desc += "Trend data:\n"
                for i, item in enumerate(chart_data):
                    x_val = item.get("month", item.get("category", f"Point {i+1}"))
                    y_val = item.get("sales", item.get("value", 0))
                    chart_desc += f"  {x_val}: {y_val}\n"
            
            elif chart_type == "pie":
                total = sum(item.get("value", item.get("sales", 0)) for item in chart_data)
                chart_desc = f"Pie Chart: {title}\n"
                chart_desc += "Percentage breakdown:\n"
                for item in chart_data:
                    label = item.get("category", item.get("month", "Unknown"))
                    value = item.get("value", item.get("sales", 0))
                    percentage = (value / total * 100) if total > 0 else 0
                    chart_desc += f"  {label}: {percentage:.1f}% ({value})\n"
            
            elif chart_type == "scatter":
                chart_desc = f"Scatter Plot: {title}\n"
                chart_desc += "Data points (X, Y):\n"
                for i, item in enumerate(chart_data):
                    x_val = item.get("x", i)
                    y_val = item.get("y", item.get("value", item.get("sales", 0)))
                    chart_desc += f"  ({x_val}, {y_val})\n"
            
            else:
                return f"Error: Unknown chart type '{chart_type}'. Supported: bar, line, pie, scatter"
            
            # Add metadata
            chart_desc += f"\nChart Configuration:\n"
            chart_desc += f"  Data points: {len(chart_data)}\n"
            chart_desc += f"  Chart type: {chart_type}\n"
            chart_desc += f"  Data source: {data_source}\n"
            
            if options:
                chart_desc += f"  Options: {json.dumps(options)}\n"
            
            return chart_desc
        
        except Exception as e:
            return f"Data visualization error: {str(e)}"

# Example 5: Workflow Automation Tool
class WorkflowInput(BaseModel):
    workflow_name: str = Field(description="Name of the workflow to execute")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Workflow parameters")
    steps: Optional[List[str]] = Field(default=[], description="Custom workflow steps")

class WorkflowAutomationTool(BaseTool):
    name: str = "workflow_automation"
    description: str = "Execute predefined workflows and custom automation sequences"
    args_schema: Type[BaseModel] = WorkflowInput
    
    def get_workflows(self):
        """Get predefined workflows"""
        return {
            "data_analysis": [
                "Load data from source",
                "Clean and validate data",
                "Perform statistical analysis",
                "Generate visualizations",
                "Create summary report"
            ],
            "content_creation": [
                "Research topic and gather sources",
                "Create content outline",
                "Write initial draft",
                "Review and edit content",
                "Format and publish"
            ],
            "user_onboarding": [
                "Create user account",
                "Send welcome email",
                "Assign default permissions",
                "Schedule onboarding call",
                "Add to team channels"
            ],
            "report_generation": [
                "Collect data from multiple sources",
                "Process and aggregate information",
                "Apply business rules and calculations",
                "Format results for presentation",
                "Distribute to stakeholders"
            ]
        }
    
    def _run(self, workflow_name: str, parameters: Dict[str, Any] = {}, steps: List[str] = []) -> str:
        try:
            workflows = self.get_workflows()
            # Use custom steps if provided, otherwise use predefined workflow
            if steps:
                workflow_steps = steps
                workflow_name = f"Custom: {workflow_name}"
            elif workflow_name in workflows:
                workflow_steps = workflows[workflow_name]
            else:
                return f"Error: Workflow '{workflow_name}' not found. Available: {list(workflows.keys())}"
            
            # Execute workflow simulation
            execution_log = f"Workflow Execution: {workflow_name}\n"
            execution_log += f"Parameters: {json.dumps(parameters)}\n"
            execution_log += f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            total_steps = len(workflow_steps)
            
            for i, step in enumerate(workflow_steps, 1):
                # Simulate step execution time
                execution_time = 0.5 + (i * 0.1)  # Mock execution time
                
                execution_log += f"Step {i}/{total_steps}: {step}\n"
                execution_log += f"  Status: ✅ Completed\n"
                execution_log += f"  Duration: {execution_time:.1f}s\n"
                
                # Add step-specific details based on parameters
                if "data" in step.lower() and parameters.get("data_source"):
                    execution_log += f"  Data source: {parameters['data_source']}\n"
                elif "email" in step.lower() and parameters.get("email"):
                    execution_log += f"  Email sent to: {parameters['email']}\n"
                elif "report" in step.lower() and parameters.get("format"):
                    execution_log += f"  Report format: {parameters['format']}\n"
                
                execution_log += "\n"
            
            # Workflow summary
            total_time = sum(0.5 + (i * 0.1) for i in range(1, total_steps + 1))
            execution_log += f"Workflow Summary:\n"
            execution_log += f"  Total steps: {total_steps}\n"
            execution_log += f"  Total time: {total_time:.1f}s\n"
            execution_log += f"  Status: ✅ Success\n"
            execution_log += f"  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            return execution_log
        
        except Exception as e:
            return f"Workflow execution error: {str(e)}"

# Demo function to showcase all advanced tools
def advanced_tools_demo():
    """
    Demonstrate all advanced tools with the agent
    """
    print("=== Advanced Agent Tools Demo ===")
    
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate
    
    llm = create_azure_chat_openai(temperature=0.3)
    
    # Create all advanced tools
    tools = [
        DatabaseQueryTool(),
        FileProcessingTool(),
        APIIntegrationTool(),
        DataVisualizationTool(),
        WorkflowAutomationTool()
    ]
    
    # Create agent with advanced tools
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an advanced AI assistant with access to powerful tools for:
        - Database operations and queries
        - File processing and analysis
        - API integrations with retry logic
        - Data visualization and charting
        - Workflow automation and orchestration
        
        Use these tools to help users with complex tasks. Always explain what you're doing and provide detailed results."""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Test queries for advanced tools
    test_queries = [
        "Find all users in the database with admin role",
        "Read and analyze the data.csv file",
        "Get weather information from the API",
        "Create a bar chart showing product sales data",
        "Execute the data analysis workflow with CSV data source"
    ]
    
    print("🚀 Advanced Tools Agent Results:")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🎯 Advanced Query {i}: {query}")
        print("-" * 60)
        
        try:
            result = agent_executor.invoke({"input": query})
            print(f"✅ Advanced Result: {result['output']}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    print("Advanced Agent Tools - Specialized Tools for Complex Tasks")
    print("=" * 70)
    
    try:
        # Test Azure OpenAI connection
        print("🔍 Testing Azure OpenAI connection...")
        test_llm = create_azure_chat_openai()
        test_response = test_llm.invoke("Hello!")
        print(f"✅ Connection successful: {test_response.content[:50]}...")
        print()
        
        # Run advanced tools demo
        advanced_tools_demo()
        
        print("🎉 Advanced agent tools demo completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Configured Azure OpenAI settings in .env file")
        print("2. Valid Azure OpenAI deployments")
        print("3. Sufficient quota in your Azure OpenAI resource")
