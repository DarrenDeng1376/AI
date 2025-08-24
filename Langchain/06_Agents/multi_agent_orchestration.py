"""
Multi-Agent Orchestration - Advanced Coordination Patterns

This module demonstrates sophisticated multi-agent patterns:
1. Hierarchical agent management with supervisors
2. Workflow-based agent coordination
3. Load balancing and resource management
4. Error handling and recovery in multi-agent systems
5. Real-time collaboration between agents
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from azure_config import create_azure_chat_openai
from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from typing import Dict, Any, List, Optional
import json
import time
import asyncio
from datetime import datetime
from enum import Enum

load_dotenv()

class AgentStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class AgentOrchestrator:
    """Central orchestrator for managing multiple agents"""
    
    def __init__(self):
        self.agents = {}
        self.task_queue = []
        self.agent_status = {}
        self.performance_metrics = {}
        self.load_balancer = LoadBalancer()
    
    def register_agent(self, agent_id: str, agent: AgentExecutor, capabilities: List[str]):
        """Register an agent with the orchestrator"""
        self.agents[agent_id] = {
            "executor": agent,
            "capabilities": capabilities,
            "load": 0,
            "last_used": datetime.now()
        }
        self.agent_status[agent_id] = AgentStatus.IDLE
        self.performance_metrics[agent_id] = {
            "tasks_completed": 0,
            "avg_response_time": 0,
            "success_rate": 1.0,
            "errors": 0
        }
    
    def assign_task(self, task: Dict[str, Any]) -> Optional[str]:
        """Assign task to the best available agent"""
        required_capability = task.get("capability", "general")
        
        # Find agents with required capability
        suitable_agents = [
            agent_id for agent_id, agent_info in self.agents.items()
            if required_capability in agent_info["capabilities"] 
            and self.agent_status[agent_id] == AgentStatus.IDLE
        ]
        
        if not suitable_agents:
            return None
        
        # Use load balancer to select best agent
        selected_agent = self.load_balancer.select_agent(suitable_agents, self.agents, task)
        
        # Update agent status
        self.agent_status[selected_agent] = AgentStatus.BUSY
        self.agents[selected_agent]["load"] += 1
        self.agents[selected_agent]["last_used"] = datetime.now()
        
        return selected_agent
    
    def complete_task(self, agent_id: str, success: bool, response_time: float):
        """Mark task as completed and update metrics"""
        self.agent_status[agent_id] = AgentStatus.IDLE
        self.agents[agent_id]["load"] -= 1
        
        metrics = self.performance_metrics[agent_id]
        metrics["tasks_completed"] += 1
        
        # Update average response time
        current_avg = metrics["avg_response_time"]
        task_count = metrics["tasks_completed"]
        metrics["avg_response_time"] = ((current_avg * (task_count - 1)) + response_time) / task_count
        
        # Update success rate
        if success:
            metrics["success_rate"] = (metrics["success_rate"] * (task_count - 1) + 1.0) / task_count
        else:
            metrics["errors"] += 1
            metrics["success_rate"] = (metrics["success_rate"] * (task_count - 1)) / task_count
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report of all agents"""
        return {
            "agents": {
                agent_id: {
                    "status": self.agent_status[agent_id].value,
                    "load": self.agents[agent_id]["load"],
                    "capabilities": self.agents[agent_id]["capabilities"],
                    "metrics": self.performance_metrics[agent_id]
                }
                for agent_id in self.agents
            },
            "total_agents": len(self.agents),
            "active_agents": sum(1 for status in self.agent_status.values() if status != AgentStatus.OFFLINE),
            "total_tasks_completed": sum(metrics["tasks_completed"] for metrics in self.performance_metrics.values())
        }

class LoadBalancer:
    """Load balancer for distributing tasks across agents"""
    
    def select_agent(self, available_agents: List[str], agents_info: Dict, task: Dict[str, Any]) -> str:
        """Select the best agent based on load balancing strategy"""
        strategy = task.get("strategy", "round_robin")
        
        if strategy == "least_loaded":
            return min(available_agents, key=lambda agent_id: agents_info[agent_id]["load"])
        
        elif strategy == "performance":
            # Select based on success rate and response time
            def score_agent(agent_id):
                metrics = agents_info[agent_id]
                # Higher success rate and lower response time = better score
                return metrics.get("success_rate", 1.0) / (metrics.get("avg_response_time", 1.0) + 1)
            
            return max(available_agents, key=score_agent)
        
        else:  # round_robin (default)
            # Simple round-robin based on last used time
            return min(available_agents, key=lambda agent_id: agents_info[agent_id]["last_used"])

# Example 1: Hierarchical Agent System
def create_hierarchical_system():
    """Create a hierarchical system with supervisor and worker agents"""
    print("=== Example 1: Hierarchical Agent System ===")
    
    # Create different types of agents
    llm = create_azure_chat_openai(temperature=0.3)
    orchestrator = AgentOrchestrator()
    
    # Supervisor Agent Tools
    @tool
    def delegate_task(task_description: str, priority: str = "medium", agent_type: str = "general") -> str:
        """Delegate a task to appropriate worker agent and execute it"""
        # Create task object
        task = {
            "description": task_description,
            "priority": getattr(TaskPriority, priority.upper(), TaskPriority.MEDIUM),
            "capability": agent_type,
            "timestamp": datetime.now().isoformat()
        }
        
        # Find and assign agent
        selected_agent = orchestrator.assign_task(task)
        
        if not selected_agent:
            return f"❌ No available agent for task: {task_description}"
        
        # Execute task on selected agent
        try:
            start_time = time.time()
            agent_executor = orchestrator.agents[selected_agent]["executor"]
            
            # Create appropriate input for the agent based on type
            if agent_type == "research":
                agent_input = f"Please research: {task_description}"
            elif agent_type == "analysis":
                agent_input = f"Please analyze: {task_description}"
            elif agent_type == "writing":
                agent_input = f"Please write: {task_description}"
            else:
                agent_input = task_description
            
            # Execute the task
            result = agent_executor.invoke({"input": agent_input})
            response_time = time.time() - start_time
            
            # Mark task as completed
            orchestrator.complete_task(selected_agent, True, response_time)
            
            return f"✅ Task completed by {selected_agent}:\n{result['output']}"
            
        except Exception as e:
            orchestrator.complete_task(selected_agent, False, time.time() - start_time)
            return f"❌ Error executing task on {selected_agent}: {str(e)}"
    
    @tool
    def monitor_agents() -> str:
        """Monitor status and performance of all agents"""
        status_report = orchestrator.get_status_report()
        return f"Agent Status Report:\n{json.dumps(status_report, indent=2)}"
    
    # Worker Agent Tools
    @tool
    def research_task(topic: str) -> str:
        """Perform research on a given topic"""
        time.sleep(1)  # Simulate work
        return f"Research completed on '{topic}': Found 5 relevant sources and key insights."
    
    @tool
    def analysis_task(data: str) -> str:
        """Perform data analysis"""
        time.sleep(1.5)  # Simulate work
        return f"Analysis completed on data: Identified trends and generated 3 key recommendations."
    
    @tool
    def writing_task(content_type: str, topic: str) -> str:
        """Create written content"""
        time.sleep(2)  # Simulate work
        return f"Created {content_type} about '{topic}': 500 words with proper structure and citations."
    
    # Create Supervisor Agent
    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a supervisor agent responsible for task delegation and monitoring.

Your role is to:
1. Receive requests from users
2. Determine which type of agent can best handle the task
3. Delegate tasks using the delegate_task tool
4. Monitor system status when requested

Available agent types:
- research: For gathering information, investigating topics, finding sources
- analysis: For analyzing data, finding patterns, making calculations  
- writing: For creating content, reports, summaries, documentation
- general: For basic tasks that don't require specialization

When delegating tasks:
- Use delegate_task(task_description, priority, agent_type)
- Priority can be: "low", "medium", "high", "critical"
- Be specific about what the agent should do
- Choose the most appropriate agent type

For monitoring requests, use the monitor_agents tool."""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    supervisor_tools = [delegate_task, monitor_agents]
    supervisor_agent = create_tool_calling_agent(llm, supervisor_tools, supervisor_prompt)
    supervisor_executor = AgentExecutor(agent=supervisor_agent, tools=supervisor_tools, verbose=False)
    
    # Create Worker Agents
    research_tools = [research_task]
    research_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research specialist. Focus on gathering comprehensive information."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    research_agent = create_tool_calling_agent(llm, research_tools, research_prompt)
    research_executor = AgentExecutor(agent=research_agent, tools=research_tools, verbose=False)
    
    analysis_tools = [analysis_task]
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a data analyst. Focus on finding patterns and insights."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    analysis_agent = create_tool_calling_agent(llm, analysis_tools, analysis_prompt)
    analysis_executor = AgentExecutor(agent=analysis_agent, tools=analysis_tools, verbose=False)
    
    writing_tools = [writing_task]
    writing_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a content writer. Focus on creating clear, engaging content."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    writing_agent = create_tool_calling_agent(llm, writing_tools, writing_prompt)
    writing_executor = AgentExecutor(agent=writing_agent, tools=writing_tools, verbose=False)
    
    # Register agents with orchestrator
    orchestrator.register_agent("researcher", research_executor, ["research", "general"])
    orchestrator.register_agent("analyst", analysis_executor, ["analysis", "general"])
    orchestrator.register_agent("writer", writing_executor, ["writing", "general"])
    
    # Test hierarchical system
    test_requests = [
        "Research artificial intelligence trends for 2024",
        "Analyze sales performance data from last quarter", 
        "Write a summary of our project findings",
        "Show me the current status of all agents"
    ]
    
    print("🏢 Hierarchical Agent System Results:")
    
    for i, request in enumerate(test_requests, 1):
        print(f"\n📋 Request {i}: {request}")
        print("-" * 50)
        
        try:
            result = supervisor_executor.invoke({"input": request})
            print(f"🎯 Supervisor Response: {result['output']}")
            
            # Note: Task completion is now handled automatically in the delegate_task tool
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Show final status
    print(f"\n📊 Final System Status:")
    status = orchestrator.get_status_report()
    print(json.dumps(status, indent=2))
    
    print("\n" + "="*60 + "\n")

# Example 2: Workflow-Based Coordination
def create_workflow_system():
    """Create a workflow-based agent coordination system"""
    print("=== Example 2: Workflow-Based Agent Coordination ===")
    
    class WorkflowEngine:
        def __init__(self):
            self.workflows = {}
            self.active_workflows = {}
        
        def define_workflow(self, workflow_id: str, steps: List[Dict[str, Any]]):
            """Define a new workflow"""
            self.workflows[workflow_id] = {
                "steps": steps,
                "created": datetime.now(),
                "executions": 0
            }
        
        def execute_workflow(self, workflow_id: str, initial_data: Dict[str, Any]) -> Dict[str, Any]:
            """Execute a workflow"""
            if workflow_id not in self.workflows:
                return {"error": f"Workflow {workflow_id} not found"}
            
            workflow = self.workflows[workflow_id]
            steps = workflow["steps"]
            
            execution_id = f"{workflow_id}_{int(time.time())}"
            self.active_workflows[execution_id] = {
                "workflow_id": workflow_id,
                "status": "running",
                "current_step": 0,
                "data": initial_data,
                "started": datetime.now(),
                "results": []
            }
            
            execution = self.active_workflows[execution_id]
            
            try:
                for i, step in enumerate(steps):
                    execution["current_step"] = i
                    
                    step_result = self._execute_step(step, execution["data"])
                    execution["results"].append(step_result)
                    
                    # Update data for next step
                    if "output" in step_result:
                        execution["data"].update(step_result["output"])
                
                execution["status"] = "completed"
                workflow["executions"] += 1
                
                return {
                    "execution_id": execution_id,
                    "status": "success",
                    "results": execution["results"],
                    "final_data": execution["data"]
                }
            
            except Exception as e:
                execution["status"] = "failed"
                execution["error"] = str(e)
                return {"execution_id": execution_id, "status": "failed", "error": str(e)}
        
        def _execute_step(self, step: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
            """Execute a single workflow step"""
            step_type = step["type"]
            step_name = step["name"]
            
            print(f"  📍 Executing step: {step_name}")
            
            if step_type == "agent_task":
                agent_id = step["agent"]
                task = step["task"]
                
                # Simulate agent execution
                time.sleep(step.get("duration", 1))
                
                if agent_id == "researcher":
                    result = f"Research completed: {task}"
                elif agent_id == "analyst":
                    result = f"Analysis completed: {task}"
                elif agent_id == "writer":
                    result = f"Content created: {task}"
                else:
                    result = f"Task completed by {agent_id}: {task}"
                
                return {
                    "step": step_name,
                    "agent": agent_id,
                    "result": result,
                    "output": {f"{step_name}_result": result}
                }
            
            elif step_type == "decision":
                condition = step["condition"]
                # Simple condition evaluation
                if condition in data and data[condition]:
                    return {"step": step_name, "decision": "true", "output": {"decision_result": True}}
                else:
                    return {"step": step_name, "decision": "false", "output": {"decision_result": False}}
            
            elif step_type == "aggregation":
                # Aggregate results from previous steps
                results = [result for result in data.values() if isinstance(result, str)]
                aggregated = f"Aggregated {len(results)} results: " + "; ".join(results[:3])
                return {"step": step_name, "aggregation": aggregated, "output": {"final_result": aggregated}}
            
            else:
                return {"step": step_name, "error": f"Unknown step type: {step_type}"}
    
    # Create workflow engine
    workflow_engine = WorkflowEngine()
    
    # Define sample workflows
    content_creation_workflow = [
        {
            "name": "research_phase",
            "type": "agent_task",
            "agent": "researcher",
            "task": "Gather information on the topic",
            "duration": 2
        },
        {
            "name": "analysis_phase",
            "type": "agent_task",
            "agent": "analyst",
            "task": "Analyze research findings",
            "duration": 1.5
        },
        {
            "name": "writing_phase",
            "type": "agent_task",
            "agent": "writer",
            "task": "Create content based on analysis",
            "duration": 3
        },
        {
            "name": "final_review",
            "type": "aggregation",
            "inputs": ["research_phase", "analysis_phase", "writing_phase"]
        }
    ]
    
    workflow_engine.define_workflow("content_creation", content_creation_workflow)
    
    data_processing_workflow = [
        {
            "name": "data_validation",
            "type": "agent_task",
            "agent": "analyst",
            "task": "Validate input data quality",
            "duration": 1
        },
        {
            "name": "quality_check",
            "type": "decision",
            "condition": "data_validation_result"
        },
        {
            "name": "main_analysis",
            "type": "agent_task",
            "agent": "analyst",
            "task": "Perform main data analysis",
            "duration": 2
        },
        {
            "name": "report_generation",
            "type": "agent_task",
            "agent": "writer",
            "task": "Generate analysis report",
            "duration": 1.5
        }
    ]
    
    workflow_engine.define_workflow("data_processing", data_processing_workflow)
    
    # Test workflow execution
    test_workflows = [
        {
            "id": "content_creation",
            "data": {"topic": "Machine Learning", "target_audience": "beginners"}
        },
        {
            "id": "data_processing", 
            "data": {"dataset": "sales_data.csv", "data_validation_result": "Valid data format"}
        }
    ]
    
    print("🔄 Workflow-Based Coordination Results:")
    
    for workflow_test in test_workflows:
        workflow_id = workflow_test["id"]
        initial_data = workflow_test["data"]
        
        print(f"\n🚀 Executing Workflow: {workflow_id}")
        print(f"Initial Data: {json.dumps(initial_data)}")
        print("-" * 50)
        
        try:
            result = workflow_engine.execute_workflow(workflow_id, initial_data)
            
            if result["status"] == "success":
                print(f"✅ Workflow completed successfully!")
                print(f"Final Result: {result['final_data'].get('final_result', 'No final result')}")
            else:
                print(f"❌ Workflow failed: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            print(f"❌ Workflow execution error: {e}")
    
    print("\n" + "="*60 + "\n")

# Example 3: Real-time Agent Collaboration
async def create_realtime_collaboration():
    """Create a real-time collaboration system between agents using actual LangChain agents"""
    print("=== Example 3: Real-time Agent Collaboration ===")
    
    # Create LangChain agents for collaboration
    llm = create_azure_chat_openai(temperature=0.7)
    
    # Collaboration tools
    @tool
    def share_findings(finding_type: str, content: str, priority: str = "medium") -> str:
        """Share research findings or analysis results with the team"""
        return f"Shared {finding_type}: {content} (Priority: {priority})"
    
    @tool
    def request_assistance(task: str, required_expertise: str, urgency: str = "normal") -> str:
        """Request assistance from team members with specific expertise"""
        return f"Assistance requested for '{task}' requiring {required_expertise} expertise (Urgency: {urgency})"
    
    @tool
    def update_project_status(task: str, status: str, progress_percentage: int, notes: str = "") -> str:
        """Update the status of a project task"""
        return f"Task '{task}' updated to {status} ({progress_percentage}% complete). Notes: {notes}"
    
    @tool
    def collaborate_on_solution(problem: str, proposed_solution: str, feedback_request: str) -> str:
        """Propose a solution and request feedback from team members"""
        return f"Solution proposed for '{problem}': {proposed_solution}. Feedback needed on: {feedback_request}"
    
    # Create specialized agents
    collaboration_tools = [share_findings, request_assistance, update_project_status, collaborate_on_solution]
    
    # Data Analyst Agent
    analyst_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Senior Data Analyst in a collaborative team environment.

Your expertise includes:
- Data collection, cleaning, and validation
- Statistical analysis and pattern recognition
- Data visualization and reporting
- Customer segmentation and behavior analysis

In collaboration:
- Share your findings using share_findings tool
- Request help when you need domain expertise using request_assistance tool
- Update your task progress using update_project_status tool
- Propose data-driven solutions using collaborate_on_solution tool

Be proactive in sharing insights and asking for clarification when needed.
Keep your responses focused and data-driven."""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    analyst_agent = create_tool_calling_agent(llm, collaboration_tools, analyst_prompt)
    analyst_executor = AgentExecutor(agent=analyst_agent, tools=collaboration_tools, verbose=False)
    
    # Business Expert Agent
    business_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Business Strategy Expert in a collaborative team environment.

Your expertise includes:
- Market analysis and business strategy
- Customer requirements and business objectives
- ROI analysis and business case development
- Stakeholder management and communication

In collaboration:
- Share business insights using share_findings tool
- Request technical or analytical support using request_assistance tool
- Update business-related task progress using update_project_status tool
- Propose business solutions using collaborate_on_solution tool

Focus on business value, strategic alignment, and practical implementation.
Always consider the business impact of technical decisions."""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    business_agent = create_tool_calling_agent(llm, collaboration_tools, business_prompt)
    business_executor = AgentExecutor(agent=business_agent, tools=collaboration_tools, verbose=False)
    
    # Technical Lead Agent  
    technical_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Technical Lead in a collaborative team environment.

Your expertise includes:
- System architecture and technical design
- Infrastructure setup and optimization
- Technology selection and implementation
- Technical risk assessment and mitigation

In collaboration:
- Share technical insights using share_findings tool
- Request business or analytical input using request_assistance tool
- Update technical task progress using update_project_status tool
- Propose technical solutions using collaborate_on_solution tool

Focus on scalability, reliability, and technical feasibility.
Consider both current needs and future growth."""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    technical_agent = create_tool_calling_agent(llm, collaboration_tools, technical_prompt)
    technical_executor = AgentExecutor(agent=technical_agent, tools=collaboration_tools, verbose=False)
    
    # Collaboration Hub with real agent integration
    class RealTimeCollaborationHub:
        def __init__(self):
            self.agents = {
                "data_analyst": analyst_executor,
                "business_expert": business_executor, 
                "technical_lead": technical_executor
            }
            self.conversation_history = []
            self.shared_workspace = {}
            
        async def agent_response(self, agent_name: str, message: str, context: str = "") -> str:
            """Get response from a specific agent"""
            if agent_name not in self.agents:
                return f"Agent {agent_name} not found"
            
            agent = self.agents[agent_name]
            
            # Build input with context from conversation history
            full_input = f"{context}\n\n{message}" if context else message
            
            try:
                response = agent.invoke({"input": full_input})
                return response["output"]
            except Exception as e:
                return f"Error from {agent_name}: {str(e)}"
        
        def add_to_history(self, agent_name: str, message: str, response: str):
            """Add interaction to conversation history"""
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "agent": agent_name,
                "input": message,
                "output": response
            })
        
        def get_conversation_context(self, last_n: int = 3) -> str:
            """Get recent conversation context"""
            if not self.conversation_history:
                return ""
            
            recent = self.conversation_history[-last_n:]
            context_parts = []
            
            for entry in recent:
                context_parts.append(f"{entry['agent']}: {entry['output']}")
            
            return "Recent team conversation:\n" + "\n".join(context_parts)
    
    # Initialize collaboration
    hub = RealTimeCollaborationHub()
    
    print("🤝 Real-time Agent Collaboration with LangChain:")
    print("Participants: Data Analyst, Business Expert, Technical Lead")
    print("-" * 60)
    
    # Collaboration scenario: Q4 Performance Analysis Project
    collaboration_steps = [
        {
            "agent": "business_expert",
            "message": "We need to analyze Q4 performance data to identify growth opportunities for 2024. The board wants insights on customer segments and revenue optimization."
        },
        {
            "agent": "data_analyst", 
            "message": "I'll analyze the Q4 data. What specific metrics should I focus on, and do we have access to customer behavior data?"
        },
        {
            "agent": "technical_lead",
            "message": "I can set up the data pipeline and analysis infrastructure. What's our timeline and what tools should we prioritize?"
        },
        {
            "agent": "business_expert",
            "message": "Timeline is 2 weeks. Focus on customer lifetime value, churn prediction, and segment profitability. We need actionable recommendations."
        },
        {
            "agent": "data_analyst",
            "message": "I've found 3 distinct customer segments with different value patterns. The high-value segment shows declining engagement. Should I dive deeper?"
        },
        {
            "agent": "technical_lead",
            "message": "I'll create a real-time monitoring dashboard for these segments. What visualization requirements do we have for the board presentation?"
        }
    ]
    
    # Execute real collaboration
    for i, step in enumerate(collaboration_steps, 1):
        agent_name = step["agent"]
        message = step["message"]
        
        print(f"\n🎯 Step {i}: {agent_name.replace('_', ' ').title()}")
        print(f"💬 Input: {message}")
        print("-" * 40)
        
        # Get context from previous conversations
        context = hub.get_conversation_context() if i > 1 else ""
        
        # Get agent response
        response = await hub.agent_response(agent_name, message, context)
        
        # Add to history
        hub.add_to_history(agent_name, message, response)
        
        print(f"🤖 Agent Response: {response}")
        
        # Simulate brief pause for real-time feel
        await asyncio.sleep(0.5)
    
    # Show collaboration summary
    print(f"\n📊 Collaboration Summary:")
    print(f"Total interactions: {len(hub.conversation_history)}")
    print(f"Agents participated: {len(set(entry['agent'] for entry in hub.conversation_history))}")
    
    # Show final conversation history
    print(f"\n� Complete Conversation History:")
    for entry in hub.conversation_history:
        agent_name = entry['agent'].replace('_', ' ').title()
        print(f"\n{agent_name}: {entry['output'][:100]}...")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    print("Multi-Agent Orchestration - Advanced Coordination Patterns")
    print("=" * 70)
    
    try:
        # Test Azure OpenAI connection
        print("🔍 Testing Azure OpenAI connection...")
        test_llm = create_azure_chat_openai()
        test_response = test_llm.invoke("Hello!")
        print(f"✅ Connection successful: {test_response.content[:50]}...")
        print()
        
        # Run orchestration examples
        # create_hierarchical_system()
        # create_workflow_system()
        
        # Run async collaboration example
        print("🔄 Running async collaboration example...")
        asyncio.run(create_realtime_collaboration())
        
        print("🎉 Multi-agent orchestration examples completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Configured Azure OpenAI settings in .env file")
        print("2. Valid Azure OpenAI deployments")
        print("3. Sufficient quota in your Azure OpenAI resource")
