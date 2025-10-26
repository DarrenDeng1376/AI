"""
Week 1 - Day 1-2: Hello World with LangGraph
Basic setup and simple graph creation
"""
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# Define the state structure
class SimpleState(TypedDict):
    message: str
    steps: List[str]

# Define node functions
def start_node(state: SimpleState) -> SimpleState:
    """First node that initializes the message"""
    print("→ Starting the workflow...")
    state["message"] = "Hello, LangGraph!"
    state["steps"] = ["Started workflow"]
    return state

def process_node(state: SimpleState) -> SimpleState:
    """Process the message"""
    print("→ Processing message...")
    state["message"] = state["message"].upper()
    state["steps"].append("Processed message to uppercase")
    return state

def end_node(state: SimpleState) -> SimpleState:
    """Final node that displays the result"""
    print("→ Finalizing...")
    state["message"] = f"🎉 {state['message']}"
    state["steps"].append("Added emoji decoration")
    return state

def build_hello_world_graph():
    """Build and run the hello world graph"""
    # Create the graph
    workflow = StateGraph(SimpleState)
    
    # Add nodes
    workflow.add_node("start", start_node)
    workflow.add_node("process", process_node)
    workflow.add_node("end", end_node)
    
    # Define the flow
    workflow.set_entry_point("start")
    workflow.add_edge("start", "process")
    workflow.add_edge("process", "end")
    workflow.add_edge("end", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app

def run_hello_world():
    """Run the hello world example"""
    print("🚀 Running Hello World Example")
    print("=" * 40)
    
    app = build_hello_world_graph()
    
    # Initial state
    initial_state = {"message": "", "steps": []}
    
    # Run the graph
    result = app.invoke(initial_state)
    
    print(f"\n📝 Final Message: {result['message']}")
    print(f"📋 Steps taken: {result['steps']}")
    print(f"🔢 Number of steps: {len(result['steps'])}")
    
    return result

if __name__ == "__main__":
    run_hello_world()