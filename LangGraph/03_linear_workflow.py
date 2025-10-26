"""
Week 1 - Day 5: Linear Workflow with 3 Nodes
Building multi-node linear workflows
"""
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import time

class ProcessingState(TypedDict):
    input_data: str
    processed_data: List[str]
    metadata: Dict[str, any]
    execution_time: float

def input_receiver(state: ProcessingState) -> ProcessingState:
    """Receive and validate input data"""
    print("📥 Receiving input data...")
    start_time = time.time()
    
    # Validate input
    if not state["input_data"] or not state["input_data"].strip():
        state["input_data"] = "Default input data"
    
    state["metadata"] = {
        "received_at": time.time(),
        "input_length": len(state["input_data"]),
        "node": "input_receiver"
    }
    
    state["execution_time"] = time.time() - start_time
    return state

def data_processor(state: ProcessingState) -> ProcessingState:
    """Process the input data"""
    print("⚙️ Processing data...")
    start_time = time.time()
    
    input_text = state["input_data"]
    
    # Multiple processing steps
    processing_steps = []
    processing_steps.append(f"Original: {input_text}")
    processing_steps.append(f"Uppercase: {input_text.upper()}")
    processing_steps.append(f"Length: {len(input_text)} characters")
    processing_steps.append(f"Words: {len(input_text.split())} words")
    
    # Add reverse for fun
    processing_steps.append(f"Reversed: {input_text[::-1]}")
    
    state["processed_data"] = processing_steps
    state["metadata"]["processing_completed_at"] = time.time()
    state["metadata"]["node"] = "data_processor"
    state["execution_time"] += time.time() - start_time
    
    return state

def output_generator(state: ProcessingState) -> ProcessingState:
    """Generate final output"""
    print("📤 Generating output...")
    start_time = time.time()
    
    # Create summary
    summary = {
        "total_processing_steps": len(state["processed_data"]),
        "final_output_ready": True,
        "completion_time": time.time()
    }
    
    state["metadata"]["summary"] = summary
    state["metadata"]["node"] = "output_generator"
    state["execution_time"] += time.time() - start_time
    
    return state

def build_linear_workflow():
    """Build a 3-node linear workflow"""
    workflow = StateGraph(ProcessingState)
    
    # Add all nodes
    workflow.add_node("receive_input", input_receiver)
    workflow.add_node("process_data", data_processor)
    workflow.add_node("generate_output", output_generator)
    
    # Linear flow
    workflow.set_entry_point("receive_input")
    workflow.add_edge("receive_input", "process_data")
    workflow.add_edge("process_data", "generate_output")
    workflow.add_edge("generate_output", END)
    
    return workflow.compile()

def run_linear_workflow():
    """Run the linear workflow example"""
    print("🚀 Running 3-Node Linear Workflow Example")
    print("=" * 50)
    
    app = build_linear_workflow()
    
    # Test with different inputs
    test_inputs = [
        "Hello, this is a test message for LangGraph!",
        "Python programming is fun and powerful.",
        "Artificial Intelligence transforms our world.",
        "Short"
    ]
    
    for i, input_text in enumerate(test_inputs, 1):
        print(f"\n🔹 Processing Input {i}: '{input_text}'")
        print("-" * 40)
        
        initial_state = {
            "input_data": input_text,
            "processed_data": [],
            "metadata": {},
            "execution_time": 0.0
        }
        
        result = app.invoke(initial_state)
        
        # Display results
        print(f"✅ Processing completed in {result['execution_time']:.4f} seconds")
        print(f"📊 Processing steps generated: {len(result['processed_data'])}")
        print(f"📋 Sample processed data:")
        for j, step in enumerate(result['processed_data'][:3], 1):
            print(f"   {j}. {step}")
        
        if len(result['processed_data']) > 3:
            print(f"   ... and {len(result['processed_data']) - 3} more steps")

if __name__ == "__main__":
    run_linear_workflow()