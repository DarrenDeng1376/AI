"""
Week 1 - Day 7: Advanced State Manipulation
Complex state operations and transformations
"""
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
import json
from datetime import datetime

class AdvancedState(TypedDict):
    raw_data: Any
    processed_data: Dict[str, Any]
    transformations: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    timestamps: Dict[str, str]

def data_collector(state: AdvancedState) -> AdvancedState:
    """Collect and initialize data"""
    print("📊 Collecting data...")
    
    # Initialize transformations log
    if "transformations" not in state or not state["transformations"]:
        state["transformations"] = []
    
    # Initialize timestamps
    if "timestamps" not in state or not state["timestamps"]:
        state["timestamps"] = {}
    
    state["timestamps"]["collection_start"] = datetime.now().isoformat()
    
    # Log the transformation
    state["transformations"].append({
        "step": "data_collection",
        "action": "initialized_data_structures",
        "timestamp": datetime.now().isoformat()
    })
    
    return state

def data_analyzer(state: AdvancedState) -> AdvancedState:
    """Analyze the raw data"""
    print("🔍 Analyzing data...")
    
    raw_data = state["raw_data"]
    analysis = {}
    
    # Type-based analysis
    data_type = type(raw_data).__name__
    analysis["data_type"] = data_type
    analysis["data_length"] = len(str(raw_data))
    
    # Content analysis based on type
    if isinstance(raw_data, str):
        analysis["word_count"] = len(raw_data.split())
        analysis["character_count"] = len(raw_data)
        analysis["is_uppercase"] = raw_data.isupper()
        
    elif isinstance(raw_data, (list, tuple)):
        analysis["element_count"] = len(raw_data)
        analysis["element_types"] = [type(item).__name__ for item in raw_data]
        
    elif isinstance(raw_data, dict):
        analysis["key_count"] = len(raw_data)
        analysis["keys"] = list(raw_data.keys())
        
    elif isinstance(raw_data, (int, float)):
        analysis["is_numeric"] = True
        analysis["is_positive"] = raw_data > 0
        
    state["processed_data"] = analysis
    
    # Log transformation
    state["transformations"].append({
        "step": "data_analysis",
        "action": f"analyzed_{data_type}",
        "insights": list(analysis.keys()),
        "timestamp": datetime.now().isoformat()
    })
    
    return state

def statistics_calculator(state: AdvancedState) -> AdvancedState:
    """Calculate statistics about the transformations"""
    print("📈 Calculating statistics...")
    
    transformations = state["transformations"]
    
    stats = {
        "total_transformations": len(transformations),
        "transformation_steps": [t["step"] for t in transformations],
        "unique_actions": list(set(t["action"] for t in transformations)),
        "processing_duration_estimate": len(transformations) * 0.1,  # simulated
    }
    
    # Add data-specific stats
    if "processed_data" in state:
        stats.update({
            "analysis_metrics_count": len(state["processed_data"]),
            "data_complexity": "high" if stats["total_transformations"] > 2 else "medium"
        })
    
    state["statistics"] = stats
    
    # Log this transformation too
    state["transformations"].append({
        "step": "statistics_calculation",
        "action": "calculated_workflow_stats",
        "metrics_generated": len(stats),
        "timestamp": datetime.now().isoformat()
    })
    
    return state

def final_reporter(state: AdvancedState) -> AdvancedState:
    """Generate final report"""
    print("📋 Generating final report...")
    
    # Add completion timestamp
    state["timestamps"]["completion"] = datetime.now().isoformat()
    
    # Create summary
    summary = {
        "total_processing_time": "simulated",
        "data_processed_successfully": True,
        "final_data_state": "enriched_with_analysis",
        "transformations_applied": len(state["transformations"])
    }
    
    state["processed_data"]["summary"] = summary
    
    # Final transformation log
    state["transformations"].append({
        "step": "final_report",
        "action": "generated_comprehensive_report",
        "summary_created": True,
        "timestamp": datetime.now().isoformat()
    })
    
    return state

def build_advanced_workflow():
    """Build advanced state manipulation workflow"""
    workflow = StateGraph(AdvancedState)
    
    # Add nodes
    workflow.add_node("collect", data_collector)
    workflow.add_node("analyze", data_analyzer)
    workflow.add_node("calculate_stats", statistics_calculator)
    workflow.add_node("report", final_reporter)
    
    # Define flow
    workflow.set_entry_point("collect")
    workflow.add_edge("collect", "analyze")
    workflow.add_edge("analyze", "calculate_stats")
    workflow.add_edge("calculate_stats", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()

def run_advanced_example():
    """Run the advanced state manipulation example"""
    print("🚀 Running Advanced State Manipulation Example")
    print("=" * 55)
    
    app = build_advanced_workflow()
    
    # Test with different data types
    test_cases = [
        {"name": "String Data", "data": "Hello World! This is a test string for LangGraph."},
        {"name": "List Data", "data": [1, 2, 3, "four", 5.0]},
        {"name": "Dictionary Data", "data": {"name": "John", "age": 30, "city": "New York"}},
        {"name": "Numeric Data", "data": 42},
    ]
    
    for test_case in test_cases:
        print(f"\n🎯 Testing with: {test_case['name']}")
        print("-" * 40)
        
        initial_state = {
            "raw_data": test_case["data"],
            "processed_data": {},
            "transformations": [],
            "statistics": {},
            "timestamps": {}
        }
        
        result = app.invoke(initial_state)
        
        # Display results
        print(f"✅ Processing completed!")
        print(f"📊 Transformations applied: {len(result['transformations'])}")
        print(f"📈 Statistics calculated: {len(result['statistics'])} metrics")
        print(f"🕒 Timestamps recorded: {len(result['timestamps'])}")
        
        # Show sample of processed data
        print(f"\n🔍 Sample Analysis:")
        for key, value in list(result["processed_data"].items())[:3]:
            print(f"   {key}: {value}")
        
        if len(result["transformations"]) > 0:
            print(f"\n🔄 Recent transformations:")
            for transform in result["transformations"][-2:]:
                print(f"   - {transform['step']}: {transform['action']}")

if __name__ == "__main__":
    run_advanced_example()