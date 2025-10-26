"""
Week 1 - Day 3-4: State Management with TypedDict
Learning state schemas and type-safe states
"""
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Optional

# Complex state example with multiple data types
class UserProfileState(TypedDict):
    user_id: str
    user_name: str
    age: Optional[int]
    preferences: Dict[str, any]
    conversation_history: List[Dict[str, str]]
    current_step: str
    processed: bool

def validate_user_input(state: UserProfileState) -> UserProfileState:
    """Validate and clean user input"""
    print(f"→ Validating user: {state.get('user_name', 'Unknown')}")
    
    # Clean user name
    if "user_name" in state:
        state["user_name"] = state["user_name"].strip().title()
    
    # Set default preferences if not provided
    if "preferences" not in state or not state["preferences"]:
        state["preferences"] = {"theme": "light", "notifications": True}
    
    state["current_step"] = "validation_complete"
    return state

def enrich_user_profile(state: UserProfileState) -> UserProfileState:
    """Add additional information to user profile"""
    print("→ Enriching user profile...")
    
    # Calculate user category based on age
    if state.get("age"):
        if state["age"] < 18:
            state["preferences"]["category"] = "minor"
        elif state["age"] < 65:
            state["preferences"]["category"] = "adult"
        else:
            state["preferences"]["category"] = "senior"
    
    # Add welcome message to history
    welcome_msg = {
        "role": "system", 
        "content": f"Welcome {state['user_name']}! Profile enriched."
    }
    state["conversation_history"].append(welcome_msg)
    
    state["current_step"] = "enrichment_complete"
    return state

def finalize_profile(state: UserProfileState) -> UserProfileState:
    """Final processing and mark as complete"""
    print("→ Finalizing profile...")
    
    state["processed"] = True
    state["current_step"] = "complete"
    
    # Add completion message
    completion_msg = {
        "role": "system",
        "content": f"Profile for {state['user_name']} completed successfully!"
    }
    state["conversation_history"].append(completion_msg)
    
    return state

def build_user_profile_graph():
    """Build user profile processing graph"""
    workflow = StateGraph(UserProfileState)
    
    # Add nodes
    workflow.add_node("validate", validate_user_input)
    workflow.add_node("enrich", enrich_user_profile)
    workflow.add_node("finalize", finalize_profile)
    
    # Define flow
    workflow.set_entry_point("validate")
    workflow.add_edge("validate", "enrich")
    workflow.add_edge("enrich", "finalize")
    workflow.add_edge("finalize", END)
    
    return workflow.compile()

def run_state_management_example():
    """Run the state management example"""
    print("🚀 Running State Management Example")
    print("=" * 50)
    
    app = build_user_profile_graph()
    
    # Test with different user data
    test_users = [
        {
            "user_id": "001",
            "user_name": "  john doe  ",  # Intentional whitespace
            "age": 25,
            "preferences": {"theme": "dark"},
            "conversation_history": [],
            "current_step": "",
            "processed": False
        },
        {
            "user_id": "002", 
            "user_name": "alice",
            "age": 70,
            "preferences": {},
            "conversation_history": [],
            "current_step": "",
            "processed": False
        }
    ]
    
    for i, user_data in enumerate(test_users, 1):
        print(f"\n👤 Processing User {i}: {user_data['user_name']}")
        print("-" * 30)
        
        result = app.invoke(user_data)
        
        print(f"✅ Final State:")
        print(f"   Name: {result['user_name']}")
        print(f"   Category: {result['preferences'].get('category', 'unknown')}")
        print(f"   Processed: {result['processed']}")
        print(f"   Steps in history: {len(result['conversation_history'])}")
        print(f"   Final step: {result['current_step']}")

if __name__ == "__main__":
    run_state_management_example()