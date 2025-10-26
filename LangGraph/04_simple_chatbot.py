"""
Week 1 - Day 6: Simple Chatbot with Basic Error Handling
Building a conversational agent with state
"""
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional

class ChatState(TypedDict):
    messages: List[Dict[str, str]]
    user_input: str
    bot_response: str
    error: Optional[str]
    conversation_count: int

def validate_input(state: ChatState) -> ChatState:
    """Validate user input"""
    print("🔍 Validating user input...")
    
    user_input = state["user_input"]
    
    # Check for empty input
    if not user_input or not user_input.strip():
        state["error"] = "Empty input received"
        return state
    
    # Check for inappropriate length
    if len(user_input) > 500:
        state["error"] = "Input too long (max 500 characters)"
        return state
    
    # Add to message history
    state["messages"].append({"role": "user", "content": user_input})
    state["conversation_count"] += 1
    
    return state

def generate_response(state: ChatState) -> ChatState:
    """Generate bot response (simulated)"""
    if state.get("error"):
        # Skip response generation if there's an error
        return state
    
    print("🤖 Generating response...")
    
    user_input = state["user_input"].lower()
    
    # Simple rule-based responses
    if "hello" in user_input or "hi" in user_input:
        response = "Hello! How can I help you today?"
    elif "how are you" in user_input:
        response = "I'm just a bot, but I'm functioning well! How about you?"
    elif "bye" in user_input or "goodbye" in user_input:
        response = "Goodbye! Feel free to chat again anytime!"
    elif "help" in user_input:
        response = "I can chat with you about simple topics. Try asking about greetings, how I am, or saying goodbye!"
    elif "?" in user_input:
        response = "That's an interesting question! I'm still learning to answer complex questions."
    else:
        response = f"I received your message: '{state['user_input']}'. I'm a simple bot learning to communicate!"
    
    state["bot_response"] = response
    state["messages"].append({"role": "assistant", "content": response})
    
    return state

def handle_error(state: ChatState) -> ChatState:
    """Handle any errors that occurred"""
    if state.get("error"):
        print("⚠️ Handling error...")
        error_message = f"Error: {state['error']}. Please try again."
        state["bot_response"] = error_message
        state["messages"].append({"role": "assistant", "content": error_message})
    
    return state

def build_chatbot_graph():
    """Build the chatbot graph"""
    workflow = StateGraph(ChatState)
    
    # Add nodes
    workflow.add_node("validate", validate_input)
    workflow.add_node("generate", generate_response)
    workflow.add_node("error_handler", handle_error)
    
    # Define flow with conditional routing
    workflow.set_entry_point("validate")
    workflow.add_edge("validate", "generate")
    workflow.add_edge("generate", "error_handler")
    workflow.add_edge("error_handler", END)
    
    return workflow.compile()

def run_chatbot():
    """Run the simple chatbot"""
    print("🤖 Welcome to Simple LangGraph Chatbot!")
    print("=" * 45)
    print("Type 'quit' to exit the conversation")
    print("-" * 45)
    
    app = build_chatbot_graph()
    
    # Initial state
    state = {
        "messages": [
            {"role": "assistant", "content": "Hello! I'm a simple chatbot built with LangGraph. How can I help you?"}
        ],
        "user_input": "",
        "bot_response": "",
        "error": None,
        "conversation_count": 0
    }
    
    # Display welcome message
    print(f"Assistant: {state['messages'][0]['content']}")
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Thank you for chatting! Goodbye!")
                break
            
            # Update state with new input
            state["user_input"] = user_input
            state["error"] = None
            
            # Process through graph
            result = app.invoke(state)
            
            # Display bot response
            print(f"Assistant: {result['bot_response']}")
            
            # Update state for next iteration (keep message history)
            state = {
                "messages": result["messages"],
                "user_input": "",
                "bot_response": "",
                "error": None,
                "conversation_count": result["conversation_count"]
            }
            
            print(f"(Conversation count: {state['conversation_count']})")
            
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            continue

if __name__ == "__main__":
    run_chatbot()