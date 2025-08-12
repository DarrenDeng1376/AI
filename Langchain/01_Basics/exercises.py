"""
LangChain Basics - Exercises

Complete these exercises to solidify your understanding of LangChain basics.
Try to solve them before looking at the solutions.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Exercise 1: Language Translator Chain
def exercise_1_translator():
    """
    Create a chain that translates text from one language to another.
    
    Requirements:
    - Use a prompt template with variables for 'text' and 'target_language'
    - Create a chain that can translate to any language
    - Test with multiple languages
    
    Example usage:
    translator.run(text="Hello, how are you?", target_language="Spanish")
    """
    print("=== Exercise 1: Language Translator ===")
    
    # YOUR CODE HERE
    # Hint: Use PromptTemplate and LLMChain
    pass


# Exercise 2: Personal Assistant Chain
def exercise_2_personal_assistant():
    """
    Create a personal assistant that remembers user preferences.
    
    Requirements:
    - Ask for the user's name and favorite topics
    - Remember this information in subsequent conversations
    - Provide personalized responses based on stored information
    
    Test the memory by asking questions in sequence.
    """
    print("=== Exercise 2: Personal Assistant ===")
    
    # YOUR CODE HERE
    # Hint: Use ConversationChain with ConversationBufferMemory
    pass


# Exercise 3: Content Generation Pipeline
def exercise_3_content_pipeline():
    """
    Create a multi-step content generation pipeline.
    
    Requirements:
    - Step 1: Generate a blog topic based on a keyword
    - Step 2: Create an outline for the blog post
    - Step 3: Write the introduction paragraph
    - Use SequentialChain to connect all steps
    
    Input: A single keyword
    Output: Topic, outline, and introduction
    """
    print("=== Exercise 3: Content Pipeline ===")
    
    # YOUR CODE HERE
    # Hint: Create multiple LLMChains and combine with SequentialChain
    pass


# Exercise 4: Smart Question Answerer
def exercise_4_smart_qa():
    """
    Create a smart Q&A system with context awareness.
    
    Requirements:
    - Accept a context paragraph and a question
    - Provide answers based strictly on the given context
    - If the answer isn't in the context, say so
    - Format the response nicely
    """
    print("=== Exercise 4: Smart Q&A ===")
    
    context = """
    LangChain is a framework for developing applications powered by language models. 
    It was created to help developers build applications that can connect language models 
    to other sources of data and allow them to interact with their environment. 
    The framework provides tools for prompt management, chains, memory, and agents.
    """
    
    question = "What is LangChain used for?"
    
    # YOUR CODE HERE
    # Hint: Create a prompt template that includes both context and question
    pass


# Exercise 5: Dynamic Prompt Builder
def exercise_5_dynamic_prompts():
    """
    Build a system that creates different prompts based on the task type.
    
    Requirements:
    - Support different task types: "summarize", "translate", "explain"
    - Each task should have a different prompt template
    - Dynamically select the right template based on task type
    - Process the input accordingly
    """
    print("=== Exercise 5: Dynamic Prompts ===")
    
    tasks = [
        {"type": "summarize", "content": "Artificial intelligence is transforming industries..."},
        {"type": "translate", "content": "Hello world", "target": "French"},
        {"type": "explain", "content": "quantum computing"}
    ]
    
    # YOUR CODE HERE
    # Hint: Create a dictionary of prompt templates for different task types
    pass


if __name__ == "__main__":
    print("LangChain Basics - Exercises")
    print("Complete each exercise before checking the solutions!")
    print("=" * 60)
    
    # Uncomment exercises as you work on them
    # exercise_1_translator()
    # exercise_2_personal_assistant()
    # exercise_3_content_pipeline()
    # exercise_4_smart_qa()
    # exercise_5_dynamic_prompts()
    
    print("\nTips:")
    print("1. Start with simple prompt templates")
    print("2. Test each component separately before combining")
    print("3. Use verbose=True to see what's happening")
    print("4. Check the solutions.py file if you get stuck")
