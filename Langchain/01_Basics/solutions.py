"""
LangChain Basics - Exercise Solutions

Solutions to the exercises in exercises.py
Try to solve the exercises yourself before looking at these!
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Solution 1: Language Translator Chain
def solution_1_translator():
    """
    Solution: Language Translator Chain
    """
    print("=== Solution 1: Language Translator ===")
    
    from langchain_openai import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    
    llm = OpenAI(temperature=0.3)  # Lower temperature for more consistent translations
    
    # Create prompt template
    template = """
    Translate the following text from English to {target_language}:
    
    Text: {text}
    
    Translation:
    """
    
    prompt = PromptTemplate(
        input_variables=["text", "target_language"],
        template=template
    )
    
    # Create translator chain
    translator = LLMChain(llm=llm, prompt=prompt)
    
    # Test translations
    test_cases = [
        {"text": "Hello, how are you?", "target_language": "Spanish"},
        {"text": "The weather is beautiful today", "target_language": "French"},
        {"text": "I love programming", "target_language": "German"}
    ]
    
    for case in test_cases:
        result = translator.run(**case)
        print(f"Original: {case['text']}")
        print(f"{case['target_language']}: {result.strip()}")
        print()


# Solution 2: Personal Assistant Chain
def solution_2_personal_assistant():
    """
    Solution: Personal Assistant with Memory
    """
    print("=== Solution 2: Personal Assistant ===")
    
    from langchain_openai import OpenAI
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import PromptTemplate
    
    llm = OpenAI(temperature=0.7)
    
    # Custom prompt for personal assistant
    template = """
    You are a helpful personal assistant. Remember information about the user 
    and provide personalized responses based on what you know about them.
    
    Current conversation:
    {history}
    Human: {input}
    AI Assistant:"""
    
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )
    
    # Create conversation with memory
    conversation = ConversationChain(
        llm=llm,
        prompt=prompt,
        memory=ConversationBufferMemory(),
        verbose=True
    )
    
    # Simulate conversation
    responses = [
        "Hi! My name is Sarah and I love reading science fiction novels.",
        "What's my name?",
        "Can you recommend a book based on my interests?",
        "What do you remember about me?"
    ]
    
    for user_input in responses:
        print(f"User: {user_input}")
        response = conversation.predict(input=user_input)
        print(f"Assistant: {response}\n")


# Solution 3: Content Generation Pipeline
def solution_3_content_pipeline():
    """
    Solution: Multi-step Content Generation Pipeline
    """
    print("=== Solution 3: Content Pipeline ===")
    
    from langchain_openai import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain, SequentialChain
    
    llm = OpenAI(temperature=0.7)
    
    # Step 1: Topic Generation
    topic_template = """
    Generate an engaging blog topic about {keyword}.
    Make it specific and interesting for readers.
    
    Topic:"""
    
    topic_prompt = PromptTemplate(
        input_variables=["keyword"],
        template=topic_template
    )
    
    topic_chain = LLMChain(
        llm=llm,
        prompt=topic_prompt,
        output_key="topic"
    )
    
    # Step 2: Outline Generation
    outline_template = """
    Create a detailed outline for this blog topic: {topic}
    
    Include:
    - Introduction
    - 3-4 main sections
    - Conclusion
    
    Outline:"""
    
    outline_prompt = PromptTemplate(
        input_variables=["topic"],
        template=outline_template
    )
    
    outline_chain = LLMChain(
        llm=llm,
        prompt=outline_prompt,
        output_key="outline"
    )
    
    # Step 3: Introduction Writing
    intro_template = """
    Write an engaging introduction paragraph for this blog post:
    
    Topic: {topic}
    Outline: {outline}
    
    Introduction:"""
    
    intro_prompt = PromptTemplate(
        input_variables=["topic", "outline"],
        template=intro_template
    )
    
    intro_chain = LLMChain(
        llm=llm,
        prompt=intro_prompt,
        output_key="introduction"
    )
    
    # Combine all chains
    content_pipeline = SequentialChain(
        chains=[topic_chain, outline_chain, intro_chain],
        input_variables=["keyword"],
        output_variables=["topic", "outline", "introduction"]
    )
    
    # Test the pipeline
    result = content_pipeline({"keyword": "artificial intelligence"})
    
    print(f"Generated Topic: {result['topic']}")
    print(f"\nOutline: {result['outline']}")
    print(f"\nIntroduction: {result['introduction']}")


# Solution 4: Smart Question Answerer
def solution_4_smart_qa():
    """
    Solution: Context-Aware Q&A System
    """
    print("=== Solution 4: Smart Q&A ===")
    
    from langchain_openai import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    
    llm = OpenAI(temperature=0.1)  # Low temperature for factual answers
    
    # QA prompt template
    template = """
    Based on the following context, answer the question. If the answer cannot be found 
    in the context, say "I cannot answer this question based on the given context."
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
    
    qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
    
    context = """
    LangChain is a framework for developing applications powered by language models. 
    It was created to help developers build applications that can connect language models 
    to other sources of data and allow them to interact with their environment. 
    The framework provides tools for prompt management, chains, memory, and agents.
    """
    
    questions = [
        "What is LangChain used for?",
        "Who created LangChain?",  # Not in context
        "What tools does LangChain provide?"
    ]
    
    for question in questions:
        result = qa_chain.run(context=context, question=question)
        print(f"Q: {question}")
        print(f"A: {result.strip()}\n")


# Solution 5: Dynamic Prompt Builder
def solution_5_dynamic_prompts():
    """
    Solution: Dynamic Prompt Selection System
    """
    print("=== Solution 5: Dynamic Prompts ===")
    
    from langchain_openai import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    
    llm = OpenAI(temperature=0.7)
    
    # Define prompt templates for different tasks
    prompt_templates = {
        "summarize": PromptTemplate(
            input_variables=["content"],
            template="Summarize the following text in 2-3 sentences:\n\n{content}\n\nSummary:"
        ),
        "translate": PromptTemplate(
            input_variables=["content", "target"],
            template="Translate the following text to {target}:\n\n{content}\n\nTranslation:"
        ),
        "explain": PromptTemplate(
            input_variables=["content"],
            template="Explain the concept of {content} in simple terms that a beginner can understand:\n\nExplanation:"
        )
    }
    
    def process_task(task_type, **kwargs):
        """Process a task with the appropriate prompt template"""
        if task_type not in prompt_templates:
            return f"Unknown task type: {task_type}"
        
        prompt = prompt_templates[task_type]
        chain = LLMChain(llm=llm, prompt=prompt)
        
        return chain.run(**kwargs)
    
    # Test cases
    tasks = [
        {
            "type": "summarize", 
            "content": "Artificial intelligence is transforming industries across the globe. From healthcare to finance, AI is being used to automate processes, make predictions, and solve complex problems. Machine learning, a subset of AI, allows computers to learn from data without being explicitly programmed."
        },
        {
            "type": "translate", 
            "content": "Hello world", 
            "target": "French"
        },
        {
            "type": "explain", 
            "content": "quantum computing"
        }
    ]
    
    for task in tasks:
        task_type = task.pop("type")
        result = process_task(task_type, **task)
        print(f"Task: {task_type.upper()}")
        print(f"Result: {result.strip()}")
        print("-" * 40)


if __name__ == "__main__":
    print("LangChain Basics - Solutions")
    print("=" * 50)
    
    try:
        solution_1_translator()
        print()
        solution_2_personal_assistant()
        print()
        solution_3_content_pipeline()
        print()
        solution_4_smart_qa()
        print()
        solution_5_dynamic_prompts()
        
        print("🎉 All solutions completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have set your OPENAI_API_KEY in the .env file")
