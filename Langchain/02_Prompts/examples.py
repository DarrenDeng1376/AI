"""
Advanced Prompt Engineering Examples - Using Azure OpenAI

This module demonstrates advanced prompt engineering techniques:
1. Structured prompts with output parsers
2. Few-shot learning prompts
3. Chain of thought reasoning
4. Role-based prompts
5. Template composition
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from azure_config import create_azure_chat_openai
from dotenv import load_dotenv

load_dotenv()

# Example 1: Structured Output with Parsers
def structured_output_example():
    """
    Demonstrates how to get structured data from LLM responses
    """
    print("=== Example 1: Structured Output ===")
    
    from langchain.prompts import PromptTemplate
    from langchain.output_parsers import PydanticOutputParser
    from langchain.chains import LLMChain
    from pydantic import BaseModel, Field
    from typing import List
    
    # Define the data structure we want
    class PersonInfo(BaseModel):
        name: str = Field(description="person's full name")
        age: int = Field(description="person's age in years")
        occupation: str = Field(description="person's job or profession")
        hobbies: List[str] = Field(description="list of hobbies or interests")
    
    # Create parser
    parser = PydanticOutputParser(pydantic_object=PersonInfo)
    
    # Create prompt with format instructions
    template = """
    Extract information about the person from the following text.
    
    {format_instructions}
    
    Text: {text}
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    llm = create_azure_chat_openai(temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Test text
    text = """
    Meet Sarah Johnson, a 28-year-old software engineer who works at Google. 
    In her free time, she enjoys hiking, photography, and playing guitar. 
    She also volunteers at local animal shelters on weekends.
    """
    
    # Get structured output
    response = chain.run(text=text)
    parsed_output = parser.parse(response)
    
    print(f"Structured Output:")
    print(f"Name: {parsed_output.name}")
    print(f"Age: {parsed_output.age}")
    print(f"Occupation: {parsed_output.occupation}")
    print(f"Hobbies: {', '.join(parsed_output.hobbies)}")
    print("\n" + "="*50 + "\n")


# Example 2: Few-Shot Learning
def few_shot_learning_example():
    """
    Shows how to use few-shot learning for classification tasks
    """
    print("=== Example 2: Few-Shot Learning ===")
    
    from langchain.prompts import FewShotPromptTemplate, PromptTemplate
    from langchain.chains import LLMChain
    
    # Define examples for sentiment classification
    examples = [
        {
            "text": "I love this product! It works perfectly.",
            "sentiment": "positive"
        },
        {
            "text": "This is the worst purchase I've ever made.",
            "sentiment": "negative"
        },
        {
            "text": "The product is okay, nothing special.",
            "sentiment": "neutral"
        },
        {
            "text": "Amazing quality and fast delivery!",
            "sentiment": "positive"
        }
    ]
    
    # Create example template
    example_template = """
    Text: {text}
    Sentiment: {sentiment}
    """
    
    example_prompt = PromptTemplate(
        input_variables=["text", "sentiment"],
        template=example_template
    )
    
    # Create few-shot prompt
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Classify the sentiment of the following texts as positive, negative, or neutral.\n\n",
        suffix="Text: {input}\nSentiment:",
        input_variables=["input"]
    )
    
    llm = create_azure_chat_openai(temperature=0)
    chain = LLMChain(llm=llm, prompt=few_shot_prompt)
    
    # Test cases
    test_texts = [
        "The service was decent but could be better.",
        "Absolutely fantastic experience!",
        "I want my money back, this is terrible."
    ]
    
    for text in test_texts:
        result = chain.run(input=text)
        print(f"Text: {text}")
        print(f"Sentiment: {result.strip()}")
        print()


# Example 3: Chain of Thought Reasoning
def chain_of_thought_example():
    """
    Demonstrates chain of thought prompting for complex reasoning
    """
    print("=== Example 3: Chain of Thought Reasoning ===")
    
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    
    # Chain of thought template
    template = """
    Solve this step by step, showing your reasoning at each step.
    
    Problem: {problem}
    
    Let me think through this step by step:
    
    Step 1: [Identify what we know and what we need to find]
    Step 2: [Break down the problem into smaller parts]
    Step 3: [Solve each part]
    Step 4: [Combine the results]
    
    Solution:
    """
    
    prompt = PromptTemplate(
        input_variables=["problem"],
        template=template
    )
    
    llm = create_azure_chat_openai(temperature=0.1)
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Test with a math word problem
    problem = """
    A company has 120 employees. 40% work in engineering, 30% work in sales, 
    20% work in marketing, and the rest work in administration. If the company 
    hires 30 new employees and wants to maintain the same proportions, 
    how many new employees should be hired for each department?
    """
    
    result = chain.run(problem=problem)
    print(f"Problem: {problem}")
    print(f"Solution:\n{result}")
    print("\n" + "="*50 + "\n")


# Example 4: Role-Based Prompts
def role_based_prompts_example():
    """
    Shows how to use role-based prompts for different perspectives
    """
    print("=== Example 4: Role-Based Prompts ===")
    
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    
    # Define different expert roles
    roles = {
        "doctor": "You are an experienced medical doctor with 15 years of practice.",
        "lawyer": "You are a senior lawyer specializing in corporate law.",
        "teacher": "You are an elementary school teacher with expertise in child education.",
        "engineer": "You are a senior software engineer with expertise in system architecture."
    }
    
    template = """
    {role}
    
    Please answer the following question from your professional perspective:
    
    Question: {question}
    
    Answer:
    """
    
    prompt = PromptTemplate(
        input_variables=["role", "question"],
        template=template
    )
    
    llm = create_azure_chat_openai(temperature=0.7)
    chain = LLMChain(llm=llm, prompt=prompt)
    
    question = "How can artificial intelligence impact your field in the next 5 years?"
    
    for role_name, role_description in roles.items():
        result = chain.run(role=role_description, question=question)
        print(f"=== {role_name.upper()} PERSPECTIVE ===")
        print(f"{result.strip()}")
        print()


# Example 5: Template Composition
def template_composition_example():
    """
    Demonstrates how to combine multiple templates
    """
    print("=== Example 5: Template Composition ===")
    
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain, SequentialChain
    
    llm = create_azure_chat_openai(temperature=0.7)
    
    # Template 1: Extract key points
    extraction_template = """
    Extract the 3 most important points from this text:
    
    Text: {text}
    
    Key Points:
    1.
    2.
    3.
    """
    
    extraction_prompt = PromptTemplate(
        input_variables=["text"],
        template=extraction_template
    )
    
    extraction_chain = LLMChain(
        llm=llm,
        prompt=extraction_prompt,
        output_key="key_points"
    )
    
    # Template 2: Generate questions
    questions_template = """
    Based on these key points, generate 3 thoughtful questions for further discussion:
    
    Key Points: {key_points}
    
    Discussion Questions:
    1.
    2.
    3.
    """
    
    questions_prompt = PromptTemplate(
        input_variables=["key_points"],
        template=questions_template
    )
    
    questions_chain = LLMChain(
        llm=llm,
        prompt=questions_prompt,
        output_key="questions"
    )
    
    # Template 3: Create action items
    actions_template = """
    Based on the key points and questions, suggest 3 concrete action items:
    
    Key Points: {key_points}
    Questions: {questions}
    
    Action Items:
    1.
    2.
    3.
    """
    
    actions_prompt = PromptTemplate(
        input_variables=["key_points", "questions"],
        template=actions_template
    )
    
    actions_chain = LLMChain(
        llm=llm,
        prompt=actions_prompt,
        output_key="actions"
    )
    
    # Combine all chains
    overall_chain = SequentialChain(
        chains=[extraction_chain, questions_chain, actions_chain],
        input_variables=["text"],
        output_variables=["key_points", "questions", "actions"]
    )
    
    # Test text
    text = """
    Climate change is one of the most pressing issues of our time. Rising global temperatures 
    are causing ice caps to melt, sea levels to rise, and weather patterns to become more extreme. 
    The primary cause is human activity, particularly the burning of fossil fuels which releases 
    greenhouse gases into the atmosphere. Solutions include transitioning to renewable energy, 
    improving energy efficiency, and implementing carbon pricing policies. Individual actions 
    like reducing consumption and supporting sustainable practices also play a crucial role.
    """
    
    result = overall_chain({"text": text})
    
    print("ORIGINAL TEXT:")
    print(text[:200] + "...\n")
    
    print("KEY POINTS:")
    print(result["key_points"])
    print("\nDISCUSSION QUESTIONS:")
    print(result["questions"])
    print("\nACTION ITEMS:")
    print(result["actions"])


if __name__ == "__main__":
    print("Advanced Prompt Engineering Examples with Azure OpenAI")
    print("=" * 60)
    
    try:
        # Test Azure OpenAI connection
        print("🔍 Testing Azure OpenAI connection...")
        test_llm = create_azure_chat_openai()
        test_response = test_llm.invoke("Hello!")
        print(f"✅ Connection successful: {test_response.content}")
        print()
        
        # Run examples (structured_output_example commented out as it requires specific schema parsing)
        # structured_output_example()
        few_shot_learning_example()
        chain_of_thought_example()
        role_based_prompts_example()
        template_composition_example()
        
        print("🎉 All prompt engineering examples completed!")
        print("Next: Explore 03_LLMs/ to learn about different language model configurations")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Configured Azure OpenAI settings in .env file")
        print("2. Installed required packages: pip install pydantic")
        print("3. Your Azure deployment supports the models you're using")
