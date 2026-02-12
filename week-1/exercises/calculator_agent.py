import json
import math
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# Multiple math functions for calculate
def add(a, b):
    return a + b
def multiply(a, b):
    return a * b
def divide(a, b):
    return a / b
def subtract(a, b):
    return a - b
def power(a, b):
    return a ** b

def square_root(a):
    if a < 0:
        return "Error: Can't do square root for negative numbers"
    return math.sqrt(a)

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a":{"type":"number", "description":"First number"},
                    "b":{"type":"number", "description":"Second number"}
                },
                "required": ["a", "b"]
            }

        }
    },
    {
        "type": "function",
        "function": {
            "name": "subtract",
            "description": "Subtract second number from first",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "divide",
            "description": "Divide first number by second",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "Numerator"},
                    "b": {"type": "number", "description": "Denominator"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "power",
            "description": "Calculate base raised to exponent",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "Base number"},
                    "b": {"type": "number", "description": "Exponent"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "square_root",
            "description": "Calculate square root of a number",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "Number to calculate square root of"}
                },
                "required": ["a"]
            }
        }
    }
]

# Map function names to actual functions
available_functions = {
    "add": add,
    "multiply": multiply,
    "divide": divide,
    "subtract": subtract,
    "power": power,
    "square_root": square_root,
}

def calculator_agent(user_input):
    """Agent that can perform calculations"""
    messages = [{"role": "user", "content": user_input}]

    print(f"\n User: {user_input}")
    print("="*60)

    max_iterations = 5
    iteration =0

    while iteration < max_iterations:
        iteration = iteration + 1

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message
        tool_calls = getattr(response_message, "tool_calls", None)

        # If no tool calls, we have final answer
        if not tool_calls:
            print("No tools available")
            print(f"\n Final answer: {response_message.content}")
            return response_message.content

        #Execute tool calls
        messages.append(response_message)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"\nIteration {iteration}: Calling {function_name}({function_args})")

            # Execute function
            function_to_call = available_functions[function_name]
            function_response = function_to_call(**function_args)

            print(f"\n Successful answer: {function_response}")

            # Add function result to messages
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": str(function_response)
            })

    return "Max iterations reached"

def main():
    """Interactive calculator agent"""
    print("=" * 60)
    print("CALCULATOR AGENT")
    print("=" * 60)
    print("Ask me to perform calculations!")
    print("Type 'quit' to exit\n")


    # Interactive mode
    print("\n\nINTERACTIVE MODE:")
    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ['quit', 'exit']:
            break

        if not user_input:
            continue

        calculator_agent(user_input)

if __name__ == "__main__":
    main()










