import json

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# Define a simple function
def get_weather(location, unit="celsius"):
    """Get weather for a location (mock function)"""
    # In real app, this would call a weather API
    weather_data = {
        "location": location,
        "temperature": 22,
        "unit": unit,
        "condition": "Sunny"
    }
    return json.dumps(weather_data)

# Define function schema for OpenAI
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. London"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

def run_conversation():
    user_input = input("Please enter a location: ")
    messages = [
        {"role": "user", "content": f"what is the weather in {user_input}? "}
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto" #Let model decide when to call
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    print("Step 1: Model's response")
    print(f"Tool calls: {tool_calls}\n")

    # Step 2: Execute the function if model wants to call it
    if tool_calls:
        print("Response message", str(response_message))
        messages.append(response_message)  # Add assistant's response

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"Step 2: Calling function: {function_name}")
            print(f"Arguments: {function_args}\n")

            # Call the actual function
            if function_name == "get_weather":
                function_response = get_weather(
                    location=function_args.get("location"),
                    unit=function_args.get("unit", "celsius")
                )

            # Add function response to messages
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response
            })

        # Step 3: Get final response from model
        print("Step 3: Getting final response\n")
        second_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        final_answer = second_response.choices[0].message.content
        print(f"Final Answer: {final_answer}")
        return final_answer

    return response_message.content


if __name__ == "__main__":
    run_conversation()