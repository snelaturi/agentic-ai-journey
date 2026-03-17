from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
import requests
from datetime import datetime
import json

load_dotenv()

print("=" * 70)
print(" " * 20 + "CUSTOM TOOLS BASICS")
print("=" * 70)


# ===== TOOL 1: Database Query Tool (Mock) =====
@tool
def query_database(query: str) -> str:
    """
    Execute a database query and return results.
    Use this to fetch data from the database.

    Args:
        query: SQL query or data request description
    """
    # Mock database
    mock_db = {
        "users": [
            {"id": 1, "name": "John Doe", "email": "john@example.com", "role": "admin"},
            {"id": 2, "name": "Jane Smith", "email": "jane@example.com", "role": "user"},
            {"id": 3, "name": "Bob Johnson", "email": "bob@example.com", "role": "user"}
        ],
        "orders": [
            {"order_id": 101, "user_id": 1, "product": "Laptop", "amount": 1200},
            {"order_id": 102, "user_id": 2, "product": "Mouse", "amount": 25},
            {"order_id": 103, "user_id": 1, "product": "Keyboard", "amount": 75}
        ]
    }

    # Simple query parsing
    query_lower = query.lower()

    if "user" in query_lower:
        return json.dumps(mock_db["users"], indent=2)
    elif "order" in query_lower:
        return json.dumps(mock_db["orders"], indent=2)
    else:
        return json.dumps({"error": "Table not found"})


# ===== TOOL 2: File System Tool =====
@tool
def read_file(filename: str) -> str:
    """
    Read contents of a file.

    Args:
        filename: Name of file to read
    """
    try:
        with open(filename, 'r') as f:
            content = f.read()
        return f"File contents:\n{content}"
    except FileNotFoundError:
        return f"Error: File '{filename}' not found"
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def write_file(filename: str, content: str) -> str:
    """
    Write content to a file.

    Args:
        filename: Name of file to write to
        content: Content to write
    """
    try:
        with open(filename, 'w') as f:
            f.write(content)
        return f"✓ Successfully wrote to {filename}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


# ===== TOOL 3: API Integration Tool =====
@tool
def get_weather(city: str) -> str:
    """
    Get current weather for a city.

    Args:
        city: City name (e.g., 'London', 'New York')
    """
    # Mock weather data (replace with real API if you have key)
    weather_data = {
        "london": {"temp": 15, "condition": "Cloudy", "humidity": 70},
        "new york": {"temp": 22, "condition": "Sunny", "humidity": 50},
        "tokyo": {"temp": 18, "condition": "Rainy", "humidity": 80}
    }

    city_lower = city.lower()
    if city_lower in weather_data:
        data = weather_data[city_lower]
        return json.dumps({
            "city": city,
            "temperature": f"{data['temp']}°C",
            "condition": data["condition"],
            "humidity": f"{data['humidity']}%"
        }, indent=2)
    else:
        return json.dumps({
            "city": city,
            "temperature": "20°C",
            "condition": "Unknown",
            "note": "Mock data for unknown city"
        })


# ===== TOOL 4: Data Processing Tool =====
@tool
def process_data(operation: str, data: str) -> str:
    """
    Process data with various operations.

    Args:
        operation: Operation to perform (sum, average, count, filter)
        data: Comma-separated numbers or JSON data
    """
    try:
        if operation == "sum":
            numbers = [float(x.strip()) for x in data.split(",")]
            result = sum(numbers)
            return f"Sum: {result}"

        elif operation == "average":
            numbers = [float(x.strip()) for x in data.split(",")]
            result = sum(numbers) / len(numbers)
            return f"Average: {result:.2f}"

        elif operation == "count":
            items = [x.strip() for x in data.split(",")]
            return f"Count: {len(items)}"

        else:
            return f"Unknown operation: {operation}"

    except Exception as e:
        return f"Error processing data: {str(e)}"


# ===== TOOL 5: Date/Time Tool =====
@tool
def get_datetime_info(query: str = "") -> str:
    """
    Get current date and time information.

    Args:
        query: Optional - 'date', 'time', 'both', or empty for both
    """
    now = datetime.now()

    if query.lower() == "date":
        return now.strftime("%Y-%m-%d")
    elif query.lower() == "time":
        return now.strftime("%H:%M:%S")
    else:
        return now.strftime("%Y-%m-%d %H:%M:%S")


# ===== CREATE AGENT WITH ALL TOOLS =====
print("\n🔧 Setting up agent with custom tools...")

tools = [
    query_database,
    read_file,
    write_file,
    get_weather,
    process_data,
    get_datetime_info
]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant with access to various tools.

Available tools:
- Database queries
- File operations (read/write)
- Weather information
- Data processing
- Date/time information

Use the appropriate tool for each task."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

print("✓ Agent ready with 6 custom tools!\n")

# ===== TEST CASES =====
print("=" * 70)
print("TESTING CUSTOM TOOLS")
print("=" * 70)
#
test_queries = [
    "What users are in the database?",
    "What's the weather in London?",
    "Calculate the sum of 10, 20, 30, 40",
    "What's the current date and time?",
    "Get all orders from the database"
]

for i, query in enumerate(test_queries, 1):
    print(f"\n{'=' * 70}")
    print(f"Test {i}: {query}")
    print('=' * 70)

    try:
        result = agent_executor.invoke({"input": query})
        print(f"\n✅ Result:")
        print(result['output'])
    except Exception as e:
        print(f"❌ Error: {e}")
# input_query = input("User: ")
# try:
#     result = agent_executor.invoke({"input": input_query})
#     print(f"\n✅ Result:")
#     print(result['output'])
# except Exception as e:
#     print(f"❌ Error: {e}")

print("\n" + "=" * 70)
print("Custom Tools Demo Complete!")
print("=" * 70)