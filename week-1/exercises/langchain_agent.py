from dotenv import load_dotenv
import datetime
import json
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Tool 1: Get current time
@tool
def get_current_time():
    """Get the current date and time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Tool 2: Calculate compound interest
@tool
def calculate_investment(principal: float, rate: float, years: int) -> str:
    """
    Calculate investment returns with compound interest.

    Args:
        principal: Initial investment amount in dollars
        rate: Annual interest rate as percentage (e.g., 8 for 8%)
        years: Number of years to invest
    """
    rate_decimal = rate / 100
    final_amount = principal * ((1 + rate_decimal) ** years)
    profit = final_amount - principal

    result = {
        "principal": principal,
        "rate": f"{rate}%",
        "years": years,
        "final_amount": round(final_amount, 2),
        "profit": round(profit, 2)
    }
    return json.dumps(result)


# Tool 3: Mock stock price
@tool
def get_stock_price(symbol: str) -> str:
    """
    Get current stock price for a given ticker symbol.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'MSFT')
    """
    mock_prices = {
        "AAPL": {"price": 178.50, "change": "+2.3%"},
        "TSLA": {"price": 242.80, "change": "-1.2%"},
        "MSFT": {"price": 378.90, "change": "+0.8%"},
        "GOOGL": {"price": 140.25, "change": "+1.5%"},
    }

    symbol = symbol.upper()
    data = mock_prices.get(symbol, {"price": 100.00, "change": "0.0%"})

    result = {
        "symbol": symbol,
        "price": data["price"],
        "change": data["change"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return json.dumps(result)

# Create tools list
tools = [get_current_time, calculate_investment, get_stock_price]

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful financial assistant. You can:
    - Tell the current time
    - Calculate investment returns
    - Get stock prices

    Always provide clear, concise answers."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Create agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # See agent thinking!
    handle_parsing_errors=True,
    max_iterations=5
)


def main():
    print("=" * 60)
    print("FINANCIAL ASSISTANT AGENT")
    print("=" * 60)
    print("I can help with time, investment calculations, and stock prices!")
    print("Type 'quit' to exit\n")

    # # Test queries
    # test_queries = [
    #     "What time is it?",
    #     "What's Apple's stock price?",
    #     "If I invest $10,000 at 8% for 5 years, what will I have?",
    #     "Get me Tesla's stock price and calculate returns on $5000 at 10% for 3 years"
    # ]
    #
    # print("Running test queries:\n")
    # for query in test_queries:
    #     print(f"\n{'=' * 60}")
    #     print(f"Query: {query}")
    #     print('=' * 60)
    #     try:
    #         response = agent_executor.invoke({"input": query})
    #         print(f"\nAgent: {response['output']}")
    #     except Exception as e:
    #         print(f"Error: {e}")
    #     print('=' * 60)

    # Interactive mode
    print("\n\nINTERACTIVE MODE:")
    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ['quit', 'exit']:
            break

        if not user_input:
            continue

        try:
            response = agent_executor.invoke({"input": user_input})
            print(f"\nAgent: {response['output']}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()