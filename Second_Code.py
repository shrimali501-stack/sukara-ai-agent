from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolMessage
import math
import json

load_dotenv()

# ── Tools ──────────────────────────────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression like '2 + 2' or 'math.sqrt(144)'."""
    try:
        result = eval(expression, {"math": math, "__builtins__": {}})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def percentage(value: float, percent: float) -> str:
    """Calculate what percent% of value is. E.g. percentage(200, 15) = 15% of 200."""
    result = (percent / 100) * value
    return f"{percent}% of {value} = {result}"

@tool
def compound_interest(principal: float, rate: float, time: float, n: float = 12) -> str:
    """Calculate compound interest.
    principal = initial amount, rate = annual rate (e.g. 8 for 8%),
    time = years, n = compounding frequency per year (default 12 = monthly)."""
    r = rate / 100
    amount = principal * (1 + r / n) ** (n * time)
    interest_earned = amount - principal
    return (
        f"Principal:        ₹{principal:,.2f}\n"
        f"Rate:             {rate}% per annum\n"
        f"Time:             {time} years\n"
        f"Final amount:     ₹{amount:,.2f}\n"
        f"Interest earned:  ₹{interest_earned:,.2f}"
    )

@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between units: km/miles, kg/lbs, celsius/fahrenheit, liters/gallons."""
    conversions = {
        ("km", "miles"):          lambda v: v * 0.621371,
        ("miles", "km"):          lambda v: v * 1.60934,
        ("kg", "lbs"):            lambda v: v * 2.20462,
        ("lbs", "kg"):            lambda v: v / 2.20462,
        ("celsius", "fahrenheit"):lambda v: v * 9/5 + 32,
        ("fahrenheit", "celsius"):lambda v: (v - 32) * 5/9,
        ("liters", "gallons"):    lambda v: v * 0.264172,
        ("gallons", "liters"):    lambda v: v * 3.78541,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    return f"Conversion from {from_unit} to {to_unit} not supported."

tools = [calculator, percentage, compound_interest, unit_converter]
tools_map = {t.name: t for t in tools}

# ── Model with tools bound ─────────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# ── Agent loop (no AgentExecutor needed) ───────────────────────────────────────

def run_agent(history: list[BaseMessage]) -> str:
    while True:
        response = llm_with_tools.invoke(history)
        history.append(response)

        # If no tool calls — we have the final answer
        if not response.tool_calls:
            return response.content

        # Execute each tool the model requested
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"  [using tool: {tool_name} with {tool_args}]")

            tool_result = tools_map[tool_name].invoke(tool_args)

            # Feed the result back into history
            history.append(ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            ))

# ── Conversation loop ──────────────────────────────────────────────────────────

chat_history: list[BaseMessage] = [
    SystemMessage(content=(
        "You are a smart calculation assistant. "
        "Always use the available tools to compute answers — never guess numbers. "
        "Show your reasoning clearly and explain each step."
    ))
]

print("Calculator Agent ready. Try:")
print("  - What is 15% of 85000?")
print("  - If I invest ₹50000 at 8% for 5 years, what do I get?")
print("  - Convert 100 km to miles")
print("  - What is the square root of 1764?\n")
print("Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "quit":
        break

    chat_history.append(HumanMessage(content=user_input))
    answer = run_agent(chat_history)
    chat_history.append(AIMessage(content=answer))
    print(f"\nAgent: {answer}\n")