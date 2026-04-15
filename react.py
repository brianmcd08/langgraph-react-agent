from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()


@tool
def triple(num: float) -> float:
    """Given a number, returns the number multiplied by 3

    Args:
        num (float): _description_

    Returns:
        float: _description_
    """
    return float(num) * 3


tools = [TavilySearch(max_results=1), triple]
llm = ChatAnthropic(
    model_name="claude-haiku-4-5-20251001",
    timeout=0,
    stop=[],
    max_tokens_to_sample=256,
    temperature=0,
).bind_tools(tools)  # for function-calling
