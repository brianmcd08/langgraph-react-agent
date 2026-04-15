from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage
from langgraph.graph import END, MessagesState, StateGraph

from nodes import run_agent_reasoning, tool_node

load_dotenv()


def should_continue(state: MessagesState) -> str:
    # check last message
    last_message = state["messages"][LAST]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return END
    return ACT


AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1

flow = StateGraph(MessagesState)
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

flow.add_conditional_edges(
    AGENT_REASON,
    should_continue,
    {
        END: END,
        ACT: ACT,
    },
)

flow.add_edge(ACT, AGENT_REASON)
app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")


if __name__ == "__main__":
    print("ReAct LangGraph with Function Calling")
    res = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What is the weather in Tokyo? List it and then triple it."
                )
            ]
        }
    )

    print(res["messages"][LAST].content)
