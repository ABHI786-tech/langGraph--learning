from dotenv import load_dotenv

# Load .env FIRST so LangSmith variables are set before LangChain/LangGraph import
load_dotenv()

import os
from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found. Add it to your .env file.")


# ---------- State ----------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------- Model ----------
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
    huggingfacehub_api_token=token,
)
model = ChatHuggingFace(llm=llm)


# ---------- Node ----------
def chat_node(state: ChatState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}


# ---------- Graph ----------
checkpointer = MemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


# ---------- Quick terminal test / LangSmith check ----------
if __name__ == "__main__":
    from langsmith import utils

    print("LangSmith tracing enabled:", utils.tracing_is_enabled())
    print("LangSmith project:", os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT"))

    thread_id = "test-thread-1"
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_turn",
    }

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue
        result = chatbot.invoke(
            {"messages": [HumanMessage(content=user_input)]}, config=config
        )
        print("Bot:", result["messages"][-1].content, "\n")