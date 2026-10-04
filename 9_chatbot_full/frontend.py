from dotenv import load_dotenv

load_dotenv()

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

# from backend import chatbot
from tool_backend import chatbot


st.set_page_config(page_title="Chatbot", page_icon="💬")


# ---------- Helpers ----------
def generate_thread_id() -> str:
    return str(uuid.uuid4())


def init_state():
    st.session_state.setdefault("message_history", [])
    st.session_state.setdefault("thread_id", generate_thread_id())
    st.session_state.setdefault("chat_threads", [])  # only chats that have messages
    st.session_state.setdefault("thread_titles", {})  # thread_id -> sidebar title


def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []


def load_conversation(thread_id: str):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def switch_thread(thread_id: str):
    history = []
    for message in load_conversation(thread_id):
        if isinstance(message, HumanMessage):
            history.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            history.append({"role": "assistant", "content": message.content})

    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = history


def register_thread(thread_id: str, first_message: str):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)
        title = first_message.strip().replace("\n", " ")
        st.session_state["thread_titles"][thread_id] = (
            title[:30] + "..." if len(title) > 30 else title
        )


def build_config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        # metadata.thread_id makes the run appear under the Threads tab in LangSmith
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_turn",
    }


def stream_reply(user_input: str, config: dict):
    """Yield only the assistant's text tokens from the chat node."""
    for chunk, metadata in chatbot.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="messages",
    ):
        if metadata.get("langgraph_node") == "chat_node" and chunk.content:
            yield chunk.content


# ---------- State ----------
init_state()

# ---------- Sidebar ----------
st.sidebar.title("Chatbot")
st.sidebar.button("➕ New chat", on_click=reset_chat, use_container_width=True)
st.sidebar.header("Conversation history")

for tid in reversed(st.session_state["chat_threads"]):
    is_active = tid == st.session_state["thread_id"]
    st.sidebar.button(
        st.session_state["thread_titles"].get(tid, "New chat"),
        key=f"thread_{tid}",
        on_click=switch_thread,
        args=(tid,),
        type="primary" if is_active else "secondary",
        use_container_width=True,
    )

# ---------- Chat history ----------
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------- Input ----------
user_input = st.chat_input("Type here")

if user_input:
    config = build_config(st.session_state["thread_id"])

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            ai_message = st.write_stream(stream_reply(user_input, config))
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    st.session_state["message_history"].append({"role": "user", "content": user_input})
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
    register_thread(st.session_state["thread_id"], user_input)
    st.rerun()
