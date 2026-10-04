from dotenv import load_dotenv

load_dotenv()

import uuid

import streamlit as st
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from tool_backend import chatbot, retrieve_all_threads

st.set_page_config(page_title="LangGraph Chatbot", page_icon="💬")


# =========================== Utilities ===========================
def generate_thread_id() -> str:
    # LangGraph/LangSmith expect a plain string, not a UUID object
    return str(uuid.uuid4())


def add_thread(thread_id: str):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def text_of(content) -> str:
    """Message content can be a str or a list of blocks; return plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def load_conversation(thread_id: str):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def messages_to_history(messages) -> list[dict]:
    """
    Convert LangChain messages into UI messages.
    - ToolMessages are not shown as bubbles; their tool names are attached
      to the assistant answer that follows them.
    - AI messages with no text (tool-call requests) are skipped, so no empty bubbles.
    """
    history = []
    pending_tools: list[str] = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": text_of(msg.content), "tools": []})
            pending_tools = []
        elif isinstance(msg, ToolMessage):
            name = getattr(msg, "name", None) or "tool"
            if name not in pending_tools:
                pending_tools.append(name)
        elif isinstance(msg, AIMessage):
            text = text_of(msg.content)
            if text.strip():
                history.append(
                    {"role": "assistant", "content": text, "tools": pending_tools}
                )
                pending_tools = []

    return history


def switch_thread(thread_id: str):
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = messages_to_history(
        load_conversation(thread_id)
    )


def thread_title(thread_id: str) -> str:
    """Use the first user message as the sidebar title (cached)."""
    titles = st.session_state["thread_titles"]
    if thread_id not in titles:
        title = "New chat"
        for msg in load_conversation(thread_id):
            if isinstance(msg, HumanMessage):
                raw = text_of(msg.content).strip().replace("\n", " ")
                title = raw[:30] + "..." if len(raw) > 30 else raw
                break
        if title != "New chat":  # only cache once the chat really has a message
            titles[thread_id] = title
        return title
    return titles[thread_id]


def render_message(message: dict):
    with st.chat_message(message["role"]):
        tools = message.get("tools") or []
        if tools:
            with st.status(
                "✅ Used " + ", ".join(f"`{t}`" for t in tools),
                state="complete",
                expanded=False,
            ):
                st.caption("Tool call finished.")
        st.markdown(message["content"])


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = [str(t) for t in retrieve_all_threads()]

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}

add_thread(st.session_state["thread_id"])


# ============================ Sidebar ============================
st.sidebar.title("LangGraph Chatbot")
st.sidebar.button("➕ New Chat", on_click=reset_chat, use_container_width=True)

st.sidebar.header("My Conversations")
for tid in reversed(st.session_state["chat_threads"]):
    st.sidebar.button(
        thread_title(tid),
        key=f"thread_{tid}",
        on_click=switch_thread,
        args=(tid,),
        type="primary" if tid == st.session_state["thread_id"] else "secondary",
        use_container_width=True,
    )


# ============================ Main UI ============================
for message in st.session_state["message_history"]:
    render_message(message)


def stream_reply(user_input: str, config: dict, status_area, tools_used: list):
    """
    Yield assistant text tokens. Tool activity is shown in `status_area`
    (a container placed ABOVE the answer text).
    """
    status_box = None

    def show_tool(name: str):
        nonlocal status_box
        if name in tools_used:
            return
        tools_used.append(name)
        label = "🔧 Using " + ", ".join(f"`{t}`" for t in tools_used) + " …"
        if status_box is None:
            status_box = status_area.status(label, expanded=True)
        else:
            status_box.update(label=label, state="running", expanded=True)

    for chunk, _metadata in chatbot.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="messages",
    ):
        # 1) The model decided to call a tool -> show it immediately
        for tc in getattr(chunk, "tool_call_chunks", None) or []:
            if tc.get("name"):
                show_tool(tc["name"])

        # 2) Tool finished
        if isinstance(chunk, ToolMessage):
            show_tool(getattr(chunk, "name", None) or "tool")

        # 3) Stream ONLY assistant text (tool-call chunks have empty content)
        elif isinstance(chunk, AIMessage):
            text = text_of(chunk.content)
            if text:
                yield text

    if status_box is not None:
        status_box.update(
            label="✅ Used " + ", ".join(f"`{t}`" for t in tools_used),
            state="complete",
            expanded=False,
        )


user_input = st.chat_input("Type here")

if user_input:
    config = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # Fixed order: tool status first, answer text below it.
        # To show the tools BELOW the answer, swap these two lines.
        status_area = st.container()
        text_area = st.container()

        tools_used: list[str] = []
        try:
            with text_area:
                ai_message = st.write_stream(
                    stream_reply(user_input, config, status_area, tools_used)
                )
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    st.session_state["message_history"].append(
        {"role": "user", "content": user_input, "tools": []}
    )
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message or "", "tools": tools_used}
    )
    st.rerun()  # refresh sidebar title for new chats