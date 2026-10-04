import streamlit as st
from chatbot_backend import chatbot
from langchain_core.messages import HumanMessage

# with st.chat_message("user"):
#     st.text("Hi")


# with st.chat_message("assistant"):
#     st.text("hi! how can i help you")
# message_history = []

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

config = {"configurable": {"thread_id": "thread_1"}}
user_input = st.chat_input("type here")

if user_input:

    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    with st.chat_message("assistant"):

        ai_message = st.write_stream(
            message_chunk.content
            for message_chunk, medata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config={"configurable": {"thread_id": "thread_1"}},
                stream_mode="messages",
            )
        )

    st.session_state['message_history'].append({"role": "assistant", "content": ai_message})
    