import streamlit as st
from chatbot import chatbot
from langchain_core.messages import  HumanMessage


#  with st.chat_message("user"):
#     st.text("Hi")


# with st.chat_message("assistant"):
#     st.text("hi! how can i help you")
# message_history = []

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

for message in st.session_state['message_history']:
    with st.chat_message(message["role"]):
        st.text(message["content"])

config = {'configurable': {'thread_id': 'thread_1'}}
user_input = st.chat_input("type here")

if user_input:

    st.session_state['message_history'].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)



    response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=config)
    ai_message = response['messages'][-1].content

    st.session_state['message_history'].append({"role": "assistant", "content": ai_message})
    with st.chat_message("assistant"):
        st.text(ai_message)
