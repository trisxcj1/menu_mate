# Imports
import streamlit as st
import random

import litellm

from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler    
from langchain import PromptTemplate, LLMChain

from llm_utils import LLMHelper

llmh__i = LLMHelper()


# Main
st.set_page_config(
    page_title='MenuMate 🧑🏽‍🍳'
)

st.title('MenuMate 🍲')


if 'messages' not in st.session_state:
    st.session_state['messages'] = [{
        'role': 'assistant',
        'content': 'Hi! How can I help today 🍲?'
    }]

    st.markdown('## Hi! How can I help today 🍲?')
    new_chat_container_p1 = st.empty()
    new_chat_container_p2 = st.empty()

    with new_chat_container_p1.container():
        suggestion_1, suggestion_2 = st.columns(2)

        suggestion_1.markdown("Tell me about Brazilian Lemonade")
        suggestion_2.markdown("What is Roquefort cheese made from?")

    with new_chat_container_p2.container():
        suggestion_3, suggestion_4 = st.columns(2)
        
        suggestion_3.markdown("What is the main ingredient in guacamole?")
        suggestion_4.markdown("Help me plan an Italian-themed dinner")

for message in st.session_state.messages[1:]:
    st.chat_message(message['role']).write(message['content'])
    # if message['content'] != 'Hi! How can I help today 🍲?':
    #     st.chat_message(message['role']).write(message['content'])

if user_input := st.chat_input():
    st.session_state.messages.append({'role': 'user', 'content': user_input})
    st.chat_message('user').write(user_input)

    app_reply = llmh__i.generate_llm_response(user_input)
    st.session_state.messages.append({'role': 'assistant', 'content': app_reply})
    st.chat_message('assistant').write(app_reply)
