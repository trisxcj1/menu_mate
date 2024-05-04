# Imports
import streamlit as st
import pandas as pd

import random
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import img2pdf
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import PyPDF2
from PyPDF2 import PdfReader, PdfWriter, PdfMerger
import os

import litellm

from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler    
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory

import litellm

from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler    
from langchain import PromptTemplate, LLMChain
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory

conversation_memory = ConversationBufferMemory()


if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
else:
    for message in st.session_state.chat_history:
        conversation_memory.save_context(
            {'input': message['human']},
            {'output': message['AI']}
        )
model_options = [
    'mistral',
    'stablelm-zephyr',
    'llama3'
]

llm_prompt_template = PromptTemplate(
    input_variables=['history', 'input'],
    template="""
    You are a food expert. Read the input prompt below and continue
    the conversation in a friendly tone with the most appropriate response.
    All responses must be related to food and must be less than 50 words.
    
    conversation history:
    {history}
    
    human:{input}
    AI:
    """
)


llm = Ollama(
    model=model_options[1],
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]),
    temperature=0.9
)

conversation_cain = LLMChain(
    llm=llm,
    prompt=llm_prompt_template,
    memory=conversation_memory,
    verbose=True
)

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


if user_input := st.chat_input():
    st.session_state.messages.append({'role': 'user', 'content': user_input})
    st.chat_message('user').write(user_input)
    
    app_reply = conversation_cain(user_input)

    st.session_state.messages.append({'role': 'assistant', 'content': app_reply['text']})
    st.chat_message('assistant').write(app_reply['text'])
    message = {'human': user_input, 'AI':app_reply['text']}
    st.session_state.chat_history.append(message)
    