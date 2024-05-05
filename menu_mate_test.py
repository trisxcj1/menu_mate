# Imports
import streamlit as st
import pandas as pd
import os

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

# for pdf chat
from langchain.embeddings import OpenAIEmbeddings, OllamaEmbeddings, GPT4AllEmbeddings
# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
# Import chroma as the vector store 
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

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


# os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
llm = Ollama(
    model=model_options[1],
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]),
    temperature=0.9
)
embeddings = GPT4AllEmbeddings()
# loader = PyPDFLoader('data/resume.pdf')
# pages = loader.load_and_split()
# url = 'https://menupages.com/12-chairs/342-wythe-ave-brooklyn'
# loader = WebBaseLoader(url)
# data = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1500,
#     chunk_overlap=100
# )
# all_splits = text_splitter.split_documents(data)

# Main
st.set_page_config(
    page_title='MenuMate 🧑🏽‍🍳'
)

st.title('MenuMate 🍲')

question = st.text_input('Ask something')

# store = Chroma.from_documents(
#     pages,
#     embeddings,
#     collection_name='restaurantsmenus'
# )

# vectorstore = Chroma.from_documents(
#     all_splits,
#     embeddings
# )
# from langchain import hub
# QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")

# from langchain.chains import RetrievalQA
# qa_chain = RetrievalQA.from_chain_type(
#     llm,
#     retriever=vectorstore.as_retriever(),
#     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},

# )

# vectorstore_info = VectorStoreInfo(
#     name='tristn_joseph_resume',
#     description='a resume',
#     vectorstore=store
# )
# result = qa_chain(
#     {'query': f'What are the menu items at {url}?'}
# )
# st.write(result['result'])

# toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# agent_executor = create_vectorstore_agent(
#     llm=llm,
#     toolkit=toolkit,
#     verbose=True
# )

# # Main
# st.set_page_config(
#     page_title='MenuMate 🧑🏽‍🍳'
# )

# st.title('MenuMate 🍲')

# question = st.text_input('Ask something')

# if question:
#     response = agent_executor.run(question)
#     st.write(response)
    
    # with st.expander('Doc Similarity Search'):
    #     search = store.similarity_search_with_score(question)
    #     st.write(search[0][0].page_content)

conversation_cain = LLMChain(
    llm=llm,
    prompt=llm_prompt_template,
    memory=conversation_memory,
    verbose=True
)


# if 'messages' not in st.session_state:
#     st.session_state['messages'] = [{
#         'role': 'assistant',
#         'content': 'Hi! How can I help today 🍲?'
#     }]

#     st.markdown('## Hi! How can I help today 🍲?')
#     new_chat_container_p1 = st.empty()
#     new_chat_container_p2 = st.empty()

#     with new_chat_container_p1.container():
#         suggestion_1, suggestion_2 = st.columns(2)

#         suggestion_1.markdown("Tell me about Brazilian Lemonade")
#         suggestion_2.markdown("What is Roquefort cheese made from?")

#     with new_chat_container_p2.container():
#         suggestion_3, suggestion_4 = st.columns(2)
        
#         suggestion_3.markdown("What is the main ingredient in guacamole?")
#         suggestion_4.markdown("Help me plan an Italian-themed dinner")

# for message in st.session_state.messages[1:]:
#     st.chat_message(message['role']).write(message['content'])


# if user_input := st.chat_input():
#     st.session_state.messages.append({'role': 'user', 'content': user_input})
#     st.chat_message('user').write(user_input)
    
#     app_reply = conversation_cain(user_input)

#     st.session_state.messages.append({'role': 'assistant', 'content': app_reply['text']})
#     st.chat_message('assistant').write(app_reply['text'])
#     message = {'human': user_input, 'AI':app_reply['text']}
#     st.session_state.chat_history.append(message)
    