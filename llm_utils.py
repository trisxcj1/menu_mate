# Imports
import litellm

from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler    
from langchain import PromptTemplate, LLMChain
from langchain.chains import LLMChain, SequentialChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import GPT4AllEmbeddings
from langchain import hub

from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from bs4 import BeautifulSoup

import streamlit as st

# LLM Utils
class LLMHelper:
    """
    """
    url = 'https://menupages.com/shake-shack/23rd-st-madison-ave-new-york'
    
    llm = Ollama(
        model='mistral',
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]),
        temperature=0.9
    )

    question_classification_llm_prompt_template = PromptTemplate(
        input_variables=['history', 'input'],
        template="""
        ### Task:
        Classify a user's question as either 'general' or 'specific'
        
        ### Instruction:
        You are a classifier that analyzes a user's input about food-related prompts
        and determines whether that user is asking a general question about food,
        or if the user is asking about food that relates to a specific restaurant or menu.
        
        Your output should either be G for general or S for specific - nothing else!
        
        You should only classify a response as S if the user is asking whether a specific item
        is available at a specific place. Otherwise, classify it as G.
        
        Each response should be exactly one character, either G or S.

        ### Conversation history:
        {history}

        human:{input}
        AI:
        """
    )
    
    basic_llm_prompt_template = PromptTemplate(
        input_variables=['history', 'input'],
        template="""
        ### Instruction:
        You are a helpful assistant who is also a food expert. Read the prompt below and continue
        the conversation in a friendly tone with the most appropriate response.
        Be conversational! All responses must be related to food and must be less than 200 words.

        ### Conversation history:
        {history}

        human:{input}
        AI:
        """
    )
    
    loader = WebBaseLoader(url)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100
    )
    all_splits = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(
        all_splits,
        GPT4AllEmbeddings()
    )
    QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")
    
    def __init__(self):
        """
        class [ LLMHelper ]

        Provides:
        - Methods to easily interact with LLM models for this application.
        """

        print('Instantiated class: [ {0} ].'.format(type(self).__name__))
        print(self.__doc__)

        pass
    
    def generate_question_classification_response(self, input_message, conversation_memory):
        """
        """
        conversation_chain = LLMChain(
            llm=self.llm,
            prompt=self.question_classification_llm_prompt_template,
            memory=conversation_memory,
            verbose=True
        )
        llm_response = conversation_chain(input_message)
        # llm_response['text']
        return llm_response['text']
        
    def generate_basic_llm_response(self, input_message, conversation_memory):
        """
        """
        conversation_chain = LLMChain(
            llm=self.llm,
            prompt=self.basic_llm_prompt_template,
            memory=conversation_memory,
            verbose=True
        )
        llm_response = conversation_chain(input_message)
        return llm_response
    
    def generate_rag_qa_llm_response(self, input_message, conversation_memory):
        """
        """
        qa_chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=self.vectorstore.as_retriever(),
            chain_type_kwargs={'prompt': self.QA_CHAIN_PROMPT}
        )
        
        result = qa_chain(
            {'query': f'{input_message} Check {self.url} to find the answer'}
        )
        return result
        