# # Imports 
# import litellm

# from langchain.llms import Ollama
# from langchain.callbacks.manager import CallbackManager
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler    
# from langchain import PromptTemplate, LLMChain
# from langchain.chains import LLMChain, SequentialChain 
# from langchain.memory import ConversationBufferMemory

# # LLM Utils
# class LLMHelper:
#     """
#     """

#     llm = Ollama(
#         model='mistral',
#         callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]),
#         temperature=0.9
#     )

#     llm_prompt_template = PromptTemplate(
#         input_variables=['food_question'],
#         template="""
#         ### Instruction:
#         You are a food expert. Read the prompt below and continue
#         the conversation with the most appropriate response.
#         All responses must be related to food.

#         ### Prompt:
#         {food_question}

#         ### Response:
#         """
#     )

#     food_memory = ConversationBufferMemory(input_key='food_question', memory_key='chat_history')
    
#     def __init__(self):
#         """
#         class [ LLMHelper ]

#         Provides:
#         - Methods to easily interact with LLM models for this application.
#         """

#         print('Instantiated class: [ {0} ].'.format(type(self).__name__))
#         print(self.__doc__)

#         pass
    
#     def generate_llm_response(self, input_message):
#         """
#         """
#         llm_chain = LLMChain(
#             llm=self.llm,
#             prompt=self.llm_prompt_template,
#             verbose=False,
#             output_key='food_response',
#             memory=self.food_memory
#         )
        
#         food_response = llm_chain.run(input_message)
#         return food_response

# Imports 
import litellm

from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler    
from langchain import PromptTemplate, LLMChain
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory

import streamlit as st

# LLM Utils
class LLMHelper:
    """
    """

    llm = Ollama(
        model='mistral',
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]),
        temperature=0.9
    )

    llm_prompt_template = PromptTemplate(
        input_variables=['history', 'input'],
        template="""
        ### Instruction:
        You are a helpful assistant who is also a food expert. Read the prompt below and continue
        the conversation in a friendly tone with the most appropriate response.
        All responses must be related to food and must be less than 50 words.

        ### Conversation history:
        {history}

        human:{input}
        AI:
        """
    )
    
    def __init__(self):
        """
        class [ LLMHelper ]

        Provides:
        - Methods to easily interact with LLM models for this application.
        """

        print('Instantiated class: [ {0} ].'.format(type(self).__name__))
        print(self.__doc__)

        pass
    
    def generate_llm_response(self, input_message, conversation_memory):
        """
        """
        conversation_chain = LLMChain(
            llm=self.llm,
            prompt=self.llm_prompt_template,
            memory=conversation_memory,
            verbose=True
        )
        llm_response = conversation_chain(input_message)
        return llm_response
        