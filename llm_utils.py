# Imports 
import litellm

from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler    
from langchain import PromptTemplate, LLMChain
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory

# LLM Utils
class LLMHelper:
    """
    """

    llm = Ollama(
        model='mistral',
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]),
        temperature=0.9
    )

    llm_prompt = PromptTemplate(
        input_variables=['user_input'],
        template="""
        ### Instruction:
        You are a food expert. Read the prompt below and continue
        the conversation with the most appropriate response.

        ### Prompt:
        {user_input}

        ### Response:
        """
    )

    user_input_memory = ConversationBufferMemory(input_key='user_input', memory_key='chat_history')
    
    def __init__(self):
        """
        class [ LLMHelper ]

        Provides:
        - Methods to easily interact with LLM models for this application.
        """

        print('Instantiated class: [ {0} ].'.format(type(self).__name__))
        print(self.__doc__)

        pass
    
    def generate_llm_response(self, input_message):
        """
        """
        llm_chain = LLMChain(
            prompt=self.llm_prompt,
            llm=self.llm,
            output_key='user_input',
            memory=self.user_input_memory
        )
        
        llm_response = llm_chain.run(input_message)
        return llm_response

    
    
