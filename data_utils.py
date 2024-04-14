# Imports
import pandas as pd
import numpy as np

# from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.llms import Ollama

# Data Utils
class DataHelper:
    """
    """
    agent = create_csv_agent(
        Ollama(
            model='mistral',
            temperature=0
        ),
        'data/data.csv',
        verbose=False,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )

    def __init__(self):
        pass

    def answer_question_using_data(self, question):
        """
        """
        try:
            response = self.agent.run(question)
        except ValueError as e:
            return f"Error: {e}"
        return response

