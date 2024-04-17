# Imports
import streamlit as st

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

from llm_utils import LLMHelper
from data_utils import DataHelper

llmh__i = LLMHelper()
dh__i = DataHelper()

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


# data_question = st.text_input('Ask DATA a question')
# if data_question:
#     data_response = dh__i.answer_question_using_data(data_question)
#     st.write(data_response)

url = 'https://www.umbrellasbeachbargrenada.com/menu'

restaurant_info = "The following menu is for Umbrellas Beach Bar Grenada and this restaurant has opening hours Monday - Friday, 8am - 8pm"

# output_pdf = 'menu.pdf'
# def save_webpage_to_pdf(url, output_pdf):
#     try:
#         pdfkit.from_url(url, output_pdf)
#         return True
#     except Exception as e:
#         st.error(f"Error saving PDF: {e}")
#         return False

# if save_webpage_to_pdf(url, output_pdf):
#     with open(output_pdf, 'rb') as f:
#         pdf_bytes = f.read()
#     st.write(pdf_bytes, format='pdf')
def capture_webpage_to_pdf(url):
    # Initialize Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run Chrome in headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    # Initialize the web driver (e.g., ChromeDriver)
    driver = webdriver.Chrome(options=chrome_options)  # Make sure chromedriver is in your PATH or specify its location

    try:
        # Open the web page
        driver.get(url)
        
        # Wait for the page to load
        time.sleep(2)  # Adjust the wait time as needed
        
        # Capture a screenshot as binary data
        total_height = driver.execute_script("return document.body.scrollHeight")
        driver.set_window_size(1200, total_height)
        screenshot = driver.get_screenshot_as_png()

        # Convert the screenshot image to PDF
        pdf_bytes = img2pdf.convert(screenshot)

        return pdf_bytes

    except Exception as e:
        st.error(f"Error capturing PDF: {e}")
        return None

    finally:
        # Close the web driver
        driver.quit()

umbrellas_pdf_bytes = capture_webpage_to_pdf(url)
other_pdf_bytes = capture_webpage_to_pdf('https://menupages.com/fishermans-cove/2137-nostrand-ave-brooklyn/')
# pdf_writer = PdfWriter()
# new_page = PyPDF2.pdf.PageObject.createBlankPage(width=1200, height=200)
# new_page.mergePage(PyPDF2.pdf.PageObject.createTextObject("hello mate"))
# pdf_writer.addPage(new_page)

# pdf_reader = PdfReader(BytesIO(pdf_bytes))

# for page_num in range(pdf_reader.numPages):
#     pdf_writer.addPage(pdf_reader.getPage(page_num))

# output_pdf_bytes = BytesIO()
# pdf_writer.write(output_pdf_bytes)
# output_pdf_bytes = output_pdf_bytes.getvalue()
  
# pdf_bytes = capture_webpage_to_pdf(url)

output_bytes = other_pdf_bytes + umbrellas_pdf_bytes

if output_bytes:
    st.download_button(
            "Download PDF",
            output_bytes,
            "umbrellas_menu.pdf",
            "Click here to download the PDF file"
        )