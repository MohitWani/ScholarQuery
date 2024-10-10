# frontend.py (Streamlit frontend)
import streamlit as st
from agent import run_agent
import requests
from langchain_groq.chat_models import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.memory import ConversationBufferMemory
import streamlit as st
import requests



from dotenv import load_dotenv
import os 

load_dotenv()

Groq_api_key = os.environ['Groq_API_key']

tools= load_tools(
    ['arxiv'],
)

llm=ChatGroq(groq_api_key=Groq_api_key,
         model_name="mixtral-8x7b-32768")

# Streamlit app interface
st.title("Ask Me About Research Paper.")

# File Upload
# st.header("Upload Document")
# uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# if uploaded_file is not None:
#     # Send the file to the backend for processing
#     files = {"file": uploaded_file.getvalue()}
#     response = requests.post("http://127.0.0.1:8000/upload/", files=files)
#     st.success("Document uploaded! " + response.json().get("message"))


agent_executor = run_agent(tools, llm)

st.title("Ask Me About Research Paper.")

query = st.text_input("Your question 👇")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if st.button('Ask'):
    response = agent_executor.invoke({"input":query})
    st.write(response['output'])

    st.session_state['chat_history'].append((response['history'][0], response['history'][1]))
    for i, (user_chat, assistant_chat) in enumerate(st.session_state['chat_history']):
        st.write(f"**User**: {user_chat}")  
        st.write(f"**Assistant**: {assistant_chat}")