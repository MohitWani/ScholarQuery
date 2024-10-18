# frontend.py (Streamlit frontend)
import streamlit as st
from utils.agent import run_agent
import streamlit as st
import requests

# Streamlit app interface

# created a two tabs, one for RAG and other for Agent and tools.
tab1, tab2 = st.tabs(['ASK QUESTION TO YOUR DOCUMENTS.','ASK QUESTION TO ARXIV AGENT.'])
# File Upload
with tab1:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Send the file to the backend for processing
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        response = requests.post("http://localhost:8000/retrieval", files=files)
        st.success("Document uploaded! ")

    prompt = st.chat_input("Write your query here")
    if prompt:
        chat_res = requests.post("http://localhost:8000/Generation",json={"prompt":prompt})
        chat_output = chat_res.json()
        st.write(f"Prompt: {prompt}")
        st.write(chat_output['response'].strip())


with tab2:
    # agent_executor = run_agent()

    st.title("Ask Me About Research Paper.")
    # Input box for Query Input.
    query = st.chat_input("Your question 👇")

    # It stores the chat history for both user and AI response.
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    if query is not None:
        st.write(f"query: {query}")
        # response = agent_executor.invoke({"input":query})
        response = requests.post("http://localhost:8000/agent", json={"query":query})
        res = response.json()
        st.write(res)

        st.session_state['chat_history'].append((response['history'][0], response['history'][1]))
        for i, (user_chat, assistant_chat) in enumerate(st.session_state['chat_history']):
            st.write(f"**User**: {user_chat}")  
            st.write(f"**Assistant**: {assistant_chat}")