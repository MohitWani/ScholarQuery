# frontend.py (Streamlit frontend)
import streamlit as st
from utils.agent import run_agent
import streamlit as st
import requests

# Streamlit app interface


# File Upload
# st.header("Upload Document")
# uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# if uploaded_file is not None:
#     # Send the file to the backend for processing
#     files = {"file": uploaded_file.getvalue()}
#     response = requests.post("http://127.0.0.1:8000/upload/", files=files)
#     st.success("Document uploaded! " + response.json().get("message"))



st.title("Ask Me About Research Paper.")
agent_executor = run_agent()

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