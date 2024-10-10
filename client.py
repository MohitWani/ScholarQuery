# frontend.py (Streamlit frontend)
import streamlit as st
import requests

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

# Retrieval Section
st.header("Ask the Assistant")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False
user_input = st.text_input(
        "Your question 👇",
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
    )
if st.button("Ask"):
    response = requests.post("http://127.0.0.1:8000/query/", json={"prompt": user_input})
    assistant_reply = response.json().get("response")
    
    # Display chat history
    st.session_state['chat_history'].append((user_input, assistant_reply))
    for i, (user_chat, assistant_chat) in enumerate(st.session_state['chat_history']):
        st.write(f"**User**: {user_chat}")
        st.write(f"**Assistant**: {assistant_chat}")
