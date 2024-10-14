import requests
import streamlit as st


def get_rag_response(input_text):
    response = requests.post("http://localhost:8000/RAG")
    json = {'input':{'topic':input_text}}

    return requests.json()


def get_agent_response(input_text):
    response = requests.post("http://localhost:8000/RAG")
    json = {'input':{'topic':input_text}}

    return requests.json()