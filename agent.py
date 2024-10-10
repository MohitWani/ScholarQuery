from langchain_groq.chat_models import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain import hub
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

prompt = hub.pull("hwchase17/react")

memory = ConversationBufferMemory(return_messages=True)

agent = create_react_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)


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


# if __name__=='__main__':
#     query = "what are the inputs to transformers in Attention is all you need paper?"
#     result = agent_executor.invoke({"input":query})

#     print(result['output'])
#     print(result['history'])
