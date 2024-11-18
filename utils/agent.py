from langchain_groq.chat_models import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain import hub

from dotenv import load_dotenv
import os 

load_dotenv()

Groq_api_key = os.environ['Groq_API_key']

tools= load_tools(
    ['arxiv'],
)

llm=ChatGroq(groq_api_key=Groq_api_key,
         model_name="mixtral-8x7b-32768")

# Langchain Agent and Arxiv tools
def run_agent(query, tools=tools, model=llm):
    prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm=model, prompt=prompt, tools=tools)
    agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)
    result = agent_executor.invoke({"input":query})

    return result






# if __name__=='__main__':
#     query = "what are the inputs to transformers in Attention is all you need paper?"
#     result = run_agent(query)

#     print(result['output'])
    # print(result['history'])
