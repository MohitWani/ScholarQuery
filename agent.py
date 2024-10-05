from langchain_groq.chat_models import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain import hub
from dotenv import load_dotenv
import os 

Groq_api_key = os.environ['Groq_API_key']

tools= load_tools(
    ['arxiv'],
)

llm=ChatGroq(groq_api_key=Groq_api_key,
         model_name="mixtral-8x7b-32768")

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm=llm, prompt=prompt, tools=tools)

agent_executor = AgentExecutor(llm=llm, tools=tools, verbose=True)

if __name__=='__main__':
    query = ""
    result = agent_executor.invoke({"input":query})

    print(result)