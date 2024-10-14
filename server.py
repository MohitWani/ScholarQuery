from fastapi import FastAPI
from langserve import add_routes
from utils.Generation import generation_step
from utils.agent import run_agent
import uvicorn
from langchain.schema.runnable import RunnableLambda

runable1 = RunnableLambda(generation_step)
runnable2 = RunnableLambda(run_agent)
app = FastAPI(
    title="Agent and RAG.",
    version="1.0",
    description="/Server"
    )

add_routes(
    app,
    runnable=runable1,
    path="/RAG"
)

add_routes(
    app,
    runnable=runnable2,
    path="/Agent"
)

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)

