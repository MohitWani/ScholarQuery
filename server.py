from fastapi import FastAPI
from langserve import add_routes
from utils.Retrieval import load_document, splitter, create_vectorstore
from utils.Generation import multi_query_retriever, reciprocal_rank_fusion, generation_step
from utils.agent import run_agent
import uvicorn
from langchain.schema.runnable import RunnableLambda,Runnable



# retrieval_chain = load_document | splitter | create_vectorstore
# gen_chain = multi_query_retriever | reciprocal_rank_fusion | generation_step



app = FastAPI(
    title="Agent and RAG.",
    version="1.0",
    description="/Server"
    )

#runable1 = RunnableLambda(generation_step)

# add_routes(
#     app,
#     retrieval_chain,
#     path="/RAG"
# )

#runable2 = RunnableLambda(run_agent)
add_routes(
    app,
    RunnableLambda(run_agent),
    path="/Agent"
)

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)

