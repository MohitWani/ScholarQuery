from fastapi import FastAPI
from utils.agent import run_agent
import uvicorn
from utils.Retrieval import load_document, splitter, create_vectorstore
from utils.Generation import multi_query_retriever, reciprocal_rank_fusion, generation_step, model
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS


app = FastAPI(
    title="Agent and RAG.",
    version="1.0",
    description="/Server"
    )

@app.post("/retrieval")
async def retrieval(file):
    doc = load_document(file)

    splits = splitter(doc)

    create_vectorstore()

    return {"message":"Retrieval Step is Successfull"}

@app.post("/Generation")
async def generation(input_text):
    llm = model()
    embedding = GPT4AllEmbeddings()
    db = FAISS.load_local('D:/my_projects/ScholarQuery/faiss_index', embedding, allow_dangerous_deserialization=True)
    retrieval = db.as_retriever()

    retriever = multi_query_retriever(llm,retrieval,input_text)
    rank = reciprocal_rank_fusion(retriever)
    response = generation_step(llm,rank,input_text)

    return response

@app.post("/agent")
async def agents(question):
    agent_exec = run_agent()
    return agent_exec.invoke({'input':question})


if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)

