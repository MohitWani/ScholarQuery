from fastapi import FastAPI, UploadFile, File
from utils.agent import run_agent
import uvicorn
from utils.Retrieval import load_document, splitter, create_vectorstore
from utils.Generation import multi_query_retriever, reciprocal_rank_fusion, generation_step, model
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel


app = FastAPI(
    title="Agent and RAG.",
    version="1.0",
    description="/Server"
    )

@app.post("/retrieval")
async def retrieval(file: UploadFile = File(...)):
    try:
        doc = load_document(file.file)

        splits = splitter(doc)

        create_vectorstore(splits)
        return {"message":"Retrieval Step is Successfull"}
    except Exception as e:
        return {"error": str(e)}

class QueryInput(BaseModel):
    prompt: str
    query: str

@app.post("/Generation")
async def generation(input: QueryInput):
    llm = model()
    embedding = GPT4AllEmbeddings()
    db = FAISS.load_local('D:/my_projects/ScholarQuery/faiss_index', embedding, allow_dangerous_deserialization=True)
    retrieval = db.as_retriever()

    query_results = multi_query_retriever(llm,retrieval,input.prompt)
    reranked_results = reciprocal_rank_fusion(query_results)
    response = generation_step(llm,reranked_results,input.prompt)

    return {'response':response}
    

@app.post("/agent")
async def agents(input: QueryInput):
    agent_exec = run_agent()
    response = agent_exec.run(input.query)
    return {'response':response}


if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)

