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

# PIPELINE for retrieval step with steps // loading -> splitting -> Embed -> vector store.
@app.post("/retrieval")
async def retrieval(file: UploadFile = File(...)):
    try:
        doc = load_document(file.file)

        splits = splitter(doc)

        create_vectorstore(splits)
        return {"message":"Retrieval Step is Successfull"}
    except Exception as e:
        return {"error": str(e)}

# QueryInput Class with Pydantic Base model for datatype restriction.
class QueryInput(BaseModel):
    
    query: str

# PIPELINE for Generation step with steps // Multi Query -> RAG Fusion -> Generation.
@app.post("/Generation")
def generation(input: QueryInput):
    llm = model()
    embedding = GPT4AllEmbeddings()
    # change the file path as per your path.
    db = FAISS.load_local('D:/my_projects/ScholarQuery/faiss_index', embedding, allow_dangerous_deserialization=True)
    retrieval = db.as_retriever()

    query_results = multi_query_retriever(llm,retrieval,input.query)
    reranked_results = reciprocal_rank_fusion(query_results)
    response = generation_step(llm,reranked_results,input.query)

    return {'output':response}


class QueryInput(BaseModel):
    
    agent_query: str

# PIPELINE for Agent.
@app.post("/agent")
def agents(input: QueryInput):
    responce = run_agent(input.agent_query)
    return {'response':responce}


if __name__=="__main__":
    # Run the server.
    uvicorn.run(app,host="localhost",port=8000)

