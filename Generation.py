from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
#from langchain_groq import ChatGroq
from langchain_core.load.load import loads
from langchain_core.load.dump import dumps
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint

import logging
logging.basicConfig()

from dotenv import load_dotenv
import os

load_dotenv()

huggingface_token = os.environ['HuggingFace_token']


"""BELOW CODE IS BELONG TO THE ADVANCE RAG AND IT CONTAIN STEPS LIKE // QUERY TRANSLATION->ROUTING->SELF QUERY RETRIEVER->RAG FUSION"""
# Output parser will split the LLM result into a list of queries
class LineListOutputParser(BaseOutputParser[list[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> list[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))

# Multi Query Retriever Function for Query Translation.
def multi_query_retriever(llm, retriever, query):

    QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five 
            different versions of the given user question to retrieve relevant documents from a vector 
            database. By generating multiple perspectives on the user question, your goal is to help
            the user overcome some of the limitations of the distance-based similarity search. 
            Provide these alternative questions separated by newlines.
            Original question: {question}""",
        )
    
    output_parser = LineListOutputParser()
    chain = QUERY_PROMPT | llm | output_parser

    multi_query = MultiQueryRetriever(
        retriever=retriever, llm_chain=chain, parser_key="lines"
    )

    #logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

    return multi_query.invoke(query)

# RAG Fusion step to reranked a retrieved documents.
def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results
    

"""BELOW CODE IS BELONG TO THE NAIVE RAG OR SIMPLE RAG"""

# This is a part of Naive RAG which is a simple RAG.// which use Query->Retriever->llm->Output.
def Naive_retriever(llm,retriever,query):
    template = """Answer the question of user by using the context:

                {context}

                Question: {question}
                """
    # llm = ChatGroq(
    #                 groq_api_key="",
    #                 model_name="mixtral-8x7b-32768"
    #             )

    prompt = ChatPromptTemplate.from_template(template)

    

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(query)

# Function to retrieve similar documents from the vector database // Using similarity search
def get_relavant_doc(query,db):
    
    query_embedding = GPT4AllEmbeddings().embed_query(query)
    
    similar_docs = db.similarity_search_by_vector(query_embedding)

    return similar_docs

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

def model():
    model_kwargs={
            "max_new_tokens": 512,
            "top_k": 2,
            "temperature": 0.1,
            "repetition_penalty": 1.03,
        }
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        input_type=model_kwargs,
        huggingfacehub_api_token= huggingface_token,
    )
    return llm

if __name__=="__main__":
    embedding = GPT4AllEmbeddings()
    db = FAISS.load_local('D:/my_projects/ScholarQuery/faiss_index', embedding, allow_dangerous_deserialization=True)
    retrieval = db.as_retriever()

    query = "Which Technology is use in this paper?"
    llm = model()

    # docs = get_relavant_doc(query,db)
    
    # formatted_docs = format_docs(docs)

    #result = Naive_retriever(llm, retrieval, query)
    result = multi_query_retriever(llm, retrieval, query)

    rank = reciprocal_rank_fusion(result)
    print(rank)
