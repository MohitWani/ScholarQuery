from langchain_core.prompts import ChatPromptTemplate
#from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

huggingface_token = os.environ['HuggingFace_token']

def get_relavant_doc(query,db):
    
    query_embedding = GPT4AllEmbeddings().embed_query(query)
    
    similar_docs = db.similarity_search_by_vector(query_embedding)

    return similar_docs

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


def get_output(llm,retriever,query):
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

    result = get_output(llm, retrieval, query)

    print(result)
