from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS


def get_relavant_doc(query, persist_directory, ):
    
    relavent_docs = retrieval.invoke(query)
    return relavent_docs


def get_output(retrieval):
    template = """Answer the question of user by using the context:

                {context}

                Question: {question}
                """
    llm = ChatGroq(
                    groq_api_key="",
                    model_name="mixtral-8x7b-32768"
                )

    prompt = ChatPromptTemplate(template)

    chain = (
                {"context": retrieval | get_relavant_doc, "question":RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser
            )
    return chain

if __name__=="__main__":
    embedding = GPT4AllEmbeddings()
    retrieval = FAISS.load_local('persist_directory', 'embedding')