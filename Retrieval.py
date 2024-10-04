from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

document = PyPDFLoader("document.pdf").load()

doc_split = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlape = 200,
)

docs = doc_split.split_documents(documents=document)

docs = [doc for doc in docs]

embedding = GPT4AllEmbeddings()

persist_db = "D:\my_projects\ScholarQuery"
vectorstore = FAISS.from_documents(docs, embedding, persist_directory=persist_db)

vectorstore.save_local(persist_db)