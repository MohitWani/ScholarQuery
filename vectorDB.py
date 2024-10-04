
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma


def store_doc(document):
    loader = PyPDFLoader(document)
    doc = loader.load()

    spliter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    split_doc = spliter.split_documents(doc)

    embedding = GoogleGenerativeAIEmbeddings()
    vectorDB = Chroma()