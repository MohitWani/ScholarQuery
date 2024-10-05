from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
import os


def load_document(file_path):
    document = PyPDFLoader(file_path).load()
    print("Document Loaded successfully...")
    return document

def splitter(document):
    doc_split = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap = 200,
    )

    docs_metadata = doc_split.split_documents(documents=document)

    docs = [doc for doc in docs_metadata]
    print("Chunks are created successfully...")
    return docs


def create_vectorstore(docs, embedding, path_tosave):

    vectorstore = FAISS.from_documents(docs, embedding)

    path = path_tosave
    folder = "faiss_index"

    persist_db = os.path.join(path, folder)
    os.makedirs(persist_db, exist_ok=True)

    vectorstore.save_local(persist_db)
    return "Vector database is Saved."

if __name__=="__main__":
    embedding = GPT4AllEmbeddings()

    document = load_document('D:\my_projects\ScholarQuery\Tacotron TTS.pdf')
    split = splitter(document=document)
    print(len(split))
    create_vectorstore(split, embedding, 'D:\my_projects\ScholarQuery')