"""
This module contains the code for processing the 
input data and storing their vector representations.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstores/db_faiss/'

def create_vector_db():
    print("Loading data from directory...")
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print("Splitting texts in data...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    print("Embedding the split texts...")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    
    print("Creating local vector store...")
    db = FAISS.from_documents(texts, embeddings)

    print("Writing vector to local file")
    db.save_local(DB_FAISS_PATH)
    print("\n\n\t\tAll done")

if __name__ == '__main__':
    create_vector_db()

