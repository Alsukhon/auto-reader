import os
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
import numpy as np

def main():
    # load PDF
    loader = PyPDFLoader("data/sample.pdf")
    docs = loader.load()  # list of Document(page_content, metadata)

    # split into chunks (~500 characters)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # create embeddings using a local sentence-transformers model
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = [embedder.encode(doc.page_content) for doc in chunks]

    # build & save FAISS index
    #    (we wrap the chunks in langchain Document objects again so FAISS.from_documents works)
    #    Note: FAISS.from_documents expects a list of langchain.Document, plus an "Embeddings" class;
    #    we'll create a dummy LangChain embeddings class that wraps our sentence-transformers embedder.
    
    class SBERTEmbeddings(Embeddings):
        def __init__(self, model):
            self.model = model

        def embed_documents(self, texts):
            return [self.model.encode(t) for t in texts]

        def embed_query(self, text):
            return self.model.encode(text)

    sbert = SBERTEmbeddings(embedder)
    db = FAISS.from_documents(chunks, sbert)
    db.save_local("faiss_index")

    print("✅ Embeddings created and FAISS index saved to './faiss_index'")

if __name__ == "__main__":
    main()
