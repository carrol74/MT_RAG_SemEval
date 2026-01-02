import os
from config import *
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class MultiDomainVectorStore:
    def __init__(self, 
                 model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 persist_name = MD_VECTOR_STORE_NAME):
        """
        Initialize vector store for multiple domains
        
        Args:
            model_name: Model name for embeddings
            persist_name: Name for the directory to save the vector store
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.persist_directory = os.path.join(DATA_ROOT, DATA_VECTOR_ROOT, persist_name)
        self.vectorstore = None
    
    def build_from_documents(self, 
                             documents,
                             collection_name = COLLECTION_NAME):
        """
        Build vector store from multiple JSONL files using JSONLoader
        
        Args:
            documents: List of Document objects
            collection_name: Name for the collection
        """
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=collection_name
        )
        self.vectorstore.persist()
        print("Vector store built and persisted")
        return self.vectorstore
    
    def load_existing(self, collection_name = COLLECTION_NAME):
        """Load an existing vector store"""
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        print("Loaded existing vector store")
        return self.vectorstore