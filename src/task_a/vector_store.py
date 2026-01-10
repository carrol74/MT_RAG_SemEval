import os
from src.task_a.config import *
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class MTVectorStore:
    def __init__(self,
                 persist_name,
                 model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
                 ):
        """
        Initialize vector store for multiple domains
        
        Args:
            model_name: Model name for embeddings
            persist_name: Name for the directory to save the vector store
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.persist_directory = os.path.join(DATA_ROOT, DATA_VECTOR_ROOT, persist_name)
        self.vector_store = None
        self.collection_name = persist_name
    
    def build_from_documents(self, 
                             documents):
        """
        Build vector store from multiple JSONL files using JSONLoader
        
        Args:
            documents: List of Document objects
            collection_name: Name for the collection
        """
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        # self.vector_store.persist()
        print("Vector store built and persisted")
    
    def load_existing(self):
        """Load an existing vector store"""
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        print("Loaded existing vector store")