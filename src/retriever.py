import os
from config import *
from vector_store import MultiDomainVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
class MTHybridRetriever:
    def __init__(self, domain, alpha=0.5):
        """
        Initialize retriever for multiple domains
        
        Args:
            vector_store: Instance of Chroma vector store
            bm25_retriever: Instance of BM25Retriever
            hybrid_retriever:
        """
        self.vector_store = None
        self.bm25_retriever = None
        self.domain = domain
        self.alpha = alpha

    #TODO: not explicitly pass documents, we should initialize them inside the model

    def index_documents(self, documents, isUpdate=False):
        ...

    def build_vector_retriever(self, documents, isUpdate=False):
        ...

    def build_bm25_retriever(self, documents):
        ...

    def search(self, query, k=2):
        """
        Build hybrid retriever combining vector store and BM25 retriever
        
        Args:
            alpha: Weighting factor between vector store and BM25
        """
        ...