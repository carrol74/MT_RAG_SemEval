import os
from config import *
from vector_store import MultiDomainVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
class MTHybridRetriever:
    def __init__(self):
        """
        Initialize retriever for multiple domains
        
        Args:
            vector_store: Instance of Chroma vector store
            bm25_retriever: Instance of BM25Retriever
            hybrid_retriever:
        """
        self.vector_store = None
        self.bm25_retriever = None
        self.hybrid_retriever = None

    def build_vector_store(self, documents):
        self.vector_store = MultiDomainVectorStore.build_from_documents(documents)                

    def build_bm25_retriever(self, documents, k=2):
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = k

    def build_hybrid_retriever(self, documents, alpha=0.3, k=2):
        """
        Build hybrid retriever combining vector store and BM25 retriever
        
        Args:
            alpha: Weighting factor between vector store and BM25
        """
        self.build_bm25_retriever(documents, k=k)
        self.build_vector_store(documents)
        chroma_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        self.hybrid_retriever = ...
