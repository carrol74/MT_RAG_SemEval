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

    def build_vector_retriever(self, documents):
        self.vector_store = MultiDomainVectorStore.build_from_documents(documents) 

    def build_bm25_retriever(self, documents):
        self.bm25_retriever = BM25Retriever.from_documents(documents)

    def hybrid_retrieve(self, query, alpha=0.3, k=2, domain=None):
        """
        Build hybrid retriever combining vector store and BM25 retriever
        
        Args:
            alpha: Weighting factor between vector store and BM25
        """
        if self.vector_store is None or self.bm25_retriever is None:
            raise ValueError("Both vector store and BM25 retriever must be initialized")
        vector_result = self.vector_store.similarity_search(query, k=k, filter={"domain": domain} if domain else None)
        self.bm25_retriever.k = k
        bm25_result = self.bm25_retriever.invoke(query)
        combined_result = []
        # TODO: Implement weighted combination logic based on alpha