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
        self.build_vector_retriever(documents, isUpdate=isUpdate)
        self.build_bm25_retriever(documents)

    def build_vector_retriever(self, documents, isUpdate=False):
        if not isUpdate and os.path.exists(os.path.join(DATA_ROOT, DATA_VECTOR_ROOT, f"{self.domain}")):
            self.vector_store = MultiDomainVectorStore.load_existing(collection_name=self.domain)
        else:
            self.vector_store = MultiDomainVectorStore.build_from_documents(documents)

    def build_bm25_retriever(self, documents):
        self.bm25_retriever = BM25Retriever.from_documents(documents)

    def search(self, query, k=2):
        """
        Build hybrid retriever combining vector store and BM25 retriever
        
        Args:
            alpha: Weighting factor between vector store and BM25
        """
        if self.vector_store is None or self.bm25_retriever is None:
            raise ValueError("Both vector store and BM25 retriever must be initialized")
        vector_result = self.vector_store.similarity_search(query, k=k)
        self.bm25_retriever.k = k
        bm25_result = self.bm25_retriever.invoke(query)
        combined_result = []
        # TODO: Implement weighted combination logic based on alpha