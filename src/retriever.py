import os
from src.config import *
from src.vector_store import MTVectorStore
from langchain_community.retrievers import BM25Retriever
class MTHybridRetriever:
    def __init__(self, domain, alpha=0.5):
        """
        Initialize retriever for multiple domains
        
        Args:
            domain: Domain name
            alpha: Weighting factor for hybrid retrieval
        """
        self.vector_store = None
        self.bm25_retriever = None
        self.domain = domain
        self.alpha = alpha

    def index_documents(self, documents, isUpdate=False):
        self.build_vector_retriever(documents, isUpdate=isUpdate)
        self.build_bm25_retriever(documents)

    def build_vector_retriever(self, documents, isUpdate=False):
        if not isUpdate and os.path.exists(os.path.join(DATA_ROOT, DATA_VECTOR_ROOT, f"{self.domain}")):
            self.vector_store = MTVectorStore.load_existing(collection_name=self.domain)
        else:
            self.vector_store = MTVectorStore.build_from_documents(documents)

    def build_bm25_retriever(self, documents):
        self.bm25_retriever = BM25Retriever.from_documents(documents)

    def search(self, query, k=2):
        """
        Build hybrid retriever combining vector store and BM25 retriever

        """
        if self.vector_store is None or self.bm25_retriever is None:
            raise ValueError("Both vector store and BM25 retriever must be initialized")
        dense_results = self.vector_store.similarity_search(query, k=k)
        self.bm25_retriever.k = k
        bm25_results = self.bm25_retriever.invoke(query)
        fused_hits = self.reciprocal_rank_fusion(dense_results, bm25_results)
        return fused_hits[:k]

    def reciprocal_rank_fusion(self, dense_res, sparse_res, k_param: int = 60):
        """
        The Core Algorithm.
        Merges two lists of documents based on their rank.
        """
        # Dictionary to hold the final scores: {doc_id: score}
        scores = {}

        # 1. Process Dense Results
        for rank, doc in enumerate(dense_res):
            doc_id = doc.metadata.get(CORPUS_KEY)
            if not doc_id: continue
            
            # Double Weight for Dense Results
            # Score = 2 / (60 + rank)
            # Rank 0 -> 2/60 = 0.0333
            # Rank 1 -> 2/61 = 0.0327
            scores[doc_id] += 2.0 / (k_param + rank)

        # 2. Process Sparse Results
        for rank, doc in enumerate(sparse_res):
            doc_id = doc.metadata.get(CORPUS_KEY)
            if not doc_id: continue
            
            # Add to existing score. If doc was in Dense list, it gets a boost!
            scores[doc_id] += 1.0 / (k_param + rank)

        # 3. Sort by Final Score (Highest first)
        sorted(scores.items(), key=lambda item: item[1], reverse=True)
        
        # 4. Return the sorted documents with scores
        return scores.items()