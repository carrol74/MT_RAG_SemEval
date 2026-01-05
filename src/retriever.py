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
        self.mtstore = None
        self.bm25_retriever = None
        self.domain = domain
        self.alpha = alpha

    def index_documents(self, documents, isUpdate=False):
        self.build_vector_retriever(documents, isUpdate=isUpdate)
        self.build_bm25_retriever(documents)

    def build_vector_retriever(self, documents, isUpdate=False):
        self.mtstore = MTVectorStore(persist_name=self.domain)
        if not isUpdate and os.path.exists(os.path.join(DATA_ROOT, DATA_VECTOR_ROOT, f"{self.domain}")):
            self.mtstore.load_existing(collection_name=self.domain)
        else:
            self.mtstore.build_from_documents(documents, collection_name=self.domain)
    def build_bm25_retriever(self, documents):
        self.bm25_retriever = BM25Retriever.from_documents(documents)

    def search(self, query, k=2):
        """
        Build hybrid retriever combining vector store and BM25 retriever

        """
        if self.mtstore is None or self.bm25_retriever is None:
            raise ValueError("Both vector store and BM25 retriever must be initialized")
        dense_results = self.mtstore.vector_store.similarity_search(query, k=k)
        self.bm25_retriever.k = k
        bm25_results = self.bm25_retriever.invoke(query)
        fused_hits = self.reciprocal_rank_fusion(dense_results, bm25_results)
        return dict(fused_hits[:k])

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
            if doc_id not in scores:
                scores[doc_id] = 0.0
            
            # Double Weight for Dense Results
            # Score = 2 / (60 + rank)
            # Rank 0 -> 2/60 = 0.0333
            # Rank 1 -> 2/61 = 0.0327
            scores[doc_id] += 2.0 / (k_param + rank)

        # 2. Process Sparse Results
        for rank, doc in enumerate(sparse_res):
            doc_id = doc.metadata.get(CORPUS_KEY)
            if not doc_id: continue
            if doc_id not in scores:
                scores[doc_id] = 0.0
            # Add to existing score.
            scores[doc_id] += 1.0 / (k_param + rank)

        # 3. Sort by Final Score (Highest first)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)