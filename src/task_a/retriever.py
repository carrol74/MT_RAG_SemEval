import os
from src.config import *
from src.vector_store import MTVectorStore
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
class MTHybridRetriever:
    def __init__(self, domain, alpha=0.8):
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
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)

    def index_documents(self, documents, isUpdate=False):
        self.build_vector_retriever(documents, isUpdate=isUpdate)
        self.build_bm25_retriever(documents)

    def build_vector_retriever(self, documents, isUpdate=False):
        self.mtstore = MTVectorStore(persist_name=self.domain)
        if not isUpdate and os.path.exists(os.path.join(DATA_ROOT, DATA_VECTOR_ROOT, f"{self.domain}")):
            self.mtstore.load_existing()
        else:
            self.mtstore.build_from_documents(documents)
    def build_bm25_retriever(self, documents):
        self.bm25_retriever = BM25Retriever.from_documents(documents)

    def search(self, query, k=2, return_content=False):
        """
        Build hybrid retriever combining vector store and BM25 retriever

        """
        top_k = 3 * k
        # rrf for top 3k
        if self.mtstore is None or self.bm25_retriever is None:
            raise ValueError("Both vector store and BM25 retriever must be initialized")
        dense_results = self.mtstore.vector_store.similarity_search(query, k=top_k)
        self.bm25_retriever.k = top_k
        bm25_results = self.bm25_retriever.invoke(query)
        fused_hits, doc_contents = self.reciprocal_rank_fusion(dense_results, bm25_results)
        #rerank for top 2k
        reranked_hits = fused_hits[:2 * k]
        cross_inputs = [[query, doc_contents[doc_id]] for doc_id, _ in reranked_hits]
        cross_scores = self.cross_encoder.predict(cross_inputs)
        reranked_with_scores = list(zip([doc_id for doc_id, _ in reranked_hits], cross_scores))
        reranked_with_scores.sort(key=lambda x: x[1], reverse=True)
        final_hits = reranked_with_scores[:k]
        if return_content:
            return {
                doc_id: {
                    "text": doc_contents[doc_id],
                    "score": float(score)
                } 
                for doc_id, score in final_hits
            }
        clean_results = {
            doc_id: float(score) 
            for doc_id, score in final_hits
        }
        return clean_results

    def reciprocal_rank_fusion(self, dense_res, sparse_res, k_param: int = 60):
        """
        The Core Algorithm.
        Merges two lists of documents based on their rank.
        """
        # Dictionary to hold the final scores: {doc_id: score}
        scores = {}
        doc_contents = {}

        # 1. Process Dense Results
        for rank, doc in enumerate(dense_res):
            doc_id = doc.metadata.get(CORPUS_KEY)
            if not doc_id: continue
            
            # Double Weight for Dense Results
            # Score = alpha / (60 + rank)
            # Rank 0 -> 0.8/60 = 0.0133
            # Rank 1 -> 0.8/61 = 0.0131
            scores[doc_id] = self.alpha / (k_param + rank)
            doc_contents[doc_id] = doc.page_content

        # # 2. Process Sparse Results
        for rank, doc in enumerate(sparse_res):
            doc_id = doc.metadata.get(CORPUS_KEY)
            if not doc_id: continue
            if doc_id not in scores:
                scores[doc_id] = 0.0
            # Add to existing score.
            scores[doc_id] += (1 - self.alpha)/ (k_param + rank)
            doc_contents[doc_id] = doc.page_content

        # 3. Sort by Final Score (Highest first)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True), doc_contents