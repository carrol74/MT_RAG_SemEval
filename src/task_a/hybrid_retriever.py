import torch
import faiss
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pickle
from src.task_a.config import *

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForMaskedLM


class HybridDenseSpladeRetriever:
    """
    Dense + SPLADE hybrid retriever with RRF fusion and optional cross-encoder reranking.

    Corpus format:
        corpus = [
            {"id": str, "text": str},
            ...
        ]
    """

    def __init__(
        self,
        alpha,
        dense_model_name=FAISS_MODEL_NAME,
        splade_model_name="naver/splade_v2_distil",
        reranker_model_name=CROSS_ENCODER_MODEL_NAME,
        splade_doc_topk=120,
        splade_query_topk=50,
        rrf_k=60,
        device=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Dense
        self.dense_model = SentenceTransformer(dense_model_name)

        # SPLADE
        self.splade_tokenizer = AutoTokenizer.from_pretrained(splade_model_name)
        self.splade_model = AutoModelForMaskedLM.from_pretrained(splade_model_name)
        self.splade_model.to(self.device)
        self.splade_model.eval()

        # Reranker
        self.reranker = CrossEncoder(reranker_model_name)

        # Config
        self.splade_doc_topk = splade_doc_topk
        self.splade_query_topk = splade_query_topk
        self.rrf_k = rrf_k
        self.alpha = alpha

        # Internal state
        self.doc_ids = []
        self.doc_texts = {}
        self.dense_index = None
        self.splade_postings = defaultdict(list)

    ############################################################
    # SPLADE helpers
    ############################################################

    def _splade_encode(self, texts):
        inputs = self.splade_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.splade_model(**inputs).logits

        return torch.max(torch.log1p(torch.relu(logits)), dim=1).values

    @staticmethod
    def _prune_sparse(vec, top_k):
        values, indices = torch.topk(vec, top_k)
        return {
            int(i): float(v)
            for i, v in zip(indices, values)
            if v > 0
        }

    ############################################################
    # Build index
    ############################################################

    def build(self, corpus, batch_size=32):
        """
        corpus: Dict[str, str]  (doc_id -> text)
        """

        # ---- store corpus ----
        self.doc_ids = list(corpus.keys())
        texts = [d['text'] for d in corpus.values()]
        self.doc_texts = {doc_id: corpus[doc_id]['text'] for doc_id in self.doc_ids}


        # ---- dense index ----
        dense_embs = self.dense_model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        dim = dense_embs.shape[1]
        self.dense_index = faiss.IndexFlatIP(dim)
        self.dense_index.add(dense_embs)

        # ---- SPLADE index ----
        self.splade_postings.clear()

        for i in tqdm(range(0, len(texts), batch_size), desc="Building SPLADE index"):
            batch = texts[i:i + batch_size]
            vecs = self._splade_encode(batch)

            for j, vec in enumerate(vecs):
                doc_id = self.doc_ids[i + j]
                sparse = self._prune_sparse(vec, self.splade_doc_topk)

                for token_id, weight in sparse.items():
                    self.splade_postings[token_id].append((doc_id, weight))

    ############################################################
    # Retrieval components
    ############################################################

    def _dense_search(self, query, k):
        q_emb = self.dense_model.encode(
            [query],
            normalize_embeddings=True
        )
        scores, indices = self.dense_index.search(q_emb, k)
        return {
            self.doc_ids[idx]: float(scores[0][i])
            for i, idx in enumerate(indices[0])
        }

    def _splade_search(self, query, k):
        q_vec = self._splade_encode([query])[0]
        q_sparse = self._prune_sparse(q_vec, self.splade_query_topk)

        scores = defaultdict(float)
        for token_id, qw in q_sparse.items():
            for doc_id, dw in self.splade_postings.get(token_id, []):
                scores[doc_id] += qw * dw

        return dict(
            sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        )

    ############################################################
    # Fusion + reranking
    ############################################################

    def _rrf_fusion(self, dense_hits, sparse_hits):
        scores = defaultdict(float)

        dense_rank = sorted(dense_hits, key=dense_hits.get, reverse=True)
        sparse_rank = sorted(sparse_hits, key=sparse_hits.get, reverse=True)

        for i, doc_id in enumerate(dense_rank):
            scores[doc_id] += self.alpha / (self.rrf_k + i + 1)

        for i, doc_id in enumerate(sparse_rank):
            scores[doc_id] += (1 - self.alpha) / (self.rrf_k + i + 1)

        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    def _rerank(self, query, doc_ids):
        pairs = [(query, self.doc_texts[d]) for d in doc_ids]
        scores = self.reranker.predict(pairs)
        return dict(
            sorted(
                zip(doc_ids, scores),
                key=lambda x: x[1],
                reverse=True
            )
        )

    def save(self, dense_index_path, dense_ids_path, splade_index_path):
        faiss.write_index(self.dense_index, dense_index_path)
        np.save(dense_ids_path, np.array(self.doc_ids))
        with open(splade_index_path, "wb") as f:
            pickle.dump({
                "splade_postings": self.splade_postings,
                "doc_texts": self.doc_texts
            }, f)

    def load(self, dense_index_path, dense_ids_path, splade_index_path):
        self.dense_index = faiss.read_index(dense_index_path)
        self.doc_ids = np.load(dense_ids_path).tolist()
        with open(splade_index_path, "rb") as f:
            data = pickle.load(f)
            self.splade_postings = data["splade_postings"]
            self.doc_texts = data["doc_texts"]

    ############################################################
    # Public API
    ############################################################

    def search(self, query, k=10, return_content=False):
        """
        query: str
        k: final results
        retrieve_k: candidates per retriever
        """
        dense_hits = self._dense_search(query, 3 * k)
        splade_hits = self._splade_search(query, 3 * k)

        fused = self._rrf_fusion(dense_hits, splade_hits)
        
        candidates = list(fused.keys())[: 2 * k]
        reranked = self._rerank(query, candidates)
        final_hits = list(reranked.items())[:k]

        # 4. Return format
        if return_content:
            return [
                {
                    "document_id": doc_id,
                    "text": self.doc_texts[doc_id],
                    "score": float(score),
                }
                for doc_id, score in final_hits
            ]

        clean_results = {
            doc_id: float(score) 
            for doc_id, score in final_hits
        }
        return clean_results