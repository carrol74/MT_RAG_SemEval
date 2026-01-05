import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple, DefaultDict
from collections import defaultdict
import json

from rank_bm25 import BM25Okapi
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.mtrag.data_loader import Corpus


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize_regex(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())

def tokenize_split(text: str) -> List[str]:
    return text.lower().split()


@dataclass
class RetrievalContext:
    document_id: str
    score: float
    text: str = ""
    title: str = ""

@dataclass
class DenseIndex:
    index: faiss.Index
    doc_ids: List[str]
    titles: List[str]
    texts: List[str]


class BM25Retriever:
    """
    Minimal BM25 retriever over a passage corpus.
    """

    def __init__(
        self,
        corpus: Corpus,
        tokenizer: Callable[[str], List[str]] = tokenize_regex,
        store_text: bool = True,
        store_title: bool = True,
    ):
        self.corpus = corpus
        self.tokenizer = tokenizer
        tokenized_corpus = [tokenizer(t) for t in corpus.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.store_text = store_text
        self.store_title = store_title

    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        q_tok = self.tokenizer(query)
        scores = self.bm25.get_scores(q_tok)
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        contexts: List[Dict[str, Any]] = []
        for i in idxs:
            c = {
                "document_id": self.corpus.doc_ids[i],
                "score": float(scores[i]),
            }
            if self.store_title:
                c["title"] = self.corpus.titles[i]
            if self.store_text:
                c["text"] = self.corpus.texts[i]
            contexts.append(c)
        return contexts

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm


class DenseRetriever:
    """
    Dense retriever (embedding) using SentenceTransformer + FAISS (cosine via inner product on normalized vectors).
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: Path = Path("runs/index"),
        batch_size: int = 64,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def _cache_paths(self, collection: str) -> Tuple[Path, Path]:
        safe = collection.replace("/", "_")
        idx_path = self.cache_dir / f"{safe}__{self.model_name.replace('/', '_')}.faiss"
        meta_path = self.cache_dir / f"{safe}__{self.model_name.replace('/', '_')}.meta.json"
        return idx_path, meta_path

    def build_or_load(self, collection: str, corpus: Corpus) -> DenseIndex:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        idx_path, meta_path = self._cache_paths(collection)

        if idx_path.exists() and meta_path.exists():
            index = faiss.read_index(str(idx_path))
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return DenseIndex(
                index=index,
                doc_ids=meta["doc_ids"],
                titles=meta["titles"],
                texts=meta["texts"],
            )

        # 1) encode corpus
        embs = self.model.encode(
            corpus.texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype("float32")

        # 2) cosine similarity via inner product on normalized vectors
        embs = _l2_normalize(embs)
        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embs)

        faiss.write_index(index, str(idx_path))
        meta = {
            "doc_ids": corpus.doc_ids,
            "titles": corpus.titles,
            "texts": corpus.texts,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

        return DenseIndex(index=index, doc_ids=corpus.doc_ids, titles=corpus.titles, texts=corpus.texts)

    def retrieve(self, dense_index: DenseIndex, query: str, k: int = 10) -> List[Dict[str, Any]]:
        q = self.model.encode([query], convert_to_numpy=True).astype("float32")
        q = _l2_normalize(q)
        scores, idxs = dense_index.index.search(q, k)
        scores = scores[0]
        idxs = idxs[0]

        contexts: List[Dict[str, Any]] = []
        for score, i in zip(scores, idxs):
            if i < 0:
                continue
            contexts.append(
                {
                    "document_id": dense_index.doc_ids[i],
                    "score": float(score),
                    "title": dense_index.titles[i],
                    "text": dense_index.texts[i],
                }
            )
        return contexts
    
def rrf_fuse(
    runs: List[List[Dict[str, Any]]],
    k0: int = 60,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion.
    runs: list of ranked lists, each item has at least {"document_id": ..., "score": ...}
    Returns a fused ranked list with new "score" (rrf score).
    """
    fused: DefaultDict[str, float] = defaultdict(float)
    payload: Dict[str, Dict[str, Any]] = {}

    for run in runs:
        for rank, item in enumerate(run, start=1):
            docid = item["document_id"]
            fused[docid] += 1.0 / (k0 + rank)
            # record reusable item information
            if docid not in payload:
                payload[docid] = dict(item)

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
    out = []
    for docid, score in ranked:
        it = payload.get(docid, {"document_id": docid})
        it["document_id"] = docid
        it["score"] = float(score)
        out.append(it)
    return out