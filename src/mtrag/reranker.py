from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from sentence_transformers import CrossEncoder


@dataclass
class RerankResult:
    document_id: str
    score: float
    title: str = ""
    text: str = ""

class CrossEncoderReranker:
    """
    Cross-Encoder reranker for (query, passage_text) pairs.

    Input: list of contexts with "document_id" and "text"
    Output: re-scored / re-ordered contexts
    """

    def __init__(self, model_name: str, device: Optional[str] = None, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, contexts: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        pairs = []
        valid_ctx = []
        for c in contexts:
            txt = (c.get("text") or "").strip()
            if not txt:
                continue
            pairs.append((query, txt))
            valid_ctx.append(c)

        if not valid_ctx:
            return contexts[:top_k]

        scores = self.model.predict(pairs, batch_size=self.batch_size)

        for c, s in zip(valid_ctx, scores):
            c["score"] = float(s)

        valid_ctx.sort(key=lambda x: x["score"], reverse=True)
        return valid_ctx[:top_k]
