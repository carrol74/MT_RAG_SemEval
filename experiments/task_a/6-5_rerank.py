import json
from pathlib import Path
from tqdm import tqdm

from src.mtrag.data_loader import load_tasks_jsonl, load_corpus_jsonl
from src.mtrag.query_builder import last_user_turn
from src.mtrag.retriever import BM25Retriever, tokenize_regex
from src.mtrag.retriever import DenseRetriever, rrf_fuse
from src.mtrag.reranker import CrossEncoderReranker

SAMPLE_INPUT = Path("human/mtrageval/sample_data/retrieval_taskac_input.jsonl")
OUT_PRED = Path("runs/task_a/hybrid_bm25_bge_rerank_sample10/preds.jsonl")

COLLECTION_TO_CORPUS = {
    "mt-rag-clapnq-elser-512-100-20240503": Path("corpora/passage_level/clapnq.jsonl"),
    "mt-rag-govt-elser-512-100-20240611": Path("corpora/passage_level/govt.jsonl"),
}

BGE_MODEL = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def main():
    OUT_PRED.parent.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks_jsonl(SAMPLE_INPUT)

    # retrievers
    bm25_map = {}
    dense = DenseRetriever(model_name=BGE_MODEL)
    dense_index_map = {}

    for collection, corpus_path in COLLECTION_TO_CORPUS.items():
        if not any(t["Collection"] == collection for t in tasks):
            continue
        corpus = load_corpus_jsonl(corpus_path)
        bm25_map[collection] = BM25Retriever(corpus, tokenizer=tokenize_regex)
        dense_index_map[collection] = dense.build_or_load(collection, corpus)

    # reranker（可用 GPU：device="cuda"）
    reranker = CrossEncoderReranker(model_name=RERANK_MODEL, device=None, batch_size=32)

    # 召回候选多一些，rerank 输出少一些
    bm25_k = 200
    dense_k = 200
    fused_k = 200      # 先融合得到 200 条候选
    final_k = 10       # 最终输出 top-10（你也可以改 100 看 recall）

    with OUT_PRED.open("w", encoding="utf-8") as w:
        for t in tqdm(tasks, desc="Hybrid(BM25+BGE)+Rerank sample10"):
            collection = t["Collection"]
            query = last_user_turn(t)

            bm25_ctx = bm25_map[collection].retrieve(query, k=bm25_k)
            dense_ctx = dense.retrieve(dense_index_map[collection], query, k=dense_k)

            fused = rrf_fuse([bm25_ctx, dense_ctx], k0=60, top_k=fused_k)

            # rerank 需要 fused 里有 text；你目前 BM25/Dense 都返回 text/title 才能用
            reranked = reranker.rerank(query, fused, top_k=final_k)

            out = dict(t)
            out["contexts"] = reranked
            w.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[OK] wrote {OUT_PRED}")

if __name__ == "__main__":
    main()
