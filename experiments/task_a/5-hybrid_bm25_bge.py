import json
from pathlib import Path
from tqdm import tqdm

from src.mtrag.data_loader import load_tasks_jsonl, load_corpus_jsonl
from src.mtrag.query_builder import last_user_turn
from src.mtrag.retriever import DenseRetriever
from src.mtrag.retriever import rrf_fuse
from src.mtrag.retriever import BM25Retriever, tokenize_regex  # or tokenize_split

SAMPLE_INPUT = Path("human/mtrageval/sample_data/retrieval_taskac_input.jsonl")
OUT_PRED = Path("runs/task_a/hybrid_bm25_bge_sample10/preds.jsonl")


COLLECTION_TO_CORPUS = {
    "mt-rag-clapnq-elser-512-100-20240503": Path("corpora/passage_level/clapnq.jsonl"),
    "mt-rag-ibmcloud-elser-512-100-20240502": Path("corpora/passage_level/cloud.jsonl"),
    "mt-rag-fiqa-beir-elser-512-100-20240501": Path("corpora/passage_level/fiqa.jsonl"),
    "mt-rag-govt-elser-512-100-20240611": Path("corpora/passage_level/govt.jsonl"),
}

BGE_MODEL = "BAAI/bge-base-en-v1.5"

def main():
    OUT_PRED.parent.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks_jsonl(SAMPLE_INPUT)

    # 1. build BM25 retriever for each collection
    bm25_map = {}
    for collection, corpus_path in COLLECTION_TO_CORPUS.items():
        if not any(t["Collection"] == collection for t in tasks):
            continue
        corpus = load_corpus_jsonl(corpus_path)
        bm25_map[collection] = BM25Retriever(corpus, tokenizer=tokenize_regex)

    # 2. build dense retriever for each collection (FAISS index)
    dense = DenseRetriever(model_name=BGE_MODEL)
    dense_index_map = {}
    for collection, corpus_path in COLLECTION_TO_CORPUS.items():
        if not any(t["Collection"] == collection for t in tasks):
            continue
        corpus = load_corpus_jsonl(corpus_path)
        dense_index_map[collection] = dense.build_or_load(collection, corpus)

    # 3. retrieval and RRF fusion
    bm25_k = 200
    dense_k = 200
    out_k = 10

    with OUT_PRED.open("w", encoding="utf-8") as w:
        for t in tqdm(tasks, desc="Hybrid BM25+BGE (RRF)"):
            collection = t["Collection"]
            query = last_user_turn(t)

            bm25_ctx = bm25_map[collection].retrieve(query, k=bm25_k)
            dense_ctx = dense.retrieve(dense_index_map[collection], query, k=dense_k)

            fused = rrf_fuse([bm25_ctx, dense_ctx], k0=60, top_k=out_k)

            out = dict(t)
            out["contexts"] = fused
            w.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[OK] wrote {OUT_PRED}")

if __name__ == "__main__":
    main()
