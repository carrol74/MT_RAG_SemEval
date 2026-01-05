import json
from pathlib import Path
from tqdm import tqdm

from src.mtrag.data_loader import load_tasks_jsonl, load_corpus_jsonl
from src.mtrag.query_builder import last_user_turn
from src.mtrag.retriever import DenseRetriever

SAMPLE_INPUT = Path("human/mtrageval/sample_data/retrieval_taskac_input.jsonl")
OUT_PRED = Path("runs/task_a/dense_bge_sample10/preds.jsonl")

COLLECTION_TO_CORPUS = {
    "mt-rag-clapnq-elser-512-100-20240503": Path("corpora/passage_level/clapnq.jsonl"),
    "mt-rag-govt-elser-512-100-20240611": Path("corpora/passage_level/govt.jsonl"),
}

# BGE (embedding model): 你可以先用 base 版对齐官方思路
MODEL_NAME = "BAAI/bge-base-en-v1.5"

def main():
    OUT_PRED.parent.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks_jsonl(SAMPLE_INPUT)

    retriever = DenseRetriever(model_name=MODEL_NAME)

    # 为出现过的 collection 建/载 index
    indices = {}
    for collection, corpus_path in COLLECTION_TO_CORPUS.items():
        if not any(t["Collection"] == collection for t in tasks):
            continue
        corpus = load_corpus_jsonl(corpus_path)
        indices[collection] = retriever.build_or_load(collection, corpus)

    k = 10
    with OUT_PRED.open("w", encoding="utf-8") as w:
        for t in tqdm(tasks, desc=f"Dense BGE sample10 k={k}"):
            collection = t["Collection"]
            query = last_user_turn(t)
            contexts = retriever.retrieve(indices[collection], query=query, k=k)
            out = dict(t)
            out["contexts"] = contexts
            w.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[OK] wrote {OUT_PRED}")

if __name__ == "__main__":
    main()
