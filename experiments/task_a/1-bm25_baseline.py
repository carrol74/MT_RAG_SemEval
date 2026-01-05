import json
from pathlib import Path
from tqdm import tqdm

from src.mtrag.data_loader import load_corpus_jsonl, load_tasks_jsonl
from src.mtrag.retriever import BM25Retriever, tokenize_regex
from src.mtrag.query_builder import last_user_turn

SAMPLE_INPUT = Path("/data/users/ruyu/MLforNLP/project/mt-rag-benchmark/human/mtrageval/sample_data/retrieval_taskac_input.jsonl")
OUT_PRED = Path("/data/users/ruyu/MLforNLP/project/mt-rag-benchmark/runs/task_a/bm25_sample/preds.jsonl")

COLLECTION_TO_CORPUS_JSONL = {
    "mt-rag-clapnq-elser-512-100-20240503": Path("corpora/passage_level/clapnq.jsonl"),
    "mt-rag-govt-elser-512-100-20240611": Path("corpora/passage_level/govt.jsonl"),
}

def main():
    OUT_PRED.parent.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks_jsonl(SAMPLE_INPUT)

    retrievers = {}
    for collection, corpus_path in COLLECTION_TO_CORPUS_JSONL.items():
        if not any(t.get("Collection") == collection for t in tasks):
            continue
        corpus = load_corpus_jsonl(corpus_path)
        retrievers[collection] = BM25Retriever(corpus, tokenizer=tokenize_regex)

    k = 10
    with OUT_PRED.open("w", encoding="utf-8") as w:
        for t in tqdm(tasks, desc="BM25 sample10 (module)"):
            collection = t["Collection"]
            query = last_user_turn(t)
            contexts = retrievers[collection].retrieve(query, k=k)

            out = dict(t)
            out["contexts"] = contexts
            w.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[OK] wrote predictions: {OUT_PRED}")

if __name__ == "__main__":
    main()
