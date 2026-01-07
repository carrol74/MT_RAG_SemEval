from pathlib import Path

from beir.retrieval.evaluation import EvaluateRetrieval

from src.mtrag.data_loader import load_corpus_jsonl
from src.mtrag.retriever import BM25Retriever, DenseRetriever, rrf_fuse, tokenize_regex
from src.mtrag.beir_io import load_beir_queries, load_beir_qrels_tsv
from src.mtrag.utils import get_device


# ====== domain: "clapnq" | "cloud" | "fiqa" | "govt" ======
DOMAIN = "clapnq"

# query variant: "lastturn" | "rewrite" | "rewrite_new"
QUERY_VARIANT = "rewrite_new"

# paths (BEIR format)
CORPUS_JSONL = Path(f"corpora/passage_level/{DOMAIN}.jsonl")
QRELS_TSV = Path(f"human/retrieval_tasks/{DOMAIN}/qrels/dev.tsv")

if QUERY_VARIANT == "lastturn":
    QUERIES_JSONL = Path(f"human/retrieval_tasks/{DOMAIN}/{DOMAIN}_lastturn.jsonl")
elif QUERY_VARIANT == "rewrite":
    QUERIES_JSONL = Path(f"human/retrieval_tasks/{DOMAIN}/{DOMAIN}_rewrite.jsonl")
elif QUERY_VARIANT == "rewrite_new":
    QUERIES_JSONL = Path(f"corpora/rewrite_query/{DOMAIN}/{DOMAIN}_rewrite_new.jsonl")
else:
    raise ValueError(f"Unknown QUERY_VARIANT: {QUERY_VARIANT}")

print(f"[INFO] Running BEIR hybrid BM25+BGE on domain: {DOMAIN} (queries={QUERY_VARIANT})")

# dense model
BGE_MODEL = "BAAI/bge-base-en-v1.5"

# retrieval sizes
BM25_K = 200
DENSE_K = 200
FUSED_K = 10
RRF_K0 = 60

DEVICE = get_device()
print(f"[INFO] Using device: {DEVICE}")


def main():
    # 1) load corpus / queries / qrels (BEIR format)
    corpus = load_corpus_jsonl(CORPUS_JSONL)
    queries = load_beir_queries(QUERIES_JSONL)
    qrels = load_beir_qrels_tsv(QRELS_TSV)

    # 2) build retrievers
    bm25 = BM25Retriever(corpus, tokenizer=tokenize_regex)

    dense = DenseRetriever(model_name=BGE_MODEL, device=DEVICE)
    dense_index = dense.build_or_load(collection=f"{DOMAIN}_passage_level", corpus=corpus)  # cache key

    # 3) run retrieval + fusion -> build "run" dict for BEIR evaluator
    # run: {qid: {docid: score}}
    run = {}

    for qid, query in queries.items():
        bm25_ctx = bm25.retrieve(query, k=BM25_K)
        dense_ctx = dense.retrieve(dense_index, query, k=DENSE_K)

        fused = rrf_fuse([bm25_ctx, dense_ctx], k0=RRF_K0, top_k=FUSED_K)

        run[qid] = {c["document_id"]: float(c["score"]) for c in fused}

    # 4) evaluate with BEIR (Recall/nDCG at k)
    evaluator = EvaluateRetrieval()
    k_values = [1, 3, 5, 10]
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, run, k_values)

    print("\n==== BEIR Eval (Hybrid BM25+BGE, RRF) ====")
    for k in k_values:
        print(f"R@{k}: {recall[f'Recall@{k}']:.4f}   nDCG@{k}: {ndcg[f'NDCG@{k}']:.4f}")


if __name__ == "__main__":
    main()
