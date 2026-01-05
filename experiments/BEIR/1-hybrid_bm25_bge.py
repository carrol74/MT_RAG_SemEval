from pathlib import Path

from beir.retrieval.evaluation import EvaluateRetrieval

from src.mtrag.data_loader import load_corpus_jsonl
from src.mtrag.retriever import BM25Retriever, DenseRetriever, rrf_fuse, tokenize_regex
from src.mtrag.beir_io import load_beir_queries, load_beir_qrels_tsv
from src.mtrag.utils import get_device


# ====== clapnq (lastturn) ======
DOMAIN = "clapnq"
CORPUS_JSONL = Path("corpora/passage_level/clapnq.jsonl")
QUERIES_JSONL = Path("human/retrieval_tasks/clapnq/clapnq_lastturn.jsonl")
QRELS_TSV = Path("human/retrieval_tasks/clapnq/qrels/dev.tsv")

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
    dense_index = dense.build_or_load(collection=f"{DOMAIN}_passage_level", corpus=corpus)  # collection used for cache paths

    # 3) run retrieval + fusion -> build "run" dict for BEIR evaluator
    # run: {qid: {docid: score}}
    run = {}

    for qid, query in queries.items():
        bm25_ctx = bm25.retrieve(query, k=BM25_K)
        dense_ctx = dense.retrieve(dense_index, query, k=DENSE_K)

        fused = rrf_fuse([bm25_ctx, dense_ctx], k0=RRF_K0, top_k=FUSED_K)

        run[qid] = {c["document_id"]: float(c["score"]) for c in fused}

    # 4) evaluate with BEIR (Recall/nDCG at k)
    evaluator = EvaluateRetrieval()  # default uses pytrec_eval internally
    k_values = [1, 3, 5, 10]
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, run, k_values)

    # print key numbers similar to README table
    print("\n==== BEIR Eval (Hybrid BM25+BGE, RRF) ====")
    for k in k_values:
        print(f"R@{k}: {recall[f'Recall@{k}']:.4f}   nDCG@{k}: {ndcg[f'NDCG@{k}']:.4f}")


if __name__ == "__main__":
    main()
