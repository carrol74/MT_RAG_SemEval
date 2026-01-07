import src.query_rewriter as qr
import random
import logging
import os
from src.config import *
from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from src.retrieve_beir_model import MTRetrieveModel
from src.Utils.beir_process import documents_from_corpus, parse_questions

# from src.format import as fp
from src.retriever import MTHybridRetriever

def test_retriever():
    domain = "clapnq"
    data_path = os.path.join("./data", "raw", domain)  # contains corpus.jsonl, queries.jsonl, qrels.jsonl
    corpus, queries, qrels = GenericDataLoader(
        data_folder=data_path,
        corpus_file=f"{domain}.jsonl",
        query_file=f"{domain}_questions.jsonl",
    ).load(split="dev")

    random_query_id = random.choice(list(queries.keys()))
    random_query = queries[random_query_id]
    question, history = parse_questions(random_query)
    rewriter = qr.MTQueryRewriter()
    rewritten_query = rewriter.rewrite_query(question, history)
    print(f"Original Query: {random_query}")
    print(f"Rewritten Query: {rewritten_query}")
    retreiver = MTHybridRetriever(domain=domain)
    documents = documents_from_corpus(corpus)
    retreiver.index_documents(documents, isUpdate=False)
    results = retreiver.search(rewritten_query, k=5)
    print(f"Top 5 Retrieved Documents for rewritten query:")
    # for doc_id, score in results.items():
    #     print(f"Doc ID: {doc_id}, Score: {score}, Title: {corpus[doc_id].get('title')}")
    doc_ids = list(results.keys())
    for doc_id in doc_ids:
        score = results[doc_id]
        print(f"Doc ID: {doc_id}, Score: {score}, Title: {corpus[doc_id].get('title')}")

def run_task_a():
    for domain in DOMAINS:
        data_path = os.path.join("./data", "raw", domain)  # contains corpus.jsonl, queries.jsonl, qrels.jsonl
        corpus, queries, qrels = GenericDataLoader(
            data_folder=data_path,
            corpus_file=f"{domain}.jsonl",
            query_file=f"{domain}_questions.jsonl",
            # query_file=f"{domain}_rewrite.jsonl",
        ).load(split="dev")

        logging.basicConfig(
            format="%(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
            handlers=[LoggingHandler()],
        )

        model = MTRetrieveModel(domain=domain)
        retriever = EvaluateRetrieval(model)
        results = retriever.retrieve(corpus, queries)
        logging.info(f"Retriever evaluation for k in: {retriever.k_values}")
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

        ### Retrieval Example ####
        # query_id, scores_dict = random.choice(list(results.items()))
        # logging.info(f"Query : {queries[query_id]}\n")

        # scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
        # for rank in range(10):
        #     doc_id = scores[rank][0]
        #     logging.info(f"Rank {rank + 1}: {doc_id} [{corpus[doc_id].get('title')}] - {corpus[doc_id].get('text')}\n")

def main():
    run_task_a()
    # test_retriever()

if __name__ == "__main__":
    main()