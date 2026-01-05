import src.query_rewriter as qr
import random
import logging
import os
from src.config import *
from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from src.retrieve_a_model import MTRetrieveModel
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

if __name__ == "__main__":
    main()