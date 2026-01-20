import logging
import os
import json
import argparse

from src.task_a.hybrid_retriever import HybridDenseSpladeRetriever
from src.task_a.config import *
from src.task_a.retriever import MTHybridRetriever
from src.task_a.query_rewriter import MTQueryRewriter
from src.Utils.beir_process import documents_from_corpus, parse_questions
from src.Utils.format_eval_process import process_data, TaskType

from beir import LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

def build_retriever_for_domain(domain, corpus, alpha):
    # retriever = MTHybridRetriever(domain=domain)
    # documents = documents_from_corpus(corpus)
    # retriever.index_documents(documents, isUpdate=False)
    retriever = HybridDenseSpladeRetriever(alpha=alpha)
    if not os.path.exists(os.path.join(DATA_ROOT, DATA_HYBRID_ROOT, f"{domain}_dense.index")) or \
        not os.path.exists(os.path.join(DATA_ROOT, DATA_HYBRID_ROOT, f"{domain}_dense_ids.npy")) or \
        not os.path.exists(os.path.join(DATA_ROOT, DATA_HYBRID_ROOT, f"{domain}_splade.index")):
        logging.info(f"Building hybrid retriever for domain: {domain} with alpha={alpha}")
        retriever.build(corpus=corpus)
        retriever.save(
            dense_index_path=os.path.join(DATA_ROOT, DATA_HYBRID_ROOT, f"{domain}_dense.index"),
            dense_ids_path=os.path.join(DATA_ROOT, DATA_HYBRID_ROOT, f"{domain}_dense_ids.npy"),
            splade_index_path=os.path.join(DATA_ROOT, DATA_HYBRID_ROOT, f"{domain}_splade.index"),
        )
    else:
        logging.info(f"Loading existing hybrid retriever for domain: {domain} with alpha={alpha}")
        retriever.load(
            dense_index_path=os.path.join(DATA_ROOT, DATA_HYBRID_ROOT, f"{domain}_dense.index"),
            dense_ids_path=os.path.join(DATA_ROOT, DATA_HYBRID_ROOT, f"{domain}_dense_ids.npy"),
            splade_index_path=os.path.join(DATA_ROOT, DATA_HYBRID_ROOT, f"{domain}_splade.index"),
        )
    
    return retriever

def save_predictions(predictions, output_path):
    """
    Sample predictions format:
    {
        "conversation_id": "dd6b6ffd177f2b311abe676261279d2f",
        "task_id": "dd6b6ffd177f2b311abe676261279d2f::2",
        "Collection": "mt-rag-clapnq-elser-512-100-20240503",
        "input": [
            {
            "speaker": "user",
            "text": "where do the arizona cardinals play this week"
            }
        ]
        "contexts":
            [
                {
                    "document_id": "822086267_7384-8758-0-1374",
                    "text": "...",
                    "score": 27.759
                }, ...
            ],
    }

    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for prediction in predictions:
            f.write(json.dumps(prediction) + '\n')

def parse_args():
    parser = argparse.ArgumentParser(description="Run Task A: Retrieval with Query Rewriting")
    parser.add_argument("--input_file", type=str, default="", help="Input file for retrieval tasks")
    parser.add_argument("--output_file", type=str, default="", help="Output file for retrieval predictions")
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )
    retrievers = {}
    domains_alpha = {
        "clapnq": 0.7,
        "fiqa": 0.5,
        "cloud": 0.8,
        "govt": 0.95,
    }
    for domain in DOMAINS:
        corpus, _, _ = GenericDataLoader(
            data_folder=os.path.join(DATA_ROOT, DATA_RAW_ROOT, domain),
            corpus_file=f"{domain}.jsonl",
            query_file=f"{domain}_questions.jsonl",
        ).load(split="dev")
        retriever = build_retriever_for_domain(domain, corpus, alpha=domains_alpha[domain])
        retrievers[domain] = retriever

    data_path = args.input_file

    tasks = process_data(
        input_file_path=data_path,
        task_type=TaskType.TaskTypeA
    )

    # domains_map = {
    #     "clapnq": "clapnq",
    #     "fiqa": "fiqa",
    #     "ibmcloud": "cloud",
    #     "govt": "govt",
    # }
    domains_map = DOMAINS_MAP

    predictions = []
    for task in tasks:
        # "Collection": "mt-rag-clapnq-elser-512-100-20240503",
        domain = domains_map.get(task["Collection"], None)
        retriever = retrievers.get(domain, None)
        if retriever is None:
            logging.warning(f"No retriever found for domain: {domain}. Skipping task.")
            continue
        history = task["input"][:-1]
        last_query = task["input"][-1]["text"]
        query_rewriter = MTQueryRewriter(REWRITE_MODEL_NAME)
        rewritten_query = query_rewriter.rewrite_query(history, last_query)
        context_list = retriever.search(rewritten_query, k=5, return_content=True)
        prediction = {
            "conversation_id": task["conversation_id"],
            "task_id": task["task_id"],
            "Collection": task["Collection"],
            "input": task["input"],
            "contexts": context_list
        }
        predictions.append(prediction)
    save_predictions(predictions, args.output_file)
    logging.info(f"Predictions saved to {args.output_file}")

if __name__ == "__main__":
    main()