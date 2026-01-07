import logging
import os

from src.config import *
from src.retriever import MTHybridRetriever
from src.query_rewriter import MTQueryRewriter
from src.Utils.beir_process import documents_from_corpus, parse_questions
from src.Utils.format_eval_process import process_data, TaskType

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

def build_retriever_for_domain(domain, corpus):
    retriever = MTHybridRetriever(domain=domain)
    documents = documents_from_corpus(corpus)
    retriever.index_documents(documents, isUpdate=False)
    return retriever

def batch_tasks(tasks):
    # decide on the domains involved

    ...


def save_predictions(tasks, predictions, output_path):
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
    # save by matching task_id?
    ...

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )
    retrievers = {}
    for domain in DOMAINS:
        corpus, queries, qrels = GenericDataLoader(
            data_folder=os.path.join("./data", "raw", domain),
            corpus_file=f"{domain}.jsonl",
            query_file=f"{domain}_questions.jsonl",
        ).load(split="dev")
        retriever = build_retriever_for_domain(domain, corpus)
        retrievers[domain] = retriever

    data_path = os.path.join("./data", "generation_tasks", "reference.jsonl")

    tasks = process_data(
        input_file_path=data_path,
        task_type=TaskType.TaskTypeA
    )

    predictions = []
    for task in tasks:
        # "Collection": "mt-rag-clapnq-elser-512-100-20240503",
        domain = task["Collection"].split("-")[2]  # extract domain from collection name
        retriever = retrievers.get(domain, None)
        if retriever is None:
            logging.warning(f"No retriever found for domain: {domain}. Skipping task.")
            continue
        query = task["input"][-1]["text"]
        retrieved_docs = retriever.search(query, k=5)
        context_list = []
        for doc_id, score in retrieved_docs.items():
            context_list.append({
                "document_id": doc_id,
                "text": "",  # Placeholder for document text if needed
                "score": score
            })
        prediction = {
            "conversation_id": task["conversation_id"],
            "task_id": task["task_id"],
            "Collection": task["Collection"],
            "input": task["input"],
            "contexts": context_list
        }
        predictions.append(prediction)
        
        