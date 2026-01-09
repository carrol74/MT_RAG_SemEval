import logging
import os
import json

from src.config import *
from src.retriever import MTHybridRetriever
from src.query_rewriter import MTQueryRewriter
from src.Utils.beir_process import documents_from_corpus, parse_questions
from src.Utils.format_eval_process import process_data, TaskType
from src.generator import MTRAGGenerator

from beir import LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

def build_retriever_for_domain(domain, corpus):
    retriever = MTHybridRetriever(domain=domain)
    documents = documents_from_corpus(corpus)
    retriever.index_documents(documents, isUpdate=False)
    return retriever

def batch_tasks(tasks):
    ...


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

def run_task_a():
    retrievers = {}
    for domain in DOMAINS:
        corpus, queries, qrels = GenericDataLoader(
            data_folder=os.path.join(DATA_ROOT, DATA_RAW_ROOT, domain),
            corpus_file=f"{domain}.jsonl",
            query_file=f"{domain}_questions.jsonl",
        ).load(split="dev")
        retriever = build_retriever_for_domain(domain, corpus)
        retrievers[domain] = retriever

    data_path = os.path.join("mt-rag-benchmark/human/generation_tasks", "RAG.jsonl")
    # data_path = os.path.join(DATA_ROOT, "sample_data", "retrieval_taskac_input.jsonl")
    domain_map = {
        "mt-rag-clapnq-elser-512-100-20240503": "clapnq",
        "mt-rag-govt-elser-512-100-20240611": "govt",
        "mt-rag-fiqa-beir-elser-512-100-20240501": "fiqa",
        "mt-rag-ibmcloud-elser-512-100-20240502": "cloud",
    }
    tasks = process_data(
        input_file_path=data_path,
        task_type=TaskType.TaskTypeA
    )

    predictions = []
    for task in tasks:
        # "Collection": "mt-rag-clapnq-elser-512-100-20240503",
        domain = domain_map.get(task["Collection"], None)
        retriever = retrievers.get(domain, None)
        if retriever is None:
            logging.warning(f"No retriever found for domain: {domain}. Skipping task.")
            continue
        # history = [input["text"] for input in task["input"] if input["speaker"] == "user"]
        # last_query = history[-1]
        query_rewriter = MTQueryRewriter(REWRITE_MODEL_NAME)
        # rewritten_query = query_rewriter.rewrite_query(history, last_query)
        rewritten_query = query_rewriter.format_rewrite(inputs=task["input"])
        retrieved_docs = retriever.search(rewritten_query, k=10, return_content=True)
        context_list = []
        for doc_id, doc_info in retrieved_docs.items():
            context_list.append({
                "document_id": doc_id,
                "text": doc_info["text"],
                "score": doc_info["score"]
            })
        prediction = {
            "conversation_id": task["conversation_id"],
            "task_id": task["task_id"],
            "Collection": task["Collection"],
            "input": task["input"],
            "contexts": context_list
        }
        predictions.append(prediction)
    output_path = os.path.join(DATA_ROOT, "processed", "retrieval_taskA_RAG.jsonl")
    save_predictions(predictions, output_path)
    logging.info(f"Predictions saved to {output_path}")

def run_task_b():
    input_path = os.path.join("mt-rag-benchmark/human", "generation_tasks", "reference.jsonl")
    tasks = process_data(
        input_file_path=input_path,
        task_type=TaskType.TaskTypeB
    )

    generator = MTRAGGenerator(
        model_name=GENERATE_MODEL_NAME
    )

    for task in tasks:
        query = task["input"][-1]["text"]  # last user input
        contexts = task["contexts"]
        answer = generator.generate_answer(query, contexts)
        task["predictions"] = [{'text': answer}]
        logging.info(f"Generated answer :{answer}")

    output_path = os.path.join(DATA_ROOT, "processed", "generation_taskB_predictions.jsonl")
    save_predictions(tasks, output_path)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )
    run_task_a()
    # run_task_b()