import src.query_rewriter as qr
import random
import logging
import os
from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from src.retrieve_a_model import MTRetrieveModel
def main():
    # rewriter = qr.MTQueryRewriter()
    data_path = os.path.join("data", "raw", "clapnq")  # contains corpus.jsonl, queries.jsonl, qrels.jsonl

    corpus, queries, qrels = GenericDataLoader(
        data_folder=data_path,
        corpus_file="clapnq.jsonl",
        query_file="clapnq_questions.jsonl",
    ).load(split="dev")

    # query_ids = list(queries.keys())
    # queries = [queries[qid] for qid in query_ids]
    # index = random.randint(0, len(queries)-1)
    # history, last_question = re.parse_questions(queries[index])
    # response = rewriter.rewrite_query(last_question, history)
    # print("----------------------"----------")
    # print("Rewritten question:", response)
    # retriever = MTHybridRetriever(domain="clapnq")
    # documents = re.documents_from_corpus(corpus)
    # retriever.index_documents(documents)
    # results = retriever.search(response, k=5)
    # print("--------------------------------")
    # print("Top 5 retrieved documents with scores:", results)
    # print("Dev docs for query:", qrels[query_ids[index]])

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )

    model = MTRetrieveModel(domain="clapnq")
    retriever = EvaluateRetrieval(model)
    results = retriever.retrieve(corpus, queries)
    logging.info(f"Retriever evaluation for k in: {retriever.k_values}")
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    print("NDCG:", ndcg)
    print("MAP:", _map)
    print("Recall:", recall)
    print("Precision:", precision)

    #### Retrieval Example ####
    # query_id, scores_dict = random.choice(list(results.items()))
    # logging.info(f"Query : {queries[query_id]}\n")

    # scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
    # for rank in range(10):
    #     doc_id = scores[rank][0]
    #     logging.info(f"Rank {rank + 1}: {doc_id} [{corpus[doc_id].get('title')}] - {corpus[doc_id].get('text')}\n")

if __name__ == "__main__":
    main()