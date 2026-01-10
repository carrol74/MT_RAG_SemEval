import logging

from task_a.config import *
from src.retriever import MTHybridRetriever
from src.query_rewriter import MTQueryRewriter
from src.Utils.beir_process import documents_from_corpus, parse_questions

class MTRetrieveModel:
    """
    A Wrapper class that makes your custom pipeline look like a BEIR model.
    It implements the 'search' function required by EvaluateRetrieval.
    """
    def __init__(self, domain):
        self.retriever = MTHybridRetriever(domain=domain)
        self.domain = domain
    def search(self, corpus, queries, top_k, score_function, **kwargs):
        """
        Input: 
            corpus: dict {doc_id: {text, title}}
            queries: dict {qid: query_text}
            top_k: int (How many to retrieve)
        Output: 
            results: dict {qid: {doc_id: score}}
        """
        documents = documents_from_corpus(corpus)
        self.retriever.index_documents(documents, isUpdate=False)

        query_ids = list(queries.keys())
        queries = [queries[qid] for qid in query_ids]

        logging.info(f"Starting Retrieval for {len(queries)} queries in domain: {self.domain}...")
        results = {}
        rewriter = MTQueryRewriter()
        # TODO: parallelize this loop if needed
        for i, query in enumerate(queries):
            history, last_question = parse_questions(query)
            rewritten_query = rewriter.rewrite_query(last_question, history)

            retrieved_docs = self.retriever.search(rewritten_query, k=top_k)
            # retrieved_docs = self.retriever.search(query, k=top_k)
            results[query_ids[i]] = retrieved_docs
        logging.info("Retrieval completed.")
        return results