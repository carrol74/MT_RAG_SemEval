import logging

from src.config import *
from src.retriever import MTHybridRetriever
from src.query_rewriter import MTQueryRewriter

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- BEIR Imports ---
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.reranking import Rerank

# --- Logging Setup ---
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

class MTRetrieveModel:
    """
    A Wrapper class that makes your custom pipeline look like a BEIR model.
    It implements the 'search' function required by EvaluateRetrieval.
    """
    def __init__(self, domain):
        self.retriever = MTHybridRetriever(domain=domain)
        self.domain = domain

    def search(self, corpus, queries, top_k, **kwargs):
        """
        Input: 
            corpus: dict {doc_id: {text, title}}
            queries: dict {qid: query_text}
            top_k: int (How many to retrieve)
        Output: 
            results: dict {qid: {doc_id: score}}
        """
        documents = documents_from_corpus(corpus)
        self.retriever.index_documents(documents)

        query_ids = list(queries.keys())
        queries = [queries[qid] for qid in query_ids]

        logging.info(f"Starting Retrieval for {len(queries)} queries in domain: {self.domain}...")
        results = {}
        #TODO: Iterate over queries and retrieve top_k documents
        return results
    
def documents_from_corpus(corpus, chunking=False):
    """
    Convert BEIR corpus dict to list of Document objects
    Args:
        corpus: BEIR corpus dictionary
    """
    # TODO: seems like it would chunk by \n
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True
    )
    documents = []
    for corpus_id, doc in corpus.items():
        documents.append(
            Document(
                page_content=doc["text"],
                metadata={
                    CORPUS_KEY: corpus_id,
                    "title": doc.get("title", "")
                }
            )
        )
    if chunking:
        documents = text_splitter.split_documents(documents)
    return documents

def parse_questions(raw_text):
    lines = raw_text.split('\n')
    turns = [line.replace("|user|: ", "").strip() for line in lines if line.strip()]
    
    if not turns:
        return None
    # Everything before the last line is History
    history_turns = turns[:-1] 
    # The very last line is the Question we want to rewrite
    last_question = turns[-1]
    return history_turns, last_question