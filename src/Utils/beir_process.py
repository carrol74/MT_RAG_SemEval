from src.task_a.config import *

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

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