# data path
DATA_ROOT = "data/raw"
DATA_RAW_ROOT = "raw"
DATA_VECTOR_ROOT = "processed"
DOMAINS = ["clapnq", "cloud", "fiqa", "govt"]

# model config
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
COLLECTION_NAME = "multi_domain_corpus"
MD_VECTOR_STORE_NAME = "md_vector_store"