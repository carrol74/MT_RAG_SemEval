# data path
DATA_ROOT = "data"
DATA_RAW_ROOT = "raw"
DATA_VECTOR_ROOT = "processed"
DOMAINS = ["clapnq", "cloud", "fiqa", "govt"]
DOMAINS_MAP = {
        "mt-rag-clapnq-elser-512-100-20240503": "clapnq",
        "mt-rag-govt-elser-512-100-20240611": "govt",
        "mt-rag-fiqa-beir-elser-512-100-20240501": "fiqa",
        "mt-rag-ibmcloud-elser-512-100-20240502": "cloud",
    }

#data key
CORPUS_KEY = "id"

# model config
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
# EMBEDDING_MODEL_NAME = 'BAAI/bge-base-en-v1.5'
# REWRITE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
REWRITE_MODEL_NAME = "meta-llama/Llama-3.2-3B"
# REWRITE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"