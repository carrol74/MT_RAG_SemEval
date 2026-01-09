# data path
DATA_ROOT = "data"
DATA_RAW_ROOT = "raw"
DATA_VECTOR_ROOT = "processed"
DOMAINS = ["clapnq", "cloud", "fiqa", "govt"]

#data key
CORPUS_KEY = "id"

# model config
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
# EMBEDDING_MODEL_NAME = 'BAAI/bge-base-en-v1.5'
# REWRITE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
REWRITE_MODEL_NAME = "meta-llama/Llama-3.2-3B"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GENERATE_MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507"