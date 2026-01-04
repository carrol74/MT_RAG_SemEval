# data path
DATA_ROOT = "data/raw"
DATA_RAW_ROOT = "raw"
DATA_VECTOR_ROOT = "processed"
DOMAINS = ["clapnq", "cloud", "fiqa", "govt"]

# model config
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# evaluation config
K_VALUES = [1, 3, 5, 10]  # Metrics to calculate