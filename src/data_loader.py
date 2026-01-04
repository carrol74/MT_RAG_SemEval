from src.config import *
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# class MTRAGDataLoader:
#     def __init__(self, data_root = os.path.join(DATA_ROOT, DATA_RAW_ROOT)):
#         """
#         Initialize data loader for multiple domains
        
#         Args:
#             data_root: Root directory containing raw data
#         """
#         self.data_root = data_root
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=CHUNK_SIZE,
#             chunk_overlap=CHUNK_OVERLAP,
#             add_start_index=True
#         )
    
    # def load_jsonl_with_loader(self, filepath):
    #     """
    #     Load JSONL file using LangChain's JSONLoader

    #     """
    #     loader = JSONLoader(
    #         file_path=filepath,
    #         jq_schema='.text',  # Extract the 'text' field as content
    #         text_content=False,
    #         json_lines=True
    #     )
        
    #     documents = loader.load()
    #     print(f"Loaded {len(documents)}")
    #     return documents
    
    # def documents_from_files(self, 
    #                          name,
    #                          use_chunking = True):
    #     """
    #     Load documents from multiple JSONL files using JSONLoader
    #     Args:
    #         name: Base name of the JSONL file (without .jsonl)
    #         use_chunking: Whether to split long documents into chunks
    #     """
    #     filepath = os.path.join(DATA_ROOT, DATA_RAW_ROOT, f"{name}.jsonl")
    #     try:
    #         docs = self.load_jsonl_with_loader(filepath, name)
    #     except Exception as e:
    #         print(f"Error loading {filepath}: {e}")

    #     if use_chunking:
    #         split_docs = self.text_splitter.split_documents(docs)
    #         print(f"Created {len(split_docs)} chunks from {len(docs)} documents")
    #         docs = split_docs
    #     return docs