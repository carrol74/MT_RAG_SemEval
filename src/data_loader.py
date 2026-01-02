import os
from config import *
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class MTRAGDataLoader:
    def __init__(self, data_root = os.path.join(DATA_ROOT, DATA_RAW_ROOT)):
        """
        Initialize data loader for multiple domains
        
        Args:
            data_root: Root directory containing raw data
        """
        self.data_root = data_root
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True
        )
    
    def load_jsonl_with_loader(self, filepath, domain):
        """
        Load JSONL file using LangChain's JSONLoader
        
        Args:
            filepath: Path to JSONL file
            domain: Domain label for this corpus
        """
        loader = JSONLoader(
            file_path=filepath,
            jq_schema='.text',  # Extract the 'text' field as content
            text_content=False,
            json_lines=True
        )
        
        documents = loader.load()
        documents.metadata["domain"] = domain
            
        print(f"Loaded {len(documents)} documents from {domain}")
        return documents
    
    def documents_from_files(self, 
                             file_domain,
                             use_chunking = True):
        """
        Load documents from multiple JSONL files using JSONLoader
        Args:
            file_domain: List of domains corresponding to files
            use_chunking: Whether to split long documents into chunks
        """
        all_documents = []
        
        # Load all documents using JSONLoader
        print("Loading JSONL files with LangChain JSONLoader...")
        for domain in file_domain:
            filepath = os.path.join(DATA_ROOT, DATA_RAW_ROOT, f"{domain}.jsonl")
            print(f"\nProcessing {filepath} (domain: {domain})...")
            try:
                docs = self.load_jsonl_with_loader(filepath, domain)
                all_documents.extend(docs)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
        print(f"\n{'='*50}")
        print(f"Total documents loaded: {len(all_documents)}")
        print(f"{'='*50}")

        if use_chunking:
            split_docs = self.text_splitter.split_documents(all_documents)
            print(f"Created {len(split_docs)} chunks from {len(all_documents)} documents")
            all_documents = split_docs
        return all_documents