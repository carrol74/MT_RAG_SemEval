import os
import logging
from tqdm import tqdm
from typing import Dict, List
from pathlib import Path
from collections import defaultdict

# Reuse existing components
from src.task_a.config import *
from src.task_b.config import TaskBConfig
from src.task_b.prompt_builder import PromptBuilder
from src.task_b.generator import GeneratorFactory

# Retrieval components from Task A
from src.task_a.query_rewriter import MTQueryRewriter
from src.task_a.retriever import MTHybridRetriever
from src.Utils.beir_process import documents_from_corpus, parse_questions
from src.Utils.format_eval_process import process_data, save_predictions, TaskType
from beir.datasets.data_loader import GenericDataLoader

class MTRAGPipeline:
    def __init__(self, config: TaskBConfig):
        self.config = config
        self.prompt_builder = PromptBuilder(config.prompt)
        self.generator = GeneratorFactory.create_generator(config.model)
        self.query_rewriter = MTQueryRewriter()

        self.retrievers = {} 
        self.corpora = {}

    def _load_retriever_for_domain(self, domain):
        """Initializes the retriever and loads corpus for a specific domain"""
        if domain in self.retrievers:
            return self.retrievers[domain], self.corpora[domain]

        print(f"Loading resources for domain: {domain}...")
        
        # Load Corpus (needed for BM25 and to get text content)
        data_path = os.path.join("./data", "raw", domain)
        # Assuming standard BEIR format as used in run_task_a.py
        corpus, _, _ = GenericDataLoader(
            data_folder=data_path,
            corpus_file=f"{domain}.jsonl",
            query_file=f"{domain}_questions.jsonl" 
        ).load(split="dev")
        
        # Initialize Retriever
        retriever = MTHybridRetriever(domain=domain)
        documents = documents_from_corpus(corpus)
        retriever.index_documents(documents, isUpdate=False)
        
        self.retrievers[domain] = retriever
        self.corpora[domain] = corpus
        return retriever, corpus

    def run(self):
        # 1. Load all tasks
        print("Loading tasks...")
        tasks = process_data(
            input_file_path=self.config.data.input_file,
            task_type=TaskType.TaskTypeC
        )
        
        # 2. Load retrievers for all domains involved
        for domain in DOMAINS:
            self._load_retriever_for_domain(domain)

        predicated_tasks = []

        # 3. Process each tasks
        for task in tqdm(tasks, desc="Processing tasks"):
            domain = DOMAINS_MAP.get(task.get("Collection"), None)
            if domain not in self.retrievers:
                logging.warning(f"Domain {domain} not recognized. Skipping task.")
                continue
            
            retriever = self.retrievers[domain]

            # Parse and rewrite query
            history = [input["text"] for input in task["input"] if input["speaker"] == "user"]
            last_query = history[-1]
            rewritten_query = self.query_rewriter.rewrite_query(history, last_query)

            # Retrieve documents
            retrieved_docs = retriever.search(rewritten_query, k=self.config.top_k, return_content=True)
            task['contexts'] = retrieved_docs
            # Build prompt
            prompt = self.prompt_builder.build_prompt(retrieved_docs, task['input'])

            # Generate response
            generated_response = self.generator.generate(prompt)

            # Compile prediction
            task['predictions'] = [{'text': generated_response}]
            predicated_tasks.append(task)

        # 4. Save predictions
        save_predictions(
            predictions=predicated_tasks,
            output_path=self.config.data.output_file
        )