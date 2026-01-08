"""
Centralizes the configuration for Task B, including model, data paths, and prompt templates
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "qwen2.5-7b-instruct"  # Options: qwen2.5-7b-instruct, llama-3.1-8b-instruct, etc.
    temperature: float = 0.1    # Low temperature for factual accuracy
    max_tokens: int = 200       # ~150 words
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    def __post_init__(self):
        # Auto-load API key from environment
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_base is None:
            self.api_base = os.getenv("OPENAI_API_BASE")

@dataclass
class DataConfig:
    """Data paths configuration"""
    # Input file: reference.jsonl with reference passages
    input_file: str = "human/generation_tasks/reference.jsonl"
    
    # Output file: predictions will be added
    output_file: str = "outputs/task_b_predictions.jsonl"
    
    # Evaluation output
    eval_output: str = "outputs/task_b_evaluation.json"

@dataclass
class PromptConfig:
    """Prompt template configuration"""
    max_words: int = 150
    
    # IDK phrase when unanswerable
    idk_phrase: str = "I do not have specific information"
    
    # Prompt template (following MTRAG paper)
    template: str = """Given one or more documents and a user query, generate a response to the query using less than {max_words} words that is grounded in the provided documents. If no answer can be found in the documents, say "{idk_phrase}"

{passages}

{conversation}

RESPONSE:"""

@dataclass
class TaskBConfig:
    """Main Task B configuration"""
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    prompt: PromptConfig = PromptConfig()
    
    # Processing
    batch_size: int = 1  # Process one at a time for now
    verbose: bool = True
    
    # Save intermediate results
    save_every: int = 10


# Default config instance
DEFAULT_CONFIG = TaskBConfig()