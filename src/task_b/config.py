"""
Centralizes the configuration for Task B, including model, data paths, and prompt templates
"""

import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "qwen/qwen2.5-7b-instruct"  # Options: qwen/qwen2.5-7b-instruct, meta-llama/llama-3.1-8b-instruct, etc.
    #temperature: float = 0.1    # Low temperature for factual accuracy
    temperature: float = 0.0    # default: greedy for best accuracy
    max_tokens: int = 200       # ~150 words
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    judge_model: str = "Qwen/Qwen2.5-7B-Instruct" # Model for evaluation/judging

    
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
    eval_output: str = "outputs/task_b_evaluation.jsonl"

@dataclass
class PromptConfig:
    """Prompt template configuration"""
    max_words: int = 150
    # IDK phrase when unanswerable
    idk_phrase: str = "I do not have specific information"
    use_few_shot: bool = True  
    use_system_prompt: bool = True
    
    # Prompt template (following MTRAG paper)(zero-shot)
    zero_shot_template: str = """Given one or more documents and a user query, generate a response to the query using less than {max_words} words that is grounded in the provided documents. If no answer can be found in the documents, say "{idk_phrase}. All responses must be in English only."
    
{passages}

{conversation}

RESPONSE:"""

    # few-shot examples
    few_shot_template_improve: str = """Given one or more documents and a user query, generate a response to the query using less than {max_words} words that is grounded in the provided documents. If no answer can be found in the documents, say "{idk_phrase}",do NOT add explanations, follow-up questions, suggestions, or any additional text. All responses must be in English only.

EXAMPLE 1 (Unanswerable):
PASSAGE 1
The Arizona Cardinals play home games in Phoenix, Arizona.

User: Where do the Arizona Cardinals play THIS WEEK?
RESPONSE: I do not have specific information about which week you are referring to or the Cardinals' schedule for this specific week in the provided documents.

EXAMPLE 2 (Answerable):
PASSAGE 1
Doctor Strange is a fictional superhero appearing in American comic books published by Marvel Comics. Created by artist Steve Ditko and writer Stan Lee, the character first appeared in Strange Tales #138 in 1963. Doctor Strange serves as the Sorcerer Supreme, the primary protector of Earth against magical and mystical threats.

User: Who is Doctor Strange?
RESPONSE: Doctor Strange is a fictional superhero appearing in American comic books published by Marvel Comics. He was created by artist Steve Ditko and writer Stan Lee, first appearing in Strange Tales #138 in 1963. Doctor Strange serves as the Sorcerer Supreme and is the primary protector of Earth against magical and mystical threats.

EXAMPLE 3 (Multi-turn):
PASSAGE 1
Doctor Strange appears in several Marvel Cinematic Universe films. He had his first solo film "Doctor Strange" released in 2016. He then appeared in "Thor: Ragnarok" (2017), "Avengers: Infinity War" (2018), "Avengers: Endgame" (2019), "Spider-Man: No Way Home" (2021), and "Doctor Strange in the Multiverse of Madness" (2022).

User: Who is Doctor Strange?
Agent: Doctor Strange is a fictional superhero appearing in Marvel Comics and the Marvel Cinematic Universe.
User: How many films does he appear in?
RESPONSE: According to the passage, Doctor Strange appears in at least six Marvel Cinematic Universe films: Doctor Strange (2016), Thor: Ragnarok (2017), Avengers: Infinity War (2018), Avengers: Endgame (2019), Spider-Man: No Way Home (2021), and Doctor Strange in the Multiverse of Madness (2022).

Now answer the following:
{passages}

{conversation}

RESPONSE:"""
    
    template: str = field(default="", init=False)
    
    def __post_init__(self):
        self.template = self.few_shot_template_improve if self.use_few_shot else self.zero_shot_template


@dataclass
class TaskBConfig:
    """Main Task B configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    
    batch_size: int = 8  # 1-8 depending on GPU memory
    verbose: bool = True
    save_every: int = 50  # 10-100 depending on dataset size

    eval_provider: str = "hf" 
    eval_config_file: str = "scripts/evaluation/config.yaml" 

# Default config instance
def get_default_config() -> TaskBConfig:
    """Get default configuration"""
    return TaskBConfig()

# For backward compatibility
DEFAULT_CONFIG = get_default_config()


