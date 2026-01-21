"""
supporting multiple LLMs
"""

from abc import ABC, abstractmethod
from typing import List

# HuggingFace Transformers
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
from transformers import BitsAndBytesConfig

from src.task_b.config import ModelConfig

print(f"Available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

class BaseGenerator(ABC):
    """Base class for all generators"""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate response from prompt"""
        pass

class HuggingFaceGenerator(BaseGenerator):
    def __init__(self, config: ModelConfig):
        """
        Initialize HuggingFace generator
        
        Args:
            config: ModelConfig instance
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers not installed. Run: pip install transformers torch accelerate"
            )
        
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model {config.model_name}...")
        print(f"This may take a few minutes for large models...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            padding_side='left'
        )
        #self.tokenizer.padding_side = 'left'
        
        # Fix pad_token if missing
        if self.tokenizer.pad_token is None:
            print(f"No pad_token found, setting to eos_token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # # 4-bit quantization config
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_use_double_quant=True,
        # )
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            #quantization_config=quantization_config,# 4-bit quantization
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"Model loaded on {self.device}")
        
        # Check if tokenizer has chat template
        self.has_chat_template = hasattr(self.tokenizer, 'chat_template') and \
                                 self.tokenizer.chat_template is not None
            
    def generate_batch(self, prompts: List[str], batch_size: int = 8,
                       use_system_prompt: bool = False) -> List[str]:
        """
        Batch generation for multiple prompts
        
        Args:
            prompts: List of prompts
            batch_size: Number of prompts to process at once
            use_system_prompt: whether to use system/user separation
            
        Returns:
            List of generated responses
        """
        all_responses = []
        
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            # Format prompts
            if self.has_chat_template:
                formatted_prompts = []
                for prompt in batch_prompts:
                    if isinstance(prompt, dict) and use_system_prompt:
                        messages = [
                            {"role": "system", "content": prompt['system']},
                            {"role": "user", "content": prompt['user']}
                        ]
                    else:
                        if isinstance(prompt, dict):
                            content = prompt['user']
                        else:
                            content = prompt
                        messages = [{"role": "user", "content": content}]

                    # print(f"DEBUG: messages = {messages}")
                    # print(f"DEBUG: messages[0]['content'] type = {type(messages[0]['content'])}")
  
                    formatted = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    formatted_prompts.append(formatted)
            else:
                formatted_prompts = []
                for prompt in batch_prompts:
                    if isinstance(prompt, dict):
                        text = prompt['user']
                    else:
                        text = prompt
                    formatted_prompts.append(text)
                #formatted_prompts = batch_prompts
            
            # Tokenize batch (with padding!)
            inputs = self.tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096
            ).to(self.device)
            
            # Record input lengths for each sample
            input_lengths = inputs['attention_mask'].sum(dim=1)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    do_sample=True if self.config.temperature > 0 else False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=self.config.repetition_penalty,
                    top_p=self.config.top_p,
                    #no_repeat_ngram_size=3,
                )
            
            # Decode each output (remove input part)
            for j, output in enumerate(outputs):
                response = self.tokenizer.decode(
                    output[len(inputs['input_ids'][j]):], 
                    skip_special_tokens=True
                )
                all_responses.append(response.strip())
        
        return all_responses
    
    # Keep original generate for backward compatibility
    def generate(self, prompt: str) -> str:
        """Generate single response"""
        return self.generate_batch([prompt], batch_size=1)[0]
    
class GeneratorFactory:
    """Factory to create appropriate generator"""
    
    # Model name mappings for common models
    QWEN_MODELS = [
        'qwen', 'qwen2', 'qwen2.5', 'qwen-2', 'qwen-2.5',
        'qwen/qwen', 'qwen/qwen2', 'qwen/qwen2.5'
    ]
    
    LLAMA_MODELS = [
        'llama', 'llama-3', 'llama-3.1', 'llama-3.2',
        'meta-llama/llama', 'meta-llama/meta-llama'
    ]
    
    @staticmethod
    def create_generator(config: ModelConfig) -> BaseGenerator:
        """
        Create generator based on model name
        
        Args:
            config: Model configuration
            
        Returns:
            Appropriate generator instance
        """
        model_name = config.model_name.lower()
        
        # Qwen models
        if any(x in model_name for x in GeneratorFactory.QWEN_MODELS):
            print("Creating HuggingFace generator for Qwen...")
            return HuggingFaceGenerator(config)
        
        # Llama models
        elif any(x in model_name for x in GeneratorFactory.LLAMA_MODELS):
            print("Creating HuggingFace generator for Llama...")
            return HuggingFaceGenerator(config)
        
        # Default: assume HuggingFace model
        else:
            print(f"Creating HuggingFace generator for {config.model_name}...")
            return HuggingFaceGenerator(config)

# # Example usage
# if __name__ == "__main__":
#     from src.task_b.config import ModelConfig
    
#     # Test Qwen
#     print("\n=== Testing Qwen ===")
#     qwen_config = ModelConfig(
#         model_name="Qwen/Qwen2.5-7B-Instruct",
#         temperature=0.1,
#         max_tokens=200
#     )
#     qwen_gen = GeneratorFactory.create_generator(qwen_config)
    
#     test_prompt = """Given documents, answer the question.

# PASSAGE 1
# Doctor Strange is a fictional superhero appearing in Marvel Comics.

# User: Who is Doctor Strange?
# Agent:"""
    
#     print("\n=== Generating with Qwen ===")
#     response_qwen = qwen_gen.generate(test_prompt)
#     print(f"Response: {response_qwen}")