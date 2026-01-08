"""
supporting multiple LLMs
"""

from abc import ABC, abstractmethod

# HuggingFace Transformers
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from config import ModelConfig

class BaseGenerator(ABC):
    """Base class for all generators"""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate response from prompt"""
        pass

class HuggingFaceGenerator(BaseGenerator):
    """
    HuggingFace models generator
    Qwen, Llama, Mistral, Mixtral etc.
    """
    
    def __init__(self, config: ModelConfig):
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
            trust_remote_code=True  # Required for some models like Qwen
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",  # Automatically distribute across GPUs
            trust_remote_code=True
        )
        
        print(f"Model loaded on {self.device}")
        
        # Get chat template if available
        self.has_chat_template = hasattr(self.tokenizer, 'chat_template') and \
                                 self.tokenizer.chat_template is not None
    
    def generate(self, prompt: str) -> str:
        """Generate using HuggingFace model"""
        
        # Apply chat template if available
        if self.has_chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096  # Most models support at least 4k
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=True if self.config.temperature > 0 else False,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from response
        if full_response.startswith(formatted_prompt):
            response = full_response[len(formatted_prompt):].strip()
        else:
            # Fallback: try to remove original prompt
            response = full_response.replace(prompt, "").strip()
        
        return response

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


# Example usage
if __name__ == "__main__":
    from config import ModelConfig
    
    # Test different models
    
    # 1. Test Qwen
    print("\n=== Testing Qwen ===")
    qwen_config = ModelConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        temperature=0.1,
        max_tokens=200
    )
    qwen_gen = GeneratorFactory.create_generator(qwen_config)
    
    # 2. Test Llama
    print("\n=== Testing Llama ===")
    llama_config = ModelConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        temperature=0.1,
        max_tokens=200
    )
    llama_gen = GeneratorFactory.create_generator(llama_config)
    
    # Test generation
    test_prompt = """Given documents, answer the question.

PASSAGE 1
Doctor Strange is a fictional superhero appearing in Marvel Comics.

User: Who is Doctor Strange?
Agent:"""
    
    print("\n=== Generating with Qwen ===")
    response_qwen = qwen_gen.generate(test_prompt)
    print(f"Response: {response_qwen}")
    
    print("\n=== Generating with Llama ===")
    response_llama = llama_gen.generate(test_prompt)
    print(f"Response: {response_llama}")