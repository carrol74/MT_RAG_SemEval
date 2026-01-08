from transformers import pipeline
from src.config import *
import os
import torch
from transformers import AutoTokenizer
class MTQueryRewriter:
    def __init__(self, model_name = LLM_MODEL_NAME):
        """
        Initialize query rewriter for multiple domains
        
        Args:
            model_name: Name of the LLM model to use
        """
        hf_token = os.getenv("HF_TOKEN", None)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = pipeline(model=model_name, device=device, dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

    def rewrite_query(self, history, query):
        """
        Rewrite query based on conversation history
        
        Args:
            query: Original user query
            history: List of previous conversation turns
        """
        #TODO: Construct prompt using query and history to guide the LLM for rewriting
        # system_prompt = """
        # You are a Query Resolution Engine.
        # Your goal is to analyze the user's last question given the conversation history.
        # The question would be used to retrieve relevant documents from a knowledge base.

        # 1. REWRITE the question to be standalone (resolve pronouns like "it", "he", "that").
        # 2. OUTPUT only the rewritten question without any additional text.

        # Example Input:
        # History:
        # "where does doctor strange get his powers from"
        # Last Question:
        # "How many films does he appear in"

        # Example Output:
        # "How many films does Doctor Strange appear in?"
        # """
        system_prompt = """You are a query rewriting assistant for a retrieval system.

        Your task: Rewrite the user's last question into a clear, standalone question that includes all necessary context from the conversation history.

        Rules:
        1. Resolve pronouns (he, she, it, they, that, this) using conversation history
        2. Keep the question in natural language - don't convert to keywords
        3. Preserve the original question's intent and specificity
        4. Only add context that's clearly needed - don't add new assumptions
        5. Output ONLY the rewritten question, nothing else

        Examples:

        History:
        User: "Where does Doctor Strange get his powers from?"
        Assistant: "Doctor Strange gains his powers from studying mystic arts..."

        Last Question: "How many films does he appear in?"
        Rewritten: "How many films does Doctor Strange appear in?"

        ---

        History:
        User: "What is the capital of France?"
        Assistant: "The capital of France is Paris."

        Last Question: "What's its population?"
        Rewritten: "What is the population of Paris?"
        """
        history_text = ""
        for turn in history:
            history_text += f'{turn}\n'
        
        user_prompt = f"""History:
        {history_text if history_text else "(No previous conversation)"}

        Last Question: "{query}"

        Rewritten:"""
        
        prompt = system_prompt + "---Now Your Turn\n" + user_prompt
        # print("Prompt to LLM:", prompt)
        generation = self.model(
            prompt,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            max_new_tokens=100,
            return_full_text=False
        )
        rewritten = generation[0]['generated_text'].strip().lower()
        rewritten = rewritten.split('\n')[0]  # Take only first line
        rewritten = rewritten.strip('"\'')  # Remove quotes if added
        # # Fallback: if rewrite fails or is too short, return original
        if len(rewritten) < 3 or rewritten == query.lower():
            return query
        
        return rewritten