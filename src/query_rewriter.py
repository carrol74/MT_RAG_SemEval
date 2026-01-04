from transformers import pipeline
from src.config import *
import torch
class MTQueryRewriter:
    def __init__(self, model_name = LLM_MODEL_NAME):
        """
        Initialize query rewriter for multiple domains
        
        Args:
            model_name: Name of the LLM model to use
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = pipeline(model=model_name, device=device, dtype=torch.bfloat16)

    def rewrite_query(self, query, history):
        """
        Rewrite query based on conversation history
        
        Args:
            query: Original user query
            history: List of previous conversation turns
        """
        #TODO: Construct prompt using query and history to guide the LLM for rewriting
        system_prompt = """
        You are a Query Resolution Engine.
        Your goal is to analyze the user's last question given the conversation history.

        1. REWRITE the question to be standalone (resolve pronouns like "it", "he", "that").
        2. OUTPUT only the rewritten question without any additional text.

        Example Input:
        History:
        "where does doctor strange get his powers from"
        Last Question:
        "How many films does he appear in"

        Example Output:
        "How many films does Doctor Strange appear in?"
        """
        user_prompt = f"""
        History: {history}
        Last Question: {query}
        Provide the rewritten question below:
        """
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        print("Prompt to LLM:", prompt)
        generation = self.model(
            prompt,
            do_sample=False,
            temperature=1.0,
            top_p=1,
            max_new_tokens=50
        )
        return generation[0]['generated_text'][-1]["content"]