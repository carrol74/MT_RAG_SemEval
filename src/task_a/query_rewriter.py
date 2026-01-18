from transformers import pipeline
from src.task_a.config import *
import os
import torch
import json
from json import JSONDecodeError
class MTQueryRewriter:
    def __init__(self, model_name = REWRITE_MODEL_NAME):
        """
        Initialize query rewriter for multiple domains
        
        Args:
            model_name: Name of the LLM model to use
        """
        hf_token = os.getenv("HF_TOKEN", None)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = pipeline(model=model_name, device=device, dtype=torch.bfloat16)

    def rewrite_query(self, history, query):
        """
        Rewrite query based on conversation history
        
        Args:
            query: Original user query
            history: List of previous conversation turns
        """
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
        generation = self.model(
            prompt,
            do_sample=True,
            temperature=0.3,
            top_p=0.7,
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

    def format_rewrite(self, conversations):
        """
        Rewrite query based on conversation history using Structured JSON output.
        """
        
        # 1. NEW PROMPT
        prompt = """
        Given the following conversation, please reword the final utterance from the user into
        a single utterance that does not need the history to understand the user's intent.
        tput in proper JSON format indicating the "class" (standalone or non-standalone)
        and the "reworded version" of the last utterance. Use this format: {"class": "type of
        last utterance", "reworded version": "the last utterance rewritten into a standalone
        question, IF NEEDED"}.
        In your rewording of the last utterance, do not do any unnecessary rephrasing or
        introduction of new terms or concepts that were not mentioned in the prior part of the
        conversation. Be minimal, by staying as close as possible to the shape and meaning of
        the last user utterance. If the last user utterance is already clear and standalone, the
        vorded version should be THE SAME as the last user utterance, and the class
        should be 'standalone'.
        {conversation}}
        ASSISTANT:
        """

        # 2. CALL THE MODEL
        generation = self.model(
            prompt,
            do_sample=True,
            temperature=0.3,
            top_p=0.7,
            max_new_tokens=100,
            return_full_text=False
        )
        
        raw_output = generation[0]['generated_text'].strip()

        # 3. NEW PARSING LOGIC (The critical change)
        try:
            # Attempt to parse the string as JSON
            # Sometimes LLMs add markdown code blocks (```json ... ```), so we clean that
            clean_json = raw_output.replace("```json", "").replace("```", "").strip()
            parsed_output = json.loads(clean_json)
            
            # 4. USE THE "CLASS" LOGIC
            # If the LLM says the query is already standalone, trust the original query.
            # This prevents "over-rewriting" where the LLM changes words unnecessarily.
            if parsed_output.get("class") == "standalone":
                return conversations[-1]["text"]  # Return original last user utterance
            
            # Otherwise, return the rewritten version
            rewritten = parsed_output.get("rewritten version", conversations[-1]["text"])
            return rewritten

        except JSONDecodeError:
            # Fallback: If the model fails to output valid JSON, return original query
            # or attempt a simple regex extraction if needed.
            print(f"Warning: Failed to parse JSON. Raw output: {raw_output}")
            return conversations[-1]["text"]
        except Exception as e:
            print(f"Error in rewriting: {e}")
            return conversations[-1]["text"]