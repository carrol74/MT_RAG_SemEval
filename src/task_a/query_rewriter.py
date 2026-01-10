from transformers import pipeline
import json
from json import JSONDecodeError
from src.task_a.config import *
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
class MTQueryRewriter:
    def __init__(self, model_name = REWRITE_MODEL_NAME):
        """
        Initialize query rewriter for multiple domains
        
        Args:
            model_name: Name of the LLM model to use
        """
        hf_token = os.getenv("HF_TOKEN", None)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.model = pipeline(model=model_name, device=device, dtype=torch.bfloat16)
        # self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def rewrite_query(self, history, query):
        system_prompt = """You are a query generation assistant for a retrieval system.
        Your task: Generate 2-3 diverse search queries based on the user's question and conversation history.
        Rules:
        1. Resolve all pronouns (he, she, it, they, that, this) using conversation history
        2. Generate queries from different angles to capture relevant documents
        3. Keep queries concise and search-friendly
        4. Each query should target different aspects or phrasings

        Examples:

        History:
        User: "Who coaches the Patriots?"
        Assistant: "Bill Belichick"

        Last Question: "How many Super Bowls has he won?"
        Output:
        "Bill Belichick Super Bowl wins", "Bill Belichick championships as Patriots coach", "New England Patriots Super Bowl victories under Belichick"

        ---

        History:
        User: "What is RAG in AI?"
        Assistant: "RAG stands for Retrieval-Augmented Generation..."

        Last Question: "What are the main benefits?"
        Output:
        "benefits of retrieval-augmented generation", "advantages of RAG systems", "why use RAG in language models"
        """
        
        prompt = f"""History:
            {history}

        Last Question: "{query}"
        Output:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

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
            temperature=0.6,
            top_p=0.95,
            max_new_tokens=150,
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