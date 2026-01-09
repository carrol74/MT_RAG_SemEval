from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import *

class MTRAGGenerator:
    def __init__(self, model_name = GENERATE_MODEL_NAME):
        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto"
        )

    def format_context(self, contexts) -> str:
        """
        "contexts":
        [
            {
                "document_id": "822086267_6698-7277-0-579",
                "source": "",
                "score": 18.759138,
                "text": "2017 Arizona Cardinals season\nOn December 13 , 2016 , the NFL announced that the Cardinals will play the Los Angeles Rams as one of the NFL International Series at Twickenham Stadium in London , England ...",
                "title": "2017 Arizona Cardinals season"
            }, ...
        ],
        """
        context_str = ""
        for i, context in enumerate(contexts):
            context_str += f"Document {context['title']} :\n{context['text']}\n"
        return context_str

    def generate_answer(self, query, contexts) -> str:
        """
        Generates the final answer.
        """
        # 1. Prepare Context
        context_str = self.format_context(contexts)
        
        # 2. Construct Prompt

        prompt = f"""You are a truthful RAG assistant.
        Answer the question using the information in the provided Context.
        If the answer is not contained in the context, state that you don't know.
        Based on the following retrieved information, answer the question using less than 150 words.

        Retrieved Information:
        {context_str}

        Question: {query}

        Answer:"""

        messages = [
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
            max_new_tokens=512,  # Adjust based on your needs
            temperature=0.6,     # Control randomness
            top_p=0.95,
            do_sample=True
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # Parse thinking content (specific to Qwen3-Thinking models)
        try:
            # Find the closing </think> tag (token ID 151668)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        return content