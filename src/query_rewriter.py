from langchain.llms import HuggingFaceLLM
from config import *
class MTQueryRewriter:
    def __init__(self, model_name = LLM_MODEL_NAME):
        """
        Initialize query rewriter for multiple domains
        
        Args:
            model_name: Name of the LLM model to use
        """
        self.model = HuggingFaceLLM(model_name=model_name)

    def rewrite_query(self, query, history):
        """
        Rewrite query based on conversation history
        
        Args:
            query: Original user query
            history: List of previous conversation turns
        """
        #TODO: Construct prompt using query and history
        SYSTEM_PROMPT = """
        You are a Query Resolution Engine.
        Your goal is to analyze the user's last question given the conversation history.

        1. REWRITE the question to be standalone (resolve pronouns like "it", "he", "that").
        2. OUTPUT strictly in JSON format.

        Example Input:
        Question: "where does doctor strange get his powers from"
        Answer: "A Doctor Strange's powers come from mystical entities such as Agamotto, Cyttorak, Ikonn, Oshtur, Raggadorr, 
        and Watoomb, who lend their energies for spells. He also wields mystical artifacts including the Cloak of Levitation,
        the Eye of Agamotto, the Book of the Vishanti, and the Orb of Agamotto which give him additional powers."
        Last Q: "How many films does he appear in"

        Example Output:
        {
            "thought": "The user is asking for the film count of the character Doctor Strange, who is the subject of the previous turn. The pronoun 'he' refers to 'Doctor Strange'.",
            "rewrite": "How many films does Doctor Strange appear in?"
        }
        """
        ...